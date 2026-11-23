// Volumetric fog for the renderer core, in froxels over the render view (Wronski, "Volumetric Fog", SIGGRAPH 2014;
// Hillaire, "Physically Based and Unified Volumetric Rendering in Frostbite", SIGGRAPH 2015): exponential height fog
// plus local fog volumes, lit by the sun through its cascades, the sky's ambient and the local lights through their
// shadow maps, jittered each frame and blended with last frame's froxels (reprojected), then integrated front to back.
// The core reads the result in its forward and sky passes at render size, before TAA or the upscaler.
#include "yea_renderer_fog.h"
#include "frame.h"

#include <Common_3/Graphics/Interfaces/IGraphics.h>
#include <Common_3/Resources/ResourceLoader/Interfaces/IResourceLoader.h>
#include <Common_3/Graphics/FSL/defaults.h>
#include "Shaders/Fog.srt.h"

#include <math.h>
#include <string.h>

namespace renderer_fog
{
namespace
{
struct FogData
{
    uint32_t size[4];
    float    depth[4];
    float    view[3][4];
    float    proj[4];
    float    camPos[4];
    float    prevView[3][4];
    float    prevProj[4];
    float    height[4];
    float    albedo[4];
    float    sunDir[4];
    float    sunColor[4];
    float    ambient[4];
    float    cascadeEnd[4];
    float    misc[4];
    uint32_t counts[4];
};
struct GpuVolume
{
    float center[4], extent[4], medium[4];
};

const uint32_t kMaxFrames = 4, kMaxVolumes = 16;
const float    kSpacing = 2.0f;    // m: slices are finer than this near the camera, coarser beyond
const float    kHistory = 0.9f;    // weight of last frame's froxels
enum Pass { kLight, kIntegrate, kPasses };
const char* const kShaderNames[kPasses] = { "renderer_fog_light.comp", "renderer_fog_integrate.comp" };

Renderer*      gRenderer;
Queue*         gQueue;
uint32_t       gFrames;
Shader*        gShader[kPasses];
Pipeline*      gPipe[kPasses];
Sampler*       gShadowCmp;
DescriptorSet* gPersistent, *gSet;
Buffer*        gCb[kMaxFrames], *gVolumeBuf[kMaxFrames];
Buffer*        gDummyViews, *gDummySh;
RenderTarget*  gDummyAtlas;
Texture*       gLight[2], *gVolume; // this frame's and last frame's light (ping-pong), the integrated volume
uint32_t       gGrid[3], gPixels, gCur;
bool           gReady;

// This frame's plan, from fog_prepare to fog_run; last frame's camera for the reprojection.
FogData  gData;
bool     gOn, gHaveHistory;
uint64_t gLastFrame;
float    gPrevView[3][4], gPrevProj[4];
Buffer*  gLights, *gShapes, *gViews, *gSh;
Texture* gSunAtlas, *gLocalAtlas;

Buffer* makeBuffer(uint64_t size, uint32_t stride, uint32_t descriptors, const char* name)
{
    Buffer*        b = nullptr;
    BufferLoadDesc d = {};
    d.mDesc.mDescriptors = (DescriptorType)descriptors;
    d.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
    d.mDesc.mSize = size;
    d.mDesc.mStartState = descriptors == DESCRIPTOR_TYPE_UNIFORM_BUFFER ? RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER
                                                                         : RESOURCE_STATE_SHADER_RESOURCE;
    d.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
    if (stride)
    {
        d.mDesc.mStructStride = stride;
        d.mDesc.mElementCount = (uint32_t)(size / stride);
    }
    d.mDesc.pName = name;
    d.ppBuffer = &b;
    addResource(&d, nullptr);
    return b;
}

Texture* makeVolume(uint32_t w, uint32_t h, uint32_t depth, const char* name)
{
    TextureDesc d = {};
    d.mWidth = w;
    d.mHeight = h;
    d.mDepth = depth;
    d.mArraySize = d.mMipLevels = 1;
    d.mSampleCount = SAMPLE_COUNT_1;
    d.mFormat = TinyImageFormat_R16G16B16A16_SFLOAT;
    d.mStartState = RESOURCE_STATE_SHADER_RESOURCE;
    d.mDescriptors = (DescriptorType)(DESCRIPTOR_TYPE_TEXTURE | DESCRIPTOR_TYPE_RW_TEXTURE);
    d.pName = name;
    Texture*        t = nullptr;
    TextureLoadDesc l = {};
    l.pDesc = &d;
    l.ppTexture = &t;
    addResource(&l, nullptr);
    return t;
}

void freeVolumes()
{
    Texture** tex[] = { &gLight[0], &gLight[1], &gVolume };
    for (Texture** t : tex)
    {
        if (*t)
            removeResource(*t);
        *t = nullptr;
    }
    gGrid[0] = gGrid[1] = gGrid[2] = 0;
    gHaveHistory = false;
}

// The froxel volumes at a grid size: remade (after the GPU is done with the old ones) only when it changes.
bool ensureVolumes(uint32_t x, uint32_t y, uint32_t z)
{
    if (gVolume && gGrid[0] == x && gGrid[1] == y && gGrid[2] == z)
        return true;
    if (gVolume)
        waitQueueIdle(gQueue);
    freeVolumes();
    gLight[0] = makeVolume(x, y, z, "renderer_fog_light0");
    gLight[1] = makeVolume(x, y, z, "renderer_fog_light1");
    gVolume = makeVolume(x, y, z, "renderer_fog_volume");
    waitForAllResourceLoads();
    if (!gLight[0] || !gLight[1] || !gVolume)
    {
        freeVolumes();
        return false;
    }
    gGrid[0] = x, gGrid[1] = y, gGrid[2] = z;
    return true;
}

void barrierTex(Cmd* cmd, Texture* t, ResourceState from, ResourceState to)
{
    TextureBarrier b = {};
    b.pTexture = t;
    b.mCurrentState = from;
    b.mNewState = to;
    cmdResourceBarrier(cmd, 0, nullptr, 1, &b, 0, nullptr);
}

uint32_t rowOf(yea_handle h)
{
    uint32_t row = h & 0xFFFFFu;
    if (row >= Volumes.size() || !Volumes[row].alive || (Volumes[row].gen & 0xFFFu) != (h >> 20))
        return 0xFFFFFFFFu;
    return row;
}

void setBox(Volume& v, yea_float3 center, yea_float3 half)
{
    v.center = center;
    v.extent = { fabsf(half.x), fabsf(half.y), fabsf(half.z) };
}

float fract(float v) { return v - floorf(v); }
} // namespace

int load(int reloaded)
{
    const YeaHost* host = yea::host();
    gRenderer = (Renderer*)host->renderer;
    gQueue = (Queue*)host->queue;
    gFrames = host->framesInFlight < kMaxFrames ? host->framesInFlight : kMaxFrames;
    gReady = gOn = gHaveHistory = false;
    for (uint32_t i = 0; i < kPasses; ++i)
    {
        ShaderLoadDesc sd = {};
        sd.mComp.pFileName = kShaderNames[i];
        addShader(gRenderer, &sd, &gShader[i]);
        if (!gShader[i])
            return 1;
        PipelineDesc desc = {};
        desc.mType = PIPELINE_TYPE_COMPUTE;
        PIPELINE_LAYOUT_DESC(desc, SRT_LAYOUT_DESC(FogSrt, Persistent), SRT_LAYOUT_DESC(FogSrt, PerFrame), NULL, NULL);
        desc.mComputeDesc.pShaderProgram = gShader[i];
        desc.pName = kShaderNames[i];
        addPipeline(gRenderer, &desc, &gPipe[i]);
        if (!gPipe[i])
            return 1;
    }
    SamplerDesc sd = {};
    sd.mMinFilter = FILTER_LINEAR;
    sd.mMagFilter = FILTER_LINEAR;
    sd.mMipMapMode = MIPMAP_MODE_NEAREST;
    sd.mAddressU = sd.mAddressV = sd.mAddressW = ADDRESS_MODE_CLAMP_TO_EDGE;
    sd.mCompareFunc = CMP_LEQUAL;
    addSampler(gRenderer, &sd, &gShadowCmp);
    {
        // As the core makes its stand-in depth: on Direct3D 12 a depth format reads as a texture only through a render
        // target, and one texel high gets a 1D view on Vulkan.
        RenderTargetDesc t = {};
        t.mWidth = t.mHeight = 2;
        t.mDepth = t.mArraySize = t.mMipLevels = 1;
        t.mSampleCount = SAMPLE_COUNT_1;
        t.mFormat = TinyImageFormat_D32_SFLOAT;
        t.mStartState = RESOURCE_STATE_SHADER_RESOURCE;
        t.mDescriptors = DESCRIPTOR_TYPE_TEXTURE;
        t.pName = "renderer_fog_no_atlas";
        addRenderTarget(gRenderer, &t, &gDummyAtlas);
    }
    gDummyViews = makeBuffer(sizeof(RenderGpuShadowView) * 4, sizeof(RenderGpuShadowView), DESCRIPTOR_TYPE_BUFFER, "renderer_fog_no_views");
    gDummySh = makeBuffer(9 * 16, 16, DESCRIPTOR_TYPE_BUFFER, "renderer_fog_no_sh");
    for (uint32_t i = 0; i < gFrames; ++i)
    {
        gCb[i] = makeBuffer(512, 0, DESCRIPTOR_TYPE_UNIFORM_BUFFER, "renderer_fog_cb");
        gVolumeBuf[i] = makeBuffer(sizeof(GpuVolume) * kMaxVolumes, sizeof(GpuVolume), DESCRIPTOR_TYPE_BUFFER, "renderer_fog_volumes");
    }
    DescriptorSetDesc persistent = SRT_SET_DESC(FogSrt, Persistent, 1, 0);
    addDescriptorSet(gRenderer, &persistent, &gPersistent);
    DescriptorSetDesc perFrame = SRT_SET_DESC(FogSrt, PerFrame, gFrames * kPasses, 0);
    addDescriptorSet(gRenderer, &perFrame, &gSet);
    waitForAllResourceLoads();
    bool ok = gShadowCmp && gDummyAtlas && gDummyAtlas->pTexture && gDummyViews && gDummySh && gPersistent && gSet;
    for (uint32_t i = 0; i < gFrames; ++i)
        ok = ok && gCb[i] && gCb[i]->pCpuMappedAddress && gVolumeBuf[i] && gVolumeBuf[i]->pCpuMappedAddress;
    if (!ok)
        return 1;
    memset(gDummySh->pCpuMappedAddress, 0, 9 * 16);
    memset(gDummyViews->pCpuMappedAddress, 0, sizeof(RenderGpuShadowView) * 4);
    DescriptorData s = {};
    s.mIndex = SRT_RES_IDX(FogSrt, Persistent, gShadowCmp);
    s.ppSamplers = &gShadowCmp;
    updateDescriptorSet(gRenderer, 0, gPersistent, 1, &s);
    gReady = true;
    return 0;
}

void unload()
{
    if (!gRenderer)
        return;
    waitQueueIdle(gQueue);
    freeVolumes();
    for (uint32_t i = 0; i < kMaxFrames; ++i)
    {
        if (gCb[i])
            removeResource(gCb[i]);
        if (gVolumeBuf[i])
            removeResource(gVolumeBuf[i]);
        gCb[i] = gVolumeBuf[i] = nullptr;
    }
    Buffer** bufs[] = { &gDummyViews, &gDummySh };
    for (Buffer** b : bufs)
    {
        if (*b)
            removeResource(*b);
        *b = nullptr;
    }
    if (gDummyAtlas)
        removeRenderTarget(gRenderer, gDummyAtlas);
    gDummyAtlas = nullptr;
    if (gShadowCmp)
        removeSampler(gRenderer, gShadowCmp);
    gShadowCmp = nullptr;
    if (gSet)
        removeDescriptorSet(gRenderer, gSet);
    if (gPersistent)
        removeDescriptorSet(gRenderer, gPersistent);
    gSet = gPersistent = nullptr;
    for (uint32_t i = 0; i < kPasses; ++i)
    {
        if (gPipe[i])
            removePipeline(gRenderer, gPipe[i]);
        if (gShader[i])
            removeShader(gRenderer, gShader[i]);
        gPipe[i] = nullptr;
        gShader[i] = nullptr;
    }
    gReady = gOn = false;
}

void update(uint32_t phase, float dt) {}
void draw(uint32_t pass, const YeaFrame* frame) {}

// --- Volumes -------------------------------------------------------------------------------------------------------
yea_handle fog_volume_add(uint32_t shape, yea_float3 center, yea_float3 half_size)
{
    uint32_t row = 0xFFFFFFFFu;
    for (uint32_t i = 0; i < Volumes.size(); ++i)
        if (!Volumes[i].alive)
        {
            uint32_t gen = Volumes[i].gen + 1;
            Volumes[i] = Volume{};
            Volumes[i].gen = gen;
            row = i;
            break;
        }
    if (row == 0xFFFFFFFFu)
    {
        if (!Volumes.add())
            return 0xFFFFFFFFu;
        row = Volumes.size() - 1;
    }
    Volume& v = Volumes[row];
    v.alive = 1;
    v.shape = shape ? 1 : 0;
    v.density = 0.1f;
    v.albedo = { 1, 1, 1 };
    v.softness = 0.25f;
    setBox(v, center, half_size);
    return row | ((v.gen & 0xFFFu) << 20);
}

void fog_volume_place(yea_handle h, yea_float3 center, yea_float3 half_size)
{
    uint32_t row = rowOf(h);
    if (row != 0xFFFFFFFFu)
        setBox(Volumes[row], center, half_size);
}

void fog_volume_medium(yea_handle h, float density, yea_float3 albedo, float softness)
{
    uint32_t row = rowOf(h);
    if (row == 0xFFFFFFFFu)
        return;
    Volume& v = Volumes[row];
    v.density = density > 0 ? density : 0;
    v.albedo = { fminf(fmaxf(albedo.x, 0), 1), fminf(fmaxf(albedo.y, 0), 1), fminf(fmaxf(albedo.z, 0), 1) };
    v.softness = softness > 0 ? fminf(softness, 1) : 0;
}

void fog_volume_remove(yea_handle h)
{
    uint32_t row = rowOf(h);
    if (row != 0xFFFFFFFFu)
        Volumes[row].alive = 0;
}

void fog_volume_clear()
{
    for (Volume& v : Volumes)
        v.alive = 0;
}

// --- The core's hooks ----------------------------------------------------------------------------------------------
bool fog_on()
{
    if (!gReady || !settings.fog)
        return false;
    if (settings.density > 0)
        return true;
    for (const Volume& v : Volumes)
        if (v.alive && v.density > 0)
            return true;
    return false;
}

bool fog_prepare(uint64_t address)
{
    RenderFrame* F = (RenderFrame*)(uintptr_t)address;
    gOn = false;
    if (!gReady || !F || F->version != RENDER_FRAME_VERSION || !settings.fog || !F->lightBuffer || !F->lightShapes || !F->width ||
        !F->height)
        return false;
    FogData& d = gData;
    memset(&d, 0, sizeof d);
    // The local volumes that hold anything.
    GpuVolume* gv = (GpuVolume*)gVolumeBuf[F->frameIndex]->pCpuMappedAddress;
    uint32_t   volumes = 0;
    for (uint32_t i = 0; i < Volumes.size() && volumes < kMaxVolumes; ++i)
    {
        const Volume& v = Volumes[i];
        if (!v.alive || v.density <= 0)
            continue;
        GpuVolume& g = gv[volumes++];
        g.center[0] = v.center.x, g.center[1] = v.center.y, g.center[2] = v.center.z, g.center[3] = (float)v.shape;
        g.extent[0] = v.extent.x, g.extent[1] = v.extent.y, g.extent[2] = v.extent.z;
        g.extent[3] = 1.0f / fmaxf(v.softness, 1e-3f);
        g.medium[0] = v.albedo.x * v.density, g.medium[1] = v.albedo.y * v.density, g.medium[2] = v.albedo.z * v.density;
        g.medium[3] = v.density;
    }
    const float density = fmaxf(settings.density, 0);
    if (density <= 0 && volumes == 0)
        return false; // nothing in the air
    const uint32_t px = settings.quality == Quality::Low ? 16 : 8, slices = settings.quality == Quality::High ? 128 : 64;
    const uint32_t gx = (F->width + px - 1) / px, gy = (F->height + px - 1) / px;
    if (!ensureVolumes(gx, gy, slices))
        return false;
    gPixels = px;
    const float range = fmaxf(settings.range, 1.0f), logRange = log2f(range / kSpacing + 1.0f);

    d.size[0] = gx, d.size[1] = gy, d.size[2] = slices, d.size[3] = px;
    // Jittered each frame inside the froxel (R2 across, the golden ratio along) for the history to average; the centre
    // without it.
    const bool  temporal = settings.temporal;
    const float n = (float)(F->frameNumber % 4096u);
    d.depth[0] = kSpacing, d.depth[1] = logRange, d.depth[2] = temporal ? fract(0.5f + n * 0.6180340f) : 0.5f;
    d.depth[3] = 1.0f / (float)slices;
    for (int r = 0; r < 3; ++r)
        memcpy(d.view[r], F->view[r], sizeof d.view[r]);
    d.proj[0] = F->p00, d.proj[1] = F->p11;
    d.proj[2] = temporal ? fract(0.5f + n * 0.7548777f) : 0.5f;
    d.proj[3] = temporal ? fract(0.5f + n * 0.5698403f) : 0.5f;
    d.camPos[0] = F->camPos[0], d.camPos[1] = F->camPos[1], d.camPos[2] = F->camPos[2], d.camPos[3] = (float)F->width;
    // Last frame's camera, when there is a last frame to blend with.
    const bool history = temporal && gHaveHistory && F->frameNumber == gLastFrame + 1 && gPrevProj[3] > 0;
    memcpy(d.prevView, gPrevView, sizeof d.prevView);
    memcpy(d.prevProj, gPrevProj, sizeof d.prevProj);
    d.misc[1] = history ? kHistory : 0.0f;
    for (int r = 0; r < 3; ++r)
        memcpy(gPrevView[r], F->view[r], sizeof gPrevView[r]);
    gPrevProj[0] = F->p00, gPrevProj[1] = F->p11, gPrevProj[2] = kSpacing, gPrevProj[3] = 1.0f / logRange;
    gLastFrame = F->frameNumber;

    const yea_float3 albedo = settings.albedo;
    d.height[0] = density, d.height[1] = fmaxf(settings.height_falloff, 0), d.height[2] = settings.base_height;
    d.height[3] = fminf(fmaxf(settings.anisotropy, -0.9f), 0.95f);
    d.albedo[0] = fminf(fmaxf(albedo.x, 0), 1), d.albedo[1] = fminf(fmaxf(albedo.y, 0), 1), d.albedo[2] = fminf(fmaxf(albedo.z, 0), 1);
    d.albedo[3] = (float)F->height;
    // The sun as it arrives, shadowed by its cascades (kept while renderer_rt traces it: the core sets keepShadowMaps).
    const bool shadowMaps = F->shadowViews && F->sunAtlas && F->localAtlas;
    if (F->sunOn)
    {
        memcpy(d.sunDir, F->sunDir, sizeof F->sunDir);
        float l = sqrtf(d.sunDir[0] * d.sunDir[0] + d.sunDir[1] * d.sunDir[1] + d.sunDir[2] * d.sunDir[2]);
        for (int k = 0; k < 3; ++k)
            d.sunDir[k] = l > 0 ? d.sunDir[k] / l : 0.0f;
        d.sunDir[3] = 1.0f;
        memcpy(d.sunColor, F->sunColor, sizeof F->sunColor);
        d.sunColor[3] = shadowMaps && F->sunShadow && (!F->rtSunShadows || F->keepShadowMaps) ? (float)F->cascadeCount : 0.0f;
        memcpy(d.cascadeEnd, F->cascadeEnd, sizeof d.cascadeEnd);
    }
    // The sky's light from all round: its mean radiance (the physical sky's first harmonic, else the gradient's average:
    // over the upper half lerp(horizon, zenith, sqrt(y)) averages to a third and two-thirds, over the lower half
    // lerp(horizon, ground, saturate(-3 y)) to a sixth and five-sixths), times the core's ambient.
    if (F->skySh)
        d.ambient[3] = fmaxf(F->ambient, 1e-6f);
    else
        for (int k = 0; k < 3; ++k)
            d.ambient[k] = (F->skyHorizon[k] * (1.0f / 3.0f + 1.0f / 6.0f) + F->skyZenith[k] * (2.0f / 3.0f) + F->skyGround[k] * (5.0f / 6.0f)) *
                           0.5f * F->ambient;
    d.misc[2] = F->sunTexel;
    d.misc[3] = F->localTexel;
    d.counts[0] = F->lightCount;
    d.counts[1] = volumes;
    d.counts[2] = shadowMaps ? 1u : 0u;
    gLights = F->lightBuffer;
    gShapes = F->lightShapes;
    gViews = shadowMaps ? F->shadowViews : gDummyViews;
    gSh = F->skySh ? F->skySh : gDummySh;
    gSunAtlas = shadowMaps ? F->sunAtlas->pTexture : gDummyAtlas->pTexture;
    gLocalAtlas = shadowMaps ? F->localAtlas->pTexture : gDummyAtlas->pTexture;
    memcpy(gCb[F->frameIndex]->pCpuMappedAddress, &d, sizeof d);

    F->fogVolume = gVolume;
    F->fogParams[0] = 1.0f / kSpacing;
    F->fogParams[1] = (float)slices / logRange;
    F->fogParams[2] = 1.0f / (float)slices;
    F->fogParams[3] = 0.0f;
    F->fogUv[0] = (float)F->width / (float)(gx * px);
    F->fogUv[1] = (float)F->height / (float)(gy * px);
    // The height fog for rays the froxels don't hold.
    if (density > 0)
    {
        memcpy(F->fogMedium, d.height, sizeof d.height);
        memcpy(F->fogAlbedo, d.albedo, sizeof d.albedo);
        F->fogAlbedo[3] = range;
    }
    gOn = true;
    return true;
}

void fog_run(uint64_t address)
{
    RenderFrame* F = (RenderFrame*)(uintptr_t)address;
    if (!gOn || !gReady || !F)
        return;
    gOn = false;
    Cmd*     cmd = F->cmd;
    Texture* before = gLight[gCur], *now = gLight[gCur ^ 1];
    gCur ^= 1;
    cmdBeginDebugMarker(cmd, 0.7f, 0.7f, 0.8f, "Renderer fog");
    for (uint32_t pass = 0; pass < kPasses; ++pass)
    {
        uint32_t       index = F->frameIndex * kPasses + pass;
        Texture*       in = pass == kLight ? before : now;
        Texture*       out = pass == kLight ? now : gVolume;
        Buffer*        volumes = gVolumeBuf[F->frameIndex];
        DescriptorData p[10] = {};
        p[0].mIndex = SRT_RES_IDX(FogSrt, PerFrame, gFog);
        p[0].ppBuffers = &gCb[F->frameIndex];
        p[1].mIndex = SRT_RES_IDX(FogSrt, PerFrame, gLights);
        p[1].ppBuffers = &gLights;
        p[2].mIndex = SRT_RES_IDX(FogSrt, PerFrame, gShapes);
        p[2].ppBuffers = &gShapes;
        p[3].mIndex = SRT_RES_IDX(FogSrt, PerFrame, gViews);
        p[3].ppBuffers = &gViews;
        p[4].mIndex = SRT_RES_IDX(FogSrt, PerFrame, gVolumes);
        p[4].ppBuffers = &volumes;
        p[5].mIndex = SRT_RES_IDX(FogSrt, PerFrame, gSkySh);
        p[5].ppBuffers = &gSh;
        p[6].mIndex = SRT_RES_IDX(FogSrt, PerFrame, gSunAtlas);
        p[6].ppTextures = &gSunAtlas;
        p[7].mIndex = SRT_RES_IDX(FogSrt, PerFrame, gLocalAtlas);
        p[7].ppTextures = &gLocalAtlas;
        p[8].mIndex = SRT_RES_IDX(FogSrt, PerFrame, gIn);
        p[8].ppTextures = &in;
        p[9].mIndex = SRT_RES_IDX(FogSrt, PerFrame, gOut);
        p[9].ppTextures = &out;
        updateDescriptorSet(gRenderer, index, gSet, 10, p);
        barrierTex(cmd, out, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_UNORDERED_ACCESS);
        cmdBindPipeline(cmd, gPipe[pass]);
        cmdBindDescriptorSet(cmd, 0, gPersistent);
        cmdBindDescriptorSet(cmd, index, gSet);
        if (pass == kLight)
            cmdDispatch(cmd, (gGrid[0] + 3) / 4, (gGrid[1] + 3) / 4, (gGrid[2] + 3) / 4);
        else
            cmdDispatch(cmd, (gGrid[0] + 7) / 8, (gGrid[1] + 7) / 8, 1);
        barrierTex(cmd, out, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_SHADER_RESOURCE);
    }
    gHaveHistory = true;
    cmdEndDebugMarker(cmd);
}
} // namespace renderer_fog
