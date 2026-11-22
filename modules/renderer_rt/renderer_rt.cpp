// Ray-traced reflections for the renderer core, where the GPU has inline ray queries (yea::raytracing()).
// Owns the scene's acceleration structures: one bottom-level structure per mesh (built on the frame's command
// list when the mesh appears or changes) and one top-level structure per frame in flight, its instances
// rewritten (setInstances) and rebuilt each frame. After the depth prepass, Shaders/Trace.comp.fsl runs twice:
// the trace (mirror-like pixels at full resolution, the rest one pixel in four with half_resolution), then the
// resolve, one full-size pass that upsamples the quarter-resolution traces and accumulates the reflections over
// frames (two histories, alternating) into the texture the core's forward pass reads.
#include "yea_renderer_rt.h"
#include <yea/raytracing.h>
#include "frame.h"

#include <Common_3/Graphics/Interfaces/IGraphics.h>
#include <Common_3/Graphics/Interfaces/IRay.h>
#include <Common_3/Resources/ResourceLoader/Interfaces/IResourceLoader.h>
#include <Common_3/Graphics/FSL/defaults.h>
#include "Shaders/Trace.srt.h"

#include <math.h>
#include <string.h>

namespace renderer_rt
{
namespace
{
struct RtData
{
    float    right[4], up[4], forward[4], camPos[4], screen[4];
    float    viewProj[4][4], prevViewProj[4][4];
    float    sunDir[4], sunColor[4], skyZenith[4], skyHorizon[4], skyGround[4];
    uint32_t counts[4];
    float    params[4];
    float    temporal[4];
    uint32_t tiles[4];
    uint32_t zero[4];
    float    shadow[4];
    uint32_t shadowMode[4];
    float    taps[32][4];
    uint32_t probe[4]; // TEMPORARY (probe_rays)
};

struct Blas
{
    AccelerationStructure* as;
    Buffer*                vertices;
    Buffer*                indices;
    uint32_t               firstIndex, indexCount, vertexOffset, vertexCount;
};

const uint32_t kMaxFrames = 4;
const uint32_t kMaxMeshes = 1024;
const uint32_t kMaxRetired = 256;
const uint32_t kMaxInstances = 65536; // the core's Instance table
const uint32_t kFlagScreen = 1, kFlagDebug = 4;
// How hits are shadowed (Trace.comp.fsl SHADOWS_*): as the scene, rays for every light, none.
enum { kShadowsScene, kShadowsRays, kShadowsOff };

Renderer*            gRenderer;
Queue*               gQueue;
uint32_t             gFrames;
const YeaRaytracing* gService;
Raytracing*          gRay;
enum { kTrace, kResolve, kProbe, kShaders };
Shader*              gShader[kShaders];
Pipeline*            gPipe[kShaders];
DescriptorSet*       gSet; // two per frame in flight: the trace and the resolve
Buffer*              gCb[kMaxFrames];
// All SHADER_RESOURCE between frames.
Texture*             gTex;      // this frame's trace of the pixels traced at full resolution (only those written)
Texture*             gHalf;     // the half-resolution trace (a quarter of the pixels)
Texture*             gAccum[2]; // accumulated reflections: this frame's is written, last frame's read
Texture*             gAccumZ[2]; // R32F: the view depth each accumulated pixel was for
uint32_t             gAccumIndex;   // the one written last
uint64_t             gAccumFrame;   // the frame it was written (0: no history)
// Per 8 x 8 tile (UNORDERED_ACCESS once used), six sets of flags, so no pass reads a flag it writes: whether
// any pixel reflects (this frame's, which the trace sets, and next frame's, which the resolve clears), and
// whether the history texture's tile holds only zeros, so the resolve skips writing zeros again (four: each
// texture's as of its last write and as of this one, alternating).
Buffer*              gTiles;
ResourceState        gTilesState;
uint32_t             gTilesX, gTileCount;
uint32_t             gFreshFrames; // frames left that write every tile (both histories new)
uint32_t             gResolves;    // resolves run: picks the flag sets
uint32_t             gWidth, gHeight;
bool                 gReady;
// For the shadow maps' lookups at hits: a comparison sampler as the core's, and stand-ins bound while
// renderer_shadows draws no maps (then no light has one, and none is read).
Sampler*             gShadowCmp;
RenderTarget*        gDummyAtlas;  // as the core's gDummyDepth: a depth render target, 2 x 2
DescriptorSet*       gPersistent;  // the sampler's set
Buffer*              gDummyViews;

Blas gBlas[kMaxMeshes];
// Bottom-level structures replaced while frames in flight may still trace them: freed once those are done.
struct Retired
{
    AccelerationStructure* as;
    uint64_t               frame;
};
Retired  gRetired[kMaxRetired];
uint32_t gRetiredCount;

AccelerationStructure*             gTlas[kMaxFrames];
uint32_t                           gTlasCapacity[kMaxFrames];
Buffer*                            gInstBuf[kMaxFrames]; // uint4 per top-level instance: row, first index, first vertex
uint32_t                           gInstCapacity[kMaxFrames];
// Made in load(), never in rt_prepare or rt_run: memory allocated in an export the core calls goes on the core's
// heap, which is freed when the core reloads (renderer_rt then wrote through freed memory and crashed).
AccelerationStructureInstanceDesc* gDescs;
uint32_t                           gDescCapacity;
uint32_t*                          gInstCpu;

float    gPrevViewProj[4][4];
uint64_t gPrevFrame;

void freeTexture()
{
    Texture** all[6] = { &gTex, &gHalf, &gAccum[0], &gAccum[1], &gAccumZ[0], &gAccumZ[1] };
    for (Texture** t : all)
    {
        if (*t)
            removeResource(*t);
        *t = nullptr;
    }
    if (gTiles)
        removeResource(gTiles);
    gTiles = nullptr;
    gWidth = gHeight = 0;
    gAccumFrame = 0;
}

Texture* makeTexture(uint32_t w, uint32_t h, const char* name, TinyImageFormat format = TinyImageFormat_R16G16B16A16_SFLOAT,
                     DescriptorType descriptors = (DescriptorType)(DESCRIPTOR_TYPE_TEXTURE | DESCRIPTOR_TYPE_RW_TEXTURE))
{
    Texture*    t = nullptr;
    TextureDesc d = {};
    d.mWidth = w;
    d.mHeight = h;
    d.mDepth = d.mArraySize = d.mMipLevels = 1;
    d.mSampleCount = SAMPLE_COUNT_1;
    d.mFormat = format; // RWTex2D(half4) is rgba16f on Vulkan, RWTex2D(float) r32f
    d.mStartState = RESOURCE_STATE_SHADER_RESOURCE;
    d.mDescriptors = descriptors;
    d.pName = name;
    TextureLoadDesc l = {};
    l.pDesc = &d;
    l.ppTexture = &t;
    addResource(&l, nullptr);
    waitForAllResourceLoads();
    return t;
}

void createTexture(uint32_t w, uint32_t h)
{
    gTex = makeTexture(w, h, "renderer_rt_reflection");
    gHalf = makeTexture((w + 1) / 2, (h + 1) / 2, "renderer_rt_half");
    for (uint32_t i = 0; i < 2; ++i)
    {
        gAccum[i] = makeTexture(w, h, "renderer_rt_accum");
        gAccumZ[i] = makeTexture(w, h, "renderer_rt_accum_z", TinyImageFormat_R32_SFLOAT);
    }
    gTilesX = (w + 7) / 8;
    gTileCount = gTilesX * ((h + 7) / 8);
    BufferLoadDesc b = {};
    b.mDesc.mDescriptors = DESCRIPTOR_TYPE_RW_BUFFER;
    b.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_GPU_ONLY;
    b.mDesc.mStartState = RESOURCE_STATE_SHADER_RESOURCE;
    b.mDesc.mSize = (uint64_t)gTileCount * 6 * 4;
    b.mDesc.mStructStride = 4;
    b.mDesc.mElementCount = gTileCount * 6;
    b.mDesc.pName = "renderer_rt_tiles";
    b.ppBuffer = &gTiles;
    addResource(&b, nullptr);
    waitForAllResourceLoads();
    gTilesState = RESOURCE_STATE_SHADER_RESOURCE;
    if (!gTex || !gHalf || !gAccum[0] || !gAccum[1] || !gAccumZ[0] || !gAccumZ[1] || !gTiles)
    {
        freeTexture();
        return;
    }
    gFreshFrames = 2;
    gWidth = w;
    gHeight = h;
}

Buffer* makeMapped(uint64_t size, uint32_t stride, DescriptorType type, const char* name)
{
    Buffer*        b = nullptr;
    BufferLoadDesc d = {};
    d.mDesc.mDescriptors = type;
    d.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
    d.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
    d.mDesc.mSize = size;
    d.mDesc.mStartState = type == DESCRIPTOR_TYPE_UNIFORM_BUFFER ? RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER : RESOURCE_STATE_SHADER_RESOURCE;
    if (stride)
    {
        d.mDesc.mStructStride = stride;
        d.mDesc.mElementCount = size / stride;
    }
    d.mDesc.pName = name;
    d.ppBuffer = &b;
    addResource(&d, nullptr);
    waitForAllResourceLoads();
    return b;
}

void retire(AccelerationStructure* as, uint64_t frame)
{
    if (!as)
        return;
    if (gRetiredCount == kMaxRetired)
    {
        // Rare (many meshes changing at once): wait for the GPU and free them all.
        waitQueueIdle(gQueue);
        for (uint32_t i = 0; i < gRetiredCount; ++i)
            removeAccelerationStructure(gRay, gRetired[i].as);
        gRetiredCount = 0;
    }
    gRetired[gRetiredCount++] = { as, frame };
}

void freeRetired(uint64_t frame)
{
    uint32_t kept = 0;
    for (uint32_t i = 0; i < gRetiredCount; ++i)
    {
        if (frame > gRetired[i].frame + gFrames)
            removeAccelerationStructure(gRay, gRetired[i].as);
        else
            gRetired[kept++] = gRetired[i];
    }
    gRetiredCount = kept;
}

// The mesh's bottom-level structure, made and built (on cmd) when it is new or its geometry moved.
AccelerationStructure* meshBlas(Cmd* cmd, const RenderFrame* F, const RenderCaster& c)
{
    if (c.mesh >= kMaxMeshes || !c.indexCount || !c.vertexCount)
        return nullptr;
    Blas& b = gBlas[c.mesh];
    if (b.as && b.vertices == F->vertices && b.indices == F->indices && b.firstIndex == c.firstIndex && b.indexCount == c.indexCount &&
        b.vertexOffset == (uint32_t)c.vertexOffset && b.vertexCount == c.vertexCount)
        return b.as;
    retire(b.as, F->frameNumber);
    b = Blas{};
    AccelerationStructureGeometryDesc g = {};
    g.pVertexBuffer = F->vertices;
    g.pIndexBuffer = F->indices;
    g.mVertexOffset = (uint32_t)c.vertexOffset * F->vertexStride; // bytes
    g.mVertexCount = c.vertexCount;
    g.mVertexStride = F->vertexStride;
    g.mVertexFormat = TinyImageFormat_R32G32B32_SFLOAT;
    g.mIndexOffset = c.firstIndex * 4; // bytes
    g.mIndexCount = c.indexCount;
    g.mIndexType = INDEX_TYPE_UINT32;
    g.mFlags = ACCELERATION_STRUCTURE_GEOMETRY_FLAG_OPAQUE;
    AccelerationStructureDesc d = {};
    d.mType = ACCELERATION_STRUCTURE_TYPE_BOTTOM;
    d.mFlags = ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
    d.mBottom.mDescCount = 1;
    d.mBottom.pGeometryDescs = &g;
    addAccelerationStructure(gRay, &d, &b.as);
    if (!b.as)
        return nullptr;
    RaytracingBuildASDesc build = {};
    build.pAccelerationStructure = b.as;
    build.mIssueRWBarrier = true;
    cmdBuildAccelerationStructure(cmd, gRay, &build);
    b.vertices = F->vertices;
    b.indices = F->indices;
    b.firstIndex = c.firstIndex;
    b.indexCount = c.indexCount;
    b.vertexOffset = (uint32_t)c.vertexOffset;
    b.vertexCount = c.vertexCount;
    return b.as;
}

uint32_t grow(uint32_t need)
{
    uint32_t n = 64;
    while (n < need)
        n *= 2;
    return n;
}

void mulRows(const float a[4][4], const float b[4][4], float out[4][4])
{
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            out[r][c] = a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c] + a[r][3] * b[3][c];
}
} // namespace

int load(int reloaded)
{
    const YeaHost* host = yea::host();
    gRenderer = (Renderer*)host->renderer;
    gQueue = (Queue*)host->queue;
    gFrames = host->framesInFlight < kMaxFrames ? host->framesInFlight : kMaxFrames;
    gService = yea::raytracing();
    if (!gService)
        return 0; // no ray queries on this GPU: the core keeps screen-space reflections
    if (!initRaytracing(gRenderer, &gRay) || !gRay)
    {
        yea::host()->log(1, "renderer_rt: ray tracing could not start; screen reflections stay");
        gRay = nullptr;
        return 0;
    }
    gDescs = new AccelerationStructureInstanceDesc[kMaxInstances];
    gInstCpu = new uint32_t[kMaxInstances * 4];
    gDescCapacity = kMaxInstances;
    const char* names[kShaders] = { "renderer_rt_trace.comp", "renderer_rt_resolve.comp", "renderer_rt_probe.comp" };
    for (uint32_t k = 0; k < kShaders; ++k)
    {
        ShaderLoadDesc sd = {};
        sd.mComp.pFileName = names[k];
        addShader(gRenderer, &sd, &gShader[k]);
        if (!gShader[k])
            return 1;
        PipelineDesc desc = {};
        desc.mType = PIPELINE_TYPE_COMPUTE;
        desc.mComputeDesc.pShaderProgram = gShader[k];
        desc.pName = "renderer_rt_trace";
        PIPELINE_LAYOUT_DESC(desc, SRT_LAYOUT_DESC(RtSrt, Persistent), SRT_LAYOUT_DESC(RtSrt, PerFrame), NULL, NULL);
        addPipeline(gRenderer, &desc, &gPipe[k]);
        if (!gPipe[k])
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
        // Made as the core makes its stand-in depth: on Direct3D 12 a depth format reads as a texture only through
        // a render target (typeless, with a depth view), and one texel high gets a 1D view on Vulkan.
        RenderTargetDesc t = {};
        t.mWidth = t.mHeight = 2;
        t.mDepth = t.mArraySize = t.mMipLevels = 1;
        t.mSampleCount = SAMPLE_COUNT_1;
        t.mFormat = TinyImageFormat_D32_SFLOAT;
        t.mStartState = RESOURCE_STATE_SHADER_RESOURCE;
        t.mDescriptors = DESCRIPTOR_TYPE_TEXTURE;
        t.pName = "renderer_rt_no_atlas";
        addRenderTarget(gRenderer, &t, &gDummyAtlas);
    }
    gDummyViews = makeMapped(sizeof(RenderGpuShadowView) * 4, sizeof(RenderGpuShadowView), DESCRIPTOR_TYPE_BUFFER, "renderer_rt_no_views");
    DescriptorSetDesc persistent = SRT_SET_DESC(RtSrt, Persistent, 1, 0);
    addDescriptorSet(gRenderer, &persistent, &gPersistent);
    if (!gShadowCmp || !gDummyAtlas || !gDummyAtlas->pTexture || !gDummyViews || !gPersistent)
    {
        unload();
        return 1;
    }
    {
        DescriptorData s = {};
        s.mIndex = SRT_RES_IDX(RtSrt, Persistent, gShadowCmp);
        s.ppSamplers = &gShadowCmp;
        updateDescriptorSet(gRenderer, 0, gPersistent, 1, &s);
    }
    DescriptorSetDesc setDesc = SRT_SET_DESC(RtSrt, PerFrame, gFrames * 2, 0);
    addDescriptorSet(gRenderer, &setDesc, &gSet);
    for (uint32_t i = 0; i < gFrames; ++i)
        gCb[i] = makeMapped(sizeof(RtData) < 256 ? 256 : (sizeof(RtData) + 255) / 256 * 256, 0, DESCRIPTOR_TYPE_UNIFORM_BUFFER, "renderer_rt_cb");
    gReady = gSet != nullptr;
    yea::log(gReady ? "renderer_rt: ray-traced reflections on" : "renderer_rt: the trace pipeline failed");
    return gReady ? 0 : 1;
}

void unload()
{
    if (!gRenderer || !gRay)
        return;
    waitQueueIdle(gQueue);
    freeTexture();
    for (uint32_t i = 0; i < gRetiredCount; ++i)
        removeAccelerationStructure(gRay, gRetired[i].as);
    gRetiredCount = 0;
    for (uint32_t m = 0; m < kMaxMeshes; ++m)
        if (gBlas[m].as)
            removeAccelerationStructure(gRay, gBlas[m].as);
    for (uint32_t i = 0; i < kMaxFrames; ++i)
    {
        if (gTlas[i])
            removeAccelerationStructure(gRay, gTlas[i]);
        if (gInstBuf[i])
            removeResource(gInstBuf[i]);
        if (gCb[i])
            removeResource(gCb[i]);
    }
    if (gSet)
        removeDescriptorSet(gRenderer, gSet);
    if (gShadowCmp)
        removeSampler(gRenderer, gShadowCmp);
    if (gDummyAtlas)
        removeRenderTarget(gRenderer, gDummyAtlas);
    if (gPersistent)
        removeDescriptorSet(gRenderer, gPersistent);
    if (gDummyViews)
        removeResource(gDummyViews);
    gShadowCmp = nullptr;
    gDummyAtlas = nullptr;
    gPersistent = nullptr;
    gDummyViews = nullptr;
    for (uint32_t k = 0; k < kShaders; ++k)
    {
        if (gPipe[k])
            removePipeline(gRenderer, gPipe[k]);
        if (gShader[k])
            removeShader(gRenderer, gShader[k]);
    }
    exitRaytracing(gRenderer, gRay);
    gRay = nullptr;
}

void update(uint32_t phase, float dt) {}
void draw(uint32_t pass, const YeaFrame* frame) {}

bool rt_prepare(uint64_t address)
{
    RenderFrame* F = (RenderFrame*)(uintptr_t)address;
    if (!gReady || !settings.enabled || !F || F->version != RENDER_FRAME_VERSION || !F->width || !F->height || !F->taaOn)
        return false;
    if (yea::viewMode() || !F->surface || !F->previous || !F->drawableCount || !F->vertices || !F->indices || !F->instances ||
        !F->materials || !F->lightBuffer || !F->exposure || !F->motion || F->frameIndex >= kMaxFrames)
        return false;
    if (F->width != gWidth || F->height != gHeight)
    {
        waitQueueIdle(gQueue);
        freeTexture();
        createTexture(F->width, F->height);
    }
    if (!gTex)
        return false;

    // Camera: world basis and projection, this frame's unjittered view-projection, and last frame's.
    const float(*v)[4] = F->view;
    RtData d = {};
    for (int k = 0; k < 3; ++k)
    {
        d.right[k] = v[0][k];
        d.up[k] = v[1][k];
        d.forward[k] = v[2][k];
        d.camPos[k] = F->camPos[k];
    }
    d.right[3] = F->p00;
    d.up[3] = F->p11;
    d.forward[3] = F->nearZ;
    float maxRough = settings.max_roughness < 0 ? 0 : (settings.max_roughness > 1 ? 1 : settings.max_roughness);
    d.camPos[3] = maxRough;
    d.screen[0] = (float)F->width;
    d.screen[1] = (float)F->height;
    d.screen[2] = F->jitter[0];
    d.screen[3] = F->jitter[1];
    const float proj[4][4] = { { F->p00, 0, 0, 0 }, { 0, F->p11, 0, 0 }, { 0, 0, 0, F->nearZ }, { 0, 0, 1, 0 } };
    mulRows(proj, v, d.viewProj);
    bool havePrev = gPrevFrame + 1 == F->frameNumber;
    memcpy(d.prevViewProj, havePrev ? gPrevViewProj : d.viewProj, sizeof d.prevViewProj);
    memcpy(gPrevViewProj, d.viewProj, sizeof gPrevViewProj);
    gPrevFrame = F->frameNumber;
    for (int k = 0; k < 3; ++k)
    {
        d.sunDir[k] = F->sunDir[k];
        d.sunColor[k] = F->sunColor[k];
        d.skyZenith[k] = F->skyZenith[k];
        d.skyHorizon[k] = F->skyHorizon[k];
        d.skyGround[k] = F->skyGround[k];
    }
    d.sunDir[3] = F->sunOn ? 1.0f : 0.0f;
    d.sunColor[3] = F->ambient;
    d.counts[0] = F->allLightCount;
    d.counts[1] = (uint32_t)(F->frameNumber % 64u);
    d.counts[2] = (settings.screen_colors ? kFlagScreen : 0) | (settings.show_hits ? kFlagDebug : 0);
    // Hits are shadowed as the scene is: from the local lights' maps (the same filter, normal bias and
    // contact-hardening as the forward pass), a ray where a light casts but has no map, a ray for the sun. When the
    // scene gains a ray-traced shadow mode, picking it here (kShadowsRays) is all reflections need.
    bool maps = F->localAtlas && F->localAtlas->pTexture && F->shadowViews;
    d.shadowMode[1] = settings.reflected_shadows == ReflectedShadows::Rays  ? kShadowsRays
                      : settings.reflected_shadows == ReflectedShadows::Off ? kShadowsOff
                                                                             : kShadowsScene;
    d.shadowMode[0] = F->taps >= 4 && F->taps <= 32 ? F->taps : 16;
    d.shadowMode[2] = F->receiverPlane ? 1 : 0;
    d.shadowMode[3] = maps ? 1 : 0;
    d.shadow[0] = F->softness;
    d.shadow[1] = F->normalBias;
    d.shadow[2] = F->localTexel;
    d.shadow[3] = F->localLightSize;
    d.probe[0] = settings.probe_rays < 0 ? 0 : (uint32_t)settings.probe_rays;
    for (uint32_t i = 0; i < d.shadowMode[0]; ++i)
    {
        float r = sqrtf(((float)i + 0.5f) / (float)d.shadowMode[0]), t = (float)i * 2.39996323f;
        d.taps[i][0] = r * cosf(t);
        d.taps[i][1] = r * sinf(t);
    }
    d.counts[3] = F->drawableCount;
    d.params[0] = settings.max_distance < 1 ? 1 : settings.max_distance;
    d.params[1] = 0.05f;
    d.params[2] = settings.half_resolution ? (settings.full_below < 0 ? 0 : settings.full_below) : 2.0f; // 2: every pixel
    // History only when it was written last frame, by the same camera path (not after a jump of frames).
    d.temporal[0] = settings.history_frames < 1 ? 1.0f : (float)settings.history_frames;
    d.temporal[1] = havePrev && gAccumFrame + 1 == F->frameNumber ? 1.0f : 0.0f;
    d.temporal[2] = settings.brightest < 1 ? 1.0f : settings.brightest;
    d.temporal[3] = settings.half_resolution ? 1.0f : 0.0f;
    d.tiles[0] = gTilesX;
    d.tiles[1] = gTileCount;
    // The sets: [0, 2) "has pixels to trace" by resolve parity, [2, 6) "all zero" by resolve count mod 4 (the history
    // texture written now was last written two resolves ago). A resolve that doesn't run leaves its flags as they were.
    d.tiles[2] = gTileCount * (gResolves & 1);
    d.tiles[3] = gTileCount * ((gResolves + 1) & 1);
    d.zero[0] = gTileCount * (2 + ((gResolves + 2) & 3));
    d.zero[1] = gTileCount * (2 + (gResolves & 3));
    d.zero[2] = gFreshFrames ? 1 : 0;
    memcpy(gCb[F->frameIndex]->pCpuMappedAddress, &d, sizeof d);

    F->rtReflection = gAccum[gAccumIndex ^ 1];
    F->rtMaxRoughness = maxRough;
    F->rtDebug = settings.show_hits;
    return true;
}

void rt_run(uint64_t address)
{
    RenderFrame* F = (RenderFrame*)(uintptr_t)address;
    uint32_t w = gAccumIndex ^ 1; // written this frame; gAccumIndex holds last frame's
    if (!gReady || !F || !gTex || F->rtReflection != gAccum[w] || !F->depth || !F->depth->pTexture || !F->surface ||
        !F->surface->pTexture || !F->motion || !F->motion->pTexture || !F->previous || !F->previous->pTexture || !F->instances ||
        !F->materials || !F->lightBuffer || !F->vertices || !F->indices || !F->exposure)
        return;
    Cmd*     cmd = F->cmd;
    uint32_t fi = F->frameIndex;
    uint32_t count = F->drawableCount < gDescCapacity ? F->drawableCount : gDescCapacity;
    freeRetired(F->frameNumber);

    cmdBeginDebugMarker(cmd, 0.5f, 0.8f, 1.0f, "Renderer ray-traced reflections");
    // Room for this frame's instances (grown rarely; the old top-level structure of this frame slot is done).
    if (count > gInstCapacity[fi])
    {
        if (gInstBuf[fi])
            removeResource(gInstBuf[fi]);
        gInstCapacity[fi] = grow(count);
        gInstBuf[fi] = makeMapped((uint64_t)gInstCapacity[fi] * 16, 16, DESCRIPTOR_TYPE_BUFFER, "renderer_rt_instances");
    }

    // Instances: each live instance with its mesh's bottom-level structure (built here when new).
    uint32_t n = 0;
    for (uint32_t i = 0; i < count; ++i)
    {
        const RenderCaster& c = F->drawables[i];
        AccelerationStructure* blas = meshBlas(cmd, F, c);
        if (!blas || c.instance >= F->instanceCount)
            continue;
        AccelerationStructureInstanceDesc& desc = gDescs[n];
        desc = AccelerationStructureInstanceDesc{};
        desc.pBottomAS = blas;
        memcpy(desc.mTransform, F->instanceRows[c.instance].row, sizeof desc.mTransform);
        desc.mInstanceID = n;
        desc.mInstanceMask = c.castShadow ? 3u : 1u; // 1: seen in reflections, 2: casts shadows
        desc.mFlags = ACCELERATION_STRUCTURE_INSTANCE_FLAG_FORCE_OPAQUE;
        gInstCpu[n * 4 + 0] = c.instance;
        gInstCpu[n * 4 + 1] = c.firstIndex;
        gInstCpu[n * 4 + 2] = (uint32_t)c.vertexOffset;
        gInstCpu[n * 4 + 3] = 0;
        n++;
    }
    if (!n || !gInstBuf[fi])
    {
        cmdEndDebugMarker(cmd);
        return;
    }
    memcpy(gInstBuf[fi]->pCpuMappedAddress, gInstCpu, (size_t)n * 16);

    // This frame slot's top-level structure: made (with room to spare) when too small, then its instances
    // rewritten and rebuilt.
    if (n > gTlasCapacity[fi])
    {
        if (gTlas[fi])
            removeAccelerationStructure(gRay, gTlas[fi]);
        gTlas[fi] = nullptr;
        uint32_t cap = grow(n);
        if (cap > gDescCapacity)
            cap = gDescCapacity;
        for (uint32_t i = n; i < cap; ++i)
            gDescs[i] = gDescs[0];
        AccelerationStructureDesc d = {};
        d.mType = ACCELERATION_STRUCTURE_TYPE_TOP;
        d.mFlags = ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE; // rebuilt every frame, but 53-odd instances build in no time; every ray traverses it
        d.mTop.mDescCount = cap;
        d.mTop.pInstanceDescs = gDescs;
        addAccelerationStructure(gRay, &d, &gTlas[fi]);
        gTlasCapacity[fi] = gTlas[fi] ? cap : 0;
    }
    if (!gTlas[fi] || !gService->setInstances(gTlas[fi], gDescs, n))
    {
        cmdEndDebugMarker(cmd);
        return;
    }
    RaytracingBuildASDesc build = {};
    build.pAccelerationStructure = gTlas[fi];
    build.mIssueRWBarrier = true;
    cmdBuildAccelerationStructure(cmd, gRay, &build);
    // The core fills its light buffer after rt_prepare: the count goes in now.
    ((RtData*)gCb[fi]->pCpuMappedAddress)->counts[0] = F->allLightCount;

    Buffer*        instances = F->instances;
    Buffer*        materials = F->materials;
    Buffer*        lights = F->lightBuffer;
    Buffer*        vertices = F->vertices;
    Buffer*        indices = F->indices;
    Texture*       previous = F->previous->pTexture;
    Buffer*        exposure = F->exposure;
    Texture*       motion = F->motion->pTexture;
    // renderer_shadows has drawn its atlas by now (shadows_render runs before the prepass).
    bool           maps = F->localAtlas && F->localAtlas->pTexture && F->shadowViews;
    Texture*       atlas = maps ? F->localAtlas->pTexture : gDummyAtlas->pTexture;
    Buffer*        views = maps ? F->shadowViews : gDummyViews;
    DescriptorData p[24] = {};
    p[22].mIndex = SRT_RES_IDX(RtSrt, PerFrame, gShadowViews);
    p[22].ppBuffers = &views;
    p[23].mIndex = SRT_RES_IDX(RtSrt, PerFrame, gLocalAtlas);
    p[23].ppTextures = &atlas;
    p[0].mIndex = SRT_RES_IDX(RtSrt, PerFrame, gScene);
    p[0].ppAccelerationStructures = &gTlas[fi];
    p[1].mIndex = SRT_RES_IDX(RtSrt, PerFrame, gRt);
    p[1].ppBuffers = &gCb[fi];
    p[2].mIndex = SRT_RES_IDX(RtSrt, PerFrame, gDepth);
    p[2].ppTextures = &F->depth->pTexture;
    p[3].mIndex = SRT_RES_IDX(RtSrt, PerFrame, gSurface);
    p[3].ppTextures = &F->surface->pTexture;
    p[4].mIndex = SRT_RES_IDX(RtSrt, PerFrame, gPrevious);
    p[4].ppTextures = &previous;
    p[5].mIndex = SRT_RES_IDX(RtSrt, PerFrame, gInstances);
    p[5].ppBuffers = &instances;
    p[6].mIndex = SRT_RES_IDX(RtSrt, PerFrame, gRtInstances);
    p[6].ppBuffers = &gInstBuf[fi];
    p[7].mIndex = SRT_RES_IDX(RtSrt, PerFrame, gMaterials);
    p[7].ppBuffers = &materials;
    p[8].mIndex = SRT_RES_IDX(RtSrt, PerFrame, gLights);
    p[8].ppBuffers = &lights;
    p[9].mIndex = SRT_RES_IDX(RtSrt, PerFrame, gVertices);
    p[9].ppBuffers = &vertices;
    p[10].mIndex = SRT_RES_IDX(RtSrt, PerFrame, gIndices);
    p[10].ppBuffers = &indices;
    p[11].mIndex = SRT_RES_IDX(RtSrt, PerFrame, gExposure);
    p[11].ppBuffers = &exposure;
    p[12].mIndex = SRT_RES_IDX(RtSrt, PerFrame, gHalfIn);
    p[13].mIndex = SRT_RES_IDX(RtSrt, PerFrame, gFullIn);
    p[14].mIndex = SRT_RES_IDX(RtSrt, PerFrame, gMotion);
    p[14].ppTextures = &motion;
    p[15].mIndex = SRT_RES_IDX(RtSrt, PerFrame, gHistory);
    p[15].ppTextures = &gAccum[w ^ 1];
    p[16].mIndex = SRT_RES_IDX(RtSrt, PerFrame, gHistoryZ);
    p[16].ppTextures = &gAccumZ[w ^ 1];
    p[17].mIndex = SRT_RES_IDX(RtSrt, PerFrame, gTiles);
    p[17].ppBuffers = &gTiles;
    p[18].mIndex = SRT_RES_IDX(RtSrt, PerFrame, gOutHalf);
    p[18].ppTextures = &gHalf;
    p[19].mIndex = SRT_RES_IDX(RtSrt, PerFrame, gOutFull);
    p[19].ppTextures = &gTex;
    p[20].mIndex = SRT_RES_IDX(RtSrt, PerFrame, gOut);
    p[20].ppTextures = &gAccum[w];
    p[21].mIndex = SRT_RES_IDX(RtSrt, PerFrame, gOutZ);
    p[21].ppTextures = &gAccumZ[w];
    // The trace reads neither trace (the history stands in, in the state it is in), the resolve reads both.
    p[12].ppTextures = &gAccum[w ^ 1];
    p[13].ppTextures = &gAccum[w ^ 1];
    updateDescriptorSet(gRenderer, fi * 2, gSet, 24, p);
    p[12].ppTextures = &gHalf;
    p[13].ppTextures = &gTex;
    updateDescriptorSet(gRenderer, fi * 2 + 1, gSet, 24, p);

    // Every target the two passes write goes to UNORDERED_ACCESS at once; the tile flags stay there.
    BufferBarrier tb = { gTiles, gTilesState, RESOURCE_STATE_UNORDERED_ACCESS };
    TextureBarrier in[4] = { { gHalf, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_UNORDERED_ACCESS },
                             { gTex, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_UNORDERED_ACCESS },
                             { gAccum[w], RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_UNORDERED_ACCESS },
                             { gAccumZ[w], RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_UNORDERED_ACCESS } };
    cmdResourceBarrier(cmd, 1, &tb, 4, in, 0, nullptr);
    gTilesState = RESOURCE_STATE_UNORDERED_ACCESS;
    if (settings.probe_rays > 0)
    {
        // TEMPORARY (probe_rays): the main view's shadow rays, traced and thrown away, to measure them.
        cmdBindPipeline(cmd, gPipe[kProbe]);
        cmdBindDescriptorSet(cmd, 0, gPersistent);
        cmdBindDescriptorSet(cmd, fi * 2, gSet);
        cmdDispatch(cmd, (F->width + 7) / 8, (F->height + 7) / 8, 1);
    }
    cmdBindPipeline(cmd, gPipe[kTrace]); // a thread per 2 x 2 block
    cmdBindDescriptorSet(cmd, 0, gPersistent);
    cmdBindDescriptorSet(cmd, fi * 2, gSet);
    cmdDispatch(cmd, ((F->width + 1) / 2 + 7) / 8, ((F->height + 1) / 2 + 7) / 8, 1);
    TextureBarrier mid[2] = { { gHalf, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_SHADER_RESOURCE },
                              { gTex, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_SHADER_RESOURCE } };
    tb = { gTiles, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_UNORDERED_ACCESS };
    cmdResourceBarrier(cmd, 1, &tb, 2, mid, 0, nullptr);
    cmdBindPipeline(cmd, gPipe[kResolve]);
    cmdBindDescriptorSet(cmd, 0, gPersistent);
    cmdBindDescriptorSet(cmd, fi * 2 + 1, gSet);
    cmdDispatch(cmd, (F->width + 7) / 8, (F->height + 7) / 8, 1);
    TextureBarrier out[2] = { { gAccum[w], RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_SHADER_RESOURCE },
                              { gAccumZ[w], RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_SHADER_RESOURCE } };
    tb = { gTiles, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_UNORDERED_ACCESS };
    cmdResourceBarrier(cmd, 1, &tb, 2, out, 0, nullptr);
    if (gFreshFrames)
        gFreshFrames--;
    gResolves++;
    gAccumIndex = w;
    gAccumFrame = F->frameNumber;
    cmdEndDebugMarker(cmd);
}
} // namespace renderer_rt
