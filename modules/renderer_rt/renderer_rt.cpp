// Ray-traced reflections for the renderer core, where the GPU has inline ray queries (yea::raytracing()).
// Owns the scene's acceleration structures: one bottom-level structure per mesh (built on the frame's command
// list when the mesh appears or changes) and one top-level structure per frame in flight, its instances
// rewritten (setInstances) and rebuilt each frame. After the depth prepass, Shaders/Trace.comp.fsl runs as:
// the trace (mirror-like pixels at full resolution, the rest one pixel in four with half_resolution), then the
// reconstruction (with half_resolution: each pixel reuses its nearest half-resolution rays, weighted by its own
// glossy lobe), then the resolve, one full-size pass that accumulates the reflections over frames (two histories,
// alternating), then the filter, which averages each glossy pixel's accumulated reflection over its gloss footprint
// into the texture the core's forward pass reads (the trace's own, free by then).
#include "yea_renderer_rt.h"
#include <yea/raytracing.h>
#include "frame.h"

#include <Common_3/Graphics/Interfaces/IGraphics.h>
#include <Common_3/Graphics/Interfaces/IRay.h>
#include <Common_3/Resources/ResourceLoader/Interfaces/IResourceLoader.h>
#include <Common_3/Graphics/FSL/defaults.h>
#include "Shaders/Trace.srt.h"
#include "Shaders/Shadows.srt.h"

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
    float    denoise[4];
    float    taps[32][4];
    float    sky[4]; // x 1: renderer_sky's physical sky, y its image's mips
    uint32_t probe[4]; // TEMPORARY (probe_rays)
    float    fogMedium[4], fogAlbedo[4]; // frame.h's height fog
};

// Shaders/ShadowData.h.fsl
struct ShadowData
{
    float    right[4], up[4], forward[4], camPos[4], screen[4], sunDir[4], params[4];
    uint32_t misc[4];
    uint32_t slots[9][4];
};

struct Blas
{
    AccelerationStructure* as;
    Buffer*                positions;
    Buffer*                indices;
    uint32_t               firstIndex, indexCount, vertexOffset, vertexCount, generation;
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
enum { kTrace, kReconstruct, kResolve, kProbe, kFilter, kShaders };
Shader*              gShader[kShaders];
Pipeline*            gPipe[kShaders];
DescriptorSet*       gSet; // four per frame in flight: the trace, the reconstruction, the resolve and the filter
Buffer*              gCb[kMaxFrames];
// All SHADER_RESOURCE between frames.
Texture*             gTex;      // this frame's trace of the pixels traced at full resolution (only those written), then
                                // the filter's output, rtReflection
Texture*             gHalf;     // the half-resolution trace (a quarter of the pixels)
Texture*             gHalfRay;  // its rays' directions, for their reuse
Texture*             gAccum[2]; // accumulated reflections: this frame's is written, last frame's read
Texture*             gAccumZ[2]; // RG16F: the view depth each accumulated pixel was for, and the samples it holds
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
Sampler*             gShadowCmp, *gSkySampler;
Texture*             gNoSky;   // 2 x 2 stand-in for renderer_sky's image
Buffer*              gNoSkySh; // and its SH
RenderTarget*        gDummyAtlas;  // as the core's gDummyDepth: a depth render target, 2 x 2
DescriptorSet*       gPersistent;  // the sampler's set
Buffer*              gDummyViews;

// Ray-traced shadows (Shaders/Shadows.comp.fsl): the sun's and up to RENDER_RT_MAX_TRACED lights', each a layer of
// the arrays below (in this frame's pick order), and of rtShadows (0 the sun, 1 + n traced light n).
const uint32_t kShadowLayers = 1 + RENDER_RT_MAX_TRACED;
const uint32_t kNoLight = 0xFFFFFFFFu, kSunId = 0xFFFFFFFEu;
enum { kShadowTrace, kShadowTemporal, kShadowFilter, kShadowShaders };
Shader*        gShShader[kShadowShaders];
Pipeline*      gShPipe[kShadowShaders];
DescriptorSet* gShSet; // three per frame in flight, a pass each
Buffer*        gShCb[kMaxFrames];
Buffer*        gNoShapes; // bound while the core has no shape buffer (every light kind 0)
// All SHADER_RESOURCE between frames.
Texture*       gShRaw;      // RG16F array: this frame's rays (lit, blocker distance)
Texture*       gShHist[2];  // RGBA16F arrays: accumulated (visibility, blocker distance, samples); alternating
Texture*       gShZ[2];     // R32F: the view depth they were for
Texture*       gRtShadows;  // R16F array: rtShadows
uint32_t       gShWidth, gShHeight, gShLayers; // the arrays' size: layers the sun and traced_lights
uint32_t       gShIndex;    // the history written last
uint64_t       gShFrame;    // the frame it was written (0: none)
uint32_t       gShIds[kShadowLayers];  // the light (id; kSunId the sun) whose history each layer of it holds
uint32_t       gShIdCount;
// This frame's pick (rt_pick): per shadow, its light in the GPU light buffer (kNoLight the sun) and its id.
uint32_t       gPickLight[kShadowLayers], gPickId[kShadowLayers], gPickCount;
uint64_t       gPickFrame;

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
    Texture** all[7] = { &gTex, &gHalf, &gHalfRay, &gAccum[0], &gAccum[1], &gAccumZ[0], &gAccumZ[1] };
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

Texture* makeArray(uint32_t w, uint32_t h, uint32_t layers, TinyImageFormat format, const char* name)
{
    Texture*    t = nullptr;
    TextureDesc d = {};
    d.mWidth = w;
    d.mHeight = h;
    d.mDepth = d.mMipLevels = 1;
    d.mArraySize = layers;
    d.mSampleCount = SAMPLE_COUNT_1;
    d.mFormat = format;
    d.mStartState = RESOURCE_STATE_SHADER_RESOURCE;
    d.mDescriptors = (DescriptorType)(DESCRIPTOR_TYPE_TEXTURE | DESCRIPTOR_TYPE_RW_TEXTURE);
    d.pName = name;
    TextureLoadDesc l = {};
    l.pDesc = &d;
    l.ppTexture = &t;
    addResource(&l, nullptr);
    waitForAllResourceLoads();
    return t;
}

void freeShadowTextures()
{
    Texture** all[6] = { &gShRaw, &gShHist[0], &gShHist[1], &gShZ[0], &gShZ[1], &gRtShadows };
    for (Texture** t : all)
    {
        if (*t)
            removeResource(*t);
        *t = nullptr;
    }
    gShWidth = gShHeight = gShLayers = 0;
    gShFrame = 0;
    gShIdCount = 0;
}

// Formats as the shaders' views: RWTex2DArray(half2) is rg16f on Vulkan, half4 rgba16f, half r16f; RWTex2D(float) r32f.
void createShadowTextures(uint32_t w, uint32_t h, uint32_t layers)
{
    gShRaw = makeArray(w, h, layers, TinyImageFormat_R16G16_SFLOAT, "renderer_rt_shadow_rays");
    gShHist[0] = makeArray(w, h, layers, TinyImageFormat_R16G16B16A16_SFLOAT, "renderer_rt_shadow_history");
    gShHist[1] = makeArray(w, h, layers, TinyImageFormat_R16G16B16A16_SFLOAT, "renderer_rt_shadow_history");
    gShZ[0] = makeTexture(w, h, "renderer_rt_shadow_z", TinyImageFormat_R32_SFLOAT);
    gShZ[1] = makeTexture(w, h, "renderer_rt_shadow_z", TinyImageFormat_R32_SFLOAT);
    gRtShadows = makeArray(w, h, layers, TinyImageFormat_R16_SFLOAT, "renderer_rt_shadows");
    if (!gShRaw || !gShHist[0] || !gShHist[1] || !gShZ[0] || !gShZ[1] || !gRtShadows)
    {
        freeShadowTextures();
        return;
    }
    gShWidth = w;
    gShHeight = h;
    gShLayers = layers;
}

void createTexture(uint32_t w, uint32_t h)
{
    gTex = makeTexture(w, h, "renderer_rt_reflection");
    gHalf = makeTexture((w + 1) / 2, (h + 1) / 2, "renderer_rt_half");
    gHalfRay = makeTexture((w + 1) / 2, (h + 1) / 2, "renderer_rt_half_ray");
    for (uint32_t i = 0; i < 2; ++i)
    {
        gAccum[i] = makeTexture(w, h, "renderer_rt_accum");
        gAccumZ[i] = makeTexture(w, h, "renderer_rt_accum_z", TinyImageFormat_R16G16_SFLOAT);
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
    if (!gTex || !gHalf || !gHalfRay || !gAccum[0] || !gAccum[1] || !gAccumZ[0] || !gAccumZ[1] || !gTiles)
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

// The mesh's bottom-level structure, made and built (on cmd) when it is new or its geometry moved or was rewritten (the
// caster's generation: buffers are reused in place, so the same addresses can hold new data).
AccelerationStructure* meshBlas(Cmd* cmd, const RenderFrame* F, const RenderCaster& c)
{
    if (c.mesh >= kMaxMeshes || !c.indexCount || !c.vertexCount)
        return nullptr;
    Blas& b = gBlas[c.mesh];
    if (b.as && b.positions == F->positions && b.indices == F->indices && b.generation == c.generation && b.firstIndex == c.firstIndex &&
        b.indexCount == c.indexCount && b.vertexOffset == (uint32_t)c.vertexOffset && b.vertexCount == c.vertexCount)
        return b.as;
    retire(b.as, F->frameNumber);
    b = Blas{};
    AccelerationStructureGeometryDesc g = {};
    g.pVertexBuffer = F->positions;
    g.pIndexBuffer = F->indices;
    g.mVertexOffset = (uint32_t)c.vertexOffset * RENDER_POSITION_STRIDE; // bytes
    g.mVertexCount = c.vertexCount;
    g.mVertexStride = RENDER_POSITION_STRIDE;
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
    b.positions = F->positions;
    b.indices = F->indices;
    b.generation = c.generation;
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
    const char* names[kShaders] = { "renderer_rt_trace.comp", "renderer_rt_reconstruct.comp", "renderer_rt_resolve.comp", "renderer_rt_probe.comp",
                                    "renderer_rt_filter.comp" };
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
    const char* shadowNames[kShadowShaders] = { "renderer_rt_shadow_trace.comp", "renderer_rt_shadow_temporal.comp",
                                                "renderer_rt_shadow_filter.comp" };
    for (uint32_t k = 0; k < kShadowShaders; ++k)
    {
        ShaderLoadDesc sd = {};
        sd.mComp.pFileName = shadowNames[k];
        addShader(gRenderer, &sd, &gShShader[k]);
        if (!gShShader[k])
            return 1;
        PipelineDesc desc = {};
        desc.mType = PIPELINE_TYPE_COMPUTE;
        desc.mComputeDesc.pShaderProgram = gShShader[k];
        desc.pName = "renderer_rt_shadows";
        PIPELINE_LAYOUT_DESC(desc, NULL, SRT_LAYOUT_DESC(ShadowSrt, PerFrame), NULL, NULL);
        addPipeline(gRenderer, &desc, &gShPipe[k]);
        if (!gShPipe[k])
            return 1;
    }
    SamplerDesc sd = {};
    sd.mMinFilter = FILTER_LINEAR;
    sd.mMagFilter = FILTER_LINEAR;
    sd.mMipMapMode = MIPMAP_MODE_NEAREST;
    sd.mAddressU = sd.mAddressV = sd.mAddressW = ADDRESS_MODE_CLAMP_TO_EDGE;
    sd.mCompareFunc = CMP_LEQUAL;
    addSampler(gRenderer, &sd, &gShadowCmp);
    sd.mMipMapMode = MIPMAP_MODE_LINEAR;
    sd.mCompareFunc = CMP_NEVER;
    addSampler(gRenderer, &sd, &gSkySampler);
    {
        TextureDesc t = {};
        t.mWidth = t.mHeight = 2; // one texel high gets a 1D view on Vulkan
        t.mDepth = t.mArraySize = t.mMipLevels = 1;
        t.mSampleCount = SAMPLE_COUNT_1;
        t.mFormat = TinyImageFormat_R16G16B16A16_SFLOAT;
        t.mStartState = RESOURCE_STATE_SHADER_RESOURCE;
        t.mDescriptors = DESCRIPTOR_TYPE_TEXTURE;
        t.pName = "renderer_rt_no_sky";
        TextureLoadDesc tl = {};
        tl.pDesc = &t;
        tl.ppTexture = &gNoSky;
        addResource(&tl, nullptr);
        waitForAllResourceLoads();
    }
    gNoSkySh = makeMapped(9 * 16, 16, DESCRIPTOR_TYPE_BUFFER, "renderer_rt_no_sky_sh");
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
    if (!gShadowCmp || !gSkySampler || !gNoSky || !gNoSkySh || !gDummyAtlas || !gDummyAtlas->pTexture || !gDummyViews || !gPersistent)
    {
        unload();
        return 1;
    }
    {
        DescriptorData s[2] = {};
        s[0].mIndex = SRT_RES_IDX(RtSrt, Persistent, gShadowCmp);
        s[0].ppSamplers = &gShadowCmp;
        s[1].mIndex = SRT_RES_IDX(RtSrt, Persistent, gSkySampler);
        s[1].ppSamplers = &gSkySampler;
        updateDescriptorSet(gRenderer, 0, gPersistent, 2, s);
    }
    DescriptorSetDesc setDesc = SRT_SET_DESC(RtSrt, PerFrame, gFrames * 4, 0);
    addDescriptorSet(gRenderer, &setDesc, &gSet);
    for (uint32_t i = 0; i < gFrames; ++i)
        gCb[i] = makeMapped(sizeof(RtData) < 256 ? 256 : (sizeof(RtData) + 255) / 256 * 256, 0, DESCRIPTOR_TYPE_UNIFORM_BUFFER, "renderer_rt_cb");
    DescriptorSetDesc shadowSet = SRT_SET_DESC(ShadowSrt, PerFrame, gFrames * 3, 0);
    addDescriptorSet(gRenderer, &shadowSet, &gShSet);
    for (uint32_t i = 0; i < gFrames; ++i)
        gShCb[i] = makeMapped(sizeof(ShadowData) < 256 ? 256 : (sizeof(ShadowData) + 255) / 256 * 256, 0, DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                              "renderer_rt_shadow_cb");
    gNoShapes = makeMapped(sizeof(RenderGpuLightShape) * 4, sizeof(RenderGpuLightShape), DESCRIPTOR_TYPE_BUFFER, "renderer_rt_no_shapes");
    if (gNoShapes)
        memset(gNoShapes->pCpuMappedAddress, 0, sizeof(RenderGpuLightShape) * 4);
    gReady = gSet != nullptr && gShSet != nullptr && gNoShapes != nullptr;
    yea::log(gReady ? "renderer_rt: ray-traced reflections on" : "renderer_rt: the trace pipeline failed");
    return gReady ? 0 : 1;
}

void unload()
{
    if (!gRenderer || !gRay)
        return;
    waitQueueIdle(gQueue);
    freeTexture();
    freeShadowTextures();
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
        if (gShCb[i])
            removeResource(gShCb[i]);
        gCb[i] = gShCb[i] = nullptr;
    }
    if (gShSet)
        removeDescriptorSet(gRenderer, gShSet);
    if (gNoShapes)
        removeResource(gNoShapes);
    gShSet = nullptr;
    gNoShapes = nullptr;
    for (uint32_t k = 0; k < kShadowShaders; ++k)
    {
        if (gShPipe[k])
            removePipeline(gRenderer, gShPipe[k]);
        if (gShShader[k])
            removeShader(gRenderer, gShShader[k]);
        gShPipe[k] = nullptr;
        gShShader[k] = nullptr;
    }
    if (gSet)
        removeDescriptorSet(gRenderer, gSet);
    if (gShadowCmp)
        removeSampler(gRenderer, gShadowCmp);
    if (gSkySampler)
        removeSampler(gRenderer, gSkySampler);
    if (gNoSky)
        removeResource(gNoSky);
    if (gNoSkySh)
        removeResource(gNoSkySh);
    gSkySampler = nullptr;
    gNoSky = nullptr;
    gNoSkySh = nullptr;
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

// Which shadows rays trace this frame: the sun's, then up to traced_lights visible local lights that cast, area lights
// first (their soft shadows cost the shadow maps most, and rays give their real shape), then in the core's order.
// renderer_shadows then draws no cascades for a traced sun and no tiles for a traced light.
bool rt_pick(uint64_t address)
{
    RenderFrame* F = (RenderFrame*)(uintptr_t)address;
    gPickCount = 0;
    if (gReady && !settings.shadows && gShLayers)
    {
        waitQueueIdle(gQueue); // switched off: their memory back (frames in flight may still read them)
        freeShadowTextures();
    }
    if (!gReady || !settings.shadows || !F || F->version != RENDER_FRAME_VERSION || !F->width || !F->height || !F->taaOn ||
        yea::viewMode())
        return false;
    uint32_t most = settings.traced_lights < 0 ? 0 : (uint32_t)settings.traced_lights;
    most = most > RENDER_RT_MAX_TRACED ? RENDER_RT_MAX_TRACED : most;
    if (F->width != gShWidth || F->height != gShHeight || 1 + most != gShLayers)
    {
        waitQueueIdle(gQueue);
        freeShadowTextures();
        createShadowTextures(F->width, F->height, 1 + most);
    }
    if (!gRtShadows)
        return false;
    F->rtSunShadows = 0;
    F->rtLightCount = 0;
    for (uint32_t i = 0; i < F->lightCount; ++i)
        F->lights[i].traced = -1;
    if (F->sunOn && F->sunShadow)
    {
        F->rtSunShadows = 1;
        gPickLight[0] = kNoLight;
        gPickId[0] = kSunId;
        gPickCount = 1;
    }
    for (uint32_t pass = 0; pass < 2; ++pass)
        for (uint32_t i = 0; i < F->lightCount && F->rtLightCount < most; ++i)
        {
            RenderLight& l = F->lights[i];
            if (!l.castShadow || l.traced >= 0 || (pass == 0) != (l.kind >= 3))
                continue;
            l.traced = (int32_t)F->rtLightCount;
            F->rtLightSlots[F->rtLightCount++] = i;
            gPickLight[gPickCount] = i; // the visible lights lead the GPU buffer in this order
            gPickId[gPickCount++] = l.id;
        }
    gPickFrame = F->frameNumber;
    F->rtShadows = gRtShadows;
    return gPickCount > 0;
}

bool rt_prepare(uint64_t address)
{
    RenderFrame* F = (RenderFrame*)(uintptr_t)address;
    bool shadows = gPickCount && F && gPickFrame == F->frameNumber;
    if (!gReady || (!settings.enabled && !shadows) || !F || F->version != RENDER_FRAME_VERSION || !F->width || !F->height || !F->taaOn)
        return false;
    if (yea::viewMode() || !F->surface || !F->previous || !F->drawableCount || !F->positions || !F->attributes || !F->indices || !F->instances ||
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
    d.sky[0] = F->skyEnv && F->skySh ? 1.0f : 0.0f;
    d.sky[1] = (float)F->skyEnvMips;
    memcpy(d.fogMedium, F->fogMedium, sizeof d.fogMedium);
    memcpy(d.fogAlbedo, F->fogAlbedo, sizeof d.fogAlbedo);
    d.counts[0] = F->allLightCount;
    d.counts[1] = (uint32_t)F->frameNumber;
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
    d.denoise[0] = 1.0f;
    d.denoise[1] = 1.5f;
    d.denoise[2] = settings.full_below < 0 ? 0.0f : settings.full_below;
    memcpy(gCb[F->frameIndex]->pCpuMappedAddress, &d, sizeof d);

    F->rtReflection = settings.enabled ? gTex : nullptr; // the filter writes it after the resolve; none: shadows only
    F->rtMaxRoughness = maxRough;
    F->rtDebug = settings.show_hits;
    return true;
}

// The traced shadows into rtShadows (rt_run, after the top-level structure's build): rays, accumulation, filter.
void traceShadows(RenderFrame* F, Cmd* cmd, uint32_t fi)
{
    uint32_t   w = gShIndex ^ 1;
    ShadowData d = {};
    const float(*v)[4] = F->view;
    for (int k = 0; k < 3; ++k)
    {
        d.right[k] = v[0][k];
        d.up[k] = v[1][k];
        d.forward[k] = v[2][k];
        d.camPos[k] = F->camPos[k];
        d.sunDir[k] = F->sunDir[k];
    }
    d.right[3] = F->p00;
    d.up[3] = F->p11;
    d.forward[3] = F->nearZ;
    d.camPos[3] = F->rtLightRadius > 0 ? F->rtLightRadius : 0.1f;
    d.screen[0] = (float)F->width;
    d.screen[1] = (float)F->height;
    d.screen[2] = F->jitter[0];
    d.screen[3] = F->jitter[1];
    d.sunDir[3] = tanf(0.5f * (F->rtSunAngle > 0 ? F->rtSunAngle : 0.0093f)); // 0.53 degrees: the real sun
    d.params[0] = 32.0f; // frames of history at most
    d.params[1] = 0.5f;  // filter radius: half the penumbra
    d.params[2] = 8.0f;  // at most (px)
    d.misc[0] = (uint32_t)F->frameNumber;
    d.misc[1] = gPickCount;
    d.misc[2] = gShFrame && gShFrame + 1 == F->frameNumber ? 0 : 1;
    // Each shadow's history: the layer that held the same light last frame, if any.
    for (uint32_t s = 0; s < gPickCount; ++s)
    {
        d.slots[s][0] = gPickLight[s] == kNoLight ? 0 : 1 + (s - (F->rtSunShadows ? 1 : 0));
        d.slots[s][1] = 0;
        d.slots[s][2] = gPickLight[s];
        d.slots[s][3] = 1;
        for (uint32_t j = 0; j < gShIdCount; ++j)
            if (gShIds[j] == gPickId[s])
                d.slots[s][1] = j, d.slots[s][3] = 0;
    }
    memcpy(gShCb[fi]->pCpuMappedAddress, &d, sizeof d);

    Buffer*  lights = F->lightBuffer;
    Buffer*  shapes = F->lightShapes ? F->lightShapes : gNoShapes;
    Texture* depth = F->depth->pTexture;
    Texture* surface = F->surface->pTexture;
    Texture* motion = F->motion->pTexture;
    DescriptorData p[14] = {};
    p[0].mIndex = SRT_RES_IDX(ShadowSrt, PerFrame, gScene);
    p[0].ppAccelerationStructures = &gTlas[fi];
    p[1].mIndex = SRT_RES_IDX(ShadowSrt, PerFrame, gSd);
    p[1].ppBuffers = &gShCb[fi];
    p[2].mIndex = SRT_RES_IDX(ShadowSrt, PerFrame, gDepth);
    p[2].ppTextures = &depth;
    p[3].mIndex = SRT_RES_IDX(ShadowSrt, PerFrame, gSurface);
    p[3].ppTextures = &surface;
    p[4].mIndex = SRT_RES_IDX(ShadowSrt, PerFrame, gMotion);
    p[4].ppTextures = &motion;
    p[5].mIndex = SRT_RES_IDX(ShadowSrt, PerFrame, gLights);
    p[5].ppBuffers = &lights;
    p[6].mIndex = SRT_RES_IDX(ShadowSrt, PerFrame, gShapes);
    p[6].ppBuffers = &shapes;
    p[7].mIndex = SRT_RES_IDX(ShadowSrt, PerFrame, gRawOut);
    p[7].ppTextures = &gShRaw;
    p[8].mIndex = SRT_RES_IDX(ShadowSrt, PerFrame, gRaw);
    p[8].ppTextures = &gShRaw;
    p[9].mIndex = SRT_RES_IDX(ShadowSrt, PerFrame, gHistory);
    p[9].ppTextures = &gShHist[w ^ 1];
    p[10].mIndex = SRT_RES_IDX(ShadowSrt, PerFrame, gHistoryOut);
    p[10].ppTextures = &gShHist[w];
    p[11].mIndex = SRT_RES_IDX(ShadowSrt, PerFrame, gPrevZ);
    p[11].ppTextures = &gShZ[w ^ 1];
    p[12].mIndex = SRT_RES_IDX(ShadowSrt, PerFrame, gZOut);
    p[12].ppTextures = &gShZ[w];
    p[13].mIndex = SRT_RES_IDX(ShadowSrt, PerFrame, gFinal);
    p[13].ppTextures = &gRtShadows;
    updateDescriptorSet(gRenderer, fi * 3, gShSet, 14, p);
    updateDescriptorSet(gRenderer, fi * 3 + 1, gShSet, 14, p);
    p[9].ppTextures = &gShHist[w]; // the filter reads this frame's accumulation
    updateDescriptorSet(gRenderer, fi * 3 + 2, gShSet, 14, p);

    uint32_t gx = (F->width + 7) / 8, gy = (F->height + 7) / 8;
    cmdBeginDebugMarker(cmd, 0.5f, 0.8f, 1.0f, "Renderer ray-traced shadows");
    TextureBarrier in[4] = { { gShRaw, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_UNORDERED_ACCESS },
                             { gShHist[w], RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_UNORDERED_ACCESS },
                             { gShZ[w], RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_UNORDERED_ACCESS },
                             { gRtShadows, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_UNORDERED_ACCESS } };
    cmdResourceBarrier(cmd, 0, nullptr, 4, in, 0, nullptr);
    cmdBindPipeline(cmd, gShPipe[kShadowTrace]);
    cmdBindDescriptorSet(cmd, fi * 3, gShSet);
    cmdDispatch(cmd, gx, gy, gPickCount);
    TextureBarrier rays[1] = { { gShRaw, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_SHADER_RESOURCE } };
    cmdResourceBarrier(cmd, 0, nullptr, 1, rays, 0, nullptr);
    cmdBindPipeline(cmd, gShPipe[kShadowTemporal]);
    cmdBindDescriptorSet(cmd, fi * 3 + 1, gShSet);
    cmdDispatch(cmd, gx, gy, gPickCount);
    TextureBarrier hist[2] = { { gShHist[w], RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_SHADER_RESOURCE },
                               { gShZ[w], RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_SHADER_RESOURCE } };
    cmdResourceBarrier(cmd, 0, nullptr, 2, hist, 0, nullptr);
    cmdBindPipeline(cmd, gShPipe[kShadowFilter]);
    cmdBindDescriptorSet(cmd, fi * 3 + 2, gShSet);
    cmdDispatch(cmd, gx, gy, gPickCount);
    TextureBarrier out[1] = { { gRtShadows, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_SHADER_RESOURCE } };
    cmdResourceBarrier(cmd, 0, nullptr, 1, out, 0, nullptr);
    cmdEndDebugMarker(cmd);
    gShIndex = w;
    gShFrame = F->frameNumber;
    for (uint32_t s = 0; s < gPickCount; ++s)
        gShIds[s] = gPickId[s];
    gShIdCount = gPickCount;
}

void rt_run(uint64_t address)
{
    RenderFrame* F = (RenderFrame*)(uintptr_t)address;
    uint32_t w = gAccumIndex ^ 1; // written this frame; gAccumIndex holds last frame's
    if (!gReady || !F || !gTex || (F->rtReflection && F->rtReflection != gTex) || !F->depth || !F->depth->pTexture || !F->surface ||
        !F->surface->pTexture || !F->motion || !F->motion->pTexture || !F->previous || !F->previous->pTexture || !F->instances ||
        !F->materials || !F->lightBuffer || !F->positions || !F->attributes || !F->indices || !F->exposure)
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
    if (gPickCount && gPickFrame == F->frameNumber && F->rtShadows == gRtShadows)
        traceShadows(F, cmd, fi);
    if (!F->rtReflection)
    {
        cmdEndDebugMarker(cmd);
        return;
    }

    Buffer*        instances = F->instances;
    Buffer*        materials = F->materials;
    Buffer*        lights = F->lightBuffer;
    Buffer*        attributes = F->attributes;
    Buffer*        indices = F->indices;
    Texture*       previous = F->previous->pTexture;
    Buffer*        exposure = F->exposure;
    Texture*       motion = F->motion->pTexture;
    // renderer_shadows has drawn its atlas by now (shadows_render runs before the prepass).
    bool           maps = F->localAtlas && F->localAtlas->pTexture && F->shadowViews;
    Texture*       atlas = maps ? F->localAtlas->pTexture : gDummyAtlas->pTexture;
    Buffer*        views = maps ? F->shadowViews : gDummyViews;
    DescriptorData p[28] = {};
    Texture*       skyEnv = F->skyEnv && F->skySh ? F->skyEnv : gNoSky;
    Buffer*        skySh = F->skyEnv && F->skySh ? F->skySh : gNoSkySh;
    p[26].mIndex = SRT_RES_IDX(RtSrt, PerFrame, gSkyEnv);
    p[26].ppTextures = &skyEnv;
    p[27].mIndex = SRT_RES_IDX(RtSrt, PerFrame, gSkySh);
    p[27].ppBuffers = &skySh;
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
    p[9].mIndex = SRT_RES_IDX(RtSrt, PerFrame, gAttributes);
    p[9].ppBuffers = &attributes;
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
    p[24].mIndex = SRT_RES_IDX(RtSrt, PerFrame, gHalfRayIn);
    p[25].mIndex = SRT_RES_IDX(RtSrt, PerFrame, gOutHalfRay);
    p[25].ppTextures = &gHalfRay;
    // The trace reads no trace (the history stands in, in the state it is in), the reconstruction the half-resolution
    // ones, the resolve all of them.
    p[12].ppTextures = &gAccum[w ^ 1];
    p[13].ppTextures = &gAccum[w ^ 1];
    p[24].ppTextures = &gAccum[w ^ 1];
    updateDescriptorSet(gRenderer, fi * 4, gSet, 28, p);
    p[12].ppTextures = &gHalf;
    p[24].ppTextures = &gHalfRay;
    updateDescriptorSet(gRenderer, fi * 4 + 1, gSet, 28, p);
    p[13].ppTextures = &gTex;
    updateDescriptorSet(gRenderer, fi * 4 + 2, gSet, 28, p);
    // The filter reads this frame's accumulation and writes the trace texture, free by then, as rtReflection.
    p[13].ppTextures = &gAccum[w ^ 1];
    p[15].ppTextures = &gAccum[w];
    p[16].ppTextures = &gAccumZ[w];
    p[20].ppTextures = &gTex;
    p[21].ppTextures = &gAccumZ[w ^ 1];
    updateDescriptorSet(gRenderer, fi * 4 + 3, gSet, 28, p);

    // Every target the passes write goes to UNORDERED_ACCESS at once; the tile flags stay there.
    BufferBarrier tb = { gTiles, gTilesState, RESOURCE_STATE_UNORDERED_ACCESS };
    TextureBarrier in[5] = { { gHalf, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_UNORDERED_ACCESS },
                             { gHalfRay, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_UNORDERED_ACCESS },
                             { gTex, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_UNORDERED_ACCESS },
                             { gAccum[w], RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_UNORDERED_ACCESS },
                             { gAccumZ[w], RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_UNORDERED_ACCESS } };
    cmdResourceBarrier(cmd, 1, &tb, 5, in, 0, nullptr);
    gTilesState = RESOURCE_STATE_UNORDERED_ACCESS;
    if (settings.probe_rays > 0)
    {
        // TEMPORARY (probe_rays): the main view's shadow rays, traced and thrown away, to measure them.
        cmdBindPipeline(cmd, gPipe[kProbe]);
        cmdBindDescriptorSet(cmd, 0, gPersistent);
        cmdBindDescriptorSet(cmd, fi * 4, gSet);
        cmdDispatch(cmd, (F->width + 7) / 8, (F->height + 7) / 8, 1);
    }
    cmdBindPipeline(cmd, gPipe[kTrace]); // a thread per 2 x 2 block
    cmdBindDescriptorSet(cmd, 0, gPersistent);
    cmdBindDescriptorSet(cmd, fi * 4, gSet);
    cmdDispatch(cmd, ((F->width + 1) / 2 + 7) / 8, ((F->height + 1) / 2 + 7) / 8, 1);
    tb = { gTiles, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_UNORDERED_ACCESS };
    if (settings.half_resolution)
    {
        // The half-resolution pixels from the rays around them, beside the full-resolution traces.
        TextureBarrier rec[3] = { { gHalf, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_SHADER_RESOURCE },
                                  { gHalfRay, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_SHADER_RESOURCE },
                                  { gTex, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_UNORDERED_ACCESS } };
        cmdResourceBarrier(cmd, 1, &tb, 3, rec, 0, nullptr);
        cmdBindPipeline(cmd, gPipe[kReconstruct]);
        cmdBindDescriptorSet(cmd, 0, gPersistent);
        cmdBindDescriptorSet(cmd, fi * 4 + 1, gSet);
        cmdDispatch(cmd, (F->width + 7) / 8, (F->height + 7) / 8, 1);
        TextureBarrier mid[1] = { { gTex, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_SHADER_RESOURCE } };
        cmdResourceBarrier(cmd, 1, &tb, 1, mid, 0, nullptr);
    }
    else
    {
        TextureBarrier mid[3] = { { gHalf, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_SHADER_RESOURCE },
                                  { gHalfRay, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_SHADER_RESOURCE },
                                  { gTex, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_SHADER_RESOURCE } };
        cmdResourceBarrier(cmd, 1, &tb, 3, mid, 0, nullptr);
    }
    cmdBindPipeline(cmd, gPipe[kResolve]);
    cmdBindDescriptorSet(cmd, 0, gPersistent);
    cmdBindDescriptorSet(cmd, fi * 4 + 2, gSet);
    cmdDispatch(cmd, (F->width + 7) / 8, (F->height + 7) / 8, 1);
    TextureBarrier out[2] = { { gAccum[w], RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_SHADER_RESOURCE },
                              { gAccumZ[w], RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_SHADER_RESOURCE } };
    tb = { gTiles, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_UNORDERED_ACCESS };
    cmdResourceBarrier(cmd, 1, &tb, 2, out, 0, nullptr);
    TextureBarrier filter[1] = { { gTex, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_UNORDERED_ACCESS } };
    cmdResourceBarrier(cmd, 0, nullptr, 1, filter, 0, nullptr);
    cmdBindPipeline(cmd, gPipe[kFilter]);
    cmdBindDescriptorSet(cmd, 0, gPersistent);
    cmdBindDescriptorSet(cmd, fi * 4 + 3, gSet);
    cmdDispatch(cmd, (F->width + 7) / 8, (F->height + 7) / 8, 1);
    filter[0] = { gTex, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_SHADER_RESOURCE };
    cmdResourceBarrier(cmd, 0, nullptr, 1, filter, 0, nullptr);
    if (gFreshFrames)
        gFreshFrames--;
    gResolves++;
    gAccumIndex = w;
    gAccumFrame = F->frameNumber;
    cmdEndDebugMarker(cmd);
}
} // namespace renderer_rt
