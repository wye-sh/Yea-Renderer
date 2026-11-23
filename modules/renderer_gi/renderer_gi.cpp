// Dynamic diffuse global illumination for the renderer core (Majercik et al., "Dynamic Diffuse Global Illumination with
// Ray-Traced Irradiance Fields", JCGT 2019; Majercik et al. 2021's relocation and classification; as in NVIDIA's RTXGI):
// a grid of probes over the scene, each an octahedral tile of irradiance and one of distance. renderer_rt traces this
// frame's rays from every active probe (its acceleration structures and hit shading: direct light plus last frame's
// probes there, so bounces add up over frames without end); this module blends them in with hysteresis, moves probes
// out of walls and switches off those that light no one. Where nothing traced them (no ray tracing, TAA off,
// renderer_rt switched off), it captures a few probes a frame itself instead: six small faces each, rasterized and lit
// as renderer_rt lights its hits, read back as the same rays, so everything after is the same. The core reads the
// probes for diffuse ambient; fog, particles and reflection hits read them too (Shaders/GiSample.h.fsl).
#include "yea_renderer_gi.h"
#include <yea/raytracing.h>
#include "frame.h"

#include <Common_3/Graphics/Interfaces/IGraphics.h>
#include <Common_3/Resources/ResourceLoader/Interfaces/IResourceLoader.h>
#include <Common_3/Graphics/FSL/defaults.h>
#include "Shaders/Gi.srt.h"
#define MAX_TEXTURES RENDER_MAX_TEXTURES // Shaders/GiData.h.fsl's, for the texture array's size
#include "Shaders/Capture.srt.h"

#include <algorithm>
#include <math.h>
#include <string.h>

namespace renderer_gi
{
namespace
{
struct GiData
{
    float    grid[4][4];
    float    rotation[3][4];
    float    params[4];
    uint32_t counts[4];
    uint32_t raster[4];
    float    sunDir[4], sunColor[4], skyZenith[4], skyHorizon[4], skyGround[4], fogMedium[4], fogAlbedo[4];
    float    shadow[4], sunView[3][4];
    uint32_t decals[4];
    float    decalGrid[2][4];
};
struct Run
{
    uint32_t firstIndex, indexCount;
    int32_t  vertexOffset;
    uint32_t streamStart, count;
};

const uint32_t kMaxFrames = 4, kMaxX = 16, kMaxY = 8, kMaxZ = 16, kMaxRays = 256;
const uint32_t kIrrTexels = 8, kDistTexels = 16;
// The raster fallback: at most kMaxSlots probes a frame, six kFace-texel faces each (Shaders/GiData.h.fsl GI_FACE_TEXELS).
const uint32_t kMaxSlots = 16, kFace = 16, kMaxEntries = 1u << 17, kMaxRuns = 8192, kMaxCaptureLights = 64;
// The raster fallback's own sun shadow map: every shadow caster, over the probe volume and the whole scene round it.
const uint32_t kSunSize = 2048, kMaxSunEntries = 1u << 16;
enum Pass { kInit, kIrradiance, kDistance, kRays, kClassify, kPasses };
const char* const kShaderNames[kPasses] = { "renderer_gi_init.comp", "renderer_gi_irradiance.comp", "renderer_gi_distance.comp",
                                            "renderer_gi_rays.comp", "renderer_gi_classify.comp" };
// Cube faces +X -X +Y -Y +Z -Z: forward, right, up (Shaders/GiData.h.fsl giFaceAxis).
const float kFaceAxes[6][3][3] = {
    { { 1, 0, 0 }, { 0, 0, -1 }, { 0, 1, 0 } },  { { -1, 0, 0 }, { 0, 0, 1 }, { 0, 1, 0 } },
    { { 0, 1, 0 }, { 1, 0, 0 }, { 0, 0, -1 } },  { { 0, -1, 0 }, { 1, 0, 0 }, { 0, 0, 1 } },
    { { 0, 0, 1 }, { 1, 0, 0 }, { 0, 1, 0 } },   { { 0, 0, -1 }, { -1, 0, 0 }, { 0, 1, 0 } },
};

Renderer*      gRenderer;
Queue*         gQueue;
uint32_t       gFrames;
Shader*        gShader[kPasses];
Pipeline*      gPipe[kPasses];
DescriptorSet* gSet;
Buffer*        gCb[kMaxFrames];
Buffer*        gProbes, *gRays;
Texture*       gIrr, *gDist;
uint32_t       gCount[3];
float          gOrigin[3], gSpacing[3];
bool           gReady, gPendingInit, gValid;

// The raster fallback's objects.
Shader*        gCaptureShader;
Pipeline*      gCapturePipe;
DescriptorSet *gCapturePersistent, *gCaptureSet;
Sampler*       gShadowCmp;
RenderTarget * gFaces, *gFaceDepth, *gNoAtlas;
Buffer *       gStream[kMaxFrames], *gNoViews, *gNoSh;
Buffer *       gNoDecals, *gNoCells, *gNoItems; // stand in for the frame's decals while there are none
uint32_t*      gStreamCpu;
Run*           gRuns;
uint32_t       gRunStart[kMaxSlots * 6 + 1];
uint32_t       gRasterNext;
bool           gCaptureReady;
uint32_t       gTexBound[kMaxFrames * 2]; // the core's textureVersion each capture set holds (0: none yet)

// The sun's map, in two: gSun holds the still casters, drawn only when the sun, the still casters, the probe box, the view
// or the textures change; gSunDyn the moving ones (skinned, or moved twice within kMovingFrames, so appearing or one
// jump costs one redraw of the still map, as before), drawn every raster frame they are there. Both share one view, so
// a texel's depth is the lesser of the two, as one map of every caster would hold: which map a caster is in changes
// only the cost, never the light.
const uint64_t kMovingFrames = 60;
const uint32_t kMoveRows = 65536; // the core's Instance table
uint64_t *     gLastMove, *gPrevMove; // per instance row: the last two `moved` values seen
Shader*        gSunShader;
Pipeline*      gSunPipe;
RenderTarget * gSun, *gSunDyn;
Buffer *       gSunStream[kMaxFrames], *gSunDynStream[kMaxFrames];
uint32_t *     gSunStreamCpu, *gSunDynStreamCpu;
Run *          gSunRuns, *gSunDynRuns;
uint32_t       gSunRunCount, gSunDynRunCount;
float          gSunView[3][4];
float          gSunDir[3], gSunOrigin[3], gSunSpacing[3];
float          gSunStill[6], gSunBox[6]; // the box of the probes and the still casters; the box the view was fitted to
uint64_t       gSunDrawn;     // the frame the still map was drawn (0: never)
uint64_t       gSunSig;       // the still casters: their count and a sum of hashes of their instances
uint32_t       gSunTextures;
bool           gSunValid, gSunDynOn; // gSunDynOn: gSunDyn holds this frame's moving casters

// This frame's plan, from gi_prepare to gi_run.
bool     gOn, gInitNow;
uint32_t gProbeCount, gRayCount;

float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

void freeGrid()
{
    Buffer** bufs[] = { &gProbes, &gRays };
    for (Buffer** b : bufs)
    {
        if (*b)
            removeResource(*b);
        *b = nullptr;
    }
    Texture** tex[] = { &gIrr, &gDist };
    for (Texture** t : tex)
    {
        if (*t)
            removeResource(*t);
        *t = nullptr;
    }
    gCount[0] = gCount[1] = gCount[2] = 0;
    gValid = false;
}

Texture* makeAtlas(uint32_t w, uint32_t h, const char* name)
{
    TextureDesc d = {};
    d.mWidth = w;
    d.mHeight = h;
    d.mDepth = d.mArraySize = d.mMipLevels = 1;
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

Buffer* makeBuffer(uint64_t size, uint32_t stride, uint32_t descriptors, ResourceMemoryUsage usage, ResourceState start, const char* name)
{
    Buffer*        b = nullptr;
    BufferLoadDesc d = {};
    d.mDesc.mDescriptors = (DescriptorType)descriptors;
    d.mDesc.mMemoryUsage = usage;
    d.mDesc.mSize = size;
    d.mDesc.mStartState = start;
    if (usage != RESOURCE_MEMORY_USAGE_GPU_ONLY)
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

// The probes for a grid: remade (after the GPU is done with the old ones), and started again, only when it changes.
bool ensureGrid(const uint32_t n[3], const float origin[3], const float spacing[3])
{
    bool same = gProbes && n[0] == gCount[0] && n[1] == gCount[1] && n[2] == gCount[2];
    for (int k = 0; k < 3 && same; ++k)
        same = fabsf(origin[k] - gOrigin[k]) < 1e-4f && fabsf(spacing[k] - gSpacing[k]) < 1e-4f;
    if (same)
        return true;
    if (gProbes && n[0] == gCount[0] && n[1] == gCount[1] && n[2] == gCount[2])
    {
        // The same counts in another place: the same resources, started again.
        memcpy(gOrigin, origin, sizeof gOrigin);
        memcpy(gSpacing, spacing, sizeof gSpacing);
        gPendingInit = true;
        gValid = false;
        return true;
    }
    if (gProbes)
        waitQueueIdle(gQueue);
    freeGrid();
    const uint32_t probes = n[0] * n[1] * n[2], row = n[0] * n[2];
    const uint32_t rw = DESCRIPTOR_TYPE_BUFFER | DESCRIPTOR_TYPE_RW_BUFFER;
    gProbes = makeBuffer((uint64_t)probes * 16, 16, rw, RESOURCE_MEMORY_USAGE_GPU_ONLY, RESOURCE_STATE_SHADER_RESOURCE, "renderer_gi_probes");
    gRays = makeBuffer((uint64_t)probes * kMaxRays * 16, 16, rw, RESOURCE_MEMORY_USAGE_GPU_ONLY, RESOURCE_STATE_SHADER_RESOURCE, "renderer_gi_rays");
    gIrr = makeAtlas(row * kIrrTexels, n[1] * kIrrTexels, "renderer_gi_irradiance");
    gDist = makeAtlas(row * kDistTexels, n[1] * kDistTexels, "renderer_gi_distance");
    waitForAllResourceLoads();
    if (!gProbes || !gRays || !gIrr || !gDist)
    {
        freeGrid();
        return false;
    }
    memcpy(gCount, n, sizeof gCount);
    memcpy(gOrigin, origin, sizeof gOrigin);
    memcpy(gSpacing, spacing, sizeof gSpacing);
    gPendingInit = true;
    gRasterNext = 0;
    return true;
}

// The box round the scene's shadow casters, snapped outwards to whole metres and kept while the scene stays within
// it and doesn't shrink by more than two metres a side (so spinning or moving things don't restart the probes).
// Heights come from the casters' spheres too, except big ones (a floor, a terrain patch, a long wall: three times the
// median radius and over 2 m), whose sphere reaches far above and below what they are: those count a metre about their
// centre, so a wide floor doesn't stretch the probes tens of metres into the air and the ground.
bool fitBox(const RenderFrame* F, float lo[3], float hi[3])
{
    static float keptLo[3], keptHi[3];
    static bool  kept;
    static float radii[1024];
    uint32_t       sampled = 0;
    const uint32_t step = F->casterCount / 1024 + 1;
    for (uint32_t i = 0; i < F->casterCount && sampled < 1024; i += step)
        if (F->casters[i].sphere[3] <= 200.0f)
            radii[sampled++] = F->casters[i].sphere[3];
    if (!sampled)
        return false;
    std::nth_element(radii, radii + sampled / 2, radii + sampled);
    const float big = fmaxf(3.0f * radii[sampled / 2], 2.0f);
    float a[3] = { 1e30f, 1e30f, 1e30f }, b[3] = { -1e30f, -1e30f, -1e30f };
    for (uint32_t i = 0; i < F->casterCount; ++i)
    {
        const float* s = F->casters[i].sphere;
        if (s[3] > 200.0f)
            continue; // a ground plane reaching the horizon, say: it bounds nothing useful
        for (int k = 0; k < 3; ++k)
        {
            const float r = k == 1 && s[3] > big ? 1.0f : s[3];
            a[k] = fminf(a[k], s[k] - r);
            b[k] = fmaxf(b[k], s[k] + r);
        }
    }
    if (a[0] > b[0])
        return false;
    for (int k = 0; k < 3; ++k)
    {
        a[k] = floorf(a[k]);
        b[k] = ceilf(b[k]);
    }
    b[1] = fmaxf(b[1], a[1] + 2.0f); // at least two metres of height (a flat scene still gets light above it)
    bool refit = !kept;
    for (int k = 0; k < 3 && !refit; ++k)
        refit = a[k] < keptLo[k] || b[k] > keptHi[k] || a[k] > keptLo[k] + 2.0f || b[k] < keptHi[k] - 2.0f;
    if (refit)
    {
        memcpy(keptLo, a, sizeof keptLo);
        memcpy(keptHi, b, sizeof keptHi);
        kept = true;
    }
    memcpy(lo, keptLo, sizeof keptLo);
    memcpy(hi, keptHi, sizeof keptHi);
    return true;
}

uint32_t hashU(uint32_t x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// A uniformly random rotation (Shoemake's random unit quaternion) for this frame's rays.
void randomRotation(uint64_t frame, float rows[3][4])
{
    const float u1 = (float)(hashU((uint32_t)frame * 3u + 1u) >> 8) / 16777216.0f;
    const float u2 = (float)(hashU((uint32_t)frame * 3u + 2u) >> 8) / 16777216.0f * 6.2831853f;
    const float u3 = (float)(hashU((uint32_t)frame * 3u + 3u) >> 8) / 16777216.0f * 6.2831853f;
    const float a = sqrtf(1.0f - u1), b = sqrtf(u1);
    const float x = a * sinf(u2), y = a * cosf(u2), z = b * sinf(u3), w = b * cosf(u3);
    const float m[3][3] = { { 1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y) },
                            { 2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x) },
                            { 2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y) } };
    for (int r = 0; r < 3; ++r)
    {
        rows[r][0] = m[r][0], rows[r][1] = m[r][1], rows[r][2] = m[r][2];
        rows[r][3] = 0;
    }
}

void barrierTex(Cmd* cmd, Texture* t, ResourceState from, ResourceState to)
{
    TextureBarrier b = {};
    b.pTexture = t;
    b.mCurrentState = from;
    b.mNewState = to;
    cmdResourceBarrier(cmd, 0, nullptr, 1, &b, 0, nullptr);
}

void barrierBuf(Cmd* cmd, Buffer* b, ResourceState from, ResourceState to)
{
    BufferBarrier bb = { b, from, to };
    cmdResourceBarrier(cmd, 1, &bb, 0, nullptr, 0, nullptr);
}

void barrierRt(Cmd* cmd, RenderTarget* rt, ResourceState from, ResourceState to)
{
    RenderTargetBarrier b = { rt, from, to };
    cmdResourceBarrier(cmd, 0, nullptr, 0, nullptr, 1, &b);
}

void dispatch(const RenderFrame* F, Pass pass, uint32_t groups)
{
    const uint32_t index = F->frameIndex * kPasses + pass;
    DescriptorData p[7] = {};
    p[0].mIndex = SRT_RES_IDX(GiSrt, PerFrame, gGi);
    p[0].ppBuffers = &gCb[F->frameIndex];
    p[1].mIndex = SRT_RES_IDX(GiSrt, PerFrame, gRaysRW);
    p[1].ppBuffers = &gRays;
    p[2].mIndex = SRT_RES_IDX(GiSrt, PerFrame, gIrrOut);
    p[2].ppTextures = &gIrr;
    p[3].mIndex = SRT_RES_IDX(GiSrt, PerFrame, gDistOut);
    p[3].ppTextures = &gDist;
    p[4].mIndex = SRT_RES_IDX(GiSrt, PerFrame, gProbesRW);
    p[4].ppBuffers = &gProbes;
    p[5].mIndex = SRT_RES_IDX(GiSrt, PerFrame, gFaces);
    p[5].ppTextures = &gFaces->pTexture;
    Buffer* sh = F->skySh ? F->skySh : gNoSh;
    p[6].mIndex = SRT_RES_IDX(GiSrt, PerFrame, gSkySh);
    p[6].ppBuffers = &sh;
    updateDescriptorSet(gRenderer, index, gSet, 7, p);
    cmdBindPipeline(F->cmd, gPipe[pass]);
    cmdBindDescriptorSet(F->cmd, index, gSet);
    cmdDispatch(F->cmd, groups, 1, 1);
}

// The capture's (sun false) or the sun map's pipeline: the frame's positions and attributes (normal, uv0, colour for the
// maps), then the instance-rate stream; both sides drawn (a probe must see the backs of one-sided surfaces to know it is
// inside something; the sun map, so thin one-sided casters still cast).
Pipeline* capturePipeline(bool sun)
{
    VertexLayout l = {};
    l.mBindingCount = 3;
    l.mBindings[0].mStride = RENDER_POSITION_STRIDE;
    l.mBindings[1].mStride = RENDER_ATTRIBUTE_STRIDE;
    l.mBindings[2].mStride = 4;
    l.mBindings[2].mRate = VERTEX_BINDING_RATE_INSTANCE;
    l.mAttribCount = 5;
    l.mAttribs[0] = {};
    l.mAttribs[0].mSemantic = SEMANTIC_POSITION;
    l.mAttribs[0].mFormat = TinyImageFormat_R32G32B32_SFLOAT;
    l.mAttribs[1] = {};
    l.mAttribs[1].mSemantic = SEMANTIC_NORMAL;
    l.mAttribs[1].mFormat = TinyImageFormat_R16G16_SNORM; // octahedral
    l.mAttribs[1].mBinding = 1;
    l.mAttribs[1].mLocation = 1;
    l.mAttribs[2] = {};
    l.mAttribs[2].mSemantic = SEMANTIC_TEXCOORD1;
    l.mAttribs[2].mFormat = TinyImageFormat_R32_UINT;
    l.mAttribs[2].mBinding = 2;
    l.mAttribs[2].mLocation = 2;
    l.mAttribs[3] = {};
    l.mAttribs[3].mSemantic = SEMANTIC_TEXCOORD0;
    l.mAttribs[3].mFormat = TinyImageFormat_R32G32_SFLOAT;
    l.mAttribs[3].mBinding = 1;
    l.mAttribs[3].mLocation = 3;
    l.mAttribs[3].mOffset = 8;
    l.mAttribs[4] = {};
    l.mAttribs[4].mSemantic = SEMANTIC_COLOR;
    l.mAttribs[4].mFormat = TinyImageFormat_R8G8B8A8_UNORM;
    l.mAttribs[4].mBinding = 1;
    l.mAttribs[4].mLocation = 4;
    l.mAttribs[4].mOffset = 16;
    RasterizerStateDesc raster = {};
    raster.mCullMode = CULL_MODE_NONE;
    DepthStateDesc depth = {};
    depth.mDepthTest = true;
    depth.mDepthWrite = true;
    depth.mDepthFunc = sun ? CMP_LEQUAL : CMP_GEQUAL; // the sun's map: 0 nearest the sun; the faces: reversed-Z
    if (sun)
    {
        raster.mDepthBias = 64;
        raster.mSlopeScaledDepthBias = 2.0f;
    }
    TinyImageFormat format = TinyImageFormat_R16G16B16A16_SFLOAT;
    PipelineDesc    desc = {};
    desc.mType = PIPELINE_TYPE_GRAPHICS;
    PIPELINE_LAYOUT_DESC(desc, SRT_LAYOUT_DESC(GiCaptureSrt, Persistent), SRT_LAYOUT_DESC(GiCaptureSrt, PerFrame), NULL, NULL);
    GraphicsPipelineDesc& g = desc.mGraphicsDesc;
    g.mPrimitiveTopo = PRIMITIVE_TOPO_TRI_LIST;
    g.mRenderTargetCount = sun ? 0 : 1;
    g.pColorFormats = sun ? nullptr : &format;
    g.mSampleCount = SAMPLE_COUNT_1;
    g.mDepthStencilFormat = TinyImageFormat_D32_SFLOAT;
    g.pShaderProgram = sun ? gSunShader : gCaptureShader;
    g.pRasterizerState = &raster;
    g.pDepthState = &depth;
    g.pVertexLayout = &l;
    desc.pName = sun ? "renderer_gi_sun" : "renderer_gi_capture";
    Pipeline* p = nullptr;
    addPipeline(gRenderer, &desc, &p);
    return p;
}

// The raster fallback's objects: the faces and their depth, the capture's pipeline and sets, its stand-ins.
bool makeCapture()
{
    RenderTargetDesc d = {};
    d.mWidth = kMaxSlots * 6 * kFace;
    d.mHeight = kFace;
    d.mDepth = d.mArraySize = d.mMipLevels = 1;
    d.mSampleCount = SAMPLE_COUNT_1;
    d.mFormat = TinyImageFormat_R16G16B16A16_SFLOAT;
    d.mStartState = RESOURCE_STATE_SHADER_RESOURCE;
    d.mClearValue = { { 0, 0, 0, 0 } }; // alpha 0: nothing drawn there, the sky
    d.mDescriptors = DESCRIPTOR_TYPE_TEXTURE;
    d.pName = "renderer_gi_faces";
    addRenderTarget(gRenderer, &d, &gFaces);
    RenderTargetDesc z = d;
    z.mFormat = TinyImageFormat_D32_SFLOAT;
    z.mStartState = RESOURCE_STATE_DEPTH_WRITE;
    z.mClearValue = {};
    z.mClearValue.depth = 0.0f; // reversed-Z
    z.mDescriptors = DESCRIPTOR_TYPE_UNDEFINED;
    z.pName = "renderer_gi_face_depth";
    addRenderTarget(gRenderer, &z, &gFaceDepth);
    // As the core makes its stand-in depth: on Direct3D 12 a depth format reads as a texture only through a render
    // target, and one texel high gets a 1D view on Vulkan.
    RenderTargetDesc t = {};
    t.mWidth = t.mHeight = 2;
    t.mDepth = t.mArraySize = t.mMipLevels = 1;
    t.mSampleCount = SAMPLE_COUNT_1;
    t.mFormat = TinyImageFormat_D32_SFLOAT;
    t.mStartState = RESOURCE_STATE_SHADER_RESOURCE;
    t.mDescriptors = DESCRIPTOR_TYPE_TEXTURE;
    t.pName = "renderer_gi_no_atlas";
    addRenderTarget(gRenderer, &t, &gNoAtlas);
    const ResourceMemoryUsage cpu = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
    gNoViews = makeBuffer(sizeof(RenderGpuShadowView) * 4, sizeof(RenderGpuShadowView), DESCRIPTOR_TYPE_BUFFER, cpu,
                          RESOURCE_STATE_SHADER_RESOURCE, "renderer_gi_no_views");
    gNoSh = makeBuffer(9 * 16, 16, DESCRIPTOR_TYPE_BUFFER, cpu, RESOURCE_STATE_SHADER_RESOURCE, "renderer_gi_no_sh");
    gNoDecals = makeBuffer(176, 176, DESCRIPTOR_TYPE_BUFFER, cpu, RESOURCE_STATE_SHADER_RESOURCE, "renderer_gi_no_decals");
    gNoCells = makeBuffer(8, 8, DESCRIPTOR_TYPE_BUFFER, cpu, RESOURCE_STATE_SHADER_RESOURCE, "renderer_gi_no_decal_cells");
    gNoItems = makeBuffer(4, 4, DESCRIPTOR_TYPE_BUFFER, cpu, RESOURCE_STATE_SHADER_RESOURCE, "renderer_gi_no_decal_items");
    for (uint32_t i = 0; i < gFrames; ++i)
        gStream[i] = makeBuffer((uint64_t)kMaxEntries * 4, 0, DESCRIPTOR_TYPE_VERTEX_BUFFER, cpu, RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
                                "renderer_gi_stream");
    SamplerDesc sd = {};
    sd.mMinFilter = FILTER_LINEAR;
    sd.mMagFilter = FILTER_LINEAR;
    sd.mMipMapMode = MIPMAP_MODE_NEAREST;
    sd.mAddressU = sd.mAddressV = sd.mAddressW = ADDRESS_MODE_CLAMP_TO_EDGE;
    sd.mCompareFunc = CMP_LEQUAL;
    addSampler(gRenderer, &sd, &gShadowCmp);
    ShaderLoadDesc shd = {};
    shd.mVert.pFileName = "renderer_gi_capture.vert";
    shd.mFrag.pFileName = "renderer_gi_capture.frag";
    addShader(gRenderer, &shd, &gCaptureShader);
    if (gCaptureShader)
        gCapturePipe = capturePipeline(false);
    ShaderLoadDesc sun = {};
    sun.mVert.pFileName = "renderer_gi_sun.vert";
    sun.mFrag.pFileName = "renderer_gi_sun.frag";
    addShader(gRenderer, &sun, &gSunShader);
    if (gSunShader)
        gSunPipe = capturePipeline(true);
    RenderTargetDesc sd2 = {};
    sd2.mWidth = sd2.mHeight = kSunSize;
    sd2.mDepth = sd2.mArraySize = sd2.mMipLevels = 1;
    sd2.mSampleCount = SAMPLE_COUNT_1;
    sd2.mFormat = TinyImageFormat_D32_SFLOAT;
    sd2.mStartState = RESOURCE_STATE_SHADER_RESOURCE;
    sd2.mClearValue = {};
    sd2.mClearValue.depth = 1.0f; // nothing between: lit
    sd2.mDescriptors = DESCRIPTOR_TYPE_TEXTURE;
    sd2.pName = "renderer_gi_sun";
    addRenderTarget(gRenderer, &sd2, &gSun);
    sd2.pName = "renderer_gi_sun_moving";
    addRenderTarget(gRenderer, &sd2, &gSunDyn);
    for (uint32_t i = 0; i < gFrames; ++i)
    {
        gSunStream[i] = makeBuffer((uint64_t)kMaxSunEntries * 4, 0, DESCRIPTOR_TYPE_VERTEX_BUFFER, cpu, RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
                                   "renderer_gi_sun_stream");
        gSunDynStream[i] = makeBuffer((uint64_t)kMaxSunEntries * 4, 0, DESCRIPTOR_TYPE_VERTEX_BUFFER, cpu,
                                      RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER, "renderer_gi_sun_moving_stream");
    }
    DescriptorSetDesc persistent = SRT_SET_DESC(GiCaptureSrt, Persistent, 1, 0);
    addDescriptorSet(gRenderer, &persistent, &gCapturePersistent);
    // Two a frame: the capture's, and the sun map's (which can't bind the map it draws).
    DescriptorSetDesc perFrame = SRT_SET_DESC(GiCaptureSrt, PerFrame, gFrames * 2, 0);
    addDescriptorSet(gRenderer, &perFrame, &gCaptureSet);
    waitForAllResourceLoads();
    bool ok = gFaces && gFaceDepth && gNoAtlas && gNoViews && gNoSh && gNoDecals && gNoCells && gNoItems && gShadowCmp && gCapturePipe && gSunPipe && gSun && gSunDyn && gCapturePersistent && gCaptureSet &&
              gNoViews->pCpuMappedAddress && gNoSh->pCpuMappedAddress;
    for (uint32_t i = 0; i < gFrames; ++i)
        ok = ok && gStream[i] && gStream[i]->pCpuMappedAddress && gSunStream[i] && gSunStream[i]->pCpuMappedAddress && gSunDynStream[i] &&
                  gSunDynStream[i]->pCpuMappedAddress;
    if (!ok)
        return false;
    memset(gNoViews->pCpuMappedAddress, 0, sizeof(RenderGpuShadowView) * 4);
    memset(gNoSh->pCpuMappedAddress, 0, 9 * 16);
    DescriptorData s = {};
    s.mIndex = SRT_RES_IDX(GiCaptureSrt, Persistent, gShadowCmp);
    s.ppSamplers = &gShadowCmp;
    updateDescriptorSet(gRenderer, 0, gCapturePersistent, 1, &s);
    gStreamCpu = new uint32_t[kMaxEntries];
    gRuns = new Run[kMaxRuns];
    gSunStreamCpu = new uint32_t[kMaxSunEntries];
    gSunRuns = new Run[kMaxRuns];
    gSunDynStreamCpu = new uint32_t[kMaxSunEntries];
    gSunDynRuns = new Run[kMaxRuns];
    gLastMove = new uint64_t[kMoveRows]();
    gPrevMove = new uint64_t[kMoveRows]();
    memset(gTexBound, 0, sizeof gTexBound);
    gSunDrawn = 0;
    gSunValid = false;
    return true;
}

void freeCapture()
{
    delete[] gStreamCpu;
    delete[] gRuns;
    delete[] gSunStreamCpu;
    delete[] gSunRuns;
    delete[] gSunDynStreamCpu;
    delete[] gSunDynRuns;
    delete[] gLastMove;
    delete[] gPrevMove;
    gLastMove = gPrevMove = nullptr;
    gSunDynStreamCpu = nullptr;
    gSunDynRuns = nullptr;
    gSunDynOn = false;
    gStreamCpu = nullptr;
    gRuns = nullptr;
    gSunStreamCpu = nullptr;
    gSunRuns = nullptr;
    gSunValid = false;
    RenderTarget** rts[] = { &gFaces, &gFaceDepth, &gNoAtlas, &gSun, &gSunDyn };
    for (RenderTarget** r : rts)
    {
        if (*r)
            removeRenderTarget(gRenderer, *r);
        *r = nullptr;
    }
    Buffer** bufs[] = { &gNoViews, &gNoSh, &gNoDecals, &gNoCells, &gNoItems, &gStream[0], &gStream[1], &gStream[2], &gStream[3],
                        &gSunStream[0], &gSunStream[1], &gSunStream[2], &gSunStream[3],
                        &gSunDynStream[0], &gSunDynStream[1], &gSunDynStream[2], &gSunDynStream[3] };
    for (Buffer** b : bufs)
    {
        if (*b)
            removeResource(*b);
        *b = nullptr;
    }
    DescriptorSet** sets[] = { &gCapturePersistent, &gCaptureSet };
    for (DescriptorSet** d : sets)
    {
        if (*d)
            removeDescriptorSet(gRenderer, *d);
        *d = nullptr;
    }
    if (gCapturePipe)
        removePipeline(gRenderer, gCapturePipe);
    if (gCaptureShader)
        removeShader(gRenderer, gCaptureShader);
    if (gSunPipe)
        removePipeline(gRenderer, gSunPipe);
    if (gSunShader)
        removeShader(gRenderer, gSunShader);
    gSunPipe = nullptr;
    gSunShader = nullptr;
    if (gShadowCmp)
        removeSampler(gRenderer, gShadowCmp);
    gCapturePipe = nullptr;
    gCaptureShader = nullptr;
    gShadowCmp = nullptr;
    gCaptureReady = false;
}

// This frame's raster captures: `count` probes from `start`, six faces each, every caster within a face (allowing for
// how far relocation may have moved the probe) into the instance stream, in runs of one mesh. Returns the entries.
uint32_t planCapture(const RenderFrame* F, uint32_t start, uint32_t count)
{
    const uint32_t nx = gCount[0], nz = gCount[2], probes = gCount[0] * gCount[1] * gCount[2];
    const float    slack = 0.45f * sqrtf(gSpacing[0] * gSpacing[0] + gSpacing[1] * gSpacing[1] + gSpacing[2] * gSpacing[2]);
    uint32_t       used = 0, runs = 0;
    for (uint32_t slot = 0; slot < count; ++slot)
    {
        const uint32_t i = (start + slot) % probes;
        const uint32_t c[3] = { i % nx, i / (nx * nz), (i / nx) % nz };
        float          e[3];
        for (int k = 0; k < 3; ++k)
            e[k] = gOrigin[k] + (float)c[k] * gSpacing[k];
        for (uint32_t f = 0; f < 6; ++f)
        {
            const uint32_t view = slot * 6 + f;
            const float(*b)[3] = kFaceAxes[f]; // forward, right, up
            gRunStart[view] = runs;
            Run* run = nullptr;
            for (uint32_t k = 0; k < F->casterCount && used < kMaxEntries; ++k)
            {
                const RenderCaster& cs = F->casters[k];
                float x = cs.sphere[0] - e[0], y = cs.sphere[1] - e[1], z = cs.sphere[2] - e[2];
                float rad = (cs.sphere[3] + slack) * 1.4142136f;
                float fw = b[0][0] * x + b[0][1] * y + b[0][2] * z;
                float rt = b[1][0] * x + b[1][1] * y + b[1][2] * z;
                float up = b[2][0] * x + b[2][1] * y + b[2][2] * z;
                if (fw - fabsf(rt) < -rad || fw - fabsf(up) < -rad)
                    continue; // outside the 90 degree face (planes scaled by sqrt 2)
                if (!run || run->firstIndex != cs.firstIndex || run->vertexOffset != cs.vertexOffset || run->indexCount != cs.indexCount)
                {
                    if (runs >= kMaxRuns)
                        break;
                    run = &gRuns[runs++];
                    *run = { cs.firstIndex, cs.indexCount, cs.vertexOffset, used, 0 };
                }
                gStreamCpu[used++] = cs.instance | (view << RENDER_VIEW_SHIFT) | (cs.twoSided ? 0x80000000u : 0u);
                run->count++;
            }
        }
    }
    gRunStart[count * 6] = runs;
    return used;
}

// The sun's casters split into still and moving (skinned, or moved within kMovingFrames: a caster that stops moves into
// the still map once), each in runs of one mesh (the core sorts them by mesh). Returns whether the still map must be
// drawn again; gSunDynOn says whether the moving map is drawn this frame. The view is orthographic along the sun over
// the box round the probe volume and every shadow caster (spheres over 200 m left out, as fitBox does), whole: outside
// its rectangle no caster lies between a point and the sun, so the captures read it as lit, exactly. It is refitted only
// when the still box changes or a moving caster leaves the box it was fitted to (which then takes the moving casters'
// spheres with a metre and a quarter of their radius to spare), so a moving caster never redraws the still casters.
void sunRuns(const RenderCaster& cs, Run* runs, uint32_t* count, uint32_t* stream, uint32_t* used, Run** run)
{
    if (*used >= kMaxSunEntries)
        return;
    if (!*run || (*run)->firstIndex != cs.firstIndex || (*run)->vertexOffset != cs.vertexOffset || (*run)->indexCount != cs.indexCount)
    {
        if (*count >= kMaxRuns)
            return;
        *run = &runs[(*count)++];
        **run = { cs.firstIndex, cs.indexCount, cs.vertexOffset, *used, 0 };
    }
    stream[(*used)++] = cs.instance;
    (*run)->count++;
}

bool fitSun(const RenderFrame* F)
{
    gSunDynOn = false;
    if (!F->sunOn || !F->casters)
    {
        gSunValid = false;
        return false;
    }
    const uint64_t now = F->frameNumber;
    for (uint32_t i = 0; i < F->casterCount; ++i)
    {
        const RenderCaster& c = F->casters[i];
        if (c.instance < kMoveRows && gLastMove[c.instance] != c.moved)
            gPrevMove[c.instance] = gLastMove[c.instance], gLastMove[c.instance] = c.moved;
    }
    auto moving = [&](const RenderCaster& c) {
        if (c.dynamic)
            return true;
        if (c.instance >= kMoveRows)
            return c.moved + kMovingFrames > now;
        return gPrevMove[c.instance] && gPrevMove[c.instance] + 2 * kMovingFrames > now && c.moved + kMovingFrames > now;
    };
    uint64_t latest = F->casterRemoved, sig = 0;
    float    still[6], dyn[6];
    uint32_t stillCount = 0;
    for (int k = 0; k < 3; ++k)
    {
        still[k] = gOrigin[k];
        still[k + 3] = gOrigin[k] + gSpacing[k] * (float)(gCount[k] - 1);
        dyn[k] = 1e30f, dyn[k + 3] = -1e30f;
    }
    for (uint32_t i = 0; i < F->casterCount; ++i)
    {
        const RenderCaster& c = F->casters[i];
        const float*        sp = c.sphere;
        const bool          m = moving(c);
        if (!m)
        {
            stillCount++;
            latest = c.moved > latest ? c.moved : latest;
            sig += (uint64_t)(c.instance + 1) * 0x9E3779B97F4A7C15ull;
        }
        if (sp[3] > 200.0f)
            continue;
        float* box = m ? dyn : still;
        for (int k = 0; k < 3; ++k)
        {
            box[k] = fminf(box[k], sp[k] - sp[3]);
            box[k + 3] = fmaxf(box[k + 3], sp[k] + sp[3]);
        }
    }
    sig += stillCount;
    bool refit = !gSunValid || memcmp(still, gSunStill, sizeof still) != 0;
    for (int k = 0; k < 3 && !refit; ++k)
        refit = dyn[k] < gSunBox[k] || dyn[k + 3] > gSunBox[k + 3];
    bool same = !refit && latest < gSunDrawn && sig == gSunSig && F->textureVersion == gSunTextures &&
                F->sunDir[0] * gSunDir[0] + F->sunDir[1] * gSunDir[1] + F->sunDir[2] * gSunDir[2] > 0.999999f;
    for (int k = 0; k < 3 && same; ++k)
        same = gSunOrigin[k] == gOrigin[k] && gSunSpacing[k] == gSpacing[k];
    if (!same)
    {
        // The box: the still one, and round the moving casters some room to move in.
        memcpy(gSunStill, still, sizeof still);
        memcpy(gSunBox, still, sizeof still);
        for (uint32_t i = 0; i < F->casterCount; ++i)
        {
            const RenderCaster& c = F->casters[i];
            const float*        sp = c.sphere;
            if (!moving(c) || sp[3] > 200.0f)
                continue;
            const float r = sp[3] * 1.25f + 1.0f;
            for (int k = 0; k < 3; ++k)
            {
                gSunBox[k] = fminf(gSunBox[k], sp[k] - r);
                gSunBox[k + 3] = fmaxf(gSunBox[k + 3], sp[k] + r);
            }
        }
        // The light's frame: f along the light, r and u across it.
        const float f[3] = { -F->sunDir[0], -F->sunDir[1], -F->sunDir[2] };
        const float up[3] = { fabsf(f[1]) < 0.99f ? 0.0f : 1.0f, fabsf(f[1]) < 0.99f ? 1.0f : 0.0f, 0.0f };
        float       r[3] = { up[1] * f[2] - up[2] * f[1], up[2] * f[0] - up[0] * f[2], up[0] * f[1] - up[1] * f[0] };
        const float rl = sqrtf(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
        for (int k = 0; k < 3; ++k)
            r[k] /= rl;
        const float  u[3] = { f[1] * r[2] - f[2] * r[1], f[2] * r[0] - f[0] * r[2], f[0] * r[1] - f[1] * r[0] };
        const float* axes[3] = { r, u, f };
        float        mn[3] = { 1e30f, 1e30f, 1e30f }, mx[3] = { -1e30f, -1e30f, -1e30f };
        for (int c = 0; c < 8; ++c)
        {
            const float p[3] = { gSunBox[c & 1 ? 3 : 0], gSunBox[c & 2 ? 4 : 1], gSunBox[c & 4 ? 5 : 2] };
            for (int a = 0; a < 3; ++a)
            {
                const float d = axes[a][0] * p[0] + axes[a][1] * p[1] + axes[a][2] * p[2];
                mn[a] = fminf(mn[a], d);
                mx[a] = fmaxf(mx[a], d);
            }
        }
        // A texel's margin round the rectangle (so the edge isn't sampled), half a metre before the nearest caster.
        for (int a = 0; a < 2; ++a)
        {
            const float pad = (mx[a] - mn[a]) * (2.0f / (float)kSunSize) + 0.01f;
            mn[a] -= pad;
            mx[a] += pad;
        }
        mn[2] -= 0.5f;
        mx[2] += 0.01f;
        for (int a = 0; a < 3; ++a)
        {
            const float scale = a < 2 ? 2.0f / (mx[a] - mn[a]) : 1.0f / (mx[a] - mn[a]);
            const float offset = a < 2 ? -(mx[a] + mn[a]) * 0.5f * scale : -mn[a] * scale;
            gSunView[a][0] = axes[a][0] * scale, gSunView[a][1] = axes[a][1] * scale, gSunView[a][2] = axes[a][2] * scale;
            gSunView[a][3] = offset;
        }
    }
    // The casters into their runs: the still ones only when their map is drawn, the moving ones every frame.
    uint32_t used = 0, dynUsed = 0;
    Run *    run = nullptr, *dynRun = nullptr;
    if (!same)
        gSunRunCount = 0;
    gSunDynRunCount = 0;
    for (uint32_t i = 0; i < F->casterCount; ++i)
    {
        const RenderCaster& c = F->casters[i];
        if (moving(c))
            sunRuns(c, gSunDynRuns, &gSunDynRunCount, gSunDynStreamCpu, &dynUsed, &dynRun);
        else if (!same)
            sunRuns(c, gSunRuns, &gSunRunCount, gSunStreamCpu, &used, &run);
    }
    if (dynUsed)
        memcpy(gSunDynStream[F->frameIndex]->pCpuMappedAddress, gSunDynStreamCpu, (size_t)dynUsed * 4);
    gSunDynOn = dynUsed != 0;
    if (same)
        return false;
    memcpy(gSunStream[F->frameIndex]->pCpuMappedAddress, gSunStreamCpu, (size_t)used * 4);
    memcpy(gSunDir, F->sunDir, sizeof gSunDir);
    memcpy(gSunOrigin, gOrigin, sizeof gSunOrigin);
    memcpy(gSunSpacing, gSpacing, sizeof gSunSpacing);
    gSunSig = sig;
    gSunTextures = F->textureVersion;
    gSunDrawn = now;
    gSunValid = true;
    return true;
}

// The capture's lighting constants: the sun, sky, fog and shadow maps as the forward pass has them this frame.
void fillCapture(GiData* d, const RenderFrame* F, uint32_t start, uint32_t count)
{
    const bool maps = F->shadowViews && F->sunAtlas && F->localAtlas;
    d->raster[0] = start;
    d->raster[1] = count;
    const uint32_t lights = F->allLightCount > F->lightCount ? F->allLightCount : F->lightCount;
    d->raster[2] = F->lightBuffer && F->lightShapes ? (lights < kMaxCaptureLights ? lights : kMaxCaptureLights) : 0;
    d->raster[3] = (maps ? 1u : 0u) | (gValid ? 2u : 0u) | (F->skySh ? 4u : 0u);
    memcpy(d->sunDir, F->sunDir, sizeof F->sunDir);
    d->sunDir[3] = F->sunOn ? 1.0f : 0.0f;
    memcpy(d->sunColor, F->sunColor, sizeof F->sunColor);
    d->sunColor[3] = F->ambient;
    memcpy(d->skyZenith, F->skyZenith, sizeof F->skyZenith);
    memcpy(d->skyHorizon, F->skyHorizon, sizeof F->skyHorizon);
    memcpy(d->skyGround, F->skyGround, sizeof F->skyGround);
    memcpy(d->fogMedium, F->fogMedium, sizeof d->fogMedium);
    memcpy(d->fogAlbedo, F->fogAlbedo, sizeof d->fogAlbedo);
    d->shadow[0] = gSunValid ? 1.0f : 0.0f;
    d->shadow[1] = 1.0f / (float)kSunSize;
    d->shadow[2] = F->localTexel;
    d->shadow[3] = gSunValid && gSunDynOn ? 1.0f : 0.0f;
    memcpy(d->sunView, gSunView, sizeof d->sunView);
}

// The capture's descriptors into set `index`, the sun's maps bound as `sun` and `sunDyn` (stand-ins while they are drawn).
void writeCaptureSet(const RenderFrame* F, uint32_t index, Texture* sun, Texture* sunDyn)
{
    const bool maps = F->shadowViews && F->sunAtlas && F->localAtlas;
    Buffer*    views = maps ? F->shadowViews : gNoViews;
    Buffer*    sh = F->skySh ? F->skySh : gNoSh;
    Texture*   local = maps ? F->localAtlas->pTexture : gNoAtlas->pTexture;
    Buffer*    lights = F->lightBuffer ? F->lightBuffer : gNoViews; // unread without lights (raster.z 0)
    Buffer*    shapes = F->lightShapes ? F->lightShapes : gNoViews;
    DescriptorData d[18] = {};
    const bool     decals = F->worldDecalCount && F->worldDecals && F->decalCells && F->decalItems;
    Buffer*        decalBufs[3] = { decals ? F->worldDecals : gNoDecals, decals ? F->decalCells : gNoCells,
                                    decals ? F->decalItems : gNoItems };
    d[13].mIndex = SRT_RES_IDX(GiCaptureSrt, PerFrame, gDecals);
    d[13].ppBuffers = &decalBufs[0];
    d[14].mIndex = SRT_RES_IDX(GiCaptureSrt, PerFrame, gDecalCells);
    d[14].ppBuffers = &decalBufs[1];
    d[15].mIndex = SRT_RES_IDX(GiCaptureSrt, PerFrame, gDecalItems);
    d[15].ppBuffers = &decalBufs[2];
    d[0].mIndex = SRT_RES_IDX(GiCaptureSrt, PerFrame, gGi);
    d[0].ppBuffers = &gCb[F->frameIndex];
    d[1].mIndex = SRT_RES_IDX(GiCaptureSrt, PerFrame, gInstances);
    d[1].ppBuffers = (Buffer**)&F->instances;
    d[2].mIndex = SRT_RES_IDX(GiCaptureSrt, PerFrame, gMaterials);
    d[2].ppBuffers = (Buffer**)&F->materials;
    d[3].mIndex = SRT_RES_IDX(GiCaptureSrt, PerFrame, gLights);
    d[3].ppBuffers = &lights;
    d[4].mIndex = SRT_RES_IDX(GiCaptureSrt, PerFrame, gShapes);
    d[4].ppBuffers = &shapes;
    d[5].mIndex = SRT_RES_IDX(GiCaptureSrt, PerFrame, gViews);
    d[5].ppBuffers = &views;
    d[6].mIndex = SRT_RES_IDX(GiCaptureSrt, PerFrame, gSkySh);
    d[6].ppBuffers = &sh;
    d[7].mIndex = SRT_RES_IDX(GiCaptureSrt, PerFrame, gGiSun);
    d[7].ppTextures = &sun;
    d[8].mIndex = SRT_RES_IDX(GiCaptureSrt, PerFrame, gLocalAtlas);
    d[8].ppTextures = &local;
    d[9].mIndex = SRT_RES_IDX(GiCaptureSrt, PerFrame, gGiIrr);
    d[9].ppTextures = &gIrr;
    d[10].mIndex = SRT_RES_IDX(GiCaptureSrt, PerFrame, gGiDist);
    d[10].ppTextures = &gDist;
    d[11].mIndex = SRT_RES_IDX(GiCaptureSrt, PerFrame, gGiProbes);
    d[11].ppBuffers = &gProbes;
    d[12].mIndex = SRT_RES_IDX(GiCaptureSrt, PerFrame, gMaterialMaps);
    d[12].ppBuffers = (Buffer**)&F->materialMaps;
    d[16].mIndex = SRT_RES_IDX(GiCaptureSrt, PerFrame, gGiSunDyn);
    d[16].ppTextures = &sunDyn;
    uint32_t bindings = 17;
    if (gTexBound[index] != F->textureVersion) // 1024 descriptors: only when the core's slots changed
    {
        d[17].mIndex = SRT_RES_IDX(GiCaptureSrt, PerFrame, gTextures);
        d[17].ppTextures = (Texture**)F->textures;
        d[17].mCount = RENDER_MAX_TEXTURES;
        gTexBound[index] = F->textureVersion;
        bindings = 18;
    }
    updateDescriptorSet(gRenderer, index, gCaptureSet, bindings, d);
}

// One of the sun's maps (fitSun's runs into it, cleared to lit first).
void recordSunMap(const RenderFrame* F, RenderTarget* map, Buffer* stream, const Run* runs, uint32_t runCount)
{
    Cmd* cmd = F->cmd;
    barrierRt(cmd, map, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_DEPTH_WRITE);
    BindRenderTargetsDesc bind = {};
    bind.mDepthStencil = { map, LOAD_ACTION_CLEAR };
    bind.mDepthStencil.mStoreAction = STORE_ACTION_STORE;
    cmdBindRenderTargets(cmd, &bind);
    cmdSetViewport(cmd, 0, 0, (float)kSunSize, (float)kSunSize, 0, 1);
    cmdSetScissor(cmd, 0, 0, kSunSize, kSunSize);
    cmdBindPipeline(cmd, gSunPipe);
    cmdBindDescriptorSet(cmd, 0, gCapturePersistent);
    cmdBindDescriptorSet(cmd, F->frameIndex * 2 + 1, gCaptureSet);
    cmdBindIndexBuffer(cmd, F->indices, INDEX_TYPE_UINT32, 0);
    for (uint32_t r = 0; r < runCount; ++r)
    {
        const Run& run = runs[r];
        Buffer*    bufs[3] = { F->positions, F->attributes, stream };
        uint32_t   strides[3] = { RENDER_POSITION_STRIDE, RENDER_ATTRIBUTE_STRIDE, 4 };
        uint64_t   offsets[3] = { 0, 0, (uint64_t)run.streamStart * 4 };
        cmdBindVertexBuffer(cmd, 3, bufs, strides, offsets);
        cmdDrawIndexedInstanced(cmd, run.indexCount, run.firstIndex, run.count, (uint32_t)run.vertexOffset, 0);
    }
    cmdBindRenderTargets(cmd, nullptr);
    barrierRt(cmd, map, RESOURCE_STATE_DEPTH_WRITE, RESOURCE_STATE_SHADER_RESOURCE);
}

// The sun's maps that change this frame: the still one when fitSun says so, the moving one whenever it holds any.
void recordSun(const RenderFrame* F, bool still)
{
    const uint32_t fi = F->frameIndex;
    writeCaptureSet(F, fi * 2 + 1, gNoAtlas->pTexture, gNoAtlas->pTexture);
    if (still)
        recordSunMap(F, gSun, gSunStream[fi], gSunRuns, gSunRunCount);
    if (gSunDynOn)
        recordSunMap(F, gSunDyn, gSunDynStream[fi], gSunDynRuns, gSunDynRunCount);
}

void recordCapture(const RenderFrame* F, uint32_t count)
{
    Cmd*           cmd = F->cmd;
    const uint32_t fi = F->frameIndex;
    writeCaptureSet(F, fi * 2, gSun->pTexture, gSunDyn->pTexture);
    barrierRt(cmd, gFaces, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_RENDER_TARGET);
    BindRenderTargetsDesc bind = {};
    bind.mRenderTargetCount = 1;
    bind.mRenderTargets[0] = { gFaces, LOAD_ACTION_CLEAR };
    bind.mDepthStencil = { gFaceDepth, LOAD_ACTION_CLEAR };
    bind.mDepthStencil.mStoreAction = STORE_ACTION_DONTCARE;
    cmdBindRenderTargets(cmd, &bind);
    cmdBindPipeline(cmd, gCapturePipe);
    cmdBindDescriptorSet(cmd, 0, gCapturePersistent);
    cmdBindDescriptorSet(cmd, fi * 2, gCaptureSet);
    cmdBindIndexBuffer(cmd, F->indices, INDEX_TYPE_UINT32, 0);
    for (uint32_t view = 0; view < count * 6; ++view)
    {
        if (gRunStart[view] == gRunStart[view + 1])
            continue; // nothing in that face: all sky (the clear)
        cmdSetViewport(cmd, (float)(view * kFace), 0, (float)kFace, (float)kFace, 0, 1);
        cmdSetScissor(cmd, view * kFace, 0, kFace, kFace);
        for (uint32_t r = gRunStart[view]; r < gRunStart[view + 1]; ++r)
        {
            const Run& run = gRuns[r];
            Buffer*    bufs[3] = { F->positions, F->attributes, gStream[fi] };
            uint32_t   strides[3] = { RENDER_POSITION_STRIDE, RENDER_ATTRIBUTE_STRIDE, 4 };
            uint64_t   offsets[3] = { 0, 0, (uint64_t)run.streamStart * 4 };
            cmdBindVertexBuffer(cmd, 3, bufs, strides, offsets);
            cmdDrawIndexedInstanced(cmd, run.indexCount, run.firstIndex, run.count, (uint32_t)run.vertexOffset, 0);
        }
    }
    cmdBindRenderTargets(cmd, nullptr);
    barrierRt(cmd, gFaces, RESOURCE_STATE_RENDER_TARGET, RESOURCE_STATE_SHADER_RESOURCE);
}
} // namespace

int load(int reloaded)
{
    const YeaHost* host = yea::host();
    gRenderer = (Renderer*)host->renderer;
    gQueue = (Queue*)host->queue;
    gFrames = host->framesInFlight < kMaxFrames ? host->framesInFlight : kMaxFrames;
    gReady = gOn = gValid = gCaptureReady = false;
    for (uint32_t i = 0; i < kPasses; ++i)
    {
        ShaderLoadDesc sd = {};
        sd.mComp.pFileName = kShaderNames[i];
        addShader(gRenderer, &sd, &gShader[i]);
        if (!gShader[i])
            return 1;
        PipelineDesc desc = {};
        desc.mType = PIPELINE_TYPE_COMPUTE;
        PIPELINE_LAYOUT_DESC(desc, NULL, SRT_LAYOUT_DESC(GiSrt, PerFrame), NULL, NULL);
        desc.mComputeDesc.pShaderProgram = gShader[i];
        desc.pName = kShaderNames[i];
        addPipeline(gRenderer, &desc, &gPipe[i]);
        if (!gPipe[i])
            return 1;
    }
    DescriptorSetDesc setDesc = SRT_SET_DESC(GiSrt, PerFrame, gFrames * kPasses, 0);
    addDescriptorSet(gRenderer, &setDesc, &gSet);
    for (uint32_t i = 0; i < gFrames; ++i)
        gCb[i] = makeBuffer(512, 0, DESCRIPTOR_TYPE_UNIFORM_BUFFER, RESOURCE_MEMORY_USAGE_CPU_TO_GPU, RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
                            "renderer_gi_cb");
    waitForAllResourceLoads();
    bool ok = gSet != nullptr;
    for (uint32_t i = 0; i < gFrames; ++i)
        ok = ok && gCb[i] && gCb[i]->pCpuMappedAddress;
    // The raster fallback, for frames nothing traced (always, without ray tracing).
    if (!ok || !makeCapture())
        return 1;
    gCaptureReady = true;
    gReady = true;
    yea::log(yea::raytracing() ? "global illumination: ray-traced probes" : "global illumination: rasterized probes (no ray tracing)");
    return 0;
}

void unload()
{
    if (!gRenderer)
        return;
    waitQueueIdle(gQueue);
    freeGrid();
    freeCapture();
    for (uint32_t i = 0; i < kMaxFrames; ++i)
    {
        if (gCb[i])
            removeResource(gCb[i]);
        gCb[i] = nullptr;
    }
    if (gSet)
        removeDescriptorSet(gRenderer, gSet);
    gSet = nullptr;
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

void gi_volume(yea_float3 center, yea_float3 size)
{
    volume->center = center;
    volume->size = { fabsf(size.x), fabsf(size.y), fabsf(size.z) };
    volume->set = size.x != 0 && size.y != 0 && size.z != 0;
}

bool gi_prepare(uint64_t address)
{
    RenderFrame* F = (RenderFrame*)(uintptr_t)address;
    gOn = gInitNow = false;
    if (!F || F->version != RENDER_FRAME_VERSION)
        return false;
    F->giProbeCount = 0;
    if (!gReady || !settings.gi)
        return false;
    // The box: the game's, else round the scene.
    float lo[3], hi[3];
    if (volume->set)
    {
        const float c[3] = { volume->center.x, volume->center.y, volume->center.z }, s[3] = { volume->size.x, volume->size.y, volume->size.z };
        for (int k = 0; k < 3; ++k)
            lo[k] = c[k] - s[k] * 0.5f, hi[k] = c[k] + s[k] * 0.5f;
    }
    else if (!fitBox(F, lo, hi))
        return false;
    // As many probes as keep them about evenly spaced, at most 16 x 8 x 16, no closer than half a metre.
    const uint32_t maxN[3] = { kMaxX, kMaxY, kMaxZ };
    float          size[3], longest = 0;
    for (int k = 0; k < 3; ++k)
    {
        size[k] = fmaxf(hi[k] - lo[k], 0.5f);
        longest = fmaxf(longest, size[k]);
    }
    const float target = fmaxf(longest / (float)(kMaxX - 1), 0.5f);
    uint32_t    n[3];
    float       spacing[3];
    for (int k = 0; k < 3; ++k)
    {
        uint32_t c = (uint32_t)ceilf(size[k] / target) + 1;
        n[k] = c < 2 ? 2 : (c > maxN[k] ? maxN[k] : c);
        spacing[k] = size[k] / (float)(n[k] - 1);
    }
    if (!ensureGrid(n, lo, spacing))
        return false;
    gProbeCount = n[0] * n[1] * n[2];
    gRayCount = settings.quality == Quality::Low ? 64 : settings.quality == Quality::High ? 256 : 128;

    GiData d = {};
    d.grid[0][0] = lo[0], d.grid[0][1] = lo[1], d.grid[0][2] = lo[2], d.grid[0][3] = (float)(F->frameNumber & 7u);
    d.grid[1][0] = spacing[0], d.grid[1][1] = spacing[1], d.grid[1][2] = spacing[2], d.grid[1][3] = (float)gProbeCount;
    d.grid[2][0] = (float)n[0], d.grid[2][1] = (float)n[1], d.grid[2][2] = (float)n[2], d.grid[2][3] = (float)gRayCount;
    d.grid[3][0] = 1.0f / (float)(n[0] * n[2] * kIrrTexels), d.grid[3][1] = 1.0f / (float)(n[1] * kIrrTexels);
    d.grid[3][2] = 1.0f / (float)(n[0] * n[2] * kDistTexels), d.grid[3][3] = 1.0f / (float)(n[1] * kDistTexels);
    randomRotation(F->frameNumber, d.rotation);
    const float maxSpacing = fmaxf(spacing[0], fmaxf(spacing[1], spacing[2]));
    d.params[0] = clampf(settings.hysteresis, 0, 0.995f);
    d.params[1] = 1.5f * sqrtf(spacing[0] * spacing[0] + spacing[1] * spacing[1] + spacing[2] * spacing[2]);
    d.params[2] = 0.25f;
    d.params[3] = 0.15f * maxSpacing;
    d.counts[0] = gProbeCount;
    d.counts[1] = gRayCount;
    // Decals, painted in the raster captures as on screen (they catch up as the captures go round).
    d.decals[0] = F->worldDecals && F->decalCells && F->decalItems ? F->worldDecalCount : 0;
    memcpy(d.decalGrid, F->decalGrid, sizeof d.decalGrid);

    memcpy(F->giGrid, d.grid, sizeof F->giGrid);
    memcpy(F->giRotation, d.rotation, sizeof F->giRotation);
    // A new grid starts this frame (gi_run clears it): nothing to trace until next frame.
    gInitNow = gPendingInit;
    F->giProbeCount = gInitNow ? 0 : gProbeCount;
    F->giRaysPerProbe = gRayCount;
    F->giProbeData = gProbes;
    F->giRayOut = gRays;
    F->giIrradiance = gValid ? gIrr : nullptr;
    F->giDistance = gValid ? gDist : nullptr;
    F->giTraced = 0;
    F->giDebug = settings.show_gi ? 1u : 0u;
    memcpy(gCb[F->frameIndex]->pCpuMappedAddress, &d, sizeof d);
    gOn = true;
    return true;
}

void gi_run(uint64_t address)
{
    RenderFrame* F = (RenderFrame*)(uintptr_t)address;
    if (!gOn || !gReady || !F)
        return;
    gOn = false;
    const bool traced = F->giTraced && !gInitNow;
    // Nothing traced the rays this frame (no ray tracing, TAA off, renderer_rt off): capture a few probes instead.
    uint32_t start = 0, count = 0;
    if (!gInitNow && !traced)
    {
        if (!gCaptureReady || !F->positions || !F->attributes || !F->indices || !F->instances || !F->materials || !F->materialMaps ||
            !F->textures)
            return; // the probes keep what they have
        const uint32_t want = settings.quality == Quality::Low ? 4 : settings.quality == Quality::High ? 16 : 8;
        count = want < gProbeCount ? want : gProbeCount;
        start = gRasterNext % gProbeCount;
        gRasterNext = (start + count) % gProbeCount;
        const uint32_t used = planCapture(F, start, count);
        memcpy(gStream[F->frameIndex]->pCpuMappedAddress, gStreamCpu, (size_t)used * 4);
    }
    const bool raster = count > 0;
    const bool drawSun = raster && fitSun(F);
    GiData* d = (GiData*)gCb[F->frameIndex]->pCpuMappedAddress;
    d->counts[2] = traced || raster ? 1u : 0u;
    d->counts[3] = raster ? 1u : 0u;
    if (raster)
    {
        fillCapture(d, F, start, count);
        // Each probe is captured only once a round, from a still image: little noise to average away, so less history.
        d->params[0] = fminf(d->params[0], 0.6f);
    }
    Cmd* cmd = F->cmd;
    cmdBeginDebugMarker(cmd, 0.9f, 0.6f, 0.3f, "Renderer GI");
    if (drawSun || (raster && gSunDynOn))
        recordSun(F, drawSun);
    if (raster)
        recordCapture(F, count);
    barrierTex(cmd, gIrr, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_UNORDERED_ACCESS);
    barrierTex(cmd, gDist, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_UNORDERED_ACCESS);
    barrierBuf(cmd, gProbes, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_UNORDERED_ACCESS);
    barrierBuf(cmd, gRays, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_UNORDERED_ACCESS);
    if (gInitNow)
    {
        dispatch(F, kInit, gProbeCount);
        gPendingInit = false;
        gValid = false;
    }
    else
    {
        const uint32_t probes = raster ? count : gProbeCount;
        if (raster)
        {
            dispatch(F, kRays, count);
            barrierBuf(cmd, gRays, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_UNORDERED_ACCESS);
        }
        dispatch(F, kIrradiance, probes);
        dispatch(F, kDistance, probes);
        // Classification rewrites the probes' states and offsets, which both blends read.
        barrierBuf(cmd, gProbes, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_UNORDERED_ACCESS);
        dispatch(F, kClassify, (probes + 63) / 64);
        gValid = true;
    }
    barrierTex(cmd, gIrr, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_SHADER_RESOURCE);
    barrierTex(cmd, gDist, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_SHADER_RESOURCE);
    barrierBuf(cmd, gProbes, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_SHADER_RESOURCE);
    barrierBuf(cmd, gRays, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_SHADER_RESOURCE);
    cmdEndDebugMarker(cmd);
    // Valid from now in this frame's forward pass too.
    F->giIrradiance = gValid ? gIrr : nullptr;
    F->giDistance = gValid ? gDist : nullptr;
}
} // namespace renderer_gi
