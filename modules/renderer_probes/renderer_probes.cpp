// Reflection probes for the renderer core. A probe is captured as a cube (6 faces of the scene, lit
// diffusely, alpha = coverage), halved level by level, then prefiltered into one slice of an octahedral
// 2D array: mip m holds the GGX lobe of roughness m / (mips - 1). The forward pass blends the probes whose
// boxes hold the pixel, projects reflections onto those boxes (parallax), and lets the live sky show
// through uncovered texels. Materials with reflections off skip them.
// Dirty probes are captured first; `updates_per_frame` more are re-captured in turn so moving things show.
#include "yea_renderer_probes.h"
#include "frame.h"

#include <Common_3/Graphics/Interfaces/IGraphics.h>
#include <Common_3/Resources/ResourceLoader/Interfaces/IResourceLoader.h>
#include <Common_3/Graphics/FSL/defaults.h>
#include "Shaders/Capture.srt.h"
#include "Shaders/Filter.srt.h"

#include <math.h>
#include <string.h>

namespace renderer_probes
{
namespace
{
struct CaptureData
{
    float    sunDir[4];
    float    sunColor[4];
    float    skyZenith[4];
    float    skyHorizon[4];
    float    skyGround[4];
    uint32_t counts[4];
};
struct CaptureView
{
    float row[4][4];
    float eye[4];
};
struct PassData
{
    uint32_t info[4];
    float    params[4];
};
struct Run
{
    uint32_t firstIndex, indexCount;
    int32_t  vertexOffset;
    uint32_t streamStart, count;
};

const uint32_t kMaxFrames = 4, kMaxProbes = 16, kMaxPerFrame = 4, kMaxLevels = 8;
const uint32_t kMaxEntries = 1u << 18, kMaxRuns = 4096, kMaxCaptureLights = 64;
const uint32_t kPrefilterSets = kMaxLevels * 6; // set index of mip 0's prefilter; downsample sets before it
const uint32_t kPassSets = kMaxLevels * 6 + kMaxLevels;
const float    kNear = 0.05f;

Renderer*      gRenderer;
Queue*         gQueue;
uint32_t       gFrames;
Shader *       gCaptureShader, *gDownShader, *gFilterShader;
Pipeline *     gCapturePipe, *gDownPipe, *gFilterPipe;
DescriptorSet *gCaptureSet, *gPassSet;
Buffer *       gCaptureCb[kMaxFrames], *gViewBuf[kMaxFrames], *gStreamBuf[kMaxFrames], *gProbeBuf[kMaxFrames];
Buffer*        gPassCb[kPassSets];
RenderTarget * gChain[kMaxLevels], *gCaptureDepth, *gProbeRt;
uint32_t       gOcta, gLevels; // probe image size and its mips (also the capture levels)
bool           gReady;

uint64_t gCapturedAt[kMaxProbes]; // frame each probe slot was captured in (0: never)
uint32_t gCapturedGen[kMaxProbes];
uint32_t gNext;                   // round robin
uint32_t gTodo[kMaxPerFrame], gTodoCount;
uint32_t gRunStart[kMaxPerFrame * 6 + 1];

// Built here, copied once into mapped memory.
uint32_t*        gStreamCpu;
Run*             gRuns;
CaptureView      gViewsCpu[kMaxPerFrame * 6];
RenderGpuProbe gProbeCpu[kMaxProbes];

// Cube faces +X -X +Y -Y +Z -Z: forward, right, up (Shaders/Filter.frag.fsl faceDir).
const float kFace[6][3][3] = {
    { { 1, 0, 0 }, { 0, 0, -1 }, { 0, 1, 0 } },  { { -1, 0, 0 }, { 0, 0, 1 }, { 0, 1, 0 } },
    { { 0, 1, 0 }, { 1, 0, 0 }, { 0, 0, -1 } },  { { 0, -1, 0 }, { 1, 0, 0 }, { 0, 0, 1 } },
    { { 0, 0, 1 }, { 1, 0, 0 }, { 0, 1, 0 } },   { { 0, 0, -1 }, { -1, 0, 0 }, { 0, 1, 0 } },
};

// --- Probe rows -----------------------------------------------------------------------------------
uint32_t rowOf(yea_handle h)
{
    uint32_t row = h & 0xFFFFFu;
    if (row >= Probes.size() || !Probes[row].alive || (Probes[row].gen & 0xFFFu) != (h >> 20))
        return 0xFFFFFFFFu;
    return row;
}

void setProbe(Probe& p, yea_float3 pos, yea_float3 lo, yea_float3 hi)
{
    p.pos = pos;
    p.boxMin = { lo.x < hi.x ? lo.x : hi.x, lo.y < hi.y ? lo.y : hi.y, lo.z < hi.z ? lo.z : hi.z };
    p.boxMax = { lo.x < hi.x ? hi.x : lo.x, lo.y < hi.y ? hi.y : lo.y, lo.z < hi.z ? hi.z : lo.z };
    p.changed = yea::host()->frame();
}

// --- GPU objects ------------------------------------------------------------------------------------
uint32_t snapResolution(int32_t r) { return r >= 512 ? 512 : r >= 256 ? 256 : r >= 128 ? 128 : 64; }

Shader* loadShader(const char* vert, const char* frag)
{
    ShaderLoadDesc d = {};
    d.mVert.pFileName = vert;
    d.mFrag.pFileName = frag;
    Shader* s = nullptr;
    addShader(gRenderer, &d, &s);
    return s;
}

Buffer* makeBuffer(uint64_t size, uint32_t stride, DescriptorType type, ResourceState start, const char* name)
{
    Buffer*        b = nullptr;
    BufferLoadDesc d = {};
    d.mDesc.mDescriptors = type;
    d.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
    d.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
    d.mDesc.mSize = size;
    d.mDesc.mStartState = start;
    d.mDesc.pName = name;
    if (stride)
    {
        d.mDesc.mStructStride = stride;
        d.mDesc.mElementCount = (uint32_t)(size / stride);
    }
    d.ppBuffer = &b;
    addResource(&d, nullptr);
    return b;
}

void freeTargets()
{
    for (RenderTarget*& t : gChain)
        if (t)
        {
            removeRenderTarget(gRenderer, t);
            t = nullptr;
        }
    if (gCaptureDepth)
        removeRenderTarget(gRenderer, gCaptureDepth);
    if (gProbeRt)
        removeRenderTarget(gRenderer, gProbeRt);
    gCaptureDepth = gProbeRt = nullptr;
    gOcta = gLevels = 0;
    memset(gCapturedAt, 0, sizeof gCapturedAt);
}

void createTargets(uint32_t octa)
{
    uint32_t levels = 0;
    for (uint32_t s = octa; s >= 8 && levels < kMaxLevels; s >>= 1)
        levels++;
    uint32_t face = octa / 2;

    RenderTargetDesc d = {};
    d.mDepth = d.mMipLevels = 1;
    d.mSampleCount = SAMPLE_COUNT_1;
    d.mFormat = TinyImageFormat_R16G16B16A16_SFLOAT;
    d.mStartState = RESOURCE_STATE_SHADER_RESOURCE;
    d.mClearValue = { { 0, 0, 0, 0 } };
    d.mArraySize = 6;
    d.mDescriptors = (DescriptorType)(DESCRIPTOR_TYPE_TEXTURE_CUBE | DESCRIPTOR_TYPE_RENDER_TARGET_ARRAY_SLICES);
    d.pName = "renderer_probe_capture";
    for (uint32_t k = 0; k < levels; ++k)
    {
        d.mWidth = d.mHeight = face >> k;
        addRenderTarget(gRenderer, &d, &gChain[k]);
    }
    d.mWidth = d.mHeight = octa;
    d.mArraySize = kMaxProbes;
    d.mMipLevels = levels;
    d.mDescriptors =
        (DescriptorType)(DESCRIPTOR_TYPE_TEXTURE | DESCRIPTOR_TYPE_RENDER_TARGET_ARRAY_SLICES | DESCRIPTOR_TYPE_RENDER_TARGET_MIP_SLICES);
    d.pName = "renderer_probes";
    addRenderTarget(gRenderer, &d, &gProbeRt);

    RenderTargetDesc z = {};
    z.mWidth = z.mHeight = face;
    z.mDepth = z.mArraySize = z.mMipLevels = 1;
    z.mSampleCount = SAMPLE_COUNT_1;
    z.mFormat = TinyImageFormat_D32_SFLOAT;
    z.mStartState = RESOURCE_STATE_DEPTH_WRITE;
    z.mClearValue.depth = 0.0f; // reversed-Z
    z.pName = "renderer_probe_depth";
    addRenderTarget(gRenderer, &z, &gCaptureDepth);
    gOcta = octa;
    gLevels = levels;

    // The downsample and prefilter passes never change until the next resize: their constants and
    // descriptor sets are written once here (the GPU is idle).
    for (uint32_t k = 1; k < levels; ++k)
        for (uint32_t f = 0; f < 6; ++f)
        {
            uint32_t idx = (k - 1) * 6 + f;
            PassData p = {};
            p.info[0] = f;
            p.info[2] = k;
            memcpy(gPassCb[idx]->pCpuMappedAddress, &p, sizeof p);
            DescriptorData dd[2] = {};
            dd[0].mIndex = SRT_RES_IDX(FilterSrt, PerFrame, gPass);
            dd[0].ppBuffers = &gPassCb[idx];
            dd[1].mIndex = SRT_RES_IDX(FilterSrt, PerFrame, gSource);
            dd[1].ppTextures = &gChain[k - 1]->pTexture;
            updateDescriptorSet(gRenderer, idx, gPassSet, 2, dd);
        }
    for (uint32_t m = 0; m < levels; ++m)
    {
        uint32_t idx = kPrefilterSets + m;
        float    r = levels > 1 ? (float)m / (float)(levels - 1) : 0.0f;
        PassData p = {};
        p.info[2] = m;
        p.params[0] = r;
        p.params[1] = r * r;
        p.params[2] = m == 0 ? 1.0f : 48.0f;
        memcpy(gPassCb[idx]->pCpuMappedAddress, &p, sizeof p);
        DescriptorData dd[2] = {};
        dd[0].mIndex = SRT_RES_IDX(FilterSrt, PerFrame, gPass);
        dd[0].ppBuffers = &gPassCb[idx];
        dd[1].mIndex = SRT_RES_IDX(FilterSrt, PerFrame, gSource);
        dd[1].ppTextures = &gChain[m]->pTexture;
        updateDescriptorSet(gRenderer, idx, gPassSet, 2, dd);
    }
}

Pipeline* fullscreenPipeline(Shader* s, const char* name)
{
    RasterizerStateDesc raster = {};
    raster.mCullMode = CULL_MODE_NONE;
    DepthStateDesc  depth = {};
    TinyImageFormat format = TinyImageFormat_R16G16B16A16_SFLOAT;
    PipelineDesc    desc = {};
    desc.mType = PIPELINE_TYPE_GRAPHICS;
    PIPELINE_LAYOUT_DESC(desc, NULL, SRT_LAYOUT_DESC(FilterSrt, PerFrame), NULL, NULL);
    GraphicsPipelineDesc& g = desc.mGraphicsDesc;
    g.mPrimitiveTopo = PRIMITIVE_TOPO_TRI_LIST;
    g.mRenderTargetCount = 1;
    g.pColorFormats = &format;
    g.mSampleCount = SAMPLE_COUNT_1;
    g.mDepthStencilFormat = TinyImageFormat_UNDEFINED;
    g.pShaderProgram = s;
    g.pRasterizerState = &raster;
    g.pDepthState = &depth;
    desc.pName = name;
    Pipeline* p = nullptr;
    addPipeline(gRenderer, &desc, &p);
    return p;
}

Pipeline* capturePipeline()
{
    VertexLayout l = {};
    l.mBindingCount = 2;
    l.mBindings[0].mStride = 24; // RenderFrame::vertexStride: float3 position, float3 normal
    l.mBindings[1].mStride = 4;
    l.mBindings[1].mRate = VERTEX_BINDING_RATE_INSTANCE;
    l.mAttribCount = 3;
    l.mAttribs[0] = {};
    l.mAttribs[0].mSemantic = SEMANTIC_POSITION;
    l.mAttribs[0].mFormat = TinyImageFormat_R32G32B32_SFLOAT;
    l.mAttribs[1] = {};
    l.mAttribs[1].mSemantic = SEMANTIC_NORMAL;
    l.mAttribs[1].mFormat = TinyImageFormat_R32G32B32_SFLOAT;
    l.mAttribs[1].mLocation = 1;
    l.mAttribs[1].mOffset = 12;
    l.mAttribs[2] = {};
    l.mAttribs[2].mSemantic = SEMANTIC_TEXCOORD1;
    l.mAttribs[2].mFormat = TinyImageFormat_R32_UINT;
    l.mAttribs[2].mBinding = 1;
    l.mAttribs[2].mLocation = 2;

    RasterizerStateDesc raster = {};
    raster.mCullMode = CULL_MODE_NONE;
    DepthStateDesc depth = {};
    depth.mDepthTest = true;
    depth.mDepthWrite = true;
    depth.mDepthFunc = CMP_GEQUAL;
    TinyImageFormat format = TinyImageFormat_R16G16B16A16_SFLOAT;
    PipelineDesc    desc = {};
    desc.mType = PIPELINE_TYPE_GRAPHICS;
    PIPELINE_LAYOUT_DESC(desc, NULL, SRT_LAYOUT_DESC(CaptureSrt, PerFrame), NULL, NULL);
    GraphicsPipelineDesc& g = desc.mGraphicsDesc;
    g.mPrimitiveTopo = PRIMITIVE_TOPO_TRI_LIST;
    g.mRenderTargetCount = 1;
    g.pColorFormats = &format;
    g.mSampleCount = SAMPLE_COUNT_1;
    g.mDepthStencilFormat = TinyImageFormat_D32_SFLOAT;
    g.pShaderProgram = gCaptureShader;
    g.pRasterizerState = &raster;
    g.pDepthState = &depth;
    g.pVertexLayout = &l;
    desc.pName = "renderer_probe_capture";
    Pipeline* p = nullptr;
    addPipeline(gRenderer, &desc, &p);
    return p;
}

void barrierRt(Cmd* cmd, RenderTarget* rt, ResourceState from, ResourceState to)
{
    RenderTargetBarrier b = { rt, from, to };
    cmdResourceBarrier(cmd, 0, nullptr, 0, nullptr, 1, &b);
}

void bindSlice(Cmd* cmd, RenderTarget* rt, uint32_t slice, uint32_t mip, LoadActionType load, RenderTarget* depth)
{
    BindRenderTargetsDesc bind = {};
    bind.mRenderTargetCount = 1;
    bind.mRenderTargets[0] = { rt, load };
    bind.mRenderTargets[0].mArraySlice = slice;
    bind.mRenderTargets[0].mUseArraySlice = 1;
    bind.mRenderTargets[0].mMipSlice = mip;
    bind.mRenderTargets[0].mUseMipSlice = mip ? 1 : 0;
    if (depth)
    {
        bind.mDepthStencil = { depth, LOAD_ACTION_CLEAR };
        bind.mDepthStencil.mStoreAction = STORE_ACTION_DONTCARE;
    }
    cmdBindRenderTargets(cmd, &bind);
    uint32_t size = (rt->mWidth >> mip) ? (rt->mWidth >> mip) : 1;
    cmdSetViewport(cmd, 0, 0, (float)size, (float)size, 0, 1);
    cmdSetScissor(cmd, 0, 0, size, size);
}
} // namespace

// --- Exports ----------------------------------------------------------------------------------------
yea_handle probe_add(yea_float3 pos, yea_float3 box_min, yea_float3 box_max)
{
    uint32_t row = 0xFFFFFFFFu;
    for (uint32_t i = 0; i < Probes.size(); ++i)
        if (!Probes[i].alive)
        {
            uint32_t gen = Probes[i].gen + 1;
            Probes[i] = Probe{};
            Probes[i].gen = gen;
            row = i;
            break;
        }
    if (row == 0xFFFFFFFFu)
    {
        if (!Probes.add())
            return 0xFFFFFFFFu;
        row = Probes.size() - 1;
    }
    Probe& p = Probes[row];
    p.alive = 1;
    setProbe(p, pos, box_min, box_max);
    return row | ((p.gen & 0xFFFu) << 20);
}

void probe_place(yea_handle h, yea_float3 pos, yea_float3 box_min, yea_float3 box_max)
{
    uint32_t row = rowOf(h);
    if (row != 0xFFFFFFFFu)
        setProbe(Probes[row], pos, box_min, box_max);
}

void probe_remove(yea_handle h)
{
    uint32_t row = rowOf(h);
    if (row != 0xFFFFFFFFu)
        Probes[row].alive = 0;
}

void probe_clear()
{
    for (Probe& p : Probes)
        p.alive = 0;
}

uint32_t probe_count()
{
    uint32_t n = 0;
    for (const Probe& p : Probes)
        n += p.alive;
    return n;
}

// --- Module -----------------------------------------------------------------------------------------
int load(int reloaded)
{
    const YeaHost* host = yea::host();
    gRenderer = (Renderer*)host->renderer;
    gQueue = (Queue*)host->queue;
    gFrames = host->framesInFlight < kMaxFrames ? host->framesInFlight : kMaxFrames;

    gCaptureShader = loadShader("renderer_probe_capture.vert", "renderer_probe_capture.frag");
    gDownShader = loadShader("renderer_probe_fullscreen.vert", "renderer_probe_downsample.frag");
    gFilterShader = loadShader("renderer_probe_fullscreen.vert", "renderer_probe_prefilter.frag");
    if (!gCaptureShader || !gDownShader || !gFilterShader)
        return 1;
    gCapturePipe = capturePipeline();
    gDownPipe = fullscreenPipeline(gDownShader, "renderer_probe_downsample");
    gFilterPipe = fullscreenPipeline(gFilterShader, "renderer_probe_prefilter");

    DescriptorSetDesc captureDesc = SRT_SET_DESC(CaptureSrt, PerFrame, gFrames, 0);
    addDescriptorSet(gRenderer, &captureDesc, &gCaptureSet);
    DescriptorSetDesc passDesc = SRT_SET_DESC(FilterSrt, PerFrame, kPassSets, 0);
    addDescriptorSet(gRenderer, &passDesc, &gPassSet);
    const ResourceState cb = RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER, sr = RESOURCE_STATE_SHADER_RESOURCE;
    for (uint32_t i = 0; i < gFrames; ++i)
    {
        gCaptureCb[i] = makeBuffer(256, 0, DESCRIPTOR_TYPE_UNIFORM_BUFFER, cb, "renderer_probe_capture_cb");
        gViewBuf[i] = makeBuffer(sizeof gViewsCpu, sizeof(CaptureView), DESCRIPTOR_TYPE_BUFFER, sr, "renderer_probe_views");
        gStreamBuf[i] = makeBuffer((uint64_t)kMaxEntries * 4, 0, DESCRIPTOR_TYPE_VERTEX_BUFFER, cb, "renderer_probe_stream");
        gProbeBuf[i] = makeBuffer(sizeof gProbeCpu, sizeof(RenderGpuProbe), DESCRIPTOR_TYPE_BUFFER, sr, "renderer_probe_data");
    }
    for (uint32_t i = 0; i < kPassSets; ++i)
        gPassCb[i] = makeBuffer(256, 0, DESCRIPTOR_TYPE_UNIFORM_BUFFER, cb, "renderer_probe_pass_cb");
    waitForAllResourceLoads();
    gStreamCpu = new uint32_t[kMaxEntries];
    gRuns = new Run[kMaxRuns];
    memset(gCapturedAt, 0, sizeof gCapturedAt);
    gReady = gCapturePipe && gDownPipe && gFilterPipe && gCaptureSet && gPassSet;
    return gReady ? 0 : 1;
}

void unload()
{
    if (!gRenderer)
        return;
    waitQueueIdle(gQueue);
    freeTargets();
    for (uint32_t i = 0; i < gFrames; ++i)
    {
        Buffer* bufs[4] = { gCaptureCb[i], gViewBuf[i], gStreamBuf[i], gProbeBuf[i] };
        for (Buffer* b : bufs)
            if (b)
                removeResource(b);
    }
    for (Buffer* b : gPassCb)
        if (b)
            removeResource(b);
    DescriptorSet* sets[2] = { gCaptureSet, gPassSet };
    for (DescriptorSet* s : sets)
        if (s)
            removeDescriptorSet(gRenderer, s);
    Pipeline* pipes[3] = { gCapturePipe, gDownPipe, gFilterPipe };
    for (Pipeline* p : pipes)
        if (p)
            removePipeline(gRenderer, p);
    Shader* shaders[3] = { gCaptureShader, gDownShader, gFilterShader };
    for (Shader* s : shaders)
        if (s)
            removeShader(gRenderer, s);
}

void update(uint32_t phase, float dt) {}
void draw(uint32_t pass, const YeaFrame* frame) {}

bool probes_prepare(uint64_t address)
{
    RenderFrame* F = (RenderFrame*)(uintptr_t)address;
    gTodoCount = 0;
    if (!gReady || !settings.enabled || !F || F->version != RENDER_FRAME_VERSION || !F->materials || !F->vertices)
        return false;
    uint32_t octa = snapResolution(settings.resolution);
    if (octa != gOcta)
    {
        waitQueueIdle(gQueue);
        freeTargets();
        createTargets(octa);
    }
    if (!gProbeRt || !gCaptureDepth || !gLevels)
        return false;

    // Which to capture: never captured or moved first, then the next in turn. None while F1 shows
    // Wireframe or Overdraw, which would change the capture's pipelines: the probes keep what they had.
    uint32_t rows = Probes.size() < kMaxProbes ? Probes.size() : kMaxProbes;
    uint32_t todoRows = yea::viewMode() ? 0 : rows;
    uint32_t extra = settings.updates_per_frame < 0 ? 0 : (uint32_t)settings.updates_per_frame;
    uint32_t budget = extra > kMaxPerFrame ? kMaxPerFrame : (extra ? extra : 1);
    for (uint32_t i = 0; i < todoRows; ++i)
    {
        const Probe& p = Probes[i];
        if (!p.alive)
        {
            gCapturedAt[i] = 0;
            continue;
        }
        bool dirty = !gCapturedAt[i] || gCapturedGen[i] != p.gen || p.changed >= gCapturedAt[i];
        if (dirty && gTodoCount < budget)
            gTodo[gTodoCount++] = i;
    }
    uint32_t base = gNext;
    for (uint32_t n = 0; n < todoRows && gTodoCount < (extra < kMaxPerFrame ? extra : kMaxPerFrame); ++n)
    {
        uint32_t i = (base + n) % rows;
        bool     queued = false;
        for (uint32_t t = 0; t < gTodoCount; ++t)
            queued |= gTodo[t] == i;
        if (Probes[i].alive && gCapturedAt[i] && !queued)
        {
            gTodo[gTodoCount++] = i;
            gNext = i + 1;
        }
    }

    // Views and the instance stream, face by face; instances outside a face's frustum are left out.
    uint32_t used = 0, runs = 0;
    for (uint32_t t = 0; t < gTodoCount; ++t)
    {
        const Probe& p = Probes[gTodo[t]];
        float        e[3] = { p.pos.x, p.pos.y, p.pos.z };
        for (uint32_t f = 0; f < 6; ++f)
        {
            uint32_t     view = t * 6 + f;
            const float(*b)[3] = kFace[f]; // forward, right, up
            CaptureView& v = gViewsCpu[view];
            const float* rows3[3] = { b[1], b[2], b[0] };
            float        d[3];
            for (int r = 0; r < 3; ++r)
                d[r] = -(rows3[r][0] * e[0] + rows3[r][1] * e[1] + rows3[r][2] * e[2]);
            float m[4][4] = { { b[1][0], b[1][1], b[1][2], d[0] }, // x = right
                              { b[2][0], b[2][1], b[2][2], d[1] }, // y = up
                              { 0, 0, 0, kNear },                  // reversed-Z, infinite far
                              { b[0][0], b[0][1], b[0][2], d[2] } };
            memcpy(v.row, m, sizeof m);
            v.eye[0] = e[0];
            v.eye[1] = e[1];
            v.eye[2] = e[2];
            v.eye[3] = 1;

            gRunStart[view] = runs;
            Run* run = nullptr;
            for (uint32_t i = 0; i < F->drawableCount && used < kMaxEntries; ++i)
            {
                const RenderCaster& c = F->drawables[i];
                float x = c.sphere[0] - e[0], y = c.sphere[1] - e[1], z = c.sphere[2] - e[2], rad = c.sphere[3] * 1.4142136f;
                float fw = b[0][0] * x + b[0][1] * y + b[0][2] * z;
                float rt = b[1][0] * x + b[1][1] * y + b[1][2] * z;
                float up = b[2][0] * x + b[2][1] * y + b[2][2] * z;
                if (fw - fabsf(rt) < -rad || fw - fabsf(up) < -rad)
                    continue; // outside the 90 degree face (planes scaled by sqrt 2)
                if (!run || run->firstIndex != c.firstIndex || run->vertexOffset != c.vertexOffset)
                {
                    if (runs >= kMaxRuns)
                        break;
                    run = &gRuns[runs++];
                    *run = { c.firstIndex, c.indexCount, c.vertexOffset, used, 0 };
                }
                gStreamCpu[used++] = c.instance | (view << RENDER_VIEW_SHIFT);
                run->count++;
            }
        }
        gCapturedAt[gTodo[t]] = F->frameNumber;
        gCapturedGen[gTodo[t]] = p.gen;
    }
    gRunStart[gTodoCount * 6] = runs;

    if (gTodoCount)
    {
        CaptureData cd = {};
        memcpy(cd.sunDir, F->sunDir, sizeof F->sunDir);
        cd.sunDir[3] = F->sunOn ? 1.0f : 0.0f;
        memcpy(cd.sunColor, F->sunColor, sizeof F->sunColor);
        memcpy(cd.skyZenith, F->skyZenith, sizeof F->skyZenith);
        memcpy(cd.skyHorizon, F->skyHorizon, sizeof F->skyHorizon);
        memcpy(cd.skyGround, F->skyGround, sizeof F->skyGround);
        cd.skyGround[3] = F->ambient;
        cd.counts[0] = F->lightBuffer ? (F->lightCount < kMaxCaptureLights ? F->lightCount : kMaxCaptureLights) : 0;
        memcpy(gCaptureCb[F->frameIndex]->pCpuMappedAddress, &cd, sizeof cd);
        memcpy(gViewBuf[F->frameIndex]->pCpuMappedAddress, gViewsCpu, sizeof(CaptureView) * gTodoCount * 6);
        memcpy(gStreamBuf[F->frameIndex]->pCpuMappedAddress, gStreamCpu, (size_t)used * 4);
    }

    // Every probe that has an image (including those captured below this frame).
    uint32_t count = 0;
    for (uint32_t i = 0; i < rows; ++i)
    {
        const Probe& p = Probes[i];
        if (!p.alive || !gCapturedAt[i])
            continue;
        RenderGpuProbe& g = gProbeCpu[count++];
        g = RenderGpuProbe{ { p.pos.x, p.pos.y, p.pos.z, (float)i },
                              { p.boxMin.x, p.boxMin.y, p.boxMin.z, 0 },
                              { p.boxMax.x, p.boxMax.y, p.boxMax.z, 0 } };
    }
    if (!count)
        return false;
    // Smallest box first: the shader lets each probe cover what the ones before it left.
    auto volume = [](const RenderGpuProbe& g) {
        return (g.boxMax[0] - g.boxMin[0]) * (g.boxMax[1] - g.boxMin[1]) * (g.boxMax[2] - g.boxMin[2]);
    };
    for (uint32_t i = 1; i < count; ++i)
        for (uint32_t j = i; j > 0 && volume(gProbeCpu[j]) < volume(gProbeCpu[j - 1]); --j)
        {
            RenderGpuProbe t = gProbeCpu[j];
            gProbeCpu[j] = gProbeCpu[j - 1];
            gProbeCpu[j - 1] = t;
        }
    memcpy(gProbeBuf[F->frameIndex]->pCpuMappedAddress, gProbeCpu, sizeof(RenderGpuProbe) * count);
    F->probes = gProbeRt;
    F->probeData = gProbeBuf[F->frameIndex];
    F->probeCount = count;
    F->probeMips = gLevels;
    F->probeIntensity = settings.intensity < 0 ? 0 : settings.intensity;
    F->probeBlend = settings.blend < 0.01f ? 0.01f : settings.blend;
    F->probeParallax = settings.parallax;
    return true;
}

void probes_render(uint64_t address)
{
    RenderFrame* F = (RenderFrame*)(uintptr_t)address;
    if (!gReady || !F || !gTodoCount || F->probes != gProbeRt || !F->instances || !F->indices)
        return;
    Cmd*     cmd = F->cmd;
    uint32_t fi = F->frameIndex;
    Buffer*  lights = F->lightBuffer ? F->lightBuffer : gProbeBuf[fi]; // unread when there are none
    DescriptorData d[5] = {};
    d[0].mIndex = SRT_RES_IDX(CaptureSrt, PerFrame, gCapture);
    d[0].ppBuffers = &gCaptureCb[fi];
    d[1].mIndex = SRT_RES_IDX(CaptureSrt, PerFrame, gInstances);
    d[1].ppBuffers = &F->instances;
    d[2].mIndex = SRT_RES_IDX(CaptureSrt, PerFrame, gViews);
    d[2].ppBuffers = &gViewBuf[fi];
    d[3].mIndex = SRT_RES_IDX(CaptureSrt, PerFrame, gMaterials);
    d[3].ppBuffers = &F->materials;
    d[4].mIndex = SRT_RES_IDX(CaptureSrt, PerFrame, gLights);
    d[4].ppBuffers = &lights;
    updateDescriptorSet(gRenderer, fi, gCaptureSet, 5, d);

    cmdBeginDebugMarker(cmd, 0.5f, 0.9f, 0.9f, "Renderer reflection probes");
    for (uint32_t t = 0; t < gTodoCount; ++t)
    {
        // The scene into the six faces.
        barrierRt(cmd, gChain[0], RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_RENDER_TARGET);
        for (uint32_t f = 0; f < 6; ++f)
        {
            uint32_t view = t * 6 + f;
            bindSlice(cmd, gChain[0], f, 0, LOAD_ACTION_CLEAR, gCaptureDepth);
            cmdBindPipeline(cmd, gCapturePipe);
            cmdBindDescriptorSet(cmd, fi, gCaptureSet);
            cmdBindIndexBuffer(cmd, F->indices, INDEX_TYPE_UINT32, 0);
            for (uint32_t r = gRunStart[view]; r < gRunStart[view + 1]; ++r)
            {
                const Run& run = gRuns[r];
                Buffer*    bufs[2] = { F->vertices, gStreamBuf[fi] };
                uint32_t   strides[2] = { F->vertexStride, 4 };
                uint64_t   offsets[2] = { 0, (uint64_t)run.streamStart * 4 };
                cmdBindVertexBuffer(cmd, 2, bufs, strides, offsets);
                cmdDrawIndexedInstanced(cmd, run.indexCount, run.firstIndex, run.count, (uint32_t)run.vertexOffset, 0);
            }
            cmdBindRenderTargets(cmd, nullptr);
        }
        barrierRt(cmd, gChain[0], RESOURCE_STATE_RENDER_TARGET, RESOURCE_STATE_SHADER_RESOURCE);

        // Halved level by level.
        for (uint32_t k = 1; k < gLevels; ++k)
        {
            barrierRt(cmd, gChain[k], RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_RENDER_TARGET);
            for (uint32_t f = 0; f < 6; ++f)
            {
                bindSlice(cmd, gChain[k], f, 0, LOAD_ACTION_DONTCARE, nullptr);
                cmdBindPipeline(cmd, gDownPipe);
                cmdBindDescriptorSet(cmd, (k - 1) * 6 + f, gPassSet);
                cmdDraw(cmd, 3, 0);
                cmdBindRenderTargets(cmd, nullptr);
            }
            barrierRt(cmd, gChain[k], RESOURCE_STATE_RENDER_TARGET, RESOURCE_STATE_SHADER_RESOURCE);
        }

        // Prefiltered into the probe's slice, a mip per roughness step.
        barrierRt(cmd, gProbeRt, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_RENDER_TARGET);
        for (uint32_t m = 0; m < gLevels; ++m)
        {
            bindSlice(cmd, gProbeRt, gTodo[t], m, LOAD_ACTION_DONTCARE, nullptr);
            cmdBindPipeline(cmd, gFilterPipe);
            cmdBindDescriptorSet(cmd, kPrefilterSets + m, gPassSet);
            cmdDraw(cmd, 3, 0);
            cmdBindRenderTargets(cmd, nullptr);
        }
        barrierRt(cmd, gProbeRt, RESOURCE_STATE_RENDER_TARGET, RESOURCE_STATE_SHADER_RESOURCE);
    }
    cmdEndDebugMarker(cmd);
}
} // namespace renderer_probes
