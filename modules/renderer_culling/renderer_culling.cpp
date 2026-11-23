// Light culling for the renderer core: Z-bins on the CPU (the visible lights come sorted by view
// depth; each bin keeps the first and last light that reaches it), then a compute pass that writes
// one bitmask of lights per screen tile, tested against the tile's depth range from the prepass. The core's decals are
// binned the same way, into their own masks and bins (any order: each bin keeps its first and last decal).
#include "yea_renderer_culling.h"
#include "frame.h"

#include <Common_3/Graphics/Interfaces/IGraphics.h>
#include <Common_3/Resources/ResourceLoader/Interfaces/IResourceLoader.h>
#include <Common_3/Graphics/FSL/defaults.h>
#define SAMPLE_COUNT 1
#include "Shaders/Cull.srt.h"
#undef SAMPLE_COUNT

#include <math.h>

namespace renderer_culling
{
namespace
{
struct CullData
{
    float    view[3][4];
    float    proj[4];
    uint32_t size[4];
    uint32_t counts[4];
};

const uint32_t kMaxFrames = 4;
const uint32_t kMaxBins = 8192;

Renderer*      gRenderer;
Queue*         gQueue;
uint32_t       gFrames;
Shader*        gShader[4];
Pipeline*      gPipe[4];
DescriptorSet* gSet;
Buffer*        gCb[kMaxFrames];
Buffer*        gBins[kMaxFrames];
Buffer*        gDecalBins[kMaxFrames];
Buffer*        gNoDecals; // stands in for the core's decal spheres while it has none
Buffer*        gMasks, *gSideMasks, *gDecalMasks; // gSideMasks: by the tiles' sides alone (frame v27 sideMasks)
uint64_t       gDecalMaskBytes;
uint64_t       gMaskBytes;
ResourceState  gMaskState;
bool           gReady;
Buffer*        gDecalSpheres; // this frame's (cull_prepare)
uint32_t       gBinsCpu[kMaxBins]; // built here, then copied once into the mapped buffer
uint32_t       gDecalBinsCpu[kMaxBins];
const uint32_t kDecalWords = RENDER_MAX_DECALS / 32;

uint32_t sampleIndex(uint32_t s) { return s == 8 ? 3 : s == 4 ? 2 : s == 2 ? 1 : 0; }

Buffer* makeBuffer(uint64_t size, uint32_t stride, uint32_t descriptors, ResourceMemoryUsage usage, ResourceState start, const char* name)
{
    Buffer*        b = nullptr;
    BufferLoadDesc d = {};
    d.mDesc.mDescriptors = (DescriptorType)descriptors;
    d.mDesc.mMemoryUsage = usage;
    d.mDesc.mSize = size;
    d.mDesc.pName = name;
    d.mDesc.mStartState = start;
    if (usage != RESOURCE_MEMORY_USAGE_GPU_ONLY)
        d.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
    if (stride)
    {
        d.mDesc.mStructStride = stride;
        d.mDesc.mElementCount = (uint32_t)(size / stride);
    }
    d.ppBuffer = &b;
    addResource(&d, nullptr);
    return b;
}

uint32_t snapTile(int32_t t) { return t >= 32 ? 32 : t >= 16 ? 16 : 8; }
} // namespace

int load(int reloaded)
{
    const YeaHost* host = yea::host();
    gRenderer = (Renderer*)host->renderer;
    gQueue = (Queue*)host->queue;
    gFrames = host->framesInFlight < kMaxFrames ? host->framesInFlight : kMaxFrames;
    const char* names[4] = { "renderer_cull_1.comp", "renderer_cull_2.comp", "renderer_cull_4.comp", "renderer_cull_8.comp" };
    for (int i = 0; i < 4; ++i)
    {
        ShaderLoadDesc sd = {};
        sd.mComp.pFileName = names[i];
        addShader(gRenderer, &sd, &gShader[i]);
        if (!gShader[i])
            return 1;
        PipelineDesc desc = {};
        desc.mType = PIPELINE_TYPE_COMPUTE;
        PIPELINE_LAYOUT_DESC(desc, NULL, SRT_LAYOUT_DESC(CullSrt, PerFrame), NULL, NULL);
        desc.mComputeDesc.pShaderProgram = gShader[i];
        desc.pName = "renderer_cull";
        addPipeline(gRenderer, &desc, &gPipe[i]);
    }
    DescriptorSetDesc setDesc = SRT_SET_DESC(CullSrt, PerFrame, gFrames, 0);
    addDescriptorSet(gRenderer, &setDesc, &gSet);
    for (uint32_t i = 0; i < gFrames; ++i)
    {
        gCb[i] = makeBuffer(256, 0, DESCRIPTOR_TYPE_UNIFORM_BUFFER, RESOURCE_MEMORY_USAGE_CPU_TO_GPU,
                            RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER, "renderer_cull_cb");
        gBins[i] = makeBuffer(kMaxBins * 4, 4, DESCRIPTOR_TYPE_BUFFER, RESOURCE_MEMORY_USAGE_CPU_TO_GPU, RESOURCE_STATE_SHADER_RESOURCE,
                              "renderer_zbins");
        gDecalBins[i] = makeBuffer(kMaxBins * 4, 4, DESCRIPTOR_TYPE_BUFFER, RESOURCE_MEMORY_USAGE_CPU_TO_GPU, RESOURCE_STATE_SHADER_RESOURCE,
                                   "renderer_decal_zbins");
    }
    gNoDecals = makeBuffer(32, 16, DESCRIPTOR_TYPE_BUFFER, RESOURCE_MEMORY_USAGE_CPU_TO_GPU, RESOURCE_STATE_SHADER_RESOURCE, "renderer_cull_no_decals");
    waitForAllResourceLoads();
    gReady = true;
    return 0;
}

void unload()
{
    if (!gRenderer)
        return;
    waitQueueIdle(gQueue);
    for (int i = 0; i < 4; ++i)
    {
        if (gPipe[i])
            removePipeline(gRenderer, gPipe[i]);
        if (gShader[i])
            removeShader(gRenderer, gShader[i]);
    }
    for (uint32_t i = 0; i < gFrames; ++i)
    {
        if (gCb[i])
            removeResource(gCb[i]);
        if (gBins[i])
            removeResource(gBins[i]);
        if (gDecalBins[i])
            removeResource(gDecalBins[i]);
        gBins[i] = gDecalBins[i] = gCb[i] = nullptr;
    }
    if (gMasks)
        removeResource(gMasks);
    if (gSideMasks)
        removeResource(gSideMasks);
    if (gDecalMasks)
        removeResource(gDecalMasks);
    if (gNoDecals)
        removeResource(gNoDecals);
    gMasks = gSideMasks = gDecalMasks = gNoDecals = nullptr;
    gMaskBytes = gDecalMaskBytes = 0;
    if (gSet)
        removeDescriptorSet(gRenderer, gSet);
}

void update(uint32_t phase, float dt) {}
void draw(uint32_t pass, const YeaFrame* frame) {}

// Frame versions (6.1.0): this module's frame layout, for the core; and whether the core's is the same (an older or newer
// core, cached or reverted, is left alone: no frame read, no GPU work), logged once a change.
uint32_t frame_version() { return RENDER_FRAME_VERSION; }
bool coreMatches()
{
    static uint32_t logged; // the core's version last logged as different (0: none)
    const uint32_t  core = renderer::frame_version();
    if (core == RENDER_FRAME_VERSION)
    {
        logged = 0;
        return true;
    }
    if (logged != core + 1)
    {
        char line[160];
        snprintf(line, sizeof line, "renderer_culling: the core's frame is v%u, this module's v%u: standing aside", core, RENDER_FRAME_VERSION);
        yea::log(line);
        logged = core + 1;
    }
    return false;
}

bool cull_prepare(uint64_t address)
{
    if (!coreMatches())
        return false;
    RenderFrame* F = (RenderFrame*)(uintptr_t)address;
    if (!gReady || !settings.enabled || !F || F->version != RENDER_FRAME_VERSION)
        return false;
    if (yea::viewMode()) // F1's Wireframe or Overdraw: stand aside so they read right
        return false;
    uint32_t maxLights = settings.max_lights < 16 ? 16 : (settings.max_lights > 4096 ? 4096 : (uint32_t)settings.max_lights);
    if (F->lightCount > maxLights)
        F->lightCount = maxLights; // the nearest ones stay
    uint32_t ts = snapTile(settings.tile_size);
    uint32_t tilesX = (F->width + ts - 1) / ts, tilesY = (F->height + ts - 1) / ts;
    uint32_t words = (maxLights + 31) / 32;
    uint64_t bytes = (uint64_t)tilesX * tilesY * words * 4;
    uint64_t decalBytes = (uint64_t)tilesX * tilesY * kDecalWords * 4;
    if (bytes > gMaskBytes || decalBytes > gDecalMaskBytes)
    {
        waitQueueIdle(gQueue);
        if (gMasks)
            removeResource(gMasks);
        if (gSideMasks)
            removeResource(gSideMasks);
        if (gDecalMasks)
            removeResource(gDecalMasks);
        gMaskBytes = bytes + bytes / 4;
        gDecalMaskBytes = decalBytes + decalBytes / 4;
        gMasks = makeBuffer(gMaskBytes, 4, DESCRIPTOR_TYPE_BUFFER | DESCRIPTOR_TYPE_RW_BUFFER, RESOURCE_MEMORY_USAGE_GPU_ONLY,
                            RESOURCE_STATE_SHADER_RESOURCE, "renderer_tile_masks");
        gSideMasks = makeBuffer(gMaskBytes, 4, DESCRIPTOR_TYPE_BUFFER | DESCRIPTOR_TYPE_RW_BUFFER, RESOURCE_MEMORY_USAGE_GPU_ONLY,
                                RESOURCE_STATE_SHADER_RESOURCE, "renderer_side_masks");
        gDecalMasks = makeBuffer(gDecalMaskBytes, 4, DESCRIPTOR_TYPE_BUFFER | DESCRIPTOR_TYPE_RW_BUFFER, RESOURCE_MEMORY_USAGE_GPU_ONLY,
                                 RESOURCE_STATE_SHADER_RESOURCE, "renderer_decal_masks");
        waitForAllResourceLoads();
        gMaskState = RESOURCE_STATE_SHADER_RESOURCE;
        if (!gMasks || !gSideMasks || !gDecalMasks)
        {
            gMaskBytes = gDecalMaskBytes = 0;
            return false;
        }
    }

    // Z-bins: linear in view depth up to the farthest light or decal.
    uint32_t bins = settings.z_bins < 64 ? 64 : (settings.z_bins > (int32_t)kMaxBins ? kMaxBins : (uint32_t)settings.z_bins);
    float    zMax = 1e-3f;
    for (uint32_t i = 0; i < F->lightCount; ++i)
    {
        float z = F->lights[i].viewZ + F->lights[i].sphere[3];
        zMax = z > zMax ? z : zMax;
    }
    const uint32_t decals = F->decalSpheres && F->decalSphereBuffer ? (F->decalCount < RENDER_MAX_DECALS ? F->decalCount : RENDER_MAX_DECALS) : 0;
    float          decalZ[RENDER_MAX_DECALS];
    for (uint32_t i = 0; i < decals; ++i)
    {
        const float* c = F->decalSpheres[i];
        decalZ[i] = F->view[2][0] * c[0] + F->view[2][1] * c[1] + F->view[2][2] * c[2] + F->view[2][3];
        zMax = decalZ[i] + c[3] > zMax ? decalZ[i] + c[3] : zMax;
    }
    float     scale = (float)bins / zMax;
    uint32_t* out = gBinsCpu;
    for (uint32_t b = 0; b < bins; ++b)
        out[b] = 0xFFFFu; // first 0xFFFF, last 0: empty
    for (uint32_t i = 0; i < F->lightCount; ++i)
    {
        const RenderLight& l = F->lights[i];
        float z0 = (l.viewZ - l.sphere[3]) * scale, z1 = (l.viewZ + l.sphere[3]) * scale;
        int32_t b0 = z0 < 0 ? 0 : (int32_t)z0, b1 = z1 < 0 ? 0 : (int32_t)z1;
        b0 = b0 >= (int32_t)bins ? (int32_t)bins - 1 : b0;
        b1 = b1 >= (int32_t)bins ? (int32_t)bins - 1 : b1;
        for (int32_t b = b0; b <= b1; ++b)
        {
            uint32_t first = out[b] & 0xFFFFu, last = out[b] >> 16;
            first = i < first ? i : first;
            last = i > last ? i : last;
            out[b] = first | (last << 16);
        }
    }

    for (uint32_t b = 0; b < bins; ++b)
        gDecalBinsCpu[b] = 0xFFFFu;
    for (uint32_t i = 0; i < decals; ++i)
    {
        float   r = F->decalSpheres[i][3];
        float   z0 = (decalZ[i] - r) * scale, z1 = (decalZ[i] + r) * scale;
        int32_t b0 = z0 < 0 ? 0 : (int32_t)z0, b1 = z1 < 0 ? 0 : (int32_t)z1;
        b0 = b0 >= (int32_t)bins ? (int32_t)bins - 1 : b0;
        b1 = b1 >= (int32_t)bins ? (int32_t)bins - 1 : b1;
        for (int32_t b = b0; b <= b1; ++b)
        {
            uint32_t first = gDecalBinsCpu[b] & 0xFFFFu, last = gDecalBinsCpu[b] >> 16;
            first = i < first ? i : first;
            last = i > last ? i : last;
            gDecalBinsCpu[b] = first | (last << 16);
        }
    }

    memcpy(gBins[F->frameIndex]->pCpuMappedAddress, gBinsCpu, (size_t)bins * 4);
    memcpy(gDecalBins[F->frameIndex]->pCpuMappedAddress, gDecalBinsCpu, (size_t)bins * 4);

    CullData* cd = (CullData*)gCb[F->frameIndex]->pCpuMappedAddress;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c)
            cd->view[r][c] = F->view[r][c];
    cd->proj[0] = F->p00;
    cd->proj[1] = F->p11;
    cd->proj[2] = F->nearZ;
    cd->size[0] = F->width;
    cd->size[1] = F->height;
    cd->size[2] = ts;
    cd->size[3] = tilesX;
    cd->counts[0] = F->lightCount;
    cd->counts[1] = words;
    cd->counts[2] = decals;

    F->tileMasks = gMasks;
    F->sideMasks = gSideMasks;
    F->zbins = gBins[F->frameIndex];
    F->tileSize = ts;
    F->tilesX = tilesX;
    F->tilesY = tilesY;
    F->wordsPerTile = words;
    F->binCount = bins;
    F->binScale = scale;
    F->heatmap = settings.heatmap;
    F->decalMasks = gDecalMasks;
    F->decalBins = gDecalBins[F->frameIndex];
    gDecalSpheres = decals ? F->decalSphereBuffer : gNoDecals;
    return true;
}

void cull_run(uint64_t address)
{
    if (!coreMatches())
        return;
    RenderFrame* F = (RenderFrame*)(uintptr_t)address;
    if (!gReady || !F || F->tileMasks != gMasks)
        return;
    Cmd* cmd = F->cmd;
    DescriptorData p[7] = {};
    p[0].mIndex = SRT_RES_IDX(CullSrt, PerFrame, gCull);
    p[0].ppBuffers = &gCb[F->frameIndex];
    p[1].mIndex = SRT_RES_IDX(CullSrt, PerFrame, gLights);
    p[1].ppBuffers = &F->lightBuffer;
    p[2].mIndex = SRT_RES_IDX(CullSrt, PerFrame, gDepth);
    p[2].ppTextures = &F->depth->pTexture;
    p[3].mIndex = SRT_RES_IDX(CullSrt, PerFrame, gTileMasks);
    p[3].ppBuffers = &gMasks;
    p[4].mIndex = SRT_RES_IDX(CullSrt, PerFrame, gDecals);
    p[4].ppBuffers = &gDecalSpheres;
    p[5].mIndex = SRT_RES_IDX(CullSrt, PerFrame, gDecalMasks);
    p[5].ppBuffers = &gDecalMasks;
    p[6].mIndex = SRT_RES_IDX(CullSrt, PerFrame, gSideMasks);
    p[6].ppBuffers = &gSideMasks;
    updateDescriptorSet(gRenderer, F->frameIndex, gSet, 7, p);

    cmdBeginDebugMarker(cmd, 0.3f, 1.0f, 0.5f, "Renderer light culling");
    BufferBarrier b[3] = { { gMasks, gMaskState, RESOURCE_STATE_UNORDERED_ACCESS }, { gDecalMasks, gMaskState, RESOURCE_STATE_UNORDERED_ACCESS },
                           { gSideMasks, gMaskState, RESOURCE_STATE_UNORDERED_ACCESS } };
    cmdResourceBarrier(cmd, 3, b, 0, nullptr, 0, nullptr);
    cmdBindPipeline(cmd, gPipe[sampleIndex(F->sampleCount)]);
    cmdBindDescriptorSet(cmd, F->frameIndex, gSet);
    cmdDispatch(cmd, F->tilesX, F->tilesY, 1);
    b[0] = { gMasks, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_SHADER_RESOURCE };
    b[1] = { gDecalMasks, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_SHADER_RESOURCE };
    b[2] = { gSideMasks, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_SHADER_RESOURCE };
    cmdResourceBarrier(cmd, 3, b, 0, nullptr, 0, nullptr);
    gMaskState = RESOURCE_STATE_SHADER_RESOURCE;
    cmdEndDebugMarker(cmd);
}
} // namespace renderer_culling
