// GPU particles for the renderer core: emitters the game places, emitted, simulated (gravity, drag, divergence-free
// turbulence), lit (the sun through its cascade, the sky's ambient, the local lights through their shadow maps, with a
// Henyey-Greenstein phase) and sorted back to front (bitonic) in compute, then drawn after the forward pass at render
// size as soft camera-facing sprites, behind the volumetric fog in front of them, before TAA or the upscaler, into the
// core's see-through pass (frame.h blendRuns): the sorted list split on the GPU at each blended surface's depth, each run
// drawn indirectly just before its surface. Nothing runs while no emitter exists and no particle is alive.
#include "yea_renderer_particles.h"
#include "frame.h"

#include <Common_3/Graphics/Interfaces/IGraphics.h>
#include <Common_3/Resources/ResourceLoader/Interfaces/IResourceLoader.h>
#include <Common_3/Graphics/FSL/defaults.h>
#include "Shaders/Particles.srt.h"
#include "Shaders/Draw.srt.h"

#include <math.h>
#include <string.h>

namespace renderer_particles
{
namespace
{
struct PartFrame
{
    float    view[3][4];
    float    camPos[4], proj[4], screen[4], soft[4], sunDir[4], sunColor[4], cascadeEnd[4], ambient[4], texel[4], fog[4], fog2[4];
    uint32_t counts[4], spawn[4];
    float    gi[4][4];
    uint32_t runs[4];
    float    blend[RENDER_MAX_BLEND_RUNS];
};
struct GpuEmitter
{
    float pos[4], extent[4], dir[4], motion[4], colorStart[4], colorEnd[4], size[4], life[4];
};
struct GpuParticle
{
    float    pos[4], vel[4];
    uint32_t info[4];
};

const uint32_t kMaxFrames = 4, kSlots = 40, kMaxEmitters = 64, kRing = 8192;
enum Pass { kInit, kEmit, kSimulate, kPad, kSortLocal, kSortGlobal, kSplit, kPasses };
const char* const kShaderNames[kPasses] = { "renderer_particles_init.comp", "renderer_particles_emit.comp", "renderer_particles_simulate.comp",
                                            "renderer_particles_pad.comp", "renderer_particles_sort_local.comp",
                                            "renderer_particles_sort_global.comp", "renderer_particles_split.comp" };
const uint32_t kRuns = RENDER_MAX_BLEND_RUNS + 1;

Renderer*      gRenderer;
Queue*         gQueue;
uint32_t       gFrames;
Shader*        gShader[kPasses], *gDrawShader;
Pipeline*      gPipe[kPasses], *gDrawPipe;
TinyImageFormat gDrawFormat;
Sampler*       gShadowCmp;
DescriptorSet* gPersistent, *gSet, *gDrawSet;
Buffer*        gFrameCb[kMaxFrames], *gStepCb[kMaxFrames * kSlots], *gEmitterBuf[kMaxFrames], *gSpawnBuf[kMaxFrames];
Buffer*        gDummyViews, *gDummyBins, *gDummySh, *gDummyLights, *gDummyShapes;
RenderTarget*  gDummyAtlas;
Texture*       gDummyFog;
Texture*       gNoGi;
Texture*       gGiIrr;  // this frame's, or gNoGi
Texture*       gGiDist;
Buffer*        gGiProbes;
// The pool: particles, their light, free slots, the draw list, counters ([0] free, [1] to draw). UNORDERED_ACCESS
// between frames; SHADER_RESOURCE only while drawn.
Buffer*        gPool, *gLit, *gDead, *gList, *gCounters;
uint32_t       gCap;
// The draw's runs (frame.h blendRuns), written by the split pass: per run its indirect draw (vertices, 1, 0, 0) and its
// first list entry, read as an instance-rate vertex stream at the run's offset (never SV_InstanceID).
// UNORDERED_ACCESS between frames.
Buffer*        gRunArgs, *gRunStart;
uint32_t       gRunCount; // this frame's: blendRuns + 1
bool           gReady, gInit;

// This frame's plan, from particles_prepare to particles_run and particles_draw.
PartFrame gData;
bool      gOn, gDrawn;
uint32_t  gSortN, gSlot;
bool      gSort;
Buffer*   gLights, *gShapes, *gBins, *gViews, *gSh;
Texture*  gSunAtlas, *gLocalAtlas, *gFogTex;

// What was emitted lately, for a bound on the particles alive (no particle outlives its record): each frame's count,
// the time it was made and the longest life any of them may have, and whether any is alpha-blended (to sort).
struct Record
{
    double   time;
    float    life;
    uint32_t count;
    bool     blended;
};
Record   gRing[kRing];
uint32_t gRingHead;
double   gNow;
float    gPendingDt;
uint32_t gLiveBound;

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

Buffer* makeMapped(uint64_t size, uint32_t stride, uint32_t descriptors, const char* name)
{
    return makeBuffer(size, stride, descriptors, RESOURCE_MEMORY_USAGE_CPU_TO_GPU,
                      descriptors == DESCRIPTOR_TYPE_UNIFORM_BUFFER ? RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER : RESOURCE_STATE_SHADER_RESOURCE,
                      name);
}

void freePool()
{
    Buffer** bufs[] = { &gPool, &gLit, &gDead, &gList, &gCounters };
    for (Buffer** b : bufs)
    {
        if (*b)
            removeResource(*b);
        *b = nullptr;
    }
    gCap = 0;
}

// The pool at a size (a power of two, so the sort's list fits): remade, empty, only when it changes.
bool ensurePool(uint32_t cap)
{
    if (gPool && gCap == cap)
        return true;
    if (gPool)
        waitQueueIdle(gQueue);
    freePool();
    const uint32_t      rw = DESCRIPTOR_TYPE_BUFFER | DESCRIPTOR_TYPE_RW_BUFFER;
    const ResourceState uav = RESOURCE_STATE_UNORDERED_ACCESS;
    gPool = makeBuffer((uint64_t)cap * sizeof(GpuParticle), sizeof(GpuParticle), rw, RESOURCE_MEMORY_USAGE_GPU_ONLY, uav, "renderer_particles_pool");
    gLit = makeBuffer((uint64_t)cap * 16, 16, rw, RESOURCE_MEMORY_USAGE_GPU_ONLY, uav, "renderer_particles_lit");
    gDead = makeBuffer((uint64_t)cap * 4, 4, rw, RESOURCE_MEMORY_USAGE_GPU_ONLY, uav, "renderer_particles_free");
    gList = makeBuffer((uint64_t)cap * 8, 8, rw, RESOURCE_MEMORY_USAGE_GPU_ONLY, uav, "renderer_particles_list");
    gCounters = makeBuffer(16, 4, rw, RESOURCE_MEMORY_USAGE_GPU_ONLY, uav, "renderer_particles_counters");
    waitForAllResourceLoads();
    if (!gPool || !gLit || !gDead || !gList || !gCounters)
    {
        freePool();
        return false;
    }
    gCap = cap;
    gInit = true;
    memset(gRing, 0, sizeof gRing);
    return true;
}


void uavBarrier(Cmd* cmd)
{
    BufferBarrier b[5] = { { gPool, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_UNORDERED_ACCESS },
                           { gLit, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_UNORDERED_ACCESS },
                           { gDead, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_UNORDERED_ACCESS },
                           { gList, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_UNORDERED_ACCESS },
                           { gCounters, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_UNORDERED_ACCESS } };
    cmdResourceBarrier(cmd, 5, b, 0, nullptr, 0, nullptr);
}

void poolBarrier(Cmd* cmd, ResourceState from, ResourceState to)
{
    BufferBarrier b[5] = { { gPool, from, to }, { gLit, from, to }, { gDead, from, to }, { gList, from, to }, { gCounters, from, to } };
    cmdResourceBarrier(cmd, 5, b, 0, nullptr, 0, nullptr);
}

// One compute pass in this frame's next slot (the whole set bound there; kj its sort step), then a barrier.
// A 2 x 2 RGBA16F stand-in for renderer_gi's atlases while GI is off (one texel high gets a 1D view on Vulkan).
Texture* makeNoGi(const char* name)
{
    TextureDesc d = {};
    d.mWidth = d.mHeight = 2;
    d.mDepth = d.mArraySize = d.mMipLevels = 1;
    d.mSampleCount = SAMPLE_COUNT_1;
    d.mFormat = TinyImageFormat_R16G16B16A16_SFLOAT;
    d.mStartState = RESOURCE_STATE_SHADER_RESOURCE;
    d.mDescriptors = DESCRIPTOR_TYPE_TEXTURE;
    d.pName = name;
    Texture*        t = nullptr;
    TextureLoadDesc l = {};
    l.pDesc = &d;
    l.ppTexture = &t;
    addResource(&l, nullptr);
    return t;
}

void dispatch(const RenderFrame* F, Pass pass, uint32_t groups, uint32_t k, uint32_t j)
{
    if (gSlot >= kSlots || !groups)
        return;
    uint32_t index = F->frameIndex * kSlots + gSlot++;
    uint32_t step[4] = { k, j, 0, 0 };
    memcpy(gStepCb[index]->pCpuMappedAddress, step, sizeof step);
    Buffer*        emitters = gEmitterBuf[F->frameIndex], *spawn = gSpawnBuf[F->frameIndex];
    DescriptorData p[21] = {};
    p[0].mIndex = SRT_RES_IDX(PartSrt, PerFrame, gFrame);
    p[0].ppBuffers = &gFrameCb[F->frameIndex];
    p[1].mIndex = SRT_RES_IDX(PartSrt, PerFrame, gStep);
    p[1].ppBuffers = &gStepCb[index];
    p[2].mIndex = SRT_RES_IDX(PartSrt, PerFrame, gEmitters);
    p[2].ppBuffers = &emitters;
    p[3].mIndex = SRT_RES_IDX(PartSrt, PerFrame, gSpawn);
    p[3].ppBuffers = &spawn;
    p[4].mIndex = SRT_RES_IDX(PartSrt, PerFrame, gLights);
    p[4].ppBuffers = &gLights;
    p[5].mIndex = SRT_RES_IDX(PartSrt, PerFrame, gShapes);
    p[5].ppBuffers = &gShapes;
    p[6].mIndex = SRT_RES_IDX(PartSrt, PerFrame, gZBins);
    p[6].ppBuffers = &gBins;
    p[7].mIndex = SRT_RES_IDX(PartSrt, PerFrame, gViews);
    p[7].ppBuffers = &gViews;
    p[8].mIndex = SRT_RES_IDX(PartSrt, PerFrame, gSkySh);
    p[8].ppBuffers = &gSh;
    p[9].mIndex = SRT_RES_IDX(PartSrt, PerFrame, gSunAtlas);
    p[9].ppTextures = &gSunAtlas;
    p[10].mIndex = SRT_RES_IDX(PartSrt, PerFrame, gLocalAtlas);
    p[10].ppTextures = &gLocalAtlas;
    p[11].mIndex = SRT_RES_IDX(PartSrt, PerFrame, gParticles);
    p[11].ppBuffers = &gPool;
    p[12].mIndex = SRT_RES_IDX(PartSrt, PerFrame, gLit);
    p[12].ppBuffers = &gLit;
    p[13].mIndex = SRT_RES_IDX(PartSrt, PerFrame, gDead);
    p[13].ppBuffers = &gDead;
    p[14].mIndex = SRT_RES_IDX(PartSrt, PerFrame, gList);
    p[14].ppBuffers = &gList;
    p[15].mIndex = SRT_RES_IDX(PartSrt, PerFrame, gCounters);
    p[15].ppBuffers = &gCounters;
    p[16].mIndex = SRT_RES_IDX(PartSrt, PerFrame, gGiIrradiance);
    p[16].ppTextures = &gGiIrr;
    p[17].mIndex = SRT_RES_IDX(PartSrt, PerFrame, gGiDistance);
    p[17].ppTextures = &gGiDist;
    p[18].mIndex = SRT_RES_IDX(PartSrt, PerFrame, gGiProbes);
    p[18].ppBuffers = &gGiProbes;
    p[19].mIndex = SRT_RES_IDX(PartSrt, PerFrame, gRunArgs);
    p[19].ppBuffers = &gRunArgs;
    p[20].mIndex = SRT_RES_IDX(PartSrt, PerFrame, gRunStart);
    p[20].ppBuffers = &gRunStart;
    updateDescriptorSet(gRenderer, index, gSet, 21, p);
    Cmd* cmd = F->cmd;
    cmdBindPipeline(cmd, gPipe[pass]);
    cmdBindDescriptorSet(cmd, 0, gPersistent);
    cmdBindDescriptorSet(cmd, index, gSet);
    cmdDispatch(cmd, groups, 1, 1);
    uavBarrier(cmd);
}

uint32_t rowOf(yea_handle h)
{
    uint32_t row = h & 0xFFFFFu;
    if (row >= Emitters.size() || !Emitters[row].alive || (Emitters[row].gen & 0xFFFu) != (h >> 20))
        return 0xFFFFFFFFu;
    return row;
}

float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

uint32_t pow2Ceil(uint32_t v)
{
    uint32_t p = 1;
    while (p < v)
        p <<= 1;
    return p;
}

bool anyEmitter()
{
    for (const Emitter& e : Emitters)
        if (e.alive)
            return true;
    return false;
}
} // namespace

int load(int reloaded)
{
    const YeaHost* host = yea::host();
    gRenderer = (Renderer*)host->renderer;
    gQueue = (Queue*)host->queue;
    gFrames = host->framesInFlight < kMaxFrames ? host->framesInFlight : kMaxFrames;
    gReady = gOn = gDrawn = false;
    gInit = true;
    gLiveBound = 0;
    memset(gRing, 0, sizeof gRing);
    for (uint32_t i = 0; i < kPasses; ++i)
    {
        ShaderLoadDesc sd = {};
        sd.mComp.pFileName = kShaderNames[i];
        addShader(gRenderer, &sd, &gShader[i]);
        if (!gShader[i])
            return 1;
        PipelineDesc desc = {};
        desc.mType = PIPELINE_TYPE_COMPUTE;
        PIPELINE_LAYOUT_DESC(desc, SRT_LAYOUT_DESC(PartSrt, Persistent), SRT_LAYOUT_DESC(PartSrt, PerFrame), NULL, NULL);
        desc.mComputeDesc.pShaderProgram = gShader[i];
        desc.pName = kShaderNames[i];
        addPipeline(gRenderer, &desc, &gPipe[i]);
        if (!gPipe[i])
            return 1;
    }
    {
        ShaderLoadDesc sd = {};
        sd.mVert.pFileName = "renderer_particles.vert";
        sd.mFrag.pFileName = "renderer_particles.frag";
        addShader(gRenderer, &sd, &gDrawShader);
        if (!gDrawShader)
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
        t.pName = "renderer_particles_no_atlas";
        addRenderTarget(gRenderer, &t, &gDummyAtlas);
    }
    {
        TextureDesc d = {};
        d.mWidth = d.mHeight = d.mDepth = 2;
        d.mArraySize = d.mMipLevels = 1;
        d.mSampleCount = SAMPLE_COUNT_1;
        d.mFormat = TinyImageFormat_R16G16B16A16_SFLOAT;
        d.mStartState = RESOURCE_STATE_SHADER_RESOURCE;
        d.mDescriptors = DESCRIPTOR_TYPE_TEXTURE;
        d.pName = "renderer_particles_no_fog";
        TextureLoadDesc l = {};
        l.pDesc = &d;
        l.ppTexture = &gDummyFog;
        addResource(&l, nullptr);
    }
    gNoGi = makeNoGi("renderer_particles_no_gi");
    gDummyViews = makeMapped(sizeof(RenderGpuShadowView) * 4, sizeof(RenderGpuShadowView), DESCRIPTOR_TYPE_BUFFER, "renderer_particles_no_views");
    gDummyBins = makeMapped(16, 4, DESCRIPTOR_TYPE_BUFFER, "renderer_particles_no_bins");
    gDummySh = makeMapped(9 * 16, 16, DESCRIPTOR_TYPE_BUFFER, "renderer_particles_no_sh");
    gDummyLights = makeMapped(sizeof(RenderGpuLight) * 2, sizeof(RenderGpuLight), DESCRIPTOR_TYPE_BUFFER, "renderer_particles_no_lights");
    gDummyShapes = makeMapped(sizeof(RenderGpuLightShape) * 2, sizeof(RenderGpuLightShape), DESCRIPTOR_TYPE_BUFFER, "renderer_particles_no_shapes");
    for (uint32_t i = 0; i < gFrames; ++i)
    {
        gFrameCb[i] = makeMapped(1024, 0, DESCRIPTOR_TYPE_UNIFORM_BUFFER, "renderer_particles_cb");
        gEmitterBuf[i] = makeMapped(sizeof(GpuEmitter) * kMaxEmitters, sizeof(GpuEmitter), DESCRIPTOR_TYPE_BUFFER, "renderer_particles_emitters");
        gSpawnBuf[i] = makeMapped(16 * kMaxEmitters, 16, DESCRIPTOR_TYPE_BUFFER, "renderer_particles_spawn");
    }
    for (uint32_t i = 0; i < gFrames * kSlots; ++i)
        gStepCb[i] = makeMapped(256, 0, DESCRIPTOR_TYPE_UNIFORM_BUFFER, "renderer_particles_step");
    gRunArgs = makeBuffer(kRuns * 16, 4, DESCRIPTOR_TYPE_RW_BUFFER | DESCRIPTOR_TYPE_INDIRECT_BUFFER, RESOURCE_MEMORY_USAGE_GPU_ONLY,
                          RESOURCE_STATE_UNORDERED_ACCESS, "renderer_particles_run_args");
    gRunStart = makeBuffer(kRuns * 4, 4, DESCRIPTOR_TYPE_RW_BUFFER | DESCRIPTOR_TYPE_VERTEX_BUFFER, RESOURCE_MEMORY_USAGE_GPU_ONLY,
                           RESOURCE_STATE_UNORDERED_ACCESS, "renderer_particles_run_start");
    DescriptorSetDesc persistent = SRT_SET_DESC(PartSrt, Persistent, 1, 0);
    addDescriptorSet(gRenderer, &persistent, &gPersistent);
    DescriptorSetDesc perFrame = SRT_SET_DESC(PartSrt, PerFrame, gFrames * kSlots, 0);
    addDescriptorSet(gRenderer, &perFrame, &gSet);
    DescriptorSetDesc drawDesc = SRT_SET_DESC(DrawSrt, PerFrame, gFrames, 0);
    addDescriptorSet(gRenderer, &drawDesc, &gDrawSet);
    waitForAllResourceLoads();
    bool ok = gNoGi && gShadowCmp && gDummyAtlas && gDummyAtlas->pTexture && gDummyFog && gDummyViews && gDummyBins && gDummySh && gDummyLights &&
              gDummyShapes && gPersistent && gSet && gDrawSet && gRunArgs && gRunStart;
    for (uint32_t i = 0; i < gFrames; ++i)
        ok = ok && gFrameCb[i] && gEmitterBuf[i] && gSpawnBuf[i] && gFrameCb[i]->pCpuMappedAddress && gEmitterBuf[i]->pCpuMappedAddress &&
             gSpawnBuf[i]->pCpuMappedAddress;
    for (uint32_t i = 0; i < gFrames * kSlots; ++i)
        ok = ok && gStepCb[i] && gStepCb[i]->pCpuMappedAddress;
    if (!ok)
        return 1;
    memset(gDummyViews->pCpuMappedAddress, 0, sizeof(RenderGpuShadowView) * 4);
    memset(gDummyBins->pCpuMappedAddress, 0xFF, 16);
    memset(gDummySh->pCpuMappedAddress, 0, 9 * 16);
    memset(gDummyLights->pCpuMappedAddress, 0, sizeof(RenderGpuLight) * 2);
    memset(gDummyShapes->pCpuMappedAddress, 0, sizeof(RenderGpuLightShape) * 2);
    DescriptorData s = {};
    s.mIndex = SRT_RES_IDX(PartSrt, Persistent, gShadowCmp);
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
    freePool();
    for (uint32_t i = 0; i < kMaxFrames; ++i)
    {
        Buffer** per[] = { &gFrameCb[i], &gEmitterBuf[i], &gSpawnBuf[i] };
        for (Buffer** b : per)
        {
            if (*b)
                removeResource(*b);
            *b = nullptr;
        }
    }
    for (uint32_t i = 0; i < kMaxFrames * kSlots; ++i)
        if (gStepCb[i])
            removeResource(gStepCb[i]), gStepCb[i] = nullptr;
    Buffer** bufs[] = { &gDummyViews, &gDummyBins, &gDummySh, &gDummyLights, &gDummyShapes, &gRunArgs, &gRunStart };
    for (Buffer** b : bufs)
    {
        if (*b)
            removeResource(*b);
        *b = nullptr;
    }
    if (gDummyFog)
        removeResource(gDummyFog);
    if (gNoGi)
        removeResource(gNoGi);
    gNoGi = nullptr;
    gDummyFog = nullptr;
    if (gDummyAtlas)
        removeRenderTarget(gRenderer, gDummyAtlas);
    gDummyAtlas = nullptr;
    if (gShadowCmp)
        removeSampler(gRenderer, gShadowCmp);
    gShadowCmp = nullptr;
    DescriptorSet** sets[] = { &gPersistent, &gSet, &gDrawSet };
    for (DescriptorSet** s : sets)
    {
        if (*s)
            removeDescriptorSet(gRenderer, *s);
        *s = nullptr;
    }
    for (uint32_t i = 0; i < kPasses; ++i)
    {
        if (gPipe[i])
            removePipeline(gRenderer, gPipe[i]);
        if (gShader[i])
            removeShader(gRenderer, gShader[i]);
        gPipe[i] = nullptr;
        gShader[i] = nullptr;
    }
    if (gDrawPipe)
        removePipeline(gRenderer, gDrawPipe);
    if (gDrawShader)
        removeShader(gRenderer, gDrawShader);
    gDrawPipe = nullptr;
    gDrawShader = nullptr;
    gReady = gOn = gDrawn = false;
}

void update(uint32_t phase, float dt)
{
    if (phase == YEA_PHASE_SIMULATE)
        gPendingDt += dt;
}
void draw(uint32_t pass, const YeaFrame* frame) {}

// --- Emitters ------------------------------------------------------------------------------------------------------
yea_handle particle_emitter_add(yea_float3 pos)
{
    uint32_t row = 0xFFFFFFFFu;
    for (uint32_t i = 0; i < Emitters.size(); ++i)
        if (!Emitters[i].alive)
        {
            uint32_t gen = Emitters[i].gen + 1;
            Emitters[i] = Emitter{};
            Emitters[i].gen = gen;
            row = i;
            break;
        }
    if (row == 0xFFFFFFFFu)
    {
        if (!Emitters.add())
            return 0xFFFFFFFFu;
        row = Emitters.size() - 1;
    }
    Emitter& e = Emitters[row];
    e.alive = 1;
    e.pos = pos;
    e.dir = { 0, 1, 0 };
    e.rate = 20;
    e.speed = 0.5f;
    e.spread = 20;
    e.life_min = 2, e.life_max = 3;
    e.color_start = e.color_end = { 1, 1, 1 };
    e.opacity = 0.5f;
    e.size_start = 0.05f, e.size_end = 0.15f;
    return row | ((e.gen & 0xFFFu) << 20);
}

void particle_emitter_place(yea_handle h, yea_float3 pos, yea_float3 dir)
{
    uint32_t row = rowOf(h);
    if (row == 0xFFFFFFFFu)
        return;
    Emitters[row].pos = pos;
    Emitters[row].dir = dir;
}

void particle_emitter_shape(yea_handle h, uint32_t shape, yea_float3 extent)
{
    uint32_t row = rowOf(h);
    if (row == 0xFFFFFFFFu)
        return;
    Emitters[row].shape = shape > 2 ? 0 : shape;
    Emitters[row].extent = { fabsf(extent.x), fabsf(extent.y), fabsf(extent.z) };
}

void particle_emitter_rate(yea_handle h, float per_second, uint32_t burst)
{
    uint32_t row = rowOf(h);
    if (row == 0xFFFFFFFFu)
        return;
    Emitters[row].rate = per_second > 0 ? per_second : 0;
    Emitters[row].burst += burst;
}

void particle_emitter_motion(yea_handle h, float speed, float spread_deg, float speed_jitter, float gravity, float drag, float turbulence)
{
    uint32_t row = rowOf(h);
    if (row == 0xFFFFFFFFu)
        return;
    Emitter& e = Emitters[row];
    e.speed = speed > 0 ? speed : 0;
    e.spread = clampf(spread_deg, 0, 180);
    e.speed_jitter = clampf(speed_jitter, 0, 1);
    e.gravity = gravity;
    e.drag = drag > 0 ? drag : 0;
    e.turbulence = turbulence > 0 ? turbulence : 0;
}

void particle_emitter_life(yea_handle h, float min_s, float max_s)
{
    uint32_t row = rowOf(h);
    if (row == 0xFFFFFFFFu)
        return;
    float lo = min_s > 0.01f ? min_s : 0.01f, hi = max_s > lo ? max_s : lo;
    Emitters[row].life_min = lo;
    Emitters[row].life_max = hi;
}

void particle_emitter_look(yea_handle h, yea_float3 color_start, yea_float3 color_end, float size_start, float size_end, float opacity)
{
    uint32_t row = rowOf(h);
    if (row == 0xFFFFFFFFu)
        return;
    Emitter& e = Emitters[row];
    e.color_start = { fmaxf(color_start.x, 0), fmaxf(color_start.y, 0), fmaxf(color_start.z, 0) };
    e.color_end = { fmaxf(color_end.x, 0), fmaxf(color_end.y, 0), fmaxf(color_end.z, 0) };
    e.size_start = fmaxf(size_start, 0);
    e.size_end = fmaxf(size_end, 0);
    e.opacity = clampf(opacity, 0, 1);
}

void particle_emitter_light(yea_handle h, float emissive, float anisotropy, bool additive)
{
    uint32_t row = rowOf(h);
    if (row == 0xFFFFFFFFu)
        return;
    Emitters[row].emissive = fmaxf(emissive, 0);
    Emitters[row].anisotropy = clampf(anisotropy, -0.9f, 0.95f);
    Emitters[row].additive = additive ? 1 : 0;
}

void particle_emitter_remove(yea_handle h)
{
    uint32_t row = rowOf(h);
    if (row != 0xFFFFFFFFu)
        Emitters[row].alive = 0;
}

void particle_emitter_clear()
{
    for (Emitter& e : Emitters)
        e.alive = 0;
}

// --- The core's hooks ----------------------------------------------------------------------------------------------
bool particles_on() { return gReady && settings.particles && (gLiveBound > 0 || anyEmitter()); }

bool particles_prepare(uint64_t address)
{
    RenderFrame* F = (RenderFrame*)(uintptr_t)address;
    gOn = gDrawn = false;
    float dt = gPendingDt;
    gPendingDt = 0;
    if (!F || F->version != RENDER_FRAME_VERSION)
        return false;
    if (!gReady || !settings.particles || F->sampleCount != 1 || !F->hdr || !F->depth || !F->width || !F->height)
        return false;
    // Reset (or a first frame): the pool starts empty.
    if (pool->epoch == 0)
    {
        pool->epoch = 1;
        gInit = true;
        memset(gRing, 0, sizeof gRing);
    }
    dt = clampf(dt, 0, 0.1f);
    gNow += dt;
    // This frame's emission, emitter by emitter, and what of the pool can still be alive.
    const uint32_t cap = pow2Ceil((uint32_t)clampf((float)settings.max_particles, 1024, 65536));
    GpuEmitter*    ge = (GpuEmitter*)gEmitterBuf[F->frameIndex]->pCpuMappedAddress;
    uint32_t*      sp = (uint32_t*)gSpawnBuf[F->frameIndex]->pCpuMappedAddress;
    memset(ge, 0, sizeof(GpuEmitter) * kMaxEmitters);
    uint32_t total = 0, spawns = 0;
    float    longest = 0;
    bool     blended = false;
    for (uint32_t i = 0; i < Emitters.size() && i < kMaxEmitters; ++i)
    {
        Emitter&    e = Emitters[i];
        GpuEmitter& g = ge[i];
        g.pos[0] = e.pos.x, g.pos[1] = e.pos.y, g.pos[2] = e.pos.z, g.pos[3] = (float)e.shape;
        g.extent[0] = e.extent.x, g.extent[1] = e.extent.y, g.extent[2] = e.extent.z, g.extent[3] = cosf(e.spread * 0.01745329f);
        g.dir[0] = e.dir.x, g.dir[1] = e.dir.y, g.dir[2] = e.dir.z, g.dir[3] = e.speed;
        g.motion[0] = e.gravity, g.motion[1] = e.drag, g.motion[2] = e.turbulence, g.motion[3] = e.speed_jitter;
        g.colorStart[0] = e.color_start.x, g.colorStart[1] = e.color_start.y, g.colorStart[2] = e.color_start.z, g.colorStart[3] = e.opacity;
        g.colorEnd[0] = e.color_end.x, g.colorEnd[1] = e.color_end.y, g.colorEnd[2] = e.color_end.z, g.colorEnd[3] = e.emissive;
        g.size[0] = e.size_start, g.size[1] = e.size_end, g.size[2] = e.anisotropy, g.size[3] = e.additive ? 1.0f : 0.0f;
        g.life[0] = e.life_min, g.life[1] = e.life_max;
        if (!e.alive)
            continue;
        float    want = e.carry + e.rate * dt;
        uint32_t n = (uint32_t)want;
        e.carry = want - (float)n;
        n += e.burst;
        e.burst = 0;
        if (total + n > cap)
            n = cap - total;
        if (!n)
            continue;
        sp[spawns * 4 + 0] = i;
        sp[spawns * 4 + 1] = total;
        sp[spawns * 4 + 2] = n;
        sp[spawns * 4 + 3] = 0;
        spawns++;
        total += n;
        longest = e.life_max > longest ? e.life_max : longest;
        blended = blended || !e.additive;
    }
    if (total)
    {
        gRingHead = (gRingHead + 1) % kRing;
        gRing[gRingHead] = { gNow, longest, total, blended };
    }
    uint32_t bound = 0;
    bool     sort = false;
    for (uint32_t i = 0; i < kRing; ++i)
        if (gRing[i].count && gNow - gRing[i].time <= (double)gRing[i].life + 0.2)
        {
            bound += gRing[i].count;
            sort = sort || gRing[i].blended;
        }
    gLiveBound = bound;
    if (!bound && !gInit)
        return false; // nothing alive, nothing made
    if (!ensurePool(cap))
        return false;
    bound = bound < cap ? bound : cap;
    gSortN = pow2Ceil(bound > 1024 ? bound : 1024);
    gSortN = gSortN < gCap ? gSortN : gCap;
    // Blended surfaces split the list at their depths: it is sorted then, additive particles too (behind a pane, they are
    // seen through it).
    const uint32_t runs = F->blendRuns < RENDER_MAX_BLEND_RUNS ? F->blendRuns : RENDER_MAX_BLEND_RUNS;
    gSort = (sort || runs > 0) && bound > 1;
    gRunCount = runs + 1;

    PartFrame& d = gData;
    memset(&d, 0, sizeof d);
    for (int r = 0; r < 3; ++r)
        memcpy(d.view[r], F->view[r], sizeof d.view[r]);
    d.camPos[0] = F->camPos[0], d.camPos[1] = F->camPos[1], d.camPos[2] = F->camPos[2], d.camPos[3] = F->nearZ;
    // The core's projection: x and y scaled by p00 and p11, offset by the jitter in ndc (pixel y runs down).
    d.proj[0] = F->p00, d.proj[1] = F->p11;
    d.proj[2] = F->jitter[0] * 2.0f / (float)F->width;
    d.proj[3] = -F->jitter[1] * 2.0f / (float)F->height;
    d.screen[0] = (float)F->width, d.screen[1] = (float)F->height, d.screen[2] = 1.0f / (float)F->width, d.screen[3] = 1.0f / (float)F->height;
    d.soft[0] = fmaxf(settings.soft_distance, 0), d.soft[1] = fmaxf(settings.near_fade, 0), d.soft[2] = (float)fmod(gNow, 4096.0), d.soft[3] = dt;
    const bool shadowMaps = F->shadowViews && F->sunAtlas && F->localAtlas;
    if (F->sunOn)
    {
        float l = sqrtf(F->sunDir[0] * F->sunDir[0] + F->sunDir[1] * F->sunDir[1] + F->sunDir[2] * F->sunDir[2]);
        for (int k = 0; k < 3; ++k)
            d.sunDir[k] = l > 0 ? F->sunDir[k] / l : 0.0f;
        d.sunDir[3] = 1.0f;
        memcpy(d.sunColor, F->sunColor, sizeof F->sunColor);
        d.sunColor[3] = shadowMaps && F->sunShadow && (!F->rtSunShadows || F->keepShadowMaps) ? (float)F->cascadeCount : 0.0f;
        memcpy(d.cascadeEnd, F->cascadeEnd, sizeof d.cascadeEnd);
    }
    // The sky's light from all round (renderer_fog's: the SH's first band, else the gradient's exact average).
    if (F->skySh)
        d.ambient[3] = fmaxf(F->ambient, 1e-6f);
    else
        for (int k = 0; k < 3; ++k)
            d.ambient[k] = (F->skyHorizon[k] * 0.5f + F->skyZenith[k] * (2.0f / 3.0f) + F->skyGround[k] * (5.0f / 6.0f)) * 0.5f * F->ambient;
    d.texel[0] = F->sunTexel, d.texel[1] = F->localTexel, d.texel[2] = F->zbins ? F->binScale : 0.0f, d.texel[3] = (float)(F->frameNumber % 65536u);
    if (F->fogVolume)
    {
        memcpy(d.fog, F->fogParams, sizeof d.fog);
        d.fog2[0] = F->fogUv[0], d.fog2[1] = F->fogUv[1];
    }
    const bool lights = F->lightBuffer && F->lightShapes;
    d.counts[0] = lights ? F->lightCount : 0;
    d.counts[1] = F->binCount;
    d.counts[2] = (shadowMaps ? 1u : 0u) | (F->zbins && F->binCount ? 2u : 0u);
    d.counts[3] = gCap;
    d.spawn[0] = total, d.spawn[1] = spawns, d.spawn[2] = gSortN, d.spawn[3] = (uint32_t)F->frameNumber * 2654435761u;
    d.runs[0] = runs;
    memcpy(d.blend, F->blendDepths, sizeof(float) * runs);
    memcpy(gFrameCb[F->frameIndex]->pCpuMappedAddress, &d, sizeof d);
    gLights = lights ? F->lightBuffer : gDummyLights;
    gShapes = lights ? F->lightShapes : gDummyShapes;
    gBins = F->zbins ? F->zbins : gDummyBins;
    gViews = shadowMaps ? F->shadowViews : gDummyViews;
    gSh = F->skySh ? F->skySh : gDummySh;
    gSunAtlas = shadowMaps ? F->sunAtlas->pTexture : gDummyAtlas->pTexture;
    gLocalAtlas = shadowMaps ? F->localAtlas->pTexture : gDummyAtlas->pTexture;
    gFogTex = F->fogVolume ? F->fogVolume : gDummyFog;
    gOn = true;
    return true;
}

void particles_run(uint64_t address)
{
    RenderFrame* F = (RenderFrame*)(uintptr_t)address;
    if (!gOn || !F)
        return;
    gSlot = 0;
    // renderer_gi's probes (set by gi_prepare, after particles_prepare): last frame's, when they hold light.
    const bool gi = F->giProbeCount && F->giIrradiance && F->giDistance && F->giProbeData;
    PartFrame* pf = (PartFrame*)gFrameCb[F->frameIndex]->pCpuMappedAddress;
    pf->counts[2] = (pf->counts[2] & ~4u) | (gi ? 4u : 0u);
    memcpy(pf->gi, F->giGrid, sizeof pf->gi);
    gGiIrr = gi ? F->giIrradiance : gNoGi;
    gGiDist = gi ? F->giDistance : gNoGi;
    gGiProbes = gi ? F->giProbeData : gDummySh;
    Cmd* cmd = F->cmd;
    cmdBeginDebugMarker(cmd, 0.9f, 0.8f, 0.6f, "Renderer particles");
    if (gInit)
    {
        dispatch(F, kInit, (gCap + 255) / 256, 0, 0);
        gInit = false;
    }
    dispatch(F, kEmit, gData.spawn[0] ? (gData.spawn[0] + 255) / 256 : 1, 0, 0); // also starts the draw list
    dispatch(F, kSimulate, (gCap + 255) / 256, 0, 0);
    if (gSort)
    {
        // Bitonic, farthest first: each 1024 in shared memory, then per larger stage its strides of 1024 and more one
        // pass each and the rest in shared memory again.
        dispatch(F, kPad, (gSortN + 255) / 256, 0, 0);
        dispatch(F, kSortLocal, gSortN / 1024, 0, 0);
        for (uint32_t k = 2048; k <= gSortN; k <<= 1)
        {
            for (uint32_t j = k >> 1; j >= 1024; j >>= 1)
                dispatch(F, kSortGlobal, (gSortN / 2 + 255) / 256, k, j);
            dispatch(F, kSortLocal, gSortN / 1024, k, 0);
        }
    }
    dispatch(F, kSplit, 1, 0, 0); // the draw's runs
    cmdEndDebugMarker(cmd);
}

namespace
{
// The pipeline for the core's see-through pass (hdr, reactive, transparency), made once, again if the format changes.
bool ensureDrawPipe(TinyImageFormat format)
{
    if (gDrawPipe && gDrawFormat == format)
        return true;
    if (gDrawPipe)
    {
        waitQueueIdle(gQueue);
        removePipeline(gRenderer, gDrawPipe);
        gDrawPipe = nullptr;
    }
    RasterizerStateDesc raster = {};
    raster.mCullMode = CULL_MODE_NONE;
    DepthStateDesc depth = {};
    // Premultiplied over the scene (additive particles write alpha 0); the reactive mask adds coverage; the transparency
    // mask is blended surfaces' alone.
    BlendStateDesc blend = {};
    blend.mSrcFactors[0] = BC_ONE;
    blend.mDstFactors[0] = BC_ONE_MINUS_SRC_ALPHA;
    blend.mSrcAlphaFactors[0] = BC_ONE;
    blend.mDstAlphaFactors[0] = BC_ONE_MINUS_SRC_ALPHA;
    blend.mSrcFactors[1] = blend.mDstFactors[1] = blend.mSrcAlphaFactors[1] = blend.mDstAlphaFactors[1] = BC_ONE;
    blend.mSrcFactors[2] = blend.mSrcAlphaFactors[2] = BC_ZERO;
    blend.mDstFactors[2] = blend.mDstAlphaFactors[2] = BC_ONE;
    for (int t = 0; t < 3; ++t)
        blend.mBlendModes[t] = blend.mBlendAlphaModes[t] = BM_ADD;
    blend.mColorWriteMasks[0] = COLOR_MASK_RED | COLOR_MASK_GREEN | COLOR_MASK_BLUE;
    blend.mColorWriteMasks[1] = COLOR_MASK_ALL;
    blend.mColorWriteMasks[2] = COLOR_MASK_NONE;
    blend.mRenderTargetMask = (BlendStateTargets)(BLEND_STATE_TARGET_0 | BLEND_STATE_TARGET_1 | BLEND_STATE_TARGET_2);
    blend.mIndependentBlend = true;
    TinyImageFormat formats[3] = { format, TinyImageFormat_R8_UNORM, TinyImageFormat_R8_UNORM };
    // The run's first list entry: an instance-rate stream bound at the run's offset.
    VertexLayout layout = {};
    layout.mBindingCount = 1;
    layout.mBindings[0].mStride = 4;
    layout.mBindings[0].mRate = VERTEX_BINDING_RATE_INSTANCE;
    layout.mAttribCount = 1;
    layout.mAttribs[0].mSemantic = SEMANTIC_TEXCOORD0;
    layout.mAttribs[0].mFormat = TinyImageFormat_R32_UINT;
    layout.mAttribs[0].mBinding = 0;
    layout.mAttribs[0].mLocation = 0;
    layout.mAttribs[0].mOffset = 0;
    PipelineDesc desc = {};
    desc.mType = PIPELINE_TYPE_GRAPHICS;
    PIPELINE_LAYOUT_DESC(desc, NULL, SRT_LAYOUT_DESC(DrawSrt, PerFrame), NULL, NULL);
    GraphicsPipelineDesc& g = desc.mGraphicsDesc;
    g.mPrimitiveTopo = PRIMITIVE_TOPO_TRI_LIST;
    g.mRenderTargetCount = 3;
    g.pColorFormats = formats;
    g.mSampleCount = SAMPLE_COUNT_1;
    g.mDepthStencilFormat = TinyImageFormat_UNDEFINED;
    g.pShaderProgram = gDrawShader;
    g.pRasterizerState = &raster;
    g.pDepthState = &depth;
    g.pBlendState = &blend;
    g.pVertexLayout = &layout;
    desc.pName = "renderer_particles_draw";
    addPipeline(gRenderer, &desc, &gDrawPipe);
    gDrawFormat = format;
    return gDrawPipe != nullptr;
}

void runBarrier(Cmd* cmd, bool toDraw)
{
    const ResourceState uav = RESOURCE_STATE_UNORDERED_ACCESS;
    BufferBarrier b[2] = { { gRunArgs, toDraw ? uav : RESOURCE_STATE_INDIRECT_ARGUMENT, toDraw ? RESOURCE_STATE_INDIRECT_ARGUMENT : uav },
                           { gRunStart, toDraw ? uav : RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
                             toDraw ? RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER : uav } };
    cmdResourceBarrier(cmd, 2, b, 0, nullptr, 0, nullptr);
}
} // namespace

// Before the core binds its see-through pass: the pipeline, this frame's set and the buffers ready to draw. false: no
// particles this frame (particles_draw and particles_draw_end then do nothing).
bool particles_draw_begin(uint64_t address)
{
    RenderFrame* F = (RenderFrame*)(uintptr_t)address;
    gDrawn = false;
    if (!gOn || !F || !F->hdr || !F->depth)
        return false;
    gOn = false;
    if (!ensureDrawPipe(F->hdr->mFormat))
        return false;
    Buffer*        emitters = gEmitterBuf[F->frameIndex];
    Texture*       depthTex = F->depth->pTexture;
    DescriptorData p[8] = {};
    p[0].mIndex = SRT_RES_IDX(DrawSrt, PerFrame, gFrame);
    p[0].ppBuffers = &gFrameCb[F->frameIndex];
    p[1].mIndex = SRT_RES_IDX(DrawSrt, PerFrame, gEmitters);
    p[1].ppBuffers = &emitters;
    p[2].mIndex = SRT_RES_IDX(DrawSrt, PerFrame, gParticles);
    p[2].ppBuffers = &gPool;
    p[3].mIndex = SRT_RES_IDX(DrawSrt, PerFrame, gLit);
    p[3].ppBuffers = &gLit;
    p[4].mIndex = SRT_RES_IDX(DrawSrt, PerFrame, gList);
    p[4].ppBuffers = &gList;
    p[5].mIndex = SRT_RES_IDX(DrawSrt, PerFrame, gCounters);
    p[5].ppBuffers = &gCounters;
    p[6].mIndex = SRT_RES_IDX(DrawSrt, PerFrame, gDepth);
    p[6].ppTextures = &depthTex;
    p[7].mIndex = SRT_RES_IDX(DrawSrt, PerFrame, gFogVolume);
    p[7].ppTextures = &gFogTex;
    updateDescriptorSet(gRenderer, F->frameIndex, gDrawSet, 8, p);
    poolBarrier(F->cmd, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_SHADER_RESOURCE);
    runBarrier(F->cmd, true);
    gDrawn = true;
    return true;
}

// Inside the core's see-through pass: run `run` of the sorted list (frame.h blendRuns), drawn indirectly from the split
// pass's counts, first vertex 0.
void particles_draw(uint64_t address, uint32_t run)
{
    RenderFrame* F = (RenderFrame*)(uintptr_t)address;
    if (!gDrawn || !F || run >= gRunCount)
        return;
    Cmd* cmd = F->cmd;
    cmdBindPipeline(cmd, gDrawPipe);
    cmdBindDescriptorSet(cmd, F->frameIndex, gDrawSet);
    uint32_t stride = 4;
    uint64_t offset = (uint64_t)run * 4;
    cmdBindVertexBuffer(cmd, 1, &gRunStart, &stride, &offset);
    cmdExecuteIndirect(cmd, INDIRECT_DRAW, 1, gRunArgs, (uint64_t)run * 16, nullptr, 0);
}

// After the core unbinds its pass.
void particles_draw_end(uint64_t address)
{
    RenderFrame* F = (RenderFrame*)(uintptr_t)address;
    if (!gDrawn || !F)
        return;
    runBarrier(F->cmd, false);
    poolBarrier(F->cmd, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_UNORDERED_ACCESS);
    gDrawn = false;
}
} // namespace renderer_particles
