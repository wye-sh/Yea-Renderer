// Ambient occlusion for the renderer core: GTAO from the depth prepass (Shaders/Gtao.comp.fsl), then
// `denoise` passes of a depth-aware 3x3 blur, at full or half size, into an RG16F texture (visibility, view depth) the
// forward pass reads for its ambient light (diffuse with multi-bounce, specular with specular occlusion); at half
// size the forward pass upsamples it along edges itself (F->aoScale 2), so no full-size pass is needed.
#include "yea_renderer_ao.h"
#include "frame.h"

#include <Common_3/Graphics/Interfaces/IGraphics.h>
#include <Common_3/Resources/ResourceLoader/Interfaces/IResourceLoader.h>
#include <Common_3/Graphics/FSL/defaults.h>
#define SAMPLE_COUNT 1
#include "Shaders/Gtao.srt.h"
#undef SAMPLE_COUNT

#include <string.h>

namespace renderer_ao
{
namespace
{
struct AoData
{
    float    proj[4];
    float    params[4];
    float    screen[4];
    uint32_t counts[4];
};

const uint32_t kMaxFrames = 4;
const uint32_t kSetsPerFrame = 2; // 0: read B, write A (GTAO, and blurs back into A); 1: read A, write B

Renderer*      gRenderer;
Queue*         gQueue;
uint32_t       gFrames;
Shader*        gGtaoShader[4];
Shader*        gBlurShader[4];
Pipeline*      gGtaoPipe[4];
Pipeline*      gBlurPipe[4];
DescriptorSet* gSet;
Buffer*        gCb[kMaxFrames];
Texture*       gTex[2]; // A, B (at the occlusion's own size): SHADER_RESOURCE between passes
uint32_t       gWidth, gHeight, gScale; // the screen's size; 1 full size, 2 half
uint32_t       gPasses; // denoise passes this frame
bool           gReady;

uint32_t sampleIndex(uint32_t s) { return s == 8 ? 3 : s == 4 ? 2 : s == 2 ? 1 : 0; }

uint32_t clampu(int32_t v, uint32_t lo, uint32_t hi) { return v < (int32_t)lo ? lo : (v > (int32_t)hi ? hi : (uint32_t)v); }

void freeTextures()
{
    for (Texture*& t : gTex)
        if (t)
        {
            removeResource(t);
            t = nullptr;
        }
    gWidth = gHeight = gScale = 0;
}

Texture* output() { return gTex[gPasses & 1]; } // each denoise pass swaps A and B

void createTextures(uint32_t w, uint32_t h, uint32_t scale)
{
    for (Texture*& t : gTex)
    {
        TextureDesc d = {};
        d.mWidth = (w + scale - 1) / scale;
        d.mHeight = (h + scale - 1) / scale;
        d.mDepth = d.mArraySize = d.mMipLevels = 1;
        d.mSampleCount = SAMPLE_COUNT_1;
        d.mFormat = TinyImageFormat_R16G16_SFLOAT; // RWTex2D(half2) is rg16f on Vulkan: x visibility, y view depth
        d.mStartState = RESOURCE_STATE_SHADER_RESOURCE;
        d.mDescriptors = (DescriptorType)(DESCRIPTOR_TYPE_TEXTURE | DESCRIPTOR_TYPE_RW_TEXTURE);
        d.pName = "renderer_ao";
        TextureLoadDesc l = {};
        l.pDesc = &d;
        l.ppTexture = &t;
        addResource(&l, nullptr);
    }
    waitForAllResourceLoads();
    gWidth = w;
    gHeight = h;
    gScale = scale;
}

void barrier(Cmd* cmd, Texture* t, ResourceState from, ResourceState to)
{
    TextureBarrier b = {};
    b.pTexture = t;
    b.mCurrentState = from;
    b.mNewState = to;
    cmdResourceBarrier(cmd, 0, nullptr, 1, &b, 0, nullptr);
}

Shader* loadCompute(const char* name)
{
    ShaderLoadDesc d = {};
    d.mComp.pFileName = name;
    Shader* s = nullptr;
    addShader(gRenderer, &d, &s);
    return s;
}

Pipeline* computePipeline(Shader* s, const char* name)
{
    PipelineDesc desc = {};
    desc.mType = PIPELINE_TYPE_COMPUTE;
    PIPELINE_LAYOUT_DESC(desc, NULL, SRT_LAYOUT_DESC(GtaoSrt, PerFrame), NULL, NULL);
    desc.mComputeDesc.pShaderProgram = s;
    desc.pName = name;
    Pipeline* p = nullptr;
    addPipeline(gRenderer, &desc, &p);
    return p;
}
} // namespace

int load(int reloaded)
{
    const YeaHost* host = yea::host();
    gRenderer = (Renderer*)host->renderer;
    gQueue = (Queue*)host->queue;
    gFrames = host->framesInFlight < kMaxFrames ? host->framesInFlight : kMaxFrames;
    const char* gtao[4] = { "renderer_gtao_1.comp", "renderer_gtao_2.comp", "renderer_gtao_4.comp", "renderer_gtao_8.comp" };
    const char* blur[4] = { "renderer_ao_denoise_1.comp", "renderer_ao_denoise_2.comp", "renderer_ao_denoise_4.comp",
                            "renderer_ao_denoise_8.comp" };
    for (int i = 0; i < 4; ++i)
    {
        gGtaoShader[i] = loadCompute(gtao[i]);
        gBlurShader[i] = loadCompute(blur[i]);
        if (!gGtaoShader[i] || !gBlurShader[i])
            return 1;
        gGtaoPipe[i] = computePipeline(gGtaoShader[i], "renderer_gtao");
        gBlurPipe[i] = computePipeline(gBlurShader[i], "renderer_ao_denoise");
    }
    DescriptorSetDesc setDesc = SRT_SET_DESC(GtaoSrt, PerFrame, gFrames * kSetsPerFrame, 0);
    addDescriptorSet(gRenderer, &setDesc, &gSet);
    for (uint32_t i = 0; i < gFrames; ++i)
    {
        BufferLoadDesc d = {};
        d.mDesc.mDescriptors = DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        d.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
        d.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
        d.mDesc.mSize = 256;
        d.mDesc.mStartState = RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
        d.mDesc.pName = "renderer_ao_cb";
        d.ppBuffer = &gCb[i];
        addResource(&d, nullptr);
    }
    waitForAllResourceLoads();
    gReady = gSet != nullptr;
    return gReady ? 0 : 1;
}

void unload()
{
    if (!gRenderer)
        return;
    waitQueueIdle(gQueue);
    freeTextures();
    for (uint32_t i = 0; i < gFrames; ++i)
        if (gCb[i])
            removeResource(gCb[i]);
    if (gSet)
        removeDescriptorSet(gRenderer, gSet);
    for (int i = 0; i < 4; ++i)
    {
        if (gGtaoPipe[i])
            removePipeline(gRenderer, gGtaoPipe[i]);
        if (gBlurPipe[i])
            removePipeline(gRenderer, gBlurPipe[i]);
        if (gGtaoShader[i])
            removeShader(gRenderer, gGtaoShader[i]);
        if (gBlurShader[i])
            removeShader(gRenderer, gBlurShader[i]);
    }
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
        snprintf(line, sizeof line, "renderer_ao: the core's frame is v%u, this module's v%u: standing aside", core, RENDER_FRAME_VERSION);
        yea::log(line);
        logged = core + 1;
    }
    return false;
}

bool ao_prepare(uint64_t address)
{
    if (!coreMatches())
        return false;
    RenderFrame* F = (RenderFrame*)(uintptr_t)address;
    if (!gReady || !settings.enabled || !F || F->version != RENDER_FRAME_VERSION || !F->width || !F->height)
        return false;
    if (yea::viewMode()) // F1's Wireframe or Overdraw: stand aside so they read right
        return false;
    uint32_t scale = settings.half_resolution ? 2 : 1;
    if (F->width != gWidth || F->height != gHeight || scale != gScale)
    {
        waitQueueIdle(gQueue);
        freeTextures();
        createTextures(F->width, F->height, scale);
    }
    if (!gTex[0] || !gTex[1])
        return false;

    gPasses = clampu(settings.denoise, 0, 3);
    float    radius = settings.radius < 0.05f ? 0.05f : settings.radius;
    float    falloff = settings.falloff < 0.05f ? 0.05f : (settings.falloff > 1 ? 1 : settings.falloff);
    AoData   d = {};
    d.proj[0] = F->p00;
    d.proj[1] = F->p11;
    d.proj[2] = F->nearZ;
    d.proj[3] = radius;
    d.params[0] = radius * (1 - falloff);
    d.params[1] = 1.0f / (radius * falloff);
    d.params[2] = settings.intensity < 0 ? 0 : settings.intensity;
    d.params[3] = (float)F->height * 0.25f; // the farthest a horizon is searched, in pixels
    d.screen[0] = (float)F->width;
    d.screen[1] = (float)F->height;
    d.screen[2] = F->jitter[0];
    d.screen[3] = F->jitter[1];
    d.counts[0] = clampu(settings.slices, 1, 4);
    d.counts[1] = clampu(settings.steps, 2, 12);
    d.counts[2] = F->taaOn ? (uint32_t)(F->frameNumber % 64) : 0; // TAA averages the noise over frames
    d.counts[3] = gScale;
    memcpy(gCb[F->frameIndex]->pCpuMappedAddress, &d, sizeof d);

    F->ao = output();
    F->aoOn = 1;
    F->aoScale = gScale;
    F->aoSpecular = settings.specular_occlusion < 0 ? 0 : (settings.specular_occlusion > 1 ? 1 : settings.specular_occlusion);
    F->aoMultiBounce = settings.multi_bounce;
    F->aoDebug = settings.show_ao;
    return true;
}

void ao_run(uint64_t address)
{
    if (!coreMatches())
        return;
    RenderFrame* F = (RenderFrame*)(uintptr_t)address;
    if (!gReady || !F || F->ao != output() || !F->depth)
        return;
    Cmd*     cmd = F->cmd;
    uint32_t fi = F->frameIndex, s = sampleIndex(F->sampleCount);
    uint32_t aw = (F->width + gScale - 1) / gScale, ah = (F->height + gScale - 1) / gScale;
    uint32_t gx = (aw + 7) / 8, gy = (ah + 7) / 8;
    for (uint32_t k = 0; k < 2u; ++k)
    {
        Texture*       in = gTex[k ^ 1];
        Texture*       out = gTex[k];
        DescriptorData p[4] = {};
        p[0].mIndex = SRT_RES_IDX(GtaoSrt, PerFrame, gAo);
        p[0].ppBuffers = &gCb[fi];
        p[1].mIndex = SRT_RES_IDX(GtaoSrt, PerFrame, gDepth);
        p[1].ppTextures = &F->depth->pTexture;
        p[2].mIndex = SRT_RES_IDX(GtaoSrt, PerFrame, gIn);
        p[2].ppTextures = &in;
        p[3].mIndex = SRT_RES_IDX(GtaoSrt, PerFrame, gOut);
        p[3].ppTextures = &out;
        updateDescriptorSet(gRenderer, fi * kSetsPerFrame + k, gSet, 4, p);
    }

    cmdBeginDebugMarker(cmd, 0.6f, 0.6f, 0.6f, "Renderer ambient occlusion");
    barrier(cmd, gTex[0], RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_UNORDERED_ACCESS);
    cmdBindPipeline(cmd, gGtaoPipe[s]);
    cmdBindDescriptorSet(cmd, fi * kSetsPerFrame, gSet);
    cmdDispatch(cmd, gx, gy, 1);
    barrier(cmd, gTex[0], RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_SHADER_RESOURCE);
    for (uint32_t i = 0; i < gPasses; ++i)
    {
        uint32_t k = (i & 1) ^ 1;
        Texture* out = gTex[k];
        barrier(cmd, out, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_UNORDERED_ACCESS);
        cmdBindPipeline(cmd, gBlurPipe[s]);
        cmdBindDescriptorSet(cmd, fi * kSetsPerFrame + k, gSet);
        cmdDispatch(cmd, gx, gy, 1);
        barrier(cmd, out, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_SHADER_RESOURCE);
    }
    cmdEndDebugMarker(cmd);
}
} // namespace renderer_ao
