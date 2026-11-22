// Ambient occlusion for the renderer core: GTAO from the depth prepass (Shaders/Gtao.comp.fsl), then
// `denoise` passes of a depth-aware 3x3 blur, at full or half size (then a depth-aware upsample), into a
// full-size R32F visibility texture the forward pass reads for its ambient light (diffuse with multi-bounce, specular with specular occlusion).
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
const uint32_t kSetsPerFrame = 3; // 0: read B, write A (GTAO, and blurs back into A); 1: read A, write B; 2: upsample

Renderer*      gRenderer;
Queue*         gQueue;
uint32_t       gFrames;
Shader*        gGtaoShader[4];
Shader*        gBlurShader[4];
Shader*        gUpShader[4];
Pipeline*      gGtaoPipe[4];
Pipeline*      gBlurPipe[4];
Pipeline*      gUpPipe[4];
DescriptorSet* gSet;
Buffer*        gCb[kMaxFrames];
Texture*       gTex[2]; // A, B (at the occlusion's own size): SHADER_RESOURCE between passes
Texture*       gFull;   // the upsampled result, at half size only
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
    if (gFull)
        removeResource(gFull);
    gFull = nullptr;
    gWidth = gHeight = gScale = 0;
}

Texture* output() { return gScale == 2 ? gFull : gTex[gPasses & 1]; } // each denoise pass swaps A and B

void createTextures(uint32_t w, uint32_t h, uint32_t scale)
{
    Texture** all[3] = { &gTex[0], &gTex[1], &gFull };
    for (uint32_t i = 0; i < (scale == 2 ? 3u : 2u); ++i)
    {
        Texture*&   t = *all[i];
        TextureDesc d = {};
        d.mWidth = i < 2 ? (w + scale - 1) / scale : w;
        d.mHeight = i < 2 ? (h + scale - 1) / scale : h;
        d.mDepth = d.mArraySize = d.mMipLevels = 1;
        d.mSampleCount = SAMPLE_COUNT_1;
        d.mFormat = TinyImageFormat_R32_SFLOAT; // RWTex2D(float) is r32f on Vulkan
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
    const char* up[4] = { "renderer_ao_upsample_1.comp", "renderer_ao_upsample_2.comp", "renderer_ao_upsample_4.comp",
                          "renderer_ao_upsample_8.comp" };
    for (int i = 0; i < 4; ++i)
    {
        gGtaoShader[i] = loadCompute(gtao[i]);
        gBlurShader[i] = loadCompute(blur[i]);
        gUpShader[i] = loadCompute(up[i]);
        if (!gGtaoShader[i] || !gBlurShader[i] || !gUpShader[i])
            return 1;
        gGtaoPipe[i] = computePipeline(gGtaoShader[i], "renderer_gtao");
        gBlurPipe[i] = computePipeline(gBlurShader[i], "renderer_ao_denoise");
        gUpPipe[i] = computePipeline(gUpShader[i], "renderer_ao_upsample");
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
        if (gUpPipe[i])
            removePipeline(gRenderer, gUpPipe[i]);
        if (gUpShader[i])
            removeShader(gRenderer, gUpShader[i]);
    }
}

void update(uint32_t phase, float dt) {}
void draw(uint32_t pass, const YeaFrame* frame) {}

bool ao_prepare(uint64_t address)
{
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
    if (!gTex[0] || !gTex[1] || (gScale == 2 && !gFull))
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
    F->aoSpecular = settings.specular_occlusion < 0 ? 0 : (settings.specular_occlusion > 1 ? 1 : settings.specular_occlusion);
    F->aoMultiBounce = settings.multi_bounce;
    F->aoDebug = settings.show_ao;
    return true;
}

void ao_run(uint64_t address)
{
    RenderFrame* F = (RenderFrame*)(uintptr_t)address;
    if (!gReady || !F || F->ao != output() || !F->depth)
        return;
    Cmd*     cmd = F->cmd;
    uint32_t fi = F->frameIndex, s = sampleIndex(F->sampleCount);
    uint32_t aw = (F->width + gScale - 1) / gScale, ah = (F->height + gScale - 1) / gScale;
    uint32_t gx = (aw + 7) / 8, gy = (ah + 7) / 8;
    for (uint32_t k = 0; k < (gScale == 2 ? 3u : 2u); ++k)
    {
        Texture*       in = k == 2 ? gTex[gPasses & 1] : gTex[k ^ 1];
        Texture*       out = k == 2 ? gFull : gTex[k];
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
    if (gScale == 2)
    {
        barrier(cmd, gFull, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_UNORDERED_ACCESS);
        cmdBindPipeline(cmd, gUpPipe[s]);
        cmdBindDescriptorSet(cmd, fi * kSetsPerFrame + 2, gSet);
        cmdDispatch(cmd, (F->width + 7) / 8, (F->height + 7) / 8, 1);
        barrier(cmd, gFull, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_SHADER_RESOURCE);
    }
    cmdEndDebugMarker(cmd);
}
} // namespace renderer_ao
