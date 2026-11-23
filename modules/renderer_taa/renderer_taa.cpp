// Temporal antialiasing for the renderer core. taa_prepare picks this frame's sub-pixel jitter (a
// Halton 2,3 sequence) and the history target to write; the core renders jittered at one sample with
// motion vectors; taa_resolve blends the frame into the reprojected history (Shaders/Taa.comp.fsl).
// The two history targets swap every frame; the resolved one is what the core composites.
#include "yea_renderer_taa.h"
#include "frame.h"

#include <Common_3/Graphics/Interfaces/IGraphics.h>
#include <Common_3/Resources/ResourceLoader/Interfaces/IResourceLoader.h>
#include <Common_3/Graphics/FSL/defaults.h>
#include "Shaders/Taa.srt.h"

#include <string.h>

namespace renderer_taa
{
namespace
{
struct TaaData
{
    float    params[4];
    float    texel[4];
    uint32_t size[4];
};

const uint32_t kMaxFrames = 4;

Renderer*      gRenderer;
Queue*         gQueue;
uint32_t       gFrames;
Shader*        gShader;
Pipeline*      gPipe;
DescriptorSet* gSet;
Buffer*        gCb[kMaxFrames];
RenderTarget*  gHistory[2];
Texture*       gNoReactive; // stands in for frame.h reactive when no particles drew
uint32_t       gWidth, gHeight;
uint32_t       gCurrent;
uint64_t       gResolvedFrame; // the frame the current history was resolved in
bool           gReady;

float halton(uint32_t i, uint32_t b)
{
    float f = 1, r = 0;
    for (; i; i /= b)
    {
        f /= (float)b;
        r += f * (float)(i % b);
    }
    return r;
}

void freeHistory()
{
    for (RenderTarget*& t : gHistory)
        if (t)
        {
            removeRenderTarget(gRenderer, t);
            t = nullptr;
        }
    gWidth = gHeight = 0;
}

void createHistory(uint32_t w, uint32_t h)
{
    RenderTargetDesc d = {};
    d.mWidth = w;
    d.mHeight = h;
    d.mDepth = d.mArraySize = d.mMipLevels = 1;
    d.mSampleCount = SAMPLE_COUNT_1;
    d.mFormat = TinyImageFormat_R16G16B16A16_SFLOAT;
    d.mStartState = RESOURCE_STATE_SHADER_RESOURCE;
    d.mDescriptors = (DescriptorType)(DESCRIPTOR_TYPE_TEXTURE | DESCRIPTOR_TYPE_RW_TEXTURE);
    d.pName = "renderer_taa_history";
    addRenderTarget(gRenderer, &d, &gHistory[0]);
    addRenderTarget(gRenderer, &d, &gHistory[1]);
    gWidth = w;
    gHeight = h;
    gResolvedFrame = 0;
}

void barrier(Cmd* cmd, RenderTarget* rt, ResourceState from, ResourceState to)
{
    RenderTargetBarrier b = { rt, from, to };
    cmdResourceBarrier(cmd, 0, nullptr, 0, nullptr, 1, &b);
}
} // namespace

int load(int reloaded)
{
    const YeaHost* host = yea::host();
    gRenderer = (Renderer*)host->renderer;
    gQueue = (Queue*)host->queue;
    gFrames = host->framesInFlight < kMaxFrames ? host->framesInFlight : kMaxFrames;

    ShaderLoadDesc sd = {};
    sd.mComp.pFileName = "renderer_taa.comp";
    addShader(gRenderer, &sd, &gShader);
    if (!gShader)
        return 1;

    PipelineDesc desc = {};
    desc.mType = PIPELINE_TYPE_COMPUTE;
    PIPELINE_LAYOUT_DESC(desc, NULL, SRT_LAYOUT_DESC(TaaSrt, PerFrame), NULL, NULL);
    desc.mComputeDesc.pShaderProgram = gShader;
    desc.pName = "renderer_taa";
    addPipeline(gRenderer, &desc, &gPipe);

    DescriptorSetDesc setDesc = SRT_SET_DESC(TaaSrt, PerFrame, gFrames, 0);
    addDescriptorSet(gRenderer, &setDesc, &gSet);
    for (uint32_t i = 0; i < gFrames; ++i)
    {
        BufferLoadDesc d = {};
        d.mDesc.mDescriptors = DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        d.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
        d.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
        d.mDesc.mSize = 256;
        d.mDesc.mStartState = RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
        d.mDesc.pName = "renderer_taa_cb";
        d.ppBuffer = &gCb[i];
        addResource(&d, nullptr);
    }
    {
        TextureDesc t = {};
        t.mWidth = t.mHeight = 2; // one texel high gets a 1D view on Vulkan
        t.mDepth = t.mArraySize = t.mMipLevels = 1;
        t.mSampleCount = SAMPLE_COUNT_1;
        t.mFormat = TinyImageFormat_R8_UNORM;
        t.mStartState = RESOURCE_STATE_SHADER_RESOURCE;
        t.mDescriptors = DESCRIPTOR_TYPE_TEXTURE;
        t.pName = "renderer_taa_no_reactive";
        TextureLoadDesc tl = {};
        tl.pDesc = &t;
        tl.ppTexture = &gNoReactive;
        addResource(&tl, nullptr);
    }
    waitForAllResourceLoads();
    gReady = gPipe && gSet && gNoReactive;
    return gReady ? 0 : 1;
}

void unload()
{
    if (!gRenderer)
        return;
    waitQueueIdle(gQueue);
    freeHistory();
    for (uint32_t i = 0; i < gFrames; ++i)
        if (gCb[i])
            removeResource(gCb[i]);
    if (gNoReactive)
        removeResource(gNoReactive);
    gNoReactive = nullptr;
    if (gSet)
        removeDescriptorSet(gRenderer, gSet);
    if (gPipe)
        removePipeline(gRenderer, gPipe);
    if (gShader)
        removeShader(gRenderer, gShader);
}

void update(uint32_t phase, float dt) {}
void draw(uint32_t pass, const YeaFrame* frame) {}

bool taa_prepare(uint64_t address)
{
    RenderFrame* F = (RenderFrame*)(uintptr_t)address;
    if (!gReady || !F || F->version != RENDER_FRAME_VERSION)
        return false;
    // renderer_upscale's frame: it resolves in TAA's place, so the history is let go; the sharpening is still this
    // module's (the core's tonemap pass, after the upscale).
    if (F->upscaleOn)
    {
        if (gHistory[0] || gHistory[1])
        {
            waitQueueIdle(gQueue);
            freeHistory();
        }
        F->sharpen = settings.sharpen < 0 ? 0 : (settings.sharpen > 1 ? 1 : settings.sharpen);
        return false;
    }
    if (!settings.enabled || !F->width || !F->height)
        return false;
    if (yea::viewMode()) // F1's Wireframe or Overdraw: stand aside so they read right
        return false;
    if (F->width != gWidth || F->height != gHeight)
    {
        waitQueueIdle(gQueue);
        freeHistory();
        createHistory(F->width, F->height);
    }
    if (!gHistory[0] || !gHistory[1])
        return false;

    uint32_t phases = settings.jitter_phases < 2 ? 2 : (settings.jitter_phases > 16 ? 16 : (uint32_t)settings.jitter_phases);
    uint32_t i = (uint32_t)(F->frameNumber % phases) + 1;
    F->jitter[0] = (halton(i, 2) - 0.5f) * settings.jitter;
    F->jitter[1] = (halton(i, 3) - 0.5f) * settings.jitter;
    F->taaOn = 1;
    F->sharpen = settings.sharpen < 0 ? 0 : (settings.sharpen > 1 ? 1 : settings.sharpen);
    gCurrent ^= 1;
    F->resolved = gHistory[gCurrent];

    // Without last frame's resolve (first frame, a resize, a gap) the history is not used.
    bool     valid = gResolvedFrame && gResolvedFrame + 1 == F->frameNumber;
    F->previous = valid ? gHistory[gCurrent ^ 1] : nullptr;
    float    keep = settings.history_weight < 0 ? 0 : (settings.history_weight > 0.99f ? 0.99f : settings.history_weight);
    TaaData  d = {};
    d.params[0] = valid ? 1.0f - keep : 1.0f;
    d.params[1] = settings.clamp_width;
    d.params[2] = F->reactive ? 1.0f : 0.0f;
    d.texel[0] = 1.0f / (float)F->width;
    d.texel[1] = 1.0f / (float)F->height;
    d.size[0] = F->width;
    d.size[1] = F->height;
    memcpy(gCb[F->frameIndex]->pCpuMappedAddress, &d, sizeof d);
    return true;
}

void taa_resolve(uint64_t address)
{
    RenderFrame* F = (RenderFrame*)(uintptr_t)address;
    if (!gReady || !F || F->resolved != gHistory[gCurrent] || !F->hdr || !F->motion || !F->depth || !F->exposure)
        return;
    Cmd*           cmd = F->cmd;
    RenderTarget*  out = gHistory[gCurrent];
    Texture*       reactive = F->reactive ? F->reactive : gNoReactive;
    DescriptorData p[8] = {};
    p[0].mIndex = SRT_RES_IDX(TaaSrt, PerFrame, gTaa);
    p[0].ppBuffers = &gCb[F->frameIndex];
    p[1].mIndex = SRT_RES_IDX(TaaSrt, PerFrame, gHdr);
    p[1].ppTextures = &F->hdr->pTexture;
    p[2].mIndex = SRT_RES_IDX(TaaSrt, PerFrame, gMotion);
    p[2].ppTextures = &F->motion->pTexture;
    p[3].mIndex = SRT_RES_IDX(TaaSrt, PerFrame, gDepth);
    p[3].ppTextures = &F->depth->pTexture;
    p[4].mIndex = SRT_RES_IDX(TaaSrt, PerFrame, gHistory);
    p[4].ppTextures = &gHistory[gCurrent ^ 1]->pTexture;
    p[5].mIndex = SRT_RES_IDX(TaaSrt, PerFrame, gExposure);
    p[5].ppBuffers = &F->exposure;
    p[6].mIndex = SRT_RES_IDX(TaaSrt, PerFrame, gOut);
    p[6].ppTextures = &out->pTexture;
    p[7].mIndex = SRT_RES_IDX(TaaSrt, PerFrame, gReactive);
    p[7].ppTextures = &reactive;
    updateDescriptorSet(gRenderer, F->frameIndex, gSet, 8, p);

    cmdBeginDebugMarker(cmd, 0.8f, 0.4f, 1.0f, "Renderer TAA");
    barrier(cmd, out, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_UNORDERED_ACCESS);
    cmdBindPipeline(cmd, gPipe);
    cmdBindDescriptorSet(cmd, F->frameIndex, gSet);
    cmdDispatch(cmd, (F->width + 7) / 8, (F->height + 7) / 8, 1);
    barrier(cmd, out, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_SHADER_RESOURCE);
    cmdEndDebugMarker(cmd);
    gResolvedFrame = F->frameNumber;
}
} // namespace renderer_taa
