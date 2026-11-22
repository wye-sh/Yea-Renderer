// Screen-space reflections for the renderer core. This module keeps a copy of the camera depth (full or
// half size, Shaders/Depth.comp.fsl) and the settings; the core's forward pass traces each glossy
// pixel's reflection against that copy and takes its colour from last frame's anti-aliased image
// (renderer_taa's `previous`), blended over the reflection probes (renderer's Shaders/Forward.frag.fsl).
#include "yea_renderer_ssr.h"
#include "frame.h"

#include <Common_3/Graphics/Interfaces/IGraphics.h>
#include <Common_3/Resources/ResourceLoader/Interfaces/IResourceLoader.h>
#include <Common_3/Graphics/FSL/defaults.h>
#include "Shaders/Depth.srt.h"

#include <string.h>

namespace renderer_ssr
{
namespace
{
struct SsrDepthData
{
    uint32_t size[4];
    uint32_t info[4];
};

const uint32_t kMaxFrames = 4;

Renderer*      gRenderer;
Queue*         gQueue;
uint32_t       gFrames;
Shader*        gShader;
Pipeline*      gPipe;
DescriptorSet* gSet;
Buffer*        gCb[kMaxFrames];
Texture*       gTex; // SHADER_RESOURCE between frames
uint32_t       gWidth, gHeight;
bool           gReady;

float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

void freeTexture()
{
    if (gTex)
        removeResource(gTex);
    gTex = nullptr;
    gWidth = gHeight = 0;
}

void createTexture(uint32_t w, uint32_t h)
{
    TextureDesc d = {};
    d.mWidth = w;
    d.mHeight = h;
    d.mDepth = d.mArraySize = d.mMipLevels = 1;
    d.mSampleCount = SAMPLE_COUNT_1;
    d.mFormat = TinyImageFormat_R32_SFLOAT; // RWTex2D(float) is r32f on Vulkan
    d.mStartState = RESOURCE_STATE_SHADER_RESOURCE;
    d.mDescriptors = (DescriptorType)(DESCRIPTOR_TYPE_TEXTURE | DESCRIPTOR_TYPE_RW_TEXTURE);
    d.pName = "renderer_ssr_depth";
    TextureLoadDesc l = {};
    l.pDesc = &d;
    l.ppTexture = &gTex;
    addResource(&l, nullptr);
    waitForAllResourceLoads();
    gWidth = w;
    gHeight = h;
}

void barrier(Cmd* cmd, Texture* t, ResourceState from, ResourceState to)
{
    TextureBarrier b = {};
    b.pTexture = t;
    b.mCurrentState = from;
    b.mNewState = to;
    cmdResourceBarrier(cmd, 0, nullptr, 1, &b, 0, nullptr);
}
} // namespace

int load(int reloaded)
{
    const YeaHost* host = yea::host();
    gRenderer = (Renderer*)host->renderer;
    gQueue = (Queue*)host->queue;
    gFrames = host->framesInFlight < kMaxFrames ? host->framesInFlight : kMaxFrames;
    ShaderLoadDesc sd = {};
    sd.mComp.pFileName = "renderer_ssr_depth.comp";
    addShader(gRenderer, &sd, &gShader);
    if (!gShader)
        return 1;
    PipelineDesc desc = {};
    desc.mType = PIPELINE_TYPE_COMPUTE;
    PIPELINE_LAYOUT_DESC(desc, NULL, SRT_LAYOUT_DESC(SsrDepthSrt, PerFrame), NULL, NULL);
    desc.mComputeDesc.pShaderProgram = gShader;
    desc.pName = "renderer_ssr_depth";
    addPipeline(gRenderer, &desc, &gPipe);
    DescriptorSetDesc setDesc = SRT_SET_DESC(SsrDepthSrt, PerFrame, gFrames, 0);
    addDescriptorSet(gRenderer, &setDesc, &gSet);
    for (uint32_t i = 0; i < gFrames; ++i)
    {
        BufferLoadDesc d = {};
        d.mDesc.mDescriptors = DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        d.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
        d.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
        d.mDesc.mSize = 256;
        d.mDesc.mStartState = RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
        d.mDesc.pName = "renderer_ssr_cb";
        d.ppBuffer = &gCb[i];
        addResource(&d, nullptr);
    }
    waitForAllResourceLoads();
    gReady = gPipe && gSet;
    return gReady ? 0 : 1;
}

void unload()
{
    if (!gRenderer)
        return;
    waitQueueIdle(gQueue);
    freeTexture();
    for (uint32_t i = 0; i < gFrames; ++i)
        if (gCb[i])
            removeResource(gCb[i]);
    if (gSet)
        removeDescriptorSet(gRenderer, gSet);
    if (gPipe)
        removePipeline(gRenderer, gPipe);
    if (gShader)
        removeShader(gRenderer, gShader);
}

void update(uint32_t phase, float dt) {}
void draw(uint32_t pass, const YeaFrame* frame) {}

bool ssr_prepare(uint64_t address)
{
    RenderFrame* F = (RenderFrame*)(uintptr_t)address;
    if (!gReady || !settings.enabled || !F || F->version != RENDER_FRAME_VERSION || !F->width || !F->height)
        return false;
    // Last frame's anti-aliased image is what gets reflected: without TAA (or on its first frame) none.
    if (!F->taaOn || !F->previous || F->sampleCount != 1 || yea::viewMode())
        return false;
    bool     half = settings.half_resolution;
    uint32_t w = half ? (F->width + 1) / 2 : F->width, h = half ? (F->height + 1) / 2 : F->height;
    if (w != gWidth || h != gHeight)
    {
        waitQueueIdle(gQueue);
        freeTexture();
        createTexture(w, h);
    }
    if (!gTex)
        return false;
    SsrDepthData d = {};
    d.size[0] = w;
    d.size[1] = h;
    d.size[2] = F->width;
    d.size[3] = F->height;
    d.info[0] = half ? 1 : 0;
    memcpy(gCb[F->frameIndex]->pCpuMappedAddress, &d, sizeof d);

    F->ssrDepth = gTex;
    F->ssrOn = 1;
    F->ssrSteps = (uint32_t)(settings.steps < 8 ? 8 : (settings.steps > 96 ? 96 : settings.steps));
    F->ssrDebug = settings.show_hits;
    F->ssrMaxRoughness = clampf(settings.max_roughness, 0.0f, 1.0f);
    F->ssrThickness = clampf(settings.thickness, 0.01f, 1e6f);
    F->ssrMaxDistance = clampf(settings.max_distance, 0.5f, 1e6f);
    F->ssrEdgeFade = clampf(settings.edge_fade, 0.001f, 0.5f);
    F->ssrIntensity = clampf(settings.intensity, 0.0f, 1.0f);
    F->ssrScale = half ? 0.5f : 1.0f;
    return true;
}

void ssr_run(uint64_t address)
{
    RenderFrame* F = (RenderFrame*)(uintptr_t)address;
    if (!gReady || !F || F->ssrDepth != gTex || !gTex || !F->depth)
        return;
    Cmd*           cmd = F->cmd;
    DescriptorData p[3] = {};
    p[0].mIndex = SRT_RES_IDX(SsrDepthSrt, PerFrame, gSsr);
    p[0].ppBuffers = &gCb[F->frameIndex];
    p[1].mIndex = SRT_RES_IDX(SsrDepthSrt, PerFrame, gDepth);
    p[1].ppTextures = &F->depth->pTexture;
    p[2].mIndex = SRT_RES_IDX(SsrDepthSrt, PerFrame, gOut);
    p[2].ppTextures = &gTex;
    updateDescriptorSet(gRenderer, F->frameIndex, gSet, 3, p);

    cmdBeginDebugMarker(cmd, 0.5f, 0.8f, 1.0f, "Renderer SSR depth");
    barrier(cmd, gTex, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_UNORDERED_ACCESS);
    cmdBindPipeline(cmd, gPipe);
    cmdBindDescriptorSet(cmd, F->frameIndex, gSet);
    cmdDispatch(cmd, (gWidth + 7) / 8, (gHeight + 7) / 8, 1);
    barrier(cmd, gTex, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_SHADER_RESOURCE);
    cmdEndDebugMarker(cmd);
}
} // namespace renderer_ssr
