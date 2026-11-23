// Variable-rate shading for the renderer core: after the depth prepass, Shaders/Rate.comp.fsl picks a
// shading rate per tile into Yea's rate image (yea::shadingRate()); the core binds the forward pass with
// it. On a GPU without variable-rate shading there is no rate image and this module does nothing.
#include "yea_renderer_vrs.h"
#include "frame.h"
#include <yea/shadingrate.h>

#include <Common_3/Graphics/Interfaces/IGraphics.h>
#include <Common_3/Resources/ResourceLoader/Interfaces/IResourceLoader.h>
#include <Common_3/Graphics/FSL/defaults.h>
#include "Shaders/Rate.srt.h"

#include <string.h>

namespace renderer_vrs
{
namespace
{
struct VrsData
{
    uint32_t size[4];
    uint32_t info[4];
    float    params[4];
    float    params2[4];
};

const uint32_t kMaxFrames = 4;

Renderer*             gRenderer;
Queue*                gQueue;
uint32_t              gFrames;
const YeaShadingRate* gRate;   // nullptr: no variable-rate shading on this GPU
Shader*               gShader;       // Rate.comp.fsl: a rate per tile into gPre
Pipeline*             gPipe;
Shader*               gUpShader;     // the same, reading last frame's image at display size (renderer_upscale renders smaller)
Pipeline*             gUpPipe;
bool                  gUpscaled;     // this frame renders smaller than last frame's image
Shader*               gDilateShader; // Dilate.comp.fsl: gPre into the rate image and gCopy
Pipeline*             gDilatePipe;
Texture*              gPre;          // R32_UINT, UNORDERED_ACCESS always
DescriptorSet*        gSet;
Buffer*               gCb[kMaxFrames];
Texture*              gImage;  // Yea's rate image: SHADING_RATE_SOURCE after vrs_run (made in UNORDERED_ACCESS)
Texture*              gCopy;   // the same rates for Show Rates (R32_UINT), SHADER_RESOURCE between frames
ResourceState         gImageState;
uint32_t              gWidth, gHeight, gTile;
bool                  gReady;

float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

void freeImages()
{
    if (gImage)
        removeResource(gImage);
    if (gCopy)
        removeResource(gCopy);
    if (gPre)
        removeResource(gPre);
    gImage = gCopy = gPre = nullptr;
    gWidth = gHeight = 0;
}

void createImages(uint32_t w, uint32_t h)
{
    uint32_t tw = (w + gTile - 1) / gTile, th = (h + gTile - 1) / gTile;
    gRate->addImage(gRenderer, w, h, (void**)&gImage);
    gImageState = RESOURCE_STATE_UNORDERED_ACCESS;
    TextureDesc d = {};
    d.mWidth = tw;
    d.mHeight = th;
    d.mDepth = d.mArraySize = d.mMipLevels = 1;
    d.mSampleCount = SAMPLE_COUNT_1;
    d.mFormat = TinyImageFormat_R32_UINT; // RWTex2D(uint) is r32ui on Vulkan
    d.mStartState = RESOURCE_STATE_SHADER_RESOURCE;
    d.mDescriptors = (DescriptorType)(DESCRIPTOR_TYPE_TEXTURE | DESCRIPTOR_TYPE_RW_TEXTURE);
    d.pName = "renderer_vrs_rates";
    TextureLoadDesc l = {};
    l.pDesc = &d;
    l.ppTexture = &gCopy;
    addResource(&l, nullptr);
    d.mStartState = RESOURCE_STATE_UNORDERED_ACCESS;
    d.mDescriptors = DESCRIPTOR_TYPE_RW_TEXTURE;
    d.pName = "renderer_vrs_pre";
    l.ppTexture = &gPre;
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
    gRate = yea::shadingRate();
    if (!gRate)
        return 0; // nothing to do on this GPU: the core draws at full rate
    gTile = gRate->tile ? gRate->tile : 16;
    ShaderLoadDesc sd = {};
    // Group memory sized for the GPU's tile: a variant per tile size.
    sd.mComp.pFileName = gTile <= 8 ? "renderer_vrs_rate_8.comp" : gTile <= 16 ? "renderer_vrs_rate_16.comp" : "renderer_vrs_rate_32.comp";
    addShader(gRenderer, &sd, &gShader);
    ShaderLoadDesc ud = {};
    ud.mComp.pFileName =
        gTile <= 8 ? "renderer_vrs_rate_8_up.comp" : gTile <= 16 ? "renderer_vrs_rate_16_up.comp" : "renderer_vrs_rate_32_up.comp";
    addShader(gRenderer, &ud, &gUpShader);
    if (!gShader || !gUpShader)
        return 1;
    PipelineDesc desc = {};
    desc.mType = PIPELINE_TYPE_COMPUTE;
    PIPELINE_LAYOUT_DESC(desc, NULL, SRT_LAYOUT_DESC(VrsSrt, PerFrame), NULL, NULL);
    desc.mComputeDesc.pShaderProgram = gShader;
    desc.pName = "renderer_vrs_rate";
    addPipeline(gRenderer, &desc, &gPipe);
    desc.mComputeDesc.pShaderProgram = gUpShader;
    desc.pName = "renderer_vrs_rate_up";
    addPipeline(gRenderer, &desc, &gUpPipe);
    ShaderLoadDesc dd = {};
    dd.mComp.pFileName = "renderer_vrs_dilate.comp";
    addShader(gRenderer, &dd, &gDilateShader);
    if (!gDilateShader)
        return 1;
    desc.mComputeDesc.pShaderProgram = gDilateShader;
    desc.pName = "renderer_vrs_dilate";
    addPipeline(gRenderer, &desc, &gDilatePipe);
    DescriptorSetDesc setDesc = SRT_SET_DESC(VrsSrt, PerFrame, gFrames, 0);
    addDescriptorSet(gRenderer, &setDesc, &gSet);
    for (uint32_t i = 0; i < gFrames; ++i)
    {
        BufferLoadDesc d = {};
        d.mDesc.mDescriptors = DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        d.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
        d.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
        d.mDesc.mSize = 256;
        d.mDesc.mStartState = RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
        d.mDesc.pName = "renderer_vrs_cb";
        d.ppBuffer = &gCb[i];
        addResource(&d, nullptr);
    }
    waitForAllResourceLoads();
    gReady = gPipe && gUpPipe && gDilatePipe && gSet;
    return gReady ? 0 : 1;
}

void unload()
{
    if (!gRenderer || !gRate)
        return;
    waitQueueIdle(gQueue);
    freeImages();
    for (uint32_t i = 0; i < gFrames; ++i)
        if (gCb[i])
            removeResource(gCb[i]);
    if (gSet)
        removeDescriptorSet(gRenderer, gSet);
    if (gPipe)
        removePipeline(gRenderer, gPipe);
    if (gUpPipe)
        removePipeline(gRenderer, gUpPipe);
    if (gUpShader)
        removeShader(gRenderer, gUpShader);
    if (gDilatePipe)
        removePipeline(gRenderer, gDilatePipe);
    if (gDilateShader)
        removeShader(gRenderer, gDilateShader);
    if (gShader)
        removeShader(gRenderer, gShader);
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
        snprintf(line, sizeof line, "renderer_vrs: the core's frame is v%u, this module's v%u: standing aside", core, RENDER_FRAME_VERSION);
        yea::log(line);
        logged = core + 1;
    }
    return false;
}

bool vrs_prepare(uint64_t address)
{
    if (!coreMatches())
        return false;
    RenderFrame* F = (RenderFrame*)(uintptr_t)address;
    if (!gReady || !settings.enabled || !F || F->version != RENDER_FRAME_VERSION || !F->width || !F->height)
        return false;
    // Rates come from last frame's anti-aliased image: without TAA (or on its first frame) full rate.
    if (!F->taaOn || !F->previous || !F->motion || F->sampleCount != 1 || yea::viewMode())
        return false;
    if (F->width != gWidth || F->height != gHeight)
    {
        waitQueueIdle(gQueue);
        freeImages();
        createImages(F->width, F->height);
    }
    if (!gImage || !gCopy || !gPre)
        return false;
    VrsData d = {};
    d.size[0] = F->width;
    d.size[1] = F->height;
    d.size[2] = (F->width + gTile - 1) / gTile;
    d.size[3] = (F->height + gTile - 1) / gTile;
    d.info[0] = gTile;
    d.info[1] = gRate->rates;
    d.info[2] = settings.allow_4x4;
    d.info[3] = 1;
    d.params[0] = 0.012f * clampf(settings.aggressiveness, 0.0f, 10.0f);
    d.params[1] = 0.25f * clampf(settings.motion, 0.0f, 10.0f);
    d.params[2] = 0.01f;
    d.params[3] = settings.keep_curved ? 2e-6f : 0.0f; // 1/z off a plane across a tile, relative
    d.params2[0] = clampf(settings.glossy_below, 0.0f, 1.0f);
    gUpscaled = F->displayWidth != F->width || F->displayHeight != F->height;
    d.params2[1] = 1.0f / (float)F->width;
    d.params2[2] = 1.0f / (float)F->height;
    memcpy(gCb[F->frameIndex]->pCpuMappedAddress, &d, sizeof d);
    F->rateImage = gImage;
    F->rateCopy = gCopy;
    F->rateTile = gTile;
    F->vrsDebug = settings.show_rates;
    return true;
}

void vrs_run(uint64_t address)
{
    if (!coreMatches())
        return;
    RenderFrame* F = (RenderFrame*)(uintptr_t)address;
    if (!gReady || !F || F->rateImage != gImage || !gImage || !F->depth || !F->previous || !F->motion || !F->exposure || !F->gloss)
        return;
    Cmd*           cmd = F->cmd;
    DescriptorData p[9] = {};
    p[0].mIndex = SRT_RES_IDX(VrsSrt, PerFrame, gVrs);
    p[0].ppBuffers = &gCb[F->frameIndex];
    p[1].mIndex = SRT_RES_IDX(VrsSrt, PerFrame, gPrevious);
    p[1].ppTextures = &F->previous->pTexture;
    p[2].mIndex = SRT_RES_IDX(VrsSrt, PerFrame, gMotion);
    p[2].ppTextures = &F->motion->pTexture;
    p[3].mIndex = SRT_RES_IDX(VrsSrt, PerFrame, gDepth);
    p[3].ppTextures = &F->depth->pTexture;
    p[4].mIndex = SRT_RES_IDX(VrsSrt, PerFrame, gExposure);
    p[4].ppBuffers = &F->exposure;
    p[5].mIndex = SRT_RES_IDX(VrsSrt, PerFrame, gRate);
    p[5].ppTextures = &gImage;
    p[6].mIndex = SRT_RES_IDX(VrsSrt, PerFrame, gRateCopy);
    p[6].ppTextures = &gCopy;
    p[7].mIndex = SRT_RES_IDX(VrsSrt, PerFrame, gPre);
    p[7].ppTextures = &gPre;
    p[8].mIndex = SRT_RES_IDX(VrsSrt, PerFrame, gGloss);
    p[8].ppTextures = &F->gloss->pTexture;
    updateDescriptorSet(gRenderer, F->frameIndex, gSet, 9, p);

    cmdBeginDebugMarker(cmd, 1.0f, 0.5f, 0.2f, "Renderer VRS rates");
    if (gImageState != RESOURCE_STATE_UNORDERED_ACCESS)
        barrier(cmd, gImage, gImageState, RESOURCE_STATE_UNORDERED_ACCESS);
    barrier(cmd, gCopy, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_UNORDERED_ACCESS);
    cmdBindPipeline(cmd, gUpscaled ? gUpPipe : gPipe);
    cmdBindDescriptorSet(cmd, F->frameIndex, gSet);
    uint32_t tw = (F->width + gTile - 1) / gTile, th = (F->height + gTile - 1) / gTile;
    cmdDispatch(cmd, tw, th, 1);
    barrier(cmd, gPre, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_UNORDERED_ACCESS);
    cmdBindPipeline(cmd, gDilatePipe);
    cmdBindDescriptorSet(cmd, F->frameIndex, gSet);
    cmdDispatch(cmd, (tw + 7) / 8, (th + 7) / 8, 1);
    barrier(cmd, gImage, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_SHADING_RATE_SOURCE);
    barrier(cmd, gCopy, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_SHADER_RESOURCE);
    gImageState = RESOURCE_STATE_SHADING_RATE_SOURCE;
    cmdEndDebugMarker(cmd);
}
} // namespace renderer_vrs
