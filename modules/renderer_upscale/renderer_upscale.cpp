// Upscaling for the renderer core: the scene renders smaller and a vendor upscaler brings it back to the frame's size
// in TAA's place. upscale_prepare runs first of all: when an upscaler runs, it lowers the frame's width and height to
// the render size and picks the jitter (the upscaler's own sequence); upscale_run then upscales hdr (with depth,
// motion and exposure) into one of two display-size targets (`resolved`; the other is `previous`). With TAA picked, or
// where no upscaler can run, upscale_prepare returns false and the core runs exactly as without this module.
//
// FSR (Windows, Direct3D 12): AMD's FSR SDK 2.4, through its signed loader and upscaler DLLs (Assets/), loaded only
// while FSR is picked and let go when it isn't. AMD's API headers are in third_party/ffx (MIT).
#include "yea_renderer_upscale.h"
#include "frame.h"

#include <Common_3/Graphics/Interfaces/IGraphics.h>
#include <Common_3/Resources/ResourceLoader/Interfaces/IResourceLoader.h>
#include <Common_3/Graphics/FSL/defaults.h>
#include "Shaders/Exposure.srt.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
#include "third_party/ffx/api/include/ffx_api.h"
#include "third_party/ffx/api/include/ffx_api_types.h"
#include "third_party/ffx/upscalers/include/ffx_upscale.h"
// From AMD's dx12/ffx_api_dx12.h (MIT, like the headers above): the backend description alone, as that header's inline
// helpers don't build against MinGW's Direct3D 12 headers.
#define FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12 FFX_API_MAKE_BACKEND_SUB_ID(FFX_API_BACKEND_ID_DX12, 0x02)
struct ffxCreateBackendDX12Desc
{
    ffxCreateContextDescHeader header;
    ID3D12Device*              device; ///< Device on which the backend will run.
};
#endif

namespace renderer_upscale
{
namespace
{
const uint32_t kMaxFrames = 4;

Renderer*      gRenderer;
Queue*         gQueue;
uint32_t       gFrames;
Shader*        gExposureShader;
Pipeline*      gExposurePipe;
DescriptorSet* gExposureSet;
Texture*       gExposureTex;  // R32F 1 x 1, SHADER_RESOURCE between frames
RenderTarget*  gOut[2];       // display-size RGBA16F, SHADER_RESOURCE between frames
uint32_t       gOutWidth, gOutHeight;
uint32_t       gCurrent;
uint64_t       gUpscaledFrame; // the frame gOut[gCurrent] was last written in
uint32_t       gRenderWidth, gRenderHeight;
uint32_t       gQuality;       // the quality the last frame rendered at (a change resets the history)
uint32_t       gJitterIndex;
double         gLastTime;
bool           gReady;
#if !defined(_WIN32)
bool gSaidFallback; // the fallback to TAA was logged
#endif

// Lines said from upscale_prepare (the core's call, which would file them under renderer) wait here for this module's
// own update(), and are said there under renderer_upscale. A few, fixed size: no allocation.
char     gSay[4][192];
int      gSayLevel[4]; // 0 news, 1 a warning
uint32_t gSayCount;
void say(const char* line, int level = 1)
{
    if (gSayCount < 4)
    {
        gSayLevel[gSayCount] = level;
        snprintf(gSay[gSayCount++], sizeof gSay[0], "%s", line);
    }
}

// The vendors' per-axis ratios (display / render) for Native, Quality, Balanced, Performance, Ultra Performance.
const float kRatio[5] = { 1.0f, 1.5f, 1.7f, 2.0f, 3.0f };

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

void barrierRt(Cmd* cmd, RenderTarget* rt, ResourceState from, ResourceState to)
{
    RenderTargetBarrier b = { rt, from, to };
    cmdResourceBarrier(cmd, 0, nullptr, 0, nullptr, 1, &b);
}

void barrierTex(Cmd* cmd, Texture* t, ResourceState from, ResourceState to)
{
    TextureBarrier b = {};
    b.pTexture = t;
    b.mCurrentState = from;
    b.mNewState = to;
    cmdResourceBarrier(cmd, 0, nullptr, 1, &b, 0, nullptr);
}

void freeOutputs()
{
    for (RenderTarget*& t : gOut)
        if (t)
        {
            removeRenderTarget(gRenderer, t);
            t = nullptr;
        }
    gOutWidth = gOutHeight = 0;
    gUpscaledFrame = 0;
}

bool createOutputs(uint32_t w, uint32_t h)
{
    RenderTargetDesc d = {};
    d.mWidth = w;
    d.mHeight = h;
    d.mDepth = d.mArraySize = d.mMipLevels = 1;
    d.mSampleCount = SAMPLE_COUNT_1;
    d.mFormat = TinyImageFormat_R16G16B16A16_SFLOAT;
    d.mStartState = RESOURCE_STATE_SHADER_RESOURCE;
    d.mDescriptors = (DescriptorType)(DESCRIPTOR_TYPE_TEXTURE | DESCRIPTOR_TYPE_RW_TEXTURE);
    d.pName = "renderer_upscale_out";
    addRenderTarget(gRenderer, &d, &gOut[0]);
    addRenderTarget(gRenderer, &d, &gOut[1]);
    gOutWidth = w;
    gOutHeight = h;
    gUpscaledFrame = 0;
    return gOut[0] && gOut[1];
}

#if defined(_WIN32)
// --- FSR ---------------------------------------------------------------------------------------------------------
HMODULE              gFfxDll;
PfnFfxCreateContext  gFfxCreate;
PfnFfxDestroyContext gFfxDestroy;
PfnFfxDispatch       gFfxDispatch;
ffxContext           gFfx;
uint32_t             gFfxWidth, gFfxHeight; // the display size the context was made for
bool                 gFfxFailed;            // the DLLs or the context were refused (said once; tried again on a new pick)

void fsrDestroyContext()
{
    if (gFfx && gFfxDestroy)
    {
        waitQueueIdle(gQueue);
        gFfxDestroy(&gFfx, nullptr);
    }
    gFfx = nullptr;
    gFfxWidth = gFfxHeight = 0;
}

void fsrUnload()
{
    fsrDestroyContext();
    if (gFfxDll)
        FreeLibrary(gFfxDll);
    gFfxDll = nullptr;
    gFfxCreate = nullptr;
    gFfxDestroy = nullptr;
    gFfxDispatch = nullptr;
}

// AMD's loader from this module's assets; it finds the upscaler DLL beside itself.
bool fsrLoad()
{
    if (gFfxDll)
        return true;
    if (gFfxFailed || gRenderer->mRendererApi != RENDERER_API_D3D12)
        return false;
    char path[1024];
    if (!yea::asset("amd_fidelityfx_loader_dx12.dll", path, sizeof path))
    {
        say("renderer_upscale: FSR's files are missing; TAA instead");
        gFfxFailed = true;
        return false;
    }
    wchar_t wide[1024];
    if (!MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, wide, 1024))
    {
        char line[192];
        snprintf(line, sizeof line, "renderer_upscale: FSR's path isn't UTF-8 (Windows error %lu); TAA instead", GetLastError());
        say(line);
        gFfxFailed = true;
        return false;
    }
    if (!(gFfxDll = LoadLibraryExW(wide, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH)))
    {
        // AMD's DLLs need Microsoft's Visual C++ 2015-2022 runtime: name the parts this PC lacks.
        const DWORD    error = GetLastError();
        const wchar_t* runtime[3] = { L"vcruntime140.dll", L"vcruntime140_1.dll", L"msvcp140.dll" };
        char           missing[96] = "";
        for (const wchar_t* dll : runtime)
        {
            HMODULE m = LoadLibraryExW(dll, nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
            if (m)
                FreeLibrary(m);
            else
                snprintf(missing + strlen(missing), sizeof missing - strlen(missing), " %ls", dll);
        }
        char line[192];
        snprintf(line, sizeof line, "renderer_upscale: FSR's loader didn't load (Windows error %lu; Visual C++ runtime missing:%s); TAA instead",
                 error, missing[0] ? missing : " none");
        say(line);
        gFfxFailed = true;
        return false;
    }
    gFfxCreate = (PfnFfxCreateContext)(void*)GetProcAddress(gFfxDll, "ffxCreateContext");
    gFfxDestroy = (PfnFfxDestroyContext)(void*)GetProcAddress(gFfxDll, "ffxDestroyContext");
    gFfxDispatch = (PfnFfxDispatch)(void*)GetProcAddress(gFfxDll, "ffxDispatch");
    if (!gFfxCreate || !gFfxDestroy || !gFfxDispatch)
    {
        say("renderer_upscale: FSR's loader lacks its functions; TAA instead");
        fsrUnload();
        gFfxFailed = true;
        return false;
    }
    return true;
}

// A context for this display size: it renders at any size up to it (Native included).
bool fsrContext(uint32_t dw, uint32_t dh)
{
    if (gFfx && gFfxWidth == dw && gFfxHeight == dh)
        return true;
    fsrDestroyContext();
    ffxCreateBackendDX12Desc backend = {};
    backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;
    backend.device = gRenderer->mDx.pDevice;
    ffxCreateContextDescUpscaleVersion version = {};
    version.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION;
    version.header.pNext = &backend.header;
    version.version = FFX_UPSCALER_VERSION;
    ffxCreateContextDescUpscale desc = {};
    desc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    desc.header.pNext = &version.header;
    // Linear HDR colour, reversed-Z depth with an infinite far plane, motion at render size without the jitter.
    desc.flags = FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE | FFX_UPSCALE_ENABLE_DEPTH_INVERTED | FFX_UPSCALE_ENABLE_DEPTH_INFINITE;
    desc.maxRenderSize = { dw, dh };
    desc.maxUpscaleSize = { dw, dh };
    if (gFfxCreate(&gFfx, &desc.header, nullptr) != FFX_API_RETURN_OK || !gFfx)
    {
        say("renderer_upscale: FSR refused this GPU; TAA instead");
        gFfx = nullptr;
        gFfxFailed = true;
        return false;
    }
    gFfxWidth = dw;
    gFfxHeight = dh;
    return true;
}

FfxApiResource fsrResource(void* res, uint32_t format, uint32_t w, uint32_t h, uint32_t usage, uint32_t state)
{
    FfxApiResource r = {};
    r.resource = res;
    r.description.type = FFX_API_RESOURCE_TYPE_TEXTURE2D;
    r.description.format = format;
    r.description.width = w;
    r.description.height = h;
    r.description.depth = 1;
    r.description.mipCount = 1;
    r.description.flags = FFX_API_RESOURCE_FLAGS_NONE;
    r.description.usage = usage;
    r.state = state;
    return r;
}

// The Forge keeps its descriptor heaps, root signatures and bound tables in the Cmd; FSR's dispatch sets its own.
// Its heaps back (DescriptorHeap's first member is its ID3D12DescriptorHeap) and the tables forgotten, so the next
// cmdBindDescriptorSet binds them again.
struct ForgeHeap
{
    ID3D12DescriptorHeap* pHeap;
};
void restoreForgeState(Cmd* cmd)
{
    ID3D12DescriptorHeap* heaps[2] = { ((ForgeHeap*)cmd->mDx.pBoundHeaps[0])->pHeap, ((ForgeHeap*)cmd->mDx.pBoundHeaps[1])->pHeap };
    cmd->mDx.pCmdList->SetDescriptorHeaps(2, heaps);
    cmd->mDx.pCmdList->SetGraphicsRootSignature(gRenderer->mDx.pGraphicsRootSignature);
    cmd->mDx.pCmdList->SetComputeRootSignature(gRenderer->mDx.pComputeRootSignature);
    memset(cmd->mDx.mBoundDescriptorSets, 0, sizeof(cmd->mDx.mBoundDescriptorSets));
}

bool fsrRun(const RenderFrame* F, RenderTarget* out, bool reset, float dtMs)
{
    const uint32_t srv = FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ;
    ffxDispatchDescUpscale d = {};
    d.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
    d.commandList = F->cmd->mDx.pCmdList;
    d.color = fsrResource(F->hdr->pTexture->mDx.pResource, FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT, F->width, F->height,
                          FFX_API_RESOURCE_USAGE_READ_ONLY, srv);
    d.depth = fsrResource(F->depth->pTexture->mDx.pResource, FFX_API_SURFACE_FORMAT_R32_FLOAT, F->width, F->height,
                          FFX_API_RESOURCE_USAGE_DEPTHTARGET, srv);
    d.motionVectors = fsrResource(F->motion->pTexture->mDx.pResource, FFX_API_SURFACE_FORMAT_R16G16_FLOAT, F->width, F->height,
                                  FFX_API_RESOURCE_USAGE_READ_ONLY, srv);
    d.exposure = fsrResource(gExposureTex->mDx.pResource, FFX_API_SURFACE_FORMAT_R32_FLOAT, 1, 1, FFX_API_RESOURCE_USAGE_READ_ONLY, srv);
    d.output = fsrResource(out->pTexture->mDx.pResource, FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT, F->displayWidth,
                           F->displayHeight, FFX_API_RESOURCE_USAGE_UAV, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
    d.jitterOffset = { F->jitter[0], F->jitter[1] };
    // The core's motion is uv now - uv last frame (y down); FSR wants pixels towards last frame.
    d.motionVectorScale = { -(float)F->width, -(float)F->height };
    d.renderSize = { F->width, F->height };
    d.upscaleSize = { F->displayWidth, F->displayHeight };
    d.enableSharpening = false; // the Renderer's own sharpening runs after tonemapping
    d.frameTimeDelta = dtMs;
    d.preExposure = 1.0f;
    d.reset = reset;
    d.cameraNear = F->nearZ;
    d.cameraFar = 3.4e38f; // infinite
    d.cameraFovAngleVertical = 2.0f * atanf(1.0f / F->p11);
    d.viewSpaceToMetersFactor = 1.0f;
    bool ok = gFfxDispatch(&gFfx, &d.header) == FFX_API_RETURN_OK;
    restoreForgeState(F->cmd);
    return ok;
}
#endif
} // namespace

int load(int reloaded)
{
    const YeaHost* host = yea::host();
    gRenderer = (Renderer*)host->renderer;
    gQueue = (Queue*)host->queue;
    gFrames = host->framesInFlight < kMaxFrames ? host->framesInFlight : kMaxFrames;

    ShaderLoadDesc sd = {};
    sd.mComp.pFileName = "renderer_upscale_exposure.comp";
    addShader(gRenderer, &sd, &gExposureShader);
    if (!gExposureShader)
        return 1;
    PipelineDesc desc = {};
    desc.mType = PIPELINE_TYPE_COMPUTE;
    PIPELINE_LAYOUT_DESC(desc, NULL, SRT_LAYOUT_DESC(UpscaleExposureSrt, PerFrame), NULL, NULL);
    desc.mComputeDesc.pShaderProgram = gExposureShader;
    desc.pName = "renderer_upscale_exposure";
    addPipeline(gRenderer, &desc, &gExposurePipe);
    DescriptorSetDesc setDesc = SRT_SET_DESC(UpscaleExposureSrt, PerFrame, gFrames, 0);
    addDescriptorSet(gRenderer, &setDesc, &gExposureSet);
    TextureDesc t = {};
    t.mWidth = t.mHeight = 1;
    t.mDepth = t.mArraySize = t.mMipLevels = 1;
    t.mSampleCount = SAMPLE_COUNT_1;
    t.mFormat = TinyImageFormat_R32_SFLOAT;
    t.mStartState = RESOURCE_STATE_SHADER_RESOURCE;
    t.mDescriptors = (DescriptorType)(DESCRIPTOR_TYPE_TEXTURE | DESCRIPTOR_TYPE_RW_TEXTURE);
    t.pName = "renderer_upscale_exposure";
    TextureLoadDesc tl = {};
    tl.pDesc = &t;
    tl.ppTexture = &gExposureTex;
    addResource(&tl, nullptr);
    waitForAllResourceLoads();
    gReady = gExposurePipe && gExposureSet && gExposureTex;
    return gReady ? 0 : 1;
}

void unload()
{
    if (!gRenderer)
        return;
    waitQueueIdle(gQueue);
#if defined(_WIN32)
    fsrUnload();
#endif
    freeOutputs();
    if (gExposureTex)
        removeResource(gExposureTex);
    gExposureTex = nullptr;
    if (gExposureSet)
        removeDescriptorSet(gRenderer, gExposureSet);
    gExposureSet = nullptr;
    if (gExposurePipe)
        removePipeline(gRenderer, gExposurePipe);
    gExposurePipe = nullptr;
    if (gExposureShader)
        removeShader(gRenderer, gExposureShader);
    gExposureShader = nullptr;
    gReady = false;
}

void update(uint32_t phase, float dt)
{
    for (uint32_t i = 0; i < gSayCount; ++i)
        yea::host()->log(gSayLevel[i], gSay[i]);
    gSayCount = 0;
}
void draw(uint32_t pass, const YeaFrame* frame) {}

bool upscale_prepare(uint64_t address)
{
    RenderFrame* F = (RenderFrame*)(uintptr_t)address;
    if (!gReady || !F || F->version != RENDER_FRAME_VERSION || !F->width || !F->height)
        return false;
    bool fsr = settings.upscaler == Upscaler::FSR && !yea::viewMode(); // F1's view modes: plain frames
#if defined(_WIN32)
    if (!fsr && (gFfxDll || gFfxFailed))
    {
        fsrUnload(); // let go of AMD's files while FSR isn't picked
        gFfxFailed = false;
    }
    if (fsr && (!fsrLoad() || !fsrContext(F->displayWidth, F->displayHeight)))
        fsr = false;
#else
    if (fsr && !gSaidFallback)
    {
        say("renderer_upscale: FSR isn't available on this platform yet; TAA instead");
        gSaidFallback = true;
    }
    fsr = false;
#endif
    // Which upscaler runs, and its sizes: said when they change (what F1's knob picked, and what could run).
    {
        const uint32_t q = (uint32_t)settings.quality_mode < 5 ? (uint32_t)settings.quality_mode : 1;
        const uint32_t dw = F->displayWidth, dh = F->displayHeight;
        const uint32_t w = fsr ? (uint32_t)((float)dw / kRatio[q]) : dw, h = fsr ? (uint32_t)((float)dh / kRatio[q]) : dh;
        static uint32_t said[6] = { ~0u, 0, 0, 0, 0, 0 };
        const uint32_t  now[6] = { (uint32_t)fsr, (uint32_t)settings.upscaler, w, h, dw, dh };
        if (memcmp(said, now, sizeof now))
        {
            memcpy(said, now, sizeof now);
            char line[192];
            if (fsr)
                snprintf(line, sizeof line, "renderer_upscale: FSR picked; FSR renders %ux%u, upscaled to %ux%u", w, h, dw, dh);
            else
                snprintf(line, sizeof line, "renderer_upscale: %s picked; TAA runs at %ux%u",
                         settings.upscaler == Upscaler::FSR ? "FSR" : "TAA", dw, dh);
            say(line, 0);
        }
    }
    if (!fsr)
    {
        if (gOut[0] || gOut[1])
        {
            waitQueueIdle(gQueue);
            freeOutputs();
        }
        return false;
    }
    const uint32_t dw = F->displayWidth, dh = F->displayHeight;
    if (dw != gOutWidth || dh != gOutHeight)
    {
        waitQueueIdle(gQueue);
        freeOutputs();
        if (!createOutputs(dw, dh))
        {
            freeOutputs();
            return false;
        }
    }
    uint32_t q = (uint32_t)settings.quality_mode < 5 ? (uint32_t)settings.quality_mode : 1;
    float    ratio = kRatio[q];
    uint32_t w = (uint32_t)((float)dw / ratio), h = (uint32_t)((float)dh / ratio);
    w = w < 1 ? 1 : (w > dw ? dw : w);
    h = h < 1 ? 1 : (h > dh ? dh : h);

    // The upscaler's jitter: Halton (2, 3) over 8 x ratio^2 phases, as AMD's FSR picks it.
    int32_t phases = (int32_t)(8.0f * ((float)dw / (float)w) * ((float)dw / (float)w));
    phases = phases < 1 ? 1 : phases;
    uint32_t i = 1 + (gJitterIndex++ % (uint32_t)phases);
    F->width = w;
    F->height = h;
    F->jitter[0] = halton(i, 2) - 0.5f;
    F->jitter[1] = halton(i, 3) - 0.5f;
    F->taaOn = 1;
    F->upscaleOn = 1;
    F->mipBias = log2f((float)w / (float)dw);
    // History stays valid only from the frame before at the same render size.
    bool valid = gUpscaledFrame && gUpscaledFrame + 1 == F->frameNumber && w == gRenderWidth && h == gRenderHeight && q == gQuality;
    gCurrent ^= 1;
    F->resolved = gOut[gCurrent];
    F->previous = valid ? gOut[gCurrent ^ 1] : nullptr;
    gRenderWidth = w;
    gRenderHeight = h;
    gQuality = q;
    return true;
}

void upscale_run(uint64_t address)
{
    RenderFrame* F = (RenderFrame*)(uintptr_t)address;
    if (!gReady || !F || !F->upscaleOn || F->resolved != gOut[gCurrent] || !F->hdr || !F->motion || !F->depth || !F->exposure)
        return;
    Cmd*           cmd = F->cmd;
    RenderTarget*  out = gOut[gCurrent];
    DescriptorData p[2] = {};
    p[0].mIndex = SRT_RES_IDX(UpscaleExposureSrt, PerFrame, gExposure);
    p[0].ppBuffers = &F->exposure;
    p[1].mIndex = SRT_RES_IDX(UpscaleExposureSrt, PerFrame, gOut);
    p[1].ppTextures = &gExposureTex;
    updateDescriptorSet(gRenderer, F->frameIndex, gExposureSet, 2, p);

    cmdBeginDebugMarker(cmd, 0.3f, 0.9f, 0.6f, "Renderer upscale");
    barrierTex(cmd, gExposureTex, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_UNORDERED_ACCESS);
    cmdBindPipeline(cmd, gExposurePipe);
    cmdBindDescriptorSet(cmd, F->frameIndex, gExposureSet);
    cmdDispatch(cmd, 1, 1, 1);
    barrierTex(cmd, gExposureTex, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_SHADER_RESOURCE);
    barrierRt(cmd, out, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_UNORDERED_ACCESS);
    double now = yea::host()->time();
    float  dtMs = gLastTime > 0 ? (float)((now - gLastTime) * 1000.0) : 16.7f;
    gLastTime = now;
    bool ok = false;
#if defined(_WIN32)
    ok = gFfx && fsrRun(F, out, F->previous == nullptr, dtMs);
#else
    (void)dtMs;
#endif
    barrierRt(cmd, out, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_SHADER_RESOURCE);
    cmdEndDebugMarker(cmd);
    if (ok)
        gUpscaledFrame = F->frameNumber;
    else
        gUpscaledFrame = 0;
}
} // namespace renderer_upscale
