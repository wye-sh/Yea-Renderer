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
#include <yea/device.h>

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
#else
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything" // AMD's headers, as AMD wrote them
#include "third_party/ffx_vk/include/FidelityFX/host/ffx_fsr3upscaler.h"
#include "third_party/ffx_vk/include/FidelityFX/host/backends/vk/ffx_vk.h"
#pragma clang diagnostic pop
bool fsrVkLoadEntryPoints(VkInstance instance); // fsr_vk.cpp
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

// Lines said from upscale_prepare (the core's call, which would file them under renderer) wait here for this module's
// own update(), and are said there under renderer_upscale. A few, fixed size: no allocation.
char     gSay[4][1024];
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
HMODULE              gFfxUpscalerDll;     // loaded first, so the loader finds it
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
    if (gFfxUpscalerDll)
        FreeLibrary(gFfxUpscalerDll);
    gFfxDll = nullptr;
    gFfxUpscalerDll = nullptr;
    gFfxCreate = nullptr;
    gFfxDestroy = nullptr;
    gFfxDispatch = nullptr;
}

// One of AMD's DLLs from this module's assets, by its full path. A module's Assets folder isn't on Windows' DLL search
// path, and yea::asset joins with '/', which Windows' loader doesn't take as a path: backslashes, and the DLL's own
// folder searched for its dependencies (never SetDllDirectory or the current directory: both are process-wide).
HMODULE fsrLoadAsset(const char* name)
{
    char    line[1400];
    wchar_t wide[1024] = L"";
    // At the top of Assets/ (where a Yea from 133 puts its platform's files), else still in windows/.
    for (int at = 0; at < 2; ++at)
    {
        char file[128], path[1024];
        snprintf(file, sizeof file, "%s%s", at ? "windows/" : "", name);
        wchar_t w[1024];
        if (!yea::asset(file, path, sizeof path) || !MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, w, 1024))
            continue;
        for (wchar_t* c = w; *c; ++c)
            if (*c == L'/')
                *c = L'\\';
        if (GetFileAttributesW(w) == INVALID_FILE_ATTRIBUTES)
            continue;
        wcscpy(wide, w);
        if (at)
        {
            snprintf(line, sizeof line, "renderer_upscale: FSR's %s is still in Assets/windows/ on this Yea", name);
            say(line);
        }
        break;
    }
    if (!wide[0])
    {
        // Name what this version's Assets/ holds instead.
        char    found[512] = "";
        wchar_t pattern[1024];
        if (MultiByteToWideChar(CP_UTF8, 0, yea::assets(), -1, pattern, 1000))
        {
            wcscat(pattern, L"\\*");
            for (wchar_t* c = pattern; *c; ++c)
                if (*c == L'/')
                    *c = L'\\';
            WIN32_FIND_DATAW f;
            HANDLE           h = FindFirstFileW(pattern, &f);
            if (h != INVALID_HANDLE_VALUE)
            {
                do
                    if (f.cFileName[0] != L'.')
                        snprintf(found + strlen(found), sizeof found - strlen(found), " %ls%s", f.cFileName,
                                 (f.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) ? "/" : "");
                while (FindNextFileW(h, &f));
                FindClose(h);
            }
        }
        snprintf(line, sizeof line, "renderer_upscale: FSR's %s isn't in this module's assets (%s holds:%s); TAA instead", name,
                 yea::assets(), found[0] ? found : " nothing");
        say(line);
        return nullptr;
    }
    HMODULE m = LoadLibraryExW(wide, nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!m)
    {
        // Name every import of AMD's two DLLs that this PC can't load (the Visual C++ runtime among them).
        const DWORD    error = GetLastError();
        const wchar_t* imports[] = { L"kernel32.dll", L"dxgi.dll", L"version.dll", L"vcruntime140.dll", L"vcruntime140_1.dll",
                                     L"msvcp140.dll", L"api-ms-win-crt-runtime-l1-1-0.dll", L"api-ms-win-crt-heap-l1-1-0.dll",
                                     L"api-ms-win-crt-string-l1-1-0.dll", L"api-ms-win-crt-locale-l1-1-0.dll",
                                     L"api-ms-win-crt-stdio-l1-1-0.dll", L"api-ms-win-crt-convert-l1-1-0.dll",
                                     L"api-ms-win-crt-math-l1-1-0.dll", L"api-ms-win-crt-environment-l1-1-0.dll" };
        char           missing[512] = "";
        for (const wchar_t* dll : imports)
        {
            HMODULE d = LoadLibraryExW(dll, nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
            if (d)
                FreeLibrary(d);
            else
                snprintf(missing + strlen(missing), sizeof missing - strlen(missing), " %ls", dll);
        }
        snprintf(line, sizeof line, "renderer_upscale: FSR's %s didn't load from %ls (Windows error %lu; imports missing:%s); TAA instead",
                 name, wide, error, missing[0] ? missing : " none");
        say(line);
    }
    return m;
}

// AMD's upscaler, then its loader. The loader opens the upscaler by bare name, which Windows answers with the one
// already loaded.
bool fsrLoad()
{
    if (gFfxDll)
        return true;
    if (gFfxFailed || gRenderer->mRendererApi != RENDERER_API_D3D12)
        return false;
    if (!gFfxUpscalerDll)
        gFfxUpscalerDll = fsrLoadAsset("amd_fidelityfx_upscaler_dx12.dll");
    if (gFfxUpscalerDll)
        gFfxDll = fsrLoadAsset("amd_fidelityfx_loader_dx12.dll");
    if (!gFfxDll)
    {
        fsrUnload();
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
#else
// --- FSR on Vulkan: AMD's FSR 3.1 upscaler and Vulkan backend, compiled in from AMD's MIT source (fsr_vk.cpp) ------
// The backend's working memory: allocated in this module's own update() and freed there or in unload() (never in
// upscale_prepare, which the core calls: an allocation there would be the core's). Nothing else of AMD's allocates.
uint8_t*               gFfxScratchBlock; // as allocated
uint8_t*               gFfxScratch;      // 64-byte aligned within it (AMD's backend keeps SSE-aligned members there)
size_t                 gFfxScratchSize;
FfxInterface           gFfxInterface;
FfxGetDeviceCapabilitiesFunc gFfxDeviceCaps; // the backend's own query, which the capability switch wraps
FfxFsr3UpscalerContext gFfx;
bool                   gFfxLive;
uint32_t               gFfxWidth, gFfxHeight; // the display size the context was made for
bool                   gFfxFailed;            // refused (said once; tried again on a new pick)
bool                   gVkLoaded;             // AMD's direct Vulkan calls are resolved
// FSR 3.1's upscaler leaves three of its textures to its caller (AMD shares them with frame generation): dilated
// depth, dilated motion and the reconstructed previous depth, made here in a backend context of their own.
const size_t           kFfxContexts = FFX_FSR3UPSCALER_CONTEXT_COUNT + 1;
uint32_t               gFfxSharedId;
FfxResourceInternal    gFfxShared[3];         // reconstructedPrevNearestDepth, dilatedDepth, dilatedMotionVectors
bool                   gFfxSharedLive;
int                    gFfxSharedMade;        // how many of gFfxShared exist

// AMD's backend reports what the GPU supports, not what Yea's device switched on, so FSR's shader choice is capped
// by yea::device(): its 16-bit shaders only with 16-bit maths and storage on, a wave size only with wave-size control.
const YeaDevice* gDevice; // taken in load()

FfxErrorCode fsrDeviceCaps(FfxInterface* backend, FfxDeviceCapabilities* caps)
{
    FfxErrorCode e = gFfxDeviceCaps(backend, caps);
    const uint32_t on = gDevice ? gDevice->features : 0;
    const uint32_t fp16 = YEA_DEVICE_FLOAT16 | YEA_DEVICE_STORAGE16;
    caps->fp16Supported = caps->fp16Supported && (on & fp16) == fp16;
    if (!(on & YEA_DEVICE_WAVE_SIZE)) // one fixed width: AMD's backend then never asks for a subgroup size
        caps->waveLaneCountMin = caps->waveLaneCountMax = gDevice ? gDevice->waveMin : gRenderer->pGpu->mWaveLaneCount;
    // Extensions only AMD's crash breadcrumbs use (off here), which Yea's device doesn't enable either.
    caps->deviceCoherentMemorySupported = false;
    caps->bufferMarkerSupported = false;
    caps->extendedSynchronizationSupported = false;
    return e;
}

void fsrDestroyContext()
{
    if (gFfxLive || gFfxSharedLive)
        waitQueueIdle(gQueue);
    if (gFfxSharedLive)
    {
        for (int i = 0; i < gFfxSharedMade; ++i)
            gFfxInterface.fpDestroyResource(&gFfxInterface, gFfxShared[i], gFfxSharedId);
        gFfxInterface.fpDestroyBackendContext(&gFfxInterface, gFfxSharedId);
        gFfxSharedMade = 0;
    }
    gFfxSharedLive = false;
    if (gFfxLive)
        ffxFsr3UpscalerContextDestroy(&gFfx);
    gFfxLive = false;
    gFfxWidth = gFfxHeight = 0;
}

// Called from this module's update(): the backend's memory while FSR is picked, let go when it isn't.
void fsrMemory(bool want)
{
    if (want && !gFfxScratch && !gFfxFailed && gRenderer->mRendererApi == RENDERER_API_VULKAN)
    {
        gVkLoaded = gVkLoaded || fsrVkLoadEntryPoints(gRenderer->pContext->mVk.pInstance);
        if (!gVkLoaded)
        {
            say("renderer_upscale: FSR couldn't reach Vulkan; TAA instead");
            gFfxFailed = true;
            return;
        }
        gFfxScratchSize = ffxGetScratchMemorySizeVK(gRenderer->pGpu->mVk.pGpu, kFfxContexts);
        gFfxScratchBlock = new uint8_t[gFfxScratchSize + 63];
        gFfxScratch = (uint8_t*)(((uintptr_t)gFfxScratchBlock + 63) & ~(uintptr_t)63);
        memset(gFfxScratch, 0, gFfxScratchSize);
    }
    if (!want)
    {
        fsrDestroyContext();
        delete[] gFfxScratchBlock;
        gFfxScratchBlock = gFfxScratch = nullptr;
        gFfxScratchSize = 0;
        gFfxFailed = false;
    }
}

bool fsrContext(uint32_t dw, uint32_t dh)
{
    if (gFfxLive && gFfxWidth == dw && gFfxHeight == dh)
        return true;
    if (!gFfxScratch || gFfxFailed)
        return false;
    fsrDestroyContext();
    VkDeviceContext device = { gRenderer->mVk.pDevice, gRenderer->pGpu->mVk.pGpu, nullptr }; // its own vkGetDeviceProcAddr
    memset(gFfxScratch, 0, gFfxScratchSize);
    if (ffxGetInterfaceVK(&gFfxInterface, ffxGetDeviceVK(&device), gFfxScratch, gFfxScratchSize, kFfxContexts) != FFX_OK)
    {
        say("renderer_upscale: FSR's Vulkan backend refused this GPU; TAA instead");
        gFfxFailed = true;
        return false;
    }
    gFfxDeviceCaps = gFfxInterface.fpGetDeviceCapabilities;
    gFfxInterface.fpGetDeviceCapabilities = fsrDeviceCaps;
    FfxFsr3UpscalerContextDescription desc = {};
    // Linear HDR colour, reversed-Z depth with an infinite far plane, motion at render size without the jitter.
    desc.flags = FFX_FSR3UPSCALER_ENABLE_HIGH_DYNAMIC_RANGE | FFX_FSR3UPSCALER_ENABLE_DEPTH_INVERTED | FFX_FSR3UPSCALER_ENABLE_DEPTH_INFINITE;
    desc.maxRenderSize = { dw, dh };
    desc.maxUpscaleSize = { dw, dh };
    desc.backendInterface = gFfxInterface;
    const FfxErrorCode e = ffxFsr3UpscalerContextCreate(&gFfx, &desc);
    if (e != FFX_OK)
    {
        char line[96];
        snprintf(line, sizeof line, "renderer_upscale: FSR refused this GPU (error 0x%x); TAA instead", (unsigned)e);
        say(line);
        gFfxFailed = true;
        return false;
    }
    gFfxLive = true;
    FfxFsr3UpscalerSharedResourceDescriptions shared = {};
    bool ok = ffxFsr3UpscalerGetSharedResourceDescriptions(&gFfx, &shared) == FFX_OK &&
              gFfxInterface.fpCreateBackendContext(&gFfxInterface, FFX_EFFECT_SHAREDRESOURCES, nullptr, &gFfxSharedId) == FFX_OK;
    gFfxSharedLive = ok;
    const FfxCreateResourceDescription* descs[3] = { &shared.reconstructedPrevNearestDepth, &shared.dilatedDepth, &shared.dilatedMotionVectors };
    for (int i = 0; ok && i < 3; ++i)
        if ((ok = gFfxInterface.fpCreateResource(&gFfxInterface, descs[i], gFfxSharedId, &gFfxShared[i]) == FFX_OK))
            gFfxSharedMade = i + 1;
    if (!ok)
    {
        say("renderer_upscale: FSR's working textures couldn't be made; TAA instead");
        fsrDestroyContext();
        gFfxFailed = true;
        return false;
    }
    gFfxWidth = dw;
    gFfxHeight = dh;
    return true;
}

FfxResource fsrResource(VkImage image, FfxSurfaceFormat format, uint32_t w, uint32_t h, FfxResourceUsage usage,
                        FfxResourceStates state, const wchar_t* name)
{
    FfxResourceDescription d = {};
    d.type = FFX_RESOURCE_TYPE_TEXTURE2D;
    d.format = format;
    d.width = w;
    d.height = h;
    d.depth = 1;
    d.mipCount = 1;
    d.flags = FFX_RESOURCE_FLAGS_NONE;
    d.usage = usage;
    return ffxGetResourceVK((void*)image, d, name, state);
}

bool fsrRun(const RenderFrame* F, RenderTarget* out, bool reset, float dtMs)
{
    const FfxResourceStates srv = FFX_RESOURCE_STATE_COMPUTE_READ;
    FfxFsr3UpscalerDispatchDescription d = {};
    d.commandList = ffxGetCommandListVK(F->cmd->mVk.pCmdBuf);
    d.color = fsrResource(F->hdr->pTexture->mVk.pImage, FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT, F->width, F->height,
                          FFX_RESOURCE_USAGE_READ_ONLY, srv, L"renderer_hdr");
    d.depth = fsrResource(F->depth->pTexture->mVk.pImage, FFX_SURFACE_FORMAT_R32_FLOAT, F->width, F->height,
                          FFX_RESOURCE_USAGE_DEPTHTARGET, srv, L"renderer_depth");
    d.motionVectors = fsrResource(F->motion->pTexture->mVk.pImage, FFX_SURFACE_FORMAT_R16G16_FLOAT, F->width, F->height,
                                  FFX_RESOURCE_USAGE_READ_ONLY, srv, L"renderer_motion");
    d.exposure = fsrResource(gExposureTex->mVk.pImage, FFX_SURFACE_FORMAT_R32_FLOAT, 1, 1, FFX_RESOURCE_USAGE_READ_ONLY, srv,
                             L"renderer_upscale_exposure");
    d.output = fsrResource(out->pTexture->mVk.pImage, FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT, F->displayWidth, F->displayHeight,
                           FFX_RESOURCE_USAGE_UAV, FFX_RESOURCE_STATE_UNORDERED_ACCESS, L"renderer_upscale_out");
    d.reconstructedPrevNearestDepth = gFfxInterface.fpGetResource(&gFfxInterface, gFfxShared[0]);
    d.dilatedDepth = gFfxInterface.fpGetResource(&gFfxInterface, gFfxShared[1]);
    d.dilatedMotionVectors = gFfxInterface.fpGetResource(&gFfxInterface, gFfxShared[2]);
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
    bool ok = ffxFsr3UpscalerContextDispatch(&gFfx, &d) == FFX_OK;
    F->cmd->mVk.pBoundPipeline = VK_NULL_HANDLE; // FSR bound its own pipelines and descriptor sets
    F->cmd->mVk.pBoundPipelineLayout = VK_NULL_HANDLE;
    return ok;
}
#endif

// FSR ready for this display size (its files, its context).
bool fsrReady(uint32_t dw, uint32_t dh)
{
#if defined(_WIN32)
    return fsrLoad() && fsrContext(dw, dh);
#else
    return fsrContext(dw, dh);
#endif
}

// FSR not running this frame; `wanted` when it was asked for (and refused: that is said once, not tried again).
void fsrLetGo(bool wanted)
{
#if defined(_WIN32)
    if (gFfxDll)
        fsrUnload(); // AMD's files are let go while FSR isn't needed
    if (!wanted)
        gFfxFailed = false;
#else
    (void)wanted;
    fsrDestroyContext(); // its memory goes in update()
#endif
}

// Which upscaler runs this frame.
enum class Active : uint32_t
{
    Taa, // none: the core's own TAA
    Fsr
};
Active gActive, gLastActive;
} // namespace

int load(int reloaded)
{
    const YeaHost* host = yea::host();
    gRenderer = (Renderer*)host->renderer;
    gQueue = (Queue*)host->queue;
    gFrames = host->framesInFlight < kMaxFrames ? host->framesInFlight : kMaxFrames;
#if !defined(_WIN32)
    gDevice = yea::device();
#endif

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
    t.mFlags = TEXTURE_CREATION_FLAG_FORCE_2D; // a 2D texture, as the shaders read it (1 x 1 would otherwise be made 1D)
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
#else
    fsrMemory(false);
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
#if !defined(_WIN32)
    // FSR's working memory, in this module's own call: while an upscaler is picked.
    if (gReady)
        fsrMemory(settings.upscaler != Upscaler::TAA && !yea::viewMode());
#endif
}
void draw(uint32_t pass, const YeaFrame* frame) {}

bool upscale_prepare(uint64_t address)
{
    RenderFrame* F = (RenderFrame*)(uintptr_t)address;
    if (!gReady || !F || F->version != RENDER_FRAME_VERSION || !F->width || !F->height)
        return false;
    const uint32_t dw = F->displayWidth, dh = F->displayHeight;
    const uint32_t q = (uint32_t)settings.quality_mode < 5 ? (uint32_t)settings.quality_mode : 1;
    const bool     picked = settings.upscaler != Upscaler::TAA && !yea::viewMode(); // F1's view modes: plain frames
    // FSR where it can run, else TAA.
    uint32_t w = 0, h = 0;
    gActive = Active::Taa;
    if (picked && gActive == Active::Taa && fsrReady(dw, dh))
    {
        gActive = Active::Fsr;
        w = (uint32_t)((float)dw / kRatio[q]);
        h = (uint32_t)((float)dh / kRatio[q]);
    }
    if (gActive != Active::Fsr)
        fsrLetGo(picked && gActive == Active::Taa);
    // Which upscaler runs, and its sizes: said when they change (what F1's knobs picked, and what could run).
    {
        static uint32_t said[6] = { ~0u, 0, 0, 0, 0, 0 };
        const uint32_t  now[6] = { (uint32_t)gActive, (uint32_t)settings.upscaler, w, h, dw, dh };
        if (memcmp(said, now, sizeof now))
        {
            memcpy(said, now, sizeof now);
            const char* names[2] = { "TAA", "FSR" };
            char        line[192];
            if (gActive == Active::Taa)
                snprintf(line, sizeof line, "renderer_upscale: %s picked; TAA runs at %ux%u", names[now[1] % 2], dw, dh);
            else
                snprintf(line, sizeof line, "renderer_upscale: %s picked; %s renders %ux%u, upscaled to %ux%u", names[now[1] % 2],
                         names[now[0] % 2], w, h, dw, dh);
            say(line, 0);
        }
    }
    if (gActive == Active::Taa)
    {
        if (gOut[0] || gOut[1])
        {
            waitQueueIdle(gQueue);
            freeOutputs();
        }
        gLastActive = Active::Taa;
        return false;
    }
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
    w = w < 1 ? 1 : (w > dw ? dw : w);
    h = h < 1 ? 1 : (h > dh ? dh : h);

    // The upscaler's jitter: Halton (2, 3) over 8 x ratio^2 phases, as AMD and NVIDIA both advise.
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
    // History stays valid only from the frame before, by the same upscaler at the same render size.
    bool valid = gUpscaledFrame && gUpscaledFrame + 1 == F->frameNumber && w == gRenderWidth && h == gRenderHeight &&
                 q == gQuality && gActive == gLastActive;
    gCurrent ^= 1;
    F->resolved = gOut[gCurrent];
    F->previous = valid ? gOut[gCurrent ^ 1] : nullptr;
    gRenderWidth = w;
    gRenderHeight = h;
    gQuality = q;
    gLastActive = gActive;
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
    const bool reset = F->previous == nullptr;
    bool       ok = false;
    ok = fsrRun(F, out, reset, dtMs);
    barrierRt(cmd, out, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_SHADER_RESOURCE);
    cmdEndDebugMarker(cmd);
    if (ok)
        gUpscaledFrame = F->frameNumber;
    else
        gUpscaledFrame = 0;
}
} // namespace renderer_upscale
