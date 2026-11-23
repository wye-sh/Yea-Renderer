// A physical sky for the renderer core (Hillaire, "A Scalable and Production Ready Sky and Atmosphere Rendering
// Technique", EGSR 2020): Earth's atmosphere as Rayleigh and Mie scattering and ozone absorption over a planet, with
// the transmittance and multiple-scattering LUTs (rebuilt only when the atmosphere changes), and each frame the
// sky-view LUT, the aerial perspective volume, the sky's reflection image (an octahedral map with GGX mips) and its
// ambient as 9 spherical harmonics. The sun's light reaching the camera is worked out here on the CPU with the same
// model, so the core hands every module the sun as it arrives (red at sunset, gone below the horizon).
#include "yea_renderer_sky.h"
#include "frame.h"

#include <Common_3/Graphics/Interfaces/IGraphics.h>
#include <Common_3/Resources/ResourceLoader/Interfaces/IResourceLoader.h>
#include <Common_3/Graphics/FSL/defaults.h>
#include "Shaders/Sky.srt.h"
#include "Shaders/Env.srt.h"

#include <math.h>
#include <string.h>

namespace renderer_sky
{
namespace
{
struct SkyData
{
    float    radii[4], rayleigh[4], mie[4], ozone[4], sunDir[4], sunColor[4], camPos[4];
    float    camRight[4], camUp[4], camForward[4];
    uint32_t size[4];
    float    misc[4];
};

const float    kGround = 6360.0f, kTop = 6460.0f, kAerialDepth = 32.0f;
const uint32_t kMaxFrames = 4, kSlots = 8, kEnvSize = 128, kEnvMips = 6;
const uint32_t kTransW = 256, kTransH = 64, kMulti = 32, kViewW = 192, kViewH = 108, kAerial = 32;
enum Pass { kTrans, kMultiPass, kView, kAerialPass, kSh, kPasses };
const char* const kShaderNames[kPasses] = { "renderer_sky_transmittance.comp", "renderer_sky_multiscatter.comp",
                                            "renderer_sky_view.comp", "renderer_sky_aerial.comp", "renderer_sky_sh.comp" };

Renderer*      gRenderer;
Queue*         gQueue;
uint32_t       gFrames;
Shader*        gShader[kPasses], *gEnvShader;
Pipeline*      gPipe[kPasses], *gEnvPipe;
DescriptorSet* gSet, *gEnvSet;
Buffer*        gCb[kMaxFrames * kSlots], *gEnvCb[kMaxFrames * kEnvMips];
Texture*       gTransTex, *gMultiTex, *gViewTex, *gAerialTex;
Texture*       gDummy, *gDummyRw, *gDummyRw3; // stand-ins: read (SHADER_RESOURCE), written (UNORDERED_ACCESS, 2D and 3D)
Buffer*        gSh, *gDummyBuf;              // gDummyBuf: UNORDERED_ACCESS
RenderTarget*  gEnv;
bool           gReady;

// This frame's plan, from sky_prepare to sky_run.
SkyData  gData;
bool     gOn, gBake;  // gBake: the transmittance and multiple scattering need (re)building
float    gBakedWith[12]; // the atmosphere they were built with
uint32_t gSlot;

float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

Texture* makeTexture(uint32_t w, uint32_t h, uint32_t depth, ResourceState start, const char* name)
{
    TextureDesc d = {};
    d.mWidth = w;
    d.mHeight = h;
    d.mDepth = depth;
    d.mArraySize = d.mMipLevels = 1;
    d.mSampleCount = SAMPLE_COUNT_1;
    d.mFormat = TinyImageFormat_R16G16B16A16_SFLOAT;
    d.mStartState = start;
    d.mDescriptors = (DescriptorType)(DESCRIPTOR_TYPE_TEXTURE | DESCRIPTOR_TYPE_RW_TEXTURE);
    d.pName = name;
    Texture*        t = nullptr;
    TextureLoadDesc l = {};
    l.pDesc = &d;
    l.ppTexture = &t;
    addResource(&l, nullptr);
    return t;
}

Buffer* makeBuffer(uint64_t size, uint32_t stride, uint32_t descriptors, ResourceMemoryUsage usage, ResourceState start, const char* name)
{
    Buffer*        b = nullptr;
    BufferLoadDesc d = {};
    d.mDesc.mDescriptors = (DescriptorType)descriptors;
    d.mDesc.mMemoryUsage = usage;
    d.mDesc.mSize = size;
    d.mDesc.mStartState = start;
    d.mDesc.pName = name;
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

void barrierTex(Cmd* cmd, Texture* t, ResourceState from, ResourceState to)
{
    TextureBarrier b = {};
    b.pTexture = t;
    b.mCurrentState = from;
    b.mNewState = to;
    cmdResourceBarrier(cmd, 0, nullptr, 1, &b, 0, nullptr);
}

void barrierBuf(Cmd* cmd, Buffer* buf, ResourceState from, ResourceState to)
{
    BufferBarrier b = { buf, from, to };
    cmdResourceBarrier(cmd, 1, &b, 0, nullptr, 0, nullptr);
}

void barrierRt(Cmd* cmd, RenderTarget* rt, ResourceState from, ResourceState to)
{
    RenderTargetBarrier b = { rt, from, to };
    cmdResourceBarrier(cmd, 0, nullptr, 0, nullptr, 1, &b);
}

// One compute pass: its constants in this frame's next slot, what it reads and writes bound there; `out` (2D or 3D)
// or `sh` is the one it writes.
void dispatch(const RenderFrame* F, Pass pass, uint32_t w, uint32_t h, Texture* out, bool volume, Buffer* sh)
{
    if (gSlot >= kSlots)
        return;
    uint32_t index = F->frameIndex * kSlots + gSlot++;
    SkyData  d = gData;
    d.size[0] = w;
    d.size[1] = h;
    memcpy(gCb[index]->pCpuMappedAddress, &d, sizeof d);
    // Each reads only what earlier passes made (never what it writes).
    Texture*       trans = pass == kTrans ? gDummy : gTransTex;
    Texture*       multi = pass == kTrans || pass == kMultiPass ? gDummy : gMultiTex;
    Texture*       env = pass == kSh ? gEnv->pTexture : gDummy;
    Texture*       out2 = out && !volume ? out : gDummyRw;
    Texture*       out3 = out && volume ? out : gDummyRw3;
    Buffer*        shOut = sh ? sh : gDummyBuf;
    DescriptorData p[8] = {};
    p[0].mIndex = SRT_RES_IDX(SkySrt, PerFrame, gSky);
    p[0].ppBuffers = &gCb[index];
    p[1].mIndex = SRT_RES_IDX(SkySrt, PerFrame, gTransmittance);
    p[1].ppTextures = &trans;
    p[2].mIndex = SRT_RES_IDX(SkySrt, PerFrame, gMultiScatter);
    p[2].ppTextures = &multi;
    p[3].mIndex = SRT_RES_IDX(SkySrt, PerFrame, gSkyView);
    p[3].ppTextures = &gDummy;
    p[4].mIndex = SRT_RES_IDX(SkySrt, PerFrame, gEnvSource);
    p[4].ppTextures = &env;
    p[5].mIndex = SRT_RES_IDX(SkySrt, PerFrame, gOut);
    p[5].ppTextures = &out2;
    p[6].mIndex = SRT_RES_IDX(SkySrt, PerFrame, gOutVolume);
    p[6].ppTextures = &out3;
    p[7].mIndex = SRT_RES_IDX(SkySrt, PerFrame, gShOut);
    p[7].ppBuffers = &shOut;
    updateDescriptorSet(gRenderer, index, gSet, 8, p);
    Cmd* cmd = F->cmd;
    if (out)
        barrierTex(cmd, out, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_UNORDERED_ACCESS);
    if (sh)
        barrierBuf(cmd, sh, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_UNORDERED_ACCESS);
    cmdBindPipeline(cmd, gPipe[pass]);
    cmdBindDescriptorSet(cmd, index, gSet);
    cmdDispatch(cmd, (w + 7) / 8, (h + 7) / 8, 1);
    if (out)
        barrierTex(cmd, out, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_SHADER_RESOURCE);
    if (sh)
        barrierBuf(cmd, sh, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_SHADER_RESOURCE);
}

// --- The sun's light through the atmosphere, on the CPU (Shaders/Atmosphere.h.fsl's model and the transmittance
// pass's integral, at 64 steps): what reaches a height r km from the centre along cosine mu with the vertical. ---
void transmittance(float r, float mu, float out[3])
{
    out[0] = out[1] = out[2] = 0;
    if (mu < 0 && r * r * (mu * mu - 1) + gData.radii[0] * gData.radii[0] >= 0)
        return; // the planet is in the way
    float t = -r * mu + sqrtf(fmaxf(r * r * (mu * mu - 1) + gData.radii[1] * gData.radii[1], 0));
    float od[3] = { 0, 0, 0 }, dt = fmaxf(t, 0) / 64;
    for (int i = 0; i < 64; ++i)
    {
        float s = (i + 0.5f) * dt;
        float h = sqrtf(r * r + s * s + 2 * r * mu * s) - gData.radii[0];
        float ra = expf(-h / gData.rayleigh[3]), mi = expf(-h / gData.mie[3]);
        float oz = fmaxf(0, 1 - fabsf(h - 25) / 15);
        for (int k = 0; k < 3; ++k)
            od[k] += (gData.rayleigh[k] * ra + (gData.mie[0] + gData.mie[1]) * mi + gData.ozone[k] * oz) * dt;
    }
    for (int k = 0; k < 3; ++k)
        out[k] = expf(-od[k]);
}

void freeAll()
{
    Texture** tex[] = { &gTransTex, &gMultiTex, &gViewTex, &gAerialTex, &gDummy, &gDummyRw, &gDummyRw3 };
    for (Texture** t : tex)
    {
        if (*t)
            removeResource(*t);
        *t = nullptr;
    }
    Buffer** bufs[] = { &gSh, &gDummyBuf };
    for (Buffer** b : bufs)
    {
        if (*b)
            removeResource(*b);
        *b = nullptr;
    }
    for (uint32_t i = 0; i < kMaxFrames * kSlots; ++i)
        if (gCb[i])
            removeResource(gCb[i]), gCb[i] = nullptr;
    for (uint32_t i = 0; i < kMaxFrames * kEnvMips; ++i)
        if (gEnvCb[i])
            removeResource(gEnvCb[i]), gEnvCb[i] = nullptr;
    if (gEnv)
        removeRenderTarget(gRenderer, gEnv);
    gEnv = nullptr;
}
} // namespace

int load(int reloaded)
{
    const YeaHost* host = yea::host();
    gRenderer = (Renderer*)host->renderer;
    gQueue = (Queue*)host->queue;
    gFrames = host->framesInFlight < kMaxFrames ? host->framesInFlight : kMaxFrames;
    gReady = gOn = false;
    gBake = true;
    for (uint32_t i = 0; i < kPasses; ++i)
    {
        ShaderLoadDesc sd = {};
        sd.mComp.pFileName = kShaderNames[i];
        addShader(gRenderer, &sd, &gShader[i]);
        if (!gShader[i])
            return 1;
        PipelineDesc desc = {};
        desc.mType = PIPELINE_TYPE_COMPUTE;
        PIPELINE_LAYOUT_DESC(desc, NULL, SRT_LAYOUT_DESC(SkySrt, PerFrame), NULL, NULL);
        desc.mComputeDesc.pShaderProgram = gShader[i];
        desc.pName = kShaderNames[i];
        addPipeline(gRenderer, &desc, &gPipe[i]);
        if (!gPipe[i])
            return 1;
    }
    {
        ShaderLoadDesc sd = {};
        sd.mVert.pFileName = "renderer_sky_fullscreen.vert";
        sd.mFrag.pFileName = "renderer_sky_env.frag";
        addShader(gRenderer, &sd, &gEnvShader);
        if (!gEnvShader)
            return 1;
        RasterizerStateDesc raster = {};
        raster.mCullMode = CULL_MODE_NONE;
        DepthStateDesc  depth = {};
        TinyImageFormat format = TinyImageFormat_R16G16B16A16_SFLOAT;
        PipelineDesc    desc = {};
        desc.mType = PIPELINE_TYPE_GRAPHICS;
        PIPELINE_LAYOUT_DESC(desc, NULL, SRT_LAYOUT_DESC(EnvSrt, PerFrame), NULL, NULL);
        GraphicsPipelineDesc& g = desc.mGraphicsDesc;
        g.mPrimitiveTopo = PRIMITIVE_TOPO_TRI_LIST;
        g.mRenderTargetCount = 1;
        g.pColorFormats = &format;
        g.mSampleCount = SAMPLE_COUNT_1;
        g.mDepthStencilFormat = TinyImageFormat_UNDEFINED;
        g.pShaderProgram = gEnvShader;
        g.pRasterizerState = &raster;
        g.pDepthState = &depth;
        desc.pName = "renderer_sky_env";
        addPipeline(gRenderer, &desc, &gEnvPipe);
        if (!gEnvPipe)
            return 1;
    }
    DescriptorSetDesc setDesc = SRT_SET_DESC(SkySrt, PerFrame, gFrames * kSlots, 0);
    addDescriptorSet(gRenderer, &setDesc, &gSet);
    DescriptorSetDesc envDesc = SRT_SET_DESC(EnvSrt, PerFrame, gFrames * kEnvMips, 0);
    addDescriptorSet(gRenderer, &envDesc, &gEnvSet);
    const ResourceState sr = RESOURCE_STATE_SHADER_RESOURCE, uav = RESOURCE_STATE_UNORDERED_ACCESS;
    for (uint32_t i = 0; i < gFrames * kSlots; ++i)
        gCb[i] = makeBuffer(256, 0, DESCRIPTOR_TYPE_UNIFORM_BUFFER, RESOURCE_MEMORY_USAGE_CPU_TO_GPU,
                            RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER, "renderer_sky_cb");
    for (uint32_t i = 0; i < gFrames * kEnvMips; ++i)
        gEnvCb[i] = makeBuffer(256, 0, DESCRIPTOR_TYPE_UNIFORM_BUFFER, RESOURCE_MEMORY_USAGE_CPU_TO_GPU,
                               RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER, "renderer_sky_env_cb");
    gTransTex = makeTexture(kTransW, kTransH, 1, sr, "renderer_sky_transmittance");
    gMultiTex = makeTexture(kMulti, kMulti, 1, sr, "renderer_sky_multiscatter");
    gViewTex = makeTexture(kViewW, kViewH, 1, sr, "renderer_sky_view");
    gAerialTex = makeTexture(kAerial, kAerial, kAerial, sr, "renderer_sky_aerial");
    gDummy = makeTexture(2, 2, 1, sr, "renderer_sky_dummy"); // 2 x 2: a 1 x 1 texture gets a 1D view
    gDummyRw = makeTexture(2, 2, 1, uav, "renderer_sky_dummy_rw");
    gDummyRw3 = makeTexture(2, 2, 2, uav, "renderer_sky_dummy_rw3");
    gSh = makeBuffer(9 * 16, 16, DESCRIPTOR_TYPE_BUFFER | DESCRIPTOR_TYPE_RW_BUFFER, RESOURCE_MEMORY_USAGE_GPU_ONLY, sr, "renderer_sky_sh");
    gDummyBuf = makeBuffer(16, 16, DESCRIPTOR_TYPE_BUFFER | DESCRIPTOR_TYPE_RW_BUFFER, RESOURCE_MEMORY_USAGE_GPU_ONLY, uav, "renderer_sky_dummy_buf");
    RenderTargetDesc rd = {};
    rd.mWidth = rd.mHeight = kEnvSize;
    rd.mDepth = rd.mArraySize = 1;
    rd.mMipLevels = kEnvMips;
    rd.mSampleCount = SAMPLE_COUNT_1;
    rd.mFormat = TinyImageFormat_R16G16B16A16_SFLOAT;
    rd.mStartState = sr;
    rd.mClearValue = { { 0, 0, 0, 0 } };
    rd.mDescriptors = (DescriptorType)(DESCRIPTOR_TYPE_TEXTURE | DESCRIPTOR_TYPE_RENDER_TARGET_MIP_SLICES);
    rd.pName = "renderer_sky_env";
    addRenderTarget(gRenderer, &rd, &gEnv);
    waitForAllResourceLoads();
    bool ok = gSet && gEnvSet && gTransTex && gMultiTex && gViewTex && gAerialTex && gDummy && gDummyRw && gDummyRw3 && gSh &&
              gDummyBuf && gEnv;
    for (uint32_t i = 0; i < gFrames * kSlots; ++i)
        ok = ok && gCb[i] && gCb[i]->pCpuMappedAddress;
    for (uint32_t i = 0; i < gFrames * kEnvMips; ++i)
        ok = ok && gEnvCb[i] && gEnvCb[i]->pCpuMappedAddress;
    if (!ok)
        return 1;
    // The reflection image's mip passes read only the sky-view LUT: their sets never change.
    for (uint32_t i = 0; i < gFrames * kEnvMips; ++i)
    {
        DescriptorData p[2] = {};
        p[0].mIndex = SRT_RES_IDX(EnvSrt, PerFrame, gSky);
        p[0].ppBuffers = &gEnvCb[i];
        p[1].mIndex = SRT_RES_IDX(EnvSrt, PerFrame, gSkyView);
        p[1].ppTextures = &gViewTex;
        updateDescriptorSet(gRenderer, i, gEnvSet, 2, p);
    }
    gReady = true;
    return 0;
}

void unload()
{
    if (!gRenderer)
        return;
    waitQueueIdle(gQueue);
    freeAll();
    if (gSet)
        removeDescriptorSet(gRenderer, gSet);
    if (gEnvSet)
        removeDescriptorSet(gRenderer, gEnvSet);
    gSet = gEnvSet = nullptr;
    for (uint32_t i = 0; i < kPasses; ++i)
    {
        if (gPipe[i])
            removePipeline(gRenderer, gPipe[i]);
        if (gShader[i])
            removeShader(gRenderer, gShader[i]);
        gPipe[i] = nullptr;
        gShader[i] = nullptr;
    }
    if (gEnvPipe)
        removePipeline(gRenderer, gEnvPipe);
    if (gEnvShader)
        removeShader(gRenderer, gEnvShader);
    gEnvPipe = nullptr;
    gEnvShader = nullptr;
    gReady = gOn = false;
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
        snprintf(line, sizeof line, "renderer_sky: the core's frame is v%u, this module's v%u: standing aside", core, RENDER_FRAME_VERSION);
        yea::log(line);
        logged = core + 1;
    }
    return false;
}

bool sky_prepare(uint64_t address)
{
    if (!coreMatches())
        return false;
    RenderFrame* F = (RenderFrame*)(uintptr_t)address;
    gOn = false;
    if (!gReady || !F || F->version != RENDER_FRAME_VERSION || !settings.physical_sky || !F->sunOn)
        return false;
    SkyData& d = gData;
    memset(&d, 0, sizeof d);
    // Earth's (Hillaire 2020, table 1), in km; the knobs scale each.
    const float air = fmaxf(settings.air_density, 0), haze = fmaxf(settings.haze, 0), oz = fmaxf(settings.ozone, 0);
    // The camera's height: the scene's metres above the ground at y = 0, at least a metre up.
    float height = fmaxf(F->camPos[1], 1.0f) * 0.001f;
    d.radii[0] = kGround;
    d.radii[1] = kTop;
    d.radii[2] = kGround + fminf(height, kTop - kGround - 0.01f);
    d.radii[3] = fmaxf(settings.aerial_perspective, 0);
    d.rayleigh[0] = 5.802e-3f * air, d.rayleigh[1] = 13.558e-3f * air, d.rayleigh[2] = 33.1e-3f * air, d.rayleigh[3] = 8.0f;
    d.mie[0] = 3.996e-3f * haze, d.mie[1] = 4.40e-3f * haze, d.mie[2] = 0.8f, d.mie[3] = 1.2f;
    d.ozone[0] = 0.650e-3f * oz, d.ozone[1] = 1.881e-3f * oz, d.ozone[2] = 0.085e-3f * oz, d.ozone[3] = clampf(settings.ground_albedo, 0, 1);
    float s[3] = { F->sunDir[0], F->sunDir[1], F->sunDir[2] };
    float sl = sqrtf(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
    sl = sl > 0 ? 1.0f / sl : 0.0f;
    d.sunDir[0] = s[0] * sl, d.sunDir[1] = s[1] * sl, d.sunDir[2] = s[2] * sl, d.sunDir[3] = 1.0f;
    // The sun above the atmosphere: the light's colour times its intensity.
    memcpy(d.sunColor, F->sunColor, sizeof F->sunColor);
    d.camPos[0] = F->camPos[0], d.camPos[1] = F->camPos[1], d.camPos[2] = F->camPos[2];
    d.camPos[3] = kAerialDepth;
    // The view's axes (rows of world -> view), scaled by the projection, for the froxels' directions.
    const float ip00 = F->p00 > 0 ? 1.0f / F->p00 : 1.0f, ip11 = F->p11 > 0 ? 1.0f / F->p11 : 1.0f;
    for (int k = 0; k < 3; ++k)
    {
        d.camRight[k] = F->view[0][k] * ip00;
        d.camUp[k] = F->view[1][k] * ip11;
        d.camForward[k] = F->view[2][k];
    }
    // The tables depend only on the atmosphere: rebuilt when it changes.
    float with[12] = { d.rayleigh[0], d.rayleigh[1], d.rayleigh[2], d.mie[0], d.mie[1], d.mie[2], d.ozone[0], d.ozone[1], d.ozone[2], d.ozone[3], 0, 0 };
    gBake = gBake || memcmp(with, gBakedWith, sizeof with) != 0;
    memcpy(gBakedWith, with, sizeof with);
    // What of the sun reaches the camera.
    float t[3];
    transmittance(d.radii[2], d.sunDir[1], t);
    memcpy(F->sunTransmittance, t, sizeof t);
    F->skyView = gViewTex;
    F->skyEnv = gEnv->pTexture;
    F->skyEnvMips = kEnvMips;
    F->skySh = gSh;
    F->skyAerial = gAerialTex;
    F->skyAerialDepth = kAerialDepth;
    memcpy(F->skyRadii, d.radii, sizeof d.radii);
    gOn = true;
    return true;
}

void sky_run(uint64_t address)
{
    if (!coreMatches())
        return;
    RenderFrame* F = (RenderFrame*)(uintptr_t)address;
    if (!gOn || !gReady || !F)
        return;
    gOn = false;
    gSlot = 0;
    Cmd* cmd = F->cmd;
    cmdBeginDebugMarker(cmd, 0.4f, 0.7f, 1.0f, "Renderer sky");
    if (gBake)
    {
        dispatch(F, kTrans, kTransW, kTransH, gTransTex, false, nullptr);
        dispatch(F, kMultiPass, kMulti, kMulti, gMultiTex, false, nullptr);
        gBake = false;
    }
    dispatch(F, kView, kViewW, kViewH, gViewTex, false, nullptr);
    dispatch(F, kAerialPass, kAerial, kAerial, gAerialTex, true, nullptr);
    // The reflection image: mip 0 the sky-view LUT as is, each smaller one the GGX lobe of its roughness.
    barrierRt(cmd, gEnv, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_RENDER_TARGET);
    for (uint32_t m = 0; m < kEnvMips; ++m)
    {
        uint32_t index = F->frameIndex * kEnvMips + m;
        SkyData  d = gData;
        d.misc[0] = (float)m / (float)(kEnvMips - 1);
        memcpy(gEnvCb[index]->pCpuMappedAddress, &d, sizeof d);
        BindRenderTargetsDesc bind = {};
        bind.mRenderTargetCount = 1;
        bind.mRenderTargets[0] = { gEnv, LOAD_ACTION_DONTCARE };
        bind.mRenderTargets[0].mMipSlice = m;
        bind.mRenderTargets[0].mUseMipSlice = m ? 1 : 0;
        cmdBindRenderTargets(cmd, &bind);
        uint32_t size = kEnvSize >> m;
        cmdSetViewport(cmd, 0, 0, (float)size, (float)size, 0, 1);
        cmdSetScissor(cmd, 0, 0, size, size);
        cmdBindPipeline(cmd, gEnvPipe);
        cmdBindDescriptorSet(cmd, index, gEnvSet);
        cmdDraw(cmd, 3, 0);
        cmdBindRenderTargets(cmd, nullptr);
    }
    barrierRt(cmd, gEnv, RESOURCE_STATE_RENDER_TARGET, RESOURCE_STATE_SHADER_RESOURCE);
    dispatch(F, kSh, 8, 8, nullptr, false, gSh);
    cmdEndDebugMarker(cmd);
}
} // namespace renderer_sky
