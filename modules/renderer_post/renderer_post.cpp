// Post-processing for the renderer core, on the anti-aliased (or upscaled) scene-linear HDR image at display size,
// before the core's tonemapper: depth of field (a thin lens, half-size two-layer gather, Shaders/Dof.comp.fsl), motion blur (McGuire 2012 with Jimenez 2014's refinements, Shaders/Blur.comp.fsl), bloom (a 13-tap downsample chain and tent upsamples, Jimenez 2014), a vignette, and
// colour grading through a 32^3 LUT in ACEScct (white balance, contrast, lift / gamma / gain, saturation). With every
// effect neutral, post_prepare returns 0 and nothing runs.
#include "yea_renderer_post.h"
#include "frame.h"

#include <Common_3/Graphics/Interfaces/IGraphics.h>
#include <Common_3/Resources/ResourceLoader/Interfaces/IResourceLoader.h>
#include <Common_3/Graphics/FSL/defaults.h>
#include "Shaders/Post.srt.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

namespace renderer_post
{
namespace
{
struct PostData
{
    uint32_t size[4];
    float    bloom[4];
    float    look[4];
    float    blur[4];     // motion blur: xy motion (uv) to blur radius (display pixels), z the tile size (the largest radius), w near
    float    blurSize[4]; // xy display size, zw render size (motion and depth)
    float    dof[4];      // depth of field: x focal length (m), y its blur scale, z largest radius (px), w set focus (0: auto)
    float    dof2[4];     // x autofocus easing, y 1: no focus history, z half-size tile (px)
    float    wb[3][4];
    float    lift[4];
    float    gamma[4];
    float    gain[4];
};

const uint32_t kMaxFrames = 4, kSlots = 32, kLevels = 6;
enum Pass
{
    kDownFirst, kDown, kUp, kLut, kCombine, kBlurTile, kBlurNeighbor, kBlurGather,
    kDofFocus, kDofCoc, kDofTile, kDofNeighbor, kDofGather, kDofCombine, kPasses
};
const char* const kShaderNames[kPasses] = { "renderer_post_down_first.comp", "renderer_post_down.comp", "renderer_post_up.comp",
                                            "renderer_post_lut.comp",        "renderer_post_combine.comp", "renderer_post_blur_tile.comp",
                                            "renderer_post_blur_neighbor.comp", "renderer_post_blur_gather.comp",
                                            "renderer_post_dof_focus.comp",    "renderer_post_dof_coc.comp",
                                            "renderer_post_dof_tile.comp",     "renderer_post_dof_neighbor.comp",
                                            "renderer_post_dof_gather.comp",   "renderer_post_dof_combine.comp" };

Renderer*      gRenderer;
Queue*         gQueue;
uint32_t       gFrames;
Shader*        gShader[kPasses];
Pipeline*      gPipe[kPasses];
DescriptorSet* gSet;
Buffer*        gCb[kMaxFrames * kSlots];
Texture*       gDummy; // 1 x 1, for the slots a pass doesn't read
Texture*       gLut;   // 1024 x 32: the grading LUT
Texture*       gOut;   // display size: what the tonemapper reads
Texture*       gDown[kLevels];
Texture*       gUp[kLevels - 1];
Texture*       gBlur;               // display size: the motion-blurred scene, made only while motion blur is on
Texture*       gTiles, *gNeighbors; // per tile: the longest velocity and the shortest length; and over its neighbours
uint32_t       gTile, gTilesX, gTilesY;
// Depth of field's targets, made only while it is on: half size (colour and blur; the background and foreground
// layers), its tiles, the focus distance (1 x 1, this frame's and last's) and the full-size result.
Texture*       gDofHalf, *gDofFar, *gDofNear, *gDofTiles, *gDofNeighbors, *gFocus[2], *gDofOut;
uint32_t       gDofTile, gDofTilesX, gDofTilesY, gFocusIndex;
bool           gFocusFresh; // no focus history yet: autofocus starts at its target
Texture*       gFocusRead;  // what gFocus binds
float          gDt = 1.0f / 60.0f;
uint32_t       gWidth, gHeight, gLevels;
bool           gReady;

// This frame's choices, from post_prepare to post_run.
struct Plan
{
    bool     on, bloom, grade, bake, blur, dof;
    Texture* in;
    PostData data;
} gPlan;
PostData gLutData;  // what the LUT was baked with
bool     gLutValid; // the LUT holds gLutData's grade

float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

Texture* makeTexture(uint32_t w, uint32_t h, const char* name)
{
    TextureDesc d = {};
    d.mWidth = w;
    d.mHeight = h;
    d.mDepth = d.mArraySize = d.mMipLevels = 1;
    d.mSampleCount = SAMPLE_COUNT_1;
    d.mFormat = TinyImageFormat_R16G16B16A16_SFLOAT;
    d.mStartState = RESOURCE_STATE_SHADER_RESOURCE;
    d.mDescriptors = (DescriptorType)(DESCRIPTOR_TYPE_TEXTURE | DESCRIPTOR_TYPE_RW_TEXTURE);
    d.pName = name;
    Texture*        t = nullptr;
    TextureLoadDesc l = {};
    l.pDesc = &d;
    l.ppTexture = &t;
    addResource(&l, nullptr);
    return t;
}

void freeTexture(Texture*& t)
{
    if (t)
        removeResource(t);
    t = nullptr;
}

void freeBlur()
{
    freeTexture(gBlur);
    freeTexture(gTiles);
    freeTexture(gNeighbors);
    gTile = gTilesX = gTilesY = 0;
}

void freeDof()
{
    Texture** all[] = { &gDofHalf, &gDofFar, &gDofNear, &gDofTiles, &gDofNeighbors, &gFocus[0], &gFocus[1], &gDofOut };
    for (Texture** t : all)
        freeTexture(*t);
    gDofTile = gDofTilesX = gDofTilesY = 0;
    gFocusRead = nullptr;
}

void freeSized()
{
    freeDof();
    freeBlur();
    freeTexture(gOut);
    for (uint32_t i = 0; i < kLevels; ++i)
        freeTexture(gDown[i]);
    for (uint32_t i = 0; i + 1 < kLevels; ++i)
        freeTexture(gUp[i]);
    gWidth = gHeight = gLevels = 0;
}

uint32_t levelSize(uint32_t s, uint32_t level) { return (s >> (level + 1)) > 1 ? s >> (level + 1) : 1; }

bool makeSized(uint32_t w, uint32_t h)
{
    gOut = makeTexture(w, h, "renderer_post_out");
    gLevels = 0;
    for (uint32_t i = 0; i < kLevels && levelSize(w, i) >= 2 && levelSize(h, i) >= 2; ++i, ++gLevels)
    {
        gDown[i] = makeTexture(levelSize(w, i), levelSize(h, i), "renderer_post_bloom_down");
        if (i > 0)
            gUp[i - 1] = makeTexture(levelSize(w, i - 1), levelSize(h, i - 1), "renderer_post_bloom_up");
    }
    waitForAllResourceLoads();
    gWidth = w;
    gHeight = h;
    bool ok = gOut && gLevels >= 2;
    for (uint32_t i = 0; i < gLevels; ++i)
        ok = ok && gDown[i] && (i == 0 || gUp[i - 1]);
    return ok;
}

// Motion blur's targets for tiles of `tile` pixels, at the current size.
bool makeBlur(uint32_t tile)
{
    gTilesX = (gWidth + tile - 1) / tile;
    gTilesY = (gHeight + tile - 1) / tile;
    gBlur = makeTexture(gWidth, gHeight, "renderer_post_blur");
    gTiles = makeTexture(gTilesX, gTilesY, "renderer_post_blur_tiles");
    gNeighbors = makeTexture(gTilesX, gTilesY, "renderer_post_blur_neighbors");
    waitForAllResourceLoads();
    gTile = tile;
    return gBlur && gTiles && gNeighbors;
}

// Depth of field's targets for half-size tiles of `tile` pixels, at the current size.
bool makeDof(uint32_t tile)
{
    uint32_t hw = (gWidth + 1) / 2, hh = (gHeight + 1) / 2;
    gDofTilesX = (hw + tile - 1) / tile;
    gDofTilesY = (hh + tile - 1) / tile;
    gDofHalf = makeTexture(hw, hh, "renderer_post_dof_half");
    gDofFar = makeTexture(hw, hh, "renderer_post_dof_far");
    gDofNear = makeTexture(hw, hh, "renderer_post_dof_near");
    gDofTiles = makeTexture(gDofTilesX, gDofTilesY, "renderer_post_dof_tiles");
    gDofNeighbors = makeTexture(gDofTilesX, gDofTilesY, "renderer_post_dof_neighbors");
    gFocus[0] = makeTexture(2, 2, "renderer_post_focus"); // 2 x 2: a 1 x 1 texture gets a 1D view
    gFocus[1] = makeTexture(2, 2, "renderer_post_focus");
    gDofOut = makeTexture(gWidth, gHeight, "renderer_post_dof");
    waitForAllResourceLoads();
    gDofTile = tile;
    gFocusFresh = true;
    return gDofHalf && gDofFar && gDofNear && gDofTiles && gDofNeighbors && gFocus[0] && gFocus[1] && gDofOut;
}

void barrier(Cmd* cmd, Texture* t, ResourceState from, ResourceState to)
{
    TextureBarrier b = {};
    b.pTexture = t;
    b.mCurrentState = from;
    b.mNewState = to;
    cmdResourceBarrier(cmd, 0, nullptr, 1, &b, 0, nullptr);
}

// --- White balance: a von Kries scaling in Bradford's cone space, from the white the knobs pick (on the CIE daylight
// locus, temperature in even steps of mired, tint across it) to the neutral one, carried into linear Rec.709. ---
typedef float M3[3][3];
void mul(const M3 a, const M3 b, M3 out)
{
    M3 r;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
    memcpy(out, r, sizeof r);
}
void inverse(const M3 m, M3 out)
{
    float c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1], c01 = m[1][2] * m[2][0] - m[1][0] * m[2][2], c02 = m[1][0] * m[2][1] - m[1][1] * m[2][0];
    float det = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02, k = 1.0f / det;
    out[0][0] = c00 * k;
    out[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * k;
    out[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * k;
    out[1][0] = c01 * k;
    out[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * k;
    out[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * k;
    out[2][0] = c02 * k;
    out[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * k;
    out[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * k;
}

// The white's XYZ (Y = 1) at temperature kelvin, its CIE 1960 v moved by dv.
void whiteXyz(float kelvin, float dv, float xyz[3])
{
    double t = kelvin, x = t <= 7000 ? -4.6070e9 / (t * t * t) + 2.9678e6 / (t * t) + 0.09911e3 / t + 0.244063
                                     : -2.0064e9 / (t * t * t) + 1.9018e6 / (t * t) + 0.24748e3 / t + 0.237040;
    double y = -3.0 * x * x + 2.87 * x - 0.275;
    double d = -2.0 * x + 12.0 * y + 3.0, u = 4.0 * x / d, v = 6.0 * y / d + dv;
    d = 2.0 * u - 8.0 * v + 4.0;
    x = 3.0 * u / d;
    y = 2.0 * v / d;
    xyz[0] = (float)(x / y);
    xyz[1] = 1.0f;
    xyz[2] = (float)((1.0 - x - y) / y);
}

void whiteBalance(float temperature, float tint, float out[3][4])
{
    const M3 bradford = { { 0.8951f, 0.2664f, -0.1614f }, { -0.7502f, 1.7135f, 0.0367f }, { 0.0389f, -0.0685f, 1.0296f } };
    const M3 toXyz = { { 0.4124f, 0.3576f, 0.1805f }, { 0.2126f, 0.7152f, 0.0722f }, { 0.0193f, 0.1192f, 0.9505f } };
    // Warmer: the image is balanced for a bluer (hotter) white than the scene's, so it reads warm; 60 mired a unit.
    float kelvin = 1e6f / (1e6f / 6500.0f - temperature * 60.0f);
    float src[3], dst[3], ls[3], ld[3];
    whiteXyz(kelvin, tint * 0.02f, src);
    whiteXyz(6500.0f, 0.0f, dst);
    for (int i = 0; i < 3; ++i)
    {
        ls[i] = bradford[i][0] * src[0] + bradford[i][1] * src[1] + bradford[i][2] * src[2];
        ld[i] = bradford[i][0] * dst[0] + bradford[i][1] * dst[1] + bradford[i][2] * dst[2];
    }
    M3 scale = { { ld[0] / ls[0], 0, 0 }, { 0, ld[1] / ls[1], 0 }, { 0, 0, ld[2] / ls[2] } }, inv, fromXyz, m;
    inverse(bradford, inv);
    inverse(toXyz, fromXyz);
    mul(scale, bradford, m);
    mul(inv, m, m);
    mul(m, toXyz, m);
    mul(fromXyz, m, m);
    for (int i = 0; i < 3; ++i)
    {
        out[i][0] = m[i][0];
        out[i][1] = m[i][1];
        out[i][2] = m[i][2];
        out[i][3] = 0;
    }
}

// A tint colour scaled to luma 1, so it changes the hue of a wheel and not its strength.
void tinted(yea_float3 c, float amount, float out[4])
{
    float l = 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
    l = l > 1e-4f ? 1.0f / l : 0.0f;
    out[0] = amount * c.x * l;
    out[1] = amount * c.y * l;
    out[2] = amount * c.z * l;
}

// The grade's constants, and whether they change anything.
bool gradeData(PostData& d)
{
    whiteBalance(clampf(settings.temperature, -1, 1), clampf(settings.tint, -1, 1), d.wb);
    tinted(settings.lift_color, clampf(settings.lift, -0.5f, 0.5f), d.lift);
    tinted(settings.gamma_color, clampf(settings.gamma, 0.25f, 4), d.gamma);
    tinted(settings.gain_color, clampf(settings.gain, 0, 2), d.gain);
    for (int i = 0; i < 3; ++i)
        d.gamma[i] = d.gamma[i] < 0.05f ? 0.05f : d.gamma[i];
    d.lift[3] = clampf(settings.contrast, 0, 2);
    d.gamma[3] = clampf(settings.saturation, 0, 2);
    bool neutral = fabsf(settings.temperature) < 1e-4f && fabsf(settings.tint) < 1e-4f && fabsf(d.lift[3] - 1) < 1e-4f && fabsf(d.gamma[3] - 1) < 1e-4f;
    for (int i = 0; i < 3; ++i)
        neutral = neutral && fabsf(d.lift[i]) < 1e-4f && fabsf(d.gamma[i] - 1) < 1e-4f && fabsf(d.gain[i] - 1) < 1e-4f;
    return !neutral;
}

// One dispatch: its constants in this frame's next slot, its reads and its output bound there.
uint32_t gSlot;
void dispatch(RenderFrame* F, Pass pass, const PostData& d, Texture* src, Texture* src2, Texture* out, uint32_t w, uint32_t h,
              Texture* src3 = nullptr, Texture* out2 = nullptr)
{
    if (gSlot >= kSlots)
        return;
    uint32_t index = F->frameIndex * kSlots + gSlot++;
    memcpy(gCb[index]->pCpuMappedAddress, &d, sizeof d);
    Texture*       lut = gLut;
    Texture*       motion = F->motion ? F->motion->pTexture : gDummy;
    Buffer*        exposure = F->exposure;
    Texture*       focus = gFocusRead ? gFocusRead : gDummy;
    Texture*       second = out2 ? out2 : gDummy;
    DescriptorData p[11] = {};
    p[0].mIndex = SRT_RES_IDX(PostSrt, PerFrame, gPost);
    p[0].ppBuffers = &gCb[index];
    p[1].mIndex = SRT_RES_IDX(PostSrt, PerFrame, gSrc);
    p[1].ppTextures = src ? &src : &gDummy;
    p[2].mIndex = SRT_RES_IDX(PostSrt, PerFrame, gSrc2);
    p[2].ppTextures = src2 ? &src2 : &gDummy;
    p[3].mIndex = SRT_RES_IDX(PostSrt, PerFrame, gLut);
    p[3].ppTextures = pass == kLut ? &gDummy : &lut;
    p[4].mIndex = SRT_RES_IDX(PostSrt, PerFrame, gDepth);
    p[4].ppTextures = &F->depth->pTexture;
    p[5].mIndex = SRT_RES_IDX(PostSrt, PerFrame, gMotion);
    p[5].ppTextures = &motion;
    p[6].mIndex = SRT_RES_IDX(PostSrt, PerFrame, gExposure);
    p[6].ppBuffers = &exposure;
    p[7].mIndex = SRT_RES_IDX(PostSrt, PerFrame, gOut);
    p[7].ppTextures = &out;
    p[8].mIndex = SRT_RES_IDX(PostSrt, PerFrame, gSrc3);
    p[8].ppTextures = src3 ? &src3 : &gDummy;
    p[9].mIndex = SRT_RES_IDX(PostSrt, PerFrame, gFocus);
    p[9].ppTextures = &focus;
    p[10].mIndex = SRT_RES_IDX(PostSrt, PerFrame, gOut2);
    p[10].ppTextures = &second;
    updateDescriptorSet(gRenderer, index, gSet, 11, p);
    Cmd* cmd = F->cmd;
    barrier(cmd, out, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_UNORDERED_ACCESS);
    if (out2)
        barrier(cmd, out2, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_UNORDERED_ACCESS);
    cmdBindPipeline(cmd, gPipe[pass]);
    cmdBindDescriptorSet(cmd, index, gSet);
    cmdDispatch(cmd, (w + 7) / 8, (h + 7) / 8, 1);
    barrier(cmd, out, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_SHADER_RESOURCE);
    if (out2)
        barrier(cmd, out2, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_SHADER_RESOURCE);
}
} // namespace

int load(int reloaded)
{
    const YeaHost* host = yea::host();
    gRenderer = (Renderer*)host->renderer;
    gQueue = (Queue*)host->queue;
    gFrames = host->framesInFlight < kMaxFrames ? host->framesInFlight : kMaxFrames;
    gReady = false;
    gLutValid = false;
    for (uint32_t i = 0; i < kPasses; ++i)
    {
        ShaderLoadDesc sd = {};
        sd.mComp.pFileName = kShaderNames[i];
        addShader(gRenderer, &sd, &gShader[i]);
        if (!gShader[i])
            return 1;
        PipelineDesc desc = {};
        desc.mType = PIPELINE_TYPE_COMPUTE;
        PIPELINE_LAYOUT_DESC(desc, NULL, SRT_LAYOUT_DESC(PostSrt, PerFrame), NULL, NULL);
        desc.mComputeDesc.pShaderProgram = gShader[i];
        desc.pName = kShaderNames[i];
        addPipeline(gRenderer, &desc, &gPipe[i]);
        if (!gPipe[i])
            return 1;
    }
    DescriptorSetDesc setDesc = SRT_SET_DESC(PostSrt, PerFrame, gFrames * kSlots, 0);
    addDescriptorSet(gRenderer, &setDesc, &gSet);
    for (uint32_t i = 0; i < gFrames * kSlots; ++i)
    {
        BufferLoadDesc d = {};
        d.mDesc.mDescriptors = DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        d.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
        d.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
        d.mDesc.mSize = 256;
        d.mDesc.mStartState = RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
        d.mDesc.pName = "renderer_post_cb";
        d.ppBuffer = &gCb[i];
        addResource(&d, nullptr);
    }
    gDummy = makeTexture(1, 1, "renderer_post_dummy");
    gLut = makeTexture(1024, 32, "renderer_post_lut");
    waitForAllResourceLoads();
    bool ok = gSet && gDummy && gLut;
    for (uint32_t i = 0; i < gFrames * kSlots; ++i)
        ok = ok && gCb[i] && gCb[i]->pCpuMappedAddress;
    gReady = ok;
    return ok ? 0 : 1;
}

void unload()
{
    if (!gRenderer)
        return;
    waitQueueIdle(gQueue);
    freeSized();
    freeTexture(gDummy);
    freeTexture(gLut);
    for (uint32_t i = 0; i < kMaxFrames * kSlots; ++i)
        if (gCb[i])
            removeResource(gCb[i]), gCb[i] = nullptr;
    if (gSet)
        removeDescriptorSet(gRenderer, gSet);
    gSet = nullptr;
    for (uint32_t i = 0; i < kPasses; ++i)
    {
        if (gPipe[i])
            removePipeline(gRenderer, gPipe[i]);
        if (gShader[i])
            removeShader(gRenderer, gShader[i]);
        gPipe[i] = nullptr;
        gShader[i] = nullptr;
    }
    gReady = false;
}

void update(uint32_t phase, float dt)
{
    if (dt > 0)
        gDt = dt; // autofocus eases by the frame's length
}
void draw(uint32_t pass, const YeaFrame* frame) {}

uint64_t post_prepare(uint64_t address)
{
    RenderFrame* F = (RenderFrame*)(uintptr_t)address;
    gPlan.on = false;
    if (!gReady || !F || F->version != RENDER_FRAME_VERSION || yea::viewMode() || !F->depth || !F->exposure)
        return 0;
    // The anti-aliased or upscaled image, else the scene itself when it is one sample at display size.
    uint32_t w = F->displayWidth, h = F->displayHeight;
    Texture* in = F->resolved ? F->resolved->pTexture
                              : (F->hdr && F->sampleCount == 1 && F->width == w && F->height == h ? F->hdr->pTexture : nullptr);
    if (!in || w < 8 || h < 8)
        return 0;
    PostData& d = gPlan.data;
    memset(&d, 0, sizeof d);
    float intensity = settings.bloom ? clampf(settings.bloom_intensity, 0, 1) : 0.0f;
    float vignette = settings.vignette ? clampf(settings.vignette_intensity, 0, 1) : 0.0f;
    bool  grade = settings.grading && gradeData(d);
    // Motion blur needs the motion vectors the core renders with TAA or an upscaler, at one sample.
    float shutter = settings.motion_blur ? clampf(settings.motion_blur_amount, 0, 1) : 0.0f;
    bool  blur = shutter > 0 && F->taaOn && F->motion && F->sampleCount == 1;
    bool  dof = settings.depth_of_field && F->p11 > 0;
    if (intensity <= 0 && vignette <= 0 && !grade && !blur && !dof)
        return 0;
    if (w != gWidth || h != gHeight)
    {
        waitQueueIdle(gQueue);
        freeSized();
        if (!makeSized(w, h))
        {
            freeSized();
            return 0;
        }
    }
    // Tiles of about a 24th of the height: the largest blur radius (a streak of up to 8% of the height, as UE's default
    // limit of a few percent of the width).
    uint32_t tile = (h + 12) / 24;
    tile = tile < 8 ? 8 : (tile > 96 ? 96 : tile);
    if (!blur && gBlur)
    {
        waitQueueIdle(gQueue); // frames in flight may still read them
        freeBlur();
    }
    if (blur && gTile != tile)
    {
        waitQueueIdle(gQueue);
        freeBlur();
        if (!makeBlur(tile))
        {
            freeBlur();
            blur = false;
        }
    }
    // Depth of field: a thin lens on a 24 mm high (full-frame) sensor, its focal length from the field of view
    // (f = S p11 / 2) and aperture from the f-stop; blur radius in display pixels f^2 H / (N S (zf - f)) (1 - zf / z).
    // Largest radius 2.5% of the height (at most 64 pixels); half-size tiles at least that wide.
    const float sensor = 0.024f;
    float       focal = 0.5f * sensor * F->p11;
    float       largest = 0.025f * (float)h < 64.0f ? 0.025f * (float)h : 64.0f;
    uint32_t    dofTile = ((uint32_t)ceilf(largest * 0.5f / 8.0f)) * 8;
    dofTile = dofTile < 8 ? 8 : dofTile;
    if (!dof && gDofOut)
    {
        waitQueueIdle(gQueue);
        freeDof();
    }
    if (dof && gDofTile != dofTile)
    {
        waitQueueIdle(gQueue);
        freeDof();
        if (!makeDof(dofTile))
        {
            freeDof();
            dof = false;
        }
    }
    d.dof[0] = focal;
    d.dof[1] = focal * focal * (float)h / (clampf(settings.f_stop, 1, 32) * sensor);
    d.dof[2] = largest;
    d.dof[3] = settings.focus == Focus::Manual ? (settings.focus_distance > 2 * focal ? settings.focus_distance : 2 * focal) : 0.0f;
    d.dof2[0] = 1.0f - expf(-gDt / 0.25f); // eases over about a quarter of a second
    d.dof2[1] = gFocusFresh ? 1.0f : 0.0f;
    d.dof2[2] = (float)dofTile;
    d.size[0] = w;
    d.size[1] = h;
    d.blur[0] = 0.5f * shutter * (float)w;
    d.blur[1] = 0.5f * shutter * (float)h;
    d.blur[2] = (float)tile;
    d.blur[3] = F->nearZ;
    d.blurSize[0] = (float)w;
    d.blurSize[1] = (float)h;
    d.blurSize[2] = (float)F->width;
    d.blurSize[3] = (float)F->height;
    d.bloom[0] = intensity;
    d.bloom[1] = clampf(settings.bloom_radius, 0.25f, 4);
    d.bloom[2] = 1.0f / (float)gLevels;
    d.look[0] = vignette;
    d.look[1] = grade ? 1.0f : 0.0f;
    d.look[2] = F->p00 > 0 ? 1.0f / F->p00 : 1.0f;
    d.look[3] = F->p11 > 0 ? 1.0f / F->p11 : 1.0f;
    gPlan.bake = grade && (!gLutValid || memcmp(gLutData.wb, d.wb, sizeof(PostData) - offsetof(PostData, wb)) != 0);
    gPlan.on = true;
    gPlan.bloom = intensity > 0;
    gPlan.blur = blur;
    gPlan.dof = dof;
    gPlan.grade = grade;
    gPlan.in = in;
    return (uint64_t)(uintptr_t)gOut;
}

void post_run(uint64_t address)
{
    RenderFrame* F = (RenderFrame*)(uintptr_t)address;
    if (!gPlan.on || !gReady || !F || !gOut)
        return;
    gPlan.on = false;
    gSlot = 0;
    Cmd* cmd = F->cmd;
    cmdBeginDebugMarker(cmd, 1.0f, 0.8f, 0.4f, "Renderer post");
    const PostData& base = gPlan.data;
    Texture*        scene = gPlan.in;
    if (gPlan.dof)
    {
        // The focus (from last frame's, eased), each half-size pixel's colour and blur, the foreground's reach per
        // tile, the two layers, then the full-size image.
        uint32_t hw = (gWidth + 1) / 2, hh = (gHeight + 1) / 2;
        Texture* last = gFocus[gFocusIndex];
        gFocusIndex ^= 1;
        gFocusRead = nullptr;
        dispatch(F, kDofFocus, base, nullptr, last, gFocus[gFocusIndex], 8, 8);
        gFocusRead = gFocus[gFocusIndex];
        gFocusFresh = false;
        PostData d = base;
        d.size[0] = hw;
        d.size[1] = hh;
        dispatch(F, kDofCoc, d, scene, nullptr, gDofHalf, hw, hh);
        d.size[0] = gDofTilesX;
        d.size[1] = gDofTilesY;
        d.size[2] = hw;
        d.size[3] = hh;
        dispatch(F, kDofTile, d, gDofHalf, nullptr, gDofTiles, gDofTilesX * 8, gDofTilesY * 8);
        dispatch(F, kDofNeighbor, d, gDofTiles, nullptr, gDofNeighbors, gDofTilesX, gDofTilesY);
        d.size[0] = hw;
        d.size[1] = hh;
        dispatch(F, kDofGather, d, gDofHalf, gDofNeighbors, gDofFar, hw, hh, nullptr, gDofNear);
        dispatch(F, kDofCombine, base, scene, gDofFar, gDofOut, gWidth, gHeight, gDofNear);
        gFocusRead = nullptr;
        scene = gDofOut;
    }
    if (gPlan.blur)
    {
        // Each tile's motion, then over its neighbours, then the gather at full size; bloom and the rest read it.
        PostData d = base;
        d.size[0] = gTilesX;
        d.size[1] = gTilesY;
        d.size[2] = (uint32_t)base.blurSize[2];
        d.size[3] = (uint32_t)base.blurSize[3];
        dispatch(F, kBlurTile, d, nullptr, nullptr, gTiles, gTilesX * 8, gTilesY * 8);
        dispatch(F, kBlurNeighbor, d, gTiles, nullptr, gNeighbors, gTilesX, gTilesY);
        dispatch(F, kBlurGather, base, scene, gNeighbors, gBlur, gWidth, gHeight);
        scene = gBlur;
    }
    if (gPlan.bake)
    {
        dispatch(F, kLut, base, nullptr, nullptr, gLut, 1024, 32);
        memcpy(&gLutData, &base, sizeof base);
        gLutValid = true;
    }
    if (gPlan.bloom)
    {
        // Down the chain from the full image, then back up it, each level adding its own blur.
        PostData d = base;
        for (uint32_t i = 0; i < gLevels; ++i)
        {
            uint32_t sw = i ? levelSize(gWidth, i - 1) : gWidth, sh = i ? levelSize(gHeight, i - 1) : gHeight;
            d.size[0] = levelSize(gWidth, i);
            d.size[1] = levelSize(gHeight, i);
            d.size[2] = sw;
            d.size[3] = sh;
            dispatch(F, i ? kDown : kDownFirst, d, i ? gDown[i - 1] : scene, nullptr, gDown[i], d.size[0], d.size[1]);
        }
        for (uint32_t i = gLevels - 1; i-- > 0;)
        {
            d.size[0] = levelSize(gWidth, i);
            d.size[1] = levelSize(gHeight, i);
            d.size[2] = levelSize(gWidth, i + 1);
            d.size[3] = levelSize(gHeight, i + 1);
            Texture* below = i + 2 < gLevels ? gUp[i + 1] : gDown[i + 1];
            dispatch(F, kUp, d, below, gDown[i], gUp[i], d.size[0], d.size[1]);
        }
    }
    dispatch(F, kCombine, base, scene, gPlan.bloom ? gUp[0] : nullptr, gOut, gWidth, gHeight);
    cmdEndDebugMarker(cmd);
}
} // namespace renderer_post
