// Shadows for the renderer core. The sun: up to four stabilised cascades (PSSM splits, a bounding
// sphere per slice so the size never changes as the camera turns, texel-snapped) in one atlas, each
// redrawn only when its view or a caster over it changed. Spot and point lights: one cached atlas of tiles; a light's tiles are
// redrawn only when it or a caster near it moved, a few lights per frame.
#include "yea_renderer_shadows.h"
#include "frame.h"
#include "mathx.h"

#include <yea/device.h>
#include <stdio.h>
#include <chrono>

#include <Common_3/Graphics/Interfaces/IGraphics.h>
#include <Common_3/Resources/ResourceLoader/Interfaces/IResourceLoader.h>
#include <Common_3/Graphics/FSL/defaults.h>
#define MAX_TEXTURES RENDER_MAX_TEXTURES
#include "Shaders/Shadow.srt.h"

using namespace mx;

namespace renderer_shadows
{
namespace
{
const uint32_t kMaxFrames = 4;
const uint32_t kSunViews = RENDER_MAX_CASCADES;
const uint32_t kCell = 128;              // the local atlas is handed out in squares of 128 px cells
const uint32_t kMaxSide = 64;            // cells per side at the largest atlas (8192)
const uint32_t kGroups = 64;             // shadowed lights at most, each with six view slots
const uint32_t kMaxViews = kSunViews + kGroups * 6;
const uint32_t kMaxEntries = 1u << 20;   // caster entries per frame

struct Group
{
    uint32_t              id;
    uint32_t              valid;
    uint32_t              faces;
    uint32_t              used; // this frame
    uint64_t              rendered;
    uint64_t              seen;
    float                 key[10];
    RenderGpuShadowView views[6];
    uint8_t               k[6];   // each face's square, in cells per side (sized by its own footprint on screen)
    uint32_t              nfaces; // squares held (0: none)
    float                 need[6];  // texels each face wants (faceWant), refreshed a few lights a frame
    uint32_t              needOk;   // need[] filled
    uint8_t               cx[6], cy[6];
};

Renderer*      gRenderer;
Queue*         gQueue;
uint32_t       gFrames;
bool           gReady;
Shader *       gShader, *gClearShader;
Pipeline *     gPipe, *gClearPipe;
// Alpha-tested casters (RenderCaster::masked): uv0 and the base colour map, cut out as the camera's passes cut them. Opaque
// casters stay on gPipe's positions-only path.
Shader*        gMaskedShader;
Pipeline*      gMaskedPipe;
uint32_t       gTexBound[kMaxFrames]; // the frame's textureVersion each set holds (0: none yet)
bool           gMaskedNow;            // this frame draws alpha-tested casters on gMaskedPipe (the maps are bound)
float          gPipeSlope = -1;
float          gGuardKey = -1; // what the point lights' face margins were drawn for
DescriptorSet* gSet;
Buffer *       gViewBuf[kMaxFrames], *gStreamBuf[kMaxFrames];
RenderTarget * gSunAtlas, *gLocalAtlas;
uint32_t       gSunSize, gSunRes, gLocalSize;
bool           gLocalFresh; // cleared once when made, so tiles never drawn read as empty
Group          gGroups[kGroups];
uint64_t       gOcc[kMaxSide]; // the local atlas's cells in use, a row each
uint32_t       gSide;          // cells per side
// The views are built and read here, then copied once into the mapped buffer (reading mapped
// memory back is very slow on write-combined heaps).
RenderGpuShadowView gViewsCpu[kMaxViews];
uint32_t*             gStreamCpu;

// Between prepare and render.
uint32_t gCascades;
float    gCascadeEnd[RENDER_MAX_CASCADES];
// While F1 shows Wireframe or Overdraw nothing is drawn (Wireframe would draw the maps as edges): the
// cascades and tiles keep what they had, and every light is redrawn once the mode ends.
bool gFrozen, gWasFrozen;
bool gLightShadowsOn; // this frame: point and spot lights get shadow maps (light_shadows_on)
// The cascades as last drawn: their views, and a sum of hashes of the casters over each (order-free,
// so the casters' order can change). A cascade whose view and casters match keeps what it has.
uint32_t              gSunValid; // bit per cascade: drawn and kept since
uint32_t              gSunDirty; // bit per cascade: to draw this frame
RenderGpuShadowView gSunDrawn[kSunViews];
uint64_t              gSunSig[kSunViews];
uint32_t gNeedCursor; // the next light whose faces' needs are refreshed
uint32_t gPending[kGroups];
uint32_t gPendingCount;
uint32_t gStreamUsed;

uint64_t mix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

uint32_t clampu(int32_t v, uint32_t lo, uint32_t hi) { return v < (int32_t)lo ? lo : ((uint32_t)v > hi ? hi : (uint32_t)v); }

uint64_t rowMask(uint32_t x, uint32_t k) { return (k >= 64 ? ~0ull : ((1ull << k) - 1)) << x; }

bool cellsFree(uint32_t x, uint32_t y, uint32_t k)
{
    for (uint32_t r = y; r < y + k; ++r)
        if (gOcc[r] & rowMask(x, k))
            return false;
    return true;
}

void cellsMark(uint32_t x, uint32_t y, uint32_t k, bool on)
{
    for (uint32_t r = y; r < y + k; ++r)
        gOcc[r] = on ? gOcc[r] | rowMask(x, k) : gOcc[r] & ~rowMask(x, k);
}

uint32_t compactBits(uint32_t v)
{
    v &= 0x55555555u;
    v = (v | (v >> 1)) & 0x33333333u;
    v = (v | (v >> 2)) & 0x0F0F0F0Fu;
    v = (v | (v >> 4)) & 0x00FF00FFu;
    return (v | (v >> 8)) & 0x0000FFFFu;
}

// A free k x k square aligned to k, the first in Morton order: small squares fill one corner of a
// larger one before the next, which leaves room for large ones.
bool cellsAlloc(uint32_t k, uint32_t* x, uint32_t* y)
{
    uint32_t n = gSide / k;
    for (uint32_t i = 0; i < n * n; ++i)
    {
        uint32_t bx = compactBits(i) * k, by = compactBits(i >> 1) * k;
        if (cellsFree(bx, by, k))
        {
            *x = bx;
            *y = by;
            return true;
        }
    }
    return false;
}

void groupFree(Group& g)
{
    for (uint32_t f = 0; f < g.nfaces; ++f)
        cellsMark(g.cx[f], g.cy[f], g.k[f], false);
    g.nfaces = 0;
    memset(g.k, 0, sizeof g.k);
}

// A square for each of a group's faces at its size in k (cells per side), largest first so they pack, each at the
// largest smaller size that fits when its own doesn't.
bool groupAlloc(Group& g, uint32_t faces, const uint32_t* k)
{
    groupFree(g);
    uint32_t order[6];
    for (uint32_t f = 0; f < faces; ++f)
    {
        uint32_t i = f;
        for (; i > 0 && k[order[i - 1]] < k[f]; --i)
            order[i] = order[i - 1];
        order[i] = f;
    }
    for (uint32_t i = 0; i < faces; ++i)
    {
        uint32_t f = order[i], x = 0, y = 0, c = k[f];
        while (c >= 1 && !cellsAlloc(c, &x, &y))
            c /= 2;
        if (!c)
        {
            for (uint32_t j = 0; j < i; ++j)
                cellsMark(g.cx[order[j]], g.cy[order[j]], g.k[order[j]], false);
            memset(g.k, 0, sizeof g.k);
            return false;
        }
        cellsMark(x, y, c, true);
        g.cx[f] = (uint8_t)x;
        g.cy[f] = (uint8_t)y;
        g.k[f] = (uint8_t)c;
    }
    g.nfaces = faces;
    return true;
}

uint32_t pow2Floor(uint32_t v)
{
    uint32_t p = 1;
    while (p * 2 <= v)
        p *= 2;
    return p;
}

void freeTarget(RenderTarget*& t)
{
    if (t)
        removeRenderTarget(gRenderer, t);
    t = nullptr;
}

RenderTarget* makeAtlas(uint32_t size, const char* name)
{
    RenderTargetDesc d = {};
    d.mWidth = d.mHeight = size;
    d.mDepth = d.mArraySize = d.mMipLevels = 1;
    d.mSampleCount = SAMPLE_COUNT_1;
    d.mFormat = TinyImageFormat_D32_SFLOAT;
    d.mStartState = RESOURCE_STATE_SHADER_RESOURCE;
    d.mDescriptors = DESCRIPTOR_TYPE_TEXTURE;
    d.mClearValue = {};
    d.mClearValue.depth = 1.0f;
    d.pName = name;
    RenderTarget* rt = nullptr;
    addRenderTarget(gRenderer, &d, &rt);
    return rt;
}

void makePipelines(float slope)
{
    if (gPipe)
        removePipeline(gRenderer, gPipe);
    if (gClearPipe)
        removePipeline(gRenderer, gClearPipe);
    if (gMaskedPipe)
        removePipeline(gRenderer, gMaskedPipe);
    gPipe = gClearPipe = gMaskedPipe = nullptr;
    VertexLayout l = {};
    l.mBindingCount = 2; // the frame's positions (nothing else is read), then the instance-rate stream
    l.mBindings[0].mStride = RENDER_POSITION_STRIDE;
    l.mBindings[1].mStride = 4;
    l.mBindings[1].mRate = VERTEX_BINDING_RATE_INSTANCE;
    l.mAttribCount = 2;
    l.mAttribs[0].mSemantic = SEMANTIC_POSITION;
    l.mAttribs[0].mFormat = TinyImageFormat_R32G32B32_SFLOAT;
    l.mAttribs[0].mBinding = 0;
    l.mAttribs[0].mLocation = 0;
    l.mAttribs[1].mSemantic = SEMANTIC_TEXCOORD1;
    l.mAttribs[1].mFormat = TinyImageFormat_R32_UINT;
    l.mAttribs[1].mBinding = 1;
    l.mAttribs[1].mLocation = 1;

    RasterizerStateDesc raster = {};
    raster.mCullMode = CULL_MODE_NONE;
    raster.mSlopeScaledDepthBias = slope;
#if !defined(_WIN32)
    raster.mDepthBias = 1; // Vulkan's backend turns depth bias on only with a constant term; 1 is one ulp of float depth
#endif
    raster.mDepthClampEnable = true; // casters behind a cascade's near plane still cast
    DepthStateDesc depth = {};
    depth.mDepthTest = true;
    depth.mDepthWrite = true;
    depth.mDepthFunc = CMP_LEQUAL;

    PipelineDesc desc = {};
    desc.mType = PIPELINE_TYPE_GRAPHICS;
    PIPELINE_LAYOUT_DESC(desc, NULL, SRT_LAYOUT_DESC(ShadowSrt, PerFrame), NULL, NULL);
    GraphicsPipelineDesc& g = desc.mGraphicsDesc;
    g.mPrimitiveTopo = PRIMITIVE_TOPO_TRI_LIST;
    g.mRenderTargetCount = 0;
    g.mSampleCount = SAMPLE_COUNT_1;
    g.mDepthStencilFormat = TinyImageFormat_D32_SFLOAT;
    g.pRasterizerState = &raster;
    g.pDepthState = &depth;
    g.pVertexLayout = &l;
    g.pShaderProgram = gShader;
    desc.pName = "renderer_shadow";
    addPipeline(gRenderer, &desc, &gPipe);
    if (gMaskedShader)
    {
        // Positions, then the attributes (uv0 at 8, the colour at 16), then the instance-rate stream.
        VertexLayout m = {};
        m.mBindingCount = 3;
        m.mBindings[0].mStride = RENDER_POSITION_STRIDE;
        m.mBindings[1].mStride = RENDER_ATTRIBUTE_STRIDE;
        m.mBindings[2].mStride = 4;
        m.mBindings[2].mRate = VERTEX_BINDING_RATE_INSTANCE;
        m.mAttribCount = 4;
        m.mAttribs[0] = l.mAttribs[0];
        m.mAttribs[1] = l.mAttribs[1];
        m.mAttribs[1].mBinding = 2;
        m.mAttribs[2].mSemantic = SEMANTIC_TEXCOORD0;
        m.mAttribs[2].mFormat = TinyImageFormat_R32G32_SFLOAT;
        m.mAttribs[2].mBinding = 1;
        m.mAttribs[2].mLocation = 2;
        m.mAttribs[2].mOffset = 8;
        m.mAttribs[3].mSemantic = SEMANTIC_COLOR;
        m.mAttribs[3].mFormat = TinyImageFormat_R8G8B8A8_UNORM;
        m.mAttribs[3].mBinding = 1;
        m.mAttribs[3].mLocation = 3;
        m.mAttribs[3].mOffset = 16;
        g.pVertexLayout = &m;
        g.pShaderProgram = gMaskedShader;
        desc.pName = "renderer_shadow_masked";
        addPipeline(gRenderer, &desc, &gMaskedPipe);
    }

    RasterizerStateDesc plain = {};
    plain.mCullMode = CULL_MODE_NONE;
    DepthStateDesc always = {};
    always.mDepthTest = true;
    always.mDepthWrite = true;
    always.mDepthFunc = CMP_ALWAYS;
    g.pRasterizerState = &plain;
    g.pDepthState = &always;
    g.pVertexLayout = nullptr;
    g.pShaderProgram = gClearShader;
    desc.pName = "renderer_shadow_clear";
    addPipeline(gRenderer, &desc, &gClearPipe);
    gPipeSlope = slope;
}

// The view's rows and the first three columns of their inverse, from which the forward pass gets a
// receiver plane's depth slope in the map with three dot products (planes stay planes under projection).
void setRows(RenderGpuShadowView& v, const float m[4][4])
{
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            v.row[r][c] = m[r][c];
    const float* a = &m[0][0];
    float        inv[16];
    inv[0] = a[5] * a[10] * a[15] - a[5] * a[11] * a[14] - a[9] * a[6] * a[15] + a[9] * a[7] * a[14] + a[13] * a[6] * a[11] - a[13] * a[7] * a[10];
    inv[4] = -a[4] * a[10] * a[15] + a[4] * a[11] * a[14] + a[8] * a[6] * a[15] - a[8] * a[7] * a[14] - a[12] * a[6] * a[11] + a[12] * a[7] * a[10];
    inv[8] = a[4] * a[9] * a[15] - a[4] * a[11] * a[13] - a[8] * a[5] * a[15] + a[8] * a[7] * a[13] + a[12] * a[5] * a[11] - a[12] * a[7] * a[9];
    inv[12] = -a[4] * a[9] * a[14] + a[4] * a[10] * a[13] + a[8] * a[5] * a[14] - a[8] * a[6] * a[13] - a[12] * a[5] * a[10] + a[12] * a[6] * a[9];
    inv[1] = -a[1] * a[10] * a[15] + a[1] * a[11] * a[14] + a[9] * a[2] * a[15] - a[9] * a[3] * a[14] - a[13] * a[2] * a[11] + a[13] * a[3] * a[10];
    inv[5] = a[0] * a[10] * a[15] - a[0] * a[11] * a[14] - a[8] * a[2] * a[15] + a[8] * a[3] * a[14] + a[12] * a[2] * a[11] - a[12] * a[3] * a[10];
    inv[9] = -a[0] * a[9] * a[15] + a[0] * a[11] * a[13] + a[8] * a[1] * a[15] - a[8] * a[3] * a[13] - a[12] * a[1] * a[11] + a[12] * a[3] * a[9];
    inv[13] = a[0] * a[9] * a[14] - a[0] * a[10] * a[13] - a[8] * a[1] * a[14] + a[8] * a[2] * a[13] + a[12] * a[1] * a[10] - a[12] * a[2] * a[9];
    inv[2] = a[1] * a[6] * a[15] - a[1] * a[7] * a[14] - a[5] * a[2] * a[15] + a[5] * a[3] * a[14] + a[13] * a[2] * a[7] - a[13] * a[3] * a[6];
    inv[6] = -a[0] * a[6] * a[15] + a[0] * a[7] * a[14] + a[4] * a[2] * a[15] - a[4] * a[3] * a[14] - a[12] * a[2] * a[7] + a[12] * a[3] * a[6];
    inv[10] = a[0] * a[5] * a[15] - a[0] * a[7] * a[13] - a[4] * a[1] * a[15] + a[4] * a[3] * a[13] + a[12] * a[1] * a[7] - a[12] * a[3] * a[5];
    inv[14] = -a[0] * a[5] * a[14] + a[0] * a[6] * a[13] + a[4] * a[1] * a[14] - a[4] * a[2] * a[13] - a[12] * a[1] * a[6] + a[12] * a[2] * a[5];
    inv[3] = -a[1] * a[6] * a[11] + a[1] * a[7] * a[10] + a[5] * a[2] * a[11] - a[5] * a[3] * a[10] - a[9] * a[2] * a[7] + a[9] * a[3] * a[6];
    inv[7] = a[0] * a[6] * a[11] - a[0] * a[7] * a[10] - a[4] * a[2] * a[11] + a[4] * a[3] * a[10] + a[8] * a[2] * a[7] - a[8] * a[3] * a[6];
    inv[11] = -a[0] * a[5] * a[11] + a[0] * a[7] * a[9] + a[4] * a[1] * a[11] - a[4] * a[3] * a[9] - a[8] * a[1] * a[7] + a[8] * a[3] * a[5];
    inv[15] = a[0] * a[5] * a[10] - a[0] * a[6] * a[9] - a[4] * a[1] * a[10] + a[4] * a[2] * a[9] + a[8] * a[1] * a[6] - a[8] * a[2] * a[5];
    float det = a[0] * inv[0] + a[1] * inv[4] + a[2] * inv[8] + a[3] * inv[12];
    float s = fabsf(det) > 1e-30f ? 1.0f / det : 0.0f;
    // inv is row-major (inv[r * 4 + c]); column c is inv[c], inv[4 + c], inv[8 + c], inv[12 + c].
    for (int c = 0; c < 3; ++c)
        for (int r = 0; r < 4; ++r)
            v.inv[c][r] = inv[r * 4 + c] * s;
}

// A perspective view (standard Z: near 0, far 1) from eye along f.
void perspective(RenderGpuShadowView& v, v3 eye, v3 f, float tanHalf, float nearZ, float farZ)
{
    v3 right, up;
    basis(f, &right, &up);
    float view[4][4], m[4][4];
    lookRows(eye, f, right, up, view);
    float a = farZ / (farZ - nearZ), b = -farZ * nearZ / (farZ - nearZ);
    float proj[4][4] = { { 1 / tanHalf, 0, 0, 0 }, { 0, 1 / tanHalf, 0, 0 }, { 0, 0, a, b }, { 0, 0, 1, 0 } };
    mul(proj, view, m);
    setRows(v, m);
}

const v3 kFaces[6] = { { 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 }, { 0, -1, 0 }, { 0, 0, 1 }, { 0, 0, -1 } };

// A light's penumbra reaches a share of its tile, 8% to 25% with its radius (the forward pass's lightShadowSize and
// localShadow take the same): 0 without contact-hardening.
float penumbraShare(const RenderLight& l)
{
    if (!settings.contact_hardening)
        return 0.0f;
    float size = l.size > 0 ? l.size : settings.light_size;
    return size > 0 ? clampf(size * 0.5f, 0.08f, 0.25f) : 0.0f;
}

void lightKey(const RenderLight& l, float key[10])
{
    key[0] = l.pos[0];
    key[1] = l.pos[1];
    key[2] = l.pos[2];
    key[3] = l.dir[0];
    key[4] = l.dir[1];
    key[5] = l.dir[2];
    key[6] = l.range;
    key[7] = l.cosOuter;
    key[8] = (float)l.type;
    key[9] = penumbraShare(l);
}

// When the faces' sizes change, at most every 20 s: how many faces hold each size, and how full the atlas is.
double   gLogTime = -1e9; // when the faces were last logged (s)
uint32_t gLogSizes[5];
uint32_t gWanted; // cells the faces wanted this frame, before they were fitted into the atlas
uint32_t gLocalAuto;      // the lights' atlas size it has grown to (0: not yet chosen)
bool     gRefusedLogged; // the last growth asked for was refused and logged (log the next refusal only after a growth)
double   gOverSince = -1; // since when the faces have wanted more than fits (s), or -1
bool     gRedrawAll;      // the atlas was just made larger: every light is redrawn this frame
double secondsNow() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

void logAtlas()
{
    uint32_t sizes[5] = {}, used = 0;
    for (const Group& g : gGroups)
        for (uint32_t f = 0; f < g.nfaces; ++f)
        {
            uint32_t i = 0;
            while ((2u << i) <= g.k[f] && i < 4)
                ++i;
            sizes[i]++;
            used += g.k[f] * g.k[f];
        }
    double now = secondsNow();
    if (!memcmp(sizes, gLogSizes, sizeof sizes) || now < gLogTime + 20.0)
        return;
    memcpy(gLogSizes, sizes, sizeof sizes);
    gLogTime = now;
    char line[160];
    snprintf(line, sizeof line, "renderer_shadows: light faces at 128/256/512/1024/2048: %u/%u/%u/%u/%u; local atlas %u%% full (%u), %u%% wanted",
             sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], gSide ? used * 100 / (gSide * gSide) : 0, gSide * kCell,
             gSide ? gWanted * 100 / (gSide * gSide) : 0);
    yea::log(line);
}

// Clips s to where a + b * s >= 0.
void clipRay(float a, float b, float* s0, float* s1)
{
    if (b > 0.0f)
        *s0 = maxf(*s0, -a / b);
    else if (b < 0.0f)
        *s1 = minf(*s1, -a / b);
    else if (a < 0.0f)
        *s1 = -1.0f;
}

// A caster seen by the camera, as a receiver of light shadows: its mesh's bounding sphere through its instance's
// transform, an ellipsoid (a floor is a flat disc, not a sphere around it).
struct Receiver
{
    v3    c;       // world centre
    float r;       // world bounding sphere's radius
    float inv[9];  // world -> object, rows
    float radius;  // object units
};
const uint32_t kReceivers = 1024;
Receiver       gRecv[kReceivers];
uint32_t       gRecvCount;
bool           gRecvAll; // every caster in view fitted

// The casters the camera sees, each with its inverse transform.
void gatherReceivers(const RenderFrame* F, const mx::Plane* frustum)
{
    gRecvCount = 0;
    gRecvAll = true;
    uint32_t meshes = renderer::Meshes.size();
    for (uint32_t i = 0; i < F->casterCount; ++i)
    {
        const RenderCaster& k = F->casters[i];
        v3                  c = V(k.sphere[0], k.sphere[1], k.sphere[2]);
        if (k.instance >= F->instanceCount || k.mesh >= meshes || !sphereInside(frustum, 5, c, k.sphere[3]))
            continue;
        if (gRecvCount == kReceivers)
        {
            gRecvAll = false;
            break;
        }
        const float(*m)[4] = F->instanceRows[k.instance].row;
        float a = m[0][0], b = m[0][1], cc = m[0][2], d = m[1][0], e = m[1][1], f = m[1][2], g = m[2][0], h = m[2][1], q = m[2][2];
        float A = e * q - f * h, B = f * g - d * q, C = d * h - e * g, det = a * A + b * B + cc * C;
        if (fabsf(det) < 1e-12f)
            continue;
        float     s = 1.0f / det;
        Receiver& o = gRecv[gRecvCount++];
        o.c = c;
        o.r = k.sphere[3];
        o.radius = renderer::Meshes[k.mesh].radius;
        float inv[9] = { A * s, (cc * h - b * q) * s, (b * f - cc * e) * s, B * s, (a * q - cc * g) * s, (cc * d - a * f) * s,
                         C * s, (b * g - a * h) * s, (a * e - b * d) * s };
        memcpy(o.inv, inv, sizeof inv);
    }
}

// The texels across a face (from the light at p along dir, half-angle tangent t, out to range) that match the screen's
// pixels where the face's texels are largest on what the camera sees. A texel s from the light is 2 * t * s / texels
// across and a pixel z deep is z / focal, so a face wants 2 * t * focal * s / z texels at its largest s / z over the
// receivers in it: the casters in view (recv, those within the light's range), where a 5 x 5 grid of rays across the
// face meets them inside the camera's view and the light's range, plus the centres of casters too small for the rays to
// be sure of. Empty space never asks for texels: a face over nothing but air keeps the smallest square. 0: no receiver
// in view; -1: the camera sees none of the face, or it is wholly behind a one-sided light.
// lit: a one-sided light's normal (rects, discs), which lights nothing behind it; else null.
float faceWant(const RenderFrame* F, v3 p, v3 dir, float t, float range, float focal, const uint32_t* recv, uint32_t recvCount, const v3* lit)
{
    auto toView = [&](v3 q, float w) {
        return V(F->view[0][0] * q.x + F->view[0][1] * q.y + F->view[0][2] * q.z + F->view[0][3] * w,
                 F->view[1][0] * q.x + F->view[1][1] * q.y + F->view[1][2] * q.z + F->view[1][3] * w,
                 F->view[2][0] * q.x + F->view[2][1] * q.y + F->view[2][2] * q.z + F->view[2][3] * w);
    };
    // The camera's view: z >= near and |x| * p00 <= z, |y| * p11 <= z, each as a + b * s >= 0 along a ray.
    v3    o = toView(p, 1.0f), right, up;
    float px = F->p00, py = F->p11, nearZ = F->nearZ;
    basis(dir, &right, &up);
    // Nothing of the face in view: its pyramid (light, then the four corners at range) wholly outside one side.
    v3 corner[4];
    for (uint32_t c = 0; c < 4; ++c)
        corner[c] = toView((dir + right * (c & 1 ? t : -t) + up * (c & 2 ? t : -t)) * range, 0.0f) + o;
    for (uint32_t side = 0; side < 5; ++side)
    {
        auto inside = [&](v3 q) {
            float v[5] = { q.z - nearZ, q.z - px * q.x, q.z + px * q.x, q.z - py * q.y, q.z + py * q.y };
            return v[side] >= 0.0f;
        };
        bool any = inside(o);
        for (uint32_t c = 0; c < 4 && !any; ++c)
            any = inside(corner[c]);
        if (!any)
            return -1.0f;
    }
    // The receivers that reach into the face: their spheres against its four side planes.
    uint32_t mine[kReceivers], count = 0;
    v3       side[4] = { normalize(dir * t - right), normalize(dir * t + right), normalize(dir * t - up), normalize(dir * t + up) };
    if (lit)
    {
        // Wholly behind a one-sided light: it lights nothing there.
        bool front = false;
        for (uint32_t c = 0; c < 4 && !front; ++c)
            front = dot(dir + right * (c & 1 ? t : -t) + up * (c & 2 ? t : -t), *lit) > 0.0f;
        if (!front)
            return -1.0f;
    }
    for (uint32_t i = 0; i < recvCount; ++i)
    {
        const Receiver& r = gRecv[recv[i]];
        v3              d = r.c - p;
        bool            in = dot(d, dir) > -r.r && (!lit || dot(d, *lit) > -r.r);
        for (uint32_t k = 0; k < 4 && in; ++k)
            in = dot(d, side[k]) > -r.r;
        if (in)
            mine[count++] = recv[i];
    }
    float best = 0.0f;
    for (uint32_t i = 0; i < count; ++i)
    {
        // A small one by its centre, at its nearest depth.
        const Receiver& r = gRecv[mine[i]];
        v3              d = r.c - p;
        float           dist = length(d);
        if (r.r > 0.25f * dist || dist > range || dot(d, dir) * t < fmaxf(fabsf(dot(d, right)), fabsf(dot(d, up))))
            continue;
        v3 q = toView(r.c, 1.0f);
        if (q.z - r.r > nearZ && fabsf(q.x) * px <= q.z + r.r * px && fabsf(q.y) * py <= q.z + r.r * py)
            best = maxf(best, dist / (q.z - r.r));
    }
    for (int i = 0; i < 5; ++i)
        for (int k = 0; k < 5; ++k)
        {
            v3    rw = normalize(dir + right * (t * (i * 0.5f - 1.0f)) + up * (t * (k * 0.5f - 1.0f)));
            if (lit && dot(rw, *lit) <= 0.0f)
                continue;
            v3    r = toView(rw, 0.0f);
            float s0 = 0.0f, s1 = range;
            clipRay(o.z - nearZ, r.z, &s0, &s1);
            clipRay(o.z - px * o.x, r.z - px * r.x, &s0, &s1);
            clipRay(o.z + px * o.x, r.z + px * r.x, &s0, &s1);
            clipRay(o.z - py * o.y, r.z - py * r.y, &s0, &s1);
            clipRay(o.z + py * o.y, r.z + py * r.y, &s0, &s1);
            if (s0 > s1)
                continue;
            for (uint32_t j = 0; j < count; ++j)
            {
                // Where the ray is inside the receiver's ellipsoid: |inv (p + s rw - c)| <= radius.
                const Receiver& e = gRecv[mine[j]];
                v3              po = p - e.c;
                const float*    m = e.inv;
                v3              a = V(m[0] * po.x + m[1] * po.y + m[2] * po.z, m[3] * po.x + m[4] * po.y + m[5] * po.z, m[6] * po.x + m[7] * po.y + m[8] * po.z);
                v3              b = V(m[0] * rw.x + m[1] * rw.y + m[2] * rw.z, m[3] * rw.x + m[4] * rw.y + m[5] * rw.z, m[6] * rw.x + m[7] * rw.y + m[8] * rw.z);
                float           A = dot(b, b), B = dot(a, b), C = dot(a, a) - e.radius * e.radius, disc = B * B - A * C;
                if (disc < 0.0f || A <= 0.0f)
                    continue;
                float sq = sqrtf(disc), e0 = maxf((-B - sq) / A, s0), e1 = minf((-B + sq) / A, s1);
                if (e0 > e1)
                    continue;
                best = maxf(best, maxf(e0 / (o.z + r.z * e0), e1 / (o.z + r.z * e1)));
            }
        }
    return 2.0f * t * focal * best;
}

void buildLocalViews(Group& g, const RenderLight& l)
{
    float atlas = (float)(gSide * kCell);
    v3    p = V(l.pos[0], l.pos[1], l.pos[2]);
    float nearZ = clampf(l.range * 0.005f, 0.01f, 0.2f);
    g.faces = l.type == 1 ? 6 : 1;
    for (uint32_t f = 0; f < g.faces; ++f)
    {
        RenderGpuShadowView& v = g.views[f];
        float                  tanHalf, tileRes = (float)(g.k[f] * kCell);
        if (l.type == 1)
        {
            // Past 90 degrees by the widest filter the shader may use at the face's edge (where it switches faces),
            // so every tap lands in this face's own drawn texels: the PCF disk (softness, a quarter of it under
            // contact-hardening, whose penumbra reaches min(24, the light's share of the tile) texels and whose blocker search
            // 12), plus 2 for the bilinear compare and the 2 x 2 gathers. A narrower margin clamps the taps to the
            // tile's border texels, which draws a false shadow line along each face edge: a square around the
            // light's foot that moves with it.
            float share = penumbraShare(l);
            float guard = share > 0 ? maxf(minf(24.0f, share * tileRes) + 0.25f * settings.softness, 12.0f) + 2.0f : settings.softness + 2.0f;
            tanHalf = tileRes / maxf(tileRes - 2.0f * guard, tileRes * 0.5f);
            perspective(v, p, kFaces[f], tanHalf, nearZ, l.range);
        }
        else
        {
            float outer = acosf(clampf(l.cosOuter, -1, 1));
            tanHalf = tanf(minf(outer * 1.05f + 0.02f, 1.48f));
            perspective(v, p, normalize(V(l.dir[0], l.dir[1], l.dir[2])), tanHalf, nearZ, l.range);
        }
        v.rect[0] = (float)(g.cx[f] * kCell) / atlas;
        v.rect[1] = (float)(g.cy[f] * kCell) / atlas;
        v.rect[2] = v.rect[3] = tileRes / atlas;
        v.info[0] = 2 * tanHalf / tileRes;
        v.info[1] = p.x; // where it was drawn from: the shader picks a point light's face by it
        v.info[2] = p.y;
        v.info[3] = p.z;
    }
}

// The version of a caster to draw into a view: from the one the screen draws (never finer, so a caster never stands
// proud of what the camera sees and shadows it), the simplest whose error stays under half a texel where the caster is
// nearest the light. The sun's texels are one size (info.x); a light's grow with distance (info.x at 1 m, from info.yzw).
uint32_t shadowVersion(const RenderCaster& c, bool sun, const RenderGpuShadowView& v)
{
    uint32_t m = c.mesh, rows = renderer::MeshLods.size() < renderer::Meshes.size() ? renderer::MeshLods.size() : renderer::Meshes.size();
    if (m >= rows || renderer::MeshLods[m].next >= rows || renderer::Meshes[m].radius <= 0)
        return m;
    float texel = v.info[0];
    if (!sun)
        texel *= maxf(length(V(c.sphere[0] - v.info[1], c.sphere[1] - v.info[2], c.sphere[2] - v.info[3])) - c.sphere[3], 0.01f);
    float allowed = 0.5f * texel * renderer::Meshes[m].radius / c.sphere[3]; // object units (radius: the finest's, as the core's sphere)
    for (uint32_t n = renderer::MeshLods[m].next; n < rows && renderer::MeshLods[n].error <= allowed; n = renderer::MeshLods[n].next)
        m = n;
    return m;
}

// Draws the casters inside a view's frustum, one instanced draw per run of the same mesh version. The casters come
// sorted by the version the screen draws; each such run is drawn once per version it takes here, simplest last.
// masked: only the alpha-tested casters (gMaskedPipe bound, with the attributes), else only the others (gPipe).
void drawCasters(Cmd* cmd, const RenderFrame* F, uint32_t view, const RenderGpuShadowView& v, uint32_t* stream, Buffer* streamBuf,
                 bool masked)
{
    mx::Plane planes[6];
    clipPlanes(v.row, planes);
    const bool sun = view < kSunViews;
    auto draw = [&](uint32_t mesh, uint32_t first, uint32_t count) {
        if (!count)
            return;
        const renderer::Mesh& m = renderer::Meshes[mesh];
        if (masked)
        {
            Buffer*  bufs[3] = { F->positions, F->attributes, streamBuf };
            uint32_t strides[3] = { RENDER_POSITION_STRIDE, RENDER_ATTRIBUTE_STRIDE, 4 };
            uint64_t offsets[3] = { 0, 0, (uint64_t)first * 4 };
            cmdBindVertexBuffer(cmd, 3, bufs, strides, offsets);
        }
        else
        {
            Buffer*  bufs[2] = { F->positions, streamBuf };
            uint32_t strides[2] = { RENDER_POSITION_STRIDE, 4 };
            uint64_t offsets[2] = { 0, (uint64_t)first * 4 };
            cmdBindVertexBuffer(cmd, 2, bufs, strides, offsets);
        }
        cmdDrawIndexedInstanced(cmd, m.indexCount, m.firstIndex, count, m.firstVertex, 0);
    };
    // Without the masked pipeline (its shaders refused), alpha-tested casters draw whole with the others.
    auto ours = [&](const RenderCaster& c) { return gMaskedNow ? (c.masked != 0) == masked : !masked; };
    for (uint32_t i = 0; i < F->casterCount && gStreamUsed < kMaxEntries;)
    {
        // The run of casters the screen draws with one version (the core sorts them by it).
        uint32_t end = i + 1;
        while (end < F->casterCount && F->casters[end].mesh == F->casters[i].mesh)
            ++end;
        // Each version this run takes, finest first: a pass finds the finest not drawn yet, then draws the casters taking it.
        // Versions follow their finer ones in the Mesh table, so "not drawn yet" is a row past the last drawn.
        uint32_t floor = F->casters[i].mesh;
        for (;;)
        {
            uint32_t pick = 0xFFFFFFFFu;
            for (uint32_t k = i; k < end; ++k)
            {
                const RenderCaster& c = F->casters[k];
                uint32_t m = shadowVersion(c, sun, v);
                if (ours(c) && m >= floor && m < pick && sphereInside(planes, 6, V(c.sphere[0], c.sphere[1], c.sphere[2]), c.sphere[3]))
                    pick = m;
            }
            if (pick == 0xFFFFFFFFu)
                break;
            uint32_t first = gStreamUsed;
            for (uint32_t k = i; k < end && gStreamUsed < kMaxEntries; ++k)
            {
                const RenderCaster& c = F->casters[k];
                if (ours(c) && shadowVersion(c, sun, v) == pick && sphereInside(planes, 6, V(c.sphere[0], c.sphere[1], c.sphere[2]), c.sphere[3]))
                    stream[gStreamUsed++] = c.instance | (view << RENDER_VIEW_SHIFT);
            }
            draw(pick, first, gStreamUsed - first);
            floor = pick + 1;
        }
        i = end;
    }
}
} // namespace

int load(int reloaded)
{
    const YeaHost* host = yea::host();
    gRenderer = (Renderer*)host->renderer;
    gQueue = (Queue*)host->queue;
    gFrames = host->framesInFlight < kMaxFrames ? host->framesInFlight : kMaxFrames;
    ShaderLoadDesc sd = {};
    sd.mVert.pFileName = "renderer_shadow.vert";
    addShader(gRenderer, &sd, &gShader);
    ShaderLoadDesc cd = {};
    cd.mVert.pFileName = "renderer_shadow_clear.vert";
    addShader(gRenderer, &cd, &gClearShader);
    if (!gShader || !gClearShader)
        return 1;
    ShaderLoadDesc md = {};
    md.mVert.pFileName = "renderer_shadow_masked.vert";
    md.mFrag.pFileName = "renderer_shadow_masked.frag";
    addShader(gRenderer, &md, &gMaskedShader); // refused: alpha-tested casters cast whole shadows
    for (uint32_t i = 0; i < kMaxFrames; ++i)
        gTexBound[i] = 0;
    DescriptorSetDesc setDesc = SRT_SET_DESC(ShadowSrt, PerFrame, gFrames, 0);
    addDescriptorSet(gRenderer, &setDesc, &gSet);
    for (uint32_t i = 0; i < gFrames; ++i)
    {
        BufferLoadDesc d = {};
        d.mDesc.mDescriptors = DESCRIPTOR_TYPE_BUFFER;
        d.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
        d.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
        d.mDesc.mStartState = RESOURCE_STATE_SHADER_RESOURCE;
        d.mDesc.mSize = sizeof(RenderGpuShadowView) * kMaxViews;
        d.mDesc.mStructStride = sizeof(RenderGpuShadowView);
        d.mDesc.mElementCount = kMaxViews;
        d.mDesc.pName = "renderer_shadow_views";
        d.ppBuffer = &gViewBuf[i];
        addResource(&d, nullptr);
        BufferLoadDesc s = {};
        s.mDesc.mDescriptors = DESCRIPTOR_TYPE_VERTEX_BUFFER;
        s.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
        s.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
        s.mDesc.mStartState = RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
        s.mDesc.mSize = (uint64_t)kMaxEntries * 4;
        s.mDesc.pName = "renderer_shadow_stream";
        s.ppBuffer = &gStreamBuf[i];
        addResource(&s, nullptr);
    }
    waitForAllResourceLoads();
    gStreamCpu = new uint32_t[kMaxEntries];
    gReady = true;
    return 0;
}

void unload()
{
    if (!gRenderer)
        return;
    waitQueueIdle(gQueue);
    if (gPipe)
        removePipeline(gRenderer, gPipe);
    if (gClearPipe)
        removePipeline(gRenderer, gClearPipe);
    if (gMaskedPipe)
        removePipeline(gRenderer, gMaskedPipe);
    if (gMaskedShader)
        removeShader(gRenderer, gMaskedShader);
    freeTarget(gSunAtlas);
    freeTarget(gLocalAtlas);
    for (uint32_t i = 0; i < gFrames; ++i)
    {
        if (gViewBuf[i])
            removeResource(gViewBuf[i]);
        if (gStreamBuf[i])
            removeResource(gStreamBuf[i]);
    }
    if (gSet)
        removeDescriptorSet(gRenderer, gSet);
    if (gShader)
        removeShader(gRenderer, gShader);
    if (gClearShader)
        removeShader(gRenderer, gClearShader);
}

void update(uint32_t phase, float dt) {}
void draw(uint32_t pass, const YeaFrame* frame) {}

bool shadows_prepare(uint64_t address)
{
    RenderFrame* F = (RenderFrame*)(uintptr_t)address;
    gLightShadowsOn = false;
    if (!gReady || !F || F->version != RENDER_FRAME_VERSION)
        return false;
    // The lights' real sizes for renderer_rt's traced penumbrae: always, whatever the maps' own filtering (pcss,
    // contact_hardening) and even with the maps off.
    F->rtSunAngle = settings.sun_size * 3.14159265f / 180.0f;
    F->rtLightRadius = settings.light_size;
    if (!settings.enabled && !settings.light_shadows)
        return false;

    // Atlases and pipelines follow the knobs.
    uint32_t cascades = clampu(settings.cascades, 1, RENDER_MAX_CASCADES);
    uint32_t res = pow2Floor(clampu(settings.sun_resolution, 512, 4096));
    uint32_t sunSize = cascades > 1 ? res * 2 : res;
    // The lights' atlas sizes itself: from 4096 (or the knob, if smaller), doubled up to the knob once the faces have
    // wanted more than fits for a second, as far as the card's memory allows. It never shrinks while the game runs (only
    // when the knob is lowered), so a light's detail doesn't come and go. Memory is asked only then: the driver's free
    // budget (yea::deviceMemory) must hold the growth with a margin left for the rest of the game (an eighth of the
    // budget, at least 512 MB); where it can't say, cards of 6 GB or more grow to 8192. Refused, it asks again in 10 s.
    uint32_t localMax = pow2Floor(clampu(settings.local_atlas_resolution, 1024, 8192));
    if (!gLocalAuto || gLocalAuto > localMax)
        gLocalAuto = localMax < 4096 ? localMax : 4096;
    double now = secondsNow();
    if (gLocalAuto < localMax && gSide && gWanted > gSide * gSide * 9 / 10)
    {
        if (gOverSince < 0)
            gOverSince = now;
        else if (now - gOverSince >= 1.0)
        {
            uint64_t        extra = 3ull * gLocalAuto * gLocalAuto * 4; // 4 bytes a texel, the new atlas less the old
            YeaDeviceMemory m = {};
            bool            known = yea::deviceMemory(&m);
            uint64_t        total = m.total ? m.total : (gRenderer->pGpu ? gRenderer->pGpu->mVRAM : 0);
            uint64_t        margin = m.budget / 8 > (512ull << 20) ? m.budget / 8 : (512ull << 20);
            uint64_t        free = m.budget > m.usage ? m.budget - m.usage : 0;
            bool            room = known ? free >= extra + margin : total >= (6ull << 30);
            char            line[200];
            if (room)
            {
                snprintf(line, sizeof line, "renderer_shadows: the lights' atlas grows to %u (%u MB): its faces wanted %u%% of %u; %s %u MB free",
                         gLocalAuto * 2, (gLocalAuto * 2 / 1024) * (gLocalAuto * 2 / 1024) * 4, gWanted * 100 / (gSide * gSide), gLocalAuto,
                         known ? "the card has" : "no budget known, total", (uint32_t)((known ? free : total) >> 20));
                gLocalAuto *= 2;
                gRedrawAll = true; // every light at once in the new atlas, so none is left without its shadow
                gOverSince = -1;
            }
            else
            {
                snprintf(line, sizeof line, "renderer_shadows: the lights' atlas stays %u: its faces want %u%% of it, but %s %u MB free", gLocalAuto,
                         gWanted * 100 / (gSide * gSide), known ? "the card has" : "no budget known, total", (uint32_t)((known ? free : total) >> 20));
                gOverSince = now + 9.0; // ask again in 10 s
            }
            if (room || !gRefusedLogged)
                yea::log(line);
            gRefusedLogged = !room;
        }
    }
    else
        gOverSince = -1;
    uint32_t localSize = gLocalAuto;
    float    slope = settings.slope_bias;
    if (sunSize != gSunSize || localSize != gLocalSize || slope != gPipeSlope)
    {
        waitQueueIdle(gQueue);
        if (sunSize != gSunSize)
        {
            freeTarget(gSunAtlas);
            gSunAtlas = makeAtlas(sunSize, "renderer_sun_atlas");
            gSunSize = sunSize;
            gCascades = 0; // nothing in it yet
            gSunValid = 0;
        }
        if (localSize != gLocalSize)
        {
            freeTarget(gLocalAtlas);
            gLocalAtlas = makeAtlas(localSize, "renderer_local_atlas");
            gLocalFresh = true;
            gLocalSize = localSize;
            memset(gGroups, 0, sizeof gGroups);
            memset(gOcc, 0, sizeof gOcc);
            gSide = localSize / kCell;
        }
        if (slope != gPipeSlope)
        {
            makePipelines(slope);
            for (Group& g : gGroups)
                g.valid = 0;
            gSunValid = 0;
        }
    }
    gSunRes = res;
    // The point lights' faces are drawn wider for wider filters (buildLocalViews): redraw them when those change.
    float guardKey = settings.softness + (settings.contact_hardening ? 1000.0f : 0.0f);
    if (guardKey != gGuardKey)
    {
        for (Group& g : gGroups)
            g.valid = 0;
        gGuardKey = guardKey;
    }
    if (!gSunAtlas || !gLocalAtlas || !gPipe)
        return false;

    RenderGpuShadowView* views = gViewsCpu;
    gFrozen = yea::viewMode() != 0;
    if (gWasFrozen && !gFrozen)
    {
        for (Group& g : gGroups)
            g.rendered = 0; // stale: redrawn oldest first, a few each frame
        gSunValid = 0;
    }
    gSunDirty = 0;
    gWasFrozen = gFrozen;

    // --- Sun cascades -------------------------------------------------------------------------
    if (!gFrozen)
    {
        memset(views, 0, sizeof(RenderGpuShadowView) * kSunViews);
        gCascades = 0;
    }
    // Traced: renderer_rt has it, unless fog or particles read the cascades (keepShadowMaps).
    if (!gFrozen && settings.enabled && F->sunOn && F->sunShadow && (!F->rtSunShadows || F->keepShadowMaps))
    {
        gCascades = cascades;
        v3    toSun = normalize(V(F->sunDir[0], F->sunDir[1], F->sunDir[2]));
        v3    f = -toSun, lr, lu;
        basis(f, &lr, &lu);
        v3    camPos = V(F->camPos[0], F->camPos[1], F->camPos[2]);
        v3    camFwd = V(F->view[2][0], F->view[2][1], F->view[2][2]);
        float k = 1 / (F->p00 * F->p00) + 1 / (F->p11 * F->p11);
        float nearZ = F->nearZ, farZ = maxf(settings.shadow_distance, nearZ * 2);
        float lambda = clampf(settings.cascade_split, 0, 1);
        v3    sc = V(F->sceneSphere[0], F->sceneSphere[1], F->sceneSphere[2]);
        float sr = F->sceneSphere[3];
        float splitNear = nearZ;
        for (uint32_t i = 0; i < cascades; ++i)
        {
            float t = (float)(i + 1) / (float)cascades;
            float splitFar = (nearZ + (farZ - nearZ) * t) * (1 - lambda) + nearZ * powf(farZ / nearZ, t) * lambda;
            // The slice's bounding sphere: its size depends only on the projection, so it never shimmers.
            float c = 0.5f * (splitNear + splitFar) * (1 + k), r;
            if (c >= splitFar)
            {
                c = splitFar;
                r = splitFar * sqrtf(k);
            }
            else
                r = maxf(sqrtf((splitFar - c) * (splitFar - c) + splitFar * splitFar * k),
                         sqrtf((c - splitNear) * (c - splitNear) + splitNear * splitNear * k));
            float half = r * (1 + 4.0f / (float)res);
            float texel = 2 * half / (float)res;
            v3    centre = camPos + camFwd * c;
            float cx = floorf(dot(lr, centre) / texel) * texel;
            float cy = floorf(dot(lu, centre) / texel) * texel;
            float cz = dot(f, centre);
            float zmin = minf(cz - half, dot(f, sc) - sr), zmax = maxf(cz + half, dot(f, sc) + sr);
            float depth = maxf(zmax - zmin, 1e-3f);
            float m[4][4] = { { lr.x / half, lr.y / half, lr.z / half, -cx / half },
                              { lu.x / half, lu.y / half, lu.z / half, -cy / half },
                              { f.x / depth, f.y / depth, f.z / depth, -zmin / depth },
                              { 0, 0, 0, 1 } };
            RenderGpuShadowView& v = views[i];
            setRows(v, m);
            float scale = cascades > 1 ? 0.5f : 1.0f;
            v.rect[0] = (float)(i % 2) * scale;
            v.rect[1] = (float)(i / 2) * scale;
            v.rect[2] = v.rect[3] = scale;
            v.info[0] = texel;
            v.info[1] = depth;
            gCascadeEnd[i] = splitFar;
            splitNear = splitFar;
        }
        for (uint32_t i = 0; i < cascades; ++i)
        {
            // The casters over the cascade (an orthographic view: a caster beside it can't shade it).
            const RenderGpuShadowView& v = views[i];
            uint64_t sig = 0;
            for (uint32_t c = 0; c < F->casterCount; ++c)
            {
                const RenderCaster& k = F->casters[c];
                float x = v.row[0][0] * k.sphere[0] + v.row[0][1] * k.sphere[1] + v.row[0][2] * k.sphere[2] + v.row[0][3];
                float y = v.row[1][0] * k.sphere[0] + v.row[1][1] * k.sphere[1] + v.row[1][2] * k.sphere[2] + v.row[1][3];
                float r = k.sphere[3] * length(V(v.row[0][0], v.row[0][1], v.row[0][2])) + 1e-3f;
                if (fabsf(x) > 1 + r || fabsf(y) > 1 + r)
                    continue;
                uint32_t b[4];
                memcpy(b, k.sphere, sizeof b);
                uint64_t h = mix64(k.moved);
                h = mix64(h ^ ((uint64_t)b[0] << 32 | b[1]));
                h = mix64(h ^ ((uint64_t)b[2] << 32 | b[3]));
                h = mix64(h ^ ((uint64_t)k.firstIndex << 32 | k.indexCount));
                sig += mix64(h ^ (uint32_t)k.vertexOffset);
            }
            if (!(gSunValid & (1u << i)) || sig != gSunSig[i] || memcmp(&v, &gSunDrawn[i], sizeof v) != 0)
            {
                gSunDirty |= 1u << i;
                gSunDrawn[i] = v;
                gSunSig[i] = sig;
            }
        }
        gSunValid = (1u << cascades) - 1;
    }
    else if (!gFrozen)
        gSunValid = 0;

    // --- Local lights: pick, assign tiles, decide which to redraw ------------------------------
    uint32_t wanted = settings.light_shadows ? clampu(settings.shadowed_lights, 0, kGroups) : 0;
    gLightShadowsOn = wanted > 0;
    uint32_t picked[kGroups];
    float    pickedScore[kGroups];
    uint32_t pickedCount = 0;
    v3       camPos = V(F->camPos[0], F->camPos[1], F->camPos[2]);
    for (uint32_t i = 0; i < F->lightCount; ++i)
    {
        RenderLight& l = F->lights[i];
        l.shadow = -1;
        if (!l.castShadow || !wanted || (l.traced >= 0 && !F->keepShadowMaps)) // traced: renderer_rt has them, unless fog or particles read the maps
            continue;
        v3    d = V(l.pos[0], l.pos[1], l.pos[2]) - camPos;
        float dist = maxf(length(d) - l.range, 1.0f);
        float score = l.intensity * l.range * l.range / (dist * dist);
        if (pickedCount < wanted)
        {
            picked[pickedCount] = i;
            pickedScore[pickedCount++] = score;
            continue;
        }
        uint32_t worst = 0;
        for (uint32_t j = 1; j < pickedCount; ++j)
            if (pickedScore[j] < pickedScore[worst])
                worst = j;
        if (score > pickedScore[worst])
        {
            picked[worst] = i;
            pickedScore[worst] = score;
        }
    }
    for (Group& g : gGroups)
        g.used = 0;
    int32_t groupOf[kGroups];
    for (uint32_t j = 0; j < pickedCount; ++j)
    {
        groupOf[j] = -1;
        for (uint32_t g = 0; g < kGroups; ++g)
            if (gGroups[g].seen && gGroups[g].id == F->lights[picked[j]].id && !gGroups[g].used)
            {
                groupOf[j] = (int32_t)g;
                gGroups[g].used = 1;
                break;
            }
    }
    // Lights no longer shadowed give their squares back.
    for (Group& g : gGroups)
        if (!g.used && g.nfaces)
        {
            groupFree(g);
            g.valid = 0;
        }
    for (uint32_t j = 0; j < pickedCount; ++j)
    {
        if (groupOf[j] >= 0)
            continue;
        uint32_t best = kGroups;
        for (uint32_t g = 0; g < kGroups; ++g)
            if (!gGroups[g].used && (best == kGroups || gGroups[g].seen < gGroups[best].seen))
                best = g;
        Group& g = gGroups[best];
        memset(&g, 0, sizeof g);
        g.id = F->lights[picked[j]].id;
        g.used = 1;
        groupOf[j] = (int32_t)best;
    }

    // --- Tile sizes: each face's texels matched to the screen's pixels where it shows ----------------
    // Each face is sized on its own by its footprint on screen (faceWant), so the faces over what the camera sees
    // get the texels and those it can't see keep the smallest square (still drawn: reflections and ray-traced
    // shadows read them). A face's size changes only when the wanted one doubles or halves, so sizes don't pop as
    // lights move; the largest are halved until all fit 90% of the atlas, and only as many lights are moved to
    // new squares as are redrawn this frame, so none loses its shadow meanwhile.
    uint32_t budget = gRedrawAll ? kGroups : clampu(settings.updates_per_frame, 1, kGroups);
    gRedrawAll = false;
    uint32_t target[kGroups][6];
    float    need[kGroups][6]; // cells wanted
    {
        uint32_t maxCells = clampu((int32_t)(pow2Floor(clampu(settings.max_tile_resolution, kCell, 2048)) / kCell), 1, gSide);
        float    focal = F->p11 * (float)F->displayHeight * 0.5f; // the shown size, so tiles keep their size at any render scale
        uint32_t area = 0;
        mx::Plane frustum[6];
        {
            float vp[4][4], proj[4][4] = { { F->p00, 0, 0, 0 }, { 0, F->p11, 0, 0 }, { 0, 0, 0, F->nearZ }, { 0, 0, 1, 0 } };
            float view[4][4] = { { F->view[0][0], F->view[0][1], F->view[0][2], F->view[0][3] },
                                 { F->view[1][0], F->view[1][1], F->view[1][2], F->view[1][3] },
                                 { F->view[2][0], F->view[2][1], F->view[2][2], F->view[2][3] },
                                 { 0, 0, 0, 1 } };
            mul(proj, view, vp);
            clipPlanes(vp, frustum);
            frustum[4] = frustum[5]; // reversed-Z: the sixth is the near plane, the fifth degenerate
        }
        // The faces' needs cost rays against receivers: lights new to a group at once, and four more a frame in turn.
        // Sizes change only when a need doubles or halves, so a few frames' wait never shows.
        bool     gathered = false;
        uint32_t refresh = 4, start = pickedCount ? gNeedCursor % pickedCount : 0;
        for (uint32_t n = 0; n < pickedCount; ++n)
        {
            uint32_t           j = (start + n) % pickedCount;
            Group&             g = gGroups[groupOf[j]];
            if (g.needOk && !refresh)
                continue;
            if (g.needOk)
                refresh--;
            if (!gathered)
                gatherReceivers(F, frustum);
            gathered = true;
            const RenderLight& l = F->lights[picked[j]];
            v3       p = V(l.pos[0], l.pos[1], l.pos[2]);
            uint32_t recv[kReceivers], recvCount = 0;
            for (uint32_t i = 0; i < gRecvCount; ++i)
                if (length(gRecv[i].c - p) < l.range + gRecv[i].r)
                    recv[recvCount++] = i;
            // Past the receivers fitted (gatherReceivers), or too many in reach for the rays, the whole light's size on
            // screen for every face it shows.
            float whole = gRecvAll && recvCount <= 64 ? -1.0f : 2.0f * l.range * focal / maxf(length(p - camPos), l.range);
            for (uint32_t f = 0; f < (l.type == 1 ? 6u : 1u); ++f)
            {
                v3    dir = l.type == 1 ? kFaces[f] : normalize(V(l.dir[0], l.dir[1], l.dir[2]));
                float t = l.type == 1 ? 1.0f : tanf(minf(acosf(clampf(l.cosOuter, -1, 1)) * 1.05f + 0.02f, 1.48f));
                v3    lit = V(l.dir[0], l.dir[1], l.dir[2]);
                bool  oneSided = l.kind == 3 || l.kind == 4;
                float want = faceWant(F, p, dir, t, l.range, focal, recv, whole >= 0.0f ? 0 : recvCount, oneSided ? &lit : nullptr);
                g.need[f] = want < 0.0f ? 0.0f : whole >= 0.0f ? whole : want;
            }
            g.needOk = 1;
        }
        gNeedCursor = start + 4;
        for (uint32_t j = 0; j < pickedCount; ++j)
        {
            const RenderLight& l = F->lights[picked[j]];
            const Group&       g = gGroups[groupOf[j]];
            uint32_t faces = l.type == 1 ? 6 : 1;
            bool shown[6] = {};
            for (uint32_t f = 0; f < faces; ++f)
            {
                float want = g.need[f] * 2.0f / (float)kCell; // two texels a pixel: a filtered edge as sharp as the screen shows it
                need[j][f] = want;
                uint32_t k = want >= 1.0f ? pow2Floor((uint32_t)want) : 1;
                uint32_t held = g.nfaces == faces ? g.k[f] : 0;
                if (held && want > 0.0f && want < held * 2.0f)
                    k = want >= held * 0.5f ? held : k * 2; // shrink with a margin, so it doesn't grow straight back
                // Soft lights are sized as every other: a penumbra is only wide where its blocker is far from the
                // receiver, and sharp at contact however big the light, so a coarser tile would step those edges.
                target[j][f] = k < maxCells ? k : maxCells;
                shown[f] = want > 0.0f;
            }
            // A shown face at least half its largest neighbour (every face but the opposite one), so a shadow's
            // sharpness changes by at most one step across a seam.
            for (uint32_t pass = 0; pass < 2 && faces == 6; ++pass)
                for (uint32_t f = 0; f < 6; ++f)
                    for (uint32_t n = 0; n < 6 && shown[f]; ++n)
                        if (n / 2 != f / 2 && target[j][f] * 2 < target[j][n])
                            target[j][f] = target[j][n] / 2;
            for (uint32_t f = 0; f < faces; ++f)
                area += target[j][f] * target[j][f];
        }
        uint32_t capacity = gSide * gSide * 9 / 10;
        gWanted = area;
        while (area > capacity)
        {
            // The face with the most texels for what it needs gives way, so all end at about the same texels a
            // pixel; one growing before one keeping its size, then the dimmer light's.
            int32_t  pj = -1;
            uint32_t pf = 0;
            float    worst = 0.0f;
            for (uint32_t j = 0; j < pickedCount; ++j)
            {
                const Group& g = gGroups[groupOf[j]];
                uint32_t     faces = F->lights[picked[j]].type == 1 ? 6 : 1;
                for (uint32_t f = 0; f < faces; ++f)
                {
                    uint32_t tk = target[j][f];
                    if (tk <= 1)
                        continue;
                    float over = (float)tk / maxf(need[j][f], 0.01f) * (tk > (g.nfaces == faces ? g.k[f] : 0u) ? 2.0f : 1.0f);
                    if (pj < 0 || over > worst || (over == worst && pickedScore[j] < pickedScore[pj]))
                    {
                        pj = (int32_t)j;
                        pf = f;
                        worst = over;
                    }
                }
            }
            if (pj < 0)
                break;
            uint32_t tk = target[pj][pf];
            area -= tk * tk - tk * tk / 4;
            target[pj][pf] = tk / 2;
        }
    }
    uint32_t moved = 0;
    for (uint32_t pass = 0; pass < 2 && !gFrozen; ++pass)
        for (uint32_t j = 0; j < pickedCount && moved < budget; ++j)
        {
            // Lights without a shadow first, then those changing size.
            Group&   g = gGroups[groupOf[j]];
            uint32_t faces = F->lights[picked[j]].type == 1 ? 6 : 1;
            bool     same = g.nfaces == faces;
            for (uint32_t f = 0; f < faces && same; ++f)
                same = g.k[f] == target[j][f];
            if ((pass == 0) != (g.valid == 0) || same)
                continue;
            if (pass == 1)
                groupFree(g); // its squares can be part of the new ones
            g.valid = 0;
            if (groupAlloc(g, faces, target[j]))
                moved++;
        }
    logAtlas();

    uint64_t latestMove = 0;
    for (uint32_t i = 0; i < F->casterCount; ++i)
        latestMove = F->casters[i].moved > latestMove ? F->casters[i].moved : latestMove;
    uint32_t dirty[kGroups], dirtyCount = 0;
    for (uint32_t j = 0; j < pickedCount; ++j)
    {
        Group&               g = gGroups[groupOf[j]];
        const RenderLight& l = F->lights[picked[j]];
        g.seen = F->frameNumber;
        float key[10];
        lightKey(l, key);
        bool stale = !g.valid || memcmp(key, g.key, sizeof key) != 0 || F->casterRemoved >= g.rendered;
        if (!stale && latestMove >= g.rendered)
        {
            v3 p = V(l.pos[0], l.pos[1], l.pos[2]);
            for (uint32_t i = 0; i < F->casterCount && !stale; ++i)
            {
                const RenderCaster& c = F->casters[i];
                if (c.moved >= g.rendered && length(V(c.sphere[0], c.sphere[1], c.sphere[2]) - p) < l.range + c.sphere[3])
                    stale = true;
            }
        }
        if (stale && !gFrozen && g.nfaces)
            dirty[dirtyCount++] = j;
    }
    // The oldest shadows first (lights with none yet before all), so every moving light takes its
    // turn; a fixed order would redraw the same few every frame and leave the rest behind.
    auto age = [&](uint32_t j) { const Group& g = gGroups[groupOf[j]]; return g.valid ? g.rendered : 0; };
    for (uint32_t a = 1; a < dirtyCount; ++a)
        for (uint32_t b = a; b > 0 && age(dirty[b]) < age(dirty[b - 1]); --b)
        {
            uint32_t t = dirty[b];
            dirty[b] = dirty[b - 1];
            dirty[b - 1] = t;
        }
    gPendingCount = 0;
    for (uint32_t d = 0; d < dirtyCount && gPendingCount < budget; ++d)
    {
        uint32_t             j = dirty[d];
        Group&               g = gGroups[groupOf[j]];
        const RenderLight& l = F->lights[picked[j]];
        buildLocalViews(g, l);
        lightKey(l, g.key);
        g.valid = 1;
        g.rendered = F->frameNumber + 1;
        gPending[gPendingCount++] = (uint32_t)groupOf[j];
    }
    for (uint32_t j = 0; j < pickedCount; ++j)
        if (gGroups[groupOf[j]].valid)
            F->lights[picked[j]].shadow = (int32_t)(kSunViews + groupOf[j] * 6);
    for (uint32_t g = 0; g < kGroups; ++g)
        if (gGroups[g].valid)
            memcpy(&views[kSunViews + g * 6], gGroups[g].views, sizeof gGroups[g].views);

    memcpy(gViewBuf[F->frameIndex]->pCpuMappedAddress, gViewsCpu, sizeof gViewsCpu);

    F->sunAtlas = gSunAtlas;
    F->localAtlas = gLocalAtlas;
    F->shadowViews = gViewBuf[F->frameIndex];
    F->cascadeCount = gCascades;
    memcpy(F->cascadeEnd, gCascadeEnd, sizeof gCascadeEnd);
    F->softness = settings.softness;
    F->lightSize = settings.pcss ? tanf(settings.sun_size * 3.14159265f / 180.0f) : 0.0f;
    F->normalBias = settings.normal_bias;
    F->taps = clampu(settings.filter_taps, 4, 32);
    F->showCascades = (settings.show_cascades ? 1u : 0u) | (settings.show_atlas ? 2u : 0u);
    F->sunTexel = 1.0f / (float)gSunSize;
    F->localTexel = 1.0f / (float)gLocalSize;
    F->localLightSize = settings.contact_hardening ? (settings.light_size > 0 ? settings.light_size : -1.0f) : 0.0f;
    F->receiverPlane = settings.receiver_plane;
    return true;
}

bool light_shadows_on() { return gLightShadowsOn; }

void shadows_render(uint64_t address)
{
    RenderFrame* F = (RenderFrame*)(uintptr_t)address;
    if (!gReady || !F || F->sunAtlas != gSunAtlas)
        return;
    Cmd*      cmd = F->cmd;
    uint32_t  fi = F->frameIndex;
    uint32_t* stream = gStreamCpu;
    gStreamUsed = 0;
    bool geometry = F->positions && F->indices && F->instances && F->casterCount;

    DescriptorData p[4] = {};
    p[0].mIndex = SRT_RES_IDX(ShadowSrt, PerFrame, gInstances);
    p[0].ppBuffers = &F->instances;
    p[1].mIndex = SRT_RES_IDX(ShadowSrt, PerFrame, gViews);
    p[1].ppBuffers = &gViewBuf[fi];
    // The material maps for alpha-tested casters; the texture array only when the core's slots changed (1024 of them).
    const bool maps = gMaskedPipe && F->materialMaps && F->textures;
    uint32_t   bindings = 2;
    if (maps)
    {
        p[2].mIndex = SRT_RES_IDX(ShadowSrt, PerFrame, gMaterialMaps);
        p[2].ppBuffers = (Buffer**)&F->materialMaps;
        bindings = 3;
        if (gTexBound[fi] != F->textureVersion)
        {
            p[3].mIndex = SRT_RES_IDX(ShadowSrt, PerFrame, gTextures);
            p[3].ppTextures = (Texture**)F->textures;
            p[3].mCount = RENDER_MAX_TEXTURES;
            gTexBound[fi] = F->textureVersion;
            bindings = 4;
        }
    }
    if (F->instances)
        updateDescriptorSet(gRenderer, fi, gSet, bindings, p);
    bool anyMasked = false;
    for (uint32_t i = 0; maps && i < F->casterCount && !anyMasked; ++i)
        anyMasked = F->casters[i].masked != 0;
    gMaskedNow = anyMasked;
    // A view's casters: the opaque ones on gPipe, then the alpha-tested ones on gMaskedPipe (left bound as gPipe after).
    auto drawView = [&](uint32_t view, const RenderGpuShadowView& v) {
        drawCasters(cmd, F, view, v, stream, gStreamBuf[fi], false);
        if (!anyMasked)
            return;
        cmdBindPipeline(cmd, gMaskedPipe);
        cmdBindDescriptorSet(cmd, fi, gSet);
        cmdBindIndexBuffer(cmd, F->indices, INDEX_TYPE_UINT32, 0);
        drawCasters(cmd, F, view, v, stream, gStreamBuf[fi], true);
        cmdBindPipeline(cmd, gPipe);
        cmdBindDescriptorSet(cmd, fi, gSet);
        cmdBindIndexBuffer(cmd, F->indices, INDEX_TYPE_UINT32, 0);
    };
    const RenderGpuShadowView* views = gViewsCpu;

    if (gCascades && !gFrozen && gSunDirty)
    {
        cmdBeginDebugMarker(cmd, 1.0f, 0.9f, 0.4f, "Renderer sun shadows");
        RenderTargetBarrier b = { gSunAtlas, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_DEPTH_WRITE };
        cmdResourceBarrier(cmd, 0, nullptr, 0, nullptr, 1, &b);
        // All of them: one clear. Some: the others kept, and each one drawn cleared first.
        bool all = !geometry || gSunDirty == (1u << gCascades) - 1;
        BindRenderTargetsDesc bind = {};
        bind.mDepthStencil = { gSunAtlas, all ? LOAD_ACTION_CLEAR : LOAD_ACTION_LOAD };
        bind.mDepthStencil.mStoreAction = STORE_ACTION_STORE;
        cmdBindRenderTargets(cmd, &bind);
        if (geometry)
        {
            cmdBindPipeline(cmd, gPipe);
            cmdBindDescriptorSet(cmd, fi, gSet);
            cmdBindIndexBuffer(cmd, F->indices, INDEX_TYPE_UINT32, 0);
            for (uint32_t i = 0; i < gCascades; ++i)
            {
                if (!(gSunDirty & (1u << i)))
                    continue;
                const RenderGpuShadowView& v = views[i];
                float x = v.rect[0] * gSunSize, y = v.rect[1] * gSunSize, s = v.rect[2] * gSunSize;
                cmdSetViewport(cmd, x, y, s, s, 0, 1);
                cmdSetScissor(cmd, (uint32_t)x, (uint32_t)y, (uint32_t)s, (uint32_t)s);
                if (!all)
                {
                    cmdBindPipeline(cmd, gClearPipe);
                    cmdBindDescriptorSet(cmd, fi, gSet);
                    cmdDraw(cmd, 3, 0);
                    cmdBindPipeline(cmd, gPipe);
                    cmdBindDescriptorSet(cmd, fi, gSet);
                    cmdBindIndexBuffer(cmd, F->indices, INDEX_TYPE_UINT32, 0);
                }
                drawView(i, v);
            }
        }
        cmdBindRenderTargets(cmd, nullptr);
        b = { gSunAtlas, RESOURCE_STATE_DEPTH_WRITE, RESOURCE_STATE_SHADER_RESOURCE };
        cmdResourceBarrier(cmd, 0, nullptr, 0, nullptr, 1, &b);
        cmdEndDebugMarker(cmd);
    }

    if (gPendingCount || gLocalFresh)
    {
        cmdBeginDebugMarker(cmd, 1.0f, 0.6f, 0.4f, "Renderer local shadows");
        RenderTargetBarrier b = { gLocalAtlas, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_DEPTH_WRITE };
        cmdResourceBarrier(cmd, 0, nullptr, 0, nullptr, 1, &b);
        BindRenderTargetsDesc bind = {};
        bind.mDepthStencil = { gLocalAtlas, gLocalFresh ? LOAD_ACTION_CLEAR : LOAD_ACTION_LOAD };
        gLocalFresh = false;
        bind.mDepthStencil.mStoreAction = STORE_ACTION_STORE;
        cmdBindRenderTargets(cmd, &bind);
        for (uint32_t k = 0; k < gPendingCount; ++k)
        {
            const Group& g = gGroups[gPending[k]];
            for (uint32_t f = 0; f < g.faces; ++f)
            {
                uint32_t idx = gPending[k] * 6 + f;
                float    tile = (float)(g.k[f] * kCell);
                float    x = (float)(g.cx[f] * kCell), y = (float)(g.cy[f] * kCell);
                cmdSetViewport(cmd, x, y, tile, tile, 0, 1);
                cmdSetScissor(cmd, (uint32_t)x, (uint32_t)y, (uint32_t)tile, (uint32_t)tile);
                cmdBindPipeline(cmd, gClearPipe);
                cmdBindDescriptorSet(cmd, fi, gSet);
                cmdDraw(cmd, 3, 0);
                if (geometry)
                {
                    cmdBindPipeline(cmd, gPipe);
                    cmdBindDescriptorSet(cmd, fi, gSet);
                    cmdBindIndexBuffer(cmd, F->indices, INDEX_TYPE_UINT32, 0);
                    drawView(kSunViews + idx, g.views[f]);
                }
            }
        }
        cmdBindRenderTargets(cmd, nullptr);
        b = { gLocalAtlas, RESOURCE_STATE_DEPTH_WRITE, RESOURCE_STATE_SHADER_RESOURCE };
        cmdResourceBarrier(cmd, 0, nullptr, 0, nullptr, 1, &b);
        cmdEndDebugMarker(cmd);
    }
    // The GPU reads the stream when this command list runs, after this copy.
    if (gStreamUsed)
        memcpy(gStreamBuf[fi]->pCpuMappedAddress, gStreamCpu, (size_t)gStreamUsed * 4);
}
} // namespace renderer_shadows
