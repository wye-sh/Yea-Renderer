// Shadows for the renderer core. The sun: up to four stabilised cascades (PSSM splits, a bounding
// sphere per slice so the size never changes as the camera turns, texel-snapped), redrawn every
// frame into one atlas. Spot and point lights: one cached atlas of tiles; a light's tiles are
// redrawn only when it or a caster near it moved, a few lights per frame.
#include "yea_renderer_shadows.h"
#include "frame.h"
#include "mathx.h"

#include <Common_3/Graphics/Interfaces/IGraphics.h>
#include <Common_3/Resources/ResourceLoader/Interfaces/IResourceLoader.h>
#include <Common_3/Graphics/FSL/defaults.h>
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
const uint32_t kGroups = 42;             // shadowed lights at most, each with six view slots
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
    float                 key[9];
    RenderGpuShadowView views[6];
    uint32_t              cells;  // each face's square, in cells per side (0: none held)
    uint32_t              nfaces; // squares held
    uint8_t               cx[6], cy[6];
};

Renderer*      gRenderer;
Queue*         gQueue;
uint32_t       gFrames;
bool           gReady;
Shader *       gShader, *gClearShader;
Pipeline *     gPipe, *gClearPipe;
float          gPipeSlope = -1;
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
uint32_t gPending[kGroups];
uint32_t gPendingCount;
uint32_t gStreamUsed;

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
        cellsMark(g.cx[f], g.cy[f], g.cells, false);
    g.cells = g.nfaces = 0;
}

// Squares for a group's faces, at k cells per side or the largest smaller size that fits.
bool groupAlloc(Group& g, uint32_t faces, uint32_t k)
{
    groupFree(g);
    for (; k >= 1; k /= 2)
    {
        uint32_t f = 0, x, y;
        for (; f < faces && cellsAlloc(k, &x, &y); ++f)
        {
            cellsMark(x, y, k, true);
            g.cx[f] = (uint8_t)x;
            g.cy[f] = (uint8_t)y;
        }
        g.cells = k;
        g.nfaces = f;
        if (f == faces)
            return true;
        groupFree(g);
    }
    return false;
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
    VertexLayout l = {};
    l.mBindingCount = 2;
    l.mBindings[0].mStride = 24;
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

void lightKey(const RenderLight& l, float key[9])
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
}

void buildLocalViews(Group& g, const RenderLight& l)
{
    float tileRes = (float)(g.cells * kCell), atlas = (float)(gSide * kCell);
    v3    p = V(l.pos[0], l.pos[1], l.pos[2]);
    float nearZ = clampf(l.range * 0.005f, 0.01f, 0.2f);
    g.faces = l.type == 1 ? 6 : 1;
    for (uint32_t f = 0; f < g.faces; ++f)
    {
        RenderGpuShadowView& v = g.views[f];
        float                  tanHalf;
        if (l.type == 1)
        {
            tanHalf = 1.05f; // a little past 90 degrees, so filtering at a face's edge stays inside
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

// Draws the casters inside a view's frustum, one instanced draw per run of the same mesh.
void drawCasters(Cmd* cmd, const RenderFrame* F, uint32_t view, const RenderGpuShadowView& v, uint32_t* stream, Buffer* streamBuf)
{
    mx::Plane planes[6];
    clipPlanes(v.row, planes);
    uint32_t runStart = gStreamUsed, runCount = 0;
    const RenderCaster* run = nullptr;
    auto flush = [&]() {
        if (!runCount)
            return;
        Buffer*  bufs[2] = { F->vertices, streamBuf };
        uint32_t strides[2] = { F->vertexStride, 4 };
        uint64_t offsets[2] = { 0, (uint64_t)runStart * 4 };
        cmdBindVertexBuffer(cmd, 2, bufs, strides, offsets);
        cmdDrawIndexedInstanced(cmd, run->indexCount, run->firstIndex, runCount, (uint32_t)run->vertexOffset, 0);
    };
    for (uint32_t i = 0; i < F->casterCount && gStreamUsed < kMaxEntries; ++i)
    {
        const RenderCaster& c = F->casters[i];
        if (!sphereInside(planes, 6, V(c.sphere[0], c.sphere[1], c.sphere[2]), c.sphere[3]))
            continue;
        if (!run || run->firstIndex != c.firstIndex || run->vertexOffset != c.vertexOffset)
        {
            flush();
            run = &c;
            runStart = gStreamUsed;
            runCount = 0;
        }
        stream[gStreamUsed++] = c.instance | (view << RENDER_VIEW_SHIFT);
        runCount++;
    }
    flush();
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
    if (!gReady || (!settings.enabled && !settings.light_shadows) || !F || F->version != RENDER_FRAME_VERSION)
        return false;

    // Atlases and pipelines follow the knobs.
    uint32_t cascades = clampu(settings.cascades, 1, RENDER_MAX_CASCADES);
    uint32_t res = pow2Floor(clampu(settings.sun_resolution, 512, 4096));
    uint32_t sunSize = cascades > 1 ? res * 2 : res;
    uint32_t localSize = pow2Floor(clampu(settings.local_atlas_resolution, 1024, 8192));
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
        }
    }
    gSunRes = res;
    if (!gSunAtlas || !gLocalAtlas || !gPipe)
        return false;

    RenderGpuShadowView* views = gViewsCpu;
    gFrozen = yea::viewMode() != 0;
    if (gWasFrozen && !gFrozen)
        for (Group& g : gGroups)
            g.rendered = 0; // stale: redrawn oldest first, a few each frame
    gWasFrozen = gFrozen;

    // --- Sun cascades -------------------------------------------------------------------------
    if (!gFrozen)
    {
        memset(views, 0, sizeof(RenderGpuShadowView) * kSunViews);
        gCascades = 0;
    }
    if (!gFrozen && settings.enabled && F->sunOn && F->sunShadow)
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
    }

    // --- Local lights: pick, assign tiles, decide which to redraw ------------------------------
    uint32_t wanted = settings.light_shadows ? clampu(settings.shadowed_lights, 0, kGroups) : 0;
    uint32_t picked[kGroups];
    float    pickedScore[kGroups];
    uint32_t pickedCount = 0;
    v3       camPos = V(F->camPos[0], F->camPos[1], F->camPos[2]);
    for (uint32_t i = 0; i < F->lightCount; ++i)
    {
        RenderLight& l = F->lights[i];
        l.shadow = -1;
        if (!l.castShadow || !wanted)
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
        if (!g.used && g.cells)
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

    // --- Tile sizes: texels matched to screen pixels where the light's reach is ------------------
    // A face spans twice the light's range; seen from the camera that is 2 * r pixels across, so it
    // wants about 2 * r texels. Sizes change only when the wanted one doubles or halves, the largest
    // are halved until all fit 90% of the atlas, and only as many
    // lights are moved to new squares as are redrawn this frame, so none loses its shadow meanwhile.
    uint32_t budget = clampu(settings.updates_per_frame, 1, kGroups);
    uint32_t target[kGroups];
    {
        uint32_t maxCells = clampu((int32_t)(pow2Floor(clampu(settings.max_tile_resolution, kCell, 2048)) / kCell), 1, gSide);
        float    focal = F->p11 * (float)F->height * 0.5f;
        uint32_t area = 0;
        for (uint32_t j = 0; j < pickedCount; ++j)
        {
            const RenderLight& l = F->lights[picked[j]];
            const Group&       g = gGroups[groupOf[j]];
            float dist = length(V(l.pos[0], l.pos[1], l.pos[2]) - camPos);
            float want = maxf(2.0f * l.range * focal / maxf(dist, l.range) / (float)kCell, 1.0f);
            uint32_t k = pow2Floor((uint32_t)want);
            if (g.cells && want < g.cells * 2.0f)
                k = want >= g.cells * 0.5f ? g.cells : k * 2; // shrink one step at a time
            target[j] = k < maxCells ? k : maxCells;
            area += (l.type == 1 ? 6 : 1) * target[j] * target[j];
        }
        uint32_t capacity = gSide * gSide * 9 / 10;
        while (area > capacity)
        {
            int32_t pick = -1;
            for (uint32_t j = 0; j < pickedCount; ++j)
            {
                if (target[j] <= 1)
                    continue;
                if (pick < 0 || target[j] > target[pick])
                {
                    pick = (int32_t)j;
                    continue;
                }
                if (target[j] < target[pick])
                    continue;
                // Same size: keep lights at the size they have; then the dimmest gives way.
                bool growJ = target[j] > gGroups[groupOf[j]].cells, growP = target[pick] > gGroups[groupOf[pick]].cells;
                if ((growJ && !growP) || (growJ == growP && pickedScore[j] < pickedScore[pick]))
                    pick = (int32_t)j;
            }
            if (pick < 0)
                break;
            uint32_t faces = F->lights[picked[pick]].type == 1 ? 6 : 1;
            area -= faces * (target[pick] * target[pick] - target[pick] * target[pick] / 4);
            target[pick] /= 2;
        }
    }
    uint32_t moved = 0;
    for (uint32_t pass = 0; pass < 2 && !gFrozen; ++pass)
        for (uint32_t j = 0; j < pickedCount && moved < budget; ++j)
        {
            // Lights without a shadow first, then those changing size.
            Group&   g = gGroups[groupOf[j]];
            uint32_t faces = F->lights[picked[j]].type == 1 ? 6 : 1;
            if ((pass == 0) != (g.valid == 0) || (g.cells == target[j] && g.nfaces == faces))
                continue;
            if (pass == 1)
                groupFree(g); // its squares can be part of the new ones
            g.valid = 0;
            if (groupAlloc(g, faces, target[j]))
                moved++;
        }

    uint64_t latestMove = 0;
    for (uint32_t i = 0; i < F->casterCount; ++i)
        latestMove = F->casters[i].moved > latestMove ? F->casters[i].moved : latestMove;
    uint32_t dirty[kGroups], dirtyCount = 0;
    for (uint32_t j = 0; j < pickedCount; ++j)
    {
        Group&               g = gGroups[groupOf[j]];
        const RenderLight& l = F->lights[picked[j]];
        g.seen = F->frameNumber;
        float key[9];
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
        if (stale && !gFrozen && g.cells)
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
    F->localLightSize = settings.contact_hardening ? settings.light_size : 0.0f;
    F->receiverPlane = settings.receiver_plane;
    return true;
}

void shadows_render(uint64_t address)
{
    RenderFrame* F = (RenderFrame*)(uintptr_t)address;
    if (!gReady || !F || F->sunAtlas != gSunAtlas)
        return;
    Cmd*      cmd = F->cmd;
    uint32_t  fi = F->frameIndex;
    uint32_t* stream = gStreamCpu;
    gStreamUsed = 0;
    bool geometry = F->vertices && F->indices && F->instances && F->casterCount;

    DescriptorData p[2] = {};
    p[0].mIndex = SRT_RES_IDX(ShadowSrt, PerFrame, gInstances);
    p[0].ppBuffers = &F->instances;
    p[1].mIndex = SRT_RES_IDX(ShadowSrt, PerFrame, gViews);
    p[1].ppBuffers = &gViewBuf[fi];
    if (F->instances)
        updateDescriptorSet(gRenderer, fi, gSet, 2, p);
    const RenderGpuShadowView* views = gViewsCpu;

    if (gCascades && !gFrozen)
    {
        cmdBeginDebugMarker(cmd, 1.0f, 0.9f, 0.4f, "Renderer sun shadows");
        RenderTargetBarrier b = { gSunAtlas, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_DEPTH_WRITE };
        cmdResourceBarrier(cmd, 0, nullptr, 0, nullptr, 1, &b);
        BindRenderTargetsDesc bind = {};
        bind.mDepthStencil = { gSunAtlas, LOAD_ACTION_CLEAR };
        bind.mDepthStencil.mStoreAction = STORE_ACTION_STORE;
        cmdBindRenderTargets(cmd, &bind);
        if (geometry)
        {
            cmdBindPipeline(cmd, gPipe);
            cmdBindDescriptorSet(cmd, fi, gSet);
            cmdBindIndexBuffer(cmd, F->indices, INDEX_TYPE_UINT32, 0);
            for (uint32_t i = 0; i < gCascades; ++i)
            {
                const RenderGpuShadowView& v = views[i];
                float x = v.rect[0] * gSunSize, y = v.rect[1] * gSunSize, s = v.rect[2] * gSunSize;
                cmdSetViewport(cmd, x, y, s, s, 0, 1);
                cmdSetScissor(cmd, (uint32_t)x, (uint32_t)y, (uint32_t)s, (uint32_t)s);
                drawCasters(cmd, F, i, v, stream, gStreamBuf[fi]);
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
            float        tile = (float)(g.cells * kCell);
            for (uint32_t f = 0; f < g.faces; ++f)
            {
                uint32_t idx = gPending[k] * 6 + f;
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
                    drawCasters(cmd, F, kSunViews + idx, g.views[f], stream, gStreamBuf[fi]);
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
