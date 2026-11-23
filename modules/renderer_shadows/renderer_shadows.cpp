// Shadows for the renderer core. The sun: up to four stabilised cascades (PSSM splits, a bounding
// sphere per slice so the size never changes as the camera turns, texel-snapped) in one atlas, each
// redrawn only when its view or a caster over it changed. Spot and point lights: one cached atlas of tiles; a light's tiles are
// redrawn only when it or a caster near it moved, a few lights per frame.
#include "yea_renderer_shadows.h"
#include "frame.h"
#include "mathx.h"

#include <yea/device.h>
#include <yea/pipelines.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

#include <Common_3/Graphics/Interfaces/IGraphics.h>
#include <Common_3/Resources/ResourceLoader/Interfaces/IResourceLoader.h>
#include <Common_3/Graphics/FSL/defaults.h>
#define MAX_TEXTURES RENDER_MAX_TEXTURES
#include "Shaders/Shadow.srt.h"
#include "Shaders/Tint.srt.h"

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
const uint32_t kListMax = 64;            // a face's caster list (Group::list)

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
    uint8_t               asked[6];      // the size each face was asked for (k may be smaller: the atlas had no room)
    uint32_t              resizeWait;    // frames a resize has been wanted (sizes change only once it persists)
    uint64_t              faceDrawn[6];  // the frame after each face was last drawn (a whole redraw, or a moving-caster refresh)
    // Each face's casters (by instance row) that can be in it: those in its frustum when it was last drawn whole, and every
    // caster that has moved into it since, so a refresh walks only these (kListMax at most; listCount above it: all).
    uint32_t              list[6][kListMax];
    uint32_t              listCount[6];
    float                 reach; // the farthest its faces reach from the light (their far corners)
    // Each face's dirty rectangle (6.6.5): ndc x0, y0, x1, y1 of where casters that moved since it was drawn were then and
    // are now, while dirtyOn; and a world sphere holding all of them (radius 1e30: the whole face, anywhere).
    uint8_t               dirtyOn[6];
    float                 dirty[6][4];
    float                 dirtySphere[6][4];
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
// A new slope's pipelines (gPipe's, gMaskedPipe's, gClearPipe's) while they compile on Yea's threads: the old ones
// draw till all three are in (none on the first frames: no shadows yet). A failed compile queues all three again,
// kSlopeTries times at most.
const YeaPipelines* gPipes;
float               gNextSlope = -1; // queued, -1 none
uint64_t            gNextJob[3];
Pipeline*           gNext[3];
bool                gNextFailed;
uint32_t            gNextTries;
const uint32_t      kSlopeTries = 3;
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
// Faces of lights not redrawn whole this frame whose view holds a caster that moved since the face was drawn: drawn again
// at the view they have, so a spinning object's own shadow follows its pose every frame instead of the few frames a whole
// redraw takes to come round (6.6.5). Group * 6 + face; every such face that can show, each frame (none held back).
const uint32_t kMaxRefresh = kGroups * 6, kResizeFrames = 30;
uint32_t gRefresh[kMaxRefresh];
float    gRefreshRect[kMaxRefresh][4]; // the part of each that is drawn again: its dirty rectangle (ndc)
uint32_t gRefreshCount;
// Every caster's sphere last frame, by instance row (gPrevFrame: that frame; 0 none), for the dirty rectangles: where a
// moving caster was. Grown, never shrunk; gMovedNow lists the casters that moved since.
struct PrevCaster
{
    float    sphere[4];
    uint64_t frame;
    uint32_t caster; // its index in that frame's casters
};
PrevCaster* gPrev;
uint32_t*   gMovedNow;
uint32_t    gPrevCap, gPrevCount, gMovedCap;
uint64_t    gPrevFrame;
// drawCasters culls to these instead of the view's own planes while gSub (a refresh's dirty rectangle).
bool      gSub;
mx::Plane gSubPlanes[6];
// drawCasters walks only these casters (indices in the frame's, ascending) while gOnly: a refresh's face list.
bool     gOnly;
uint32_t gOnlyIdx[kListMax], gOnlyCount;
// The counters logged with the faces (logAtlas): since its last line.
struct ShadowStats
{
    uint32_t frames, resized, placed, short_, redrawn, refreshed, listed, walked;
    uint32_t lagSum, lagCount, lagMax; // frames a face holding a moving caster went without a draw, at its draw
    double   area;                     // the share of their squares those refreshes drew again, summed
} gStats;
uint32_t gStreamUsed;

// --- Glass in the shadow maps (frame v26 tintDepth, tintColor, shadowTints) --------------------------------------------
// A deep shadow map of the glass over each shadow view that sees any: layer k holds the k-th glass surface out from the
// light (depth peeling), the last layer every surface behind it too (depth min, what they let through multiplied, how many
// added up). A view's layers are squares of the tint maps over just its glass's footprint, at the view's own texel size or
// coarser; they are drawn when the view is, or when its glass changes. Its last layer's surfaces are counted (an occlusion
// query of the texels holding more than one): where any is, the view gets one layer more, up to RENDER_MAX_TINT_LAYERS, so
// stacked panes stay exact and a view with one pane on each light ray pays for two.
const uint32_t kTintCells = 64;     // the tint maps' cells per side
const uint32_t kTintStart = 2;      // layers a view starts with
const uint32_t kMaxTintPasses = 1024;
const uint32_t kTintCounts = 32;    // views counted a frame
const uint32_t kTintPeel = 1, kTintLocal = 2, kTintRest = 4; // TintPass info.y (Shaders/ShadowTypes.h.fsl TINT_PEEL, TINT_LOCAL)

struct TintPassCpu
{
    float    light[4];
    uint32_t info[4];
    uint32_t info2[4];
    float    extra[4];
};

struct TintView
{
    uint32_t layers;  // squares held (0: none), each k x k cells in the tint maps, plus one in the scratch
    uint32_t want;    // layers it needs: kTintStart, more as its counts find them
    uint32_t k;
    uint8_t  cx[RENDER_MAX_TINT_LAYERS], cy[RENDER_MAX_TINT_LAYERS], sx, sy;
    uint64_t sig;     // the glass over it and its look when last drawn
    float    area[4]; // as drawn (RenderGpuShadowTint area)
    uint32_t drawn;   // its squares hold what area says
    uint32_t serial;  // changes with its squares, so a count is taken for the layers it measured
    uint32_t wantK;   // the square side it last asked for (cells), which it may have got smaller
};

TintView              gTint[kMaxViews];
RenderGpuShadowTint gTintCpu[kMaxViews];
uint64_t              gTintOcc[kTintCells], gScratchOcc[kTintCells];
uint32_t              gTintSize, gTintCell;
RenderTarget *        gTintDepth, *gTintColor, *gScratchZ, *gScratchColor, *gScratchDepth;
Shader *              gTintShader, *gTintRestShader, *gTintCopyShader;
Pipeline *            gTintPipe, *gTintLastPipe, *gTintCopyPipe, *gTintCountPipe;
DescriptorSet*        gTintSet;
Buffer *              gTintBuf[kMaxFrames], *gPassBuf[kMaxFrames], *gTriBuf;
QueryPool*            gCountPool[kMaxFrames];
uint32_t              gCountView[kMaxFrames][kTintCounts], gCountSerial[kMaxFrames][kTintCounts], gCountUsed[kMaxFrames];
uint32_t              gTintTexBound[kMaxFrames];
TintPassCpu           gPassCpu[kMaxTintPasses];
uint32_t              gPassCount;
struct TintDraw
{
    uint32_t view, pass, counted; // its first pass (layer 0; then a layer each, then the two counts)
};
TintDraw gTintDraw[kMaxViews];
uint32_t gTintDrawCount, gTintCountCount; // views drawn this frame, and of them counted
bool     gTintOn;                         // this frame's frame has the tint maps
uint32_t gTintRefused;                    // views left without their tint (the tint maps full)
uint32_t gTintShrink, gTintRoomy;         // squares capped at (kTintCells / 2) >> shrink cells; frames the maps sat roomy
uint64_t gNoGlassSince;                   // frame glass last cast a shadow
uint32_t gTintSerial;
uint32_t gTintTexVersion; // the frame's textureVersion the views' tint was drawn with

uint64_t mix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

uint32_t clampu(int32_t v, uint32_t lo, uint32_t hi) { return v < (int32_t)lo ? lo : ((uint32_t)v > hi ? hi : (uint32_t)v); }

uint64_t rowMask(uint32_t x, uint32_t k) { return (k >= 64 ? ~0ull : ((1ull << k) - 1)) << x; }

bool cellsFree(uint32_t x, uint32_t y, uint32_t k, const uint64_t* occ = gOcc)
{
    for (uint32_t r = y; r < y + k; ++r)
        if (occ[r] & rowMask(x, k))
            return false;
    return true;
}

void cellsMark(uint32_t x, uint32_t y, uint32_t k, bool on, uint64_t* occ = gOcc)
{
    for (uint32_t r = y; r < y + k; ++r)
        occ[r] = on ? occ[r] | rowMask(x, k) : occ[r] & ~rowMask(x, k);
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
bool cellsAlloc(uint32_t k, uint32_t* x, uint32_t* y, uint64_t* occ = gOcc, uint32_t side = gSide)
{
    uint32_t n = side / k;
    for (uint32_t i = 0; i < n * n; ++i)
    {
        uint32_t bx = compactBits(i) * k, by = compactBits(i >> 1) * k;
        if (cellsFree(bx, by, k, occ))
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
        g.asked[f] = (uint8_t)k[f];
        gStats.short_ += c < k[f] ? 1 : 0;
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

// A slope's pipelines into out (the casters', the alpha-tested casters', the clear's): each queued on Yea's compile
// threads into jobs (when it has them), else made now.
void makePipelines(float slope, Pipeline* out[3], uint64_t jobs[3])
{
    auto make = [&](PipelineDesc& d, uint32_t i) {
        if (!gPipes || !(jobs[i] = gPipes->add(&d)))
            addPipeline(gRenderer, &d, &out[i]);
    };
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
    make(desc, 0);
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
        make(desc, 1);
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
    make(desc, 2);
}

// The next slope's pipelines dropped: queued compiles cancelled, made ones (never bound) removed.
void dropNext()
{
    for (uint32_t i = 0; i < 3; ++i)
    {
        if (gNextJob[i])
            gPipes->cancel(gNextJob[i]);
        if (gNext[i])
            removePipeline(gRenderer, gNext[i]);
        gNextJob[i] = 0;
        gNext[i] = nullptr;
    }
    gNextFailed = false;
}

// Each frame: a new slope's pipelines queued (made here without Yea's compile threads), and swapped in once all three
// are made; true then (the maps redraw with it). Never waits for a compile.
bool slopePipelines(float slope)
{
    if (slope != gPipeSlope && slope != gNextSlope)
    {
        dropNext();
        gNextSlope = slope;
        gNextTries = 1;
        makePipelines(slope, gNext, gNextJob);
    }
    if (gNextSlope < 0 || gNextTries > kSlopeTries) // none queued, or given up on this slope
        return false;
    bool waiting = false;
    for (uint32_t i = 0; i < 3; ++i)
        if (gNextJob[i])
        {
            void*     out = nullptr;
            const int r = gPipes->ready(gNextJob[i], &out);
            if (r == 0)
            {
                waiting = true;
                continue;
            }
            gNextJob[i] = 0;
            gNext[i] = (Pipeline*)out;
            gNextFailed = gNextFailed || r < 0;
        }
    if (waiting)
        return false;
    if (gNextFailed || !gNext[0] || !gNext[2] || (gMaskedShader && !gNext[1]))
    {
        char line[200];
        snprintf(line, sizeof line, "renderer_shadows: the GPU refused a shadow pipeline (%u of %u tries)%s", gNextTries, kSlopeTries,
                 gNextTries < kSlopeTries ? ": queued again" : gPipe ? ": the last slope's stay" : ": shadows are off");
        yea::log(line);
        dropNext();
        if (++gNextTries <= kSlopeTries)
            makePipelines(gNextSlope, gNext, gNextJob);
        return false;
    }
    if (gPipe || gClearPipe || gMaskedPipe)
        waitQueueIdle(gQueue); // the old ones, still in flight (a slope change is a settings change)
    if (gPipe)
        removePipeline(gRenderer, gPipe);
    if (gMaskedPipe)
        removePipeline(gRenderer, gMaskedPipe);
    if (gClearPipe)
        removePipeline(gRenderer, gClearPipe);
    gPipe = gNext[0], gMaskedPipe = gNext[1], gClearPipe = gNext[2];
    gNext[0] = gNext[1] = gNext[2] = nullptr;
    gPipeSlope = gNextSlope;
    gNextSlope = -1;
    return true;
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
uint32_t gLogGlass[RENDER_MAX_TINT_LAYERS + 2]; // views of glass at each layer count, then those refused
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
    // Glass: how many views hold it at each layer count (the counts grow them), and how full the tint maps are.
    uint32_t glass[RENDER_MAX_TINT_LAYERS + 2] = {}, cells = 0, views = 0;
    for (uint32_t v = 0; v < kMaxViews; ++v)
        if (gTintCpu[v].count)
        {
            glass[gTintCpu[v].count]++;
            cells += gTint[v].layers * gTint[v].k * gTint[v].k;
            views++;
        }
    glass[RENDER_MAX_TINT_LAYERS + 1] = gTintRefused;
    double now = secondsNow();
    gStats.frames++;
    if (now < gLogTime + 20.0 || (!used && !views))
        return;
    memcpy(gLogSizes, sizes, sizeof sizes);
    memcpy(gLogGlass, glass, sizeof glass);
    char line[640];
    int  n = snprintf(line, sizeof line, "renderer_shadows: light faces at 128/256/512/1024/2048: %u/%u/%u/%u/%u; local atlas %u%% full (%u), %u%% wanted",
                      sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], gSide ? used * 100 / (gSide * gSide) : 0, gSide * kCell,
                      gSide ? gWanted * 100 / (gSide * gSide) : 0);
    if (views || gTintRefused)
    {
        n += snprintf(line + n, sizeof line - n, "; glass over %u views, at 1-%u layers:", views, RENDER_MAX_TINT_LAYERS);
        for (uint32_t i = 1; i <= RENDER_MAX_TINT_LAYERS; ++i)
            n += snprintf(line + n, sizeof line - n, "%s%u", i > 1 ? "/" : " ", glass[i]);
        n += snprintf(line + n, sizeof line - n, ", tint maps %u%% full (%u)", cells * 100 / (kTintCells * kTintCells), gTintSize);
        if (gTintRefused)
            n += snprintf(line + n, sizeof line - n, ", %u views without (no room)", gTintRefused);
    }
    // Per second since the last line: faces resized and newly placed, faces given a smaller square than asked, lights
    // redrawn whole, faces refreshed for moving casters, and how many frames old the map of a face over a moving caster was, each frame (average/max; 1 is drawn every frame); the
    // casters each refresh walked from its face's list, and refreshes that walked every caster (a list too long).
    const double secs = gLogTime > 0 ? now - gLogTime : 20.0;
    const ShadowStats& t = gStats;
    snprintf(line + n, sizeof line - n, "; per s: %.1f resized, %.1f placed, %.1f short, %.0f lights redrawn, %.0f moving-caster faces (%.1f casters walked each, %.1f walked all), their age %.1f/%u frames, %.0f%% of their area drawn",
             t.resized / secs, t.placed / secs, t.short_ / secs, t.redrawn / secs, t.refreshed / secs,
             t.refreshed > t.walked ? (double)t.listed / (t.refreshed - t.walked) : 0.0, t.walked / secs, t.lagCount ? (double)t.lagSum / t.lagCount : 0.0, t.lagMax,
             t.refreshed ? t.area * 100.0 / t.refreshed : 0.0);
    gStats = ShadowStats{};
    gLogTime = now;
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
    g.reach = 0;
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
        g.reach = maxf(g.reach, l.range * sqrtf(1 + 2 * tanHalf * tanHalf));
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
// While gOnly, only the casters gOnlyIdx lists (a refresh's face list, ascending, so the runs stay whole).
// The stream is bound once, each draw reading its entries from its first instance on (an instance-rate attribute: both
// APIs add the first instance to it, unlike SV_InstanceID).
void drawCasters(Cmd* cmd, const RenderFrame* F, uint32_t view, const RenderGpuShadowView& v, uint32_t* stream, Buffer* streamBuf,
                 bool masked)
{
    // A refresh culls to its rectangle's planes and the face's own: a large sphere near the light can pass the first alone
    // while wholly outside the face (side planes meet at the light), and would draw nothing.
    mx::Plane planes[12];
    clipPlanes(v.row, planes);
    const uint32_t planeCount = gSub ? 12 : 6;
    if (gSub)
        memcpy(planes + 6, gSubPlanes, sizeof gSubPlanes);
    const bool     sun = view < kSunViews;
    const uint32_t count = gOnly ? gOnlyCount : F->casterCount;
    auto           at = [&](uint32_t k) -> const RenderCaster& { return F->casters[gOnly ? gOnlyIdx[k] : k]; };
    if (masked)
    {
        Buffer*  bufs[3] = { F->positions, F->attributes, streamBuf };
        uint32_t strides[3] = { RENDER_POSITION_STRIDE, RENDER_ATTRIBUTE_STRIDE, 4 };
        uint64_t offsets[3] = { 0, 0, 0 };
        cmdBindVertexBuffer(cmd, 3, bufs, strides, offsets);
    }
    else
    {
        Buffer*  bufs[2] = { F->positions, streamBuf };
        uint32_t strides[2] = { RENDER_POSITION_STRIDE, 4 };
        uint64_t offsets[2] = { 0, 0 };
        cmdBindVertexBuffer(cmd, 2, bufs, strides, offsets);
    }
    // Without the masked pipeline (its shaders refused), alpha-tested casters draw whole with the others. Skinned casters
    // (dynamic) draw after the runs, each on its own.
    auto mine = [&](const RenderCaster& c) { return gMaskedNow ? (c.masked != 0) == masked : !masked; };
    auto inside = [&](const RenderCaster& c) { return sphereInside(planes, planeCount, V(c.sphere[0], c.sphere[1], c.sphere[2]), c.sphere[3]); };
    // Each caster of a run culled once: the version it takes here, or none (kNone: not this pass's, or outside).
    const uint32_t kNone = 0xFFFFFFFFu;
    uint32_t       version[256];
    for (uint32_t i = 0; i < count && gStreamUsed < kMaxEntries;)
    {
        // The run of casters the screen draws with one version (the core sorts them by it), at most 256 at a time.
        const uint32_t mesh = at(i).mesh;
        uint32_t       end = i + 1;
        while (end < count && end - i < 256 && at(end).mesh == mesh)
            ++end;
        for (uint32_t k = i; k < end; ++k)
        {
            const RenderCaster& c = at(k);
            version[k - i] = !c.dynamic && mine(c) && inside(c) ? shadowVersion(c, sun, v) : kNone;
        }
        // Each version this run takes, finest first: a pass finds the finest not drawn yet, then draws the casters taking it.
        // Versions follow their finer ones in the Mesh table, so "not drawn yet" is a row past the last drawn.
        uint32_t floor = mesh;
        for (;;)
        {
            uint32_t pick = kNone;
            for (uint32_t k = i; k < end; ++k)
                if (version[k - i] >= floor && version[k - i] < pick)
                    pick = version[k - i];
            if (pick == kNone)
                break;
            const uint32_t first = gStreamUsed;
            for (uint32_t k = i; k < end && gStreamUsed < kMaxEntries; ++k)
                if (version[k - i] == pick)
                    stream[gStreamUsed++] = at(k).instance | (view << RENDER_VIEW_SHIFT);
            const renderer::Mesh& m = renderer::Meshes[pick];
            if (gStreamUsed > first)
                cmdDrawIndexedInstanced(cmd, m.indexCount, m.firstIndex, gStreamUsed - first, m.firstVertex, first);
            floor = pick + 1;
        }
        i = end;
    }
    // Skinned casters: only the version the core posed exists, at its own vertex offset.
    for (uint32_t k = 0; k < count && gStreamUsed < kMaxEntries; ++k)
    {
        const RenderCaster& c = at(k);
        if (!c.dynamic || !mine(c) || !inside(c))
            continue;
        const uint32_t first = gStreamUsed;
        stream[gStreamUsed++] = c.instance | (view << RENDER_VIEW_SHIFT);
        cmdDrawIndexedInstanced(cmd, c.indexCount, c.firstIndex, 1, (uint32_t)c.vertexOffset, first);
    }
}
// --- Glass tint (frame v26) ------------------------------------------------------------------------------------------

uint64_t hashBytes(uint64_t h, const void* p, size_t n)
{
    const uint8_t* b = (const uint8_t*)p;
    for (size_t i = 0; i < n; ++i)
        h = (h ^ b[i]) * 0x100000001B3ull;
    return h;
}

uint32_t pow2Ceil(uint32_t v)
{
    uint32_t p = 1;
    while (p < v)
        p *= 2;
    return p;
}

// A see-through caster's shape, place and look: its views' tint is redrawn when any changes.
uint64_t glassLook(const RenderFrame* F, const RenderCaster& c)
{
    uint64_t h = mix64(c.moved ^ ((uint64_t)c.generation << 32));
    h = hashBytes(h, c.sphere, sizeof c.sphere);
    h = mix64(h ^ ((uint64_t)c.firstIndex << 32 | c.indexCount) ^ ((uint64_t)(uint32_t)c.vertexOffset << 7) ^ c.twoSided);
    uint32_t m = F->instanceRows[c.instance].material;
    if (m < renderer::Materials.size())
        h = hashBytes(h, &renderer::Materials[m], sizeof(renderer::Material));
    if (m < renderer::MaterialBlends.size())
        h = hashBytes(h, &renderer::MaterialBlends[m], sizeof(renderer::MaterialBlend));
    if (m < renderer::MaterialRefractions.size())
        h = hashBytes(h, &renderer::MaterialRefractions[m], sizeof(renderer::MaterialRefraction));
    if (m < renderer::MaterialMaps.size())
        h = hashBytes(h, &renderer::MaterialMaps[m], sizeof(renderer::MaterialMap));
    return mix64(h);
}

// Whether a caster can tint a view (inside its frustum), and where in the view's ndc, conservatively: lo, hi (x, y).
bool glassOver(const RenderGpuShadowView& v, const mx::Plane* planes, bool persp, const RenderCaster& c, float lo[2], float hi[2])
{
    const float* s = c.sphere;
    if (!sphereInside(planes, 6, V(s[0], s[1], s[2]), s[3]))
        return false;
    float x = v.row[0][0] * s[0] + v.row[0][1] * s[1] + v.row[0][2] * s[2] + v.row[0][3];
    float y = v.row[1][0] * s[0] + v.row[1][1] * s[1] + v.row[1][2] * s[2] + v.row[1][3];
    float rx = s[3] * length(V(v.row[0][0], v.row[0][1], v.row[0][2]));
    float ry = s[3] * length(V(v.row[1][0], v.row[1][1], v.row[1][2]));
    lo[0] = x - rx, hi[0] = x + rx, lo[1] = y - ry, hi[1] = y + ry;
    if (persp)
    {
        // w: the distance along the face's axis (row 3, a unit vector); across the light's near plane, the whole face.
        float w = v.row[3][0] * s[0] + v.row[3][1] * s[1] + v.row[3][2] * s[2] + v.row[3][3];
        float n = w - s[3], f = w + s[3];
        if (n <= 1e-3f)
            lo[0] = lo[1] = -1.0f, hi[0] = hi[1] = 1.0f;
        else
        {
            lo[0] = minf(lo[0] / n, lo[0] / f), hi[0] = maxf(hi[0] / n, hi[0] / f);
            lo[1] = minf(lo[1] / n, lo[1] / f), hi[1] = maxf(hi[1] / n, hi[1] / f);
        }
    }
    lo[0] = maxf(lo[0], -1.0f), lo[1] = maxf(lo[1], -1.0f), hi[0] = minf(hi[0], 1.0f), hi[1] = minf(hi[1], 1.0f);
    return lo[0] < hi[0] && lo[1] < hi[1];
}

RenderTarget* makeTintTarget(uint32_t size, TinyImageFormat format, const char* name, float clear, float clearAlpha, ResourceState start)
{
    RenderTargetDesc d = {};
    d.mWidth = d.mHeight = size;
    d.mDepth = d.mArraySize = d.mMipLevels = 1;
    d.mSampleCount = SAMPLE_COUNT_1;
    d.mFormat = format;
    d.mStartState = start;
    d.mDescriptors = DESCRIPTOR_TYPE_TEXTURE;
    d.mClearValue = {};
    if (format == TinyImageFormat_D32_SFLOAT)
        d.mClearValue.depth = clear;
    else
    {
        d.mClearValue.r = d.mClearValue.g = d.mClearValue.b = clear;
        d.mClearValue.a = clearAlpha;
    }
    d.pName = name;
    RenderTarget* rt = nullptr;
    addRenderTarget(gRenderer, &d, &rt);
    return rt;
}

void freeTintTargets()
{
    freeTarget(gTintDepth);
    freeTarget(gTintColor);
    freeTarget(gScratchZ);
    freeTarget(gScratchColor);
    freeTarget(gScratchDepth);
    gTintSize = 0;
    memset(gTint, 0, sizeof gTint);
    memset(gTintCpu, 0, sizeof gTintCpu);
    memset(gTintOcc, 0, sizeof gTintOcc);
    memset(gScratchOcc, 0, sizeof gScratchOcc);
    gTintShrink = gTintRoomy = 0;
}

// The tint maps (depth R32F, colour RGB10A2: white, nothing in front) and the scratch a layer is drawn into (depth R32F,
// colour RGBA16F: alpha the surfaces it holds; a depth buffer), all size x size.
bool tintTargets(uint32_t size)
{
    if (gTintSize == size && gTintDepth && gTintColor && gScratchZ && gScratchColor && gScratchDepth)
        return true;
    if (gTintSize)
        waitQueueIdle(gQueue);
    freeTintTargets();
    gTintDepth = makeTintTarget(size, TinyImageFormat_R32_SFLOAT, "renderer_tint_depth", 1.0f, 1.0f, RESOURCE_STATE_SHADER_RESOURCE);
    gTintColor = makeTintTarget(size, TinyImageFormat_R10G10B10A2_UNORM, "renderer_tint_color", 1.0f, 1.0f, RESOURCE_STATE_SHADER_RESOURCE);
    gScratchZ = makeTintTarget(size, TinyImageFormat_R32_SFLOAT, "renderer_tint_scratch_depth", 1.0f, 1.0f, RESOURCE_STATE_SHADER_RESOURCE);
    gScratchColor = makeTintTarget(size, TinyImageFormat_R16G16B16A16_SFLOAT, "renderer_tint_scratch_color", 1.0f, 0.0f, RESOURCE_STATE_SHADER_RESOURCE);
    gScratchDepth = makeTintTarget(size, TinyImageFormat_D32_SFLOAT, "renderer_tint_scratch_z", 1.0f, 0.0f, RESOURCE_STATE_DEPTH_WRITE);
    if (!gTintDepth || !gTintColor || !gScratchZ || !gScratchColor || !gScratchDepth)
    {
        freeTintTargets();
        return false;
    }
    gTintSize = size;
    gTintCell = size / kTintCells;
    return true;
}

void tintFree(TintView& t)
{
    for (uint32_t l = 0; l < t.layers; ++l)
        cellsMark(t.cx[l], t.cy[l], t.k, false, gTintOcc);
    if (t.layers)
        cellsMark(t.sx, t.sy, t.k, false, gScratchOcc);
    t.layers = 0;
    t.drawn = 0;
}

// layers squares of k x k cells in the tint maps and one in the scratch, or none.
bool tintAlloc(TintView& t, uint32_t k, uint32_t layers)
{
    uint32_t x, y, l = 0;
    for (; l < layers && cellsAlloc(k, &x, &y, gTintOcc, kTintCells); ++l)
    {
        cellsMark(x, y, k, true, gTintOcc);
        t.cx[l] = (uint8_t)x, t.cy[l] = (uint8_t)y;
    }
    bool ok = l == layers && cellsAlloc(k, &x, &y, gScratchOcc, kTintCells);
    if (!ok)
    {
        while (l-- > 0)
            cellsMark(t.cx[l], t.cy[l], k, false, gTintOcc);
        return false;
    }
    cellsMark(x, y, k, true, gScratchOcc);
    t.sx = (uint8_t)x, t.sy = (uint8_t)y;
    t.k = k;
    t.layers = layers;
    t.drawn = 0;
    t.serial = ++gTintSerial;
    return true;
}

uint32_t tintOrigin(uint32_t cx, uint32_t cy) { return (cx * gTintCell) | ((cy * gTintCell) << 16); }

// After the views are final (shadows_prepare): each view's glass, its squares, and which views' tint is drawn this frame.
void prepareTints(RenderFrame* F)
{
    const uint32_t fi = F->frameIndex;
    gTintDrawCount = gTintCountCount = gPassCount = 0;
    gTintOn = false;
    // Last time this frame's slot ran, its counts: a view whose last layer held more than one surface somewhere gets
    // another layer; one whose last layer held nothing gives one back (never under kTintStart).
    for (uint32_t i = 0; i < gCountUsed[fi]; ++i)
    {
        TintView& t = gTint[gCountView[fi][i]];
        if (t.serial != gCountSerial[fi][i] || !t.want)
            continue;
        QueryData more = {}, any = {};
        getQueryData(gRenderer, gCountPool[fi], 2 * i, &more);
        getQueryData(gRenderer, gCountPool[fi], 2 * i + 1, &any);
        if (more.mOcclusionCounts > 0 && t.want < RENDER_MAX_TINT_LAYERS)
            t.want++;
        else if (any.mOcclusionCounts == 0 && t.want > kTintStart)
            t.want--;
    }
    gCountUsed[fi] = 0;

    bool glass = false;
    for (uint32_t i = 0; i < F->seeThroughCount && !glass; ++i)
        glass = F->seeThrough[i].castShadow != 0;
    const bool on = settings.glass_shadows && glass && gTintPipe && F->instanceRows && F->materials && F->materialMaps && F->refraction &&
                    F->dfg && F->textures && F->positions && F->attributes && F->indices && F->instances;
    if (on)
        gNoGlassSince = F->frameNumber;
    if (!on)
    {
        for (uint32_t v = 0; v < kMaxViews; ++v)
        {
            if (gTint[v].layers)
                tintFree(gTint[v]);
            gTint[v].want = 0;
            gTintCpu[v].count = 0;
        }
        // The maps go once glass has cast no shadow for 10 s (a pane hidden a moment keeps them).
        if (gTintSize && F->frameNumber > gNoGlassSince + 600)
        {
            waitQueueIdle(gQueue);
            freeTintTargets();
        }
        gTintRefused = 0;
        return;
    }
    if (!gFrozen && tintTargets(pow2Floor(clampu(settings.glass_resolution, 1024, 4096))))
    {
        bool groupDrawn[kGroups] = {};
        for (uint32_t k = 0; k < gPendingCount; ++k)
            groupDrawn[gPending[k]] = true;
        const bool textures = F->textureVersion != gTintTexVersion;
        gTintTexVersion = F->textureVersion;
        uint32_t refused = 0;
        v3       sunRay = -normalize(V(F->sunDir[0], F->sunDir[1], F->sunDir[2]));
        for (uint32_t view = 0; view < kMaxViews; ++view)
        {
            const bool sun = view < kSunViews;
            bool       active, redraw;
            if (sun)
            {
                active = view < gCascades && (gSunValid & (1u << view));
                redraw = (gSunDirty >> view) & 1u;
            }
            else
            {
                const Group& g = gGroups[(view - kSunViews) / 6];
                active = g.valid && g.nfaces && (view - kSunViews) % 6 < g.faces;
                redraw = groupDrawn[(view - kSunViews) / 6];
            }
            TintView& t = gTint[view];
            // The glass over the view: its footprint, and a sum of hashes of it (order-free).
            const RenderGpuShadowView& v = gViewsCpu[view];
            float                        lo[2] = { 2, 2 }, hi[2] = { -2, -2 };
            uint64_t                     sig = 0;
            bool                         any = false;
            if (active)
            {
                mx::Plane planes[6];
                clipPlanes(v.row, planes);
                for (uint32_t i = 0; i < F->seeThroughCount; ++i)
                {
                    const RenderCaster& c = F->seeThrough[i];
                    float               a[2], b[2];
                    if (!c.castShadow || !glassOver(v, planes, !sun, c, a, b))
                        continue;
                    any = true;
                    lo[0] = minf(lo[0], a[0]), lo[1] = minf(lo[1], a[1]), hi[0] = maxf(hi[0], b[0]), hi[1] = maxf(hi[1], b[1]);
                    sig += glassLook(F, c);
                }
            }
            if (!any)
            {
                if (t.layers)
                    tintFree(t);
                t.want = 0;
                gTintCpu[view].count = 0;
                continue;
            }
            // The footprint in atlas texels with two of margin, inside the view's tile, at the view's own texel size, or
            // coarser by powers of two when it would take more than half the maps' side (less while they are crowded).
            const float S = sun ? (float)gSunSize : (float)(gSide * kCell);
            const float tx = v.rect[0] * S, ty = v.rect[1] * S, tn = v.rect[2] * S;
            float       ax = maxf(floorf((v.rect[0] + (lo[0] * 0.5f + 0.5f) * v.rect[2]) * S) - 2.0f, tx);
            float       ay = maxf(floorf((v.rect[1] + (0.5f - hi[1] * 0.5f) * v.rect[3]) * S) - 2.0f, ty);
            float       bx = minf(ceilf((v.rect[0] + (hi[0] * 0.5f + 0.5f) * v.rect[2]) * S) + 2.0f, tx + tn);
            float       by = minf(ceilf((v.rect[1] + (0.5f - lo[1] * 0.5f) * v.rect[3]) * S) + 2.0f, ty + tn);
            float       side = maxf(bx - ax, by - ay);
            uint32_t    wantK = 1;
            for (uint32_t m = 0;; ++m)
            {
                wantK = pow2Ceil((uint32_t)ceilf(side / (float)(1u << m) / (float)gTintCell));
                if (wantK <= ((kTintCells / 2) >> gTintShrink))
                    break;
            }
            if (!t.want)
                t.want = kTintStart;
            bool fresh = false;
            if (t.layers != t.want || t.wantK != wantK)
            {
                // Its squares at the size wanted, else smaller (coarser texels), else fewer layers (never under two).
                tintFree(t);
                t.wantK = wantK;
                for (uint32_t layers = t.want; layers >= kTintStart && !t.layers; --layers)
                    for (uint32_t k = wantK; k >= 1 && !tintAlloc(t, k, layers); k /= 2)
                        ;
                fresh = true;
            }
            if (!t.layers)
            {
                refused++;
                gTintCpu[view].count = 0;
                continue;
            }
            uint32_t m = 0;
            while (side / (float)(1u << m) > (float)(t.k * gTintCell))
                ++m;
            const float step = (float)(1u << m);
            float       area[4] = { floorf(ax / step) * step / S, floorf(ay / step) * step / S, S / step, (float)(t.k * gTintCell) };
            bool        changed = fresh || !t.drawn || sig != t.sig || memcmp(area, t.area, sizeof area) != 0 || textures;
            if ((redraw || changed) && gPassCount + t.layers + 3 <= kMaxTintPasses)
            {
                TintDraw& d = gTintDraw[gTintDrawCount++];
                d.view = view;
                d.pass = gPassCount;
                d.counted = gTintCountCount < kTintCounts ? gTintCountCount + 1 : 0;
                if (d.counted)
                {
                    gCountView[fi][gTintCountCount] = view;
                    gCountSerial[fi][gTintCountCount] = t.serial;
                    gTintCountCount++;
                }
                // Passes: each layer, the two counts, then the last layer's rest.
                for (uint32_t l = 0; l < t.layers + 3; ++l)
                {
                    TintPassCpu& p = gPassCpu[gPassCount++];
                    uint32_t     layer = l < t.layers ? l : t.layers - 1; // the counts and the rest are the last layer's
                    memset(&p, 0, sizeof p);
                    if (sun)
                        p.light[0] = sunRay.x, p.light[1] = sunRay.y, p.light[2] = sunRay.z;
                    else
                        p.light[0] = v.info[1], p.light[1] = v.info[2], p.light[2] = v.info[3], p.light[3] = 1.0f;
                    p.info[0] = view;
                    p.info[1] = (layer ? kTintPeel : 0u) | (sun ? 0u : kTintLocal);
                    p.info[2] = layer ? tintOrigin(t.cx[layer - 1], t.cy[layer - 1]) : 0u;
                    p.info[3] = tintOrigin(t.sx, t.sy);
                    p.info2[0] = tintOrigin(t.cx[layer], t.cy[layer]);
                    p.extra[0] = l == t.layers ? 1.5f : l == t.layers + 1 ? 0.5f : 0.0f;
                    if (l == t.layers + 2)
                        p.info[1] |= kTintRest;
                }
                t.sig = sig;
                memcpy(t.area, area, sizeof area);
                t.drawn = 1;
            }
            RenderGpuShadowTint& out = gTintCpu[view];
            memset(&out, 0, sizeof out);
            if (!t.drawn)
                continue;
            memcpy(out.area, t.area, sizeof out.area);
            for (uint32_t l = 0; l < t.layers; ++l)
                out.layer[l] = tintOrigin(t.cx[l], t.cy[l]);
            out.count = t.layers;
        }
        gTintRefused = refused;
        // Crowded (a view found no room): every view's squares are capped a power of two smaller, so the largest go
        // coarser, as the light atlas halves its most over-served face. Roomy (under an eighth full, so four times the
        // area still fits) for 2 s: the cap goes back up a step.
        uint32_t cells = 0;
        for (uint32_t v = 0; v < kMaxViews; ++v)
            cells += gTint[v].layers * gTint[v].k * gTint[v].k;
        if (refused)
        {
            gTintShrink += gTintShrink < 5; // 32 >> 5: one cell
            gTintRoomy = 0;
        }
        else if (gTintShrink && cells * 8 < kTintCells * kTintCells)
        {
            if (++gTintRoomy >= 120)
                gTintShrink--, gTintRoomy = 0;
        }
        else
            gTintRoomy = 0;
    }
    if (!gTintSize)
        return;
    bool shown = false;
    for (uint32_t v = 0; v < kMaxViews && !shown; ++v)
        shown = gTintCpu[v].count != 0;
    if (!shown && !gTintDrawCount)
        return;
    memcpy(gTintBuf[fi]->pCpuMappedAddress, gTintCpu, sizeof gTintCpu);
    if (gPassCount)
        memcpy(gPassBuf[fi]->pCpuMappedAddress, gPassCpu, sizeof(TintPassCpu) * gPassCount);
    gTintOn = true;
    F->tintDepth = gTintDepth;
    F->tintColor = gTintColor;
    F->shadowTints = gTintBuf[fi];
}

// The see-through casters over a view, into the scratch (pass: the layer's TintPass), batched where they share vertices.
void drawGlass(Cmd* cmd, const RenderFrame* F, const RenderGpuShadowView& v, bool persp, uint32_t pass, uint32_t* stream, Buffer* streamBuf)
{
    mx::Plane planes[6];
    clipPlanes(v.row, planes);
    float lo[2], hi[2];
    auto  takes = [&](const RenderCaster& c) { return c.castShadow && glassOver(v, planes, persp, c, lo, hi); };
    for (uint32_t i = 0; i < F->seeThroughCount && gStreamUsed < kMaxEntries;)
    {
        const RenderCaster& c = F->seeThrough[i];
        uint32_t            end = i + 1;
        if (!takes(c))
        {
            i = end;
            continue;
        }
        const uint32_t first = gStreamUsed;
        stream[gStreamUsed++] = c.instance | (pass << RENDER_VIEW_SHIFT) | (c.twoSided ? 0x80000000u : 0u);
        for (; !c.dynamic && end < F->seeThroughCount && gStreamUsed < kMaxEntries; ++end)
        {
            const RenderCaster& n = F->seeThrough[end];
            if (n.dynamic || n.firstIndex != c.firstIndex || n.indexCount != c.indexCount || n.vertexOffset != c.vertexOffset)
                break;
            if (takes(n))
                stream[gStreamUsed++] = n.instance | (pass << RENDER_VIEW_SHIFT) | (n.twoSided ? 0x80000000u : 0u);
        }
        Buffer*  bufs[3] = { F->positions, F->attributes, streamBuf };
        uint32_t strides[3] = { RENDER_POSITION_STRIDE, RENDER_ATTRIBUTE_STRIDE, 4 };
        uint64_t offsets[3] = { 0, 0, (uint64_t)first * 4 };
        cmdBindVertexBuffer(cmd, 3, bufs, strides, offsets);
        cmdDrawIndexedInstanced(cmd, c.indexCount, c.firstIndex, gStreamUsed - first, (uint32_t)c.vertexOffset, 0);
        i = end;
    }
}

// In shadows_render after both atlases: each drawn view's layers, nearest the light first, into the scratch, each then
// copied into its square of the tint maps; the last layer's two counts with it.
void renderTints(Cmd* cmd, const RenderFrame* F, uint32_t fi, uint32_t* stream)
{
    if (gCountPool[fi])
        cmdResetQuery(cmd, gCountPool[fi], 0, 2 * kTintCounts);
    if (!gTintOn || !gTintDrawCount || gFrozen || F->tintDepth != gTintDepth)
        return;
    cmdBeginDebugMarker(cmd, 0.6f, 0.9f, 1.0f, "Renderer glass shadows");
    DescriptorData p[13] = {};
    Texture*       tex[5] = { gTintDepth->pTexture, gScratchZ->pTexture, gScratchColor->pTexture, gSunAtlas->pTexture, gLocalAtlas->pTexture };
    Buffer*        bufs[7] = { F->instances, gViewBuf[fi], gPassBuf[fi], F->materials, F->materialMaps, F->refraction, F->dfg };
    p[0].mIndex = SRT_RES_IDX(TintSrt, PerFrame, gInstances);
    p[1].mIndex = SRT_RES_IDX(TintSrt, PerFrame, gViews);
    p[2].mIndex = SRT_RES_IDX(TintSrt, PerFrame, gPasses);
    p[3].mIndex = SRT_RES_IDX(TintSrt, PerFrame, gMaterials);
    p[4].mIndex = SRT_RES_IDX(TintSrt, PerFrame, gMaterialMaps);
    p[5].mIndex = SRT_RES_IDX(TintSrt, PerFrame, gRefraction);
    p[6].mIndex = SRT_RES_IDX(TintSrt, PerFrame, gDfg);
    for (uint32_t i = 0; i < 7; ++i)
        p[i].ppBuffers = &bufs[i];
    p[7].mIndex = SRT_RES_IDX(TintSrt, PerFrame, gTintPrev);
    p[8].mIndex = SRT_RES_IDX(TintSrt, PerFrame, gScratchZ);
    p[9].mIndex = SRT_RES_IDX(TintSrt, PerFrame, gScratchColor);
    p[10].mIndex = SRT_RES_IDX(TintSrt, PerFrame, gSunAtlas);
    p[11].mIndex = SRT_RES_IDX(TintSrt, PerFrame, gLocalAtlas);
    for (uint32_t i = 0; i < 5; ++i)
        p[7 + i].ppTextures = &tex[i];
    uint32_t bindings = 12;
    if (gTintTexBound[fi] != F->textureVersion)
    {
        p[12].mIndex = SRT_RES_IDX(TintSrt, PerFrame, gTextures);
        p[12].ppTextures = (Texture**)F->textures;
        p[12].mCount = RENDER_MAX_TEXTURES;
        gTintTexBound[fi] = F->textureVersion;
        bindings = 13;
    }
    updateDescriptorSet(gRenderer, fi, gTintSet, bindings, p);

    uint32_t layers = 0;
    for (uint32_t d = 0; d < gTintDrawCount; ++d)
        layers = gTint[gTintDraw[d].view].layers > layers ? gTint[gTintDraw[d].view].layers : layers;
    uint32_t counted = 0;
    for (uint32_t l = 0; l < layers; ++l)
    {
        RenderTargetBarrier b[2] = { { gScratchZ, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_RENDER_TARGET },
                                     { gScratchColor, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_RENDER_TARGET } };
        cmdResourceBarrier(cmd, 0, nullptr, 0, nullptr, 2, b);
        BindRenderTargetsDesc bind = {};
        bind.mRenderTargetCount = 2;
        bind.mRenderTargets[0] = { gScratchZ, LOAD_ACTION_CLEAR };
        bind.mRenderTargets[1] = { gScratchColor, LOAD_ACTION_CLEAR };
        bind.mDepthStencil = { gScratchDepth, LOAD_ACTION_CLEAR };
        bind.mDepthStencil.mStoreAction = STORE_ACTION_DONTCARE;
        cmdBindRenderTargets(cmd, &bind);
        for (uint32_t d = 0; d < gTintDrawCount; ++d)
        {
            const TintView& t = gTint[gTintDraw[d].view];
            if (l >= t.layers)
                continue;
            const RenderGpuShadowView& v = gViewsCpu[gTintDraw[d].view];
            cmdBindPipeline(cmd, gTintPipe);
            cmdBindDescriptorSet(cmd, fi, gTintSet);
            cmdBindIndexBuffer(cmd, F->indices, INDEX_TYPE_UINT32, 0);
            const float sx = (float)(t.sx * gTintCell), sy = (float)(t.sy * gTintCell), side = (float)(t.k * gTintCell);
            cmdSetViewport(cmd, sx + (v.rect[0] - t.area[0]) * t.area[2], sy + (v.rect[1] - t.area[1]) * t.area[2], v.rect[2] * t.area[2],
                           v.rect[3] * t.area[2], 0, 1);
            cmdSetScissor(cmd, (uint32_t)sx, (uint32_t)sy, (uint32_t)side, (uint32_t)side);
            drawGlass(cmd, F, v, gTintDraw[d].view >= kSunViews, gTintDraw[d].pass + l, stream, gStreamBuf[fi]);
        }
        cmdBindRenderTargets(cmd, nullptr);
        {
            // The last layer's rest, for views whose last layer this is: the kept depth is read, the colour blended.
            RenderTargetBarrier r = { gScratchZ, RESOURCE_STATE_RENDER_TARGET, RESOURCE_STATE_SHADER_RESOURCE };
            cmdResourceBarrier(cmd, 0, nullptr, 0, nullptr, 1, &r);
            BindRenderTargetsDesc rest = {};
            rest.mRenderTargetCount = 1;
            rest.mRenderTargets[0] = { gScratchColor, LOAD_ACTION_LOAD };
            bool bound = false;
            for (uint32_t d = 0; d < gTintDrawCount; ++d)
            {
                const TintView& t = gTint[gTintDraw[d].view];
                if (l + 1 != t.layers)
                    continue;
                if (!bound)
                    cmdBindRenderTargets(cmd, &rest);
                bound = true;
                const RenderGpuShadowView& v = gViewsCpu[gTintDraw[d].view];
                cmdBindPipeline(cmd, gTintLastPipe);
                cmdBindDescriptorSet(cmd, fi, gTintSet);
                cmdBindIndexBuffer(cmd, F->indices, INDEX_TYPE_UINT32, 0);
                const float sx = (float)(t.sx * gTintCell), sy = (float)(t.sy * gTintCell), side = (float)(t.k * gTintCell);
                cmdSetViewport(cmd, sx + (v.rect[0] - t.area[0]) * t.area[2], sy + (v.rect[1] - t.area[1]) * t.area[2], v.rect[2] * t.area[2],
                               v.rect[3] * t.area[2], 0, 1);
                cmdSetScissor(cmd, (uint32_t)sx, (uint32_t)sy, (uint32_t)side, (uint32_t)side);
                drawGlass(cmd, F, v, gTintDraw[d].view >= kSunViews, gTintDraw[d].pass + t.layers + 2, stream, gStreamBuf[fi]);
            }
            if (bound)
                cmdBindRenderTargets(cmd, nullptr);
        }
        RenderTargetBarrier c[3] = { { gScratchColor, RESOURCE_STATE_RENDER_TARGET, RESOURCE_STATE_SHADER_RESOURCE },
                                     { gTintDepth, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_RENDER_TARGET },
                                     { gTintColor, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_RENDER_TARGET } };
        cmdResourceBarrier(cmd, 0, nullptr, 0, nullptr, 3, c);
        BindRenderTargetsDesc copy = {};
        copy.mRenderTargetCount = 2;
        copy.mRenderTargets[0] = { gTintDepth, LOAD_ACTION_LOAD };
        copy.mRenderTargets[1] = { gTintColor, LOAD_ACTION_LOAD };
        cmdBindRenderTargets(cmd, &copy);
        auto tri = [&](uint32_t pass) {
            const uint32_t first = gStreamUsed;
            stream[gStreamUsed++] = pass << RENDER_VIEW_SHIFT;
            Buffer*  vb[2] = { gTriBuf, gStreamBuf[fi] };
            uint32_t strides[2] = { 8, 4 };
            uint64_t offsets[2] = { 0, (uint64_t)first * 4 };
            cmdBindVertexBuffer(cmd, 2, vb, strides, offsets);
            cmdDrawInstanced(cmd, 3, 0, 1, 0);
        };
        for (uint32_t d = 0; d < gTintDrawCount && gStreamUsed + 3 <= kMaxEntries; ++d)
        {
            const TintView& t = gTint[gTintDraw[d].view];
            if (l >= t.layers)
                continue;
            const float x = (float)(t.cx[l] * gTintCell), y = (float)(t.cy[l] * gTintCell), side = (float)(t.k * gTintCell);
            cmdSetViewport(cmd, x, y, side, side, 0, 1);
            cmdSetScissor(cmd, (uint32_t)x, (uint32_t)y, (uint32_t)side, (uint32_t)side);
            cmdBindPipeline(cmd, gTintCopyPipe);
            cmdBindDescriptorSet(cmd, fi, gTintSet);
            tri(gTintDraw[d].pass + l);
            if (l + 1 != t.layers || !gTintDraw[d].counted || !gCountPool[fi])
                continue;
            // Its counts: texels whose last layer holds more than one surface, then more than none.
            cmdBindPipeline(cmd, gTintCountPipe);
            cmdBindDescriptorSet(cmd, fi, gTintSet);
            for (uint32_t q = 0; q < 2; ++q)
            {
                QueryDesc qd = { 2 * (gTintDraw[d].counted - 1) + q };
                cmdBeginQuery(cmd, gCountPool[fi], &qd);
                tri(gTintDraw[d].pass + t.layers + q);
                cmdEndQuery(cmd, gCountPool[fi], &qd);
            }
            counted = gTintDraw[d].counted > counted ? gTintDraw[d].counted : counted;
        }
        cmdBindRenderTargets(cmd, nullptr);
        RenderTargetBarrier e[2] = { { gTintDepth, RESOURCE_STATE_RENDER_TARGET, RESOURCE_STATE_SHADER_RESOURCE },
                                     { gTintColor, RESOURCE_STATE_RENDER_TARGET, RESOURCE_STATE_SHADER_RESOURCE } };
        cmdResourceBarrier(cmd, 0, nullptr, 0, nullptr, 2, e);
    }
    if (counted)
    {
        cmdResolveQuery(cmd, gCountPool[fi], 0, 2 * counted);
        gCountUsed[fi] = counted;
    }
    cmdEndDebugMarker(cmd);
}



// In load(): the tint passes' shaders, pipelines and buffers. Refused, glass casts no tinted shadows (glass_shadows reads
// as off).
void makeTintPasses()
{
    ShaderLoadDesc td = {};
    td.mVert.pFileName = "renderer_shadow_tint.vert";
    td.mFrag.pFileName = "renderer_shadow_tint.frag";
    addShader(gRenderer, &td, &gTintShader);
    td.mFrag.pFileName = "renderer_shadow_tint_rest.frag";
    addShader(gRenderer, &td, &gTintRestShader);
    ShaderLoadDesc cd = {};
    cd.mVert.pFileName = "renderer_shadow_tint_copy.vert";
    cd.mFrag.pFileName = "renderer_shadow_tint_copy.frag";
    addShader(gRenderer, &cd, &gTintCopyShader);
    if (!gTintShader || !gTintRestShader || !gTintCopyShader)
        return;
    DescriptorSetDesc setDesc = SRT_SET_DESC(TintSrt, PerFrame, gFrames, 0);
    addDescriptorSet(gRenderer, &setDesc, &gTintSet);
    if (!gTintSet)
        return;

    // The core's scene layout (renderer.cpp sceneLayout, without motion): positions, attributes, the instance-rate stream.
    VertexLayout l = {};
    l.mBindingCount = 3;
    l.mBindings[0].mStride = RENDER_POSITION_STRIDE;
    l.mBindings[1].mStride = RENDER_ATTRIBUTE_STRIDE;
    l.mBindings[2].mStride = 4;
    l.mBindings[2].mRate = VERTEX_BINDING_RATE_INSTANCE;
    l.mAttribCount = 6;
    const ShaderSemantic  sem[6] = { SEMANTIC_POSITION, SEMANTIC_NORMAL, SEMANTIC_TEXCOORD1, SEMANTIC_TANGENT, SEMANTIC_TEXCOORD0, SEMANTIC_COLOR };
    const TinyImageFormat fmt[6] = { TinyImageFormat_R32G32B32_SFLOAT, TinyImageFormat_R16G16_SNORM,     TinyImageFormat_R32_UINT,
                                     TinyImageFormat_R16G16_SINT,      TinyImageFormat_R32G32_SFLOAT,    TinyImageFormat_R8G8B8A8_UNORM };
    const uint32_t        binding[6] = { 0, 1, 2, 1, 1, 1 }, offset[6] = { 0, 0, 0, 4, 8, 16 };
    for (uint32_t i = 0; i < 6; ++i)
    {
        l.mAttribs[i].mSemantic = sem[i];
        l.mAttribs[i].mFormat = fmt[i];
        l.mAttribs[i].mBinding = binding[i];
        l.mAttribs[i].mLocation = i;
        l.mAttribs[i].mOffset = offset[i];
    }
    RasterizerStateDesc raster = {};
    raster.mCullMode = CULL_MODE_NONE;
    raster.mDepthClampEnable = true; // glass nearer than a cascade's near plane still tints
    DepthStateDesc nearest = {};
    nearest.mDepthTest = true;
    nearest.mDepthWrite = true;
    nearest.mDepthFunc = CMP_LESS;
    DepthStateDesc  none = {};
    TinyImageFormat scratch[2] = { TinyImageFormat_R32_SFLOAT, TinyImageFormat_R16G16B16A16_SFLOAT };
    TinyImageFormat maps[2] = { TinyImageFormat_R32_SFLOAT, TinyImageFormat_R10G10B10A2_UNORM };

    PipelineDesc desc = {};
    desc.mType = PIPELINE_TYPE_GRAPHICS;
    PIPELINE_LAYOUT_DESC(desc, NULL, SRT_LAYOUT_DESC(TintSrt, PerFrame), NULL, NULL);
    GraphicsPipelineDesc& g = desc.mGraphicsDesc;
    g.mPrimitiveTopo = PRIMITIVE_TOPO_TRI_LIST;
    g.mRenderTargetCount = 2;
    g.pColorFormats = scratch;
    g.mSampleCount = SAMPLE_COUNT_1;
    g.mDepthStencilFormat = TinyImageFormat_D32_SFLOAT;
    g.pRasterizerState = &raster;
    g.pDepthState = &nearest;
    g.pVertexLayout = &l;
    g.pShaderProgram = gTintShader;
    desc.pName = "renderer_shadow_tint";
    addPipeline(gRenderer, &desc, &gTintPipe);
    // The last layer's rest, into the scratch colour alone (one blend state, so nothing rests on per-target blending):
    // what each surface past the kept one lets through multiplied in, and one added to the count for each.
    BlendStateDesc blend = {};
    blend.mRenderTargetMask = BLEND_STATE_TARGET_0;
    blend.mSrcFactors[0] = BC_ZERO;
    blend.mDstFactors[0] = BC_SRC_COLOR;
    blend.mSrcAlphaFactors[0] = blend.mDstAlphaFactors[0] = BC_ONE;
    blend.mBlendModes[0] = blend.mBlendAlphaModes[0] = BM_ADD;
    blend.mColorWriteMasks[0] = COLOR_MASK_ALL;
    TinyImageFormat restFormat = TinyImageFormat_R16G16B16A16_SFLOAT;
    g.mRenderTargetCount = 1;
    g.pColorFormats = &restFormat;
    g.mDepthStencilFormat = TinyImageFormat_UNDEFINED;
    g.pDepthState = &none;
    g.pBlendState = &blend;
    g.pShaderProgram = gTintRestShader;
    desc.pName = "renderer_shadow_tint_rest";
    addPipeline(gRenderer, &desc, &gTintLastPipe);
    g.mRenderTargetCount = 2;

    VertexLayout cl = {};
    cl.mBindingCount = 2;
    cl.mBindings[0].mStride = 8;
    cl.mBindings[1].mStride = 4;
    cl.mBindings[1].mRate = VERTEX_BINDING_RATE_INSTANCE;
    cl.mAttribCount = 2;
    cl.mAttribs[0].mSemantic = SEMANTIC_POSITION;
    cl.mAttribs[0].mFormat = TinyImageFormat_R32G32_SFLOAT;
    cl.mAttribs[0].mBinding = 0;
    cl.mAttribs[0].mLocation = 0;
    cl.mAttribs[1].mSemantic = SEMANTIC_TEXCOORD1;
    cl.mAttribs[1].mFormat = TinyImageFormat_R32_UINT;
    cl.mAttribs[1].mBinding = 1;
    cl.mAttribs[1].mLocation = 1;
    RasterizerStateDesc plain = {};
    plain.mCullMode = CULL_MODE_NONE;
    g.pRasterizerState = &plain;
    g.pColorFormats = maps;
    g.mDepthStencilFormat = TinyImageFormat_UNDEFINED;
    g.pBlendState = nullptr;
    g.pVertexLayout = &cl;
    g.pShaderProgram = gTintCopyShader;
    desc.pName = "renderer_shadow_tint_copy";
    addPipeline(gRenderer, &desc, &gTintCopyPipe);
    BlendStateDesc mute = {};
    mute.mRenderTargetMask = (BlendStateTargets)(BLEND_STATE_TARGET_0 | BLEND_STATE_TARGET_1);
    mute.mIndependentBlend = true;
    mute.mColorWriteMasks[0] = mute.mColorWriteMasks[1] = COLOR_MASK_NONE;
    for (uint32_t t = 0; t < 2; ++t)
    {
        mute.mSrcFactors[t] = mute.mSrcAlphaFactors[t] = BC_ONE;
        mute.mDstFactors[t] = mute.mDstAlphaFactors[t] = BC_ZERO;
        mute.mBlendModes[t] = mute.mBlendAlphaModes[t] = BM_ADD;
    }
    g.pBlendState = &mute;
    desc.pName = "renderer_shadow_tint_count";
    addPipeline(gRenderer, &desc, &gTintCountPipe);
    if (!gTintPipe || !gTintLastPipe || !gTintCopyPipe || !gTintCountPipe)
    {
        yea::host()->log(1, "renderer_shadows: the glass tint pipelines were refused; glass casts untinted light");
        return;
    }

    for (uint32_t i = 0; i < gFrames; ++i)
    {
        BufferLoadDesc d = {};
        d.mDesc.mDescriptors = DESCRIPTOR_TYPE_BUFFER;
        d.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
        d.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
        d.mDesc.mStartState = RESOURCE_STATE_SHADER_RESOURCE;
        d.mDesc.mSize = sizeof(RenderGpuShadowTint) * kMaxViews;
        d.mDesc.mStructStride = sizeof(RenderGpuShadowTint);
        d.mDesc.mElementCount = kMaxViews;
        d.mDesc.pName = "renderer_shadow_tints";
        d.ppBuffer = &gTintBuf[i];
        addResource(&d, nullptr);
        d.mDesc.mSize = sizeof(TintPassCpu) * kMaxTintPasses;
        d.mDesc.mStructStride = sizeof(TintPassCpu);
        d.mDesc.mElementCount = kMaxTintPasses;
        d.mDesc.pName = "renderer_shadow_tint_passes";
        d.ppBuffer = &gPassBuf[i];
        addResource(&d, nullptr);
        QueryPoolDesc qd = {};
        qd.pName = "renderer_shadow_tint_counts";
        qd.mType = QUERY_TYPE_OCCLUSION;
        qd.mQueryCount = 2 * kTintCounts;
        initQueryPool(gRenderer, &qd, &gCountPool[i]);
        gCountUsed[i] = 0;
        gTintTexBound[i] = 0;
    }
    BufferLoadDesc t = {};
    t.mDesc.mDescriptors = DESCRIPTOR_TYPE_VERTEX_BUFFER;
    t.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
    t.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
    t.mDesc.mStartState = RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
    t.mDesc.mSize = 3 * 8;
    t.mDesc.pName = "renderer_shadow_tint_triangle";
    t.ppBuffer = &gTriBuf;
    addResource(&t, nullptr);
}

void freeTintPasses()
{
    freeTintTargets();
    for (uint32_t i = 0; i < gFrames; ++i)
    {
        if (gTintBuf[i])
            removeResource(gTintBuf[i]);
        if (gPassBuf[i])
            removeResource(gPassBuf[i]);
        if (gCountPool[i])
            exitQueryPool(gRenderer, gCountPool[i]);
    }
    if (gTriBuf)
        removeResource(gTriBuf);
    Pipeline* pipes[4] = { gTintPipe, gTintLastPipe, gTintCopyPipe, gTintCountPipe };
    for (Pipeline* p : pipes)
        if (p)
            removePipeline(gRenderer, p);
    if (gTintSet)
        removeDescriptorSet(gRenderer, gTintSet);
    if (gTintShader)
        removeShader(gRenderer, gTintShader);
    if (gTintCopyShader)
        removeShader(gRenderer, gTintCopyShader);
    if (gTintRestShader)
        removeShader(gRenderer, gTintRestShader);
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
    makeTintPasses();
    waitForAllResourceLoads();
    if (gTriBuf && gTriBuf->pCpuMappedAddress)
    {
        const float tri[6] = { -1, -1, 3, -1, -1, 3 }; // a triangle over the square
        memcpy(gTriBuf->pCpuMappedAddress, tri, sizeof tri);
    }
    gStreamCpu = new uint32_t[kMaxEntries];
    gPipes = yea::pipelines(); // the pipelines are queued by the first shadows_prepare (the host drops load()'s)
    gReady = true;
    return 0;
}

void unload()
{
    if (!gRenderer)
        return;
    waitQueueIdle(gQueue);
    dropNext();
    gNextSlope = -1;
    if (gPipe)
        removePipeline(gRenderer, gPipe);
    if (gClearPipe)
        removePipeline(gRenderer, gClearPipe);
    if (gMaskedPipe)
        removePipeline(gRenderer, gMaskedPipe);
    if (gMaskedShader)
        removeShader(gRenderer, gMaskedShader);
    free(gPrev);
    free(gMovedNow);
    gPrev = nullptr, gMovedNow = nullptr;
    gPrevCap = gMovedCap = 0;
    gPrevFrame = 0;
    freeTarget(gSunAtlas);
    freeTarget(gLocalAtlas);
    freeTintPasses();
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

// The slope's pipelines, here rather than in shadows_prepare: queued before the core's first frame (prepare comes only
// once the core draws, behind all its warm-up), so the first picture has its shadows; and a compile is this module's to
// take (one queued under the core's call, as prepare is, isn't).
void update(uint32_t phase, float dt)
{
    if (gReady && slopePipelines(settings.slope_bias))
    {
        for (Group& g : gGroups)
            g.valid = 0;
        gSunValid = 0;
    }
}
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
        snprintf(line, sizeof line, "renderer_shadows: the core's frame is v%u, this module's v%u: standing aside", core, RENDER_FRAME_VERSION);
        yea::log(line);
        logged = core + 1;
    }
    return false;
}

// The part of view v's square a sphere covers, as an ndc rectangle (x0, y0, x1, y1) added to r (or set when !on): its exact
// bounds (Mara and McGuire 2013, "2D Polyhedral Bounds of a Clipped, Perspective-Projected 3D Sphere"), each axis from the
// sphere's tangents in view space; all of the square when it reaches behind the light.
void rectAdd(const RenderGpuShadowView& v, v3 c, float rad, float r[4], bool on)
{
    float       x0 = -1, y0 = -1, x1 = 1, y1 = 1;
    const float z = v.row[3][0] * c.x + v.row[3][1] * c.y + v.row[3][2] * c.z + v.row[3][3];
    if (z > rad * 1.001f + 1e-4f)
    {
        auto axis = [&](const float* row, float* lo, float* hi) {
            const float s = sqrtf(row[0] * row[0] + row[1] * row[1] + row[2] * row[2]); // 1 / tan(half angle)
            const float a = (row[0] * c.x + row[1] * c.y + row[2] * c.z + row[3]) / s;
            const float t = sqrtf(a * a + z * z - rad * rad);
            *lo = s * (a * t - z * rad) / (z * t + a * rad);
            *hi = s * (a * t + z * rad) / (z * t - a * rad);
        };
        axis(v.row[0], &x0, &x1);
        axis(v.row[1], &y0, &y1);
    }
    x0 = maxf(x0, -1.0f), y0 = maxf(y0, -1.0f), x1 = minf(x1, 1.0f), y1 = minf(y1, 1.0f);
    if (!on)
        r[0] = x0, r[1] = y0, r[2] = x1, r[3] = y1;
    else
        r[0] = minf(r[0], x0), r[1] = minf(r[1], y0), r[2] = maxf(r[2], x1), r[3] = maxf(r[3], y1);
}

// Grows sphere s (or sets it when !on) to hold sphere (c, r).
void sphereAdd(float s[4], v3 c, float r, bool on)
{
    v3    a = V(s[0], s[1], s[2]);
    float d = length(c - a);
    if (on && d + r <= s[3])
        return;
    if (!on || d + s[3] <= r)
    {
        s[0] = c.x, s[1] = c.y, s[2] = c.z, s[3] = r;
        return;
    }
    float nr = (d + r + s[3]) * 0.5f;
    v3    nc = a + (c - a) * ((nr - s[3]) / d);
    s[0] = nc.x, s[1] = nc.y, s[2] = nc.z, s[3] = nr;
}

bool shadows_prepare(uint64_t address)
{
    if (!coreMatches())
        return false;
    RenderFrame* F = (RenderFrame*)(uintptr_t)address;
    gLightShadowsOn = false;
    gRefreshCount = 0;
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
    if (sunSize != gSunSize || localSize != gLocalSize)
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
    // A held face resizes only once the new size has been wanted kResizeFrames frames running (as has a face that got a
    // smaller square than it asked for, to try again): the squeeze above can trade sizes between faces from one frame to
    // the next as needs refresh, and each resize moves the light's squares and spends a redraw.
    for (uint32_t j = 0; j < pickedCount && !gFrozen; ++j)
    {
        Group&   g = gGroups[groupOf[j]];
        uint32_t faces = F->lights[picked[j]].type == 1 ? 6 : 1;
        bool     settled = g.nfaces == faces;
        for (uint32_t f = 0; f < faces && settled; ++f)
            settled = g.asked[f] == target[j][f] && g.k[f] == g.asked[f];
        g.resizeWait = g.valid && !settled ? g.resizeWait + 1 : 0;
    }
    uint32_t moved = 0;
    for (uint32_t pass = 0; pass < 2 && !gFrozen; ++pass)
        for (uint32_t j = 0; j < pickedCount && moved < budget; ++j)
        {
            // Lights without a shadow first, then those changing size.
            Group&   g = gGroups[groupOf[j]];
            uint32_t faces = F->lights[picked[j]].type == 1 ? 6 : 1;
            if ((pass == 0) != (g.valid == 0) || (pass == 0 ? g.nfaces == faces : g.resizeWait < kResizeFrames))
                continue;
            if (pass == 1)
                groupFree(g); // its squares can be part of the new ones
            g.valid = 0;
            g.resizeWait = 0;
            if (groupAlloc(g, faces, target[j]))
            {
                moved++;
                (pass == 0 ? gStats.placed : gStats.resized) += faces;
            }
        }
    logAtlas();

    uint32_t dirty[kGroups], dirtyCount = 0;
    for (uint32_t j = 0; j < pickedCount; ++j)
    {
        Group&               g = gGroups[groupOf[j]];
        const RenderLight& l = F->lights[picked[j]];
        g.seen = F->frameNumber;
        float key[10];
        lightKey(l, key);
        // Moved casters don't make a light stale (6.6.5): the faces they are in are refreshed below, each on its own.
        bool stale = !g.valid || memcmp(key, g.key, sizeof key) != 0 || F->casterRemoved >= g.rendered;
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
        for (uint32_t f = 0; f < 6; ++f)
            g.faceDrawn[f] = g.rendered;
        for (uint32_t f = 0; f < g.faces; ++f)
        {
            // Its list: the casters in the face's frustum as it is drawn now.
            mx::Plane planes[6];
            clipPlanes(g.views[f].row, planes);
            uint32_t n = 0;
            for (uint32_t i = 0; i < F->casterCount && n <= kListMax; ++i)
            {
                const RenderCaster& c = F->casters[i];
                if (!sphereInside(planes, 6, V(c.sphere[0], c.sphere[1], c.sphere[2]), c.sphere[3]))
                    continue;
                if (n < kListMax)
                    g.list[f][n] = c.instance;
                n++;
            }
            g.listCount[f] = n;
        }
        gPending[gPendingCount++] = (uint32_t)groupOf[j];
    }
    gStats.redrawn += gPendingCount;

    // Dirty rectangles (6.6.5): each caster that moved since last frame adds where it was and where it is to the faces either
    // reaches, so a refresh clears and draws again only that part of a face, exactly as a whole redraw would leave it (as
    // virtual shadow maps invalidate the pages under an object's last and present bounds). A whole redraw empties a face's.
    // Rows that may have changed hands (instances added or removed, or a frame without this) make every face dirty whole.
    bool whole[kGroups] = {};
    for (uint32_t k = 0; k < gPendingCount; ++k)
        whole[gPending[k]] = true;
    {
        const bool all = !gPrevFrame || gPrevFrame + 1 != F->frameNumber || F->instanceCount != gPrevCount ||
                         F->casterRemoved > gPrevFrame;
        if (F->instanceCount > gPrevCap || F->casterCount > gMovedCap)
        {
            gPrevCap = F->instanceCount > gPrevCap ? F->instanceCount + F->instanceCount / 2 + 64 : gPrevCap;
            gMovedCap = F->casterCount > gMovedCap ? F->casterCount + F->casterCount / 2 + 64 : gMovedCap;
            PrevCaster* prev = (PrevCaster*)realloc(gPrev, (size_t)gPrevCap * sizeof *gPrev);
            uint32_t*   moved = (uint32_t*)realloc(gMovedNow, (size_t)gMovedCap * sizeof *gMovedNow);
            if (prev)
                gPrev = prev;
            if (moved)
                gMovedNow = moved;
            if (!prev || !moved)
                gPrevCap = gMovedCap = 0;
        }
        uint32_t movedCount = 0;
        for (uint32_t i = 0; i < F->casterCount && gMovedCap; ++i)
            if (F->casters[i].moved > gPrevFrame)
                gMovedNow[movedCount++] = i;
        for (uint32_t gi = 0; gi < kGroups; ++gi)
        {
            Group& g = gGroups[gi];
            if (whole[gi] || !g.valid)
            {
                memset(g.dirtyOn, 0, sizeof g.dirtyOn);
                continue;
            }
            if (all || !gMovedCap)
            {
                // Rows may have changed hands: every face dirty whole, its list unknown (walk every caster) till redrawn.
                for (uint32_t f = 0; f < g.faces; ++f)
                {
                    g.dirty[f][0] = g.dirty[f][1] = -1, g.dirty[f][2] = g.dirty[f][3] = 1;
                    g.dirtySphere[f][3] = 1e30f;
                    g.dirtyOn[f] = 1;
                    g.listCount[f] = kListMax + 1;
                }
                continue;
            }
            // Only the moved casters within the light's reach (its faces' far corners) are tested against its faces.
            const v3    light = V(g.key[0], g.key[1], g.key[2]);
            const float range = g.reach;
            for (uint32_t m = 0; m < movedCount; ++m)
            {
                const RenderCaster& c = F->casters[gMovedNow[m]];
                const PrevCaster&   p = gPrev[c.instance];
                const v3            at = V(c.sphere[0], c.sphere[1], c.sphere[2]);
                const v3            was = V(p.sphere[0], p.sphere[1], p.sphere[2]);
                const bool          nowNear = length(at - light) < range + c.sphere[3];
                const bool          wasNear = p.frame == gPrevFrame && length(was - light) < range + p.sphere[3];
                if (!nowNear && !wasNear)
                    continue;
                for (uint32_t f = 0; f < g.faces; ++f)
                {
                    mx::Plane planes[6];
                    clipPlanes(g.views[f].row, planes);
                    if (nowNear && sphereInside(planes, 6, at, c.sphere[3]))
                    {
                        rectAdd(g.views[f], at, c.sphere[3], g.dirty[f], g.dirtyOn[f]);
                        sphereAdd(g.dirtySphere[f], at, c.sphere[3], g.dirtyOn[f]);
                        g.dirtyOn[f] = 1;
                        // Into the face's list, once.
                        uint32_t& n = g.listCount[f];
                        bool      listed = n > kListMax;
                        for (uint32_t i = 0; i < n && !listed; ++i)
                            listed = g.list[f][i] == c.instance;
                        if (!listed)
                        {
                            if (n < kListMax)
                                g.list[f][n] = c.instance;
                            n++;
                        }
                    }
                    if (wasNear && sphereInside(planes, 6, was, p.sphere[3]))
                    {
                        rectAdd(g.views[f], was, p.sphere[3], g.dirty[f], g.dirtyOn[f]);
                        sphereAdd(g.dirtySphere[f], was, p.sphere[3], g.dirtyOn[f]);
                        g.dirtyOn[f] = 1;
                    }
                }
            }
        }
        for (uint32_t i = 0; i < F->casterCount && gPrevCap; ++i)
        {
            const RenderCaster& c = F->casters[i];
            if (c.instance < gPrevCap)
            {
                memcpy(gPrev[c.instance].sphere, c.sphere, sizeof c.sphere);
                gPrev[c.instance].frame = F->frameNumber;
                gPrev[c.instance].caster = i;
            }
        }
        gPrevFrame = F->frameNumber;
        gPrevCount = F->instanceCount;
    }

    // Faces with a dirty rectangle, of lights not redrawn whole this frame: that part of them again, at the view they have
    // (the light's drawn position, so the cube faces still agree).
    gRefreshCount = 0;
    if (!gFrozen)
    {
        // Only where it can show: a dirty face counts while its sphere, or the shadow it throws along the light out to the
        // light's range (a capsule from it to its far end, widening as the light's rays spread), reaches the camera's view.
        // One that can't stays as drawn, still dirty, and is refreshed once it can.
        mx::Plane view[6];
        {
            float vp[4][4], proj[4][4] = { { F->p00, 0, 0, 0 }, { 0, F->p11, 0, 0 }, { 0, 0, 0, F->nearZ }, { 0, 0, 1, 0 } };
            float cam[4][4] = { { F->view[0][0], F->view[0][1], F->view[0][2], F->view[0][3] },
                                { F->view[1][0], F->view[1][1], F->view[1][2], F->view[1][3] },
                                { F->view[2][0], F->view[2][1], F->view[2][2], F->view[2][3] },
                                { 0, 0, 0, 1 } };
            mul(proj, cam, vp);
            clipPlanes(vp, view);
            view[4] = view[5]; // reversed-Z: the sixth is the near plane, the fifth degenerate
        }
        auto shows = [&](const float* s, v3 light, float range) {
            if (s[3] >= 1e29f)
                return true;
            v3    a = V(s[0], s[1], s[2]);
            float r = s[3], dist = length(a - light);
            float reach = maxf(range - dist, 0.0f);
            v3    b = dist > 1e-4f ? a + (a - light) * (reach / dist) : a;
            float rb = r * (dist + reach) / maxf(dist - r, 0.05f); // the widest the rays past it spread
            for (uint32_t i = 0; i < 6; ++i)
                if (dot(view[i].n, a) + view[i].d < -r && dot(view[i].n, b) + view[i].d < -rb)
                    return false;
            return true;
        };
        uint32_t cand[kGroups * 6], candCount = 0;
        for (uint32_t j = 0; j < pickedCount; ++j)
        {
            const uint32_t gi = (uint32_t)groupOf[j];
            const Group&   g = gGroups[gi];
            if (whole[gi] || !g.valid || !g.nfaces)
                continue;
            const float range = F->lights[picked[j]].range;
            for (uint32_t f = 0; f < g.faces; ++f)
                if (g.dirtyOn[f] && g.dirty[f][0] < g.dirty[f][2] && g.dirty[f][1] < g.dirty[f][3] &&
                    shows(g.dirtySphere[f], V(g.views[f].info[1], g.views[f].info[2], g.views[f].info[3]), range))
                    cand[candCount++] = gi * 6 + f;
        }
        for (uint32_t c = 0; c < candCount; ++c)
        {
            Group&         g = gGroups[cand[c] / 6];
            const uint32_t f = cand[c] % 6;
            const uint32_t lag = (uint32_t)(F->frameNumber + 1 - g.faceDrawn[f]);
            gStats.lagSum += lag, gStats.lagCount++, gStats.lagMax = lag > gStats.lagMax ? lag : gStats.lagMax;
            memcpy(gRefreshRect[gRefreshCount], g.dirty[f], sizeof g.dirty[f]);
            gStats.area += (g.dirty[f][2] - g.dirty[f][0]) * (g.dirty[f][3] - g.dirty[f][1]) * 0.25;
            gRefresh[gRefreshCount++] = cand[c];
            g.dirtyOn[f] = 0;
            g.faceDrawn[f] = F->frameNumber + 1;
        }
        gStats.refreshed += gRefreshCount;
    }
    for (uint32_t j = 0; j < pickedCount; ++j)
        if (gGroups[groupOf[j]].valid)
            F->lights[picked[j]].shadow = (int32_t)(kSunViews + groupOf[j] * 6);
    for (uint32_t g = 0; g < kGroups; ++g)
        if (gGroups[g].valid)
            memcpy(&views[kSunViews + g * 6], gGroups[g].views, sizeof gGroups[g].views);

    memcpy(gViewBuf[F->frameIndex]->pCpuMappedAddress, gViewsCpu, sizeof gViewsCpu);
    prepareTints(F);

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
    if (!coreMatches())
        return;
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

    // One face's square, cleared and its casters drawn.
    // rect: only that part of it (a refresh's dirty rectangle, ndc), widened by two texels, its casters culled to it, and
    // only those in its list walked (they are every caster that can be in the face).
    auto drawFace = [&](uint32_t idx, const float* rect) {
        const uint32_t f = idx % 6;
        const Group&   g = gGroups[idx / 6];
        float          tile = (float)(g.k[f] * kCell);
        float          x = (float)(g.cx[f] * kCell), y = (float)(g.cy[f] * kCell);
        float          x0 = x, y0 = y, x1 = x + tile, y1 = y + tile;
        if (rect)
        {
            // ndc y up is the square's top row (Lighting.h.fsl's viewToAtlas).
            x0 = maxf(x, floorf(x + (rect[0] * 0.5f + 0.5f) * tile) - 2), x1 = minf(x + tile, ceilf(x + (rect[2] * 0.5f + 0.5f) * tile) + 2);
            y0 = maxf(y, floorf(y + (0.5f - rect[3] * 0.5f) * tile) - 2), y1 = minf(y + tile, ceilf(y + (0.5f - rect[1] * 0.5f) * tile) + 2);
            if (x1 <= x0 || y1 <= y0)
                return;
            const float nx0 = (x0 - x) / tile * 2 - 1, nx1 = (x1 - x) / tile * 2 - 1;
            const float ny0 = 1 - (y1 - y) / tile * 2, ny1 = 1 - (y0 - y) / tile * 2;
            const float cx = (nx0 + nx1) * 0.5f, hx = (nx1 - nx0) * 0.5f, cy = (ny0 + ny1) * 0.5f, hy = (ny1 - ny0) * 0.5f;
            float       sub[4][4];
            memcpy(sub, g.views[f].row, sizeof sub);
            for (uint32_t i = 0; i < 4; ++i)
                sub[0][i] = (sub[0][i] - cx * sub[3][i]) / hx, sub[1][i] = (sub[1][i] - cy * sub[3][i]) / hy;
            clipPlanes(sub, gSubPlanes);
            gSub = true;
            if (g.listCount[f] <= kListMax)
            {
                // The list's casters by their index this frame, ascending (as the casters come), those gone left out.
                gOnly = true;
                gOnlyCount = 0;
                for (uint32_t i = 0; i < g.listCount[f]; ++i)
                {
                    const uint32_t inst = g.list[f][i];
                    if (inst >= gPrevCap || gPrev[inst].frame != F->frameNumber)
                        continue;
                    uint32_t c = gPrev[inst].caster, k = gOnlyCount++;
                    for (; k > 0 && gOnlyIdx[k - 1] > c; --k)
                        gOnlyIdx[k] = gOnlyIdx[k - 1];
                    gOnlyIdx[k] = c;
                }
                gStats.listed += gOnlyCount;
            }
            else
                gStats.walked++;
        }
        cmdSetViewport(cmd, x, y, tile, tile, 0, 1);
        cmdSetScissor(cmd, (uint32_t)x0, (uint32_t)y0, (uint32_t)(x1 - x0), (uint32_t)(y1 - y0));
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
        gSub = gOnly = false;
    };

    if (gPendingCount || gRefreshCount || gLocalFresh)
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
            for (uint32_t f = 0; f < gGroups[gPending[k]].faces; ++f)
                drawFace(gPending[k] * 6 + f, nullptr);
        for (uint32_t k = 0; k < gRefreshCount; ++k)
            drawFace(gRefresh[k], gRefreshRect[k]);
        cmdBindRenderTargets(cmd, nullptr);
        b = { gLocalAtlas, RESOURCE_STATE_DEPTH_WRITE, RESOURCE_STATE_SHADER_RESOURCE };
        cmdResourceBarrier(cmd, 0, nullptr, 0, nullptr, 1, &b);
        cmdEndDebugMarker(cmd);
    }
    renderTints(cmd, F, fi, stream);
    // The GPU reads the stream when this command list runs, after this copy.
    if (gStreamUsed)
        memcpy(gStreamBuf[fi]->pCpuMappedAddress, gStreamCpu, (size_t)gStreamUsed * 4);
}
} // namespace renderer_shadows
