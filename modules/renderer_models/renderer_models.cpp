// Models for the renderer (AGENTS.md beside the component): the registry of every model the view's modules declare
// (ModelLink.h's MODEL lines, through api()), their meshes and materials in the renderer's scene, the game's placements
// of them as the renderer's instances, and the Model Viewer's preview (drawn by the core's preview_draw into this
// module's own targets).
#include "yea_renderer_models.h"
#include "knobs.h"
#include "models.h"
#include "retarget.h"
#include "blend.h"
#include "rmodel.h"
#include "anim.h"

#include <Common_3/Graphics/Interfaces/IGraphics.h>
#include <yea/ui.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace renderer_models
{
Entry*   gEntries;
uint32_t gEntryCount;
Preview  gPreview;

namespace
{
Renderer*      gRenderer;
Queue*         gQueue;
uint64_t       gInstance;
RenderTarget * gHdr, *gDepth, *gShadow, *gOut;
uint32_t       gTargetW, gTargetH;
RenderPreview* gDraw; // the core's preview request, kept (made in load())
const uint32_t kShadowSize = 1024;
// Skinned models: a skeleton's nodes and palette at most (the cooker allows more; the loader refuses them).
constexpr uint32_t kMaxNodes = 1024, kMaxBones = 1024;
anim::Pose*        gPoses;   // scratch for one placement's evaluation (made in load()): its clip's poses,
anim::Pose*        gFrom;    // the clip it fades from,
float*             gWorld;   // each node's model-space matrix (12 floats),
float*             gPalette; // and each bone's (12 floats)
float *            gWeights, *gFromWeights; // a placement's morph weights, and those of the clip it fades from
// Retargeting (made in load()): a source's sampled poses and weights, pose()'s scratch, and the maps between two
// skeletons (each made the first time a placement plays one's clips on the other, in this module's update).
anim::Pose*        gSrcPoses;
anim::Pose*        gLayerPoses; // a layer's clip as sampled
// Additive layers' first frames (made in load()): each clip's, sampled once and kept while its models stay as read.
constexpr uint32_t kLayerRefs = 8;
struct LayerRef
{
    uint32_t key, source, clip, still, stamp, live;
};
LayerRef           gLayerRefKey[kLayerRefs];
anim::Pose*        gLayerRefs; // kLayerRefs x kMaxNodes
uint32_t           gLayerRefNext;
uint8_t*           gLayerMask;              // the nodes it covers
constexpr uint32_t kMaskPending = 0xFFFFFFFEu; // a layer's bone named while its model loads
anim::Pose*        gBlendPoses;   // a blend space's sample as sampled (made in load())
float*             gBlendWeights; // and its morph weights
// Blend spaces' triangulations, two a placement row (made in load()): kept while their samples stay the same.
struct BlendTris
{
    uint32_t placement, set, n;
    blend::Tris tris;
};
constexpr uint32_t kBlendRows = 4096, kPreviewRow = kBlendRows; // and one more: the Model Viewer's
BlendTris*         gBlendTris;
static_assert(kViewerSamples == blend::kMaxSamples && kViewerNodes == kMaxNodes, "models.h's sizes");
float*             gSrcWeights;
retarget::Scratch  gRt;
// Inertialized transitions (the transitions knob's Inertial; made in load()): a placement in one keeps its offsets
// (anim::Inertial a node, then two floats a morph target: its offset and rate) in the pool, in a slot found by its
// handle (0: the Model Viewer's), taken on its first pose after the switch and dropped once its fade is over.
constexpr float    kInertStep = 1e-3f;     // seconds: how far either side of the switch its clips' rates are read
constexpr uint32_t kInertFloats = 1u << 18; // 1 MB: about 14 thousand nodes in transitions at once
constexpr uint32_t kInertSlots = 256;
struct InertSlot
{
    uint32_t placement, clip, from, start, nodes, targets;
    float    base, last, t1; // the fade's time when taken and when last posed; how long its offsets take to go
};
float*             gInertPool;
InertSlot*         gInertSlot;
uint32_t           gInertCount;
anim::Pose *       gInertNew, *gInertOld; // the new clip a step on, the old a step before
float *            gInertNewW, *gInertOldW;
struct MapSlot
{
    uint32_t      tKey, sKey, tStamp, sStamp, used, live, ok;
    float         tUp, sUp; // their Up Axis knobs, as made
    retarget::Map map;
};
constexpr uint32_t kMaps = 32;
MapSlot*           gMaps;
uint32_t           gStamp, gTick;

void freeAnim(Entry& e)
{
    free(e.anim);
    e.anim = nullptr;
    e.skel = nullptr, e.nodes = nullptr, e.bones = nullptr, e.clips = nullptr, e.targets = nullptr;
    e.motionNode = kNoClip;
    e.animStamp = 0;
    e.legs = 0, e.arms = 0, e.aimCount = 0;
    free(e.boneBox);
    e.boneBox = nullptr;
}

uint32_t fnv(const char* a, const char* b)
{
    uint32_t h = 2166136261u;
    for (const char* s : { a, "/", b })
        for (; *s; ++s)
            h = (h ^ (uint8_t)*s) * 16777619u;
    return h ? h : 1;
}
void copy(char* out, size_t cap, const char* in) { snprintf(out, cap, "%s", in ? in : ""); }

Entry* byId(uint32_t id)
{
    for (uint32_t i = 0; i < gEntryCount; ++i)
        if (gEntries[i].id == id)
            return &gEntries[i];
    return nullptr;
}
Entry* byKey(uint32_t key)
{
    for (uint32_t i = 0; i < gEntryCount; ++i)
        if (gEntries[i].key == key)
            return &gEntries[i];
    return nullptr;
}
uint32_t placementRow(uint32_t h)
{
    uint32_t row = h & 0xFFFFFu;
    return row < Placements.size() && Placements[row].alive && Placements[row].gen == (h >> 20) ? row : 0xFFFFFFFFu;
}
float gGlow = -1; // the emissive_nits the parts' materials have

void dropInstances(uint32_t row)
{
    for (uint32_t i = PartInstances.size(); i-- > 0;)
        if (PartInstances[i].placement == row)
        {
            renderer::instance_remove(PartInstances[i].instance);
            PartInstances.remove(i);
        }
}

// --- api(): what the game's modules call (in their own update) ---------------------------------------------------
void apiBegin(const char* owner)
{
    for (uint32_t i = 0; i < gEntryCount; ++i)
        if (!strcmp(gEntries[i].owner, owner))
            gEntries[i].defined = 0;
}
uint32_t apiDefine(const mdl::Spec* s, char* error, uint32_t bytes)
{
    if (!s || !s->owner || !s->name || !s->source)
        return copy(error, bytes, "no owner, name or source"), 0;
    const uint32_t key = fnv(s->owner, s->name);
    Entry*         e = byKey(key);
    if (!e)
    {
        if (gEntryCount >= kMaxModels)
            return copy(error, bytes, "more than 512 models in the view"), 0;
        e = &gEntries[gEntryCount++];
        *e = Entry{};
        e->key = key;
        static uint32_t next = 0;
        e->id = ++next ? next : ++next;
    }
    if (e->source[0] && strcmp(e->source, s->source)) // a new file: its parts made again
    {
        e->state = 0;
        freeAnim(*e);
        for (uint32_t i = ModelParts.size(); i-- > 0;)
            if (ModelParts[i].key == key)
                ModelParts.remove(i);
    }
    copy(e->owner, sizeof e->owner, s->owner);
    copy(e->name, sizeof e->name, s->name);
    copy(e->source, sizeof e->source, s->source);
    copy(e->assets, sizeof e->assets, s->assets);
    memcpy(e->knobs, s->knobs, sizeof e->knobs);
    e->defined = 1;
    for (uint32_t i = 0; i < Placements.size(); ++i)
        if (Placements[i].alive && Placements[i].key == key)
            Placements[i].dirty = 1;
    return e->id;
}
void apiEnd(const char* owner)
{
    for (uint32_t i = 0; i < gEntryCount;)
        if (!strcmp(gEntries[i].owner, owner) && !gEntries[i].defined)
        {
            for (uint32_t r = 0; r < Placements.size(); ++r)
                if (Placements[r].alive && Placements[r].key == gEntries[i].key)
                    Placements[r].alive = 0, Placements[r].dirty = 1; // its instances go in update()
            freeAnim(gEntries[i]);
            memmove(&gEntries[i], &gEntries[i + 1], sizeof(Entry) * (gEntryCount - i - 1));
            --gEntryCount;
        }
        else
            ++i;
}
void apiKnobs(uint32_t model, const float* values)
{
    Entry* e = byId(model);
    if (!e || !values)
        return;
    memcpy(e->knobs, values, sizeof e->knobs);
    for (uint32_t i = 0; i < Placements.size(); ++i)
        if (Placements[i].alive && Placements[i].key == e->key)
            Placements[i].dirty = 1;
}
uint32_t apiPlace(uint32_t model, const float* pos, const float* rot, float scale)
{
    Entry* e = byId(model);
    if (!e)
        return 0;
    uint32_t row = 0xFFFFFFFFu;
    for (uint32_t i = 0; i < Placements.size() && row == 0xFFFFFFFFu; ++i)
        if (!Placements[i].alive)
            row = i;
    if (row == 0xFFFFFFFFu)
    {
        if (!Placements.add())
            return 0;
        row = Placements.size() - 1;
        Placements[row].gen = 0;
    }
    Placement& p = Placements[row];
    const uint32_t gen = (p.gen + 1) & 0xFFFu;
    p = Placement{};
    p.gen = gen;
    p.alive = 1;
    p.key = e->key;
    p.pos = { pos[0], pos[1], pos[2] };
    p.rot = { rot[0], rot[1], rot[2], rot[3] };
    p.scale = scale;
    p.dirty = 1;
    while (PlacementAnims.size() <= row && PlacementAnims.add())
        ;
    if (row < PlacementAnims.size())
    {
        PlacementAnim& a = PlacementAnims[row];
        a = PlacementAnim{};
        a.clip = 0; // its first clip, looping (a model without clips stands in its rest pose)
        a.speed = 1, a.loop = 1, a.from = kNoClip;
    }
    while (PlacementMotions.size() <= row && PlacementMotions.add())
        ;
    if (row < PlacementMotions.size())
        PlacementMotions[row] = PlacementMotion{};
    while (PlacementRetargets.size() <= row && PlacementRetargets.add())
        ;
    if (row < PlacementRetargets.size())
        PlacementRetargets[row] = PlacementRetarget{};
    while (PlacementGrounds.size() <= row && PlacementGrounds.add())
        ;
    if (row < PlacementGrounds.size())
        PlacementGrounds[row] = PlacementGround{};
    while (PlacementBlends.size() <= row && PlacementBlends.add())
        ;
    if (row < PlacementBlends.size())
        PlacementBlends[row] = PlacementBlend{};
    while (PlacementAims.size() <= row && PlacementAims.add())
        ;
    if (row < PlacementAims.size())
        PlacementAims[row] = PlacementAim{};
    while (PlacementHands.size() <= row && PlacementHands.add())
        ;
    if (row < PlacementHands.size())
        PlacementHands[row] = PlacementHand{};
    return row | gen << 20;
}
void apiMove(uint32_t h, const float* pos, const float* rot, float scale)
{
    uint32_t row = placementRow(h);
    if (row == 0xFFFFFFFFu)
        return;
    Placement& p = Placements[row];
    const yea_float3 np = { pos[0], pos[1], pos[2] };
    const yea_float4 nr = { rot[0], rot[1], rot[2], rot[3] };
    if (memcmp(&p.pos, &np, sizeof np) || memcmp(&p.rot, &nr, sizeof nr) || p.scale != scale)
        p.pos = np, p.rot = nr, p.scale = scale, p.dirty = 1;
}
void apiRemove(uint32_t h)
{
    uint32_t row = placementRow(h);
    if (row != 0xFFFFFFFFu)
        Placements[row].alive = 0, Placements[row].dirty = 1;
}
int      apiPlaced(uint32_t h) { return placementRow(h) != 0xFFFFFFFFu; }
uint32_t apiCount() { return gEntryCount; }
int      apiInfo(uint32_t index, mdl::Info* out)
{
    if (index >= gEntryCount || !out)
        return 0;
    const Entry& e = gEntries[index];
    *out = mdl::Info{};
    out->owner = e.owner;
    out->name = e.name;
    out->source = e.source;
    out->id = e.id;
    out->kind = e.kind == 1 && e.anim ? 2 : e.kind;
    out->state = e.state;
    out->parts = e.parts;
    out->lods = lodCount(e);
    out->triangles = triangles(e);
    for (uint32_t i = 0; i < Placements.size(); ++i)
        out->placed += Placements[i].alive && Placements[i].key == e.key;
    const float d[3] = { e.hi[0] - e.lo[0], e.hi[1] - e.lo[1], e.hi[2] - e.lo[2] };
    out->radius = 0.5f * sqrtf(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    out->error = e.error;
    return 1;
}
// A placement's skinned model and its animation row, or nullptr.
Entry* animOf(uint32_t h, PlacementAnim** out)
{
    const uint32_t row = placementRow(h);
    Entry*         e = row != 0xFFFFFFFFu ? byKey(Placements[row].key) : nullptr;
    if (!e || !e->anim || row >= PlacementAnims.size())
        return nullptr;
    *out = &PlacementAnims[row];
    return e;
}
// A placement's rows beside it in the tables made after Placement (placed before they existed, say before a reload).
void grow(uint32_t row)
{
    while (row != 0xFFFFFFFFu && PlacementMotions.size() <= row && PlacementMotions.add())
        ;
    while (row != 0xFFFFFFFFu && PlacementRetargets.size() <= row && PlacementRetargets.add())
        ;
    while (row != 0xFFFFFFFFu && PlacementGrounds.size() <= row && PlacementGrounds.add())
        ;
    while (row != 0xFFFFFFFFu && PlacementBlends.size() <= row && PlacementBlends.add())
        ;
    while (row != 0xFFFFFFFFu && PlacementAims.size() <= row && PlacementAims.add())
        ;
    while (row != 0xFFFFFFFFu && PlacementHands.size() <= row && PlacementHands.add())
        ;
}
uint32_t clipHash(const char* name, size_t max)
{
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < max && name[i]; ++i)
        h = (h ^ (uint8_t)name[i]) * 16777619u;
    return h ? h : 1;
}
uint32_t clipNamed(const Entry& e, const char* name)
{
    for (uint32_t c = 0; name && name[0] && c < e.skel->clipCount; ++c)
        if (!strncmp(e.clips[c].name, name, sizeof e.clips[c].name))
            return c;
    return kNoClip;
}
uint32_t clipHashed(const Entry& e, uint32_t hash)
{
    for (uint32_t c = 0; c < e.skel->clipCount; ++c)
        if (clipHash(e.clips[c].name, sizeof e.clips[c].name) == hash)
            return c;
    return kNoClip;
}
// Clip c (kNoClip: the rest pose) of model `src` (its key; 0 the placement's own) from now, faded from what played
// over `fade` seconds; the same clip keeps its time. key and fromKey: whose clip plays, and whose it fades from.
void startClip(PlacementAnim& a, uint32_t& key, uint32_t& fromKey, uint32_t c, uint32_t src, float speed, int loop, float fade)
{
    const bool same = c == a.clip && src == key;
    if (!same && fade > 0)
        a.from = a.clip, a.fromTime = a.time, a.fromSpeed = a.speed, a.fromLoop = a.loop, a.fade = fade, a.faded = 0, fromKey = key;
    else if (!same)
        a.from = kNoClip;
    if (!same)
        a.time = 0;
    a.clip = c, a.speed = speed, a.loop = loop ? 1 : 0, key = src;
}
// Its clip (and the one it fades from) moved on by dt: ce and fe the models they are of (nullptr: not ready).
void advanceClip(const Entry* ce, const Entry* fe, PlacementAnim& a, float dt)
{
    auto step = [&](const Entry* e, uint32_t clip, float& time, float speed, uint32_t loop) {
        if (!e || clip >= e->skel->clipCount)
            return;
        const float d = e->clips[clip].duration;
        time += dt * speed;
        if (d > 0)
            time = loop ? (fmodf(time, d) + (time < 0 ? d : 0)) : (time < 0 ? 0 : (time > d ? d : time));
    };
    step(ce, a.clip, a.time, a.speed, a.loop);
    if (a.from != kNoClip)
    {
        step(fe, a.from, a.fromTime, a.fromSpeed, a.fromLoop);
        a.faded += dt;
        if (a.faded >= a.fade)
            a.from = kNoClip;
    }
}
void leaveBlend(uint32_t row, PlacementAnim& a);
int apiPlay(uint32_t h, const char* clip, float speed, int loop, float fade)
{
    PlacementAnim* a;
    Entry*         e = animOf(h, &a);
    if (const uint32_t row = placementRow(h); row < PlacementAnims.size())
        leaveBlend(row, PlacementAnims[row]);
    if (!e)
    {
        // Its model still loading: the clip is taken by name once it has (a placement of a static model ignores it).
        const uint32_t row = placementRow(h);
        Entry*         m = row != 0xFFFFFFFFu ? byKey(Placements[row].key) : nullptr;
        if (!m || m->state != 0 || row >= PlacementAnims.size())
            return 0;
        PlacementAnim& p = PlacementAnims[row];
        p.want = clip && clip[0] ? clipHash(clip, 48) : 0;
        p.clip = clip && clip[0] ? 0 : kNoClip;
        p.speed = speed, p.loop = loop ? 1 : 0, p.time = 0, p.from = kNoClip;
        grow(row);
        if (row < PlacementRetargets.size())
            PlacementRetargets[row] = PlacementRetarget{};
        return 1;
    }
    a->want = 0;
    const uint32_t c = clipNamed(*e, clip);
    if (c == kNoClip && clip && clip[0])
        return 0;
    const uint32_t row = placementRow(h);
    grow(row);
    if (row >= PlacementRetargets.size())
        return 0;
    startClip(*a, PlacementRetargets[row].clip, PlacementRetargets[row].from, c, 0, speed, loop, fade);
    return 1;
}
int apiPlayFrom(uint32_t h, uint32_t model, const char* clip, float speed, int loop, float fade)
{
    const uint32_t row = placementRow(h);
    grow(row);
    Entry*         t = row != 0xFFFFFFFFu ? byKey(Placements[row].key) : nullptr;
    const Entry*   s = byId(model);
    if (!t || !s || row >= PlacementAnims.size() || row >= PlacementRetargets.size() || t->state == 2 || s->state == 2 ||
        (t->state == 1 && !t->anim) || (s->state == 1 && !s->anim))
        return 0;
    if (s == t)
        return apiPlay(h, clip, speed, loop, fade);
    PlacementAnim&     a = PlacementAnims[row];
    PlacementRetarget& r = PlacementRetargets[row];
    leaveBlend(row, a);
    if (t->state != 1 || s->state != 1) // either still loading: taken by name once both are
    {
        a.want = clip && clip[0] ? clipHash(clip, 48) : 0;
        a.clip = kNoClip, a.speed = speed, a.loop = loop ? 1 : 0, a.time = 0, a.from = kNoClip;
        r.clip = s->key, r.from = 0;
        return 1;
    }
    a.want = 0;
    const uint32_t c = clipNamed(*s, clip);
    if (c == kNoClip && clip && clip[0])
        return 0;
    startClip(a, r.clip, r.from, c, s->key, speed, loop, fade);
    return 1;
}
float apiTime(uint32_t h)
{
    PlacementAnim* a;
    return animOf(h, &a) ? a->time : 0.0f;
}
// --- Root motion ---------------------------------------------------------------------------------------------
// The motion node carries a clip's travel: its translation across the model's up (its height stays, so a walk still
// bobs) and its turn about that up (the twist of its rotation). In place, both stay at the clip's first frame; what
// they did is handed to the game. Its parent is taken as still (its rest pose) and turning and scaling evenly.
yea_float4 upTurn(const Entry& e);
yea_float4 qmul(yea_float4 a, yea_float4 b);
void       pendingMotion(uint32_t row, const Entry& e, float* out);
void       placedNow(uint32_t row, const Entry& e, yea_float3& pos, yea_float4& rot, float& scale);
void       qrot(yea_float4 q, const float* v, float* out)
{
    const float u[3] = { q.x, q.y, q.z };
    const float t[3] = { 2 * (u[1] * v[2] - u[2] * v[1]), 2 * (u[2] * v[0] - u[0] * v[2]), 2 * (u[0] * v[1] - u[1] * v[0]) };
    out[0] = v[0] + q.w * t[0] + (u[1] * t[2] - u[2] * t[1]);
    out[1] = v[1] + q.w * t[1] + (u[2] * t[0] - u[0] * t[2]);
    out[2] = v[2] + q.w * t[2] + (u[0] * t[1] - u[1] * t[0]);
}
void mat3(const float* m, const float* v, float* out)
{
    for (int r = 0; r < 3; ++r)
        out[r] = m[r * 3] * v[0] + m[r * 3 + 1] * v[1] + m[r * 3 + 2] * v[2];
}
float dot3(const float* a, const float* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
void  across(const float* v, const float* up, float* out) // v without its part along up
{
    const float d = dot3(v, up);
    for (int k = 0; k < 3; ++k)
        out[k] = v[k] - d * up[k];
}
void turn(const float* up, float angle, const float* v, float* out) // v turned by angle about up
{
    const float h = angle * 0.5f, s = sinf(h);
    qrot({ up[0] * s, up[1] * s, up[2] * s, cosf(h) }, v, out);
}
float wrapPi(float a)
{
    a = fmodf(a + 3.14159265f, 6.28318531f);
    return (a < 0 ? a + 6.28318531f : a) - 3.14159265f;
}
// Its motion node: the nearest common ancestor of the bones its clips move, and that one's parent at rest.
void findMotion(Entry& e)
{
    e.motionNode = kNoClip;
    const uint32_t n = e.skel->nodeCount;
    auto           depth = [&](uint32_t x) {
        uint32_t d = 0;
        for (int32_t p = e.nodes[x].parent; p >= 0; p = e.nodes[p].parent)
            ++d;
        return d;
    };
    auto moves = [&](uint32_t x) { // a clip moves it (0.1 mm) or turns it (0.2 degrees), sampled across each
        for (uint32_t c = 0; c < e.skel->clipCount; ++c)
        {
            anim::Pose p0, p;
            const uint8_t* data = e.anim + (e.clips[c].offset - e.animBase);
            if (!anim::sampleNodeAt(data, e.clips[c], x, 0, false, p0))
                continue;
            for (int k = 1; k <= 16; ++k)
            {
                anim::sampleNodeAt(data, e.clips[c], x, e.clips[c].duration * (float)k / 16, false, p);
                const float dt = fabsf(p.t[0] - p0.t[0]) + fabsf(p.t[1] - p0.t[1]) + fabsf(p.t[2] - p0.t[2]);
                const float dq = fabsf(p.r[0] * p0.r[0] + p.r[1] * p0.r[1] + p.r[2] * p0.r[2] + p.r[3] * p0.r[3]);
                if (dt > 1e-4f || dq < 0.999999f)
                    return true;
            }
        }
        return false;
    };
    // over the bones the clips move (a bone that only stands, such as a skinned mesh's own node, carries nothing)
    uint32_t lca = kNoClip;
    for (uint32_t b = 0; b < e.skel->boneCount; ++b)
    {
        uint32_t x = e.bones[b].node;
        bool     moved = false;
        for (int32_t q = (int32_t)x; q >= 0 && !moved; q = e.nodes[q].parent)
            moved = moves((uint32_t)q);
        if (!moved)
            continue;
        if (lca == kNoClip)
        {
            lca = x;
            continue;
        }
        uint32_t y = lca, dx = depth(x), dy = depth(y);
        for (; dx > dy; --dx)
            x = (uint32_t)e.nodes[x].parent;
        for (; dy > dx; --dy)
            y = (uint32_t)e.nodes[y].parent;
        while (x != y && e.nodes[x].parent >= 0 && e.nodes[y].parent >= 0)
            x = (uint32_t)e.nodes[x].parent, y = (uint32_t)e.nodes[y].parent;
        if (x != y)
            return; // two roots: no one node carries them
        lca = x;
    }
    if (lca == kNoClip || !gPoses || !gWorld)
        return;
    anim::restPoses(e.nodes, n, gPoses);
    float P[12] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 };
    int32_t chain[64], depthN = 0; // the parent's model space at rest: its ancestors' poses, root first
    for (int32_t q = e.nodes[lca].parent; q >= 0 && depthN < 64; q = e.nodes[q].parent)
        chain[depthN++] = q;
    for (int32_t i = depthN; i-- > 0;)
    {
        float l[12];
        anim::poseMatrix(gPoses[chain[i]], l);
        anim::mul34(P, l, P);
    }
    const float* m = P;
    const float  c[9] = { m[5] * m[10] - m[6] * m[9], m[6] * m[8] - m[4] * m[10], m[4] * m[9] - m[5] * m[8],
                          m[2] * m[9] - m[1] * m[10], m[0] * m[10] - m[2] * m[8], m[1] * m[8] - m[0] * m[9],
                          m[1] * m[6] - m[2] * m[5], m[2] * m[4] - m[0] * m[6], m[0] * m[5] - m[1] * m[4] };
    const float det = m[0] * c[0] + m[1] * c[1] + m[2] * c[2];
    if (!(det > 1e-12f)) // flat or mirrored: its turns would read backwards
        return;
    for (int r = 0; r < 3; ++r)
        for (int k = 0; k < 3; ++k)
            e.motionLin[r * 3 + k] = m[r * 4 + k], e.motionInv[r * 3 + k] = c[k * 3 + r] / det;
    e.motionT[0] = m[3], e.motionT[1] = m[7], e.motionT[2] = m[11];
    e.motionNode = lca;
}
// The model's up (what its placement turns to +Y) and the same in the motion node's parent's space.
void motionAxes(const Entry& e, float* up, float* local)
{
    const yea_float4 t = upTurn(e);
    const float      y[3] = { 0, 1, 0 };
    qrot({ -t.x, -t.y, -t.z, t.w }, y, up);
    mat3(e.motionInv, up, local);
    const float l = sqrtf(dot3(local, local));
    for (int k = 0; k < 3; ++k)
        local[k] = l > 0 ? local[k] / l : 0;
}
float twist(const float* r, const float* u) { return 2.0f * atan2f(dot3(r, u), r[3]); } // its turn about u
// The motion node at a clip's time: its model-space position and its turn about the model's up.
bool motionAt(const Entry& e, uint32_t clip, float time, bool loop, const float* u, float* pos, float& yaw)
{
    anim::Pose p;
    if (clip >= e.skel->clipCount || !anim::sampleNodeAt(e.anim + (e.clips[clip].offset - e.animBase), e.clips[clip], e.motionNode, time, loop, p))
        return false;
    mat3(e.motionLin, p.t, pos);
    for (int k = 0; k < 3; ++k)
        pos[k] += e.motionT[k];
    yaw = twist(p.r, u);
    return true;
}
// A clip's sampled pose in place: its motion node's travel across the up and its turn about it as on its first frame.
void inPlace(const Entry& e, uint32_t clip, anim::Pose* poses)
{
    if (e.motionNode == kNoClip || clip >= e.skel->clipCount)
        return;
    float up[3], u[3];
    motionAxes(e, up, u);
    anim::Pose ref;
    if (!anim::sampleNodeAt(e.anim + (e.clips[clip].offset - e.animBase), e.clips[clip], e.motionNode, 0, false, ref))
        return;
    anim::Pose& p = poses[e.motionNode];
    float       d[3] = { p.t[0] - ref.t[0], p.t[1] - ref.t[1], p.t[2] - ref.t[2] }, h[3];
    across(d, u, h);
    for (int k = 0; k < 3; ++k)
        p.t[k] -= h[k];
    const float a = 0.5f * (twist(ref.r, u) - twist(p.r, u)), s = sinf(a);
    const yea_float4 q = qmul({ u[0] * s, u[1] * s, u[2] * s, cosf(a) }, { p.r[0], p.r[1], p.r[2], p.r[3] });
    p.r[0] = q.x, p.r[1] = q.y, p.r[2] = q.z, p.r[3] = q.w;
}
// What a clip's motion node did from `from` over `step` seconds of its time (looping as it does), in the placement's own
// frame as it stood at the start: out[0..2] model-space travel, out[3] its turn. With the pivot: the placement turns
// about its origin, the node stood at its first frame's place, so the turn carries that place round too.
void clipMotion(const Entry& e, uint32_t clip, float from, float step, bool loop, float* out)
{
    out[0] = out[1] = out[2] = out[3] = 0;
    if (e.motionNode == kNoClip || clip >= e.skel->clipCount || step == 0)
        return;
    const float d = e.clips[clip].duration;
    if (!(d > 0))
        return;
    float up[3], u[3], p0[3], y0;
    motionAxes(e, up, u);
    if (!motionAt(e, clip, 0, false, u, p0, y0))
        return;
    float pivot[3];
    across(p0, up, pivot);
    float pieces[8][2];
    int   count = 0;
    const float raw = from + step;
    if (!loop)
        pieces[count][0] = from, pieces[count][1] = raw < 0 ? 0 : (raw > d ? d : raw), ++count;
    else
    {
        int wraps = (int)floorf(raw / d);
        wraps = wraps > 3 ? 3 : (wraps < -3 ? -3 : wraps);
        const float now = raw - (float)wraps * d;
        if (!wraps)
            pieces[count][0] = from, pieces[count][1] = now, ++count;
        else
        {
            const float end = wraps > 0 ? d : 0, start = wraps > 0 ? 0 : d;
            pieces[count][0] = from, pieces[count][1] = end, ++count;
            for (int w = 1; w < (wraps > 0 ? wraps : -wraps); ++w)
                pieces[count][0] = start, pieces[count][1] = end, ++count;
            pieces[count][0] = start, pieces[count][1] = now, ++count;
        }
    }
    for (int i = 0; i < count; ++i)
    {
        float pa[3], pb[3], ya, yb;
        if (!motionAt(e, clip, pieces[i][0], false, u, pa, ya) || !motionAt(e, clip, pieces[i][1], false, u, pb, yb))
            continue;
        const float dy = wrapPi(yb - ya);
        float       dp[3] = { pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2] }, h[3], t[3], rp[3];
        across(dp, up, h);
        turn(up, wrapPi(y0 - ya), h, t);   // its travel, as the in-place node faces
        turn(up, dy, pivot, rp);           // the pivot: turned about the placement's origin, the node must stay put
        for (int k = 0; k < 3; ++k)
            t[k] += pivot[k] - rp[k];
        float acc[3]; // in the frame the pieces before this one left it facing
        turn(up, out[3], t, acc);
        for (int k = 0; k < 3; ++k)
            out[k] += acc[k];
        out[3] += dy;
    }
}

// --- Retargeting ----------------------------------------------------------------------------------------------
// The model a clip is of: the placement's own (key 0), another skinned one (retargeted), nullptr while it isn't ready.
const Entry* sourceOf(const Entry& e, uint32_t key)
{
    if (!key || key == e.key)
        return &e;
    const Entry* s = byKey(key);
    return s && s->anim && s->state == 1 ? s : nullptr;
}
retarget::Skel skelOf(const Entry& e)
{
    const yea_float4 u = upTurn(e);
    return { e.nodes, e.skel->nodeCount, { u.x, u.y, u.z, u.w }, e.targets, e.targets ? e.skel->targetCount : 0,
             e.motionNode };
}
// The map from s's skeleton onto t's: found, else (build: only from this module's update, as it allocates) made.
const retarget::Map* mapFor(const Entry& t, const Entry& s, bool build)
{
    if (!gMaps || !t.anim || !s.anim)
        return nullptr;
    const float tUp = t.knobs[mdl::KnobUpAxis], sUp = s.knobs[mdl::KnobUpAxis];
    for (uint32_t i = 0; i < kMaps; ++i)
    {
        MapSlot& m = gMaps[i];
        if (m.live && m.tKey == t.key && m.sKey == s.key && m.tStamp == t.animStamp && m.sStamp == s.animStamp && m.tUp == tUp &&
            m.sUp == sUp)
        {
            m.used = gTick;
            return m.ok ? &m.map : nullptr;
        }
    }
    if (!build)
        return nullptr;
    MapSlot* slot = &gMaps[0];
    for (uint32_t i = 0; i < kMaps; ++i)
        if (!gMaps[i].live || gMaps[i].used < slot->used)
            slot = &gMaps[i];
    if (slot->live)
        retarget::release(slot->map);
    *slot = MapSlot{};
    slot->tKey = t.key, slot->sKey = s.key, slot->tStamp = t.animStamp, slot->sStamp = s.animStamp, slot->tUp = tUp, slot->sUp = sUp;
    slot->live = 1, slot->used = gTick;
    slot->ok = retarget::build(skelOf(t), skelOf(s), slot->map) ? 1 : 0;
    char line[200];
    if (!slot->ok)
        snprintf(line, sizeof line, "renderer_models: %s (%s) can't play %s's clips: their skeletons have no bones in common", t.name,
                 t.owner, s.name);
    else
        snprintf(line, sizeof line,
                 "renderer_models: %s (%s) plays %s's clips: %u chains, %u more bones by name, %u limbs by IK, pelvis height x%.3f",
                 t.name, t.owner, s.name, slot->map.chains, slot->map.byName, slot->map.ikCount, slot->map.ratio);
    yea::log(line);
    return slot->ok ? &slot->map : nullptr;
}
// Clip `clip` of model s at `time` as e's local poses (in place: its root motion left out) and its morph weights into w
// (e's targets; nullptr skips them): s's own when s is e, else retargeted onto e's skeleton. False: the rest pose.
bool sampleAs(const Entry& e, const Entry* s, uint32_t clip, float time, bool loop, bool still, anim::Pose* out, float* w)
{
    const uint32_t n = e.skel->nodeCount, nt = w && e.targets ? e.skel->targetCount : 0;
    auto           rest = [&] {
        anim::restPoses(e.nodes, n, out);
        for (uint32_t t = 0; t < nt; ++t)
            w[t] = e.targets[t].weight;
        return false;
    };
    if (!s || clip >= s->skel->clipCount)
        return rest();
    const uint8_t* data = s->anim + (s->clips[clip].offset - s->animBase);
    if (s == &e)
    {
        if (nt && !anim::sampleWeights(data, e.clips[clip], n, nt, time, loop, w))
            for (uint32_t t = 0; t < nt; ++t)
                w[t] = e.targets[t].weight;
        if (!anim::sampleClip(data, e.clips[clip], n, time, loop, out))
            return rest();
        if (still)
            inPlace(e, clip, out);
        return true;
    }
    const retarget::Map* m = mapFor(e, *s, false);
    const uint32_t       sn = s->skel->nodeCount, st = s->targets ? s->skel->targetCount : 0;
    if (!m || !anim::sampleClip(data, s->clips[clip], sn, time, loop, gSrcPoses))
        return rest();
    if (still)
        inPlace(*s, clip, gSrcPoses);
    retarget::pose(*m, skelOf(e), skelOf(*s), gSrcPoses, out, gRt);
    const bool weighed = nt && st && anim::sampleWeights(data, s->clips[clip], sn, st, time, loop, gSrcWeights);
    for (uint32_t t = 0; t < nt; ++t)
        w[t] = weighed && m->morph[t] != retarget::kNone ? gSrcWeights[m->morph[t]] : e.targets[t].weight;
    return true;
}
// What clip `clip` of model s does from `from` over `step` (clipMotion), as e's.
void motionOf(const Entry& e, const Entry* s, uint32_t clip, float from, float step, bool loop, float* out)
{
    out[0] = out[1] = out[2] = out[3] = 0;
    if (!s)
        return;
    clipMotion(*s, clip, from, step, loop, out);
    if (s == &e)
        return;
    if (const retarget::Map* m = mapFor(e, *s, false))
        retarget::motion(*m, out);
    else
        out[0] = out[1] = out[2] = out[3] = 0;
}

// --- Blend spaces (mdl::blendSpace) ---------------------------------------------------------------------------------
// One set of a placement's samples, weighed at a point: the rows that weigh, their weights and their clips' lengths.
struct BlendSet
{
    uint32_t n;
    uint32_t clip[blend::kMaxSamples], source[blend::kMaxSamples]; // each sample's clip of model source (key; 0 e's)
    float    w[blend::kMaxSamples], dur[blend::kMaxSamples];
    float    length; // its weighted length (seconds): one cycle of its phase at speed 1
};
// Set `set` of placement h (row `row`) weighed at (x, y), from e (whose clips its samples' sources are retargeted onto):
// samples still waiting on their names, or whose clip isn't there, are left out. False: none to weigh. h 0: the Model
// Viewer's (row kPreviewRow), its samples the panel's list.
bool blendWeigh(const Entry& e, uint32_t row, uint32_t h, uint32_t set, uint32_t dims, float x, float y, BlendSet& b)
{
    b.n = 0, b.length = 0;
    float p[blend::kMaxSamples * 2];
    auto  take = [&](uint32_t source, uint32_t clip, float sx, float sy) {
        const Entry* src = sourceOf(e, source);
        if (!src || clip >= src->skel->clipCount)
            return;
        b.clip[b.n] = clip, b.source[b.n] = source, b.dur[b.n] = src->clips[clip].duration;
        p[b.n * 2] = sx, p[b.n * 2 + 1] = sy, ++b.n;
    };
    for (uint32_t i = 0; set && !h && i < gPreview.blendCount && i < blend::kMaxSamples; ++i)
        take(gPreview.blendSource, gPreview.blendClip[i], gPreview.blendX[i], gPreview.blendY[i]);
    for (uint32_t i = 0; set && h && i < PlacementBlendSamples.size() && b.n < blend::kMaxSamples; ++i)
    {
        const PlacementBlendSample& S = PlacementBlendSamples[i];
        if (S.placement == h && S.set == set && !S.want)
            take(S.source, S.clip, S.x, S.y);
    }
    if (!b.n)
        return false;
    if (dims == 2 && gBlendTris && row <= kBlendRows)
    {
        BlendTris& t = gBlendTris[row * 2 + (set & 1)];
        if (t.placement != h || t.set != set || t.n != b.n)
            t.placement = h, t.set = set, t.n = b.n, blend::triangulate(p, b.n, t.tris);
        blend::weights2(p, b.n, t.tris, x, y, b.w);
    }
    else
    {
        float xs[blend::kMaxSamples];
        for (uint32_t k = 0; k < b.n; ++k)
            xs[k] = p[k * 2];
        blend::weights1(xs, b.n, x, b.w);
    }
    for (uint32_t k = 0; k < b.n; ++k)
        b.length += b.w[k] * b.dur[k];
    return true;
}
// A blend space at `phase` (0 to 1) as e's local poses into out and morph weights into w (nullptr skips them): each
// sample's clip at the phase times its own length, summed by weight (rotations turned to the first one's side, then
// made unit), as UE's blend spaces sync their samples by normalised time. False: nothing to weigh (out untouched).
bool blendSample(const Entry& e, const BlendSet& b, float phase, bool loop, bool still, anim::Pose* out, float* w)
{
    const uint32_t n = e.skel->nodeCount, nt = w && e.targets ? e.skel->targetCount : 0;
    float          total = 0;
    for (uint32_t k = 0; k < b.n; ++k)
    {
        const float wk = b.w[k];
        if (!(wk > 1e-6f))
            continue;
        const bool  first = total == 0;
        anim::Pose* to = first ? out : gBlendPoses;
        float*      tw = first ? w : gBlendWeights;
        sampleAs(e, sourceOf(e, b.source[k]), b.clip[k], phase * b.dur[k], loop, still, to, nt ? tw : nullptr);
        for (uint32_t j = 0; j < n; ++j)
        {
            anim::Pose& o = out[j];
            if (first)
            {
                for (int c = 0; c < 3; ++c)
                    o.t[c] *= wk, o.s[c] *= wk;
                for (int c = 0; c < 4; ++c)
                    o.r[c] *= wk;
                continue;
            }
            const anim::Pose& q = gBlendPoses[j];
            const float       side = o.r[0] * q.r[0] + o.r[1] * q.r[1] + o.r[2] * q.r[2] + o.r[3] * q.r[3] < 0 ? -wk : wk;
            for (int c = 0; c < 3; ++c)
                o.t[c] += wk * q.t[c], o.s[c] += wk * q.s[c];
            for (int c = 0; c < 4; ++c)
                o.r[c] += side * q.r[c];
        }
        for (uint32_t t = 0; t < nt; ++t)
            w[t] = first ? w[t] * wk : w[t] + wk * gBlendWeights[t];
        total += wk;
    }
    if (!(total > 0))
        return false;
    for (uint32_t j = 0; j < n; ++j)
    {
        anim::Pose& o = out[j];
        for (int c = 0; c < 3; ++c)
            o.t[c] /= total, o.s[c] /= total;
        const float l = sqrtf(o.r[0] * o.r[0] + o.r[1] * o.r[1] + o.r[2] * o.r[2] + o.r[3] * o.r[3]);
        for (int c = 0; c < 4; ++c)
            o.r[c] = l > 0 ? o.r[c] / l : (c == 3 ? 1.0f : 0.0f);
    }
    for (uint32_t t = 0; t < nt; ++t)
        w[t] /= total;
    return true;
}
// What a blend space's samples do from `phase` over `step` of it (clipMotion's, each over its own length), summed by
// weight, as e's: out[0..2] travel, out[3] turn.
void blendMotion(const Entry& e, const BlendSet& b, float phase, float step, bool loop, float* out)
{
    out[0] = out[1] = out[2] = out[3] = 0;
    float total = 0;
    for (uint32_t k = 0; k < b.n; ++k)
    {
        if (!(b.w[k] > 1e-6f))
            continue;
        float m[4];
        motionOf(e, sourceOf(e, b.source[k]), b.clip[k], phase * b.dur[k], step * b.dur[k], loop, m);
        for (int c = 0; c < 4; ++c)
            out[c] += b.w[k] * m[c];
        total += b.w[k];
    }
    for (int c = 0; total > 0 && c < 4; ++c)
        out[c] /= total;
}
// The blend space a placement's clip (from: the one it fades from) is: its set and point now. False: not a blend space.
// A row's blend space: a placement's, or the Model Viewer's (kPreviewRow); nullptr none.
PlacementBlend* blendFor(uint32_t row)
{
    return row < PlacementBlends.size() ? &PlacementBlends[row] : row == kPreviewRow ? &gPreview.blend : nullptr;
}
bool blendOf(const PlacementAnim& a, uint32_t row, bool from, uint32_t& set, uint32_t& dims, float& x, float& y, float& phase)
{
    const PlacementBlend* bp = blendFor(row);
    if ((from ? a.from : a.clip) != kBlendClip || !bp)
        return false;
    const PlacementBlend& b = *bp;
    set = from ? b.from : b.set, dims = from ? b.fromDims : b.dims, x = from ? b.fromX : b.x, y = from ? b.fromY : b.y;
    phase = from ? b.fromPhase : b.phase;
    return set != 0;
}
// What a placement's clip (from: the one it fades from) does over dt, as clipMotion says: a clip's, or its blend space's.
void playMotion(const Entry& e, uint32_t row, const PlacementAnim& a, bool from, const Entry* src, float dt, float* out)
{
    uint32_t set, dims;
    float    x, y, phase;
    BlendSet b;
    if (blendOf(a, row, from, set, dims, x, y, phase))
    {
        out[0] = out[1] = out[2] = out[3] = 0;
        const uint32_t h = row == kPreviewRow ? 0 : row | (Placements[row].gen & 0xFFFu) << 20;
        if (blendWeigh(e, row, h, set, dims, x, y, b) && b.length > 0)
            blendMotion(e, b, phase, dt * (from ? a.fromSpeed : a.speed) / b.length, from ? a.fromLoop : a.loop, out);
        return;
    }
    if (from)
        motionOf(e, src, a.from, a.fromTime, dt * a.fromSpeed, a.fromLoop, out);
    else
        motionOf(e, src, a.clip, a.time, dt * a.speed, a.loop, out);
}
// A placement leaving its blend space for a clip: the space kept as what it fades from (play()'s fade, if any).
void leaveBlend(uint32_t row, PlacementAnim& a)
{
    PlacementBlend* bp = blendFor(row);
    if (a.clip != kBlendClip || !bp)
        return;
    PlacementBlend& b = *bp;
    b.from = b.set, b.fromDims = b.dims, b.fromX = b.x, b.fromY = b.y, b.fromPhase = b.phase;
    b.set = 0, b.hash = 0;
}
// A placement (or the Viewer) taking blend space `set` (its samples already there) as its clip, faded from what played:
// another blend space in step with it, or a clip (`ready`: its models loaded, else nothing to fade from). rClip and
// rFrom: whose clips its clip and the one it fades from are (PlacementRetarget's, or the Viewer's).
void enterBlend(PlacementAnim& a, PlacementBlend& b, uint32_t& rClip, uint32_t& rFrom, uint32_t set, uint32_t dims, uint32_t hash,
                float speed, int loop, float fade, bool ready)
{
    if (a.clip == kBlendClip && b.set) // another blend space: faded from the old, in step with it
    {
        b.from = b.set, b.fromDims = b.dims, b.fromX = b.x, b.fromY = b.y, b.fromPhase = b.phase;
        a.from = fade > 0 ? kBlendClip : kNoClip, a.fromSpeed = a.speed, a.fromLoop = a.loop, a.fade = fade, a.faded = 0, rFrom = 0;
    }
    else // from a clip: faded from it (while the model still loads, nothing to fade from), the phase from the start
    {
        if (fade > 0 && ready && a.clip != kNoClip && !a.want)
            a.from = a.clip, a.fromTime = a.time, a.fromSpeed = a.speed, a.fromLoop = a.loop, a.fade = fade, a.faded = 0, rFrom = rClip;
        else
            a.from = kNoClip;
        b.phase = 0, b.x = b.tx, b.y = b.ty;
    }
    b.set = set, b.dims = dims, b.hash = hash;
    a.clip = kBlendClip, a.want = 0, a.time = 0, a.speed = speed, a.loop = loop ? 1 : 0, rClip = 0;
}
// Its blend space moved on by dt: the point followed toward the one asked for, and the phase (its own, and that of the
// one it fades from) by the weighted length of what plays, so every sample keeps step. h its handle (0 the Viewer's).
void stepBlend(const Entry& e, uint32_t row, uint32_t h, PlacementAnim& a, PlacementBlend& b, float dt)
{
    const float k = b.smooth > 0 ? 1 - expf(-3.0f * dt / b.smooth) : 1.0f; // most of the way in `smooth` s
    b.x += (b.tx - b.x) * k, b.y += (b.ty - b.y) * k;
    if (a.clip != kBlendClip)
        b.set = 0, b.hash = 0;
    if (a.from != kBlendClip)
        b.from = 0;
    BlendSet w;
    auto     step = [&](float& phase, float speed, uint32_t loop, uint32_t set, uint32_t dims, float x, float y) {
        if (!set || !blendWeigh(e, row, h, set, dims, x, y, w) || !(w.length > 0))
            return 0.0f;
        phase += dt * speed / w.length;
        phase = loop ? phase - floorf(phase) : fminf(fmaxf(phase, 0.0f), 1.0f);
        return phase * w.length;
    };
    if (b.set)
        a.time = step(b.phase, a.speed, a.loop, b.set, b.dims, b.x, b.y); // clipTime: its phase times its length
    if (b.from)
        step(b.fromPhase, a.fromSpeed, a.fromLoop, b.from, b.fromDims, b.fromX, b.fromY);
}

// --- Foot IK (mdl::footIk) -----------------------------------------------------------------------------------------
constexpr float kFootRate = 15.0f;      // 1/s: how fast a lifted foot's ground glides to a new one (a step's edge)
constexpr float kFootTilt = 0.7853982f; // the most a foot tilts to its ground (45 degrees)
bool gFootMoved; // foot IK, look-at or hand IK moved the placement evaluate() posed last (its bounds then come from its bones' boxes)

// Its front, arms and aiming chain (findLegs, from the chains it found and its rest pose in gRt).
void findUpper(Entry& e, const retarget::Found& f)
{
    static const float idq[4] = { 0, 0, 0, 1 }, Y[3] = { 0, 1, 0 }, back[3] = { 0, 0, -1 };
    e.arms = 0, e.aimCount = 0;
    const yea_float4 ut = upTurn(e);
    const float      uc[4] = { -ut.x, -ut.y, -ut.z, ut.w };
    float            upR[3], fr[3] = { 0, 0, 0 };
    retarget::qrot(uc, Y, upR);
    // its front: the way its toes point at rest (both feet, level), else the cooker's -Z
    for (int side = 0; side < 2; ++side)
        if (f.count[retarget::ChainToeL + side] && f.count[retarget::ChainLegL + side])
        {
            float a[3], b[3], d[3];
            retarget::at(gRt.tw, f.node[retarget::ChainLegL + side][f.count[retarget::ChainLegL + side] - 1], a);
            retarget::at(gRt.tw, f.node[retarget::ChainToeL + side][0], b);
            retarget::sub(b, a, d);
            const float u = retarget::dot(d, upR);
            for (int j = 0; j < 3; ++j)
                fr[j] += d[j] - u * upR[j];
        }
    if (!retarget::unit(fr))
        retarget::qrot(uc, back, fr);
    memcpy(e.front, fr, 12);
    // its arms: upper arm, forearm (the hand's parent) and hand
    bool arms = true;
    for (int side = 0; side < 2 && arms; ++side)
    {
        const uint32_t  count = f.count[retarget::ChainArmL + side];
        const uint32_t* chain = f.node[retarget::ChainArmL + side];
        Entry::Leg&     L = e.arm[side];
        arms = count >= 3;
        if (!arms)
            break;
        L.c = chain[count - 1], L.b = (uint32_t)e.nodes[L.c].parent, L.a = chain[0];
        arms = L.b != L.a;
        retarget::at(gRt.tw, L.c, L.rest);
        const int32_t pa = e.nodes[L.a].parent;
        L.hasParent = pa >= 0 ? 1 : 0;
        memcpy(L.parentRest, pa >= 0 ? &gRt.tq[pa * 4] : idq, 16);
    }
    e.arms = arms ? 1 : 0;
    // its aiming chain: the top two of its spine, then its neck to its head (the head last), each a share of the turn
    // (the head the most) and a limit (spine 15, neck 30, head 50 degrees)
    const uint32_t cs = f.count[retarget::ChainSpine], ch = f.count[retarget::ChainHead];
    if (!ch)
        return;
    for (uint32_t i = cs > 2 ? cs - 2 : 0; i < cs; ++i)
        e.aim[e.aimCount] = f.node[retarget::ChainSpine][i], e.aimShare[e.aimCount] = 1.0f, e.aimLimit[e.aimCount++] = 0.2617994f;
    for (uint32_t i = ch > 4 ? ch - 4 : 0; i < ch; ++i)
    {
        const bool head = i == ch - 1;
        e.aim[e.aimCount] = f.node[retarget::ChainHead][i], e.aimShare[e.aimCount] = head ? 2.0f : 1.5f;
        e.aimLimit[e.aimCount++] = head ? 0.8726646f : 0.5235988f;
    }
    // the head's forward in its own frame: the model's front, through its turn at rest
    float hc[4];
    retarget::qconj(&gRt.tq[e.aim[e.aimCount - 1] * 4], hc), retarget::qrot(hc, fr, e.aimFwd);
}
// Its legs, found from its bones' names as retargeting finds them (loadAnim, this module's update: it allocates).
void findLegs(Entry& e)
{
    e.legs = 0, e.pelvis = kNoClip, e.arms = 0, e.aimCount = 0;
    const uint32_t  n = e.skel->nodeCount;
    retarget::Name* names = (retarget::Name*)malloc(n * sizeof(retarget::Name));
    if (!names || !gLayerPoses || !gRt.tw)
        return free(names);
    for (uint32_t i = 0; i < n; ++i)
        retarget::parse(e.nodes[i].name, names[i]);
    retarget::Found f;
    retarget::chains(e.nodes, n, names, f);
    free(names);
    anim::restPoses(e.nodes, n, gLayerPoses);
    retarget::worlds(e.nodes, n, gLayerPoses, gRt.tw, gRt.tq);
    findUpper(e, f);
    for (int side = 0; side < 2; ++side)
    {
        const uint32_t  count = f.count[retarget::ChainLegL + side];
        const uint32_t* chain = f.node[retarget::ChainLegL + side];
        if (count < 3)
            return;
        Entry::Leg& L = e.leg[side];
        L.c = chain[count - 1], L.b = (uint32_t)e.nodes[L.c].parent, L.a = chain[0];
        if (L.b == L.a)
            return;
        retarget::at(gRt.tw, L.c, L.rest);
        const int32_t pa = e.nodes[L.a].parent;
        static const float idq[4] = { 0, 0, 0, 1 };
        L.hasParent = pa >= 0 ? 1 : 0;
        memcpy(L.parentRest, pa >= 0 ? &gRt.tq[pa * 4] : idq, 16);
    }
    // the pelvis both thighs hang from (its own bone, else their parent), the legs apart (neither inside the other)
    const uint32_t pelvis = f.count[retarget::ChainPelvis] ? f.node[retarget::ChainPelvis][0] : (uint32_t)e.nodes[e.leg[0].a].parent;
    uint32_t       tmp[retarget::kChainMax];
    if (pelvis >= n || !retarget::path(e.nodes, pelvis, e.leg[0].a, tmp) || !retarget::path(e.nodes, pelvis, e.leg[1].a, tmp) ||
        retarget::path(e.nodes, e.leg[0].a, e.leg[1].c, tmp) || retarget::path(e.nodes, e.leg[1].a, e.leg[0].c, tmp))
        return;
    e.pelvis = pelvis, e.legs = 1;
}

// Each bone's box from its model's vertices (positions, skin; morph refs and deltas, or nullptr), in this module's
// update (it allocates). A vertex counts for every bone it weighs on, grown by what its morph targets can move it (each
// delta at the larger of 1 and the furthest weight its clips reach).
void boneBoxes(Entry& e, const float* positions, const RmodelSkinVertex* skin, uint32_t vertices, const uint32_t* refs,
               const RmodelMorphDelta* deltas, uint32_t deltaCount)
{
    free(e.boneBox);
    const uint32_t nb = e.skel->boneCount;
    e.boneBox = nb ? (float*)malloc(nb * 6 * sizeof(float)) : nullptr;
    if (!e.boneBox)
        return;
    for (uint32_t b = 0; b < nb; ++b)
        for (int j = 0; j < 3; ++j)
            e.boneBox[b * 6 + j] = 1e30f, e.boneBox[b * 6 + 3 + j] = -1e30f;
    const uint32_t nt = e.targets ? e.skel->targetCount : 0;
    for (uint32_t v = 0; v < vertices; ++v)
    {
        float lo[3], hi[3], ext[3] = { 0, 0, 0 };
        if (refs && deltas)
            for (uint32_t k = refs[v] & 0xFFFFFFu, end = k + (refs[v] >> 24); k < end && k < deltaCount; ++k)
            {
                const uint32_t t = deltas[k].packed & 0xFFu;
                const float    w = t < nt ? fmaxf(1.0f, fmaxf(fabsf(e.targets[t].lo), fabsf(e.targets[t].hi))) : 1.0f;
                for (int j = 0; j < 3; ++j)
                    ext[j] += fabsf(deltas[k].p[j]) * w;
            }
        for (int j = 0; j < 3; ++j)
            lo[j] = positions[v * 3 + j] - ext[j], hi[j] = positions[v * 3 + j] + ext[j];
        for (int i = 0; i < 4; ++i)
            if (skin[v].weights[i] && skin[v].bones[i] < nb)
                for (int j = 0; j < 3; ++j)
                {
                    float* box = &e.boneBox[skin[v].bones[i] * 6];
                    box[j] = fminf(box[j], lo[j]), box[3 + j] = fmaxf(box[3 + j], hi[j]);
                }
    }
}
// The bounds of a pose (its palette, bones of e): each bone's box under its matrix, together. False: no boxes.
bool poseBounds(const Entry& e, const float* palette, float* lo, float* hi)
{
    for (int j = 0; j < 3; ++j)
        lo[j] = 1e30f, hi[j] = -1e30f;
    for (uint32_t b = 0; e.boneBox && b < e.skel->boneCount; ++b)
    {
        const float* box = &e.boneBox[b * 6];
        const float* m = &palette[b * 12];
        if (box[0] > box[3])
            continue;
        for (int c = 0; c < 8; ++c)
        {
            const float p[3] = { box[c & 1 ? 3 : 0], box[c & 2 ? 4 : 1], box[c & 4 ? 5 : 2] };
            for (int r = 0; r < 3; ++r)
            {
                const float x = m[r * 4] * p[0] + m[r * 4 + 1] * p[1] + m[r * 4 + 2] * p[2] + m[r * 4 + 3];
                lo[r] = fminf(lo[r], x), hi[r] = fmaxf(hi[r], x);
            }
        }
    }
    return lo[0] <= hi[0];
}

// Foot IK over a placement's pose in gPoses (local), as UE's foot placement does: each foot's ankle brought to where it
// stands on the ground the game gave under it (the clip's lift above its rest height kept, so a swinging foot still
// swings), the foot tilted to that ground while the clip has it down; the pelvis moved first to the lower foot's ground,
// and lowered further where a foot is still out of reach; then two-bone IK on each leg. On flat ground at the
// placement's height it changes nothing. dt < 0: as it stands (mdl::joint), its state left alone.
void footIk(const Entry& e, uint32_t row, float dt)
{
    if (!e.legs || row >= PlacementGrounds.size())
        return;
    PlacementGround& g = PlacementGrounds[row];
    const bool       step = dt >= 0;
    if (step)
    {
        const float to = g.on ? 1.0f : 0.0f, d = to - g.weight;
        g.weight = g.fade > 0 ? g.weight + (d > 0 ? 1.0f : -1.0f) * fminf(fabsf(d), dt / g.fade) : to;
        if (g.weight <= 0 && !g.on)
            g.weight = 0, g.pelvis = 0, g.seen = 0, g.at = 0;
    }
    if (g.weight <= 0 && !g.on)
        return;
    yea_float3 pos;
    yea_float4 rot;
    float      s;
    placedNow(row, e, pos, rot, s);
    if (!(s > 1e-6f))
        return;
    const uint32_t   n = e.skel->nodeCount;
    const yea_float4 rc = { -rot.x, -rot.y, -rot.z, rot.w }, ut = upTurn(e), uc = { -ut.x, -ut.y, -ut.z, ut.w };
    const float      P[3] = { pos.x, pos.y, pos.z }, Y[3] = { 0, 1, 0 }, back[3] = { 0, 0, -1 }, idq[4] = { 0, 0, 0, 1 };
    float            upM[3], upR[3], front[3]; // the world's up in model space; the model's own up and front there
    qrot(rc, Y, upM), qrot(uc, Y, upR), qrot(uc, back, front);
    retarget::worlds(e.nodes, n, gPoses, gRt.tw, gRt.tq);
    const float w = g.weight < 1 ? g.weight * g.weight * (3 - 2 * g.weight) : 1.0f; // eased, as clip fades
    const float k = step ? 1 - expf(-kFootRate * dt) : 0.0f;
    float* const gp[2] = { &g.pointL.x, &g.pointR.x }, *const gn[2] = { &g.normalL.x, &g.normalR.x };
    float* const sp[2] = { &g.planeL.x, &g.planeR.x }, *const sn[2] = { &g.upL.x, &g.upR.x };
    float* const at[2] = { &g.footL.x, &g.footR.x };
    const uint32_t seenBefore = g.seen;
    float          A[2][3], T[2][3], tilt[2][4];
    bool           has[2] = { false, false };
    for (int i = 0; i < 2; ++i)
    {
        const Entry::Leg& L = e.leg[i];
        const uint32_t    bit = 1u << i;
        float             Am[3];
        retarget::at(gRt.tw, L.c, Am);
        for (int j = 0; j < 3; ++j) // world
        {
            float v[3];
            qrot(rot, Am, v);
            A[i][j] = P[j] + s * v[j];
        }
        memcpy(T[i], A[i], 12), memcpy(tilt[i], idq, 16);
        if (step)
            memcpy(at[i], A[i], 12), g.at |= bit;
        // its ankle's height above the ground at rest, and how far the clip lifts it from there (both model space)
        const float h0 = fmaxf(s * retarget::dot(L.rest, upR), 0.0f), lift = s * retarget::dot(Am, upR) - h0;
        const float band = h0 > 0.01f * s ? h0 : 0.05f * s, x = fminf(fmaxf(lift / band, 0.0f), 1.0f);
        const float down = 1 - x * x * (3 - 2 * x); // 1 while the clip has the foot down, 0 once it lifts it an ankle's height
        const float kf = k + (1 - k) * down;         // a new ground: glided to in the air, stood on at once when down
        if (step && !(g.set & bit))
            g.seen &= ~bit;
        else if (step && !(g.seen & bit))
            memcpy(sp[i], gp[i], 12), memcpy(sn[i], gn[i], 12), g.seen |= bit;
        else if (step) // a new ground followed smoothly; the same plane, wherever its point is, stays the same plane
            for (int j = 0; j < 3; ++j)
                sp[i][j] += (gp[i][j] - sp[i][j]) * kf, sn[i][j] += (gn[i][j] - sn[i][j]) * kf;
        float nn[3] = { sn[i][0], sn[i][1], sn[i][2] };
        if (!(g.seen & bit) || !retarget::unit(nn) || nn[1] < 0.05f)
            continue; // no ground under it, or a wall: it follows its clip
        // the ground straight below the clip's ankle, and the foot's tilt to it (at most kFootTilt)
        const float Qy = sp[i][1] - (nn[0] * (A[i][0] - sp[i][0]) + nn[2] * (A[i][2] - sp[i][2])) / nn[1];
        float       nc[3] = { nn[0], nn[1], nn[2] };
        if (nc[1] < cosf(kFootTilt))
        {
            const float f = sinf(kFootTilt) / sqrtf(nc[0] * nc[0] + nc[2] * nc[2]);
            nc[0] *= f, nc[2] *= f, nc[1] = cosf(kFootTilt);
        }
        float       nb[3], t[3] = { A[i][0], Qy + lift, A[i][2] };
        for (int j = 0; j < 3; ++j)
            nb[j] = Y[j] + (nc[j] - Y[j]) * down;
        retarget::unit(nb);
        for (int j = 0; j < 3; ++j)
            t[j] += nb[j] * h0;
        if (!(fabsf(t[1] - A[i][1]) <= g.reach))
            continue; // ground further off than it may reach: left alone
        memcpy(T[i], t, 12);
        retarget::arc(Y, nb, tilt[i]);
        has[i] = true;
    }
    // the pelvis: to the lower foot's ground (followed smoothly; at once when the ground first comes)
    float target = 0;
    for (int i = 0, any = 0; i < 2; ++i)
        if (has[i])
            target = any++ ? fminf(target, T[i][1] - A[i][1]) : T[i][1] - A[i][1];
    if (step)
        g.pelvis = (seenBefore & 3) ? g.pelvis + (target - g.pelvis) * k : target;
    if (!has[0] && !has[1] && g.pelvis == 0)
        return;
    gFootMoved = true;
    float Gm[2][3], drop = 0;
    retarget::shift(e.nodes, n, e.pelvis, upM, g.pelvis * w / s, gRt.tw, gPoses);
    for (int i = 0; i < 2; ++i)
    {
        const Entry::Leg& L = e.leg[i];
        float             G[3], a[3], b[3], c[3];
        for (int j = 0; j < 3; ++j)
            G[j] = (A[i][j] + (T[i][j] - A[i][j]) * w - P[j]) / s;
        qrot(rc, G, Gm[i]);
        retarget::at(gRt.tw, L.a, a), retarget::at(gRt.tw, L.b, b), retarget::at(gRt.tw, L.c, c);
        const float x = retarget::reachDrop(a, b, c, Gm[i], upM);
        drop = x > drop ? x : drop;
    }
    if (drop > 0) // a foot still out of reach: the pelvis lowered by the least that brings both within it
        retarget::shift(e.nodes, n, e.pelvis, upM, -drop, gRt.tw, gPoses);
    const float rq[4] = { rot.x, rot.y, rot.z, rot.w }, rcq[4] = { rc.x, rc.y, rc.z, rc.w };
    for (int i = 0; i < 2; ++i)
    {
        const Entry::Leg& L = e.leg[i];
        float             q[4], m[4], end[4];
        retarget::qnlerp(idq, tilt[i], w, q);
        retarget::qmul(rcq, q, m), retarget::qmul(m, rq, m); // the tilt in model space
        retarget::qmul(m, &gRt.tq[L.c * 4], end), retarget::qnorm(end);
        retarget::limb(e.nodes, gRt.tw, gRt.tq, L.a, L.b, L.c, Gm[i], front, L.hasParent ? L.parentRest : nullptr, gPoses, end);
    }
}

// --- Look-at and hand IK (mdl::lookAt, mdl::handIk) -------------------------------------------------------------
// A weight faded toward `to` over `fade` seconds by dt (at once without a fade).
float fadeTo(float w, float to, float fade, float dt)
{
    const float d = to - w;
    return fade > 0 ? w + (d > 0 ? 1.0f : -1.0f) * fminf(fabsf(d), dt / fade) : to;
}
// The turn q scaled to `share` of its angle and kept within `most` radians.
void turnPart(float* q, float share, float most)
{
    if (q[3] < 0)
        for (int j = 0; j < 4; ++j)
            q[j] = -q[j];
    const float sh = sqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]);
    if (!(sh > 1e-9f))
    {
        q[0] = q[1] = q[2] = 0, q[3] = 1;
        return;
    }
    const float a = fminf(2 * atan2f(sh, q[3]) * share, most), k = sinf(a * 0.5f) / sh;
    q[0] *= k, q[1] *= k, q[2] *= k, q[3] = cosf(a * 0.5f);
}
// gRt's worlds of `from` and the nodes down to `to`, from gPoses, after from's pose changed (the rest stay as they were).
void chainWorlds(const Entry& e, uint32_t from, uint32_t to)
{
    uint32_t       p[retarget::kChainMax];
    const uint32_t c = retarget::path(e.nodes, from, to, p);
    if (!c)
        return retarget::worlds(e.nodes, to + 1, gPoses, gRt.tw, gRt.tq);
    for (uint32_t i = 0; i < c; ++i)
    {
        const uint32_t nd = p[i];
        const int32_t  pa = e.nodes[nd].parent;
        float          local[12];
        anim::poseMatrix(gPoses[nd], local);
        if (pa >= 0)
            anim::mul34(&gRt.tw[pa * 12], local, &gRt.tw[nd * 12]), retarget::qmul(&gRt.tq[pa * 4], gPoses[nd].r, &gRt.tq[nd * 4]);
        else
            memcpy(&gRt.tw[nd * 12], local, sizeof local), memcpy(&gRt.tq[nd * 4], gPoses[nd].r, 16);
        retarget::qnorm(&gRt.tq[nd * 4]);
    }
}
// Look-at over a placement's pose in gPoses (local), as UE's LookAt spread down a chain: from the top of the spine to
// the head, each bone turns its share of what is left between where the head faces and the target (the head the rest),
// within its limit; the target first brought within `limit` of where the clip has the head face. dt < 0: as it stands.
void aimIk(const Entry& e, uint32_t row, float dt)
{
    if (!e.aimCount || row >= PlacementAims.size())
        return;
    PlacementAim& g = PlacementAims[row];
    if (dt >= 0)
        g.weight = fadeTo(g.weight, g.on ? g.most : 0.0f, g.fade, dt);
    if (!(g.weight > 0))
        return;
    yea_float3 pos;
    yea_float4 rot;
    float      s;
    placedNow(row, e, pos, rot, s);
    if (!(s > 1e-6f))
        return;
    const uint32_t head = e.aim[e.aimCount - 1];
    const float    rc[4] = { -rot.x, -rot.y, -rot.z, rot.w }, v[3] = { (g.target.x - pos.x) / s, (g.target.y - pos.y) / s, (g.target.z - pos.z) / s };
    float          T[3], keep[6][4], f0[3], remaining = 0;
    retarget::qrot(rc, v, T); // the target, model space
    retarget::worlds(e.nodes, head + 1, gPoses, gRt.tw, gRt.tq); // parents first: all the chain needs
    retarget::qrot(&gRt.tq[head * 4], e.aimFwd, f0); // where the clip has the head face
    for (uint32_t k = 0; k < e.aimCount; ++k)
        memcpy(keep[k], gPoses[e.aim[k]].r, 16), remaining += e.aimShare[k];
    for (uint32_t k = 0; k < e.aimCount; ++k)
    {
        float H[3], to[3], fw[3], q[4];
        retarget::at(gRt.tw, head, H), retarget::sub(T, H, to), retarget::qrot(&gRt.tq[head * 4], e.aimFwd, fw);
        if (!retarget::unit(to))
            break;
        // within the limit of where the clip faces
        const float c = retarget::dot(to, f0);
        if (c < cosf(g.limit))
        {
            float side[3];
            for (int j = 0; j < 3; ++j)
                side[j] = to[j] - c * f0[j];
            static const float X[3] = { 1, 0, 0 }, Y[3] = { 0, 1, 0 };
            if (!retarget::unit(side)) // straight behind: turn across it, level where it can
                retarget::cross(f0, Y, side), retarget::unit(side) || (retarget::cross(f0, X, side), retarget::unit(side));
            for (int j = 0; j < 3; ++j)
                to[j] = cosf(g.limit) * f0[j] + sinf(g.limit) * side[j];
        }
        retarget::arc(fw, to, q);
        turnPart(q, e.aimShare[k] / remaining, e.aimLimit[k]);
        remaining -= e.aimShare[k];
        const uint32_t nd = e.aim[k];
        const int32_t  pa = e.nodes[nd].parent;
        static const float idq[4] = { 0, 0, 0, 1 };
        float          m[4], pc[4];
        retarget::qmul(q, &gRt.tq[nd * 4], m);
        retarget::qconj(pa >= 0 ? &gRt.tq[pa * 4] : idq, pc), retarget::qmul(pc, m, gPoses[nd].r), retarget::qnorm(gPoses[nd].r);
        if (k + 1 < e.aimCount)
            chainWorlds(e, nd, head);
    }
    const float w = g.weight < 1 ? g.weight * g.weight * (3 - 2 * g.weight) : 1.0f; // eased, as clip fades
    for (uint32_t k = 0; w < 1 && k < e.aimCount; ++k)
        retarget::qnlerp(keep[k], gPoses[e.aim[k]].r, w, gPoses[e.aim[k]].r);
    gFootMoved = true;
}
// Hand IK over gPoses: each hand on brought (eased by its weight) from where the clip put it to its point, turned to
// its turn if given, by two-bone IK on its arm (the elbow bending as the clip bends it, else to the model's back).
void handIk(const Entry& e, uint32_t row, float dt)
{
    if (!e.arms || row >= PlacementHands.size())
        return;
    PlacementHand& g = PlacementHands[row];
    if (dt >= 0)
        g.weightL = fadeTo(g.weightL, g.on & 1 ? 1.0f : 0.0f, g.fadeL, dt), g.weightR = fadeTo(g.weightR, g.on & 2 ? 1.0f : 0.0f, g.fadeR, dt);
    if (!(g.weightL > 0) && !(g.weightR > 0))
        return;
    yea_float3 pos;
    yea_float4 rot;
    float      s;
    placedNow(row, e, pos, rot, s);
    if (!(s > 1e-6f))
        return;
    const float    rc[4] = { -rot.x, -rot.y, -rot.z, rot.w };
    const float    bend[3] = { -e.front[0], -e.front[1], -e.front[2] }; // elbows to its back
    retarget::worlds(e.nodes, (e.arm[0].c > e.arm[1].c ? e.arm[0].c : e.arm[1].c) + 1, gPoses, gRt.tw, gRt.tq);
    for (int i = 0; i < 2; ++i)
    {
        const float wt = i ? g.weightR : g.weightL;
        if (!(wt > 0))
            continue;
        const float        w = wt < 1 ? wt * wt * (3 - 2 * wt) : 1.0f;
        const Entry::Leg&  L = e.arm[i];
        const yea_float3&  P = i ? g.pointR : g.pointL;
        const yea_float4&  R = i ? g.rotR : g.rotL;
        const float        v[3] = { (P.x - pos.x) / s, (P.y - pos.y) / s, (P.z - pos.z) / s };
        float              G[3], C[3], end[4];
        retarget::qrot(rc, v, G), retarget::at(gRt.tw, L.c, C);
        for (int j = 0; j < 3; ++j)
            G[j] = C[j] + (G[j] - C[j]) * w;
        const bool turned = (g.turned >> i) & 1;
        if (turned) // its world turn in model space, eased from the clip's
        {
            const float r[4] = { R.x, R.y, R.z, R.w };
            float       m[4];
            retarget::qmul(rc, r, m), retarget::qnorm(m), retarget::qnlerp(&gRt.tq[L.c * 4], m, w, end);
        }
        retarget::limb(e.nodes, gRt.tw, gRt.tq, L.a, L.b, L.c, G, bend, L.hasParent ? L.parentRest : nullptr, gPoses, turned ? end : nullptr);
    }
    gFootMoved = true;
}

// A layer over gPoses: clip `clip` of e's source `source` at `time`, from node `mask` down (kNoClip: every node),
// overriding the pose there or added to it, at `weight` (eased, as a layer's fade is). False: no such clip.
bool applyLayer(const Entry& e, uint32_t source, uint32_t clip, float time, bool loop, bool still, uint32_t mask, float weight,
                bool additive)
{
    const uint32_t n = e.skel->nodeCount;
    const Entry*   s = sourceOf(e, source);
    if (!s || clip >= s->skel->clipCount || !sampleAs(e, s, clip, time, loop, still, gLayerPoses, nullptr))
        return false;
    const float w = weight < 1 ? weight * weight * (3 - 2 * weight) : 1.0f; // eased, as clip fades
    anim::layerMask(e.nodes, n, mask, gLayerMask);
    if (additive)
    {
        // (a retargeted clip's also by both models' Up Axis, as its map is)
        const uint32_t up = s == &e ? 0u : uint32_t(e.knobs[mdl::KnobUpAxis]) * 3u + uint32_t(s->knobs[mdl::KnobUpAxis]) + 1u;
        const LayerRef want = { e.key, source, clip, (still ? 1u : 0u) | up << 1, e.animStamp * 65599u ^ s->animStamp, 1 };
        uint32_t       k = 0;
        while (k < kLayerRefs && memcmp(&gLayerRefKey[k], &want, sizeof want))
            ++k;
        if (k == kLayerRefs) // not kept yet: sampled into the oldest slot
        {
            k = gLayerRefNext, gLayerRefNext = (gLayerRefNext + 1) % kLayerRefs;
            gLayerRefKey[k] = LayerRef{};
            if (!sampleAs(e, s, clip, 0, false, still, gLayerRefs + k * kMaxNodes, nullptr))
                return true;
            gLayerRefKey[k] = want;
        }
        anim::layerAdd(gPoses, gLayerPoses, gLayerRefs + k * kMaxNodes, gLayerMask, n, w);
    }
    else
        anim::layerOverride(gPoses, gLayerPoses, gLayerMask, n, w);
    return true;
}

// An inertialized transition's slot: a placement's (nullptr: none), or one taken for it with room for n nodes and nt
// morph targets (nullptr: the pool or the slots are full, so it crossfades).
InertSlot* inertFind(uint32_t placement)
{
    for (uint32_t i = 0; i < gInertCount; ++i)
        if (gInertSlot[i].placement == placement)
            return &gInertSlot[i];
    return nullptr;
}
InertSlot* inertTake(uint32_t placement, uint32_t n, uint32_t nt)
{
    const uint32_t need = n * 18 + nt * 2;
    if (!gInertPool || gInertCount >= kInertSlots || need > kInertFloats)
        return nullptr;
    uint32_t at = 0; // the first gap it fits in
    for (bool moved = true; moved;)
    {
        moved = false;
        for (uint32_t i = 0; i < gInertCount; ++i)
        {
            const InertSlot& o = gInertSlot[i];
            if (at < o.start + o.nodes * 18 + o.targets * 2 && o.start < at + need)
                at = o.start + o.nodes * 18 + o.targets * 2, moved = true;
        }
    }
    if (at + need > kInertFloats)
        return nullptr;
    InertSlot& sl = gInertSlot[gInertCount++];
    sl = InertSlot{ placement, kNoClip, kNoClip, at, n, nt, 0, 0, 0 };
    return &sl;
}

// One placement's pose now into gPoses (its clip, faded from the one before), and its morph weights into gWeights
// (the same, then what the game holds: `held` weights, `isHeld` where it does): false when it isn't skinned.
// clipKey and fromKey: the models (keys; 0 its own) the clip and the one it fades from are of.
bool evaluate(const Entry& e, const PlacementAnim& a, const float* held = nullptr, const uint8_t* isHeld = nullptr, uint32_t placement = 0,
              bool still = false, uint32_t clipKey = 0, uint32_t fromKey = 0, float dt = -1,
              const Preview* view = nullptr) // still: in place (root motion on); view: the Model Viewer's, with its layer
{
    const uint32_t n = e.skel->nodeCount, nt = e.targets ? e.skel->targetCount : 0;
    const bool     fading = a.from != kNoClip && a.fade > 0 && a.faded < a.fade;
    const uint32_t row = placement ? placementRow(placement) : view ? kPreviewRow : 0xFFFFFFFFu;
    // its clip (or the one it fades from), or its blend space, `shift` seconds of it from now
    auto side = [&](bool from, anim::Pose* out, float* w, float shift) {
        uint32_t set, dims;
        float    x, y, phase;
        BlendSet b;
        const float  speed = from ? a.fromSpeed : a.speed;
        const bool   loop = from ? a.fromLoop : a.loop;
        if (row != 0xFFFFFFFFu && blendOf(a, row, from, set, dims, x, y, phase))
        {
            const bool weighed = blendWeigh(e, row, placement, set, dims, x, y, b);
            if (weighed && shift != 0 && b.length > 0)
            {
                phase += shift * speed / b.length;
                phase = loop ? phase - floorf(phase) : (phase < 0 ? 0 : (phase > 1 ? 1 : phase));
            }
            if (!weighed || !blendSample(e, b, phase, loop, still, out, w))
                sampleAs(e, nullptr, 0, 0, false, false, out, w); // nothing to weigh yet: the rest pose
        }
        else if (from)
            sampleAs(e, sourceOf(e, fromKey), a.from, a.fromTime + shift * speed, loop, still, out, w);
        else
            sampleAs(e, sourceOf(e, clipKey), a.clip, a.time + shift * speed, loop, still, out, w);
    };
    side(false, gPoses, nt ? gWeights : nullptr, 0);
    // Inertialized: the old pose's offsets from the new, taken on its first pose after the switch (the old one as it
    // showed, a transition still under way included), then only the new clip sampled.
    InertSlot* inert = nullptr;
    if (fading && settings.transitions == Transitions::Inertial && gInertPool)
    {
        InertSlot* sl = inertFind(placement);
        if (sl && (sl->nodes != n || sl->targets != nt)) // another model now: start afresh
            *sl = gInertSlot[--gInertCount], sl = nullptr;
        const bool fresh = !sl || sl->clip != a.clip || sl->from != a.from || a.faded < sl->last;
        const InertSlot was = sl ? *sl : InertSlot{ 0xFFFFFFFFu };
        if (!sl)
            sl = inertTake(placement, n, nt);
        if (sl && fresh)
        {
            const float h = kInertStep;
            side(false, gInertNew, nt ? gInertNewW : nullptr, h);
            side(true, gFrom, nt ? gFromWeights : nullptr, 0);
            side(true, gInertOld, nt ? gInertOldW : nullptr, -h);
            anim::Inertial* off = (anim::Inertial*)(gInertPool + sl->start);
            float*          wo = gInertPool + sl->start + n * 18;
            if (was.placement == placement && was.t1 > 0) // the transition it cut short, as it stood
            {
                const float t = was.last - was.base + a.faded;
                anim::inertialApply(gFrom, off, n, t, was.t1);
                anim::inertialApply(gInertOld, off, n, t - h, was.t1);
                for (uint32_t k = 0; k < nt; ++k)
                    gFromWeights[k] += anim::inertialDecay(wo[k * 2], wo[k * 2 + 1], t, was.t1),
                        gInertOldW[k] += anim::inertialDecay(wo[k * 2], wo[k * 2 + 1], t - h, was.t1);
            }
            anim::inertialOffsets(gFrom, gInertOld, gPoses, gInertNew, h, n, off);
            for (uint32_t k = 0; k < nt; ++k)
            {
                const float x = gFromWeights[k] - gWeights[k];
                wo[k * 2] = x, wo[k * 2 + 1] = (x - (gInertOldW[k] - (2 * gWeights[k] - gInertNewW[k]))) / h;
            }
            sl->clip = a.clip, sl->from = a.from, sl->base = a.faded, sl->t1 = a.fade - a.faded;
        }
        if (sl)
            sl->last = a.faded, inert = sl;
    }
    const bool cross = fading && !inert;
    if (cross)
        side(true, gFrom, nt ? gFromWeights : nullptr, 0);
    if (nt)
    {
        if (inert)
        {
            const float* wo = gInertPool + inert->start + n * 18;
            for (uint32_t t = 0; t < nt; ++t)
                gWeights[t] += anim::inertialDecay(wo[t * 2], wo[t * 2 + 1], a.faded - inert->base, inert->t1);
        }
        if (cross)
        {
            float f = a.faded / a.fade;
            f = f * f * (3 - 2 * f);
            for (uint32_t t = 0; t < nt; ++t)
                gWeights[t] = gFromWeights[t] + (gWeights[t] - gFromWeights[t]) * f;
        }
        for (uint32_t t = 0; held && t < nt; ++t)
            if (isHeld[t])
                gWeights[t] = held[t];
        for (uint32_t i = 0; placement && i < MorphOverrides.size(); ++i)
            if (MorphOverrides[i].placement == placement && MorphOverrides[i].target < nt)
                gWeights[MorphOverrides[i].target] = MorphOverrides[i].weight;
    }
    if (cross)
    {
        float f = a.faded / a.fade;
        f = f * f * (3 - 2 * f); // eased
        anim::blendPoses(gFrom, gPoses, n, f);
        memcpy(gPoses, gFrom, sizeof(anim::Pose) * n);
    }
    if (inert)
        anim::inertialApply(gPoses, (const anim::Inertial*)(gInertPool + inert->start), n, a.faded - inert->base, inert->t1);
    // Its layers over that, in slot order (older first within a slot, so a slot's new clip crossfades over its old).
    uint32_t order[16], count = 0;
    for (uint32_t i = 0; placement && gLayerPoses && i < PlacementLayers.size() && count < 16; ++i)
    {
        const PlacementLayer& L = PlacementLayers[i];
        if (L.placement != placement || L.weight <= 0 || L.want || L.mask == kMaskPending)
            continue;
        uint32_t k = count++;
        for (; k > 0; --k) // insertion by (slot, seq)
        {
            const PlacementLayer& P = PlacementLayers[order[k - 1]];
            if (P.slot < L.slot || (P.slot == L.slot && P.seq < L.seq))
                break;
            order[k] = order[k - 1];
        }
        order[k] = i;
    }
    for (uint32_t k = 0; k < count; ++k)
    {
        const PlacementLayer& L = PlacementLayers[order[k]];
        applyLayer(e, L.source, L.clip, L.time, L.loop, still, L.mask, L.weight, L.additive != 0);
    }
    if (view && view->layerClip != kNoClip && view->layerWeight > 0 && gLayerPoses) // the Model Viewer's layer
        applyLayer(e, view->layerSource, view->layerClip, view->layerTime, true, still, view->layerBone, view->layerWeight,
                   view->layerAdditive != 0);
    gFootMoved = false;
    if (placement) // its feet on the game's ground (dt: moved on in animate(); -1 as it stands)
        footIk(e, placementRow(placement), dt), aimIk(e, placementRow(placement), dt), handIk(e, placementRow(placement), dt);
    anim::palette(e.nodes, n, e.bones, e.skel->boneCount, gPoses, gWorld, gPalette);
    return true;
}
int        apiJoint(uint32_t h, const char* name, float* out)
{
    PlacementAnim* a;
    Entry*         e = animOf(h, &a);
    if (!e || !name || !out)
        return 0;
    uint32_t node = kNoClip;
    for (uint32_t i = 0; i < e->skel->nodeCount && node == kNoClip; ++i)
        if (!strncmp(e->nodes[i].name, name, sizeof e->nodes[i].name))
            node = i;
    if (node == kNoClip)
        return 0;
    const uint32_t row = placementRow(h);
    const bool     re = row < PlacementRetargets.size();
    evaluate(*e, *a, nullptr, nullptr, h, row < PlacementMotions.size() && PlacementMotions[row].on, re ? PlacementRetargets[row].clip : 0,
             re ? PlacementRetargets[row].from : 0);
    yea_float3 pos;
    yea_float4 r;
    float      s;
    placedNow(row, *e, pos, r, s);
    const anim::Pose placed = { { pos.x, pos.y, pos.z }, { r.x, r.y, r.z, r.w }, { s, s, s } };
    float            m[12];
    anim::poseMatrix(placed, m);
    anim::mul34(m, &gWorld[node * 12], out);
    return 1;
}
void     apiUnlayer(uint32_t h, uint32_t slot, float fade);
uint32_t nodeNamed(const Entry& e, const char* name)
{
    for (uint32_t i = 0; name && e.nodes && i < e.skel->nodeCount; ++i)
        if (!strncmp(e.nodes[i].name, name, sizeof e.nodes[i].name))
            return i;
    return kNoClip;
}
int apiLayer(uint32_t h, uint32_t slot, uint32_t model, const char* clip, const char* bone, float weight, int additive, float speed,
             int loop, float fade)
{
    const uint32_t row = placementRow(h);
    const Entry*   t = row != 0xFFFFFFFFu ? byKey(Placements[row].key) : nullptr;
    const Entry*   s = model ? byId(model) : t;
    if (!t || !s || !clip || !clip[0] || t->state == 2 || s->state == 2 || (t->state == 1 && !t->anim) || (s->state == 1 && !s->anim))
        return 0;
    if (!(weight > 0))
    {
        apiUnlayer(h, slot, fade);
        return 1;
    }
    const bool     ready = t->state == 1 && s->state == 1;
    const uint32_t key = s == t ? 0 : s->key;
    uint32_t       c = kNoClip, mask = bone && bone[0] ? kMaskPending : kNoClip;
    if (ready)
    {
        c = clipNamed(*s, clip);
        if (c == kNoClip)
            return 0;
        if (bone && bone[0] && (mask = nodeNamed(*t, bone)) == kNoClip)
            return 0;
    }
    const uint32_t want = ready ? 0 : clipHash(clip, 48), maskHash = bone && bone[0] ? clipHash(bone, 48) : 0;
    uint32_t       seq = 0;
    for (uint32_t i = 0; i < PlacementLayers.size(); ++i)
    {
        PlacementLayer& L = PlacementLayers[i];
        if (L.placement != h || L.slot != slot)
            continue;
        seq = L.seq + 1 > seq ? L.seq + 1 : seq;
        if (L.target <= 0)
            continue; // already leaving
        if (L.source == key && L.clip == c && L.want == want && L.maskHash == maskHash && L.additive == (additive ? 1u : 0u))
        {
            L.target = weight, L.fade = fade, L.speed = speed, L.loop = loop ? 1 : 0; // the same layer: re-weighed, its time kept
            return 1;
        }
        L.target = 0, L.fade = fade; // another clip in the slot: the old fades out under the new
    }
    PlacementLayer* L = PlacementLayers.add();
    if (!L)
        return 0;
    *L = PlacementLayer{};
    L->placement = h, L->slot = slot, L->seq = seq, L->source = key, L->clip = c, L->want = want, L->mask = mask,
    L->maskHash = maskHash, L->additive = additive ? 1 : 0, L->speed = speed, L->loop = loop ? 1 : 0;
    L->weight = fade > 0 ? 0.0f : weight, L->target = weight, L->fade = fade;
    return 1;
}
void apiUnlayer(uint32_t h, uint32_t slot, float fade)
{
    for (uint32_t i = 0; i < PlacementLayers.size(); ++i)
        if (PlacementLayers[i].placement == h && PlacementLayers[i].slot == slot)
            PlacementLayers[i].target = 0, PlacementLayers[i].fade = fade;
}
uint32_t morphNamed(const Entry& e, const char* name)
{
    for (uint32_t t = 0; e.targets && name && t < e.skel->targetCount; ++t)
        if (!strncmp(e.targets[t].name, name, sizeof e.targets[t].name))
            return t;
    return kNoClip;
}
int apiMorph(uint32_t h, const char* target, float weight)
{
    PlacementAnim* a;
    Entry*         e = animOf(h, &a);
    const uint32_t t = e ? morphNamed(*e, target) : kNoClip;
    if (t == kNoClip || !(weight == weight))
        return 0;
    for (uint32_t i = 0; i < MorphOverrides.size(); ++i)
        if (MorphOverrides[i].placement == h && MorphOverrides[i].target == t)
            return MorphOverrides[i].weight = weight, 1;
    MorphOverride* o = MorphOverrides.add();
    if (!o)
        return 0;
    o->placement = h, o->target = t, o->weight = weight;
    return 1;
}
void apiUnmorph(uint32_t h, const char* target)
{
    PlacementAnim* a;
    Entry*         e = animOf(h, &a);
    const uint32_t t = e && target ? morphNamed(*e, target) : kNoClip;
    for (uint32_t i = MorphOverrides.size(); i-- > 0;)
        if (MorphOverrides[i].placement == h && (!target || MorphOverrides[i].target == t))
            MorphOverrides.remove(i);
}
uint32_t apiMorphs(uint32_t model)
{
    const Entry* e = byId(model);
    return e && e->targets ? e->skel->targetCount : 0;
}
const char* apiMorphTarget(uint32_t model, uint32_t index, float* weight)
{
    const Entry* e = byId(model);
    if (!e || !e->targets || index >= e->skel->targetCount)
        return nullptr;
    if (weight)
        *weight = e->targets[index].weight;
    return e->targets[index].name;
}
int apiRootMotion(uint32_t h, int on)
{
    // Taken while its model still loads (as a clip asked for by name is); gathering starts with its first frame.
    const uint32_t row = placementRow(h);
    grow(row);
    const Entry*   e = row != 0xFFFFFFFFu ? byKey(Placements[row].key) : nullptr;
    if (!e || row >= PlacementMotions.size() || (e->state == 1 && !e->anim) || e->state > 1)
        return 0;
    PlacementMotion& m = PlacementMotions[row];
    const bool       was = m.on != 0;
    m.on = on ? 1 : 0, m.acc = { 0, 0, 0 }, m.yaw = 0;
    // Switched on partway through a clip: what it did since its first frame is pending, so the model stays where it
    // stands (in place, the node goes back to that frame's spot) and the game's first take moves it there.
    if (on && !was && e->state == 1 && row < PlacementAnims.size() && row < PlacementRetargets.size())
    {
        const PlacementAnim& a = PlacementAnims[row];
        float                d[4];
        uint32_t             set, dims;
        float                x, y, phase;
        BlendSet             b;
        if (blendOf(a, row, false, set, dims, x, y, phase)) // a blend space: what its samples did since its phase's start
        {
            d[0] = d[1] = d[2] = d[3] = 0;
            if (blendWeigh(*e, row, h, set, dims, x, y, b))
                blendMotion(*e, b, 0, phase, false, d);
        }
        else
            motionOf(*e, sourceOf(*e, PlacementRetargets[row].clip), a.clip, 0, a.time, false, d);
        m.acc = { d[0], d[1], d[2] }, m.yaw = d[3];
    }
    Placements[row].dirty = 1;
    return 1;
}
int apiTakeMotion(uint32_t h, float* out)
{
    const uint32_t row = placementRow(h);
    out[0] = out[1] = out[2] = out[3] = 0;
    if (row >= PlacementMotions.size() || !PlacementMotions[row].on)
        return 0;
    const Entry* e = byKey(Placements[row].key);
    if (e)
        pendingMotion(row, *e, out);
    PlacementMotion& m = PlacementMotions[row];
    m.acc = { 0, 0, 0 }, m.yaw = 0;
    Placements[row].dirty = 1; // drawn where the game puts it now
    return 1;
}
int apiFootIk(uint32_t h, int on, float reach, float fade)
{
    // Taken while its model still loads; 0 once it has loaded without two legs.
    const uint32_t row = placementRow(h);
    grow(row);
    const Entry* e = row != 0xFFFFFFFFu ? byKey(Placements[row].key) : nullptr;
    if (!e || row >= PlacementGrounds.size() || e->state > 1 || (e->state == 1 && !e->legs))
        return 0;
    PlacementGround& g = PlacementGrounds[row];
    g.on = on ? 1 : 0, g.reach = reach > 0 ? reach : 0.5f, g.fade = fade > 0 ? fade : 0.0f;
    return 1;
}
int apiLookAt(uint32_t h, int on, const float* target, float weight, float limit, float fade)
{
    // Taken while its model still loads; 0 once it has loaded without a head chain.
    const uint32_t row = placementRow(h);
    grow(row);
    const Entry* e = row != 0xFFFFFFFFu ? byKey(Placements[row].key) : nullptr;
    if (!e || row >= PlacementAims.size() || e->state > 1 || (e->state == 1 && !e->aimCount))
        return 0;
    PlacementAim& g = PlacementAims[row];
    g.on = on ? 1 : 0, g.fade = fade > 0 ? fade : 0.0f;
    if (on)
        g.most = weight > 0 ? (weight < 1 ? weight : 1.0f) : 0.0f, g.limit = limit > 0 ? fminf(limit, 3.1415926f) : 1.2217305f;
    if (target)
        g.target = { target[0], target[1], target[2] };
    return 1;
}
int apiHandIk(uint32_t h, uint32_t hand, int on, const float* point, const float* rot, float fade)
{
    const uint32_t row = placementRow(h);
    grow(row);
    const Entry* e = row != 0xFFFFFFFFu ? byKey(Placements[row].key) : nullptr;
    if (!e || hand > 1 || row >= PlacementHands.size() || e->state > 1 || (e->state == 1 && !e->arms))
        return 0;
    PlacementHand& g = PlacementHands[row];
    const uint32_t bit = 1u << hand;
    g.on = on ? g.on | bit : g.on & ~bit;
    (hand ? g.fadeR : g.fadeL) = fade > 0 ? fade : 0.0f;
    if (on && point)
        (hand ? g.pointR : g.pointL) = { point[0], point[1], point[2] };
    if (on && rot)
        (hand ? g.rotR : g.rotL) = { rot[0], rot[1], rot[2], rot[3] }, g.turned |= bit;
    else if (on)
        g.turned &= ~bit;
    return 1;
}
int apiFootGround(uint32_t h, uint32_t foot, const float* point, const float* normal)
{
    const uint32_t row = placementRow(h);
    grow(row);
    if (row >= PlacementGrounds.size() || foot > 1)
        return 0;
    PlacementGround& g = PlacementGrounds[row];
    if (!point)
        return g.set &= ~(1u << foot), 1;
    const float up[3] = { 0, 1, 0 };
    const float* nr = normal ? normal : up;
    (foot ? g.pointR : g.pointL) = { point[0], point[1], point[2] };
    (foot ? g.normalR : g.normalL) = { nr[0], nr[1], nr[2] };
    g.set |= 1u << foot;
    return 1;
}
int apiFootAt(uint32_t h, uint32_t foot, float* out)
{
    const uint32_t row = placementRow(h);
    if (row >= PlacementGrounds.size() || foot > 1 || !(PlacementGrounds[row].at & (1u << foot)))
        return 0;
    const yea_float3& f = foot ? PlacementGrounds[row].footR : PlacementGrounds[row].footL;
    out[0] = f.x, out[1] = f.y, out[2] = f.z;
    return 1;
}
int apiBlendSpace(uint32_t h, uint32_t model, const mdl::BlendSample* samples, uint32_t count, uint32_t dims, float speed, int loop,
                  float fade)
{
    const uint32_t row = placementRow(h);
    grow(row);
    const Entry* t = row != 0xFFFFFFFFu ? byKey(Placements[row].key) : nullptr;
    const Entry* s = model ? byId(model) : t;
    if (!t || !s || !samples || !count || count > blend::kMaxSamples || (dims != 1 && dims != 2) || row >= PlacementAnims.size() ||
        row >= PlacementBlends.size() || row >= PlacementRetargets.size() || t->state == 2 || s->state == 2 ||
        (t->state == 1 && !t->anim) || (s->state == 1 && !s->anim))
        return 0;
    const bool     ready = t->state == 1 && s->state == 1;
    const uint32_t key = s == t ? 0 : s->key;
    uint32_t       clips[blend::kMaxSamples], hash = 2166136261u ^ key ^ dims * 16777619u;
    for (uint32_t i = 0; i < count; ++i)
    {
        if (!samples[i].clip || !samples[i].clip[0])
            return 0;
        clips[i] = ready ? clipNamed(*s, samples[i].clip) : kNoClip;
        if (ready && clips[i] == kNoClip)
            return 0;
        uint32_t xy[2];
        memcpy(xy, &samples[i].x, 4), memcpy(xy + 1, &samples[i].y, 4);
        hash = ((hash ^ clipHash(samples[i].clip, 48)) * 16777619u ^ xy[0]) * 16777619u ^ xy[1];
    }
    hash = hash ? hash : 1;
    PlacementAnim&     a = PlacementAnims[row];
    PlacementBlend&    b = PlacementBlends[row];
    PlacementRetarget& r = PlacementRetargets[row];
    if (a.clip == kBlendClip && b.set && b.hash == hash) // the same samples: playing on, its phase kept
    {
        a.speed = speed, a.loop = loop ? 1 : 0;
        return 1;
    }
    if (PlacementBlendSamples.size() + count > PlacementBlendSamples.capacity())
        return 0;
    const uint32_t set = ++b.next ? b.next : ++b.next;
    for (uint32_t i = 0; i < count; ++i)
    {
        PlacementBlendSample* S = PlacementBlendSamples.add();
        *S = PlacementBlendSample{};
        S->placement = h, S->set = set, S->source = key, S->clip = clips[i], S->want = ready ? 0 : clipHash(samples[i].clip, 48);
        S->x = samples[i].x, S->y = samples[i].y;
    }
    enterBlend(a, b, r.clip, r.from, set, dims, hash, speed, loop, fade, ready);
    return 1;
}
void apiBlendAt(uint32_t h, float x, float y, float smooth)
{
    const uint32_t row = placementRow(h);
    grow(row);
    if (row >= PlacementBlends.size())
        return;
    PlacementBlend& b = PlacementBlends[row];
    b.tx = x, b.ty = y, b.smooth = smooth > 0 ? smooth : 0.0f;
    if (!(smooth > 0) || !b.set)
        b.x = x, b.y = y; // at once (and where a blend space starts)
}
uint32_t apiClips(uint32_t model)
{
    const Entry* e = byId(model);
    return e && e->anim ? e->skel->clipCount : 0;
}
const char* apiClip(uint32_t model, uint32_t index, float* duration)
{
    const Entry* e = byId(model);
    if (!e || !e->anim || index >= e->skel->clipCount)
        return nullptr;
    if (duration)
        *duration = e->clips[index].duration;
    return e->clips[index].name;
}
mdl::Api gApi = { mdl::kApi, sizeof(mdl::Api), 0,       apiBegin, apiDefine, apiEnd,   apiKnobs, apiPlace, apiMove,
                  apiRemove, apiPlaced,        apiCount, apiInfo,  apiPlay,   apiTime,  apiJoint, apiClips, apiClip,
                  apiMorph,  apiUnmorph,       apiMorphs, apiMorphTarget, apiRootMotion, apiTakeMotion, apiPlayFrom,
                  apiLayer,  apiUnlayer,       apiFootIk, apiFootGround, apiFootAt,
                  apiBlendSpace, apiBlendAt, apiLookAt, apiHandIk };

// --- Building models in the renderer's scene (this module's own update) ------------------------------------------

// A built-in shape's mesh, or kNone for a source that isn't one.
uint32_t builtin(const char* s)
{
    if (!strcmp(s, "box"))
        return renderer::mesh_box();
    if (!strcmp(s, "sphere"))
        return renderer::mesh_sphere(64);
    if (!strcmp(s, "cylinder"))
        return renderer::mesh_cylinder(48);
    if (!strcmp(s, "cone"))
        return renderer::mesh_cone(48);
    if (!strcmp(s, "torus"))
        return renderer::mesh_torus(0.15f, 64);
    if (!strcmp(s, "plane"))
        return renderer::mesh_plane();
    return 0xFFFFFFFEu;
}

// The bounds of a mesh's vertices (object units), from the renderer's Vertex table.
void bounds(uint32_t mesh, float lo[3], float hi[3])
{
    lo[0] = lo[1] = lo[2] = 0, hi[0] = hi[1] = hi[2] = 0;
    if (mesh >= renderer::Meshes.size())
        return;
    const renderer::Mesh& m = renderer::Meshes[mesh];
    bool                  any = false;
    for (uint32_t v = m.firstVertex; v < m.firstVertex + m.vertexCount && v < renderer::Vertexes.size(); ++v)
    {
        const yea_float3 p = renderer::Vertexes[v].pos;
        const float      q[3] = { p.x, p.y, p.z };
        for (int k = 0; k < 3; ++k)
        {
            lo[k] = any && lo[k] < q[k] ? lo[k] : q[k];
            hi[k] = any && hi[k] > q[k] ? hi[k] : q[k];
        }
        any = true;
    }
}

bool cooked(const char* source)
{
    const size_t n = strlen(source);
    return n > 7 && !strcmp(source + n - 7, ".rmodel");
}

void fail(Entry& e, const char* why)
{
    e.state = 2;
    snprintf(e.error, sizeof e.error, "%s", why);
    char line[300];
    snprintf(line, sizeof line, "renderer_models: model %s (%s): %s", e.name, e.owner, e.error);
    yea::log(line);
}

// A cooked model (rmodel.h) from its owner's Assets/: each part's LOD chain as the renderer's meshes, linked finest
// first, and a material per part. Read here, in this module's own update; the renderer copies the data into its tables.
void loadCooked(Entry& e)
{
    freeAnim(e);
    char path[1300];
    snprintf(path, sizeof path, "%s/%s", e.assets, e.source);
    FILE* f = e.assets[0] ? fopen(path, "rb") : nullptr;
    if (!f)
        return fail(e, "its file isn't in its module's Assets/");
    fseek(f, 0, SEEK_END);
    const long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t* data = size >= (long)sizeof(RmodelHeader) ? (uint8_t*)malloc((size_t)size) : nullptr;
    const bool read = data && fread(data, 1, (size_t)size, f) == (size_t)size;
    fclose(f);
    RmodelHeader h = {};
    if (read)
        memcpy(&h, data, sizeof h);
    auto inside = [&](uint64_t at, uint64_t bytes) { return at + bytes <= (uint64_t)size; };
    const char* bad = !read                                   ? "its file couldn't be read"
                      : h.magic != kRmodelMagic               ? "its file isn't a cooked model"
                      : h.version < kRmodelOldest || h.version > kRmodelVersion ? "it was cooked for another version: cook it again"
                      : !h.partCount || h.partCount > kMaxParts ? "it has no parts, or more than 64"
                      : !inside(h.materials, (uint64_t)h.materialCount * sizeof(RmodelMaterial)) ||
                              !inside(h.parts, (uint64_t)h.partCount * sizeof(RmodelPart)) ||
                              !inside(h.lods, (uint64_t)h.lodCount * sizeof(RmodelLod)) ||
                              !inside(h.positions, (uint64_t)h.vertexCount * 12) || !inside(h.attributes, (uint64_t)h.vertexCount * 20) ||
                              !inside(h.indices, (uint64_t)h.indexCount * 4) ||
                              !inside(h.textures, (uint64_t)h.textureCount * sizeof(RmodelTexture))
                          ? "its file is cut short"
                          : nullptr;
    const RmodelPart*     parts = (const RmodelPart*)(data + (bad ? 0 : h.parts));
    const RmodelLod*      lods = (const RmodelLod*)(data + (bad ? 0 : h.lods));
    const RmodelMaterial* mats = (const RmodelMaterial*)(data + (bad ? 0 : h.materials));
    const RmodelTexture*  texs = (const RmodelTexture*)(data + (bad ? 0 : h.textures));
    for (uint32_t t = 0; !bad && t < h.textureCount; ++t)
        if (texs[t].format < RmodelBC7Srgb || texs[t].format > RmodelBC5 || !inside(texs[t].offset, texs[t].bytes))
            bad = "its file is damaged";
    for (uint32_t m = 0; !bad && m < h.materialCount; ++m)
        for (int32_t t : mats[m].textures)
            if (t >= (int32_t)h.textureCount)
                bad = "its file is damaged";
    // Skinned (v3): its skeleton and clips, kept (the span from the skeleton to the textures).
    const bool          skinned = !bad && h.version >= 3 && (h.flags & RmodelSkinned);
    const RmodelMorphs* morphs = nullptr;
    if (skinned)
    {
        const RmodelSkeleton* sk = inside(h.skeleton, sizeof(RmodelSkeleton)) && h.skeleton < h.textures ? (const RmodelSkeleton*)(data + h.skeleton) : nullptr;
        bad = !sk || !inside(h.skin, (uint64_t)h.vertexCount * sizeof(RmodelSkinVertex)) ? "its file is cut short"
              : !sk->nodeCount || sk->nodeCount > kMaxNodes || !sk->boneCount || sk->boneCount > kMaxBones
                  ? "its skeleton has no bones, or more than 1024 nodes or bones"
              : sk->nodes < h.skeleton || sk->bones < h.skeleton || sk->clips < h.skeleton ||
                      sk->nodes + (uint64_t)sk->nodeCount * sizeof(RmodelNode) > h.textures ||
                      sk->bones + (uint64_t)sk->boneCount * sizeof(RmodelBone) > h.textures ||
                      sk->clips + (uint64_t)sk->clipCount * sizeof(RmodelClip) > h.textures
                  ? "its file is damaged"
                  : nullptr;
        const RmodelNode* nodes = bad ? nullptr : (const RmodelNode*)(data + sk->nodes);
        const RmodelBone* bones = bad ? nullptr : (const RmodelBone*)(data + sk->bones);
        const RmodelClip* clips = bad ? nullptr : (const RmodelClip*)(data + sk->clips);
        for (uint32_t i = 0; !bad && i < sk->nodeCount; ++i)
            if (nodes[i].parent >= (int32_t)i)
                bad = "its file is damaged";
        for (uint32_t i = 0; !bad && i < sk->boneCount; ++i)
            if (bones[i].node >= sk->nodeCount)
                bad = "its file is damaged";
        // Morph targets (v4): their names inside the kept span, every vertex's deltas inside the file.
        const uint32_t      nt = !bad && h.version >= 4 ? sk->targetCount : 0;
        const RmodelMorphs* mo = nullptr;
        if (nt)
        {
            mo = nt <= kRmodelMaxTargets && sk->targets >= h.skeleton &&
                         sk->targets + sizeof(RmodelMorphs) + (uint64_t)nt * sizeof(RmodelMorphTarget) <= h.textures
                     ? (const RmodelMorphs*)(data + sk->targets)
                     : nullptr;
            bad = !mo || !inside(mo->refs, (uint64_t)h.vertexCount * 4) || !inside(mo->deltas, (uint64_t)mo->deltaCount * sizeof(RmodelMorphDelta))
                      ? "its file is damaged"
                      : nullptr;
            const uint32_t*         refs = bad ? nullptr : (const uint32_t*)(data + mo->refs);
            const RmodelMorphDelta* ds = bad ? nullptr : (const RmodelMorphDelta*)(data + mo->deltas);
            for (uint32_t v = 0; !bad && v < h.vertexCount; ++v)
                if ((uint64_t)(refs[v] & 0xFFFFFFu) + (refs[v] >> 24) > mo->deltaCount)
                    bad = "its file is damaged";
            for (uint32_t d = 0; !bad && d < mo->deltaCount; ++d)
                if ((ds[d].packed & 0xFFu) >= nt)
                    bad = "its file is damaged";
        }
        morphs = mo;
        // Every clip inside the file and its data whole in its codec's layout (anim.h clipValid), and every bone a vertex
        // names in the skeleton.
        for (uint32_t i = 0; !bad && i < sk->clipCount; ++i)
            if (clips[i].offset < h.skeleton || clips[i].offset + clips[i].bytes > h.textures || !clips[i].frames ||
                !anim::clipValid(data + clips[i].offset, clips[i], sk->nodeCount, nt))
                bad = "its file is damaged";
        const RmodelSkinVertex* sv = bad ? nullptr : (const RmodelSkinVertex*)(data + h.skin);
        for (uint32_t v = 0; !bad && v < h.vertexCount; ++v)
            for (int k = 0; k < 4; ++k)
                if (sv[v].weights[k] && sv[v].bones[k] >= sk->boneCount)
                    bad = "its file is damaged";
        if (!bad)
        {
            freeAnim(e);
            e.animBase = h.skeleton;
            e.anim = (uint8_t*)malloc(h.textures - h.skeleton);
            if (!e.anim)
                bad = "out of memory for its animation";
            else
            {
                memcpy(e.anim, data + h.skeleton, h.textures - h.skeleton);
                e.skel = (const RmodelSkeleton*)e.anim;
                e.nodes = (const RmodelNode*)(e.anim + (e.skel->nodes - h.skeleton));
                e.bones = (const RmodelBone*)(e.anim + (e.skel->bones - h.skeleton));
                e.clips = (const RmodelClip*)(e.anim + (e.skel->clips - h.skeleton));
                e.targets = morphs ? (const RmodelMorphTarget*)(e.anim + (e.skel->targets - h.skeleton) + sizeof(RmodelMorphs)) : nullptr;
                findMotion(e);
                findLegs(e);
                boneBoxes(e, (const float*)(data + h.positions), sv, h.vertexCount, morphs ? (const uint32_t*)(data + morphs->refs) : nullptr,
                          morphs ? (const RmodelMorphDelta*)(data + morphs->deltas) : nullptr, morphs ? morphs->deltaCount : 0);
                e.animStamp = ++gStamp;
            }
        }
    }
    for (uint32_t k = 0; !bad && k < h.partCount; ++k)
    {
        const RmodelPart& p = parts[k];
        if (p.material >= h.materialCount || !p.lodCount || p.firstLod + p.lodCount > h.lodCount)
            bad = "its file is damaged";
        for (uint32_t l = p.firstLod; !bad && l < p.firstLod + p.lodCount; ++l)
            if (lods[l].firstVertex + (uint64_t)lods[l].vertexCount > h.vertexCount || lods[l].firstIndex + (uint64_t)lods[l].indexCount > h.indexCount)
                bad = "its file is damaged";
    }
    if (bad)
    {
        free(data);
        return fail(e, bad);
    }
    // Its textures: handed to the renderer by file and offset (it reads them in its own frames), each once.
    uint32_t texHandle[256];
    for (uint32_t t = 0; t < 256; ++t)
        texHandle[t] = 0xFFFFFFFEu;
    auto texture = [&](int32_t t) -> uint32_t {
        if (t < 0 || t >= 256 || (uint32_t)t >= h.textureCount)
            return 0xFFFFFFFFu;
        if (texHandle[t] == 0xFFFFFFFEu)
        {
            const RmodelTexture& x = texs[t];
            texHandle[t] = renderer::texture_add((uint64_t)(uintptr_t)path, x.offset, x.bytes, x.format, x.width, x.height, x.mips);
        }
        return texHandle[t];
    };
    uint32_t triangles = 0, made = 0;
    for (uint32_t k = 0; k < h.partCount; ++k)
    {
        const RmodelPart& p = parts[k];
        uint32_t          chain[32], n = 0;
        for (uint32_t l = p.firstLod; l < p.firstLod + p.lodCount && n < 32; ++l)
        {
            const RmodelLod& lod = lods[l];
            const uint32_t   m = renderer::mesh_packed((uint64_t)(uintptr_t)(data + h.positions + (size_t)lod.firstVertex * 12),
                                                       (uint64_t)(uintptr_t)(data + h.attributes + (size_t)lod.firstVertex * 20), lod.vertexCount,
                                                       (uint64_t)(uintptr_t)(data + h.indices + (size_t)lod.firstIndex * 4), lod.indexCount);
            if (m == 0xFFFFFFFFu)
                break; // the renderer's tables are full: the versions made so far still chain
            if (skinned) // its bones, and the bounds every clip reaches (the part's, as cooked)
            {
                renderer::mesh_skin(m, (uint64_t)(uintptr_t)(data + h.skin + (size_t)lod.firstVertex * sizeof(RmodelSkinVertex)),
                                    { p.lo[0], p.lo[1], p.lo[2] }, { p.hi[0], p.hi[1], p.hi[2] });
                if (morphs) // its morph targets' deltas (the renderer keeps those this version's vertices name)
                    renderer::mesh_morph(m, (uint64_t)(uintptr_t)(data + morphs->refs + (size_t)lod.firstVertex * 4),
                                         (uint64_t)(uintptr_t)(data + morphs->deltas), morphs->deltaCount);
            }
            chain[n++] = m;
        }
        if (!n)
            break;
        for (uint32_t i = 0; i < n; ++i)
            renderer::mesh_lod(chain[i], lods[p.firstLod + i].error, i + 1 < n ? chain[i + 1] : 0xFFFFFFFFu);
        const RmodelMaterial& mt = mats[p.material];
        const yea_handle material = renderer::material_add({ mt.flatColor[0], mt.flatColor[1], mt.flatColor[2] }, mt.flatRoughness, mt.flatMetallic);
        ModelPart* row = material == 0xFFFFFFFFu ? nullptr : ModelParts.add();
        if (!row)
            break;
        // Emission as glTF has it (factor times strength), emissive_nits cd/m2 for 1 (update() follows the knob); its maps,
        // factors and cutoff.
        const float glow = mt.emissiveStrength * settings.emissive_nits;
        renderer::material_emissive(material, { mt.emissive[0], mt.emissive[1], mt.emissive[2] }, glow);
        row->emissive = { mt.emissive[0] * mt.emissiveStrength, mt.emissive[1] * mt.emissiveStrength, mt.emissive[2] * mt.emissiveStrength };
        renderer::material_factors(material, { mt.baseColor[0], mt.baseColor[1], mt.baseColor[2], mt.baseColor[3] }, mt.roughness,
                                   mt.metallic, mt.normalScale, mt.occlusionStrength);
        renderer::material_maps(material, texture(mt.textures[0]), texture(mt.textures[1]), texture(mt.textures[2]),
                                texture(mt.textures[3]));
        if (mt.alpha == RmodelMask)
            renderer::material_alpha(material, mt.alphaCutoff > 0 ? mt.alphaCutoff : 0.5f);
        else if (mt.alpha == RmodelBlend) // blended: its factors' alpha (above) times its map's, as glTF says
            renderer::material_blend(material, 1.0f);
        row->key = e.key;
        row->part = k;
        row->mesh = e.mesh[k] = chain[0];
        row->material = e.material[k] = material;
        row->twoSided = e.twoSided[k] = mt.doubleSided ? 1 : 0;
        row->lo = { h.lo[0], h.lo[1], h.lo[2] };
        row->hi = { h.hi[0], h.hi[1], h.hi[2] };
        triangles += lods[p.firstLod].indexCount / 3;
        ++made;
    }
    free(data);
    if (made < h.partCount)
    {
        freeAnim(e);
        // Partly made: what was made stays in the renderer's tables until its scene starts over; nothing of it is drawn.
        for (uint32_t i = ModelParts.size(); i-- > 0;)
            if (ModelParts[i].key == e.key)
                ModelParts.remove(i);
        return fail(e, "the renderer's mesh tables are full");
    }
    e.parts = h.partCount;
    memcpy(e.lo, h.lo, sizeof e.lo);
    memcpy(e.hi, h.hi, sizeof e.hi);
    e.state = 1;
    char line[300];
    snprintf(line, sizeof line, "renderer_models: model %s (%s): %u parts, %u triangles, %u versions in all", e.name, e.owner, e.parts,
             triangles, h.lodCount);
    yea::log(line);
}

// A skinned model's skeleton and clips alone, from its file (its parts are in the renderer already).
void loadAnim(Entry& e)
{
    char path[1300];
    snprintf(path, sizeof path, "%s/%s", e.assets, e.source);
    FILE* f = e.assets[0] ? fopen(path, "rb") : nullptr;
    if (!f)
        return;
    RmodelHeader h = {};
    fseek(f, 0, SEEK_END);
    const long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    const bool ok = fread(&h, sizeof h, 1, f) == 1 && h.magic == kRmodelMagic && h.version >= 3 && h.version <= kRmodelVersion &&
                    (h.flags & RmodelSkinned) && h.skeleton < h.textures && h.textures <= (uint64_t)size;
    if (ok && (e.anim = (uint8_t*)malloc(h.textures - h.skeleton)) != nullptr)
    {
        fseek(f, (long)h.skeleton, SEEK_SET);
        const RmodelSkeleton* sk = (const RmodelSkeleton*)e.anim;
        if (fread(e.anim, 1, h.textures - h.skeleton, f) != h.textures - h.skeleton || sk->nodeCount > kMaxNodes || sk->boneCount > kMaxBones)
            freeAnim(e);
        else
        {
            e.animBase = h.skeleton;
            e.skel = sk;
            e.nodes = (const RmodelNode*)(e.anim + (sk->nodes - h.skeleton));
            e.bones = (const RmodelBone*)(e.anim + (sk->bones - h.skeleton));
            e.clips = (const RmodelClip*)(e.anim + (sk->clips - h.skeleton));
            const bool morphed = h.version >= 4 && sk->targetCount && sk->targetCount <= kRmodelMaxTargets && sk->targets >= h.skeleton &&
                                 sk->targets + sizeof(RmodelMorphs) + (uint64_t)sk->targetCount * sizeof(RmodelMorphTarget) <= h.textures;
            e.targets = morphed ? (const RmodelMorphTarget*)(e.anim + (sk->targets - h.skeleton) + sizeof(RmodelMorphs)) : nullptr;
            findMotion(e);
            findLegs(e);
            // its bones' boxes, from its vertices (and morph deltas) read again
            const RmodelMorphs* mo = e.targets ? (const RmodelMorphs*)(e.anim + (sk->targets - h.skeleton)) : nullptr;
            const uint64_t      nv = h.vertexCount, nd = mo ? mo->deltaCount : 0;
            const bool fits = h.positions + nv * 12 <= (uint64_t)size && h.skin + nv * sizeof(RmodelSkinVertex) <= (uint64_t)size &&
                              (!mo || (mo->refs + nv * 4 <= (uint64_t)size && mo->deltas + nd * sizeof(RmodelMorphDelta) <= (uint64_t)size));
            uint8_t* tmp = fits ? (uint8_t*)malloc(nv * (12 + sizeof(RmodelSkinVertex) + (mo ? 4 : 0)) + nd * sizeof(RmodelMorphDelta)) : nullptr;
            float*            pos = (float*)tmp;
            RmodelSkinVertex* sv = (RmodelSkinVertex*)(tmp + nv * 12);
            uint32_t*         refs = mo ? (uint32_t*)(tmp + nv * (12 + sizeof(RmodelSkinVertex))) : nullptr;
            RmodelMorphDelta* ds = mo ? (RmodelMorphDelta*)(tmp + nv * (16 + sizeof(RmodelSkinVertex))) : nullptr;
            auto at = [&](uint64_t off, void* out, uint64_t bytes) { return !fseek(f, (long)off, SEEK_SET) && fread(out, 1, bytes, f) == bytes; };
            if (tmp && at(h.positions, pos, nv * 12) && at(h.skin, sv, nv * sizeof(RmodelSkinVertex)) &&
                (!mo || (at(mo->refs, refs, nv * 4) && at(mo->deltas, ds, nd * sizeof(RmodelMorphDelta)))))
                boneBoxes(e, pos, sv, h.vertexCount, refs, ds, (uint32_t)nd);
            free(tmp);
            e.animStamp = ++gStamp;
        }
    }
    fclose(f);
}

// Its parts in the renderer: found again (made in this scene before this module restarted) or made now.
void build(Entry& e)
{
    e.parts = 0;
    e.kind = cooked(e.source) ? 1 : 0;
    for (uint32_t i = 0; i < ModelParts.size(); ++i)
        if (ModelParts[i].key == e.key && ModelParts[i].part < kMaxParts)
        {
            const ModelPart& p = ModelParts[i];
            e.mesh[p.part] = p.mesh;
            e.material[p.part] = p.material;
            e.twoSided[p.part] = p.twoSided;
            e.parts = e.parts > p.part + 1 ? e.parts : p.part + 1;
            if (p.part == 0)
                memcpy(e.lo, &p.lo, sizeof e.lo), memcpy(e.hi, &p.hi, sizeof e.hi);
        }
    if (e.parts)
    {
        // Made before this module restarted: a skinned model's skeleton and clips are read again from its file.
        if (e.kind == 1 && !e.anim)
            loadAnim(e);
        e.state = 1;
        return;
    }
    if (e.kind == 1)
        return renderer::scene_epoch() ? loadCooked(e) : void();
    const uint32_t mesh = builtin(e.source);
    if (mesh == 0xFFFFFFFEu)
    {
        char why[200];
        snprintf(why, sizeof why, "%s isn't a built-in shape (box, sphere, cylinder, cone, torus, plane) or a cooked .rmodel", e.source);
        return fail(e, why);
    }
    if (mesh == 0xFFFFFFFFu)
        return; // the renderer isn't running: tried again next frame
    const yea_handle material = renderer::material_add({ 0.72f, 0.72f, 0.72f }, 0.45f, 0.0f);
    if (material == 0xFFFFFFFFu)
        return;
    ModelPart* p = ModelParts.add();
    if (!p)
        return;
    e.parts = 1;
    e.twoSided[0] = p->twoSided = 0;
    e.mesh[0] = p->mesh = mesh;
    e.material[0] = p->material = material;
    p->key = e.key;
    p->part = 0;
    bounds(mesh, e.lo, e.hi);
    p->lo = { e.lo[0], e.lo[1], e.lo[2] };
    p->hi = { e.hi[0], e.hi[1], e.hi[2] };
    e.state = 1;
}

yea_float4 qmul(yea_float4 a, yea_float4 b)
{
    return { a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y, a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
             a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w, a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z };
}
// The turn that stands a model up as its Up Axis knob says: Z up in its file turned to Y up.
yea_float4 upTurn(const Entry& e)
{
    return e.knobs[mdl::KnobUpAxis] > 1.5f ? yea_float4{ -0.70710678f, 0, 0, 0.70710678f } : yea_float4{ 0, 0, 0, 1 };
}
// The root motion its clips gathered that the game hasn't taken yet, in the world: out[0..2] metres, out[3] radians
// about +Y (zeros while root motion is off).
void pendingMotion(uint32_t row, const Entry& e, float* out)
{
    out[0] = out[1] = out[2] = out[3] = 0;
    if (row >= PlacementMotions.size() || !PlacementMotions[row].on)
        return;
    const PlacementMotion& m = PlacementMotions[row];
    const Placement&       p = Placements[row];
    const float            s = p.scale * e.knobs[mdl::KnobScale];
    const float            v[3] = { m.acc.x * s, m.acc.y * s, m.acc.z * s };
    qrot(qmul(p.rot, upTurn(e)), v, out);
    out[3] = m.yaw;
}
// Where a placement's model draws: as placed, then moved on by what root motion gathered since the game last took it,
// so its pose and its place agree on every frame (the game moves it a frame later).
void placedNow(uint32_t row, const Entry& e, yea_float3& pos, yea_float4& rot, float& scale)
{
    const Placement& p = Placements[row];
    float            m[4];
    pendingMotion(row, e, m);
    const float h = m[3] * 0.5f;
    pos = { p.pos.x + m[0], p.pos.y + m[1], p.pos.z + m[2] };
    rot = qmul(m[3] != 0 ? yea_float4{ 0, sinf(h), 0, cosf(h) } : yea_float4{ 0, 0, 0, 1 }, qmul(p.rot, upTurn(e)));
    scale = p.scale * e.knobs[mdl::KnobScale];
}

// Each placement's instances, made, moved and tuned to match it; those of placements that went, removed.
void syncPlacements()
{
    for (uint32_t row = 0; row < Placements.size(); ++row)
    {
        Placement& p = Placements[row];
        if (!p.dirty)
            continue;
        Entry* e = p.alive ? byKey(p.key) : nullptr;
        if (!e || e->state != 1)
        {
            dropInstances(row);
            p.dirty = p.alive && e && e->state == 0; // a model still being built: again next frame
            continue;
        }
        uint32_t have = 0;
        for (uint32_t i = 0; i < PartInstances.size(); ++i)
            have += PartInstances[i].placement == row;
        if (have != e->parts)
        {
            dropInstances(row);
            for (uint32_t k = 0; k < e->parts; ++k)
            {
                yea_handle h = renderer::instance_add(e->mesh[k], e->material[k]);
                PartInstance* pi = h == 0xFFFFFFFFu ? nullptr : PartInstances.add();
                if (!pi)
                    break;
                pi->placement = row;
                pi->instance = h;
            }
        }
        yea_float3 pos;
        yea_float4 rot;
        float      s;
        placedNow(row, *e, pos, rot, s);
        for (uint32_t k = 0; k < e->parts; ++k)
            renderer::material_double_sided(e->material[k], e->twoSided[k] || e->knobs[mdl::KnobDoubleSided] > 0.5f);
        for (uint32_t i = 0; i < PartInstances.size(); ++i)
            if (PartInstances[i].placement == row)
            {
                const yea_handle h = PartInstances[i].instance;
                renderer::instance_place(h, pos, rot, { s, s, s });
                renderer::instance_shadow(h, e->knobs[mdl::KnobCastShadows] > 0.5f);
                renderer::instance_lod_quality(h, e->knobs[mdl::KnobLodQuality]);
            }
        p.dirty = 0;
    }
}

// Every skinned placement's clip moved on by dt, then its palette handed to the renderer for each of its instances.
void animate(float dt)
{
    ++gTick;
    const uint32_t rows = Placements.size() < PlacementAnims.size() ? Placements.size() : PlacementAnims.size();
    // Blend spaces' samples (first, so a blend space asked for while its model loaded starts on the frame a clip would):
    // names taken once their models load (and the maps of others' clips made), those of sets no
    // longer played and of placements that are gone dropped.
    for (uint32_t i = PlacementBlendSamples.size(); i-- > 0;)
    {
        PlacementBlendSample& S = PlacementBlendSamples[i];
        const uint32_t        row = placementRow(S.placement);
        const Entry*          e = row != 0xFFFFFFFFu ? byKey(Placements[row].key) : nullptr;
        const bool live = e && row < PlacementBlends.size() && row < PlacementAnims.size() && e->state != 2 && (e->state != 1 || e->anim) &&
                          ((S.set == PlacementBlends[row].set && PlacementAnims[row].clip == kBlendClip) ||
                           (S.set == PlacementBlends[row].from && PlacementAnims[row].from == kBlendClip));
        if (!live)
        {
            PlacementBlendSamples.remove(i);
            continue;
        }
        const Entry* s = e->state == 1 ? sourceOf(*e, S.source) : nullptr;
        if (!s)
            continue; // either still loading
        if (S.want)
        {
            S.clip = clipHashed(*s, S.want), S.want = 0;
            if (S.clip == kNoClip)
            {
                char line[160];
                snprintf(line, sizeof line, "renderer_models: %s has no clip a blend space of %s asked for: that sample is left out", s->name, e->name);
                yea::log(line);
            }
        }
        if (s != e)
            mapFor(*e, *s, true);
    }
    for (uint32_t row = 0; row < rows; ++row)
    {
        const Entry* e = Placements[row].alive ? byKey(Placements[row].key) : nullptr;
        if (!e || !e->anim || e->state != 1)
            continue;
        PlacementAnim& a = PlacementAnims[row];
        const uint32_t ck = row < PlacementRetargets.size() ? PlacementRetargets[row].clip : 0;
        const uint32_t fk = row < PlacementRetargets.size() ? PlacementRetargets[row].from : 0;
        const Entry *  ce = sourceOf(*e, ck), *fe = sourceOf(*e, fk);
        if (a.want && ce) // asked for by name while it (or the model whose clips it plays) loaded
        {
            const uint32_t c = clipHashed(*ce, a.want);
            a.clip = c != kNoClip ? c : (ce == e ? 0 : kNoClip), a.want = 0;
        }
        if (ce && ce != e) // another model's clips: the map between their skeletons (made here, once)
            mapFor(*e, *ce, true);
        if (fe && fe != e && a.from != kNoClip)
            mapFor(*e, *fe, true);
        // Root motion: what its clips do this frame, gathered (faded between two clips as their poses are).
        if (row < PlacementMotions.size() && PlacementMotions[row].on)
        {
            float to[4], from[4] = { 0, 0, 0, 0 }, m[4];
            playMotion(*e, row, a, false, ce, dt, to);
            const bool fading = a.from != kNoClip && a.fade > 0 && a.faded < a.fade;
            float      f = fading ? a.faded / a.fade : 1.0f;
            f = f * f * (3 - 2 * f);
            if (fading)
                playMotion(*e, row, a, true, fe, dt, from);
            for (int k = 0; k < 4; ++k)
                m[k] = from[k] + (to[k] - from[k]) * f;
            PlacementMotion& pm = PlacementMotions[row];
            float            up[3], u[3], acc[3];
            motionAxes(*e, up, u);
            turn(up, pm.yaw, m, acc);
            pm.acc.x += acc[0], pm.acc.y += acc[1], pm.acc.z += acc[2];
            pm.yaw += m[3];
            Placements[row].dirty = 1; // drawn moved on by it (syncPlacements, next)
        }
        advanceClip(ce, fe, a, dt);
        // Its blend space: the point followed toward the game's, and the phase moved on.
        if (row < PlacementBlends.size())
            stepBlend(*e, row, row | (Placements[row].gen & 0xFFFu) << 20, a, PlacementBlends[row], dt);
    }
    // The Model Viewer's own clip (its model's, whether placed or not), while it plays.
    if (const Entry* e = byId(gPreview.model); e && e->anim && e->state == 1)
    {
        PlacementAnim& a = gPreview.anim;
        if (gPreview.animModel != e->id) // another model chosen: its first clip, from the start
        {
            a = PlacementAnim{};
            a.from = kNoClip, a.clip = e->skel->clipCount ? 0 : kNoClip, a.speed = 1, a.loop = 1;
            memset(gPreview.morphHeld, 0, sizeof gPreview.morphHeld);
            gPreview.animModel = e->id;
            gPreview.clipsFrom = gPreview.srcClip = gPreview.srcFrom = 0;
            gPreview.layerClip = gPreview.layerBone = kNoClip, gPreview.layerSource = 0, gPreview.layerTime = 0;
            gPreview.blend = PlacementBlend{}, gPreview.blendCount = 0, gPreview.blendPick = 0, gPreview.blendSource = 0;
        }
        const Entry* from = gPreview.clipsFrom ? byId(gPreview.clipsFrom) : e; // whose clips the panel lists
        if (!from || !from->anim || from->state != 1)
            from = e, gPreview.clipsFrom = 0;
        // Its blend space: played when the panel asks (or its samples changed while it plays, then at once, in step),
        // left for the clip it played before when the panel stops it, driven to the panel's point.
        PlacementBlend& pb = gPreview.blend;
        if (const Entry* bs = gPreview.blendCount ? sourceOf(*e, gPreview.blendSource) : nullptr; bs && bs != e)
            mapFor(*e, *bs, true);
        if (gPreview.blendPick == 2)
            gPreview.pickClip = a.clip == kBlendClip ? gPreview.blendWas : 0, gPreview.blendPick = 0;
        if (gPreview.blendCount > blend::kMaxSamples)
            gPreview.blendCount = blend::kMaxSamples;
        {
            uint32_t hash = 2166136261u ^ gPreview.blendSource ^ gPreview.blendDims * 16777619u;
            for (uint32_t i = 0; i < gPreview.blendCount; ++i)
            {
                uint32_t xy[2];
                memcpy(xy, &gPreview.blendX[i], 4), memcpy(xy + 1, &gPreview.blendY[i], 4);
                hash = ((hash ^ gPreview.blendClip[i]) * 16777619u ^ xy[0]) * 16777619u ^ xy[1];
            }
            hash = hash ? hash : 1;
            const bool playing = a.clip == kBlendClip && pb.set;
            if (!gPreview.blendCount && playing) // its last sample taken out: the clip before
                gPreview.pickClip = gPreview.blendWas;
            else if (gPreview.blendCount && ((gPreview.blendPick == 1 && !playing) || (playing && pb.hash != hash)))
            {
                if (!playing)
                    gPreview.blendWas = a.clip == kNoClip ? 1 : a.clip + 2;
                const uint32_t set = ++pb.next ? pb.next : ++pb.next;
                pb.tx = gPreview.blendPx, pb.ty = gPreview.blendPy, pb.smooth = 0;
                enterBlend(a, pb, gPreview.srcClip, gPreview.srcFrom, set, gPreview.blendDims == 2 ? 2 : 1, hash, a.speed, 1,
                           playing ? 0.0f : gPreview.fade, true);
            }
            gPreview.blendPick = 0;
        }
        pb.tx = gPreview.blendPx, pb.ty = gPreview.blendPy, pb.smooth = 0;
        if (gPreview.pickClip) // picked in the panel: 1 the rest pose, clip c at c + 2
        {
            leaveBlend(kPreviewRow, a);
            const uint32_t c = gPreview.pickClip - 2;
            startClip(a, gPreview.srcClip, gPreview.srcFrom, gPreview.pickClip >= 2 && c < from->skel->clipCount ? c : kNoClip,
                      from == e ? 0 : from->key, a.speed, 1, gPreview.fade);
            gPreview.pickClip = 0;
        }
        const Entry *ce = sourceOf(*e, gPreview.srcClip), *fe = sourceOf(*e, gPreview.srcFrom);
        if (ce && ce != e)
            mapFor(*e, *ce, true);
        if (fe && fe != e && a.from != kNoClip)
            mapFor(*e, *fe, true);
        a.speed = gPreview.speed;
        advanceClip(ce, fe, a, gPreview.playing ? dt : 0.0f);
        stepBlend(*e, kPreviewRow, 0, a, pb, gPreview.playing ? dt : 0.0f);
        // How far what plays travels (a clip, or the blend space at its point), shown under the clips.
        gPreview.motionShown = 0;
        if (a.clip == kBlendClip && pb.set)
        {
            BlendSet b;
            float    m[4];
            if (e->motionNode != kNoClip && blendWeigh(*e, kPreviewRow, 0, pb.set, pb.dims, pb.x, pb.y, b) && b.length > 0)
            {
                blendMotion(*e, b, 0, 1, true, m);
                const float sc = e->knobs[mdl::KnobScale] > 0 ? e->knobs[mdl::KnobScale] : 1.0f;
                gPreview.motionSpeed = sc * sqrtf(m[0] * m[0] + m[1] * m[1] + m[2] * m[2]) / b.length;
                gPreview.motionTurn = m[3] * 57.2957795f / b.length, gPreview.motionShown = 1;
            }
        }
        else if (a.clip != kNoClip)
            gPreview.motionShown = clipTravel(*e, gPreview.srcClip, a.clip, gPreview.motionSpeed, gPreview.motionTurn);
        // Its layer: picked in the panel (from the models whose clips it lists), its time moved on as its clip's.
        if (gPreview.layerPick)
        {
            const uint32_t c = gPreview.layerPick - 2;
            gPreview.layerClip = gPreview.layerPick >= 2 && c < from->skel->clipCount ? c : kNoClip;
            gPreview.layerSource = from == e ? 0 : from->key, gPreview.layerTime = 0, gPreview.layerPick = 0;
        }
        if (gPreview.layerBone != kNoClip && gPreview.layerBone >= e->skel->nodeCount)
            gPreview.layerBone = kNoClip;
        if (const Entry* ls = gPreview.layerClip != kNoClip ? sourceOf(*e, gPreview.layerSource) : nullptr;
            ls && gPreview.layerClip < ls->skel->clipCount)
        {
            if (ls != e)
                mapFor(*e, *ls, true);
            const float d = ls->clips[gPreview.layerClip].duration;
            gPreview.layerTime += (gPreview.playing ? dt : 0.0f) * gPreview.speed;
            if (d > 0)
                gPreview.layerTime = fmodf(gPreview.layerTime, d) + (gPreview.layerTime < 0 ? d : 0);
        }
        else
            gPreview.layerClip = kNoClip;
    }
    // Layers: names taken once their models load, times and weights moved on, faded-out ones and those of placements
    // that are gone dropped.
    for (uint32_t i = PlacementLayers.size(); i-- > 0;)
    {
        PlacementLayer& L = PlacementLayers[i];
        const uint32_t  row = placementRow(L.placement);
        const Entry*    e = row != 0xFFFFFFFFu && Placements[row].alive ? byKey(Placements[row].key) : nullptr;
        if (!e || e->state == 2 || (e->state == 1 && !e->anim))
        {
            PlacementLayers.remove(i);
            continue;
        }
        const Entry* s = e->state == 1 ? sourceOf(*e, L.source) : nullptr;
        if (!s)
            continue; // either still loading
        bool gone = false;
        if (L.want)
        {
            L.clip = clipHashed(*s, L.want), L.want = 0;
            gone = L.clip == kNoClip;
        }
        if (L.mask == kMaskPending)
        {
            L.mask = kNoClip;
            for (uint32_t n = 0; n < e->skel->nodeCount && L.mask == kNoClip; ++n)
                if (clipHash(e->nodes[n].name, sizeof e->nodes[n].name) == L.maskHash)
                    L.mask = n;
            gone = gone || L.mask == kNoClip; // no such bone: the layer goes rather than cover the whole body
        }
        const float d = L.target - L.weight;
        L.weight = L.fade > 0 ? L.weight + (d > 0 ? 1.0f : -1.0f) * fminf(fabsf(d), dt / L.fade) : L.target;
        if (gone || (L.weight <= 0 && L.target <= 0))
        {
            PlacementLayers.remove(i);
            continue;
        }
        if (s != e)
            mapFor(*e, *s, true);
        const float dur = s->clips[L.clip].duration;
        L.time += dt * L.speed;
        if (dur > 0)
            L.time = L.loop ? (fmodf(L.time, dur) + (L.time < 0 ? dur : 0)) : (L.time < 0 ? 0 : (L.time > dur ? dur : L.time));
    }
    // Held weights of placements that are gone go with them.
    for (uint32_t i = MorphOverrides.size(); i-- > 0;)
        if (placementRow(MorphOverrides[i].placement) == 0xFFFFFFFFu)
            MorphOverrides.remove(i);
    // Inertialized transitions that are over (or whose placement is gone) give back their room.
    for (uint32_t i = gInertCount; i-- > 0;)
    {
        const uint32_t       h = gInertSlot[i].placement, r = h ? placementRow(h) : 0xFFFFFFFFu;
        const PlacementAnim* pa = h ? (r < PlacementAnims.size() ? &PlacementAnims[r] : nullptr) : &gPreview.anim;
        if (!pa || pa->from == kNoClip || !(pa->fade > 0) || pa->faded >= pa->fade)
            gInertSlot[i] = gInertSlot[--gInertCount];
    }
    uint32_t last = 0xFFFFFFFFu, bones = 0, targets = 0; // a placement's parts come together: posed once for all of them
    bool     grown = false; // and foot IK moved it: its pose's bounds in lo, hi (the core keeps its mesh's sphere too)
    float    lo[3], hi[3];
    for (uint32_t i = 0; i < PartInstances.size(); ++i)
    {
        const uint32_t row = PartInstances[i].placement;
        if (row != last)
        {
            last = row, bones = 0, targets = 0, grown = false;
            const Entry* e = row < rows && Placements[row].alive ? byKey(Placements[row].key) : nullptr;
            const bool re = row < PlacementRetargets.size();
            if (e && e->anim && e->state == 1 &&
                evaluate(*e, PlacementAnims[row], nullptr, nullptr, row | (Placements[row].gen & 0xFFFu) << 20,
                         row < PlacementMotions.size() && PlacementMotions[row].on, re ? PlacementRetargets[row].clip : 0,
                         re ? PlacementRetargets[row].from : 0, dt))
            {
                bones = e->skel->boneCount, targets = e->targets ? e->skel->targetCount : 0;
                grown = gFootMoved && poseBounds(*e, gPalette, lo, hi);
            }
        }
        if (bones)
            renderer::instance_pose(PartInstances[i].instance, (uint64_t)(uintptr_t)gPalette, bones);
        if (bones && grown)
            renderer::instance_bounds(PartInstances[i].instance, { lo[0], lo[1], lo[2] }, { hi[0], hi[1], hi[2] });
        if (bones && targets)
            renderer::instance_morph(PartInstances[i].instance, (uint64_t)(uintptr_t)gWeights, targets);
    }
}

// --- The preview's targets (this module's own update, never its draw) --------------------------------------------
void freeTargets()
{
    if (gHdr || gDepth || gShadow || gOut)
        waitQueueIdle(gQueue);
    RenderTarget** all[4] = { &gHdr, &gDepth, &gShadow, &gOut };
    for (RenderTarget** t : all)
        if (*t)
            removeRenderTarget(gRenderer, *t), *t = nullptr;
    gTargetW = gTargetH = 0;
    gPreview.outTexture = nullptr;
    gPreview.outW = gPreview.outH = 0;
}
void makeTargets(uint32_t w, uint32_t h)
{
    freeTargets();
    RenderTargetDesc d = {};
    d.mWidth = w;
    d.mHeight = h;
    d.mDepth = d.mArraySize = d.mMipLevels = 1;
    d.mSampleCount = SAMPLE_COUNT_4;
    d.mFormat = TinyImageFormat_R16G16B16A16_SFLOAT;
    d.mStartState = RESOURCE_STATE_SHADER_RESOURCE;
    d.mDescriptors = DESCRIPTOR_TYPE_TEXTURE;
    d.mClearValue = { { 0, 0, 0, 0 } };
    d.pName = "renderer_models_preview_hdr";
    addRenderTarget(gRenderer, &d, &gHdr);
    d.mFormat = TinyImageFormat_D32_SFLOAT;
    d.mStartState = RESOURCE_STATE_DEPTH_WRITE;
    d.mDescriptors = DESCRIPTOR_TYPE_UNDEFINED;
    d.mClearValue = {};
    d.mClearValue.depth = 0.0f; // reversed Z
    d.pName = "renderer_models_preview_depth";
    addRenderTarget(gRenderer, &d, &gDepth);
    d.mWidth = d.mHeight = kShadowSize;
    d.mSampleCount = SAMPLE_COUNT_1;
    d.mStartState = RESOURCE_STATE_SHADER_RESOURCE;
    d.mDescriptors = DESCRIPTOR_TYPE_TEXTURE;
    d.mClearValue.depth = 1.0f;
    d.pName = "renderer_models_preview_shadow";
    addRenderTarget(gRenderer, &d, &gShadow);
    d.mWidth = w;
    d.mHeight = h;
    d.mFormat = TinyImageFormat_R8G8B8A8_SRGB;
    d.mDescriptors = DESCRIPTOR_TYPE_TEXTURE;
    d.mClearValue = { { 0, 0, 0, 0 } };
    d.pName = "renderer_models_preview";
    addRenderTarget(gRenderer, &d, &gOut);
    if (!gHdr || !gDepth || !gShadow || !gOut)
    {
        yea::host()->log(1, "renderer_models: the GPU refused the Model Viewer's preview targets");
        freeTargets();
        return;
    }
    gTargetW = w, gTargetH = h;
    gPreview.outTexture = gOut->pTexture;
    gPreview.outW = w, gPreview.outH = h;
}
} // namespace

bool clipTravel(const Entry& e, uint32_t src, uint32_t c, float& speed, float& turn)
{
    const Entry* s = sourceOf(e, src);
    if (!s || !s->anim || c >= s->skel->clipCount || s->motionNode == kNoClip || !(s->clips[c].duration > 0))
        return false;
    const float d = s->clips[c].duration, sc = e.knobs[mdl::KnobScale] > 0 ? e.knobs[mdl::KnobScale] : 1.0f;
    float       m[4];
    motionOf(e, s, c, 0, d, true, m);
    speed = sc * sqrtf(m[0] * m[0] + m[1] * m[1] + m[2] * m[2]) / d, turn = m[3] * 57.2957795f / d;
    return true;
}
uint32_t lodCount(const Entry& e)
{
    if (!e.parts)
        return 0;
    const uint32_t rows = renderer::MeshLods.size() < renderer::Meshes.size() ? renderer::MeshLods.size() : renderer::Meshes.size();
    uint32_t       n = 0;
    for (uint32_t m = e.mesh[0]; m < rows && n < 32; m = renderer::MeshLods[m].next)
        ++n;
    return n ? n : 1;
}
uint32_t triangles(const Entry& e)
{
    uint32_t n = 0;
    for (uint32_t k = 0; k < e.parts; ++k)
        n += e.mesh[k] < renderer::Meshes.size() ? renderer::Meshes[e.mesh[k]].indexCount / 3 : 0;
    return n;
}

uint64_t api(uint32_t version) { return version == mdl::kApi ? (uint64_t)(uintptr_t)&gApi : 0; }

int load(int reloaded)
{
    knobs::check(true);
    const YeaHost* host = yea::host();
    gRenderer = (Renderer*)host->renderer;
    gQueue = (Queue*)host->queue;
    gEntries = new Entry[kMaxModels];
    gEntryCount = 0;
    gDraw = new RenderPreview;
    gPoses = new anim::Pose[kMaxNodes];
    gFrom = new anim::Pose[kMaxNodes];
    gWorld = new float[kMaxNodes * 12];
    gPalette = new float[kMaxBones * 12];
    gWeights = new float[kRmodelMaxTargets];
    gFromWeights = new float[kRmodelMaxTargets];
    gSrcPoses = new anim::Pose[kMaxNodes];
    gLayerPoses = new anim::Pose[kMaxNodes];
    gLayerRefs = new anim::Pose[kLayerRefs * kMaxNodes];
    memset(gLayerRefKey, 0, sizeof gLayerRefKey), gLayerRefNext = 0;
    gLayerMask = new uint8_t[kMaxNodes];
    gBlendPoses = new anim::Pose[kMaxNodes];
    gBlendWeights = new float[kRmodelMaxTargets];
    gBlendTris = new BlendTris[(kBlendRows + 1) * 2]();
    gSrcWeights = new float[kRmodelMaxTargets];
    gInertPool = new float[kInertFloats];
    gInertSlot = new InertSlot[kInertSlots];
    gInertCount = 0;
    gInertNew = new anim::Pose[kMaxNodes], gInertOld = new anim::Pose[kMaxNodes];
    gInertNewW = new float[kRmodelMaxTargets], gInertOldW = new float[kRmodelMaxTargets];
    gRt = { new float[kMaxNodes * 12], new float[kMaxNodes * 4], new float[kMaxNodes * 12], new float[kMaxNodes * 4] };
    gMaps = new MapSlot[kMaps]();
    gPreview = Preview{};
    gPreview.yaw = 35.0f;
    gPreview.pitch = 20.0f;
    gPreview.distance = 1.35f;
    gPreview.lod = -1;
    gPreview.turntable = 1;
    gPreview.playing = 1, gPreview.speed = 1.0f, gPreview.fade = 0.2f, gPreview.inPlace = 1;
    gPreview.layerClip = gPreview.layerBone = kNoClip, gPreview.layerWeight = 1.0f;
    gApi.instance = (uint64_t)host->frame() * 2654435761u ^ (uint64_t)(uintptr_t)gEntries; // models register again
    gInstance = gApi.instance;
    return 0;
}

void unload()
{
    freeTargets();
    for (uint32_t i = 0; i < gEntryCount; ++i)
        freeAnim(gEntries[i]);
    delete[] gEntries;
    delete[] gPoses;
    delete[] gFrom;
    delete[] gWorld;
    delete[] gPalette;
    delete[] gWeights;
    delete[] gFromWeights;
    gWeights = gFromWeights = nullptr;
    for (uint32_t i = 0; gMaps && i < kMaps; ++i)
        if (gMaps[i].live)
            retarget::release(gMaps[i].map);
    delete[] gMaps;
    delete[] gSrcPoses;
    delete[] gSrcWeights;
    delete[] gLayerPoses;
    delete[] gLayerRefs;
    delete[] gLayerMask;
    delete[] gBlendPoses;
    delete[] gBlendWeights;
    delete[] gBlendTris;
    delete[] gInertPool, delete[] gInertSlot, delete[] gInertNew, delete[] gInertOld, delete[] gInertNewW, delete[] gInertOldW;
    gInertPool = nullptr, gInertSlot = nullptr, gInertNew = gInertOld = nullptr, gInertNewW = gInertOldW = nullptr, gInertCount = 0;
    gBlendPoses = nullptr, gBlendWeights = nullptr, gBlendTris = nullptr;
    gLayerPoses = gLayerRefs = nullptr, gLayerMask = nullptr;
    delete[] gRt.sw, delete[] gRt.sq, delete[] gRt.tw, delete[] gRt.tq;
    gMaps = nullptr, gSrcPoses = nullptr, gSrcWeights = nullptr, gRt = {};
    gPoses = gFrom = nullptr;
    gWorld = gPalette = nullptr;
    gEntries = nullptr;
    gEntryCount = 0;
    delete gDraw;
    gDraw = nullptr;
}

void update(uint32_t phase, float dt)
{
    if (phase == YEA_PHASE_LATE)
        knobs::check(false);
    if (getenv("YEA_FIXED_DT")) dt = 1.0f / 60.0f; // SCRATCH-DUMP
    if (phase != YEA_PHASE_LATE) // after the game's modules have placed and moved their models this frame
    {
        if (phase == YEA_PHASE_SIMULATE && gPreview.turntable)
            gPreview.yaw = fmodf(gPreview.yaw + dt * 12.0f, 360.0f);
        return;
    }
    // The renderer's scene went (cleared, Reset, its material layout changed): every part and instance with it.
    const uint32_t epoch = renderer::scene_epoch();
    if (!epoch)
        return; // the renderer isn't running: its imports are stubs
    if (epoch != scene->epoch)
    {
        ModelParts.clear();
        PartInstances.clear();
        for (uint32_t i = 0; i < gEntryCount; ++i)
            gEntries[i].state = gEntries[i].state == 2 ? 2 : 0, gEntries[i].parts = 0;
        for (uint32_t i = 0; i < Placements.size(); ++i)
            Placements[i].dirty = Placements[i].alive;
        scene->epoch = epoch;
    }
    // Models made from an older cooked format (stage 1's flat materials): their instances go and they are made again, with
    // their texture maps. What they had in the renderer's tables stays there, undrawn, until its scene starts over.
    if (format->version != kRmodelVersion)
    {
        for (uint32_t i = 0; i < PartInstances.size(); ++i)
            renderer::instance_remove(PartInstances[i].instance);
        PartInstances.clear();
        ModelParts.clear();
        for (uint32_t i = 0; i < gEntryCount; ++i)
            gEntries[i].state = 0, gEntries[i].parts = 0;
        for (uint32_t i = 0; i < Placements.size(); ++i)
            Placements[i].dirty = Placements[i].alive;
        format->version = kRmodelVersion;
    }
    for (uint32_t i = 0; i < gEntryCount; ++i)
        if (gEntries[i].state == 0)
            build(gEntries[i]);
    syncPlacements();
    animate(dt);
    syncPlacements(); // those root motion moved on
    // emissive_nits, live: every cooked part's glow again when it changes (and once after this module restarts).
    if (settings.emissive_nits != gGlow)
    {
        gGlow = settings.emissive_nits;
        for (uint32_t i = 0; i < ModelParts.size(); ++i)
            renderer::material_emissive(ModelParts[i].material, ModelParts[i].emissive, gGlow);
    }

    // The preview's targets at the size the panel last asked for.
    const uint32_t w = gPreview.width, h = gPreview.height;
    if (w && h && (w != gTargetW || h != gTargetH))
        makeTargets(w, h);
}

void draw(uint32_t pass, const YeaFrame* frame)
{
    if (pass != YEA_PASS_RENDER || !gPreview.shown)
        return;
    gPreview.shown = 0; // drawn only on frames the panel showed it
    Entry* e = byId(gPreview.model);
    if (!e || e->state != 1 || !gOut || !gDraw)
        return;
    RenderPreview& p = *gDraw;
    p = RenderPreview{};
    p.version = RENDER_PREVIEW_VERSION;
    p.frameIndex = frame->frameIndex;
    p.cmd = (Cmd*)frame->cmd;
    p.width = gTargetW;
    p.height = gTargetH;
    p.hdr = gHdr, p.depth = gDepth, p.shadow = gShadow, p.out = gOut;
    // The model at the origin, scaled by its Scale knob and stood up by its Up Axis, its bottom on the floor.
    const float      s = e->knobs[mdl::KnobScale] > 0 ? e->knobs[mdl::KnobScale] : 1.0f;
    const yea_float4 q = upTurn(*e);
    const float      x = q.x, y = q.y, z = q.z, w = q.w;
    const float      r[3][3] = { { 1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w) },
                                 { 2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w) },
                                 { 2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y) } };
    // Its bounds turned: the centre, the radius and the lowest point.
    float lo[3] = { 1e30f, 1e30f, 1e30f }, hi[3] = { -1e30f, -1e30f, -1e30f };
    for (int c = 0; c < 8; ++c)
    {
        const float o[3] = { c & 1 ? e->hi[0] : e->lo[0], c & 2 ? e->hi[1] : e->lo[1], c & 4 ? e->hi[2] : e->lo[2] };
        for (int k = 0; k < 3; ++k)
        {
            const float v = s * (r[k][0] * o[0] + r[k][1] * o[1] + r[k][2] * o[2]);
            lo[k] = v < lo[k] ? v : lo[k];
            hi[k] = v > hi[k] ? v : hi[k];
        }
    }
    const float ext[3] = { hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2] };
    const float radius = fmaxf(0.5f * sqrtf(ext[0] * ext[0] + ext[1] * ext[1] + ext[2] * ext[2]), 1e-3f);
    const float ctr[3] = { 0.5f * (lo[0] + hi[0]), 0.5f * (lo[1] + hi[1]), 0.5f * (lo[2] + hi[2]) };
    p.parts = e->parts;
    for (uint32_t k = 0; k < e->parts; ++k)
    {
        p.mesh[k] = e->mesh[k];
        p.material[k] = e->material[k];
        p.twoSided[k] = e->twoSided[k] || e->knobs[mdl::KnobDoubleSided] > 0.5f;
        for (int row = 0; row < 3; ++row)
        {
            for (int c = 0; c < 3; ++c)
                p.transform[k][row * 4 + c] = r[row][c] * s;
            p.transform[k][row * 4 + 3] = -ctr[row]; // centred on the origin
        }
    }
    p.center[0] = p.center[1] = p.center[2] = 0;
    p.radius = radius;
    p.floorY = lo[1] - ctr[1];
    // The camera on its orbit, the key light high and to its left (it turns with the model's view, as a studio's does).
    const float yaw = gPreview.yaw * 0.01745329f, pitch = gPreview.pitch * 0.01745329f;
    const float fov = 30.0f, dist = radius * gPreview.distance / sinf(fov * 0.5f * 0.01745329f); // 1: its sphere fills the height
    p.fov = fov;
    p.eye[0] = dist * cosf(pitch) * sinf(yaw);
    p.eye[1] = dist * sinf(pitch);
    p.eye[2] = -dist * cosf(pitch) * cosf(yaw);
    const float ky = yaw - 0.8f, kp = 0.9f;
    p.sunDir[0] = cosf(kp) * sinf(ky);
    p.sunDir[1] = sinf(kp);
    p.sunDir[2] = -cosf(kp) * cosf(ky);
    p.sunLux = 6.0f;
    p.skyLuminance = 1.2f;
    p.lodQuality = e->knobs[mdl::KnobLodQuality];
    p.lod = gPreview.lod;
    if (e->anim && gPreview.animModel == e->id &&
        evaluate(*e, gPreview.anim, gPreview.morph, gPreview.morphHeld, 0, gPreview.inPlace != 0, gPreview.srcClip,
                 gPreview.srcFrom, -1, &gPreview)) // posed as its clip (and its layer) is now
    {
        p.palette = (uint64_t)(uintptr_t)gPalette, p.bones = e->skel->boneCount;
        // Its skeleton for the panel's overlay: each node placed as the preview places the model, through the
        // preview's camera (the core's previewCamera: looking at the origin, right = up x forward).
        gPreview.boneNodes = 0;
        if (gPreview.bones && p.parts && gTargetW && gTargetH)
        {
            const float* m = p.transform[0];
            const float  el = sqrtf(p.eye[0] * p.eye[0] + p.eye[1] * p.eye[1] + p.eye[2] * p.eye[2]);
            const float  f[3] = { -p.eye[0] / el, -p.eye[1] / el, -p.eye[2] / el }, hp[3] = { fabsf(f[1]) < 0.99f ? 0.0f : 1.0f,
                                                                                          fabsf(f[1]) < 0.99f ? 1.0f : 0.0f, 0 };
            float rt[3] = { hp[1] * f[2] - hp[2] * f[1], hp[2] * f[0] - hp[0] * f[2], hp[0] * f[1] - hp[1] * f[0] };
            const float rl = sqrtf(rt[0] * rt[0] + rt[1] * rt[1] + rt[2] * rt[2]);
            for (float& c : rt)
                c /= rl;
            const float up[3] = { f[1] * rt[2] - f[2] * rt[1], f[2] * rt[0] - f[0] * rt[2], f[0] * rt[1] - f[1] * rt[0] };
            const float p11 = 1.0f / tanf(p.fov * 0.5f * 0.01745329f), p00 = p11 * float(gTargetH) / float(gTargetW);
            const uint32_t n = e->skel->nodeCount < kViewerNodes ? e->skel->nodeCount : kViewerNodes;
            for (uint32_t i = 0; i < n; ++i)
            {
                const float* g = &gWorld[i * 12];
                float        d[3];
                for (int k = 0; k < 3; ++k)
                    d[k] = m[k * 4] * g[3] + m[k * 4 + 1] * g[7] + m[k * 4 + 2] * g[11] + m[k * 4 + 3] - p.eye[k];
                const float z = f[0] * d[0] + f[1] * d[1] + f[2] * d[2];
                const float xv = rt[0] * d[0] + rt[1] * d[1] + rt[2] * d[2], yv = up[0] * d[0] + up[1] * d[1] + up[2] * d[2];
                gPreview.boneSeen[i] = z > 1e-4f;
                gPreview.boneXY[i * 2] = z > 1e-4f ? 0.5f + 0.5f * p00 * xv / z : 0;
                gPreview.boneXY[i * 2 + 1] = z > 1e-4f ? 0.5f - 0.5f * p11 * yv / z : 0;
            }
            gPreview.boneNodes = n;
        }
        if (e->targets)
        {
            p.weights = (uint64_t)(uintptr_t)gWeights, p.targets = e->skel->targetCount;
            memcpy(gPreview.morphNow, gWeights, e->skel->targetCount * sizeof(float));
        }
    }
    if (renderer::preview_draw((uint64_t)(uintptr_t)&p))
        gPreview.lodShown = p.lodShown, gPreview.triangles = p.triangles;
}
} // namespace renderer_models
