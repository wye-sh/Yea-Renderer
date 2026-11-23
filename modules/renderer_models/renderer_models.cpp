// Models for the renderer (AGENTS.md beside the component): the registry of every model the view's modules declare
// (ModelLink.h's MODEL lines, through api()), their meshes and materials in the renderer's scene, the game's placements
// of them as the renderer's instances, and the Model Viewer's preview (drawn by the core's preview_draw into this
// module's own targets).
#include "yea_renderer_models.h"
#include "models.h"
#include "retarget.h"
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
anim::Pose *       gLayerPoses, *gLayerRef; // a layer's clip as sampled, and its first frame (additive)
uint8_t*           gLayerMask;              // the nodes it covers
constexpr uint32_t kMaskPending = 0xFFFFFFFEu; // a layer's bone named while its model loads
float*             gSrcWeights;
retarget::Scratch  gRt;
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
int apiPlay(uint32_t h, const char* clip, float speed, int loop, float fade)
{
    PlacementAnim* a;
    Entry*         e = animOf(h, &a);
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

// One placement's pose now into gPoses (its clip, faded from the one before), and its morph weights into gWeights
// (the same, then what the game holds: `held` weights, `isHeld` where it does): false when it isn't skinned.
// clipKey and fromKey: the models (keys; 0 its own) the clip and the one it fades from are of.
bool evaluate(const Entry& e, const PlacementAnim& a, const float* held = nullptr, const uint8_t* isHeld = nullptr, uint32_t placement = 0,
              bool still = false, uint32_t clipKey = 0, uint32_t fromKey = 0) // still: in place (root motion on)
{
    const uint32_t n = e.skel->nodeCount, nt = e.targets ? e.skel->targetCount : 0;
    const bool     fading = a.from != kNoClip && a.fade > 0 && a.faded < a.fade;
    sampleAs(e, sourceOf(e, clipKey), a.clip, a.time, a.loop, still, gPoses, nt ? gWeights : nullptr);
    if (fading)
        sampleAs(e, sourceOf(e, fromKey), a.from, a.fromTime, a.fromLoop, still, gFrom, nt ? gFromWeights : nullptr);
    if (nt)
    {
        if (fading)
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
    if (fading)
    {
        float f = a.faded / a.fade;
        f = f * f * (3 - 2 * f); // eased
        anim::blendPoses(gFrom, gPoses, n, f);
        memcpy(gPoses, gFrom, sizeof(anim::Pose) * n);
    }
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
        const Entry*          s = sourceOf(e, L.source);
        if (!s || L.clip >= s->skel->clipCount || !sampleAs(e, s, L.clip, L.time, L.loop, still, gLayerPoses, nullptr))
            continue;
        const float w = L.weight < 1 ? L.weight * L.weight * (3 - 2 * L.weight) : 1.0f; // eased, as clip fades
        anim::layerMask(e.nodes, n, L.mask, gLayerMask);
        if (L.additive)
        {
            if (sampleAs(e, s, L.clip, 0, false, still, gLayerRef, nullptr))
                anim::layerAdd(gPoses, gLayerPoses, gLayerRef, gLayerMask, n, w);
        }
        else
            anim::layerOverride(gPoses, gLayerPoses, gLayerMask, n, w);
    }
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
                  apiLayer,  apiUnlayer };

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
        for (uint32_t i = 0; !bad && i < sk->clipCount; ++i)
            if (clips[i].offset < h.skeleton || clips[i].offset + clips[i].bytes > h.textures || !clips[i].frames ||
                clips[i].bytes < (uint64_t)(sk->nodeCount * 3 + nt) * sizeof(RmodelTrack))
                bad = "its file is damaged";
        for (uint32_t i = 0; !bad && i < sk->clipCount; ++i)
            for (uint32_t t = 0; !bad && t < nt; ++t)
            {
                const RmodelTrack& tr = ((const RmodelTrack*)(data + clips[i].offset))[sk->nodeCount * 3 + t];
                if (tr.data != 0xFFFFFFFFu && tr.data + (uint64_t)clips[i].frames * 2 > clips[i].bytes)
                    bad = "its file is damaged";
            }
        // Every track's samples inside its clip, and every bone a vertex names in the skeleton.
        for (uint32_t i = 0; !bad && i < sk->clipCount; ++i)
            for (uint32_t t = 0; !bad && t < sk->nodeCount * 3; ++t)
            {
                const RmodelTrack& tr = ((const RmodelTrack*)(data + clips[i].offset))[t];
                if (tr.data != 0xFFFFFFFFu && tr.data + (uint64_t)clips[i].frames * 6 > clips[i].bytes)
                    bad = "its file is damaged";
            }
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
            motionOf(*e, ce, a.clip, a.time, dt * a.speed, a.loop, to);
            const bool fading = a.from != kNoClip && a.fade > 0 && a.faded < a.fade;
            float      f = fading ? a.faded / a.fade : 1.0f;
            f = f * f * (3 - 2 * f);
            if (fading)
                motionOf(*e, fe, a.from, a.fromTime, dt * a.fromSpeed, a.fromLoop, from);
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
        }
        const Entry* from = gPreview.clipsFrom ? byId(gPreview.clipsFrom) : e; // whose clips the panel lists
        if (!from || !from->anim || from->state != 1)
            from = e, gPreview.clipsFrom = 0;
        if (gPreview.pickClip) // picked in the panel: 1 the rest pose, clip c at c + 2
        {
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
    uint32_t last = 0xFFFFFFFFu, bones = 0, targets = 0; // a placement's parts come together: posed once for all of them
    for (uint32_t i = 0; i < PartInstances.size(); ++i)
    {
        const uint32_t row = PartInstances[i].placement;
        if (row != last)
        {
            last = row, bones = 0, targets = 0;
            const Entry* e = row < rows && Placements[row].alive ? byKey(Placements[row].key) : nullptr;
            const bool re = row < PlacementRetargets.size();
            if (e && e->anim && e->state == 1 &&
                evaluate(*e, PlacementAnims[row], nullptr, nullptr, row | (Placements[row].gen & 0xFFFu) << 20,
                         row < PlacementMotions.size() && PlacementMotions[row].on, re ? PlacementRetargets[row].clip : 0,
                         re ? PlacementRetargets[row].from : 0))
                bones = e->skel->boneCount, targets = e->targets ? e->skel->targetCount : 0;
        }
        if (bones)
            renderer::instance_pose(PartInstances[i].instance, (uint64_t)(uintptr_t)gPalette, bones);
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
    gLayerRef = new anim::Pose[kMaxNodes];
    gLayerMask = new uint8_t[kMaxNodes];
    gSrcWeights = new float[kRmodelMaxTargets];
    gRt = { new float[kMaxNodes * 12], new float[kMaxNodes * 4], new float[kMaxNodes * 12], new float[kMaxNodes * 4] };
    gMaps = new MapSlot[kMaps]();
    gPreview = Preview{};
    gPreview.yaw = 35.0f;
    gPreview.pitch = 20.0f;
    gPreview.distance = 1.35f;
    gPreview.lod = -1;
    gPreview.turntable = 1;
    gPreview.playing = 1, gPreview.speed = 1.0f, gPreview.fade = 0.2f, gPreview.inPlace = 1;
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
    delete[] gLayerRef;
    delete[] gLayerMask;
    gLayerPoses = gLayerRef = nullptr, gLayerMask = nullptr;
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
                 gPreview.srcFrom)) // posed as its clip is now
    {
        p.palette = (uint64_t)(uintptr_t)gPalette, p.bones = e->skel->boneCount;
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
