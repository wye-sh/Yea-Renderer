// Models for a game's module, through the Renderer's renderer_models module. Include this in a module's .cpp after its
// yea_<module>.h; its module.yea needs `import renderer_models u64 api(u32 version) = 0`. Each MODEL line then gets
// every knob of renderer_models' each block (chair_scale, chair_up_axis, chair_lod_quality, chair_cast_shadows,
// chair_double_sided), tuned in the Model Viewer.
//
//   MODEL(chair, "models/chair.rmodel");   // a cooked model in this module's Assets/ (tools/model_cook writes it)
//   MODEL(crate, "box");                   // a built-in shape: box, sphere, cylinder, cone, torus, plane
//
//   void update(uint32_t phase, float dt)
//   {
//       mdl::update();                                       // once per update: registers the models, sends knobs
//       if (!gChair) gChair = mdl::place(chair, { 0, 0, 2 }); // a whole model, placed
//       mdl::move(gChair, pos, rot);
//   }
//
// Skinned models play their first clip, looping, once placed. mdl::play(h, "Run") switches clip (fading over a
// fraction of a second by default), mdl::clipTime(h) gives how far in it is, and mdl::joint(h, "hand_r", m) a joint's
// world transform (3x4 rows), to attach something to it. Clips and joints are named as in the source file. A model
// with morph targets (blend shapes: a face's smile, a blink) follows its clip's weights; mdl::morph(h, "Smile", 0.8f)
// holds one at a weight over them, mdl::unmorph(h, "Smile") (or every one, with no name) hands it back to the clip.
// Retargeting: mdl::play(h, animations, "Run") plays another skinned model's clip on this one, as UE's IK Retargeter
// does (its chains matched by their bones' names, the pelvis scaled by the two's heights, IK on the legs and arms so
// feet stay planted), so one model can hold the clips every character uses:
//   MODEL(hero, "models/Hero.rmodel");   MODEL(moves, "models/Moves.rmodel");
//   mdl::play(heroPlacement, moves, "Run");
// Layers (UE's Layered Blend per Bone and additive animation): mdl::layer(h, 1, "Wave", "spine_01") plays Wave over
// whatever h plays, from that bone down (the legs keep walking); additive (true) adds what a clip does since its first
// frame on top instead (a breath, a flinch, a lean), at its weight. Slots apply in order; a new clip in a slot crossfades
// from the old; mdl::unlayer(h, 1) fades it out. Layers carry no root motion and leave morph weights to the base clip.
//   mdl::layer(hero, 1, "Wave", "spine_01", 1.0f);  mdl::layer(hero, 2, "Breathe", nullptr, 0.6f, true);
// Root motion: mdl::rootMotion(h, true) plays its clips in place (the travel and turn of the node that carries them left
// out), and each frame mdl::takeRootMotion(h, m) hands over what was left out, for the game to move it by (as UE's
// root motion: the game owns where the character is, so it can collide, follow a path or be networked):
//   float m[4];
//   if (mdl::takeRootMotion(h, m)) { pos += { m[0], m[1], m[2] }; yaw += m[3]; mdl::move(h, pos, yawQuat(yaw)); }
// Foot IK (UE's foot placement): mdl::footIk(h, true) plants its feet on the game's ground, on slopes and steps: each
// leg bent so its foot stands on the ground under it, tilted to it while the clip has it down, and the pelvis lowered to
// the lower foot. The renderer has no physics, so the game traces its own ground: each frame, for each foot (0 left,
// 1 right), mdl::footAt says where its clip puts it, and mdl::footGround hands over the ground found below:
//   for (int f = 0; f < 2; ++f)
//       if (yea_float3 p; mdl::footAt(h, f, &p) && myTrace(p, &hit, &normal)) mdl::footGround(h, f, hit, normal);
// Blend spaces (UE's Blend Space and Blend Space 1D): mdl::blendSpace(h, samples, n, dims) plays clips placed in a space
// of one or two dimensions (speed; speed and direction) as a placement's clip, crossfaded from what it played as play()
// does; each frame mdl::blendAt(h, x, y) says where in the space it is, and the clips nearest that point blend (1D:
// the two either side; 2D: the three corners of its triangle), every one in step, so feet keep their rhythm as a
// walk becomes a run. play() leaves it; root motion, retargeting, layers and foot IK work on it as on a clip:
//   static const mdl::BlendSample kMove[] = { { "Idle", 0, 0 }, { "Walk", 0, 1.5f }, { "Run", 0, 4 },
//                                             { "StrafeL", -1.5f, 0 }, { "StrafeR", 1.5f, 0 }, { "Back", 0, -1.5f } };
//   mdl::blendSpace(h, kMove, 6, 2);            // once (the same samples again change nothing)
//   mdl::blendAt(h, sideways, forward, 0.1f);   // each frame: metres a second, followed over 0.1 s
// Look-at and hand IK (UE's LookAt and Two Bone IK): mdl::lookAt(h, target) turns the head, neck and the top of the
// spine toward a point, each a share of the turn within its own limit; mdl::handIk(h, 1, grip, gripTurn) brings the
// right hand to a point (and turn), the elbow bending as the clip bends it, to hold a sword or grab a ledge:
//   mdl::lookAt(h, enemyHead);   mdl::handIk(h, 0, ledgePoint);   mdl::stopHandIk(h, 0);
// Placements are kept by renderer_models (through its reloads); Reset State empties them, so place again whenever
// mdl::placed(h) is false. The handle is 0 while renderer_models isn't running.
#pragma once

#include <yea/base.h>

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

namespace mdl
{
constexpr uint32_t kApi = 1; // raised when Api or what it takes changes shape

// Every model's knobs, in this order (renderer_models' each block makes them for each MODEL line).
enum Knob : uint32_t
{
    KnobScale,
    KnobUpAxis, // 0 Auto, 1 Y, 2 Z
    KnobLodQuality,
    KnobCastShadows,
    KnobDoubleSided,
    kKnobs
};
struct KnobSpec
{
    const char* suffix;
    const char* title;
};
constexpr KnobSpec kKnobSpecs[kKnobs] = { { "scale", "Scale" },
                                          { "up_axis", "Up Axis" },
                                          { "lod_quality", "LOD Quality" },
                                          { "cast_shadows", "Cast Shadows" },
                                          { "double_sided", "Double Sided" } };

struct Spec
{
    const char* owner;  // the module's folder name
    const char* name;   // the MODEL line's
    const char* source; // a built-in shape's name, or its file under the owner's Assets/
    const char* assets; // the owner's Assets/ folder
    float       knobs[kKnobs];
};
// For a panel: one registered model.
struct Info
{
    const char* owner;
    const char* name;
    const char* source;
    uint32_t    id;
    uint32_t    kind;      // 0 built-in shape, 1 static model, 2 skinned
    uint32_t    state;     // 0 waiting, 1 ready, 2 failed (why: `error`)
    uint32_t    parts, lods;
    uint32_t    triangles; // its finest version's
    uint32_t    placed;    // placements now
    float       radius;    // its bounding sphere's, as cooked (before its scale)
    const char* error;
};

// One clip of a blend space: its name, and where it stands (x; and y in two dimensions).
struct BlendSample
{
    const char* clip;
    float       x, y;
};

struct Api
{
    uint32_t version, size;
    uint64_t instance; // changes whenever renderer_models starts: models are registered again
    void (*begin)(const char* owner);
    uint32_t (*define)(const Spec* spec, char* error, uint32_t errorBytes); // its id, or 0 and why
    void (*end)(const char* owner); // the owner's models not defined since begin() go, with their placements
    void (*knobs)(uint32_t model, const float* values); // kKnobs
    // A placement of a whole model: its handle (0 none). pos metres, rot a quaternion (x, y, z, w), scale on top of
    // its Scale knob.
    uint32_t (*place)(uint32_t model, const float* pos, const float* rot, float scale);
    void (*move)(uint32_t placement, const float* pos, const float* rot, float scale);
    void (*remove)(uint32_t placement);
    int (*placed)(uint32_t placement); // 1 while it stands (Reset State and its model going end it)
    // For a panel (the Model Viewer): every model, in the order declared.
    uint32_t (*count)(void);
    int (*info)(uint32_t index, Info* out);
    // Skinned models (later in Api: check `size`). play: by name (nullptr or "" the rest pose); speed 1 as authored,
    // fade seconds from what played; 0 when it has no such clip.
    int (*play)(uint32_t placement, const char* clip, float speed, int loop, float fade);
    float (*time)(uint32_t placement); // seconds into its clip
    int (*joint)(uint32_t placement, const char* name, float* out); // 12 floats, 3x4 rows, world; 0 no such joint
    uint32_t (*clips)(uint32_t model); // its clips, for a panel
    const char* (*clip)(uint32_t model, uint32_t index, float* duration);
    // Morph targets (later in Api: check `size`). morph: a target's weight held over its clip's; 0 no such target.
    // unmorph: a target (nullptr: every one) back to its clip's.
    int (*morph)(uint32_t placement, const char* target, float weight);
    void (*unmorph)(uint32_t placement, const char* target);
    uint32_t (*morphs)(uint32_t model); // its targets, for a panel
    const char* (*morphTarget)(uint32_t model, uint32_t index, float* weight); // its name, and its default weight
    // Root motion (later in Api: check `size`). rootMotion: on, its clips play in place; 0 when it has none to give.
    // takeMotion: what they left out since the last take, world (out[0..2] metres, out[3] radians about +Y).
    int (*rootMotion)(uint32_t placement, int on);
    int (*takeMotion)(uint32_t placement, float* out);
    // Retargeting (later in Api: check `size`): another skinned model's clip, played on this one's skeleton; 0 when
    // either isn't skinned or the source has no such clip.
    int (*playFrom)(uint32_t placement, uint32_t model, const char* clip, float speed, int loop, float fade);
    // Layers (later in Api: check `size`). layer: slot's clip (model 0: its own) over the placement's pose from `bone`
    // down (nullptr or "": all), overriding it or (additive) added to it, faded to `weight` over `fade` seconds; 0 when
    // the clip or bone isn't there. unlayer: the slot faded out and gone.
    int (*layer)(uint32_t placement, uint32_t slot, uint32_t model, const char* clip, const char* bone, float weight, int additive,
                 float speed, int loop, float fade);
    void (*unlayer)(uint32_t placement, uint32_t slot, float fade);
    // Foot IK (later in Api: check `size`). footIk: on or off, faded over `fade` seconds; `reach` metres the most a foot
    // moves from its clip's place; 0 when it has no two legs. footGround: the ground under a foot (0 left, 1 right),
    // world point and normal (point nullptr: none). footAt: where its clip put the foot last frame, world; 0 none yet.
    int (*footIk)(uint32_t placement, int on, float reach, float fade);
    int (*footGround)(uint32_t placement, uint32_t foot, const float* point, const float* normal);
    int (*footAt)(uint32_t placement, uint32_t foot, float* out);
    // Blend spaces (later in Api: check `size`). blendSpace: `count` samples (1 to 16; model 0: its own clips) in `dims`
    // (1 or 2) dimensions as its clip, faded to over `fade` seconds; 0 when a clip isn't there. blendAt: the point it
    // plays, followed over `smooth` seconds (0: at once).
    int (*blendSpace)(uint32_t placement, uint32_t model, const BlendSample* samples, uint32_t count, uint32_t dims, float speed,
                      int loop, float fade);
    void (*blendAt)(uint32_t placement, float x, float y, float smooth);
    // Look-at and hand IK (later in Api: check `size`). lookAt: on or off, the target (world; nullptr keeps the last),
    // faded to `weight` (0 to 1) over `fade` seconds, the head turned at most `limit` radians from where its clip points
    // it; 0 when it has no spine, neck or head found. handIk: hand (0 left, 1 right) on or off, reaching `point` (world;
    // nullptr keeps the last) and turned to `rot` there (world, x y z w; nullptr: the clip's), faded over `fade`
    // seconds; 0 when it has no two arms.
    int (*lookAt)(uint32_t placement, int on, const float* target, float weight, float limit, float fade);
    int (*handIk)(uint32_t placement, uint32_t hand, int on, const float* point, const float* rot, float fade);
};
inline bool hasAnim(const Api* a) { return a->size >= offsetof(Api, morph); }
inline bool hasMorph(const Api* a) { return a->size >= offsetof(Api, rootMotion); }
inline bool hasRootMotion(const Api* a) { return a->size >= offsetof(Api, playFrom); }
inline bool hasRetarget(const Api* a) { return a->size >= offsetof(Api, layer); }
inline bool hasLayers(const Api* a) { return a->size >= offsetof(Api, footIk); }
inline bool hasFootIk(const Api* a) { return a->size >= offsetof(Api, blendSpace); }
inline bool hasBlendSpaces(const Api* a) { return a->size >= offsetof(Api, lookAt); }
inline bool hasAim(const Api* a) { return a->size >= sizeof(Api); }

struct Model;
namespace detail
{
inline Model*   gModels = nullptr; // this module's, newest first
inline uint64_t gInstance = 0;     // renderer_models' instance they were registered with
inline bool     gWarned = false;
} // namespace detail

struct Model
{
    const char*         name;
    const char*         source;
    const float* const* knobs;   // its float knobs (scale, lod_quality)
    const int32_t*      upAxis;  // its choice
    const bool* const*  flags;   // cast_shadows, double_sided
    const char*         file;    // where it was declared: its module's folder names its owner
    uint32_t            id = 0;
    float               sent[kKnobs] = {};
    Model*              next = nullptr;
    Model(const char* n, const char* s, const float* const* k, const int32_t* u, const bool* const* f, const char* where)
        : name(n), source(s), knobs(k), upAxis(u), flags(f), file(where)
    {
        next = detail::gModels, detail::gModels = this;
    }
    Model(const Model&) = delete;
    void values(float* out) const
    {
        out[KnobScale] = *knobs[0];
        out[KnobUpAxis] = (float)*upAxis;
        out[KnobLodQuality] = *knobs[1];
        out[KnobCastShadows] = *flags[0] ? 1.0f : 0.0f;
        out[KnobDoubleSided] = *flags[1] ? 1.0f : 0.0f;
    }
};

namespace detail
{
// ".../modules/props/Props.cpp" -> "props"
inline void ownerOf(const char* file, char* out, size_t bytes)
{
    const char* end = nullptr;
    const char* start = file;
    for (const char* p = file; *p; ++p)
        if (*p == '/' || *p == '\\')
            start = end ? end + 1 : file, end = p;
    size_t n = end ? size_t(end - start) : 0;
    n = n < bytes - 1 ? n : bytes - 1;
    memcpy(out, start, n);
    out[n] = 0;
}
// Each model from `m` on, the oldest first, so the registry keeps the order declared.
inline void define(const Api* a, Model* m, const char* owner, const char* assets)
{
    if (!m)
        return;
    define(a, m->next, owner, assets);
    Spec spec = { owner, m->name, m->source, assets, {} };
    m->values(spec.knobs);
    memcpy(m->sent, spec.knobs, sizeof m->sent);
    char error[200] = "";
    m->id = a->define(&spec, error, sizeof error);
    if (!m->id)
    {
        char message[300];
        snprintf(message, sizeof message, "model %s: %s", m->name, error);
        yea::host()->log(2, message);
    }
}
// renderer_models' calls, with this module's models registered; nullptr while it isn't running.
inline const Api* api()
{
    const Api* a = (const Api*)(uintptr_t)::renderer_models::api(kApi);
    if (!a)
    {
        if (!gWarned && ::renderer_models::api(0))
            yea::host()->log(1, "renderer_models is a different version: rebuild this module"), gWarned = true;
        return nullptr;
    }
    if (a->instance != gInstance)
    {
        gInstance = a->instance;
        char owner[64];
        ownerOf(gModels ? gModels->file : "", owner, sizeof owner);
        a->begin(owner);
        define(a, gModels, owner, yea::assets()); // this module's own update: its own Assets/
        a->end(owner);
    }
    return a;
}
} // namespace detail

#define MDL_KNOBS(p) { &settings.p##_scale, &settings.p##_lod_quality }
#define MDL_FLAGS(p) { &settings.p##_cast_shadows, &settings.p##_double_sided }
// MODEL(name, "source"): the handle is `name`; its knobs are settings.name_scale and the rest of kKnobSpecs.
#define MODEL(name, source)                                                                                        \
    namespace name##_model {                                                                                       \
    static const float* const kKnobs[] = MDL_KNOBS(name);                                                          \
    static const bool* const  kFlags[] = MDL_FLAGS(name);                                                          \
    ::mdl::Model model(#name, source, kKnobs, &settings.name##_up_axis, kFlags, __FILE__);                         \
    }                                                                                                              \
    ::mdl::Model& name = name##_model::model

// Once per update (any phase): registers this module's models with renderer_models (again whenever it restarts) and
// sends knob changes.
inline void update()
{
    const Api* a = detail::api();
    if (!a)
        return;
    for (Model* m = detail::gModels; m; m = m->next)
    {
        float now[kKnobs];
        m->values(now);
        if (m->id && memcmp(now, m->sent, sizeof now))
        {
            memcpy(m->sent, now, sizeof now);
            a->knobs(m->id, now);
        }
    }
}
inline uint32_t place(const Model& m, yea_float3 pos, yea_float4 rot = { 0, 0, 0, 1 }, float scale = 1.0f)
{
    const Api*  a = detail::api();
    const float p[3] = { pos.x, pos.y, pos.z }, r[4] = { rot.x, rot.y, rot.z, rot.w };
    return a && m.id ? a->place(m.id, p, r, scale) : 0;
}
inline void move(uint32_t placement, yea_float3 pos, yea_float4 rot = { 0, 0, 0, 1 }, float scale = 1.0f)
{
    const Api*  a = detail::api();
    const float p[3] = { pos.x, pos.y, pos.z }, r[4] = { rot.x, rot.y, rot.z, rot.w };
    if (a && placement)
        a->move(placement, p, r, scale);
}
inline void remove(uint32_t placement)
{
    const Api* a = detail::api();
    if (a && placement)
        a->remove(placement);
}
inline bool placed(uint32_t placement)
{
    const Api* a = detail::api();
    return a && placement && a->placed(placement);
}
// A skinned placement's clip: by name, speed 1 as authored, faded in over `fade` seconds. False: no such clip (or not
// skinned, or renderer_models isn't running). While its model is still loading it is taken by name once loaded (true).
inline bool play(uint32_t placement, const char* clip, float speed = 1.0f, bool loop = true, float fade = 0.2f)
{
    const Api* a = detail::api();
    return a && placement && hasAnim(a) && a->play(placement, clip, speed, loop ? 1 : 0, fade);
}
inline float clipTime(uint32_t placement)
{
    const Api* a = detail::api();
    return a && placement && hasAnim(a) ? a->time(placement) : 0.0f;
}
// A joint's world transform this frame (3x4 rows: x, y and z axes in its columns, then its position), to attach
// something to it. False when it has none of that name.
inline bool joint(uint32_t placement, const char* name, float out[12])
{
    const Api* a = detail::api();
    return a && placement && hasAnim(a) && a->joint(placement, name, out);
}
// A morph target held at `weight` (usually 0 to 1) over what its clip says. False: no such target (or no morphs, or
// renderer_models isn't running).
inline bool morph(uint32_t placement, const char* target, float weight)
{
    const Api* a = detail::api();
    return a && placement && hasMorph(a) && a->morph(placement, target, weight);
}
// Another skinned model's clip (`from`'s, by name; nullptr or "" the rest pose) played on this one, retargeted onto its
// skeleton, crossfaded as play() does. False: either isn't skinned, or `from` has no such clip.
inline bool play(uint32_t placement, const Model& from, const char* clip, float speed = 1.0f, bool loop = true, float fade = 0.2f)
{
    const Api* a = detail::api();
    return a && placement && from.id && hasRetarget(a) && a->playFrom(placement, from.id, clip, speed, loop ? 1 : 0, fade);
}
// A clip as layer `slot` over what it plays (its own clips, or `from`'s retargeted onto it): from `bone` down (nullptr
// or "": the whole body), overriding the pose there or, `additive`, adding what the clip does since its first frame;
// faded to `weight` over `fade` seconds. False: no such clip or bone (or no layers in this renderer_models).
inline bool layer(uint32_t placement, uint32_t slot, const char* clip, const char* bone = nullptr, float weight = 1.0f,
                  bool additive = false, float fade = 0.2f, float speed = 1.0f, bool loop = true)
{
    const Api* a = detail::api();
    return a && placement && hasLayers(a) && a->layer(placement, slot, 0, clip, bone, weight, additive ? 1 : 0, speed, loop ? 1 : 0, fade);
}
inline bool layer(uint32_t placement, uint32_t slot, const Model& from, const char* clip, const char* bone = nullptr,
                  float weight = 1.0f, bool additive = false, float fade = 0.2f, float speed = 1.0f, bool loop = true)
{
    const Api* a = detail::api();
    return a && placement && from.id && hasLayers(a) &&
           a->layer(placement, slot, from.id, clip, bone, weight, additive ? 1 : 0, speed, loop ? 1 : 0, fade);
}
// Layer `slot` faded out over `fade` seconds, then gone.
inline void unlayer(uint32_t placement, uint32_t slot, float fade = 0.2f)
{
    const Api* a = detail::api();
    if (a && placement && hasLayers(a))
        a->unlayer(placement, slot, fade);
}
// Root motion on (its clips play in place) or off (as authored). False: it isn't skinned.
inline bool rootMotion(uint32_t placement, bool on)
{
    const Api* a = detail::api();
    return a && placement && hasRootMotion(a) && a->rootMotion(placement, on ? 1 : 0);
}
// The motion its clips left out since the last take, in the world: out[0..2] metres to move it by, out[3] radians to
// turn it by about +Y (placed upright). False (and zeros) while root motion is off.
inline bool takeRootMotion(uint32_t placement, float out[4])
{
    const Api* a = detail::api();
    out[0] = out[1] = out[2] = out[3] = 0;
    return a && placement && hasRootMotion(a) && a->takeMotion(placement, out);
}
// Foot IK on (faded in over `fade` seconds) or off: its feet planted on the ground footGround gives, never moved more
// than `reach` metres from where its clip puts them (ground further off is left alone). False: it hasn't two legs.
inline bool footIk(uint32_t placement, bool on, float reach = 0.5f, float fade = 0.2f)
{
    const Api* a = detail::api();
    return a && placement && hasFootIk(a) && a->footIk(placement, on ? 1 : 0, reach, fade);
}
// The ground under `foot` (0 left, 1 right): a point on it and its normal, world, as the game's trace found it below
// footAt. Kept until given again; a new one is followed smoothly (a step's edge never pops the foot).
inline bool footGround(uint32_t placement, int foot, yea_float3 point, yea_float3 normal = { 0, 1, 0 })
{
    const Api*  a = detail::api();
    const float p[3] = { point.x, point.y, point.z }, n[3] = { normal.x, normal.y, normal.z };
    return a && placement && hasFootIk(a) && a->footGround(placement, (uint32_t)foot, p, n);
}
// No ground under `foot` (nothing below it): it follows its clip.
inline void footNoGround(uint32_t placement, int foot)
{
    const Api* a = detail::api();
    if (a && placement && hasFootIk(a))
        a->footGround(placement, (uint32_t)foot, nullptr, nullptr);
}
// Where its clip put `foot` (its ankle, world) last frame, before foot IK: trace the ground below it. False before
// foot IK's first frame.
inline bool footAt(uint32_t placement, int foot, yea_float3* out)
{
    const Api* a = detail::api();
    float      p[3];
    if (!a || !placement || !hasFootIk(a) || !a->footAt(placement, (uint32_t)foot, p))
        return false;
    *out = { p[0], p[1], p[2] };
    return true;
}
// A blend space as its clip: `count` samples in `dims` (1 or 2) dimensions, its own clips or `from`'s retargeted onto it,
// faded to over `fade` seconds; the same samples again keep playing where they are. False: a clip isn't there (or no
// blend spaces in this renderer_models).
inline bool blendSpace(uint32_t placement, const BlendSample* samples, uint32_t count, int dims = 1, float fade = 0.2f, float speed = 1.0f,
                       bool loop = true)
{
    const Api* a = detail::api();
    return a && placement && hasBlendSpaces(a) && a->blendSpace(placement, 0, samples, count, (uint32_t)dims, speed, loop ? 1 : 0, fade);
}
inline bool blendSpace(uint32_t placement, const Model& from, const BlendSample* samples, uint32_t count, int dims = 1, float fade = 0.2f,
                       float speed = 1.0f, bool loop = true)
{
    const Api* a = detail::api();
    return a && placement && from.id && hasBlendSpaces(a) &&
           a->blendSpace(placement, from.id, samples, count, (uint32_t)dims, speed, loop ? 1 : 0, fade);
}
// Look-at: its head, neck and the top of its spine turned toward `target` (world), each bone a share of the turn and
// never past its own limit, the head at most `limitDegrees` from where its clip points it; faded to `weight` over
// `fade` seconds. Call it again as the target moves (it follows at once). False: no spine, neck or head found.
inline bool lookAt(uint32_t placement, yea_float3 target, float weight = 1.0f, float limitDegrees = 70.0f, float fade = 0.2f)
{
    const Api*  a = detail::api();
    const float t[3] = { target.x, target.y, target.z };
    return a && placement && hasAim(a) && a->lookAt(placement, 1, t, weight, limitDegrees * 0.01745329f, fade);
}
// Look-at faded out over `fade` seconds: its clip's own head again.
inline void stopLookAt(uint32_t placement, float fade = 0.2f)
{
    const Api* a = detail::api();
    if (a && placement && hasAim(a))
        a->lookAt(placement, 0, nullptr, 0, 0, fade);
}
// Hand IK: `hand` (0 left, 1 right) reaching `point` (world: its wrist there, as mdl::joint places it) with the elbow
// bent the way its clip bends it, faded in over `fade` seconds; with `rotation` (world), the hand also turned to it (a
// grip on a hilt, a palm on a ledge). Call it again as the point moves. False: no two arms found.
inline bool handIk(uint32_t placement, int hand, yea_float3 point, float fade = 0.2f)
{
    const Api*  a = detail::api();
    const float p[3] = { point.x, point.y, point.z };
    return a && placement && hasAim(a) && a->handIk(placement, (uint32_t)hand, 1, p, nullptr, fade);
}
inline bool handIk(uint32_t placement, int hand, yea_float3 point, yea_float4 rotation, float fade = 0.2f)
{
    const Api*  a = detail::api();
    const float p[3] = { point.x, point.y, point.z }, r[4] = { rotation.x, rotation.y, rotation.z, rotation.w };
    return a && placement && hasAim(a) && a->handIk(placement, (uint32_t)hand, 1, p, r, fade);
}
// Hand IK faded out over `fade` seconds: the hand follows its clip again.
inline void stopHandIk(uint32_t placement, int hand, float fade = 0.2f)
{
    const Api* a = detail::api();
    if (a && placement && hasAim(a))
        a->handIk(placement, (uint32_t)hand, 0, nullptr, nullptr, fade);
}
// Where in its blend space it plays (y for two dimensions), followed over `smooth` seconds (0: at once).
inline void blendAt(uint32_t placement, float x, float y = 0.0f, float smooth = 0.0f)
{
    const Api* a = detail::api();
    if (a && placement && hasBlendSpaces(a))
        a->blendAt(placement, x, y, smooth);
}
// A held target (nullptr: every one) back to its clip's weight.
inline void unmorph(uint32_t placement, const char* target = nullptr)
{
    const Api* a = detail::api();
    if (a && placement && hasMorph(a))
        a->unmorph(placement, target);
}
} // namespace mdl
