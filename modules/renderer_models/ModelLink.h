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
// Placements are kept by renderer_models (through its reloads); Reset State empties them, so place again whenever
// mdl::placed(h) is false. The handle is 0 while renderer_models isn't running.
#pragma once

#include <yea/base.h>

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
};

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
} // namespace mdl
