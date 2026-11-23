// Models for the renderer (AGENTS.md beside the component): the registry of every model the view's modules declare
// (ModelLink.h's MODEL lines, through api()), their meshes and materials in the renderer's scene, the game's placements
// of them as the renderer's instances, and the Model Viewer's preview (drawn by the core's preview_draw into this
// module's own targets).
#include "yea_renderer_models.h"
#include "models.h"
#include "rmodel.h"

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
    out->kind = e.kind;
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
mdl::Api gApi = { mdl::kApi, sizeof(mdl::Api), 0, apiBegin, apiDefine, apiEnd, apiKnobs, apiPlace, apiMove, apiRemove, apiPlaced,
                  apiCount, apiInfo };

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
                      : h.version != kRmodelVersion           ? "it was cooked for another version: cook it again"
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
        const float      s = p.scale * e->knobs[mdl::KnobScale];
        const yea_float4 rot = qmul(p.rot, upTurn(*e));
        for (uint32_t k = 0; k < e->parts; ++k)
            renderer::material_double_sided(e->material[k], e->twoSided[k] || e->knobs[mdl::KnobDoubleSided] > 0.5f);
        for (uint32_t i = 0; i < PartInstances.size(); ++i)
            if (PartInstances[i].placement == row)
            {
                const yea_handle h = PartInstances[i].instance;
                renderer::instance_place(h, p.pos, rot, { s, s, s });
                renderer::instance_shadow(h, e->knobs[mdl::KnobCastShadows] > 0.5f);
                renderer::instance_lod_quality(h, e->knobs[mdl::KnobLodQuality]);
            }
        p.dirty = 0;
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
    gPreview = Preview{};
    gPreview.yaw = 35.0f;
    gPreview.pitch = 20.0f;
    gPreview.distance = 1.35f;
    gPreview.lod = -1;
    gPreview.turntable = 1;
    gApi.instance = (uint64_t)host->frame() * 2654435761u ^ (uint64_t)(uintptr_t)gEntries; // models register again
    gInstance = gApi.instance;
    return 0;
}

void unload()
{
    freeTargets();
    delete[] gEntries;
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
    if (renderer::preview_draw((uint64_t)(uintptr_t)&p))
        gPreview.lodShown = p.lodShown, gPreview.triangles = p.triangles;
}
} // namespace renderer_models
