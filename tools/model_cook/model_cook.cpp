// The Renderer's model cooker: a glTF or GLB file in, a .rmodel (modules/renderer_models/rmodel.h) out, with a report.
// Agents run it in their orb when a model arrives; it is never shipped or compiled into a module.
//
//   build.sh                                   (once: builds build/tools/model_cook)
//   build/tools/model_cook in.glb out.rmodel
//
// What it does, in order:
// - Reads the default scene's mesh nodes (cgltf; EXT_meshopt_compression decoded), each primitive's triangles with its
//   node's world transform, and normalises them: glTF is metres and +Y up (any node scale, a centimetre file's 0.01,
//   is applied); the renderer is left-handed, so Z is negated (the model's front, glTF's +Z, faces -Z: towards a camera
//   looking down +Z) and the triangles' order flipped, so they stay counter-clockwise around their outward normal.
// - Missing normals: flat (as glTF says). Missing tangents: generated MikkTSpace-compatible (meshoptimizer), so normal
//   maps baked against MikkTSpace shade right. Given tangents keep their handedness through the mirrors.
// - Merges every primitive of one material into one part, welds equal vertices, orders for the vertex cache and fetch.
// - Builds each part's LOD chain with meshoptimizer's attribute-aware simplifier: normals and UVs weighted, so shading
//   and texturing hold; UV and normal seams and hard edges (split vertices) collapse only along themselves; vertices on
//   a border with another part (a material border) are locked, so parts never open a crack between them. Each level
//   aims at half the triangles of the one before and records its error in metres, which only grows down the chain;
//   the chain stops when the simplifier can't take a tenth off, the error passes a quarter of the part's size, or a
//   level has under 64 triangles.
// - Writes the file and prints the report: units and axes, parts, triangles and error per LOD, what it generated or
//   ignored. Skins, animations, Draco and other extensions come with later stages and are listed as ignored.
// - Texture maps (rmodel v2): each image used as base colour, ORM, normal or emissive is decoded (stb_image) and written
//   with its full mip chain, BC7 (sRGB for colour) or BC5 for normals (bc7e, Apache-2.0).
#define CGLTF_IMPLEMENTATION
#include "vendor/cgltf/cgltf.h"
#include "vendor/meshoptimizer/meshoptimizer.h"
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#include "vendor/stb/stb_image.h"

#include "../../modules/renderer_models/rmodel.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace
{
struct Vertex // a corner with everything that tells two apart (welded by bytes)
{
    float    p[3];
    float    n[3];
    float    t[4];
    float    uv[2];
    uint32_t color;
};

struct Part
{
    uint32_t              material;
    std::vector<Vertex>   corners; // three per triangle, before welding
    std::vector<Vertex>   vertices;
    std::vector<uint32_t> indices;
    float                 lo[3] = { 1e30f, 1e30f, 1e30f }, hi[3] = { -1e30f, -1e30f, -1e30f };
    struct Level
    {
        std::vector<float>    positions;  // float3
        std::vector<uint32_t> attributes; // 5 each
        std::vector<uint32_t> indices;
        float                 error;
    };
    std::vector<Level> levels;
};

std::vector<std::string> gNotes, gWarnings;

void note(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
void note(const char* fmt, ...)
{
    char    s[512];
    va_list a;
    va_start(a, fmt);
    vsnprintf(s, sizeof s, fmt, a);
    va_end(a);
    gNotes.push_back(s);
}
void warn(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
void warn(const char* fmt, ...)
{
    char    s[512];
    va_list a;
    va_start(a, fmt);
    vsnprintf(s, sizeof s, fmt, a);
    va_end(a);
    for (const std::string& w : gWarnings)
        if (w == s)
            return;
    gWarnings.push_back(s);
}

// The same packing as the renderer's (scene.cpp octPack): of the nearby roundings, the one that decodes closest.
uint32_t octPack(float x, float y, float z, int parity)
{
    float l = fabsf(x) + fabsf(y) + fabsf(z);
    if (!(l > 0))
        x = 0, y = 0, z = 1, l = 1;
    float u = x / l, v = y / l;
    if (z < 0)
    {
        float ou = u;
        u = (1 - fabsf(v)) * (ou >= 0 ? 1.0f : -1.0f);
        v = (1 - fabsf(ou)) * (v >= 0 ? 1.0f : -1.0f);
    }
    float   len = sqrtf(x * x + y * y + z * z);
    float   best = -2;
    int32_t bu = 0, bv = 0;
    int32_t fu = (int32_t)floorf(u * 32767.0f), fv = (int32_t)floorf(v * 32767.0f);
    for (int32_t i = 0; i < 2; ++i)
        for (int32_t j = -1; j < 3; ++j)
        {
            int32_t cu = fu + i, cv = fv + j;
            cu = cu < -32767 ? -32767 : (cu > 32767 ? 32767 : cu);
            cv = cv < -32767 ? -32767 : (cv > 32767 ? 32767 : cv);
            if ((parity < 0 && !(cv & 1)) || (parity > 0 && (cv & 1)) || (parity == 0 && (j < 0 || j > 1)))
                continue;
            float dx = (float)cu / 32767.0f, dy = (float)cv / 32767.0f, dz = 1 - fabsf(dx) - fabsf(dy);
            float t = dz < 0 ? -dz : 0;
            dx += dx >= 0 ? -t : t;
            dy += dy >= 0 ? -t : t;
            float d = (dx * x + dy * y + dz * z) / (sqrtf(dx * dx + dy * dy + dz * dz) * len);
            if (d > best)
                best = d, bu = cu, bv = cv;
        }
    return (uint32_t)(uint16_t)(int16_t)bu | ((uint32_t)(uint16_t)(int16_t)bv << 16);
}

void normalize3(float* v)
{
    float l = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (l > 1e-20f)
        v[0] /= l, v[1] /= l, v[2] /= l;
    else
        v[0] = 0, v[1] = 1, v[2] = 0;
}

// EXT_meshopt_compression: each compressed view decoded into its own data (cgltf reads it from there).
bool decodeMeshopt(cgltf_data* data)
{
    for (cgltf_size i = 0; i < data->buffer_views_count; ++i)
    {
        cgltf_buffer_view& view = data->buffer_views[i];
        if (!view.has_meshopt_compression)
            continue;
        const cgltf_meshopt_compression& mc = view.meshopt_compression;
        const uint8_t*                   src = mc.buffer->data ? (const uint8_t*)mc.buffer->data + mc.offset : nullptr;
        if (!src)
            return false;
        void* out = malloc(mc.count * mc.stride);
        int   r = -1;
        if (mc.mode == cgltf_meshopt_compression_mode_attributes)
            r = meshopt_decodeVertexBuffer(out, mc.count, mc.stride, src, mc.size);
        else if (mc.mode == cgltf_meshopt_compression_mode_triangles)
            r = meshopt_decodeIndexBuffer(out, mc.count, mc.stride, src, mc.size);
        else if (mc.mode == cgltf_meshopt_compression_mode_indices)
            r = meshopt_decodeIndexSequence(out, mc.count, mc.stride, src, mc.size);
        if (r != 0)
        {
            free(out);
            return false;
        }
        if (mc.filter == cgltf_meshopt_compression_filter_octahedral)
            meshopt_decodeFilterOct(out, mc.count, mc.stride);
        else if (mc.filter == cgltf_meshopt_compression_filter_quaternion)
            meshopt_decodeFilterQuat(out, mc.count, mc.stride);
        else if (mc.filter == cgltf_meshopt_compression_filter_exponential)
            meshopt_decodeFilterExp(out, mc.count, mc.stride);
        view.data = out; // freed by cgltf_free
    }
    return true;
}

const cgltf_accessor* attribute(const cgltf_primitive& p, cgltf_attribute_type type, int index)
{
    for (cgltf_size i = 0; i < p.attributes_count; ++i)
        if (p.attributes[i].type == type && p.attributes[i].index == index)
            return p.attributes[i].data;
    return nullptr;
}

// One primitive's triangles, in the renderer's space, appended to its material's part.
void addPrimitive(const cgltf_data* data, const cgltf_primitive& prim, const float* m, std::vector<Part>& parts, bool& flatNormals,
                  bool& madeTangents)
{
    if (prim.type != cgltf_primitive_type_triangles && prim.type != cgltf_primitive_type_triangle_strip &&
        prim.type != cgltf_primitive_type_triangle_fan)
    {
        warn("points and lines are ignored (the renderer draws triangles)");
        return;
    }
    if (prim.has_draco_mesh_compression)
    {
        warn("a Draco-compressed primitive was skipped (Draco comes with FBX and OBJ, a later stage)");
        return;
    }
    const cgltf_accessor* pos = attribute(prim, cgltf_attribute_type_position, 0);
    if (!pos || !pos->count)
        return;
    const cgltf_accessor* nor = attribute(prim, cgltf_attribute_type_normal, 0);
    const cgltf_accessor* tan = attribute(prim, cgltf_attribute_type_tangent, 0);
    const cgltf_accessor* uv0 = attribute(prim, cgltf_attribute_type_texcoord, 0);
    const cgltf_accessor* col = attribute(prim, cgltf_attribute_type_color, 0);
    if (attribute(prim, cgltf_attribute_type_joints, 0))
        warn("skinning is ignored: the model is cooked in its bind pose (skinned models come with animation)");
    if (prim.targets_count)
        warn("morph targets are ignored");
    if (attribute(prim, cgltf_attribute_type_texcoord, 1))
        warn("a second UV set is ignored");

    // Indices, as a triangle list.
    std::vector<uint32_t> idx;
    const cgltf_size      n = prim.indices ? prim.indices->count : pos->count;
    std::vector<uint32_t> raw(n);
    for (cgltf_size i = 0; i < n; ++i)
        raw[i] = prim.indices ? (uint32_t)cgltf_accessor_read_index(prim.indices, i) : (uint32_t)i;
    if (prim.type == cgltf_primitive_type_triangles)
        idx.assign(raw.begin(), raw.end() - n % 3);
    else
        for (cgltf_size i = 2; i < n; ++i)
        {
            uint32_t a = prim.type == cgltf_primitive_type_triangle_fan ? raw[0] : raw[i - 2], b = raw[i - 1], c = raw[i];
            if (prim.type == cgltf_primitive_type_triangle_strip && (i & 1))
                std::swap(a, b);
            idx.push_back(a), idx.push_back(b), idx.push_back(c);
        }

    // The world transform (column-major), its normal matrix (inverse transpose, by cofactors over the determinant).
    const float a00 = m[0], a10 = m[1], a20 = m[2], a01 = m[4], a11 = m[5], a21 = m[6], a02 = m[8], a12 = m[9], a22 = m[10];
    const float c[3][3] = { { a11 * a22 - a12 * a21, a12 * a20 - a10 * a22, a10 * a21 - a11 * a20 },
                            { a02 * a21 - a01 * a22, a00 * a22 - a02 * a20, a01 * a20 - a00 * a21 },
                            { a01 * a12 - a02 * a11, a02 * a10 - a00 * a12, a00 * a11 - a01 * a10 } };
    const float det = a00 * c[0][0] + a01 * c[0][1] + a02 * c[0][2];
    const float ds = det < 0 ? -1.0f : 1.0f;
    // glTF: a positive determinant keeps counter-clockwise fronts; negating Z mirrors once more. Order flips when det > 0.
    const bool flip = det > 0;

    const uint32_t mat = prim.material ? (uint32_t)(prim.material - data->materials) : (uint32_t)data->materials_count;
    Part*          part = nullptr;
    for (Part& p : parts)
        if (p.material == mat)
            part = &p;
    if (!part)
    {
        parts.push_back(Part{});
        part = &parts.back();
        part->material = mat;
    }
    const size_t first = part->corners.size();
    for (size_t t = 0; t + 2 < idx.size(); t += 3)
    {
        const uint32_t tri[3] = { idx[t], flip ? idx[t + 2] : idx[t + 1], flip ? idx[t + 1] : idx[t + 2] };
        if (tri[0] >= pos->count || tri[1] >= pos->count || tri[2] >= pos->count)
        {
            warn("a primitive indexes past its vertices: those triangles were dropped");
            continue;
        }
        for (uint32_t k = 0; k < 3; ++k)
        {
            const uint32_t v = tri[k];
            Vertex         o = {};
            float          p[3] = {}, q[4] = { 0, 0, 0, 1 };
            cgltf_accessor_read_float(pos, v, p, 3);
            o.p[0] = m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12];
            o.p[1] = m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13];
            o.p[2] = -(m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14]);
            if (nor && cgltf_accessor_read_float(nor, v, p, 3))
            {
                for (int r = 0; r < 3; ++r)
                    o.n[r] = (c[r][0] * p[0] + c[r][1] * p[1] + c[r][2] * p[2]) * ds;
                o.n[2] = -o.n[2];
                normalize3(o.n);
            }
            if (tan && cgltf_accessor_read_float(tan, v, q, 4))
            {
                o.t[0] = m[0] * q[0] + m[4] * q[1] + m[8] * q[2];
                o.t[1] = m[1] * q[0] + m[5] * q[1] + m[9] * q[2];
                o.t[2] = -(m[2] * q[0] + m[6] * q[1] + m[10] * q[2]);
                normalize3(o.t);
                o.t[3] = (q[3] < 0 ? -1.0f : 1.0f) * -ds; // two mirrors (Z, and the node's when det < 0) each flip it
            }
            if (uv0)
                cgltf_accessor_read_float(uv0, v, o.uv, 2);
            o.color = 0xFFFFFFFFu;
            if (col)
            {
                float rgba[4] = { 1, 1, 1, 1 };
                cgltf_accessor_read_float(col, v, rgba, col->type == cgltf_type_vec3 ? 3 : 4);
                o.color = 0;
                for (int ch = 0; ch < 4; ++ch)
                {
                    float x = rgba[ch] < 0 ? 0 : (rgba[ch] > 1 ? 1 : rgba[ch]);
                    o.color |= (uint32_t)(x * 255.0f + 0.5f) << (8 * ch);
                }
            }
            part->corners.push_back(o);
        }
    }
    if (!nor) // flat: each triangle's own normal
    {
        flatNormals = true;
        for (size_t t = first; t + 2 < part->corners.size(); t += 3)
        {
            const float* a = part->corners[t].p;
            const float* b = part->corners[t + 1].p;
            const float* d = part->corners[t + 2].p;
            float e1[3] = { b[0] - a[0], b[1] - a[1], b[2] - a[2] }, e2[3] = { d[0] - a[0], d[1] - a[1], d[2] - a[2] };
            float nn[3] = { e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0] };
            normalize3(nn);
            for (int k = 0; k < 3; ++k)
                memcpy(part->corners[t + k].n, nn, 12);
        }
    }
    const size_t count = part->corners.size() - first;
    if (!tan && count)
    {
        if (uv0)
        {
            madeTangents = true;
            std::vector<float> out(count * 4);
            meshopt_generateTangents(out.data(), nullptr, count, part->corners[first].p, count, sizeof(Vertex), part->corners[first].n,
                                     sizeof(Vertex), part->corners[first].uv, sizeof(Vertex), meshopt_TangentCompatible);
            for (size_t i = 0; i < count; ++i)
                memcpy(part->corners[first + i].t, &out[i * 4], 16);
        }
        else // no UVs to follow: any tangent across the normal, so nothing reads garbage
            for (size_t i = first; i < part->corners.size(); ++i)
            {
                Vertex& o = part->corners[i];
                float   ax[3] = { fabsf(o.n[0]) < 0.9f ? 1.0f : 0.0f, fabsf(o.n[0]) < 0.9f ? 0.0f : 1.0f, 0 };
                float   d = ax[0] * o.n[0] + ax[1] * o.n[1];
                o.t[0] = ax[0] - o.n[0] * d, o.t[1] = ax[1] - o.n[1] * d, o.t[2] = -o.n[2] * d, o.t[3] = 1;
                normalize3(o.t);
            }
    }
}

void visit(const cgltf_data* data, const cgltf_node* node, std::vector<Part>& parts, bool& flatNormals, bool& madeTangents,
           uint32_t& meshNodes)
{
    if (node->mesh)
    {
        float m[16];
        cgltf_node_transform_world(node, m);
        ++meshNodes;
        for (cgltf_size i = 0; i < node->mesh->primitives_count; ++i)
            addPrimitive(data, node->mesh->primitives[i], m, parts, flatNormals, madeTangents);
    }
    if (node->skin)
        warn("skinning is ignored: the model is cooked in its bind pose (skinned models come with animation)");
    for (cgltf_size i = 0; i < node->children_count; ++i)
        visit(data, node->children[i], parts, flatNormals, madeTangents, meshNodes);
}

// Each part's corners welded into vertices and indices, ordered for the GPU.
void weld(Part& p)
{
    const size_t          n = p.corners.size();
    std::vector<uint32_t> remap(n);
    size_t                unique = meshopt_generateVertexRemap(remap.data(), nullptr, n, p.corners.data(), n, sizeof(Vertex));
    p.vertices.resize(unique);
    p.indices.resize(n);
    meshopt_remapVertexBuffer(p.vertices.data(), p.corners.data(), n, sizeof(Vertex), remap.data());
    meshopt_remapIndexBuffer(p.indices.data(), nullptr, n, remap.data());
    meshopt_optimizeVertexCache(p.indices.data(), p.indices.data(), n, unique);
    meshopt_optimizeVertexFetch(p.vertices.data(), p.indices.data(), n, p.vertices.data(), unique, sizeof(Vertex));
    p.corners.clear();
    p.corners.shrink_to_fit();
    for (const Vertex& v : p.vertices)
        for (int k = 0; k < 3; ++k)
            p.lo[k] = v.p[k] < p.lo[k] ? v.p[k] : p.lo[k], p.hi[k] = v.p[k] > p.hi[k] ? v.p[k] : p.hi[k];
}

uint64_t positionKey(const float* p)
{
    uint32_t b[3];
    memcpy(b, p, 12);
    uint64_t h = 1469598103934665603ull;
    for (uint32_t x : b)
        h = (h ^ x) * 1099511628211ull;
    return h;
}

// One level: its indices into the part's vertices, compacted into a range of its own.
Part::Level compact(const Part& p, const uint32_t* indices, size_t count, float error)
{
    Part::Level           l;
    std::vector<uint32_t> idx(indices, indices + count);
    meshopt_optimizeVertexCache(idx.data(), idx.data(), count, p.vertices.size());
    std::vector<Vertex> verts(p.vertices.size());
    size_t              used = meshopt_optimizeVertexFetch(verts.data(), idx.data(), count, p.vertices.data(), p.vertices.size(), sizeof(Vertex));
    l.indices = idx;
    l.error = error;
    for (size_t i = 0; i < used; ++i)
    {
        const Vertex& v = verts[i];
        l.positions.insert(l.positions.end(), v.p, v.p + 3);
        int      sign = v.t[3] < 0 ? -1 : 1;
        uint32_t uv[2];
        memcpy(uv, v.uv, 8);
        l.attributes.push_back(octPack(v.n[0], v.n[1], v.n[2], 0));
        l.attributes.push_back(octPack(v.t[0], v.t[1], v.t[2], sign));
        l.attributes.push_back(uv[0]);
        l.attributes.push_back(uv[1]);
        l.attributes.push_back(v.color);
    }
    return l;
}

void buildLods(Part& p, const std::unordered_map<uint64_t, uint32_t>& owners)
{
    const size_t n = p.vertices.size();
    // Material borders: a vertex whose position another part also has stays where it is.
    std::vector<unsigned char> lock(n, 0);
    for (size_t i = 0; i < n; ++i)
    {
        auto it = owners.find(positionKey(p.vertices[i].p));
        if (it != owners.end() && it->second == 0xFFFFFFFFu)
            lock[i] = meshopt_SimplifyVertex_Lock;
    }
    // Normals and UVs weigh in the error, so shading and texturing hold.
    std::vector<float> attrs(n * 5);
    for (size_t i = 0; i < n; ++i)
    {
        memcpy(&attrs[i * 5], p.vertices[i].n, 12);
        memcpy(&attrs[i * 5 + 3], p.vertices[i].uv, 8);
    }
    const float weights[5] = { 0.5f, 0.5f, 0.5f, 0.5f, 0.5f };
    const float scale = meshopt_simplifyScale(p.vertices[0].p, n, sizeof(Vertex)); // relative error to metres

    p.levels.push_back(compact(p, p.indices.data(), p.indices.size(), 0));
    size_t                prev = p.indices.size();
    float                 prevError = 0;
    std::vector<uint32_t> out(p.indices.size());
    for (int level = 1; level < 8 && prev / 3 >= 64; ++level)
    {
        float  relative = 0;
        size_t target = (prev / 6) * 3;
        size_t got = meshopt_simplifyWithAttributes(out.data(), p.indices.data(), p.indices.size(), p.vertices[0].p, n, sizeof(Vertex),
                                                    attrs.data(), 5 * sizeof(float), weights, 5, lock.data(), target, 0.25f, 0,
                                                    &relative);
        if (got == 0 || got > prev * 9 / 10)
            break;
        float error = relative * scale;
        error = error > prevError ? error : prevError;
        p.levels.push_back(compact(p, out.data(), got, error));
        prev = got;
        prevError = error;
        if (relative >= 0.25f * 0.999f)
            break;
    }
}

std::string gDir; // the source file's folder, for images beside a .gltf

float srgbToLinear(float c) { return c <= 0.04045f ? c / 12.92f : powf((c + 0.055f) / 1.055f, 2.4f); }

// The mean of a texture's channels over its whole image (rgb sRGB-decoded when srgb), or false when it can't be read.
bool textureMean(const cgltf_texture* tex, bool srgb, float out[4])
{
    const cgltf_image* img = tex ? tex->image : nullptr;
    if (!img)
        return false;
    int      w = 0, h = 0, n = 0;
    uint8_t* px = nullptr;
    if (img->buffer_view)
        px = stbi_load_from_memory(cgltf_buffer_view_data(img->buffer_view), (int)img->buffer_view->size, &w, &h, &n, 4);
    else if (img->uri && strncmp(img->uri, "data:", 5))
    {
        std::string path = gDir + img->uri;
        cgltf_decode_uri(&path[gDir.size()]);
        path.resize(strlen(path.c_str()));
        px = stbi_load(path.c_str(), &w, &h, &n, 4);
    }
    if (!px)
    {
        warn("an image (%s) couldn't be read: its material's factors stand in", img->uri ? img->uri : img->name ? img->name : "embedded");
        return false;
    }
    float lut[256];
    for (int i = 0; i < 256; ++i)
        lut[i] = srgb ? srgbToLinear(i / 255.0f) : i / 255.0f;
    double sum[4] = {};
    for (size_t i = 0, count = (size_t)w * h; i < count; ++i)
        for (int c = 0; c < 4; ++c)
            sum[c] += c < 3 ? lut[px[i * 4 + c]] : px[i * 4 + c] / 255.0f;
    for (int c = 0; c < 4; ++c)
        out[c] = (float)(sum[c] / ((double)w * h));
    stbi_image_free(px);
    return true;
}

} // namespace
#include "cook_textures.h"
namespace
{
size_t align16(size_t x) { return (x + 15) & ~(size_t)15; }
} // namespace

int main(int argc, char** argv)
{
    if (argc == 5 && !strcmp(argv[3], "--max-size"))
        cook::gMaxSize = (uint32_t)atoi(argv[4]);
    if (argc != 3 && !(argc == 5 && cook::gMaxSize >= 4))
    {
        fprintf(stderr, "usage: model_cook <in.gltf|in.glb> <out.rmodel> [--max-size <pixels, at least 4>]\n");
        return 2;
    }
    gDir = argv[1];
    gDir.resize(gDir.find_last_of("/\\") == std::string::npos ? 0 : gDir.find_last_of("/\\") + 1);
    cgltf_options options = {};
    cgltf_data*   data = nullptr;
    if (cgltf_parse_file(&options, argv[1], &data) != cgltf_result_success)
    {
        fprintf(stderr, "error: %s isn't a glTF or GLB file cgltf can read\n", argv[1]);
        return 1;
    }
    if (cgltf_load_buffers(&options, data, argv[1]) != cgltf_result_success || cgltf_validate(data) != cgltf_result_success)
    {
        fprintf(stderr, "error: %s: its buffers are missing or it isn't valid glTF\n", argv[1]);
        cgltf_free(data);
        return 1;
    }
    for (cgltf_size i = 0; i < data->extensions_required_count; ++i)
        if (strcmp(data->extensions_required[i], "EXT_meshopt_compression") && strcmp(data->extensions_required[i], "KHR_mesh_quantization"))
            warn("it requires %s, which the cooker doesn't read yet", data->extensions_required[i]);
    if (!decodeMeshopt(data))
    {
        fprintf(stderr, "error: its EXT_meshopt_compression data didn't decode\n");
        cgltf_free(data);
        return 1;
    }

    std::vector<Part> parts;
    bool              flatNormals = false, madeTangents = false;
    uint32_t          meshNodes = 0;
    const cgltf_scene* scene = data->scene ? data->scene : (data->scenes_count ? &data->scenes[0] : nullptr);
    if (scene)
        for (cgltf_size i = 0; i < scene->nodes_count; ++i)
            visit(data, scene->nodes[i], parts, flatNormals, madeTangents, meshNodes);
    else
        for (cgltf_size i = 0; i < data->nodes_count; ++i)
            if (!data->nodes[i].parent)
                visit(data, &data->nodes[i], parts, flatNormals, madeTangents, meshNodes);
    for (size_t i = parts.size(); i-- > 0;)
        if (parts[i].corners.empty())
            parts.erase(parts.begin() + i);
    if (parts.empty())
    {
        fprintf(stderr, "error: %s has no triangles in its scene\n", argv[1]);
        cgltf_free(data);
        return 1;
    }
    if (parts.size() > 64)
    {
        fprintf(stderr, "error: %zu materials; a model may have 64 (merge materials in the source)\n", parts.size());
        cgltf_free(data);
        return 1;
    }
    for (Part& p : parts)
        weld(p);
    std::unordered_map<uint64_t, uint32_t> owners; // a position's part, or 0xFFFFFFFF when shared by two
    for (uint32_t k = 0; k < parts.size(); ++k)
        for (const Vertex& v : parts[k].vertices)
        {
            auto r = owners.emplace(positionKey(v.p), k);
            if (!r.second && r.first->second != k)
                r.first->second = 0xFFFFFFFFu;
        }
    for (Part& p : parts)
        buildLods(p, owners);

    // Materials, in the parts' order.
    std::vector<RmodelMaterial> mats;
    uint32_t                    textured = 0;
    for (const Part& p : parts)
    {
        RmodelMaterial o = {};
        o.baseColor[0] = o.baseColor[1] = o.baseColor[2] = o.baseColor[3] = 1;
        o.roughness = o.metallic = 1;
        o.emissiveStrength = 1;
        for (int t = 0; t < 4; ++t)
            o.textures[t] = -1;
        if (p.material < data->materials_count)
        {
            const cgltf_material& m = data->materials[p.material];
            snprintf(o.name, sizeof o.name, "%s", m.name ? m.name : "");
            if (m.has_pbr_metallic_roughness)
            {
                memcpy(o.baseColor, m.pbr_metallic_roughness.base_color_factor, 16);
                o.roughness = m.pbr_metallic_roughness.roughness_factor;
                o.metallic = m.pbr_metallic_roughness.metallic_factor;
                float mean[4];
                memcpy(o.flatColor, o.baseColor, 12);
                o.flatRoughness = o.roughness, o.flatMetallic = o.metallic;
                if (textureMean(m.pbr_metallic_roughness.base_color_texture.texture, true, mean))
                    for (int c = 0; c < 3; ++c)
                        o.flatColor[c] *= mean[c];
                if (textureMean(m.pbr_metallic_roughness.metallic_roughness_texture.texture, false, mean))
                    o.flatRoughness *= mean[1], o.flatMetallic *= mean[2]; // glTF: roughness in green, metallic in blue
            }
            else if (m.has_pbr_specular_glossiness)
                warn("material %s uses specular-glossiness: cooked as a plain metallic-roughness material", o.name);
            memcpy(o.emissive, m.emissive_factor, 12);
            if (m.has_emissive_strength)
                o.emissiveStrength = m.emissive_strength.emissive_strength;
            o.alpha = m.alpha_mode == cgltf_alpha_mode_mask ? RmodelMask : (m.alpha_mode == cgltf_alpha_mode_blend ? RmodelBlend : RmodelOpaque);
            o.alphaCutoff = m.alpha_cutoff;
            o.doubleSided = m.double_sided;
            if (o.alpha == RmodelBlend)
                warn("material %s is alpha blend: drawn as a mask at its cutoff (0.5) until transparency comes", o.name);
            o.normalScale = m.normal_texture.texture ? m.normal_texture.scale : 1;
            o.occlusionStrength = m.occlusion_texture.texture ? m.occlusion_texture.scale : 1;
            const float cutoff = o.alpha == RmodelOpaque ? 0 : (o.alpha == RmodelBlend ? 0.5f : o.alphaCutoff);
            const cgltf_texture_view views[5] = { m.pbr_metallic_roughness.base_color_texture, m.pbr_metallic_roughness.metallic_roughness_texture,
                                                  m.occlusion_texture, m.normal_texture, m.emissive_texture };
            for (const cgltf_texture_view& v : views)
            {
                if (v.texture && v.texcoord != 0)
                    warn("material %s reads a texture with UV set %d: the cooker keeps only the first set, so it reads that", o.name, v.texcoord);
                if (v.texture && v.has_transform)
                    warn("material %s moves a texture (KHR_texture_transform): ignored", o.name);
            }
            if (m.has_pbr_metallic_roughness)
            {
                o.textures[0] = cook::cook(m.pbr_metallic_roughness.base_color_texture.texture, nullptr, cook::KindBase, cutoff);
                o.textures[1] = cook::cook(m.pbr_metallic_roughness.metallic_roughness_texture.texture, m.occlusion_texture.texture,
                                           cook::KindOrm, 0);
            }
            o.textures[2] = cook::cook(m.normal_texture.texture, nullptr, cook::KindNormal, 0);
            o.textures[3] = cook::cook(m.emissive_texture.texture, nullptr, cook::KindEmissive, 0);
            for (int t = 0; t < 4; ++t)
                textured += o.textures[t] >= 0;
        }
        else
            {
            snprintf(o.name, sizeof o.name, "(default)");
            o.normalScale = o.occlusionStrength = 1;
        }
        if (p.material >= data->materials_count || !data->materials[p.material].has_pbr_metallic_roughness)
            memcpy(o.flatColor, o.baseColor, 12), o.flatRoughness = o.roughness, o.flatMetallic = o.metallic;
        mats.push_back(o);
    }
    if (data->animations_count)
        warn("%zu animation%s ignored (animation is a later stage)", (size_t)data->animations_count, data->animations_count == 1 ? " is" : "s are");

    // Layout.
    RmodelHeader h = {};
    h.magic = kRmodelMagic;
    h.version = kRmodelVersion;
    h.headerBytes = sizeof h;
    h.materialCount = (uint32_t)mats.size();
    h.partCount = (uint32_t)parts.size();
    std::vector<RmodelPart> fparts;
    std::vector<RmodelLod>  flods;
    for (int k = 0; k < 3; ++k)
        h.lo[k] = 1e30f, h.hi[k] = -1e30f;
    for (uint32_t i = 0; i < parts.size(); ++i)
    {
        const Part& p = parts[i];
        RmodelPart  fp = {};
        fp.material = i;
        fp.firstLod = (uint32_t)flods.size();
        fp.lodCount = (uint32_t)p.levels.size();
        memcpy(fp.lo, p.lo, 12);
        memcpy(fp.hi, p.hi, 12);
        for (int k = 0; k < 3; ++k)
            h.lo[k] = p.lo[k] < h.lo[k] ? p.lo[k] : h.lo[k], h.hi[k] = p.hi[k] > h.hi[k] ? p.hi[k] : h.hi[k];
        fparts.push_back(fp);
        for (const Part::Level& l : p.levels)
        {
            RmodelLod fl = {};
            fl.firstVertex = h.vertexCount;
            fl.vertexCount = (uint32_t)(l.positions.size() / 3);
            fl.firstIndex = h.indexCount;
            fl.indexCount = (uint32_t)l.indices.size();
            fl.error = l.error;
            h.vertexCount += fl.vertexCount;
            h.indexCount += fl.indexCount;
            flods.push_back(fl);
        }
    }
    h.lodCount = (uint32_t)flods.size();
    size_t at = align16(sizeof h);
    h.materials = (uint32_t)at, at = align16(at + mats.size() * sizeof(RmodelMaterial));
    h.parts = (uint32_t)at, at = align16(at + fparts.size() * sizeof(RmodelPart));
    h.lods = (uint32_t)at, at = align16(at + flods.size() * sizeof(RmodelLod));
    h.positions = (uint32_t)at, at = align16(at + (size_t)h.vertexCount * 12);
    h.attributes = (uint32_t)at, at = align16(at + (size_t)h.vertexCount * 20);
    h.indices = (uint32_t)at, at = align16(at + (size_t)h.indexCount * 4);
    h.textureCount = (uint32_t)cook::gTextures.size();
    h.textures = (uint32_t)at, at = align16(at + cook::gTextures.size() * sizeof(RmodelTexture));
    for (cook::Cooked& c : cook::gTextures)
        c.t.offset = at, at = align16(at + c.data.size());
    if (at > 100u * 1024 * 1024)
    {
        fprintf(stderr, "error: the cooked model would be %.1f MB; Yea allows 100 MB a file\n", at / 1048576.0);
        cgltf_free(data);
        return 1;
    }
    h.fileBytes = (uint32_t)at;
    std::vector<uint8_t> file(at, 0);
    memcpy(&file[0], &h, sizeof h);
    memcpy(&file[h.materials], mats.data(), mats.size() * sizeof(RmodelMaterial));
    memcpy(&file[h.parts], fparts.data(), fparts.size() * sizeof(RmodelPart));
    memcpy(&file[h.lods], flods.data(), flods.size() * sizeof(RmodelLod));
    size_t li = 0;
    for (const Part& p : parts)
        for (const Part::Level& l : p.levels)
        {
            const RmodelLod& fl = flods[li++];
            memcpy(&file[h.positions + (size_t)fl.firstVertex * 12], l.positions.data(), l.positions.size() * 4);
            memcpy(&file[h.attributes + (size_t)fl.firstVertex * 20], l.attributes.data(), l.attributes.size() * 4);
            memcpy(&file[h.indices + (size_t)fl.firstIndex * 4], l.indices.data(), l.indices.size() * 4);
        }
    for (size_t i = 0; i < cook::gTextures.size(); ++i)
    {
        memcpy(&file[h.textures + i * sizeof(RmodelTexture)], &cook::gTextures[i].t, sizeof(RmodelTexture));
        memcpy(&file[cook::gTextures[i].t.offset], cook::gTextures[i].data.data(), cook::gTextures[i].data.size());
    }
    FILE* f = fopen(argv[2], "wb");
    if (!f || fwrite(file.data(), 1, file.size(), f) != file.size())
    {
        fprintf(stderr, "error: couldn't write %s\n", argv[2]);
        if (f)
            fclose(f);
        cgltf_free(data);
        return 1;
    }
    fclose(f);

    // The report.
    const float size[3] = { h.hi[0] - h.lo[0], h.hi[1] - h.lo[1], h.hi[2] - h.lo[2] };
    const float largest = fmaxf(size[0], fmaxf(size[1], size[2]));
    note("units: metres, as glTF is; node scales applied. Size %.3g x %.3g x %.3g m (x, y, z)", size[0], size[1], size[2]);
    if (largest > 500.0f)
        warn("it is %.0f m across: if it should be smaller, the file's units are off (set its Scale knob)", largest);
    else if (largest < 0.01f)
        warn("it is %.1f mm across: if it should be larger, the file's units are off (set its Scale knob)", largest * 1000.0f);
    note("axes: +Y up kept; Z negated into the renderer's left-handed space (its front, glTF's +Z, now faces -Z), triangle order flipped to match");
    if (flatNormals)
        note("normals: some primitives had none, so they got flat normals (glTF's rule)");
    if (madeTangents)
        note("tangents: generated where missing (MikkTSpace-compatible)");
    printf("cooked %s -> %s (%.1f KB)\n", argv[1], argv[2], file.size() / 1024.0);
    printf("%u mesh node%s, %zu part%s (one per material):\n", meshNodes, meshNodes == 1 ? "" : "s", parts.size(), parts.size() == 1 ? "" : "s");
    for (uint32_t i = 0; i < parts.size(); ++i)
    {
        printf("  %s:", mats[i].name[0] ? mats[i].name : "(unnamed)");
        for (const Part::Level& l : parts[i].levels)
            printf(" %zu", l.indices.size() / 3);
        printf(" triangles; error");
        for (const Part::Level& l : parts[i].levels)
            printf(" %.2g", l.error * 1000.0f);
        printf(" mm; drawn until textures as colour %.2f %.2f %.2f, roughness %.2f, metallic %.2f\n", mats[i].flatColor[0],
               mats[i].flatColor[1], mats[i].flatColor[2], mats[i].flatRoughness, mats[i].flatMetallic);
    }
    if (!cook::gTextures.empty())
    {
        size_t bytes = 0;
        for (const cook::Cooked& c : cook::gTextures)
            bytes += c.data.size();
        printf("%zu texture%s (%u map%s used), %.1f MB with every mip:", cook::gTextures.size(), cook::gTextures.size() == 1 ? "" : "s",
               textured, textured == 1 ? "" : "s", bytes / 1048576.0);
        static const char* kinds[] = { "colour", "ORM", "normal", "emissive" };
        for (const cook::Cooked& c : cook::gTextures)
            printf(" %s %ux%u%s;", kinds[c.kind], c.t.width, c.t.height, c.b ? " (occlusion merged)" : "");
        printf("\n");
        if (cook::gSkipped)
            note("--max-size %u: %u top mip level%s left out over all textures (the source images are unchanged)", cook::gMaxSize, cook::gSkipped,
                 cook::gSkipped == 1 ? "" : "s");
    }
    for (const std::string& s : gNotes)
        printf("%s\n", s.c_str());
    for (const std::string& s : gWarnings)
        printf("warning: %s\n", s.c_str());
    cgltf_free(data);
    return 0;
}
