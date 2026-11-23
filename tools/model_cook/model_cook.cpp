// The Renderer's model cooker: a glTF/GLB, FBX or OBJ file in, a .rmodel (modules/renderer_models/rmodel.h) out, with a
// report. Agents run it in their orb when a model arrives; it is never shipped or compiled into a module.
//
//   build.sh                                   (once: builds build/tools/model_cook, fetching and building Draco)
//   build/tools/model_cook in.glb out.rmodel   (or .gltf, .fbx, .obj)
//
// What it does, in order:
// - glTF (cgltf): the default scene's mesh nodes, each primitive with its node's world transform (EXT_meshopt_compression
//   decoded; KHR_draco_mesh_compression by Google's Draco decoder). glTF is metres and +Y up (node scales applied).
// - FBX and OBJ (ufbx, vendor/ufbx, MIT or public domain): ufbx converts the file into glTF's space (right-handed, +Y up,
//   metres, the geometry scaled) and reads its materials (an OBJ's .mtl too) as metallic-roughness; faces are
//   triangulated, UVs flipped (their origin is the image's bottom left, glTF's its top left), vertex colours left out.
//   A value the file doesn't give takes the map alone, or Blender's default without one; opacity 0 without a map (old
//   exporters' OBJ d 0) is opaque; an opacity map that is the base colour's own image cuts out, other opacity blends.
// - Then every mesh alike: the renderer is left-handed, so Z is negated (the model's front, glTF's +Z, faces -Z: towards
//   a camera looking down +Z) and the triangles' order flipped, so they stay counter-clockwise around their outward normal.
// - Missing normals: flat (as glTF says). Missing tangents (always for FBX and OBJ): generated MikkTSpace-compatible
//   (meshoptimizer), so normal maps baked against MikkTSpace shade right. Given tangents keep their handedness.
// - Merges every mesh of one material into one part, welds equal vertices, orders for the vertex cache and fetch.
// - Builds each part's LOD chain with meshoptimizer's attribute-aware simplifier: normals and UVs weighted, so shading
//   and texturing hold; UV and normal seams and hard edges (split vertices) collapse only along themselves; vertices on
//   a border with another part (a material border) are locked, so parts never open a crack between them. Each level
//   aims at half the triangles of the one before and records its error in metres, which only grows down the chain;
//   the chain stops when the simplifier can't take a tenth off, the error passes a quarter of the part's size, or a
//   level has under 64 triangles.
// - Writes the file and prints the report: units and axes, parts, triangles and error per LOD, what it generated or
//   ignored. Skins and animations come with a later stage and are listed as ignored.
// - Texture maps (rmodel v2): each image used as base colour, ORM, normal or emissive is decoded (stb_image) and written
//   with its full mip chain, BC7 (sRGB for colour) or BC5 for normals (bc7e, Apache-2.0).
#define CGLTF_IMPLEMENTATION
#include "vendor/cgltf/cgltf.h"
#include "vendor/meshoptimizer/meshoptimizer.h"
#include "vendor/ufbx/ufbx.h"
#include <draco/compression/decode.h> // Google's Draco decoder: fetched and built by build.sh
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

#include <memory>
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
uint32_t                 gDraco; // primitives decoded from Draco

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

// A mesh from any source, before it goes into the parts: glTF's space (right-handed, +Y up, metres; ufbx converts FBX
// and OBJ into it), its world transform, counter-clockwise fronts.
struct SrcMesh
{
    float                 m[16];    // world transform, column-major
    uint32_t              material; // index into the source's materials (their count: the default material)
    std::vector<float>    pos, nor, tan, uv; // per vertex: 3, 3, 4 (w the bitangent's sign), 2; nor, tan, uv may be empty
    std::vector<uint32_t> col;      // rgba8 per vertex; empty: white
    std::vector<uint32_t> idx;      // a triangle list
};

// One mesh's triangles, in the renderer's space, appended to its material's part.
void addMesh(const SrcMesh& src, std::vector<Part>& parts, bool& flatNormals, bool& madeTangents)
{
    const float* m = src.m;
    const size_t vcount = src.pos.size() / 3;
    const bool   nor = src.nor.size() == vcount * 3 && vcount, tan = src.tan.size() == vcount * 4 && vcount;
    const bool   uv0 = src.uv.size() == vcount * 2 && vcount, col = src.col.size() == vcount && vcount;

    // The world transform (column-major), its normal matrix (inverse transpose, by cofactors over the determinant).
    const float a00 = m[0], a10 = m[1], a20 = m[2], a01 = m[4], a11 = m[5], a21 = m[6], a02 = m[8], a12 = m[9], a22 = m[10];
    const float c[3][3] = { { a11 * a22 - a12 * a21, a12 * a20 - a10 * a22, a10 * a21 - a11 * a20 },
                            { a02 * a21 - a01 * a22, a00 * a22 - a02 * a20, a01 * a20 - a00 * a21 },
                            { a01 * a12 - a02 * a11, a02 * a10 - a00 * a12, a00 * a11 - a01 * a10 } };
    const float det = a00 * c[0][0] + a01 * c[0][1] + a02 * c[0][2];
    const float ds = det < 0 ? -1.0f : 1.0f;
    // A positive determinant keeps counter-clockwise fronts; negating Z mirrors once more. Order flips when det > 0.
    const bool flip = det > 0;

    Part* part = nullptr;
    for (Part& p : parts)
        if (p.material == src.material)
            part = &p;
    if (!part)
    {
        parts.push_back(Part{});
        part = &parts.back();
        part->material = src.material;
    }
    const size_t first = part->corners.size();
    for (size_t t = 0; t + 2 < src.idx.size(); t += 3)
    {
        const uint32_t tri[3] = { src.idx[t], flip ? src.idx[t + 2] : src.idx[t + 1], flip ? src.idx[t + 1] : src.idx[t + 2] };
        if (tri[0] >= vcount || tri[1] >= vcount || tri[2] >= vcount)
        {
            warn("a mesh indexes past its vertices: those triangles were dropped");
            continue;
        }
        for (uint32_t k = 0; k < 3; ++k)
        {
            const uint32_t v = tri[k];
            Vertex         o = {};
            const float*   p = &src.pos[v * 3];
            o.p[0] = m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12];
            o.p[1] = m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13];
            o.p[2] = -(m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14]);
            if (nor)
            {
                const float* n = &src.nor[v * 3];
                for (int r = 0; r < 3; ++r)
                    o.n[r] = (c[r][0] * n[0] + c[r][1] * n[1] + c[r][2] * n[2]) * ds;
                o.n[2] = -o.n[2];
                normalize3(o.n);
            }
            if (tan)
            {
                const float* q = &src.tan[v * 4];
                o.t[0] = m[0] * q[0] + m[4] * q[1] + m[8] * q[2];
                o.t[1] = m[1] * q[0] + m[5] * q[1] + m[9] * q[2];
                o.t[2] = -(m[2] * q[0] + m[6] * q[1] + m[10] * q[2]);
                normalize3(o.t);
                o.t[3] = (q[3] < 0 ? -1.0f : 1.0f) * -ds; // two mirrors (Z, and the node's when det < 0) each flip it
            }
            if (uv0)
                o.uv[0] = src.uv[v * 2], o.uv[1] = src.uv[v * 2 + 1];
            o.color = col ? src.col[v] : 0xFFFFFFFFu;
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

uint32_t packColor(const float* rgba)
{
    uint32_t o = 0;
    for (int ch = 0; ch < 4; ++ch)
    {
        float x = rgba[ch] < 0 ? 0 : (rgba[ch] > 1 ? 1 : rgba[ch]);
        o |= (uint32_t)(x * 255.0f + 0.5f) << (8 * ch);
    }
    return o;
}

// A primitive's index list as triangles (strips and fans unrolled).
void triangleList(cgltf_primitive_type type, const std::vector<uint32_t>& raw, std::vector<uint32_t>& idx)
{
    const size_t n = raw.size();
    if (type == cgltf_primitive_type_triangles)
        idx.assign(raw.begin(), raw.end() - n % 3);
    else
        for (size_t i = 2; i < n; ++i)
        {
            uint32_t a = type == cgltf_primitive_type_triangle_fan ? raw[0] : raw[i - 2], b = raw[i - 1], c = raw[i];
            if (type == cgltf_primitive_type_triangle_strip && (i & 1))
                std::swap(a, b);
            idx.push_back(a), idx.push_back(b), idx.push_back(c);
        }
}

// KHR_draco_mesh_compression: the primitive's attributes and indices from its Draco data (Google's decoder, fetched and
// built by build.sh). cgltf leaves each attribute's Draco id as an index into the accessors.
bool decodeDraco(const cgltf_data* data, const cgltf_primitive& prim, SrcMesh& out)
{
    const cgltf_draco_mesh_compression& dc = prim.draco_mesh_compression;
    const uint8_t* bytes = dc.buffer_view ? (const uint8_t*)cgltf_buffer_view_data(dc.buffer_view) : nullptr;
    if (!bytes)
        return false;
    draco::DecoderBuffer buf;
    buf.Init((const char*)bytes, dc.buffer_view->size);
    draco::Decoder dec;
    auto           r = dec.DecodeMeshFromBuffer(&buf);
    if (!r.ok())
        return false;
    std::unique_ptr<draco::Mesh> mesh = std::move(r).value();
    const uint32_t               points = mesh->num_points();
    auto read = [&](cgltf_attribute_type type, int index, int comps, std::vector<float>& dst) {
        for (cgltf_size i = 0; i < dc.attributes_count; ++i)
            if (dc.attributes[i].type == type && dc.attributes[i].index == index)
            {
                const draco::PointAttribute* a = mesh->GetAttributeByUniqueId((uint32_t)(dc.attributes[i].data - data->accessors));
                if (!a)
                    return false;
                dst.assign((size_t)points * comps, 1.0f);
                for (uint32_t v = 0; v < points; ++v)
                    a->ConvertValue<float>(a->mapped_index(draco::PointIndex(v)), (int8_t)(comps < a->num_components() ? comps : a->num_components()),
                                           &dst[(size_t)v * comps]);
                return true;
            }
        return false;
    };
    if (!read(cgltf_attribute_type_position, 0, 3, out.pos))
        return false;
    read(cgltf_attribute_type_normal, 0, 3, out.nor);
    read(cgltf_attribute_type_tangent, 0, 4, out.tan);
    read(cgltf_attribute_type_texcoord, 0, 2, out.uv);
    std::vector<float> col;
    if (read(cgltf_attribute_type_color, 0, 4, col))
        for (uint32_t v = 0; v < points; ++v)
            out.col.push_back(packColor(&col[(size_t)v * 4]));
    std::vector<uint32_t> raw;
    raw.reserve((size_t)mesh->num_faces() * 3);
    for (draco::FaceIndex f(0); f < mesh->num_faces(); ++f)
        for (int k = 0; k < 3; ++k)
            raw.push_back(mesh->face(f)[k].value());
    triangleList(cgltf_primitive_type_triangles, raw, out.idx);
    return true;
}

const cgltf_accessor* attribute(const cgltf_primitive& p, cgltf_attribute_type type, int index)
{
    for (cgltf_size i = 0; i < p.attributes_count; ++i)
        if (p.attributes[i].type == type && p.attributes[i].index == index)
            return p.attributes[i].data;
    return nullptr;
}

// One glTF primitive as a SrcMesh (false: nothing to draw).
bool gltfPrimitive(const cgltf_data* data, const cgltf_primitive& prim, const float* world, SrcMesh& out)
{
    if (prim.type != cgltf_primitive_type_triangles && prim.type != cgltf_primitive_type_triangle_strip &&
        prim.type != cgltf_primitive_type_triangle_fan)
    {
        warn("points and lines are ignored (the renderer draws triangles)");
        return false;
    }
    memcpy(out.m, world, sizeof out.m);
    out.material = prim.material ? (uint32_t)(prim.material - data->materials) : (uint32_t)data->materials_count;
    if (attribute(prim, cgltf_attribute_type_joints, 0))
        warn("skinning is ignored: the model is cooked in its bind pose (skinned models come with animation)");
    if (prim.targets_count)
        warn("morph targets are ignored");
    if (attribute(prim, cgltf_attribute_type_texcoord, 1))
        warn("a second UV set is ignored");
    if (prim.has_draco_mesh_compression)
    {
        if (!decodeDraco(data, prim, out))
        {
            warn("a Draco-compressed primitive didn't decode: it was skipped");
            return false;
        }
        gDraco++;
        return true;
    }
    const cgltf_accessor* pos = attribute(prim, cgltf_attribute_type_position, 0);
    if (!pos || !pos->count)
        return false;
    auto read = [](const cgltf_accessor* a, int comps, std::vector<float>& dst) {
        if (!a)
            return false;
        dst.assign(a->count * comps, 1.0f);
        for (cgltf_size v = 0; v < a->count; ++v)
            cgltf_accessor_read_float(a, v, &dst[v * comps], comps);
        return true;
    };
    read(pos, 3, out.pos);
    const cgltf_accessor* nor = attribute(prim, cgltf_attribute_type_normal, 0);
    const cgltf_accessor* tan = attribute(prim, cgltf_attribute_type_tangent, 0);
    if (nor && nor->count == pos->count)
        read(nor, 3, out.nor);
    if (tan && tan->count == pos->count)
        read(tan, 4, out.tan);
    const cgltf_accessor* uv0 = attribute(prim, cgltf_attribute_type_texcoord, 0);
    if (uv0 && uv0->count == pos->count)
        read(uv0, 2, out.uv);
    const cgltf_accessor* col = attribute(prim, cgltf_attribute_type_color, 0);
    if (col && col->count == pos->count)
        for (cgltf_size v = 0; v < col->count; ++v)
        {
            float rgba[4] = { 1, 1, 1, 1 };
            cgltf_accessor_read_float(col, v, rgba, col->type == cgltf_type_vec3 ? 3 : 4);
            out.col.push_back(packColor(rgba));
        }
    const cgltf_size      n = prim.indices ? prim.indices->count : pos->count;
    std::vector<uint32_t> raw(n);
    for (cgltf_size i = 0; i < n; ++i)
        raw[i] = prim.indices ? (uint32_t)cgltf_accessor_read_index(prim.indices, i) : (uint32_t)i;
    triangleList(prim.type, raw, out.idx);
    return true;
}

void visit(const cgltf_data* data, const cgltf_node* node, std::vector<SrcMesh>& meshes, uint32_t& meshNodes)
{
    if (node->mesh)
    {
        float m[16];
        cgltf_node_transform_world(node, m);
        ++meshNodes;
        for (cgltf_size i = 0; i < node->mesh->primitives_count; ++i)
        {
            meshes.emplace_back();
            if (!gltfPrimitive(data, node->mesh->primitives[i], m, meshes.back()))
                meshes.pop_back();
        }
    }
    if (node->skin)
        warn("skinning is ignored: the model is cooked in its bind pose (skinned models come with animation)");
    for (cgltf_size i = 0; i < node->children_count; ++i)
        visit(data, node->children[i], meshes, meshNodes);
}

// FBX and OBJ (ufbx): every mesh node's faces, triangulated, one SrcMesh per node and material. ufbx has already turned
// the file into glTF's space (right-handed, +Y up, metres, geometry scaled). Their UVs start at the image's bottom left,
// glTF's at its top left: v flipped. Tangents are always generated (MikkTSpace-compatible, as the renderer's normal maps
// expect); vertex colours are left out (FBX files often keep other data in them).
void ufbxMeshes(const ufbx_scene* scene, std::vector<SrcMesh>& meshes, uint32_t& meshNodes, bool& droppedColors)
{
    for (size_t ni = 0; ni < scene->nodes.count; ++ni)
    {
        const ufbx_node* node = scene->nodes.data[ni];
        const ufbx_mesh* mesh = node->mesh;
        if (!mesh || !mesh->num_triangles)
            continue;
        ++meshNodes;
        if (mesh->skin_deformers.count)
            warn("skinning is ignored: the model is cooked in its bind pose (skinned models come with animation)");
        if (mesh->blend_deformers.count)
            warn("blend shapes (morph targets) are ignored");
        if (mesh->num_point_faces || mesh->num_line_faces)
            warn("points and lines are ignored (the renderer draws triangles)");
        if (mesh->uv_sets.count > 1)
            warn("a second UV set is ignored");
        droppedColors = droppedColors || mesh->vertex_color.exists;
        const ufbx_matrix& g = node->geometry_to_world;
        const float        world[16] = { (float)g.cols[0].x, (float)g.cols[0].y, (float)g.cols[0].z, 0, (float)g.cols[1].x, (float)g.cols[1].y,
                                         (float)g.cols[1].z, 0, (float)g.cols[2].x, (float)g.cols[2].y, (float)g.cols[2].z, 0,
                                         (float)g.cols[3].x, (float)g.cols[3].y, (float)g.cols[3].z, 1 };
        std::vector<uint32_t> tri(mesh->max_face_triangles * 3);
        const size_t          groups = mesh->materials.count ? mesh->materials.count : 1;
        for (size_t mi = 0; mi < groups; ++mi)
        {
            SrcMesh out;
            memcpy(out.m, world, sizeof out.m);
            const ufbx_material* mat = mi < mesh->materials.count ? mesh->materials.data[mi] : nullptr;
            out.material = mat ? mat->typed_id : (uint32_t)scene->materials.count;
            for (size_t f = 0; f < mesh->faces.count; ++f)
            {
                if ((mesh->face_material.count ? mesh->face_material.data[f] : 0) != mi || mesh->faces.data[f].num_indices < 3)
                    continue;
                const uint32_t n = ufbx_triangulate_face(tri.data(), tri.size(), mesh, mesh->faces.data[f]);
                for (uint32_t k = 0; k < n * 3; ++k)
                {
                    const uint32_t ix = tri[k];
                    const ufbx_vec3 p = ufbx_get_vertex_vec3(&mesh->vertex_position, ix);
                    out.pos.push_back((float)p.x), out.pos.push_back((float)p.y), out.pos.push_back((float)p.z);
                    if (mesh->vertex_normal.exists)
                    {
                        const ufbx_vec3 q = ufbx_get_vertex_vec3(&mesh->vertex_normal, ix);
                        out.nor.push_back((float)q.x), out.nor.push_back((float)q.y), out.nor.push_back((float)q.z);
                    }
                    if (mesh->vertex_uv.exists)
                    {
                        const ufbx_vec2 t = ufbx_get_vertex_vec2(&mesh->vertex_uv, ix);
                        out.uv.push_back((float)t.x), out.uv.push_back(1.0f - (float)t.y);
                    }
                    out.idx.push_back((uint32_t)out.idx.size());
                }
            }
            if (!out.idx.empty())
                meshes.push_back(std::move(out));
        }
    }
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

} // namespace
#include "cook_textures.h"
namespace
{
size_t align16(size_t x) { return (x + 15) & ~(size_t)15; }

// A material from any source: glTF's metallic-roughness model, its maps as images (ORM: occlusion, roughness and
// metallic each from an image's channel).
struct SrcMaterial
{
    std::string           name = "(default)";
    float                 baseColor[4] = { 1, 1, 1, 1 };
    float                 roughness = 1, metallic = 1;
    float                 emissive[3] = {};
    float                 emissiveStrength = 1;
    uint32_t              alpha = RmodelOpaque;
    float                 cutoff = 0.5f;
    bool                  doubleSided = false;
    float                 normalScale = 1, occlusionStrength = 1;
    const cook::ImageRef* base = nullptr;
    const cook::ImageRef* normal = nullptr;
    const cook::ImageRef* emissiveMap = nullptr;
    const cook::ImageRef* orm[3] = {};
    int                   ormCh[3] = {};
};

const cook::ImageRef* gltfImage(const cgltf_texture* tex)
{
    const cgltf_image* img = tex ? tex->image : nullptr;
    if (!img)
        return nullptr;
    cook::ImageRef r;
    r.label = img->uri ? img->uri : img->name ? img->name : "embedded";
    if (img->buffer_view)
        r.data = (const uint8_t*)cgltf_buffer_view_data(img->buffer_view), r.size = img->buffer_view->size;
    else if (img->uri && strncmp(img->uri, "data:", 5))
    {
        r.path = gDir + img->uri;
        cgltf_decode_uri(&r.path[gDir.size()]);
        r.path.resize(strlen(r.path.c_str()));
    }
    return cook::imageRef(img, r);
}

std::vector<SrcMaterial> gltfMaterials(const cgltf_data* data)
{
    std::vector<SrcMaterial> out(data->materials_count);
    for (cgltf_size i = 0; i < data->materials_count; ++i)
    {
        const cgltf_material& m = data->materials[i];
        SrcMaterial&          o = out[i];
        o.name = m.name ? m.name : "";
        if (m.has_pbr_metallic_roughness)
        {
            memcpy(o.baseColor, m.pbr_metallic_roughness.base_color_factor, 16);
            o.roughness = m.pbr_metallic_roughness.roughness_factor;
            o.metallic = m.pbr_metallic_roughness.metallic_factor;
            o.base = gltfImage(m.pbr_metallic_roughness.base_color_texture.texture);
            const cook::ImageRef* mr = gltfImage(m.pbr_metallic_roughness.metallic_roughness_texture.texture);
            o.orm[1] = o.orm[2] = mr, o.ormCh[1] = 1, o.ormCh[2] = 2; // glTF: roughness in green, metallic in blue
        }
        else if (m.has_pbr_specular_glossiness)
            warn("material %s uses specular-glossiness: cooked as a plain metallic-roughness material", o.name.c_str());
        o.orm[0] = gltfImage(m.occlusion_texture.texture); // red; none: white (glTF leaves the red of a lone metallic-roughness image unused)
        o.occlusionStrength = m.occlusion_texture.texture ? m.occlusion_texture.scale : 1;
        memcpy(o.emissive, m.emissive_factor, 12);
        if (m.has_emissive_strength)
            o.emissiveStrength = m.emissive_strength.emissive_strength;
        o.emissiveMap = gltfImage(m.emissive_texture.texture);
        o.normal = gltfImage(m.normal_texture.texture);
        o.normalScale = m.normal_texture.texture ? m.normal_texture.scale : 1;
        o.alpha = m.alpha_mode == cgltf_alpha_mode_mask ? RmodelMask : (m.alpha_mode == cgltf_alpha_mode_blend ? RmodelBlend : RmodelOpaque);
        o.cutoff = m.alpha_cutoff;
        o.doubleSided = m.double_sided;
        const cgltf_texture_view views[5] = { m.pbr_metallic_roughness.base_color_texture, m.pbr_metallic_roughness.metallic_roughness_texture,
                                              m.occlusion_texture, m.normal_texture, m.emissive_texture };
        for (const cgltf_texture_view& v : views)
        {
            if (v.texture && v.texcoord != 0)
                warn("material %s reads a texture with UV set %d: the cooker keeps only the first set, so it reads that", o.name.c_str(), v.texcoord);
            if (v.texture && v.has_transform)
                warn("material %s moves a texture (KHR_texture_transform): ignored", o.name.c_str());
        }
    }
    return out;
}

// An FBX or OBJ texture's image: embedded in the file, or a file found by its absolute path, its path relative to the
// source, or its name beside the source (exported files often keep the author's own absolute paths).
const cook::ImageRef* ufbxImage(const ufbx_texture* t)
{
    if (t && t->type != UFBX_TEXTURE_FILE && t->file_textures.count)
        t = t->file_textures.data[0];
    if (!t || (t->type != UFBX_TEXTURE_FILE && !t->content.size))
        return nullptr;
    cook::ImageRef r;
    r.label = t->relative_filename.length ? t->relative_filename.data : t->filename.length ? t->filename.data : t->name.data;
    if (t->content.size)
    {
        r.data = (const uint8_t*)t->content.data, r.size = t->content.size;
        return cook::imageRef(t, r);
    }
    std::string name = t->filename.data;
    name = name.substr(name.find_last_of("/\\") == std::string::npos ? 0 : name.find_last_of("/\\") + 1);
    const std::string tries[3] = { t->absolute_filename.data, t->relative_filename.length ? gDir + t->relative_filename.data : "",
                                   name.empty() ? "" : gDir + name };
    for (const std::string& path : tries)
        if (!path.empty() && !r.path.size())
            if (FILE* f = fopen(path.c_str(), "rb"))
                fclose(f), r.path = path;
    if (r.path.empty())
        r.path = tries[0].empty() ? tries[2] : tries[0]; // load() says it couldn't be read
    for (const cook::ImageRef& i : cook::gImages) // one image per file, however many textures name it
        if (!i.data && i.path == r.path)
            return &i;
    return cook::imageRef(t, r);
}

std::vector<SrcMaterial> ufbxMaterials(const ufbx_scene* scene)
{
    std::vector<SrcMaterial> out(scene->materials.count);
    for (size_t i = 0; i < scene->materials.count; ++i)
    {
        const ufbx_material&         m = *scene->materials.data[i];
        const ufbx_material_pbr_maps& p = m.pbr;
        SrcMaterial&                 o = out[i];
        auto tex = [](const ufbx_material_map& map) { return map.texture && map.texture_enabled ? ufbxImage(map.texture) : nullptr; };
        o.name = m.name.data;
        // A value the file doesn't give reads 0 in ufbx: with a map, the map alone (factor 1); without, Blender's defaults.
        auto value = [](const ufbx_material_map& map, const cook::ImageRef* img, float none) {
            return map.has_value ? (float)map.value_real : img ? 1.0f : none;
        };
        o.base = tex(p.base_color);
        const float factor = p.base_factor.has_value ? (float)p.base_factor.value_real : 1.0f;
        for (int c = 0; c < 3; ++c)
            o.baseColor[c] = (p.base_color.has_value ? (float)p.base_color.value_vec4.v[c] : o.base ? 1.0f : 0.8f) * factor;
        o.roughness = value(p.roughness, tex(p.roughness), 0.5f);
        o.metallic = value(p.metalness, tex(p.metalness), 0.0f);
        if (m.features.roughness_as_glossiness.enabled && p.glossiness.texture)
            warn("material %s has a glossiness map: ignored (its roughness value stands)", o.name.c_str());
        else
            o.orm[1] = tex(p.roughness);
        o.orm[2] = tex(p.metalness);
        o.orm[0] = tex(p.ambient_occlusion);
        o.emissiveMap = tex(p.emission_color);
        const float ef = p.emission_factor.has_value ? (float)p.emission_factor.value_real : 1.0f;
        for (int c = 0; c < 3; ++c)
            o.emissive[c] = (p.emission_color.has_value ? (float)p.emission_color.value_vec3.v[c] : o.emissiveMap ? 1.0f : 0.0f) * ef;
        o.normal = tex(p.normal_map);
        o.doubleSided = m.features.double_sided.enabled;
        // FBX and OBJ don't say mask or blend: an opacity map that is the base colour's own image cuts out (leaves and
        // fences keep their alpha there); otherwise less than full opacity blends.
        const cook::ImageRef* opacity = tex(p.opacity);
        float                 alpha = p.opacity.has_value ? (float)p.opacity.value_real : 1.0f;
        if (alpha <= 0.0f && !opacity)
        {
            warn("material %s says it is fully transparent: cooked opaque (old exporters write OBJ's d 0 by mistake)", o.name.c_str());
            alpha = 1.0f;
        }
        if (opacity && opacity == o.base)
            o.alpha = RmodelMask, o.cutoff = 0.5f;
        else if (opacity || alpha < 0.999f)
        {
            o.alpha = RmodelBlend;
            o.baseColor[3] = alpha < 0 ? 0 : alpha;
            if (opacity)
                warn("material %s has an opacity map of its own: ignored (the base colour's alpha stands)", o.name.c_str());
        }
    }
    return out;
}
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
    std::string ext = argv[1];
    ext = ext.substr(ext.find_last_of('.') == std::string::npos ? ext.size() : ext.find_last_of('.') + 1);
    for (char& ch : ext)
        ch = (char)tolower((unsigned char)ch);
    cgltf_data* data = nullptr;
    ufbx_scene* scene = nullptr;
    auto        done = [&](int code) {
        if (data)
            cgltf_free(data);
        if (scene)
            ufbx_free_scene(scene);
        return code;
    };

    std::vector<SrcMesh>     meshes;
    std::vector<SrcMaterial> srcMats;
    uint32_t                 meshNodes = 0;
    bool                     droppedColors = false;
    const bool               isUfbx = ext == "fbx" || ext == "obj";
    if (isUfbx)
    {
        // Into glTF's space: right-handed, +Y up, metres, with the geometry itself scaled (so node scales stay 1).
        ufbx_load_opts opts = {};
        opts.target_axes = ufbx_axes_right_handed_y_up;
        opts.target_unit_meters = 1.0f;
        opts.space_conversion = UFBX_SPACE_CONVERSION_MODIFY_GEOMETRY;
        opts.obj_search_mtl_by_filename = true;
        opts.load_external_files = true; // an OBJ's .mtl
        ufbx_error error;
        scene = ufbx_load_file(argv[1], &opts, &error);
        if (!scene)
        {
            fprintf(stderr, "error: %s isn't an FBX or OBJ file ufbx can read: %s\n", argv[1], error.description.data);
            return 1;
        }
        ufbxMeshes(scene, meshes, meshNodes, droppedColors);
        srcMats = ufbxMaterials(scene);
        if (scene->anim_curves.count)
            warn("its animation is ignored (animation is a later stage)");
        const double unit = scene->settings.original_unit_meters;
        if (ext == "fbx" && unit > 0 && fabs(unit - 1.0) > 1e-6)
            note("units: the file's %g m a unit scaled to metres", unit);
        note("axes: the file's converted to +Y up, then Z negated into the renderer's left-handed space, triangle order flipped "
             "to match; UVs flipped to start at the image's top");
        if (droppedColors)
            note("vertex colours: left out (FBX files often hold other data in them)");
    }
    else if (ext == "gltf" || ext == "glb")
    {
        cgltf_options options = {};
        if (cgltf_parse_file(&options, argv[1], &data) != cgltf_result_success)
        {
            fprintf(stderr, "error: %s isn't a glTF or GLB file cgltf can read\n", argv[1]);
            return 1;
        }
        if (cgltf_load_buffers(&options, data, argv[1]) != cgltf_result_success || cgltf_validate(data) != cgltf_result_success)
        {
            fprintf(stderr, "error: %s: its buffers are missing or it isn't valid glTF\n", argv[1]);
            return done(1);
        }
        for (cgltf_size i = 0; i < data->extensions_required_count; ++i)
            if (strcmp(data->extensions_required[i], "EXT_meshopt_compression") && strcmp(data->extensions_required[i], "KHR_mesh_quantization") &&
                strcmp(data->extensions_required[i], "KHR_draco_mesh_compression"))
                warn("it requires %s, which the cooker doesn't read yet", data->extensions_required[i]);
        if (!decodeMeshopt(data))
        {
            fprintf(stderr, "error: its EXT_meshopt_compression data didn't decode\n");
            return done(1);
        }
        const cgltf_scene* sc = data->scene ? data->scene : (data->scenes_count ? &data->scenes[0] : nullptr);
        if (sc)
            for (cgltf_size i = 0; i < sc->nodes_count; ++i)
                visit(data, sc->nodes[i], meshes, meshNodes);
        else
            for (cgltf_size i = 0; i < data->nodes_count; ++i)
                if (!data->nodes[i].parent)
                    visit(data, &data->nodes[i], meshes, meshNodes);
        srcMats = gltfMaterials(data);
        if (data->animations_count)
            warn("%zu animation%s ignored (animation is a later stage)", (size_t)data->animations_count, data->animations_count == 1 ? " is" : "s are");
        note("units: metres, as glTF is; node scales applied");
        note("axes: +Y up kept; Z negated into the renderer's left-handed space (its front, glTF's +Z, now faces -Z), triangle order flipped to match");
        if (gDraco)
            note("Draco: %u primitive%s decoded", gDraco, gDraco == 1 ? "" : "s");
    }
    else
    {
        fprintf(stderr, "error: %s: the cooker reads .gltf, .glb, .fbx and .obj\n", argv[1]);
        return 2;
    }

    std::vector<Part> parts;
    bool              flatNormals = false, madeTangents = false;
    for (const SrcMesh& m : meshes)
        addMesh(m, parts, flatNormals, madeTangents);
    meshes.clear();
    for (size_t i = parts.size(); i-- > 0;)
        if (parts[i].corners.empty())
            parts.erase(parts.begin() + i);
    if (parts.empty())
    {
        fprintf(stderr, "error: %s has no triangles in its scene\n", argv[1]);
        return done(1);
    }
    if (parts.size() > 64)
    {
        fprintf(stderr, "error: %zu materials; a model may have 64 (merge materials in the source)\n", parts.size());
        return done(1);
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
        const SrcMaterial src = p.material < srcMats.size() ? srcMats[p.material] : SrcMaterial{};
        RmodelMaterial    o = {};
        snprintf(o.name, sizeof o.name, "%s", src.name.c_str());
        memcpy(o.baseColor, src.baseColor, 16);
        o.roughness = src.roughness, o.metallic = src.metallic;
        memcpy(o.emissive, src.emissive, 12);
        o.emissiveStrength = src.emissiveStrength;
        o.alpha = src.alpha;
        o.alphaCutoff = src.cutoff;
        o.doubleSided = src.doubleSided;
        o.normalScale = src.normalScale;
        o.occlusionStrength = src.occlusionStrength;
        if (o.alpha == RmodelBlend)
            warn("material %s is alpha blend: drawn opaque until the renderer has transparency", o.name);
        // The flat look (each map averaged) for a renderer without the textures.
        memcpy(o.flatColor, o.baseColor, 12);
        o.flatRoughness = o.roughness, o.flatMetallic = o.metallic;
        float mean[4];
        if (cook::mean(src.base, true, mean))
            for (int c = 0; c < 3; ++c)
                o.flatColor[c] *= mean[c];
        if (cook::mean(src.orm[1], false, mean))
            o.flatRoughness *= mean[src.ormCh[1]];
        if (cook::mean(src.orm[2], false, mean))
            o.flatMetallic *= mean[src.ormCh[2]];
        // Only a mask keeps its coverage in every mip; a blend's alpha is its opacity, as painted.
        o.textures[0] = cook::cook(src.base, cook::KindBase, o.alpha == RmodelMask ? o.alphaCutoff : 0);
        o.textures[1] = cook::cookOrm(src.orm[0], src.ormCh[0], src.orm[1], src.ormCh[1], src.orm[2], src.ormCh[2]);
        o.textures[2] = cook::cook(src.normal, cook::KindNormal, 0);
        o.textures[3] = cook::cook(src.emissiveMap, cook::KindEmissive, 0);
        for (int t = 0; t < 4; ++t)
            textured += o.textures[t] >= 0;
        mats.push_back(o);
    }

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
        return done(1);
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
        return done(1);
    }
    fclose(f);

    // The report.
    const float size[3] = { h.hi[0] - h.lo[0], h.hi[1] - h.lo[1], h.hi[2] - h.lo[2] };
    const float largest = fmaxf(size[0], fmaxf(size[1], size[2]));
    note("size: %.3g x %.3g x %.3g m (x, y, z)", size[0], size[1], size[2]);
    if (largest > 500.0f)
        warn("it is %.0f m across: if it should be smaller, the file's units are off (set its Scale knob)", largest);
    else if (largest < 0.01f)
        warn("it is %.1f mm across: if it should be larger, the file's units are off (set its Scale knob)", largest * 1000.0f);
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
            printf(" %s %ux%u%s;", kinds[c.kind], c.t.width, c.t.height,
                   cook::ormImages(c) > 1 ? " (merged from its maps)" : "");
        printf("\n");
        if (cook::gSkipped)
            note("--max-size %u: %u top mip level%s left out over all textures (the source images are unchanged)", cook::gMaxSize, cook::gSkipped,
                 cook::gSkipped == 1 ? "" : "s");
    }
    for (const std::string& s : gNotes)
        printf("%s\n", s.c_str());
    for (const std::string& s : gWarnings)
        printf("warning: %s\n", s.c_str());
    return done(0);
}
