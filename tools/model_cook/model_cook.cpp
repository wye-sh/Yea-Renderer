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
// - Skins and animations (rmodel v3): a model with a skin, or with a clip that moves a node holding a mesh, is cooked
//   skinned. Its vertices stay in their bind space (a glTF skinned mesh's own, an FBX mesh's geometry space) with up to
//   4 bones each (the strongest, renormalised); a mesh with no skin is bound whole to its own node, so it follows its
//   node's animation. Every node of the file goes in the skeleton with its rest pose; every clip (glTF animations, FBX
//   takes) is evaluated exactly from the file and encoded with the Renderer's own codec (cook_anim.h), at the lowest
//   rate that keeps every vertex within a millimetre. Parts' bounds hold every frame of every clip. Morph targets are
//   ignored.
// - Writes the file and prints the report: units and axes, parts, triangles and error per LOD, the skeleton and clips,
//   what it generated or ignored.
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
#include "cook_anim.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <map>
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
    uint16_t bones[4]; // skinned models only (all zero otherwise)
    uint16_t weights[4];
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
        std::vector<RmodelSkinVertex> skin; // skinned models
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
    std::vector<uint16_t> bones, weights; // 4 each per vertex (skeleton bones, unorm16), skinned models only
    std::vector<float>    dracoJoints, dracoWeights; // a Draco primitive's JOINTS_n/WEIGHTS_n, set after set (4 a vertex)
    std::vector<uint32_t> idx;      // a triangle list
};

// The skeleton and clips of a skinned model (rmodel v3), in the renderer's space.
struct Clip
{
    std::string                                   name;
    float                                         duration;
    std::function<void(float, anim::Pose*)>       source; // every node's exact local pose at a time
    bool                                          stepped = false; // it has step keys (jumps)
    float                                         keyRate = 0;     // the file's keys are evenly spaced at it (0: they aren't)
};
struct Skeleton
{
    bool                            on = false;
    std::vector<RmodelNode>         nodes;
    std::vector<RmodelBone>         bones;
    std::vector<Clip>               clips;
    std::vector<bool>               meshNode; // per node: it holds a mesh
    std::map<std::string, uint32_t> boneKeys;
    uint32_t bone(uint32_t node, const float* invBind)
    {
        RmodelBone b = {};
        memcpy(b.invBind, invBind, sizeof b.invBind);
        b.node = node;
        std::string key((const char*)&b, sizeof b);
        auto it = boneKeys.find(key);
        if (it != boneKeys.end())
            return it->second;
        bones.push_back(b);
        return boneKeys[key] = (uint32_t)bones.size() - 1;
    }
};
Skeleton gSkel;
const float kIdentity34[12] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 };

// Up to n influences to the strongest 4, renormalised to unorm16 that sum to exactly 65535 (false: they weigh nothing).
bool packInfluences(const uint32_t* bone, const float* weight, int n, uint16_t* ob, uint16_t* ow)
{
    uint32_t b[16];
    float    w[16];
    int      m = 0;
    for (int i = 0; i < n && i < 16; ++i)
    {
        if (!(weight[i] > 0))
            continue;
        int j = 0;
        while (j < m && b[j] != bone[i])
            ++j;
        if (j == m)
            b[m] = bone[i], w[m++] = 0;
        w[j] += weight[i];
    }
    for (int i = 1; i < m; ++i) // strongest first
        for (int j = i; j > 0 && w[j] > w[j - 1]; --j)
            std::swap(w[j], w[j - 1]), std::swap(b[j], b[j - 1]);
    m = m < 4 ? m : 4;
    float sum = 0;
    for (int i = 0; i < m; ++i)
        sum += w[i];
    if (!(sum > 0))
        return false;
    int total = 0;
    for (int i = 0; i < 4; ++i)
    {
        ob[i] = i < m ? (uint16_t)b[i] : 0;
        ow[i] = i < m ? (uint16_t)(w[i] / sum * 65535.0f + 0.5f) : 0;
        total += ow[i];
    }
    ow[0] = (uint16_t)(ow[0] + 65535 - total);
    return true;
}

// glTF's space to the renderer's (Z negated): a pose, and a column-major 4x4 to 3x4 rows.
void toRenderer(anim::Pose& p)
{
    p.t[2] = -p.t[2];
    p.r[0] = -p.r[0], p.r[1] = -p.r[1];
}
void rowsToRenderer(const float* colMajor, float* rows)
{
    static const float sgn[4] = { 1, 1, -1, 1 };
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c)
            rows[r * 4 + c] = colMajor[c * 4 + r] * sgn[r] * sgn[c];
}

// A column-major affine 4x4 as scale, rotation and translation (a negative determinant flips the x scale).
void decompose(const float* m, anim::Pose& p)
{
    p.t[0] = m[12], p.t[1] = m[13], p.t[2] = m[14];
    float c[3][3];
    for (int k = 0; k < 3; ++k)
    {
        p.s[k] = sqrtf(m[k * 4] * m[k * 4] + m[k * 4 + 1] * m[k * 4 + 1] + m[k * 4 + 2] * m[k * 4 + 2]);
        for (int r = 0; r < 3; ++r)
            c[k][r] = p.s[k] > 0 ? m[k * 4 + r] / p.s[k] : (r == k ? 1.0f : 0.0f);
    }
    const float det = c[0][0] * (c[1][1] * c[2][2] - c[2][1] * c[1][2]) - c[1][0] * (c[0][1] * c[2][2] - c[2][1] * c[0][2]) +
                      c[2][0] * (c[0][1] * c[1][2] - c[1][1] * c[0][2]);
    if (det < 0)
    {
        p.s[0] = -p.s[0];
        for (int r = 0; r < 3; ++r)
            c[0][r] = -c[0][r];
    }
    // R[row][col] = c[col][row]
    const float tr = c[0][0] + c[1][1] + c[2][2];
    float*      q = p.r;
    if (tr > 0)
    {
        float s = sqrtf(tr + 1.0f) * 2;
        q[3] = 0.25f * s, q[0] = (c[1][2] - c[2][1]) / s, q[1] = (c[2][0] - c[0][2]) / s, q[2] = (c[0][1] - c[1][0]) / s;
    }
    else if (c[0][0] > c[1][1] && c[0][0] > c[2][2])
    {
        float s = sqrtf(1.0f + c[0][0] - c[1][1] - c[2][2]) * 2;
        q[3] = (c[1][2] - c[2][1]) / s, q[0] = 0.25f * s, q[1] = (c[1][0] + c[0][1]) / s, q[2] = (c[2][0] + c[0][2]) / s;
    }
    else if (c[1][1] > c[2][2])
    {
        float s = sqrtf(1.0f + c[1][1] - c[0][0] - c[2][2]) * 2;
        q[3] = (c[2][0] - c[0][2]) / s, q[0] = (c[1][0] + c[0][1]) / s, q[1] = 0.25f * s, q[2] = (c[2][1] + c[1][2]) / s;
    }
    else
    {
        float s = sqrtf(1.0f + c[2][2] - c[0][0] - c[1][1]) * 2;
        q[3] = (c[0][1] - c[1][0]) / s, q[0] = (c[2][0] + c[0][2]) / s, q[1] = (c[2][1] + c[1][2]) / s, q[2] = 0.25f * s;
    }
}

void setNode(RmodelNode& n, int32_t parent, const anim::Pose& p, const char* name, uint32_t index)
{
    n = {};
    n.parent = parent;
    memcpy(n.t, p.t, 12), memcpy(n.r, p.r, 16), memcpy(n.s, p.s, 12);
    if (name && name[0])
        snprintf(n.name, sizeof n.name, "%s", name);
    else
        snprintf(n.name, sizeof n.name, "node%u", index);
}

// The rest pose composed through the hierarchy against the file's own world matrices (rows, renderer space): the
// worst difference, so a node whose transform the cooker read wrongly shows up in the report.
float checkHierarchy(const std::vector<float>& fileWorld)
{
    const uint32_t          n = (uint32_t)gSkel.nodes.size();
    std::vector<anim::Pose> rest(n);
    std::vector<float>      world(n * 12), pal;
    anim::restPoses(gSkel.nodes.data(), n, rest.data());
    anim::palette(gSkel.nodes.data(), n, nullptr, 0, rest.data(), world.data(), pal.data());
    float worst = 0;
    for (size_t i = 0; i < world.size(); ++i)
        worst = fmaxf(worst, fabsf(world[i] - fileWorld[i]) / (1.0f + fabsf(fileWorld[i])));
    return worst;
}

// Does any clip move a node that holds a mesh, or one above it? (A model without a skin is only cooked skinned then.)
bool clipsMoveMeshes()
{
    const uint32_t    n = (uint32_t)gSkel.nodes.size();
    std::vector<bool> relevant(n, false);
    for (uint32_t i = n; i-- > 0;) // children after parents: walk back, marking parents of relevant nodes
    {
        relevant[i] = relevant[i] || gSkel.meshNode[i];
        if (relevant[i] && gSkel.nodes[i].parent >= 0)
            relevant[gSkel.nodes[i].parent] = true;
    }
    std::vector<anim::Pose> a(n), b(n);
    for (const Clip& c : gSkel.clips)
    {
        c.source(0, a.data());
        for (float t = 1.0f / 30; t <= c.duration + 1e-4f; t += 1.0f / 30)
        {
            c.source(t, b.data());
            for (uint32_t i = 0; i < n; ++i)
                if (relevant[i] && memcmp(&a[i], &b[i], sizeof a[i]))
                    for (int k = 0; k < 3; ++k)
                        if (fabsf(a[i].t[k] - b[i].t[k]) > 1e-6f || fabsf(a[i].s[k] - b[i].s[k]) > 1e-6f ||
                            fabsf(a[i].r[0] * b[i].r[0] + a[i].r[1] * b[i].r[1] + a[i].r[2] * b[i].r[2] + a[i].r[3] * b[i].r[3]) < 1.0f - 1e-7f)
                            return true;
        }
    }
    return false;
}

// One mesh's triangles, in the renderer's space, appended to its material's part.
void addMesh(const SrcMesh& src, std::vector<Part>& parts, bool& flatNormals, bool& madeTangents)
{
    const float* m = src.m;
    const size_t vcount = src.pos.size() / 3;
    const bool   nor = src.nor.size() == vcount * 3 && vcount, tan = src.tan.size() == vcount * 4 && vcount;
    const bool   uv0 = src.uv.size() == vcount * 2 && vcount, col = src.col.size() == vcount && vcount;
    const bool   skinned = src.bones.size() == vcount * 4 && src.weights.size() == vcount * 4 && vcount;

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
            if (skinned)
                memcpy(o.bones, &src.bones[v * 4], 8), memcpy(o.weights, &src.weights[v * 4], 8);
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
    for (int set = 0; set < 2; ++set)
    {
        std::vector<float> j, w;
        if (read(cgltf_attribute_type_joints, set, 4, j) && read(cgltf_attribute_type_weights, set, 4, w))
        {
            out.dracoJoints.insert(out.dracoJoints.end(), j.begin(), j.end());
            out.dracoWeights.insert(out.dracoWeights.end(), w.begin(), w.end());
        }
    }
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
// Skinned models: `palette` maps the skin's joints to skeleton bones (null: bound whole to `rigidBone`).
bool gltfPrimitive(const cgltf_data* data, const cgltf_primitive& prim, const float* world, SrcMesh& out,
                   const std::vector<uint32_t>* palette = nullptr, uint32_t rigidBone = 0xFFFFFFFFu);
void gltfSkin(const cgltf_primitive& prim, SrcMesh& out, const std::vector<uint32_t>* palette, uint32_t rigidBone)
{
    if (rigidBone == 0xFFFFFFFFu && !palette)
        return;
    const size_t vcount = out.pos.size() / 3;
    std::vector<float> joints, weights; // 4 a vertex, per set
    int                sets = 0;
    if (palette && !out.dracoJoints.empty())
    {
        sets = (int)(out.dracoJoints.size() / (vcount * 4));
        joints = out.dracoJoints, weights = out.dracoWeights;
    }
    else if (palette)
        for (int set = 0; set < 2; ++set)
        {
            const cgltf_accessor* j = attribute(prim, cgltf_attribute_type_joints, set);
            const cgltf_accessor* w = attribute(prim, cgltf_attribute_type_weights, set);
            if (!j || !w || j->count != vcount || w->count != vcount)
                break;
            joints.resize((size_t)(sets + 1) * vcount * 4), weights.resize(joints.size());
            for (size_t v = 0; v < vcount; ++v)
            {
                cgltf_uint ji[4] = {};
                cgltf_accessor_read_uint(j, v, ji, 4);
                for (int k = 0; k < 4; ++k)
                    joints[(sets * vcount + v) * 4 + k] = (float)ji[k];
                cgltf_accessor_read_float(w, v, &weights[(sets * vcount + v) * 4], 4);
            }
            ++sets;
        }
    if (sets > 1)
        note("skin: up to %d influences a vertex in the file, the strongest 4 kept", sets * 4);
    out.bones.assign(vcount * 4, 0);
    out.weights.assign(vcount * 4, 0);
    bool unweighted = false, badJoint = false;
    for (size_t v = 0; v < vcount; ++v)
    {
        uint32_t b[8];
        float    w[8];
        int      n = 0;
        for (int set = 0; set < sets; ++set)
            for (int k = 0; k < 4; ++k)
            {
                const uint32_t j = (uint32_t)joints[(set * vcount + v) * 4 + k];
                const float    wt = weights[(set * vcount + v) * 4 + k];
                if (!(wt > 0))
                    continue;
                if (j >= palette->size())
                {
                    badJoint = true;
                    continue;
                }
                b[n] = (*palette)[j], w[n++] = wt;
            }
        if (!packInfluences(b, w, n, &out.bones[v * 4], &out.weights[v * 4]))
        {
            unweighted = unweighted || palette;
            const uint32_t rb = rigidBone != 0xFFFFFFFFu ? rigidBone : 0;
            const float    one = 1;
            packInfluences(&rb, &one, 1, &out.bones[v * 4], &out.weights[v * 4]);
        }
    }
    if (unweighted)
        warn("some skinned vertices weigh nothing: they follow their mesh's node");
    if (badJoint)
        warn("some skin influences name joints the skin doesn't have: left out");
}

bool gltfPrimitive(const cgltf_data* data, const cgltf_primitive& prim, const float* world, SrcMesh& out,
                   const std::vector<uint32_t>* palette, uint32_t rigidBone)
{
    if (prim.type != cgltf_primitive_type_triangles && prim.type != cgltf_primitive_type_triangle_strip &&
        prim.type != cgltf_primitive_type_triangle_fan)
    {
        warn("points and lines are ignored (the renderer draws triangles)");
        return false;
    }
    memcpy(out.m, world, sizeof out.m);
    out.material = prim.material ? (uint32_t)(prim.material - data->materials) : (uint32_t)data->materials_count;
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
        gltfSkin(prim, out, palette, rigidBone);
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
    gltfSkin(prim, out, palette, rigidBone);
    return true;
}

// Skinned glTF models: every node into the skeleton (parents first), with the map from cgltf's nodes to its.
std::vector<uint32_t> gNodeOf; // cgltf node index -> skeleton node
void gltfSkeleton(const cgltf_data* data)
{
    gNodeOf.assign(data->nodes_count, 0xFFFFFFFFu);
    std::vector<const cgltf_node*> order;
    std::vector<float>             fileWorld;
    std::function<void(const cgltf_node*)> add = [&](const cgltf_node* node) {
        const uint32_t index = (uint32_t)order.size();
        gNodeOf[node - data->nodes] = index;
        order.push_back(node);
        anim::Pose p;
        float      m[16];
        if (node->has_matrix)
            decompose(node->matrix, p);
        else
        {
            const float t0[3] = { 0, 0, 0 }, r0[4] = { 0, 0, 0, 1 }, s0[3] = { 1, 1, 1 };
            memcpy(p.t, node->has_translation ? node->translation : t0, 12);
            memcpy(p.r, node->has_rotation ? node->rotation : r0, 16);
            memcpy(p.s, node->has_scale ? node->scale : s0, 12);
        }
        toRenderer(p);
        gSkel.nodes.emplace_back();
        setNode(gSkel.nodes.back(), node->parent ? (int32_t)gNodeOf[node->parent - data->nodes] : -1, p, node->name, index);
        gSkel.meshNode.push_back(node->mesh != nullptr);
        cgltf_node_transform_world(node, m);
        fileWorld.resize(fileWorld.size() + 12);
        rowsToRenderer(m, &fileWorld[fileWorld.size() - 12]);
        for (cgltf_size i = 0; i < node->children_count; ++i)
            add(node->children[i]);
    };
    for (cgltf_size i = 0; i < data->nodes_count; ++i)
        if (!data->nodes[i].parent)
            add(&data->nodes[i]);
    const float worst = checkHierarchy(fileWorld);
    if (worst > 1e-3f)
        warn("the skeleton's rest pose strays %.3g from the file's node transforms: tell the Renderer's agents", worst);

    // Clips: each animation's translation, rotation and scale channels, evaluated as glTF says (step, linear with
    // spherical interpolation for rotations, cubic spline), held at their first and last keys outside them.
    struct Track
    {
        uint32_t           node;
        int                path, interp; // 0 t, 1 r, 2 s; 0 step, 1 linear, 2 cubic
        std::vector<float> in, out;
    };
    bool morphs = false;
    for (cgltf_size a = 0; a < data->animations_count; ++a)
    {
        const cgltf_animation& an = data->animations[a];
        auto                   tracks = std::make_shared<std::vector<Track>>();
        float                  duration = 0;
        for (cgltf_size c = 0; c < an.channels_count; ++c)
        {
            const cgltf_animation_channel& ch = an.channels[c];
            if (!ch.target_node || !ch.sampler)
                continue;
            if (ch.target_path == cgltf_animation_path_type_weights)
            {
                morphs = true;
                continue;
            }
            Track tr;
            tr.node = gNodeOf[ch.target_node - data->nodes];
            tr.path = ch.target_path == cgltf_animation_path_type_translation ? 0 : (ch.target_path == cgltf_animation_path_type_rotation ? 1 : 2);
            if (ch.target_path != cgltf_animation_path_type_translation && ch.target_path != cgltf_animation_path_type_rotation &&
                ch.target_path != cgltf_animation_path_type_scale)
                continue;
            tr.interp = ch.sampler->interpolation == cgltf_interpolation_type_step ? 0 : (ch.sampler->interpolation == cgltf_interpolation_type_linear ? 1 : 2);
            const int comps = tr.path == 1 ? 4 : 3;
            tr.in.resize(ch.sampler->input->count);
            for (cgltf_size k = 0; k < tr.in.size(); ++k)
                cgltf_accessor_read_float(ch.sampler->input, k, &tr.in[k], 1);
            tr.out.resize(ch.sampler->output->count * comps);
            for (cgltf_size k = 0; k < ch.sampler->output->count; ++k)
                cgltf_accessor_read_float(ch.sampler->output, k, &tr.out[k * comps], comps);
            if (tr.in.empty() || tr.out.size() < tr.in.size() * comps * (tr.interp == 2 ? 3 : 1))
                continue;
            duration = fmaxf(duration, tr.in.back());
            tracks->push_back(std::move(tr));
        }
        if (tracks->empty())
            continue;
        Clip clip;
        clip.name = an.name && an.name[0] ? an.name : "clip" + std::to_string(a);
        clip.duration = duration;
        float step = 0; // the smallest gap between keys, then whether every key falls on it
        for (const Track& tr : *tracks)
        {
            clip.stepped = clip.stepped || (tr.interp == 0 && tr.in.size() > 1);
            for (size_t k = 1; k < tr.in.size(); ++k)
                if (tr.in[k] - tr.in[k - 1] > 1e-4f && (step == 0 || tr.in[k] - tr.in[k - 1] < step))
                    step = tr.in[k] - tr.in[k - 1];
        }
        bool even = step > 0;
        for (const Track& tr : *tracks)
            for (size_t k = 0; k < tr.in.size() && even; ++k)
            {
                const float x = (tr.in[k] - tr.in[0]) / step;
                even = fabsf(x - roundf(x)) < 0.01f && fabsf(tr.in[0] - (*tracks)[0].in[0]) < 1e-4f;
            }
        clip.keyRate = even ? 1.0f / step : 0;
        clip.source = [tracks](float t, anim::Pose* poses) {
            anim::restPoses(gSkel.nodes.data(), (uint32_t)gSkel.nodes.size(), poses);
            for (const Track& tr : *tracks)
            {
                const int    comps = tr.path == 1 ? 4 : 3, stride = tr.interp == 2 ? comps * 3 : comps, at = tr.interp == 2 ? comps : 0;
                const size_t n = tr.in.size();
                float        v[4];
                auto         value = [&](size_t k) { return &tr.out[k * stride + at]; };
                if (t <= tr.in[0] || n == 1)
                    memcpy(v, value(0), comps * 4);
                else if (t >= tr.in[n - 1])
                    memcpy(v, value(n - 1), comps * 4);
                else
                {
                    const size_t k = (size_t)(std::upper_bound(tr.in.begin(), tr.in.end(), t) - tr.in.begin()) - 1;
                    const float  td = tr.in[k + 1] - tr.in[k], f = td > 0 ? (t - tr.in[k]) / td : 0;
                    const float *a = value(k), *b = value(k + 1);
                    if (tr.interp == 0)
                        memcpy(v, a, comps * 4);
                    else if (tr.interp == 2)
                    {
                        const float* bo = &tr.out[k * stride + 2 * comps]; // a's out-tangent
                        const float* ai = &tr.out[(k + 1) * stride];       // b's in-tangent
                        const float  f2 = f * f, f3 = f2 * f;
                        for (int i = 0; i < comps; ++i)
                            v[i] = (2 * f3 - 3 * f2 + 1) * a[i] + td * (f3 - 2 * f2 + f) * bo[i] + (-2 * f3 + 3 * f2) * b[i] + td * (f3 - f2) * ai[i];
                    }
                    else if (tr.path == 1) // slerp
                    {
                        float d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3], s = 1;
                        if (d < 0)
                            d = -d, s = -1;
                        float wa = 1 - f, wb = f;
                        if (d < 0.9995f)
                        {
                            const float th = acosf(d), sn = sinf(th);
                            wa = sinf((1 - f) * th) / sn, wb = sinf(f * th) / sn;
                        }
                        for (int i = 0; i < 4; ++i)
                            v[i] = a[i] * wa + b[i] * wb * s;
                    }
                    else
                        for (int i = 0; i < comps; ++i)
                            v[i] = a[i] + (b[i] - a[i]) * f;
                }
                anim::Pose& p = poses[tr.node];
                if (tr.path == 0)
                    p.t[0] = v[0], p.t[1] = v[1], p.t[2] = -v[2];
                else if (tr.path == 2)
                    memcpy(p.s, v, 12);
                else
                {
                    float len = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]);
                    len = len > 0 ? len : 1;
                    p.r[0] = -v[0] / len, p.r[1] = -v[1] / len, p.r[2] = v[2] / len, p.r[3] = v[3] / len;
                }
            }
        };
        gSkel.clips.push_back(std::move(clip));
    }
    if (morphs)
        warn("morph target animation is ignored");
}

void visit(const cgltf_data* data, const cgltf_node* node, std::vector<SrcMesh>& meshes, uint32_t& meshNodes)
{
    if (node->mesh)
    {
        float m[16];
        cgltf_node_transform_world(node, m);
        ++meshNodes;
        // Skinned models keep each mesh in its own space: a skinned mesh's bind space (glTF ignores its node's
        // transform), or its node's, bound whole to that node.
        std::vector<uint32_t> palette;
        uint32_t              rigid = 0xFFFFFFFFu;
        if (gSkel.on)
        {
            static const float id[16] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };
            memcpy(m, id, sizeof m);
            rigid = gSkel.bone(gNodeOf[node - data->nodes], kIdentity34);
            if (node->skin)
                for (cgltf_size j = 0; j < node->skin->joints_count; ++j)
                {
                    float ibm[16], rows[12];
                    memcpy(ibm, id, sizeof ibm);
                    if (node->skin->inverse_bind_matrices)
                        cgltf_accessor_read_float(node->skin->inverse_bind_matrices, j, ibm, 16);
                    rowsToRenderer(ibm, rows);
                    palette.push_back(gSkel.bone(gNodeOf[node->skin->joints[j] - data->nodes], rows));
                }
        }
        for (cgltf_size i = 0; i < node->mesh->primitives_count; ++i)
        {
            meshes.emplace_back();
            if (!gltfPrimitive(data, node->mesh->primitives[i], m, meshes.back(), node->skin && gSkel.on ? &palette : nullptr, rigid))
                meshes.pop_back();
        }
    }
    for (cgltf_size i = 0; i < node->children_count; ++i)
        visit(data, node->children[i], meshes, meshNodes);
}

// FBX and OBJ (ufbx): every mesh node's faces, triangulated, one SrcMesh per node and material. ufbx has already turned
// the file into glTF's space (right-handed, +Y up, metres, geometry scaled). Their UVs start at the image's bottom left,
// glTF's at its top left: v flipped. Tangents are always generated (MikkTSpace-compatible, as the renderer's normal maps
// expect); vertex colours are left out (FBX files often keep other data in them).
std::vector<uint32_t> gUfbxNodeOf; // ufbx node typed_id -> skeleton node

void ufbxRows(const ufbx_matrix& m, float* rows) // ufbx's matrix (glTF's space) as 3x4 rows in the renderer's
{
    float c[16] = {};
    for (int col = 0; col < 4; ++col)
        c[col * 4] = (float)m.cols[col].x, c[col * 4 + 1] = (float)m.cols[col].y, c[col * 4 + 2] = (float)m.cols[col].z;
    c[15] = 1;
    rowsToRenderer(c, rows);
}

anim::Pose ufbxPose(const ufbx_transform& t)
{
    anim::Pose p = { { (float)t.translation.x, (float)t.translation.y, (float)t.translation.z },
                     { (float)t.rotation.x, (float)t.rotation.y, (float)t.rotation.z, (float)t.rotation.w },
                     { (float)t.scale.x, (float)t.scale.y, (float)t.scale.z } };
    toRenderer(p);
    return p;
}

// Skinned FBX models: every node into the skeleton (parents first), and each take as a clip, evaluated by ufbx.
void ufbxSkeleton(const ufbx_scene* scene)
{
    gUfbxNodeOf.assign(scene->nodes.count, 0xFFFFFFFFu);
    std::vector<const ufbx_node*> order;
    std::vector<float>            fileWorld;
    std::function<void(const ufbx_node*)> add = [&](const ufbx_node* node) {
        const uint32_t index = (uint32_t)order.size();
        gUfbxNodeOf[node->typed_id] = index;
        order.push_back(node);
        gSkel.nodes.emplace_back();
        setNode(gSkel.nodes.back(), node->parent ? (int32_t)gUfbxNodeOf[node->parent->typed_id] : -1, ufbxPose(node->local_transform),
                node->name.data, index);
        gSkel.meshNode.push_back(node->mesh != nullptr);
        fileWorld.resize(fileWorld.size() + 12);
        ufbxRows(node->node_to_world, &fileWorld[fileWorld.size() - 12]);
        for (size_t i = 0; i < node->children.count; ++i)
            add(node->children.data[i]);
    };
    add(scene->root_node);
    const float worst = checkHierarchy(fileWorld);
    if (worst > 1e-3f)
        warn("the skeleton's rest pose strays %.3g from the file's node transforms: tell the Renderer's agents", worst);
    auto nodes = std::make_shared<std::vector<const ufbx_node*>>(order);
    for (size_t i = 0; i < scene->anim_stacks.count; ++i)
    {
        const ufbx_anim_stack* st = scene->anim_stacks.data[i];
        if (!(st->time_end - st->time_begin > 1e-6))
            continue; // an empty take (3ds Max's "(internal)")
        Clip                   clip;
        clip.keyRate = (float)scene->settings.frames_per_second; // FBX curves are mostly keyed on the file's frames
        clip.name = st->name.length ? st->name.data : "clip" + std::to_string(i);
        clip.duration = (float)fmax(0.0, st->time_end - st->time_begin);
        const ufbx_anim* an = st->anim;
        const double     begin = st->time_begin;
        clip.source = [nodes, an, begin](float t, anim::Pose* poses) {
            for (size_t n = 0; n < nodes->size(); ++n)
                poses[n] = ufbxPose(ufbx_evaluate_transform(an, (*nodes)[n], begin + t));
        };
        gSkel.clips.push_back(std::move(clip));
    }
}

void ufbxMeshes(const ufbx_scene* scene, std::vector<SrcMesh>& meshes, uint32_t& meshNodes, bool& droppedColors)
{
    for (size_t ni = 0; ni < scene->nodes.count; ++ni)
    {
        const ufbx_node* node = scene->nodes.data[ni];
        const ufbx_mesh* mesh = node->mesh;
        if (!mesh || !mesh->num_triangles)
            continue;
        ++meshNodes;
        if (mesh->skin_deformers.count > 1)
            warn("a mesh has more than one skin: only its first is used");
        const ufbx_skin_deformer* skin = gSkel.on && mesh->skin_deformers.count ? mesh->skin_deformers.data[0] : nullptr;
        std::vector<uint32_t>     palette;
        uint32_t                  rigid = 0;
        if (gSkel.on)
        {
            float rows[12];
            ufbxRows(node->geometry_to_node, rows);
            rigid = gSkel.bone(gUfbxNodeOf[node->typed_id], rows);
            if (skin)
                for (size_t c = 0; c < skin->clusters.count; ++c)
                {
                    const ufbx_skin_cluster* cl = skin->clusters.data[c];
                    ufbxRows(cl->geometry_to_bone, rows);
                    palette.push_back(cl->bone_node ? gSkel.bone(gUfbxNodeOf[cl->bone_node->typed_id], rows) : rigid);
                }
        }
        if (mesh->blend_deformers.count)
            warn("blend shapes (morph targets) are ignored");
        if (mesh->num_point_faces || mesh->num_line_faces)
            warn("points and lines are ignored (the renderer draws triangles)");
        if (mesh->uv_sets.count > 1)
            warn("a second UV set is ignored");
        droppedColors = droppedColors || mesh->vertex_color.exists;
        ufbx_matrix identity = {};
        identity.m00 = identity.m11 = identity.m22 = 1;
        const ufbx_matrix& g = gSkel.on ? identity : node->geometry_to_world; // skinned: geometry space
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
                    if (gSkel.on)
                    {
                        uint32_t b[16];
                        float    w[16];
                        int      n = 0;
                        if (skin)
                        {
                            const ufbx_skin_vertex& sv = skin->vertices.data[mesh->vertex_indices.data[ix]];
                            for (uint32_t i = 0; i < sv.num_weights && n < 16; ++i)
                            {
                                const ufbx_skin_weight& sw = skin->weights.data[sv.weight_begin + i];
                                b[n] = palette[sw.cluster_index], w[n++] = (float)sw.weight;
                            }
                        }
                        uint16_t ob[4], ow[4];
                        if (!packInfluences(b, w, n, ob, ow))
                        {
                            const float one = 1;
                            packInfluences(&rigid, &one, 1, ob, ow);
                        }
                        out.bones.insert(out.bones.end(), ob, ob + 4);
                        out.weights.insert(out.weights.end(), ow, ow + 4);
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
        if (gSkel.on)
        {
            RmodelSkinVertex sv;
            memcpy(sv.bones, v.bones, 8), memcpy(sv.weights, v.weights, 8);
            l.skin.push_back(sv);
        }
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
    // Skinned: where a triangle spans two strongest bones (an influence border), its vertices stay, so coarser levels
    // bend at the same joints as the finest; and the strongest bone's weight weighs in, so blended vertices don't
    // collapse onto rigid ones.
    if (gSkel.on)
        for (size_t t = 0; t + 2 < p.indices.size(); t += 3)
        {
            const uint32_t a = p.indices[t], b = p.indices[t + 1], c = p.indices[t + 2];
            const uint16_t ba = p.vertices[a].bones[0], bb = p.vertices[b].bones[0], bc = p.vertices[c].bones[0];
            if (ba != bb || bb != bc)
                lock[a] = lock[b] = lock[c] = meshopt_SimplifyVertex_Lock;
        }
    // Normals and UVs weigh in the error, so shading and texturing hold.
    const size_t       stride = gSkel.on ? 6 : 5;
    std::vector<float> attrs(n * stride);
    for (size_t i = 0; i < n; ++i)
    {
        memcpy(&attrs[i * stride], p.vertices[i].n, 12);
        memcpy(&attrs[i * stride + 3], p.vertices[i].uv, 8);
        if (gSkel.on)
            attrs[i * stride + 5] = p.vertices[i].weights[0] / 65535.0f;
    }
    const float weights[6] = { 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 1.0f };
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
                                                    attrs.data(), stride * sizeof(float), weights, stride, lock.data(), target, 0.25f, 0,
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
        if (scene->skin_deformers.count || scene->anim_curves.count)
        {
            ufbxSkeleton(scene);
            gSkel.on = scene->skin_deformers.count || clipsMoveMeshes();
        }
        ufbxMeshes(scene, meshes, meshNodes, droppedColors);
        srcMats = ufbxMaterials(scene);
        if (gSkel.on && !scene->anim_stacks.count)
            note("skinned, with no takes: it stands in its rest pose");
        if (gSkel.on && scene->anim_stacks.count)
            note("takes: the file's frame rate is %g a second; clips start at each take's start", scene->settings.frames_per_second);
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
        if (data->skins_count || data->animations_count)
        {
            gltfSkeleton(data);
            gSkel.on = data->skins_count || clipsMoveMeshes();
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
    if (!gSkel.on && !gSkel.clips.empty())
        note("its %zu clip%s move%s no mesh: cooked static", gSkel.clips.size(), gSkel.clips.size() == 1 ? "" : "s",
             gSkel.clips.size() == 1 ? "s" : "");
    if (gSkel.on && gSkel.bones.size() > 65535)
    {
        fprintf(stderr, "error: %zu bones; a model may have 65535\n", gSkel.bones.size());
        return done(1);
    }

    // Skinned: each part's bounds hold the rest pose and every frame of every clip (padded for the frames between),
    // and every clip encoded within a millimetre at the skin.
    std::vector<cook_anim::Encoded> clips;
    if (gSkel.on)
    {
        std::vector<cook_anim::SkinnedVertex> verts;
        std::vector<uint32_t>                 vpart;
        for (uint32_t k = 0; k < parts.size(); ++k)
        {
            const Part::Level& l = parts[k].levels[0];
            for (size_t i = 0; i < l.skin.size(); ++i)
            {
                cook_anim::SkinnedVertex v;
                memcpy(v.p, &l.positions[i * 3], 12);
                memcpy(v.bones, l.skin[i].bones, 8), memcpy(v.weights, l.skin[i].weights, 8);
                verts.push_back(v);
                vpart.push_back(k);
            }
        }
        const uint32_t          nn = (uint32_t)gSkel.nodes.size(), nb = (uint32_t)gSkel.bones.size();
        std::vector<float>      lo(parts.size() * 3, 1e30f), hi(parts.size() * 3, -1e30f), world(nn * 12), pal(nb * 12), posed;
        std::vector<anim::Pose> rest(nn);
        anim::restPoses(gSkel.nodes.data(), nn, rest.data());
        anim::palette(gSkel.nodes.data(), nn, gSkel.bones.data(), nb, rest.data(), world.data(), pal.data());
        cook_anim::skin(verts, pal.data(), posed);
        for (size_t i = 0; i < verts.size(); ++i)
            for (int k = 0; k < 3; ++k)
                lo[vpart[i] * 3 + k] = fminf(lo[vpart[i] * 3 + k], posed[i * 3 + k]), hi[vpart[i] * 3 + k] = fmaxf(hi[vpart[i] * 3 + k], posed[i * 3 + k]);
        float size = 0, mlo[3] = { 1e30f, 1e30f, 1e30f }, mhi[3] = { -1e30f, -1e30f, -1e30f };
        for (uint32_t k = 0; k < parts.size(); ++k)
            for (int a = 0; a < 3; ++a)
                mlo[a] = fminf(mlo[a], lo[k * 3 + a]), mhi[a] = fmaxf(mhi[a], hi[k * 3 + a]);
        for (int a = 0; a < 3; ++a)
            size = fmaxf(size, mhi[a] - mlo[a]);
        const float tolerance = fmaxf(cook_anim::kMaxError, size * cook_anim::kRelativeError);
        for (const Clip& c : gSkel.clips)
            clips.push_back(cook_anim::encode(c.name.c_str(), c.duration, gSkel.nodes.data(), nn, gSkel.bones.data(), nb, verts, vpart, lo, hi,
                                              c.source, tolerance, c.stepped, c.keyRate));
        for (size_t i = 0; i < clips.size(); ++i)
            if (gSkel.clips[i].stepped)
                note("clip %s has step keys: each jump blends over one frame (%.1f ms)", clips[i].clip.name, 1000.0f / clips[i].clip.rate);
        for (uint32_t k = 0; k < parts.size(); ++k)
        {
            float big = 0;
            for (int a = 0; a < 3; ++a)
                big = fmaxf(big, hi[k * 3 + a] - lo[k * 3 + a]);
            for (int a = 0; a < 3; ++a)
                parts[k].lo[a] = lo[k * 3 + a] - big * 0.01f - cook_anim::kMaxError, parts[k].hi[a] = hi[k * 3 + a] + big * 0.01f + cook_anim::kMaxError;
        }
    }

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
            note("material %s is alpha blend: blended (renderer 4.8.0 and later)", o.name);
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
    RmodelSkeleton sk = {};
    if (gSkel.on)
    {
        h.flags |= RmodelSkinned;
        h.skin = (uint32_t)at, at = align16(at + (size_t)h.vertexCount * sizeof(RmodelSkinVertex));
        h.skeleton = (uint32_t)at, at = align16(at + sizeof sk);
        sk.nodeCount = (uint32_t)gSkel.nodes.size(), sk.boneCount = (uint32_t)gSkel.bones.size(), sk.clipCount = (uint32_t)clips.size();
        sk.nodes = (uint32_t)at, at = align16(at + gSkel.nodes.size() * sizeof(RmodelNode));
        sk.bones = (uint32_t)at, at = align16(at + gSkel.bones.size() * sizeof(RmodelBone));
        sk.clips = (uint32_t)at, at = align16(at + clips.size() * sizeof(RmodelClip));
        for (cook_anim::Encoded& c : clips)
            c.clip.offset = at, at = align16(at + c.data.size());
    }
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
            if (gSkel.on)
                memcpy(&file[h.skin + (size_t)fl.firstVertex * sizeof(RmodelSkinVertex)], l.skin.data(), l.skin.size() * sizeof(RmodelSkinVertex));
        }
    if (gSkel.on)
    {
        memcpy(&file[h.skeleton], &sk, sizeof sk);
        memcpy(&file[sk.nodes], gSkel.nodes.data(), gSkel.nodes.size() * sizeof(RmodelNode));
        memcpy(&file[sk.bones], gSkel.bones.data(), gSkel.bones.size() * sizeof(RmodelBone));
        for (size_t i = 0; i < clips.size(); ++i)
        {
            memcpy(&file[sk.clips + i * sizeof(RmodelClip)], &clips[i].clip, sizeof(RmodelClip));
            memcpy(&file[clips[i].clip.offset], clips[i].data.data(), clips[i].data.size());
        }
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
    if (gSkel.on)
    {
        size_t bytes = 0, raw = 0;
        float  worst = 0, over = 0;
        for (const cook_anim::Encoded& c : clips)
            bytes += c.data.size(), raw += c.rawBytes, worst = fmaxf(worst, c.error), over = fmaxf(over, c.error / c.tolerance);
        printf("skinned: %zu nodes, %zu bones, %zu clip%s (%.1f KB, %.1fx smaller than floats; the skin within %.2f mm of the file)%s\n",
               gSkel.nodes.size(), gSkel.bones.size(), clips.size(), clips.size() == 1 ? "" : "s", bytes / 1024.0, bytes ? (double)raw / bytes : 0.0,
               worst * 1000.0f, clips.empty() ? "" : ":");
        for (const cook_anim::Encoded& c : clips)
            printf("  %s: %.2f s, %u frames at %.0f a second, %.1f KB, within %.2f mm\n", c.clip.name, c.clip.duration, c.clip.frames,
                   c.clip.frames > 1 ? c.clip.rate : 0.0f, c.data.size() / 1024.0, c.error * 1000.0f);
        if (over > 1.0f)
            warn("a clip strays %.1fx its tolerance (%.2g mm) at its best rate: very fast motion", over, clips.empty() ? 0.0 : clips[0].tolerance * 1000.0);
    }
    for (const std::string& s : gNotes)
        printf("%s\n", s.c_str());
    for (const std::string& s : gWarnings)
        printf("warning: %s\n", s.c_str());
    return done(0);
}
