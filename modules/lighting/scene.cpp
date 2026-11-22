// The scene exports: meshes (built-in shapes and custom), materials, instances, lights, camera.
// Exports only touch tables and state; the GPU side picks the changes up when it draws.
#include "yea_lighting.h"
#include "scene.h"

#include <math.h>

namespace lighting
{
namespace
{
const uint32_t kNoMesh = 0xFFFFFFFFu;

struct Build
{
    uint32_t active;
    uint32_t firstVertex;
    uint32_t firstIndex;
    uint32_t failed;
};
Build* gBuild;

Build* build() { return gBuild; } // taken in bindState()

template<class T> uint32_t rowOf(const yea::Table<T>& t, yea_handle h)
{
    uint32_t row = h & 0xFFFFFu;
    if (row >= t.size() || !t[row].alive || (t[row].gen & 0xFFFu) != (h >> 20))
        return 0xFFFFFFFFu;
    return row;
}

// A fresh row: a dead one reused, else a new one; nullptr when full.
template<class T> T* allocRow(const yea::Table<T>& t, uint32_t* rowOut)
{
    for (uint32_t i = 0; i < t.size(); ++i)
        if (!t[i].alive)
        {
            uint32_t gen = t[i].gen + 1;
            t[i] = T{};
            t[i].gen = gen;
            t[i].alive = 1;
            *rowOut = i;
            return &t[i];
        }
    T* r = t.add();
    if (!r)
        return nullptr;
    r->alive = 1;
    *rowOut = t.size() - 1;
    return r;
}

template<class T> yea_handle handleOf(const yea::Table<T>& t, uint32_t row) { return row | ((t[row].gen & 0xFFFu) << 20); }

uint32_t vertex(float px, float py, float pz, float nx, float ny, float nz)
{
    Vertex* v = Vertexes.add();
    if (!v)
    {
        build()->failed = 1;
        return 0;
    }
    v->pos = { px, py, pz };
    v->normal = { nx, ny, nz };
    return Vertexes.size() - 1 - build()->firstVertex;
}

void tri(uint32_t a, uint32_t b, uint32_t c)
{
    if (Indexes.size() + 3 > Indexes.capacity())
    {
        build()->failed = 1;
        return;
    }
    Indexes.add()->v = a;
    Indexes.add()->v = b;
    Indexes.add()->v = c;
}

void quad(uint32_t a, uint32_t b, uint32_t c, uint32_t d)
{
    tri(a, b, c);
    tri(a, c, d);
}

// Shapes are made between begin() and finish().
bool begin()
{
    Build* b = build();
    if (b->active)
        return false;
    b->active = 1;
    b->failed = 0;
    b->firstVertex = Vertexes.size();
    b->firstIndex = Indexes.size();
    return true;
}

uint32_t finish()
{
    Build* b = build();
    if (!b->active)
        return kNoMesh;
    b->active = 0;
    uint32_t vcount = Vertexes.size() - b->firstVertex;
    uint32_t icount = Indexes.size() - b->firstIndex;
    Mesh*    m = b->failed || vcount == 0 || icount == 0 ? nullptr : Meshes.add();
    if (!m)
    {
        // Roll back what this mesh added.
        (*Vertexes.slot)->count = b->firstVertex;
        (*Indexes.slot)->count = b->firstIndex;
        return kNoMesh;
    }
    float lo[3] = { 1e30f, 1e30f, 1e30f }, hi[3] = { -1e30f, -1e30f, -1e30f };
    for (uint32_t i = 0; i < vcount; ++i)
    {
        const yea_float3& p = Vertexes[b->firstVertex + i].pos;
        const float       q[3] = { p.x, p.y, p.z };
        for (int k = 0; k < 3; ++k)
        {
            lo[k] = q[k] < lo[k] ? q[k] : lo[k];
            hi[k] = q[k] > hi[k] ? q[k] : hi[k];
        }
    }
    m->firstVertex = b->firstVertex;
    m->vertexCount = vcount;
    m->firstIndex = b->firstIndex;
    m->indexCount = icount;
    m->center = { (lo[0] + hi[0]) * 0.5f, (lo[1] + hi[1]) * 0.5f, (lo[2] + hi[2]) * 0.5f };
    float r2 = 0;
    for (uint32_t i = 0; i < vcount; ++i)
    {
        const yea_float3& p = Vertexes[b->firstVertex + i].pos;
        float dx = p.x - m->center.x, dy = p.y - m->center.y, dz = p.z - m->center.z;
        float d2 = dx * dx + dy * dy + dz * dz;
        r2 = d2 > r2 ? d2 : r2;
    }
    m->radius = sqrtf(r2);
    return Meshes.size() - 1;
}

uint32_t clampSegments(uint32_t s, uint32_t lo, uint32_t hi) { return s < lo ? lo : (s > hi ? hi : s); }
} // namespace

// --- Meshes -----------------------------------------------------------------------------------

uint32_t mesh_box()
{
    if (!begin())
        return kNoMesh;
    // Each face: normal n, axes u and v; corners at n/2 +- u/2 +- v/2.
    const float faces[6][9] = {
        { 1, 0, 0, 0, 0, 1, 0, 1, 0 },  { -1, 0, 0, 0, 0, -1, 0, 1, 0 }, { 0, 1, 0, 1, 0, 0, 0, 0, 1 },
        { 0, -1, 0, 1, 0, 0, 0, 0, -1 }, { 0, 0, 1, -1, 0, 0, 0, 1, 0 }, { 0, 0, -1, 1, 0, 0, 0, 1, 0 },
    };
    for (const auto& f : faces)
    {
        uint32_t base = 0;
        for (int c = 0; c < 4; ++c)
        {
            float su = (c == 1 || c == 2) ? 0.5f : -0.5f;
            float sv = (c >= 2) ? 0.5f : -0.5f;
            float px = f[0] * 0.5f + f[3] * su + f[6] * sv;
            float py = f[1] * 0.5f + f[4] * su + f[7] * sv;
            float pz = f[2] * 0.5f + f[5] * su + f[8] * sv;
            uint32_t i = vertex(px, py, pz, f[0], f[1], f[2]);
            if (c == 0)
                base = i;
        }
        quad(base, base + 1, base + 2, base + 3);
    }
    return finish();
}

uint32_t mesh_sphere(uint32_t segments)
{
    uint32_t seg = clampSegments(segments, 6, 256);
    uint32_t rings = seg / 2;
    if (!begin())
        return kNoMesh;
    for (uint32_t r = 0; r <= rings; ++r)
    {
        float t = 3.14159265f * (float)r / (float)rings;
        for (uint32_t s = 0; s <= seg; ++s)
        {
            float p = 6.28318531f * (float)s / (float)seg;
            float nx = sinf(t) * cosf(p), ny = cosf(t), nz = sinf(t) * sinf(p);
            vertex(nx * 0.5f, ny * 0.5f, nz * 0.5f, nx, ny, nz);
        }
    }
    for (uint32_t r = 0; r < rings; ++r)
        for (uint32_t s = 0; s < seg; ++s)
        {
            uint32_t a = r * (seg + 1) + s, b = a + seg + 1;
            quad(a, a + 1, b + 1, b);
        }
    return finish();
}

uint32_t mesh_plane()
{
    if (!begin())
        return kNoMesh;
    uint32_t a = vertex(-0.5f, 0, -0.5f, 0, 1, 0);
    vertex(0.5f, 0, -0.5f, 0, 1, 0);
    vertex(0.5f, 0, 0.5f, 0, 1, 0);
    vertex(-0.5f, 0, 0.5f, 0, 1, 0);
    quad(a, a + 3, a + 2, a + 1);
    return finish();
}

namespace
{
// A surface of revolution around y: rings of (radius, y, normal radial, normal y), plus caps.
uint32_t revolve(uint32_t segments, const float (*profile)[4], uint32_t count, bool capBottom, bool capTop)
{
    uint32_t seg = clampSegments(segments, 3, 256);
    if (!begin())
        return kNoMesh;
    for (uint32_t k = 0; k < count; ++k)
        for (uint32_t s = 0; s <= seg; ++s)
        {
            float p = 6.28318531f * (float)s / (float)seg;
            float c = cosf(p), sn = sinf(p);
            vertex(profile[k][0] * c, profile[k][1], profile[k][0] * sn, profile[k][2] * c, profile[k][3], profile[k][2] * sn);
        }
    for (uint32_t k = 0; k + 1 < count; ++k)
        for (uint32_t s = 0; s < seg; ++s)
        {
            uint32_t a = k * (seg + 1) + s, b = a + seg + 1;
            quad(a, b, b + 1, a + 1);
        }
    for (int capIndex = 0; capIndex < 2; ++capIndex)
    {
        bool top = capIndex == 1;
        if (top ? !capTop : !capBottom)
            continue;
        const float* ring = profile[top ? count - 1 : 0];
        float        ny = top ? 1.0f : -1.0f;
        uint32_t     centre = vertex(0, ring[1], 0, 0, ny, 0);
        for (uint32_t s = 0; s <= seg; ++s)
        {
            float p = 6.28318531f * (float)s / (float)seg;
            vertex(ring[0] * cosf(p), ring[1], ring[0] * sinf(p), 0, ny, 0);
        }
        for (uint32_t s = 0; s < seg; ++s)
            if (top)
                tri(centre, centre + 2 + s, centre + 1 + s);
            else
                tri(centre, centre + 1 + s, centre + 2 + s);
    }
    return finish();
}
} // namespace

uint32_t mesh_cylinder(uint32_t segments)
{
    const float profile[2][4] = { { 0.5f, -0.5f, 1, 0 }, { 0.5f, 0.5f, 1, 0 } };
    return revolve(segments, profile, 2, true, true);
}

uint32_t mesh_cone(uint32_t segments)
{
    // Side normal of a cone with radius 0.5 and height 1: (1, 0.5) normalised.
    const float nr = 0.894427f, ny = 0.447214f;
    const float profile[2][4] = { { 0.5f, -0.5f, nr, ny }, { 0.0f, 0.5f, nr, ny } };
    return revolve(segments, profile, 2, true, false);
}

uint32_t mesh_torus(float thickness, uint32_t segments)
{
    uint32_t seg = clampSegments(segments, 6, 256);
    uint32_t sides = seg / 2 < 4 ? 4 : seg / 2;
    float    minor = thickness < 0.01f ? 0.01f : (thickness > 0.49f ? 0.49f : thickness);
    float    major = 0.5f - minor;
    if (!begin())
        return kNoMesh;
    for (uint32_t i = 0; i <= seg; ++i)
    {
        float u = 6.28318531f * (float)i / (float)seg;
        for (uint32_t j = 0; j <= sides; ++j)
        {
            float v = 6.28318531f * (float)j / (float)sides;
            float nx = cosf(v) * cosf(u), ny = sinf(v), nz = cosf(v) * sinf(u);
            vertex(major * cosf(u) + minor * nx, minor * ny, major * sinf(u) + minor * nz, nx, ny, nz);
        }
    }
    for (uint32_t i = 0; i < seg; ++i)
        for (uint32_t j = 0; j < sides; ++j)
        {
            uint32_t a = i * (sides + 1) + j, b = a + sides + 1;
            quad(a, a + 1, b + 1, b);
        }
    return finish();
}

bool mesh_begin() { return begin(); }

bool meshBuilding() { return build()->active != 0; }

uint32_t mesh_vertex(yea_float3 pos, yea_float3 normal)
{
    if (!build()->active)
        return 0;
    return vertex(pos.x, pos.y, pos.z, normal.x, normal.y, normal.z);
}

void mesh_triangle(uint32_t a, uint32_t b, uint32_t c)
{
    Build* bl = build();
    if (!bl->active)
        return;
    uint32_t n = Vertexes.size() - bl->firstVertex;
    if (a >= n || b >= n || c >= n)
    {
        bl->failed = 1;
        return;
    }
    tri(a, b, c);
}

uint32_t mesh_end() { return finish(); }

// --- Materials --------------------------------------------------------------------------------

yea_handle material_add(yea_float3 albedo, float roughness, float metallic)
{
    uint32_t  row;
    Material* m = allocRow(Materials, &row);
    if (!m)
        return 0xFFFFFFFFu;
    m->reflectance = 0.5f;
    m->reflections = 1;
    yea_handle h = handleOf(Materials, row);
    material_set(h, albedo, roughness, metallic);
    return h;
}

void material_set(yea_handle h, yea_float3 albedo, float roughness, float metallic)
{
    uint32_t row = rowOf(Materials, h);
    if (row == 0xFFFFFFFFu)
        return;
    Material& m = Materials[row];
    m.albedo = albedo;
    m.roughness = roughness;
    m.metallic = metallic;
}

void material_emissive(yea_handle h, yea_float3 color, float luminance)
{
    uint32_t row = rowOf(Materials, h);
    if (row == 0xFFFFFFFFu)
        return;
    Materials[row].emissive = { color.x * luminance, color.y * luminance, color.z * luminance };
}

void material_reflections(yea_handle h, bool on)
{
    uint32_t row = rowOf(Materials, h);
    if (row != 0xFFFFFFFFu)
        Materials[row].reflections = on;
}

uint32_t materialRow(yea_handle h)
{
    uint32_t row = rowOf(Materials, h);
    return row == 0xFFFFFFFFu ? 0 : row;
}

// --- Instances --------------------------------------------------------------------------------

yea_handle instance_add(uint32_t mesh, yea_handle material)
{
    uint32_t  row;
    Instance* i = allocRow(Instances, &row);
    if (!i)
        return 0xFFFFFFFFu;
    i->mesh = mesh;
    i->material = materialRow(material);
    i->rot = { 0, 0, 0, 1 };
    i->scale = { 1, 1, 1 };
    i->shadow = 1;
    i->moved = yea::host()->frame();
    return handleOf(Instances, row);
}

void instance_place(yea_handle h, yea_float3 pos, yea_float4 rotation, yea_float3 scale)
{
    uint32_t row = rowOf(Instances, h);
    if (row == 0xFFFFFFFFu)
        return;
    Instance& i = Instances[row];
    if (memcmp(&i.pos, &pos, sizeof pos) || memcmp(&i.rot, &rotation, sizeof rotation) || memcmp(&i.scale, &scale, sizeof scale))
        i.moved = yea::host()->frame();
    i.pos = pos;
    i.rot = rotation;
    i.scale = scale;
}

void instance_material(yea_handle h, yea_handle material)
{
    uint32_t row = rowOf(Instances, h);
    if (row != 0xFFFFFFFFu)
        Instances[row].material = materialRow(material);
}

void instance_shadow(yea_handle h, bool cast)
{
    uint32_t row = rowOf(Instances, h);
    if (row != 0xFFFFFFFFu && Instances[row].shadow != (uint32_t)cast)
    {
        Instances[row].shadow = cast;
        Instances[row].moved = yea::host()->frame();
    }
}

void instance_remove(yea_handle h)
{
    uint32_t row = rowOf(Instances, h);
    if (row == 0xFFFFFFFFu)
        return;
    Instances[row].alive = 0;
    Instances[row].moved = yea::host()->frame();
    sceneState()->removed = yea::host()->frame();
}

uint32_t instance_count()
{
    uint32_t n = 0;
    for (const Instance& i : Instances)
        n += i.alive;
    return n;
}

// --- Lights -----------------------------------------------------------------------------------

yea_handle light_add(uint32_t type)
{
    uint32_t row;
    Light*   l = allocRow(Lights, &row);
    if (!l)
        return 0xFFFFFFFFu;
    l->type = type > 2 ? 1 : type;
    l->dir = { 0, -1, 0 };
    l->color = { 1, 1, 1 };
    l->intensity = type == 0 ? 10000.0f : 100.0f;
    l->range = 10.0f;
    l->inner = 20.0f;
    l->outer = 30.0f;
    return handleOf(Lights, row);
}

void light_place(yea_handle h, yea_float3 pos, yea_float3 dir)
{
    uint32_t row = rowOf(Lights, h);
    if (row == 0xFFFFFFFFu)
        return;
    Lights[row].pos = pos;
    float l = sqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
    if (l > 1e-6f)
        Lights[row].dir = { dir.x / l, dir.y / l, dir.z / l };
}

void light_color(yea_handle h, yea_float3 color, float intensity)
{
    uint32_t row = rowOf(Lights, h);
    if (row == 0xFFFFFFFFu)
        return;
    Lights[row].color = color;
    Lights[row].intensity = intensity < 0 ? 0 : intensity;
}

void light_range(yea_handle h, float range)
{
    uint32_t row = rowOf(Lights, h);
    if (row != 0xFFFFFFFFu)
        Lights[row].range = range < 0.01f ? 0.01f : range;
}

void light_cone(yea_handle h, float inner_deg, float outer_deg)
{
    uint32_t row = rowOf(Lights, h);
    if (row == 0xFFFFFFFFu)
        return;
    float outer = outer_deg < 1 ? 1 : (outer_deg > 89 ? 89 : outer_deg);
    float inner = inner_deg < 0 ? 0 : (inner_deg > outer ? outer : inner_deg);
    Lights[row].inner = inner;
    Lights[row].outer = outer;
}

void light_shadow(yea_handle h, bool cast)
{
    uint32_t row = rowOf(Lights, h);
    if (row != 0xFFFFFFFFu)
        Lights[row].shadow = cast;
}

void light_remove(yea_handle h)
{
    uint32_t row = rowOf(Lights, h);
    if (row != 0xFFFFFFFFu)
        Lights[row].alive = 0;
}

uint32_t light_count()
{
    uint32_t n = 0;
    for (const Light& l : Lights)
        n += l.alive;
    return n;
}

// State blocks are taken once, in load(): the host files a block under the module that is running when it
// is asked for, so one asked for inside an export (called from another module's update) would be that
// module's, not this one's.
SceneState*  gScene;
CameraState* gCamera;

void bindState()
{
    int fresh = 0;
    gBuild = (Build*)yea::host()->state("lighting.build", 1, sizeof(Build), &fresh);
    gScene = (SceneState*)yea::host()->state("lighting.scene", 1, sizeof(SceneState), &fresh);
    gCamera = (CameraState*)yea::host()->state("lighting.camera", 1, sizeof(CameraState), &fresh);
    if (fresh)
    {
        gCamera->eye = { 0, 3, -8 };
        gCamera->target = { 0, 0, 0 };
        gCamera->fov = 60;
    }
}

SceneState* sceneState() { return gScene; }

void clear()
{
    Meshes.clear();
    Vertexes.clear();
    Indexes.clear();
    Materials.clear();
    Instances.clear();
    Lights.clear();
    build()->active = 0;
    sceneState()->removed = yea::host()->frame();
}

// --- Camera -----------------------------------------------------------------------------------

CameraState* cameraState() { return gCamera; }

void camera(yea_float3 eye, yea_float3 target, float fov_deg)
{
    CameraState* c = cameraState();
    c->eye = eye;
    c->target = target;
    c->fov = fov_deg < 5 ? 5 : (fov_deg > 150 ? 150 : fov_deg);
}
} // namespace lighting
