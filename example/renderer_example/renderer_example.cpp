// The Renderer component's example view: a floor, a metal and a plastic sphere row sweeping
// roughness, assorted shapes, a sun, many moving coloured point lights, sweeping spot lights, four area lights (a
// softbox, a pendant disc, a fluorescent tube and a glowing sphere), two reflection probes, and a free-look camera (hold the mouse button to look, the move keys to fly).
// Reset: the scene is rebuilt with the lights and camera where they start.
#include "yea_renderer_example.h"

#include <math.h>
#include <stdio.h>

namespace renderer_example
{
namespace
{
const uint32_t kMaxLights = 512, kMaxSpots = 16, kMaxSpin = 16, kAreas = 4;
const float    kPi = 3.14159265f;

struct State
{
    uint32_t   built;
    float      t;
    yea_float3 eye;
    float      yaw, pitch; // look direction: (cos yaw cos pitch, sin pitch, sin yaw cos pitch)
    uint32_t   pictureUnused; // was the card-picture flag; kept so the state's layout stays the same
    yea_handle sun;
    uint32_t   lightCount;
    yea_handle lights[kMaxLights];
    yea_handle markers[kMaxLights];
    yea_handle markerMaterials[kMaxLights];
    uint32_t   spotCount;
    yea_handle spots[kMaxSpots];
    uint32_t   spinCount;
    yea_handle spin[kMaxSpin];
    yea_float3 spinPos[kMaxSpin];
    yea_float3 spinScale[kMaxSpin];
    yea_float3 spinAxis[kMaxSpin];
    uint32_t   sphereMesh;
    yea_handle spotMarkers[kMaxSpots];
    yea_handle spotMarkerMaterials[kMaxSpots];
    uint32_t   instances, lightsAlive; // the renderer's live counts once this frame's scene was in place
    uint32_t   cylinderMesh;           // the spot lamps' housings and lenses
    yea_handle housingMaterial;
    yea_handle spotHousings[kMaxSpots];
    yea_handle areas[kAreas];          // the area lights and their emitters' materials
    yea_handle areaMaterials[kAreas];
};
State* gState;
bool   gFreshVersion; // this version of the module hasn't built its scene yet: build it once (new meshes reach a running game)

// Mouse look: the host holds the mouse (hidden, kept in the window) while the left button stays down
// after a press on the game; it lets go itself when F1 opens, focus is lost, the view switches or resets.
bool   gLooking;
bool   gPicture; // Yea is taking the card picture; a global so a reload never leaves it stuck on

// Where the camera starts: 17 m from (0, 0.8, 0) at yaw 0.6 and 0.42 rad up, looking at it.
const float kStartYaw = 0.6f, kStartPitch = 0.42f, kStartDist = 17.0f;

void startCamera(State* s)
{
    const float yaw = kStartYaw, pitch = kStartPitch, dist = kStartDist;
    s->eye = { cosf(yaw) * cosf(pitch) * dist, 0.8f + sinf(pitch) * dist, sinf(yaw) * cosf(pitch) * dist };
    s->yaw = yaw + kPi;
    s->pitch = -pitch;
}

State* state()
{
    if (!gState)
    {
        int fresh = 0;
        gState = (State*)yea::host()->state("renderer_example", 6, sizeof(State), &fresh);
        if (fresh)
            startCamera(gState);
    }
    return gState;
}

float frac(float x) { return x - floorf(x); }

yea_float3 hue(float h)
{
    float r = fabsf(h * 6 - 3) - 1, g = 2 - fabsf(h * 6 - 2), b = 2 - fabsf(h * 6 - 4);
    auto  c = [](float v) { return v < 0 ? 0 : (v > 1 ? 1 : v); };
    return { c(r), c(g), c(b) };
}

yea_float4 axisAngle(yea_float3 a, float angle)
{
    float l = sqrtf(a.x * a.x + a.y * a.y + a.z * a.z), s = sinf(angle * 0.5f) / (l > 0 ? l : 1);
    return { a.x * s, a.y * s, a.z * s, cosf(angle * 0.5f) };
}

const yea_float4 kNoRotation = { 0, 0, 0, 1 };

yea_float3 cross(yea_float3 a, yea_float3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
yea_float3 normalized(yea_float3 a)
{
    float l = sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
    return { a.x / l, a.y / l, a.z / l };
}

// The rotation taking the mesh's x, y and z to these (orthonormal, right-handed) axes.
yea_float4 basis(yea_float3 x, yea_float3 y, yea_float3 z)
{
    float tr = x.x + y.y + z.z;
    if (tr > 0)
    {
        float k = 0.5f / sqrtf(tr + 1);
        return { (y.z - z.y) * k, (z.x - x.z) * k, (x.y - y.x) * k, 0.25f / k };
    }
    if (x.x > y.y && x.x > z.z)
    {
        float k = 2 * sqrtf(1 + x.x - y.y - z.z);
        return { 0.25f * k, (y.x + x.y) / k, (z.x + x.z) / k, (y.z - z.y) / k };
    }
    if (y.y > z.z)
    {
        float k = 2 * sqrtf(1 + y.y - x.x - z.z);
        return { (y.x + x.y) / k, 0.25f * k, (z.y + y.z) / k, (z.x - x.z) / k };
    }
    float k = 2 * sqrtf(1 + z.z - x.x - y.y);
    return { (z.x + x.z) / k, (z.y + y.z) / k, 0.25f * k, (x.y - y.x) / k };
}

// The area lights: type (3 rect, 4 disc, 5 sphere, 6 tube), centre, facing (a tube's axis), the rect's or disc's width
// axis, width and height as light_size takes them, range and colour.
struct AreaLight
{
    uint32_t   type;
    yea_float3 pos, dir, across;
    float      width, height, range;
    yea_float3 color;
};
const AreaLight kAreaLights[kAreas] = {
    { 3, { -9.3f, 1.8f, -3.2f }, { 1, -0.15f, 0.1f }, { 0.1f, 0, -1 }, 1.4f, 0.9f, 10, { 1.0f, 0.95f, 0.88f } }, // softbox
    { 4, { -5.1f, 3.4f, 2.5f }, { 0, -1, 0 }, { 1, 0, 0 }, 0.9f, 0.9f, 9, { 1.0f, 0.8f, 0.55f } },               // pendant
    { 6, { 0.0f, 0.65f, -6.85f }, { 1, 0, 0 }, { 0, 1, 0 }, 4.0f, 0.03f, 8, { 0.8f, 0.92f, 1.0f } },              // tube
    { 5, { 0.5f, 1.2f, 2.2f }, { 0, -1, 0 }, { 1, 0, 0 }, 0.3f, 0, 8, { 0.4f, 0.8f, 1.0f } },                     // sphere
};

// The emitting surface's area as light_color's candela count it (seen face-on: a tube's and a sphere's outline).
float areaOf(const AreaLight& a)
{
    switch (a.type)
    {
    case 3: return a.width * a.height;
    case 4: return kPi * 0.25f * a.width * a.height;
    case 5: return kPi * a.width * a.width;
    default: return a.width * 2 * a.height;
    }
}

void place(uint32_t mesh, yea_handle material, yea_float3 pos, yea_float3 scale, yea_float4 rot = kNoRotation)
{
    renderer::instance_place(renderer::instance_add(mesh, material), pos, rot, scale);
}

void addSpin(State* s, uint32_t mesh, yea_handle material, yea_float3 pos, yea_float3 scale, yea_float3 axis)
{
    if (s->spinCount >= kMaxSpin)
        return;
    uint32_t i = s->spinCount++;
    s->spin[i] = renderer::instance_add(mesh, material);
    s->spinPos[i] = pos;
    s->spinScale[i] = scale;
    s->spinAxis[i] = axis;
    renderer::instance_place(s->spin[i], pos, kNoRotation, scale);
}

// Reflection probes: one over the sphere rows, one for the whole yard; the floor is their boxes' bottom.
void addProbes()
{
    renderer_probes::probe_add({ 0, 1.0f, -2.8f }, { -9.0f, -0.5f, -6.0f }, { 9.0f, 4.0f, 0.5f });
    renderer_probes::probe_add({ 0, 2.0f, 1.5f }, { -16.0f, -0.5f, -16.0f }, { 16.0f, 10.0f, 16.0f });
}

// The whole scene, from nothing. Not while the renderer is unloaded (a crash, a revert, a reload): its imports are
// stubs then, and a scene built against them would hold invalid handles that never move what the renderer has.
void build(State* s)
{
    s->built = 0;
    renderer::clear();
    renderer_probes::probe_clear();
    s->lightCount = s->spotCount = s->spinCount = 0;
    // Segments so the silhouettes stay round in the close-ups (a sphere filling the screen at 2560x1356 is under a
    // third of a pixel off a true circle at 128); farther away the renderer draws simpler versions of each shape, so
    // this costs only up close. The light orbs are small, so 32 keeps their triangles a useful size.
    uint32_t sphere = renderer::mesh_sphere(128);
    if (sphere == 0xFFFFFFFFu)
        return; // the renderer isn't loaded: try again next frame
    uint32_t box = renderer::mesh_box(), plane = renderer::mesh_plane();
    uint32_t cylinder = renderer::mesh_cylinder(96), cone = renderer::mesh_cone(96), torus = renderer::mesh_torus(0.15f, 128);
    s->sphereMesh = renderer::mesh_sphere(32); // the light orbs
    s->cylinderMesh = cylinder;
    s->housingMaterial = renderer::material_add({ 0.06f, 0.06f, 0.065f }, 0.45f, 0.0f); // a dark painted lamp body

    yea_handle floor = renderer::material_add({ 0.42f, 0.42f, 0.42f }, 0.65f, 0);
    place(plane, floor, { 0, 0, 0 }, { 60, 1, 60 });

    // Roughness sweeps: gold metal at the back, red plastic in front of it.
    for (int i = 0; i < 8; ++i)
    {
        float      r = 0.04f + 0.96f * (float)i / 7.0f;
        float      x = -7.0f + 2.0f * (float)i;
        yea_handle gold = renderer::material_add({ 1.0f, 0.77f, 0.34f }, r, 1);
        yea_handle red = renderer::material_add({ 0.75f, 0.06f, 0.05f }, r, 0);
        place(sphere, gold, { x, 0.75f, -4.0f }, { 1.5f, 1.5f, 1.5f });
        place(sphere, red, { x, 0.75f, -1.6f }, { 1.5f, 1.5f, 1.5f });
    }

    // Assorted shapes.
    yea_handle white = renderer::material_add({ 0.85f, 0.85f, 0.82f }, 0.4f, 0);
    yea_handle stone = renderer::material_add({ 0.55f, 0.5f, 0.45f }, 0.85f, 0);
    yea_handle steel = renderer::material_add({ 0.9f, 0.91f, 0.92f }, 0.25f, 1);
    yea_handle copper = renderer::material_add({ 0.95f, 0.64f, 0.54f }, 0.3f, 1);
    yea_handle blue = renderer::material_add({ 0.08f, 0.2f, 0.7f }, 0.2f, 0);
    yea_handle green = renderer::material_add({ 0.1f, 0.5f, 0.15f }, 0.55f, 0);

    for (int i = 0; i < 8; ++i)
    {
        float a = (float)i / 8.0f * 2 * kPi;
        place(cylinder, stone, { cosf(a) * 11.0f, 2.0f, sinf(a) * 11.0f }, { 0.8f, 4.0f, 0.8f });
    }
    place(box, white, { -5.0f, 0.6f, 2.5f }, { 1.2f, 1.2f, 1.2f }, axisAngle({ 0, 1, 0 }, 0.4f));
    place(box, white, { -5.2f, 1.6f, 2.4f }, { 0.8f, 0.8f, 0.8f }, axisAngle({ 0, 1, 0 }, 1.1f));
    place(box, copper, { 5.5f, 0.5f, 2.0f }, { 3.0f, 1.0f, 1.0f });
    place(cone, blue, { -2.0f, 0.75f, 3.5f }, { 1.2f, 1.5f, 1.2f });
    place(cone, green, { 2.5f, 1.0f, 4.0f }, { 1.4f, 2.0f, 1.4f });
    place(cylinder, steel, { 0.5f, 0.4f, 2.2f }, { 1.6f, 0.8f, 1.6f });
    place(box, stone, { 0.0f, 0.25f, -7.5f }, { 18.0f, 0.5f, 1.5f });
    addSpin(s, torus, steel, { 4.0f, 1.6f, 5.5f }, { 2.6f, 2.6f, 2.6f }, { 1, 0.3f, 0 });
    addSpin(s, torus, copper, { -4.5f, 1.4f, 5.5f }, { 2.4f, 2.4f, 2.4f }, { 0, 1, 0.5f });
    addSpin(s, box, blue, { 8.5f, 1.2f, -1.0f }, { 1.2f, 1.2f, 1.2f }, { 1, 1, 0 });
    addSpin(s, torus, white, { -8.5f, 1.3f, -1.0f }, { 2.2f, 2.2f, 2.2f }, { 0, 0, 1 });

    addProbes();

    // Area lights, each with an emitter of its exact shape that casts no shadow (it would shadow its own light, as the
    // orbs would), and dark fittings: the softbox's backing and stand, the pendant's canopy and cord, the tube's
    // brackets. The fittings beside an emitter cast none either; the stand's pole and foot do.
    for (uint32_t i = 0; i < kAreas; ++i)
    {
        const AreaLight& a = kAreaLights[i];
        yea_float3       z = normalized(a.dir), x = normalized(a.across), y = cross(z, x);
        s->areas[i] = renderer::light_add(a.type);
        renderer::light_place(s->areas[i], a.pos, z);
        renderer::light_size(s->areas[i], a.width, a.height, x);
        renderer::light_range(s->areas[i], a.range);
        renderer::light_shadow(s->areas[i], true);
        s->areaMaterials[i] = renderer::material_add({ 0, 0, 0 }, 1, 0);
        yea_float3 behind = { a.pos.x - z.x * 0.006f, a.pos.y - z.y * 0.006f, a.pos.z - z.z * 0.006f };
        yea_handle e;
        if (a.type == 3) // a thin slab, its face on the light's plane
        {
            e = renderer::instance_add(box, s->areaMaterials[i]);
            renderer::instance_place(e, behind, basis(x, y, z), { a.width, a.height, 0.01f });
            yea_handle back = renderer::instance_add(box, s->housingMaterial);
            renderer::instance_place(back, { a.pos.x - z.x * 0.065f, a.pos.y - z.y * 0.065f, a.pos.z - z.z * 0.065f }, basis(x, y, z),
                                     { a.width + 0.12f, a.height + 0.12f, 0.1f });
            renderer::instance_shadow(back, false);
            yea_float3 foot = { a.pos.x - z.x * 0.25f, 0, a.pos.z - z.z * 0.25f };
            place(cylinder, s->housingMaterial, { foot.x, a.pos.y * 0.5f, foot.z }, { 0.05f, a.pos.y, 0.05f });
            place(cylinder, s->housingMaterial, { foot.x, 0.02f, foot.z }, { 0.5f, 0.04f, 0.5f });
        }
        else if (a.type == 4) // a glowing diffuser 4 cm deep, its rim showing under a slim canopy hung from a cord
        {
            yea_float4 q = basis(x, z, cross(x, z));
            e = renderer::instance_add(cylinder, s->areaMaterials[i]);
            renderer::instance_place(e, { a.pos.x - z.x * 0.02f, a.pos.y - z.y * 0.02f, a.pos.z - z.z * 0.02f }, q, { a.width, 0.04f, a.height });
            yea_handle canopy = renderer::instance_add(cylinder, s->housingMaterial);
            renderer::instance_place(canopy, { a.pos.x - z.x * 0.06f, a.pos.y - z.y * 0.06f, a.pos.z - z.z * 0.06f }, q,
                                     { a.width * 0.96f, 0.04f, a.height * 0.96f });
            renderer::instance_shadow(canopy, false);
            yea_handle cord = renderer::instance_add(cylinder, s->housingMaterial);
            renderer::instance_place(cord, { a.pos.x, a.pos.y + 2.09f, a.pos.z }, kNoRotation, { 0.015f, 4.0f, 0.015f });
            renderer::instance_shadow(cord, false);
        }
        else if (a.type == 5)
        {
            e = renderer::instance_add(sphere, s->areaMaterials[i]);
            renderer::instance_place(e, a.pos, kNoRotation, { 2 * a.width, 2 * a.width, 2 * a.width });
        }
        else // a tube on two brackets
        {
            e = renderer::instance_add(cylinder, s->areaMaterials[i]);
            renderer::instance_place(e, a.pos, basis(x, z, cross(x, z)), { 2 * a.height, a.width, 2 * a.height });
            for (int side = -1; side <= 1; side += 2)
            {
                yea_float3 b = { a.pos.x + z.x * a.width * 0.4f * side, a.pos.y - 0.075f, a.pos.z + z.z * a.width * 0.4f * side };
                yea_handle bracket = renderer::instance_add(box, s->housingMaterial);
                renderer::instance_place(bracket, b, kNoRotation, { 0.04f, 0.15f, 0.04f });
                renderer::instance_shadow(bracket, false);
            }
        }
        renderer::instance_shadow(e, false);
    }

    s->sun = renderer::light_add(0);
    renderer::light_shadow(s->sun, true);
    s->built = 1;
}

void syncLights(State* s)
{
    uint32_t want = settings.light_count < 0 ? 0 : (uint32_t)settings.light_count;
    want = want > kMaxLights ? kMaxLights : want;
    while (s->lightCount < want)
    {
        uint32_t i = s->lightCount;
        s->lights[i] = renderer::light_add(1);
        s->markerMaterials[i] = renderer::material_add({ 0, 0, 0 }, 1, 0);
        s->markers[i] = renderer::instance_add(s->sphereMesh, s->markerMaterials[i]);
        renderer::instance_shadow(s->markers[i], false);
        s->lightCount++;
    }
    while (s->lightCount > want)
    {
        uint32_t i = --s->lightCount;
        renderer::light_remove(s->lights[i]);
        renderer::instance_remove(s->markers[i]);
    }
    uint32_t spots = settings.spot_count < 0 ? 0 : (uint32_t)settings.spot_count;
    spots = spots > kMaxSpots ? kMaxSpots : spots;
    while (s->spotCount < spots)
    {
        uint32_t   i = s->spotCount++;
        yea_handle h = renderer::light_add(2);
        renderer::light_cone(h, 14, 24);
        renderer::light_shadow(h, true);
        s->spots[i] = h;
        // A lamp: a short dark body behind the light, aimed along it, and a glowing lens disc at its front. Neither casts
        // a shadow (it would shadow its own light), as the point lights' orbs don't.
        s->spotMarkerMaterials[i] = renderer::material_add({ 0, 0, 0 }, 1, 0);
        s->spotMarkers[i] = renderer::instance_add(s->cylinderMesh, s->spotMarkerMaterials[i]);
        renderer::instance_shadow(s->spotMarkers[i], false);
        s->spotHousings[i] = renderer::instance_add(s->cylinderMesh, s->housingMaterial);
        renderer::instance_shadow(s->spotHousings[i], false);
    }
    while (s->spotCount > spots)
    {
        uint32_t i = --s->spotCount;
        renderer::light_remove(s->spots[i]);
        renderer::instance_remove(s->spotMarkers[i]);
        renderer::instance_remove(s->spotHousings[i]);
    }
}

void animate(State* s)
{
    float t = s->t;
    float el = settings.sun_elevation * kPi / 180, az = settings.sun_azimuth * kPi / 180;
    renderer::light_place(s->sun, { 0, 0, 0 }, { -cosf(el) * cosf(az), -sinf(el), -cosf(el) * sinf(az) });
    renderer::light_color(s->sun, settings.sun_color, settings.sun_intensity);

    uint32_t shadowed = settings.shadowed_points < 0 ? 0 : (uint32_t)settings.shadowed_points;
    for (uint32_t i = 0; i < s->lightCount; ++i)
    {
        float      fi = (float)i;
        float      radius = 2.5f + 8.0f * frac(fi * 0.618034f);
        float      speed = settings.light_speed * (0.4f + frac(fi * 0.371f)) * ((i & 1) ? 1.0f : -1.0f) * 3.0f / radius;
        float      a = fi * 2.39996f + t * speed;
        float      y = 0.8f + 2.0f * frac(fi * 0.7311f) + 0.25f * sinf(t * 0.9f + fi);
        yea_float3 p = { cosf(a) * radius, y, sinf(a) * radius };
        yea_float3 c = hue(frac(fi * 0.618034f + 0.05f));
        renderer::light_place(s->lights[i], p, { 0, -1, 0 });
        renderer::light_color(s->lights[i], c, settings.light_intensity);
        renderer::light_range(s->lights[i], settings.light_range);
        renderer::light_shadow(s->lights[i], i < shadowed);
        renderer::material_emissive(s->markerMaterials[i], c, settings.light_intensity * 0.6f + 1.0f);
        renderer::instance_place(s->markers[i], p, kNoRotation, { 0.12f, 0.12f, 0.12f });
    }
    for (uint32_t i = 0; i < s->spotCount; ++i)
    {
        float      a = (float)i / (float)(s->spotCount ? s->spotCount : 1) * 2 * kPi;
        yea_float3 p = { cosf(a) * 7.0f, 6.5f, sinf(a) * 7.0f };
        float      b = a + t * 0.5f * settings.light_speed / 0.4f;
        yea_float3 target = { cosf(b) * 3.0f, 0, sinf(b) * 3.0f };
        yea_float3 c = (i & 1) ? yea_float3{ 0.7f, 0.85f, 1.0f } : yea_float3{ 1.0f, 0.85f, 0.6f };
        renderer::light_place(s->spots[i], p, { target.x - p.x, target.y - p.y, target.z - p.z });
        renderer::light_color(s->spots[i], c, settings.spot_intensity);
        renderer::material_emissive(s->spotMarkerMaterials[i], c, settings.spot_intensity * 0.6f + 1.0f);
        // The lamp turned so its axis (the cylinder's y) points along the light: the body 0.3 m long behind the light,
        // the lens 0.2 m across just in front of it.
        yea_float3 d = { target.x - p.x, target.y - p.y, target.z - p.z };
        float      dl = sqrtf(d.x * d.x + d.y * d.y + d.z * d.z);
        d = { d.x / dl, d.y / dl, d.z / dl };
        yea_float4 q = axisAngle({ d.z, 0, -d.x }, acosf(d.y < -1 ? -1 : (d.y > 1 ? 1 : d.y))); // +y onto d
        renderer::instance_place(s->spotHousings[i], { p.x - d.x * 0.16f, p.y - d.y * 0.16f, p.z - d.z * 0.16f }, q, { 0.26f, 0.3f, 0.26f });
        renderer::instance_place(s->spotMarkers[i], { p.x + d.x * 0.0f, p.y + d.y * 0.0f, p.z + d.z * 0.0f }, q, { 0.2f, 0.02f, 0.2f });
        renderer::light_range(s->spots[i], 20); // reaches the far edge of the sweep (11.9 m) at 77% of the inverse square
    }
    for (uint32_t i = 0; i < kAreas; ++i)
    {
        // Each emitter's luminance (the pendant's its own): each light's candela is it times the emitter's area.
        const AreaLight& a = kAreaLights[i];
        float            lum = a.type == 4 ? settings.pendant_luminance : settings.area_luminance;
        renderer::light_color(s->areas[i], a.color, lum * areaOf(a));
        renderer::material_emissive(s->areaMaterials[i], a.color, lum);
    }
    for (uint32_t i = 0; i < s->spinCount; ++i)
        renderer::instance_place(s->spin[i], s->spinPos[i], axisAngle(s->spinAxis[i], t * 0.6f + (float)i), s->spinScale[i]);
}

void moveCamera(State* s, float dt)
{
    // Look: yea_mouse_dx/dy (pixels this frame, up positive) while the host holds the mouse for us, 0.0025
    // rad per pixel at sensitivity 1; else The Forge's look_x/look_y (F/H/T/G, the right stick; per second).
    // move_x A/D, move_y S/W, move_up Q/E or the left stick. Turning right lowers yaw; looking up raises pitch.
    const YeaHost* host = yea::host();
    bool           down = yea::held(actions::look);
    // A press on the game starts it (with F1 open too); the host lets go when we stop asking.
    gLooking = down && (gLooking || yea::pressed(actions::look)) && yea::captureMouse(true);
    if (gLooking)
    {
        s->yaw -= host->input("yea_mouse_dx") * 0.0025f * settings.look_sensitivity;
        s->pitch += host->input("yea_mouse_dy") * 0.0025f * settings.look_sensitivity;
    }
    else if (!down) // the mouse part of look_x/look_y is only there while the button is held
    {
        float look = settings.look_sensitivity * dt;
        s->yaw -= host->input("look_x") * look;
        s->pitch += host->input("look_y") * look;
    }
    // Just short of straight down or up (89.5 degrees): the view's heading comes from yaw alone, so it never flips.
    const float kMaxPitch = 1.5621f;
    s->pitch = s->pitch < -kMaxPitch ? -kMaxPitch : (s->pitch > kMaxPitch ? kMaxPitch : s->pitch);

    yea_float3 f = { cosf(s->yaw) * cosf(s->pitch), sinf(s->pitch), sinf(s->yaw) * cosf(s->pitch) };
    yea_float3 r = { sinf(s->yaw), 0, -cosf(s->yaw) }; // up x forward: the camera's right
    float      step = settings.move_speed * dt, mx = host->input("move_x"), my = host->input("move_y");
    float      mu = host->input("move_up");
    s->eye.x += (r.x * mx + f.x * my) * step;
    s->eye.y += (f.y * my + mu) * step;
    s->eye.z += (r.z * mx + f.z * my) * step;

    if (settings.benchmark_view)
    {
        // The start view, or a close-up, at a fixed field of view; the user's own camera is kept for after.
        if (settings.benchmark_camera == BenchmarkCamera::Gold_Closeup)
            renderer::camera({ -4.2f, 1.4f, -1.2f }, { -5.0f, 0.75f, -4.0f }, 40);
        else if (settings.benchmark_camera == BenchmarkCamera::Mirror_Closeup)
            renderer::camera({ -7.0f, 1.0f, -2.6f }, { -7.0f, 0.75f, -4.0f }, 40);
        else
        {
            const float yaw = kStartYaw, pitch = kStartPitch, dist = kStartDist;
            yea_float3  eye = { cosf(yaw) * cosf(pitch) * dist, 0.8f + sinf(pitch) * dist, sinf(yaw) * cosf(pitch) * dist };
            renderer::camera(eye, { 0, 0.8f, 0 }, 55);
        }
        return;
    }
    if (gPicture)
    {
        // A fixed angle for the project card.
        const float yaw = 0.75f, pitch = 0.38f, dist = 15.0f;
        yea_float3  eye = { cosf(yaw) * cosf(pitch) * dist, 0.8f + sinf(pitch) * dist, sinf(yaw) * cosf(pitch) * dist };
        renderer::camera(eye, { 0, 0.8f, 0 }, settings.fov);
        return;
    }
    renderer::camera(s->eye, { s->eye.x + f.x, s->eye.y + f.y, s->eye.z + f.z }, settings.fov);
}
} // namespace

bool picture(bool on)
{
    gPicture = on;
    return true;
}

int load(int reloaded)
{
    gState = nullptr;
    gFreshVersion = true;
    gLooking = false;
    gPicture = false;
    return 0;
}

void unload() { yea::captureMouse(false); }

void update(uint32_t phase, float dt)
{
    if (phase != YEA_PHASE_SIMULATE)
        return;
    State* s = state();
    // The renderer's scene must be the one this module built: after a renderer crash, revert or reload that lost or
    // replaced it (or one that came back while this module had built against stubs), every handle held here is stale
    // and nothing would move, so build it all again.
    if (!s->built || gFreshVersion || renderer::instance_count() != s->instances || renderer::light_count() != s->lightsAlive)
    {
        char line[160];
        snprintf(line, sizeof line, "renderer_example: building the scene (built %u; renderer has %u instances, %u lights; expected %u, %u)",
                 s->built, renderer::instance_count(), renderer::light_count(), s->instances, s->lightsAlive);
        build(s);
        if (s->built)
        {
            gFreshVersion = false;
            yea::log(line);
        }
    }
    if (!s->built)
        return; // the renderer isn't loaded
    if (renderer_probes::probe_count() == 0)
        addProbes(); // renderer_probes came up after the scene was built: its import was a stub then
    if (!settings.benchmark_view)
        s->t += dt; // Benchmark View holds the lights and shapes still, so timings repeat
    syncLights(s);
    animate(s);
    moveCamera(s, dt);
    s->instances = renderer::instance_count();
    s->lightsAlive = renderer::light_count();
}

void draw(uint32_t pass, const YeaFrame* frame) {}
} // namespace renderer_example
