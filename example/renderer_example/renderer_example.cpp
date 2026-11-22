// The Renderer component's example view: a floor, a metal and a plastic sphere row sweeping
// roughness, assorted shapes, a sun, many moving coloured point lights, sweeping spot lights, two
// reflection probes, and a free-look camera (hold the mouse button to look, the move keys to fly).
// Reset: the scene is rebuilt with the lights and camera where they start.
#include "yea_renderer_example.h"

#include <math.h>

namespace renderer_example
{
namespace
{
const uint32_t kMaxLights = 512, kMaxSpots = 16, kMaxSpin = 16;
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
};
State* gState;

// Mouse look: the host holds the mouse (hidden, kept in the window) while the left button stays down
// after a press on the game; it lets go itself when F1 opens, focus is lost, the view switches or resets.
bool   gLooking;
bool   gPicture; // Yea is taking the card picture; a global so a reload never leaves it stuck on

// Where the camera starts: 17 m from (0, 0.8, 0) at yaw 0.6 and 0.42 rad up, looking at it.
void startCamera(State* s)
{
    const float yaw = 0.6f, pitch = 0.42f, dist = 17.0f;
    s->eye = { cosf(yaw) * cosf(pitch) * dist, 0.8f + sinf(pitch) * dist, sinf(yaw) * cosf(pitch) * dist };
    s->yaw = yaw + kPi;
    s->pitch = -pitch;
}

State* state()
{
    if (!gState)
    {
        int fresh = 0;
        gState = (State*)yea::host()->state("renderer_example", 3, sizeof(State), &fresh);
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

void build(State* s)
{
    renderer::clear();
    renderer_probes::probe_clear();
    s->lightCount = s->spotCount = s->spinCount = 0;
    uint32_t sphere = renderer::mesh_sphere(48), box = renderer::mesh_box(), plane = renderer::mesh_plane();
    uint32_t cylinder = renderer::mesh_cylinder(40), cone = renderer::mesh_cone(40), torus = renderer::mesh_torus(0.15f, 64);
    s->sphereMesh = sphere;

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
    yea_handle glow = renderer::material_add({ 0.05f, 0.05f, 0.05f }, 0.5f, 0);
    renderer::material_emissive(glow, { 0.4f, 0.8f, 1.0f }, 40);

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
    place(sphere, glow, { 0.5f, 1.2f, 2.2f }, { 0.6f, 0.6f, 0.6f });
    place(box, stone, { 0.0f, 0.25f, -7.5f }, { 18.0f, 0.5f, 1.5f });
    addSpin(s, torus, steel, { 4.0f, 1.6f, 5.5f }, { 2.6f, 2.6f, 2.6f }, { 1, 0.3f, 0 });
    addSpin(s, torus, copper, { -4.5f, 1.4f, 5.5f }, { 2.4f, 2.4f, 2.4f }, { 0, 1, 0.5f });
    addSpin(s, box, blue, { 8.5f, 1.2f, -1.0f }, { 1.2f, 1.2f, 1.2f }, { 1, 1, 0 });
    addSpin(s, torus, white, { -8.5f, 1.3f, -1.0f }, { 2.2f, 2.2f, 2.2f }, { 0, 0, 1 });

    addProbes();

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
        yea_handle h = renderer::light_add(2);
        renderer::light_cone(h, 14, 24);
        renderer::light_shadow(h, true);
        s->spots[s->spotCount++] = h;
    }
    while (s->spotCount > spots)
        renderer::light_remove(s->spots[--s->spotCount]);
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
        renderer::light_place(s->spots[i], p, { target.x - p.x, target.y - p.y, target.z - p.z });
        renderer::light_color(s->spots[i], (i & 1) ? yea_float3{ 0.7f, 0.85f, 1.0f } : yea_float3{ 1.0f, 0.85f, 0.6f },
                              settings.spot_intensity);
        renderer::light_range(s->spots[i], 16);
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
    s->pitch = s->pitch < -1.55f ? -1.55f : (s->pitch > 1.55f ? 1.55f : s->pitch);

    yea_float3 f = { cosf(s->yaw) * cosf(s->pitch), sinf(s->pitch), sinf(s->yaw) * cosf(s->pitch) };
    yea_float3 r = { sinf(s->yaw), 0, -cosf(s->yaw) }; // up x forward: the camera's right
    float      step = settings.move_speed * dt, mx = host->input("move_x"), my = host->input("move_y");
    float      mu = host->input("move_up");
    s->eye.x += (r.x * mx + f.x * my) * step;
    s->eye.y += (f.y * my + mu) * step;
    s->eye.z += (r.z * mx + f.z * my) * step;

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
    if (!s->built || renderer::instance_count() == 0)
        build(s);
    else if (renderer_probes::probe_count() == 0)
        addProbes(); // renderer_probes came up after the scene was built: its import was a stub then
    s->t += dt;
    syncLights(s);
    animate(s);
    moveCamera(s, dt);
}

void draw(uint32_t pass, const YeaFrame* frame) {}
} // namespace renderer_example
