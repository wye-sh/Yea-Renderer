# Lighting (component)

3D renderer for a view: forward+ PBR with a depth prepass, compute light culling, sun cascades and a
local-light shadow atlas, contact shadows, GTAO, reflection probes, TAA, auto exposure and tonemapping.
It records the whole 3D frame in the host's `YEA_PASS_RENDER` (nothing bound; its own targets and depth)
and composites it in `YEA_PASS_WORLD` with a `YEA_PIPELINE_PICTURE` pipeline, under the host's world quads.

## Modules

- `lighting`: the core. Scene tables and exports, camera, depth prepass, forward shading, sky, exposure,
  tonemap, composite. Code: `lighting.cpp` (frame), `scene.cpp` (exports), `Shaders/`.
- `lighting_culling`: Z-bins (CPU, lights sorted by view depth) and per-tile light bitmasks (compute,
  tile depth range from the prepass). Switch off: every pixel loops all visible lights (capped at 256).
- `lighting_shadows`: sun cascades (stabilised, PCF or PCSS) and a cached 16 x 16 tile atlas for spot and
  point lights (6 tiles per point light), redrawn only when the light or a nearby caster moves.
- `lighting_taa`: temporal antialiasing. Picks a Halton sub-pixel jitter each frame; the core then renders
  at one sample (TAA replaces `msaa`) with a motion vector target (RG16F, from last frame's instance rows and
  camera); the resolve reprojects two ping-pong RGBA16F histories (Catmull-Rom), clips them to the 3x3
  neighbourhood's spread in YCoCg, blends. Sharpening is in the core's composite, after TAA, never in history.
  Switch off: the core renders with `msaa` as before.
- `lighting_ao`: GTAO (Jimenez 2016) from the prepass depth alone (normals rebuilt from depth), then
  `denoise` depth-aware 3x3 blurs, into a full-size R32F texture. The forward pass darkens sky ambient by it:
  diffuse with the multi-bounce fit, specular with Lagarde's specular occlusion. Direct light is untouched.
  Runs between the prepass and the forward pass, beside culling. Noise rotates per frame only with TAA on.
- `lighting_contact`: sun contact shadows. A screen-space march from each pixel towards the sun over
  `length` metres against the prepass depth, into a full-size R32F texture the forward pass multiplies the
  sun's shadow by. Catches small and thin shadows the cascades miss. Off when there is no sun.
- `lighting_probes`: reflection probes games place (its own `Probe` table, up to 16). Each is captured as a
  cube (the scene lit diffusely: sky ambient, sun without shadow, the nearest 64 local lights; alpha =
  coverage), halved level by level, prefiltered (GGX) into one slice of an RGBA16F octahedral 2D array, a mip
  per roughness step. Dirty probes are captured first (at most `updates_per_frame`, or 1), then that many more
  in turn. The forward pass blends the probes whose boxes hold the pixel, smallest box first, projects onto
  the box (parallax), and lets the live sky show where the capture saw none.
- The core calls the others through optional imports with a `LightingFrame*` (as `u64`): `frame.h`, copied
  identically into `lighting`, `lighting_culling`, `lighting_shadows`, `lighting_taa`, `lighting_ao`, `lighting_contact` and `lighting_probes` (`mathx.h` into the
  first three). Change all copies together and raise `LIGHTING_FRAME_VERSION`. `taa_prepare` runs first,
  before the camera, so the jitter reaches the projection.

## Hook-up points (exports of `lighting`)

Units: metres; sun intensity in lux, point and spot in candela. Colours linear RGB.
- Meshes: `mesh_box/sphere/plane/cylinder/cone/torus` (unit size, centred), or `mesh_begin`,
  `mesh_vertex(pos, normal)` (returns a local index), `mesh_triangle`, `mesh_end` → mesh id. Meshes are
  uploaded when they change (a stall: build them up front).
- Materials: `material_add(albedo, roughness, metallic)`, `material_set`, `material_emissive(m, color, nits)`,
  `material_reflections(m, on)`: off, the material shows no reflected image (probes or sky), only the sky's
  soft light, however shiny or metallic it is.
- Instances: `instance_add(mesh, material)`, `instance_place(i, pos, quat, scale)`, `instance_material`,
  `instance_shadow(i, cast)`, `instance_remove`. Only call `instance_place` when it moved: a move redraws
  local shadows near it.
- Lights: `light_add(type)` (0 sun, 1 point, 2 spot), `light_place(l, pos, dir)`, `light_color(l, rgb,
  intensity)`, `light_range`, `light_cone(inner°, outer°)`, `light_shadow(l, cast)`, `light_remove`. The
  first sun gets the cascades; up to 4 suns.
- `camera(eye, target, fov°)`, `clear()` (removes everything but probes).
- Reflection probes (exports of `lighting_probes`): `probe_add(pos, box_min, box_max)` (capture point and
  the world box where it is used and reflections are projected onto; put the box's bottom just under the
  floor), `probe_place`, `probe_remove`, `probe_clear`, `probe_count`.
- Handles are `row | generation << 20`; stale handles are ignored.

Minimal link module: `use lighting`, a `preset` for every knob, and in `update(SIMULATE)` add meshes,
materials, instances and lights once (when `instance_count()` is 0), then move what moves and set the camera.

## Knobs

`lighting`: exposure_ev, auto_exposure, manual_ev100, adapt_speed, tonemapper (0 PBR Neutral, 1 filmic,
2 Reinhard, 3 none), msaa (1/2/4/8), specular_aa, sky/horizon/ground colours, sky_luminance, ambient,
sun_disk (degrees), near_plane. `lighting_culling`: enabled, max_lights, tile_size (8/16/32), z_bins,
heatmap. `lighting_shadows`: enabled, cascades, shadow_distance, cascade_split, sun_resolution, softness,
filter_taps, pcss, sun_size, normal_bias, slope_bias, shadowed_lights, local_atlas_resolution,
updates_per_frame, show_cascades, show_atlas (the light shadow atlas, top right). `lighting_taa`: enabled, history_weight, clamp_width, sharpen, jitter,
jitter_phases. `lighting_ao`: enabled, radius, intensity, falloff, slices, steps, denoise, specular_occlusion,
multi_bounce, show_ao. `lighting_contact`: enabled, length, thickness, steps, intensity, max_distance,
show_contact. `lighting_probes`: enabled, resolution (64-512), updates_per_frame, intensity, blend, parallax.

## Reset

All scene data is in `lighting`'s tables and state: Reset empties the scene; the feature's link module
builds it again. GPU objects and shadow caches are rebuilt on load.

## Cost

Per frame: one depth prepass and one forward pass (draws grouped by mesh, instanced), one culling
dispatch per 16 px tile, a GTAO dispatch and `denoise` blur dispatches, one contact shadow dispatch (all full size), per re-captured probe 6 scene draws at half its
resolution plus 6 small passes per mip, sun cascades redrawn, at most `updates_per_frame` local lights redrawn (oldest shadow first), two
exposure dispatches, one full-screen TAA resolve (when on), one full-screen composite. No per-frame allocation. GPU memory at the defaults:
HDR + depth at 4x MSAA, sun atlas 4096² D32 (64 MB), local atlas 4096² (64 MB).

## Pitfalls

- F1's Wireframe and Overdraw (`yea::viewMode()`, development only): TAA, AO, contact shadows and light culling
  step aside; the core skips the prepass (the forward pass writes depth) and the sky, clears HDR to black,
  uses exposure 1 without tonemapping or dither (no sRGB encode in Overdraw) and holds auto exposure;
  shadow maps and probes are not redrawn while a mode is on, and every light's shadow is redrawn after.
- A light's shadow is drawn from where it was when its tiles were last redrawn, which lags a moving light
  when more lights move than `updates_per_frame`. Shading must use that drawn position (point lights pick
  their cube face from it), never the light's current one, or points project off the tile's edge.
- 2D world quads the host draws after this view are over the 3D image; there is no depth between them.
- Instances draw with an instance-rate stream of indices, never `SV_InstanceID` (D3D12/Vulkan rule).
- Shadows use standard Z (near 0); the camera uses reversed-Z with an infinite far plane.
- With TAA on, the prepass uses the motion vertex shader too, so both passes compute the same depth.
- Motion vectors match an instance by its table row and generation; a reused row starts with no motion.
- Take state blocks (`host()->state`) only in your own `load()`: the host files a block under the module
  running when it is asked for, so one asked for inside an export another module calls is that module's
  (the camera once never reached the renderer this way). `lighting` binds its blocks in `bindState()`.
- A probe sees the objects inside its box too, so a shiny object reflects itself where it is near the probe.

## Example view (`example/lighting_example`)

Floor, gold and red sphere rows sweeping roughness, pillars and assorted shapes, spinning tori, a sun,
`light_count` moving coloured point lights (markers glow in their colour; the first `shadowed_points` cast
shadows), `spot_count` sweeping shadowed spots, two reflection probes (the sphere rows, the whole yard), a free-look camera: holding its `look` action (left mouse button) after a press on the game
asks the host for the mouse each frame (`yea::captureMouse(true)`: hidden, kept in the window) and turns by
`yea_mouse_dx`/`yea_mouse_dy` (`look_sensitivity`); the move keys fly (WASD, Q/E down/up, `move_speed`).
The host lets go when the button is up (it is no longer asked), F1 opens, focus is lost, or on a view switch
or Reset. Reset: lights and camera
back to the start. Exports `picture` (a fixed angle).
