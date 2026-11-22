# Renderer (component)

3D renderer for a view: forward+ PBR with a depth prepass, compute light culling, sun cascades and a
local-light shadow atlas, contact shadows, GTAO, reflection probes, TAA, auto exposure and tonemapping.
It records the whole 3D frame in the host's `YEA_PASS_RENDER` (nothing bound; its own targets and depth)
and composites it in `YEA_PASS_WORLD` with a `YEA_PIPELINE_PICTURE` pipeline, under the host's world quads.

## Modules

- `renderer`: the core. Scene tables and exports, camera, depth prepass, forward shading, sky, exposure,
  tonemap, composite. Code: `renderer.cpp` (frame), `scene.cpp` (exports), `Shaders/`.
- `renderer_culling`: Z-bins (CPU, lights sorted by view depth) and per-tile light bitmasks (compute,
  tile depth range from the prepass). Switch off: every pixel loops all visible lights (capped at 256).
- `renderer_shadows`: sun cascades (stabilised, PCF or PCSS) and a cached atlas for spot and point lights
  (6 tiles per point light), redrawn only when the light or a nearby caster moves. Tiles are squares of
  128 to max_tile_resolution px, sized to the light's reach on screen and halved, largest first, until
  all fit 90% of the atlas; a light changes size only when its wanted size doubles or halves.
- `renderer_taa`: temporal antialiasing. Picks a Halton sub-pixel jitter each frame; the core then renders
  at one sample (TAA replaces `msaa`) with a motion vector target (RG16F, from last frame's instance rows and
  camera); the resolve reprojects two ping-pong RGBA16F histories (Catmull-Rom), clips them to the 3x3
  neighbourhood's spread in YCoCg, blends. Sharpening is in the core's composite, after TAA, never in history.
  Switch off: the core renders with `msaa` as before.
- `renderer_ao`: GTAO (Jimenez 2016) from the prepass depth alone (normals rebuilt from depth), then
  `denoise` depth-aware 3x3 blurs, into a full-size R32F texture. The forward pass darkens sky ambient by it:
  diffuse with the multi-bounce fit, specular with Lagarde's specular occlusion. Direct light is untouched.
  Runs between the prepass and the forward pass, beside culling. Noise rotates per frame only with TAA on.
- `renderer_contact`: sun contact shadows. A screen-space march from each pixel towards the sun over
  `length` metres against the prepass depth, into a full-size R32F texture the forward pass multiplies the
  sun's shadow by. Catches small and thin shadows the cascades miss. Off when there is no sun.
- `renderer_probes`: reflection probes games place (its own `Probe` table, up to 16). Each is captured as a
  cube (the scene lit diffusely: sky ambient, sun without shadow, the nearest 64 local lights; alpha =
  coverage), halved level by level, prefiltered (GGX) into one slice of an RGBA16F octahedral 2D array, a mip
  per roughness step. Dirty probes are captured first (at most `updates_per_frame`, or 1), then that many more
  in turn. The forward pass blends the probes whose boxes hold the pixel, smallest box first, projects onto
  the box (parallax), and lets the live sky show where the capture saw none.
- `renderer_ssr`: screen-space reflections over the probes. It copies the prepass depth (full or half size);
  the core's forward pass marches each glossy pixel's reflected ray across the screen against that copy
  (1/depth linear in screen space, thickness test, 4 halvings), and takes the colour from last frame's
  TAA-resolved image (`previous`), reprojected. Where nothing is found, off screen, near the screen's edges
  or above `max_roughness`, the probes show. Needs TAA: without it there is no previous image and it is off.
  Only on screen: what is behind the camera or hidden keeps the probes' (box-projected) reflection.
- `renderer_vrs`: variable-rate shading of the forward pass, through Yea's shading-rate service
  (`yea::shadingRate()`, nullptr on GPUs without it: then the module does nothing). After the prepass, one
  compute group per rate tile (16 px on an RTX 2080) measures last frame's TAA-resolved image as seen
  (exposed and compressed): the RMS luminance step between horizontal and between vertical neighbours.
  Below `aggressiveness`' threshold, raised by motion, that axis goes coarse (2x1, 1x2, 2x2; 4x4 with
  allow_4x4). A crease or edge in this frame's depth keeps the tile at 1x1. Writes Yea's R8_UINT rate image
  and an R32_UINT copy (Show Rates); the core binds the forward pass with the service's `bindRenderTargets`.
  Needs TAA and one sample; off in F1's view modes. Vulkan caveat: FSL declares `RWTex2D(uint)` as `r32ui`,
  so the R8_UINT image is written through a mismatched format qualifier (fine on D3D12; unverified on Vulkan
  VRS hardware).
- The core calls the others through optional imports with a `RenderFrame*` (as `u64`): `frame.h`, copied
  identically into `renderer`, `renderer_culling`, `renderer_shadows`, `renderer_taa`, `renderer_ao`, `renderer_contact`, `renderer_probes`, `renderer_ssr` and `renderer_vrs` (`mathx.h` into the
  first three). Change all copies together and raise `RENDER_FRAME_VERSION`. `taa_prepare` runs first,
  before the camera, so the jitter reaches the projection.

## Hook-up points (exports of `renderer`)

Units: metres; sun intensity in lux, point and spot in candela. Colours linear RGB.
- Meshes: `mesh_box/sphere/plane/cylinder/cone/torus` (unit size, centred), or `mesh_begin`,
  `mesh_vertex(pos, normal)` (returns a local index), `mesh_triangle`, `mesh_end` → mesh id. Meshes are
  uploaded when they change (a stall: build them up front). A triangle faces where cross(b - a, c - a)
  points (counter-clockwise around its outward normal); back faces are culled.
- Materials: `material_add(albedo, roughness, metallic)`, `material_set`, `material_emissive(m, color, nits)`,
  `material_reflections(m, on)`: off, the material shows no reflected image (probes or sky), only the sky's
  soft light, however shiny or metallic it is. `material_double_sided(m, on)` (off by default): drawn from
  both sides, for open or thin surfaces (a lone plane, leaves, cloth).
- Instances: `instance_add(mesh, material)`, `instance_place(i, pos, quat, scale)`, `instance_material`,
  `instance_shadow(i, cast)`, `instance_remove`. Only call `instance_place` when it moved: a move redraws
  local shadows near it.
- Lights: `light_add(type)` (0 sun, 1 point, 2 spot), `light_place(l, pos, dir)`, `light_color(l, rgb,
  intensity)`, `light_range`, `light_cone(inner°, outer°)`, `light_shadow(l, cast)`, `light_remove`. The
  first sun gets the cascades; up to 4 suns.
- `camera(eye, target, fov°)`, `clear()` (removes everything but probes).
- Reflection probes (exports of `renderer_probes`): `probe_add(pos, box_min, box_max)` (capture point and
  the world box where it is used and reflections are projected onto; put the box's bottom just under the
  floor), `probe_place`, `probe_remove`, `probe_clear`, `probe_count`.
- Handles are `row | generation << 20`; stale handles are ignored.

Minimal link module: `use renderer`, a `preset` for every knob, and in `update(SIMULATE)` add meshes,
materials, instances and lights once (when `instance_count()` is 0), then move what moves and set the camera.

## Knobs

`renderer`: exposure_ev, auto_exposure, manual_ev100, adapt_speed, tonemapper (0 PBR Neutral, 1 filmic,
2 Reinhard, 3 none), msaa (1/2/4/8), specular_aa, sky/horizon/ground colours, sky_luminance, ambient,
sun_disk (degrees), near_plane, gpu_timings (logs each pass's GPU time every 10 s: avg/max ms). `renderer_culling`: enabled, max_lights, tile_size (8/16/32), z_bins,
heatmap. `renderer_shadows`: enabled (the sun's), cascades, shadow_distance, cascade_split, sun_resolution, softness,
filter_taps, pcss, sun_size, normal_bias, slope_bias, light_shadows, shadowed_lights, local_atlas_resolution,
updates_per_frame, max_tile_resolution, show_cascades, show_atlas (the light shadow atlas, top right). `renderer_taa`: enabled, history_weight, clamp_width, sharpen, jitter,
jitter_phases. `renderer_ao`: enabled, half_resolution (GTAO and its blur at half size, then a depth-aware upsample), radius, intensity, falloff, slices, steps, denoise, specular_occlusion,
multi_bounce, show_ao. `renderer_contact`: enabled, length, thickness, steps, intensity, max_distance,
show_contact. `renderer_probes`: enabled, resolution (64-512), updates_per_frame, intensity, blend, parallax.
`renderer_ssr`: enabled, half_resolution, max_roughness, thickness, max_distance, steps, edge_fade, intensity,
show_hits (tints where reflections were found). `renderer_vrs`: enabled, aggressiveness, motion, allow_4x4,
show_rates (green half rate, blue 2x2, magenta coarser).

## Reset

All scene data is in `renderer`'s tables and state: Reset empties the scene; the feature's link module
builds it again. GPU objects and shadow caches are rebuilt on load.

## Cost

Per frame: one depth prepass and one forward pass (draws grouped by mesh and sidedness, instanced; back
faces culled except for double-sided materials; a leaner forward shader unless PCSS or a debug view is on), one culling
dispatch per 16 px tile, a GTAO dispatch and `denoise` blur dispatches (at half size with half_resolution, plus a full-size upsample), one contact shadow dispatch (all full size; its march takes no more steps than the ray spans pixels), one
rate dispatch per 16 px tile with variable-rate shading (the forward pass then shades smooth tiles at 2x2), one
depth copy for screen reflections (their march, up to `steps` plus 4 depth reads, runs in the forward pass for
glossy pixels with reflections on, and shows in its timing), per re-captured probe 6 scene draws at half its
resolution plus 6 small passes per mip, sun cascades redrawn, at most `updates_per_frame` local lights redrawn (oldest shadow first), two
exposure dispatches, one full-screen TAA resolve (when on), one full-screen composite. No per-frame allocation. GPU memory at the defaults:
HDR + depth at 4x MSAA, sun atlas 4096² D32 (64 MB), local atlas 4096² (64 MB), a screen-size R32F depth copy
for screen reflections (a quarter of it at half resolution).

## Pitfalls

- Back faces are culled in the prepass, forward pass and probe captures; the camera space is left-handed,
  so front faces are clockwise on screen (`FRONT_FACE_CW`). Shadow maps cull nothing: a single-sided
  surface must cast from both sides (a ceiling plane lit from above), and the biases were tuned so.
- Changing the Material table's layout empties it on reload; `load()` then clears the scene so its
  owner builds it again rather than drawing instances whose materials are gone.
- Debug views must not feed auto exposure: Show AO and Show Contact write raw values shown at exposure 1, and
  exposure holds while they or the heatmap are on. A view scaled by 1/exposure is metered by the histogram,
  which lowers exposure every frame until the HDR image saturates (Show AO once read flat grey this way).
- F1's Wireframe and Overdraw (`yea::viewMode()`, development only): TAA, AO, contact shadows and light culling
  step aside; the core skips the prepass (the forward pass writes depth) and the sky, clears HDR to black,
  uses exposure 1 without tonemapping or dither (no sRGB encode in Overdraw) and holds auto exposure;
  shadow maps and probes are not redrawn while a mode is on, and every light's shadow is redrawn after.
  Wireframe draws each line in its material's albedo plus emissive, unlit.
- Contact shadows refine the sun's shadow maps and are off whenever those are. They march from two pixels
  off the surface (normal rebuilt from depth): from the surface itself, silhouettes darken in thin lines.
- A module that fills the scene can update before the others have loaded, when its imports are stubs
  that do nothing; the scene lives in tables, so it is not built again. Check counts each frame and
  add what is missing (the example adds its probes whenever `probe_count()` is 0).
- A light's shadow is drawn from where it was when its tiles were last redrawn, which lags a moving light
  when more lights move than `updates_per_frame`. Shading must use that drawn position (point lights pick
  their cube face from it, kept in the first face's `info.yzw`), never the light's current one, or points project off the tile's edge.
- 2D world quads the host draws after this view are over the 3D image; there is no depth between them.
- Instances draw with an instance-rate stream of indices, never `SV_InstanceID` (D3D12/Vulkan rule).
- Shadows use standard Z (near 0); the camera uses reversed-Z with an infinite far plane.
- With TAA on, the prepass uses the motion vertex shader too, so both passes compute the same depth.
- Motion vectors match an instance by its table row and generation; a reused row starts with no motion.
- Take state blocks (`host()->state`) only in your own `load()`: the host files a block under the module
  running when it is asked for, so one asked for inside an export another module calls is that module's
  (the camera once never reached the renderer this way). `renderer` binds its blocks in `bindState()`.
- A probe sees the objects inside its box too, so a shiny object reflects itself where it is near the probe.

## Example view (`example/renderer_example`)

Floor, gold and red sphere rows sweeping roughness, pillars and assorted shapes, spinning tori, a sun,
`light_count` moving coloured point lights (markers glow in their colour; the first `shadowed_points` cast
shadows), `spot_count` sweeping shadowed spots, two reflection probes (the sphere rows, the whole yard), a free-look camera: holding its `look` action (left mouse button) after a press on the game
asks the host for the mouse each frame (`yea::captureMouse(true)`: hidden, kept in the window) and turns by
`yea_mouse_dx`/`yea_mouse_dy` (`look_sensitivity`); the move keys fly (WASD, Q/E down/up, `move_speed`).
The host lets go when the button is up (it is no longer asked), F1 opens, focus is lost, or on a view switch
or Reset. Reset: lights and camera
back to the start. `benchmark_view` holds the camera at its start view (field of view 55) without moving the
user's own, so GPU Timings compare between versions. Exports `picture` (a fixed angle).
