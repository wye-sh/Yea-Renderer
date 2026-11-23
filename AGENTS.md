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
- `renderer_shadows`: sun cascades (stabilised, PCF or PCSS; each redrawn only when its view or a caster over it
  changed) and a cached atlas for spot and point lights
  (6 tiles per point light), redrawn only when the light or a nearby caster moves. Tiles are squares of
  128 to max_tile_resolution px, sized to the light's reach on screen and halved, largest first, until
  all fit 90% of the atlas; a light changes size only when its wanted size doubles or halves.
  Sampling (in the core's Lighting.h.fsl): every filter tap compares against the receiver's own depth
  there (receiver-plane bias: the surface's tangent plane carried into the view's ndc by the first three
  columns of its inverse, `inv` in the shadow view, three dot products; no screen derivatives, as lights are
  shaded in loops), so the normal offset can stay small. Point and spot lights harden at contact (PCSS: 4 gathers on
  the search disk's rim first, 16 depths that decide fully lit or umbra; only penumbrae gather 4 more, the 32
  depths averaged raw and turned into the blockers' distance with one divide, giving a penumbra from `light_size`;
  keep divides out of the rim, which every pixel runs). With TAA, cascade transitions
  pick one cascade per pixel (dithered) and the filters' noise turns each frame.
- `renderer_taa`: temporal antialiasing. Picks a Halton sub-pixel jitter each frame; the core then renders
  at one sample (TAA replaces `msaa`) with a motion vector target (RG16F, from last frame's instance rows and
  camera); the resolve (compute, `Shaders/Taa.comp.fsl`: each 8x8 group reads its pixels and a one-pixel border
  into shared memory once) reprojects two ping-pong RGBA16F histories (5-tap Catmull-Rom), clips them to the 3x3
  neighbourhood's spread in YCoCg, blends. Sharpening is the core's, after TAA, never in history: its tonemap pass (`Shaders/Tonemap.comp.fsl`) tonemaps each
  pixel once and sharpens into a packed R32_UINT the composite reads (tonemappers in `Shaders/Tonemap.h.fsl`).
  Switch off: the core renders with `msaa` as before.
- `renderer_ao`: GTAO (Jimenez 2016) from the prepass depth alone (normals rebuilt from depth), then
  `denoise` depth-aware 3x3 blurs, into an RG16F texture at full or half render size (`F->aoScale`; x visibility, and
  every pass carries the view depth in y, so the blurs read no depth). At half size the forward pass upsamples it
  itself (the four nearest texels, weighted by depth against its own view depth): no full-size pass. It darkens sky ambient by it:
  diffuse with the multi-bounce fit, specular with Lagarde's specular occlusion. Direct light is untouched.
  Runs between the prepass and the forward pass, beside culling. Noise rotates per frame only with TAA on.
- `renderer_contact`: sun contact shadows, Bend Studio's screen space shadows (Apache-2.0; `third_party/bend_sss/bend_sss_cpu.h`
  unmodified, `Shaders/BendSss.h.fsl` our FSL port of its GPU half, its header listing every change: keep that list current,
  as the licence requires). 64-thread groups each take one stretch of a ray from the sun's point on screen, share their depth
  reads in groupshared memory, and test 60 pixels towards the sun, into a full-size R32F texture the forward pass multiplies
  the sun's shadow by. Bend's dispatch list is merged into one dispatch (a table in the constant buffer). Off when there is no sun.
- `renderer_probes`: reflection probes games place (its own `Probe` table, up to 16). Each is captured as a
  cube (the scene lit diffusely: sky ambient, sun without shadow, the nearest 64 local lights; alpha =
  coverage), halved level by level, prefiltered (GGX) into one slice of an RGBA16F octahedral 2D array, a mip
  per roughness step. Dirty probes are captured first (at most `updates_per_frame`, or 1), then that many more
  in turn. The forward pass blends the probes whose boxes hold the pixel, smallest box first, projects onto
  the box (parallax), and lets the live sky show where the capture saw none.
- `renderer_ssr`: screen-space reflections over the probes. It copies the prepass depth (full or half size);
  the core's forward pass marches each glossy pixel's reflected ray across the screen against that copy
  (1/depth linear in screen space, the march's noise rotated per frame for TAA). Where the ray first goes
  behind what is seen, 4 halvings find the crossing, and only there is thickness judged: `thickness` on a
  continuous surface, but where the seen depth jumps nearer (an object's outline, where objects are thin) only
  a ray barely behind counts, else it marches on. Glossy rays follow a GGX microfacet normal drawn per pixel
  and frame, so TAA averages them into the roughness's blur, and takes the colour from last frame's
  TAA-resolved image (`previous`), reprojected. Where nothing is found, off screen, near the screen's edges
  or above `max_roughness`, the probes show. Needs TAA: without it there is no previous image and it is off.
  Only on screen: what is behind the camera or hidden keeps the probes' (box-projected) reflection.
- `renderer_vrs`: variable-rate shading of the forward pass, through Yea's shading-rate service
  (`yea::shadingRate()`, nullptr on GPUs without it: then the module does nothing). After the prepass, one
  compute group per rate tile (16 px on an RTX 2080; a shader variant per tile size, 8/16/32, so group memory
  fits the tile and enough groups run at once) loads the tile once into group memory and measures
  last frame's TAA-resolved image as seen (exposed and compressed): the RMS luminance step between
  horizontal and between vertical neighbours. While it runs, the core's depth prepass also writes each
  pixel's reflection roughness (R8, `gloss` in frame.h); a tile with a pixel smoother than glossy_below
  stays at 1x1 at any distance. Below `aggressiveness`' threshold (raised by motion only when `motion` > 0) that axis goes
  coarse (2x1, 1x2, 2x2; 4x4 with allow_4x4). From this frame's depth, a crease or edge keeps the tile at
  1x1, and with keep_curved so does a curved surface (1/z off a plane across the tile; a tile mostly
  surrounded by curved ones too, as it lies on one flat facet of a curved mesh). A second small pass spreads
  full rate from tiles kept for their contrast to their neighbours (last frame's image lags). Writes Yea's R8_UINT rate image
  and an R32_UINT copy (Show Rates); the core binds the forward pass with the service's `bindRenderTargets`.
  Needs TAA and one sample; off in F1's view modes. On Direct3D 12 it needs Yea 111 or later: older Yeas
  made the rate image typeless, and binding it lost the GPU device. Vulkan caveat: FSL declares `RWTex2D(uint)` as `r32ui`,
  so the R8_UINT image is written through a mismatched format qualifier (fine on D3D12; unverified on Vulkan
  VRS hardware).
- `renderer_rt`: ray-traced reflections, through Yea's ray-query service (`yea::raytracing()`, nullptr on GPUs
  without inline ray queries: the module then does nothing and `renderer_ssr` runs as before; with it, SSR is
  off). Owns the acceleration structures: one bottom-level structure per mesh (built on the frame's command list
  when the mesh is new or its geometry moved; replaced ones freed once no frame in flight uses them) and one
  top-level structure per frame in flight, its instances rewritten with `setInstances` and rebuilt every frame
  (mask 1 every drawable, 2 shadow casters). The core's vertex and index buffers get the device-address and
  build-input flags and a uint view when the service exists. While it runs, the prepass writes `surface`
  (RGBA16F: x reflection roughness, yzw the normal; renderer_vrs reads x as its gloss). After the prepass, one
  full-size compute dispatch traces each pixel below `max_roughness` along one GGX visible-normal sample (R2
  low-discrepancy sequence over frames, IGN across pixels), each sample capped at `brightest` times white. A hit visible on screen now (its depth matches there) takes last frame's resolved colour
  (`screen_colors`), fully lit with every shadow; elsewhere it is shaded at the hit: emission, the sun and up to
  64 local lights (the core's light buffer now holds the lights outside the view after the visible ones,
  `allLightCount`), sky ambient. Hits are shadowed as the scene is (`reflected_shadows` Scene): a local light with a
  map in renderer_shadows' atlas from that map, filtered as the forward pass does (`Shaders/Shadow.h.fsl` copies the
  core's localShadow: keep them in step), a light that casts in the scene but has no map (outside the view, over the
  budget; the core's LightData params.w) by a ray, the sun by a ray; at most 8 rays per hit. A scene ray-traced shadow
  mode would pick `kShadowsRays` in rt_prepare. Rays traces every light (to compare), Off none. Each thread lights its own
  hits (gathering them per group to light was no faster on an RTX 2080: Benchmark v7). A miss sees the sky. The forward
  pass blends it over the probes, fading out over the roughness limit's last third. Two passes, both
  `Shaders/Trace.comp.fsl`. The trace runs a thread per 2x2 block, so no lane idles: pixels smoother than `full_below`
  (0.15: mirrors) are traced at every pixel, so mirrors are exactly as at full resolution, and with `half_resolution`
  (on) one other pixel of each block (a different one each frame, for TAA to gather) into a half-size texture.
  Then a full-size reconstruction rebuilds every other glossy pixel from the 3x3 nearest half-resolution rays (their
  directions kept in an RGBA16F half-size texture), each turned towards its hit from this pixel and weighted by this
  pixel's GGX lobe over the ray's own sampling probability (Stachowiak 2015's ray reuse), by depth and normal match and
  by inverse brightness (pixels no ray fits are traced there). One full-size resolve then accumulates the reflections
  over frames (two RGBA16F histories plus RG16F view depth and samples held, alternating, kept only where alpha > 0;
  `rtReflection` is this frame's): history reprojected from the pixel's centre (not its jittered sample) by the
  reflection's virtual image (smooth) or the surface (rough), dropped where the surface's depth doesn't match,
  clipped to the neighbourhood's spread in YCoCg (at half resolution, of the 5x5 traced rays around), blended in
  compressed colour at 1/samples: up to `history_frames` (full from roughness 0.125, 4 on moving things, less on
  mirrors), up to 128 (`STILL_SAMPLES`) where neither the surface nor its reflection moved by 0.01 pixel and the
  history lay less than 1.5 box half-widths outside the clip box (`STILL_SNAP`; past it the history is changed and
  drops back to the cap at once), so a still view settles (3.6.3: before it, half-resolution reflections crawled in
  a still view with a diagonal stipple; a box widened by each pixel's own spread over time held them stiller but
  let a switched-off light linger). Each pixel's rays start from a hash of its position (interleaved gradient noise
  lined up into diagonal streaks).
  Without the accumulation, 1 spp reflections flickered (fireflies). The trace marks each 8x8 tile holding a pixel below
  `max_roughness` (`gTiles`, flag sets double-buffered so no thread reads a flag another rewrites); the resolve skips
  the rest and writes their zeros only when the history there isn't zero already. Every GPU object is freed
  in `unload()` (device-loss recovery unloads modules); nothing GPU-side is in tables.
- `renderer_upscale`: render scale and upscaling. `upscale_prepare` runs first of all; when it upscales it lowers
  `width`/`height` to the render size (every scene pass, the prepass, culling, AO, contact, SSR, VRS, RT, the forward
  pass and exposure run at it; jitter is in its pixels), and `upscale_run` takes the TAA resolve's place, writing
  `resolved` at `displayWidth`/`displayHeight`, the frame's size, where the tonemap pass and composite run. TAA then
  frees its history and stands aside (its `sharpen` still applies, after the upscale). Not upscaling (native size, the
  default), it returns false and the frame is exactly as without it: no pass, copy or target added. Shadow tiles size
  by `displayHeight`. renderer_vrs reads last frame's image through `_up` shader variants only while upscaled.
  Upscalers (`upscaler`: TAA, FSR, DLSS): DLSS through Yea's host service (`yea::dlss()`, Windows, NVIDIA RTX; the
  `service dlss when upscaler DLSS` line ties NVIDIA's files, which are Yea's own licence entries), taken in `load()`,
  evaluated from `upscale_run`, released in `unload()`; Native is DLAA. What a machine can't run is hidden by Yea's
  `needs` lines (DLSS: `dlss`; FSR: `msvc 14.40`, as AMD builds with toolset 14.44 and its DLLs can crash in std::mutex
  before 14.40, and `offer`, which `load()` says from FSR's files alone: Direct3D 12 and both DLLs in Assets/ on Windows,
  always on Linux; Transformer: `fp8`), so what's picked runs; the code checks none of it. Only run-time failures
  change it: a DLSS evaluate that fails gives way to FSR where Yea says FSR runs here (the upscaler's player setting's
  `values`), else TAA; FSR's loader, context or dispatch failing gives way to TAA (logged once, with the reason).
  FSR on Windows (Direct3D 12): AMD's signed FSR SDK 2.4 DLLs in `Assets/windows/` (a Yea from 133 puts them at the
  top of Assets/), loaded only while picked, upscaler first, each by its full path with backslashes and
  `LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS` (Assets/ isn't on Windows' DLL search path, and
  the loader opens the upscaler by bare name; never SetDllDirectory, which is process-wide), through AMD's MIT API
  headers (`third_party/ffx`); FSR 4 on Radeon RX 6000 and later, FSR 3.1 elsewhere. FSR on Linux (Vulkan): AMD's
  FSR 3.1 upscaler and Vulkan backend compiled in from AMD's MIT source (`third_party/ffx_vk`, SDK 1.1.4, built by
  `fsr_vk.cpp`; its few edits are marked `Modified for Yea's Renderer` in place). Its shaders are headers in
  `third_party/ffx_vk/generated`, made by `fsr_vk_shaders.py` (glslangValidator; run it after changing AMD's GLSL).
  Its 16-bit and wave64 shaders run only where `yea::device()` says the device has them on. Its working memory is
  allocated in this module's own `update()`, never in `upscale_prepare`. Jitter: Halton 2,3 over 8 x ratio² phases;
  the render size is the vendors' ratio per quality mode (DLSS's own `renderSize` for DLSS); the core's exposure is copied into a 1x1 texture (one tiny
  dispatch); the upscalers' own sharpening is off. After one records, The Forge's bound state is forgotten so its next binds
  are made in full. Lines said from `upscale_prepare` wait for this module's `update()`, so they are filed under
  renderer_upscale: which upscaler is picked and runs, its sizes, and why FSR can't run.
  Nothing of Microsoft's ships.
- The core calls the others through optional imports with a `RenderFrame*` (as `u64`): `frame.h`, copied
  identically into `renderer`, `renderer_culling`, `renderer_shadows`, `renderer_taa`, `renderer_ao`, `renderer_contact`, `renderer_probes`, `renderer_ssr`, `renderer_vrs`, `renderer_rt` and `renderer_upscale` (`mathx.h` into the
  first three). Change all copies together and raise `RENDER_FRAME_VERSION`. `upscale_prepare`, then `taa_prepare` run first,
  before the camera, so the jitter reaches the projection.

## Hook-up points (exports of `renderer`)

Units: metres; sun intensity in lux, point and spot in candela. Colours linear RGB.
- Meshes: `mesh_box/sphere/plane/cylinder/cone/torus` (unit size, centred), or `mesh_begin`,
  `mesh_vertex(pos, normal)` (returns a local index), `mesh_triangle`, `mesh_end` → mesh id. Meshes are
  uploaded when they change (a stall: build them up front). A triangle faces where cross(b - a, c - a)
  points (counter-clockwise around its outward normal); back faces are culled.
- Mesh versions (LOD): the curved built-in shapes make a chain of simpler versions after the mesh they return
  (`MeshLod` table: each row's `next` and its surface `error` in object units; ~1/sqrt(2) the segments each, so the
  error doubles). Per instance the core draws the simplest within `lod_error` display pixels at its nearest point
  (off screen: its nearest distance, which bounds reflections); casters, drawables, renderer_rt's BLAS (one per
  version) and probes use that version. renderer_shadows goes coarser per view, within half a texel, never finer
  than the screen's (a finer caster would shadow the screen's surface). A version change counts as a move for the
  cached shadow maps. Custom meshes have one version. Request finer segments for close-ups freely: far away it costs
  nothing.
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
materials, instances and lights once, then move what moves and set the camera. Build again whenever the
renderer's `instance_count()`/`light_count()` differ from what you left last frame (a crash, revert or reload
lost or replaced its scene: your handles are stale), and never while `mesh_sphere` etc. return 0xFFFFFFFF
(the renderer is unloaded and its imports are stubs), as the example does.

## Knobs

`renderer`: exposure_ev, auto_exposure, manual_ev100, adapt_speed, tonemapper (a choice: Khronos_PBR_Neutral,
AgX (our own constants, in Tonemap.h.fsl), Hable_Filmic, Reinhard, None, ACES_Fitted (Hill's fit, MIT; its code
compiles only while the shader's TONEMAPPER_ACES_FITTED is 1, which a build baking another tonemapper sets to 0),
Tony_McMapface (its table, Assets/tony_mc_mapface.dds, is read only while it is picked; Khronos PBR Neutral stands in
when missing); the shader's mode is its index, so new options go at the end; both licences are tied to their options), msaa (1/2/4/8), specular_aa, sky/horizon/ground colours, sky_luminance, ambient,
sun_disk (degrees), near_plane, gpu_timings (logs each pass's GPU time every 10 s: avg/max ms), lod_error (display pixels; 0: always the finest). `renderer_culling`: enabled, max_lights, tile_size (8/16/32), z_bins,
heatmap. `renderer_shadows`: enabled (the sun's), cascades, shadow_distance, cascade_split, sun_resolution, softness,
filter_taps, pcss, sun_size, normal_bias, slope_bias, receiver_plane, light_shadows, shadowed_lights, local_atlas_resolution,
updates_per_frame, max_tile_resolution, contact_hardening, light_size (m), show_cascades, show_atlas (the light shadow atlas, top right). `renderer_taa`: enabled, history_weight, clamp_width, sharpen, jitter,
jitter_phases. `renderer_upscale`: upscaler (TAA, FSR, DLSS), quality_mode (Native, Quality 67%, Balanced 59%,
Performance 50%, Ultra_Performance 33%; for an upscaler only), dlss_model (Auto: Transformer where the GPU runs FP8, CNN
elsewhere, CNN until Yea reports FP8; Transformer: NVIDIA's default per mode, K/M/L; CNN: preset E, two to five times cheaper on RTX 20 and 30, falling back to the default in a mode where it's refused and the default then runs, said once per mode; Transformer is
hidden without FP8). `renderer_ao`: enabled, half_resolution (GTAO and its blur at half size, upsampled along edges in the forward pass), radius, intensity, falloff, slices, steps, denoise, specular_occlusion,
multi_bounce, show_ao. `renderer_contact`: enabled, surface_thickness (a part of the distance), edge_threshold, contrast, ignore_edges,
intensity, max_distance, show_contact. `renderer_probes`: enabled, resolution (64-512), updates_per_frame, intensity, blend, parallax.
`renderer_ssr`: enabled, half_resolution, max_roughness, thickness, max_distance, steps, edge_fade, intensity,
show_hits (tints where reflections were found). `renderer_vrs`: enabled, aggressiveness, motion (0), keep_curved, glossy_below, allow_4x4,
show_rates (green half rate, blue 2x2, magenta coarser). `renderer_rt`: enabled, max_roughness, half_resolution, full_below, screen_colors,
reflected_shadows (Scene, Rays, Off), history_frames, brightest, max_distance, show_hits (green: hit seen on screen, blue: lit at the hit, grey: sky),
probe_rays (temporary: 0 off; traces 1 sun ray plus k-1 local-light rays per pixel and discards them, to measure ray-traced
shadows' cost on the user's GPU; remove once measured).
Players of a released game tune (`player` lines): upscaler, quality_mode, dlss_model, exposure_ev (brightness), lod_error,
TAA's sharpen, shadows' sun_resolution, shadow_distance, pcss and light_shadows, and enabled for AO, contact shadows, SSR,
ray-traced reflections and VRS, with half_resolution for AO, SSR and ray-traced reflections, and probes' resolution.
Debug views and show_ knobs never. renderer_rt's `enabled` needs `raytracing` and renderer_vrs' `shadingrate` (hidden and
false without them); each still returns from `load()` before touching its service when that is nullptr, as `load()` runs
either way.

## Reset

All scene data is in `renderer`'s tables and state: Reset empties the scene; the feature's link module
builds it again. GPU objects and shadow caches are rebuilt on load.

## Cost

Per frame: one depth prepass and one forward pass (draws grouped by mesh and sidedness, instanced; back
faces culled except for double-sided materials; a leaner forward shader unless PCSS or a debug view is on), one culling
dispatch per 16 px tile, a GTAO dispatch and `denoise` blur dispatches (at half size with half_resolution; the forward pass upsamples), one contact shadow dispatch (all full size; 64-pixel stretches of rays from the sun's point on screen, two bilinear depth reads per pixel), one
rate dispatch per 16 px tile and a small dilation dispatch with variable-rate shading (the forward pass then shades smooth tiles at 2x2), one
depth copy for screen reflections (their march, up to `steps` plus 4 depth reads per crossing, runs in the forward pass for
glossy pixels with reflections on, and shows in its timing), per re-captured probe 6 scene draws at half its
resolution plus 6 small passes per mip, sun cascades whose view or casters changed redrawn, at most `updates_per_frame` local lights redrawn (oldest shadow first), two
exposure dispatches, with ray-traced reflections (instead of the SSR depth copy and march) one top-level
rebuild, a trace dispatch (a closest-hit ray per glossy pixel, a quarter of them with half_resolution except
mirror-like ones; where the hit isn't on screen a shadow map lookup per lit light with a map, a ray for the sun and for mapless casters), with half_resolution a
full-size reconstruction (nine rays reused per glossy pixel) and a full-size resolve (temporal accumulation; tiles with no glossy pixel cost a read or two), one full-screen TAA resolve (when on; with an upscaler, its dispatch at the display size and a one-thread
exposure copy instead, every scene pass then at the render size), one full-screen composite. No per-frame allocation. GPU memory at the defaults:
HDR + depth at 4x MSAA, sun atlas 4096² D32 (64 MB), local atlas 4096² (64 MB), a screen-size R32F depth copy
for screen reflections (a quarter of it at half resolution).

## Pitfalls

- Third-party material: every addition or removal updates both `licenses.json` and `LICENSES.txt` (its name,
  what uses it, source, verbatim licence text with its copyright line) in the same commit.
- Code run per light per pixel (`brdfLight`, the shadow filters) stays free of sin, cos, pow and per-tap sqrt: the
  filters' taps are `FrameData.taps` (a Vogel disk the core builds for `counts.w`), turned by the pixel's `Surface.rot`.
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
- Contact shadows refine the sun's shadow maps and are off whenever those are. Bend's thickness and edge threshold are parts
  of the depth (reversed Z: near / z), so of the view distance, not metres.
- A module that fills the scene can update before the others have loaded, when its imports are stubs
  that do nothing; the scene lives in tables, so it is not built again. Check counts each frame and
  add what is missing (the example adds its probes whenever `probe_count()` is 0).
- A light's shadow is drawn from where it was when its tiles were last redrawn, which lags a moving light
  when more lights move than `updates_per_frame`. Shading must use that drawn position (point lights pick
  their cube face from it, kept in the first face's `info.yzw`), never the light's current one, or points project off the tile's edge.
- 2D world quads the host draws after this view are over the 3D image; there is no depth between them.
- FSR's dispatch sets its own descriptor heaps and root signatures on the command list; `restoreForgeState` sets
  The Forge's back and forgets its bound tables (The Forge sets heaps only at `beginCmd`). Any new vendor dispatch must
  do the same.
- A new module of the component isn't added to the example's Base Settings by `yea publish`: `yea feature set "Base
  Settings" <every module> --view Renderer`.
- Sizes: `width`/`height` are the render size, `displayWidth`/`displayHeight` the frame's. Anything shown at the frame's
  size or chosen by how big things look on screen (shadow resolution) uses the display size; anything reading this
  frame's scene targets uses the render size. At native size they are equal, so a mix-up shows only when upscaled.
- Instances draw with an instance-rate stream of indices, never `SV_InstanceID` (D3D12/Vulkan rule).
- Shadows use standard Z (near 0); the camera uses reversed-Z with an infinite far plane.
- With TAA on, the prepass uses the motion vertex shader too, so both passes compute the same depth.
- Motion vectors match an instance by its table row and generation; a reused row starts with no motion.
- Take state blocks (`host()->state`) only in your own `load()`: the host files a block under the module
  running when it is asked for, so one asked for inside an export another module calls is that module's
  (the camera once never reached the renderer this way). `renderer` binds its blocks in `bindState()`.
- The same goes for `new`: memory allocated inside an export the core calls (`*_prepare`, `*_run`) goes on the core's
  heap and is freed when the core reloads. Allocate in your own `load()` (renderer_rt once crashed writing there).
  So does `yea::assets()`: asked for inside the core's call it is the core's folder. Take it in your own `load()`
  (renderer_upscale looked for FSR's DLLs in renderer's Assets/ this way, and FSR never ran before 3.1.2).
- A stand-in for a depth texture (bound while its module draws none) is a render target (`addRenderTarget`, D32,
  2x2) as the core's `gDummyDepth`, never `addResource`: Direct3D 12 reads a depth format only through a render
  target's typeless resource. Samplers go in a set of their own (`Persistent`, written once), as the core's.
  renderer_rt's 1x1 `addResource` D32 stand-in, bound while renderer_shadows was reloading, most likely crashed
  Direct3D 12 in 2.2.1.
- A probe sees the objects inside its box too, so a shiny object reflects itself where it is near the probe.

## Example view (`example/renderer_example`)

Floor, gold and red sphere rows sweeping roughness, pillars and assorted shapes, spinning tori, a sun,
`light_count` moving coloured point lights (markers glow in their colour; the first `shadowed_points` cast
shadows), `spot_count` sweeping shadowed spots (each with a small orb in its colour), two reflection probes (the sphere rows, the whole yard), a free-look camera: holding its `look` action (left mouse button) after a press on the game
asks the host for the mouse each frame (`yea::captureMouse(true)`: hidden, kept in the window) and turns by
`yea_mouse_dx`/`yea_mouse_dy` (`look_sensitivity`); the move keys fly (WASD, Q/E down/up, `move_speed`).
The host lets go when the button is up (it is no longer asked), F1 opens, focus is lost, or on a view switch
or Reset. Reset: lights and camera
back to the start. `benchmark_view` holds the camera at its start view (field of view 55), or at the gold or
mirror sphere's close-up (`benchmark_camera`, field of view 40), without moving the user's own, and holds the lights and shapes still, so GPU Timings compare between versions; the project's
`modules/test_renderer_bench` (a test on the user's machine) drives it. Exports `picture` (a fixed angle).
