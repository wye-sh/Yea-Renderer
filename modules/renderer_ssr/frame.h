// The frame contract between the renderer modules: renderer (the core) fills a RenderFrame each
// frame and passes its address (as u64) to the other renderer modules (culling, shadows, taa, ao, contact, probes, ssr, vrs, rt, upscale, sky, fog, particles, gi, post). The same
// file sits in each of them; keep the copies identical and raise RENDER_FRAME_VERSION on any change.
#pragma once
#include <stdint.h>

struct Renderer;
struct Cmd;
struct Buffer;
struct RenderTarget;
struct Texture;

#define RENDER_FRAME_VERSION 28u
#define RENDER_RT_MAX_TRACED 8u
#define RENDER_MAX_CASCADES  4u
#define RENDER_MAX_TEXTURES  1024u // texture maps the shaders' array holds (the core's Texture table)
#define RENDER_MAX_BLEND_RUNS 64u  // depths blended surfaces split the particles at (frame v23)
#define RENDER_MAX_DECALS     256u // decals drawn in a frame (frame v23)
#define RENDER_MAX_WORLD_DECALS 1024u // decals captures and ray hits see (frame v25)
#define RENDER_VIEW_SHIFT    20u // shadow instance stream entry: instance | view << 20
#define RENDER_MAX_TINT_LAYERS 8u // glass layers a shadow view's tint keeps at most (frame v26)
// Vertex streams (positions, attributes): bytes per vertex.
#define RENDER_POSITION_STRIDE  12u // float3
#define RENDER_ATTRIBUTE_STRIDE 20u // 5 u32: [0] normal, [1] tangent, [2..3] uv0 (float2), [4] colour (rgba8 unorm, r lowest).
                                    // Normal and tangent: octahedral snorm16x2 (x the low 16 bits; value / 32767, at least
                                    // -1); the tangent's bitangent sign is y's lowest bit (odd: -1; bitangent =
                                    // cross(normal, tangent) * sign). Decode (f the two values): n = (f.x, f.y,
                                    // 1 - |f.x| - |f.y|), t = saturate(-n.z), n.xy += n.xy >= 0 ? -t : t, normalize(n).

// GPU layouts (match Shaders/Common.h.fsl in renderer). Matrices are 4 rows; shaders dot(row, p).
struct RenderGpuInstance
{
    float    row[3][4];   // world, 3x4 rows
    uint32_t material;
    uint32_t flags;
    uint32_t pad[2];
};

struct RenderGpuLight
{
    float pos[4];    // xyz, w range
    float color[4];  // rgb * intensity (candela), w type (1 point, 2 spot)
    float dir[4];    // xyz spot direction, w spot scale
    float params[4]; // x spot offset, y first shadow view (-1 none), z 1/range^2, w 1: casts shadows in the scene (it may still have no map), 2 + n: ray-traced, its shadow in
                     // rtShadows slice 1 + n
    float sphere[4]; // culling sphere: xyz centre, w radius
};

// Each light's shape, parallel to RenderGpuLight (lightShapes). Area lights are point lights to everything else
// (color.w 1, a cube of shadow tiles, color the candela seen face-on); only their shading and shadow size read this.
// Kinds: 0 point or spot, 3 rect, 4 disc (an ellipse), 5 sphere, 6 tube. Metres. A rect or disc emits along dir
// (one side); a tube lies along dir.
struct RenderGpuLightShape
{
    float ex[4]; // rect, disc: half the width along the light's tangent (a vector); tube: half its length along dir. w: kind
    float ey[4]; // rect, disc: half the height (a vector, dir x tangent); tube: x its radius. w: kinds 3 to 6 1 / area
                 // (luminance = color * w; a sphere's radius is sqrt(1 / (pi w)), the others' shadow radius the same);
                 // kind 0 its source radius for shadows (0: none set, rtLightRadius and renderer_shadows' light_size)
};

struct RenderGpuShadowView
{
    float row[4][4]; // world -> clip (ndc xy in -1..1, z 0..1)
    float rect[4];   // tile in the atlas: u, v, width, height
    float info[4];   // x world size of a texel (at distance 1 when perspective); sun: y depth range; lights: yzw drawn from
    float inv[3][4]; // columns 0-2 of row's inverse: a world plane p (n, -n.P) is dot(inv[i], p) x + ... = 0 in the view's ndc
};

// Glass over a shadow view (frame v26, renderer's Shaders/Common.h.fsl ShadowTint): the square of the tint maps that covers
// it. Map uv (the atlas's, as viewToAtlas) u is the square's texel (u - area.xy) * area.z; layer i's copy of the square
// starts at texel (layer[i] & 0xFFFF, layer[i] >> 16) of tintDepth and tintColor.
struct RenderGpuShadowTint
{
    float    area[4];  // xy the map uv where the square starts, z its texels per map uv, w its side in texels
    uint32_t layer[RENDER_MAX_TINT_LAYERS];
    uint32_t count;    // layers (0: no glass over this view)
    uint32_t pad[3];
};

// A reflection probe as the shaders read it (renderer's Shaders/Common.h.fsl ProbeData).
struct RenderGpuProbe
{
    float pos[4];    // xyz capture point, w slice in the probe array
    float boxMin[4]; // influence and parallax box
    float boxMax[4];
};

// A shadow-casting instance, as the core sorted them: by mesh (drawables: by mesh, then two-sided).
struct RenderCaster
{
    uint32_t instance;   // row in the frame's instance buffer
    uint32_t firstIndex;
    uint32_t indexCount;
    int32_t  vertexOffset;
    float    sphere[4];  // world bounding sphere
    uint64_t moved;      // frame number it last moved, appeared or changed mesh
    uint32_t twoSided;   // its material is double-sided: draw it without back-face culling
    uint32_t mesh;       // the core's mesh row
    uint32_t vertexCount; // the mesh's vertices, from vertexOffset
    uint32_t castShadow;  // it casts shadows (every caster does; a drawable may not)
    uint32_t generation;  // when its mesh's data was last written to the buffers (a frame number's low 32 bits, never 0):
                          // the same generation, buffers and ranges mean the same data
    uint32_t masked;      // its material is alpha-tested (cut out where its base colour's alpha is under its cutoff): draw it
                          // with uv0 and the material maps (textures, materialMaps)
    uint32_t dynamic;     // frame v24: the core skins it into vertices of its own every frame (vertexOffset, new each frame,
                          // with mesh the version it skinned): draw it on its own at exactly that version, never a coarser
                          // one; ray tracing refits rather than rebuilds. moved is this frame while it animates
};

// A visible local light, sorted by view depth (nearest first).
struct RenderLight
{
    float    pos[3];
    float    range;
    float    dir[3];
    float    cosOuter;
    float    sphere[4];
    float    viewZ;      // view depth of the sphere centre
    float    intensity;  // for picking which lights get shadows
    uint32_t type;       // 1 point, 2 spot
    uint32_t id;         // stable while the light exists (its handle)
    uint32_t castShadow;
    int32_t  shadow;     // set by renderer_shadows: first shadow view, or -1
    int32_t  traced;     // set by renderer_rt's rt_pick: n, its shadow traced into rtShadows slice 1 + n; or -1 (renderer_shadows
                         // gives it no tiles)
    uint32_t source;     // the core's own reference (its light row)
    uint32_t kind;       // its shape, as RenderGpuLightShape's: 0 point or spot, 3 rect, 4 disc (both light only the side dir
                         // points to), 5 sphere, 6 tube
    float    size;       // its radius for contact-hardening shadows (an area light's: a disc of its area), or 0: light_size
};

struct RenderFrame
{
    uint32_t version;
    uint32_t frameIndex;
    uint32_t framesInFlight;
    uint32_t width, height;
    uint32_t sampleCount;
    uint64_t frameNumber;
    Renderer* renderer;
    Cmd*      cmd;

    // Camera: view (world -> view, +z forward), projection (reversed-Z, infinite far).
    float view[4][4]; // rows
    float camPos[3];
    float nearZ;
    float p00, p11;   // projection scale x, y

    // The sun: the first directional light.
    uint32_t sunOn;
    uint32_t sunShadow;
    float    sunDir[3]; // towards the sun

    // Geometry: two vertex streams (RENDER_POSITION_STRIDE, RENDER_ATTRIBUTE_STRIDE) and u32 indices, relative to a
    // mesh's vertexOffset. GPU-only, kept in vertex (or index) and shader-resource state together: never barrier them.
    // With ray queries also device-addressed acceleration-structure inputs, readable as Buffer(uint) (4-byte stride).
    Buffer*               positions;
    Buffer*               attributes;
    Buffer*               indices;
    Buffer*               instances;   // RenderGpuInstance[]
    const RenderGpuInstance* instanceRows; // the same rows on the CPU, instanceCount of them
    uint32_t              instanceCount;
    const RenderCaster* casters;
    uint32_t              casterCount;
    uint64_t              casterRemoved; // frame a caster was last removed
    float                 sceneSphere[4];

    // Local lights (in), capped by culling.
    RenderLight* lights;
    uint32_t       lightCount;

    // Camera depth (reversed-Z), in SHADER_RESOURCE state while renderer_culling runs.
    RenderTarget* depth;
    Buffer*       lightBuffer; // RenderGpuLight[], this frame's: lightCount visible ones, then the rest up to allLightCount
    uint32_t      allLightCount; // also the lights outside the view (no shadows), for what reflections see
    Buffer*       lightShapes;   // RenderGpuLightShape[], in lightBuffer's order

    // Out from renderer_culling.
    Buffer*  tileMasks;     // u32[tilesX * tilesY * wordsPerTile], SHADER_RESOURCE after cull_run
    Buffer*  zbins;         // u32[binCount]: low 16 bits first light, high 16 bits last
    uint32_t tileSize, tilesX, tilesY, wordsPerTile, binCount;
    float    binScale;      // bins per view-space unit
    uint32_t heatmap;

    // Out from renderer_shadows.
    RenderTarget* sunAtlas;    // SHADER_RESOURCE after shadows_render
    RenderTarget* localAtlas;
    Buffer*       shadowViews; // RenderGpuShadowView[]
    uint32_t      cascadeCount;
    float         cascadeEnd[RENDER_MAX_CASCADES]; // view depth where each cascade ends
    float         softness;     // PCF radius in texels
    float         lightSize;    // sun size for PCSS (0: plain PCF)
    float         normalBias;   // in texels
    uint32_t      taps;
    uint32_t      showCascades; // bit 0: tint the cascades; bit 1: show the light shadow atlas
    float         sunTexel;     // 1 / sun atlas size
    float         localTexel;   // 1 / local atlas size
    float         localLightSize; // point and spot lights' radius (m) for contact-hardening, 0: plain PCF; negative: contact-
                                  // hardening on with no default radius (lights with their own size use it)
    uint32_t      receiverPlane;  // tilt each PCF tap's depth compare with the receiver's plane

    // TAA (renderer_taa). taa_prepare runs right after upscale_prepare, with only the fields above the camera set: it sets
    // jitter, taaOn, sharpen and resolved, and the core then renders at one sample with motion vectors.
    float         jitter[2];  // sub-pixel camera offset this frame, in pixels
    uint32_t      taaOn;
    float         sharpen;    // composite sharpening, 0..1
    RenderTarget* hdr;        // the frame's HDR colour, SHADER_RESOURCE when taa_resolve runs
    RenderTarget* motion;     // RG16F: uv now - uv last frame, SHADER_RESOURCE; depth too. While renderer_rt runs the prepass
                              // writes it (the sky 0), so it is this frame's in rt_run; the forward and sky passes write it again
    Buffer*       exposure;   // float[]: [1] the exposure in use
    RenderTarget* resolved;   // the target taa_resolve writes the anti-aliased HDR image to; SHADER_RESOURCE after
    RenderTarget* previous;   // last frame's resolved image (linear, like hdr), SHADER_RESOURCE all frame; or nullptr

    // Ambient occlusion (renderer_ao). ao_prepare runs after cull_prepare (jitter and taaOn are set);
    // ao_run right after it, after the depth prepass, with depth in SHADER_RESOURCE.
    Texture* ao;            // RG16F, x visibility (1: open), y view depth; at 1 / aoScale of width and height,
                            // SHADER_RESOURCE after ao_run
    uint32_t aoOn;
    uint32_t aoScale;       // 1: ao is full size; 2: half size, and the forward pass upsamples it along edges
    float    aoSpecular;    // specular occlusion, 0..1
    uint32_t aoMultiBounce; // brighter occlusion on bright surfaces (light bouncing between them)
    uint32_t aoDebug;       // show the occlusion alone

    // Contact shadows (renderer_contact), for the sun. contact_prepare runs after ao_prepare; contact_run
    // after the depth prepass, with depth in SHADER_RESOURCE.
    Texture* contact;      // R32F sun visibility (1: lit), full size, SHADER_RESOURCE after contact_run
    uint32_t contactOn;
    uint32_t contactDebug; // show the contact shadows alone

    // Reflection probes (renderer_probes). probes_prepare runs after contact_prepare, with the fields below
    // up to `ambient` set; probes_render right after shadows_render, before the depth prepass.
    const RenderCaster* drawables;     // every live instance (casting shadows or not), by mesh
    uint32_t              drawableCount;
    Buffer*               materials;     // this frame's materials (renderer's Shaders/Common.h.fsl MaterialData)
    float                 sunColor[3];   // the sun's colour * lux
    float                 skyZenith[3], skyHorizon[3], skyGround[3]; // luminance
    float                 ambient;
    // Out: RGBA16F octahedral 2D array, a slice per probe and a mip per roughness step; rgb is premultiplied
    // by coverage a, and the sky shows through where a < 1.
    RenderTarget*         probes;
    Buffer*               probeData;     // RenderGpuProbe[probeCount]
    uint32_t              probeCount;
    uint32_t              probeMips;
    float                 probeIntensity;
    float                 probeBlend;    // metres over which a probe fades in from its box's sides
    uint32_t              probeParallax;

    // Screen-space reflections (renderer_ssr). ssr_prepare runs after probes_prepare, and only with TAA (it
    // reflects `previous`); ssr_run after the depth prepass, with depth in SHADER_RESOURCE. The core's
    // forward pass traces against ssrDepth.
    Texture* ssrDepth;        // R32F camera depth (reversed-Z) at ssrScale of the screen, SHADER_RESOURCE after ssr_run
    uint32_t ssrOn;
    uint32_t ssrSteps;
    uint32_t ssrDebug;        // show where reflections were found on screen
    float    ssrMaxRoughness;
    float    ssrThickness;    // metres
    float    ssrMaxDistance;  // metres
    float    ssrEdgeFade;     // fraction of the screen
    float    ssrIntensity;
    float    ssrScale;        // 1 or 0.5

    // Variable-rate shading (renderer_vrs), only on GPUs with it (yea::shadingRate()) and with TAA. vrs_prepare
    // runs after ssr_prepare; vrs_run after the depth prepass, with depth in SHADER_RESOURCE, and leaves
    // rateImage in SHADING_RATE_SOURCE for the forward pass (the core binds it with the service's
    // bindRenderTargets).
    Texture* rateImage;       // Yea's R8_UINT rate image, a YEA_RATE_* per rateTile px tile
    Texture* rateCopy;        // the same rates, R32_UINT, SHADER_RESOURCE after vrs_run (Show Rates)
    RenderTarget* gloss;      // .x: each pixel's reflection roughness (1: none), from the prepass, SHADER_RESOURCE in vrs_run
                              // (R8, or `surface` while renderer_rt runs)
    uint32_t rateTile;
    uint32_t vrsDebug;        // tint the screen by rate

    // Ray-traced reflections (renderer_rt), only on GPUs with inline ray queries (yea::raytracing()) and with TAA;
    // screen-space reflections are then off. rt_prepare runs after probes_prepare; rt_run after the depth
    // prepass, with depth and surface in SHADER_RESOURCE. The forward pass reads rtReflection.
    RenderTarget* surface;    // RGBA16F from the prepass: x reflection roughness (specular AA in; 1: none), yzw the
                              // shading normal (towards the viewer)
    Texture* rtReflection;    // RGBA16F, full size, SHADER_RESOURCE after rt_run: rgb what the pixel's reflected ray
                              // sees, a its distance (0: not traced, RENDER_RT_MISS: the sky)
    float    rtMaxRoughness;  // traced below it, fading out over its last third
    uint32_t rtDebug;         // show where rays were traced (green: hit seen on screen, blue: shaded at the hit)
    // Ray-traced shadows (renderer_rt). rt_pick runs just before shadows_prepare, with the visible lights sorted and taaOn
    // known: it picks the lights (and the sun) whose shadows it traces, sets their `traced`, rtSunShadows, rtLightCount and
    // rtLightSlots, and renderer_shadows then draws no cascades while rtSunShadows is set and no tiles for traced lights.
    // rt_run fills rtShadows; the forward pass reads it only while rt_prepare also returned true and rtShadows is set.
    Texture* rtShadows;       // R16F 2D array at width x height, at least 2 slices (a 1-slice texture gets a 2D view, which a
                              // Tex2DArray slot rejects), SHADER_RESOURCE after rt_run: slice 0 the sun, slice 1 + n traced
                              // light n (1 lit, 0 shadowed)
    uint32_t rtSunShadows;    // 1: the first sun's shadow is traced (slice 0)
    uint32_t rtLightCount;    // traced local lights
    uint32_t rtLightSlots[RENDER_RT_MAX_TRACED]; // index in `lights` of traced light n
    float    rtSunAngle;      // the sun's angular diameter (radians, renderer_shadows' sun_size), filled even without PCSS
    float    rtLightRadius;   // point and spot lights' radius (m, renderer_shadows' light_size), filled even without
                              // contact-hardening; both set by shadows_prepare (0 while renderer_shadows isn't loaded)

    // Upscaling (renderer_upscale). upscale_prepare runs first of all, before taa_prepare, with only the fields above the
    // camera set and width, height = displayWidth, displayHeight = the frame's size. When it upscales, it lowers width and
    // height to the render size (every pass above runs at it, and jitter is in its pixels), sets jitter, taaOn, sharpen,
    // resolved (its output, display size) and previous (last frame's output) as taa_prepare would, and upscaleOn; TAA then
    // steps aside and upscale_run takes taa_resolve's place, with hdr, depth, motion and exposure as there. At native size
    // without an upscaler, width and height are the display's and nothing here changes the frame.
    uint32_t displayWidth, displayHeight; // the frame's size: what the tonemap pass, the composite and screen-sized choices
                                          // (such as shadow resolution) use
    uint32_t upscaleOn;
    float    mipBias;                     // texture LOD bias for the render scale (0 at native): log2(width / displayWidth)

    // Physical sky (renderer_sky, frame v17). sky_prepare runs once the camera, sunOn, sunDir and sunColor (the sun's
    // light above the atmosphere) are set, before shadows_prepare: it sets sunTransmittance (the share of it that reaches
    // the camera; the core then scales sunColor by it, so every module sees the sun as it arrives) and the sky's images,
    // all nullptr without it (the core's gradient sky). sky_run records first of all, before shadows_render, so they are
    // ready (SHADER_RESOURCE) for probes_render, rt_run and the forward pass.
    float    sunTransmittance[3];
    Texture* skyView;      // the sky's radiance around the camera (renderer_sky's skyViewUv), lux-scaled, no sun disk
    Texture* skyEnv;       // the sky as an octahedral map (y up, as the probes'), skyEnvMips mips, mip m the GGX lobe of
                           // roughness m / (skyEnvMips - 1); the sky's reflections
    uint32_t skyEnvMips;
    Buffer*  skySh;        // 9 float4 (rgb): the sky's ambient, cosine-convolved spherical harmonics over pi (bands 0..2,
                           // order y, z, x as renderer_sky's Lut.comp.fsl), radiance-like as skyIrradiance was
    Texture* skyAerial;    // 32 x 32 x 32 RGBA16F froxels over the screen: rgb light scattered in, a transmittance, from the
                           // camera to slice depth skyAerialDepth (km) x ((slice + 1) / 32)^2
    float    skyAerialDepth;
    float    skyRadii[4];  // km: x ground, y top of the atmosphere, z the camera from the planet's centre; w aerial strength

    // Volumetric fog (renderer_fog, frame v18). fog_prepare runs after vrs_prepare, with the lights, culling, shadows and
    // sky set; fog_run records after probes_render (the shadow maps and the sky's images ready), before the depth prepass.
    Texture* fogVolume;    // RGBA16F froxels over the render view, SHADER_RESOURCE after fog_run: rgb the light scattered in
                           // and a the transmittance, from the camera to the far side of each slice; nullptr: no fog
    float    fogParams[4]; // view depth z lies at slice p = log2(z * x + 1) * y (slice k's far side at p = k + 1); z 1 / slices
    float    fogUv[2];     // render uv times these: the volume's xy (it may reach past the view's right and bottom edges)
    // Frame v19. The height fog for rays the froxels don't hold (probe captures, reflections and their sky, ray-traced
    // reflections), which take it analytically along the ray: the exact optical depth, the sun (unshadowed) and the sky's
    // mean light scattered in. Set by fog_prepare; fogMedium[0] 0 without fog.
    float    fogMedium[4]; // x density (1/m) at and below z (base height, m), y falloff above it (1/m), w anisotropy g
    float    fogAlbedo[4]; // rgb scattering albedo, w the fog's range (m): how much of it a ray that leaves (the sky) crosses
    uint32_t keepShadowMaps; // set by the core before rt_pick while renderer_fog or renderer_particles runs (frame v20; v19
                             // fogShadowMaps): renderer_shadows keeps the sun's cascades and traced lights' tiles even when
                             // renderer_rt traces them (fog and particles read the maps)

    // Particles (renderer_particles, frame v20). particles_prepare runs after fog_prepare, particles_run after fog_run.
    // They draw in the core's see-through pass (frame v23: blendRuns), before exposure, TAA and the upscaler.
    Texture* reactive;      // R8 at render size, SHADER_RESOURCE after the see-through pass: how much of each pixel is particle
                            // or blended surface (0 where none drew), for TAA and the upscalers (their reactive or bias
                            // masks); nullptr: none drawn. The core's (frame v23; renderer_particles' before).

    // Texture maps (frame v21), set before shadows_prepare. A material's maps are materialMaps[its index] (renderer's
    // Shaders/Common.h.fsl MaterialMapData: each map's slot in textures + 1, 0 none); a pass that draws masked casters or
    // shades materials binds both. textures has RENDER_MAX_TEXTURES entries, a 2 x 2 stand-in where a slot is empty or
    // still loading, every one in SHADER_RESOURCE; bind them again whenever textureVersion changes (and on a set's first
    // use), never per frame. Sample with mipBias added.
    Buffer*         materialMaps;
    Texture* const* textures;
    uint32_t        textureVersion;
    // Dynamic diffuse GI (renderer_gi, frame v22; renderer_gi's Shaders/GiSample.h.fsl, copied identically into each
    // module that reads or traces the probes, says how). gi_prepare runs before rt_prepare and sets everything here, or
    // leaves giProbeCount 0 (no GI); renderer_rt's rt_run traces the rays right after its top-level build and sets
    // giTraced; gi_run (after rt_run, before the forward pass) blends them into the probes, or, when nothing traced them
    // (no ray tracing), captures a few probes a frame by rasterizing.
    float    giGrid[4][4];      // GiSample.h.fsl: g0 origin and frame bits, g1 spacing and probes, g2 counts and rays, g3 atlas texels
    float    giRotation[3][4];  // this frame's ray rotation (rows): ray k of n is giRayDirection(k, n, rows)
    uint32_t giProbeCount, giRaysPerProbe;
    Buffer*  giProbeData;       // float4 per probe (xyz relocation offset, w state: 0 off, 1 on, 2 on but not lit yet), SHADER_RESOURCE
                                // outside gi_run
    Buffer*  giRayOut;          // float4 per ray, probe-major (rgb radiance, a distance: negative a backface, 1e27 a miss), 16-byte
                                // stride, SHADER_RESOURCE between passes; rays of probes not traced this frame are left unwritten
    Texture* giIrradiance;      // RGBA16F octahedral tiles, SHADER_RESOURCE outside gi_run: rgb irradiance / pi; last frame's in rt_run
                                // (nullptr: none yet)
    Texture* giDistance;        // RGBA16F octahedral tiles: xy mean distance and its square
    uint32_t giTraced;          // set by rt_run: this frame's rays are in giRayOut
    uint32_t giDebug;           // show only the bounced light (the core)

    // See-through surfaces (frame v23). With one sample a pixel, the core draws blended surfaces (material_blend) and
    // particles in one pass after the forward pass and the sky, back to front: hdr, reactive and transparency bound in
    // that order (R8 both; reactive and transparency cleared), depth in SHADER_RESOURCE, nothing else. particles_draw_begin
    // records before the core binds them, particles_draw(run) inside (run 0 to blendRuns, the k-th just before the
    // surface at blendDepths[k]), particles_draw_end after it unbinds. Run k holds the particles deeper than
    // blendDepths[k] and not deeper than blendDepths[k - 1]; run blendRuns those nearer than all.
    uint32_t blendRuns;                              // set before particles_prepare; particles sort whenever it isn't 0
    float    blendDepths[RENDER_MAX_BLEND_RUNS];     // view depth (m), farthest first
    Texture* transparency;  // R8 at render size, SHADER_RESOURCE after that pass: how much of each pixel blended surfaces cover
                            // (1 - what they let through), for the upscalers' transparency masks; nullptr: none drawn
    // Decals (frame v23): the core's in view, at most RENDER_MAX_DECALS, in the order they paint, set before cull_prepare.
    // renderer_culling bins them as it bins the lights (decalMasks, decalBins), or leaves both nullptr.
    uint32_t     decalCount;
    const float (*decalSpheres)[4]; // each one's bounding sphere: world centre, radius
    Buffer*      decalSphereBuffer; // the same, float4 each, SHADER_RESOURCE
    Buffer*      decalMasks;        // set by cull_prepare: RENDER_MAX_DECALS / 32 bitmask words per tile (tiles as tileMasks)
    Buffer*      decalBins;         // set by cull_prepare: per z-bin (as zbins), the first decal in it | the last << 16
    // See-through instances (frame v25): every live blended instance (material_blend), on screen or not, laid out as
    // drawables (by mesh; the core's LOD), set before shadows_prepare. They are in neither casters nor drawables, so probes,
    // GI captures and the shadow maps' depth skip them; the shadow maps' tint draws them (frame v26). castShadow is its
    // instance_shadow.
    const RenderCaster* seeThrough;
    uint32_t            seeThroughCount;
    // Two float4 a material (renderer's Shaders/Scene.srt.h gRefraction): x ior, y thickness, z shape (0 thin, 1 solid),
    // w 1 given; rgb tint (1, 1, 1 when not given). With materialMaps' misc.z (opacity, times the base colour's alpha), a
    // layer lets through (1 - coverage) * tint * (1 - its specular reflection). SHADER_RESOURCE.
    Buffer*             refraction;
    // Set by renderer_rt's rt_prepare beside rtShadows, filled by rt_run: RGBA16F at rtShadows' size and layers (alpha: renderer_rt's own), the mean colour glass lets through over a pixel's
    // shadow rays that no opaque surface blocked (visibility x tint is the light that arrives; white where no glass).
    // SHADER_RESOURCE for the forward pass. nullptr when no see-through instance with castShadow is traced this frame.
    Texture*            rtShadowTint;
    // Decals where the camera's view doesn't reach (frame v25): reflection probe and GI captures, ray hits. Every live
    // decal (with the core's decals knob on), in the order they paint, at most RENDER_MAX_WORLD_DECALS, as the core's
    // DecalData (renderer's Shaders/Decal.h.fsl, whose copy paints them: decalsAt), with a world grid over them: cell
    // (x, y, z), x fastest, spans decalGrid[0].xyz + (x, y, z) / decalGrid[0].w; decalCells[cell] = (first, count) of its
    // decals in decalItems, each an index into worldDecals, in paint order. Set before shadows_prepare; SHADER_RESOURCE.
    // worldDecalCount 0: none (the buffers are then nullptr). decalsChanged: the frame they last changed (added, moved,
    // repainted, removed, or a map of one loaded), when captures of what they paint go stale.
    uint32_t worldDecalCount;
    Buffer*  worldDecals;
    Buffer*  decalCells;    // uint2 a cell
    Buffer*  decalItems;    // uint
    float    decalGrid[2][4]; // [0] xyz the grid's world corner, w cells a metre; [1] xyz its cells along x, y, z
    uint64_t decalsChanged;
    // The core's split-sum table (renderer's Shaders/Brdf.h.fsl dfgLookup, gDfg): 128 x 128 float2 (frame v26), x sqrt(NoV),
    // y roughness; SHADER_RESOURCE, the same every frame, so passes outside the core weigh specular reflection and
    // transmission (glass) by the same numbers.
    Buffer*  dfg;
    // Glass in the shadow maps (frame v26), out from renderer_shadows: what see-through casters (seeThrough with castShadow)
    // let through to what lies behind them, from each shadow view, in layers out from the light (a deep shadow map), each
    // layer the next glass surface a light ray meets: its depth in the view (as the atlas's) in tintDepth (R32F) and what it
    // lets through, (1 - coverage) * tint * (1 - its specular reflection), in tintColor (RGB10A2, rgb). The last layer also
    // carries every surface behind it, so a view is exact while its glass needs no more layers than it has; renderer_shadows
    // gives a view more (up to RENDER_MAX_TINT_LAYERS) when it finds more. A point at depth z in a view gets the product of
    // its layers nearer than z. shadowTints: RenderGpuShadowTint[], as shadowViews. All SHADER_RESOURCE after shadows_render;
    // tintDepth and tintColor nullptr when no view has glass this frame.
    RenderTarget* tintDepth;
    RenderTarget* tintColor;
    Buffer*       shadowTints;
    // Out from renderer_culling (frame v27): the lights by the screen tiles' sides alone, for surfaces in front of the
    // opaque scene (glass), which tileMasks' depth ranges leave out: u32[tilesX * tilesY * wordsPerTile] as tileMasks, a
    // light's bit set where its sphere reaches the tile's four side planes past the near plane, sky tiles too. ANDed with
    // the z-bin's range as tileMasks are. SHADER_RESOURCE after cull_run; set whenever tileMasks is.
    Buffer*       sideMasks;
    // Set by the core before rt_prepare (frame v28): 1 while glass refracts (its refraction knob is on and the scene colour
    // chain is made, not in F1's view modes), as the see-through pass does whether or not a refracting surface is in view.
    // renderer_rt bends reflection rays through material_refraction's glass only then; else they pass straight, tinted.
    // Either way a solid (refraction shape 1) is one layer, its near side: the see-through pass never draws its far side.
    uint32_t      refractOn;
    // Set by renderer_rt in rt_prepare while it traces on-screen refraction (refractOn, rays available), filled by rt_run
    // before the forward pass; SHADER_RESOURCE by the see-through pass. nullptr: refract from the scene colour chain as
    // before. Per pixel, for the nearest refracting layer along the view (material_refraction given, the glassCounts rule;
    // non-refracting glass in front of it passes straight):
    Texture* rtRefraction;     // RGBA16F, render size: rgb what the layer lets through from behind, as the chain would hold
                               // it there (lit, fogged, deeper glass shaded; before this layer's own T), a its view depth
                               // (0: no refracting layer here)
    Texture* rtRefractLanding; // RGBA16F: where w is 1, the bent ray meets what the depth shows on screen and the straight
                               // view to it crosses as many other glass layers: read the chain there (it holds steam and the
                               // glass drawn behind), at xy pixels from the fragment's own centre (In.Position.xy + xy, this
                               // frame's jittered pixels), at mip z (refractedScene's px rule; clamp it to the chain's
                               // levels). w 0: take rtRefraction.rgb
    // The core's area-light shading data for renderer_rt's hits (as Lighting.h.fsl reads them), every frame:
    Buffer*  ltc;         // the area lights' LTC tables (gLtc: Assets/ltc.bin, 2 x 64 x 64 float4), SHADER_RESOURCE
    Buffer*  lightLooks;  // per GPU light, its image + 1 (0 none), in lightBuffer's order (uint each; written by the core
                          // later in its draw, before submission: read it on the GPU only)
    Texture* lightImages; // the rect lights' images (2D array, prefiltered mips), SHADER_RESOURCE
};
#define RENDER_RT_MISS 60000.0f
