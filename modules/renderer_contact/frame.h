// The frame contract between the renderer modules: renderer (the core) fills a RenderFrame each
// frame and passes its address (as u64) to the other renderer modules (culling, shadows, taa, ao, contact, probes). The same
// file sits in each of them; keep the copies identical and raise RENDER_FRAME_VERSION on any change.
#pragma once
#include <stdint.h>

struct Renderer;
struct Cmd;
struct Buffer;
struct RenderTarget;
struct Texture;

#define RENDER_FRAME_VERSION 5u
#define RENDER_MAX_CASCADES  4u
#define RENDER_VIEW_SHIFT    20u // shadow instance stream entry: instance | view << 20

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
    float params[4]; // x spot offset, y first shadow view (-1 none), z 1/range^2, w 0
    float sphere[4]; // culling sphere: xyz centre, w radius
};

struct RenderGpuShadowView
{
    float row[4][4]; // world -> clip (ndc xy in -1..1, z 0..1)
    float rect[4];   // tile in the atlas: u, v, width, height
    float info[4];   // x world size of a texel (at distance 1 when perspective), y depth range, z perspective, w near
};

// A reflection probe as the shaders read it (renderer's Shaders/Common.h.fsl ProbeData).
struct RenderGpuProbe
{
    float pos[4];    // xyz capture point, w slice in the probe array
    float boxMin[4]; // influence and parallax box
    float boxMax[4];
};

// A shadow-casting instance, as the core sorted them: by mesh.
struct RenderCaster
{
    uint32_t instance;   // row in the frame's instance buffer
    uint32_t firstIndex;
    uint32_t indexCount;
    int32_t  vertexOffset;
    float    sphere[4];  // world bounding sphere
    uint64_t moved;      // frame number it last moved, appeared or changed mesh
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
    uint32_t source;     // the core's own reference (its light row)
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

    // Geometry, for shadow rendering. Vertex: float3 position, float3 normal (vertexStride bytes).
    Buffer*               vertices;
    Buffer*               indices;     // u32
    Buffer*               instances;   // RenderGpuInstance[]
    uint32_t              vertexStride;
    const RenderCaster* casters;
    uint32_t              casterCount;
    uint64_t              casterRemoved; // frame a caster was last removed
    float                 sceneSphere[4];

    // Local lights (in), capped by culling.
    RenderLight* lights;
    uint32_t       lightCount;

    // Camera depth (reversed-Z), in SHADER_RESOURCE state while renderer_culling runs.
    RenderTarget* depth;
    Buffer*       lightBuffer; // RenderGpuLight[], this frame's

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

    // TAA (renderer_taa). taa_prepare runs first, with only the fields above the camera set: it sets
    // jitter, taaOn, sharpen and resolved, and the core then renders at one sample with motion vectors.
    float         jitter[2];  // sub-pixel camera offset this frame, in pixels
    uint32_t      taaOn;
    float         sharpen;    // composite sharpening, 0..1
    RenderTarget* hdr;        // the frame's HDR colour, SHADER_RESOURCE when taa_resolve runs
    RenderTarget* motion;     // RG16F: uv now - uv last frame, SHADER_RESOURCE; depth too
    Buffer*       exposure;   // float[]: [1] the exposure in use
    RenderTarget* resolved;   // the target taa_resolve writes the anti-aliased HDR image to; SHADER_RESOURCE after

    // Ambient occlusion (renderer_ao). ao_prepare runs after cull_prepare (jitter and taaOn are set);
    // ao_run right after it, after the depth prepass, with depth in SHADER_RESOURCE.
    Texture* ao;            // R32F visibility (1: open), full size, SHADER_RESOURCE after ao_run
    uint32_t aoOn;
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
};
