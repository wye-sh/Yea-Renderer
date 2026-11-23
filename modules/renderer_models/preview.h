// The Model Viewer's preview: renderer_models asks the core (renderer's preview_draw) to draw one model alone, with the
// core's BRDF, materials, meshes and tonemapper, lit by a studio rig, into renderer_models' own targets. Copied
// identically into renderer and renderer_models; raise RENDER_PREVIEW_VERSION on any change.
//
// renderer_models makes every target in its own load() or update() (never in a draw): `hdr` RGBA16F at 4 samples
// (clear 0, 0, 0, 0), `depth` D32 at 4 samples (clear 0: reversed Z), `shadow` D32 at 1 sample (clear 1), `out`
// R8G8B8A8_SRGB, each width x height but `shadow` (square, any size). All start and end in SHADER_RESOURCE but the depth
// targets, which start and end in DEPTH_WRITE. The core makes its pipelines and buffers in its own load().
#pragma once

#include <stdint.h>

#define RENDER_PREVIEW_VERSION 2u
#define RENDER_PREVIEW_PARTS 64u

struct Cmd;
struct RenderTarget;

struct RenderPreview
{
    uint32_t      version; // RENDER_PREVIEW_VERSION
    uint32_t      frameIndex;
    Cmd*          cmd;     // the host's, in renderer_models' draw(YEA_PASS_RENDER); nothing bound before or after
    uint32_t      width, height;
    RenderTarget *hdr, *depth, *shadow, *out;
    float         eye[3], at[3], fov; // degrees
    float         center[3], radius;  // the model's bounding sphere, placed (world)
    float         floorY;             // where its floor is (its lowest point)
    float         sunDir[3];          // towards the key light (normalised by the core)
    float         sunLux;             // the key light's illuminance; the studio's sky is a neutral grey from skyLuminance
    float         skyLuminance;
    float         lodQuality;         // the model's LOD Quality knob
    int32_t       lod;                // -1 as the error on screen picks (the core's lod_error), else that version
    uint32_t      parts;
    uint32_t      mesh[RENDER_PREVIEW_PARTS];          // the core's mesh ids (each the finest of its chain)
    uint32_t      material[RENDER_PREVIEW_PARTS];      // the core's material handles
    uint32_t      twoSided[RENDER_PREVIEW_PARTS];      // drawn from both sides
    float         transform[RENDER_PREVIEW_PARTS][12]; // object to world, 3 rows of 4
    // Written by preview_draw, for the panel: the version drawn (0 the finest; the deepest of any part) and the
    // triangles drawn.
    uint32_t      lodShown, triangles;
    // A skinned model posed (v2): its palette this frame (3 rows of 4 floats a bone, model space, as instance_pose's),
    // read during the call; 0 bones draws the bind pose. preview_draw poses its skinned parts into a region of its own.
    uint32_t      bones, pad;
    uint64_t      palette;
};
