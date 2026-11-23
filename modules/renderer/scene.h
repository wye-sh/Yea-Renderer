// What the scene exports (scene.cpp) share with the renderer (renderer.cpp).
#pragma once
#include "yea_renderer.h"

typedef struct Texture Texture;
typedef struct Queue   Queue;

namespace renderer
{
struct CameraState
{
    yea_float3 eye;
    yea_float3 target;
    float      fov;
};
CameraState* cameraState();

struct SceneState
{
    uint64_t removed; // frame an instance was last removed (shadow caches redraw)
};
SceneState* sceneState();
// A number every module that adds to the scene keeps: it changes whenever the scene it built is gone (clear(), Reset
// State, a material layout change), never on a reload that keeps it. Never 0.
struct EpochState
{
    uint32_t epoch;
};
bool        meshBuilding(); // a custom mesh is half made
void        bindState();    // takes the state blocks: call in load(), before anything uses them

// Rect lights' images (images.cpp).
void     lightImagesInit();                  // in load()
void     lightImagesFree();                  // in unload()
Texture* lightImagesMake();                  // the texture array (16 images of 256 x 256, with mips)
void     lightImagesUpload(Texture* tex);    // before the frame's lights: uploads when the images changed
uint32_t lightImageOf(uint32_t lightRow);    // a rect light row's image + 1 (uploaded), else 0
void     lightImageForget(uint32_t lightRow); // a light row made afresh has none

// Texture maps and material maps (textures.cpp).
void              texturesInit(Texture* standIn); // in load(): standIn fills the slots without a texture
void              texturesFree();                 // in unload(), the GPU idle
void              texturesUpdate(Queue* queue);   // in draw, before the frame: frees gone ones, loads waiting ones
Texture* const*   textureSlots();                 // RENDER_MAX_TEXTURES of them, for the shaders' array
uint32_t          textureVersion();               // changes whenever textureSlots() does
uint32_t          textureSlot(yea_handle t);      // a texture's slot + 1 once it is drawn with, else 0
void              materialMapsReset(uint32_t materialRow); // a material row made afresh has no maps (and is opaque)
float             materialOpacity(uint32_t materialRow);   // material_blend's opacity, 0: opaque
bool              materialRefraction(uint32_t materialRow, float out[8]); // ior, thickness, shape, 1 given, tint rgb, 0;
                                                                          // true when given
} // namespace renderer
