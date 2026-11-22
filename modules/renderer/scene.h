// What the scene exports (scene.cpp) share with the renderer (renderer.cpp).
#pragma once
#include "yea_renderer.h"

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
bool        meshBuilding(); // a custom mesh is half made
void        bindState();    // takes the state blocks: call in load(), before anything uses them
} // namespace renderer
