// Skinning (skin.cpp, frame v24): skinned meshes' bones (mesh_skin), each instance's palette this frame
// (instance_pose), and the compute pass that poses them into regions of their own at the end of the vertex pool.
#pragma once
#include <stdint.h>

struct Renderer;
struct Cmd;
struct Buffer;

namespace renderer
{
// Where a skinned instance draws from this frame. Its region holds two copies of its finest version's vertex count
// (`stride`): this frame's at base + copy * stride, last frame's at the other copy.
struct SkinDraw
{
    uint32_t base, stride, copy;
    int32_t  current; // base + copy * stride: the vertex offset every other pass draws its caster with
    int32_t  previous;
};

bool     skinLoad(Renderer* renderer, uint32_t frames); // in load()
void     skinUnload();                                  // in unload(), the GPU idle
// Vertices this frame's posed instances need in the dynamic region (two copies each), before uploadMeshes sizes it.
uint32_t skinNeed();
// After uploadMeshes: the dynamic region starts at `base` (vertices) and holds `capacity`; jobs start afresh.
void     skinBegin(uint32_t base, uint32_t capacity);
// In the instance loop: whether instance `row`, drawn with mesh version `lod`, is posed this frame, and where from.
bool     skinInstance(uint32_t row, uint32_t lod, SkinDraw& out);
// First in the frame's commands: the bones' buffer brought up to date, then every job, with the pool's buffers moved
// from `state` to unordered access and back.
void     skinRecord(Cmd* cmd, uint32_t fi, Buffer* positions, Buffer* attributes, uint32_t state);
bool     skinJobs(); // anything to record this frame
// The Model Viewer's preview (preview_draw): its skinned parts (meshes[k], the versions it draws) posed by `palette` into
// a region kept after the instances', with a dispatch of its own; offsets[k] each part's first vertex there, -1 where it
// draws its bind pose (not skinned, or no room or bones yet: a frame or two after it first asks). Allocates nothing.
uint32_t skinPreview(Cmd* cmd, uint32_t fi, Buffer* positions, Buffer* attributes, uint32_t state, const uint32_t* meshes,
                     uint32_t parts, const float* palette, uint32_t bones, int32_t* offsets);
} // namespace renderer
