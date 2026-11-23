// renderer_models' registry and preview, shared by renderer_models.cpp and the Model Viewer (panels/Viewer.cpp).
#pragma once

#include "ModelLink.h"
#include "preview.h"

#include <stdint.h>

namespace renderer_models
{
constexpr uint32_t kMaxModels = 512;
constexpr uint32_t kMaxParts = RENDER_PREVIEW_PARTS;

struct Entry
{
    char     owner[64], name[64], source[160], assets[1024];
    char     error[160];
    uint32_t id, key;
    uint32_t kind;  // as mdl::Info
    uint32_t state; // 0 waiting, 1 ready, 2 failed
    uint32_t defined; // in the owner's latest begin()..end()
    float    knobs[mdl::kKnobs];
    uint32_t parts;
    uint32_t mesh[kMaxParts], material[kMaxParts];
    float    lo[3], hi[3]; // its bounds as cooked (object units)
};
// Every model the view's modules declared, in the order declared (made in load()).
extern Entry*   gEntries;
extern uint32_t gEntryCount;

// The Model Viewer's preview: what the panel asks for this frame, drawn in this module's draw(YEA_PASS_RENDER).
struct Preview
{
    uint32_t shown;          // the panel showed the image this frame
    uint32_t model;          // its id
    uint32_t width, height;  // wanted (pixels); targets follow in update()
    float    yaw, pitch;     // degrees
    float    distance;       // 1: its bounding sphere fills the preview's height; more is further
    int32_t  lod;            // -1 as the error on screen picks
    uint32_t turntable;      // turning by itself
    uint32_t lodShown, triangles; // what the last draw drew
    const void* outTexture;  // what the panel shows: the target's Texture* (nullptr until made)
    uint32_t    outW, outH;
};
extern Preview gPreview;

// A model's lods (its first part's chain), counted from the renderer's MeshLod table.
uint32_t lodCount(const Entry& e);
uint32_t triangles(const Entry& e); // its finest version's, every part
} // namespace renderer_models
