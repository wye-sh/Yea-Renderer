// The cooked model file (.rmodel), written by the component's model cooker (tools/model_cook) and read by
// renderer_models. Little-endian, every section 16-byte aligned, offsets from the file's start. Raise kRmodelVersion on
// any change; the loader refuses other versions (the model is cooked again).
//
// A model is parts (one per material: the cooker merges a static model's meshes by material), each a chain of LODs, the
// finest first. Each LOD is its own vertex range (positions float3, attributes as frame.h's RENDER_ATTRIBUTE_STRIDE) and
// index range (u32, from the LOD's first vertex). Units are metres and axes the renderer's (+Y up, left-handed, the
// model's front facing -Z), whatever the source file used; triangles are counter-clockwise around their outward normal.
#pragma once

#include <stdint.h>

constexpr uint32_t kRmodelMagic = 0x4C444D52u; // "RMDL"
constexpr uint32_t kRmodelVersion = 1;

struct RmodelHeader
{
    uint32_t magic, version, headerBytes, fileBytes;
    uint32_t materialCount, partCount, lodCount, flags; // flags: reserved (skinned, clusters)
    uint32_t vertexCount, indexCount;                   // totals, over every LOD
    float    lo[3], hi[3];                              // bounds of the finest LODs
    uint32_t materials, parts, lods;                    // section offsets
    uint32_t positions, attributes, indices;
    uint32_t clusters, clusterBytes; // reserved for cluster LOD (meshlet hierarchy): 0 for now
    uint32_t reserved[4];
};

enum RmodelAlpha : uint32_t
{
    RmodelOpaque = 0,
    RmodelMask = 1,
    RmodelBlend = 2,
};

struct RmodelMaterial
{
    float    baseColor[4]; // linear rgb, alpha
    float    roughness, metallic;
    float    emissive[3];      // linear rgb as in the file (glTF emissiveFactor)
    float    emissiveStrength; // KHR_materials_emissive_strength, 1 without
    uint32_t alpha;            // RmodelAlpha
    float    alphaCutoff;
    uint32_t doubleSided;
    int32_t  textures[4]; // base colour, ORM, normal, emissive: -1 (textures come with stage 2)
    char     name[64];
    // Its look with each texture averaged over its image: the colour (linear rgb, factor times texture), roughness and
    // metallic a renderer without the textures draws (stage 1). The factors alone when it has none.
    float    flatColor[3];
    float    flatRoughness, flatMetallic;
    uint32_t pad[2];
};

struct RmodelPart
{
    uint32_t material, firstLod, lodCount, pad;
    float    lo[3], hi[3];
    uint32_t pad2[2];
};

struct RmodelLod
{
    uint32_t firstVertex, vertexCount, firstIndex, indexCount;
    float    error; // how far it strays from the finest (object units, metres): the core draws the simplest within lod_error
    uint32_t pad[3];
};

static_assert(sizeof(RmodelHeader) == 112, "rmodel header");
static_assert(sizeof(RmodelMaterial) == 160, "rmodel material");
static_assert(sizeof(RmodelPart) == 48, "rmodel part");
static_assert(sizeof(RmodelLod) == 32, "rmodel lod");
