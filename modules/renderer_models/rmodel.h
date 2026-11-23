// The cooked model file (.rmodel), written by the component's model cooker (tools/model_cook) and read by
// renderer_models. Little-endian, every section 16-byte aligned, offsets from the file's start. Raise kRmodelVersion on
// any change; the loader refuses versions it can't read (kRmodelOldest to kRmodelVersion; older ones are cooked again).
//
// A model is parts (one per material: the cooker merges a static model's meshes by material), each a chain of LODs, the
// finest first. Each LOD is its own vertex range (positions float3, attributes as frame.h's RENDER_ATTRIBUTE_STRIDE) and
// index range (u32, from the LOD's first vertex). Units are metres and axes the renderer's (+Y up, left-handed, the
// model's front facing -Z), whatever the source file used; triangles are counter-clockwise around their outward normal.
//
// v3 adds skinning and animation (flags RmodelSkinned; a v2 file is a v3 file without them, its last two header words
// zero). A skinned model's vertices are in its bind pose and carry a third stream, `skin` (RmodelSkinVertex, one per
// vertex in the same order); its skeleton (RmodelSkeleton) holds the nodes (the file's node hierarchy, parents before
// children, each with its rest pose), the bones (the skinning palette: a node and its inverse bind matrix; a vertex names
// up to 4 bones) and the clips. A posed vertex is sum(weight * world(bone.node) * bone.invBind * position). Rigid meshes
// under an animated node are skinned to one bone with weight 1, so every mesh of an animated model moves the same way.
//
// v4 adds morph targets (blend shapes): RmodelSkeleton's `targetCount` and `targets` (an RmodelMorphs header, then its
// RmodelMorphTarget names and default weights). Each vertex has a u32 in `refs` saying where its RmodelMorphDelta rows
// are (RmodelMorphs). A morphed vertex is (bind position + sum(weight[target] * delta)) before skinning; its normal the same, then
// normalised. A model with morph targets is always cooked skinned (a mesh without a skin bound whole to its node). Each
// clip holds a weight track per target after its node tracks. A v3 file is a v4 file without targets.
#pragma once

#include <stdint.h>

constexpr uint32_t kRmodelMagic = 0x4C444D52u; // "RMDL"
constexpr uint32_t kRmodelVersion = 4;
constexpr uint32_t kRmodelOldest = 2; // the oldest version the loader still reads

struct RmodelHeader
{
    uint32_t magic, version, headerBytes, fileBytes;
    uint32_t materialCount, partCount, lodCount, flags; // flags: RmodelFlags
    uint32_t vertexCount, indexCount;                   // totals, over every LOD
    float    lo[3], hi[3];                              // bounds of the finest LODs
    uint32_t materials, parts, lods;                    // section offsets
    uint32_t positions, attributes, indices;
    uint32_t clusters, clusterBytes; // reserved for cluster LOD (meshlet hierarchy): 0 for now
    uint32_t textureCount, textures; // RmodelTexture[] (its offset); each one's mips follow at its own offset
    uint32_t skin;     // RmodelSkinVertex[vertexCount] (its offset), when RmodelSkinned; else 0
    uint32_t skeleton; // RmodelSkeleton (its offset), when RmodelSkinned; else 0
};

enum RmodelFlags : uint32_t
{
    RmodelSkinned = 1, // a skin stream and a skeleton: the model moves (skinned meshes, or rigid ones under animated nodes)
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
    int32_t  textures[4]; // RmodelTexture indices, -1 none: base colour (BC7 sRGB, alpha), ORM (BC7 linear: occlusion,
                          // roughness, metallic; glTF's occlusion map merged in), normal (BC5: x, y; z rebuilt), emissive (BC7 sRGB)
    char     name[64];
    // Its look with each texture averaged over its image: the colour (linear rgb, factor times texture), roughness and
    // metallic a renderer without the textures draws (stage 1). The factors alone when it has none.
    float    flatColor[3];
    float    flatRoughness, flatMetallic;
    float    normalScale;       // glTF's normalTexture.scale: the normal map's x and y times it
    float    occlusionStrength; // glTF's occlusionTexture.strength: lerp(1, occlusion, it)
};

enum RmodelFormat : uint32_t
{
    RmodelBC7Srgb = 1,
    RmodelBC7Linear = 2,
    RmodelBC5 = 3, // two channels, unsigned (a normal's x, y as 0..1)
};

// A texture: its full mip chain, the largest first, each mip's 4x4 blocks row by row (16 bytes a block), mips packed
// one after another from `offset`. Width and height are the top mip's; a mip is max(1, size >> level) wide, in
// max(1, (size + 3) / 4) blocks. The loader may skip top mips to keep within memory (the cooker never shrinks them).
struct RmodelTexture
{
    uint32_t format; // RmodelFormat
    uint32_t width, height, mips;
    uint64_t offset, bytes; // from the file's start, every mip
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

// Skinning (v3). Bones index RmodelSkeleton's bones; weights are unorm16 and sum to exactly 65535; unused slots weigh 0.
struct RmodelSkinVertex
{
    uint16_t bones[4];
    uint16_t weights[4];
};

struct RmodelSkeleton
{
    uint32_t nodeCount, boneCount, clipCount, targetCount; // targetCount: morph targets (v4; 0 before)
    uint32_t nodes, bones, clips, targets; // offsets: RmodelNode[], RmodelBone[], RmodelClip[], RmodelMorphs (0: none)
};

// Morph targets (v4). `refs` is a u32 a vertex (h.vertexCount, the vertices' order): its deltas' first row (the low 24
// bits, from `deltas`) and their count (the top 8; 0: it doesn't morph). A vertex's deltas name distinct targets.
// RmodelMorphTarget[targetCount] follows this header.
struct RmodelMorphs
{
    uint32_t refs, deltas, deltaCount, pad; // offsets from the file's start
};
struct RmodelMorphTarget
{
    char  name[48];
    float weight; // its default (glTF's mesh.weights): what it is when no clip says
    float lo, hi; // the weights its clips reach, with the default (bounds use them)
    float pad;
};
// A vertex's offset for one target, in the model's space as the vertices are: position (metres) and normal (the morphed
// normal minus the bind one, each component snorm8 over +-2) in `packed` with the target: target | nx << 8 | ny << 16 |
// nz << 24.
struct RmodelMorphDelta
{
    float    p[3];
    uint32_t packed;
};
constexpr uint32_t kRmodelMaxTargets = 255;

// A node of the hierarchy, in the renderer's space: its rest pose relative to its parent (scale, then rotation, then
// translation). Parents come before their children; the roots' parent is -1, and their transforms are in model space.
struct RmodelNode
{
    int32_t  parent;
    float    t[3];
    float    r[4]; // quaternion x, y, z, w
    float    s[3];
    char     name[48];
    uint32_t pad;
};

// A bone of the skinning palette: its node and inverse bind matrix (3x4, rows, model space at bind time to the node's).
struct RmodelBone
{
    float    invBind[12];
    uint32_t node;
    uint32_t pad[3];
};

enum RmodelCodec : uint32_t
{
    RmodelCodecQuantized = 0, // the Renderer's own (anim.h): uniformly resampled, smallest-three rotations, quantised tracks
};

// A clip: every node's local pose resampled at `rate` frames a second, `frames` of them, the first at 0 and the last at
// `duration` (so rate = (frames - 1) / duration). Its data (`bytes` at `offset`) is in `codec`'s layout (anim.h).
struct RmodelClip
{
    char     name[48];
    float    duration, rate;
    uint32_t frames, codec;   // RmodelCodec
    uint64_t offset, bytes;
};

// RmodelCodecQuantized's tracks: three a node (translation, rotation, scale), nodeCount * 3 of them at the clip's offset,
// then (v4) one a morph target (its weight: base[0] + extent[0] * u / 65535, one u16 a frame), then their samples. A constant track has `data` 0xFFFFFFFF and its value in `base` (xyz, or the quaternion). Otherwise
// `data` is the byte offset, from the clip's offset, of `frames` samples of three u16 each: a translation or scale is
// base + extent * u / 65535; a rotation is smallest-three (anim.h's decodeRotation).
struct RmodelTrack
{
    float    base[4];
    float    extent[3];
    uint32_t data;
};

static_assert(sizeof(RmodelHeader) == 112, "rmodel header");
static_assert(sizeof(RmodelMaterial) == 160, "rmodel material");
static_assert(sizeof(RmodelPart) == 48, "rmodel part");
static_assert(sizeof(RmodelLod) == 32, "rmodel lod");
static_assert(sizeof(RmodelTexture) == 32, "rmodel texture");
static_assert(sizeof(RmodelSkinVertex) == 16, "rmodel skin vertex");
static_assert(sizeof(RmodelSkeleton) == 32, "rmodel skeleton");
static_assert(sizeof(RmodelNode) == 96, "rmodel node");
static_assert(sizeof(RmodelBone) == 64, "rmodel bone");
static_assert(sizeof(RmodelClip) == 80, "rmodel clip");
static_assert(sizeof(RmodelTrack) == 32, "rmodel track");
static_assert(sizeof(RmodelMorphs) == 16, "rmodel morphs");
static_assert(sizeof(RmodelMorphTarget) == 64, "rmodel morph target");
static_assert(sizeof(RmodelMorphDelta) == 16, "rmodel morph delta");
