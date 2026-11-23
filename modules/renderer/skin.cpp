// Skinning (skin.h). A skinned mesh is an ordinary mesh in the pool (its bind pose) with a SkinVertex row per vertex.
// renderer_models hands each animated instance its palette every frame (instance_pose); this frame's posed instances
// get regions at the end of the pool's vertex buffers, laid out in instance order, and one compute dispatch writes
// them first thing in the frame. Every pass then draws them from there: the core through skinInstance's offsets,
// every other module through its caster's vertexOffset.
#include "yea_renderer.h"
#include "scene.h"
#include "skin.h"

#include <math.h>
#include <string.h>

#include <Common_3/Graphics/Interfaces/IGraphics.h>
#include <Common_3/Resources/ResourceLoader/Interfaces/IResourceLoader.h>
#include <Common_3/Graphics/FSL/defaults.h>

#include "Shaders/Skin.srt.h"

namespace renderer
{
namespace
{
constexpr uint32_t kMaxFrames = 4;
constexpr uint32_t kPaletteFloats = 32768 * 12; // bones posed a frame, over every instance
constexpr uint32_t kMaxJobs = 4096;

struct Pose // an instance row's palette this frame
{
    uint32_t offset, bones; // into gPoseCpu (floats)
    uint64_t frame;
};
struct Last // where an instance row drew from last frame
{
    uint32_t base, lod, gen;
    uint64_t frame;
};
struct Job // skin.comp's two uint4s
{
    uint32_t src, count, dst, firstGroup;
    uint32_t dst2, skin, palette, pad;
};

Renderer*      gRenderer;
uint32_t       gFrames;
Shader*        gShader;
Pipeline*      gPipe;
DescriptorSet* gSet;
Buffer*        gPaletteBuf[kMaxFrames];
Buffer*        gJobBuf[kMaxFrames];
Buffer*        gSkinBuf; // the SkinVertex table on the GPU
uint32_t       gSkinRows, gSkinEpoch; // what it holds
float*         gPoseCpu;              // this frame's palettes (made in load(); instance_pose copies into it)
uint32_t       gPoseUsed;
uint64_t       gPoseFrame;
Pose*          gPose; // per instance row
Last*          gLast;
Job*           gJobs; // this frame's, from index 1 (index 0's first word: their count + 1)
uint32_t       gJobCount, gGroups;
uint32_t       gBase, gCapacity, gCursor;
uint64_t       gFrame;

template<class T> uint32_t rowOf(const yea::Table<T>& t, yea_handle h)
{
    uint32_t row = h & 0xFFFFFu;
    if (row >= t.size() || !t[row].alive || (t[row].gen & 0xFFFu) != (h >> 20))
        return 0xFFFFFFFFu;
    return row;
}

bool posed(uint32_t row)
{
    const Instance& in = Instances[row];
    return in.alive && gPose[row].frame == gFrame && in.mesh < Meshes.size() && in.mesh < MeshSkins.size() && MeshSkins[in.mesh].firstSkin;
}
} // namespace

// --- Exports -----------------------------------------------------------------------------------

void mesh_skin(uint32_t mesh, uint64_t skin, yea_float3 lo, yea_float3 hi)
{
    const uint32_t* s = (const uint32_t*)(uintptr_t)skin;
    if (!s || mesh >= Meshes.size())
        return;
    const Mesh& m = Meshes[mesh];
    if (SkinVertexes.size() + (uint64_t)m.vertexCount > SkinVertexes.capacity())
        return; // the table is full: it stays in its bind pose
    while (MeshSkins.size() <= mesh && MeshSkins.add())
        MeshSkins[MeshSkins.size() - 1].firstSkin = 0;
    if (mesh >= MeshSkins.size())
        return;
    MeshSkins[mesh].firstSkin = SkinVertexes.size() + 1;
    for (uint32_t v = 0; v < m.vertexCount; ++v)
    {
        SkinVertex* o = SkinVertexes.add();
        o->bones01 = s[v * 4], o->bones23 = s[v * 4 + 1], o->weights01 = s[v * 4 + 2], o->weights23 = s[v * 4 + 3];
    }
    // Its bounds are those its animation reaches, so culling and shadows never lose it mid-stride.
    Mesh& w = Meshes[mesh];
    w.center = { (lo.x + hi.x) * 0.5f, (lo.y + hi.y) * 0.5f, (lo.z + hi.z) * 0.5f };
    const float dx = hi.x - lo.x, dy = hi.y - lo.y, dz = hi.z - lo.z;
    const float r = 0.5f * sqrtf(dx * dx + dy * dy + dz * dz);
    w.radius = r > w.radius ? r : w.radius;
}

void instance_pose(yea_handle h, uint64_t palette, uint32_t bones)
{
    const uint32_t row = rowOf(Instances, h);
    const float*   p = (const float*)(uintptr_t)palette;
    if (row == 0xFFFFFFFFu || !p || !bones || !gPoseCpu)
        return;
    const uint64_t frame = yea::host()->frame();
    if (frame != gPoseFrame)
        gPoseFrame = frame, gPoseUsed = 0;
    if (gPoseUsed + bones * 12 > kPaletteFloats)
        return; // over the frame's room: this one stays in its bind pose
    memcpy(gPoseCpu + gPoseUsed, p, (size_t)bones * 12 * sizeof(float));
    gPose[row] = { gPoseUsed, bones, frame };
    gPoseUsed += bones * 12;
}

// --- The GPU side ------------------------------------------------------------------------------

bool skinLoad(Renderer* renderer, uint32_t frames)
{
    gRenderer = renderer;
    gFrames = frames < kMaxFrames ? frames : kMaxFrames;
    gPoseCpu = new float[kPaletteFloats];
    gPose = new Pose[Instances.capacity()]();
    gLast = new Last[Instances.capacity()]();
    gJobs = new Job[kMaxJobs + 1]();
    gPoseUsed = 0, gPoseFrame = 0, gSkinRows = 0, gSkinEpoch = 0;
    ShaderLoadDesc sd = {};
    sd.mComp.pFileName = "renderer_skin.comp";
    addShader(gRenderer, &sd, &gShader);
    if (!gShader)
        return false;
    PipelineDesc desc = {};
    desc.mType = PIPELINE_TYPE_COMPUTE;
    PIPELINE_LAYOUT_DESC(desc, NULL, SRT_LAYOUT_DESC(SkinSrt, PerFrame), NULL, NULL);
    desc.mComputeDesc.pShaderProgram = gShader;
    desc.pName = "renderer_skin";
    addPipeline(gRenderer, &desc, &gPipe);
    DescriptorSetDesc set = SRT_SET_DESC(SkinSrt, PerFrame, gFrames, 0);
    addDescriptorSet(gRenderer, &set, &gSet);
    for (uint32_t i = 0; i < gFrames; ++i)
    {
        BufferLoadDesc d = {};
        d.mDesc.mDescriptors = DESCRIPTOR_TYPE_BUFFER;
        d.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
        d.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
        d.mDesc.mStartState = RESOURCE_STATE_SHADER_RESOURCE;
        d.mDesc.mFormat = TinyImageFormat_R32G32B32A32_SFLOAT;
        d.mDesc.mSize = (uint64_t)kPaletteFloats * 4;
        d.mDesc.mElementCount = kPaletteFloats / 4;
        d.mDesc.mStructStride = 16;
        d.mDesc.pName = "renderer_skin_palette";
        d.ppBuffer = &gPaletteBuf[i];
        addResource(&d, nullptr);
        d.mDesc.mFormat = TinyImageFormat_R32G32B32A32_UINT;
        d.mDesc.mSize = (uint64_t)(kMaxJobs + 1) * sizeof(Job);
        d.mDesc.mElementCount = (kMaxJobs + 1) * 2;
        d.mDesc.pName = "renderer_skin_jobs";
        d.ppBuffer = &gJobBuf[i];
        addResource(&d, nullptr);
    }
    waitForAllResourceLoads();
    bool ok = gPipe && gSet;
    for (uint32_t i = 0; i < gFrames; ++i)
        ok = ok && gPaletteBuf[i] && gPaletteBuf[i]->pCpuMappedAddress && gJobBuf[i] && gJobBuf[i]->pCpuMappedAddress;
    return ok;
}

void skinUnload()
{
    for (uint32_t i = 0; i < kMaxFrames; ++i)
    {
        if (gPaletteBuf[i])
            removeResource(gPaletteBuf[i]), gPaletteBuf[i] = nullptr;
        if (gJobBuf[i])
            removeResource(gJobBuf[i]), gJobBuf[i] = nullptr;
    }
    if (gSkinBuf)
        removeResource(gSkinBuf), gSkinBuf = nullptr;
    if (gSet)
        removeDescriptorSet(gRenderer, gSet), gSet = nullptr;
    if (gPipe)
        removePipeline(gRenderer, gPipe), gPipe = nullptr;
    if (gShader)
        removeShader(gRenderer, gShader), gShader = nullptr;
    delete[] gPoseCpu;
    delete[] gPose;
    delete[] gLast;
    delete[] gJobs;
    gPoseCpu = nullptr, gPose = nullptr, gLast = nullptr, gJobs = nullptr;
}

uint32_t skinNeed()
{
    gFrame = yea::host()->frame();
    if (!gPose || !SkinVertexes.size())
        return 0;
    uint64_t need = 0;
    for (uint32_t row = 0; row < Instances.size(); ++row)
        if (posed(row))
            need += 2ull * Meshes[Instances[row].mesh].vertexCount;
    return need > 0x0FFFFFFFu ? 0x0FFFFFFFu : (uint32_t)need;
}

void skinBegin(uint32_t base, uint32_t capacity)
{
    gBase = base, gCapacity = capacity, gCursor = 0, gJobCount = 0, gGroups = 0;
}

bool skinInstance(uint32_t row, uint32_t lod, SkinDraw& out)
{
    if (!gPose || !gCapacity || !posed(row) || lod >= Meshes.size() || lod >= MeshSkins.size() || !MeshSkins[lod].firstSkin ||
        gJobCount >= kMaxJobs || !gPipe)
        return false;
    const Instance& in = Instances[row];
    const uint32_t  stride = Meshes[in.mesh].vertexCount, count = Meshes[lod].vertexCount;
    if (count > stride || gCursor + 2ull * stride > gCapacity || MeshSkins[lod].firstSkin - 1 + (uint64_t)count > SkinVertexes.size())
        return false;
    const uint32_t base = gBase + gCursor;
    gCursor += 2 * stride;
    const uint32_t copy = (uint32_t)(gFrame & 1);
    out = { base, stride, copy, (int32_t)(base + copy * stride), (int32_t)(base + (1 - copy) * stride) };
    // A region new to it (moved, another version, or not posed last frame): both copies, so last frame's reads the same.
    Last&      l = gLast[row];
    const bool fresh = l.base != base || l.lod != lod || l.gen != in.gen || l.frame + 1 != gFrame;
    l = { base, lod, in.gen, gFrame };
    Job& j = gJobs[1 + gJobCount++];
    j = { Meshes[lod].firstVertex, count, (uint32_t)out.current, gGroups,
          fresh ? (uint32_t)out.previous : 0xFFFFFFFFu, MeshSkins[lod].firstSkin - 1, gPose[row].offset / 4, 0 };
    gGroups += (count + 63) / 64;
    return true;
}

bool skinJobs() { return gJobCount > 0; }

void skinRecord(Cmd* cmd, uint32_t fi, Buffer* positions, Buffer* attributes, uint32_t stateBits)
{
    if (!gJobCount || !positions || !attributes)
        return;
    const ResourceState state = (ResourceState)stateBits;
    // The bones' buffer, made again whenever the table grew or the scene started over (rare: a model loading).
    const uint32_t rows = SkinVertexes.size(), epoch = scene_epoch();
    if (!gSkinBuf || rows != gSkinRows || epoch != gSkinEpoch)
    {
        if (gSkinBuf)
        {
            waitQueueIdle((Queue*)yea::host()->queue);
            removeResource(gSkinBuf), gSkinBuf = nullptr;
        }
        BufferLoadDesc d = {};
        d.mDesc.mDescriptors = DESCRIPTOR_TYPE_BUFFER;
        d.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_GPU_ONLY;
        d.mDesc.mStartState = RESOURCE_STATE_SHADER_RESOURCE;
        d.mDesc.mFormat = TinyImageFormat_R32G32B32A32_UINT;
        d.mDesc.mSize = (uint64_t)rows * 16;
        d.mDesc.mElementCount = rows;
        d.mDesc.mStructStride = 16;
        d.mDesc.pName = "renderer_skin_vertices";
        d.pData = SkinVertexes.begin();
        d.ppBuffer = &gSkinBuf;
        addResource(&d, nullptr);
        waitForAllResourceLoads();
        gSkinRows = rows, gSkinEpoch = epoch;
        if (!gSkinBuf)
        {
            gJobCount = 0;
            return;
        }
    }
    memcpy(gPaletteBuf[fi]->pCpuMappedAddress, gPoseCpu, (size_t)gPoseUsed * sizeof(float));
    gJobs[0] = Job{};
    gJobs[0].src = gJobCount; // skin.comp's [0].x
    memcpy(gJobBuf[fi]->pCpuMappedAddress, gJobs, (size_t)(gJobCount + 1) * sizeof(Job));
    DescriptorData p[5] = {};
    p[0].mIndex = SRT_RES_IDX(SkinSrt, PerFrame, gPositions);
    p[0].ppBuffers = &positions;
    p[1].mIndex = SRT_RES_IDX(SkinSrt, PerFrame, gAttributes);
    p[1].ppBuffers = &attributes;
    p[2].mIndex = SRT_RES_IDX(SkinSrt, PerFrame, gSkin);
    p[2].ppBuffers = &gSkinBuf;
    p[3].mIndex = SRT_RES_IDX(SkinSrt, PerFrame, gPalette);
    p[3].ppBuffers = &gPaletteBuf[fi];
    p[4].mIndex = SRT_RES_IDX(SkinSrt, PerFrame, gJobs);
    p[4].ppBuffers = &gJobBuf[fi];
    updateDescriptorSet(gRenderer, fi, gSet, 5, p);
    BufferBarrier in[2] = { { positions, state, RESOURCE_STATE_UNORDERED_ACCESS }, { attributes, state, RESOURCE_STATE_UNORDERED_ACCESS } };
    cmdResourceBarrier(cmd, 2, in, 0, nullptr, 0, nullptr);
    cmdBindPipeline(cmd, gPipe);
    cmdBindDescriptorSet(cmd, fi, gSet);
    cmdDispatch(cmd, gGroups, 1, 1);
    BufferBarrier out[2] = { { positions, RESOURCE_STATE_UNORDERED_ACCESS, state }, { attributes, RESOURCE_STATE_UNORDERED_ACCESS, state } };
    cmdResourceBarrier(cmd, 2, out, 0, nullptr, 0, nullptr);
}
} // namespace renderer
