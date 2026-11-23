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
constexpr uint32_t kPreviewBones = 1024; // the Model Viewer's model: bones posed at most
constexpr uint32_t kPreviewJobs = 64;    // RENDER_PREVIEW_PARTS

struct Pose // an instance row's palette this frame
{
    uint32_t offset, bones; // into gPoseCpu (floats)
    uint64_t frame;
    uint32_t weights, targets; // its morph weights in gPoseCpu (floats, from a float4), when weightFrame is this frame
    uint64_t weightFrame;
    float    lo[3], hi[3]; // instance_bounds', when boundsFrame is this frame
    uint64_t boundsFrame;
};
struct Last // where an instance row drew from last frame
{
    uint32_t base, lod, gen;
    uint64_t frame;
};
struct Job // skin.comp's three uint4s
{
    uint32_t src, count, dst, firstGroup;
    uint32_t dst2, skin, palette, pad;
    uint32_t morph, delta, weights, pad2; // morph: its first MorphVertex row (0xFFFFFFFF none); weights in float4s
};
constexpr uint32_t kNoMorph = 0xFFFFFFFFu;

Renderer*      gRenderer;
uint32_t       gFrames;
Shader*        gShader;
Pipeline*      gPipe;
DescriptorSet* gSet;
Buffer*        gPaletteBuf[kMaxFrames];
Buffer*        gJobBuf[kMaxFrames];
Buffer*        gSkinBuf; // the SkinVertex table on the GPU
uint32_t       gSkinRows, gSkinEpoch; // what it holds
Buffer *       gMorphVertBuf, *gMorphDeltaBuf; // the MorphVertex and MorphDelta tables on the GPU (one row at least)
uint32_t       gMorphRows, gMorphDeltaRows, gMorphEpoch;
float*         gPoseCpu;              // this frame's palettes (made in load(); instance_pose copies into it)
uint32_t       gPoseUsed;
uint64_t       gPoseFrame;
Pose*          gPose; // per instance row
Last*          gLast;
Job*           gJobs; // this frame's, from index 1 (index 0's first word: their count + 1)
uint32_t       gJobCount, gGroups;
uint32_t       gBase, gCapacity, gCursor;
uint64_t       gFrame;
// The Model Viewer's preview: its own palette, jobs and set (its call comes inside renderer_models' draw, after or before
// this frame's skin pass), and a region after the instances' sized by what it last asked for.
DescriptorSet* gPreviewSet;
Buffer*        gPreviewPaletteBuf[kMaxFrames];
Buffer*        gPreviewJobBuf[kMaxFrames];
uint32_t       gPreviewWant;      // vertices its skinned parts need
uint64_t       gPreviewAsked;     // the frame it last asked
uint32_t       gPreviewBase, gPreviewRoom;

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

void instance_bounds(yea_handle h, yea_float3 lo, yea_float3 hi)
{
    const uint32_t row = rowOf(Instances, h);
    if (row == 0xFFFFFFFFu || !gPose || !(lo.x <= hi.x && lo.y <= hi.y && lo.z <= hi.z))
        return;
    Pose& p = gPose[row];
    p.lo[0] = lo.x, p.lo[1] = lo.y, p.lo[2] = lo.z, p.hi[0] = hi.x, p.hi[1] = hi.y, p.hi[2] = hi.z;
    p.boundsFrame = yea::host()->frame();
}

void mesh_morph(uint32_t mesh, uint64_t refs, uint64_t deltas, uint32_t deltaCount)
{
    const uint32_t*         r = (const uint32_t*)(uintptr_t)refs;
    const RenderMorphDelta* d = (const RenderMorphDelta*)(uintptr_t)deltas;
    if (!r || !d || mesh >= Meshes.size() || mesh >= MeshSkins.size() || !MeshSkins[mesh].firstSkin)
        return; // morphs ride the skin pass: a mesh without a skin has none
    const uint32_t n = Meshes[mesh].vertexCount;
    uint64_t       need = 0;
    for (uint32_t v = 0; v < n; ++v)
    {
        const uint32_t first = r[v] & 0xFFFFFFu, count = r[v] >> 24;
        if ((uint64_t)first + count > deltaCount)
            return; // damaged: it stays unmorphed
        need += count;
    }
    if (MorphVertexes.size() + (uint64_t)n > MorphVertexes.capacity() || MorphDeltas.size() + need > MorphDeltas.capacity() || need >= (1u << 24))
        return; // the tables are full: it stays unmorphed
    while (MeshMorphs.size() <= mesh && MeshMorphs.add())
        MeshMorphs[MeshMorphs.size() - 1] = MeshMorph{};
    if (mesh >= MeshMorphs.size())
        return;
    MeshMorphs[mesh].firstVertex = MorphVertexes.size() + 1;
    MeshMorphs[mesh].firstDelta = MorphDeltas.size();
    // Only the deltas this version's vertices name, in its vertices' order.
    uint32_t at = 0;
    for (uint32_t v = 0; v < n; ++v)
    {
        const uint32_t first = r[v] & 0xFFFFFFu, count = r[v] >> 24;
        MorphVertexes.add()->ref = count ? at | count << 24 : 0;
        for (uint32_t k = 0; k < count; ++k)
        {
            MorphDelta* o = MorphDeltas.add();
            o->p = { d[first + k].p[0], d[first + k].p[1], d[first + k].p[2] };
            o->packed = d[first + k].packed;
        }
        at += count;
    }
}

void instance_morph(yea_handle h, uint64_t weights, uint32_t count)
{
    const uint32_t row = rowOf(Instances, h);
    const float*   w = (const float*)(uintptr_t)weights;
    if (row == 0xFFFFFFFFu || !w || !count || count > 255 || !gPoseCpu)
        return;
    const uint64_t frame = yea::host()->frame();
    if (frame != gPoseFrame)
        gPoseFrame = frame, gPoseUsed = 0;
    const uint32_t room = (count + 3) & ~3u; // whole float4s, so the palettes after it stay aligned
    if (gPoseUsed + room > kPaletteFloats)
        return;
    memcpy(gPoseCpu + gPoseUsed, w, count * sizeof(float));
    memset(gPoseCpu + gPoseUsed + count, 0, (room - count) * sizeof(float));
    gPose[row].weights = gPoseUsed, gPose[row].targets = count, gPose[row].weightFrame = frame;
    gPoseUsed += room;
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
    addDescriptorSet(gRenderer, &set, &gPreviewSet);
    gPreviewWant = 0, gPreviewAsked = 0, gPreviewBase = gPreviewRoom = 0;
    gMorphRows = gMorphDeltaRows = gMorphEpoch = 0;
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
        d.mDesc.mFormat = TinyImageFormat_R32G32B32A32_SFLOAT;
        d.mDesc.mSize = (uint64_t)kPreviewBones * 48;
        d.mDesc.mElementCount = kPreviewBones * 3;
        d.mDesc.pName = "renderer_skin_preview_palette";
        d.ppBuffer = &gPreviewPaletteBuf[i];
        addResource(&d, nullptr);
        d.mDesc.mFormat = TinyImageFormat_R32G32B32A32_UINT;
        d.mDesc.mSize = (uint64_t)(kPreviewJobs + 1) * sizeof(Job);
        d.mDesc.mElementCount = (kPreviewJobs + 1) * 2;
        d.mDesc.pName = "renderer_skin_preview_jobs";
        d.ppBuffer = &gPreviewJobBuf[i];
        addResource(&d, nullptr);
    }
    waitForAllResourceLoads();
    bool ok = gPipe && gSet && gPreviewSet;
    for (uint32_t i = 0; i < gFrames; ++i)
        ok = ok && gPaletteBuf[i] && gPaletteBuf[i]->pCpuMappedAddress && gJobBuf[i] && gJobBuf[i]->pCpuMappedAddress &&
             gPreviewPaletteBuf[i] && gPreviewPaletteBuf[i]->pCpuMappedAddress && gPreviewJobBuf[i] && gPreviewJobBuf[i]->pCpuMappedAddress;
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
        if (gPreviewPaletteBuf[i])
            removeResource(gPreviewPaletteBuf[i]), gPreviewPaletteBuf[i] = nullptr;
        if (gPreviewJobBuf[i])
            removeResource(gPreviewJobBuf[i]), gPreviewJobBuf[i] = nullptr;
    }
    if (gPreviewSet)
        removeDescriptorSet(gRenderer, gPreviewSet), gPreviewSet = nullptr;
    if (gSkinBuf)
        removeResource(gSkinBuf), gSkinBuf = nullptr;
    if (gMorphVertBuf)
        removeResource(gMorphVertBuf), gMorphVertBuf = nullptr;
    if (gMorphDeltaBuf)
        removeResource(gMorphDeltaBuf), gMorphDeltaBuf = nullptr;
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
    uint64_t need = gPreviewAsked + 2 >= gFrame ? gPreviewWant : 0; // the preview's, while it is shown
    for (uint32_t row = 0; row < Instances.size(); ++row)
        if (posed(row))
            need += 2ull * Meshes[Instances[row].mesh].vertexCount;
    return need > 0x0FFFFFFFu ? 0x0FFFFFFFu : (uint32_t)need;
}

void skinBegin(uint32_t base, uint32_t capacity)
{
    // The preview's region at the end, while it is shown and fits; the instances' from the start, before it.
    const uint32_t want = gPreviewAsked + 2 >= gFrame ? gPreviewWant : 0;
    gPreviewRoom = want <= capacity ? want : 0;
    gPreviewBase = base + capacity - gPreviewRoom;
    gBase = base, gCapacity = capacity - gPreviewRoom, gCursor = 0, gJobCount = 0, gGroups = 0;
    // The bones' buffer, made again whenever the table grew or the scene started over (rare: a model loading). Here, in
    // the core's own draw: the preview's call (renderer_models' draw) never makes it.
    const uint32_t rows = SkinVertexes.size(), epoch = scene_epoch();
    if (!gPose)
        return;
    // The morph tables likewise (a row at least each, so the skin pass always has them to bind).
    const uint32_t mrows = MorphVertexes.size(), drows = MorphDeltas.size();
    if (!gMorphVertBuf || !gMorphDeltaBuf || mrows != gMorphRows || drows != gMorphDeltaRows || epoch != gMorphEpoch)
    {
        if (gMorphVertBuf || gMorphDeltaBuf)
            waitQueueIdle((Queue*)yea::host()->queue);
        if (gMorphVertBuf)
            removeResource(gMorphVertBuf), gMorphVertBuf = nullptr;
        if (gMorphDeltaBuf)
            removeResource(gMorphDeltaBuf), gMorphDeltaBuf = nullptr;
        static const uint32_t zero[4] = {};
        BufferLoadDesc        d = {};
        d.mDesc.mDescriptors = DESCRIPTOR_TYPE_BUFFER;
        d.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_GPU_ONLY;
        d.mDesc.mStartState = RESOURCE_STATE_SHADER_RESOURCE;
        d.mDesc.mFormat = TinyImageFormat_R32_UINT;
        d.mDesc.mSize = (uint64_t)(mrows ? mrows : 1) * 4;
        d.mDesc.mElementCount = mrows ? mrows : 1;
        d.mDesc.mStructStride = 4;
        d.mDesc.pName = "renderer_morph_vertices";
        d.pData = mrows ? (const void*)MorphVertexes.begin() : (const void*)zero;
        d.ppBuffer = &gMorphVertBuf;
        addResource(&d, nullptr);
        d.mDesc.mFormat = TinyImageFormat_R32G32B32A32_UINT;
        d.mDesc.mSize = (uint64_t)(drows ? drows : 1) * 16;
        d.mDesc.mElementCount = drows ? drows : 1;
        d.mDesc.mStructStride = 16;
        d.mDesc.pName = "renderer_morph_deltas";
        d.pData = drows ? (const void*)MorphDeltas.begin() : (const void*)zero;
        d.ppBuffer = &gMorphDeltaBuf;
        addResource(&d, nullptr);
        waitForAllResourceLoads();
        gMorphRows = gMorphVertBuf && gMorphDeltaBuf ? mrows : 0xFFFFFFFFu, gMorphDeltaRows = drows, gMorphEpoch = epoch;
    }
    if ((gSkinBuf && rows == gSkinRows && epoch == gSkinEpoch) || (!rows && !gSkinBuf))
        return;
    if (gSkinBuf)
    {
        waitQueueIdle((Queue*)yea::host()->queue);
        removeResource(gSkinBuf), gSkinBuf = nullptr;
    }
    gSkinRows = 0, gSkinEpoch = epoch;
    if (!rows)
        return;
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
    gSkinRows = gSkinBuf ? rows : 0;
}

bool skinBounds(uint32_t row, float* lo, float* hi)
{
    if (!gPose || !posed(row) || gPose[row].boundsFrame != gFrame)
        return false;
    memcpy(lo, gPose[row].lo, 12), memcpy(hi, gPose[row].hi, 12);
    return true;
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
    const bool morphed = gPose[row].weightFrame == gFrame && lod < MeshMorphs.size() && MeshMorphs[lod].firstVertex &&
                         MeshMorphs[lod].firstVertex - 1 + (uint64_t)count <= MorphVertexes.size();
    j = { Meshes[lod].firstVertex, count, (uint32_t)out.current, gGroups,
          fresh ? (uint32_t)out.previous : 0xFFFFFFFFu, MeshSkins[lod].firstSkin - 1, gPose[row].offset / 4, 0,
          morphed ? MeshMorphs[lod].firstVertex - 1 : kNoMorph, morphed ? MeshMorphs[lod].firstDelta : 0, gPose[row].weights / 4,
          morphed ? gPose[row].targets : 0 };
    gGroups += (count + 63) / 64;
    return true;
}

bool skinJobs() { return gJobCount > 0; }

void skinRecord(Cmd* cmd, uint32_t fi, Buffer* positions, Buffer* attributes, uint32_t stateBits)
{
    if (!gJobCount || !positions || !attributes)
        return;
    const ResourceState state = (ResourceState)stateBits;
    if (!gSkinBuf || gSkinRows != SkinVertexes.size() || !gMorphVertBuf || !gMorphDeltaBuf || gMorphRows != MorphVertexes.size())
    { // skinBegin made them, or couldn't
        gJobCount = 0;
        return;
    }
    memcpy(gPaletteBuf[fi]->pCpuMappedAddress, gPoseCpu, (size_t)gPoseUsed * sizeof(float));
    gJobs[0] = Job{};
    gJobs[0].src = gJobCount; // skin.comp's [0].x
    memcpy(gJobBuf[fi]->pCpuMappedAddress, gJobs, (size_t)(gJobCount + 1) * sizeof(Job));
    DescriptorData p[7] = {};
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
    p[5].mIndex = SRT_RES_IDX(SkinSrt, PerFrame, gMorphVertices);
    p[5].ppBuffers = &gMorphVertBuf;
    p[6].mIndex = SRT_RES_IDX(SkinSrt, PerFrame, gMorphDeltas);
    p[6].ppBuffers = &gMorphDeltaBuf;
    updateDescriptorSet(gRenderer, fi, gSet, 7, p);
    BufferBarrier in[2] = { { positions, state, RESOURCE_STATE_UNORDERED_ACCESS }, { attributes, state, RESOURCE_STATE_UNORDERED_ACCESS } };
    cmdResourceBarrier(cmd, 2, in, 0, nullptr, 0, nullptr);
    cmdBindPipeline(cmd, gPipe);
    cmdBindDescriptorSet(cmd, fi, gSet);
    cmdDispatch(cmd, gGroups, 1, 1);
    BufferBarrier out[2] = { { positions, RESOURCE_STATE_UNORDERED_ACCESS, state }, { attributes, RESOURCE_STATE_UNORDERED_ACCESS, state } };
    cmdResourceBarrier(cmd, 2, out, 0, nullptr, 0, nullptr);
}

uint32_t skinPreview(Cmd* cmd, uint32_t fi, Buffer* positions, Buffer* attributes, uint32_t stateBits, const uint32_t* meshes,
                     uint32_t parts, const float* palette, uint32_t bones, const float* weights, uint32_t targets, int32_t* offsets)
{
    for (uint32_t k = 0; k < parts; ++k)
        offsets[k] = -1;
    targets = targets > 255 ? 0 : targets;
    if (!gPipe || !gPreviewSet || !palette || !bones || bones + (targets + 11) / 12 > kPreviewBones || !positions || !attributes)
        return 0;
    // What its skinned parts need (asked for again each frame it shows; reserved from the core's next draw on).
    uint64_t want = 0;
    for (uint32_t k = 0; k < parts && k < kPreviewJobs; ++k)
        if (meshes[k] < Meshes.size() && meshes[k] < MeshSkins.size() && MeshSkins[meshes[k]].firstSkin)
            want += Meshes[meshes[k]].vertexCount;
    gPreviewWant = want > 0x0FFFFFFFu ? 0x0FFFFFFFu : (uint32_t)want;
    gPreviewAsked = yea::host()->frame();
    if (!want || want > gPreviewRoom || !gSkinBuf || gSkinRows != SkinVertexes.size() || gSkinEpoch != scene_epoch() || !gMorphVertBuf ||
        !gMorphDeltaBuf || gMorphRows != MorphVertexes.size())
        return 0; // drawn in its bind pose until its region and bones are there (a frame or two)
    Job*     jobs = (Job*)gPreviewJobBuf[fi]->pCpuMappedAddress;
    uint32_t count = 0, groups = 0, cursor = 0;
    for (uint32_t k = 0; k < parts && k < kPreviewJobs; ++k)
    {
        const uint32_t m = meshes[k];
        if (m >= Meshes.size() || m >= MeshSkins.size() || !MeshSkins[m].firstSkin ||
            MeshSkins[m].firstSkin - 1 + (uint64_t)Meshes[m].vertexCount > SkinVertexes.size())
            continue;
        const uint32_t n = Meshes[m].vertexCount;
        offsets[k] = (int32_t)(gPreviewBase + cursor);
        const bool morphed = weights && targets && m < MeshMorphs.size() && MeshMorphs[m].firstVertex &&
                             MeshMorphs[m].firstVertex - 1 + (uint64_t)n <= MorphVertexes.size();
        jobs[1 + count++] = { Meshes[m].firstVertex, n, gPreviewBase + cursor, groups, 0xFFFFFFFFu, MeshSkins[m].firstSkin - 1, 0, 0,
                              morphed ? MeshMorphs[m].firstVertex - 1 : kNoMorph, morphed ? MeshMorphs[m].firstDelta : 0,
                              (bones * 12) / 4, morphed ? targets : 0 };
        groups += (n + 63) / 64;
        cursor += n;
    }
    if (!count)
        return 0;
    jobs[0] = Job{};
    jobs[0].src = count;
    memcpy(gPreviewPaletteBuf[fi]->pCpuMappedAddress, palette, (size_t)bones * 12 * sizeof(float));
    if (weights && targets) // the weights after the palette (whole float4s: bones * 12 floats already are)
        memcpy((float*)gPreviewPaletteBuf[fi]->pCpuMappedAddress + bones * 12, weights, (size_t)targets * sizeof(float));
    DescriptorData p[7] = {};
    p[0].mIndex = SRT_RES_IDX(SkinSrt, PerFrame, gPositions);
    p[0].ppBuffers = &positions;
    p[1].mIndex = SRT_RES_IDX(SkinSrt, PerFrame, gAttributes);
    p[1].ppBuffers = &attributes;
    p[2].mIndex = SRT_RES_IDX(SkinSrt, PerFrame, gSkin);
    p[2].ppBuffers = &gSkinBuf;
    p[3].mIndex = SRT_RES_IDX(SkinSrt, PerFrame, gPalette);
    p[3].ppBuffers = &gPreviewPaletteBuf[fi];
    p[4].mIndex = SRT_RES_IDX(SkinSrt, PerFrame, gJobs);
    p[4].ppBuffers = &gPreviewJobBuf[fi];
    p[5].mIndex = SRT_RES_IDX(SkinSrt, PerFrame, gMorphVertices);
    p[5].ppBuffers = &gMorphVertBuf;
    p[6].mIndex = SRT_RES_IDX(SkinSrt, PerFrame, gMorphDeltas);
    p[6].ppBuffers = &gMorphDeltaBuf;
    updateDescriptorSet(gRenderer, fi, gPreviewSet, 7, p);
    const ResourceState state = (ResourceState)stateBits;
    BufferBarrier in[2] = { { positions, state, RESOURCE_STATE_UNORDERED_ACCESS }, { attributes, state, RESOURCE_STATE_UNORDERED_ACCESS } };
    cmdResourceBarrier(cmd, 2, in, 0, nullptr, 0, nullptr);
    cmdBindPipeline(cmd, gPipe);
    cmdBindDescriptorSet(cmd, fi, gPreviewSet);
    cmdDispatch(cmd, groups, 1, 1);
    BufferBarrier out[2] = { { positions, RESOURCE_STATE_UNORDERED_ACCESS, state }, { attributes, RESOURCE_STATE_UNORDERED_ACCESS, state } };
    cmdResourceBarrier(cmd, 2, out, 0, nullptr, 0, nullptr);
    return count;
}
} // namespace renderer
