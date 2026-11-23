// Texture maps and material maps (models stage 2). The exports only write tables; the GPU side (texturesUpdate, in the
// renderer's own draw) reads each texture's mips from its file and uploads them, a few a frame, and again after the
// renderer restarts (the tables survive it, the GPU textures don't).
#include "yea_renderer.h"
#include "scene.h"

#include <stdio.h>
#include <string.h>
#include <chrono>

#include <yea/device.h>
#include <Common_3/Graphics/Interfaces/IGraphics.h>
#include <Common_3/Resources/ResourceLoader/Interfaces/IResourceLoader.h>

namespace renderer
{
namespace
{
const uint32_t kNone = 0xFFFFFFFFu;
const uint32_t kSlots = 1024; // Texture's capacity: a row is a slot in the shaders' array (frame.h RENDER_MAX_TEXTURES)
const uint64_t kFrameBytes = 24ull << 20; // file bytes read and uploaded in one frame at most (one texture at least)

enum : uint8_t
{
    SlotEmpty,
    SlotReady,
    SlotFailed,
};

Texture* gTex[kSlots];    // the GPU texture of each row, made for gTexGen
uint32_t   gTexGen[kSlots]; // the row's generation it was made (or failed) for
uint32_t   gTexEpoch[kSlots]; // and the scene's epoch then (clear() and Reset start rows again at generation 0)
uint8_t    gTexState[kSlots];
uint32_t   gTexSkip[kSlots]; // top mips left out
Texture* gSlots[kSlots];  // what the shaders' array holds: the texture, or the stand-in
Texture* gStandIn;
uint32_t   gVersion;        // changes whenever gSlots does (never 0)
int32_t    gDetail = -1;    // texture_detail the textures were made at
uint32_t   gLoadedNow, gFailedNow; // since the queue last emptied, for one log line
uint64_t   gBytesNow;

template<class T> uint32_t rowOf(const yea::Table<T>& t, yea_handle h)
{
    uint32_t row = h & 0xFFFFFu;
    if (row >= t.size() || !t[row].alive || (t[row].gen & 0xFFFu) != (h >> 20))
        return kNone;
    return row;
}

uint64_t mipBytes(uint32_t w, uint32_t h, uint32_t level)
{
    uint32_t mw = w >> level ? w >> level : 1, mh = h >> level ? h >> level : 1;
    return (uint64_t)((mw + 3) / 4) * ((mh + 3) / 4) * 16;
}

MaterialMap* mapRow(yea_handle m)
{
    uint32_t row = rowOf(Materials, m);
    if (row == kNone)
        return nullptr;
    while (MaterialMaps.size() <= row)
        if (!MaterialMaps.add())
            return nullptr;
    return &MaterialMaps[row];
}

void defaultFactors(MaterialMap& mm)
{
    mm.baseFactor = { 1, 1, 1, 1 };
    mm.roughness = mm.metallic = mm.normalScale = mm.occlusion = 1;
    mm.set |= 2;
}

// Loads one texture's mips (past `skip` top ones) from its file; false when the file or the GPU refused.
bool upload(uint32_t row, uint32_t skip)
{
    const TextureMap& t = TextureMaps[row];
    char path[1024];
    uint32_t n = t.pathLength < sizeof path - 1 ? t.pathLength : sizeof path - 1;
    for (uint32_t i = 0; i < n; ++i)
        path[i] = (char)TexturePaths[t.pathFirst + i].c;
    path[n] = 0;
    FILE* f = fopen(path, "rb");
    if (!f)
        return false;
    uint64_t at = t.offset;
    for (uint32_t l = 0; l < skip; ++l)
        at += mipBytes(t.width, t.height, l);
    TextureDesc d = {};
    d.mWidth = t.width >> skip ? t.width >> skip : 1;
    d.mHeight = t.height >> skip ? t.height >> skip : 1;
    d.mDepth = d.mArraySize = 1;
    d.mMipLevels = t.mips - skip;
    d.mSampleCount = SAMPLE_COUNT_1;
    d.mFormat = t.format == 1 ? TinyImageFormat_DXBC7_SRGB : t.format == 2 ? TinyImageFormat_DXBC7_UNORM : TinyImageFormat_DXBC5_UNORM;
    d.mStartState = RESOURCE_STATE_SHADER_RESOURCE;
    d.mDescriptors = DESCRIPTOR_TYPE_TEXTURE;
    d.pName = "renderer_texture_map";
    Texture*      tex = nullptr;
    TextureLoadDesc tl = {};
    tl.pDesc = &d;
    tl.ppTexture = &tex;
    addResource(&tl, nullptr);
    waitForAllResourceLoads();
    bool ok = tex != nullptr;
    if (ok)
    {
        TextureUpdateDesc u = {};
        u.pTexture = tex;
        u.mBaseMipLevel = 0;
        u.mMipLevels = d.mMipLevels;
        u.mBaseArrayLayer = 0;
        u.mLayerCount = 1;
        u.mCurrentState = RESOURCE_STATE_SHADER_RESOURCE;
        beginUpdateResource(&u);
        ok = fseek(f, (long)at, SEEK_SET) == 0;
        for (uint32_t l = 0; ok && l < d.mMipLevels; ++l)
        {
            uint32_t                 mw = d.mWidth >> l ? d.mWidth >> l : 1, mh = d.mHeight >> l ? d.mHeight >> l : 1;
            uint32_t                 rowBytes = (mw + 3) / 4 * 16, rows = (mh + 3) / 4;
            TextureSubresourceUpdate s = u.getSubresourceUpdateDesc(l, 0);
            if (!s.pMappedData || s.mRowCount < rows || s.mDstRowStride < rowBytes)
                ok = false;
            else if (s.mDstRowStride == rowBytes)
                ok = fread(s.pMappedData, rowBytes, rows, f) == rows;
            else
                for (uint32_t y = 0; ok && y < rows; ++y)
                    ok = fread(s.pMappedData + (size_t)y * s.mDstRowStride, rowBytes, 1, f) == 1;
        }
        endUpdateResource(&u);
        waitForAllResourceLoads();
        if (!ok)
            removeResource(tex), tex = nullptr;
    }
    fclose(f);
    gTex[row] = tex;
    return ok;
}
} // namespace

// --- Exports ----------------------------------------------------------------------------------------------------------

yea_handle texture_add(uint64_t pathPtr, uint64_t offset, uint64_t bytes, uint32_t format, uint32_t width, uint32_t height, uint32_t mips)
{
    const char* path = (const char*)(uintptr_t)pathPtr;
    if (!path || format < 1 || format > 3 || !width || !height || !mips || mips > 16 || (width >> (mips - 1) == 0 && height >> (mips - 1) == 0))
        return kNone;
    uint64_t need = 0;
    for (uint32_t l = 0; l < mips; ++l)
        need += mipBytes(width, height, l);
    if (need > bytes)
        return kNone;
    uint32_t len = (uint32_t)strnlen(path, 1023);
    bool     anyAlive = false;
    for (uint32_t i = 0; i < TextureMaps.size(); ++i)
        anyAlive = anyAlive || TextureMaps[i].alive;
    if (!anyAlive)
        TexturePaths.clear();
    // The file's path, shared with every texture of the same file.
    uint32_t first = kNone;
    for (uint32_t i = 0; i < TextureMaps.size() && first == kNone; ++i)
    {
        const TextureMap& t = TextureMaps[i];
        if (!t.alive || t.pathLength != len || t.pathFirst + len > TexturePaths.size())
            continue;
        uint32_t k = 0;
        while (k < len && TexturePaths[t.pathFirst + k].c == (uint8_t)path[k])
            ++k;
        if (k == len)
            first = t.pathFirst;
    }
    if (first == kNone)
    {
        if (TexturePaths.size() + len > TexturePaths.capacity())
            return kNone;
        first = TexturePaths.size();
        for (uint32_t k = 0; k < len; ++k)
            TexturePaths.add()->c = (uint8_t)path[k];
    }
    // A dead row reused, else a new one (as the scene's other tables).
    uint32_t row = kNone;
    for (uint32_t i = 0; i < TextureMaps.size() && row == kNone; ++i)
        if (!TextureMaps[i].alive)
            row = i;
    uint32_t gen = 0;
    if (row == kNone)
    {
        if (!TextureMaps.add())
            return kNone;
        row = TextureMaps.size() - 1;
    }
    else
        gen = TextureMaps[row].gen + 1;
    TextureMap& t = TextureMaps[row];
    t = TextureMap{};
    t.pathFirst = first;
    t.pathLength = len;
    t.offset = offset;
    t.bytes = bytes;
    t.format = format;
    t.width = width;
    t.height = height;
    t.mips = mips;
    t.gen = gen;
    t.alive = 1;
    return row | ((gen & 0xFFFu) << 20);
}

uint32_t texture_state(yea_handle h)
{
    uint32_t row = rowOf(TextureMaps, h);
    if (row == kNone)
        return 0;
    if (gTexGen[row] != TextureMaps[row].gen || gTexEpoch[row] != scene_epoch() || gTexState[row] == SlotEmpty)
        return 1;
    return gTexState[row] == SlotReady ? 2 : 3;
}

void texture_remove(yea_handle h)
{
    uint32_t row = rowOf(TextureMaps, h);
    if (row != kNone)
        TextureMaps[row].alive = 0;
}

void material_maps(yea_handle m, yea_handle base, yea_handle orm, yea_handle normal, yea_handle emissive)
{
    MaterialMap* mm = mapRow(m);
    if (!mm)
        return;
    mm->baseColor = base;
    mm->orm = orm;
    mm->normal = normal;
    mm->emissive = emissive;
    if (!(mm->set & 2))
        defaultFactors(*mm);
    mm->set |= 1;
}

void material_factors(yea_handle m, yea_float4 base, float roughness, float metallic, float normalScale, float occlusion)
{
    MaterialMap* mm = mapRow(m);
    if (!mm)
        return;
    mm->baseFactor = base;
    mm->roughness = roughness;
    mm->metallic = metallic;
    mm->normalScale = normalScale;
    mm->occlusion = occlusion;
    mm->set |= 2;
}

void material_alpha(yea_handle m, float cutoff)
{
    if (MaterialMap* mm = mapRow(m))
        mm->cutoff = cutoff > 0 ? cutoff : 0;
}

void material_blend(yea_handle m, float opacity)
{
    uint32_t row = rowOf(Materials, m);
    if (row == kNone)
        return;
    while (MaterialBlends.size() <= row)
        if (!MaterialBlends.add())
            return;
    MaterialBlends[row].opacity = opacity > 0 ? (opacity < 1 ? opacity : 1) : 0;
}

float materialOpacity(uint32_t row) { return row < MaterialBlends.size() ? MaterialBlends[row].opacity : 0.0f; }

// --- The renderer's side ------------------------------------------------------------------------------------------------

void materialMapsReset(uint32_t materialRow)
{
    if (materialRow < MaterialMaps.size())
        MaterialMaps[materialRow] = MaterialMap{};
    if (materialRow < MaterialBlends.size())
        MaterialBlends[materialRow] = MaterialBlend{};
}

uint32_t textureSlot(yea_handle h)
{
    uint32_t row = h == kNone ? kNone : rowOf(TextureMaps, h);
    return row != kNone && gTexGen[row] == TextureMaps[row].gen && gTexEpoch[row] == scene_epoch() && gTexState[row] == SlotReady ? row + 1 : 0;
}

void texturesInit(Texture* standIn)
{
    gStandIn = standIn;
    for (uint32_t i = 0; i < kSlots; ++i)
    {
        gTex[i] = nullptr;
        gTexState[i] = SlotEmpty;
        gSlots[i] = standIn;
    }
    // Unique to this load: a consumer's set holding an earlier load's (freed) textures never matches it.
    gVersion = (uint32_t)std::chrono::steady_clock::now().time_since_epoch().count() | 1u;
    gDetail = -1;
    gLoadedNow = gFailedNow = 0;
    gBytesNow = 0;
}

void texturesFree()
{
    for (uint32_t i = 0; i < kSlots; ++i)
    {
        if (gTex[i])
            removeResource(gTex[i]);
        gTex[i] = nullptr;
        gTexState[i] = SlotEmpty;
        gSlots[i] = gStandIn;
    }
}

void texturesUpdate(Queue* queue)
{
    if (!gStandIn)
        return;
    // Gone, made again for a newer row, or a new texture_detail: freed together, after the GPU is done with them.
    const int32_t  detail = settings.texture_detail;
    const uint32_t epoch = scene_epoch();
    bool          freeing = false;
    for (uint32_t i = 0; i < kSlots && !freeing; ++i)
        freeing = gTexState[i] != SlotEmpty && (i >= TextureMaps.size() || !TextureMaps[i].alive || gTexGen[i] != TextureMaps[i].gen || gTexEpoch[i] != epoch ||
                                                (gDetail != detail && gTexState[i] == SlotReady));
    if (freeing)
    {
        waitQueueIdle(queue);
        for (uint32_t i = 0; i < kSlots; ++i)
            if (gTexState[i] != SlotEmpty && (i >= TextureMaps.size() || !TextureMaps[i].alive || gTexGen[i] != TextureMaps[i].gen || gTexEpoch[i] != epoch ||
                                              (gDetail != detail && gTexState[i] == SlotReady)))
            {
                if (gTex[i])
                    removeResource(gTex[i]);
                gTex[i] = nullptr;
                gTexState[i] = SlotEmpty;
                gSlots[i] = gStandIn;
            }
        if (!++gVersion)
            gVersion = 1;
    }
    gDetail = detail;

    // Load what's waiting, the lowest rows first, up to kFrameBytes a frame.
    uint64_t spent = 0;
    bool     waiting = false;
    for (uint32_t i = 0; i < TextureMaps.size() && i < kSlots; ++i)
    {
        const TextureMap& t = TextureMaps[i];
        if (!t.alive || gTexState[i] != SlotEmpty)
            continue;
        if (spent >= kFrameBytes)
        {
            waiting = true;
            break;
        }
        // Top mips left out: texture_detail's, then more while the GPU's free memory can't hold it (with 256 MB to spare);
        // never below 64 texels on the longer side.
        uint32_t skip = (uint32_t)(detail < 0 ? 0 : detail), longest = t.width > t.height ? t.width : t.height;
        auto     fits = [&](uint32_t s) { return s + 1 < t.mips && (longest >> (s + 1)) >= 64; };
        while (skip && !fits(skip - 1))
            --skip;
        uint64_t bytes = 0;
        for (uint32_t l = skip; l < t.mips; ++l)
            bytes += mipBytes(t.width, t.height, l);
        YeaDeviceMemory mem = {};
        if (yea::deviceMemory(&mem) && mem.budget)
            while (fits(skip) && mem.usage + bytes + (256ull << 20) > mem.budget)
            {
                bytes -= mipBytes(t.width, t.height, skip);
                ++skip;
            }
        // Block-compressed: the top mip drawn must be whole 4 x 4 blocks.
        bool ok = ((t.width >> skip) % 4 == 0 && (t.height >> skip) % 4 == 0) && upload(i, skip);
        gTexGen[i] = t.gen;
        gTexEpoch[i] = epoch;
        gTexSkip[i] = skip;
        gTexState[i] = ok ? SlotReady : SlotFailed;
        gSlots[i] = ok ? gTex[i] : gStandIn;
        if (!++gVersion)
            gVersion = 1;
        spent += bytes;
        if (ok)
            gLoadedNow++, gBytesNow += bytes;
        else
            gFailedNow++;
    }
    if (!waiting && gLoadedNow + gFailedNow)
    {
        char line[256];
        snprintf(line, sizeof line, "renderer: %u texture maps loaded (%.1f MB on the GPU)%s", gLoadedNow, (double)gBytesNow / (1 << 20),
                 gFailedNow ? "; some could not be read (their file is missing, or not whole 4 x 4 blocks): drawn without them" : "");
        yea::host()->log(gFailedNow ? 1 : 0, line);
        gLoadedNow = gFailedNow = 0;
        gBytesNow = 0;
    }
}

Texture* const* textureSlots() { return gSlots; }
uint32_t          textureVersion() { return gVersion; }
} // namespace renderer
