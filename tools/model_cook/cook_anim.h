// The model cooker's animation encoder (rmodel v3, RmodelCodecQuantized; the layout and decoder are in
// modules/renderer_models/anim.h). Included by model_cook.cpp; never shipped.
//
// A clip is resampled at 30 frames a second, encoded, decoded again and compared with the source evaluated exactly, at
// every frame and at quarters between frames (a cubic key's bulge can pass through the halfway point), by where it puts the skin: every vertex posed both ways. When a vertex strays
// more than the tolerance (kMaxError, or kRelativeError of the model's size for big models, where 16-bit quantisation
// sets the floor), the clip is resampled at twice the rate (up to 240), so fast motion keeps its shape, as long as each
// doubling still helps. A clip with step keys (glTF STEP) is sampled at 120 or more and measured on its frames only: a
// jump blends over one frame (8 ms or less).
// The report gives each clip's rate and worst error.
#pragma once

#include "../../modules/renderer_models/anim.h"

#include <functional>
#include <vector>

namespace cook_anim
{
constexpr float kMaxError = 0.001f;        // metres at the skin
constexpr float kRelativeError = 2.5e-4f;  // of the model's largest size, when that is more

struct SkinnedVertex // a vertex at bind time and its bones, for measuring
{
    float    p[3];
    uint16_t bones[4];
    uint16_t weights[4];
};

struct Encoded
{
    RmodelClip           clip = {};
    std::vector<uint8_t> data;
    float                error = 0;   // the worst a vertex strays, metres
    float                tolerance = kMaxError;
    size_t               rawBytes = 0; // its frames as plain floats (t, r, s), for the report
};

// The skin posed by a palette: each vertex's position.
inline void skin(const std::vector<SkinnedVertex>& verts, const float* pal, std::vector<float>& out)
{
    out.resize(verts.size() * 3);
    for (size_t i = 0; i < verts.size(); ++i)
    {
        const SkinnedVertex& v = verts[i];
        float                o[3] = { 0, 0, 0 };
        for (int k = 0; k < 4; ++k)
        {
            if (!v.weights[k])
                continue;
            const float* m = &pal[v.bones[k] * 12];
            const float  w = v.weights[k] / 65535.0f;
            for (int r = 0; r < 3; ++r)
                o[r] += w * (m[r * 4] * v.p[0] + m[r * 4 + 1] * v.p[1] + m[r * 4 + 2] * v.p[2] + m[r * 4 + 3]);
        }
        memcpy(&out[i * 3], o, 12);
    }
}

// One clip at `rate`: `source(t, poses)` gives every node's exact local pose at time t (renderer space).
inline Encoded encodeAt(const char* name, float duration, float rate, uint32_t nodeCount,
                        const std::function<void(float, anim::Pose*)>& source)
{
    Encoded e;
    const uint32_t frames = duration > 0 ? (uint32_t)ceilf(duration * rate - 1e-4f) + 1 : 1;
    snprintf(e.clip.name, sizeof e.clip.name, "%s", name);
    e.clip.duration = duration > 0 ? duration : 0;
    e.clip.frames = frames;
    e.clip.rate = frames > 1 ? (float)(frames - 1) / duration : 0;
    e.clip.codec = RmodelCodecQuantized;
    std::vector<anim::Pose> exact((size_t)frames * nodeCount);
    for (uint32_t f = 0; f < frames; ++f)
        source(frames > 1 ? duration * (float)f / (float)(frames - 1) : 0, &exact[(size_t)f * nodeCount]);
    e.rawBytes = (size_t)frames * nodeCount * 40;

    std::vector<RmodelTrack> tracks(nodeCount * 3);
    std::vector<uint16_t>    samples;
    for (uint32_t n = 0; n < nodeCount; ++n)
        for (uint32_t k = 0; k < 3; ++k)
        {
            RmodelTrack& tr = tracks[n * 3 + k];
            const int    comps = k == 1 ? 4 : 3;
            auto         value = [&](uint32_t f) -> const float* {
                const anim::Pose& p = exact[(size_t)f * nodeCount + n];
                return k == 0 ? p.t : (k == 1 ? p.r : p.s);
            };
            bool constant = true;
            for (uint32_t f = 1; f < frames && constant; ++f)
            {
                const float* a = value(0);
                const float* b = value(f);
                if (k == 1)
                    constant = fabsf(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]) > 1.0f - 1e-9f;
                else
                    for (int i = 0; i < 3; ++i)
                        constant = constant && fabsf(a[i] - b[i]) <= 1e-6f * fmaxf(1.0f, fabsf(a[i]));
            }
            if (constant)
            {
                memcpy(tr.base, value(0), comps * 4);
                if (k == 1)
                {
                    float len = sqrtf(tr.base[0] * tr.base[0] + tr.base[1] * tr.base[1] + tr.base[2] * tr.base[2] + tr.base[3] * tr.base[3]);
                    for (int i = 0; i < 4; ++i)
                        tr.base[i] /= len > 0 ? len : 1;
                }
                tr.data = 0xFFFFFFFFu;
                continue;
            }
            tr.data = (uint32_t)(samples.size() * 2); // from the samples' start; made absolute below
            if (k == 1)
            {
                for (uint32_t f = 0; f < frames; ++f)
                {
                    uint16_t w[3];
                    anim::encodeRotation(value(f), w);
                    samples.insert(samples.end(), w, w + 3);
                }
                continue;
            }
            float lo[3], hi[3];
            for (int i = 0; i < 3; ++i)
                lo[i] = hi[i] = value(0)[i];
            for (uint32_t f = 1; f < frames; ++f)
                for (int i = 0; i < 3; ++i)
                    lo[i] = fminf(lo[i], value(f)[i]), hi[i] = fmaxf(hi[i], value(f)[i]);
            for (int i = 0; i < 3; ++i)
                tr.base[i] = lo[i], tr.extent[i] = hi[i] - lo[i];
            for (uint32_t f = 0; f < frames; ++f)
                for (int i = 0; i < 3; ++i)
                {
                    float u = tr.extent[i] > 0 ? (value(f)[i] - lo[i]) / tr.extent[i] : 0;
                    u = u < 0 ? 0 : (u > 1 ? 1 : u);
                    samples.push_back((uint16_t)(u * 65535.0f + 0.5f));
                }
        }
    const size_t head = tracks.size() * sizeof(RmodelTrack);
    for (RmodelTrack& tr : tracks)
        if (tr.data != 0xFFFFFFFFu)
            tr.data += (uint32_t)head;
    e.data.resize(head + samples.size() * 2);
    memcpy(e.data.data(), tracks.data(), head);
    if (!samples.empty())
        memcpy(e.data.data() + head, samples.data(), samples.size() * 2);
    e.data.resize((e.data.size() + 15) & ~(size_t)15, 0);
    e.clip.bytes = e.data.size();
    return e;
}

// The worst a vertex strays between the decoded clip and the source, at every frame and quarters between them. Also
// grows lo/hi (per part: `part[v]` names a vertex's part) by every decoded frame's pose, so culling bounds hold the motion.
inline float measure(const Encoded& e, const RmodelNode* nodes, uint32_t nodeCount, const RmodelBone* bones, uint32_t boneCount,
                     const std::vector<SkinnedVertex>& verts, const std::vector<uint32_t>& part, float (*lo)[3], float (*hi)[3],
                     const std::function<void(float, anim::Pose*)>& source, bool framesOnly)
{
    std::vector<anim::Pose> a(nodeCount), b(nodeCount);
    std::vector<float>      world(nodeCount * 12), pa(boneCount * 12), pb(boneCount * 12), sa, sb;
    float                   worst = 0;
    const uint32_t          steps = e.clip.frames > 1 ? (e.clip.frames - 1) * 4 : 0;
    for (uint32_t s = 0; s <= steps; ++s)
    {
        const float t = steps ? e.clip.duration * (float)s / (float)steps : 0;
        if (framesOnly && (s & 3))
            continue;
        anim::sampleClip(e.data.data(), e.clip, nodeCount, t, false, a.data());
        source(t, b.data());
        anim::palette(nodes, nodeCount, bones, boneCount, a.data(), world.data(), pa.data());
        anim::palette(nodes, nodeCount, bones, boneCount, b.data(), world.data(), pb.data());
        skin(verts, pa.data(), sa);
        skin(verts, pb.data(), sb);
        for (size_t i = 0; i < verts.size(); ++i)
        {
            const float dx = sa[i * 3] - sb[i * 3], dy = sa[i * 3 + 1] - sb[i * 3 + 1], dz = sa[i * 3 + 2] - sb[i * 3 + 2];
            worst = fmaxf(worst, sqrtf(dx * dx + dy * dy + dz * dz));
            if ((s & 3) == 0)
                for (int k = 0; k < 3; ++k)
                    lo[part[i]][k] = fminf(lo[part[i]][k], sa[i * 3 + k]), hi[part[i]][k] = fmaxf(hi[part[i]][k], sa[i * 3 + k]);
        }
    }
    return worst;
}

// A clip encoded at the lowest rate (30, 60, 120 or 240 frames a second) that keeps the skin within `tolerance`, or
// none does, the most accurate (a lower rate wins within a quarter of it: quantisation, not sampling, is the limit
// there). `keyRate`, when the file's keys are evenly spaced at it, is tried first: sampling on the keys is exact for
// linear keys, where a rate between them cuts their corners.
inline Encoded encode(const char* name, float duration, const RmodelNode* nodes, uint32_t nodeCount, const RmodelBone* bones,
                      uint32_t boneCount, const std::vector<SkinnedVertex>& verts, const std::vector<uint32_t>& part,
                      std::vector<float>& lo, std::vector<float>& hi, const std::function<void(float, anim::Pose*)>& source,
                      float tolerance, bool stepped, float keyRate)
{
    std::vector<float> rates;
    if (stepped)
        keyRate = 0;
    if (keyRate >= 1 && keyRate <= 240)
        rates.push_back(keyRate);
    for (float r = stepped ? 120.0f : 30.0f; r <= 240; r *= 2)
        if (r > keyRate * 1.001f || !(keyRate >= 1))
            rates.push_back(r);
    Encoded            best;
    std::vector<float> bestLo, bestHi;
    bool               have = false;
    for (float rate : rates)
    {
        std::vector<float> l = lo, h = hi;
        Encoded            e = encodeAt(name, duration, rate, nodeCount, source);
        e.tolerance = tolerance;
        e.error = measure(e, nodes, nodeCount, bones, boneCount, verts, part, (float(*)[3])l.data(), (float(*)[3])h.data(), source, stepped);
        if (!have || e.error < best.error * 0.75f)
            best = std::move(e), bestLo = std::move(l), bestHi = std::move(h), have = true;
        if (best.error <= tolerance || best.clip.frames <= 1)
            break;
    }
    lo = bestLo, hi = bestHi;
    return best;
}
} // namespace cook_anim
