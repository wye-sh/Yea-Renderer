// Animation clips in a cooked model (rmodel.h, v3; v4's morph weights): sampling a clip into local poses, and poses into the skinning
// palette. Shared by renderer_models (the runtime) and the model cooker (its encoder checks itself against this decoder).
// Header-only; nothing here allocates.
//
// RmodelCodecQuantized, the Renderer's own codec:
// - Every node's translation, rotation and scale is resampled uniformly (the cooker picks the rate per clip so the skin
//   strays at most about a millimetre between frames), the first frame at 0 and the last at the clip's duration.
// - A track that never changes is one value. Others are three u16 a frame:
//   - translations and scales over the track's own range (base + extent * u / 65535);
//   - rotations smallest-three: the largest of the quaternion's four components is dropped (made positive, and rebuilt
//     from the other three), the other three are 15 bits each over +-1/sqrt(2), and the dropped one's index is the top
//     bit of the first two words.
// - Between frames, translations and scales are lerped and rotations nlerped (on the shorter arc).
//
// RmodelCodecAcl (rmodel v5): Nicholas Frechette's Animation Compression Library (MIT; acl/ and rtm/ here hold only its
// decompression code and Realtime Math, unmodified): the clip's bytes start with an AclClip naming one ACL transform
// track list (a qvvf track a node, its output index the node, sampled at the clip's rate and frames, not looping) and,
// with morph targets, one float1f list (a track a target). The cooker picks it per clip where it is at least as
// accurate at the skin as RmodelCodecQuantized, which stays for the rest. Its decoders allocate nothing.
#pragma once

#include "rmodel.h"

#include <math.h>
#include <string.h>

#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
#endif
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wunknown-warning-option"
#pragma clang diagnostic ignored "-Wmissing-template-arg-list-after-template-kw"
#endif
#include <acl/decompression/decompress.h>
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace anim
{
struct Pose // a node's local transform: scale, then rotation, then translation
{
    float t[3], r[4], s[3];
};

constexpr float kRotationRange = 0.70710678f; // a smallest-three component's bound, 1/sqrt(2)

inline void decodeRotation(const uint16_t* w, float* q)
{
    const uint32_t largest = (w[0] >> 15) | ((w[1] >> 15) << 1);
    float          c[3], sum = 0;
    for (int k = 0; k < 3; ++k)
    {
        const uint32_t u = w[k] & 0x7FFFu;
        c[k] = ((float)u / 32767.0f * 2.0f - 1.0f) * kRotationRange;
        sum += c[k] * c[k];
    }
    for (uint32_t i = 0, k = 0; i < 4; ++i)
        q[i] = i == largest ? sqrtf(fmaxf(0.0f, 1.0f - sum)) : c[k++];
}

inline void encodeRotation(const float* qin, uint16_t* w)
{
    float q[4];
    memcpy(q, qin, 16);
    float len = sqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    len = len > 0 ? len : 1;
    uint32_t largest = 0;
    for (uint32_t i = 0; i < 4; ++i)
    {
        q[i] /= len;
        if (fabsf(q[i]) > fabsf(q[largest]))
            largest = i;
    }
    const float sign = q[largest] < 0 ? -1.0f : 1.0f;
    for (uint32_t i = 0, k = 0; i < 4; ++i)
    {
        if (i == largest)
            continue;
        float v = q[i] * sign / kRotationRange * 0.5f + 0.5f;
        v = v < 0 ? 0 : (v > 1 ? 1 : v);
        w[k++] = (uint16_t)(v * 32767.0f + 0.5f);
    }
    w[0] |= (uint16_t)((largest & 1) << 15);
    w[1] |= (uint16_t)((largest >> 1) << 15);
}

inline void nlerp(const float* a, const float* b, float f, float* out)
{
    const float d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    const float s = d < 0 ? -f : f;
    float       len = 0;
    for (int i = 0; i < 4; ++i)
        out[i] = a[i] * (1 - f) + b[i] * s, len += out[i] * out[i];
    len = len > 0 ? 1.0f / sqrtf(len) : 1.0f;
    for (int i = 0; i < 4; ++i)
        out[i] *= len;
}

// The clip's frame for a time: the two frames around it and how far between. Looping wraps the time into the clip.
inline void frameAt(const RmodelClip& c, float time, bool loop, uint32_t& f0, uint32_t& f1, float& f)
{
    if (c.frames < 2 || !(c.duration > 0)) // one frame: a pose
    {
        f0 = f1 = 0, f = 0;
        return;
    }
    if (loop)
        time = fmodf(time, c.duration), time = time < 0 ? time + c.duration : time;
    time = time < 0 ? 0 : (time > c.duration ? c.duration : time);
    const float x = time * c.rate;
    f0 = (uint32_t)x;
    f0 = f0 > c.frames - 2 ? c.frames - 2 : f0;
    f1 = f0 + 1;
    f = x - (float)f0;
    f = f < 0 ? 0 : (f > 1 ? 1 : f);
}

// RmodelCodecAcl's clip data: offsets from the clip's data of its ACL buffers (16-byte aligned; weights 0: none).
struct AclClip
{
    uint32_t transforms, weights, pad[2];
};
static_assert(sizeof(AclClip) == 16, "acl clip");

// A clip's time as ACL is asked for it: wrapped when looping, inside the clip.
inline float aclTime(const RmodelClip& c, float time, bool loop)
{
    if (!(c.duration > 0))
        return 0;
    if (loop)
        time = fmodf(time, c.duration), time = time < 0 ? time + c.duration : time;
    return time < 0 ? 0 : (time > c.duration ? c.duration : time);
}

struct AclPoses : acl::track_writer // ACL's decoded node poses into Pose[node - first]
{
    Pose*    out;
    uint32_t first, count;
    static constexpr acl::default_sub_track_mode get_default_scale_mode() { return acl::default_sub_track_mode::constant; }
    bool skip(uint32_t i) const { return i < first || i - first >= count; }
    void RTM_SIMD_CALL write_rotation(uint32_t i, rtm::quatf_arg0 q)
    {
        if (!skip(i))
            rtm::quat_store(q, out[i - first].r);
    }
    void RTM_SIMD_CALL write_translation(uint32_t i, rtm::vector4f_arg0 v)
    {
        if (!skip(i))
            rtm::vector_store3(v, out[i - first].t);
    }
    void RTM_SIMD_CALL write_scale(uint32_t i, rtm::vector4f_arg0 v)
    {
        if (!skip(i))
            rtm::vector_store3(v, out[i - first].s);
    }
};

struct AclWeights : acl::track_writer
{
    float*   out;
    uint32_t count;
    void RTM_SIMD_CALL write_float1(uint32_t i, rtm::scalarf_arg0 v)
    {
        if (i < count)
            out[i] = rtm::scalar_cast(v);
    }
};

// Nodes [first, first + count) of an ACL clip at `time`; one node alone when count is 1.
inline bool aclPoses(const uint8_t* data, const RmodelClip& c, uint32_t nodeCount, uint32_t first, uint32_t count, float time, bool loop,
                     Pose* out)
{
    const AclClip& h = *(const AclClip*)data;
    const acl::compressed_tracks* tracks = acl::make_compressed_tracks(data + h.transforms);
    acl::decompression_context<acl::default_transform_decompression_settings> ctx;
    if (!tracks || tracks->get_num_tracks() != nodeCount || !ctx.initialize(*tracks))
        return false;
    ctx.seek(aclTime(c, time, loop), acl::sample_rounding_policy::none);
    AclPoses w;
    w.out = out, w.first = first, w.count = count;
    if (count == 1)
        ctx.decompress_track(first, w);
    else
        ctx.decompress_tracks(w);
    return true;
}

// One node's local pose between two frames (frameAt's f0, f1, f).
inline void sampleNode(const uint8_t* data, uint32_t node, uint32_t f0, uint32_t f1, float f, Pose& out)
{
    const RmodelTrack* tracks = (const RmodelTrack*)data;
    for (uint32_t k = 0; k < 3; ++k)
    {
        const RmodelTrack& tr = tracks[node * 3 + k];
        float*             dst = k == 0 ? out.t : (k == 1 ? out.r : out.s);
        if (tr.data == 0xFFFFFFFFu)
        {
            memcpy(dst, tr.base, k == 1 ? 16 : 12);
            continue;
        }
        const uint16_t* a = (const uint16_t*)(data + tr.data) + f0 * 3;
        const uint16_t* b = (const uint16_t*)(data + tr.data) + f1 * 3;
        if (k == 1)
        {
            float qa[4], qb[4];
            decodeRotation(a, qa);
            decodeRotation(b, qb);
            nlerp(qa, qb, f, dst);
        }
        else
            for (int i = 0; i < 3; ++i)
                dst[i] = tr.base[i] + tr.extent[i] * ((float)a[i] * (1 - f) + (float)b[i] * f) * (1.0f / 65535.0f);
    }
}

// A clip's data (`c.bytes` at `data`) whole in its codec's layout for nodeCount nodes and targetCount morph targets: every
// quantised track's samples inside it, or ACL's buffers inside it, valid, of the right kind and track count.
inline bool aclValid(const uint8_t* data, uint64_t bytes, uint32_t at, acl::track_type8 type, uint32_t tracks)
{
    if (at < sizeof(AclClip) || (at & 15) || ((uintptr_t)(data + at) & 15) || at + (uint64_t)sizeof(acl::compressed_tracks) > bytes)
        return false;
    const acl::compressed_tracks* t = acl::make_compressed_tracks(data + at);
    return t && at + (uint64_t)t->get_size() <= bytes && !t->is_valid(false).any() && t->get_track_type() == type &&
           t->get_num_tracks() == tracks;
}
inline bool clipValid(const uint8_t* data, const RmodelClip& c, uint32_t nodeCount, uint32_t targetCount)
{
    if (c.codec == RmodelCodecAcl)
    {
        if (c.bytes < sizeof(AclClip))
            return false;
        const AclClip& h = *(const AclClip*)data;
        return aclValid(data, c.bytes, h.transforms, acl::track_type8::qvvf, nodeCount) &&
               (!targetCount || aclValid(data, c.bytes, h.weights, acl::track_type8::float1f, targetCount));
    }
    if (c.codec != RmodelCodecQuantized || c.bytes < (uint64_t)(nodeCount * 3 + targetCount) * sizeof(RmodelTrack))
        return false;
    const RmodelTrack* tracks = (const RmodelTrack*)data;
    for (uint32_t t = 0; t < nodeCount * 3 + targetCount; ++t)
        if (tracks[t].data != 0xFFFFFFFFu && tracks[t].data + (uint64_t)c.frames * (t < nodeCount * 3 ? 6 : 2) > c.bytes)
            return false;
    return true;
}

// Every node's local pose at `time` (seconds) into `out` (nodeCount poses). `data` is the clip's bytes (file + offset).
inline bool sampleClip(const uint8_t* data, const RmodelClip& c, uint32_t nodeCount, float time, bool loop, Pose* out)
{
    if (c.codec == RmodelCodecAcl)
        return aclPoses(data, c, nodeCount, 0, nodeCount, time, loop, out);
    if (c.codec != RmodelCodecQuantized)
        return false;
    uint32_t f0, f1;
    float    f;
    frameAt(c, time, loop, f0, f1, f);
    for (uint32_t n = 0; n < nodeCount; ++n)
        sampleNode(data, n, f0, f1, f, out[n]);
    return true;
}

// One node's local pose at `time` (root motion reads the motion node alone).
inline bool sampleNodeAt(const uint8_t* data, const RmodelClip& c, uint32_t node, float time, bool loop, Pose& out)
{
    if (c.codec == RmodelCodecAcl)
    {
        const acl::compressed_tracks* tracks = acl::make_compressed_tracks(data + ((const AclClip*)data)->transforms);
        return tracks && node < tracks->get_num_tracks() && aclPoses(data, c, tracks->get_num_tracks(), node, 1, time, loop, &out);
    }
    if (c.codec != RmodelCodecQuantized)
        return false;
    uint32_t f0, f1;
    float    f;
    frameAt(c, time, loop, f0, f1, f);
    sampleNode(data, node, f0, f1, f, out);
    return true;
}

// Every morph target's weight at `time` into `out` (targetCount floats): the tracks after the nodes' (v4), lerped.
inline bool sampleWeights(const uint8_t* data, const RmodelClip& c, uint32_t nodeCount, uint32_t targetCount, float time, bool loop,
                          float* out)
{
    if (c.codec == RmodelCodecAcl)
    {
        const AclClip& h = *(const AclClip*)data;
        if (!h.weights)
            return false;
        const acl::compressed_tracks* tracks = acl::make_compressed_tracks(data + h.weights);
        acl::decompression_context<acl::default_scalar_decompression_settings> ctx;
        if (!tracks || tracks->get_num_tracks() != targetCount || !ctx.initialize(*tracks))
            return false;
        ctx.seek(aclTime(c, time, loop), acl::sample_rounding_policy::none);
        AclWeights w;
        w.out = out, w.count = targetCount;
        ctx.decompress_tracks(w);
        return true;
    }
    if (c.codec != RmodelCodecQuantized)
        return false;
    uint32_t f0, f1;
    float    f;
    frameAt(c, time, loop, f0, f1, f);
    const RmodelTrack* tracks = (const RmodelTrack*)data + nodeCount * 3;
    for (uint32_t t = 0; t < targetCount; ++t)
    {
        const RmodelTrack& tr = tracks[t];
        if (tr.data == 0xFFFFFFFFu)
        {
            out[t] = tr.base[0];
            continue;
        }
        const uint16_t* u = (const uint16_t*)(data + tr.data);
        out[t] = tr.base[0] + tr.extent[0] * ((float)u[f0] * (1 - f) + (float)u[f1] * f) * (1.0f / 65535.0f);
    }
    return true;
}

// a = lerp(a, b, f) for every node: translation and scale lerped, rotation nlerped (a crossfade between two clips).
inline void blendPoses(Pose* a, const Pose* b, uint32_t count, float f)
{
    for (uint32_t n = 0; n < count; ++n)
    {
        for (int i = 0; i < 3; ++i)
            a[n].t[i] += (b[n].t[i] - a[n].t[i]) * f, a[n].s[i] += (b[n].s[i] - a[n].s[i]) * f;
        float q[4];
        nlerp(a[n].r, b[n].r, f, q);
        memcpy(a[n].r, q, 16);
    }
}

// Inertialization (Bollo, GDC 2018; UE's Inertialization node): at a transition only the new clip plays, and each node
// carries the old pose's offset from it, decayed to nothing by a quintic that ends with no speed or acceleration. Each
// of a node's nine numbers (translation, rotation as axis times angle, scale) decays on its own with its own rate, so the
// pose leaves the switch at the old pose's place and speed; a number heading for zero fast ends sooner, so none goes past
// it, and one moving away carries on a little first (the old motion's momentum) instead of stopping dead.
struct Inertial
{
    float x[9], v[9]; // t, log r (shorter arc, old * new⁻¹), s; and their rates at the switch
};
// x0 with rate v0 at the switch, after t seconds of t1 (in t1's own time, s = t / t1, so a tiny t1 stays finite).
inline float inertialDecay(float x0, float v0, float t, float t1)
{
    const float sg = x0 != 0 ? (x0 < 0 ? -1.0f : 1.0f) : (v0 < 0 ? -1.0f : 1.0f);
    x0 *= sg, v0 *= sg;
    if (!(t1 > 0) || (x0 == 0 && v0 == 0))
        return 0;
    if (v0 * t1 < -5 * x0) // heading for zero fast: over sooner, or it would go past
        t1 = -5 * x0 / v0;
    if (!(t < t1))
        return 0;
    const float V = v0 * t1, a = -8 * V - 20 * x0, Ac = a > 0 ? a : 0; // V and Ac: v0 and a0 in s's time
    const float A = -(Ac + 6 * V + 12 * x0) * 0.5f, B = (3 * Ac + 16 * V + 30 * x0) * 0.5f, C = -(3 * Ac + 12 * V + 20 * x0) * 0.5f;
    const float u = t / t1;
    return sg * (((((A * u + B) * u + C) * u + Ac * 0.5f) * u + V) * u + x0);
}
// A unit quaternion's log: axis times angle, on the shorter arc.
inline void quatLog(const float* q, float* v)
{
    const float sg = q[3] < 0 ? -1.0f : 1.0f, sh = sqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]);
    const float k = sh > 1e-9f ? 2 * atan2f(sh, sg * q[3]) / sh * sg : 2 * sg;
    for (int i = 0; i < 3; ++i)
        v[i] = q[i] * k;
}
inline void quatMul(const float* a, const float* b, float* o)
{
    const float r[4] = { a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1], a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
                         a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3], a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2] };
    memcpy(o, r, 16);
}
// The offsets at a transition: src the old pose now and srcBefore h seconds before, dst the new now and dstAfter h after
// (the new clip starts there, so its rate is taken forward and mirrored back).
inline void inertialOffsets(const Pose* src, const Pose* srcBefore, const Pose* dst, const Pose* dstAfter, float h, uint32_t count,
                            Inertial* out)
{
    for (uint32_t n = 0; n < count; ++n)
    {
        const Pose &s = src[n], &sb = srcBefore[n], &d = dst[n], &da = dstAfter[n];
        Inertial&   o = out[n];
        for (int i = 0; i < 3; ++i)
        {
            o.x[i] = s.t[i] - d.t[i], o.v[i] = (o.x[i] - (sb.t[i] - (2 * d.t[i] - da.t[i]))) / h;
            o.x[6 + i] = s.s[i] - d.s[i], o.v[6 + i] = (o.x[6 + i] - (sb.s[i] - (2 * d.s[i] - da.s[i]))) / h;
        }
        // the new pose h before, mirrored from h after: d * da⁻¹ * d
        const float di[4] = { -d.r[0], -d.r[1], -d.r[2], d.r[3] }, dai[4] = { -da.r[0], -da.r[1], -da.r[2], da.r[3] };
        float       m[4], db[4], dbi[4], q[4], now[3], before[3];
        quatMul(d.r, dai, m), quatMul(m, d.r, db);
        dbi[0] = -db[0], dbi[1] = -db[1], dbi[2] = -db[2], dbi[3] = db[3];
        quatMul(s.r, di, q), quatLog(q, now);
        quatMul(sb.r, dbi, q), quatLog(q, before);
        for (int i = 0; i < 3; ++i)
            o.x[3 + i] = now[i], o.v[3 + i] = (now[i] - before[i]) / h;
    }
}
// The offsets, `t` seconds into a transition of `t1`, put on the new pose (rotation as exp(offset) * pose).
inline void inertialApply(Pose* pose, const Inertial* off, uint32_t count, float t, float t1)
{
    for (uint32_t n = 0; n < count; ++n)
    {
        Pose&           p = pose[n];
        const Inertial& o = off[n];
        float           v[9];
        for (int i = 0; i < 9; ++i)
            v[i] = inertialDecay(o.x[i], o.v[i], t, t1);
        for (int i = 0; i < 3; ++i)
            p.t[i] += v[i], p.s[i] += v[6 + i];
        const float a = sqrtf(v[3] * v[3] + v[4] * v[4] + v[5] * v[5]);
        if (a > 0)
        {
            const float k = sinf(a * 0.5f) / a, e[4] = { v[3] * k, v[4] * k, v[5] * k, cosf(a * 0.5f) };
            quatMul(e, p.r, p.r);
        }
    }
}

// A pose as a 3x4 matrix (rows): T * R * S.
inline void poseMatrix(const Pose& p, float* m)
{
    const float x = p.r[0], y = p.r[1], z = p.r[2], w = p.r[3];
    const float r[9] = { 1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w),
                         2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                         2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y) };
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
            m[row * 4 + col] = r[row * 3 + col] * p.s[col];
        m[row * 4 + 3] = p.t[row];
    }
}

// out = a * b, 3x4 affine matrices (rows).
inline void mul34(const float* a, const float* b, float* out)
{
    float o[12];
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 4; ++col)
            o[row * 4 + col] = a[row * 4] * b[col] + a[row * 4 + 1] * b[4 + col] + a[row * 4 + 2] * b[8 + col];
        o[row * 4 + 3] += a[row * 4 + 3];
    }
    memcpy(out, o, sizeof o);
}

// Local poses to each node's model-space matrix (`world`, 12 floats a node), then the palette (12 floats a bone):
// world(bone.node) * bone.invBind.
inline void palette(const RmodelNode* nodes, uint32_t nodeCount, const RmodelBone* bones, uint32_t boneCount, const Pose* poses,
                    float* world, float* out)
{
    for (uint32_t n = 0; n < nodeCount; ++n)
    {
        float local[12];
        poseMatrix(poses[n], local);
        if (nodes[n].parent >= 0)
            mul34(&world[nodes[n].parent * 12], local, &world[n * 12]);
        else
            memcpy(&world[n * 12], local, sizeof local);
    }
    for (uint32_t b = 0; b < boneCount; ++b)
        mul34(&world[bones[b].node * 12], bones[b].invBind, &out[b * 12]);
}

// Layers (UE's Layered Blend per Bone): `in` marks the nodes a layer covers (a bone and everything under it, or all).
// Override: each covered node's pose moves toward the layer's by w (lerped, rotations nlerped). Additive: each covered
// node takes on what the layer did since its reference pose `ref` (its clip's first frame), times w, on top of its own:
// rotation base * slerp(1, ref⁻¹ * layer, w), translation base + w (layer - ref), scale base * lerp(1, layer / ref, w).
inline void layerMask(const RmodelNode* nodes, uint32_t count, uint32_t root, uint8_t* in)
{
    for (uint32_t n = 0; n < count; ++n)
        in[n] = root >= count || n == root || (nodes[n].parent >= 0 && in[nodes[n].parent]);
}
inline void layerOverride(Pose* base, const Pose* layer, const uint8_t* in, uint32_t count, float w)
{
    for (uint32_t n = 0; n < count; ++n)
        if (in[n])
        {
            for (int i = 0; i < 3; ++i)
                base[n].t[i] += (layer[n].t[i] - base[n].t[i]) * w, base[n].s[i] += (layer[n].s[i] - base[n].s[i]) * w;
            float q[4];
            nlerp(base[n].r, layer[n].r, w, q);
            memcpy(base[n].r, q, 16);
        }
}
inline void layerAdd(Pose* base, const Pose* layer, const Pose* ref, const uint8_t* in, uint32_t count, float w)
{
    for (uint32_t n = 0; n < count; ++n)
        if (in[n])
        {
            const float* a = ref[n].r;
            const float* b = layer[n].r;
            // d = a⁻¹ b (the turn the layer added), on the shorter arc, scaled by w from no turn
            float d[4] = { a[3] * b[0] - a[0] * b[3] - a[1] * b[2] + a[2] * b[1], a[3] * b[1] + a[0] * b[2] - a[1] * b[3] - a[2] * b[0],
                           a[3] * b[2] - a[0] * b[1] + a[1] * b[0] - a[2] * b[3], a[3] * b[3] + a[0] * b[0] + a[1] * b[1] + a[2] * b[2] };
            if (d[3] < 0)
                for (float& c : d)
                    c = -c;
            const float id[4] = { 0, 0, 0, 1 };
            float       dw[4];
            const float h = acosf(d[3] > 1 ? 1 : d[3]), sh = sinf(h);
            if (sh > 1e-6f) // slerp from no turn: the angle scaled
            {
                const float k = sinf(w * h) / sh;
                for (int i = 0; i < 3; ++i)
                    dw[i] = d[i] * k;
                dw[3] = cosf(w * h);
            }
            else
                nlerp(id, d, w, dw);
            const float* r = base[n].r;
            const float  o[4] = { r[3] * dw[0] + r[0] * dw[3] + r[1] * dw[2] - r[2] * dw[1], r[3] * dw[1] - r[0] * dw[2] + r[1] * dw[3] + r[2] * dw[0],
                                  r[3] * dw[2] + r[0] * dw[1] - r[1] * dw[0] + r[2] * dw[3], r[3] * dw[3] - r[0] * dw[0] - r[1] * dw[1] - r[2] * dw[2] };
            memcpy(base[n].r, o, 16);
            for (int i = 0; i < 3; ++i)
            {
                base[n].t[i] += w * (layer[n].t[i] - ref[n].t[i]);
                base[n].s[i] *= 1 + w * (ref[n].s[i] != 0 ? layer[n].s[i] / ref[n].s[i] - 1 : 0);
            }
        }
}

// The rest pose of every node.
inline void restPoses(const RmodelNode* nodes, uint32_t nodeCount, Pose* out)
{
    for (uint32_t n = 0; n < nodeCount; ++n)
    {
        memcpy(out[n].t, nodes[n].t, 12);
        memcpy(out[n].r, nodes[n].r, 16);
        memcpy(out[n].s, nodes[n].s, 12);
    }
}
} // namespace anim
