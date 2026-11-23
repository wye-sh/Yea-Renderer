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
// A codec with better ratios (ACL) can sit beside it under another RmodelCodec, behind sampleClip.
#pragma once

#include "rmodel.h"

#include <math.h>
#include <string.h>

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

// Every node's local pose at `time` (seconds) into `out` (nodeCount poses). `data` is the clip's bytes (file + offset).
inline bool sampleClip(const uint8_t* data, const RmodelClip& c, uint32_t nodeCount, float time, bool loop, Pose* out)
{
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
