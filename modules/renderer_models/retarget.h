// Retargeting: one skeleton's clips played on another's, as UE's IK Retargeter does it.
// - Chains: each skeleton's pelvis, spine, neck and head, clavicles, arms, legs and toes, found from its bones' names
//   (Mixamo's, UE's, Blender's, Bip01's and the like: "LeftUpLeg", "thigh_l", "DEF-thigh.L"), and every other bone by
//   its name (fingers, a tail). A chain maps onto the same chain of the other skeleton.
// - FK: each target bone turns as its source bone turned from its rest pose (in model space, so bones whose axes differ
//   still match), bone for bone where a chain has as many bones on both sides, else interpolated along the chain.
// - The pelvis moves as the source's, scaled by the ratio of their pelvis heights (its height above the model's origin).
// - IK: two-bone IK on the legs and arms, the knee or elbow bending in the plane FK left it in, reaching for the
//   source's foot or hand scaled the same way (the legs offset by where their feet stand at rest, so a planted foot
//   stays planted and stands on the ground; the arms not, so hands that meet still meet). Feet and hands keep FK's turn.
//   Where a leg's goal is out of its reach, the pelvis first comes down by the least that brings both within reach.
// Rest poses are the nodes' (as the file has them). Shared by nothing else: only renderer_models includes it.
#pragma once

#include "anim.h"
#include "rmodel.h"

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace retarget
{
constexpr uint32_t kNone = 0xFFFFFFFFu;
constexpr uint32_t kChainMax = 16;

// --- Quaternions (x, y, z, w) and vectors --------------------------------------------------------------------------
inline void qmul(const float* a, const float* b, float* o)
{
    const float r[4] = { a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1], a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
                         a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3], a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2] };
    memcpy(o, r, 16);
}
inline void qconj(const float* a, float* o) { o[0] = -a[0], o[1] = -a[1], o[2] = -a[2], o[3] = a[3]; }
inline void qnorm(float* q)
{
    const float l = sqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (l > 0)
        for (int i = 0; i < 4; ++i)
            q[i] /= l;
}
inline void qrot(const float* q, const float* v, float* o)
{
    const float t[3] = { 2 * (q[1] * v[2] - q[2] * v[1]), 2 * (q[2] * v[0] - q[0] * v[2]), 2 * (q[0] * v[1] - q[1] * v[0]) };
    const float r[3] = { v[0] + q[3] * t[0] + (q[1] * t[2] - q[2] * t[1]), v[1] + q[3] * t[1] + (q[2] * t[0] - q[0] * t[2]),
                         v[2] + q[3] * t[2] + (q[0] * t[1] - q[1] * t[0]) };
    memcpy(o, r, 12);
}
inline void qnlerp(const float* a, const float* b, float f, float* o)
{
    const float s = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3] < 0 ? -1.0f : 1.0f;
    for (int i = 0; i < 4; ++i)
        o[i] = a[i] + (s * b[i] - a[i]) * f;
    qnorm(o);
}
inline float dot(const float* a, const float* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
inline float len(const float* a) { return sqrtf(dot(a, a)); }
inline void  cross(const float* a, const float* b, float* o)
{
    const float r[3] = { a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0] };
    memcpy(o, r, 12);
}
inline void sub(const float* a, const float* b, float* o) { o[0] = a[0] - b[0], o[1] = a[1] - b[1], o[2] = a[2] - b[2]; }
inline bool unit(float* v)
{
    const float l = len(v);
    if (!(l > 1e-12f))
        return false;
    v[0] /= l, v[1] /= l, v[2] /= l;
    return true;
}
// The shortest turn taking unit vector a onto unit vector b.
inline void arc(const float* a, const float* b, float* q)
{
    const float d = dot(a, b);
    if (d < -0.999999f) // opposite: half a turn about any axis across a
    {
        float x[3] = { 1, 0, 0 }, ax[3];
        if (fabsf(a[0]) > 0.9f)
            x[0] = 0, x[1] = 1;
        cross(a, x, ax);
        unit(ax);
        q[0] = ax[0], q[1] = ax[1], q[2] = ax[2], q[3] = 0;
        return;
    }
    float c[3];
    cross(a, b, c);
    q[0] = c[0], q[1] = c[1], q[2] = c[2], q[3] = 1 + d;
    qnorm(q);
}
inline void at(const float* world, uint32_t n, float* p) { p[0] = world[n * 12 + 3], p[1] = world[n * 12 + 7], p[2] = world[n * 12 + 11]; }
// p in the frame of the 3x4 m (rows): m⁻¹ p. False when m is flat.
inline bool unplace(const float* m, const float* p, float* out)
{
    const float c[9] = { m[5] * m[10] - m[6] * m[9], m[6] * m[8] - m[4] * m[10], m[4] * m[9] - m[5] * m[8],
                         m[2] * m[9] - m[1] * m[10], m[0] * m[10] - m[2] * m[8], m[1] * m[8] - m[0] * m[9],
                         m[1] * m[6] - m[2] * m[5], m[2] * m[4] - m[0] * m[6], m[0] * m[5] - m[1] * m[4] };
    const float det = m[0] * c[0] + m[1] * c[1] + m[2] * c[2];
    if (fabsf(det) < 1e-20f)
        return false;
    const float d[3] = { p[0] - m[3], p[1] - m[7], p[2] - m[11] };
    for (int r = 0; r < 3; ++r) // (m⁻¹)[r][k] = c[k * 3 + r] / det
        out[r] = (c[r] * d[0] + c[3 + r] * d[1] + c[6 + r] * d[2]) / det;
    return true;
}

// Local poses to each node's model-space matrix (12 floats a node) and its turn (4 floats a node).
inline void worlds(const RmodelNode* nodes, uint32_t count, const anim::Pose* poses, float* world, float* turn)
{
    for (uint32_t n = 0; n < count; ++n)
    {
        float local[12];
        anim::poseMatrix(poses[n], local);
        const int32_t p = nodes[n].parent;
        if (p >= 0)
            anim::mul34(&world[p * 12], local, &world[n * 12]), qmul(&turn[p * 4], poses[n].r, &turn[n * 4]);
        else
            memcpy(&world[n * 12], local, sizeof local), memcpy(&turn[n * 4], poses[n].r, 16);
        qnorm(&turn[n * 4]);
    }
}

// --- Names ---------------------------------------------------------------------------------------------------------
enum Role : uint8_t
{
    RoleNone,
    RolePelvis,
    RoleNeck,
    RoleHead,
    RoleClavicle,
    RoleUpperArm,
    RoleHand,
    RoleThigh,
    RoleFoot,
    RoleToe,
    Roles
};
struct Name
{
    char    key[64]; // its side and its words: "l:hand_index_1"
    uint8_t role, side; // side 0 none, 1 left, 2 right
};
inline bool in(const char* s, const char* const* list)
{
    for (; *list; ++list)
        if (!strcmp(s, *list))
            return true;
    return false;
}
// A bone's name in words, its side and what it is.
inline void parse(const char* raw, Name& out)
{
    char name[49];
    strncpy(name, raw, 48), name[48] = 0;
    const char* s = name;
    for (const char* p = name; *p; ++p)
        if (*p == ':' || *p == '|')
            s = p + 1;
    char tok[12][24];
    int  nt = 0, ln = 0;
    char prev = 0;
    auto flush = [&] {
        if (ln && nt < 12)
            tok[nt][ln] = 0, ++nt;
        ln = 0;
    };
    for (const char* p = s; *p; ++p)
    {
        const unsigned char c = (unsigned char)*p;
        if (!isalnum(c))
        {
            flush(), prev = 0;
            continue;
        }
        if (ln && ((isupper(c) && islower((unsigned char)prev)) || (!isdigit(c) != !isdigit((unsigned char)prev))))
            flush();
        if (nt < 12 && ln < 23)
            tok[nt][ln++] = (char)tolower(c);
        prev = (char)c;
    }
    flush();
    static const char* const prefixes[] = { "mixamorig", "bip", "def", "org", "armature", "bn", "jnt", "j", "b", "bone",
                                            "cc", "skel", "rig", "valve", "biped", nullptr };
    int first = 0;
    while (nt - first > 1 && (in(tok[first], prefixes) || isdigit((unsigned char)tok[first][0])))
        ++first;
    out.side = 0;
    static const char* const skips[] = { "twist", "roll", "ik", "fk", "pole", "target", "helper", "corrective", "end", "nub",
                                         "tip", "socket", "attach", "prop", "weapon", "ctrl", "mch", "null", "dummy", nullptr };
    bool skip = false;
    char base[64] = "";
    int  k = snprintf(out.key, sizeof out.key, "n:");
    for (int i = first; i < nt; ++i)
    {
        if (!out.side && (!strcmp(tok[i], "l") || !strcmp(tok[i], "left")))
        {
            out.side = 1;
            continue;
        }
        if (!out.side && (!strcmp(tok[i], "r") || !strcmp(tok[i], "right")))
        {
            out.side = 2;
            continue;
        }
        skip = skip || in(tok[i], skips);
        if (k < (int)sizeof out.key - 1)
            k += snprintf(out.key + k, sizeof out.key - k, "%s%s", k > 2 ? "_" : "", tok[i]);
        if (!isdigit((unsigned char)tok[i][0]) && strlen(base) + strlen(tok[i]) < sizeof base)
            strcat(base, tok[i]);
    }
    out.key[0] = out.side == 1 ? 'l' : out.side == 2 ? 'r' : 'n';
    out.role = RoleNone;
    if (skip)
        return;
    static const char* const pelvis[] = { "hips", "hip", "pelvis", nullptr };
    static const char* const clavicle[] = { "clavicle", "shoulder", "collar", "collarbone", nullptr };
    static const char* const upperArm[] = { "upperarm", "arm", "uparm", "humerus", nullptr };
    static const char* const hand[] = { "hand", "wrist", nullptr };
    static const char* const thigh[] = { "thigh", "upleg", "upperleg", "femur", "hip", nullptr };
    static const char* const foot[] = { "foot", "ankle", nullptr };
    if (!out.side)
        out.role = in(base, pelvis) ? RolePelvis : strstr(base, "neck") ? RoleNeck : !strcmp(base, "head") ? RoleHead : RoleNone;
    else
        out.role = in(base, clavicle) ? RoleClavicle
                 : in(base, upperArm) ? RoleUpperArm
                 : in(base, hand)     ? RoleHand
                 : in(base, thigh)    ? RoleThigh
                 : in(base, foot)     ? RoleFoot
                 : (!strncmp(base, "toe", 3) || !strcmp(base, "ball")) ? RoleToe
                                                                        : RoleNone;
}

// --- Chains --------------------------------------------------------------------------------------------------------
enum Chain
{
    ChainPelvis,
    ChainSpine,
    ChainHead,
    ChainClavicleL,
    ChainClavicleR,
    ChainArmL,
    ChainArmR,
    ChainLegL,
    ChainLegR,
    ChainToeL,
    ChainToeR,
    Chains
};
struct Found
{
    uint32_t count[Chains];
    uint32_t node[Chains][kChainMax];
};
inline uint32_t depth(const RmodelNode* nodes, uint32_t n)
{
    uint32_t d = 0;
    for (int32_t p = nodes[n].parent; p >= 0 && d < 4096; p = nodes[p].parent)
        ++d;
    return d;
}
// The nodes from `from` down to `to` (both in), when from is to's ancestor and the chain fits; else 0.
inline uint32_t path(const RmodelNode* nodes, uint32_t from, uint32_t to, uint32_t* out)
{
    if (from == kNone || to == kNone)
        return 0;
    uint32_t up[kChainMax], n = 0;
    for (int32_t x = (int32_t)to; x >= 0 && n < kChainMax; x = nodes[x].parent)
    {
        up[n++] = (uint32_t)x;
        if ((uint32_t)x == from)
        {
            for (uint32_t i = 0; i < n; ++i)
                out[i] = up[n - 1 - i];
            return n;
        }
    }
    return 0;
}
inline void chains(const RmodelNode* nodes, uint32_t count, const Name* names, Found& f)
{
    uint32_t role[Roles][3];
    for (auto& r : role)
        r[0] = r[1] = r[2] = kNone;
    for (uint32_t n = 0; n < count; ++n) // each role's shallowest bone
    {
        uint32_t& r = role[names[n].role][names[n].side];
        if (names[n].role != RoleNone && (r == kNone || depth(nodes, n) < depth(nodes, r)))
            r = n;
    }
    memset(&f, 0, sizeof f);
    uint32_t       tmp[kChainMax];
    const uint32_t pelvis = role[RolePelvis][0], head = role[RoleHead][0], neck = role[RoleNeck][0];
    if (pelvis != kNone)
        f.count[ChainPelvis] = 1, f.node[ChainPelvis][0] = pelvis;
    const uint32_t top = neck != kNone && path(nodes, neck, head, tmp) ? neck : head;
    if (const uint32_t n = path(nodes, pelvis, top, tmp); n >= 2)
    {
        f.count[ChainSpine] = n - 2; // between the pelvis and the neck
        memcpy(f.node[ChainSpine], tmp + 1, (n - 2) * 4);
        f.count[ChainHead] = path(nodes, top, head, f.node[ChainHead]);
    }
    for (uint32_t side = 1; side <= 2; ++side)
    {
        const uint32_t clavicle = role[RoleClavicle][side], arm = role[RoleUpperArm][side], hand = role[RoleHand][side];
        const uint32_t thigh = role[RoleThigh][side], foot = role[RoleFoot][side], toe = role[RoleToe][side];
        const int      o = side == 1 ? 0 : 1;
        if (clavicle != arm && path(nodes, clavicle, arm, tmp))
            f.count[ChainClavicleL + o] = 1, f.node[ChainClavicleL + o][0] = clavicle;
        f.count[ChainArmL + o] = path(nodes, arm, hand, f.node[ChainArmL + o]);
        f.count[ChainLegL + o] = path(nodes, thigh, foot, f.node[ChainLegL + o]);
        if (foot != toe && path(nodes, foot, toe, tmp))
            f.count[ChainToeL + o] = 1, f.node[ChainToeL + o][0] = toe;
    }
}

// --- The map -------------------------------------------------------------------------------------------------------
// A model's skeleton as retargeting reads it: its nodes, its turn upright (the model's up to +Y), its morph targets.
struct Skel
{
    const RmodelNode*        nodes;
    uint32_t                 count;
    float                    up[4];
    const RmodelMorphTarget* targets;
    uint32_t                 targetCount;
    uint32_t                 motion; // its root motion node (kNone none)
};
struct NodeMap
{
    uint32_t s0, s1; // the source bones it follows (s0 kNone: its own rest pose)
    float    f;      // between them
};
struct Ik
{
    uint32_t a, b, c;   // the target's upper, lower and end (b is c's parent)
    uint32_t src;       // the source's end
    float    offset[3]; // added to the scaled source end (legs: where its foot stands at rest)
    float    bend[3];   // the knee's or elbow's side when FK leaves the limb straight, model space at rest
    uint32_t leg;       // a leg: its goal can lower the pelvis
};
struct Map
{
    uint32_t tCount, sCount;
    float    ratio;      // the target's pelvis height over the source's
    float    turn[4];    // the source's model space to the target's
    uint32_t tRoot, sRoot;
    float    rootOffset[3];
    float    up[3];      // the target's up, model space
    uint32_t ikCount;
    Ik       ik[4];
    NodeMap* nodes;      // tCount
    float *  restS, *restT; // each node's turn at rest, model space
    uint32_t chains, byName; // what it found, for its log line
    uint32_t morph[kRmodelMaxTargets]; // each target morph target's source target (kNone: its default weight)
};
inline void release(Map& m)
{
    free(m.nodes);
    m.nodes = nullptr, m.restS = m.restT = nullptr;
}
// Model-space positions along a chain at rest, from 0 at its first bone to 1 at its last.
inline void along(const float* world, const uint32_t* chain, uint32_t n, float* u)
{
    float total = 0;
    u[0] = 0;
    for (uint32_t i = 1; i < n; ++i)
    {
        float a[3], b[3], d[3];
        at(world, chain[i - 1], a), at(world, chain[i], b), sub(b, a, d);
        total += len(d), u[i] = total;
    }
    for (uint32_t i = 1; i < n; ++i)
        u[i] = total > 0 ? u[i] / total : (float)i / (float)(n - 1);
}
// Made in the module's own update (it allocates). False: the two have nothing in common.
inline bool build(const Skel& t, const Skel& s, Map& m)
{
    memset(&m, 0, sizeof m);
    m.tCount = t.count, m.sCount = s.count, m.tRoot = m.sRoot = kNone;
    const size_t tn = t.count, sn = s.count;
    uint8_t*     block = (uint8_t*)malloc(tn * sizeof(NodeMap) + (tn + sn) * 16);
    uint8_t*     tmp = (uint8_t*)malloc((tn + sn) * (sizeof(anim::Pose) + sizeof(Name) + 64));
    if (!block || !tmp)
    {
        free(block), free(tmp);
        return false;
    }
    m.nodes = (NodeMap*)block;
    m.restT = (float*)(block + tn * sizeof(NodeMap));
    m.restS = m.restT + tn * 4;
    anim::Pose* tp = (anim::Pose*)tmp;
    anim::Pose* sp = tp + tn;
    Name*       tName = (Name*)(sp + sn);
    Name*       sName = tName + tn;
    float*      tw = (float*)(sName + sn);
    float*      sw = tw + tn * 12;
    anim::restPoses(t.nodes, t.count, tp), anim::restPoses(s.nodes, s.count, sp);
    worlds(t.nodes, t.count, tp, tw, m.restT), worlds(s.nodes, s.count, sp, sw, m.restS);
    for (uint32_t n = 0; n < t.count; ++n)
        parse(t.nodes[n].name, tName[n]), m.nodes[n] = { kNone, kNone, 0 };
    for (uint32_t n = 0; n < s.count; ++n)
        parse(s.nodes[n].name, sName[n]);
    qconj(t.up, m.turn), qmul(m.turn, s.up, m.turn), qnorm(m.turn);

    Found tf, sf;
    chains(t.nodes, t.count, tName, tf), chains(s.nodes, s.count, sName, sf);
    for (int c = 0; c < Chains; ++c)
    {
        const uint32_t nt = tf.count[c], ns = sf.count[c];
        if (!nt || !ns)
            continue;
        ++m.chains;
        float ut[kChainMax], us[kChainMax];
        along(tw, tf.node[c], nt, ut), along(sw, sf.node[c], ns, us);
        for (uint32_t i = 0; i < nt; ++i)
        {
            NodeMap& nm = m.nodes[tf.node[c][i]];
            if (nt == ns || ns == 1) // bone for bone
            {
                nm.s0 = nm.s1 = sf.node[c][ns == 1 ? 0 : i];
                continue;
            }
            const float u = nt == 1 ? 0.0f : ut[i];
            uint32_t    j = 0;
            while (j + 2 < ns && us[j + 1] < u)
                ++j;
            const float span = us[j + 1] - us[j];
            nm.s0 = sf.node[c][j], nm.s1 = sf.node[c][j + 1];
            nm.f = span > 0 ? fminf(fmaxf((u - us[j]) / span, 0.0f), 1.0f) : 0.0f;
        }
    }
    // Every other bone with the same name on both sides (fingers, a tail, the root above a UE pelvis).
    for (uint32_t n = 0; n < t.count; ++n)
        if (m.nodes[n].s0 == kNone)
            for (uint32_t k = 0; k < s.count; ++k)
                if (!strcmp(tName[n].key, sName[k].key))
                {
                    m.nodes[n].s0 = m.nodes[n].s1 = k, ++m.byName;
                    break;
                }
    if (!m.chains && !m.byName)
    {
        free(tmp), release(m);
        return false;
    }
    // The retarget root: the pelvis, else the source's root motion node where the target has a bone of its name.
    if (tf.count[ChainPelvis] && sf.count[ChainPelvis])
        m.tRoot = tf.node[ChainPelvis][0], m.sRoot = sf.node[ChainPelvis][0];
    else if (s.motion != kNone)
        for (uint32_t n = 0; n < t.count && m.tRoot == kNone; ++n)
            if (m.nodes[n].s0 == s.motion)
                m.tRoot = n, m.sRoot = s.motion;
    m.ratio = 1;
    if (m.tRoot != kNone)
    {
        const float yy[3] = { 0, 1, 0 };
        float upT[3], upS[3], pt[3], ps[3], c[4];
        qconj(t.up, c), qrot(c, yy, upT);
        memcpy(m.up, upT, 12);
        qconj(s.up, c), qrot(c, yy, upS);
        at(tw, m.tRoot, pt), at(sw, m.sRoot, ps);
        const float ht = dot(pt, upT), hs = dot(ps, upS);
        if (ht > 1e-4f && hs > 1e-4f)
            m.ratio = ht / hs;
        float q[3];
        qrot(m.turn, ps, q);
        for (int k = 0; k < 3; ++k)
            m.rootOffset[k] = pt[k] - m.ratio * q[k];
    }
    // Two-bone IK on the limbs both sides have, each upper and lower distinct.
    for (int c = ChainArmL; c <= ChainLegR; ++c)
    {
        const uint32_t nt = tf.count[c], ns = sf.count[c];
        if (nt < 3 || !ns || m.ikCount >= 4)
            continue;
        Ik& k = m.ik[m.ikCount];
        k.c = tf.node[c][nt - 1];
        k.b = (uint32_t)t.nodes[k.c].parent;
        k.a = tf.node[c][0];
        if (k.b == k.a)
            continue;
        k.src = sf.node[c][ns - 1];
        const bool leg = c >= ChainLegL;
        float      pt[3], ps[3], q[3];
        at(tw, k.c, pt), at(sw, k.src, ps), qrot(m.turn, ps, q);
        for (int i = 0; i < 3; ++i)
            k.offset[i] = leg ? pt[i] - m.ratio * q[i] : 0.0f;
        // straight at rest: knees bend to the model's front (-Z, the cooker's), elbows to its back
        const float front[3] = { 0, 0, leg ? -1.0f : 1.0f };
        float       cu[4];
        qconj(t.up, cu), qrot(cu, front, k.bend);
        k.leg = leg ? 1 : 0;
        ++m.ikCount;
    }
    for (uint32_t i = 0; i < t.targetCount && i < kRmodelMaxTargets; ++i)
    {
        m.morph[i] = kNone;
        for (uint32_t j = 0; j < s.targetCount; ++j)
            if (!strncmp(t.targets[i].name, s.targets[j].name, sizeof t.targets[i].name))
                m.morph[i] = j;
    }
    free(tmp);
    return true;
}

// --- IK (pose() and footik.h) ------------------------------------------------------------------------------------
// The least the pelvis must come down along `up` for the two-bone limb A-B-C to reach G: 0 when it already can.
inline float reachDrop(const float* A, const float* B, const float* C, const float* G, const float* up)
{
    float ab[3], bc[3], v[3];
    sub(B, A, ab), sub(C, B, bc), sub(G, A, v);
    const float L = (len(ab) + len(bc)) * 0.9998f, vu = dot(v, up), vv = dot(v, v);
    if (vv <= L * L)
        return 0;
    const float disc = vu * vu - (vv - L * L); // |v + d up| = L: the smaller root; none, the nearest point
    return disc >= 0 ? -vu - sqrtf(disc) : -vu;
}
// `root` and everything under it moved by d along `up` (model space; world as worlds() made it), and root's local
// position to match. At most 1024 nodes.
inline void shift(const RmodelNode* nodes, uint32_t count, uint32_t root, const float* up, float d, float* world, anim::Pose* out)
{
    static const float id12[12] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 };
    uint8_t            under[1024]; // the root and everything under it
    for (uint32_t n = 0; n < count && n < 1024; ++n)
    {
        const int32_t pa = nodes[n].parent;
        under[n] = n == root || (pa >= 0 && under[pa]);
        if (under[n])
            for (int j = 0; j < 3; ++j)
                world[n * 12 + j * 4 + 3] += d * up[j];
    }
    const int32_t pa = nodes[root].parent;
    float         P[3];
    at(world, root, P);
    unplace(pa >= 0 ? &world[pa * 12] : id12, P, out[root].t);
}
// Two-bone IK: the upper a and lower b turned (their local turns in out) so the end c reaches G, c taking the
// model-space turn endTurn (nullptr: keeping FK's); world and turn (model space) as FK left them, not updated. The knee or elbow bends in the plane FK
// left it in, else (FK straight) to `bend` (model space at rest) turned as a's parent turned from its rest (parentRest;
// nullptr: a has no parent). False: the limb has no length or no side to bend to, and stays as it was.
inline bool limb(const RmodelNode* nodes, const float* world, const float* turn, uint32_t a, uint32_t b, uint32_t c, const float* G,
                 const float* bend, const float* parentRest, anim::Pose* out, const float* endTurn = nullptr)
{
    static const float idq[4] = { 0, 0, 0, 1 };
    float              A[3], B[3], C[3];
    at(world, a, A), at(world, b, B), at(world, c, C);
    float ab[3], bc[3], ag[3];
    sub(B, A, ab), sub(C, B, bc), sub(G, A, ag);
    const float lab = len(ab), lbc = len(bc), dg = len(ag);
    if (!(lab > 1e-6f) || !(lbc > 1e-6f) || !(dg > 1e-6f))
        return false;
    float u[3] = { ag[0] / dg, ag[1] / dg, ag[2] / dg };
    const float lo = fabsf(lab - lbc) * 1.001f + 1e-5f, hi = (lab + lbc) * 0.9999f;
    const float d = dg < lo ? lo : (dg > hi ? hi : dg);
    // the knee's side: where FK bent it (its knee off the line from its hip to its end), else, FK straight, the
    // chain's front turned as its parent turned; then square to the line to the goal
    float kd[3], ac[3];
    sub(C, A, ac);
    if (!unit(ac))
        return false;
    float du = dot(ab, ac);
    for (int j = 0; j < 3; ++j)
        kd[j] = ab[j] - du * ac[j];
    const int32_t pa = nodes[a].parent, pb = nodes[b].parent;
    if (len(kd) < 1e-3f * lab)
    {
        float q[4] = { 0, 0, 0, 1 }, cj[4];
        if (pa >= 0 && parentRest)
            qconj(parentRest, cj), qmul(&turn[pa * 4], cj, q);
        qrot(q, bend, kd);
    }
    du = dot(kd, u);
    for (int j = 0; j < 3; ++j)
        kd[j] -= du * u[j];
    if (!unit(kd))
        return false;
    float ca = (lab * lab + d * d - lbc * lbc) / (2 * lab * d);
    ca = ca < -1 ? -1 : (ca > 1 ? 1 : ca);
    const float sa = sqrtf(1 - ca * ca);
    float       B2[3], T2[3], ab2[3], qa[4], qb[4], C1[3], cb[3], tb[3], v[3];
    for (int j = 0; j < 3; ++j)
        B2[j] = A[j] + lab * (ca * u[j] + sa * kd[j]), T2[j] = A[j] + d * u[j];
    float abn[3] = { ab[0] / lab, ab[1] / lab, ab[2] / lab };
    sub(B2, A, ab2), unit(ab2), arc(abn, ab2, qa);
    sub(C, A, v), qrot(qa, v, C1);
    for (int j = 0; j < 3; ++j)
        C1[j] += A[j];
    sub(C1, B2, cb), sub(T2, B2, tb);
    if (!unit(cb) || !unit(tb))
        return false;
    arc(cb, tb, qb);
    // new turns: the upper's subtree by qa, then the end's parent's subtree by qb
    float ga[4], gbp[4], gb[4], cj[4], q[4];
    qmul(qa, &turn[a * 4], ga);
    qconj(pa >= 0 ? &turn[pa * 4] : idq, cj), qmul(cj, ga, out[a].r), qnorm(out[a].r);
    qmul(qa, &turn[pb * 4], gbp); // b's parent, turned with the upper
    qmul(qb, qa, q), qmul(q, &turn[b * 4], gb);
    qconj(gbp, cj), qmul(cj, gb, out[b].r), qnorm(out[b].r);
    qconj(gb, cj), qmul(cj, endTurn ? endTurn : &turn[c * 4], out[c].r), qnorm(out[c].r);
    return true;
}

// Scratch for pose(), kMaxNodes of each (made in the module's load()).
struct Scratch
{
    float *sw, *sq, *tw, *tq;
};

// The source's local poses (as sampled; in place when asked) as the target's.
inline void pose(const Map& m, const Skel& t, const Skel& s, const anim::Pose* src, anim::Pose* out, const Scratch& sc)
{
    worlds(s.nodes, s.count, src, sc.sw, sc.sq);
    static const float id12[12] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 }, idq[4] = { 0, 0, 0, 1 };
    float              turnC[4];
    qconj(m.turn, turnC);
    // FK, parents first: each mapped bone turned in model space as its source bone turned from its rest.
    for (uint32_t n = 0; n < t.count; ++n)
    {
        const RmodelNode& nd = t.nodes[n];
        anim::Pose&       p = out[n];
        memcpy(p.t, nd.t, 12), memcpy(p.r, nd.r, 16), memcpy(p.s, nd.s, 12);
        const int32_t pa = nd.parent;
        const float*  pq = pa >= 0 ? &sc.tq[pa * 4] : idq;
        const float*  pw = pa >= 0 ? &sc.tw[pa * 12] : id12;
        const NodeMap& nm = m.nodes[n];
        if (nm.s0 != kNone)
        {
            float d[4], d1[4], c[4], g[4];
            qconj(&m.restS[nm.s0 * 4], c), qmul(&sc.sq[nm.s0 * 4], c, d);
            if (nm.s1 != nm.s0 && nm.f > 0)
                qconj(&m.restS[nm.s1 * 4], c), qmul(&sc.sq[nm.s1 * 4], c, d1), qnlerp(d, d1, nm.f, d);
            qmul(m.turn, d, g), qmul(g, turnC, g); // the turn in the target's model space
            qmul(g, &m.restT[n * 4], g);
            qconj(pq, c), qmul(c, g, p.r), qnorm(p.r);
        }
        if (n == m.tRoot)
        {
            float ps[3], P[3];
            at(sc.sw, m.sRoot, ps), qrot(m.turn, ps, P);
            for (int k = 0; k < 3; ++k)
                P[k] = m.ratio * P[k] + m.rootOffset[k];
            unplace(pw, P, p.t);
        }
        float local[12];
        anim::poseMatrix(p, local);
        anim::mul34(pw, local, &sc.tw[n * 12]);
        qmul(pq, p.r, &sc.tq[n * 4]), qnorm(&sc.tq[n * 4]);
    }
    auto goal = [&](const Ik& k, float* G) {
        float ps[3];
        at(sc.sw, k.src, ps), qrot(m.turn, ps, G);
        for (int j = 0; j < 3; ++j)
            G[j] = m.ratio * G[j] + k.offset[j];
    };
    // The pelvis lowered by the least that brings every leg's goal within its reach (as UE's IK Rig does), so a
    // shorter-legged target's feet stay planted instead of hanging short of the ground.
    if (m.tRoot != kNone && t.count <= 1024)
    {
        float drop = 0;
        for (uint32_t i = 0; i < m.ikCount; ++i)
        {
            const Ik& k = m.ik[i];
            if (!k.leg)
                continue;
            float A[3], B[3], C[3], G[3];
            at(sc.tw, k.a, A), at(sc.tw, k.b, B), at(sc.tw, k.c, C), goal(k, G);
            const float d = reachDrop(A, B, C, G, m.up);
            drop = d > drop ? d : drop;
        }
        if (drop > 0)
            shift(t.nodes, t.count, m.tRoot, m.up, -drop, sc.tw, out);
    }
    // IK: each limb's upper and lower turned so its end reaches the goal; its end keeps FK's turn.
    for (uint32_t i = 0; i < m.ikCount; ++i)
    {
        const Ik&     k = m.ik[i];
        const int32_t pa = t.nodes[k.a].parent;
        float         G[3];
        goal(k, G);
        limb(t.nodes, sc.tw, sc.tq, k.a, k.b, k.c, G, k.bend, pa >= 0 ? &m.restT[pa * 4] : nullptr, out);
    }
}
// A source clip's root motion (model-space travel and turn) as the target's.
inline void motion(const Map& m, float* io)
{
    float v[3];
    qrot(m.turn, io, v);
    io[0] = v[0] * m.ratio, io[1] = v[1] * m.ratio, io[2] = v[2] * m.ratio;
}
} // namespace retarget
