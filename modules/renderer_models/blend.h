// Blend spaces (UE's Blend Space and Blend Space 1D): each sample's weight at a point of the space.
// - 1D: the samples sorted by x; the point between two neighbours weighs them linearly, past either end the end one.
// - 2D: the samples triangulated (Delaunay: no sample inside a triangle's circumcircle; where four lie on one circle, as
//   on a grid, the first triangles in sample order win and none overlap), the point's triangle weighs its three corners
//   barycentrically; outside every triangle, the nearest point on a triangle's edge weighs that edge's two ends. Fewer
//   than three samples, or all on one line: 1D along whichever axis they spread over.
// Weights are exact at samples (1 there, 0 for the rest), continuous across the space and sum to 1. No allocation.
// Shared by renderer_models.cpp and nothing else.
#pragma once

#include <math.h>
#include <stdint.h>

namespace blend
{
constexpr uint32_t kMaxSamples = 16;
constexpr uint32_t kMaxTris = 2 * kMaxSamples; // a triangulation of n points has at most 2n - 5

struct Tris
{
    uint32_t count;
    uint8_t  t[kMaxTris][3];
};

inline float cross2(const float* o, const float* a, const float* b) { return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]); }
// Whether p lies strictly inside the triangle a, b, c (either winding).
inline bool inside(const float* a, const float* b, const float* c, const float* p, float eps)
{
    const float s = cross2(a, b, c) > 0 ? 1.0f : -1.0f;
    return s * cross2(a, b, p) > eps && s * cross2(b, c, p) > eps && s * cross2(c, a, p) > eps;
}
// Whether segments ab and cd cross at a point inside both (not at an end).
inline bool crosses(const float* a, const float* b, const float* c, const float* d, float eps)
{
    const float d1 = cross2(a, b, c), d2 = cross2(a, b, d), d3 = cross2(c, d, a), d4 = cross2(c, d, b);
    return ((d1 > eps && d2 < -eps) || (d1 < -eps && d2 > eps)) && ((d3 > eps && d4 < -eps) || (d3 < -eps && d4 > eps));
}

// p (n points, 2 floats each) triangulated into out. Its scale: the spread of the points, for tolerances.
inline void triangulate(const float* p, uint32_t n, Tris& out)
{
    out.count = 0;
    if (n < 3 || n > kMaxSamples)
        return;
    float lo[2] = { p[0], p[1] }, hi[2] = { p[0], p[1] };
    for (uint32_t i = 1; i < n; ++i)
        for (int k = 0; k < 2; ++k)
            lo[k] = fminf(lo[k], p[i * 2 + k]), hi[k] = fmaxf(hi[k], p[i * 2 + k]);
    const float span = fmaxf(hi[0] - lo[0], hi[1] - lo[1]);
    if (!(span > 0))
        return;
    const float area = 1e-6f * span * span, len = 1e-5f * span;
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t j = i + 1; j < n; ++j)
            for (uint32_t k = j + 1; k < n && out.count < kMaxTris; ++k)
            {
                const float *A = &p[i * 2], *B = &p[j * 2], *C = &p[k * 2];
                const float  d = 2 * cross2(A, B, C);
                if (fabsf(d) <= 2 * area)
                    continue; // flat
                // its circumcircle (relative to A)
                const float bx = B[0] - A[0], by = B[1] - A[1], cx = C[0] - A[0], cy = C[1] - A[1];
                const float b2 = bx * bx + by * by, c2 = cx * cx + cy * cy;
                const float ux = (cy * b2 - by * c2) / d, uy = (bx * c2 - cx * b2) / d, r2 = ux * ux + uy * uy;
                bool ok = true;
                for (uint32_t m = 0; m < n && ok; ++m)
                {
                    if (m == i || m == j || m == k)
                        continue;
                    const float dx = p[m * 2] - A[0] - ux, dy = p[m * 2 + 1] - A[1] - uy;
                    ok = dx * dx + dy * dy >= r2 * (1 - 1e-5f) - len * len; // on the circle counts as outside
                }
                // none overlapping one already taken (four on one circle make two ways to split them)
                for (uint32_t q = 0; q < out.count && ok; ++q)
                {
                    const float* T[3] = { &p[out.t[q][0] * 2], &p[out.t[q][1] * 2], &p[out.t[q][2] * 2] };
                    const float* U[3] = { A, B, C };
                    for (int e = 0; e < 3 && ok; ++e)
                        for (int f = 0; f < 3 && ok; ++f)
                            ok = !crosses(T[e], T[(e + 1) % 3], U[f], U[(f + 1) % 3], area);
                    const float ct[2] = { (T[0][0] + T[1][0] + T[2][0]) / 3, (T[0][1] + T[1][1] + T[2][1]) / 3 };
                    const float cu[2] = { (A[0] + B[0] + C[0]) / 3, (A[1] + B[1] + C[1]) / 3 };
                    ok = ok && !inside(T[0], T[1], T[2], cu, area) && !inside(A, B, C, ct, area);
                }
                if (ok)
                    out.t[out.count][0] = (uint8_t)i, out.t[out.count][1] = (uint8_t)j, out.t[out.count][2] = (uint8_t)k, ++out.count;
            }
}

// 1D weights of n samples at x (xs their positions) into w.
inline void weights1(const float* xs, uint32_t n, float x, float* w)
{
    for (uint32_t i = 0; i < n; ++i)
        w[i] = 0;
    if (!n)
        return;
    uint32_t lo = 0, hi = 0; // the nearest at or below x, and at or above it
    bool     below = false, above = false;
    for (uint32_t i = 0; i < n; ++i)
    {
        if (xs[i] <= x && (!below || xs[i] > xs[lo]))
            lo = i, below = true;
        if (xs[i] >= x && (!above || xs[i] < xs[hi]))
            hi = i, above = true;
    }
    if (!below)
        lo = hi;
    if (!above)
        hi = lo;
    if (lo == hi || xs[hi] <= xs[lo])
    {
        w[lo] = 1;
        return;
    }
    const float f = (x - xs[lo]) / (xs[hi] - xs[lo]);
    w[lo] = 1 - f, w[hi] = f;
}

// 2D weights of n samples (p: 2 floats each, triangulated into tris) at (x, y) into w.
inline void weights2(const float* p, uint32_t n, const Tris& tris, float x, float y, float* w)
{
    for (uint32_t i = 0; i < n; ++i)
        w[i] = 0;
    if (!tris.count) // too few, or on one line: 1D along the axis they spread over most
    {
        float lo[2] = { 1e30f, 1e30f }, hi[2] = { -1e30f, -1e30f }, xs[kMaxSamples];
        for (uint32_t i = 0; i < n; ++i)
            for (int k = 0; k < 2; ++k)
                lo[k] = fminf(lo[k], p[i * 2 + k]), hi[k] = fmaxf(hi[k], p[i * 2 + k]);
        const int axis = hi[1] - lo[1] > hi[0] - lo[0] ? 1 : 0;
        for (uint32_t i = 0; i < n && i < kMaxSamples; ++i)
            xs[i] = p[i * 2 + axis];
        weights1(xs, n < kMaxSamples ? n : kMaxSamples, axis ? y : x, w);
        return;
    }
    const float P[2] = { x, y };
    float       best = 1e30f, bw[2] = { 1, 0 };
    uint32_t    be[2] = { tris.t[0][0], tris.t[0][1] };
    for (uint32_t q = 0; q < tris.count; ++q)
    {
        const uint8_t* t = tris.t[q];
        const float *  A = &p[t[0] * 2], *B = &p[t[1] * 2], *C = &p[t[2] * 2];
        const float    d = cross2(A, B, C);
        const float    u = cross2(B, C, P) / d, v = cross2(C, A, P) / d, s = 1 - u - v;
        if (u >= -1e-6f && v >= -1e-6f && s >= -1e-6f)
        {
            const float tot = fmaxf(u, 0) + fmaxf(v, 0) + fmaxf(s, 0);
            w[t[0]] = fmaxf(u, 0) / tot, w[t[1]] = fmaxf(v, 0) / tot, w[t[2]] = fmaxf(s, 0) / tot;
            return;
        }
        for (int e = 0; e < 3; ++e) // outside: the nearest point on its edges
        {
            const float *E0 = &p[t[e] * 2], *E1 = &p[t[(e + 1) % 3] * 2];
            const float  ex = E1[0] - E0[0], ey = E1[1] - E0[1], l2 = ex * ex + ey * ey;
            float        f = l2 > 0 ? ((x - E0[0]) * ex + (y - E0[1]) * ey) / l2 : 0;
            f = f < 0 ? 0 : (f > 1 ? 1 : f);
            const float dx = E0[0] + ex * f - x, dy = E0[1] + ey * f - y, dd = dx * dx + dy * dy;
            if (dd < best)
                best = dd, be[0] = t[e], be[1] = t[(e + 1) % 3], bw[0] = 1 - f, bw[1] = f;
        }
    }
    w[be[0]] += bw[0], w[be[1]] += bw[1];
}
} // namespace blend
