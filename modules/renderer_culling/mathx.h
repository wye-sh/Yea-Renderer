// Small vector math for the renderer modules (no SIMD types, so it stays plain data).
#pragma once
#include <math.h>
#include <stdint.h>

namespace mx
{
struct v3
{
    float x, y, z;
};
inline v3    V(float x, float y, float z) { return { x, y, z }; }
inline v3    operator+(v3 a, v3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
inline v3    operator-(v3 a, v3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
inline v3    operator-(v3 a) { return { -a.x, -a.y, -a.z }; }
inline v3    operator*(v3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
inline float dot(v3 a, v3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline v3    cross(v3 a, v3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
inline float length(v3 a) { return sqrtf(dot(a, a)); }
inline v3    normalize(v3 a)
{
    float l = length(a);
    return l > 1e-20f ? a * (1.0f / l) : V(0, 0, 1);
}
inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline float minf(float a, float b) { return a < b ? a : b; }
inline float maxf(float a, float b) { return a > b ? a : b; }

// An orthonormal basis with f as its third axis.
inline void basis(v3 f, v3* right, v3* up)
{
    v3 helper = fabsf(f.y) < 0.99f ? V(0, 1, 0) : V(1, 0, 0);
    *right = normalize(cross(helper, f));
    *up = cross(f, *right);
}

// A plane (n, d): inside where dot(n, p) + d >= 0.
struct Plane
{
    v3    n;
    float d;
};
inline bool sphereInside(const Plane* planes, uint32_t count, v3 c, float r)
{
    for (uint32_t i = 0; i < count; ++i)
        if (dot(planes[i].n, c) + planes[i].d < -r)
            return false;
    return true;
}
// The six planes of a clip matrix given as rows (D3D clip, 0 <= z <= w): left, right, bottom, top,
// z >= 0, z <= w. With reversed-Z the last is the near plane and the fifth is degenerate.
inline void clipPlanes(const float m[4][4], Plane* out)
{
    const float* r0 = m[0];
    const float* r1 = m[1];
    const float* r2 = m[2];
    const float* r3 = m[3];
    float p[6][4];
    for (int i = 0; i < 4; ++i)
    {
        p[0][i] = r3[i] + r0[i];
        p[1][i] = r3[i] - r0[i];
        p[2][i] = r3[i] + r1[i];
        p[3][i] = r3[i] - r1[i];
        p[4][i] = r2[i];
        p[5][i] = r3[i] - r2[i];
    }
    for (uint32_t i = 0; i < 6; ++i)
    {
        float l = sqrtf(p[i][0] * p[i][0] + p[i][1] * p[i][1] + p[i][2] * p[i][2]);
        float s = l > 0 ? 1.0f / l : 0.0f;
        out[i] = { V(p[i][0] * s, p[i][1] * s, p[i][2] * s), p[i][3] * s };
    }
}

// Row-major 4x4 product: out = a * b.
inline void mul(const float a[4][4], const float b[4][4], float out[4][4])
{
    float t[4][4];
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            t[r][c] = a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c] + a[r][3] * b[3][c];
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            out[r][c] = t[r][c];
}

// World -> view rows for a camera at eye looking along f (unit), +z forward, +y up, +x right.
inline void lookRows(v3 eye, v3 f, v3 right, v3 up, float out[4][4])
{
    float rows[4][4] = { { right.x, right.y, right.z, -dot(right, eye) },
                         { up.x, up.y, up.z, -dot(up, eye) },
                         { f.x, f.y, f.z, -dot(f, eye) },
                         { 0, 0, 0, 1 } };
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            out[r][c] = rows[r][c];
}
} // namespace mx
