// Fits the renderer's area-light tables: linearly transformed cosines (LTC; Heitz, Dupuy, Hill and Neubelt,
// "Real-Time Polygonal-Light Shading with Linearly Transformed Cosines", SIGGRAPH 2016) to the core's own specular
// BRDF (GGX D, height-correlated Smith visibility, as brdfLight in Shaders/Lighting.h.fsl), plus the horizon table
// for the sphere approximation of a clipped polygon (Hill and Heitz, "Real-Time Area Lighting: a Journey From
// Research to Production", SIGGRAPH 2016 course). Written for this renderer; it runs offline, not in the game.
//
//   c++ -O2 -std=c++17 -pthread ltc_fit.cpp -o ltc_fit && ./ltc_fit ../modules/renderer/Assets/ltc.bin
//   ./ltc_fit --check ../modules/renderer/Assets/ltc.bin    (LTC against Monte Carlo of the BRDF over test lights)
//
// Tables are kN x kN over x = perceptual roughness and y = sqrt(1 - N.V), cell centres at i / (kN - 1).
// ltc.bin: "LTC1", kN, then kN * kN float4 (inverse M: m00, m02, m20, m22, with m11 = 1; V in the xz plane, N = z),
// kN * kN float4 (x: the BRDF's integral with F = 1 (norm), y: its integral weighted by Schlick's (1 - V.H)^5, z: the
// sphere horizon table at (x: form factor, y: elevation z / 2 + 1 / 2), w 0), all little-endian float32.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <thread>
#include <vector>

namespace
{
const int    kN = 64;
const double kPi = 3.14159265358979323846;
const double kMinAlpha = 1e-4;

struct V3
{
    double x, y, z;
};
V3     operator+(V3 a, V3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
V3     operator-(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
V3     operator*(V3 a, double s) { return { a.x * s, a.y * s, a.z * s }; }
double dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
V3     cross(V3 a, V3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
double len(V3 a) { return sqrt(dot(a, a)); }
V3     norm(V3 a) { return a * (1.0 / len(a)); }

struct M3
{
    double m[3][3];
    V3     mul(V3 v) const
    {
        return { m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z, m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
                 m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z };
    }
    double det() const
    {
        return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
               m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    }
    M3 inverse() const
    {
        double d = 1.0 / det();
        M3     r;
        r.m[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * d;
        r.m[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * d;
        r.m[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * d;
        r.m[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * d;
        r.m[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * d;
        r.m[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * d;
        r.m[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * d;
        r.m[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * d;
        r.m[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * d;
        return r;
    }
};
M3 mul(const M3& a, const M3& b)
{
    M3 r;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            r.m[i][j] = a.m[i][0] * b.m[0][j] + a.m[i][1] * b.m[1][j] + a.m[i][2] * b.m[2][j];
    return r;
}

// --- The core's specular BRDF times cos (F = 1), in the frame N = z, V in the xz plane ----------------
struct Brdf
{
    double a2;
    V3     v;
    double smithV; // sqrt(NoV^2 (1 - a2) + a2)

    Brdf(double alpha, double cosV)
    {
        a2 = alpha * alpha;
        v = { sqrt(fmax(1.0 - cosV * cosV, 0.0)), 0.0, cosV };
        smithV = sqrt(cosV * cosV * (1.0 - a2) + a2);
    }
    // f cos towards l, and the pdf of sampling l from the GGX normal distribution (D(h) N.H / (4 V.H)).
    double eval(V3 l, double* pdf) const
    {
        *pdf = 0.0;
        if (l.z <= 0.0)
            return 0.0;
        V3     h = norm(v + l);
        double noh = h.z, voh = fmax(dot(v, h), 1e-12);
        double d = (noh * a2 - noh) * noh + 1.0;
        double D = a2 / (kPi * d * d);
        double gv = l.z * smithV, gl = v.z * sqrt(l.z * l.z * (1.0 - a2) + a2);
        double vis = 0.5 / fmax(gv + gl, 1e-12);
        *pdf = D * noh / (4.0 * voh);
        return D * vis * l.z;
    }
    V3 sample(double u1, double u2) const
    {
        double phi = 2.0 * kPi * u1;
        double ct = sqrt((1.0 - u2) / (1.0 + (a2 - 1.0) * u2));
        double st = sqrt(fmax(1.0 - ct * ct, 0.0));
        V3     h = { st * cos(phi), st * sin(phi), ct };
        return h * (2.0 * dot(v, h)) - v;
    }
};

// --- An LTC: the clamped cosine through M ----------------------------------------------------------
struct Ltc
{
    M3     M, invM;
    double detInv;
    void   set(const M3& m)
    {
        M = m;
        invM = m.inverse();
        detInv = fabs(invM.det());
    }
    double pdf(V3 l) const
    {
        V3     lo = invM.mul(l);
        double n = len(lo);
        lo = lo * (1.0 / n);
        return fmax(lo.z, 0.0) / kPi * detInv / (n * n * n);
    }
    V3 sample(double u1, double u2) const
    {
        double r = sqrt(u1), phi = 2.0 * kPi * u2;
        V3     lo = { r * cos(phi), r * sin(phi), sqrt(fmax(1.0 - u1, 0.0)) };
        return norm(M.mul(lo));
    }
};

double radicalInverse(uint32_t i)
{
    i = (i << 16) | (i >> 16);
    i = ((i & 0x55555555u) << 1) | ((i & 0xAAAAAAAAu) >> 1);
    i = ((i & 0x33333333u) << 2) | ((i & 0xCCCCCCCCu) >> 2);
    i = ((i & 0x0F0F0F0Fu) << 4) | ((i & 0xF0F0F0F0u) >> 4);
    i = ((i & 0x00FF00FFu) << 8) | ((i & 0xFF00FF00u) >> 8);
    return (double)i * 2.3283064365386963e-10;
}

// The BRDF's integral (norm), its Fresnel-weighted integral and its mean direction, by importance sampling.
void moments(const Brdf& b, double* normOut, double* fresOut, V3* avgOut)
{
    const uint32_t n = 1 << 16;
    double         sum = 0, fres = 0;
    V3             avg = { 0, 0, 0 };
    for (uint32_t i = 0; i < n; ++i)
    {
        V3     l = b.sample((i + 0.5) / n, radicalInverse(i));
        double pdf, f = b.eval(l, &pdf);
        if (pdf <= 0.0 || f <= 0.0)
            continue;
        double w = f / pdf;
        V3     h = norm(b.v + l);
        double x = 1.0 - fmax(dot(b.v, h), 0.0);
        sum += w;
        fres += w * x * x * x * x * x;
        avg = avg + l * w;
    }
    *normOut = sum / n;
    *fresOut = fres / n;
    avg.y = 0.0;
    *avgOut = norm(avg);
}

// How far the LTC is from the normalised BRDF: the cubed difference of the two densities, estimated with samples from
// both (balance heuristic), fixed samples so the error is a smooth function of M.
double fitError(const Ltc& ltc, const Brdf& b, double brdfNorm)
{
    const int n = 24;
    double    err = 0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            double u1 = (i + 0.5) / n, u2 = (j + 0.5) / n;
            for (int which = 0; which < 2; ++which)
            {
                V3     l = which ? b.sample(u1, u2) : ltc.sample(u1, u2);
                double pdfB, f = b.eval(l, &pdfB) / brdfNorm;
                double pdfL = ltc.pdf(l);
                double d = fabs(f - pdfL);
                double p = pdfB + pdfL;
                if (p > 0.0)
                    err += d * d * d / p;
            }
        }
    return err / (2.0 * n * n);
}

// Nelder-Mead in 3 dimensions.
template <typename F>
void nelderMead(double* x, double step, int iters, F f)
{
    const int D = 3;
    double    p[D + 1][D], v[D + 1];
    for (int i = 0; i <= D; ++i)
    {
        for (int k = 0; k < D; ++k)
            p[i][k] = x[k] + (i == k + 1 ? step : 0.0);
        v[i] = f(p[i]);
    }
    for (int it = 0; it < iters; ++it)
    {
        int lo = 0, hi = 0, nh = 0;
        for (int i = 1; i <= D; ++i)
        {
            if (v[i] < v[lo])
                lo = i;
            if (v[i] > v[hi])
                hi = i;
        }
        nh = lo;
        for (int i = 0; i <= D; ++i)
            if (i != hi && v[i] > v[nh])
                nh = i;
        if (fabs(v[hi] - v[lo]) <= 1e-9 * (fabs(v[lo]) + 1e-12))
            break;
        double c[D] = {};
        for (int i = 0; i <= D; ++i)
            if (i != hi)
                for (int k = 0; k < D; ++k)
                    c[k] += p[i][k] / D;
        double r[D], e[D], q[D];
        for (int k = 0; k < D; ++k)
            r[k] = c[k] + (c[k] - p[hi][k]);
        double vr = f(r);
        if (vr < v[lo])
        {
            for (int k = 0; k < D; ++k)
                e[k] = c[k] + 2.0 * (c[k] - p[hi][k]);
            double ve = f(e);
            if (ve < vr)
                memcpy(p[hi], e, sizeof e), v[hi] = ve;
            else
                memcpy(p[hi], r, sizeof r), v[hi] = vr;
            continue;
        }
        if (vr < v[nh])
        {
            memcpy(p[hi], r, sizeof r), v[hi] = vr;
            continue;
        }
        bool outside = vr < v[hi];
        for (int k = 0; k < D; ++k)
            q[k] = outside ? c[k] + 0.5 * (r[k] - c[k]) : c[k] + 0.5 * (p[hi][k] - c[k]);
        double vq = f(q);
        if (vq < (outside ? vr : v[hi]))
        {
            memcpy(p[hi], q, sizeof q), v[hi] = vq;
            continue;
        }
        for (int i = 0; i <= D; ++i)
            if (i != lo)
            {
                for (int k = 0; k < D; ++k)
                    p[i][k] = p[lo][k] + 0.5 * (p[i][k] - p[lo][k]);
                v[i] = f(p[i]);
            }
    }
    int lo = 0;
    for (int i = 1; i <= D; ++i)
        if (v[i] < v[lo])
            lo = i;
    memcpy(x, p[lo], sizeof(double) * D);
}

double alphaOf(int i) { return fmax(pow((double)i / (kN - 1), 2.0), kMinAlpha); } // perceptual roughness squared
double cosOf(int j)
{
    double y = (double)j / (kN - 1); // sqrt(1 - N.V)
    return fmax(1.0 - y * y, 1e-4);
}

// M for parameters x (m00, m11, m02 in the frame of the BRDF's mean direction).
M3 frameM(const double* x, V3 avg)
{
    V3 t2 = { 0, 1, 0 }, t1 = cross(t2, avg);
    M3 R = { { { t1.x, t2.x, avg.x }, { t1.y, t2.y, avg.y }, { t1.z, t2.z, avg.z } } };
    M3 L = { { { x[0], 0, x[2] }, { 0, x[1], 0 }, { 0, 0, 1 } } };
    return mul(R, L);
}

struct Cell
{
    float inv[4];
    float norm, fres;
    float err;
};

void fitColumn(int j, Cell* out)
{
    // From rough to smooth, each cell starting from the last: a smooth BRDF's lobe is hard to find from scratch.
    double x[3] = { 1, 1, 0 };
    bool   first = true;
    for (int i = kN - 1; i >= 0; --i)
    {
        Brdf   b(alphaOf(i), cosOf(j));
        double nrm, fres;
        V3     avg;
        moments(b, &nrm, &fres, &avg);
        if (first)
        {
            x[0] = x[1] = 1.0;
            x[2] = 0.0;
        }
        else if (i < kN - 1 && alphaOf(i) < 0.05 && j == 0)
            x[2] = 0.0;
        Ltc  ltc;
        auto f = [&](const double* p) {
            if (p[0] <= 1e-7 || p[1] <= 1e-7)
                return 1e30;
            ltc.set(frameM(p, avg));
            return fitError(ltc, b, nrm);
        };
        nelderMead(x, first ? 0.05 : 0.005 + 0.05 * x[0], 400, f);
        first = false;
        ltc.set(frameM(x, avg));
        M3     inv = ltc.invM;
        double s = inv.m[1][1];
        Cell&  c = out[j * kN + i];
        c.inv[0] = (float)(inv.m[0][0] / s);
        c.inv[1] = (float)(inv.m[0][2] / s);
        c.inv[2] = (float)(inv.m[2][0] / s);
        c.inv[3] = (float)(inv.m[2][2] / s);
        c.norm = (float)nrm;
        c.fres = (float)fres;
        c.err = (float)f(x);
    }
}

// The clamped cosine over a spherical cap whose unclipped form factor is f (sin^2 of its half-angle), centred at
// elevation z (the cosine of its angle from N), divided by pi: what a polygon of form-factor vector length f and
// direction z contributes once clipped to the horizon, by the sphere approximation.
double sphereHorizon(double f, double z)
{
    double sigma = asin(sqrt(fmin(fmax(f, 0.0), 1.0)));
    double omega = acos(fmin(fmax(z, -1.0), 1.0));
    if (omega + sigma <= kPi * 0.5)
        return f * z; // wholly above the horizon: the form factor, tilted
    if (omega - sigma >= kPi * 0.5)
        return 0.0;
    // Integrate over the cap numerically: a grid in its own polar coordinates.
    const int n = 256;
    double    sum = 0, ct0 = cos(sigma);
    V3        c = { sin(omega), 0, cos(omega) }, t1 = { cos(omega), 0, -sin(omega) }, t2 = { 0, 1, 0 };
    for (int a = 0; a < n; ++a)
        for (int b = 0; b < n; ++b)
        {
            double ct = 1.0 - (1.0 - ct0) * (a + 0.5) / n, st = sqrt(fmax(1.0 - ct * ct, 0.0)), phi = 2.0 * kPi * (b + 0.5) / n;
            V3     d = c * ct + t1 * (st * cos(phi)) + t2 * (st * sin(phi));
            sum += fmax(d.z, 0.0);
        }
    double solid = 2.0 * kPi * (1.0 - ct0);
    return sum / (n * n) * solid / kPi;
}

// --- Checking: the LTC integral over a polygon against Monte Carlo of the BRDF -------------------
// The polygon's integral of the clamped cosine (Lambert's edge formula, as the shader computes it), clipped to the
// horizon exactly here.
int clipHorizon(const V3* in, int n, V3* out)
{
    int m = 0;
    for (int i = 0; i < n; ++i)
    {
        V3 a = in[i], b = in[(i + 1) % n];
        if (a.z >= 0)
            out[m++] = a;
        if ((a.z >= 0) != (b.z >= 0))
            out[m++] = a + (b - a) * (a.z / (a.z - b.z));
    }
    return m;
}
double polygonCosine(const V3* p, int n)
{
    V3  q[16];
    int m = clipHorizon(p, n, q);
    double sum = 0;
    for (int i = 0; i < m; ++i)
    {
        V3     a = norm(q[i]), b = norm(q[(i + 1) % m]);
        double c = fmin(fmax(dot(a, b), -1.0), 1.0);
        V3     x = cross(a, b);
        double lx = len(x);
        if (lx > 1e-12)
            sum += acos(c) * x.z / lx;
    }
    return fabs(sum) / (2.0 * kPi);
}

// The form-factor vector of an unclipped polygon (sum of its edges' vectors), as the shader forms it.
V3 formFactor(const V3* p, int n)
{
    V3 f = { 0, 0, 0 };
    for (int i = 0; i < n; ++i)
    {
        V3     a = norm(p[i]), b = norm(p[(i + 1) % n]);
        double c = fmin(fmax(dot(a, b), -1.0), 1.0);
        V3     x = cross(a, b);
        double lx = len(x);
        if (lx > 1e-12)
            f = f + x * (acos(c) / lx);
    }
    return f * (1.0 / (2.0 * kPi));
}

// An ellipse's form factor (centre C, semi-axes V1 and V2, seen from the origin) and its cone's axis, as the shader
// computes them: the cone through it is the quadratic form Q = B^-T diag(1, 1, -1) B^-1 (B = [V1 V2 C]); its negative
// eigenvalue's vector is the axis and the two positive ones give the half-angles, tan^2 = -l3 / li, so the form
// factor is sin s1 sin s2 (sin^2 s for a circle).
double ellipseFormFactor(V3 C, V3 V1, V3 V2, V3* axis)
{
    M3 B = { { { V1.x, V2.x, C.x }, { V1.y, V2.y, C.y }, { V1.z, V2.z, C.z } } };
    M3 Bi = B.inverse();
    M3 Q;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            Q.m[i][j] = Bi.m[0][i] * Bi.m[0][j] + Bi.m[1][i] * Bi.m[1][j] - Bi.m[2][i] * Bi.m[2][j];
    // Eigenvalues of a symmetric 3 x 3 (the trigonometric solution of its cubic).
    double q = (Q.m[0][0] + Q.m[1][1] + Q.m[2][2]) / 3.0;
    double p1 = Q.m[0][1] * Q.m[0][1] + Q.m[0][2] * Q.m[0][2] + Q.m[1][2] * Q.m[1][2];
    double p2 = (Q.m[0][0] - q) * (Q.m[0][0] - q) + (Q.m[1][1] - q) * (Q.m[1][1] - q) + (Q.m[2][2] - q) * (Q.m[2][2] - q) + 2 * p1;
    double pp = sqrt(p2 / 6.0);
    M3     Bm = Q;
    for (int i = 0; i < 3; ++i)
        Bm.m[i][i] -= q;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            Bm.m[i][j] /= pp;
    double r = fmin(fmax(Bm.det() / 2.0, -1.0), 1.0), phi = acos(r) / 3.0;
    double e1 = q + 2 * pp * cos(phi), e3 = q + 2 * pp * cos(phi + 2.0 * kPi / 3.0), e2 = 3 * q - e1 - e3; // e1 >= e2 >= e3
    double t1 = -e3 / e1, t2 = -e3 / e2;
    // The axis: a vector of Q - e3 I's null space, the longest cross product of two of its rows.
    V3 r0 = { Q.m[0][0] - e3, Q.m[0][1], Q.m[0][2] }, r1 = { Q.m[1][0], Q.m[1][1] - e3, Q.m[1][2] }, r2 = { Q.m[2][0], Q.m[2][1], Q.m[2][2] - e3 };
    V3 a = cross(r0, r1), b = cross(r1, r2), c = cross(r2, r0);
    V3 best = dot(a, a) > dot(b, b) ? a : b;
    best = dot(best, best) > dot(c, c) ? best : c;
    best = norm(best);
    *axis = dot(best, C) < 0 ? best * -1.0 : best;
    return sqrt(t1 / (1 + t1) * t2 / (1 + t2));
}

double horizonLookup(const float* horizon, double f, double z)
{
    double hx = fmin(f, 1.0) * (kN - 1), hy = (z * 0.5 + 0.5) * (kN - 1);
    int    h0 = (int)hx, g0 = (int)hy, h1 = h0 + 1 < kN ? h0 + 1 : h0, g1 = g0 + 1 < kN ? g0 + 1 : g0;
    double fx = hx - h0, fy = hy - g0;
    auto   H = [&](int i, int j) { return (double)horizon[j * kN + i]; };
    return (H(h0, g0) * (1 - fx) + H(h1, g0) * fx) * (1 - fy) + (H(h0, g1) * (1 - fx) + H(h1, g1) * fx) * fy;
}

// Discs (ellipses): the clamped cosine over them, Monte Carlo against the eigen form (no LTC: this checks the geometry).
void checkDiscs(const float* horizon)
{
    struct Disc
    {
        V3          c, u, v;
        const char* name;
    } discs[] = { { { 0, 0, 2 }, { 0.8, 0, 0 }, { 0, 0.8, 0 }, "overhead disc" },
                  { { 1, 0.3, 1 }, { 0, 0.7, 0 }, { -0.3, 0, 0.3 }, "tilted ellipse" },
                  { { 1.5, 0, 0.3 }, { 0, 0.8, 0 }, { 0, 0, 0.5 }, "ellipse crossing the horizon" },
                  { { 0.2, 0, 0.6 }, { 0.5, 0, 0.2 }, { 0, 0.9, 0 }, "close and large" } };
    for (const Disc& d : discs)
    {
        // Reference: uniform samples over the ellipse's area, each its solid angle times its cosine.
        V3       n = norm(cross(d.u, d.v));
        double   area = kPi * len(d.u) * len(d.v) * fabs(dot(norm(cross(d.u, d.v)), n));
        double   ref = 0;
        const int k = 1024;
        for (int i = 0; i < k; ++i)
            for (int j = 0; j < k; ++j)
            {
                double r = sqrt((i + 0.5) / k), ph = 2 * kPi * (j + 0.5) / k;
                V3     p = d.c + d.u * (r * cos(ph)) + d.v * (r * sin(ph));
                double l2 = dot(p, p), l = sqrt(l2);
                V3     w = p * (1 / l);
                ref += fmax(w.z, 0.0) * fabs(dot(w, n)) / l2;
            }
        ref *= area / ((double)k * k) / kPi;
        V3     axis;
        double ff = ellipseFormFactor(d.c, d.u, d.v, &axis);
        double est = horizonLookup(horizon, ff, axis.z);
        printf("%-30s reference %.5f  eigen form + horizon %.5f  (%.2f%%)\n", d.name, ref, est, fabs(est - ref) / ref * 100);
    }
}

// Tubes: a capsule (a segment of half-length h along a, radius r, centre c) of uniform luminance, against the shader's
// way: the rectangle along it that faces the point, reaching to its round ends' tips, through the LTC and the sphere
// horizon (reaching only as far as the ends' area, r pi / 4, left highlights just past a tip 35% dim). Errors as for the rects (of all the BRDF reflects), and of the tube's own light.
bool hitsCapsule(V3 d, V3 c, V3 a, double h, double r)
{
    // The ray from the origin along d against the segment: the closest points of the two, clamped to each.
    V3     p0 = c - a * h, u = a * (2 * h);
    double uu = dot(u, u), ud = dot(u, d), up = dot(u, p0), dp = dot(d, p0);
    double den = uu - ud * ud; // |d| = 1
    double s = den > 1e-12 ? (ud * dp - up) / den : 0.0; // along the segment
    s = fmin(fmax(s, 0.0), 1.0);
    double t = fmax(dot(p0 + u * s, d), 0.0); // along the ray, for that point
    s = fmin(fmax((dot(d * t - p0, u)) / uu, 0.0), 1.0);
    t = fmax(dot(p0 + u * s, d), 0.0);
    V3 q = p0 + u * s - d * t;
    return dot(q, q) <= r * r;
}

double tubeShader(const double* t, const M3& inv, V3 c, V3 a, double h, double r, const float* horizon)
{
    V3     w = cross(a, c * -1.0);
    double wl = len(w);
    w = wl > 1e-12 ? w * (1.0 / wl) : norm(cross(a, fabs(a.y) < 0.99 ? V3{ 0, 1, 0 } : V3{ 1, 0, 0 }));
    V3 e1 = a * (h + r), e2 = w * r;
    V3 p[4] = { c - e1 - e2, c + e1 - e2, c + e1 + e2, c - e1 + e2 };
    for (V3& q : p)
        q = inv.mul(q);
    V3 F = formFactor(p, 4);
    if (dot(F, p[0] + p[1] + p[2] + p[3]) < 0)
        F = F * -1.0;
    double fl = len(F);
    return fl > 0 ? horizonLookup(horizon, fl, F.z / fl) * t[4] : 0.0;
}

void checkTubes(const Cell* cells, const float* horizon)
{
    struct Tube
    {
        V3          c, a;
        const char* name;
    } tubes[] = { { { 0, 0, 1.2 }, { 1, 0, 0 }, "overhead, across the view" },
                  { { 0.8, 0, 1.0 }, { 0, 0.6, 0.8 }, "in the reflection, tilted" },
                  { { 1.2, 0.3, 0.6 }, { 0, 1, 0 }, "to the side" },
                  { { 1.0, 0, 0.15 }, { 0, 0.8, 0.6 }, "low, crossing the horizon" } };
    double lengths[] = { 0.5, 1.5 }, radii[] = { 0.02, 0.1 }, roughs[] = { 0.15, 0.4, 0.7, 1.0 }, cosVs[] = { 0.95, 0.6, 0.25 };
    double worstS = 0, sumS = 0, worstD = 0, sumD = 0, worstRel = 0, sumRel = 0;
    int    count = 0, countRel = 0;
    const M3 id = { { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } } };
    for (const Tube& tb : tubes)
        for (double L : lengths)
            for (double R : radii)
            {
                V3     a = norm(tb.a);
                double h = L * 0.5;
                // Diffuse: cosine-distributed samples that hit it, against the identity transform.
                const uint32_t n = 1 << 20;
                double         refD = 0;
                for (uint32_t i = 0; i < n; ++i)
                {
                    double u1 = (i + 0.5) / n, u2 = radicalInverse(i), rr = sqrt(u1), ph = 2 * kPi * u2;
                    V3     d = { rr * cos(ph), rr * sin(ph), sqrt(fmax(1 - u1, 0.0)) };
                    refD += hitsCapsule(d, tb.c, a, h, R);
                }
                refD /= n;
                double one[6] = { 1, 0, 0, 1, 1, 0 };
                double estD = tubeShader(one, id, tb.c, a, h, R, horizon);
                worstD = fmax(worstD, fabs(estD - refD));
                sumD += fabs(estD - refD);
                for (double rough : roughs)
                    for (double cv : cosVs)
                    {
                        Brdf   b(fmax(rough * rough, kMinAlpha), cv);
                        double ref = 0;
                        for (uint32_t i = 0; i < n; ++i)
                        {
                            V3     l = b.sample((i + 0.5) / n, radicalInverse(i));
                            double pdf, f = b.eval(l, &pdf);
                            if (pdf > 0 && f > 0 && hitsCapsule(l, tb.c, a, h, R))
                                ref += f / pdf;
                        }
                        ref /= n;
                        double x = rough * (kN - 1), y = sqrt(1.0 - cv) * (kN - 1);
                        int    x0 = (int)x, y0 = (int)y, x1 = x0 + 1 < kN ? x0 + 1 : x0, y1 = y0 + 1 < kN ? y0 + 1 : y0;
                        double fx = x - x0, fy = y - y0, t[6];
                        for (int k = 0; k < 6; ++k)
                        {
                            auto g = [&](int i, int j) { const Cell& c = cells[j * kN + i]; return k < 4 ? (double)c.inv[k] : k == 4 ? c.norm : c.fres; };
                            t[k] = (g(x0, y0) * (1 - fx) + g(x1, y0) * fx) * (1 - fy) + (g(x0, y1) * (1 - fx) + g(x1, y1) * fx) * fy;
                        }
                        M3     inv = { { { t[0], 0, t[1] }, { 0, 1, 0 }, { t[2], 0, t[3] } } };
                        double est = tubeShader(t, inv, tb.c, a, h, R, horizon);
                        double share = fabs(est - ref) / t[4];
                        worstS = fmax(worstS, share);
                        sumS += share;
                        count++;
                        // Of the tube's own light, where it lights the point at least a tenth as much as the brightest
                        // case of its size does (the rest is its dim tail).
                        if (ref > 0.002)
                        {
                            double rel = fabs(est - ref) / ref;
                            worstRel = fmax(worstRel, rel);
                            sumRel += rel;
                            countRel++;
                            if (rel > 0.25)
                                printf("  %-28s L %.1f r %.2f rough %.2f N.V %.2f: reference %.5f, shader %.5f (%.0f%%)\n", tb.name, L, R,
                                       rough, cv, ref, est, rel * 100);
                        }
                    }
            }
    printf("tubes, specular, as a share of all reflected: mean %.2f%% worst %.2f%% (%d cases)\n", sumS / count * 100, worstS * 100, count);
    printf("tubes, specular, of the tube's own light (%d cases above 0.002): mean %.1f%% worst %.1f%%\n", countRel, sumRel / countRel * 100,
           worstRel * 100);
    printf("tubes, diffuse form factor: mean error %.5f, worst %.5f (absolute, %d cases)\n", sumD / 16, worstD, 16);
}

void check(const Cell* cells, const float* horizon)
{
    // A few lights (rectangles) over a few views and roughnesses; Monte Carlo with BRDF sampling.
    struct Rect
    {
        V3     c, u, v; // centre, half-extents
        const char* name;
    } rects[] = { { { 0, 0, 2 }, { 1, 0, 0 }, { 0, 0.5, 0 }, "overhead panel" },
                  { { 2, 0, 1 }, { 0, 0.6, 0 }, { -0.3, 0, 0.6 }, "panel to the side" },
                  { { -1.5, 0.5, 0.6 }, { 0.2, 0.9, 0 }, { 0, 0, 0.4 }, "panel behind the view" },
                  { { 1, 0, 0.2 }, { 0, 1, 0 }, { 0, 0, 0.5 }, "low panel crossing the horizon" } };
    double roughs[] = { 0.15, 0.4, 0.7, 1.0 };
    double cosVs[] = { 0.95, 0.6, 0.25 };
    // Errors as a share of all the BRDF reflects (its norm): what can show on screen, as a tail the LTC cuts off (its
    // support is a tilted hemisphere) is a large share of a tiny value.
    double worst = 0, sumRel = 0, worstS = 0, sumS = 0, worstOwn = 0, sumOwn = 0;
    int    count = 0, countOwn = 0;
    printf("%-30s %6s %6s %10s %10s %14s %10s %9s\n", "light", "rough", "N.V", "reference", "LTC", "error (of all)", "LTC+sph", "error");
    for (const Rect& r : rects)
        for (double rough : roughs)
            for (double cv : cosVs)
            {
                Brdf b(fmax(rough * rough, kMinAlpha), cv);
                // Reference: sample the BRDF, count hits on the rectangle.
                const uint32_t n = 1 << 20;
                double         ref = 0;
                V3             nrm = norm(cross(r.u, r.v));
                for (uint32_t i = 0; i < n; ++i)
                {
                    V3     l = b.sample((i + 0.5) / n, radicalInverse(i));
                    double pdf, f = b.eval(l, &pdf);
                    if (pdf <= 0 || f <= 0)
                        continue;
                    double dn = dot(l, nrm);
                    if (fabs(dn) < 1e-12)
                        continue;
                    double t = dot(r.c, nrm) / dn;
                    if (t <= 0)
                        continue;
                    V3 h = l * t - r.c;
                    if (fabs(dot(h, r.u)) <= dot(r.u, r.u) && fabs(dot(h, r.v)) <= dot(r.v, r.v))
                        ref += f / pdf;
                }
                ref /= n;
                // LTC: bilinear lookup as the shader does.
                double x = rough * (kN - 1), y = sqrt(1.0 - cv) * (kN - 1);
                int    x0 = (int)x, y0 = (int)y, x1 = x0 + 1 < kN ? x0 + 1 : x0, y1 = y0 + 1 < kN ? y0 + 1 : y0;
                double fx = x - x0, fy = y - y0;
                double t[6];
                for (int k = 0; k < 6; ++k)
                {
                    auto g = [&](int i, int j) { const Cell& c = cells[j * kN + i]; return k < 4 ? (double)c.inv[k] : k == 4 ? c.norm : c.fres; };
                    t[k] = (g(x0, y0) * (1 - fx) + g(x1, y0) * fx) * (1 - fy) + (g(x0, y1) * (1 - fx) + g(x1, y1) * fx) * fy;
                }
                M3 inv = { { { t[0], 0, t[1] }, { 0, 1, 0 }, { t[2], 0, t[3] } } };
                V3 corners[4] = { r.c - r.u - r.v, r.c + r.u - r.v, r.c + r.u + r.v, r.c - r.u + r.v };
                for (V3& p : corners)
                    p = inv.mul(p);
                double ltc = polygonCosine(corners, 4) * t[4];
                // The shader's way: the unclipped polygon's form-factor vector, clipped by the sphere approximation.
                V3     F = formFactor(corners, 4);
                double fl = len(F), fz = fl > 0 ? F.z / fl : 0;
                if (dot(cross(corners[1] - corners[0], corners[2] - corners[0]), corners[0]) < 0)
                    fz = -fz; // the vector's sign follows the winding: as seen from the surface, two-sided
                double hx = fmin(fl, 1.0) * (kN - 1), hy = (fz * 0.5 + 0.5) * (kN - 1);
                int    h0 = (int)hx, g0 = (int)hy, h1 = h0 + 1 < kN ? h0 + 1 : h0, g1 = g0 + 1 < kN ? g0 + 1 : g0;
                double hfx = hx - h0, hfy = hy - g0;
                auto   H = [&](int i, int j) { return (double)horizon[j * kN + i]; };
                double sph = ((H(h0, g0) * (1 - hfx) + H(h1, g0) * hfx) * (1 - hfy) + (H(h0, g1) * (1 - hfx) + H(h1, g1) * hfx) * hfy) * t[4];
                double relS = fabs(sph - ref) / t[4];
                worstS = fmax(worstS, relS);
                sumS += relS;
                double rel = fabs(ltc - ref) / t[4];
                if (ref > 0.002)
                {
                    double own = fabs(sph - ref) / ref;
                    worstOwn = fmax(worstOwn, own);
                    sumOwn += own;
                    countOwn++;
                }
                worst = fmax(worst, rel);
                sumRel += rel;
                count++;
                printf("%-30s %6.2f %6.2f %10.5f %10.5f %13.2f%% %10.5f %8.2f%%\n", r.name, rough, cv, ref, ltc, rel * 100, sph, relS * 100);
            }
    printf("error as a share of all reflected: exact clip mean %.2f%% worst %.2f%%; sphere horizon mean %.2f%% worst %.2f%% (%d cases)\n",
           sumRel / count * 100, worst * 100, sumS / count * 100, worstS * 100, count);
    printf("rects, of the light's own (%d cases above 0.002, sphere horizon): mean %.1f%% worst %.1f%%\n", countOwn, sumOwn / countOwn * 100,
           worstOwn * 100);
}
} // namespace

int main(int argc, char** argv)
{
    bool        doCheck = argc > 2 && !strcmp(argv[1], "--check");
    std::string path = argc > 1 ? argv[argc - 1] : "ltc.bin";
    std::vector<Cell> cells(kN * kN);
    if (doCheck)
    {
        FILE* f = fopen(path.c_str(), "rb");
        char  magic[4];
        int   n = 0;
        if (!f || fread(magic, 1, 4, f) != 4 || fread(&n, 4, 1, f) != 1 || n != kN)
            return printf("can't read %s\n", path.c_str()), 1;
        std::vector<float> a(kN * kN * 4), b(kN * kN * 4), horizon(kN * kN);
        if (fread(a.data(), 4, a.size(), f) != a.size() || fread(b.data(), 4, b.size(), f) != b.size())
            return 1;
        fclose(f);
        for (int i = 0; i < kN * kN; ++i)
        {
            memcpy(cells[i].inv, &a[i * 4], 16);
            cells[i].norm = b[i * 4];
            cells[i].fres = b[i * 4 + 1];
            horizon[i] = b[i * 4 + 2];
        }
        check(cells.data(), horizon.data());
        checkDiscs(horizon.data());
        checkTubes(cells.data(), horizon.data());
        return 0;
    }
    std::vector<std::thread> pool;
    unsigned                 workers = std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 4;
    for (unsigned w = 0; w < workers; ++w)
        pool.emplace_back([&, w] {
            for (int j = (int)w; j < kN; j += (int)workers)
                fitColumn(j, cells.data());
        });
    for (auto& t : pool)
        t.join();
    double worst = 0;
    for (const Cell& c : cells)
        worst = fmax(worst, c.err);
    std::vector<float> a(kN * kN * 4), b(kN * kN * 4);
    for (int j = 0; j < kN; ++j)
        for (int i = 0; i < kN; ++i)
        {
            const Cell& c = cells[j * kN + i];
            memcpy(&a[(j * kN + i) * 4], c.inv, 16);
            b[(j * kN + i) * 4 + 0] = c.norm;
            b[(j * kN + i) * 4 + 1] = c.fres;
            // The horizon table, on the same grid: x the form factor, y the elevation z / 2 + 1 / 2.
            b[(j * kN + i) * 4 + 2] = (float)sphereHorizon((double)i / (kN - 1), (double)j / (kN - 1) * 2.0 - 1.0);
            b[(j * kN + i) * 4 + 3] = 0.0f;
        }
    FILE* f = fopen(path.c_str(), "wb");
    if (!f)
        return printf("can't write %s\n", path.c_str()), 1;
    int n = kN;
    fwrite("LTC1", 1, 4, f);
    fwrite(&n, 4, 1, f);
    fwrite(a.data(), 4, a.size(), f);
    fwrite(b.data(), 4, b.size(), f);
    fclose(f);
    printf("wrote %s (%d x %d), worst fit error %.3g\n", path.c_str(), kN, kN, worst);
    return 0;
}
