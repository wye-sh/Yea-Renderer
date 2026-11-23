// Bakes the renderer's split-sum table (Brdf.h.fsl dfgLookup, frame.h dfg): for the core's GGX with height-correlated
// Smith visibility (brdfLight in Shaders/Lighting.h.fsl), the directional albedo split by Schlick's Fresnel,
// A = integral of (1 - (1 - V.H)^5) f cos and B = integral of (1 - V.H)^5 f cos over the hemisphere, so F0 A + B is the
// specular's albedo. Sampled from the distribution of visible normals (Heitz, "Sampling the GGX Distribution of Visible
// Normals", JCGT 2018), whose estimator is G2 / G1(V): it converges far faster than sampling the NDF at grazing angles.
// Written for this renderer; it runs offline, not in the game.
//
//   c++ -O2 -std=c++17 -pthread dfg_bake.cpp -o dfg_bake && ./dfg_bake ../modules/renderer/Assets/dfg.bin
//   ./dfg_bake --check ../modules/renderer/Assets/dfg.bin    (the file against 4x the samples, grazing cells first)
//
// The table is kN x kN: x = sqrt(N.V) (grazing angles, where it changes fastest, get the finest steps), y = perceptual
// roughness, cell centres at i / (kN - 1); N.V at least 1e-4, roughness at least 0.02 (as the core's own fallback).
// dfg.bin: "DFG1", kN (int32), then kN * kN float2 (A, B), row-major by y, all little-endian.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <thread>
#include <vector>

namespace
{
const int    kN = 128;
const int    kSamples = 16384;
const double kPi = 3.14159265358979323846;

double radicalInverse(uint32_t bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return (double)bits * 2.3283064365386963e-10;
}

// Smith's Lambda for GGX at cosine mu.
double lambda(double mu, double a2) { return 0.5 * (-1.0 + sqrt(1.0 + a2 * (1.0 - mu * mu) / (mu * mu))); }

void cell(int i, int j, int n, double* outA, double* outB)
{
    double x = (double)i / (kN - 1), NoV = fmax(x * x, 1e-4);
    double r = fmax((double)j / (kN - 1), 0.02), a = r * r, a2 = a * a;
    double V[3] = { sqrt(1.0 - NoV * NoV), 0.0, NoV };
    // The view direction stretched to the unit-roughness configuration, and a frame around it.
    double Vh[3] = { a * V[0], a * V[1], V[2] };
    double l = sqrt(Vh[0] * Vh[0] + Vh[1] * Vh[1] + Vh[2] * Vh[2]);
    for (double& c : Vh)
        c /= l;
    double lensq = Vh[0] * Vh[0] + Vh[1] * Vh[1];
    double T1[3] = { 1, 0, 0 };
    if (lensq > 0)
    {
        double s = 1.0 / sqrt(lensq);
        T1[0] = -Vh[1] * s, T1[1] = Vh[0] * s, T1[2] = 0;
    }
    double T2[3] = { Vh[1] * T1[2] - Vh[2] * T1[1], Vh[2] * T1[0] - Vh[0] * T1[2], Vh[0] * T1[1] - Vh[1] * T1[0] };
    double lv = lambda(NoV, a2), A = 0, B = 0;
    for (int k = 0; k < n; ++k)
    {
        double u1 = ((double)k + 0.5) / n, u2 = radicalInverse((uint32_t)k);
        double rr = sqrt(u1), phi = 2.0 * kPi * u2;
        double t1 = rr * cos(phi), t2 = rr * sin(phi);
        double s = 0.5 * (1.0 + Vh[2]);
        t2 = (1.0 - s) * sqrt(fmax(1.0 - t1 * t1, 0.0)) + s * t2;
        double tz = sqrt(fmax(1.0 - t1 * t1 - t2 * t2, 0.0));
        double Nh[3] = { t1 * T1[0] + t2 * T2[0] + tz * Vh[0], t1 * T1[1] + t2 * T2[1] + tz * Vh[1], t1 * T1[2] + t2 * T2[2] + tz * Vh[2] };
        double H[3] = { a * Nh[0], a * Nh[1], fmax(Nh[2], 0.0) };
        double hl = sqrt(H[0] * H[0] + H[1] * H[1] + H[2] * H[2]);
        if (hl <= 0)
            continue;
        for (double& c : H)
            c /= hl;
        double VoH = V[0] * H[0] + V[1] * H[1] + V[2] * H[2];
        double NoL = 2.0 * VoH * H[2] - V[2];
        if (NoL <= 0 || VoH <= 0)
            continue;
        // f cos / pdf = G2 / G1(V), with G2 = 1 / (1 + Lambda(V) + Lambda(L)) (height-correlated) and G1 = 1 / (1 + Lambda(V)).
        double w = (1.0 + lv) / (1.0 + lv + lambda(NoL, a2));
        double fc = pow(1.0 - VoH, 5.0);
        A += (1.0 - fc) * w;
        B += fc * w;
    }
    *outA = A / n;
    *outB = B / n;
}

void bake(std::vector<float>& t, int n)
{
    t.assign((size_t)kN * kN * 2, 0.0f);
    unsigned threads = std::thread::hardware_concurrency();
    threads = threads ? threads : 4;
    std::vector<std::thread> pool;
    for (unsigned w = 0; w < threads; ++w)
        pool.emplace_back([&, w] {
            for (int j = (int)w; j < kN; j += (int)threads)
                for (int i = 0; i < kN; ++i)
                {
                    double A, B;
                    cell(i, j, n, &A, &B);
                    t[((size_t)j * kN + i) * 2 + 0] = (float)A;
                    t[((size_t)j * kN + i) * 2 + 1] = (float)B;
                }
        });
    for (std::thread& th : pool)
        th.join();
}
} // namespace

int main(int argc, char** argv)
{
    bool        check = argc == 3 && !strcmp(argv[1], "--check");
    const char* path = check ? argv[2] : argc == 2 ? argv[1] : nullptr;
    if (!path)
    {
        fprintf(stderr, "usage: dfg_bake <out.bin> | dfg_bake --check <dfg.bin>\n");
        return 1;
    }
    if (check)
    {
        FILE*              f = fopen(path, "rb");
        char               magic[4] = {};
        int                n = 0;
        std::vector<float> t((size_t)kN * kN * 2);
        bool ok = f && fread(magic, 1, 4, f) == 4 && !memcmp(magic, "DFG1", 4) && fread(&n, 4, 1, f) == 1 && n == kN &&
                  fread(t.data(), 8, (size_t)kN * kN, f) == (size_t)kN * kN;
        if (f)
            fclose(f);
        if (!ok)
        {
            fprintf(stderr, "%s is not a %d x %d DFG1 table\n", path, kN, kN);
            return 1;
        }
        // Every 8th row and column, plus the grazing columns of the smooth rows, against 4x the samples.
        double worst = 0, worstGrazing = 0;
        for (int j = 0; j < kN; ++j)
            for (int i = 0; i < kN; ++i)
            {
                bool grazing = i < 12 && j < 16;
                if (!grazing && (i % 8 || j % 8))
                    continue;
                double A, B;
                cell(i, j, kSamples * 4, &A, &B);
                double e = fmax(fabs(A - t[((size_t)j * kN + i) * 2]), fabs(B - t[((size_t)j * kN + i) * 2 + 1]));
                (grazing ? worstGrazing : worst) = fmax(grazing ? worstGrazing : worst, e);
            }
        printf("max error against %d samples: %.6f at grazing smooth cells, %.6f elsewhere\n", kSamples * 4, worstGrazing, worst);
        return 0;
    }
    std::vector<float> t;
    bake(t, kSamples);
    FILE* f = fopen(path, "wb");
    if (!f)
    {
        fprintf(stderr, "can't write %s\n", path);
        return 1;
    }
    int n = kN;
    fwrite("DFG1", 1, 4, f);
    fwrite(&n, 4, 1, f);
    fwrite(t.data(), 8, (size_t)kN * kN, f);
    fclose(f);
    printf("wrote %s: %d x %d, %d samples a cell\n", path, kN, kN, kSamples);
    return 0;
}
