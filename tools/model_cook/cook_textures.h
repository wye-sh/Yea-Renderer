// The model cooker's textures (included by model_cook.cpp, after warn() and gDir): glTF images to rmodel.h's BC7 and
// BC5 mip chains.
// - Mips: each level a 2x2 box of the one above in linear light (sRGB decoded first for colour), clamped at odd edges;
//   normal maps averaged as vectors and renormalised; ORM as data. The full chain to 1x1 is always kept: the source is
//   never shrunk (--max-size names a top mip to start from, and the report says so).
// - Alpha-masked base colours keep their coverage at the cutoff in every mip (Castano 2010's alpha scaling), so leaves
//   and fences don't thin out with distance.
// - BC7 by bc7e (Binomial's port, vendor/bc7e, Apache-2.0), perceptual for sRGB; BC5 by our own BC4 encoder below
//   (endpoints from the block's range, refined by a small search). Blocks are encoded on every core.
#pragma once

#include "vendor/bc7e/basisu_bc7e_scalar.h"

#include <atomic>
#include <thread>

namespace cook
{
enum Kind : uint32_t
{
    KindBase,     // BC7 sRGB with alpha
    KindOrm,      // BC7 linear: occlusion, roughness, metallic
    KindNormal,   // BC5
    KindEmissive, // BC7 sRGB
};

struct Image
{
    int                  w = 0, h = 0;
    std::vector<uint8_t> rgba;
};

struct Cooked
{
    const cgltf_image*   a;
    const cgltf_image*   b; // ORM: the occlusion image merged into a's (nullptr: none or the same)
    uint32_t             kind;
    float                cutoff; // KindBase of a masked material: its coverage kept per mip (0: opaque)
    RmodelTexture        t;
    std::vector<uint8_t> data;
};
std::vector<Cooked> gTextures;
uint32_t            gMaxSize = 0; // --max-size: the largest top mip written (0: the source's)
uint32_t            gSkipped = 0; // top mips left out by --max-size, over every texture

bool load(const cgltf_image* img, Image& out)
{
    int      n = 0;
    uint8_t* px = nullptr;
    if (img->buffer_view)
        px = stbi_load_from_memory(cgltf_buffer_view_data(img->buffer_view), (int)img->buffer_view->size, &out.w, &out.h, &n, 4);
    else if (img->uri && strncmp(img->uri, "data:", 5))
    {
        std::string path = gDir + img->uri;
        cgltf_decode_uri(&path[gDir.size()]);
        path.resize(strlen(path.c_str()));
        px = stbi_load(path.c_str(), &out.w, &out.h, &n, 4);
    }
    if (!px)
    {
        warn("an image (%s) couldn't be read: its material draws without it", img->uri ? img->uri : img->name ? img->name : "embedded");
        return false;
    }
    out.rgba.assign(px, px + (size_t)out.w * out.h * 4);
    stbi_image_free(px);
    return true;
}

float toLinear(uint8_t c) { return srgbToLinear(c / 255.0f); }
uint8_t toSrgb(float c)
{
    c = c < 0 ? 0 : (c > 1 ? 1 : c);
    c = c <= 0.0031308f ? c * 12.92f : 1.055f * powf(c, 1 / 2.4f) - 0.055f;
    return (uint8_t)(c * 255.0f + 0.5f);
}
uint8_t toUnorm(float c) { return (uint8_t)((c < 0 ? 0 : (c > 1 ? 1 : c)) * 255.0f + 0.5f); }

// One mip level as linear floats (rgba).
struct Level
{
    uint32_t           w, h;
    std::vector<float> px;
};

Level decode(const Image& im, uint32_t kind)
{
    Level l = { (uint32_t)im.w, (uint32_t)im.h, std::vector<float>((size_t)im.w * im.h * 4) };
    float lut[256];
    for (int i = 0; i < 256; ++i)
        lut[i] = (kind == KindBase || kind == KindEmissive) ? toLinear((uint8_t)i) : i / 255.0f;
    for (size_t i = 0; i < (size_t)im.w * im.h; ++i)
    {
        for (int c = 0; c < 3; ++c)
            l.px[i * 4 + c] = lut[im.rgba[i * 4 + c]];
        l.px[i * 4 + 3] = im.rgba[i * 4 + 3] / 255.0f;
        if (kind == KindNormal)
        {
            float* v = &l.px[i * 4];
            v[0] = v[0] * 2 - 1, v[1] = v[1] * 2 - 1, v[2] = v[2] * 2 - 1;
            float n = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
            if (n > 1e-6f)
                v[0] /= n, v[1] /= n, v[2] /= n;
            else
                v[0] = 0, v[1] = 0, v[2] = 1;
        }
    }
    return l;
}

Level half(const Level& s, uint32_t kind)
{
    Level d = { s.w > 1 ? s.w / 2 : 1, s.h > 1 ? s.h / 2 : 1, {} };
    d.px.resize((size_t)d.w * d.h * 4);
    for (uint32_t y = 0; y < d.h; ++y)
        for (uint32_t x = 0; x < d.w; ++x)
        {
            // The source texels this one covers: two each way, three at an odd edge's last texel.
            uint32_t x0 = x * 2, x1 = s.w > 1 ? (x == d.w - 1 ? s.w - 1 : x * 2 + 1) : 0;
            uint32_t y0 = y * 2, y1 = s.h > 1 ? (y == d.h - 1 ? s.h - 1 : y * 2 + 1) : 0;
            float    sum[4] = {};
            uint32_t n = 0;
            for (uint32_t sy = y0 < s.h ? y0 : s.h - 1; sy <= y1; ++sy)
                for (uint32_t sx = x0 < s.w ? x0 : s.w - 1; sx <= x1; ++sx, ++n)
                    for (int c = 0; c < 4; ++c)
                        sum[c] += s.px[((size_t)sy * s.w + sx) * 4 + c];
            float* o = &d.px[((size_t)y * d.w + x) * 4];
            for (int c = 0; c < 4; ++c)
                o[c] = sum[c] / (float)n;
            if (kind == KindNormal)
            {
                float l = sqrtf(o[0] * o[0] + o[1] * o[1] + o[2] * o[2]);
                if (l > 1e-6f)
                    o[0] /= l, o[1] /= l, o[2] /= l;
                else
                    o[0] = 0, o[1] = 0, o[2] = 1;
            }
        }
    return d;
}

float coverage(const Level& l, float cutoff, float scale)
{
    size_t in = 0, n = (size_t)l.w * l.h;
    for (size_t i = 0; i < n; ++i)
        in += l.px[i * 4 + 3] * scale >= cutoff;
    return n ? (float)in / (float)n : 0;
}

// The alpha scale that gives this mip the top mip's coverage at the cutoff (a bisection; coverage only grows with it).
void keepCoverage(Level& l, float cutoff, float want)
{
    float lo = 0, hi = 4;
    for (int i = 0; i < 16; ++i)
    {
        float mid = 0.5f * (lo + hi);
        (coverage(l, cutoff, mid) < want ? lo : hi) = mid;
    }
    for (size_t i = 0; i < (size_t)l.w * l.h; ++i)
        l.px[i * 4 + 3] = fminf(l.px[i * 4 + 3] * hi, 1.0f);
}

// A BC4 block (8 bytes) for 16 values: endpoints in the 8-value mode, from the range and a small search around it.
void bc4(const uint8_t v[16], uint8_t* out)
{
    int lo = 255, hi = 0;
    for (int i = 0; i < 16; ++i)
        lo = v[i] < lo ? v[i] : lo, hi = v[i] > hi ? v[i] : hi;
    uint64_t bestErr = ~0ull, bestBits = 0;
    int      bestA = hi, bestB = lo;
    if (hi == lo)
    {
        out[0] = out[1] = (uint8_t)hi;
        memset(out + 2, 0, 6);
        return;
    }
    for (int da = -2; da <= 2; ++da)
        for (int db = -2; db <= 2; ++db)
        {
            int a = hi + da, b = lo + db;
            if (a > 255 || b < 0 || a <= b)
                continue;
            int pal[8] = { a, b };
            for (int i = 1; i < 7; ++i)
                pal[i + 1] = ((7 - i) * a + i * b + 3) / 7;
            uint64_t err = 0, bits = 0;
            for (int p = 0; p < 16; ++p)
            {
                int bi = 0, be = 1 << 30;
                for (int k = 0; k < 8; ++k)
                {
                    int e = (pal[k] - v[p]) * (pal[k] - v[p]);
                    if (e < be)
                        be = e, bi = k;
                }
                err += (uint64_t)be;
                bits |= (uint64_t)bi << (3 * p);
            }
            if (err < bestErr)
                bestErr = err, bestBits = bits, bestA = a, bestB = b;
        }
    out[0] = (uint8_t)bestA;
    out[1] = (uint8_t)bestB;
    for (int i = 0; i < 6; ++i)
        out[2 + i] = (uint8_t)(bestBits >> (8 * i));
}

// One level's blocks into out (16 bytes each).
void encode(const Level& l, uint32_t kind, uint8_t* out, const bc7e_scalar::bc7e_compress_block_params* params)
{
    const uint32_t bw = (l.w + 3) / 4, bh = (l.h + 3) / 4, total = bw * bh;
    std::atomic<uint32_t> next(0);
    auto work = [&]() {
        for (;;)
        {
            const uint32_t first = next.fetch_add(64);
            if (first >= total)
                return;
            const uint32_t count = total - first < 64 ? total - first : 64;
            uint32_t       px[64 * 16];
            for (uint32_t k = 0; k < count; ++k)
            {
                const uint32_t bx = (first + k) % bw, by = (first + k) / bw;
                for (uint32_t i = 0; i < 16; ++i)
                {
                    uint32_t     x = bx * 4 + (i & 3), y = by * 4 + (i >> 2);
                    x = x < l.w ? x : l.w - 1, y = y < l.h ? y : l.h - 1; // edges repeat past a mip's end
                    const float* s = &l.px[((size_t)y * l.w + x) * 4];
                    uint8_t      c[4];
                    if (kind == KindBase || kind == KindEmissive)
                        c[0] = toSrgb(s[0]), c[1] = toSrgb(s[1]), c[2] = toSrgb(s[2]);
                    else if (kind == KindNormal)
                        c[0] = toUnorm(s[0] * 0.5f + 0.5f), c[1] = toUnorm(s[1] * 0.5f + 0.5f), c[2] = 0;
                    else
                        c[0] = toUnorm(s[0]), c[1] = toUnorm(s[1]), c[2] = toUnorm(s[2]);
                    c[3] = kind == KindBase ? toUnorm(s[3]) : 255;
                    memcpy(&px[k * 16 + i], c, 4);
                }
                if (kind == KindNormal)
                {
                    uint8_t r[16], g[16];
                    for (int i = 0; i < 16; ++i)
                        r[i] = (uint8_t)(px[k * 16 + i] & 255), g[i] = (uint8_t)((px[k * 16 + i] >> 8) & 255);
                    bc4(r, out + (size_t)(first + k) * 16);
                    bc4(g, out + (size_t)(first + k) * 16 + 8);
                }
            }
            if (kind != KindNormal)
                bc7e_scalar::bc7e_compress_blocks(count, (uint64_t*)(out + (size_t)first * 16), px, params);
        }
    };
    std::vector<std::thread> threads;
    const unsigned           n = std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 4;
    for (unsigned i = 0; i < n; ++i)
        threads.emplace_back(work);
    for (std::thread& t : threads)
        t.join();
}

// A texture's index in gTextures (cooked once per image, kind and cutoff), or -1.
int32_t cook(const cgltf_texture* texA, const cgltf_texture* texB, uint32_t kind, float cutoff)
{
    const cgltf_image* a = texA ? texA->image : nullptr;
    const cgltf_image* b = texB ? texB->image : nullptr;
    if (b == a)
        b = nullptr;
    if (!a && kind == KindOrm && b) // occlusion alone: roughness and metallic white, their factors as they are
        a = nullptr;
    else if (!a)
        return -1;
    for (size_t i = 0; i < gTextures.size(); ++i)
        if (gTextures[i].a == a && gTextures[i].b == b && gTextures[i].kind == kind && gTextures[i].cutoff == cutoff)
            return (int32_t)i;
    Image im, occ;
    if (a && !load(a, im))
        return -1;
    if (b && !load(b, occ))
        b = nullptr;
    if (!a) // only an occlusion map: it is the ORM, green and blue white
    {
        im = occ;
        for (size_t i = 0; i < (size_t)im.w * im.h; ++i)
            im.rgba[i * 4 + 1] = im.rgba[i * 4 + 2] = 255;
    }
    else if (b) // glTF's occlusion in its own image: into red, sampled at the ORM's size (bilinear)
        for (int y = 0; y < im.h; ++y)
            for (int x = 0; x < im.w; ++x)
            {
                float fx = ((x + 0.5f) * occ.w / im.w) - 0.5f, fy = ((y + 0.5f) * occ.h / im.h) - 0.5f;
                int   x0 = (int)floorf(fx), y0 = (int)floorf(fy);
                float tx = fx - x0, ty = fy - y0, v = 0;
                for (int k = 0; k < 4; ++k)
                {
                    int sx = x0 + (k & 1), sy = y0 + (k >> 1);
                    sx = sx < 0 ? 0 : (sx >= occ.w ? occ.w - 1 : sx), sy = sy < 0 ? 0 : (sy >= occ.h ? occ.h - 1 : sy);
                    v += occ.rgba[((size_t)sy * occ.w + sx) * 4] * ((k & 1) ? tx : 1 - tx) * ((k >> 1) ? ty : 1 - ty);
                }
                im.rgba[((size_t)y * im.w + x) * 4] = (uint8_t)(v + 0.5f);
            }

    Level              top = decode(im, kind);
    const float        want = cutoff > 0 ? coverage(top, cutoff, 1) : 0;
    std::vector<Level> levels;
    levels.push_back(std::move(top));
    while (levels.back().w > 1 || levels.back().h > 1)
    {
        levels.push_back(half(levels.back(), kind));
        if (cutoff > 0)
            keepCoverage(levels.back(), cutoff, want);
    }
    size_t first = 0;
    while (gMaxSize && first + 1 < levels.size() && (levels[first].w > gMaxSize || levels[first].h > gMaxSize))
        ++first;
    gSkipped += (uint32_t)first;

    static bool inited = false;
    if (!inited)
        bc7e_scalar::bc7e_compress_block_init(), inited = true;
    bc7e_scalar::bc7e_compress_block_params params;
    bc7e_scalar::bc7e_compress_block_params_init_basic(&params, kind == KindBase || kind == KindEmissive);

    Cooked c = { a, b, kind, cutoff, {}, {} };
    c.t.format = kind == KindNormal ? RmodelBC5 : (kind == KindOrm ? RmodelBC7Linear : RmodelBC7Srgb);
    c.t.width = levels[first].w, c.t.height = levels[first].h;
    c.t.mips = (uint32_t)(levels.size() - first);
    size_t bytes = 0;
    for (size_t i = first; i < levels.size(); ++i)
        bytes += (size_t)((levels[i].w + 3) / 4) * ((levels[i].h + 3) / 4) * 16;
    c.data.resize(bytes);
    size_t at = 0;
    for (size_t i = first; i < levels.size(); ++i)
    {
        encode(levels[i], kind, c.data.data() + at, &params);
        at += (size_t)((levels[i].w + 3) / 4) * ((levels[i].h + 3) / 4) * 16;
    }
    c.t.bytes = bytes;
    gTextures.push_back(std::move(c));
    return (int32_t)(gTextures.size() - 1);
}
} // namespace cook
