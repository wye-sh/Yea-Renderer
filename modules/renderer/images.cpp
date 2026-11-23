// Images across rect lights (light_image): made through the exports, kept in the LightTexel table at 256 x 256, and
// uploaded with a prefiltered mip chain into one texture array the forward pass reads (Lighting.h.fsl's ltcImage).
#include "yea_renderer.h"
#include "scene.h"

#include <math.h>
#include <string.h>

#include <Common_3/Graphics/Interfaces/IGraphics.h>
#include <Common_3/Resources/ResourceLoader/Interfaces/IResourceLoader.h>

namespace renderer
{
namespace
{
const uint32_t kSide = 256, kTexels = kSide * kSide, kLayers = 16, kMips = 9, kMaxInput = 4096;

// The image being made: its size, the pixels given so far, and each texel's area-weighted sum (rgb, weight).
struct Making
{
    bool     active;
    uint32_t width, height, count;
} gMaking;
float*   gAccum; // kTexels x 4
float*   gMipA;  // kTexels x 3, linear: a mip level and the next
float*   gMipB;
uint32_t gUploaded = 0xFFFFFFFFu; // images in the texture (none uploaded yet: the first upload clears it too)

float toLinear(uint32_t c)
{
    float v = (float)c / 255.0f;
    return v <= 0.04045f ? v / 12.92f : powf((v + 0.055f) / 1.055f, 2.4f);
}

uint32_t toSrgb(float v)
{
    v = v < 0 ? 0 : (v > 1 ? 1 : v);
    float s = v <= 0.0031308f ? v * 12.92f : 1.055f * powf(v, 1.0f / 2.4f) - 0.055f;
    return (uint32_t)(s * 255.0f + 0.5f);
}

uint32_t imageCount() { return LightTexels.size() / kTexels; }
} // namespace

void lightImagesInit()
{
    gAccum = new float[kTexels * 4];
    gMipA = new float[kTexels * 3];
    gMipB = new float[kTexels * 3];
    gMaking.active = false;
    gUploaded = 0xFFFFFFFFu;
}

void lightImagesFree()
{
    delete[] gAccum;
    delete[] gMipA;
    delete[] gMipB;
    gAccum = gMipA = gMipB = nullptr;
}

bool light_image_begin(uint32_t width, uint32_t height)
{
    if (gMaking.active || !gAccum || width == 0 || height == 0 || width > kMaxInput || height > kMaxInput || imageCount() >= kLayers)
        return false;
    gMaking = { true, width, height, 0 };
    memset(gAccum, 0, sizeof(float) * kTexels * 4);
    return true;
}

// Each pixel adds to the texels it covers, by the area it covers of each: a box filter both ways (a small image
// keeps its pixels' hard edges, which the mips then soften).
void light_image_pixel(yea_float3 rgb)
{
    if (!gMaking.active)
        return;
    uint32_t i = gMaking.count++;
    if (i >= gMaking.width * gMaking.height)
        return;
    uint32_t px = i % gMaking.width, py = i / gMaking.width;
    float    sx = (float)kSide / (float)gMaking.width, sy = (float)kSide / (float)gMaking.height;
    float    x0 = px * sx, x1 = x0 + sx, y0 = py * sy, y1 = y0 + sy;
    for (uint32_t ty = (uint32_t)y0; ty < kSide && (float)ty < y1; ++ty)
    {
        float wy = fminf(y1, (float)(ty + 1)) - fmaxf(y0, (float)ty);
        for (uint32_t tx = (uint32_t)x0; tx < kSide && (float)tx < x1; ++tx)
        {
            float  w = wy * (fminf(x1, (float)(tx + 1)) - fmaxf(x0, (float)tx));
            float* a = gAccum + (ty * kSide + tx) * 4;
            a[0] += rgb.x * w;
            a[1] += rgb.y * w;
            a[2] += rgb.z * w;
            a[3] += w;
        }
    }
}

uint32_t light_image_end()
{
    if (!gMaking.active)
        return 0xFFFFFFFFu;
    gMaking.active = false;
    if (gMaking.count != gMaking.width * gMaking.height || LightTexels.size() + kTexels > LightTexels.capacity())
        return 0xFFFFFFFFu;
    uint32_t image = imageCount();
    for (uint32_t t = 0; t < kTexels; ++t)
    {
        const float* a = gAccum + t * 4;
        float        k = a[3] > 0 ? 1.0f / a[3] : 0.0f;
        LightTexels.add()->rgba = toSrgb(a[0] * k) | toSrgb(a[1] * k) << 8 | toSrgb(a[2] * k) << 16 | 255u << 24;
    }
    return image;
}

void light_image(yea_handle l, uint32_t image)
{
    uint32_t row = l & 0xFFFFFu;
    if (row >= Lights.size() || !Lights[row].alive || (Lights[row].gen & 0xFFFu) != (l >> 20))
        return;
    while (LightLooks.size() <= row)
        if (LightLook* k = LightLooks.add())
            k->image = 0;
        else
            return;
    LightLooks[row].image = image < imageCount() ? image + 1 : 0;
}

uint32_t lightImageOf(uint32_t lightRow)
{
    if (lightRow >= LightLooks.size() || Lights[lightRow].type != 3)
        return 0;
    uint32_t k = LightLooks[lightRow].image;
    return k && k <= imageCount() && gUploaded != 0xFFFFFFFFu && k <= gUploaded ? k : 0;
}

void lightImageForget(uint32_t lightRow)
{
    if (lightRow < LightLooks.size())
        LightLooks[lightRow].image = 0;
}

Texture* lightImagesMake()
{
    TextureDesc t = {};
    t.mWidth = t.mHeight = kSide;
    t.mDepth = 1;
    t.mArraySize = kLayers;
    t.mMipLevels = kMips;
    t.mSampleCount = SAMPLE_COUNT_1;
    t.mFormat = TinyImageFormat_R8G8B8A8_SRGB;
    t.mStartState = RESOURCE_STATE_SHADER_RESOURCE;
    t.mDescriptors = DESCRIPTOR_TYPE_TEXTURE;
    t.pName = "renderer_light_images";
    Texture*        tex = nullptr;
    TextureLoadDesc tl = {};
    tl.pDesc = &t;
    tl.ppTexture = &tex;
    addResource(&tl, nullptr);
    waitForAllResourceLoads();
    gUploaded = 0xFFFFFFFFu;
    return tex;
}

// When the images changed (one made, or all cleared; every one after a reload): the whole array again, each image
// with its mips. Each mip is the one above filtered by a [1 3 3 1] / 8 kernel each way (a Gaussian pyramid, as the
// prefiltered images of Heitz et al. 2016), so a lobe's footprint reads a smooth average at any level. A stall, as
// mesh uploads are: images are made up front.
void lightImagesUpload(Texture* tex)
{
    uint32_t n = imageCount();
    if (!tex || n == gUploaded || gMaking.active || !gMipA)
        return;
    TextureUpdateDesc u = {};
    u.pTexture = tex;
    u.mBaseMipLevel = 0;
    u.mMipLevels = kMips;
    u.mBaseArrayLayer = 0;
    u.mLayerCount = kLayers;
    u.mCurrentState = RESOURCE_STATE_SHADER_RESOURCE;
    beginUpdateResource(&u);
    for (uint32_t layer = 0; layer < kLayers; ++layer)
    {
        for (uint32_t t = 0; t < kTexels; ++t)
        {
            uint32_t c = layer < n ? LightTexels[layer * kTexels + t].rgba : 0;
            gMipA[t * 3 + 0] = toLinear(c & 255u);
            gMipA[t * 3 + 1] = toLinear((c >> 8) & 255u);
            gMipA[t * 3 + 2] = toLinear((c >> 16) & 255u);
        }
        uint32_t side = kSide;
        for (uint32_t mip = 0; mip < kMips; ++mip, side /= 2)
        {
            TextureSubresourceUpdate s = u.getSubresourceUpdateDesc(mip, layer);
            for (uint32_t y = 0; y < side && y < s.mRowCount; ++y)
            {
                uint32_t* row = (uint32_t*)(s.pMappedData + y * s.mDstRowStride);
                for (uint32_t x = 0; x < side; ++x)
                {
                    const float* c = gMipA + (y * side + x) * 3;
                    row[x] = toSrgb(c[0]) | toSrgb(c[1]) << 8 | toSrgb(c[2]) << 16 | 255u << 24;
                }
            }
            if (side == 1)
                break;
            uint32_t half = side / 2;
            const float k[4] = { 0.125f, 0.375f, 0.375f, 0.125f };
            for (uint32_t y = 0; y < half; ++y)
                for (uint32_t x = 0; x < half; ++x)
                {
                    float sum[3] = {};
                    for (int j = 0; j < 4; ++j)
                    {
                        int sy = (int)(2 * y) - 1 + j;
                        sy = sy < 0 ? 0 : (sy >= (int)side ? (int)side - 1 : sy);
                        for (int i = 0; i < 4; ++i)
                        {
                            int sx = (int)(2 * x) - 1 + i;
                            sx = sx < 0 ? 0 : (sx >= (int)side ? (int)side - 1 : sx);
                            const float* c = gMipA + (sy * side + sx) * 3;
                            float        w = k[i] * k[j];
                            sum[0] += c[0] * w;
                            sum[1] += c[1] * w;
                            sum[2] += c[2] * w;
                        }
                    }
                    memcpy(gMipB + (y * half + x) * 3, sum, sizeof sum);
                }
            float* swap = gMipA;
            gMipA = gMipB;
            gMipB = swap;
        }
    }
    endUpdateResource(&u);
    waitForAllResourceLoads();
    gUploaded = n;
}
} // namespace renderer
