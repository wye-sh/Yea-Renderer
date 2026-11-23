// The renderer core. In the host's YEA_PASS_RENDER (nothing bound) it records shadows
// (renderer_shadows), reflection probes, a depth prepass, light culling, AO and contact shadows, forward
// PBR shading and the sky into its own HDR target, auto exposure and TAA (renderer_taa). In
// YEA_PASS_WORLD, with the frame's target bound, it composites that image (tone mapped) onto it, so the
// host's own world quads draw over it.
#include "yea_renderer.h"
#include <yea/shadingrate.h>
#include <yea/raytracing.h>
#include "frame.h"
#include "preview.h"
#include "mathx.h"
#include "scene.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <Common_3/Graphics/Interfaces/IGraphics.h>
#include <Common_3/Resources/ResourceLoader/Interfaces/IResourceLoader.h>
#include <Common_3/Resources/ResourceLoader/ThirdParty/OpenSource/tinyimageformat/tinyimageformat_query.h>
#include <Common_3/Graphics/FSL/defaults.h>
#define SAMPLE_COUNT 1
#include "Shaders/Scene.srt.h"
#include "Shaders/Composite.srt.h"
#include "Shaders/Exposure.srt.h"
#include "Shaders/Tonemap.srt.h"
#undef SAMPLE_COUNT

using namespace mx;

namespace renderer
{
namespace
{
// --- GPU layouts (Shaders/Common.h.fsl, Composite.frag.fsl, ExposureCommon.h.fsl) ---------------
struct GpuFrame
{
    float    viewProj[4][4];
    float    view[4][4];
    float    camPos[4];
    float    screen[4];
    float    proj[4];
    float    sunDir[4][4];
    float    sunColor[4][4];
    float    skyZenith[4];
    float    skyHorizon[4];
    float    skyGround[4];
    uint32_t counts[4];
    uint32_t tiles[4];
    float    misc[4];
    float    cascadeEnd[4];
    float    shadow[4];
    float    shadow2[4];
    float    viewProjNoJitter[4][4];
    float    prevViewProj[4][4];
    float    aoParams[4];
    uint32_t probeInfo[4];
    float    probeParams[4];
    float    ssr[4];  // x max roughness, y thickness, z max distance, w edge fade
    float    ssr2[4]; // x intensity, y depth copy scale, z steps
    float    rt[4];   // x max roughness of ray-traced reflections
    float    atmosphere[4];  // renderer_sky: km, ground radius, top, the camera from the centre; aerial strength
    float    atmosphere2[4]; // aerial volume depth (km), the sky image's mips
    float    fog[4];  // renderer_fog: frame.h fogParams
    float    fog2[4]; // xy frame.h fogUv
    float    taps[32][4]; // xy: the shadow filters' Vogel disk, unturned
};
struct GpuMaterial
{
    float albedoRough[4];
    float params[4];
    float emissive[4];
};
struct CompositeData
{
    uint32_t mode[4];
    float    params[4];
    uint32_t rates[4]; // x Show Rates on, y tile px
};
struct ExposureData
{
    float    range[4];
    float    adapt[4];
    uint32_t size[4];
    float    dark[4];
};

const uint32_t kFlagCulled = 1, kFlagHeatmap = 2, kFlagCascades = 4, kFlagShadows = 8, kFlagAo = 16, kFlagAoDebug = 32, kFlagContact = 64,
               kFlagContactDebug = 128, kFlagUnlit = 256, kFlagSsr = 512, kFlagSsrDebug = 1024,
               kFlagReceiverPlane = 2048, kFlagRt = 4096, kFlagRtDebug = 8192, kFlagRtShadows = 16384,
               kFlagRtSun = 32768, kFlagArea = 65536, kFlagSky = 131072, kFlagFog = 262144;
const uint32_t kMaxFrames = 4;
const uint32_t kMaxUnculledLights = 256;
const uint32_t kHistogramBins = 128;

// --- GPU objects -------------------------------------------------------------------------------
Renderer* gRenderer;
Queue*    gQueue;
uint32_t  gFrames;
bool      gReady;

Shader *gForwardDepthShader, *gForwardShader, *gSkyShader, *gCompositeShader[4], *gHistogramShader[4], *gAdaptShader;
Shader *gMotionDepthShader, *gMotionShader, *gSkyMotionShader; // TAA on: with motion vectors
Shader*        gGlossShader;   // the prepass with a reflection-roughness output, for renderer_vrs
Pipeline*      gDepthGlossPipe[2];
RenderTarget*  gGloss;
Shader*        gSurfaceShader; // the prepass with roughness and normal outputs, for renderer_rt
Pipeline*      gDepthSurfacePipe[2];
RenderTarget*  gSurface;
Texture*       gLdr;          // the tonemap pass's packed output (R32_UINT), with TAA only
const YeaRaytracing* gRaytracing; // nullptr: no ray queries on this GPU
// The full forward shaders add PCSS, the debug views and Wireframe's unlit lines; the lean ones, used
// otherwise, leave them out and so need fewer registers.
Shader *gForwardFullShader, *gMotionFullShader;
Sampler*       gShadowSampler;
DescriptorSet *gScenePersistent, *gSceneFrame, *gCompositeSet, *gExposureSet;
// Scene pipelines in pairs: [0] culls back faces, [1] draws both sides (double-sided materials).
Pipeline *     gDepthPipe[2], *gForwardPipe[2], *gForwardFullPipe[2], *gForwardDepthPipe[2];
Pipeline *     gSkyPipe, *gCompositePipe, *gHistogramPipe, *gAdaptPipe, *gTonemapPipe;
Shader*        gTonemapShader;
DescriptorSet* gTonemapSet;
RenderTarget * gHdr, *gDepth, *gDummyDepth, *gMotion, *gDummyArray, *gDummyColor;
Texture*       gDummyRates;   // stands in for renderer_vrs's rate copy (R32_UINT)
Texture*       gTonyLut;      // Tony McMapface's 48^3 table (Assets/tony_mc_mapface.dds) while it is picked and loaded
bool           gTargetsFailed; // the screen targets were refused (logged once)
bool           gTonyTried;    // read the table since Tony McMapface was picked (once per pick, even when it failed)
const YeaShadingRate* gShadingRate; // nullptr: no variable-rate shading on this GPU
Buffer *       gFrameCb[kMaxFrames], *gInstanceBuf[kMaxFrames], *gMaterialBuf[kMaxFrames], *gLightBuf[kMaxFrames], *gStreamBuf[kMaxFrames];
Buffer *       gCompositeCb[kMaxFrames], *gExposureCb[kMaxFrames];
Buffer *       gLtc, *gShapeBuf[kMaxFrames], *gLookBuf[kMaxFrames];
Texture*       gLightImages; // the rect lights' images (images.cpp)
uint32_t*      gLookCpu;     // per GPU light: its image + 1, 0 none (gLightLooks)
RenderGpuLightShape* gShapeCpu;
v3                   gCamRight = { 1, 0, 0 }; // the view's right last frame, for a camera looking exactly straight down or up
Buffer *       gDfg, *gExposure, *gHistogram, *gDummyU32, *gDummyViews, *gDummyProbes;
Buffer*        gDummySh;     // stands in for renderer_sky's ambient (9 float4)
Texture*       gDummyVolume; // and its aerial perspective (2 x 2 x 2)
Buffer *       gPosBuf, *gAttrBuf, *gIndexBuf; // the vertex streams and indices (uploadMeshes)
uint32_t       gVertexCap, gIndexCap;          // what they hold
uint32_t       gMeshGen[1024];                 // each mesh row's data generation (RenderCaster::generation)
// Read as vertices (indices) and in shaders (ray tracing) at once, so nobody barriers them.
const ResourceState gGeometryState = (ResourceState)(RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER | RESOURCE_STATE_SHADER_RESOURCE);
const ResourceState gIndexState = (ResourceState)(RESOURCE_STATE_INDEX_BUFFER | RESOURCE_STATE_SHADER_RESOURCE);
Buffer *       gPrevBuf[kMaxFrames]; // last frame's instance rows, for motion vectors

uint32_t        gWidth, gHeight, gSamples;       // the targets as made (render size)
uint32_t        gDisplayWidth, gDisplayHeight;   // the frame's size the targets were made for (gLdr's)
bool            gTargetMotion;
uint32_t        gPipeSamples;                    // the pipelines as made
bool            gPipeMotion;
TinyImageFormat gFrameFormat = TinyImageFormat_UNDEFINED;
uint32_t        gFrameSamples;
ResourceState   gDepthState, gExposureState;
uint32_t        gUploadedVertices, gUploadedIndices, gUploadedMeshes, gUploadedEpoch;
uint64_t        gUploadedMeshHash; // of the Mesh rows uploaded

// --- CPU scratch, made once in load() ----------------------------------------------------------
struct InstanceInfo
{
    uint32_t mesh; // the version drawn
    uint32_t row;  // in the Instance table
    uint32_t flags; // 1 visible, 2 casts shadow, 4 double-sided
    float    sphere[4];
    uint64_t moved;
};
// Everything the GPU reads is built here and copied into the mapped buffers once: those are
// write-combined on many drivers, so reading them back (or scattered small writes) is very slow.
InstanceInfo*        gInfo;     // per GPU instance
RenderGpuInstance* gInstCpu;  // instance rows
uint32_t             gTapsMade; // the taps count gFrameCpu.taps holds
uint32_t*            gStream;   // visible instances by mesh
RenderGpuLight*    gLightCpu;
GpuMaterial*         gMaterialCpu;
GpuFrame             gFrameCpu;
RenderCaster* gCasters;
RenderCaster* gDrawables; // every instance, by mesh (reflection probes draw them all)
RenderLight*  gLocal;
uint32_t*     gOutside; // rows of lights outside the view, for what reflections see
// Visible instances by key: mesh * 2 + double-sided, so single- and double-sided ones draw apart.
uint32_t        gMeshCount[2048], gMeshStart[2048], gMeshCursor[2048];
RenderFrame   gFrame;
// Motion vectors: each instance row's world rows as last drawn, and the frame they were drawn in.
struct PrevRows
{
    float    row[3][4];
    uint64_t frame;
    uint32_t gen;
};
PrevRows* gPrevRows;   // per Instance table row
// Per Instance table row: the mesh version last drawn, and the frame that changed (cached shadows redraw it).
uint32_t* gLodMesh;
uint64_t* gLodChanged;
float (*gPrevCpu)[3][4]; // per GPU instance, copied into gPrevBuf
float    gPrevViewProj[4][4];
uint64_t gPrevViewProjFrame;

uint32_t snapSamples(int32_t s) { return s >= 8 ? 8 : s >= 4 ? 4 : s >= 2 ? 2 : 1; }
uint32_t sampleIndex(uint32_t s) { return s == 8 ? 3 : s == 4 ? 2 : s == 2 ? 1 : 0; }

Buffer* makeBuffer(uint64_t size, uint32_t stride, uint32_t descriptors, ResourceMemoryUsage usage, ResourceState start, const char* name,
                   const void* data = nullptr, bool zero = false, uint32_t flags = 0)
{
    Buffer*        b = nullptr;
    BufferLoadDesc d = {};
    d.mDesc.mDescriptors = (DescriptorType)descriptors;
    d.mDesc.mMemoryUsage = usage;
    d.mDesc.mSize = size;
    d.mDesc.pName = name;
    d.mDesc.mStartState = start;
    if (usage != RESOURCE_MEMORY_USAGE_GPU_ONLY)
        d.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
    d.mDesc.mFlags = (BufferCreationFlags)(d.mDesc.mFlags | flags);
    if (stride)
    {
        d.mDesc.mStructStride = stride;
        d.mDesc.mElementCount = (uint32_t)(size / stride);
    }
    d.pData = data;
    d.mForceReset = zero;
    d.ppBuffer = &b;
    addResource(&d, nullptr);
    if (b && usage != RESOURCE_MEMORY_USAGE_GPU_ONLY && b->pCpuMappedAddress)
        memset(b->pCpuMappedAddress, 0, size);
    return b;
}

void freeBuffer(Buffer*& b)
{
    if (b)
        removeResource(b);
    b = nullptr;
}

void freeTarget(RenderTarget*& t)
{
    if (t)
        removeRenderTarget(gRenderer, t);
    t = nullptr;
}

void freePipeline(Pipeline*& p)
{
    if (p)
        removePipeline(gRenderer, p);
    p = nullptr;
}

Shader* loadShader(const char* vert, const char* frag, const char* comp)
{
    ShaderLoadDesc d = {};
    d.mVert.pFileName = vert;
    d.mFrag.pFileName = frag;
    d.mComp.pFileName = comp;
    Shader* s = nullptr;
    addShader(gRenderer, &d, &s);
    return s;
}

// The split-sum DFG table, 32 x 32 over (N.V, perceptual roughness), with height-correlated
// Smith visibility: x scales F0, y is the bias. Our own numeric integration.
void buildDfg(float* out)
{
    const uint32_t n = 256;
    for (uint32_t j = 0; j < 32; ++j)
        for (uint32_t i = 0; i < 32; ++i)
        {
            float NoV = maxf((float)i / 31.0f, 1e-3f);
            float r = maxf((float)j / 31.0f, 0.02f);
            float a = r * r, a2 = a * a;
            v3    Vd = V(sqrtf(1 - NoV * NoV), 0, NoV);
            float A = 0, B = 0;
            for (uint32_t k = 0; k < n; ++k)
            {
                uint32_t bits = k;
                bits = (bits << 16u) | (bits >> 16u);
                bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
                bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
                bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
                bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
                float u = (float)k / (float)n, v = (float)bits * 2.3283064e-10f;
                float phi = 6.28318531f * u;
                float cosT = sqrtf((1 - v) / (1 + (a2 - 1) * v));
                float sinT = sqrtf(maxf(1 - cosT * cosT, 0));
                v3    H = V(sinT * cosf(phi), sinT * sinf(phi), cosT);
                float VoH = dot(Vd, H);
                v3    L = H * (2 * VoH) - Vd;
                float NoL = L.z, NoH = H.z;
                if (NoL > 0 && VoH > 0)
                {
                    float gv = NoL * sqrtf(NoV * NoV * (1 - a2) + a2);
                    float gl = NoV * sqrtf(NoL * NoL * (1 - a2) + a2);
                    float vis = 0.5f / (gv + gl);
                    float g = vis * 4 * NoL * VoH / NoH;
                    float fc = powf(1 - VoH, 5);
                    A += (1 - fc) * g;
                    B += fc * g;
                }
            }
            out[(j * 32 + i) * 2 + 0] = A / n;
            out[(j * 32 + i) * 2 + 1] = B / n;
        }
}

VertexLayout sceneLayout()
{
    VertexLayout l = {};
    // Positions, attributes (only the normal read so far) and the instance-rate stream: bindGeometry's three buffers.
    l.mBindingCount = 3;
    l.mBindings[0].mStride = RENDER_POSITION_STRIDE;
    l.mBindings[1].mStride = RENDER_ATTRIBUTE_STRIDE;
    l.mBindings[2].mStride = 4;
    l.mBindings[2].mRate = VERTEX_BINDING_RATE_INSTANCE;
    l.mAttribCount = 3;
    l.mAttribs[0] = {};
    l.mAttribs[0].mSemantic = SEMANTIC_POSITION;
    l.mAttribs[0].mFormat = TinyImageFormat_R32G32B32_SFLOAT;
    l.mAttribs[0].mBinding = 0;
    l.mAttribs[0].mLocation = 0;
    l.mAttribs[0].mOffset = 0;
    l.mAttribs[1].mSemantic = SEMANTIC_NORMAL;
    l.mAttribs[1].mFormat = TinyImageFormat_R16G16_SNORM; // octahedral
    l.mAttribs[1].mBinding = 1;
    l.mAttribs[1].mLocation = 1;
    l.mAttribs[1].mOffset = 0;
    l.mAttribs[2].mSemantic = SEMANTIC_TEXCOORD1;
    l.mAttribs[2].mFormat = TinyImageFormat_R32_UINT;
    l.mAttribs[2].mBinding = 2;
    l.mAttribs[2].mLocation = 2;
    l.mAttribs[2].mOffset = 0;
    return l;
}

void destroyScenePipelines()
{
    for (uint32_t s = 0; s < 2; ++s)
    {
        freePipeline(gDepthPipe[s]);
        freePipeline(gDepthGlossPipe[s]);
        freePipeline(gDepthSurfacePipe[s]);
        freePipeline(gForwardPipe[s]);
        freePipeline(gForwardFullPipe[s]);
        freePipeline(gForwardDepthPipe[s]);
    }
    freePipeline(gSkyPipe);
    freePipeline(gHistogramPipe);
    gPipeSamples = 0;
}

void createScenePipelines(uint32_t samples, bool motion)
{
    VertexLayout        layout = sceneLayout();
    // Triangles wind counter-clockwise around their outward normal (cross(b - a, c - a) points out);
    // seen through the left-handed camera that is clockwise on screen. The Forge flips Vulkan's
    // viewport, so D3D12 and Vulkan agree.
    RasterizerStateDesc raster = {};
    raster.mCullMode = CULL_MODE_BACK;
    raster.mFrontFace = FRONT_FACE_CW;
    raster.mMultiSample = samples > 1;
    TinyImageFormat formats[2] = { TinyImageFormat_R16G16B16A16_SFLOAT, TinyImageFormat_R16G16_SFLOAT };

    PipelineDesc desc = {};
    desc.mType = PIPELINE_TYPE_GRAPHICS;
    PIPELINE_LAYOUT_DESC(desc, SRT_LAYOUT_DESC(SceneSrt, Persistent), SRT_LAYOUT_DESC(SceneSrt, PerFrame), NULL, NULL);
    GraphicsPipelineDesc& g = desc.mGraphicsDesc;
    g.mPrimitiveTopo = PRIMITIVE_TOPO_TRI_LIST;
    g.mSampleCount = (SampleCount)samples;
    g.mDepthStencilFormat = TinyImageFormat_D32_SFLOAT;
    g.pRasterizerState = &raster;

    // Depth prepass: the forward vertex shader alone, so both passes write the same depth.
    DepthStateDesc depthWrite = {};
    depthWrite.mDepthTest = true;
    depthWrite.mDepthWrite = true;
    depthWrite.mDepthFunc = CMP_GEQUAL;
    g.mRenderTargetCount = 0;
    g.pColorFormats = nullptr;
    g.pDepthState = &depthWrite;
    g.pVertexLayout = &layout;
    g.pShaderProgram = motion ? gMotionDepthShader : gForwardDepthShader;
    desc.pName = "renderer_depth";
    for (uint32_t s = 0; s < 2; ++s)
    {
        raster.mCullMode = s ? CULL_MODE_NONE : CULL_MODE_BACK;
        addPipeline(gRenderer, &desc, &gDepthPipe[s]);
    }
    if (motion && gGlossShader)
    {
        TinyImageFormat glossFormat = TinyImageFormat_R8_UNORM;
        g.mRenderTargetCount = 1;
        g.pColorFormats = &glossFormat;
        g.pShaderProgram = gGlossShader;
        desc.pName = "renderer_depth_gloss";
        for (uint32_t s = 0; s < 2; ++s)
        {
            raster.mCullMode = s ? CULL_MODE_NONE : CULL_MODE_BACK;
            addPipeline(gRenderer, &desc, &gDepthGlossPipe[s]);
        }
        g.mRenderTargetCount = 0;
        g.pColorFormats = nullptr;
    }
    if (motion && gSurfaceShader)
    {
        TinyImageFormat surfaceFormats[2] = { TinyImageFormat_R16G16B16A16_SFLOAT, TinyImageFormat_R16G16_SFLOAT }; // surface, motion
        g.mRenderTargetCount = 2;
        g.pColorFormats = surfaceFormats;
        g.pShaderProgram = gSurfaceShader;
        desc.pName = "renderer_depth_surface";
        for (uint32_t s = 0; s < 2; ++s)
        {
            raster.mCullMode = s ? CULL_MODE_NONE : CULL_MODE_BACK;
            addPipeline(gRenderer, &desc, &gDepthSurfacePipe[s]);
        }
        g.mRenderTargetCount = 0;
        g.pColorFormats = nullptr;
    }

    DepthStateDesc depthTest = {};
    depthTest.mDepthTest = true;
    depthTest.mDepthWrite = false;
    depthTest.mDepthFunc = CMP_GEQUAL;
    g.mRenderTargetCount = motion ? 2 : 1;
    g.pColorFormats = formats;
    g.pDepthState = &depthTest;
    for (uint32_t s = 0; s < 2; ++s)
    {
        raster.mCullMode = s ? CULL_MODE_NONE : CULL_MODE_BACK;
        g.pDepthState = &depthTest;
        g.pShaderProgram = motion ? gMotionShader : gForwardShader;
        desc.pName = "renderer_forward";
        addPipeline(gRenderer, &desc, &gForwardPipe[s]);
        g.pShaderProgram = motion ? gMotionFullShader : gForwardFullShader;
        desc.pName = "renderer_forward_full";
        addPipeline(gRenderer, &desc, &gForwardFullPipe[s]);
        // F1's view modes skip the prepass (Wireframe would draw it as edges): the forward pass writes depth itself.
        g.pDepthState = &depthWrite;
        desc.pName = "renderer_forward_depth";
        addPipeline(gRenderer, &desc, &gForwardDepthPipe[s]);
    }
    g.pDepthState = &depthTest;
    raster.mCullMode = CULL_MODE_NONE;

    // The sky: a full-screen triangle at depth 0 (reversed-Z far), only where nothing was drawn.
    g.pVertexLayout = nullptr;
    g.pShaderProgram = motion ? gSkyMotionShader : gSkyShader;
    desc.pName = "renderer_sky";
    addPipeline(gRenderer, &desc, &gSkyPipe);

    PipelineDesc cdesc = {};
    cdesc.mType = PIPELINE_TYPE_COMPUTE;
    PIPELINE_LAYOUT_DESC(cdesc, NULL, SRT_LAYOUT_DESC(ExposureSrt, PerFrame), NULL, NULL);
    cdesc.mComputeDesc.pShaderProgram = gHistogramShader[sampleIndex(samples)];
    cdesc.pName = "renderer_histogram";
    addPipeline(gRenderer, &cdesc, &gHistogramPipe);
    gPipeSamples = samples;
    gPipeMotion = motion;
}

void createCompositePipeline(TinyImageFormat format, uint32_t frameSamples, uint32_t samples)
{
    freePipeline(gCompositePipe);
    RasterizerStateDesc raster = {};
    raster.mCullMode = CULL_MODE_NONE;
    DepthStateDesc depth = {};
    PipelineDesc   desc = {};
    desc.mType = PIPELINE_TYPE_GRAPHICS;
    PIPELINE_LAYOUT_DESC(desc, NULL, SRT_LAYOUT_DESC(CompositeSrt, PerFrame), NULL, NULL);
    GraphicsPipelineDesc& g = desc.mGraphicsDesc;
    g.mPrimitiveTopo = PRIMITIVE_TOPO_TRI_LIST;
    g.mRenderTargetCount = 1;
    g.pColorFormats = &format;
    g.mSampleCount = (SampleCount)frameSamples;
    g.mDepthStencilFormat = TinyImageFormat_UNDEFINED;
    g.pShaderProgram = gCompositeShader[sampleIndex(samples)];
    g.pRasterizerState = &raster;
    g.pDepthState = &depth;
    desc.pName = YEA_PIPELINE_PICTURE " renderer_composite"; // F1's Wireframe/Overdraw show the scene under it
    addPipeline(gRenderer, &desc, &gCompositePipe);
}

void destroyTargets()
{
    freeTarget(gHdr);
    freeTarget(gDepth);
    freeTarget(gMotion);
    freeTarget(gGloss);
    freeTarget(gSurface);
    if (gLdr)
        removeResource(gLdr);
    gLdr = nullptr;
    gWidth = gHeight = gSamples = 0;
    gDisplayWidth = gDisplayHeight = 0;
}

// The scene's targets at the render size w x h; the tonemap pass's output at the frame's size dw x dh (the same unless
// renderer_upscale renders smaller).
void createTargets(uint32_t w, uint32_t h, uint32_t samples, bool motion, uint32_t dw, uint32_t dh)
{
    RenderTargetDesc d = {};
    d.mWidth = w;
    d.mHeight = h;
    d.mDepth = 1;
    d.mArraySize = 1;
    d.mMipLevels = 1;
    d.mSampleCount = (SampleCount)samples;
    d.mFormat = TinyImageFormat_R16G16B16A16_SFLOAT;
    d.mStartState = RESOURCE_STATE_SHADER_RESOURCE;
    d.mDescriptors = DESCRIPTOR_TYPE_TEXTURE;
    d.mClearValue = { { 0, 0, 0, 0 } };
    d.pName = "renderer_hdr";
    addRenderTarget(gRenderer, &d, &gHdr);
    d.mFormat = TinyImageFormat_D32_SFLOAT;
    d.mStartState = RESOURCE_STATE_SHADER_RESOURCE;
    d.mClearValue = {};
    d.mClearValue.depth = 0.0f;
    d.pName = "renderer_depth";
    addRenderTarget(gRenderer, &d, &gDepth);
    gDepthState = RESOURCE_STATE_SHADER_RESOURCE;
    if (motion)
    {
        d.mFormat = TinyImageFormat_R16G16_SFLOAT;
        d.mClearValue = { { 0, 0, 0, 0 } };
        d.pName = "renderer_motion";
        addRenderTarget(gRenderer, &d, &gMotion);
        d.mFormat = TinyImageFormat_R8_UNORM;
        d.mClearValue = { { 1, 1, 1, 1 } };
        d.pName = "renderer_gloss";
        addRenderTarget(gRenderer, &d, &gGloss);
        if (gRaytracing)
        {
            d.mFormat = TinyImageFormat_R16G16B16A16_SFLOAT;
            d.mClearValue = { { 1, 0, 0, 0 } };
            d.pName = "renderer_surface";
            addRenderTarget(gRenderer, &d, &gSurface);
        }
        TextureDesc t = {};
        t.mWidth = dw;
        t.mHeight = dh;
        t.mDepth = t.mArraySize = t.mMipLevels = 1;
        t.mSampleCount = SAMPLE_COUNT_1;
        t.mFormat = TinyImageFormat_R32_UINT;
        t.mStartState = RESOURCE_STATE_SHADER_RESOURCE;
        t.mDescriptors = (DescriptorType)(DESCRIPTOR_TYPE_TEXTURE | DESCRIPTOR_TYPE_RW_TEXTURE);
        t.pName = "renderer_ldr";
        TextureLoadDesc tl = {};
        tl.pDesc = &t;
        tl.ppTexture = &gLdr;
        addResource(&tl, nullptr);
        waitForAllResourceLoads();
    }
    gWidth = w;
    gHeight = h;
    gDisplayWidth = dw;
    gDisplayHeight = dh;
    gSamples = samples;
    gTargetMotion = motion;
}

uint64_t meshHash(uint32_t rows)
{
    uint64_t h = 0xcbf29ce484222325ull;
    const uint8_t* p = (const uint8_t*)Meshes.begin();
    size_t         n = rows * sizeof(Mesh);
    for (size_t i = 0; i < n; ++i)
        h = (h ^ p[i]) * 0x100000001b3ull;
    return h;
}

// Writes table rows [first, first + count) of the vertex streams and indices into the GPU buffers (the queue idle).
void writeGeometry(uint32_t firstVertex, uint32_t vertexCount, uint32_t firstIndex, uint32_t indexCount)
{
    if (vertexCount)
    {
        BufferUpdateDesc pos = { gPosBuf, (uint64_t)firstVertex * RENDER_POSITION_STRIDE, (uint64_t)vertexCount * RENDER_POSITION_STRIDE };
        pos.mCurrentState = gGeometryState;
        beginUpdateResource(&pos);
        BufferUpdateDesc attr = { gAttrBuf, (uint64_t)firstVertex * RENDER_ATTRIBUTE_STRIDE, (uint64_t)vertexCount * RENDER_ATTRIBUTE_STRIDE };
        attr.mCurrentState = gGeometryState;
        beginUpdateResource(&attr);
        uint8_t* pd = (uint8_t*)pos.pMappedData;
        uint8_t* ad = (uint8_t*)attr.pMappedData;
        for (uint32_t i = 0; i < vertexCount; ++i)
        {
            const Vertex& v = Vertexes[firstVertex + i];
            memcpy(pd + (size_t)i * RENDER_POSITION_STRIDE, &v.pos, 12);
            uint32_t a[5] = { v.normal, v.tangent, 0, 0, v.color };
            memcpy(a + 2, &v.uv, 8);
            memcpy(ad + (size_t)i * RENDER_ATTRIBUTE_STRIDE, a, 20);
        }
        endUpdateResource(&pos);
        endUpdateResource(&attr);
    }
    if (indexCount)
    {
        BufferUpdateDesc ind = { gIndexBuf, (uint64_t)firstIndex * 4, (uint64_t)indexCount * 4 };
        ind.mCurrentState = gIndexState;
        beginUpdateResource(&ind);
        memcpy(ind.pMappedData, Indexes.begin() + firstIndex, (size_t)indexCount * 4);
        endUpdateResource(&ind);
    }
}

// Uploads the vertex and index tables when meshes were added (rare: a short stall is fine). Meshes only ever append
// until the scene starts over, so new rows go into the buffers' spare room; the buffers are made again, with room to
// grow, only when they are full or the scene started over.
void uploadMeshes()
{
    uint32_t meshes = Meshes.size(), epoch = scene_epoch();
    if (Vertexes.size() == gUploadedVertices && Indexes.size() == gUploadedIndices && meshes == gUploadedMeshes &&
        epoch == gUploadedEpoch && meshHash(meshes) == gUploadedMeshHash)
        return;
    if (meshBuilding())
        return;
    uint32_t vertices = Vertexes.size(), indices = Indexes.size();
    bool     append = gPosBuf && epoch == gUploadedEpoch && vertices >= gUploadedVertices && indices >= gUploadedIndices &&
                  meshes >= gUploadedMeshes && meshHash(gUploadedMeshes) == gUploadedMeshHash && vertices <= gVertexCap &&
                  indices <= gIndexCap;
    waitQueueIdle(gQueue); // frames in flight may read the buffers (a copy into them must not overlap a draw)
    uint32_t gen = (uint32_t)yea::host()->frame();
    gen = gen ? gen : 1;
    if (!append)
    {
        freeBuffer(gPosBuf);
        freeBuffer(gAttrBuf);
        freeBuffer(gIndexBuf);
        gVertexCap = gIndexCap = 0;
        gUploadedVertices = gUploadedIndices = 0;
        if (vertices && indices)
        {
            // GPU-only (fast to read every pass), with half again as much room, so the next model appends without a remake.
            // With ray queries also acceleration-structure build inputs, read as uint arrays by renderer_rt.
            uint32_t vcap = vertices + vertices / 2 < 65536 ? 65536 : vertices + vertices / 2;
            uint32_t icap = indices + indices / 2 < 196608 ? 196608 : indices + indices / 2;
            uint32_t rtFlags = gRaytracing ? (uint32_t)(BUFFER_CREATION_FLAG_SHADER_DEVICE_ADDRESS | BUFFER_CREATION_FLAG_ACCELERATION_STRUCTURE_BUILD_INPUT) : 0;
            gPosBuf = makeBuffer((uint64_t)vcap * RENDER_POSITION_STRIDE, 4, DESCRIPTOR_TYPE_VERTEX_BUFFER | DESCRIPTOR_TYPE_BUFFER,
                                 RESOURCE_MEMORY_USAGE_GPU_ONLY, gGeometryState, "renderer_positions", nullptr, false, rtFlags);
            gAttrBuf = makeBuffer((uint64_t)vcap * RENDER_ATTRIBUTE_STRIDE, 4, DESCRIPTOR_TYPE_VERTEX_BUFFER | DESCRIPTOR_TYPE_BUFFER,
                                  RESOURCE_MEMORY_USAGE_GPU_ONLY, gGeometryState, "renderer_attributes", nullptr, false, rtFlags);
            gIndexBuf = makeBuffer((uint64_t)icap * 4, 4, DESCRIPTOR_TYPE_INDEX_BUFFER | DESCRIPTOR_TYPE_BUFFER, RESOURCE_MEMORY_USAGE_GPU_ONLY,
                                   gIndexState, "renderer_indices", nullptr, false, rtFlags);
            waitForAllResourceLoads();
            if (!gPosBuf || !gAttrBuf || !gIndexBuf)
            {
                yea::host()->log(2, "renderer: could not make the mesh buffers");
                freeBuffer(gPosBuf);
                freeBuffer(gAttrBuf);
                freeBuffer(gIndexBuf);
            }
            else
                gVertexCap = vcap, gIndexCap = icap;
        }
    }
    if (gPosBuf)
    {
        writeGeometry(gUploadedVertices, vertices - gUploadedVertices, gUploadedIndices, indices - gUploadedIndices);
        waitForAllResourceLoads();
        for (uint32_t m = append ? gUploadedMeshes : 0; m < meshes && m < 1024; ++m)
            gMeshGen[m] = gen;
    }
    gUploadedVertices = gPosBuf ? vertices : 0;
    gUploadedIndices = gPosBuf ? indices : 0;
    gUploadedMeshes = meshes;
    gUploadedEpoch = epoch;
    gUploadedMeshHash = meshHash(meshes);
}

void quatRows(yea_float4 q, yea_float3 s, yea_float3 p, float out[3][4])
{
    float l = sqrtf(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    float x = 0, y = 0, z = 0, w = 1;
    if (l > 1e-8f)
    {
        x = q.x / l;
        y = q.y / l;
        z = q.z / l;
        w = q.w / l;
    }
    float r[3][3] = { { 1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w) },
                      { 2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w) },
                      { 2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y) } };
    const float sc[3] = { s.x, s.y, s.z };
    const float pos[3] = { p.x, p.y, p.z };
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
            out[i][j] = r[i][j] * sc[j];
        out[i][3] = pos[i];
    }
}

// Heap sort of the visible lights by view depth, nearest first.
void sortLights(RenderLight* a, uint32_t n)
{
    auto sift = [a](uint32_t i, uint32_t size) {
        for (;;)
        {
            uint32_t big = i, l = 2 * i + 1, r = l + 1;
            if (l < size && a[l].viewZ > a[big].viewZ)
                big = l;
            if (r < size && a[r].viewZ > a[big].viewZ)
                big = r;
            if (big == i)
                return;
            RenderLight t = a[i];
            a[i] = a[big];
            a[big] = t;
            i = big;
        }
    };
    for (uint32_t i = n / 2; i-- > 0;)
        sift(i, n);
    for (uint32_t end = n; end-- > 1;)
    {
        RenderLight t = a[0];
        a[0] = a[end];
        a[end] = t;
        sift(0, end);
    }
}

void bindGeometry(Cmd* cmd, Buffer* stream, uint32_t first)
{
    Buffer*  bufs[3] = { gPosBuf, gAttrBuf, stream };
    uint32_t strides[3] = { RENDER_POSITION_STRIDE, RENDER_ATTRIBUTE_STRIDE, 4 };
    uint64_t offsets[3] = { 0, 0, (uint64_t)first * 4 };
    cmdBindVertexBuffer(cmd, 3, bufs, strides, offsets);
}

// Single-sided instances with pipes[0] (back faces culled), then double-sided ones with pipes[1].
void drawVisible(Cmd* cmd, uint32_t fi, Pipeline* const pipes[2])
{
    uint32_t meshCount = Meshes.size() < 1024 ? Meshes.size() : 1024;
    for (uint32_t s = 0; s < 2; ++s)
    {
        bool bound = false;
        for (uint32_t m = 0; m < meshCount; ++m)
        {
            uint32_t key = m * 2 + s;
            if (!gMeshCount[key])
                continue;
            if (!bound)
            {
                cmdBindPipeline(cmd, pipes[s]);
                cmdBindDescriptorSet(cmd, 0, gScenePersistent);
                cmdBindDescriptorSet(cmd, fi, gSceneFrame);
                cmdBindIndexBuffer(cmd, gIndexBuf, INDEX_TYPE_UINT32, 0);
                bound = true;
            }
            const Mesh& mesh = Meshes[m];
            bindGeometry(cmd, gStreamBuf[fi], gMeshStart[key]);
            cmdDrawIndexedInstanced(cmd, mesh.indexCount, mesh.firstIndex, gMeshCount[key], mesh.firstVertex, 0);
        }
    }
}

void barrierRt(Cmd* cmd, RenderTarget* rt, ResourceState from, ResourceState to)
{
    RenderTargetBarrier b = { rt, from, to };
    cmdResourceBarrier(cmd, 0, nullptr, 0, nullptr, 1, &b);
}

void barrierBuf(Cmd* cmd, Buffer* buf, ResourceState from, ResourceState to)
{
    BufferBarrier b = { buf, from, to };
    cmdResourceBarrier(cmd, 1, &b, 0, nullptr, 0, nullptr);
}
} // namespace

// GPU timings (the gpu_timings knob): a timestamp pair per pass, read back framesInFlight frames later
// (that frame's fence has passed by then, so nothing waits), averaged and logged every 10 s.
namespace
{
enum Timing : uint32_t { kTimeSky, kTimeShadows, kTimeProbes, kTimePrepass, kTimeCulling, kTimeAo, kTimeContact, kTimeSsr, kTimeVrs, kTimeRt, kTimeFog, kTimeForward, kTimeExposure, kTimeTaa, kTimeUpscale, kTimePost, kTimeTonemap, kTimeComposite, kTimeTotal, kTimings };
const char* const kTimingNames[kTimings] = { "sky", "shadows", "probes", "prepass", "culling", "ao", "contact", "ssr", "vrs", "rt", "fog", "forward+sky", "exposure", "taa", "upscale", "post", "tonemap", "composite", "total" };
QueryPool*        gTimePool[kMaxFrames];
uint32_t          gTimeUsed[kMaxFrames]; // bits: what that frame recorded
double            gTimeTicks;            // per second
double            gTimeSum[kTimings], gTimeMax[kTimings];
uint32_t          gTimeCount[kTimings];
double            gTimeLogged;
bool              gTimedFrame; // this frame's render was timed (the composite follows it)

void timeBegin(Cmd* cmd, uint32_t fi, uint32_t t)
{
    if (!gTimePool[fi])
        return;
    QueryDesc d = { t };
    cmdBeginQuery(cmd, gTimePool[fi], &d);
    gTimeUsed[fi] |= 1u << t;
}
void timeEnd(Cmd* cmd, uint32_t fi, uint32_t t)
{
    if (!gTimePool[fi] || !(gTimeUsed[fi] & (1u << t)))
        return;
    QueryDesc d = { t };
    cmdEndQuery(cmd, gTimePool[fi], &d);
}
// At the start of a frame's recording: what this slot measured last time, then a fresh start.
void timeFrame(Cmd* cmd, uint32_t fi, bool on, const char* scene)
{
    if (!on || !gTimePool[fi])
    {
        gTimeUsed[fi] = 0;
        return;
    }
    for (uint32_t t = 0; t < kTimings; ++t)
    {
        if (!(gTimeUsed[fi] & (1u << t)))
            continue;
        QueryData q = {};
        getQueryData(gRenderer, gTimePool[fi], t, &q);
        if (q.mEndTimestamp <= q.mBeginTimestamp)
            continue;
        double ms = (double)(q.mEndTimestamp - q.mBeginTimestamp) * 1000.0 / gTimeTicks;
        gTimeSum[t] += ms;
        gTimeMax[t] = ms > gTimeMax[t] ? ms : gTimeMax[t];
        gTimeCount[t]++;
    }
    cmdResetQuery(cmd, gTimePool[fi], 0, kTimings);
    gTimeUsed[fi] = 0;
    double now = yea::host()->time();
    if (now - gTimeLogged >= 10.0 && gTimeCount[kTimeTotal])
    {
        char line[512];
        int  n = snprintf(line, sizeof line, "gpu ms avg/max over %u frames:", gTimeCount[kTimeTotal]);
        for (uint32_t t = 0; t < kTimings && n < (int)sizeof line - 40; ++t)
            if (gTimeCount[t])
                n += snprintf(line + n, sizeof line - n, " %s %.3f/%.3f", kTimingNames[t], gTimeSum[t] / gTimeCount[t], gTimeMax[t]);
        snprintf(line + n, sizeof line - n, " | %s", scene);
        yea::log(line);
        memset(gTimeSum, 0, sizeof gTimeSum);
        memset(gTimeMax, 0, sizeof gTimeMax);
        memset(gTimeCount, 0, sizeof gTimeCount);
        gTimeLogged = now;
    }
}

} // namespace

// One of the table's R9G9B9E5 channels (9-bit mantissa m, shared exponent e: m * 2^(e - 24)) as a half float. Every
// such value is a half exactly: at most 9 significant bits, between 2^-24 and 65408.
uint16_t halfOf(uint32_t m, uint32_t e)
{
    if (!m)
        return 0;
    uint32_t k = 8;
    while (!(m >> k))
        --k;                                // m's top bit
    int32_t E = (int32_t)(k + e) - 9;        // the half's biased exponent
    if (E <= 0)
        return (uint16_t)(m << e);         // subnormal: m * 2^(e - 24) = (m << e) * 2^-24
    return (uint16_t)((E << 10) | ((m << (10 - k)) & 0x3FFu));
}

// Tony McMapface's table: its DDS as published (DX10 header, R9G9B9E5, 48^3, one mip), read only while Tony McMapface
// is picked (a game shipped with another tonemapper leaves the file out), into a 2D RGBA16F strip of its 48 slices side
// by side (2304 x 48), filtered across slices in the shader; nullptr when the file is missing or not that table.
Texture* loadTonyLut()
{
    const uint32_t kDim = 48, kHeader = 148, kBytes = kDim * kDim * kDim * 4;
    Texture*       tex = nullptr;
    char           path[1024];
    FILE*          f = yea::asset("tony_mc_mapface.dds", path, sizeof path) ? fopen(path, "rb") : nullptr;
    uint8_t*       data = f ? (uint8_t*)malloc(kHeader + kBytes) : nullptr;
    size_t         n = data ? fread(data, 1, kHeader + kBytes + 1, f) : 0; // + 1: the file is exactly that long
    uint32_t       h[37] = {};
    if (n == kHeader + kBytes)
        memcpy(h, data, sizeof h);
    if (f)
        fclose(f);
    // "DDS ", size 124; height, width, depth; "DX10"; R9G9B9E5_SHAREDEXP (67), a 3D texture (4)
    if (h[0] == 0x20534444u && h[1] == 124 && h[3] == kDim && h[4] == kDim && h[6] == kDim && h[21] == 0x30315844u && h[32] == 67 &&
        h[33] == 4)
    {
        TextureDesc t = {};
        t.mWidth = kDim * kDim;
        t.mHeight = kDim;
        t.mDepth = t.mArraySize = t.mMipLevels = 1;
        t.mSampleCount = SAMPLE_COUNT_1;
        t.mFormat = TinyImageFormat_R16G16B16A16_SFLOAT;
        t.mStartState = RESOURCE_STATE_SHADER_RESOURCE;
        t.mDescriptors = DESCRIPTOR_TYPE_TEXTURE;
        t.pName = "renderer_tony_mc_mapface";
        TextureLoadDesc tl = {};
        tl.pDesc = &t;
        tl.ppTexture = &tex;
        addResource(&tl, nullptr);
        waitForAllResourceLoads();
        if (tex)
        {
            TextureUpdateDesc u = {};
            u.pTexture = tex;
            u.mMipLevels = u.mLayerCount = 1;
            u.mCurrentState = RESOURCE_STATE_SHADER_RESOURCE;
            beginUpdateResource(&u);
            TextureSubresourceUpdate s = u.getSubresourceUpdateDesc(0, 0);
            const uint8_t*           src = data + kHeader;
            for (uint32_t y = 0; y < kDim && y < s.mRowCount; ++y)
            {
                uint16_t* row = (uint16_t*)(s.pMappedData + y * s.mDstRowStride);
                for (uint32_t z = 0; z < kDim; ++z)
                    for (uint32_t x = 0; x < kDim; ++x)
                    {
                        uint32_t v, e;
                        memcpy(&v, src + ((z * kDim + y) * kDim + x) * 4, 4);
                        e = v >> 27;
                        uint16_t* o = row + (z * kDim + x) * 4;
                        o[0] = halfOf(v & 511u, e);
                        o[1] = halfOf((v >> 9) & 511u, e);
                        o[2] = halfOf((v >> 18) & 511u, e);
                        o[3] = 0x3C00; // 1
                    }
            }
            endUpdateResource(&u);
            waitForAllResourceLoads();
        }
    }
    free(data);
    if (!tex)
        yea::host()->log(1, "renderer: Tony McMapface's table (Assets/tony_mc_mapface.dds) didn't load: Khronos PBR Neutral stands in");
    return tex;
}

#include "preview_core.h"

int load(int reloaded)
{
    const YeaHost* host = yea::host();
    bindState(); // first: the exports use these blocks even if the GPU side fails below
    gShadingRate = yea::shadingRate();
    gRaytracing = yea::raytracing();
    // A new material layout starts its table empty: instances would point at materials that are
    // gone, so the scene starts over and its owner builds it again (as on a clear()).
    if (Materials.size() == 0 && Instances.size() > 0)
        clear();
    // So does a vertex or index layout change (the tables start empty while meshes still point into them).
    if (Meshes.size() > 0 && (Meshes[Meshes.size() - 1].firstVertex + Meshes[Meshes.size() - 1].vertexCount > Vertexes.size() ||
                              Meshes[Meshes.size() - 1].firstIndex + Meshes[Meshes.size() - 1].indexCount > Indexes.size()))
        clear();
    gRenderer = (Renderer*)host->renderer;
    gQueue = (Queue*)host->queue;
    gFrames = host->framesInFlight < kMaxFrames ? host->framesInFlight : kMaxFrames;
    getTimestampFrequency(gQueue, &gTimeTicks);
    for (uint32_t i = 0; i < gFrames && gTimeTicks > 0; ++i)
    {
        QueryPoolDesc qd = {};
        qd.pName = "renderer_timings";
        qd.mType = QUERY_TYPE_TIMESTAMP;
        qd.mQueryCount = kTimings;
        initQueryPool(gRenderer, &qd, &gTimePool[i]);
        gTimeUsed[i] = 0;
    }

    gForwardDepthShader = loadShader("renderer_forward.vert", nullptr, nullptr);
    gForwardShader = loadShader("renderer_forward.vert", "renderer_forward.frag", nullptr);
    gSkyShader = loadShader("renderer_fullscreen.vert", "renderer_sky.frag", nullptr);
    gMotionDepthShader = loadShader("renderer_forward_motion.vert", nullptr, nullptr);
    gMotionShader = loadShader("renderer_forward_motion.vert", "renderer_forward_motion.frag", nullptr);
    gForwardFullShader = loadShader("renderer_forward.vert", "renderer_forward_full.frag", nullptr);
    gMotionFullShader = loadShader("renderer_forward_motion.vert", "renderer_forward_motion_full.frag", nullptr);
    gSkyMotionShader = loadShader("renderer_fullscreen.vert", "renderer_sky_motion.frag", nullptr);
    gGlossShader = loadShader("renderer_forward_motion.vert", "renderer_gloss.frag", nullptr);
    gSurfaceShader = gRaytracing ? loadShader("renderer_forward_motion.vert", "renderer_surface.frag", nullptr) : nullptr;
    const char* composite[4] = { "renderer_composite_1.frag", "renderer_composite_2.frag", "renderer_composite_4.frag",
                                 "renderer_composite_8.frag" };
    const char* histogram[4] = { "renderer_histogram_1.comp", "renderer_histogram_2.comp", "renderer_histogram_4.comp",
                                 "renderer_histogram_8.comp" };
    bool ok = gForwardDepthShader && gForwardShader && gSkyShader && gMotionDepthShader && gMotionShader && gSkyMotionShader &&
              gForwardFullShader && gMotionFullShader && gGlossShader;
    for (int i = 0; i < 4; ++i)
    {
        gCompositeShader[i] = loadShader("renderer_fullscreen.vert", composite[i], nullptr);
        gHistogramShader[i] = loadShader(nullptr, nullptr, histogram[i]);
        ok = ok && gCompositeShader[i] && gHistogramShader[i];
    }
    gAdaptShader = loadShader(nullptr, nullptr, "renderer_adapt.comp");
    gTonemapShader = loadShader(nullptr, nullptr, "renderer_tonemap.comp");
    if (!ok || !gAdaptShader || !gTonemapShader)
    {
        host->log(2, "renderer: shaders missing");
        return 1;
    }

    SamplerDesc s = {};
    s.mMinFilter = FILTER_LINEAR;
    s.mMagFilter = FILTER_LINEAR;
    s.mMipMapMode = MIPMAP_MODE_NEAREST;
    s.mAddressU = s.mAddressV = s.mAddressW = ADDRESS_MODE_CLAMP_TO_EDGE;
    s.mCompareFunc = CMP_LEQUAL;
    addSampler(gRenderer, &s, &gShadowSampler);

    DescriptorSetDesc persistent = SRT_SET_DESC(SceneSrt, Persistent, 1, 0);
    addDescriptorSet(gRenderer, &persistent, &gScenePersistent);
    DescriptorSetDesc perFrame = SRT_SET_DESC(SceneSrt, PerFrame, gFrames, 0);
    addDescriptorSet(gRenderer, &perFrame, &gSceneFrame);
    DescriptorSetDesc compositeDesc = SRT_SET_DESC(CompositeSrt, PerFrame, gFrames, 0);
    addDescriptorSet(gRenderer, &compositeDesc, &gCompositeSet);
    DescriptorSetDesc exposureDesc = SRT_SET_DESC(ExposureSrt, PerFrame, gFrames, 0);
    addDescriptorSet(gRenderer, &exposureDesc, &gExposureSet);
    DescriptorSetDesc tonemapDesc = SRT_SET_DESC(TonemapSrt, PerFrame, gFrames, 0);
    addDescriptorSet(gRenderer, &tonemapDesc, &gTonemapSet);

    const ResourceMemoryUsage cpu = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
    const ResourceState       sr = RESOURCE_STATE_SHADER_RESOURCE;
    for (uint32_t i = 0; i < gFrames; ++i)
    {
        gFrameCb[i] = makeBuffer(sizeof(GpuFrame), 0, DESCRIPTOR_TYPE_UNIFORM_BUFFER, cpu, RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
                                 "renderer_frame");
        gCompositeCb[i] = makeBuffer(256, 0, DESCRIPTOR_TYPE_UNIFORM_BUFFER, cpu, RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
                                     "renderer_composite");
        gExposureCb[i] = makeBuffer(256, 0, DESCRIPTOR_TYPE_UNIFORM_BUFFER, cpu, RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
                                    "renderer_exposure");
        gInstanceBuf[i] = makeBuffer((uint64_t)Instances.capacity() * sizeof(RenderGpuInstance), sizeof(RenderGpuInstance),
                                     DESCRIPTOR_TYPE_BUFFER, cpu, sr, "renderer_instances");
        gMaterialBuf[i] = makeBuffer((uint64_t)Materials.capacity() * sizeof(GpuMaterial), sizeof(GpuMaterial), DESCRIPTOR_TYPE_BUFFER,
                                     cpu, sr, "renderer_materials");
        gLightBuf[i] = makeBuffer((uint64_t)Lights.capacity() * sizeof(RenderGpuLight), sizeof(RenderGpuLight), DESCRIPTOR_TYPE_BUFFER,
                                  cpu, sr, "renderer_lights");
        gShapeBuf[i] = makeBuffer((uint64_t)Lights.capacity() * sizeof(RenderGpuLightShape), sizeof(RenderGpuLightShape),
                                  DESCRIPTOR_TYPE_BUFFER, cpu, sr, "renderer_light_shapes");
        gLookBuf[i] = makeBuffer((uint64_t)Lights.capacity() * 4, 4, DESCRIPTOR_TYPE_BUFFER, cpu, sr, "renderer_light_looks");
        gStreamBuf[i] = makeBuffer((uint64_t)Instances.capacity() * 4, 0, DESCRIPTOR_TYPE_VERTEX_BUFFER, cpu,
                                   RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER, "renderer_stream");
        gPrevBuf[i] = makeBuffer((uint64_t)Instances.capacity() * 48, 48, DESCRIPTOR_TYPE_BUFFER, cpu, sr, "renderer_prev_instances");
    }
    gDummyU32 = makeBuffer(256, 4, DESCRIPTOR_TYPE_BUFFER, cpu, sr, "renderer_dummy_u32");
    gDummyViews = makeBuffer(sizeof(RenderGpuShadowView) * 4, sizeof(RenderGpuShadowView), DESCRIPTOR_TYPE_BUFFER, cpu, sr,
                             "renderer_dummy_views");
    gDummyProbes = makeBuffer(sizeof(RenderGpuProbe) * 4, sizeof(RenderGpuProbe), DESCRIPTOR_TYPE_BUFFER, cpu, sr, "renderer_dummy_probes");
    gDummySh = makeBuffer(9 * 16, 16, DESCRIPTOR_TYPE_BUFFER, cpu, sr, "renderer_dummy_sky_sh");

    float* dfg = new float[32 * 32 * 2];
    buildDfg(dfg);
    gDfg = makeBuffer(32 * 32 * 8, 8, DESCRIPTOR_TYPE_BUFFER, cpu, sr, "renderer_dfg");
    if (gDfg && gDfg->pCpuMappedAddress)
        memcpy(gDfg->pCpuMappedAddress, dfg, 32 * 32 * 8);
    // The area lights' tables (tools/ltc_fit.cpp): 64 x 64 inverse transforms, then 64 x 64 norms and the horizon
    // table, float4 each. Missing, area lights stay dark.
    gLtc = makeBuffer(2 * 64 * 64 * 16, 16, DESCRIPTOR_TYPE_BUFFER, cpu, sr, "renderer_ltc");
    if (gLtc && gLtc->pCpuMappedAddress)
    {
        char  path[1024];
        FILE* f = yea::asset("ltc.bin", path, sizeof path) ? fopen(path, "rb") : nullptr;
        char  magic[4] = {};
        int   n = 0;
        bool  ok = f && fread(magic, 1, 4, f) == 4 && !memcmp(magic, "LTC1", 4) && fread(&n, 4, 1, f) == 1 && n == 64 &&
                  fread(gLtc->pCpuMappedAddress, 16, 2 * 64 * 64, f) == 2 * 64 * 64;
        if (f)
            fclose(f);
        if (!ok)
        {
            memset(gLtc->pCpuMappedAddress, 0, 2 * 64 * 64 * 16);
            yea::host()->log(1, "renderer: Assets/ltc.bin is missing or not the area lights' tables; area lights are dark");
        }
    }
    gExposure = makeBuffer(16, 4, DESCRIPTOR_TYPE_BUFFER | DESCRIPTOR_TYPE_RW_BUFFER, RESOURCE_MEMORY_USAGE_GPU_ONLY,
                           RESOURCE_STATE_UNORDERED_ACCESS, "renderer_exposure", nullptr, true);
    gHistogram = makeBuffer(kHistogramBins * 4, 4, DESCRIPTOR_TYPE_BUFFER | DESCRIPTOR_TYPE_RW_BUFFER, RESOURCE_MEMORY_USAGE_GPU_ONLY,
                            RESOURCE_STATE_UNORDERED_ACCESS, "renderer_histogram", nullptr, true);
    waitForAllResourceLoads();
    delete[] dfg;
    gExposureState = RESOURCE_STATE_UNORDERED_ACCESS;

    RenderTargetDesc d = {};
    d.mWidth = d.mHeight = 2; // one texel high gets a 1D view on Vulkan, which the 2D slots reject
    d.mDepth = d.mArraySize = d.mMipLevels = 1;
    d.mSampleCount = SAMPLE_COUNT_1;
    d.mFormat = TinyImageFormat_D32_SFLOAT;
    d.mStartState = RESOURCE_STATE_SHADER_RESOURCE;
    d.mDescriptors = DESCRIPTOR_TYPE_TEXTURE;
    d.pName = "renderer_dummy_depth";
    addRenderTarget(gRenderer, &d, &gDummyDepth);
    d.mFormat = TinyImageFormat_R16G16B16A16_SFLOAT;
    d.pName = "renderer_dummy_color";
    addRenderTarget(gRenderer, &d, &gDummyColor);
    {
        TextureDesc t = {};
        t.mWidth = t.mHeight = 2;
        t.mDepth = t.mArraySize = t.mMipLevels = 1;
        t.mSampleCount = SAMPLE_COUNT_1;
        t.mFormat = TinyImageFormat_R32_UINT;
        t.mStartState = RESOURCE_STATE_SHADER_RESOURCE;
        t.mDescriptors = DESCRIPTOR_TYPE_TEXTURE;
        t.pName = "renderer_dummy_rates";
        TextureLoadDesc tl = {};
        tl.pDesc = &t;
        tl.ppTexture = &gDummyRates;
        addResource(&tl, nullptr);
        t.mDepth = 2;
        t.mFormat = TinyImageFormat_R16G16B16A16_SFLOAT;
        t.pName = "renderer_dummy_volume";
        tl.ppTexture = &gDummyVolume;
        addResource(&tl, nullptr);
        waitForAllResourceLoads();
    }
    // stands in for the probe array: two slices, so an array view
    d.mArraySize = 2;
    d.pName = "renderer_dummy_probes";
    addRenderTarget(gRenderer, &d, &gDummyArray);

    // Every buffer and texture made above, before anything uses them: a GPU or driver that refuses one stops the
    // module cleanly instead of crashing on it later.
    lightImagesInit();
    gLightImages = lightImagesMake();
    bool made = gDfg && gLtc && gLightImages && gExposure && gHistogram && gDummyU32 && gDummyViews && gDummyProbes && gDummySh && gDummyVolume && gDummyDepth && gDummyColor &&
                gDummyRates && gDummyArray && gScenePersistent && gSceneFrame && gCompositeSet && gExposureSet && gTonemapSet && gShadowSampler;
    for (uint32_t i = 0; i < gFrames; ++i)
        made = made && gFrameCb[i] && gCompositeCb[i] && gExposureCb[i] && gInstanceBuf[i] && gMaterialBuf[i] && gLightBuf[i] && gShapeBuf[i] && gLookBuf[i] && gLookBuf[i]->pCpuMappedAddress &&
               gStreamBuf[i] && gPrevBuf[i] && gFrameCb[i]->pCpuMappedAddress && gCompositeCb[i]->pCpuMappedAddress &&
               gExposureCb[i]->pCpuMappedAddress && gInstanceBuf[i]->pCpuMappedAddress && gMaterialBuf[i]->pCpuMappedAddress &&
               gLightBuf[i]->pCpuMappedAddress && gShapeBuf[i]->pCpuMappedAddress && gStreamBuf[i]->pCpuMappedAddress && gPrevBuf[i]->pCpuMappedAddress;
    if (!made)
    {
        yea::host()->log(2, "renderer: the GPU refused one of its buffers or textures; the Renderer is off");
        unload();
        return 1;
    }
    DescriptorData p[4] = {};
    p[3].mIndex = SRT_RES_IDX(SceneSrt, Persistent, gLightImages);
    p[3].ppTextures = &gLightImages;
    p[0].mIndex = SRT_RES_IDX(SceneSrt, Persistent, gShadowCmp);
    p[0].ppSamplers = &gShadowSampler;
    p[1].mIndex = SRT_RES_IDX(SceneSrt, Persistent, gDfg);
    p[1].ppBuffers = &gDfg;
    p[2].mIndex = SRT_RES_IDX(SceneSrt, Persistent, gLtc);
    p[2].ppBuffers = &gLtc;
    updateDescriptorSet(gRenderer, 0, gScenePersistent, 4, p);

    PipelineDesc cdesc = {};
    cdesc.mType = PIPELINE_TYPE_COMPUTE;
    PIPELINE_LAYOUT_DESC(cdesc, NULL, SRT_LAYOUT_DESC(ExposureSrt, PerFrame), NULL, NULL);
    cdesc.mComputeDesc.pShaderProgram = gAdaptShader;
    cdesc.pName = "renderer_adapt";
    addPipeline(gRenderer, &cdesc, &gAdaptPipe);
    PipelineDesc tdesc = {};
    tdesc.mType = PIPELINE_TYPE_COMPUTE;
    PIPELINE_LAYOUT_DESC(tdesc, NULL, SRT_LAYOUT_DESC(TonemapSrt, PerFrame), NULL, NULL);
    tdesc.mComputeDesc.pShaderProgram = gTonemapShader;
    tdesc.pName = "renderer_tonemap";
    addPipeline(gRenderer, &tdesc, &gTonemapPipe);

    gInfo = new InstanceInfo[Instances.capacity()];
    gInstCpu = new RenderGpuInstance[Instances.capacity()];
    gStream = new uint32_t[Instances.capacity()];
    gLightCpu = new RenderGpuLight[Lights.capacity()];
    gShapeCpu = new RenderGpuLightShape[Lights.capacity()];
    gLookCpu = new uint32_t[Lights.capacity()];
    gMaterialCpu = new GpuMaterial[Materials.capacity()];
    gCasters = new RenderCaster[Instances.capacity()];
    gDrawables = new RenderCaster[Instances.capacity()];
    gLocal = new RenderLight[Lights.capacity()];
    gOutside = new uint32_t[Lights.capacity()];
    gPrevRows = new PrevRows[Instances.capacity()];
    memset(gPrevRows, 0, sizeof(PrevRows) * Instances.capacity());
    gLodMesh = new uint32_t[Instances.capacity()];
    memset(gLodMesh, 0xFF, sizeof(uint32_t) * Instances.capacity());
    gLodChanged = new uint64_t[Instances.capacity()];
    memset(gLodChanged, 0, sizeof(uint64_t) * Instances.capacity());
    gPrevCpu = new float[Instances.capacity()][3][4];
    gPrevViewProjFrame = 0;
    previewLoad(); // the Model Viewer's preview: off on its own if it fails

    gReady = true;
    return 0;
}

void unload()
{
    if (!gRenderer)
        return;
    waitQueueIdle(gQueue);
    previewUnload();
    for (uint32_t i = 0; i < gFrames; ++i)
        if (gTimePool[i])
            exitQueryPool(gRenderer, gTimePool[i]), gTimePool[i] = nullptr;
    destroyScenePipelines();
    freePipeline(gCompositePipe);
    freePipeline(gAdaptPipe);
    freePipeline(gTonemapPipe);
    destroyTargets();
    freeTarget(gDummyDepth);
    freeTarget(gDummyArray);
    freeTarget(gDummyColor);
    if (gDummyRates)
        removeResource(gDummyRates);
    gDummyRates = nullptr;
    if (gTonyLut)
        removeResource(gTonyLut);
    gTonyLut = nullptr;
    gTonyTried = false;
    for (uint32_t i = 0; i < gFrames; ++i)
    {
        freeBuffer(gFrameCb[i]);
        freeBuffer(gCompositeCb[i]);
        freeBuffer(gExposureCb[i]);
        freeBuffer(gInstanceBuf[i]);
        freeBuffer(gMaterialBuf[i]);
        freeBuffer(gLightBuf[i]);
        freeBuffer(gShapeBuf[i]);
        freeBuffer(gLookBuf[i]);
        freeBuffer(gStreamBuf[i]);
        freeBuffer(gPrevBuf[i]);
    }
    freeBuffer(gDummyU32);
    freeBuffer(gDummyViews);
    freeBuffer(gDummyProbes);
    freeBuffer(gDummySh);
    if (gDummyVolume)
        removeResource(gDummyVolume);
    gDummyVolume = nullptr;
    freeBuffer(gDfg);
    freeBuffer(gLtc);
    if (gLightImages)
        removeResource(gLightImages);
    gLightImages = nullptr;
    lightImagesFree();
    freeBuffer(gExposure);
    freeBuffer(gHistogram);
    freeBuffer(gPosBuf);
    freeBuffer(gAttrBuf);
    freeBuffer(gIndexBuf);
    gVertexCap = gIndexCap = 0;
    DescriptorSet** sets[5] = { &gScenePersistent, &gSceneFrame, &gCompositeSet, &gExposureSet, &gTonemapSet };
    for (DescriptorSet** set : sets)
        if (*set)
            removeDescriptorSet(gRenderer, *set), *set = nullptr;
    if (gShadowSampler)
        removeSampler(gRenderer, gShadowSampler), gShadowSampler = nullptr;
    // Each one let go and forgotten, so a second unload (after a load that gave up) frees nothing twice.
    Shader** shaders[] = { &gForwardDepthShader, &gForwardShader, &gSkyShader, &gAdaptShader, &gMotionDepthShader, &gMotionShader,
                           &gSkyMotionShader, &gForwardFullShader, &gMotionFullShader, &gGlossShader, &gSurfaceShader, &gTonemapShader };
    for (Shader** s : shaders)
        if (*s)
            removeShader(gRenderer, *s), *s = nullptr;
    for (int i = 0; i < 4; ++i)
    {
        if (gCompositeShader[i])
            removeShader(gRenderer, gCompositeShader[i]), gCompositeShader[i] = nullptr;
        if (gHistogramShader[i])
            removeShader(gRenderer, gHistogramShader[i]), gHistogramShader[i] = nullptr;
    }
}

void update(uint32_t phase, float dt) {}

uint64_t gRenderedFrame; // host frame whose scene YEA_PASS_RENDER made, for YEA_PASS_WORLD's composite
uint32_t gRenderedFi;

// YEA_PASS_WORLD: the frame's target is bound; this frame's scene onto it.
void composite(const YeaFrame* yf)
{
    if (gRenderedFrame != yea::host()->frame() || !gCompositePipe || yf->width != gDisplayWidth || yf->height != gDisplayHeight)
        return;
    Cmd* cmd = (Cmd*)yf->cmd;
    cmdSetViewport(cmd, 0, 0, (float)yf->width, (float)yf->height, 0, 1);
    cmdSetScissor(cmd, 0, 0, yf->width, yf->height);
    if (gTimedFrame)
        timeBegin(cmd, gRenderedFi, kTimeComposite);
    cmdBindPipeline(cmd, gCompositePipe);
    cmdBindDescriptorSet(cmd, gRenderedFi, gCompositeSet);
    cmdDraw(cmd, 3, 0);
    if (gTimedFrame)
    {
        timeEnd(cmd, gRenderedFi, kTimeComposite);
        cmdResolveQuery(cmd, gTimePool[gRenderedFi], kTimeComposite, 1);
    }
}

void draw(uint32_t pass, const YeaFrame* yf)
{
    if (!gReady || !yf->width || !yf->height)
        return;
    if (pass == YEA_PASS_WORLD)
    {
        composite(yf);
        return;
    }
    if (pass != YEA_PASS_RENDER)
        return;
    Cmd*          cmd = (Cmd*)yf->cmd;
    RenderTarget* target = (RenderTarget*)yf->renderTarget;
    uint32_t      fi = yf->frameIndex % gFrames;
    const uint32_t dw = yf->width, dh = yf->height; // the frame's (display) size

    // --- The frame contract; an upscaler or TAA first: they pick the jitter, and replace MSAA ---
    RenderFrame& F = gFrame;
    memset(&F, 0, sizeof F);
    F.version = RENDER_FRAME_VERSION;
    F.frameIndex = fi;
    F.framesInFlight = gFrames;
    F.width = dw;
    F.height = dh;
    F.displayWidth = dw;
    F.displayHeight = dh;
    F.frameNumber = yea::host()->frame();
    F.renderer = gRenderer;
    F.cmd = cmd;
    // renderer_upscale may render smaller (F.width x F.height) and upscale to the frame's size in TAA's place; without
    // it everything runs at the frame's size, exactly as before it existed.
    const bool upscaled = renderer_upscale::upscale_prepare((uint64_t)(uintptr_t)&F) && F.upscaleOn && F.resolved && F.width &&
                          F.height && F.width <= dw && F.height <= dh;
    if (!upscaled)
    {
        F.width = dw;
        F.height = dh;
        F.upscaleOn = 0;
        F.mipBias = 0;
        F.jitter[0] = F.jitter[1] = 0;
        F.taaOn = 0;
        F.sharpen = 0;
        F.resolved = F.previous = nullptr;
    }
    // TAA also sees an upscaler's frame: it then frees its history and stands aside.
    bool taa = (renderer_taa::taa_prepare((uint64_t)(uintptr_t)&F) && F.taaOn && F.resolved && !upscaled) || upscaled;
    const uint32_t w = F.width, h = F.height; // the render size: every scene pass runs at it
    // F1's Wireframe or Overdraw (development only; 0 in a release): the scene drawn plainly so they read
    // right. No TAA, AO, contact shadows or light culling (their modules step aside), no prepass, no sky,
    // the HDR target cleared to black, exposure 1 and no tonemapping; shadow and probe caches stay frozen.
    const uint32_t viewMode = yea::viewMode();

    // --- Targets, pipelines, meshes ----------------------------------------------------------
    uint32_t samples = taa || viewMode ? 1 : snapSamples(settings.msaa);
    if (w != gWidth || h != gHeight || dw != gDisplayWidth || dh != gDisplayHeight || samples != gSamples || taa != gTargetMotion)
    {
        waitQueueIdle(gQueue);
        destroyTargets();
        createTargets(w, h, samples, taa, dw, dh);
        if (!gHdr || !gDepth || (taa && (!gMotion || !gGloss || (gRaytracing && !gSurface))))
        {
            // Refused (out of memory, say): nothing drawn this frame, tried again next frame, said once.
            if (!gTargetsFailed)
                yea::host()->log(2, "renderer: the GPU refused its screen targets; nothing is drawn until it takes them");
            gTargetsFailed = true;
            destroyTargets();
            return;
        }
        gTargetsFailed = false;
    }
    if (samples != gPipeSamples || taa != gPipeMotion)
    {
        waitQueueIdle(gQueue);
        destroyScenePipelines();
        createScenePipelines(samples, taa);
        gFrameFormat = TinyImageFormat_UNDEFINED;
    }
    if (target->mFormat != gFrameFormat || (uint32_t)target->mSampleCount != gFrameSamples)
    {
        waitQueueIdle(gQueue);
        gFrameFormat = target->mFormat;
        gFrameSamples = target->mSampleCount;
        createCompositePipeline(gFrameFormat, gFrameSamples, samples);
    }
    uploadMeshes();
    lightImagesUpload(gLightImages);

    // --- Camera -----------------------------------------------------------------------------
    CameraState* cam = cameraState();
    v3           eye = V(cam->eye.x, cam->eye.y, cam->eye.z);
    v3           fwd = normalize(V(cam->target.x, cam->target.y, cam->target.z) - eye);
    // The view keeps world up for its roll at every pitch, right from up x forward: its heading never flips as the
    // camera looks straight down or up (a helper axis switched near vertical turned the image to world x there).
    // Only exactly vertical, where up x forward vanishes, the last frame's right carries the heading over.
    v3    right = cross(V(0, 1, 0), fwd), up;
    float rightLen = length(right);
    if (rightLen > 1e-6f)
        gCamRight = right * (1.0f / rightLen);
    else
        gCamRight = normalize(gCamRight - fwd * dot(gCamRight, fwd));
    right = gCamRight;
    up = cross(fwd, right);
    float nearZ = maxf(settings.near_plane, 0.001f);
    float p11 = 1.0f / tanf(cam->fov * 0.5f * 3.14159265f / 180.0f);
    float p00 = p11 * (float)dh / (float)dw;
    float view[4][4], proj[4][4] = { { p00, 0, 0, 0 }, { 0, p11, 0, 0 }, { 0, 0, 0, nearZ }, { 0, 0, 1, 0 } }, viewProj[4][4];
    lookRows(eye, fwd, right, up, view);
    float viewProjNoJitter[4][4];
    mul(proj, view, viewProjNoJitter);
    proj[0][2] = F.jitter[0] * 2.0f / (float)w; // sub-pixel offset in NDC (pixel y runs down)
    proj[1][2] = -F.jitter[1] * 2.0f / (float)h;
    mul(proj, view, viewProj);
    bool havePrev = gPrevViewProjFrame + 1 == F.frameNumber;
    mx::Plane all[6], frustum[5];
    clipPlanes(viewProj, all);
    frustum[0] = all[0];
    frustum[1] = all[1];
    frustum[2] = all[2];
    frustum[3] = all[3];
    frustum[4] = all[5];

    // --- Instances: GPU rows, visibility, casters, sorted by mesh -----------------------------
    RenderGpuInstance* gpuInst = gInstCpu;
    uint32_t             meshCount = Meshes.size() < 1024 ? Meshes.size() : 1024;
    uint32_t             count = 0, casterTotal = 0;
    float                lo[3] = { 1e30f, 1e30f, 1e30f }, hi[3] = { -1e30f, -1e30f, -1e30f };
    memset(gMeshCount, 0, sizeof gMeshCount);
    uint32_t keyCount = meshCount * 2;
    uint32_t casterCount[1024], drawCount[2048];
    memset(casterCount, 0, sizeof casterCount);
    memset(drawCount, 0, sizeof drawCount);
    // Mesh versions: display pixels per world unit at view depth 1, and the error allowed on screen (0: always the finest).
    const float lodFocal = 0.5f * (float)dh * p11;
    const float lodError = settings.lod_error;
    const uint32_t lodRows = MeshLods.size() < meshCount ? MeshLods.size() : meshCount;
    for (uint32_t row = 0; row < Instances.size(); ++row)
    {
        const Instance& in = Instances[row];
        if (!in.alive || in.mesh >= meshCount || !gPosBuf || Meshes[in.mesh].firstIndex + Meshes[in.mesh].indexCount > gUploadedIndices)
            continue;
        RenderGpuInstance& g = gpuInst[count];
        quatRows(in.rot, in.scale, in.pos, g.row);
        if (taa)
        {
            // Where it was last frame (itself when it is new, or its row was reused).
            PrevRows& pr = gPrevRows[row];
            bool      same = pr.frame + 1 == F.frameNumber && pr.gen == in.gen;
            memcpy(gPrevCpu[count], same ? pr.row : g.row, sizeof g.row);
            memcpy(pr.row, g.row, sizeof g.row);
            pr.frame = F.frameNumber;
            pr.gen = in.gen;
        }
        g.material = in.material;
        g.flags = 0;
        const Mesh&   mesh = Meshes[in.mesh];
        InstanceInfo& info = gInfo[count];
        v3            c = V(mesh.center.x, mesh.center.y, mesh.center.z);
        info.sphere[0] = g.row[0][0] * c.x + g.row[0][1] * c.y + g.row[0][2] * c.z + g.row[0][3];
        info.sphere[1] = g.row[1][0] * c.x + g.row[1][1] * c.y + g.row[1][2] * c.z + g.row[1][3];
        info.sphere[2] = g.row[2][0] * c.x + g.row[2][1] * c.y + g.row[2][2] * c.z + g.row[2][3];
        float scale = maxf(fabsf(in.scale.x), maxf(fabsf(in.scale.y), fabsf(in.scale.z)));
        info.sphere[3] = mesh.radius * scale;
        info.moved = in.moved;
        v3 sc = V(info.sphere[0], info.sphere[1], info.sphere[2]);
        bool twoSided = in.material < Materials.size() && Materials[in.material].doubleSided;
        bool visible = sphereInside(frustum, 5, sc, info.sphere[3]);
        info.flags = (visible ? 1u : 0u) | (in.shadow ? 2u : 0u) | (twoSided ? 4u : 0u);
        // The simplest version whose error stays under lod_error display pixels where the instance is nearest the
        // camera: at its nearest view depth, stretched by how far off-axis it is (a flat projection widens things
        // towards the screen's edges). Off screen it is seen only in reflections, which are at least as far away, and
        // shadows, which pick their own (renderer_shadows): its nearest distance then.
        uint32_t lod = in.mesh;
        if (lodError > 0 && in.mesh < lodRows && MeshLods[in.mesh].next < lodRows)
        {
            float d = length(sc - eye), r = info.sphere[3];
            float z = view[2][0] * sc.x + view[2][1] * sc.y + view[2][2] * sc.z + view[2][3];
            float perUnit = visible ? lodFocal * minf(d / maxf(z, nearZ), 2.0f) / maxf(z - r, nearZ) : lodFocal / maxf(d - r, nearZ);
            float quality = row < InstanceLods.size() && InstanceLods[row].quality > 0 ? InstanceLods[row].quality : 1.0f;
            float allowed = lodError / (quality * perUnit * maxf(scale, 1e-6f)); // object units
            for (uint32_t next = MeshLods[lod].next; next < lodRows && MeshLods[next].error <= allowed &&
                                                     Meshes[next].firstIndex + Meshes[next].indexCount <= gUploadedIndices;
                 next = MeshLods[next].next)
                lod = next;
        }
        if (gLodMesh[row] != lod)
        {
            gLodMesh[row] = lod;
            gLodChanged[row] = F.frameNumber;
        }
        info.mesh = lod;
        info.row = row;
        uint32_t key = lod * 2 + (twoSided ? 1 : 0);
        drawCount[key]++;
        if (info.flags & 1)
            gMeshCount[key]++;
        if (info.flags & 2)
        {
            casterCount[lod]++;
            casterTotal++;
            const float s[3] = { sc.x, sc.y, sc.z };
            for (int k = 0; k < 3; ++k)
            {
                lo[k] = minf(lo[k], s[k] - info.sphere[3]);
                hi[k] = maxf(hi[k], s[k] + info.sphere[3]);
            }
        }
        count++;
    }
    uint32_t start = 0, casterStart[1024], casterCursor[1024], drawCursor[2048];
    for (uint32_t k = 0; k < keyCount; ++k)
    {
        gMeshStart[k] = gMeshCursor[k] = start;
        start += gMeshCount[k];
    }
    start = 0;
    for (uint32_t m = 0; m < meshCount; ++m)
    {
        casterStart[m] = casterCursor[m] = start;
        start += casterCount[m];
    }
    start = 0;
    for (uint32_t k = 0; k < keyCount; ++k)
    {
        drawCursor[k] = start;
        start += drawCount[k];
    }
    uint32_t* stream = gStream;
    for (uint32_t i = 0; i < count; ++i)
    {
        const InstanceInfo& info = gInfo[i];
        uint32_t            key = info.mesh * 2 + ((info.flags >> 2) & 1);
        if (info.flags & 1)
            stream[gMeshCursor[key]++] = i;
        {
            const Mesh&     mesh = Meshes[info.mesh];
            RenderCaster& c = gDrawables[drawCursor[key]++];
            c.instance = i;
            c.firstIndex = mesh.firstIndex;
            c.indexCount = mesh.indexCount;
            c.vertexOffset = (int32_t)mesh.firstVertex;
            memcpy(c.sphere, info.sphere, sizeof c.sphere);
            c.moved = info.moved;
            c.twoSided = (info.flags >> 2) & 1;
            c.mesh = info.mesh;
            c.vertexCount = mesh.vertexCount;
            c.castShadow = (info.flags >> 1) & 1;
            c.generation = gMeshGen[info.mesh];
        }
        if (info.flags & 2)
        {
            const Mesh&     mesh = Meshes[info.mesh];
            RenderCaster& c = gCasters[casterCursor[info.mesh]++];
            c.instance = i;
            c.firstIndex = mesh.firstIndex;
            c.indexCount = mesh.indexCount;
            c.vertexOffset = (int32_t)mesh.firstVertex;
            memcpy(c.sphere, info.sphere, sizeof c.sphere);
            c.moved = info.moved > gLodChanged[info.row] ? info.moved : gLodChanged[info.row]; // a new version redraws cached maps
            c.mesh = info.mesh;
            c.vertexCount = mesh.vertexCount;
            c.castShadow = 1;
            c.generation = gMeshGen[info.mesh];
        }
    }
    (void)casterStart;

    // --- Lights -----------------------------------------------------------------------------
    GpuFrame* gf = &gFrameCpu;
    memset(gf, 0, sizeof *gf);
    uint32_t dirCount = 0, localCount = 0, outsideCount = 0;
    bool     sunShadow = false;
    for (uint32_t row = 0; row < Lights.size(); ++row)
    {
        const Light& l = Lights[row];
        if (!l.alive || l.intensity <= 0)
            continue;
        v3 dir = normalize(V(l.dir.x, l.dir.y, l.dir.z));
        if (l.type == 0)
        {
            if (dirCount >= 4)
                continue;
            if (dirCount == 0)
                sunShadow = l.shadow != 0;
            gf->sunDir[dirCount][0] = -dir.x;
            gf->sunDir[dirCount][1] = -dir.y;
            gf->sunDir[dirCount][2] = -dir.z;
            gf->sunDir[dirCount][3] = 1;
            gf->sunColor[dirCount][0] = l.color.x * l.intensity;
            gf->sunColor[dirCount][1] = l.color.y * l.intensity;
            gf->sunColor[dirCount][2] = l.color.z * l.intensity;
            dirCount++;
            continue;
        }
        float          range = maxf(l.range, 0.01f);
        RenderLight& o = gLocal[localCount];
        o.pos[0] = l.pos.x;
        o.pos[1] = l.pos.y;
        o.pos[2] = l.pos.z;
        o.range = range;
        o.dir[0] = dir.x;
        o.dir[1] = dir.y;
        o.dir[2] = dir.z;
        o.type = l.type >= 3 ? 1 : l.type; // area lights are point lights to every other module
        o.cosOuter = cosf(l.outer * 3.14159265f / 180.0f);
        v3    p = V(l.pos.x, l.pos.y, l.pos.z);
        v3    c = p;
        float r = range;
        if (l.type == 2)
        {
            // The tightest sphere around the cone.
            float co = o.cosOuter;
            if (co > 0.70710678f)
            {
                r = range / (2 * co);
                c = p + dir * r;
            }
            else
            {
                c = p + dir * (range * co);
                r = range * sqrtf(maxf(1 - co * co, 0));
            }
        }
        else if (l.type == 3 || l.type == 4)
        {
            // One-sided: the half ball in front, inside the sphere through its rim, centred 3/8 of the range out.
            c = p + dir * (range * 0.375f);
            r = range * 1.068f;
        }
        if (!sphereInside(frustum, 5, c, r))
        {
            gOutside[outsideCount++] = row;
            continue;
        }
        o.sphere[0] = c.x;
        o.sphere[1] = c.y;
        o.sphere[2] = c.z;
        o.sphere[3] = r;
        o.viewZ = view[2][0] * c.x + view[2][1] * c.y + view[2][2] * c.z + view[2][3];
        o.intensity = l.intensity * maxf(l.color.x, maxf(l.color.y, l.color.z));
        o.id = row | ((l.gen & 0xFFFu) << 20);
        o.castShadow = l.shadow;
        o.shadow = -1;
        o.traced = -1;
        o.source = row;
        // Its shape for renderer_shadows: one-sided rects and discs, and a radius for its penumbra (an area light's: a
        // disc of its area, as the forward pass takes it).
        o.kind = l.type >= 3 ? l.type : 0;
        {
            float w = maxf(l.size.x, 1e-3f), h = maxf(l.size.y, 1e-3f);
            float area = l.type == 3 ? w * h : l.type == 4 ? 3.14159265f * w * h * 0.25f : l.type == 5 ? 3.14159265f * w * w : w * 2.0f * h;
            o.size = l.type >= 3 ? sqrtf(area / 3.14159265f) : maxf(l.size.x, 0.0f);
        }
        localCount++;
    }
    sortLights(gLocal, localCount);

    // --- The frame contract -----------------------------------------------------------------
    F.sampleCount = samples;
    memcpy(F.view, view, sizeof view);
    F.camPos[0] = eye.x;
    F.camPos[1] = eye.y;
    F.camPos[2] = eye.z;
    F.nearZ = nearZ;
    F.p00 = p00;
    F.p11 = p11;
    F.sunOn = dirCount > 0;
    F.sunShadow = sunShadow;
    F.sunDir[0] = gf->sunDir[0][0];
    F.sunDir[1] = gf->sunDir[0][1];
    F.sunDir[2] = gf->sunDir[0][2];
    // The sun's light above the atmosphere; with the physical sky (renderer_sky), what of it reaches the camera, so
    // every module (shadows' tints, probes, ray tracing, the forward pass) sees the sun as it arrives.
    memcpy(F.sunColor, gf->sunColor[0], sizeof F.sunColor);
    const bool sky = renderer_sky::sky_prepare((uint64_t)(uintptr_t)&F) && F.skyView && F.skyEnv && F.skySh && F.skyAerial;
    if (sky)
        for (int k = 0; k < 3; ++k)
        {
            F.sunColor[k] *= F.sunTransmittance[k];
            gf->sunColor[0][k] = F.sunColor[k];
        }
    F.positions = gPosBuf;
    F.attributes = gAttrBuf;
    F.indices = gIndexBuf;
    F.instances = gInstanceBuf[fi];
    F.instanceRows = gInstCpu;
    F.instanceCount = count;
    F.casters = gCasters;
    F.casterCount = casterTotal;
    F.casterRemoved = sceneState()->removed;
    if (casterTotal)
    {
        F.sceneSphere[0] = (lo[0] + hi[0]) * 0.5f;
        F.sceneSphere[1] = (lo[1] + hi[1]) * 0.5f;
        F.sceneSphere[2] = (lo[2] + hi[2]) * 0.5f;
        F.sceneSphere[3] = 0.5f * length(V(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]));
    }
    F.lights = gLocal;
    F.lightCount = localCount;
    F.depth = gDepth;
    F.lightBuffer = gLightBuf[fi];
    F.lightShapes = gShapeBuf[fi];
    F.taps = 16;
    F.softness = 1.5f;
    F.hdr = gHdr;
    F.motion = gMotion;
    F.gloss = gGloss;
    F.exposure = gExposure;

    bool culled = renderer_culling::cull_prepare((uint64_t)(uintptr_t)&F) && F.tileMasks && F.zbins && F.wordsPerTile;
    if (!culled && F.lightCount > kMaxUnculledLights)
        F.lightCount = kMaxUnculledLights;
    // Ray-traced shadows (renderer_rt): its pick comes first, so renderer_shadows draws nothing for what it traces.
    const bool rtGate = taa && !viewMode && gRaytracing && gSurface && gDepthSurfacePipe[0] && gDepthSurfacePipe[1] && F.previous;
    const bool rtPicked = rtGate && renderer_rt::rt_pick((uint64_t)(uintptr_t)&F);
    if (!rtPicked)
    {
        for (uint32_t i = 0; i < F.lightCount; ++i)
            F.lights[i].traced = -1;
        F.rtSunShadows = F.rtLightCount = 0;
    }
    bool shadows = renderer_shadows::shadows_prepare((uint64_t)(uintptr_t)&F) && F.shadowViews && F.sunAtlas && F.localAtlas;
    bool ao = renderer_ao::ao_prepare((uint64_t)(uintptr_t)&F) && F.aoOn && F.ao;
    // Contact shadows refine the sun's: off with them.
    bool contact = shadows && F.cascadeCount && renderer_contact::contact_prepare((uint64_t)(uintptr_t)&F) && F.contactOn && F.contact;
    const yea_float3 zen = settings.sky_color, hor = settings.horizon_color, gnd = settings.ground_color;
    float            L = settings.sky_luminance;
    F.drawables = gDrawables;
    F.drawableCount = count;
    F.materials = gMaterialBuf[fi];
    F.skyZenith[0] = zen.x * L, F.skyZenith[1] = zen.y * L, F.skyZenith[2] = zen.z * L;
    F.skyHorizon[0] = hor.x * L, F.skyHorizon[1] = hor.y * L, F.skyHorizon[2] = hor.z * L;
    F.skyGround[0] = gnd.x * L, F.skyGround[1] = gnd.y * L, F.skyGround[2] = gnd.z * L;
    F.ambient = settings.ambient;
    bool probes = renderer_probes::probes_prepare((uint64_t)(uintptr_t)&F) && F.probes && F.probeData && F.probeCount;
    // Ray-traced reflections where the GPU has ray queries; screen-space ones otherwise.
    F.surface = gSurface;
    // renderer_rt runs (its prepass target, rt_run) for its reflections, its traced shadows or both.
    const bool rtRun = rtGate && renderer_rt::rt_prepare((uint64_t)(uintptr_t)&F);
    bool       rt = rtRun && F.rtReflection;
    const bool rtShadows = rtRun && rtPicked && F.rtShadows;
    bool ssr = !rt && taa && !viewMode && renderer_ssr::ssr_prepare((uint64_t)(uintptr_t)&F) && F.ssrOn && F.ssrDepth && F.previous;
    bool vrs = taa && !viewMode && renderer_vrs::vrs_prepare((uint64_t)(uintptr_t)&F) && F.rateImage && F.rateCopy;
    // Volumetric fog (renderer_fog): it reads the lights, their bins and shadow views, and the sky, all set by now.
    const bool fog = !viewMode && renderer_fog::fog_prepare((uint64_t)(uintptr_t)&F) && F.fogVolume;

    // GPU lights in sorted order, with the shadow views renderer_shadows picked; then, for reflections
    // (renderer_rt), the lights outside the view, without shadows.
    RenderGpuLight* gl = gLightCpu;
    const bool      lightShadows = shadows && renderer_shadows::light_shadows_on();
    bool            anyShape = false;
    for (uint32_t i = 0; i < F.lightCount + outsideCount; ++i)
    {
        bool         inside = i < F.lightCount;
        uint32_t     row = inside ? gLocal[i].source : gOutside[i - F.lightCount];
        const Light& l = Lights[row];
        RenderGpuLight& g = gl[i];
        v3           dir = normalize(V(l.dir.x, l.dir.y, l.dir.z));
        float        range = maxf(l.range, 0.01f);
        float        cosOuter = cosf(l.outer * 3.14159265f / 180.0f);
        float        cosInner = cosf(l.inner * 3.14159265f / 180.0f);
        float        scale = 1.0f / maxf(cosInner - cosOuter, 1e-3f);
        g.pos[0] = l.pos.x;
        g.pos[1] = l.pos.y;
        g.pos[2] = l.pos.z;
        g.pos[3] = range;
        g.color[0] = l.color.x * l.intensity;
        g.color[1] = l.color.y * l.intensity;
        g.color[2] = l.color.z * l.intensity;
        g.color[3] = l.type >= 3 ? 1.0f : (float)l.type;
        g.dir[0] = dir.x;
        g.dir[1] = dir.y;
        g.dir[2] = dir.z;
        g.dir[3] = scale;
        g.params[0] = -cosOuter * scale;
        g.params[1] = shadows && inside ? (float)gLocal[i].shadow : -1.0f;
        g.params[2] = 1.0f / (range * range);
        // Casts in the scene (renderer_rt: a map, else a ray); 2 + n: traced into rtShadows slice 1 + n.
        g.params[3] = rtShadows && inside && gLocal[i].traced >= 0 ? 2.0f + (float)gLocal[i].traced : lightShadows && l.shadow ? 1.0f : 0.0f;
        // Its shape (frame.h's RenderGpuLightShape).
        RenderGpuLightShape& sh = gShapeCpu[i];
        memset(&sh, 0, sizeof sh);
        gLookCpu[i] = lightImageOf(row);
        if (l.type >= 3)
        {
            v3    t = normalize(V(l.tangent.x, l.tangent.y, l.tangent.z));
            t = normalize(t - dir * dot(t, dir)); // across, square to dir
            if (fabsf(dot(t, dir)) > 0.999f || dot(t, t) < 0.5f)
            {
                v3 up;
                basis(dir, &t, &up);
            }
            v3    b = cross(dir, t);
            float w = maxf(l.size.x, 1e-3f), h = maxf(l.size.y, 1e-3f), area = 1.0f;
            sh.ex[3] = (float)l.type;
            if (l.type <= 4)
            {
                for (int k = 0; k < 3; ++k)
                {
                    sh.ex[k] = (&t.x)[k] * w * 0.5f;
                    sh.ey[k] = (&b.x)[k] * h * 0.5f;
                }
                area = l.type == 3 ? w * h : 3.14159265f * w * h * 0.25f;
            }
            else if (l.type == 5)
                area = 3.14159265f * w * w;
            else
            {
                for (int k = 0; k < 3; ++k)
                    sh.ex[k] = (&dir.x)[k] * w * 0.5f;
                sh.ey[0] = h;
                area = w * 2.0f * h;
            }
            sh.ey[3] = 1.0f / maxf(area, 1e-6f);
            anyShape = true;
        }
        else if (l.size.x > 0)
        {
            sh.ey[3] = l.size.x;
            anyShape = true;
        }
        if (inside)
            memcpy(g.sphere, gLocal[i].sphere, sizeof g.sphere);
        else
        {
            g.sphere[0] = l.pos.x, g.sphere[1] = l.pos.y, g.sphere[2] = l.pos.z, g.sphere[3] = range;
        }
    }
    F.allLightCount = F.lightCount + outsideCount;

    // Materials.
    GpuMaterial* gm = gMaterialCpu;
    for (uint32_t i = 0; i < Materials.size(); ++i)
    {
        const Material& m = Materials[i];
        gm[i] = GpuMaterial{};
        gm[i].albedoRough[0] = m.albedo.x;
        gm[i].albedoRough[1] = m.albedo.y;
        gm[i].albedoRough[2] = m.albedo.z;
        gm[i].albedoRough[3] = m.roughness;
        gm[i].params[0] = m.metallic;
        gm[i].params[1] = 0.16f * m.reflectance * m.reflectance;
        gm[i].params[2] = m.reflections ? 1.0f : 0.0f;
        gm[i].emissive[0] = m.emissive.x;
        gm[i].emissive[1] = m.emissive.y;
        gm[i].emissive[2] = m.emissive.z;
    }

    // Frame constants.
    memcpy(gf->viewProj, viewProj, sizeof viewProj);
    memcpy(gf->viewProjNoJitter, viewProjNoJitter, sizeof viewProjNoJitter);
    memcpy(gf->prevViewProj, havePrev ? gPrevViewProj : viewProjNoJitter, sizeof viewProjNoJitter);
    memcpy(gPrevViewProj, viewProjNoJitter, sizeof viewProjNoJitter);
    gPrevViewProjFrame = F.frameNumber;
    memcpy(gf->view, view, sizeof view);
    gf->camPos[0] = eye.x;
    gf->camPos[1] = eye.y;
    gf->camPos[2] = eye.z;
    gf->camPos[3] = nearZ;
    gf->screen[0] = (float)w;
    gf->screen[1] = (float)h;
    gf->screen[2] = 1.0f / (float)w;
    gf->screen[3] = 1.0f / (float)h;
    gf->proj[0] = p00;
    gf->proj[1] = p11;
    float            skyZ[4] = { zen.x * L, zen.y * L, zen.z * L, 0 }, skyH[4] = { hor.x * L, hor.y * L, hor.z * L, 0 },
          skyG[4] = { gnd.x * L, gnd.y * L, gnd.z * L, settings.ambient };
    memcpy(gf->skyZenith, skyZ, sizeof skyZ);
    memcpy(gf->skyHorizon, skyH, sizeof skyH);
    memcpy(gf->skyGround, skyG, sizeof skyG);
    gf->counts[0] = F.lightCount;
    gf->counts[1] = dirCount;
    gf->counts[2] = (culled ? kFlagCulled : 0) | (culled && F.heatmap ? kFlagHeatmap : 0) |
                    (shadows && (F.showCascades & 1) ? kFlagCascades : 0) | (shadows ? kFlagShadows : 0) | (ao ? kFlagAo : 0) |
                    (ao && F.aoDebug ? kFlagAoDebug : 0) | (contact ? kFlagContact : 0) |
                    (contact && F.contactDebug ? kFlagContactDebug : 0) |
                    ((viewMode & YEA_VIEW_WIREFRAME) ? kFlagUnlit : 0) | (ssr ? kFlagSsr : 0) | (ssr && F.ssrDebug ? kFlagSsrDebug : 0) |
                    (shadows && F.receiverPlane ? kFlagReceiverPlane : 0) | (rt ? kFlagRt : 0) | (rt && F.rtDebug ? kFlagRtDebug : 0) |
                    (rtShadows ? kFlagRtShadows : 0) | (rtShadows && F.rtSunShadows ? kFlagRtSun : 0) | (anyShape ? kFlagArea : 0) | (sky ? kFlagSky : 0) | (fog ? kFlagFog : 0);
    if (ssr)
    {
        gf->ssr[0] = F.ssrMaxRoughness;
        gf->ssr[1] = F.ssrThickness;
        gf->ssr[2] = F.ssrMaxDistance;
        gf->ssr[3] = F.ssrEdgeFade;
        gf->ssr2[0] = F.ssrIntensity;
        gf->ssr2[1] = F.ssrScale;
        gf->ssr2[2] = (float)F.ssrSteps;
        gf->ssr2[3] = (float)((F.frameNumber % 64u) * 0.6180339887); // rotates the march's noise; TAA averages it
    }
    gf->rt[0] = rt ? F.rtMaxRoughness : 0.0f;
    if (sky)
    {
        memcpy(gf->atmosphere, F.skyRadii, sizeof F.skyRadii);
        gf->atmosphere2[0] = F.skyAerialDepth;
        gf->atmosphere2[1] = (float)F.skyEnvMips;
    }
    if (fog)
    {
        memcpy(gf->fog, F.fogParams, sizeof F.fogParams);
        gf->fog2[0] = F.fogUv[0];
        gf->fog2[1] = F.fogUv[1];
    }
    if (probes)
    {
        gf->probeInfo[0] = F.probeCount;
        gf->probeInfo[1] = F.probeMips;
        gf->probeInfo[2] = F.probeParallax;
        gf->probeParams[0] = F.probeIntensity;
        gf->probeParams[1] = F.probeBlend;
    }
    if (ao)
    {
        gf->aoParams[0] = F.aoSpecular;
        gf->aoParams[1] = F.aoMultiBounce ? 1.0f : 0.0f;
        gf->aoParams[2] = F.aoScale == 2 ? 2.0f : 1.0f; // half size: the forward pass upsamples it
    }
    gf->counts[3] = F.taps ? F.taps : 16;
    if (gTapsMade != gf->counts[3])
    {
        // Each pixel turns the disk by its own noise angle: one rotation per pixel instead of a sine, cosine
        // and square root per tap.
        gTapsMade = gf->counts[3];
        for (uint32_t i = 0; i < gTapsMade && i < 32; ++i)
        {
            float r = sqrtf(((float)i + 0.5f) / (float)gTapsMade), t = (float)i * 2.39996323f;
            gf->taps[i][0] = r * cosf(t);
            gf->taps[i][1] = r * sinf(t);
        }
    }
    if (culled)
    {
        gf->tiles[0] = F.tileSize;
        gf->tiles[1] = F.tilesX;
        gf->tiles[2] = F.wordsPerTile;
        gf->tiles[3] = F.binCount;
        gf->misc[0] = F.binScale;
    }
    gf->misc[1] = settings.specular_aa;
    gf->misc[2] = settings.sun_disk > 0 ? cosf(settings.sun_disk * 0.5f * 3.14159265f / 180.0f) : 1.0f;
    if (shadows)
    {
        for (uint32_t c = 0; c < 4; ++c)
            gf->cascadeEnd[c] = c < F.cascadeCount ? F.cascadeEnd[c] : 1e30f;
        gf->shadow[0] = (float)F.cascadeCount;
        gf->shadow[1] = F.softness;
        gf->shadow[2] = F.lightSize;
        gf->shadow[3] = F.normalBias;
        gf->shadow2[0] = F.sunTexel;
        gf->shadow2[1] = F.localTexel;
        gf->shadow2[3] = F.localLightSize;
    }
    // With TAA the shadow filters' noise turns each frame and cascade transitions are dithered: TAA averages both.
    gf->shadow2[2] = taa && !viewMode ? (float)(F.frameNumber % 64u) : -1.0f;

    // One copy per buffer into the mapped memory.
    uint32_t visibleTotal = 0;
    for (uint32_t k = 0; k < keyCount; ++k)
        visibleTotal += gMeshCount[k];
    memcpy(gInstanceBuf[fi]->pCpuMappedAddress, gInstCpu, (size_t)count * sizeof(RenderGpuInstance));
    memcpy(gStreamBuf[fi]->pCpuMappedAddress, gStream, (size_t)visibleTotal * 4);
    memcpy(gLightBuf[fi]->pCpuMappedAddress, gLightCpu, (size_t)F.allLightCount * sizeof(RenderGpuLight));
    memcpy(gShapeBuf[fi]->pCpuMappedAddress, gShapeCpu, (size_t)F.allLightCount * sizeof(RenderGpuLightShape));
    memcpy(gLookBuf[fi]->pCpuMappedAddress, gLookCpu, (size_t)F.allLightCount * 4);
    memcpy(gMaterialBuf[fi]->pCpuMappedAddress, gMaterialCpu, (size_t)Materials.size() * sizeof(GpuMaterial));
    memcpy(gFrameCb[fi]->pCpuMappedAddress, &gFrameCpu, sizeof gFrameCpu);
    if (taa)
        memcpy(gPrevBuf[fi]->pCpuMappedAddress, gPrevCpu, (size_t)count * sizeof gPrevCpu[0]);

    CompositeData composite = {};
    // Tony McMapface's table is read when it is picked and let go when it isn't; without it, Khronos PBR Neutral.
    bool tony = settings.tonemapper == Tonemapper::Tony_McMapface;
    if (tony && !gTonyTried)
    {
        gTonyLut = loadTonyLut();
        gTonyTried = true;
    }
    else if (!tony && gTonyTried)
    {
        if (gTonyLut)
        {
            waitQueueIdle(gQueue); // frames in flight may still read it
            removeResource(gTonyLut);
            gTonyLut = nullptr;
        }
        gTonyTried = false;
    }
    composite.mode[0] = (uint32_t)(settings.tonemapper < 0 || (tony && !gTonyLut) ? 0 : settings.tonemapper);
    composite.mode[1] = TinyImageFormat_IsSRGB(gFrameFormat) ? 0 : 1;
    composite.mode[2] = samples;
    composite.mode[3] = dw | (dh << 16);
    composite.params[0] = taa ? F.sharpen : 0.0f;
    // Show AO and Show Contact write their values as they are: shown at exposure 1, untonemapped.
    const bool rawDebug = (ao && F.aoDebug) || (contact && F.contactDebug);
    if (rawDebug && !viewMode)
    {
        composite.mode[0] = 4; // the tonemapper choice's None: no tonemapping
        composite.params[3] = 1.0f;
    }
    if (viewMode)
    {
        composite.mode[0] = 4; // the tonemapper choice's None: no tonemapping
        if (viewMode & YEA_VIEW_OVERDRAW)
            composite.mode[1] = 0; // Overdraw's counts go out as they are, as F1 writes its own
        composite.params[3] = 1.0f; // exposure 1, no dither
    }
    // The light shadow atlas in the top right corner (renderer_shadows' show_atlas).
    composite.rates[0] = vrs && F.vrsDebug ? 1u : 0u;
    composite.rates[1] = F.rateTile ? F.rateTile : 16u;
    // With TAA's sharpening, the tonemap pass tonemaps each pixel once and sharpens, instead of the composite
    // tonemapping every pixel five times (MSAA and no sharpening tonemap once per sample there anyway).
    const bool ldr = taa && composite.params[0] > 0.0f && gLdr && gTonemapPipe;
    composite.rates[2] = ldr ? 1u : 0u;
    composite.params[1] = shadows && (F.showCascades & 2) ? floorf(mx::minf(dh * 0.5f, dw * 0.4f)) : 0.0f;
    composite.params[2] = shadows && F.localTexel > 0 ? 1.0f / F.localTexel : 0.0f;
    memcpy(gCompositeCb[fi]->pCpuMappedAddress, &composite, sizeof composite);
    // Post-processing (renderer_post) on the resolved HDR image, before the tonemapper: the image it returns is what
    // the tonemap pass and the composite read. Not under the debug views, which show their values as they are.
    Texture* post = !rawDebug && !(culled && F.heatmap) && (taa || samples == 1)
                        ? (Texture*)(uintptr_t)renderer_post::post_prepare((uint64_t)(uintptr_t)&F)
                        : nullptr;
    ExposureData* ed = (ExposureData*)gExposureCb[fi]->pCpuMappedAddress;
    float         minLog = -8.0f, range = 26.0f;
    float         dt = (float)(yea::host()->time());
    static float  lastTime = 0;
    float         step = lastTime > 0 ? mx::clampf(dt - lastTime, 0, 0.25f) : 0;
    lastTime = dt;
    ed->range[0] = minLog;
    ed->range[1] = 1.0f / range;
    ed->range[2] = range;
    ed->adapt[0] = 1.0f - expf(-step * settings.adapt_speed);
    ed->adapt[1] = settings.exposure_ev;
    ed->adapt[2] = settings.manual_ev100;
    ed->adapt[3] = settings.auto_exposure ? 0.0f : 1.0f;
    ed->size[0] = w;
    ed->size[1] = h;
    ed->dark[0] = 3.0f; // about a scene lit as the sun sets leaves it, in this renderer's units (the sun ~500 lux)
    ed->dark[1] = mx::clampf(settings.dark_adaptation, 0, 1);

    // Descriptors for this frame.
    {
        Texture* sun = shadows ? F.sunAtlas->pTexture : gDummyDepth->pTexture;
        Texture* local = shadows ? F.localAtlas->pTexture : gDummyDepth->pTexture;
        Buffer*  masks = culled ? F.tileMasks : gDummyU32;
        Buffer*  bins = culled ? F.zbins : gDummyU32;
        Buffer*  views = shadows ? F.shadowViews : gDummyViews;
        DescriptorData p[26] = {};
        Texture*       ssrDepth = ssr ? F.ssrDepth : gDummyDepth->pTexture;
        Texture*       previous = ssr ? F.previous->pTexture : gDummyColor->pTexture;
        Texture*       probeTex = probes ? F.probes->pTexture : gDummyArray->pTexture;
        Buffer*        probeData = probes ? F.probeData : gDummyProbes;
        Texture*       contactTex = contact ? F.contact : gDummyDepth->pTexture;
        Texture*       aoTex = ao ? F.ao : gDummyDepth->pTexture;
        Texture*       rtTex = rt ? F.rtReflection : gDummyColor->pTexture;
        Texture*       rtShadowTex = rtShadows ? F.rtShadows : gDummyArray->pTexture;
        p[0].mIndex = SRT_RES_IDX(SceneSrt, PerFrame, gFrame);
        p[0].ppBuffers = &gFrameCb[fi];
        p[1].mIndex = SRT_RES_IDX(SceneSrt, PerFrame, gInstances);
        p[1].ppBuffers = &gInstanceBuf[fi];
        p[2].mIndex = SRT_RES_IDX(SceneSrt, PerFrame, gMaterials);
        p[2].ppBuffers = &gMaterialBuf[fi];
        p[3].mIndex = SRT_RES_IDX(SceneSrt, PerFrame, gLights);
        p[3].ppBuffers = &gLightBuf[fi];
        p[4].mIndex = SRT_RES_IDX(SceneSrt, PerFrame, gTileMasks);
        p[4].ppBuffers = &masks;
        p[5].mIndex = SRT_RES_IDX(SceneSrt, PerFrame, gZBins);
        p[5].ppBuffers = &bins;
        p[6].mIndex = SRT_RES_IDX(SceneSrt, PerFrame, gShadowViews);
        p[6].ppBuffers = &views;
        p[7].mIndex = SRT_RES_IDX(SceneSrt, PerFrame, gExposure);
        p[7].ppBuffers = &gExposure;
        p[8].mIndex = SRT_RES_IDX(SceneSrt, PerFrame, gSunAtlas);
        p[8].ppTextures = &sun;
        p[9].mIndex = SRT_RES_IDX(SceneSrt, PerFrame, gLocalAtlas);
        p[9].ppTextures = &local;
        p[10].mIndex = SRT_RES_IDX(SceneSrt, PerFrame, gPrevInstances);
        p[10].ppBuffers = &gPrevBuf[fi];
        p[11].mIndex = SRT_RES_IDX(SceneSrt, PerFrame, gAo);
        p[11].ppTextures = &aoTex;
        p[12].mIndex = SRT_RES_IDX(SceneSrt, PerFrame, gContactShadow);
        p[12].ppTextures = &contactTex;
        p[13].mIndex = SRT_RES_IDX(SceneSrt, PerFrame, gProbes);
        p[13].ppTextures = &probeTex;
        p[14].mIndex = SRT_RES_IDX(SceneSrt, PerFrame, gProbeData);
        p[14].ppBuffers = &probeData;
        p[15].mIndex = SRT_RES_IDX(SceneSrt, PerFrame, gSsrDepth);
        p[15].ppTextures = &ssrDepth;
        p[16].mIndex = SRT_RES_IDX(SceneSrt, PerFrame, gPrevColor);
        p[16].ppTextures = &previous;
        p[17].mIndex = SRT_RES_IDX(SceneSrt, PerFrame, gRtReflection);
        p[17].ppTextures = &rtTex;
        p[18].mIndex = SRT_RES_IDX(SceneSrt, PerFrame, gRtShadows);
        p[18].ppTextures = &rtShadowTex;
        p[19].mIndex = SRT_RES_IDX(SceneSrt, PerFrame, gLightShapes);
        p[19].ppBuffers = &gShapeBuf[fi];
        p[20].mIndex = SRT_RES_IDX(SceneSrt, PerFrame, gLightLooks);
        p[20].ppBuffers = &gLookBuf[fi];
        Texture* skyView = sky ? F.skyView : gDummyColor->pTexture;
        Texture* skyEnv = sky ? F.skyEnv : gDummyColor->pTexture;
        Buffer*  skySh = sky ? F.skySh : gDummySh;
        Texture* skyAerial = sky ? F.skyAerial : gDummyVolume;
        p[21].mIndex = SRT_RES_IDX(SceneSrt, PerFrame, gSkyView);
        p[21].ppTextures = &skyView;
        p[22].mIndex = SRT_RES_IDX(SceneSrt, PerFrame, gSkyEnv);
        p[22].ppTextures = &skyEnv;
        p[23].mIndex = SRT_RES_IDX(SceneSrt, PerFrame, gSkySh);
        p[23].ppBuffers = &skySh;
        p[24].mIndex = SRT_RES_IDX(SceneSrt, PerFrame, gSkyAerial);
        p[24].ppTextures = &skyAerial;
        Texture* fogVolume = fog ? F.fogVolume : gDummyVolume;
        p[25].mIndex = SRT_RES_IDX(SceneSrt, PerFrame, gFogVolume);
        p[25].ppTextures = &fogVolume;
        updateDescriptorSet(gRenderer, fi, gSceneFrame, 26, p);

        DescriptorData c[7] = {};
        c[0].mIndex = SRT_RES_IDX(CompositeSrt, PerFrame, gComposite);
        c[0].ppBuffers = &gCompositeCb[fi];
        c[1].mIndex = SRT_RES_IDX(CompositeSrt, PerFrame, gHdr);
        c[1].ppTextures = post ? &post : taa ? &F.resolved->pTexture : &gHdr->pTexture;
        c[2].mIndex = SRT_RES_IDX(CompositeSrt, PerFrame, gExposure);
        c[2].ppBuffers = &gExposure;
        c[3].mIndex = SRT_RES_IDX(CompositeSrt, PerFrame, gShadowAtlas);
        c[3].ppTextures = &local;
        Texture* rates = vrs ? F.rateCopy : gDummyRates;
        c[4].mIndex = SRT_RES_IDX(CompositeSrt, PerFrame, gRates);
        c[4].ppTextures = &rates;
        c[5].mIndex = SRT_RES_IDX(CompositeSrt, PerFrame, gTony);
        Texture* tonyTex = gTonyLut ? gTonyLut : gDummyColor->pTexture;
        c[5].ppTextures = &tonyTex;
        Texture* ldrTex = ldr ? gLdr : gDummyRates;
        c[6].mIndex = SRT_RES_IDX(CompositeSrt, PerFrame, gLdr);
        c[6].ppTextures = &ldrTex;
        updateDescriptorSet(gRenderer, fi, gCompositeSet, 7, c);
        if (ldr)
        {
            DescriptorData t[5] = {};
            t[0].mIndex = SRT_RES_IDX(TonemapSrt, PerFrame, gComposite);
            t[0].ppBuffers = &gCompositeCb[fi];
            t[1].mIndex = SRT_RES_IDX(TonemapSrt, PerFrame, gHdr);
            t[1].ppTextures = post ? &post : &F.resolved->pTexture;
            t[2].mIndex = SRT_RES_IDX(TonemapSrt, PerFrame, gExposure);
            t[2].ppBuffers = &gExposure;
            t[3].mIndex = SRT_RES_IDX(TonemapSrt, PerFrame, gTony);
            t[3].ppTextures = &tonyTex;
            t[4].mIndex = SRT_RES_IDX(TonemapSrt, PerFrame, gLdrOut);
            t[4].ppTextures = &gLdr;
            updateDescriptorSet(gRenderer, fi, gTonemapSet, 5, t);
        }

        DescriptorData e[4] = {};
        e[0].mIndex = SRT_RES_IDX(ExposureSrt, PerFrame, gExposureData);
        e[0].ppBuffers = &gExposureCb[fi];
        e[1].mIndex = SRT_RES_IDX(ExposureSrt, PerFrame, gHdr);
        e[1].ppTextures = &gHdr->pTexture;
        e[2].mIndex = SRT_RES_IDX(ExposureSrt, PerFrame, gHistogram);
        e[2].ppBuffers = &gHistogram;
        e[3].mIndex = SRT_RES_IDX(ExposureSrt, PerFrame, gExposureRW);
        e[3].ppBuffers = &gExposure;
        updateDescriptorSet(gRenderer, fi, gExposureSet, 4, e);
    }

    // --- Record -----------------------------------------------------------------------------
    {
        char scene[192];
        int  n = snprintf(scene, sizeof scene, "%ux%u x%u, %u lights, %u instances drawn, %u casters", w, h, samples, F.lightCount, count, casterTotal);
        if (upscaled && n > 0 && n < (int)sizeof scene)
            snprintf(scene + n, sizeof scene - n, ", upscaled to %ux%u", dw, dh);
        timeFrame(cmd, fi, settings.gpu_timings, scene);
    }
    const bool timed = settings.gpu_timings;
    if (timed)
        timeBegin(cmd, fi, kTimeTotal);
    if (gExposureState != RESOURCE_STATE_SHADER_RESOURCE)
    {
        barrierBuf(cmd, gExposure, gExposureState, RESOURCE_STATE_SHADER_RESOURCE);
        gExposureState = RESOURCE_STATE_SHADER_RESOURCE;
    }

    if (sky)
    {
        if (timed)
            timeBegin(cmd, fi, kTimeSky);
        renderer_sky::sky_run((uint64_t)(uintptr_t)&F);
        timeEnd(cmd, fi, kTimeSky);
    }
    if (shadows)
    {
        if (timed)
            timeBegin(cmd, fi, kTimeShadows);
        renderer_shadows::shadows_render((uint64_t)(uintptr_t)&F);
        timeEnd(cmd, fi, kTimeShadows);
    }
    if (probes)
    {
        if (timed)
            timeBegin(cmd, fi, kTimeProbes);
        renderer_probes::probes_render((uint64_t)(uintptr_t)&F);
        timeEnd(cmd, fi, kTimeProbes);
    }
    if (fog)
    {
        if (timed)
            timeBegin(cmd, fi, kTimeFog);
        renderer_fog::fog_run((uint64_t)(uintptr_t)&F);
        timeEnd(cmd, fi, kTimeFog);
    }

    bool geometry = gPosBuf && gIndexBuf && count > 0;
    if (gDepthState != RESOURCE_STATE_DEPTH_WRITE)
        barrierRt(cmd, gDepth, gDepthState, RESOURCE_STATE_DEPTH_WRITE);
    BindRenderTargetsDesc bind = {};
    if (timed && !viewMode)
        timeBegin(cmd, fi, kTimePrepass);
    if (!viewMode)
    {
    cmdBeginDebugMarker(cmd, 0.4f, 0.6f, 1.0f, "Renderer depth prepass");
    bind.mDepthStencil = { gDepth, LOAD_ACTION_CLEAR };
    bind.mDepthStencil.mStoreAction = STORE_ACTION_STORE;
    // With renderer_vrs, also each pixel's reflection roughness, so glossy surfaces keep full rate.
    // With renderer_rt, each pixel's reflection roughness and normal instead (renderer_vrs reads its roughness too), and the
    // motion vectors, final for rt_run (the forward pass writes the same again, as its own shading rate has them).
    RenderTarget* second = rtRun ? gSurface : vrs && gGloss && gDepthGlossPipe[0] && gDepthGlossPipe[1] ? gGloss : nullptr;
    if (second)
    {
        barrierRt(cmd, second, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_RENDER_TARGET);
        bind.mRenderTargetCount = 1;
        bind.mRenderTargets[0] = { second, LOAD_ACTION_CLEAR };
    }
    if (rtRun)
    {
        barrierRt(cmd, gMotion, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_RENDER_TARGET);
        bind.mRenderTargetCount = 2;
        bind.mRenderTargets[1] = { gMotion, LOAD_ACTION_CLEAR }; // the sky's 0 until the sky pass
    }
    cmdBindRenderTargets(cmd, &bind);
    cmdSetViewport(cmd, 0, 0, (float)w, (float)h, 0, 1);
    cmdSetScissor(cmd, 0, 0, w, h);
    if (geometry)
    {
        drawVisible(cmd, fi, rtRun ? gDepthSurfacePipe : second ? gDepthGlossPipe : gDepthPipe);
    }
    cmdBindRenderTargets(cmd, nullptr);
    if (second)
        barrierRt(cmd, second, RESOURCE_STATE_RENDER_TARGET, RESOURCE_STATE_SHADER_RESOURCE);
    if (rtRun)
        barrierRt(cmd, gMotion, RESOURCE_STATE_RENDER_TARGET, RESOURCE_STATE_SHADER_RESOURCE);
    F.gloss = second;
    cmdEndDebugMarker(cmd);
    }

    timeEnd(cmd, fi, kTimePrepass);
    if (culled || ao || contact || ssr || vrs || rtRun)
    {
        barrierRt(cmd, gDepth, RESOURCE_STATE_DEPTH_WRITE, RESOURCE_STATE_SHADER_RESOURCE);
        const uint32_t which[6] = { kTimeCulling, kTimeAo, kTimeContact, kTimeSsr, kTimeVrs, kTimeRt };
        const bool     run[6] = { culled, ao, contact, ssr, vrs, rtRun };
        for (uint32_t k = 0; k < 6; ++k)
        {
            if (!run[k])
                continue;
            if (timed)
                timeBegin(cmd, fi, which[k]);
            if (k == 0)
                renderer_culling::cull_run((uint64_t)(uintptr_t)&F);
            else if (k == 1)
                renderer_ao::ao_run((uint64_t)(uintptr_t)&F);
            else if (k == 2)
                renderer_contact::contact_run((uint64_t)(uintptr_t)&F);
            else if (k == 3)
                renderer_ssr::ssr_run((uint64_t)(uintptr_t)&F);
            else if (k == 4)
                renderer_vrs::vrs_run((uint64_t)(uintptr_t)&F);
            else
                renderer_rt::rt_run((uint64_t)(uintptr_t)&F);
            timeEnd(cmd, fi, which[k]);
        }
        barrierRt(cmd, gDepth, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_DEPTH_WRITE);
    }

    if (timed)
        timeBegin(cmd, fi, kTimeForward);

    cmdBeginDebugMarker(cmd, 1.0f, 0.8f, 0.3f, "Renderer forward");
    barrierRt(cmd, gHdr, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_RENDER_TARGET);
    if (taa)
        barrierRt(cmd, gMotion, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_RENDER_TARGET);
    bind = {};
    bind.mRenderTargetCount = taa ? 2 : 1;
    bind.mRenderTargets[0] = { gHdr, LOAD_ACTION_DONTCARE };
    bind.mRenderTargets[1] = { gMotion, LOAD_ACTION_DONTCARE }; // the forward pass and the sky cover every pixel
    bind.mDepthStencil = { gDepth, LOAD_ACTION_LOAD };
    bind.mDepthStencil.mStoreAction = STORE_ACTION_STORE;
    if (viewMode)
    {
        bind.mRenderTargets[0].mLoadAction = LOAD_ACTION_CLEAR;
        bind.mRenderTargets[0].mOverrideClearValue = 1;
        bind.mRenderTargets[0].mClearValue = {};
        bind.mDepthStencil.mLoadAction = LOAD_ACTION_CLEAR;
        bind.mDepthStencil.mOverrideClearValue = 1;
        bind.mDepthStencil.mClearValue = {}; // reversed-Z: 0 is far
    }
    // With renderer_vrs: shaded at its rates (this pass only; the next bind ends them).
    if (vrs && gShadingRate)
        gShadingRate->bindRenderTargets(cmd, &bind, F.rateImage);
    else
        cmdBindRenderTargets(cmd, &bind);
    cmdSetViewport(cmd, 0, 0, (float)w, (float)h, 0, 1);
    cmdSetScissor(cmd, 0, 0, w, h);
    if (geometry)
    {
        bool full = (gf->counts[2] & (kFlagHeatmap | kFlagCascades | kFlagAoDebug | kFlagContactDebug | kFlagSsrDebug | kFlagRtDebug)) || (shadows && F.lightSize > 0);
        drawVisible(cmd, fi, viewMode ? gForwardDepthPipe : full ? gForwardFullPipe : gForwardPipe);
    }
    if (!viewMode)
    {
        cmdBindPipeline(cmd, gSkyPipe);
        cmdBindDescriptorSet(cmd, 0, gScenePersistent);
        cmdBindDescriptorSet(cmd, fi, gSceneFrame);
        cmdDraw(cmd, 3, 0);
    }
    cmdBindRenderTargets(cmd, nullptr);
    barrierRt(cmd, gHdr, RESOURCE_STATE_RENDER_TARGET, RESOURCE_STATE_SHADER_RESOURCE);
    if (taa)
        barrierRt(cmd, gMotion, RESOURCE_STATE_RENDER_TARGET, RESOURCE_STATE_SHADER_RESOURCE);
    gDepthState = RESOURCE_STATE_DEPTH_WRITE;
    cmdEndDebugMarker(cmd);
    timeEnd(cmd, fi, kTimeForward);

    // Auto exposure: histogram, then the adapted exposure. Held still in a view mode or a debug view,
    // so it never meters (and chases) the debug image, and is where it was when it ends.
    if (!viewMode && !rawDebug && !(culled && F.heatmap))
    {
        if (timed)
            timeBegin(cmd, fi, kTimeExposure);
        barrierBuf(cmd, gExposure, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_UNORDERED_ACCESS);
        cmdBindPipeline(cmd, gHistogramPipe);
        cmdBindDescriptorSet(cmd, fi, gExposureSet);
        cmdDispatch(cmd, (w / 2 + 15) / 16, (h / 2 + 15) / 16, 1);
        barrierBuf(cmd, gHistogram, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_UNORDERED_ACCESS);
        cmdBindPipeline(cmd, gAdaptPipe);
        cmdBindDescriptorSet(cmd, fi, gExposureSet);
        cmdDispatch(cmd, 1, 1, 1);
        barrierBuf(cmd, gExposure, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_SHADER_RESOURCE);
        timeEnd(cmd, fi, kTimeExposure);
    }

    if (taa)
    {
        const uint32_t slot = upscaled ? kTimeUpscale : kTimeTaa;
        if (timed)
            timeBegin(cmd, fi, slot);
        barrierRt(cmd, gDepth, RESOURCE_STATE_DEPTH_WRITE, RESOURCE_STATE_SHADER_RESOURCE);
        gDepthState = RESOURCE_STATE_SHADER_RESOURCE;
        if (upscaled)
            renderer_upscale::upscale_run((uint64_t)(uintptr_t)&F);
        else
            renderer_taa::taa_resolve((uint64_t)(uintptr_t)&F);
        timeEnd(cmd, fi, slot);
    }
    if (post)
    {
        if (timed)
            timeBegin(cmd, fi, kTimePost);
        renderer_post::post_run((uint64_t)(uintptr_t)&F);
        timeEnd(cmd, fi, kTimePost);
    }
    if (ldr)
    {
        if (timed)
            timeBegin(cmd, fi, kTimeTonemap);
        TextureBarrier b = { gLdr, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_UNORDERED_ACCESS };
        cmdResourceBarrier(cmd, 0, nullptr, 1, &b, 0, nullptr);
        cmdBindPipeline(cmd, gTonemapPipe);
        cmdBindDescriptorSet(cmd, fi, gTonemapSet);
        cmdDispatch(cmd, (dw + 7) / 8, (dh + 7) / 8, 1);
        b = { gLdr, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_SHADER_RESOURCE };
        cmdResourceBarrier(cmd, 0, nullptr, 1, &b, 0, nullptr);
        timeEnd(cmd, fi, kTimeTonemap);
    }
    timeEnd(cmd, fi, kTimeTotal);
    for (uint32_t t = 0; t < kTimings; ++t)
        if (gTimeUsed[fi] & (1u << t))
            cmdResolveQuery(cmd, gTimePool[fi], t, 1);
    gTimedFrame = timed;

    // Nothing is left bound (every pass above unbinds); YEA_PASS_WORLD composites.
    gRenderedFrame = yea::host()->frame();
    gRenderedFi = fi;
}
} // namespace renderer
