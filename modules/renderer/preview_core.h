// The Model Viewer's preview (export preview_draw; renderer_models' preview.h): one model alone, the core's meshes,
// materials, BRDF and tonemapper, a studio rig (a key light with a shadow map, a neutral grey sky), into renderer_models'
// targets. Included by renderer.cpp inside namespace renderer, after its helpers. Its pipelines and buffers are made in
// the core's load() (previewLoad) and freed in unload(): preview_draw runs inside renderer_models' draw, where the core
// allocates nothing.
// preview.h is included at the top of renderer.cpp, outside the namespace.
#include "Shaders/Preview.srt.h"
#include "Shaders/PreviewShadow.srt.h"
#include "Shaders/PreviewResolve.srt.h"

namespace
{
struct PreviewGpu // Shaders/Preview.h.fsl's PreviewData
{
    float    viewProj[4][4];
    float    shadowProj[4][4];
    float    camPos[4];
    float    sunDir[4];
    float    sunColor[4];
    float    skyZenith[4];
    float    skyHorizon[4];
    float    skyGround[4];
    float    ground[4];
    uint32_t mode[4];
};
Shader *       gPreviewShader, *gPreviewShadowShader, *gPreviewGroundShader, *gPreviewResolveShader;
Pipeline *     gPreviewPipe[2], *gPreviewShadowPipe, *gPreviewGroundPipe, *gPreviewResolvePipe;
DescriptorSet *gPreviewPersistent, *gPreviewFrame, *gPreviewShadowFrame, *gPreviewResolveFrame;
Buffer *       gPreviewCb[kMaxFrames], *gPreviewInst[kMaxFrames], *gPreviewMat[kMaxFrames], *gPreviewStream[kMaxFrames];
bool           gPreviewReady;

void previewUnload()
{
    gPreviewReady = false;
    freePipeline(gPreviewPipe[0]);
    freePipeline(gPreviewPipe[1]);
    freePipeline(gPreviewShadowPipe);
    freePipeline(gPreviewGroundPipe);
    freePipeline(gPreviewResolvePipe);
    for (uint32_t i = 0; i < kMaxFrames; ++i)
    {
        freeBuffer(gPreviewCb[i]);
        freeBuffer(gPreviewInst[i]);
        freeBuffer(gPreviewMat[i]);
        freeBuffer(gPreviewStream[i]);
    }
    DescriptorSet** sets[4] = { &gPreviewPersistent, &gPreviewFrame, &gPreviewShadowFrame, &gPreviewResolveFrame };
    for (DescriptorSet** set : sets)
        if (*set)
            removeDescriptorSet(gRenderer, *set), *set = nullptr;
    Shader** shaders[4] = { &gPreviewShader, &gPreviewShadowShader, &gPreviewGroundShader, &gPreviewResolveShader };
    for (Shader** s : shaders)
        if (*s)
            removeShader(gRenderer, *s), *s = nullptr;
}

// In the core's load(), after gDfg and gShadowSampler: everything the preview draws with. A failure leaves the preview
// off (preview_draw returns false), never the Renderer.
void previewLoad()
{
    gPreviewShader = loadShader("renderer_preview.vert", "renderer_preview.frag", nullptr);
    gPreviewShadowShader = loadShader("renderer_preview_shadow.vert", nullptr, nullptr);
    gPreviewGroundShader = loadShader("renderer_preview_ground.vert", "renderer_preview_ground.frag", nullptr);
    gPreviewResolveShader = loadShader("renderer_fullscreen.vert", "renderer_preview_resolve.frag", nullptr);
    if (!gPreviewShader || !gPreviewShadowShader || !gPreviewGroundShader || !gPreviewResolveShader)
    {
        yea::host()->log(1, "renderer: the Model Viewer's preview shaders are missing; no preview");
        previewUnload();
        return;
    }
    DescriptorSetDesc ps = SRT_SET_DESC(PreviewSrt, Persistent, 1, 0);
    addDescriptorSet(gRenderer, &ps, &gPreviewPersistent);
    DescriptorSetDesc pf = SRT_SET_DESC(PreviewSrt, PerFrame, gFrames, 0);
    addDescriptorSet(gRenderer, &pf, &gPreviewFrame);
    DescriptorSetDesc sf = SRT_SET_DESC(PreviewShadowSrt, PerFrame, gFrames, 0);
    addDescriptorSet(gRenderer, &sf, &gPreviewShadowFrame);
    DescriptorSetDesc rf = SRT_SET_DESC(PreviewResolveSrt, PerFrame, gFrames, 0);
    addDescriptorSet(gRenderer, &rf, &gPreviewResolveFrame);
    const ResourceMemoryUsage cpu = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
    const ResourceState       sr = RESOURCE_STATE_SHADER_RESOURCE;
    bool                      made = gPreviewPersistent && gPreviewFrame && gPreviewShadowFrame && gPreviewResolveFrame;
    for (uint32_t i = 0; i < gFrames; ++i)
    {
        gPreviewCb[i] = makeBuffer(sizeof(PreviewGpu), 0, DESCRIPTOR_TYPE_UNIFORM_BUFFER, cpu, RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
                                   "renderer_preview");
        gPreviewInst[i] = makeBuffer(RENDER_PREVIEW_PARTS * sizeof(RenderGpuInstance), sizeof(RenderGpuInstance), DESCRIPTOR_TYPE_BUFFER,
                                     cpu, sr, "renderer_preview_instances");
        gPreviewMat[i] = makeBuffer(RENDER_PREVIEW_PARTS * sizeof(GpuMaterial), sizeof(GpuMaterial), DESCRIPTOR_TYPE_BUFFER, cpu, sr,
                                    "renderer_preview_materials");
        gPreviewStream[i] = makeBuffer(RENDER_PREVIEW_PARTS * 4, 0, DESCRIPTOR_TYPE_VERTEX_BUFFER, cpu,
                                       RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER, "renderer_preview_stream");
        made = made && gPreviewCb[i] && gPreviewInst[i] && gPreviewMat[i] && gPreviewStream[i] && gPreviewCb[i]->pCpuMappedAddress &&
               gPreviewInst[i]->pCpuMappedAddress && gPreviewMat[i]->pCpuMappedAddress && gPreviewStream[i]->pCpuMappedAddress;
    }
    if (!made)
    {
        yea::host()->log(1, "renderer: the GPU refused the Model Viewer's preview buffers; no preview");
        previewUnload();
        return;
    }
    waitForAllResourceLoads();
    for (uint32_t i = 0; i < gFrames; ++i)
    {
        uint32_t* s = (uint32_t*)gPreviewStream[i]->pCpuMappedAddress;
        for (uint32_t k = 0; k < RENDER_PREVIEW_PARTS; ++k)
            s[k] = k; // part k reads instance row k
    }
    DescriptorData p[2] = {};
    p[0].mIndex = SRT_RES_IDX(PreviewSrt, Persistent, gShadowCmp);
    p[0].ppSamplers = &gShadowSampler;
    p[1].mIndex = SRT_RES_IDX(PreviewSrt, Persistent, gDfg);
    p[1].ppBuffers = &gDfg;
    updateDescriptorSet(gRenderer, 0, gPreviewPersistent, 2, p);

    VertexLayout        layout = sceneLayout();
    RasterizerStateDesc raster = {};
    raster.mFrontFace = FRONT_FACE_CW; // as the core's camera: counter-clockwise around the normal is clockwise on screen
    raster.mMultiSample = true;
    DepthStateDesc depth = {};
    depth.mDepthTest = true;
    depth.mDepthWrite = true;
    depth.mDepthFunc = CMP_GEQUAL; // reversed Z
    TinyImageFormat       hdr = TinyImageFormat_R16G16B16A16_SFLOAT;
    PipelineDesc          desc = {};
    GraphicsPipelineDesc& g = desc.mGraphicsDesc;
    desc.mType = PIPELINE_TYPE_GRAPHICS;
    PIPELINE_LAYOUT_DESC(desc, SRT_LAYOUT_DESC(PreviewSrt, Persistent), SRT_LAYOUT_DESC(PreviewSrt, PerFrame), NULL, NULL);
    g.mPrimitiveTopo = PRIMITIVE_TOPO_TRI_LIST;
    g.mRenderTargetCount = 1;
    g.pColorFormats = &hdr;
    g.mSampleCount = SAMPLE_COUNT_4;
    g.mDepthStencilFormat = TinyImageFormat_D32_SFLOAT;
    g.pRasterizerState = &raster;
    g.pDepthState = &depth;
    g.pVertexLayout = &layout;
    g.pShaderProgram = gPreviewShader;
    desc.pName = "renderer_preview";
    for (uint32_t s = 0; s < 2; ++s)
    {
        raster.mCullMode = s ? CULL_MODE_NONE : CULL_MODE_BACK;
        addPipeline(gRenderer, &desc, &gPreviewPipe[s]);
    }
    // The floor: blended over what is there (its shadow, black with alpha), tested but not written.
    BlendStateDesc blend = {};
    blend.mSrcFactors[0] = BC_SRC_ALPHA;
    blend.mDstFactors[0] = BC_ONE_MINUS_SRC_ALPHA;
    blend.mSrcAlphaFactors[0] = BC_ONE;
    blend.mDstAlphaFactors[0] = BC_ONE_MINUS_SRC_ALPHA;
    blend.mBlendModes[0] = BM_ADD;
    blend.mBlendAlphaModes[0] = BM_ADD;
    blend.mColorWriteMasks[0] = COLOR_MASK_ALL;
    blend.mRenderTargetMask = BLEND_STATE_TARGET_0;
    DepthStateDesc test = depth;
    test.mDepthWrite = false;
    raster.mCullMode = CULL_MODE_NONE;
    g.pDepthState = &test;
    g.pBlendState = &blend;
    g.pVertexLayout = nullptr;
    g.pShaderProgram = gPreviewGroundShader;
    desc.pName = "renderer_preview_ground";
    addPipeline(gRenderer, &desc, &gPreviewGroundPipe);

    // The key light's shadow map: depth alone, standard Z, from both sides (as the core's shadow maps).
    PipelineDesc          sdesc = {};
    GraphicsPipelineDesc& sg = sdesc.mGraphicsDesc;
    sdesc.mType = PIPELINE_TYPE_GRAPHICS;
    PIPELINE_LAYOUT_DESC(sdesc, NULL, SRT_LAYOUT_DESC(PreviewShadowSrt, PerFrame), NULL, NULL);
    RasterizerStateDesc sraster = {};
    sraster.mCullMode = CULL_MODE_NONE;
    DepthStateDesc sdepth = {};
    sdepth.mDepthTest = true;
    sdepth.mDepthWrite = true;
    sdepth.mDepthFunc = CMP_LEQUAL;
    sg.mPrimitiveTopo = PRIMITIVE_TOPO_TRI_LIST;
    sg.mSampleCount = SAMPLE_COUNT_1;
    sg.mDepthStencilFormat = TinyImageFormat_D32_SFLOAT;
    sg.pRasterizerState = &sraster;
    sg.pDepthState = &sdepth;
    sg.pVertexLayout = &layout;
    sg.pShaderProgram = gPreviewShadowShader;
    sdesc.pName = "renderer_preview_shadow";
    addPipeline(gRenderer, &sdesc, &gPreviewShadowPipe);

    // The resolve into renderer_models' sRGB target.
    PipelineDesc          rdesc = {};
    GraphicsPipelineDesc& rg = rdesc.mGraphicsDesc;
    rdesc.mType = PIPELINE_TYPE_GRAPHICS;
    PIPELINE_LAYOUT_DESC(rdesc, NULL, SRT_LAYOUT_DESC(PreviewResolveSrt, PerFrame), NULL, NULL);
    RasterizerStateDesc rraster = {};
    rraster.mCullMode = CULL_MODE_NONE;
    DepthStateDesc  rdepth = {};
    TinyImageFormat out = TinyImageFormat_R8G8B8A8_SRGB;
    rg.mPrimitiveTopo = PRIMITIVE_TOPO_TRI_LIST;
    rg.mRenderTargetCount = 1;
    rg.pColorFormats = &out;
    rg.mSampleCount = SAMPLE_COUNT_1;
    rg.mDepthStencilFormat = TinyImageFormat_UNDEFINED;
    rg.pRasterizerState = &rraster;
    rg.pDepthState = &rdepth;
    rg.pShaderProgram = gPreviewResolveShader;
    rdesc.pName = "renderer_preview_resolve";
    addPipeline(gRenderer, &rdesc, &gPreviewResolvePipe);

    gPreviewReady = gPreviewPipe[0] && gPreviewPipe[1] && gPreviewGroundPipe && gPreviewShadowPipe && gPreviewResolvePipe;
    if (!gPreviewReady)
    {
        yea::host()->log(1, "renderer: the Model Viewer's preview pipelines failed; no preview");
        previewUnload();
    }
}

// Rows of a 4x4 that takes world points into the camera's clip space: reversed Z, infinite far, as the core's camera.
void previewCamera(const RenderPreview& p, float out[4][4])
{
    v3 eye = V(p.eye[0], p.eye[1], p.eye[2]);
    v3 fwd = normalize(V(p.at[0], p.at[1], p.at[2]) - eye);
    v3 right, up;
    basis(fwd, &right, &up);
    float view[4][4];
    lookRows(eye, fwd, right, up, view);
    float nearZ = maxf(p.radius * 0.01f, 0.001f);
    float p11 = 1.0f / tanf(clampf(p.fov, 5.0f, 150.0f) * 0.5f * 3.14159265f / 180.0f);
    float p00 = p11 * (float)p.height / (float)p.width;
    float proj[4][4] = { { p00, 0, 0, 0 }, { 0, p11, 0, 0 }, { 0, 0, 0, nearZ }, { 0, 0, 1, 0 } };
    mul(proj, view, out);
}
} // namespace

bool preview_draw(uint64_t address)
{
    RenderPreview* p = (RenderPreview*)(uintptr_t)address;
    if (!gReady || !gPreviewReady || !p || p->version != RENDER_PREVIEW_VERSION || !p->cmd || !p->hdr || !p->depth || !p->shadow ||
        !p->out || !p->width || !p->height || !gPosBuf || !gIndexBuf)
        return false;
    Cmd*           cmd = p->cmd;
    const uint32_t fi = p->frameIndex % gFrames;
    const uint32_t parts = p->parts < RENDER_PREVIEW_PARTS ? p->parts : RENDER_PREVIEW_PARTS;

    // Each part's version: the simplest within lod_error display pixels at the model's nearest point (as the core
    // picks per instance), or the one asked for; its instance row and its material, numbered within the preview.
    RenderGpuInstance* inst = (RenderGpuInstance*)gPreviewInst[fi]->pCpuMappedAddress;
    GpuMaterial*       mats = (GpuMaterial*)gPreviewMat[fi]->pCpuMappedAddress;
    uint32_t           drawMesh[RENDER_PREVIEW_PARTS];
    const uint32_t     meshCount = Meshes.size(), lodRows = MeshLods.size() < meshCount ? MeshLods.size() : meshCount;
    const float        p11 = 1.0f / tanf(clampf(p->fov, 5.0f, 150.0f) * 0.5f * 3.14159265f / 180.0f);
    const float        dist = length(V(p->eye[0] - p->center[0], p->eye[1] - p->center[1], p->eye[2] - p->center[2]));
    const float        perUnit = 0.5f * (float)p->height * p11 / maxf(dist - p->radius, p->radius * 0.01f + 1e-4f);
    p->lodShown = 0;
    p->triangles = 0;
    for (uint32_t k = 0; k < parts; ++k)
    {
        uint32_t m = p->mesh[k];
        drawMesh[k] = 0xFFFFFFFFu;
        if (m >= meshCount || Meshes[m].firstIndex + Meshes[m].indexCount > gUploadedIndices)
            continue;
        const float* t = p->transform[k];
        float        scale = maxf(sqrtf(t[0] * t[0] + t[4] * t[4] + t[8] * t[8]),
                                  maxf(sqrtf(t[1] * t[1] + t[5] * t[5] + t[9] * t[9]), sqrtf(t[2] * t[2] + t[6] * t[6] + t[10] * t[10])));
        float        allowed = settings.lod_error / (maxf(p->lodQuality, 0.01f) * perUnit * maxf(scale, 1e-6f));
        uint32_t     depth = 0;
        for (uint32_t next = m < lodRows ? MeshLods[m].next : 0xFFFFFFFFu;
             next < lodRows && Meshes[next].firstIndex + Meshes[next].indexCount <= gUploadedIndices &&
             (p->lod >= 0 ? depth < (uint32_t)p->lod : settings.lod_error > 0 && MeshLods[next].error <= allowed);
             next = MeshLods[next].next)
            m = next, ++depth;
        drawMesh[k] = m;
        p->lodShown = depth > p->lodShown ? depth : p->lodShown;
        p->triangles += Meshes[m].indexCount / 3;
        memcpy(inst[k].row, t, sizeof inst[k].row);
        inst[k].material = k;
        inst[k].flags = 0;
        uint32_t row = p->material[k] & 0xFFFFFu;
        mats[k] = GpuMaterial{};
        mats[k].albedoRough[0] = mats[k].albedoRough[1] = mats[k].albedoRough[2] = 0.7f;
        mats[k].albedoRough[3] = 0.5f;
        mats[k].params[1] = 0.04f;
        if (row < Materials.size() && Materials[row].alive && Materials[row].gen == (p->material[k] >> 20))
        {
            const Material& mt = Materials[row];
            mats[k].albedoRough[0] = mt.albedo.x;
            mats[k].albedoRough[1] = mt.albedo.y;
            mats[k].albedoRough[2] = mt.albedo.z;
            mats[k].albedoRough[3] = mt.roughness;
            mats[k].params[0] = mt.metallic;
            mats[k].params[1] = 0.16f * mt.reflectance * mt.reflectance;
            mats[k].params[2] = mt.reflections ? 1.0f : 0.0f;
            mats[k].emissive[0] = mt.emissive.x;
            mats[k].emissive[1] = mt.emissive.y;
            mats[k].emissive[2] = mt.emissive.z;
        }
    }

    // The studio: a neutral sky, the key light, its shadow map fitted around the model's sphere (standard Z).
    PreviewGpu& c = *(PreviewGpu*)gPreviewCb[fi]->pCpuMappedAddress;
    c = PreviewGpu{};
    previewCamera(*p, c.viewProj);
    v3 L = normalize(V(p->sunDir[0], p->sunDir[1], p->sunDir[2]));
    v3 lr, lu;
    basis(-L, &lr, &lu);
    const v3    ctr = V(p->center[0], p->center[1], p->center[2]);
    const float R = maxf(p->radius, 1e-4f) * 1.05f, ground = R * 3.0f;
    // The light's box holds the model and the floor around it, whose shadow shows.
    const float side = ground * 1.1f, depthRange = side * 2.0f;
    const float sp[4][4] = { { lr.x / side, lr.y / side, lr.z / side, -dot(lr, ctr) / side },
                             { lu.x / side, lu.y / side, lu.z / side, -dot(lu, ctr) / side },
                             { -L.x / depthRange, -L.y / depthRange, -L.z / depthRange, 0.5f + dot(L, ctr) / depthRange },
                             { 0, 0, 0, 1 } };
    memcpy(c.shadowProj, sp, sizeof sp);
    c.camPos[0] = p->eye[0], c.camPos[1] = p->eye[1], c.camPos[2] = p->eye[2];
    const float sky = maxf(p->skyLuminance, 0.0f), lux = maxf(p->sunLux, 0.0f);
    c.camPos[3] = 0.6f / maxf(lux / 3.14159265f + sky, 1e-4f); // a white wall facing the key light at about 0.6
    c.sunDir[0] = L.x, c.sunDir[1] = L.y, c.sunDir[2] = L.z;
    const uint32_t shadowSize = p->shadow->mWidth ? p->shadow->mWidth : 1024;
    c.sunDir[3] = 1.0f / (float)shadowSize;
    c.sunColor[0] = c.sunColor[1] = c.sunColor[2] = lux;
    c.sunColor[3] = 2.0f * side / (float)shadowSize; // a texel, in world units
    c.skyZenith[0] = c.skyZenith[1] = c.skyZenith[2] = sky;
    c.skyHorizon[0] = c.skyHorizon[1] = c.skyHorizon[2] = sky * 0.8f;
    c.skyGround[0] = c.skyGround[1] = c.skyGround[2] = sky * 0.3f;
    c.ground[0] = ctr.x, c.ground[1] = p->floorY, c.ground[2] = ctr.z, c.ground[3] = ground;
    const bool tony = settings.tonemapper == Tonemapper::Tony_McMapface;
    c.mode[0] = (uint32_t)(settings.tonemapper < 0 || (tony && !gTonyLut) ? 0 : settings.tonemapper);
    c.mode[1] = p->width, c.mode[2] = p->height;

    {
        DescriptorData d[4] = {};
        d[0].mIndex = SRT_RES_IDX(PreviewSrt, PerFrame, gPreview);
        d[0].ppBuffers = &gPreviewCb[fi];
        d[1].mIndex = SRT_RES_IDX(PreviewSrt, PerFrame, gInstances);
        d[1].ppBuffers = &gPreviewInst[fi];
        d[2].mIndex = SRT_RES_IDX(PreviewSrt, PerFrame, gMaterials);
        d[2].ppBuffers = &gPreviewMat[fi];
        d[3].mIndex = SRT_RES_IDX(PreviewSrt, PerFrame, gShadowMap);
        d[3].ppTextures = &p->shadow->pTexture;
        updateDescriptorSet(gRenderer, fi, gPreviewFrame, 4, d);
        DescriptorData s[2] = {};
        s[0].mIndex = SRT_RES_IDX(PreviewShadowSrt, PerFrame, gPreview);
        s[0].ppBuffers = &gPreviewCb[fi];
        s[1].mIndex = SRT_RES_IDX(PreviewShadowSrt, PerFrame, gInstances);
        s[1].ppBuffers = &gPreviewInst[fi];
        updateDescriptorSet(gRenderer, fi, gPreviewShadowFrame, 2, s);
        Texture*       tonyTex = gTonyLut ? gTonyLut : gDummyColor->pTexture;
        DescriptorData r[3] = {};
        r[0].mIndex = SRT_RES_IDX(PreviewResolveSrt, PerFrame, gPreview);
        r[0].ppBuffers = &gPreviewCb[fi];
        r[1].mIndex = SRT_RES_IDX(PreviewResolveSrt, PerFrame, gHdr);
        r[1].ppTextures = &p->hdr->pTexture;
        r[2].mIndex = SRT_RES_IDX(PreviewResolveSrt, PerFrame, gTony);
        r[2].ppTextures = &tonyTex;
        updateDescriptorSet(gRenderer, fi, gPreviewResolveFrame, 3, r);
    }

    auto drawParts = [&](bool shadow, uint32_t sided) {
        cmdBindIndexBuffer(cmd, gIndexBuf, INDEX_TYPE_UINT32, 0);
        for (uint32_t k = 0; k < parts; ++k)
        {
            if (drawMesh[k] == 0xFFFFFFFFu || (!shadow && (p->twoSided[k] ? 1u : 0u) != sided))
                continue;
            const Mesh& mesh = Meshes[drawMesh[k]];
            bindGeometry(cmd, gPreviewStream[fi], k);
            cmdDrawIndexedInstanced(cmd, mesh.indexCount, mesh.firstIndex, 1, mesh.firstVertex, 0);
        }
    };
    cmdBeginDebugMarker(cmd, 0.9f, 0.8f, 0.4f, "Renderer Model Viewer preview");

    // The key light's shadow map.
    barrierRt(cmd, p->shadow, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_DEPTH_WRITE);
    BindRenderTargetsDesc bind = {};
    bind.mDepthStencil = { p->shadow, LOAD_ACTION_CLEAR };
    bind.mDepthStencil.mStoreAction = STORE_ACTION_STORE;
    bind.mDepthStencil.mClearValue = {};
    bind.mDepthStencil.mClearValue.depth = 1.0f;
    cmdBindRenderTargets(cmd, &bind);
    cmdSetViewport(cmd, 0, 0, (float)shadowSize, (float)(p->shadow->mHeight ? p->shadow->mHeight : shadowSize), 0, 1);
    cmdSetScissor(cmd, 0, 0, shadowSize, p->shadow->mHeight ? p->shadow->mHeight : shadowSize);
    cmdBindPipeline(cmd, gPreviewShadowPipe);
    cmdBindDescriptorSet(cmd, fi, gPreviewShadowFrame);
    drawParts(true, 0);
    cmdBindRenderTargets(cmd, nullptr);
    barrierRt(cmd, p->shadow, RESOURCE_STATE_DEPTH_WRITE, RESOURCE_STATE_SHADER_RESOURCE);

    // The model, then its floor, at 4 samples.
    barrierRt(cmd, p->hdr, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_RENDER_TARGET);
    bind = {};
    bind.mRenderTargetCount = 1;
    bind.mRenderTargets[0] = { p->hdr, LOAD_ACTION_CLEAR };
    bind.mRenderTargets[0].mClearValue = {}; // transparent: the panel shows through around the model
    bind.mDepthStencil = { p->depth, LOAD_ACTION_CLEAR };
    bind.mDepthStencil.mClearValue = {}; // reversed Z: 0 is far
    cmdBindRenderTargets(cmd, &bind);
    cmdSetViewport(cmd, 0, 0, (float)p->width, (float)p->height, 0, 1);
    cmdSetScissor(cmd, 0, 0, p->width, p->height);
    for (uint32_t s = 0; s < 2; ++s)
    {
        cmdBindPipeline(cmd, gPreviewPipe[s]);
        cmdBindDescriptorSet(cmd, 0, gPreviewPersistent);
        cmdBindDescriptorSet(cmd, fi, gPreviewFrame);
        drawParts(false, s);
    }
    cmdBindPipeline(cmd, gPreviewGroundPipe);
    cmdBindDescriptorSet(cmd, 0, gPreviewPersistent);
    cmdBindDescriptorSet(cmd, fi, gPreviewFrame);
    cmdDraw(cmd, 6, 0);
    cmdBindRenderTargets(cmd, nullptr);
    barrierRt(cmd, p->hdr, RESOURCE_STATE_RENDER_TARGET, RESOURCE_STATE_SHADER_RESOURCE);

    // Tonemapped and resolved into renderer_models' target, which its panel shows.
    barrierRt(cmd, p->out, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_RENDER_TARGET);
    bind = {};
    bind.mRenderTargetCount = 1;
    bind.mRenderTargets[0] = { p->out, LOAD_ACTION_DONTCARE };
    cmdBindRenderTargets(cmd, &bind);
    cmdSetViewport(cmd, 0, 0, (float)p->width, (float)p->height, 0, 1);
    cmdSetScissor(cmd, 0, 0, p->width, p->height);
    cmdBindPipeline(cmd, gPreviewResolvePipe);
    cmdBindDescriptorSet(cmd, fi, gPreviewResolveFrame);
    cmdDraw(cmd, 3, 0);
    cmdBindRenderTargets(cmd, nullptr);
    barrierRt(cmd, p->out, RESOURCE_STATE_RENDER_TARGET, RESOURCE_STATE_SHADER_RESOURCE);
    cmdEndDebugMarker(cmd);
    return true;
}
