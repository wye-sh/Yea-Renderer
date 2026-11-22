// Contact shadows for the renderer core: Bend Studio's screen space shadows (Shaders/Contact.comp.fsl and
// BendSss.h.fsl; third_party/bend_sss, Apache-2.0) from the depth prepass towards the sun, into a full-size R32F
// visibility texture the forward pass multiplies the sun's shadow by.
#include "yea_renderer_contact.h"
#include "frame.h"
#include "third_party/bend_sss/bend_sss_cpu.h"

#include <Common_3/Graphics/Interfaces/IGraphics.h>
#include <Common_3/Resources/ResourceLoader/Interfaces/IResourceLoader.h>
#include <Common_3/Graphics/FSL/defaults.h>
#define SAMPLE_COUNT 1
#include "Shaders/Contact.srt.h"
#undef SAMPLE_COUNT

#include <string.h>

namespace renderer_contact
{
namespace
{
struct ContactData
{
    float light[4];
    float params[4];
    float fade[4];
    float size[4];
    float dispatch[8][4];
};

const uint32_t kMaxFrames = 4;

Renderer*      gRenderer;
Queue*         gQueue;
uint32_t       gFrames;
Shader*        gShader[4];
Pipeline*      gPipe[4];
DescriptorSet* gSet;
Buffer*        gCb[kMaxFrames];
Texture*       gTex; // SHADER_RESOURCE between frames
uint32_t       gWidth, gHeight;
uint32_t       gGroups[kMaxFrames]; // this frame's groups along y (all of Bend's dispatches as one)
bool           gReady;

uint32_t sampleIndex(uint32_t s) { return s == 8 ? 3 : s == 4 ? 2 : s == 2 ? 1 : 0; }
float    clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

void freeTexture()
{
    if (gTex)
        removeResource(gTex);
    gTex = nullptr;
    gWidth = gHeight = 0;
}

void createTexture(uint32_t w, uint32_t h)
{
    TextureDesc d = {};
    d.mWidth = w;
    d.mHeight = h;
    d.mDepth = d.mArraySize = d.mMipLevels = 1;
    d.mSampleCount = SAMPLE_COUNT_1;
    d.mFormat = TinyImageFormat_R32_SFLOAT; // RWTex2D(float) is r32f on Vulkan
    d.mStartState = RESOURCE_STATE_SHADER_RESOURCE;
    d.mDescriptors = (DescriptorType)(DESCRIPTOR_TYPE_TEXTURE | DESCRIPTOR_TYPE_RW_TEXTURE);
    d.pName = "renderer_contact";
    TextureLoadDesc l = {};
    l.pDesc = &d;
    l.ppTexture = &gTex;
    addResource(&l, nullptr);
    waitForAllResourceLoads();
    gWidth = w;
    gHeight = h;
}

void barrier(Cmd* cmd, Texture* t, ResourceState from, ResourceState to)
{
    TextureBarrier b = {};
    b.pTexture = t;
    b.mCurrentState = from;
    b.mNewState = to;
    cmdResourceBarrier(cmd, 0, nullptr, 1, &b, 0, nullptr);
}
} // namespace

int load(int reloaded)
{
    const YeaHost* host = yea::host();
    gRenderer = (Renderer*)host->renderer;
    gQueue = (Queue*)host->queue;
    gFrames = host->framesInFlight < kMaxFrames ? host->framesInFlight : kMaxFrames;
    const char* names[4] = { "renderer_contact_1.comp", "renderer_contact_2.comp", "renderer_contact_4.comp", "renderer_contact_8.comp" };
    for (int i = 0; i < 4; ++i)
    {
        ShaderLoadDesc sd = {};
        sd.mComp.pFileName = names[i];
        addShader(gRenderer, &sd, &gShader[i]);
        if (!gShader[i])
            return 1;
        PipelineDesc desc = {};
        desc.mType = PIPELINE_TYPE_COMPUTE;
        PIPELINE_LAYOUT_DESC(desc, NULL, SRT_LAYOUT_DESC(ContactSrt, PerFrame), NULL, NULL);
        desc.mComputeDesc.pShaderProgram = gShader[i];
        desc.pName = "renderer_contact";
        addPipeline(gRenderer, &desc, &gPipe[i]);
    }
    DescriptorSetDesc setDesc = SRT_SET_DESC(ContactSrt, PerFrame, gFrames, 0);
    addDescriptorSet(gRenderer, &setDesc, &gSet);
    for (uint32_t i = 0; i < gFrames; ++i)
    {
        BufferLoadDesc d = {};
        d.mDesc.mDescriptors = DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        d.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
        d.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
        d.mDesc.mSize = 256;
        d.mDesc.mStartState = RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
        d.mDesc.pName = "renderer_contact_cb";
        d.ppBuffer = &gCb[i];
        addResource(&d, nullptr);
    }
    waitForAllResourceLoads();
    gReady = gSet != nullptr;
    return gReady ? 0 : 1;
}

void unload()
{
    if (!gRenderer)
        return;
    waitQueueIdle(gQueue);
    freeTexture();
    for (uint32_t i = 0; i < gFrames; ++i)
        if (gCb[i])
            removeResource(gCb[i]);
    if (gSet)
        removeDescriptorSet(gRenderer, gSet);
    for (int i = 0; i < 4; ++i)
    {
        if (gPipe[i])
            removePipeline(gRenderer, gPipe[i]);
        if (gShader[i])
            removeShader(gRenderer, gShader[i]);
    }
}

void update(uint32_t phase, float dt) {}
void draw(uint32_t pass, const YeaFrame* frame) {}

bool contact_prepare(uint64_t address)
{
    RenderFrame* F = (RenderFrame*)(uintptr_t)address;
    if (!gReady || !settings.enabled || !F || F->version != RENDER_FRAME_VERSION || !F->width || !F->height || !F->sunOn)
        return false;
    if (yea::viewMode()) // F1's Wireframe or Overdraw: stand aside so they read right
        return false;
    if (F->width != gWidth || F->height != gHeight)
    {
        waitQueueIdle(gQueue);
        freeTexture();
        createTexture(F->width, F->height);
    }
    if (!gTex)
        return false;

    float maxDistance = settings.max_distance < 1 ? 1 : settings.max_distance;
    // The sun's homogeneous point on screen: its direction (w 0) through the view and the jittered projection,
    // the jitter (pixels) as the ndc offset it adds per unit of view depth.
    const float(*v)[4] = F->view;
    float       dir[3];
    for (int r = 0; r < 3; ++r)
        dir[r] = v[r][0] * F->sunDir[0] + v[r][1] * F->sunDir[1] + v[r][2] * F->sunDir[2];
    float lightProj[4] = { F->p00 * dir[0] + 2.0f * F->jitter[0] / (float)F->width * dir[2],
                           F->p11 * dir[1] - 2.0f * F->jitter[1] / (float)F->height * dir[2], 0.0f, dir[2] };
    int               viewport[2] = { (int)F->width, (int)F->height };
    int               lo[2] = { 0, 0 };
    Bend::DispatchList list = Bend::BuildDispatchList(lightProj, viewport, lo, viewport);

    ContactData d = {};
    memcpy(d.light, list.LightCoordinate_Shader, sizeof d.light);
    d.params[0] = clampf(settings.surface_thickness, 0.0001f, 1.0f);
    d.params[1] = clampf(settings.edge_threshold, 0.0001f, 1.0f);
    d.params[2] = clampf(settings.contrast, 1.0f, 1e6f);
    d.params[3] = clampf(settings.intensity, 0, 1);
    d.fade[0] = F->nearZ;
    d.fade[1] = maxDistance;
    d.fade[2] = 1.0f / (maxDistance * 0.2f);
    d.fade[3] = settings.ignore_edges ? 1.0f : 0.0f;
    d.size[0] = (float)F->width;
    d.size[1] = (float)F->height;
    d.size[2] = (float)list.DispatchCount;
    d.size[3] = F->nearZ / maxDistance; // reversed Z: depth near / z
    uint32_t groups = 0;
    for (int i = 0; i < list.DispatchCount; ++i)
    {
        const Bend::DispatchData& b = list.Dispatch[i];
        d.dispatch[i][0] = (float)b.WaveOffset_Shader[0];
        d.dispatch[i][1] = (float)b.WaveOffset_Shader[1];
        d.dispatch[i][2] = (float)b.WaveCount[1];
        d.dispatch[i][3] = (float)groups;
        groups += (uint32_t)(b.WaveCount[1] * b.WaveCount[2]); // WaveCount[0] is always 64: the dispatch's x
    }
    gGroups[F->frameIndex] = groups;
    memcpy(gCb[F->frameIndex]->pCpuMappedAddress, &d, sizeof d);

    F->contact = gTex;
    F->contactOn = 1;
    F->contactDebug = settings.show_contact;
    return true;
}

void contact_run(uint64_t address)
{
    RenderFrame* F = (RenderFrame*)(uintptr_t)address;
    if (!gReady || !F || F->contact != gTex || !gTex || !F->depth)
        return;
    Cmd*           cmd = F->cmd;
    DescriptorData p[3] = {};
    p[0].mIndex = SRT_RES_IDX(ContactSrt, PerFrame, gContact);
    p[0].ppBuffers = &gCb[F->frameIndex];
    p[1].mIndex = SRT_RES_IDX(ContactSrt, PerFrame, gDepth);
    p[1].ppTextures = &F->depth->pTexture;
    p[2].mIndex = SRT_RES_IDX(ContactSrt, PerFrame, gOut);
    p[2].ppTextures = &gTex;
    updateDescriptorSet(gRenderer, F->frameIndex, gSet, 3, p);

    cmdBeginDebugMarker(cmd, 0.9f, 0.7f, 0.4f, "Renderer contact shadows");
    barrier(cmd, gTex, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_UNORDERED_ACCESS);
    cmdBindPipeline(cmd, gPipe[sampleIndex(F->sampleCount)]);
    cmdBindDescriptorSet(cmd, F->frameIndex, gSet);
    if (gGroups[F->frameIndex])
        cmdDispatch(cmd, 64, gGroups[F->frameIndex], 1);
    barrier(cmd, gTex, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_SHADER_RESOURCE);
    cmdEndDebugMarker(cmd);
}
} // namespace renderer_contact
