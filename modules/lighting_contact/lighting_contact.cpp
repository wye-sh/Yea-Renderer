// Contact shadows for the lighting core: a screen-space march towards the sun from the depth prepass
// (Shaders/Contact.comp.fsl) into a full-size R32F visibility texture the forward pass multiplies the
// sun's shadow by.
#include "yea_lighting_contact.h"
#include "frame.h"

#include <Common_3/Graphics/Interfaces/IGraphics.h>
#include <Common_3/Resources/ResourceLoader/Interfaces/IResourceLoader.h>
#include <Common_3/Graphics/FSL/defaults.h>
#define SAMPLE_COUNT 1
#include "Shaders/Contact.srt.h"
#undef SAMPLE_COUNT

#include <string.h>

namespace lighting_contact
{
namespace
{
struct ContactData
{
    float    proj[4];
    float    light[4];
    float    screen[4];
    float    params[4];
    uint32_t counts[4];
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
    d.pName = "lighting_contact";
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
    const char* names[4] = { "lighting_contact_1.comp", "lighting_contact_2.comp", "lighting_contact_4.comp", "lighting_contact_8.comp" };
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
        desc.pName = "lighting_contact";
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
        d.mDesc.pName = "lighting_contact_cb";
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
    LightingFrame* F = (LightingFrame*)(uintptr_t)address;
    if (!gReady || !settings.enabled || !F || F->version != LIGHTING_FRAME_VERSION || !F->width || !F->height || !F->sunOn)
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

    float       maxDistance = settings.max_distance < 1 ? 1 : settings.max_distance;
    int32_t     steps = settings.steps < 4 ? 4 : (settings.steps > 64 ? 64 : settings.steps);
    const float(*v)[4] = F->view;
    ContactData d = {};
    d.proj[0] = F->p00;
    d.proj[1] = F->p11;
    d.proj[2] = F->nearZ;
    d.proj[3] = clampf(settings.length, 0.01f, 10);
    for (int r = 0; r < 3; ++r)
        d.light[r] = v[r][0] * F->sunDir[0] + v[r][1] * F->sunDir[1] + v[r][2] * F->sunDir[2];
    d.light[3] = clampf(settings.thickness, 0.005f, 10);
    d.screen[0] = (float)F->width;
    d.screen[1] = (float)F->height;
    d.screen[2] = F->jitter[0];
    d.screen[3] = F->jitter[1];
    d.params[0] = clampf(settings.intensity, 0, 1);
    d.params[1] = maxDistance;
    d.params[2] = 1.0f / (maxDistance * 0.2f);
    d.counts[0] = (uint32_t)steps;
    d.counts[1] = F->taaOn ? (uint32_t)(F->frameNumber % 64) : 0; // TAA averages the noise over frames
    memcpy(gCb[F->frameIndex]->pCpuMappedAddress, &d, sizeof d);

    F->contact = gTex;
    F->contactOn = 1;
    F->contactDebug = settings.show_contact;
    return true;
}

void contact_run(uint64_t address)
{
    LightingFrame* F = (LightingFrame*)(uintptr_t)address;
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

    cmdBeginDebugMarker(cmd, 0.9f, 0.7f, 0.4f, "Lighting contact shadows");
    barrier(cmd, gTex, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_UNORDERED_ACCESS);
    cmdBindPipeline(cmd, gPipe[sampleIndex(F->sampleCount)]);
    cmdBindDescriptorSet(cmd, F->frameIndex, gSet);
    cmdDispatch(cmd, (F->width + 7) / 8, (F->height + 7) / 8, 1);
    barrier(cmd, gTex, RESOURCE_STATE_UNORDERED_ACCESS, RESOURCE_STATE_SHADER_RESOURCE);
    cmdEndDebugMarker(cmd);
}
} // namespace lighting_contact
