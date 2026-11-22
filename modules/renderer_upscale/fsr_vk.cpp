// FSR on Vulkan (Linux): AMD's FSR 3.1 upscaler and FidelityFX Vulkan backend, compiled from AMD's MIT source
// (third_party/ffx_vk, FidelityFX SDK v1.1.4; its shaders as headers made by fsr_vk_shaders.py). AMD's files are
// unchanged except their #include lines (paths within third_party/ffx_vk, and The Forge's Vulkan headers). The few
// Vulkan entry points they call directly are resolved here from the device's own loader (fsrVkLoadEntryPoints);
// the rest AMD's backend loads through vkGetDeviceProcAddr itself. renderer_upscale.cpp drives it.
#if !defined(_WIN32)

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <Common_3/Graphics/ThirdParty/OpenSource/VulkanSDK/include/vulkan/vulkan.h>
#include <dlfcn.h>

// AMD's direct calls, by their own names, file-local (The Forge's volk keeps its own).
#define FSR_VK_ENTRY_POINTS(X)                \
    X(vkGetDeviceProcAddr)                    \
    X(vkEnumerateDeviceExtensionProperties)   \
    X(vkGetPhysicalDeviceFeatures)            \
    X(vkGetPhysicalDeviceFeatures2)           \
    X(vkGetPhysicalDeviceMemoryProperties)    \
    X(vkGetPhysicalDeviceProperties)          \
    X(vkGetPhysicalDeviceProperties2)         \
    X(vkCreateBuffer)
#define FSR_VK_DECLARE(name) static PFN_##name name;
FSR_VK_ENTRY_POINTS(FSR_VK_DECLARE)

// The MSVC library extensions AMD's code calls, for its Linux build (bounded copies that always terminate).
#include <wchar.h>
#include <string.h>
#include <stdio.h>
#include <stddef.h>
#define _countof(a) (sizeof(a) / sizeof((a)[0]))
static inline int wcscpy_s(wchar_t* dst, size_t n, const wchar_t* src)
{
    if (!dst || !n)
        return 1;
    size_t i = 0;
    for (; src && src[i] && i + 1 < n; ++i)
        dst[i] = src[i];
    dst[i] = 0;
    return 0;
}
template <size_t N> static inline int wcscpy_s(wchar_t (&dst)[N], const wchar_t* src) { return wcscpy_s(dst, N, src); }
#define sprintf_s  snprintf
#define swprintf_s swprintf

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#define FFX_FSR3UPSCALER 1 // ffx_shader_blobs.cpp: FSR 3.1's upscaler alone
#include "third_party/ffx_vk/src/shared/ffx_assert.cpp"
#include "third_party/ffx_vk/src/shared/ffx_message.cpp"
#include "third_party/ffx_vk/src/shared/ffx_object_management.cpp"
#include "third_party/ffx_vk/src/shared/ffx_breadcrumbs_list.cpp"
#include "third_party/ffx_vk/src/backends/shared/ffx_shader_blobs.cpp"
#include "third_party/ffx_vk/src/backends/shared/blob_accessors/ffx_fsr3upscaler_shaderblobs.cpp"
#include "third_party/ffx_vk/src/components/fsr3upscaler/ffx_fsr3upscaler.cpp"
#include "third_party/ffx_vk/src/backends/vk/ffx_vk.cpp"
#pragma clang diagnostic pop

// AMD's frame-generation swapchain (FSR 3's frame interpolation) lives in files not taken here: the upscaler never
// calls it, but the backend's table names it.
FfxErrorCode ffxSetFrameGenerationConfigToSwapchainVK(FfxFrameGenerationConfig const*)
{
    return FFX_ERROR_INVALID_ARGUMENT;
}

// Resolves AMD's direct calls through the loader the device was made with (already in the process; never loaded
// here). False when any is missing.
bool fsrVkLoadEntryPoints(VkInstance instance)
{
    void* loader = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_NOLOAD);
    if (!loader)
        return false;
    PFN_vkGetInstanceProcAddr getInstanceProcAddr = (PFN_vkGetInstanceProcAddr)dlsym(loader, "vkGetInstanceProcAddr");
    dlclose(loader); // RTLD_NOLOAD took a reference; the host's own keeps it loaded
    if (!getInstanceProcAddr)
        return false;
    bool ok = true;
#define FSR_VK_RESOLVE(name)                                       \
    name = (PFN_##name)getInstanceProcAddr(instance, #name);       \
    ok = ok && name;
    FSR_VK_ENTRY_POINTS(FSR_VK_RESOLVE)
    return ok;
}

#endif
