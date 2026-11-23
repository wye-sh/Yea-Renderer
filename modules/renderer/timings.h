// The live timings (renderer.cpp) the Timings panel (panels/Timings.cpp) shows: each part's average a frame over the
// last half second, a part a frame skipped counting 0 there.
#pragma once

#include <stdint.h>

namespace renderer
{
struct LiveTiming
{
    const char* name;  // as the gpu and cpu lines name it
    float       ms;
    uint8_t     depth; // 1: a part of the pass before it (the see-through pass's glass, chain and particle runs)
};
// The GPU's passes as the gpu line lists them, "total" (everything the renderer records but the composite) last; none
// while gpu_timings is off.
uint32_t liveGpu(LiveTiming* out, uint32_t cap);
// The main thread's parts, largest first ("total" first).
uint32_t liveCpu(LiveTiming* out, uint32_t cap);
} // namespace renderer
