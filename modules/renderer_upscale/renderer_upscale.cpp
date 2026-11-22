// Upscaling for the renderer core: renders the scene smaller and a vendor upscaler brings it back to the frame's size in
// TAA's place. At native size it is never in the frame: upscale_prepare returns false and the core runs exactly as
// without this module. (Its upscalers come next; until then it never upscales.)
#include "yea_renderer_upscale.h"
#include "frame.h"

namespace renderer_upscale
{
int  load(int reloaded) { return 0; }
void unload() {}
void update(uint32_t phase, float dt) {}
void draw(uint32_t pass, const YeaFrame* frame) {}

bool upscale_prepare(uint64_t address) { return false; }
void upscale_run(uint64_t address) {}
} // namespace renderer_upscale
