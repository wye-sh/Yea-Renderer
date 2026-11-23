// The Timings panel: a panel in F1 (never in a release of the game). Each pass's GPU time and the main thread's parts,
// each its average a frame over the last half second, as bars against the frame's total. The figures are those of the
// renderer's own timing lines (gpu_timings), live.
#include "yea_renderer.h"
#include "../timings.h"

#include <yea/ui.h>

#include <stdio.h>

namespace renderer
{
namespace
{
LiveTiming gRows[64];

// One canvas of rows: name, a bar against `scale` ms, the ms at the right. The first row (the total) in white.
void bars(const YeaUi* u, const char* label, const LiveTiming* rows, uint32_t n, uint32_t totalRow)
{
    const float rowH = 18.0f;
    YeaUiCanvas in;
    u->canvas(label, rowH * (float)n + 6.0f, &in);
    if (in.w <= 0 || !n)
        return;
    float scale = rows[totalRow].ms > 0.001f ? rows[totalRow].ms : 1.0f;
    const float nameW = 130.0f, msW = 58.0f, barX = nameW, barW = in.w - nameW - msW - 8.0f;
    char        ms[24];
    for (uint32_t i = 0; i < n; ++i)
    {
        const LiveTiming& t = rows[i];
        const float       y = 3.0f + rowH * (float)i, mid = y + rowH * 0.5f;
        const bool        total = i == totalRow;
        const float       indent = t.depth ? 14.0f : 0.0f;
        const float       c = total ? 0.95f : t.depth ? 0.6f : 0.82f;
        u->text(4.0f + indent, mid, t.name, 13.0f, c, c, c, 1.0f);
        float w = barW * (t.ms / scale);
        w = w > barW ? barW : w;
        if (barW > 0)
        {
            u->rect(barX, y + 3.0f, barW, rowH - 6.0f, 0.16f, 0.16f, 0.19f, 1.0f, 3);
            if (w > 0.5f)
            {
                if (total)
                    u->rect(barX, y + 3.0f, w, rowH - 6.0f, 0.85f, 0.85f, 0.9f, 1.0f, 3);
                else if (t.depth)
                    u->rect(barX, y + 3.0f, w, rowH - 6.0f, 0.45f, 0.6f, 0.85f, 1.0f, 3);
                else
                    u->rect(barX, y + 3.0f, w, rowH - 6.0f, 0.35f, 0.65f, 1.0f, 1.0f, 3);
            }
        }
        snprintf(ms, sizeof ms, "%.2f ms", t.ms);
        u->text(in.w - msW, mid, ms, 13.0f, c, c, c, 1.0f);
    }
}
} // namespace

void panel_timings()
{
    const YeaUi* u = yea::ui();
    if (!u)
        return;
    u->heading("GPU");
    uint32_t n = liveGpu(gRows, 64);
    if (!n)
        u->label("GPU timings are off", 1);
    else
        bars(u, "##gpu", gRows, n, n - 1); // the gpu line's order, its total last
    u->knob("GPU Timings##gpu_timings");
    u->space(2);
    u->heading("CPU (main thread)");
    n = liveCpu(gRows, 64);
    if (!n)
        u->label("Nothing measured yet", 1);
    else
        bars(u, "##cpu", gRows, n, 0); // largest first: the total
}
} // namespace renderer
