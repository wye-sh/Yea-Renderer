// The Model Viewer: a panel in F1 (never in a release of the game). Every model the view's game declared, in a dropdown
// (its search inside it); under it the chosen model turning in an orbiting preview (drag to orbit, the wheel to zoom, a
// double-click back to the front), drawn by the renderer itself under a studio rig into this module's own target; the
// version shown (Auto as the error on screen picks, or each LOD); then its knobs. Add Models (the header's +) and files
// dropped on the panel go to the agents: the panel never writes anything.
#include "yea_renderer_models.h"
#include "../models.h"

#include <yea/ui.h>

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

namespace renderer_models
{
namespace
{
uint32_t gSel = 0; // the model shown (its id)
YeaUiRow gRows[kMaxModels];
uint32_t gIds[kMaxModels];
char     gNames[kMaxModels][80], gRights[kMaxModels][48];
char     gLodNames[9][16];
constexpr uint32_t kClipRows = 256;
YeaUiRow gClipRows[kClipRows + 1];
char     gClipNames[kClipRows + 1][64], gClipRights[kClipRows + 1][16];

// "old_chair": "Old Chair".
void pretty(const char* in, char* out, size_t cap)
{
    size_t n = 0;
    bool   start = true;
    for (; *in && n + 1 < cap; ++in)
    {
        const char c = *in == '_' ? ' ' : *in;
        out[n++] = start ? char(toupper((unsigned char)c)) : c;
        start = c == ' ';
    }
    out[n] = 0;
}
// 12345: "12,345".
void thousands(uint32_t v, char* out, size_t cap)
{
    char raw[16];
    snprintf(raw, sizeof raw, "%u", v);
    size_t len = strlen(raw), n = 0;
    for (size_t i = 0; i < len && n + 2 < cap; ++i)
    {
        out[n++] = raw[i];
        if ((len - i - 1) % 3 == 0 && i + 1 < len)
            out[n++] = ',';
    }
    out[n] = 0;
}
} // namespace

void panel_model_viewer()
{
    const YeaUi* u = yea::ui();
    if (!u)
        return;
    u->header("Add", "Add Models", ""); // pressed: the file picker, then the chat with this panel

    uint32_t rows = 0;
    int      chosen = -1;
    for (uint32_t i = 0; i < gEntryCount && rows < kMaxModels; ++i)
    {
        const Entry& e = gEntries[i];
        char         name[64], tris[24];
        pretty(e.name, name, sizeof name);
        snprintf(gNames[rows], sizeof gNames[rows], "%s##%u", name, e.id);
        thousands(triangles(e), tris, sizeof tris);
        const char* kind = e.kind == 0 ? "Shape" : e.kind == 2 ? "Skinned" : "Static";
        if (e.state == 2)
            snprintf(gRights[rows], sizeof gRights[rows], "%s \xC2\xB7 failed", kind);
        else
            snprintf(gRights[rows], sizeof gRights[rows], "%s \xC2\xB7 %s \xE2\x96\xB3", kind, tris);
        uint32_t placed = 0;
        for (uint32_t r = 0; r < Placements.size(); ++r)
            placed += Placements[r].alive && Placements[r].key == e.key;
        gRows[rows] = { gNames[rows], gRights[rows], placed > 0 };
        gIds[rows] = e.id;
        if (!gSel)
            gSel = e.id;
        if (e.id == gSel)
            chosen = int(rows);
        ++rows;
    }
    if (!rows)
    {
        u->label("No Models Yet", 1);
        return;
    }
    const int picked = u->dropdown("##model", gRows, int(rows), chosen);
    if (picked != chosen && picked >= 0 && picked < int(rows))
        gSel = gIds[picked], chosen = picked, gPreview.lod = -1;
    if (chosen < 0) // the one shown went (its module stopped): the first
    {
        gSel = gIds[0];
        return;
    }
    const Entry* sel = nullptr;
    for (uint32_t i = 0; i < gEntryCount; ++i)
        if (gEntries[i].id == gSel)
            sel = &gEntries[i];
    if (!sel)
        return;

    // The preview: drag orbits (and stops the turntable), the wheel zooms, a double-click goes back to the front.
    YeaUiCanvas in;
    u->canvas("##preview", 320.0f, &in);
    if (in.w > 0 && in.h > 0)
    {
        if (in.down && (in.dx != 0 || in.dy != 0))
        {
            gPreview.yaw += in.dx * 0.4f;
            gPreview.pitch += in.dy * 0.4f;
            gPreview.pitch = gPreview.pitch < -10.0f ? -10.0f : gPreview.pitch > 85.0f ? 85.0f : gPreview.pitch;
            gPreview.turntable = 0;
        }
        if (in.over && in.wheel != 0)
        {
            float d = gPreview.distance;
            for (float s = in.wheel; s > 0.5f; s -= 1.0f)
                d *= 1.12f;
            for (float s = in.wheel; s < -0.5f; s += 1.0f)
                d /= 1.12f;
            gPreview.distance = d < 0.4f ? 0.4f : d > 6.0f ? 6.0f : d;
        }
        if (in.doubleClick)
            gPreview.yaw = 35.0f, gPreview.pitch = 20.0f, gPreview.distance = 1.35f;
        // Rendered at twice the canvas's size (sharp on high-density screens), at most 2048 across.
        uint32_t w = (uint32_t)(in.w * 2.0f), h = (uint32_t)(in.h * 2.0f);
        if (w > 2048)
            h = h * 2048 / w, w = 2048;
        gPreview.width = w;
        gPreview.height = h;
        gPreview.model = sel->id;
        if (sel->state == 1 && gPreview.outTexture && gPreview.outW == w && gPreview.outH == h)
        {
            yea::canvasImage(gPreview.outTexture, 0, 0, in.w, in.h);
            gPreview.shown = 1;
        }
        else if (sel->state == 2)
            u->text(12.0f, in.h * 0.5f, sel->error, 13.0f, 0.95f, 0.6f, 0.55f, 1.0f);
    }
    YeaUiIcon icons[2] = { { "Resume", "Pause", "Turn", "Hold Still", gPreview.turntable ? 1 : 0 },
                           { "BackToStart", nullptr, "Front View", nullptr, -1 } };
    const int pressed = u->icons("##view", icons, 2);
    if (pressed == 0)
        gPreview.turntable = !gPreview.turntable;
    if (pressed == 1)
        gPreview.yaw = 35.0f, gPreview.pitch = 20.0f, gPreview.distance = 1.35f;

    // The version shown: Auto (as the game picks it at this size), or each one.
    const uint32_t lods = lodCount(*sel);
    if (lods > 1)
    {
        const char* names[9];
        const uint32_t n = lods < 8 ? lods : 8;
        snprintf(gLodNames[0], sizeof gLodNames[0], "Auto");
        names[0] = gLodNames[0];
        for (uint32_t i = 0; i < n; ++i)
            snprintf(gLodNames[i + 1], sizeof gLodNames[i + 1], "LOD %u", i), names[i + 1] = gLodNames[i + 1];
        const int t = u->tabs("##lod", names, int(n + 1), gPreview.lod + 1);
        gPreview.lod = t - 1;
    }
    {
        char tris[24], line[96];
        thousands(gPreview.triangles, tris, sizeof tris);
        snprintf(line, sizeof line, "LOD %u of %u \xC2\xB7 %s triangles", gPreview.lodShown, lods ? lods : 1, tris);
        u->label(line, 1);
    }

    // A skinned model's clips: which one plays in the preview, play and pause, back to its start, its speed, and how long
    // it fades from the one before.
    if (sel->anim && sel->state == 1 && gPreview.animModel == sel->id)
    {
        const PlacementAnim& a = gPreview.anim;
        const uint32_t       clips = sel->skel->clipCount < kClipRows ? sel->skel->clipCount : kClipRows;
        gClipRows[0] = { "Rest Pose", "", a.clip == kNoClip && gPreview.playing };
        for (uint32_t c = 0; c < clips; ++c)
        {
            pretty(sel->clips[c].name, gClipNames[c + 1], sizeof gClipNames[c + 1]);
            if (!gClipNames[c + 1][0])
                snprintf(gClipNames[c + 1], sizeof gClipNames[c + 1], "Clip %u", c + 1);
            snprintf(gClipRights[c + 1], sizeof gClipRights[c + 1], "%.2f s", sel->clips[c].duration);
            gClipRows[c + 1] = { gClipNames[c + 1], gClipRights[c + 1], a.clip == c && gPreview.playing };
        }
        const int chosen = a.clip == kNoClip ? 0 : int(a.clip + 1);
        const int picked = u->dropdown("##clip", gClipRows, int(clips + 1), chosen);
        if (picked != chosen && picked >= 0 && picked <= int(clips))
            gPreview.pickClip = uint32_t(picked) + 1; // 1 the rest pose, clip c at c + 2
        YeaUiIcon transport[2] = { { "Resume", "Pause", "Play", "Pause", gPreview.playing ? 1 : 0 },
                                   { "BackToStart", nullptr, "From the Start", nullptr, -1 } };
        const int t = u->icons("##clipTransport", transport, 2);
        if (t == 0)
            gPreview.playing = !gPreview.playing;
        if (t == 1)
            gPreview.anim.time = 0, gPreview.anim.from = kNoClip;
        u->slider("Speed##clip", &gPreview.speed, 0.0f, 2.0f, 0);
        u->slider("Crossfade (s)##clip", &gPreview.fade, 0.0f, 1.0f, 0);
        if (sel->motionNode != kNoClip) // its clips with their travel (as authored) or in place (root motion on)
            u->toggle("In Place##clip", &gPreview.inPlace);
        // Its morph targets: each one's weight now; dragging one holds it there over the clip, until Follow the Clip.
        if (sel->targets && sel->skel->targetCount)
        {
            u->fold("Morph Targets##morph", &gPreview.morphOpen);
            if (gPreview.morphOpen)
            {
                for (uint32_t t = 0; t < sel->skel->targetCount && t < kRmodelMaxTargets; ++t)
                {
                    char name[64], label[96];
                    pretty(sel->targets[t].name, name, sizeof name);
                    snprintf(label, sizeof label, "%s##morph%u", name[0] ? name : "Target", t);
                    float     w = gPreview.morphHeld[t] ? gPreview.morph[t] : gPreview.morphNow[t];
                    const float lo = fminf(0.0f, sel->targets[t].lo), hi = fmaxf(1.0f, sel->targets[t].hi);
                    if (u->slider(label, &w, lo, hi, 0))
                        gPreview.morph[t] = w, gPreview.morphHeld[t] = 1;
                }
                if (u->button("Follow the Clip##morph", 0, 1))
                    memset(gPreview.morphHeld, 0, sizeof gPreview.morphHeld);
            }
        }
    }

    // Its knobs (renderer_models' each block makes them for its MODEL line).
    char title[80], knob[200];
    pretty(sel->name, title, sizeof title);
    u->space(1);
    u->heading(title);
    for (uint32_t k = 0; k < mdl::kKnobs; ++k)
    {
        snprintf(knob, sizeof knob, "%s##%s.%s_%s", mdl::kKnobSpecs[k].title, sel->owner, sel->name, mdl::kKnobSpecs[k].suffix);
        u->knob(knob);
    }
}
} // namespace renderer_models
