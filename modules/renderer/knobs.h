// The module's knobs that differ from their module.yea defaults, logged as one line ("<module> knobs off their
// defaults: name=value ..."), so the values a game runs with reach the hub's reports without anyone reading F1: at load,
// then once the values have held still for 2 s after a change, at most every 10 s (Yea passes 10 lines a minute a
// module). Each frame costs a hash of the values; the line is only written when it is logged. Choices by option name.
// The same file sits in every renderer module and renderer_example; keep the copies identical.
#pragma once
#include <yea/base.h>
#include <yea/module.h>
#include <stdio.h>
#include <string.h>

extern "C" const YeaModuleDesc* yea_module(void);

namespace knobs
{
inline uint64_t gLogged, gSeen; // the values' hash last logged, and last seen
inline double   gChanged, gLast; // when they last changed, when last logged

inline uint32_t valueSize(const YeaSetting& s)
{
    return s.type == YEA_SETTING_BOOL ? (uint32_t)sizeof(bool) : s.type == YEA_SETTING_COLOR ? 12u : 4u;
}

inline uint64_t hashValues(const YeaModuleDesc* d)
{
    uint64_t h = 1469598103934665603ull;
    for (uint32_t i = 0; i < d->settingCount; ++i)
    {
        const unsigned char* p = (const unsigned char*)d->settings[i].value;
        for (uint32_t b = 0, n = valueSize(d->settings[i]); b < n; ++b)
            h = (h ^ p[b]) * 1099511628211ull;
    }
    return h;
}

inline void logLine(const YeaModuleDesc* d)
{
    char line[2048];
    int  n = snprintf(line, sizeof line, "%s knobs off their defaults:", d->name);
    const int head = n;
    for (uint32_t i = 0; i < d->settingCount && n < (int)sizeof line - 128; ++i)
    {
        const YeaSetting& s = d->settings[i];
        char              v[96];
        bool              off = false;
        if (s.type == YEA_SETTING_FLOAT)
        {
            float x = *(const float*)s.value;
            off = x != s.def[0];
            snprintf(v, sizeof v, "%g", x);
        }
        else if (s.type == YEA_SETTING_INT)
        {
            int32_t x = *(const int32_t*)s.value;
            off = x != (int32_t)s.def[0];
            snprintf(v, sizeof v, "%d", x);
            if ((s.pad & 2) && x >= 0 && x <= (int32_t)s.max)
            {
                // A choice: its options follow the name (and the category, when it has one).
                const char* p = s.name + strlen(s.name) + 1;
                if (s.pad & 1)
                    p += strlen(p) + 1;
                for (int32_t k = 0; k < x; ++k)
                    p += strlen(p) + 1;
                snprintf(v, sizeof v, "%s", p);
            }
        }
        else if (s.type == YEA_SETTING_BOOL)
        {
            bool x = *(const bool*)s.value;
            off = x != (s.def[0] != 0.0f);
            snprintf(v, sizeof v, "%s", x ? "true" : "false");
        }
        else
        {
            const float* c = (const float*)s.value;
            off = c[0] != s.def[0] || c[1] != s.def[1] || c[2] != s.def[2];
            snprintf(v, sizeof v, "%g,%g,%g", c[0], c[1], c[2]);
        }
        if (off)
            n += snprintf(line + n, sizeof line - n, " %s=%s", s.name, v);
    }
    if (n == head)
        snprintf(line + n, sizeof line - n, " none");
    yea::log(line);
}

// loaded: from load() (logs at once); else each frame.
inline void check(bool loaded)
{
    const YeaModuleDesc* d = yea_module();
    const uint64_t       h = hashValues(d);
    const double         now = yea::host()->time();
    if (loaded)
    {
        logLine(d);
        gLogged = gSeen = h;
        gChanged = gLast = now;
        return;
    }
    if (h != gSeen)
    {
        gSeen = h;
        gChanged = now;
    }
    else if (h != gLogged && now - gChanged >= 2.0 && now - gLast >= 10.0)
    {
        logLine(d);
        gLogged = h;
        gLast = now;
    }
}
} // namespace knobs
