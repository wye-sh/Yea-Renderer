#pragma once

// The local atlas's moving-caster refresh (renderer_shadows.cpp, 6.6.5): a face's still casters, cached in the static
// atlas, copied into its square before its moving casters draw over them.
BEGIN_SRT(ShadowCopySrt)
    BEGIN_SRT_SET(PerFrame)
        DECL_TEXTURE(PerFrame, Tex2D(float), gStatic)
    END_SRT_SET(PerFrame)
END_SRT(ShadowCopySrt)
