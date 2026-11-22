#pragma once

BEGIN_SRT(GtaoSrt)
    BEGIN_SRT_SET(PerFrame)
        DECL_CBUFFER(PerFrame, CBUFFER(AoData), gAo)
#if SAMPLE_COUNT > 1
        DECL_TEXTURE(PerFrame, Depth2DMS(float, SAMPLE_COUNT), gDepth)
#else
        DECL_TEXTURE(PerFrame, Depth2D(float), gDepth)
#endif
        DECL_TEXTURE(PerFrame, Tex2D(float2), gIn)   // x visibility, y view depth (m)
        DECL_RWTEXTURE(PerFrame, RWTex2D(half2), gOut) // the same
    END_SRT_SET(PerFrame)
END_SRT(GtaoSrt)
