#pragma once

BEGIN_SRT(SsrDepthSrt)
    BEGIN_SRT_SET(PerFrame)
        DECL_CBUFFER(PerFrame, CBUFFER(SsrDepthData), gSsr)
        DECL_TEXTURE(PerFrame, Depth2D(float), gDepth)
        DECL_RWTEXTURE(PerFrame, RWTex2D(float), gOut)
    END_SRT_SET(PerFrame)
END_SRT(SsrDepthSrt)
