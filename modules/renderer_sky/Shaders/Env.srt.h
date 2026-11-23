#pragma once

// The reflection image's passes (fullscreen, one mip each).
BEGIN_SRT(EnvSrt)
    BEGIN_SRT_SET(PerFrame)
        DECL_CBUFFER(PerFrame, CBUFFER(SkyData), gSky)
        DECL_TEXTURE(PerFrame, Tex2D(float4), gSkyView)
    END_SRT_SET(PerFrame)
END_SRT(EnvSrt)
