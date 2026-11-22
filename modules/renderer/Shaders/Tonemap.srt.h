// The tonemap pass: the resolved HDR image tonemapped and sharpened once per pixel, packed for the composite.
#pragma once

BEGIN_SRT(TonemapSrt)
    BEGIN_SRT_SET(PerFrame)
        DECL_CBUFFER(PerFrame, CBUFFER(CompositeData), gComposite)
        DECL_TEXTURE(PerFrame, Tex2D(float4), gHdr)
        DECL_BUFFER(PerFrame, Buffer(float), gExposure)
        DECL_TEXTURE(PerFrame, Tex2D(float4), gTony)
        DECL_RWTEXTURE(PerFrame, RWTex2D(uint), gLdrOut)
    END_SRT_SET(PerFrame)
END_SRT(TonemapSrt)
