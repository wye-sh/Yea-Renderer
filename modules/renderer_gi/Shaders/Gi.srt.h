#pragma once

// Every renderer_gi compute pass binds the same set.
BEGIN_SRT(GiSrt)
    BEGIN_SRT_SET(PerFrame)
        DECL_CBUFFER(PerFrame, CBUFFER(GiData), gGi)
        DECL_RWBUFFER(PerFrame, RWBuffer(float4), gRaysRW)    // frame.h giRayOut (the raster fallback writes it)
        DECL_RWTEXTURE(PerFrame, RWTex2D(half4), gIrrOut)     // irradiance atlas, RGBA16F
        DECL_RWTEXTURE(PerFrame, RWTex2D(half4), gDistOut)    // distance atlas, RGBA16F (xy mean, mean squared)
        DECL_RWBUFFER(PerFrame, RWBuffer(float4), gProbesRW)  // frame.h giProbeData
        DECL_TEXTURE(PerFrame, Tex2D(float4), gFaces)         // the raster fallback's captures (Capture.frag.fsl)
        DECL_BUFFER(PerFrame, Buffer(float4), gSkySh)         // renderer_sky's ambient (raster.w & 4)
    END_SRT_SET(PerFrame)
END_SRT(GiSrt)
