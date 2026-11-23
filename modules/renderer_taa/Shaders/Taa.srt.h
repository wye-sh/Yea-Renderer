// The TAA resolve.
#pragma once

BEGIN_SRT(TaaSrt)
    BEGIN_SRT_SET(PerFrame)
        DECL_CBUFFER(PerFrame, CBUFFER(TaaData), gTaa)
        DECL_TEXTURE(PerFrame, Tex2D(float4), gHdr)
        DECL_TEXTURE(PerFrame, Tex2D(float2), gMotion)
        DECL_TEXTURE(PerFrame, Depth2D(float), gDepth)
        DECL_TEXTURE(PerFrame, Tex2D(float4), gHistory)
        DECL_BUFFER(PerFrame, Buffer(float), gExposure)
        DECL_RWTEXTURE(PerFrame, RWTex2D(half4), gOut) // RGBA16F
    END_SRT_SET(PerFrame)
END_SRT(TaaSrt)
