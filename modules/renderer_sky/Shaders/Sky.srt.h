#pragma once

// One layout for every sky pass: each binds what it reads (unused slots hold a valid stand-in).
BEGIN_SRT(SkySrt)
    BEGIN_SRT_SET(PerFrame)
        DECL_CBUFFER(PerFrame, CBUFFER(SkyData), gSky)
        DECL_TEXTURE(PerFrame, Tex2D(float4), gTransmittance)
        DECL_TEXTURE(PerFrame, Tex2D(float4), gMultiScatter)
        DECL_TEXTURE(PerFrame, Tex2D(float4), gSkyView)
        DECL_TEXTURE(PerFrame, Tex2D(float4), gEnvSource) // the reflection image's mip 0 (the SH pass)
        DECL_RWTEXTURE(PerFrame, RWTex2D(half4), gOut) // RGBA16F
        DECL_RWTEXTURE(PerFrame, RWTex3D(half4), gOutVolume) // RGBA16F
        DECL_RWBUFFER(PerFrame, RWBuffer(float4), gShOut)
    END_SRT_SET(PerFrame)
END_SRT(SkySrt)
