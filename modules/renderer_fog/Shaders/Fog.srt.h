#pragma once

// Both fog passes: LIGHT reads last frame's light (gIn) and writes this frame's (gOut); INTEGRATE reads this frame's
// light (gIn) and writes the volume the core reads (gOut).
BEGIN_SRT(FogSrt)
    BEGIN_SRT_SET(Persistent)
        DECL_SAMPLER(Persistent, SamplerComparisonState, gShadowCmp)
    END_SRT_SET(Persistent)
    BEGIN_SRT_SET(PerFrame)
        DECL_CBUFFER(PerFrame, CBUFFER(FogData), gFog)
        DECL_BUFFER(PerFrame, Buffer(FogLight), gLights)
        DECL_BUFFER(PerFrame, Buffer(FogLightShape), gShapes)
        DECL_BUFFER(PerFrame, Buffer(FogShadowView), gViews)
        DECL_BUFFER(PerFrame, Buffer(FogVolume), gVolumes)
        DECL_BUFFER(PerFrame, Buffer(float4), gSkySh)
        DECL_TEXTURE(PerFrame, Depth2D(float), gSunAtlas)
        DECL_TEXTURE(PerFrame, Depth2D(float), gLocalAtlas)
        DECL_TEXTURE(PerFrame, Tex3D(float4), gIn)
        DECL_RWTEXTURE(PerFrame, RWTex3D(half4), gOut) // RGBA16F
    END_SRT_SET(PerFrame)
END_SRT(FogSrt)
