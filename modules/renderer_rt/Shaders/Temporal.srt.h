// The reflections' temporal accumulation.
#pragma once

BEGIN_SRT(TemporalSrt)
    BEGIN_SRT_SET(PerFrame)
        DECL_CBUFFER(PerFrame, CBUFFER(RtData), gRt)
        DECL_TEXTURE(PerFrame, Depth2D(float), gDepth)
        DECL_TEXTURE(PerFrame, Tex2D(float4), gSurface)
        DECL_TEXTURE(PerFrame, Tex2D(float2), gMotion)   // last frame's motion vectors (the forward pass writes them later)
        DECL_TEXTURE(PerFrame, Tex2D(float4), gCurrent)  // this frame's trace (full size)
        DECL_TEXTURE(PerFrame, Tex2D(float4), gHistory)  // last frame's accumulated reflections
        DECL_TEXTURE(PerFrame, Tex2D(float), gHistoryZ)  // and the view depth they were for
        DECL_TEXTURE(PerFrame, Tex2D(float4), gHalfIn)   // the half-resolution trace, when it ran
        DECL_BUFFER(PerFrame, Buffer(float), gExposure)
        DECL_RWTEXTURE(PerFrame, RWTex2D(half4), gOut)
        DECL_RWTEXTURE(PerFrame, RWTex2D(float), gOutZ)
    END_SRT_SET(PerFrame)
END_SRT(TemporalSrt)
