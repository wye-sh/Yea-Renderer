#pragma once

BEGIN_SRT(VrsSrt)
    BEGIN_SRT_SET(PerFrame)
        DECL_CBUFFER(PerFrame, CBUFFER(VrsData), gVrs)
        DECL_TEXTURE(PerFrame, Tex2D(float4), gPrevious)
        DECL_TEXTURE(PerFrame, Tex2D(float2), gMotion)
        DECL_TEXTURE(PerFrame, Depth2D(float), gDepth)
        DECL_BUFFER(PerFrame, Buffer(float), gExposure)
        DECL_RWTEXTURE(PerFrame, RWTex2D(uint), gRate)
        DECL_RWTEXTURE(PerFrame, RWTex2D(uint), gRateCopy)
    END_SRT_SET(PerFrame)
END_SRT(VrsSrt)
