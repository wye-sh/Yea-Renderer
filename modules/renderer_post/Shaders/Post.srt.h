#pragma once

// One layout for every post pass: each binds what it reads (unused slots hold a valid stand-in).
BEGIN_SRT(PostSrt)
    BEGIN_SRT_SET(PerFrame)
        DECL_CBUFFER(PerFrame, CBUFFER(PostData), gPost)
        DECL_TEXTURE(PerFrame, Tex2D(float4), gSrc)
        DECL_TEXTURE(PerFrame, Tex2D(float4), gSrc2)
        DECL_TEXTURE(PerFrame, Tex2D(float4), gSrc3)
        DECL_TEXTURE(PerFrame, Tex2D(float4), gFocus) // 2 x 2, texel (0, 0): x the focus distance (depth of field)
        DECL_TEXTURE(PerFrame, Tex2D(float4), gLut)
        DECL_TEXTURE(PerFrame, Depth2D(float), gDepth)
        DECL_TEXTURE(PerFrame, Tex2D(float2), gMotion)
        DECL_BUFFER(PerFrame, Buffer(float), gExposure)
        DECL_RWTEXTURE(PerFrame, RWTex2D(half4), gOut) // RGBA16F
        DECL_RWTEXTURE(PerFrame, RWTex2D(half4), gOut2) // a second output (depth of field's foreground layer)
    END_SRT_SET(PerFrame)
END_SRT(PostSrt)
