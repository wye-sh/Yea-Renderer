// With MSAA: the scene colour chain's level 0 resolved from the multisampled HDR target, and the scene's depth resolved
// beside it (refraction's landing test reads it as the see-through pass's gSceneDepth).
#pragma once

BEGIN_SRT(ChainMsSrt)
    BEGIN_SRT_SET(PerFrame)
        DECL_CBUFFER(PerFrame, CBUFFER(ChainData), gChain)
        DECL_TEXTURE(PerFrame, Tex2DMS(float4, SAMPLE_COUNT), gHdrMs)
        DECL_TEXTURE(PerFrame, Tex2DMS(float, SAMPLE_COUNT), gDepthMs)
        DECL_RWTEXTURE(PerFrame, RWTex2D(half4), gDst)      // the chain's level 0
        DECL_RWTEXTURE(PerFrame, RWTex2D(float), gDepthOut) // R32F: each pixel's nearest sample (reversed-Z: the largest)
    END_SRT_SET(PerFrame)
END_SRT(ChainMsSrt)
