// The see-through pass's scene colour chain (refraction): level 0 a copy of the scene drawn so far, each next level half
// the size, filtered.
#pragma once

BEGIN_SRT(ChainSrt)
    BEGIN_SRT_SET(PerFrame)
        DECL_CBUFFER(PerFrame, CBUFFER(ChainData), gChain)
        DECL_TEXTURE(PerFrame, Tex2D(float4), gHdrIn)      // level 0's source: the HDR target
        DECL_RWTEXTURE(PerFrame, RWTex2D(half4), gPrev)    // the level read (the chain at the level above)
        DECL_RWTEXTURE(PerFrame, RWTex2D(half4), gDst)     // the level written
    END_SRT_SET(PerFrame)
END_SRT(ChainSrt)
