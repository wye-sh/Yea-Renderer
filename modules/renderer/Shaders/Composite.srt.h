// Composite: the HDR target, tonemapped per sample, into the frame.
#pragma once

BEGIN_SRT(CompositeSrt)
    BEGIN_SRT_SET(PerFrame)
        DECL_CBUFFER(PerFrame, CBUFFER(CompositeData), gComposite)
#if SAMPLE_COUNT > 1
        DECL_TEXTURE(PerFrame, Tex2DMS(float4, SAMPLE_COUNT), gHdr)
#else
        DECL_TEXTURE(PerFrame, Tex2D(float4), gHdr)
#endif
        DECL_BUFFER(PerFrame, Buffer(float), gExposure)
        DECL_TEXTURE(PerFrame, Tex2D(float), gShadowAtlas)
        DECL_TEXTURE(PerFrame, Tex2D(uint), gRates)
        DECL_TEXTURE(PerFrame, Tex2D(float4), gTony)
        DECL_TEXTURE(PerFrame, Tex2D(uint), gLdr) // the tonemap pass's packed output (Tonemap.h.fsl)
    END_SRT_SET(PerFrame)
END_SRT(CompositeSrt)
