// Auto exposure: a luminance histogram of the HDR target, then the adapted exposure.
#pragma once

BEGIN_SRT(ExposureSrt)
    BEGIN_SRT_SET(PerFrame)
        DECL_CBUFFER(PerFrame, CBUFFER(ExposureData), gExposureData)
#if SAMPLE_COUNT > 1
        DECL_TEXTURE(PerFrame, Tex2DMS(float4, SAMPLE_COUNT), gHdr)
#else
        DECL_TEXTURE(PerFrame, Tex2D(float4), gHdr)
#endif
        DECL_RWBUFFER(PerFrame, RWBuffer(uint), gHistogram)
        DECL_RWBUFFER(PerFrame, RWBuffer(float), gExposureRW)
    END_SRT_SET(PerFrame)
END_SRT(ExposureSrt)
