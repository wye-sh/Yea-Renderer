// The Model Viewer's preview, tonemapped per sample and resolved into renderer_models' target (preview.cpp).
#pragma once

BEGIN_SRT(PreviewResolveSrt)
    BEGIN_SRT_SET(PerFrame)
        DECL_CBUFFER(PerFrame, CBUFFER(PreviewData), gPreview)
        DECL_TEXTURE(PerFrame, Tex2DMS(float4, 4), gHdr)
        DECL_TEXTURE(PerFrame, Tex2D(float4), gTony)
    END_SRT_SET(PerFrame)
END_SRT(PreviewResolveSrt)
