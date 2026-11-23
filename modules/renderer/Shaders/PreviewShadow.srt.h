// The Model Viewer's preview, its key light's shadow map (preview.cpp).
#pragma once

BEGIN_SRT(PreviewShadowSrt)
    BEGIN_SRT_SET(PerFrame)
        DECL_CBUFFER(PerFrame, CBUFFER(PreviewData), gPreview)
        DECL_BUFFER(PerFrame, Buffer(InstanceData), gInstances)
    END_SRT_SET(PerFrame)
END_SRT(PreviewShadowSrt)
