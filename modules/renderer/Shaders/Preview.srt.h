// The Model Viewer's preview, the model and its floor (preview.cpp).
#pragma once

BEGIN_SRT(PreviewSrt)
    BEGIN_SRT_SET(Persistent)
        DECL_SAMPLER(Persistent, SamplerComparisonState, gShadowCmp)
        DECL_BUFFER(Persistent, Buffer(float2), gDfg)
    END_SRT_SET(Persistent)
    BEGIN_SRT_SET(PerFrame)
        DECL_CBUFFER(PerFrame, CBUFFER(PreviewData), gPreview)
        DECL_BUFFER(PerFrame, Buffer(InstanceData), gInstances)
        DECL_BUFFER(PerFrame, Buffer(MaterialData), gMaterials)
        DECL_TEXTURE(PerFrame, Depth2D(float), gShadowMap)
        DECL_BUFFER(PerFrame, Buffer(MaterialMapData), gMaterialMaps) // each part's, as the forward pass's
        DECL_ARRAY_TEXTURES(PerFrame, Tex2D(float4), gTextures, MAX_TEXTURES)
    END_SRT_SET(PerFrame)
END_SRT(PreviewSrt)
