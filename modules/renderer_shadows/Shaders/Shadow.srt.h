#pragma once

BEGIN_SRT(ShadowSrt)
    BEGIN_SRT_SET(PerFrame)
        DECL_BUFFER(PerFrame, Buffer(InstanceData), gInstances)
        DECL_BUFFER(PerFrame, Buffer(ShadowView), gViews)
        DECL_BUFFER(PerFrame, Buffer(MaterialMapData), gMaterialMaps) // the frame's materialMaps (alpha-tested casters)
        DECL_ARRAY_TEXTURES(PerFrame, Tex2D(float4), gTextures, MAX_TEXTURES)
    END_SRT_SET(PerFrame)
END_SRT(ShadowSrt)
