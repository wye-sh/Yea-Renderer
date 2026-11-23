#pragma once

BEGIN_SRT(CaptureSrt)
    BEGIN_SRT_SET(PerFrame)
        DECL_CBUFFER(PerFrame, CBUFFER(CaptureData), gCapture)
        DECL_BUFFER(PerFrame, Buffer(InstanceData), gInstances)
        DECL_BUFFER(PerFrame, Buffer(CaptureView), gViews)
        DECL_BUFFER(PerFrame, Buffer(MaterialData), gMaterials)
        DECL_BUFFER(PerFrame, Buffer(LightData), gLights)
        DECL_BUFFER(PerFrame, Buffer(float4), gSkySh) // renderer_sky's ambient (counts.y 1), else unread
        DECL_BUFFER(PerFrame, Buffer(MaterialMapData), gMaterialMaps) // the frame's materialMaps
        DECL_BUFFER(PerFrame, Buffer(DecalData), gDecals)  // frame worldDecals (counts.z), decalCells, decalItems
        DECL_BUFFER(PerFrame, Buffer(uint2), gDecalCells)
        DECL_BUFFER(PerFrame, Buffer(uint), gDecalItems)
        DECL_ARRAY_TEXTURES(PerFrame, Tex2D(float4), gTextures, MAX_TEXTURES)
    END_SRT_SET(PerFrame)
END_SRT(CaptureSrt)
