#pragma once

BEGIN_SRT(RtSrt)
    BEGIN_SRT_SET(PerFrame)
        DECL_RAYTRACING(PerFrame, RaytracingAccelerationStructure, gScene) // first: Vulkan binds it at 0
        DECL_CBUFFER(PerFrame, CBUFFER(RtData), gRt)
        DECL_TEXTURE(PerFrame, Depth2D(float), gDepth)
        DECL_TEXTURE(PerFrame, Tex2D(float4), gSurface)
        DECL_TEXTURE(PerFrame, Tex2D(float4), gPrevious)
        DECL_BUFFER(PerFrame, Buffer(InstanceData), gInstances)
        DECL_BUFFER(PerFrame, Buffer(uint4), gRtInstances)
        DECL_BUFFER(PerFrame, Buffer(MaterialData), gMaterials)
        DECL_BUFFER(PerFrame, Buffer(LightData), gLights)
        DECL_BUFFER(PerFrame, Buffer(uint), gVertices)
        DECL_BUFFER(PerFrame, Buffer(uint), gIndices)
        DECL_BUFFER(PerFrame, Buffer(float), gExposure)
        DECL_RWTEXTURE(PerFrame, RWTex2D(half4), gOut)
    END_SRT_SET(PerFrame)
END_SRT(RtSrt)
