#pragma once

// Both passes' bindings (renderer_rt.cpp binds every slot in each; a pass leaves the other's alone).
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
        DECL_TEXTURE(PerFrame, Tex2D(float4), gHalfIn)   // resolve: the half-resolution traces
        DECL_TEXTURE(PerFrame, Tex2D(float4), gFullIn)   // resolve: the pixels traced at full resolution
        DECL_TEXTURE(PerFrame, Tex2D(float2), gMotion)   // resolve: last frame's motion vectors (the forward pass writes them later)
        DECL_TEXTURE(PerFrame, Tex2D(float4), gHistory)  // resolve: last frame's accumulated reflections
        DECL_TEXTURE(PerFrame, Tex2D(float), gHistoryZ)  // and the view depth they were for
        DECL_RWBUFFER(PerFrame, RWBuffer(uint), gTiles)  // per 8 x 8 tile, 6 flags (RtData tiles and zero): any
                                                         // pixel to trace (two frames' worth), and whether the history
                                                         // textures' tile is all zero (each before and after its write)
        DECL_RWTEXTURE(PerFrame, RWTex2D(half4), gOutHalf) // trace: the half-resolution samples
        DECL_RWTEXTURE(PerFrame, RWTex2D(half4), gOutFull) // trace: the pixels traced at full resolution
        DECL_RWTEXTURE(PerFrame, RWTex2D(half4), gOut)     // resolve: this frame's accumulated reflections
        DECL_RWTEXTURE(PerFrame, RWTex2D(float), gOutZ)
    END_SRT_SET(PerFrame)
END_SRT(RtSrt)
