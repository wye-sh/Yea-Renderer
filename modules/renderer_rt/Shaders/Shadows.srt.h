#pragma once

// The ray-traced shadow passes' bindings: every slot is bound in each pass's set; a pass leaves the others alone.
// The textures are 2D arrays with a layer per traced shadow (the sun and up to traced_lights lights).
BEGIN_SRT(ShadowSrt)
    BEGIN_SRT_SET(PerFrame)
        DECL_RAYTRACING(PerFrame, RaytracingAccelerationStructure, gScene) // first: Vulkan binds it at 0
        DECL_CBUFFER(PerFrame, CBUFFER(ShadowData), gSd)
        DECL_TEXTURE(PerFrame, Depth2D(float), gDepth)
        DECL_TEXTURE(PerFrame, Tex2D(float4), gSurface)        // y z w the normal
        DECL_TEXTURE(PerFrame, Tex2D(float2), gMotion)         // uv now - uv last frame
        DECL_BUFFER(PerFrame, Buffer(LightData), gLights)
        DECL_BUFFER(PerFrame, Buffer(LightShape), gShapes)
        DECL_RWTEXTURE(PerFrame, RWTex2DArray(half), gRawOut)  // trace: the blocker's distance (0: lit; one ray is all or nothing)
        DECL_TEXTURE(PerFrame, Tex2DArray(float), gRaw)        // temporal: the trace's
        DECL_TEXTURE(PerFrame, Tex2DArray(float2), gHistory)   // temporal: last frame's; filter: this frame's
        DECL_RWTEXTURE(PerFrame, RWTex2DArray(half2), gHistoryOut) // temporal: x visibility, y blocker distance
        DECL_TEXTURE(PerFrame, Tex2DArray(float), gSamples)    // as gHistory: the samples it holds
        DECL_RWTEXTURE(PerFrame, RWTex2DArray(half), gSamplesOut)
        DECL_TEXTURE(PerFrame, Tex2D(float), gPrevZ)           // temporal: last frame's view depth
        DECL_RWTEXTURE(PerFrame, RWTex2D(float), gZOut)        // temporal (first shadow): this frame's
        DECL_RWTEXTURE(PerFrame, RWTex2DArray(half), gFinal)   // filter: rtShadows (the rays' texture: free by then)
    END_SRT_SET(PerFrame)
END_SRT(ShadowSrt)
