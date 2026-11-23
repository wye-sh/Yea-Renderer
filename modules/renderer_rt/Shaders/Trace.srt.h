#pragma once

// Both passes' bindings (renderer_rt.cpp binds every slot in each; a pass leaves the other's alone).
BEGIN_SRT(RtSrt)
    BEGIN_SRT_SET(Persistent) // as the core's: a sampler in a set of its own, written once
        DECL_SAMPLER(Persistent, SamplerComparisonState, gShadowCmp)
        DECL_SAMPLER(Persistent, SamplerState, gSkySampler) // trilinear, clamped: the physical sky's image
    END_SRT_SET(Persistent)
    BEGIN_SRT_SET(PerFrame)
        DECL_RAYTRACING(PerFrame, RaytracingAccelerationStructure, gScene) // first: Vulkan binds it at 0
        DECL_CBUFFER(PerFrame, CBUFFER(RtData), gRt)
        DECL_TEXTURE(PerFrame, Depth2D(float), gDepth)
        DECL_TEXTURE(PerFrame, Tex2D(float4), gSurface)
        DECL_TEXTURE(PerFrame, Tex2D(float4), gPrevious)
        DECL_BUFFER(PerFrame, Buffer(InstanceData), gInstances)
        DECL_BUFFER(PerFrame, Buffer(uint4), gRtInstances) // per top-level instance; then glass's last-frame rows
        DECL_BUFFER(PerFrame, Buffer(MaterialData), gMaterials)
        DECL_BUFFER(PerFrame, Buffer(LightData), gLights)
        DECL_BUFFER(PerFrame, Buffer(uint), gAttributes) // RENDER_ATTRIBUTE_STRIDE: 5 u32 a vertex, [0] the normal
        DECL_BUFFER(PerFrame, Buffer(uint), gIndices)
        DECL_BUFFER(PerFrame, Buffer(float), gExposure)
        DECL_TEXTURE(PerFrame, Tex2D(float4), gHalfIn)   // resolve: the half-resolution traces
        DECL_TEXTURE(PerFrame, Tex2D(float4), gFullIn)   // resolve: the pixels traced at full resolution
        DECL_TEXTURE(PerFrame, Tex2D(float2), gMotion)   // resolve: last frame's motion vectors (the forward pass writes them later)
        DECL_TEXTURE(PerFrame, Tex2D(float4), gHistory)  // resolve: last frame's accumulated reflections
        DECL_TEXTURE(PerFrame, Tex2D(float2), gHistoryZ) // x the view depth they were for, y their samples
        DECL_RWBUFFER(PerFrame, RWBuffer(uint), gTiles)  // per 8 x 8 tile, 6 flags (RtData tiles and zero): any
                                                         // pixel to trace (two frames' worth), and whether the history
                                                         // textures' tile is all zero (each before and after its write)
        DECL_RWTEXTURE(PerFrame, RWTex2D(half4), gOutHalf) // trace: the half-resolution samples
        DECL_RWTEXTURE(PerFrame, RWTex2D(half4), gOutFull) // trace: the pixels traced at full resolution
        DECL_RWTEXTURE(PerFrame, RWTex2D(half4), gOut)     // resolve: this frame's accumulated reflections
        DECL_RWTEXTURE(PerFrame, RWTex2D(half2), gOutZ)
        DECL_BUFFER(PerFrame, Buffer(ShadowView), gShadowViews) // trace: renderer_shadows' views (a dummy without them)
        DECL_TEXTURE(PerFrame, Depth2D(float), gLocalAtlas)     // and its local lights' atlas
        DECL_TEXTURE(PerFrame, Tex2D(float4), gHalfRayIn)        // reconstruct: the half-resolution rays' directions, w 1 / their pdf
        DECL_RWTEXTURE(PerFrame, RWTex2D(half4), gOutHalfRay)    // trace: the same
        DECL_TEXTURE(PerFrame, Tex2D(float4), gSkyEnv)           // renderer_sky (sky.x 1): its reflection image
        DECL_BUFFER(PerFrame, Buffer(float4), gSkySh)            // and its ambient (9 SH)
        DECL_BUFFER(PerFrame, Buffer(MaterialMapData), gMaterialMaps) // cut-outs (Cutout.h.fsl): the core's material maps
        DECL_ARRAY_TEXTURES(PerFrame, Tex2D(float4), gTextures, MAX_TEXTURES) // and its texture maps (bound when they change)
        DECL_BUFFER(PerFrame, Buffer(float4), gGiProbes)         // renderer_gi (giCounts.x): its probes' offsets and states
        DECL_TEXTURE(PerFrame, Tex2D(float4), gGiIrradiance)     // last frame's probes (giCounts.w): irradiance / pi
        DECL_TEXTURE(PerFrame, Tex2D(float4), gGiDistance)       // and their distances
        DECL_RWBUFFER(PerFrame, RWBuffer(float4), gGiRayOut)     // the GI pass: each probe ray's radiance and distance
        DECL_BUFFER(PerFrame, Buffer(uint), gPositions)          // hits' texture mips: float3 a vertex (RENDER_POSITION_STRIDE)
        DECL_BUFFER(PerFrame, Buffer(float4), gRefraction)       // glass (frame refraction): two float4 a material
        DECL_BUFFER(PerFrame, Buffer(float2), gDfg)              // the core's split-sum table (frame dfg)
        DECL_BUFFER(PerFrame, Buffer(DecalData), gDecals)        // frame worldDecals (decals.x), decalCells, decalItems
        DECL_BUFFER(PerFrame, Buffer(uint2), gDecalCells)
        DECL_BUFFER(PerFrame, Buffer(uint), gDecalItems)
        DECL_TEXTURE(PerFrame, Tex2D(float4), gRefractHistory)    // the refraction pass (frame rtRefraction): last frame's
        DECL_TEXTURE(PerFrame, Tex2D(float), gRefractHistoryN)    // and the samples each pixel of it holds (0: none)
        DECL_RWTEXTURE(PerFrame, RWTex2D(half4), gOutRefract)     // this frame's rtRefraction
        DECL_RWTEXTURE(PerFrame, RWTex2D(half4), gOutRefractLand) // rtRefractLanding
        DECL_RWTEXTURE(PerFrame, RWTex2D(half), gOutRefractN)
    END_SRT_SET(PerFrame)
END_SRT(RtSrt)
