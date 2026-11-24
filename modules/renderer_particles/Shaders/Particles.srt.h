#pragma once

// Every compute pass of the particles (init, emit, simulate, pad, sort, split): each binds the whole set, in its own slot.
BEGIN_SRT(PartSrt)
    BEGIN_SRT_SET(Persistent)
        DECL_SAMPLER(Persistent, SamplerComparisonState, gShadowCmp)
    END_SRT_SET(Persistent)
    BEGIN_SRT_SET(PerFrame)
        DECL_CBUFFER(PerFrame, CBUFFER(PartFrame), gFrame)
        DECL_CBUFFER(PerFrame, CBUFFER(SortStep), gStep)
        DECL_BUFFER(PerFrame, Buffer(Emitter), gEmitters)
        DECL_BUFFER(PerFrame, Buffer(Spawn), gSpawn)
        DECL_BUFFER(PerFrame, Buffer(PartLight), gLights)
        DECL_BUFFER(PerFrame, Buffer(PartLightShape), gShapes)
        DECL_BUFFER(PerFrame, Buffer(uint), gZBins)
        DECL_BUFFER(PerFrame, Buffer(PartShadowView), gViews)
        DECL_BUFFER(PerFrame, Buffer(float4), gSkySh)
        DECL_TEXTURE(PerFrame, Tex2D(float4), gGiIrradiance) // renderer_gi's probes (GiSample.h.fsl), when the GI flag is set
        DECL_TEXTURE(PerFrame, Tex2D(float4), gGiDistance)
        DECL_BUFFER(PerFrame, Buffer(float4), gGiProbes)
        DECL_TEXTURE(PerFrame, Depth2D(float), gSunAtlas)
        DECL_TEXTURE(PerFrame, Depth2D(float), gLocalAtlas)
        DECL_RWBUFFER(PerFrame, RWBuffer(Particle), gParticles)
        DECL_RWBUFFER(PerFrame, RWBuffer(float4), gLit)     // each drawn particle's light: rgb radiance per unit albedo
        DECL_RWBUFFER(PerFrame, RWBuffer(uint), gDead)      // free pool slots, gCounters[0] of them
        DECL_RWBUFFER(PerFrame, RWBuffer(uint2), gList)     // to draw: x the depth's bits, y the slot; gCounters[1] of them
        DECL_RWBUFFER(PerFrame, RWBuffer(uint), gCounters)  // [0] free slots, [1] particles to draw
        DECL_RWBUFFER(PerFrame, RWBuffer(uint), gRunArgs)   // SPLIT: per draw run its indirect draw (vertices, 1, 0, 0)
        DECL_RWBUFFER(PerFrame, RWBuffer(uint), gRunStart)  // and its first list entry
        DECL_RWBUFFER(PerFrame, RWBuffer(uint), gRunRect)   // and its screen rectangle (render pixels: x0, y0, x1, y1; BOUNDS)
    END_SRT_SET(PerFrame)
END_SRT(PartSrt)
