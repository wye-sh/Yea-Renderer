#pragma once

// The particles' draw, after the forward pass: camera-facing quads into the HDR image and the reactive mask.
BEGIN_SRT(DrawSrt)
    BEGIN_SRT_SET(PerFrame)
        DECL_CBUFFER(PerFrame, CBUFFER(PartFrame), gFrame)
        DECL_BUFFER(PerFrame, Buffer(Emitter), gEmitters)
        DECL_BUFFER(PerFrame, Buffer(Particle), gParticles)
        DECL_BUFFER(PerFrame, Buffer(float4), gLit)
        DECL_BUFFER(PerFrame, Buffer(uint2), gList)
        DECL_BUFFER(PerFrame, Buffer(uint), gCounters)
        DECL_TEXTURE(PerFrame, Depth2D(float), gDepth)
        DECL_TEXTURE(PerFrame, Tex3D(float4), gFogVolume)
    END_SRT_SET(PerFrame)
END_SRT(DrawSrt)
