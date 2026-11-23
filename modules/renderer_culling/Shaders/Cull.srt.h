#pragma once

BEGIN_SRT(CullSrt)
    BEGIN_SRT_SET(PerFrame)
        DECL_CBUFFER(PerFrame, CBUFFER(CullData), gCull)
        DECL_BUFFER(PerFrame, Buffer(CullLight), gLights)
#if SAMPLE_COUNT > 1
        DECL_TEXTURE(PerFrame, Depth2DMS(float, SAMPLE_COUNT), gDepth)
#else
        DECL_TEXTURE(PerFrame, Depth2D(float), gDepth)
#endif
        DECL_RWBUFFER(PerFrame, RWBuffer(uint), gTileMasks)
        DECL_BUFFER(PerFrame, Buffer(float4), gDecals)         // the core's decals' spheres (frame.h decalSphereBuffer)
        DECL_RWBUFFER(PerFrame, RWBuffer(uint), gDecalMasks)   // DECAL_WORDS a tile
        DECL_RWBUFFER(PerFrame, RWBuffer(uint), gSideMasks)    // as gTileMasks, by the tile's sides alone (glass)
    END_SRT_SET(PerFrame)
END_SRT(CullSrt)
