// The local lights a pixel runs (LightCount.comp.fsl): the forward pass's own frame data, depth and light lists.
#pragma once

BEGIN_SRT(LightCountSrt)
    BEGIN_SRT_SET(PerFrame)
        DECL_CBUFFER(PerFrame, CBUFFER(FrameData), gFrame)
        DECL_TEXTURE(PerFrame, Depth2D(float), gDepth)
        DECL_BUFFER(PerFrame, Buffer(uint), gTileMasks)
        DECL_BUFFER(PerFrame, Buffer(uint), gZBins)
        DECL_BUFFER(PerFrame, Buffer(LightData), gLights)
        DECL_BUFFER(PerFrame, Buffer(LightShape), gLightShapes)
        DECL_BUFFER(PerFrame, Buffer(float), gExposure)
        DECL_RWBUFFER(PerFrame, RWBuffer(uint), gCount)
    END_SRT_SET(PerFrame)
END_SRT(LightCountSrt)
