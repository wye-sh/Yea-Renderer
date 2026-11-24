// The scene colour chain's copy rectangle after particle runs (ChainRect.comp.fsl).
#pragma once

BEGIN_SRT(ChainRectSrt)
    BEGIN_SRT_SET(PerFrame)
        DECL_CBUFFER(PerFrame, CBUFFER(ChainRectData), gRect)
        DECL_BUFFER(PerFrame, Buffer(uint), gRunRects)      // renderer_particles' run rectangles (x0, y0, x1, y1 a run)
        DECL_RWBUFFER(PerFrame, RWBuffer(uint4), gRectsOut) // each level's rectangle (Chain.comp.fsl gRects)
        DECL_RWBUFFER(PerFrame, RWBuffer(uint), gArgsOut)   // and its dispatch (groups x, y, z)
    END_SRT_SET(PerFrame)
END_SRT(ChainRectSrt)
