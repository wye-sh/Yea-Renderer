// Skinning (skin.cpp): every posed skinned instance's vertices, from its mesh in the shared pool into its own region
// of the same buffers, once a frame.
#pragma once

BEGIN_SRT(SkinSrt)
    BEGIN_SRT_SET(PerFrame)
        DECL_RWBUFFER(PerFrame, RWBuffer(uint), gPositions)  // RENDER_POSITION_STRIDE: 3 uints a vertex
        DECL_RWBUFFER(PerFrame, RWBuffer(uint), gAttributes) // RENDER_ATTRIBUTE_STRIDE: 5 uints a vertex
        DECL_BUFFER(PerFrame, Buffer(uint4), gSkin)          // a vertex's bones (u16 pairs) and weights (unorm16 pairs)
        DECL_BUFFER(PerFrame, Buffer(float4), gPalette)      // 3 rows a bone, model space
        DECL_BUFFER(PerFrame, Buffer(uint4), gJobs)          // [0].x the job count ([0] and [1] its header); job k at 2 + 2k (skin.cpp's Job)
    END_SRT_SET(PerFrame)
END_SRT(SkinSrt)
