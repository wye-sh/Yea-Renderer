#pragma once

BEGIN_SRT(FilterSrt)
    BEGIN_SRT_SET(PerFrame)
        DECL_CBUFFER(PerFrame, CBUFFER(PassData), gPass)
        DECL_TEXTURE(PerFrame, TexCube(float4), gSource)
    END_SRT_SET(PerFrame)
END_SRT(FilterSrt)
