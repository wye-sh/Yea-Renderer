// The exposure in use, copied into the 1 x 1 texture upscalers read it from.
#pragma once

BEGIN_SRT(UpscaleExposureSrt)
    BEGIN_SRT_SET(PerFrame)
        DECL_BUFFER(PerFrame, Buffer(float), gExposure)
        DECL_RWTEXTURE(PerFrame, RWTex2D(float), gOut) // R32F, 1 x 1
    END_SRT_SET(PerFrame)
END_SRT(UpscaleExposureSrt)
