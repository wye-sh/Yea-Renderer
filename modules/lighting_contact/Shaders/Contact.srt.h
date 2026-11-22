#pragma once

BEGIN_SRT(ContactSrt)
    BEGIN_SRT_SET(PerFrame)
        DECL_CBUFFER(PerFrame, CBUFFER(ContactData), gContact)
#if SAMPLE_COUNT > 1
        DECL_TEXTURE(PerFrame, Depth2DMS(float, SAMPLE_COUNT), gDepth)
#else
        DECL_TEXTURE(PerFrame, Depth2D(float), gDepth)
#endif
        DECL_RWTEXTURE(PerFrame, RWTex2D(float), gOut)
    END_SRT_SET(PerFrame)
END_SRT(ContactSrt)
