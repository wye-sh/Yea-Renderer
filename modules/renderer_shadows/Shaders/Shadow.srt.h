#pragma once

BEGIN_SRT(ShadowSrt)
    BEGIN_SRT_SET(PerFrame)
        DECL_BUFFER(PerFrame, Buffer(InstanceData), gInstances)
        DECL_BUFFER(PerFrame, Buffer(ShadowView), gViews)
    END_SRT_SET(PerFrame)
END_SRT(ShadowSrt)
