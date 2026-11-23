#pragma once

// The raster fallback (Capture.vert.fsl, Capture.frag.fsl): the scene round raster.y probes a frame, six
// GI_FACE_TEXELS faces each, lit as renderer_rt lights its probe rays' hits; and (SUN_SHADOW) the sun's own shadow map
// over everything the probes can see.
BEGIN_SRT(GiCaptureSrt)
    BEGIN_SRT_SET(Persistent)
        DECL_SAMPLER(Persistent, SamplerComparisonState, gShadowCmp)
    END_SRT_SET(Persistent)
    BEGIN_SRT_SET(PerFrame)
        DECL_CBUFFER(PerFrame, CBUFFER(GiData), gGi)
        DECL_BUFFER(PerFrame, Buffer(GiInstance), gInstances)
        DECL_BUFFER(PerFrame, Buffer(GiMaterial), gMaterials)
        DECL_BUFFER(PerFrame, Buffer(GiLight), gLights)
        DECL_BUFFER(PerFrame, Buffer(GiLightShape), gShapes)
        DECL_BUFFER(PerFrame, Buffer(GiShadowView), gViews)
        DECL_BUFFER(PerFrame, Buffer(float4), gSkySh)
        DECL_TEXTURE(PerFrame, Depth2D(float), gGiSun)  // the sun's map (shadow.x)
        DECL_TEXTURE(PerFrame, Depth2D(float), gLocalAtlas)
        DECL_TEXTURE(PerFrame, Tex2D(float4), gGiIrr)   // last frame's probes (raster.w & 2), for the bounce
        DECL_TEXTURE(PerFrame, Tex2D(float4), gGiDist)
        DECL_BUFFER(PerFrame, Buffer(float4), gGiProbes)
        DECL_BUFFER(PerFrame, Buffer(GiMaterialMap), gMaterialMaps) // the frame's materialMaps
        DECL_ARRAY_TEXTURES(PerFrame, Tex2D(float4), gTextures, MAX_TEXTURES)
    END_SRT_SET(PerFrame)
END_SRT(GiCaptureSrt)
