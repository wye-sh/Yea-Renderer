#pragma once

// The glass tint passes (renderer_shadows.cpp): see-through casters peeled layer by layer into a scratch square, then
// copied into the tint maps.
BEGIN_SRT(TintSrt)
    BEGIN_SRT_SET(PerFrame)
        DECL_BUFFER(PerFrame, Buffer(InstanceData), gInstances)
        DECL_BUFFER(PerFrame, Buffer(ShadowView), gViews)
        DECL_BUFFER(PerFrame, Buffer(TintPass), gPasses)
        DECL_BUFFER(PerFrame, Buffer(MaterialData), gMaterials)
        DECL_BUFFER(PerFrame, Buffer(MaterialMapData), gMaterialMaps)
        DECL_BUFFER(PerFrame, Buffer(float4), gRefraction) // frame.h refraction: two a material
        DECL_BUFFER(PerFrame, Buffer(float2), gDfg)        // frame.h dfg
        DECL_TEXTURE(PerFrame, Tex2D(float), gTintPrev)    // the tint maps' depth (the previous layer's square)
        DECL_TEXTURE(PerFrame, Tex2D(float), gScratchZ)    // the layer just drawn (copy passes)
        DECL_TEXTURE(PerFrame, Tex2D(float4), gScratchColor)
        DECL_TEXTURE(PerFrame, Depth2D(float), gSunAtlas)  // the opaque depth: glass behind it is left out
        DECL_TEXTURE(PerFrame, Depth2D(float), gLocalAtlas)
        DECL_ARRAY_TEXTURES(PerFrame, Tex2D(float4), gTextures, MAX_TEXTURES)
    END_SRT_SET(PerFrame)
END_SRT(TintSrt)
