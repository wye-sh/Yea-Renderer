# Renderer

The 3D rendering engine for Yea games: physically based materials, a sun and any number of
coloured point and spot lights, light culling on the GPU, soft cascaded sun shadows and cached
shadows for local lights, ambient occlusion, contact shadows, reflection probes, ray-traced reflections on GPUs with ray tracing (screen-space ones elsewhere), variable-rate
shading, automatic mesh level of detail, animated models (skinned characters with crossfaded clips, root motion, retargeting (one character's animations on another, feet kept planted), layered animation (an upper-body clip over a walk, additive breathing and flinches), foot IK (feet planted on slopes and steps), blend spaces (walk, run and strafe blended by speed and direction) and morph targets for faces and shape changes, their shadows and ray-traced shadows posed with them), global illumination (light bouncing between surfaces, ray-traced where the GPU can), a physical sky, volumetric fog with sunbeams and lamp halos, GPU particles (dust, steam, smoke, sparks), glass and other see-through surfaces (sorted with the particles, so steam behind a pane is seen through it; bending, tinting and, when rough, blurring what is behind them; seen in ray-traced reflections and casting coloured shadows, ray-traced or from shadow maps, however many panes the light passes), decals (scorches, puddles, paint, dirt; seen in reflections and colouring bounced light too), temporal antialiasing, AMD FSR upscaling, NVIDIA DLSS (Windows, GeForce RTX), automatic exposure and tonemapping. Open its example view, Renderer,
to see everything it does and tune it in F1.

MIT licensed. Our own code, apart from the third-party items in LICENSES.txt, each shipped only while the option that
uses it is picked (such as AMD's FSR DLLs, under AMD's own licence).
