#!/usr/bin/env bash
# The example view's sample models: fetches each source from Khronos' glTF-Sample-Assets at a pinned commit, checks it
# against the hashes below, and cooks it into example/renderer_example/Assets/models/ with the component's model cooker.
# The sources (about 76 MB, most of it FlightHelmet's textures) are not kept in the repository; this pins them exactly.
#
# Licences, from each model's own LICENSE.md and metadata.json at that commit: every model file is CC0-1.0 (public
# domain: no conditions, credit not required). Their LICENSE.md and metadata.json files themselves are CC-BY-4.0 and
# are not used. Credit, as metadata.json gives it: Lantern by sbtron (Microsoft, 2017); FlightHelmet converted by Gary
# Hsu (2018); WaterBottle and BoomBox by Microsoft (2017). Logos and trademarks are excluded from the licence.
#
#   example/model_sources/fetch_and_cook.sh          (from anywhere; builds the cooker when missing)
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../../../.." && pwd)"
commit=e124e8d8d7330a5372bc2b4d00d39a4a9d7997ea
base="https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/$commit/Models"
out="$here/../renderer_example/Assets/models"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
[[ -x "$root/build/tools/model_cook" ]] || "$root/components/renderer/tools/model_cook/build.sh"

# sha256, model, path under the model's folder
files="
a79458c4b02d695187a952f23a63b8bf278e7bc3d316a3c2a314f2d6974181f1 Lantern glTF-Binary/Lantern.glb
b337e526fd6a162013c2984aeec163f5fbb4f717252724dfc3f3458bd51df94b WaterBottle glTF-Binary/WaterBottle.glb
f8b918445ebdd006768232205a62f5182d2208ca57f84c6ccc084943c0bc8f15 BoomBox glTF-Binary/BoomBox.glb
c9c16c8b85749f62f38ec6069700e5388c6894d8e566f51a35cc270255402cd3 FlightHelmet glTF/FlightHelmet.gltf
9e623a27f837e1cc995d1380da5fd3becf4d29586fefba882fc0ba76b49f94bf FlightHelmet glTF/FlightHelmet.bin
0914544aabe161046f417274051e34ecc85163cfce6698581b3e1e18c96fe4f1 FlightHelmet glTF/FlightHelmet_Materials_GlassPlasticMat_BaseColor.png
c297f18f3b4259fc8b411cdc843a4a44b8cef3e3a53b21cec7b53574e5ef6872 FlightHelmet glTF/FlightHelmet_Materials_GlassPlasticMat_Normal.png
d4519704f15395c7138421a5814b478df549a2411bc6fa97dd6d02ba62dc6c54 FlightHelmet glTF/FlightHelmet_Materials_GlassPlasticMat_OcclusionRoughMetal.png
c9346ac6daa3ad34d0b36c1e7cf311ccbcc298710ea453cdf3b6ad40dc60013f FlightHelmet glTF/FlightHelmet_Materials_LeatherPartsMat_BaseColor.png
7fb69b26f80c67df41e87d7344bde9d24dda0be77546b0a6acabc4040694872a FlightHelmet glTF/FlightHelmet_Materials_LeatherPartsMat_Normal.png
b77e60f175e4632cd3d892bc9f0e748f522c9288a469585d5cb90270cd68910e FlightHelmet glTF/FlightHelmet_Materials_LeatherPartsMat_OcclusionRoughMetal.png
b55f6561f8adabc907360443aab010f3effaf05074fdccea46d8c2d66247cf16 FlightHelmet glTF/FlightHelmet_Materials_LensesMat_BaseColor.png
430775894001933044c89070b1bfd4f9b8489f74780b1ab7079efb0afca95d8d FlightHelmet glTF/FlightHelmet_Materials_LensesMat_Normal.png
35d81a71ab184f9b50f63ff8f3e0f90cfb6c99f9b2d9ed9d9887a902c16a458d FlightHelmet glTF/FlightHelmet_Materials_LensesMat_OcclusionRoughMetal.png
8e26add3ad1b2398d58638580bb1fe56f2fbae3572b4a682b6595a5a26fb1b40 FlightHelmet glTF/FlightHelmet_Materials_MetalPartsMat_BaseColor.png
a833fcc12718c02bcba97cbc9760d06c4a711ce7f36c23076e7cf6d4393a80e7 FlightHelmet glTF/FlightHelmet_Materials_MetalPartsMat_Normal.png
17f3e27b51011b8f98751dae25958e2b850fcf1467f1019bec812553e0e5eb1d FlightHelmet glTF/FlightHelmet_Materials_MetalPartsMat_OcclusionRoughMetal.png
f751d643d1f220b300705d9ad57ed284a883eb7d3788469199b752f4381dcc37 FlightHelmet glTF/FlightHelmet_Materials_RubberWoodMat_BaseColor.png
bb70753c836c59db7e245e5fe2c9db8eee833d29d67e03da348f6adb1e95142a FlightHelmet glTF/FlightHelmet_Materials_RubberWoodMat_Normal.png
2727d4638514f75fcbd29777bf03db6bd2f0f1ad926e2976cf2346aa22c0bd19 FlightHelmet glTF/FlightHelmet_Materials_RubberWoodMat_OcclusionRoughMetal.png
"
while read -r sum model path; do
    [[ -n "$sum" ]] || continue
    mkdir -p "$work/$model"
    curl -sfL -o "$work/$model/$(basename "$path")" "$base/$model/$path"
    echo "$sum  $work/$model/$(basename "$path")" | sha256sum -c --quiet -
done <<< "$files"
mkdir -p "$out"
"$root/build/tools/model_cook" "$work/Lantern/Lantern.glb" "$out/Lantern.rmodel"
"$root/build/tools/model_cook" "$work/FlightHelmet/FlightHelmet.gltf" "$out/FlightHelmet.rmodel"
"$root/build/tools/model_cook" "$work/WaterBottle/WaterBottle.glb" "$out/WaterBottle.rmodel"
"$root/build/tools/model_cook" "$work/BoomBox/BoomBox.glb" "$out/BoomBox.rmodel"
