#!/usr/bin/env bash
# Builds the model cooker (never shipped) into the repository's build/tools/model_cook. Run from anywhere.
# Google's Draco (Apache-2.0, for KHR_draco_mesh_compression) isn't kept in the repository: its release is fetched once,
# checked against its SHA-256 and built as a static library under build/tools/draco (about a minute).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../../../.." && pwd)"
out="$root/build/tools"
mkdir -p "$out"

draco_version=1.5.9
draco_sha256=5b84e0e212ccc7b03ace681760a65a66fe2481a5f54df65154cfc8f0cbb87148
draco="$out/draco-$draco_version"
if [ ! -f "$draco/build/libdraco.a" ]; then
    rm -rf "$draco" && mkdir -p "$draco"
    curl -sSfL -o "$draco/src.tar.gz" "https://github.com/google/draco/archive/refs/tags/$draco_version.tar.gz"
    echo "$draco_sha256  $draco/src.tar.gz" | sha256sum -c --quiet
    tar xzf "$draco/src.tar.gz" -C "$draco" --strip-components=1
    cmake -S "$draco" -B "$draco/build" -DCMAKE_BUILD_TYPE=Release -DDRACO_TESTS=OFF -DDRACO_JS_GLUE=OFF -DBUILD_SHARED_LIBS=OFF > "$draco/cmake.log"
    cmake --build "$draco/build" -j"$(nproc)" --target draco_static > "$draco/build.log"
fi

cc -O2 -c -o "$out/ufbx.o" "$here/vendor/ufbx/ufbx.c"
c++ -std=c++17 -O2 -Wall -Wno-unused-function -o "$out/model_cook" "$here/model_cook.cpp" "$here/vendor/bc7e/basisu_bc7e_scalar.cpp" \
    "$here"/vendor/meshoptimizer/*.cpp "$out/ufbx.o" -I"$draco/src" -I"$draco/build" "$draco/build/libdraco.a" -pthread \
    -Wno-unused-but-set-variable -Wno-maybe-uninitialized
echo "built $out/model_cook"
