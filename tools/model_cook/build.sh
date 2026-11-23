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

# Nicholas Frechette's ACL 2.1.0 and Realtime Math 2.2.0 (MIT): the cooker compresses clips with ACL's full source,
# fetched and checked like Draco (renderer_models keeps only their decompression headers, which the cooker's decoder
# check uses through this same copy).
acl="$out/acl-2.1.0" rtm="$out/rtm-2.2.0"
fetch() { # dir url sha256
    [ -d "$1/includes" ] && return
    rm -rf "$1" && mkdir -p "$1"
    curl -sSfL -o "$1/src.tar.gz" "$2"
    echo "$3  $1/src.tar.gz" | sha256sum -c --quiet
    tar xzf "$1/src.tar.gz" -C "$1" --strip-components=1
}
fetch "$acl" https://github.com/nfrechette/acl/archive/refs/tags/v2.1.0.tar.gz 0ac8473cd30eb768bae1ef58558e3088242d6fef81f727ce7b5ff5af9be74fce
fetch "$rtm" https://github.com/nfrechette/rtm/archive/refs/tags/v2.2.0.tar.gz e627c46de1895b78eee63612ce4e5ade3af28e040edab6ec6f6521366b4e8b9b

cc -O2 -c -o "$out/ufbx.o" "$here/vendor/ufbx/ufbx.c"
c++ -std=c++17 -O2 -Wall -Wno-unused-function -o "$out/model_cook" "$here/model_cook.cpp" "$here/vendor/bc7e/basisu_bc7e_scalar.cpp" \
    "$here"/vendor/meshoptimizer/*.cpp "$out/ufbx.o" -I"$acl/includes" -I"$rtm/includes" -I"$draco/src" -I"$draco/build" "$draco/build/libdraco.a" -pthread \
    -Wno-unused-but-set-variable -Wno-maybe-uninitialized
echo "built $out/model_cook"
