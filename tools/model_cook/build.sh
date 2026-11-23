#!/usr/bin/env bash
# Builds the model cooker (never shipped) into the repository's build/tools/model_cook. Run from anywhere.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../../../.." && pwd)"
mkdir -p "$root/build/tools"
c++ -std=c++17 -O2 -Wall -Wno-unused-function -o "$root/build/tools/model_cook" "$here/model_cook.cpp" "$here"/vendor/meshoptimizer/*.cpp -Wno-unused-but-set-variable
echo "built $root/build/tools/model_cook"
