#!/bin/sh
# Builds and tests everything, inside the SDK image (docker/Dockerfile):
#
#   docker build -t nativeres-sdk docker
#   docker run --rm -v "$PWD:/src" -w /src nativeres-sdk bash -lc "sh scripts/build.sh"
#
# Steps: the PC tests (core, and the replay regression test over the frame
# dumps in tests/, see replay/tests/regression.rs), the plugin
# (target/adrenaline-native-res.suprx) and, with "all", the shader compiler app.
. "$(dirname "$0")/env.sh"

echo "== Core tests (PC)"
cargo test -p nativeres-core --quiet

echo "== PC replay + regression test"
cargo build --release -p nativeres-replay --quiet
cargo test --release -p nativeres-replay --test regression --quiet -- --nocapture

echo "== Plugin (adrenaline-native-res.suprx with nativeres.prx inside)"
sh scripts/build-plugin.sh

if [ "$1" = all ]; then
  echo "== Shader compiler app"
  sh scripts/build-shadercompiler.sh
fi
echo "== Done"
