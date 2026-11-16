#!/bin/sh
# Builds and tests everything, inside the SDK image (docker/Dockerfile):
#
#   docker build -t nativepsp-sdk docker
#   docker run --rm -v "$PWD:/src" -w /src nativepsp-sdk bash -lc "sh scripts/build.sh"
#
# Steps: the PC tests (core, and the replay regression test over the frame
# dumps in tests/, see replay/tests/regression.rs), the plugin
# (target/adrenaline-native-res.suprx) and, with "all", the shader compiler app.
. "$(dirname "$0")/env.sh"

echo "== Core tests (PC)"
cargo test -p nativepsp-core --quiet

echo "== PC replay + regression test"
cargo build --release -p nativepsp-replay --quiet
cargo test --release -p nativepsp-replay --test regression --quiet -- --nocapture

echo "== Plugin (adrenaline-native-res.suprx with nativepsp.prx inside)"
sh scripts/build-plugin.sh

if [ "$1" = all ]; then
  echo "== Shader compiler app"
  sh scripts/build-shadercompiler.sh
fi
echo "== Done"
