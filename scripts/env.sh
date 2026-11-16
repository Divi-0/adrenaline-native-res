#!/bin/sh
# Shared by the build scripts (sourced): the repository root, cargo's
# download cache and the toolchain. Run the scripts inside the SDK image
# (docker/Dockerfile), see README.md "Building".
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
# Cargo's download cache inside the checkout, so it survives container runs
# (NATIVERES_CARGO_HOME to put it elsewhere).
export CARGO_HOME="${NATIVERES_CARGO_HOME:-$ROOT/.cargo-home}"
mkdir -p "$CARGO_HOME"
# The Vita and the PSP are tier-3 targets: core is built from source
# (-Z build-std), which needs nightly. Pinned in docker/Dockerfile too.
NIGHTLY=+nightly-2026-10-01
VITA_TARGET=armv7-sony-vita-newlibeabihf
PSP_TARGET=mipsel-sony-psp
