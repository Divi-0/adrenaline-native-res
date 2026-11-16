#!/bin/sh
# Builds the PSP side (psp/, Rust) into target/nativeres.prx with cargo for
# mipsel-sony-psp (rust-lld and the target's own linker script) and pspdev's
# psp-prxgen. The Vita plugin embeds the result (vita/src/psp_plugin.rs), so
# scripts/build-plugin.sh runs this first.
. "$(dirname "$0")/env.sh"
OUT=target/psp
mkdir -p "$OUT"

# -mgpopt=false: no gp-relative small data. The hooks run inside the game's
# system calls, where $gp isn't ours.
# --no-gc-sections: the import tables (.lib.stub) are referenced by nothing
# but the module info's bounds, and must stay.
# No compiler-builtins-mem: memcpy/memset are the kernel's (psp/src/sys.rs).
export CARGO_TARGET_MIPSEL_SONY_PSP_RUSTFLAGS="-C llvm-args=-mgpopt=false -C link-arg=--no-gc-sections"
cargo $NIGHTLY build --release -p nativeres-psp --target $PSP_TARGET \
  -Z build-std=core,compiler_builtins
cp target/$PSP_TARGET/release/nativeres "$OUT/nativeres.elf"

# Checks: nothing gp-relative, no panic path linked in (a panic would stop
# the game's thread), the module info where the loader looks.
if psp-readelf -r "$OUT/nativeres.elf" | grep -i "GPREL\|GOT16\|CALL16"; then
  echo "build-psp-plugin: gp-relative relocations in the PSP plugin" >&2; exit 1
fi
if psp-nm "$OUT/nativeres.elf" | grep -i "panic_bounds_check\|panic_fmt\|panicking\|unwrap_failed\|slice_index\|overflow"; then
  echo "build-psp-plugin: a panic path is linked into the PSP plugin" >&2; exit 1
fi
psp-readelf -SW "$OUT/nativeres.elf" | grep -q "sceModuleInfo" || { echo "build-psp-plugin: no module info" >&2; exit 1; }

psp-fixup-imports "$OUT/nativeres.elf"
psp-prxgen "$OUT/nativeres.elf" target/nativeres.prx
ls -l target/nativeres.prx
