#!/bin/sh
# Builds the plugin (plugin/ = the renderer vita/, carrying psp/) into
# target/adrenaline-native-res.suprx with cargo and the SDK's linker and
# tools only (no CMake, no C).
. "$(dirname "$0")/env.sh"
OUT=target/plugin
mkdir -p "$OUT"

# The PSP side first: the plugin carries it (vita/src/psp_plugin.rs).
sh scripts/build-psp-plugin.sh
# The PSP-depth programs go into a file of their own, loaded only for the
# games that use them (vita/src/shaders.rs); the plugin checks its hash.
python3 scripts/pack-depth-twins.py

# The version the user log names (vita/src/user_log.rs): the git tag
# (e.g. v0.3.0, or v0.3.0-2-gabc1234 two commits after it). safe.directory:
# in the SDK container the checkout belongs to another user.
NATIVERES_VERSION=$(git -c safe.directory='*' describe --tags --always --dirty 2>/dev/null || echo unknown)
export NATIVERES_VERSION

# core from source (the Vita is a tier-3 target); compiler_builtins brings
# memcpy/memset/memcmp/strlen, since no C library is linked.
cargo $NIGHTLY build --release -p nativeres-plugin --target $VITA_TARGET \
  -Z build-std=core,compiler_builtins -Z build-std-features=compiler-builtins-mem

# -q keeps the relocations vita-elf-create needs; module_start/module_stop
# are pulled from the archive by naming them. No --gc-sections: the
# nativeres_plan_* statics (the memory plan, read from the ELF) are
# referenced by nothing and must stay.
#
# Imports: taiHEN starts the plugin when Adrenaline's process starts, before
# some libraries are loaded. A plain import of a library not loaded yet makes
# the loader refuse the whole module, so everything but the process's basics
# is imported weak, and nothing calls a library before it is there.
#
# The capture thread's per-batch code is linked first, together, so it
# misses the instruction cache less (scripts/hot-order.py explains).
#
# vita-elf-create appends ~4 KiB of module tables after the code; code
# ending within that of the next 64 KiB boundary fails ("segment 1
# overlaps"). __sce_headroom (vitasdk's linker script) moves the data
# segment on to the next boundary then.
python3 scripts/hot-order.py target/$VITA_TARGET/release/libnativeres_plugin.a "$OUT/hot-order.ld"
arm-vita-eabi-gcc -Wl,-q -nostartfiles -nostdlib \
  -Wl,-u,module_start -Wl,-u,module_stop -Wl,-e,module_start -Wl,-z,noexecstack \
  -Wl,--section-ordering-file="$OUT/hot-order.ld" \
  -Wl,--defsym,__sce_headroom=0x2000 \
  -o "$OUT/nativeres.elf" \
  target/$VITA_TARGET/release/libnativeres_plugin.a \
  -Wl,--start-group \
  -ltaihen_stub -lSceLibKernel_stub -lSceKernelThreadMgr_stub -lSceSysmem_stub \
  -lSceKernelModulemgr_stub -lSceProcessmgr_stub \
  -lSceGxm_stub_weak -lSceDisplay_stub_weak -lSceCompat_stub_weak -lSceKernelDmacMgr_stub_weak \
  -lSceIofilemgr_stub_weak -lSceSysmodule_stub_weak -lScePower_stub_weak -lSceNet_stub_weak -lSceRtc_stub_weak \
  -Wl,--end-group
vita-elf-create -e vita/exports.yml "$OUT/nativeres.elf" "$OUT/nativeres.velf"
vita-make-fself -c "$OUT/nativeres.velf" target/adrenaline-native-res.suprx
ls -l target/adrenaline-native-res.suprx
