#!/bin/sh
# Builds the plugin (plugin/ = the renderer vita/, carrying psp/) into
# target/adrenaline-native-res.suprx with cargo and the SDK's linker and
# tools only (no CMake, no C).
. "$(dirname "$0")/env.sh"
OUT=target/plugin
mkdir -p "$OUT"

# The PSP side first: the plugin carries it (vita/src/psp_plugin.rs).
sh scripts/build-psp-plugin.sh

# core from source (the Vita is a tier-3 target); compiler_builtins brings
# memcpy/memset/memcmp/strlen, since no C library is linked.
cargo $NIGHTLY build --release -p nativepsp-plugin --target $VITA_TARGET \
  -Z build-std=core,compiler_builtins -Z build-std-features=compiler-builtins-mem

# -q keeps the relocations vita-elf-create needs; module_start/module_stop
# are pulled from the archive by naming them. No --gc-sections: the
# nativepsp_plan_* statics (the memory plan, read from the ELF) are
# referenced by nothing and must stay.
#
# Imports: taiHEN starts the plugin when Adrenaline's process starts, before
# some libraries are loaded. A plain import of a library not loaded yet makes
# the loader refuse the whole module, so everything but the process's basics
# is imported weak, and nothing calls a library before it is there.
arm-vita-eabi-gcc -Wl,-q -nostartfiles -nostdlib \
  -Wl,-u,module_start -Wl,-u,module_stop \
  -o "$OUT/nativepsp.elf" \
  target/$VITA_TARGET/release/libnativepsp_plugin.a \
  -Wl,--start-group \
  -ltaihen_stub -lSceLibKernel_stub -lSceKernelThreadMgr_stub -lSceSysmem_stub \
  -lSceKernelModulemgr_stub -lSceProcessmgr_stub \
  -lSceGxm_stub_weak -lSceDisplay_stub_weak -lSceCompat_stub_weak -lSceKernelDmacMgr_stub_weak \
  -lSceIofilemgr_stub_weak -lSceSysmodule_stub_weak -lScePower_stub_weak -lSceNet_stub_weak \
  -Wl,--end-group
vita-elf-create -e vita/exports.yml "$OUT/nativepsp.elf" "$OUT/nativepsp.velf"
vita-make-fself -c "$OUT/nativepsp.velf" target/adrenaline-native-res.suprx
ls -l target/adrenaline-native-res.suprx
