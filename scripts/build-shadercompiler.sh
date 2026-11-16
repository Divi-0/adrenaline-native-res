#!/bin/sh
# Builds the NativeRes Shader Compiler app (shadercompiler/, Rust) into
# target/nativeres_shadercompiler.vpk. Only needed to change the shaders:
# the compiled .gxp files are in vita/shaders/ (see README.md "Shaders").
#
# Install the VPK once with VitaShell (title NRES00001). It needs Sony's
# runtime shader compiler at ur0:data/libshacccg.suprx (not included).
. "$(dirname "$0")/env.sh"
OUT=target/shadercompiler
mkdir -p "$OUT"

# core from source (the Vita is a tier-3 target); compiler_builtins brings
# memcpy/memset/memcmp, since no C library is linked.
cargo $NIGHTLY build --release -p nativeres-shadercompiler --target $VITA_TARGET \
  -Z build-std=core,compiler_builtins -Z build-std-features=compiler-builtins-mem

# -q keeps the relocations vita-elf-create needs. The entry is our
# module_start (no C start files); vita-elf-create makes it the module's
# start and writes the process parameters (sceUserMainThreadStackSize).
# libshacccg is loaded by module_start before its functions are called.
arm-vita-eabi-gcc -Wl,-q -nostartfiles -nostdlib \
  -Wl,-e,module_start -Wl,-u,module_start -Wl,-u,sceUserMainThreadStackSize -Wl,--gc-sections \
  -o "$OUT/nativeres_shadercompiler.elf" \
  target/$VITA_TARGET/release/libnativeres_shadercompiler.a \
  -Wl,--start-group \
  -lSceShaccCg_stub -lSceLibKernel_stub -lSceIofilemgr_stub -lSceKernelModulemgr_stub -lSceSysmem_stub \
  -Wl,--end-group
vita-elf-create "$OUT/nativeres_shadercompiler.elf" "$OUT/nativeres_shadercompiler.velf"
# Unsafe (no -s): it loads a module from ur0:data.
vita-make-fself -c "$OUT/nativeres_shadercompiler.velf" "$OUT/eboot.bin"
vita-mksfoex -s APP_VER=01.00 -s TITLE_ID=NRES00001 "NativeRes Shader Compiler" "$OUT/param.sfo"
vita-pack-vpk -s "$OUT/param.sfo" -b "$OUT/eboot.bin" target/nativeres_shadercompiler.vpk
ls -l "$OUT/eboot.bin" target/nativeres_shadercompiler.vpk
