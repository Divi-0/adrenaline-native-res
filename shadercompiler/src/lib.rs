//! NativeRes Shader Compiler: a small Vita app, only needed to change the
//! plugin's shaders.
//!
//! Compiles every .cg file in ux0:data/nativeres/shaders/ with Sony's
//! runtime Cg compiler (ur0:data/libshacccg.suprx, extracted by tools like
//! ShaRKBR33D) and writes <name>.gxp plus <name>.gxp.txt (the compiler's
//! messages) next to it. Files ending in _v.cg are vertex programs, all
//! others fragment programs. When done it writes done.txt and quits.
//!
//! done.txt has one line per file, `<name>.cg: OK <bytes> (<ms> ms)` or
//! `<name>.cg: ERR <why>`, then a line starting with `#` (the heap's
//! numbers and the total time). Copy the sources there, start the app,
//! wait for done.txt and copy the .gxp files into vita/shaders/, which the
//! plugin embeds. Users of the plugin never need libshacccg.
//!
//! It is a separate app because loading libshacccg inside Adrenaline's PSP
//! emulator process failed (0x8002D0F3, a missing preload) and once crashed
//! Adrenaline. An app gets the system's preloaded modules libshacccg needs.
//!
//! Built by scripts/build-shadercompiler.sh (cargo, the SDK's linker and
//! tools, no C library): the compiler's malloc/free are ours
//! (`nativeres_core::class_heap` over one 32 MiB block, as big as the C
//! app's newlib heap).

#![no_std]
#![allow(non_snake_case, non_upper_case_globals)]

use core::cell::UnsafeCell;
use core::ffi::c_void;
use core::fmt::Write;
use core::panic::PanicInfo;
use core::sync::atomic::{AtomicBool, AtomicI32, Ordering};

use nativeres_core::class_heap::ClassHeap;

type SceUID = i32;

const SHADER_DIR: &str = "ux0:data/nativeres/shaders";
const SHACCCG_PATH: &[u8] = b"ur0:data/libshacccg.suprx\0";
const MAX_SOURCE: usize = 60 * 1024;
/// The compiler's memory (the C app: `_newlib_heap_size_user`).
const HEAP_BYTES: u32 = 32 * 1024 * 1024;

const SCE_O_RDONLY: i32 = 0x0001;
const SCE_O_WRONLY: i32 = 0x0002;
const SCE_O_CREAT: i32 = 0x0200;
const SCE_O_TRUNC: i32 = 0x0400;
const SCE_SEEK_SET: i32 = 0;
const SCE_SEEK_END: i32 = 2;
const SCE_KERNEL_MEMBLOCK_TYPE_USER_RW: u32 = 0x0C20_D060;

/// The main thread's stack (read by vita-elf-create into the process
/// parameters). The C app had the default (256 KiB); the compiler is deep
/// code, and the room costs nothing here.
#[no_mangle]
#[used]
pub static sceUserMainThreadStackSize: u32 = 1024 * 1024;

// ---- Sony's runtime Cg compiler (psp2/shacccg.h) ----

const SCE_SHACCCG_PROFILE_VP: u32 = 0;
const SCE_SHACCCG_PROFILE_FP: u32 = 1;
const SCE_SHACCCG_TRIVIAL: u32 = 1;
const SCE_SHACCCG_ENGLISH: u32 = 0;
const SCE_SHACCCG_DIAGNOSTIC_LEVEL_WARNING: u32 = 1;
const SCE_SHACCCG_DIAGNOSTIC_LEVEL_ERROR: u32 = 2;

#[repr(C)]
struct SceShaccCgSourceFile {
    file_name: *const u8,
    text: *const u8,
    size: u32,
}

#[repr(C)]
struct SceShaccCgSourceLocation {
    file: *const SceShaccCgSourceFile,
    line_number: u32,
    column_number: u32,
}

type OpenFile = unsafe extern "C" fn(*const u8, *const SceShaccCgSourceLocation, *const SceShaccCgCompileOptions, *mut *const u8) -> *mut SceShaccCgSourceFile;
type ReleaseFile = unsafe extern "C" fn(*const SceShaccCgSourceFile, *const SceShaccCgCompileOptions);

#[repr(C)]
struct SceShaccCgCallbackList {
    open_file: Option<OpenFile>,
    release_file: Option<ReleaseFile>,
    locate_file: *const c_void,
    absolute_path: *const c_void,
    release_file_name: *const c_void,
    file_date: *const c_void,
}

#[repr(C)]
struct SceShaccCgCompileOptions {
    main_source_file: *const u8,
    target_profile: u32,
    entry_function_name: *const u8,
    search_path_count: u32,
    search_paths: *const c_void,
    macro_definition_count: u32,
    macro_definitions: *const c_void,
    include_file_count: u32,
    include_files: *const c_void,
    suppressed_warnings_count: u32,
    suppressed_warnings: *const c_void,
    locale: u32,
    use_fx: i32,
    no_stdlib: i32,
    optimization_level: i32,
    use_fastmath: i32,
    use_fastprecision: i32,
    use_fastint: i32,
    field_48: i32,
    warnings_as_errors: i32,
    performance_warnings: i32,
    warning_level: i32,
    pedantic: i32,
    pedantic_error: i32,
    field_60: i32,
    field_64: i32,
}

#[repr(C)]
struct SceShaccCgDiagnosticMessage {
    level: u32,
    code: u32,
    location: *const SceShaccCgSourceLocation,
    message: *const u8,
}

#[repr(C)]
struct SceShaccCgCompileOutput {
    program_data: *const u8,
    program_size: u32,
    diagnostic_count: i32,
    diagnostics: *const SceShaccCgDiagnosticMessage,
}

const _: () = assert!(core::mem::size_of::<SceShaccCgSourceFile>() == 0xC);
const _: () = assert!(core::mem::size_of::<SceShaccCgSourceLocation>() == 0xC);
const _: () = assert!(core::mem::size_of::<SceShaccCgCallbackList>() == 0x18);
const _: () = assert!(core::mem::size_of::<SceShaccCgCompileOptions>() == 0x68);
const _: () = assert!(core::mem::size_of::<SceShaccCgDiagnosticMessage>() == 0x10);
const _: () = assert!(core::mem::size_of::<SceShaccCgCompileOutput>() == 0x10);

/// SceIoDirent (0x160 bytes): SceIoStat (0x58), d_name[256], d_private, a
/// spare word.
#[repr(C)]
struct SceIoDirent {
    stat: [u8; 0x58],
    name: [u8; 256],
    private: *mut c_void,
    dummy: i32,
}

const _: () = assert!(core::mem::size_of::<SceIoDirent>() == 0x160);

extern "C" {
    fn sceShaccCgInitializeCompileOptions(options: *mut SceShaccCgCompileOptions) -> i32;
    fn sceShaccCgInitializeCallbackList(callbacks: *mut SceShaccCgCallbackList, defaults: u32);
    fn sceShaccCgCompileProgram(options: *const SceShaccCgCompileOptions, callbacks: *const SceShaccCgCallbackList, unknown: i32) -> *const SceShaccCgCompileOutput;
    fn sceShaccCgDestroyCompileOutput(output: *const SceShaccCgCompileOutput);
    fn sceShaccCgSetDefaultAllocator(malloc: unsafe extern "C" fn(u32) -> *mut c_void, free: unsafe extern "C" fn(*mut c_void)) -> i32;
    fn sceShaccCgReleaseCompiler();

    fn sceKernelLoadStartModule(path: *const u8, args: u32, argp: *mut c_void, flags: i32, option: *mut c_void, status: *mut i32) -> SceUID;
    fn sceKernelExitProcess(status: i32) -> i32;
    fn sceKernelAllocMemBlock(name: *const u8, block_type: u32, size: u32, options: *mut c_void) -> SceUID;
    fn sceKernelGetMemBlockBase(uid: SceUID, base: *mut *mut c_void) -> i32;
    fn sceKernelGetProcessTimeWide() -> u64;

    fn sceIoOpen(path: *const u8, flags: i32, mode: i32) -> SceUID;
    fn sceIoRead(fd: SceUID, data: *mut c_void, size: u32) -> i32;
    fn sceIoWrite(fd: SceUID, data: *const c_void, size: u32) -> i32;
    fn sceIoLseek32(fd: SceUID, offset: i32, whence: i32) -> i32;
    fn sceIoClose(fd: SceUID) -> i32;
    fn sceIoRemove(path: *const u8) -> i32;
    fn sceIoRename(old: *const u8, new: *const u8) -> i32;
    fn sceIoDopen(path: *const u8) -> SceUID;
    fn sceIoDread(fd: SceUID, entry: *mut SceIoDirent) -> i32;
    fn sceIoDclose(fd: SceUID) -> i32;
}

// ---- The compiler's Vita language extensions ----
//
// Sony's runtime compiler ships with its Vita-specific Cg extensions
// switched off: `__nativecolor` / `__regformat` (frame-read fragment
// programs, which read the pixel under them and do the blending
// themselves: nativeres_*_read_f.cg) are then unknown words. One flag in
// the compile state turns them on; the compiler's function that fills that
// state is hooked (taiHEN, in this app's process only) to set it. Found by
// SceShaccCgExt (bythos14, Rinnegatamante; LGPL): its
// `sceShaccCgExtEnableExtensions` sets the same byte at the same place,
// for the same two builds of the module (checked by module NID).

/// libshacccg builds whose layout the hook knows (SceShaccCgExt's list).
const KNOWN_SHACCCG_NIDS: [u32; 2] = [0xEE15_880D, 0x6C3C_7547];
/// The function filling the compile state (Thumb, segment 0).
const FILL_STATE_OFFSET: u32 = 0x1F_E7A4;
/// Its seventh argument + this: the "extension support" byte.
const EXTENSION_FLAG_AT: usize = 0x14;
/// The first argument + this: a pointer to the compile options' copy;
/// its byte PEDANTIC_FLAG_AT goes to the third argument + PEDANTIC_ERROR_AT.
const PEDANTIC_OPTIONS_AT: usize = 0x90;
const PEDANTIC_FLAG_AT: usize = 0x3A;
const PEDANTIC_ERROR_AT: usize = 0x0B;

/// taihen.h tai_module_info_t.
#[repr(C)]
struct TaiModuleInfo {
    size: u32,
    modid: SceUID,
    module_nid: u32,
    name: [u8; 27],
    exports_start: usize,
    exports_end: usize,
    imports_start: usize,
    imports_end: usize,
}
const _: () = assert!(core::mem::size_of::<TaiModuleInfo>() == 56);

/// taihen.h tai_offset_args_t.
#[repr(C)]
struct TaiOffsetArgs {
    size: u32,
    modid: SceUID,
    segidx: i32,
    offset: u32,
    thumb: i32,
    source: *const c_void,
    /// For injections only (0 for a hook).
    source_size: u32,
}
const _: () = assert!(core::mem::size_of::<TaiOffsetArgs>() == 28);

/// taihen.h struct _tai_hook_user: what a tai_hook_ref_t points at.
#[repr(C)]
struct TaiHookUser {
    next: *const TaiHookUser,
    func: *const c_void,
    old: *const c_void,
}

extern "C" {
    fn taiGetModuleInfo(module: *const u8, info: *mut TaiModuleInfo) -> i32;
    fn taiHookFunctionOffsetForUser(hook: *mut usize, args: *mut TaiOffsetArgs) -> SceUID;
    fn taiHookRelease(uid: SceUID, hook: usize) -> i32;
    fn taiInjectDataForUser(args: *mut TaiOffsetArgs) -> SceUID;
    fn taiInjectRelease(uid: SceUID) -> i32;
    fn sceKernelGetModuleInfo(uid: SceUID, info: *mut SceKernelModuleInfo) -> i32;
}

/// psp2common/kernel/modulemgr.h SceKernelModuleInfo (0x1B8 bytes); only
/// the first segment's address is read.
#[repr(C)]
struct SceKernelModuleInfo {
    size: u32,
    head: [u8; 0x154 - 4],
    segments: [SceKernelSegmentInfo; 4],
    state: u32,
}

#[repr(C)]
struct SceKernelSegmentInfo {
    size: u32,
    perms: u32,
    vaddr: usize,
    memsz: u32,
    filesz: u32,
    res: u32,
}
const _: () = assert!(core::mem::size_of::<SceKernelModuleInfo>() == 0x1B8);

// ---- Intrinsics: GPU instructions as Cg functions ----
//
// Declarations with `__attribute__((Intrinsic, OpcodeName(...)))` make a
// function one instruction (FBLEND: the GPU's 8-bit blend, which GXM's
// patcher appends to blending fragment programs). Only the compiler's own
// internal source may declare them. Before every compile it loads its
// standard library text through one function (`LoadInternalString`);
// SceShaccCgExt (as above) patches the call site to call its own loader
// instead, which loads its declarations, then the standard library. The
// same patches here, with our own declarations (INTRINSICS: only what our
// frame-read programs use).

/// The compiler's loader of internal source: (its state, NUL-terminated text).
const LOAD_INTERNAL_STRING_OFFSET: usize = 0x20_11EC;
/// The standard library's text (NUL-terminated) in segment 0.
const STDLIB_TEXT_OFFSET: usize = 0x29_A778;
/// The call that loads the standard library: replaced by a call to
/// `load_internal_source` (r0 = the state, r1 = no_stdlib).
const LOAD_CALL_OFFSET: u32 = 0x20_1D02;
/// The two ways into it (standard library and none): both now set r1 =
/// no_stdlib (r3) and branch to LOAD_CALL_OFFSET.
const NOSTDLIB_PATH_OFFSET: u32 = 0x20_1D32;
const STDLIB_PATH_OFFSET: u32 = 0x20_1B9C;

/// Thumb: movw r2, #lo; movt r2, #hi; mov r0, r4; blx r2; nop. The halves
/// of the address are filled in by `movw_movt`.
const LOAD_CALL_PATCH: [u16; 7] = [0xF240, 0x0200, 0xF2C0, 0x0200, 0x4620, 0x4790, 0xBF00];
/// Thumb: mov r1, r3; b LOAD_CALL_OFFSET (from NOSTDLIB_PATH_OFFSET).
const NOSTDLIB_PATH_PATCH: [u16; 2] = [0x4619, 0xE7E5];
/// Thumb: mov r1, r3; b.w LOAD_CALL_OFFSET (from STDLIB_PATH_OFFSET).
const STDLIB_PATH_PATCH: [u16; 3] = [0x4619, 0xF000, 0xB8B0];

/// Our intrinsic declarations (SceShaccCgExt's names and constants, so
/// sources read the same; operand order as FBLEND/FWBLEND take them).
/// Blend ops: ADD 1, SUB 2, MIN 3, MAX 4. Factors: ZERO 1, ONE 2, SRC1 3,
/// ONE_MINUS_SRC1 4, SRC2 5, ONE_MINUS_SRC2 6, SRC1_ALPHA 9,
/// ONE_MINUS_SRC1_ALPHA 10, SRC2_ALPHA 11, ONE_MINUS_SRC2_ALPHA 12. Write
/// mask bits R 1, G 2, B 4, A 8.
///
/// `tex2D_gather4` (the sample instruction's raw sample data: the four
/// texels of the bilinear footprint, unfiltered; the last form also the
/// filter's weights) and `tex2D_info` (the texture unit's own numbers for a
/// coordinate: u fraction, v fraction, trilinear fraction, level of detail)
/// are declared for lab programs; no program of the renderer uses them yet.
static INTRINSICS: &[u8] = b"\
#define __BLEND_OP_ADD 1
#define __BLEND_OP_SUB 2
#define __BLEND_OP_MIN 3
#define __BLEND_OP_MAX 4
#define __BLEND_FACTOR_ZERO 1
#define __BLEND_FACTOR_ONE 2
#define __BLEND_FACTOR_SRC1 3
#define __BLEND_FACTOR_ONE_MINUS_SRC1 4
#define __BLEND_FACTOR_SRC2 5
#define __BLEND_FACTOR_ONE_MINUS_SRC2 6
#define __BLEND_FACTOR_SRC1_ALPHA 9
#define __BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA 10
#define __BLEND_FACTOR_SRC2_ALPHA 11
#define __BLEND_FACTOR_ONE_MINUS_SRC2_ALPHA 12
#define __BLEND_WMASK_RGB 7
#define __BLEND_WMASK_RGBA 15
unsigned char4 __fastblend(unsigned char4 src1, unsigned char4 src2, int colorOp, int colorFactor1, int colorFactor2, int alphaOp, int alphaFactor1, int alphaFactor2) __attribute__((Intrinsic, OpcodeName(\"FBLEND\")));
unsigned char4 __fastblend_writemask(unsigned char4 src1, unsigned char4 src2, int colorOp, int factor1, int factor2, int alphaOp, unsigned char4 passThrough, int writeMask) __attribute__((Intrinsic, OpcodeName(\"FWBLEND\")));
unsigned char4 __add_sat(unsigned char4 a, unsigned char4 b) __attribute__((Intrinsic, OpcodeName(\"ADDSAT\")));
unsigned char4 __mul_sat(unsigned char4 a, unsigned char4 b) __attribute__((Intrinsic, OpcodeName(\"MULSAT\")));
unsigned char4 __min(unsigned char4 a, unsigned char4 b) __attribute__((Intrinsic, OpcodeName(\"MIN\")));
unsigned char4 __max(unsigned char4 a, unsigned char4 b) __attribute__((Intrinsic, OpcodeName(\"MAX\")));
unsigned short __pixel_x() __attribute__((Intrinsic, OpcodeName(\"PIXEL_X\")));
unsigned short __pixel_y() __attribute__((Intrinsic, OpcodeName(\"PIXEL_Y\")));
unsigned int __getTexCtrlWord(sampler2D s, unsigned int cwIndex) __attribute__((Intrinsic, OpcodeName(\"GETTEXCTRLWORD\"), Internal));
unsigned int __getTexCtrlWord(usampler2D s, unsigned int cwIndex) __attribute__((Intrinsic, OpcodeName(\"GETTEXCTRLWORD\"), Internal));
unsigned char4 __tex2D_sinf(unsigned int cw0, unsigned int cw1, unsigned int cw2, unsigned int cw3, float2 uv) __attribute__((Intrinsic, OpcodeName(\"TEX2D_SINF\"), Internal));
unsigned char4 tex2D_info(sampler2D smp, float2 uv) { return __tex2D_sinf(__getTexCtrlWord(smp, 0), __getTexCtrlWord(smp, 1), __getTexCtrlWord(smp, 2), __getTexCtrlWord(smp, 3), uv); }
unsigned char4 tex2D_info(usampler2D smp, float2 uv) { return __tex2D_sinf(__getTexCtrlWord(smp, 0), __getTexCtrlWord(smp, 1), __getTexCtrlWord(smp, 2), __getTexCtrlWord(smp, 3), uv); }
void tex2D_gather4(sampler2D, float2, out float[4]) __attribute__((Intrinsic, OpcodeName(\"TEX2D_RSD\")));
void tex2D_gather4(sampler2D, float2, out float2[4]) __attribute__((Intrinsic, OpcodeName(\"TEX2D_RSD\")));
void tex2D_gather4(sampler2D, float2, out float4[4]) __attribute__((Intrinsic, OpcodeName(\"TEX2D_RSD\")));
void tex2D_gather4(sampler2D, float2, out half[4]) __attribute__((Intrinsic, OpcodeName(\"TEX2D_RSD\")));
void tex2D_gather4(sampler2D, float2, out half2[4]) __attribute__((Intrinsic, OpcodeName(\"TEX2D_RSD\")));
void tex2D_gather4(sampler2D, float2, out half4[4]) __attribute__((Intrinsic, OpcodeName(\"TEX2D_RSD\")));
void tex2D_gather4(usampler2D, float2, out unsigned char[4]) __attribute__((Intrinsic, OpcodeName(\"TEX2D_RSD\")));
void tex2D_gather4(usampler2D, float2, out unsigned char2[4]) __attribute__((Intrinsic, OpcodeName(\"TEX2D_RSD\")));
void tex2D_gather4(usampler2D, float2, out unsigned char4[4]) __attribute__((Intrinsic, OpcodeName(\"TEX2D_RSD\")));
void tex2D_gather4(usampler2D, float2, out unsigned short4[4]) __attribute__((Intrinsic, OpcodeName(\"TEX2D_RSD\")));
void tex2D_gather4(sampler2D, float2, out float[4], out half4) __attribute__((Intrinsic, OpcodeName(\"TEX2D_IRSD\")));
\0";

type LoadInternalString = unsafe extern "C" fn(*mut c_void, *const u8) -> bool;

/// The compiler's loader and its standard library text, set before the
/// patches go in.
static LOADER: Shared<(usize, usize)> = Shared(UnsafeCell::new((0, 0)));

/// Called by the patched compiler where it loaded its standard library.
/// `no_stdlib` is the options' byte as the compiler holds it (a register:
/// not trusted to be 0 or 1, so not a Rust bool).
unsafe extern "C" fn load_internal_source(state: *mut c_void, no_stdlib: u32) -> bool {
    let (load, stdlib) = *LOADER.0.get();
    let load: LoadInternalString = core::mem::transmute(load);
    load(state, INTRINSICS.as_ptr()) && (no_stdlib & 0xFF != 0 || load(state, stdlib as *const u8))
}

/// `movw rN, #lo; movt rN, #hi` (Thumb-2, in `patch[0..4]`) for `value`.
fn movw_movt(patch: &mut [u16], value: u32) {
    for (half, immediate) in [(0, value & 0xFFFF), (2, value >> 16)] {
        patch[half] |= ((((immediate & 0x800) >> 11) << 10) | ((immediate & 0xF000) >> 12)) as u16;
        patch[half + 1] |= ((immediate & 0xFF) | (((immediate & 0x700) >> 8) << 12)) as u16;
    }
}

/// Puts in the loader patches: their inject ids, or why not (any put in
/// are released then).
fn enable_intrinsics(modid: SceUID) -> Result<[SceUID; 3], Text<128>> {
    // Safe: our own structs; the patched code is only run by the compiler,
    // on this thread, after all three patches are in.
    unsafe {
        let mut module: SceKernelModuleInfo = core::mem::zeroed();
        module.size = core::mem::size_of::<SceKernelModuleInfo>() as u32;
        let found = sceKernelGetModuleInfo(modid, &mut module);
        if found < 0 {
            return Err(Text::of(format_args!("module info: {:08x}", found as u32)));
        }
        let segment = module.segments[0].vaddr;
        *LOADER.0.get() = ((segment + LOAD_INTERNAL_STRING_OFFSET) | 1, segment + STDLIB_TEXT_OFFSET);

        let mut call = LOAD_CALL_PATCH;
        movw_movt(&mut call, load_internal_source as *const () as usize as u32);
        let patches: [(u32, *const u16, usize); 3] = [
            (LOAD_CALL_OFFSET, call.as_ptr(), call.len()),
            (NOSTDLIB_PATH_OFFSET, NOSTDLIB_PATH_PATCH.as_ptr(), NOSTDLIB_PATH_PATCH.len()),
            (STDLIB_PATH_OFFSET, STDLIB_PATH_PATCH.as_ptr(), STDLIB_PATH_PATCH.len()),
        ];
        let mut ids = [-1; 3];
        for (index, &(offset, code, halves)) in patches.iter().enumerate() {
            let mut args = TaiOffsetArgs {
                size: core::mem::size_of::<TaiOffsetArgs>() as u32,
                modid,
                segidx: 0,
                offset,
                thumb: 0,
                source: code as *const c_void,
                source_size: (halves * 2) as u32,
            };
            let id = taiInjectDataForUser(&mut args);
            if id < 0 {
                for &done in &ids[..index] {
                    taiInjectRelease(done);
                }
                return Err(Text::of(format_args!("inject at {offset:x} failed: {:08x}", id as u32)));
            }
            ids[index] = id;
        }
        Ok(ids)
    }
}

/// The hook's continuation (taiHEN's tai_hook_ref_t), set once before the
/// compiler runs.
static FILL_STATE_REF: Shared<usize> = Shared(UnsafeCell::new(0));

type FillState = unsafe extern "C" fn(u32, u32, u32, u32, u32, u32, u32, u32) -> *mut c_void;

/// The hooked function: sets the extension byte, then goes on as taiHEN's
/// TAI_CONTINUE does (the next hook, or the original).
unsafe extern "C" fn fill_state_hook(a1: u32, a2: u32, a3: u32, a4: u32, a5: u32, a6: u32, a7: u32, a8: u32) -> *mut c_void {
    *((a7 as usize + EXTENSION_FLAG_AT) as *mut u8) = 1;
    // Extensions only an error when the options ask for pedantic errors
    // (the options' flag copied into the third argument's state; else
    // e.g. the FRAGCOLOR semantic is refused "as an extension").
    let options = *((a1 as usize + PEDANTIC_OPTIONS_AT) as *const *const u8);
    *((a3 as usize + PEDANTIC_ERROR_AT) as *mut u8) = *options.add(PEDANTIC_FLAG_AT);
    let current = *FILL_STATE_REF.0.get() as *const TaiHookUser;
    let next = (*current).next;
    let target = if next.is_null() { (*current).old } else { (*next).func };
    let target: FillState = core::mem::transmute(target);
    target(a1, a2, a3, a4, a5, a6, a7, a8)
}

/// Turns the extensions on: the hook's id and the compiler's module id, or
/// why not.
fn enable_extensions() -> Result<(SceUID, SceUID), Text<128>> {
    // Safe: taiHEN calls with our own structs; the hook is installed
    // before the compiler is first called, on this thread.
    unsafe {
        let mut info: TaiModuleInfo = core::mem::zeroed();
        info.size = core::mem::size_of::<TaiModuleInfo>() as u32;
        let found = taiGetModuleInfo(b"SceShaccCg\0".as_ptr(), &mut info);
        if found < 0 {
            return Err(Text::of(format_args!("SceShaccCg not found: {:08x}", found as u32)));
        }
        if !KNOWN_SHACCCG_NIDS.contains(&info.module_nid) {
            return Err(Text::of(format_args!("unknown libshacccg build (NID {:08x})", info.module_nid)));
        }
        let mut args = TaiOffsetArgs {
            size: core::mem::size_of::<TaiOffsetArgs>() as u32,
            modid: info.modid,
            segidx: 0,
            offset: FILL_STATE_OFFSET,
            thumb: 1,
            source: fill_state_hook as *const c_void,
            source_size: 0,
        };
        let hook = taiHookFunctionOffsetForUser(FILL_STATE_REF.0.get(), &mut args);
        if hook < 0 {
            return Err(Text::of(format_args!("hook failed: {:08x}", hook as u32)));
        }
        Ok((hook, info.modid))
    }
}

// ---- Small helpers ----

/// A static touched by one thread at a time (the app's main thread; the
/// heap under its own lock).
struct Shared<T>(UnsafeCell<T>);
// Safe: see above.
unsafe impl<T> Sync for Shared<T> {}

/// Text built in a fixed buffer, cut off when full; always NUL-terminated,
/// so it can be a path.
struct Text<const N: usize> {
    bytes: [u8; N],
    len: usize,
}

impl<const N: usize> Text<N> {
    fn new() -> Self {
        Text { bytes: [0; N], len: 0 }
    }

    fn of(args: core::fmt::Arguments) -> Self {
        let mut text = Self::new();
        let _ = text.write_fmt(args);
        text
    }

    fn as_bytes(&self) -> &[u8] {
        &self.bytes[..self.len]
    }

    /// The text with its NUL.
    fn c_str(&self) -> *const u8 {
        self.bytes.as_ptr()
    }
}

impl<const N: usize> Write for Text<N> {
    fn write_str(&mut self, text: &str) -> core::fmt::Result {
        let room = (N - 1).saturating_sub(self.len);
        let take = text.len().min(room);
        self.bytes[self.len..self.len + take].copy_from_slice(&text.as_bytes()[..take]);
        self.len += take;
        self.bytes[self.len] = 0;
        Ok(())
    }
}

/// The bytes before the first NUL.
fn until_nul(bytes: &[u8]) -> &[u8] {
    let end = bytes.iter().position(|&byte| byte == 0).unwrap_or(bytes.len());
    &bytes[..end]
}

/// A C string from the compiler as text ("?" for null).
///
/// # Safety
/// `text` must be null or point to a NUL-terminated string.
unsafe fn c_text<'a>(text: *const u8) -> &'a str {
    if text.is_null() {
        return "?";
    }
    let mut len = 0;
    while *text.add(len) != 0 {
        len += 1;
    }
    core::str::from_utf8(core::slice::from_raw_parts(text, len)).unwrap_or("(not UTF-8)")
}

fn write_all(fd: SceUID, bytes: &[u8]) -> bool {
    // Safe: `bytes` is readable for its length.
    fd >= 0 && unsafe { sceIoWrite(fd, bytes.as_ptr() as *const c_void, bytes.len() as u32) } == bytes.len() as i32
}

fn now_us() -> u64 {
    // Safe: no arguments.
    unsafe { sceKernelGetProcessTimeWide() }
}

/// done.tmp, written as we go (renamed to done.txt at the end).
static SUMMARY: AtomicI32 = AtomicI32::new(-1);

fn say(args: core::fmt::Arguments) {
    let text = Text::<512>::of(args);
    write_all(SUMMARY.load(Ordering::Relaxed), text.as_bytes());
}

// ---- The compiler's heap ----

struct HeapState {
    memory: *mut u32,
    heap: ClassHeap,
}

static HEAP: Shared<HeapState> = Shared(UnsafeCell::new(HeapState { memory: core::ptr::null_mut(), heap: ClassHeap::new(0) }));
static HEAP_LOCK: AtomicBool = AtomicBool::new(false);

/// Runs `body` on the heap under its lock (the compiler may call from any
/// thread it makes).
fn with_heap<R>(body: impl FnOnce(&mut HeapState) -> R) -> R {
    while HEAP_LOCK.compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed).is_err() {
        core::hint::spin_loop();
    }
    // Safe: we hold the lock.
    let result = body(unsafe { &mut *HEAP.0.get() });
    HEAP_LOCK.store(false, Ordering::Release);
    result
}

/// The block's words.
///
/// # Safety
/// `memory` must be the block of HEAP_BYTES (made before the compiler gets
/// the heap), used only under the lock.
unsafe fn words<'a>(memory: *mut u32) -> &'a mut [u32] {
    core::slice::from_raw_parts_mut(memory, (HEAP_BYTES / 4) as usize)
}

/// Makes the heap's block. False if there is no memory.
fn heap_init() -> bool {
    // Safe: plain system calls; the block is ours alone.
    let base = unsafe {
        let block = sceKernelAllocMemBlock(b"NativeResShaccHeap\0".as_ptr(), SCE_KERNEL_MEMBLOCK_TYPE_USER_RW, HEAP_BYTES, core::ptr::null_mut());
        let mut base: *mut c_void = core::ptr::null_mut();
        if block < 0 || sceKernelGetMemBlockBase(block, &mut base) < 0 {
            say(format_args!("ERR no memory for the compiler's heap: {:08x}\n", block as u32));
            return false;
        }
        base
    };
    with_heap(|state| {
        state.memory = base as *mut u32;
        state.heap = ClassHeap::new((HEAP_BYTES / 4) as usize);
    });
    true
}

unsafe extern "C" fn heap_malloc(size: u32) -> *mut c_void {
    with_heap(|state| {
        let memory = state.memory as *mut u8;
        // Safe: the block exists (the compiler gets the heap after it); we
        // hold the lock.
        match state.heap.allocate(unsafe { words(state.memory) }, size) {
            // Safe: the heap hands out offsets inside the block.
            Some(offset) => unsafe { memory.add(offset as usize) as *mut c_void },
            None => core::ptr::null_mut(),
        }
    })
}

unsafe extern "C" fn heap_free(pointer: *mut c_void) {
    if pointer.is_null() {
        return;
    }
    with_heap(|state| {
        let offset = (pointer as usize).wrapping_sub(state.memory as usize);
        if offset < HEAP_BYTES as usize {
            // Safe: as in heap_malloc.
            state.heap.free(unsafe { words(state.memory) }, offset as u32);
        }
    })
}

// ---- Compiling ----

static SOURCE_TEXT: Shared<[u8; MAX_SOURCE + 1]> = Shared(UnsafeCell::new([0; MAX_SOURCE + 1]));
static SOURCE_FILE: Shared<SceShaccCgSourceFile> = Shared(UnsafeCell::new(SceShaccCgSourceFile { file_name: core::ptr::null(), text: core::ptr::null(), size: 0 }));

/// The compiler asks for its one source file (no includes): the one read.
unsafe extern "C" fn open_source(
    _name: *const u8,
    _from: *const SceShaccCgSourceLocation,
    _options: *const SceShaccCgCompileOptions,
    _error: *mut *const u8,
) -> *mut SceShaccCgSourceFile {
    SOURCE_FILE.0.get()
}

unsafe extern "C" fn release_source(_file: *const SceShaccCgSourceFile, _options: *const SceShaccCgCompileOptions) {}

/// Reads `path` into SOURCE_TEXT (NUL after it): its size, or None if it
/// can't be read or is empty or bigger than MAX_SOURCE.
fn read_source(path: *const u8) -> Option<u32> {
    // Safe: plain file calls; SOURCE_TEXT is only touched by this thread
    // (the compiler reads it later, on this thread too).
    unsafe {
        let fd = sceIoOpen(path, SCE_O_RDONLY, 0);
        if fd < 0 {
            return None;
        }
        let size = sceIoLseek32(fd, 0, SCE_SEEK_END);
        sceIoLseek32(fd, 0, SCE_SEEK_SET);
        let text = &mut *SOURCE_TEXT.0.get();
        let ok = size > 0 && size as usize <= MAX_SOURCE && sceIoRead(fd, text.as_mut_ptr() as *mut c_void, size as u32) == size;
        sceIoClose(fd);
        if !ok {
            return None;
        }
        text[size as usize] = 0;
        Some(size as u32)
    }
}

/// Whether the source holds `marker` (a comment line such as
/// "// shacc: precise"): per-file compile options.
fn source_has(size: u32, marker: &[u8]) -> bool {
    // Safe: SOURCE_TEXT holds the source just read (`size` bytes); only
    // this thread touches it.
    let text: &[u8; MAX_SOURCE + 1] = unsafe { &*SOURCE_TEXT.0.get() };
    let text = &text[..size as usize];
    text.windows(marker.len()).any(|window| window == marker)
}

/// True for a vertex program: the name without ".cg" ends in "_v".
fn is_vertex_program(name: &str) -> bool {
    name.strip_suffix(".cg").is_some_and(|stem| stem.ends_with("_v"))
}

/// Compiles one file; true on success.
fn compile(name: &str) -> bool {
    let started = now_us();
    let stem = &name[..name.len() - 3];
    let input = Text::<256>::of(format_args!("{SHADER_DIR}/{name}"));
    let output_path = Text::<256>::of(format_args!("{SHADER_DIR}/{stem}.gxp"));
    let log_path = Text::<256>::of(format_args!("{SHADER_DIR}/{stem}.gxp.txt"));

    let Some(size) = read_source(input.c_str()) else {
        say(format_args!("{name}: ERR cannot read (max {MAX_SOURCE} bytes)\n"));
        return false;
    };
    // Safe: the compiler reads these only during the call below.
    unsafe {
        *SOURCE_FILE.0.get() = SceShaccCgSourceFile { file_name: input.c_str(), text: (*SOURCE_TEXT.0.get()).as_ptr(), size };
    }

    // Safe: the structs are the header's layout (sizes asserted), filled
    // by the compiler's own initializers first; the output is read only
    // until it is destroyed.
    unsafe {
        let mut options: SceShaccCgCompileOptions = core::mem::zeroed();
        sceShaccCgInitializeCompileOptions(&mut options);
        options.main_source_file = input.c_str();
        options.target_profile = if is_vertex_program(name) { SCE_SHACCCG_PROFILE_VP } else { SCE_SHACCCG_PROFILE_FP };
        options.entry_function_name = b"main\0".as_ptr();
        options.locale = SCE_SHACCCG_ENGLISH;
        options.optimization_level = 3;
        // Fast maths and fast precision by default. "// shacc: precise":
        // without fast precision (the compiler otherwise takes varyings and
        // texture samples as fp16: a vertex colour arrives truncated to
        // fp16, up to 0.25 levels low). "// shacc: exact-math": without
        // fast maths (no reassociation).
        options.use_fastmath = !source_has(size, b"// shacc: exact-math") as i32;
        options.use_fastprecision = !source_has(size, b"// shacc: precise") as i32;

        let mut callbacks: SceShaccCgCallbackList = core::mem::zeroed();
        sceShaccCgInitializeCallbackList(&mut callbacks, SCE_SHACCCG_TRIVIAL);
        callbacks.open_file = Some(open_source);
        callbacks.release_file = Some(release_source);

        let output = sceShaccCgCompileProgram(&options, &callbacks, 0);
        if output.is_null() {
            say(format_args!("{name}: ERR compiler returned nothing\n"));
            return false;
        }
        let output = &*output;
        let log = sceIoOpen(log_path.c_str(), SCE_O_WRONLY | SCE_O_CREAT | SCE_O_TRUNC, 0o666);
        let mut first_error: Option<(&str, u32)> = None;
        let count = output.diagnostic_count.max(0) as usize;
        let diagnostics: &[SceShaccCgDiagnosticMessage] =
            if output.diagnostics.is_null() || count == 0 { &[] } else { core::slice::from_raw_parts(output.diagnostics, count) };
        for diagnostic in diagnostics {
            let line_number = if diagnostic.location.is_null() { 0 } else { (*diagnostic.location).line_number };
            let message = c_text(diagnostic.message);
            let level = match diagnostic.level {
                SCE_SHACCCG_DIAGNOSTIC_LEVEL_ERROR => "error",
                SCE_SHACCCG_DIAGNOSTIC_LEVEL_WARNING => "warning",
                _ => "info",
            };
            let line = Text::<512>::of(format_args!("{level} line {line_number}: {message}\n"));
            write_all(log, line.as_bytes());
            if diagnostic.level == SCE_SHACCCG_DIAGNOSTIC_LEVEL_ERROR && first_error.is_none() {
                first_error = Some((message, line_number));
            }
        }
        if log >= 0 {
            sceIoClose(log);
        }
        let ok = if !output.program_data.is_null() && output.program_size != 0 {
            let program = core::slice::from_raw_parts(output.program_data, output.program_size as usize);
            let fd = sceIoOpen(output_path.c_str(), SCE_O_WRONLY | SCE_O_CREAT | SCE_O_TRUNC, 0o666);
            let written = write_all(fd, program);
            if fd >= 0 {
                sceIoClose(fd);
            }
            if written {
                let ms = (now_us() - started) / 1000;
                say(format_args!("{name}: OK {} ({ms} ms)\n", output.program_size));
            } else {
                say(format_args!("{name}: ERR cannot write the .gxp\n"));
            }
            written
        } else {
            let (message, line_number) = first_error.unwrap_or(("no program produced", 0));
            say(format_args!("{name}: ERR line {line_number}: {message}\n"));
            false
        };
        sceShaccCgDestroyCompileOutput(output);
        ok
    }
}

/// Compiles every .cg in SHADER_DIR.
fn compile_all() {
    let dir_path = Text::<64>::of(format_args!("{SHADER_DIR}"));
    // Safe: plain directory calls into our own entry.
    unsafe {
        let dir = sceIoDopen(dir_path.c_str());
        if dir < 0 {
            say(format_args!("ERR cannot open {SHADER_DIR}: {:08x}\n", dir as u32));
            return;
        }
        let mut entry: SceIoDirent = core::mem::zeroed();
        while sceIoDread(dir, &mut entry) > 0 {
            let name = until_nul(&entry.name);
            if let Ok(name) = core::str::from_utf8(name) {
                if name.len() > 3 && name.ends_with(".cg") {
                    compile(name);
                }
            }
        }
        sceIoDclose(dir);
    }
}

fn run() {
    let started = now_us();
    let done = Text::<64>::of(format_args!("{SHADER_DIR}/done.txt"));
    let temporary = Text::<64>::of(format_args!("{SHADER_DIR}/done.tmp"));
    // Safe: plain file calls.
    unsafe {
        sceIoRemove(done.c_str());
        SUMMARY.store(sceIoOpen(temporary.c_str(), SCE_O_WRONLY | SCE_O_CREAT | SCE_O_TRUNC, 0o666), Ordering::Relaxed);
    }

    // Safe: loads Sony's compiler; its imports of ours are bound then.
    let module = unsafe { sceKernelLoadStartModule(SHACCCG_PATH.as_ptr(), 0, core::ptr::null_mut(), 0, core::ptr::null_mut(), core::ptr::null_mut()) };
    if module < 0 {
        say(format_args!("ERR cannot load ur0:data/libshacccg.suprx: {:08x}\n", module as u32));
    } else if heap_init() {
        // Safe: the heap is ready; the compiler is loaded.
        unsafe { sceShaccCgSetDefaultAllocator(heap_malloc, heap_free) };
        let extensions = enable_extensions();
        let intrinsics = match &extensions {
            Ok((_, modid)) => {
                let intrinsics = enable_intrinsics(*modid);
                match &intrinsics {
                    Ok(_) => say(format_args!("# extensions on, intrinsics on\n")),
                    Err(why) => say(format_args!("# extensions on, intrinsics off: {}\n", core::str::from_utf8(why.as_bytes()).unwrap_or("?"))),
                }
                intrinsics.ok()
            }
            Err(why) => {
                say(format_args!("# extensions off: {}\n", core::str::from_utf8(why.as_bytes()).unwrap_or("?")));
                None
            }
        };
        compile_all();
        // Safe: the compiler is loaded; the hook and the patches are ours.
        unsafe {
            sceShaccCgReleaseCompiler();
            for id in intrinsics.into_iter().flatten() {
                taiInjectRelease(id);
            }
            if let Ok((hook, _)) = extensions {
                taiHookRelease(hook, *FILL_STATE_REF.0.get());
            }
        }
        let (peak, top, total, live, failed, bad_frees) = with_heap(|state| {
            let heap = &state.heap;
            (heap.peak(), heap.top(), heap.total_allocations(), heap.allocations(), heap.failed(), heap.bad_frees())
        });
        say(format_args!(
            "# heap peak={peak} top={top} of {HEAP_BYTES} allocations={total} left={live} failed={failed} bad_frees={bad_frees}, {} ms\n",
            (now_us() - started) / 1000
        ));
    }
    finish();
}

/// Closes done.tmp and renames it, so done.txt only ever appears complete.
fn finish() {
    let done = Text::<64>::of(format_args!("{SHADER_DIR}/done.txt"));
    let temporary = Text::<64>::of(format_args!("{SHADER_DIR}/done.tmp"));
    let summary = SUMMARY.swap(-1, Ordering::Relaxed);
    // Safe: plain file calls.
    unsafe {
        if summary >= 0 {
            sceIoClose(summary);
        }
        sceIoRename(temporary.c_str(), done.c_str());
    }
}

/// The app's entry (the process's main thread).
#[no_mangle]
pub extern "C" fn module_start(_args: u32, _argp: *mut c_void) -> i32 {
    run();
    loop {
        // Safe: ends the app.
        unsafe { sceKernelExitProcess(0) };
    }
}

/// A bug of ours: say so in done.txt, then quit.
#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    say(format_args!("PANIC: {info}\n"));
    finish();
    loop {
        // Safe: ends the app.
        unsafe { sceKernelExitProcess(1) };
    }
}
