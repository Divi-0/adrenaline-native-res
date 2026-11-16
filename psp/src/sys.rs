//! What the PSP kernel module is made of besides our code: the module info
//! and export table the PSP's loader reads, the import stubs it patches
//! (same libraries, NIDs and flags as pspsdk's for the C plugin), and the
//! few things only assembly can do (k1, VFPU copies).
//!
//! **Calling conventions.** rustc's PSP target uses MIPS o32, the PSP's
//! firmware (pspsdk's gcc) the EABI. They agree up to four integer
//! arguments (a0-a3, result in v0): all our hooks and most imports. Beyond
//! that o32 passes arguments on the stack (16($sp), 20($sp), ...), the EABI
//! in t0-t3: the three imports with more ([`sceKernelCreateSema`],
//! [`sceKernelCreateThread`], [`sceKernelAllocPartitionMemory`]) go through
//! shims that load t0/t1 from the stack and jump to the real stub.
//!
//! **No gp-relative data.** Our hooks run inside the game's system calls,
//! where $gp is not ours (the C plugin was built with -G0 for this):
//! tools/build-psp-plugin.sh turns LLVM's small-data sections off and
//! checks the ELF for gp-relative relocations.

use core::arch::{asm, global_asm};
use core::ffi::c_void;

pub type SceUID = i32;

// Module info (PSP_MODULE_INFO("NativePSP", 0x1007, 1, 0): kernel mode),
// the export table's bounds, and the syslib exports (module_start,
// module_stop, module_info), as pspsdk's crt0 and exports.exp made them.
global_asm!(
    ".set push",
    ".set noreorder",
    ".section .lib.ent.top, \"a\", @progbits",
    ".align 2",
    ".word 0",
    "__lib_ent_top:",
    ".section .lib.ent.btm, \"a\", @progbits",
    ".align 2",
    "__lib_ent_bottom:",
    ".word 0",
    ".section .lib.stub.top, \"a\", @progbits",
    ".align 2",
    ".word 0",
    "__lib_stub_top:",
    ".section .lib.stub.btm, \"a\", @progbits",
    ".align 2",
    "__lib_stub_bottom:",
    ".word 0",
    // syslib: name 0, version 0, attribute 0x8000, entry length 4, one
    // variable, two functions.
    ".section .lib.ent, \"a\", @progbits",
    ".align 2",
    ".word 0",
    ".hword 0x0000",
    ".hword 0x8000",
    ".byte 4, 1",
    ".hword 2",
    ".word __nps_syslib_exports",
    ".section .rodata.sceResident, \"a\", @progbits",
    ".align 2",
    "__nps_syslib_exports:",
    ".word 0xD632ACDB", // module_start
    ".word 0xCEE8593C", // module_stop
    ".word 0xF01D73A7", // module_info
    ".word module_start",
    ".word module_stop",
    ".word module_info",
    ".section .rodata.sceModuleInfo, \"a\", @progbits",
    ".align 4",
    ".global module_info",
    ".type module_info, @object",
    "module_info:",
    ".hword 0x1007",
    ".byte 0, 1",
    ".ascii \"NativePSP\"",
    ".space 28 - 9",
    ".word _gp",
    ".word __lib_ent_top",
    ".word __lib_ent_bottom",
    ".word __lib_stub_top",
    ".word __lib_stub_bottom",
    ".size module_info, 52",
    ".set pop",
);

/// One import library: its name, flags (version << 16 | attribute) and
/// function count, then each stub's symbol and NID in the order of the
/// stub table. Each stub is `jr $ra; nop` until the loader patches it.
macro_rules! import_library {
    ($library:literal, $flags:literal, $count:literal, $( $symbol:ident = $nid:literal ),+ $(,)?) => {
        global_asm!(concat!(
            ".set push\n.set noreorder\n",
            ".section .rodata.sceResident, \"a\", @progbits\n.align 2\n",
            "__nps_name_", $library, ":\n.asciz \"", $library, "\"\n",
            ".section .rodata.sceNid, \"a\", @progbits\n.align 2\n",
            "__nps_nids_", $library, ":\n",
            $( ".word ", stringify!($nid), "\n", )+
            ".section .sceStub.text, \"ax\", @progbits\n.align 2\n",
            "__nps_stubs_", $library, ":\n",
            $( ".global ", stringify!($symbol), "\n.type ", stringify!($symbol), ", @function\n",
               stringify!($symbol), ":\njr $ra\nnop\n", )+
            ".section .lib.stub, \"a\", @progbits\n.align 2\n",
            ".word __nps_name_", $library, "\n",
            ".word ", stringify!($flags), "\n",
            ".byte 5, 0\n",
            ".hword ", stringify!($count), "\n",
            ".word __nps_nids_", $library, "\n",
            ".word __nps_stubs_", $library, "\n",
            ".set pop\n",
        ));
    };
}

import_library!("SystemCtrlForKernel", 0x00090000, 2,
    sctrlHENFindFunction = 0x159AF5CC,
    sctrlHENPatchSyscall = 0xF988C1DC,
);
import_library!("LoadCoreForKernel", 0x00090000, 1,
    sceKernelIcacheClearAll = 0xD8779AC6,
);
import_library!("UtilsForKernel", 0x00090000, 4,
    sceKernelDcacheWritebackInvalidateAll = 0xB435DEC5,
    sceKernelDcacheWritebackRange = 0x3EE30821,
    sceKernelDcacheWritebackInvalidateRange = 0x34B9FA9E,
    sceKernelDcacheInvalidateRange = 0xBFA98062,
);
import_library!("InterruptManagerForKernel", 0x00090000, 2,
    sceKernelCpuSuspendIntr = 0x092968F4,
    sceKernelCpuResumeIntr = 0x5F10D406,
);
import_library!("ThreadManForKernel", 0x00090000, 10,
    sceKernelDelayThread = 0xCEADEB47,
    __nps_eabi_sceKernelCreateSema = 0xD6DA4BA1,
    sceKernelDeleteSema = 0x28B6489C,
    sceKernelSignalSema = 0x3F53E640,
    sceKernelWaitSema = 0x4E3A1105,
    sceKernelGetSystemTimeLow = 0x369ED59D,
    __nps_eabi_sceKernelCreateThread = 0x446D8DE6,
    sceKernelStartThread = 0xF475845D,
    sceKernelGetThreadId = 0x293B45B8,
    sceKernelReferThreadStatus = 0x17C1684E,
);
// memcpy/memset under their own names: the code rustc generates calls them
// too (no compiler_builtins copies), as the C plugin used the kernel's.
import_library!("SysclibForKernel", 0x00090000, 2,
    memcpy = 0xAB7592FF,
    memset = 0x10F3BB61,
);
import_library!("SysMemForKernel", 0x00010000, 2,
    __nps_eabi_sceKernelAllocPartitionMemory = 0x237DBD4F,
    sceKernelGetBlockHeadAddr = 0x9D9A5BA1,
);

// o32 -> EABI for the imports with five or six arguments (see the module
// docs): argument 5 to t0, 6 to t1, then the stub, which returns to our
// caller.
global_asm!(
    ".set push",
    ".set noreorder",
    ".section .text.nps_eabi, \"ax\", @progbits",
    ".align 2",
    ".global sceKernelCreateSema",
    ".type sceKernelCreateSema, @function",
    "sceKernelCreateSema:",
    "j __nps_eabi_sceKernelCreateSema",
    "lw $t0, 16($sp)",
    ".global sceKernelCreateThread",
    ".type sceKernelCreateThread, @function",
    "sceKernelCreateThread:",
    "lw $t0, 16($sp)",
    "j __nps_eabi_sceKernelCreateThread",
    "lw $t1, 20($sp)",
    ".global sceKernelAllocPartitionMemory",
    ".type sceKernelAllocPartitionMemory, @function",
    "sceKernelAllocPartitionMemory:",
    "j __nps_eabi_sceKernelAllocPartitionMemory",
    "lw $t0, 16($sp)",
    ".set pop",
);

// VFPU copy, 64 bytes per step: `nps_vfpu_copy64(from, to, steps)`, both
// 16-byte aligned. LLVM's assembler doesn't know the VFPU, so lv.q / sv.q
// are written as words (as psp-as assembles `lv.q C000, 0($a0)` ... for
// the C plugin's copy_vram_vfpu, with $a0 / $a1 as the base registers).
global_asm!(
    ".set push",
    ".set noreorder",
    ".section .text.nps_vfpu_copy64, \"ax\", @progbits",
    ".align 2",
    ".global nps_vfpu_copy64",
    ".type nps_vfpu_copy64, @function",
    "nps_vfpu_copy64:",
    "beqz $a2, 2f",
    "nop",
    "1:",
    ".word 0xD8800000", // lv.q C000, 0($a0)
    ".word 0xD8810010", // lv.q C010, 16($a0)
    ".word 0xD8820020", // lv.q C020, 32($a0)
    ".word 0xD8830030", // lv.q C030, 48($a0)
    ".word 0xF8A00000", // sv.q C000, 0($a1)
    ".word 0xF8A10010", // sv.q C010, 16($a1)
    ".word 0xF8A20020", // sv.q C020, 32($a1)
    ".word 0xF8A30030", // sv.q C030, 48($a1)
    "addiu $a2, $a2, -1",
    "addiu $a0, $a0, 64",
    "bnez $a2, 1b",
    "addiu $a1, $a1, 64",
    "2:",
    "jr $ra",
    "nop",
    ".set pop",
);

extern "C" {
    /// The address of a kernel function, by module, library and NID (0 =
    /// not found).
    pub fn sctrlHENFindFunction(module: *const u8, library: *const u8, nid: u32) -> u32;
    /// Points the system call of the function at `address` to `function`.
    pub fn sctrlHENPatchSyscall(address: u32, function: *const c_void) -> i32;

    pub fn sceKernelIcacheClearAll();
    pub fn sceKernelDcacheWritebackInvalidateAll();
    pub fn sceKernelDcacheWritebackRange(address: *const c_void, size: u32);
    pub fn sceKernelDcacheWritebackInvalidateRange(address: *const c_void, size: u32);
    pub fn sceKernelDcacheInvalidateRange(address: *const c_void, size: u32);

    pub fn sceKernelCpuSuspendIntr() -> u32;
    pub fn sceKernelCpuResumeIntr(flags: u32);

    pub fn sceKernelDelayThread(us: u32) -> i32;
    pub fn sceKernelCreateSema(name: *const u8, attr: u32, initial: i32, max: i32, option: *mut c_void) -> SceUID;
    pub fn sceKernelDeleteSema(id: SceUID) -> i32;
    pub fn sceKernelSignalSema(id: SceUID, signal: i32) -> i32;
    pub fn sceKernelWaitSema(id: SceUID, signal: i32, timeout: *mut u32) -> i32;
    pub fn sceKernelGetSystemTimeLow() -> u32;
    pub fn sceKernelCreateThread(
        name: *const u8,
        entry: unsafe extern "C" fn(u32, *mut c_void) -> i32,
        priority: i32,
        stack_size: i32,
        attr: u32,
        option: *mut c_void,
    ) -> SceUID;
    pub fn sceKernelStartThread(id: SceUID, length: u32, argp: *mut c_void) -> i32;
    pub fn sceKernelGetThreadId() -> SceUID;
    pub fn sceKernelReferThreadStatus(id: SceUID, info: *mut ThreadInfo) -> i32;

    /// SysclibForKernel's memcpy (copy method `MEMCPY`).
    pub fn memcpy(to: *mut c_void, from: *const c_void, size: usize) -> *mut c_void;

    pub fn sceKernelAllocPartitionMemory(partition: i32, name: *const u8, kind: i32, size: u32, address: *mut c_void) -> SceUID;
    pub fn sceKernelGetBlockHeadAddr(id: SceUID) -> *mut c_void;

    pub fn nps_vfpu_copy64(from: *const u32, to: *mut u32, steps: u32);
}

/// SceKernelThreadInfo (104 bytes: size, name[32], attr, ...).
#[repr(C)]
pub struct ThreadInfo {
    pub size: u32,
    pub name: [u8; 32],
    pub attr: u32,
    pub rest: [u32; 16],
}

const _: () = assert!(core::mem::size_of::<ThreadInfo>() == 104);

/// PSP_THREAD_ATTR_VFPU: the kernel saves the thread's VFPU state.
pub const THREAD_ATTR_VFPU: u32 = 0x0000_4000;
/// PSP_SMEM_High: a block at the top of the partition.
pub const SMEM_HIGH: i32 = 1;
/// The user partition.
pub const USER_PARTITION: i32 = 2;

/// Sets $k1 (pspSdkSetK1), returns the old value. In the game's system
/// call $k1 says "user-mode caller", and the kernel then refuses our
/// (kernel) objects.
#[inline(always)]
pub fn set_k1(value: u32) -> u32 {
    let old: u32;
    // Safe: only $k1 changes; the caller puts it back.
    // Fixed registers: the allocator may otherwise pick $at.
    unsafe { asm!("move $t0, $k1", "move $k1, $t1", out("$8") old, in("$9") value, options(nomem, nostack)) };
    old
}
