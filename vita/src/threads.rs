//! Our threads: the CPU time they have run, and the cores they may run on.

use crate::sys::{self, SceUID};

/// CPU time `thread` has run, in microseconds (advances at context
/// switches: good over many of them, e.g. a whole capturebench). 0 for a
/// bad id.
pub fn run_us_of(thread: SceUID) -> u64 {
    // Safe: all zeroes is a valid SceKernelThreadInfo (numbers, a null
    // entry and stack).
    let mut info: sys::SceKernelThreadInfo = unsafe { core::mem::zeroed() };
    info.size = core::mem::size_of::<sys::SceKernelThreadInfo>() as u32;
    // Safe: `info` is a valid, sized struct.
    if unsafe { sys::sceKernelGetThreadInfo(thread, &mut info) } < 0 {
        return 0;
    }
    info.run_clocks
}

/// CPU time the calling thread has run, in microseconds (for the log; it
/// only advances at context switches, so don't time anything with it).
pub fn run_us() -> u64 {
    // Safe: no arguments.
    run_us_of(unsafe { sys::sceKernelGetThreadId() })
}

/// Core 3: the kernel refuses it to apps. Only an extension can offer it
/// (`ext::Extensions::set_cores_with_core_3`); the renderer's own threads
/// stay on cores 0-2 by default.
pub const CORE_3: i32 = 0x8_0000;

/// Refused, as the kernel answers (SCE_KERNEL_ERROR_ILLEGAL_CPU_AFFINITY_MASK).
const REFUSED: i32 = 0x8002_0035_u32 as i32;

/// The calling thread runs on the cores in `mask` (0x10000 = core 0 ...
/// 0x80000 = core 3). < 0 if refused.
pub fn set_cores(mask: i32) -> i32 {
    if mask & CORE_3 != 0 {
        return crate::ext::set_cores_with_core_3(mask).unwrap_or(REFUSED);
    }
    // Safe: system calls with plain numbers.
    unsafe { sys::sceKernelChangeThreadCpuAffinityMask(sys::sceKernelGetThreadId(), mask) }
}
