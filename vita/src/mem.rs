//! The ARM EABI memory helpers (`__aeabi_memcpy` and friends) for our Rust
//! code, forwarded to Sony's SceLibKernel (`sceClibMemcpy` and friends).
//!
//! LLVM turns struct moves, slice copies and zeroing into calls to these.
//! The only other definitions in the module are compiler_builtins' (weak
//! symbols): plain Rust loops that move one word per ~5 instructions, so
//! moving a 1,160-byte `transform::Transform` took ~1,450 instructions
//! (found 2026-11-07 with the PMU counters: `prepare` ran ~1,670
//! instructions per draw on the Vita against ~640 on the PC). Sony's
//! sceClib functions are optimized ARM code. These strong definitions win
//! over the weak ones at link time.
//!
//! They must not call anything named memcpy / memmove / memset: LLVM knows
//! those as builtins and lowers the call straight back to these helpers
//! (the first try called itself forever; caught in the disassembly).
//!
//! Copies from uncached PSP memory don't come here: they go through
//! `platform::psp_copy` (VLDM bursts).
//!
//! Small aligned copies and clears (`SMALL` bytes or fewer: most struct
//! moves, e.g. a draw's 108-byte `Prepared` two or three times per PRIM
//! batch) are done here, not in sceClib: each sceClib call is an import
//! call into another module's code, which the capture thread's per-batch
//! code finds cold in the instruction cache (2026-11-17: clearing an unused
//! 512-byte array per batch cost the capture thread ~0.5 ms per frame).
//! Measured 2026-11-21 (MTX race start, deploy A/B): walk -1.0 ms per frame;
//! raising `SMALL` to 2,048 bytes gained nothing more.

use core::ffi::c_void;
use core::ptr::{read_volatile, write_volatile};

/// Aligned copies and clears up to this many bytes stay in this module.
const SMALL: usize = 256;

/// Whether a copy or clear of `size` bytes stays here. No switch to turn it
/// off: a global read per struct move would cost what it saves (judge it
/// by a build without it).
#[inline(always)]
fn small_ones_here(size: usize) -> bool {
    size <= SMALL
}

/// Copies `size` bytes, both pointers 4-byte aligned, not overlapping: 32
/// bytes per VLDM / VSTM pair (d0-d3: free to use in any call), then words,
/// then bytes. Volatile word and byte accesses, so that LLVM doesn't turn
/// the loops back into a memcpy call (this is the memcpy).
#[inline(always)]
unsafe fn copy_aligned(mut target: *mut u8, mut source: *const u8, mut size: usize) {
    while size >= 32 {
        #[cfg(target_arch = "arm")]
        core::arch::asm!(
            ".fpu neon",
            "vldmia {from}!, {{d0-d3}}",
            "vstmia {to}!, {{d0-d3}}",
            from = inout(reg) source,
            to = inout(reg) target,
            out("d0") _, out("d1") _, out("d2") _, out("d3") _,
            options(nostack, preserves_flags),
        );
        #[cfg(not(target_arch = "arm"))]
        {
            for word in 0..8 {
                write_volatile((target as *mut u32).add(word), read_volatile((source as *const u32).add(word)));
            }
            target = target.add(32);
            source = source.add(32);
        }
        size -= 32;
    }
    while size >= 4 {
        write_volatile(target as *mut u32, read_volatile(source as *const u32));
        target = target.add(4);
        source = source.add(4);
        size -= 4;
    }
    while size > 0 {
        write_volatile(target, read_volatile(source));
        target = target.add(1);
        source = source.add(1);
        size -= 1;
    }
}

/// Clears `size` bytes at `target` (4-byte aligned), as `copy_aligned`
/// copies.
#[inline(always)]
unsafe fn clear_aligned(mut target: *mut u8, mut size: usize) {
    let blocks = size / 32;
    if blocks > 0 {
        // Zeros in d0-d3, stored 32 bytes per VSTM (one asm block: the
        // registers must keep the zeros between the stores).
        #[cfg(target_arch = "arm")]
        core::arch::asm!(
            ".fpu neon",
            "vmov.i32 q0, #0",
            "vmov.i32 q1, #0",
            "2:",
            "vstmia {to}!, {{d0-d3}}",
            "subs {blocks}, {blocks}, #1",
            "bne 2b",
            to = inout(reg) target,
            blocks = inout(reg) blocks => _,
            out("d0") _, out("d1") _, out("d2") _, out("d3") _,
            options(nostack),
        );
        #[cfg(not(target_arch = "arm"))]
        for _ in 0..blocks {
            for word in 0..8 {
                write_volatile((target as *mut u32).add(word), 0);
            }
            target = target.add(32);
        }
        size -= 32 * blocks;
    }
    while size >= 4 {
        write_volatile(target as *mut u32, 0);
        target = target.add(4);
        size -= 4;
    }
    while size > 0 {
        write_volatile(target, 0);
        target = target.add(1);
        size -= 1;
    }
}

extern "C" {
    // SceLibKernel (imported by the module, see user/CMakeLists.txt).
    #[link_name = "sceClibMemcpy"]
    fn clib_memcpy(target: *mut c_void, source: *const c_void, size: usize) -> *mut c_void;
    #[link_name = "sceClibMemmove"]
    fn clib_memmove(target: *mut c_void, source: *const c_void, size: usize) -> *mut c_void;
    #[link_name = "sceClibMemset"]
    fn clib_memset(target: *mut c_void, value: i32, size: usize) -> *mut c_void;
}

/// # Safety
/// As memcpy.
#[no_mangle]
pub unsafe extern "C" fn __aeabi_memcpy(target: *mut u8, source: *const u8, size: usize) {
    // Unknown alignment: small ones whose pointers happen to be aligned
    // stay here as well (the copy tasks' small slices).
    if small_ones_here(size) && (target as usize | source as usize) & 3 == 0 {
        copy_aligned(target, source, size);
    } else {
        clib_memcpy(target as *mut c_void, source as *const c_void, size);
    }
}

/// # Safety
/// As memcpy; both 4-byte aligned.
#[no_mangle]
pub unsafe extern "C" fn __aeabi_memcpy4(target: *mut u8, source: *const u8, size: usize) {
    if small_ones_here(size) {
        copy_aligned(target, source, size);
    } else {
        clib_memcpy(target as *mut c_void, source as *const c_void, size);
    }
}

/// # Safety
/// As memcpy; both 8-byte aligned.
#[no_mangle]
pub unsafe extern "C" fn __aeabi_memcpy8(target: *mut u8, source: *const u8, size: usize) {
    if small_ones_here(size) {
        copy_aligned(target, source, size);
    } else {
        clib_memcpy(target as *mut c_void, source as *const c_void, size);
    }
}

/// # Safety
/// As memmove.
#[no_mangle]
pub unsafe extern "C" fn __aeabi_memmove(target: *mut u8, source: *const u8, size: usize) {
    clib_memmove(target as *mut c_void, source as *const c_void, size);
}

/// # Safety
/// As memmove; both 4-byte aligned.
#[no_mangle]
pub unsafe extern "C" fn __aeabi_memmove4(target: *mut u8, source: *const u8, size: usize) {
    clib_memmove(target as *mut c_void, source as *const c_void, size);
}

/// # Safety
/// As memmove; both 8-byte aligned.
#[no_mangle]
pub unsafe extern "C" fn __aeabi_memmove8(target: *mut u8, source: *const u8, size: usize) {
    clib_memmove(target as *mut c_void, source as *const c_void, size);
}

/// Note the EABI argument order: size before value.
///
/// # Safety
/// As memset.
#[no_mangle]
pub unsafe extern "C" fn __aeabi_memset(target: *mut u8, size: usize, value: i32) {
    clib_memset(target as *mut c_void, value, size);
}

/// # Safety
/// As memset; 4-byte aligned.
#[no_mangle]
pub unsafe extern "C" fn __aeabi_memset4(target: *mut u8, size: usize, value: i32) {
    clib_memset(target as *mut c_void, value, size);
}

/// # Safety
/// As memset; 8-byte aligned.
#[no_mangle]
pub unsafe extern "C" fn __aeabi_memset8(target: *mut u8, size: usize, value: i32) {
    clib_memset(target as *mut c_void, value, size);
}

/// # Safety
/// As memset with 0.
#[no_mangle]
pub unsafe extern "C" fn __aeabi_memclr(target: *mut u8, size: usize) {
    clib_memset(target as *mut c_void, 0, size);
}

/// # Safety
/// As memset with 0; 4-byte aligned.
#[no_mangle]
pub unsafe extern "C" fn __aeabi_memclr4(target: *mut u8, size: usize) {
    if small_ones_here(size) {
        clear_aligned(target, size);
    } else {
        clib_memset(target as *mut c_void, 0, size);
    }
}

/// # Safety
/// As memset with 0; 8-byte aligned.
#[no_mangle]
pub unsafe extern "C" fn __aeabi_memclr8(target: *mut u8, size: usize) {
    if small_ones_here(size) {
        clear_aligned(target, size);
    } else {
        clib_memset(target as *mut c_void, 0, size);
    }
}
