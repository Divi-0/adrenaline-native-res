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

use core::ffi::c_void;

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
    clib_memcpy(target as *mut c_void, source as *const c_void, size);
}

/// # Safety
/// As memcpy; both 4-byte aligned.
#[no_mangle]
pub unsafe extern "C" fn __aeabi_memcpy4(target: *mut u8, source: *const u8, size: usize) {
    clib_memcpy(target as *mut c_void, source as *const c_void, size);
}

/// # Safety
/// As memcpy; both 8-byte aligned.
#[no_mangle]
pub unsafe extern "C" fn __aeabi_memcpy8(target: *mut u8, source: *const u8, size: usize) {
    clib_memcpy(target as *mut c_void, source as *const c_void, size);
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
    clib_memset(target as *mut c_void, 0, size);
}

/// # Safety
/// As memset with 0; 8-byte aligned.
#[no_mangle]
pub unsafe extern "C" fn __aeabi_memclr8(target: *mut u8, size: usize) {
    clib_memset(target as *mut c_void, 0, size);
}
