//! PSP memory as the Vita sees it: where it is mapped, and copying out of
//! it quickly. Used by the capture threads and the display thread.

use core::ffi::c_void;
use core::sync::atomic::{AtomicU32, Ordering};

use nativepsp_core::memory::{place_read, PspPlace};

use crate::sys;

/// The PSP plugin's copy of VRAM (`psp_link::VRAM_COPY_ADDRESS`). Sony's
/// emulator keeps the real VRAM where we can't read it; the plugin copies
/// the ranges we ask for there (see `nativepsp_core::vram_requests`).
use nativepsp_core::psp_link::VRAM_COPY_ADDRESS;

/// Reads `psp_memory` refused (outside VRAM, RAM and the
/// scratchpad), for the capture log (`badread=`). Each one would have been
/// a crash before 2026-11-06.
pub static REFUSED_READS: AtomicU32 = AtomicU32::new(0);
/// The last refused read's address and size (for the log).
pub static LAST_REFUSED_ADDRESS: AtomicU32 = AtomicU32::new(0);
pub static LAST_REFUSED_SIZE: AtomicU32 = AtomicU32::new(0);

/// Where `size` bytes of PSP memory at physical `address` can be read on
/// the Vita, or null. VRAM addresses (also in VRAM's mirrors) lead to the
/// plugin's copy. Null for anything that isn't VRAM, RAM or the scratchpad
/// (`nativepsp_core::memory::place_read`, tested there).
///
/// We read PSP RAM directly (like the native framebuffer in menu.c) instead
/// of using ScePspemuConvertAddress(KERMIT_INPUT_MODE): that call does
/// cache maintenance with unknown effects on data the game is writing at
/// the same moment. A plain read can at worst see stale data.
pub fn psp_memory(address: u32, size: u32) -> *const u8 {
    match place_read(address, size) {
        Some(PspPlace::Vram(offset)) => sys::psp_to_vita((VRAM_COPY_ADDRESS & 0x0FFF_FFFF) + offset) as *const u8,
        Some(PspPlace::Mapped(address)) => sys::psp_to_vita(address) as *const u8,
        None => {
            LAST_REFUSED_ADDRESS.store(address, Ordering::Relaxed);
            LAST_REFUSED_SIZE.store(size, Ordering::Relaxed);
            REFUSED_READS.fetch_add(1, Ordering::Relaxed);
            core::ptr::null()
        }
    }
}

/// Copies `size` bytes out of (uncached) PSP memory. Measured on the Vita
/// for 64 KiB: word by word 2.2 ms, memcpy 0.7 ms, VLDM bursts ~0.4 ms, one
/// DMA 0.18 ms, but 64 DMAs of 1 KiB 4.5-9 ms (each DMA call has a high
/// fixed cost). So small copies use VLDM, large ones DMA.
///
/// # Safety
/// `target` and `source` must hold `size` bytes and not overlap.
pub unsafe fn psp_copy(target: *mut u8, source: *const u8, size: u32) {
    if size >= 16 * 1024 {
        sys::sceDmacMemcpy(target as *mut c_void, source as *const c_void, size);
    } else {
        copy_uncached(target, source, size as usize);
    }
}

/// Copies with VLDM loads of 128 bytes: the memory system then fetches in
/// bursts (64 KiB of PSP RAM: memcpy ~0.7 ms, this ~0.4 ms). The last
/// bytes (less than 128) go through `copy_tail`.
///
/// # Safety
/// As `psp_copy`.
unsafe fn copy_uncached(target: *mut u8, source: *const u8, size: usize) {
    let aligned = (target as usize | source as usize) & 3 == 0;
    let bulk = if aligned { size & !127 } else { 0 };
    let mut to = target;
    let mut from = source;
    let bulk_end = source.add(bulk);
    while from < bulk_end {
        #[cfg(target_arch = "arm")]
        core::arch::asm!(
            ".fpu neon",
            "vldmia {from}!, {{d0-d15}}",
            "vstmia {to}!, {{d0-d15}}",
            from = inout(reg) from,
            to = inout(reg) to,
            out("d0") _, out("d1") _, out("d2") _, out("d3") _,
            out("d4") _, out("d5") _, out("d6") _, out("d7") _,
            out("d8") _, out("d9") _, out("d10") _, out("d11") _,
            out("d12") _, out("d13") _, out("d14") _, out("d15") _,
            options(nostack, preserves_flags),
        );
        #[cfg(not(target_arch = "arm"))]
        {
            core::ptr::copy_nonoverlapping(from, to, 128);
            from = from.add(128);
            to = to.add(128);
        }
    }
    let rest = size - bulk;
    if aligned && rest > 0 {
        copy_tail(to, from, rest);
    } else {
        core::ptr::copy_nonoverlapping(from, to, rest);
    }
}

/// Copies the last `rest` (< 128) bytes of a copy from word-aligned `from`.
/// Uncached PSP memory costs per load, not per byte (Vita, 2026-11-07, 64
/// bytes: word by word ~3.1 us, memcpy ~2.3 us, one VLDM ~0.45 us; 128
/// bytes as one VLDM ~0.85 us). So the tail is one 64- or 128-byte load
/// into a buffer, reading past its end. That read stays inside the 1 KiB
/// block holding the tail's last byte, so it touches only memory that
/// exists: PSP RAM, VRAM and the plugin's VRAM copy start and end on 1 KiB
/// boundaries. Where it would leave that block, exact loads of 64, 32, 16
/// and 8 bytes, then words and bytes.
///
/// # Safety
/// As `psp_copy`; `from` word-aligned.
unsafe fn copy_tail(to: *mut u8, from: *const u8, rest: usize) {
    let mut buffer = [0u64; 16];
    let staging = buffer.as_mut_ptr() as *mut u8;
    let start = from as usize;
    let chunk = if rest <= 64 { 64 } else { 128 };
    if (start + chunk - 1) / BLOCK_BYTES == (start + rest - 1) / BLOCK_BYTES {
        load_burst(staging, from, chunk);
        core::ptr::copy_nonoverlapping(staging, to, rest);
        return;
    }
    let mut done = 0;
    for piece in [64, 32, 16, 8] {
        if rest - done >= piece {
            load_burst(staging, from.add(done), piece);
            core::ptr::copy_nonoverlapping(staging, to.add(done), piece);
            done += piece;
        }
    }
    if rest - done >= 4 {
        let word = core::ptr::read_volatile(from.add(done) as *const u32);
        core::ptr::copy_nonoverlapping(&word as *const u32 as *const u8, to.add(done), 4);
        done += 4;
    }
    core::ptr::copy_nonoverlapping(from.add(done), to.add(done), rest - done);
}

/// The blocks `copy_tail` keeps its over-reads in.
const BLOCK_BYTES: usize = 1024;

/// One VLDM of `bytes` (128, 64, 32, 16 or 8) from word-aligned `from` into
/// `to` (8-byte aligned, cached).
///
/// # Safety
/// `from` and `to` must hold `bytes` bytes.
unsafe fn load_burst(to: *mut u8, from: *const u8, bytes: usize) {
    #[cfg(target_arch = "arm")]
    macro_rules! burst {
        ($registers:literal, $($clobber:tt)*) => {
            core::arch::asm!(
                ".fpu neon",
                concat!("vldmia {from}, {{", $registers, "}}"),
                concat!("vstmia {to}, {{", $registers, "}}"),
                from = in(reg) from,
                to = in(reg) to,
                $($clobber)*
                options(nostack, preserves_flags),
            )
        };
    }
    #[cfg(target_arch = "arm")]
    match bytes {
        128 => burst!("d0-d15",
            out("d0") _, out("d1") _, out("d2") _, out("d3") _, out("d4") _, out("d5") _, out("d6") _, out("d7") _,
            out("d8") _, out("d9") _, out("d10") _, out("d11") _, out("d12") _, out("d13") _, out("d14") _, out("d15") _,),
        64 => burst!("d0-d7",
            out("d0") _, out("d1") _, out("d2") _, out("d3") _, out("d4") _, out("d5") _, out("d6") _, out("d7") _,),
        32 => burst!("d0-d3", out("d0") _, out("d1") _, out("d2") _, out("d3") _,),
        16 => burst!("d0-d1", out("d0") _, out("d1") _,),
        _ => burst!("d0", out("d0") _,),
    }
    #[cfg(not(target_arch = "arm"))]
    core::ptr::copy_nonoverlapping(from, to, bytes);
}

/// Debug "copytest": `psp_copy` against a word-by-word copy,
/// from PSP RAM at `address` (block-aligned, 4 KiB readable), for every
/// size up to 300 bytes from word offsets 0..64 into a block and up to the
/// block's end (where `copy_tail` may not over-read). Returns how many
/// copies differed (0 = all right), or -1 if the memory isn't there.
pub fn copy_selftest(address: u32) -> i32 {
    let base = psp_memory(address, 4096);
    if base.is_null() {
        return -1;
    }
    let mut failures = 0;
    // Word-aligned, like the copies' targets.
    let mut got_words = [0u32; 80];
    // Safe: the 320 bytes of `got_words`, used only through this slice.
    let got = unsafe { core::slice::from_raw_parts_mut(got_words.as_mut_ptr() as *mut u8, 320) };
    let mut expected = [0u8; 320];
    for size in 0..=300usize {
        let mut offsets: [usize; 20] = [0; 20];
        for (index, offset) in offsets.iter_mut().enumerate().take(16) {
            *offset = 4 * index;
        }
        // Copies that end at, or just before, the end of block 1.
        let end = 2 * BLOCK_BYTES;
        for (slot, back) in [0usize, 4, 8, 60].iter().enumerate() {
            offsets[16 + slot] = (end - back).saturating_sub(size) & !3;
        }
        for &offset in &offsets {
            // Safe: offset + size stays within the 4 KiB looked up above.
            unsafe {
                let source = base.add(offset);
                for (index, byte) in expected[..size].iter_mut().enumerate() {
                    *byte = core::ptr::read_volatile(source.add(index));
                }
                got[..size + 16].fill(0xA5);
                psp_copy(got.as_mut_ptr(), source, size as u32);
            }
            let tail_untouched = got[size..size + 16].iter().all(|&byte| byte == 0xA5);
            if got[..size] != expected[..size] || !tail_untouched {
                failures += 1;
            }
        }
    }
    failures
}
