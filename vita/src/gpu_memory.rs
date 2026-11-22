//! Memory blocks the GPU reads or writes, made through our block table
//! (blocks.rs) and mapped for the GPU. Two kinds:
//!
//! - `for_gpu`: memory the GPU writes and the CPU never touches (render
//!   targets, the depth buffer) or the CPU only writes small bits of (palette
//!   tables). Uncached: in cached memory, the CPU's cache lines from
//!   zero-filling the block could later be written back over what the GPU
//!   drew.
//! - `gpu_input`: memory the CPU fills once and the GPU then reads (index
//!   tables, quads). Cached: allocating uncached memory took ~25 ms per MiB.

use core::ffi::c_void;

use crate::{gxm, sys};

/// A mapped block: its id (for `free_block`) and its memory.
pub struct GpuBlock {
    #[allow(dead_code)] // nothing frees these blocks yet
    pub uid: sys::SceUID,
    pub base: *mut u8,
}

/// `bytes` of `block_type` memory under `name` (a NUL-terminated literal),
/// mapped for the GPU with `attributes`; None if there is no memory or it
/// can't be mapped.
fn allocate(name: &'static [u8], block_type: u32, bytes: u32, attributes: u32) -> Option<GpuBlock> {
    debug_assert!(name.last() == Some(&0));
    // Safe: plain system calls; the block is ours alone.
    unsafe {
        let uid = sys::alloc_block(name.as_ptr(), block_type, bytes, core::ptr::null_mut());
        if uid < 0 {
            return None;
        }
        let mut base: *mut c_void = core::ptr::null_mut();
        sys::sceKernelGetMemBlockBase(uid, &mut base);
        if gxm::sceGxmMapMemory(base, bytes, attributes) < 0 {
            sys::free_block(uid);
            return None;
        }
        Some(GpuBlock { uid, base: base as *mut u8 })
    }
}

/// Uncached main memory the GPU reads and writes (see the module's head).
pub fn for_gpu(name: &'static [u8], bytes: u32) -> Option<GpuBlock> {
    allocate(
        name,
        sys::SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE,
        bytes,
        gxm::MEMORY_ATTRIB_READ | gxm::MEMORY_ATTRIB_WRITE,
    )
}

/// CDRAM (the GPU's own memory) the GPU reads and writes: the off-screen
/// targets (targets.rs), which main memory has no room for. Whole 256 KiB
/// units (the kernel's rule for CDRAM blocks). None if that would leave
/// less than `CDRAM_RESERVE` free.
pub fn for_gpu_cdram(name: &'static [u8], bytes: u32) -> Option<GpuBlock> {
    let bytes = bytes.next_multiple_of(CDRAM_UNIT);
    if crate::blocks::free_cdram() < bytes + CDRAM_RESERVE {
        return None;
    }
    allocate(name, sys::SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RW, bytes, gxm::MEMORY_ATTRIB_READ | gxm::MEMORY_ATTRIB_WRITE)
}

/// CDRAM blocks come in whole 256 KiB.
pub const CDRAM_UNIT: u32 = 256 * 1024;

/// CDRAM left free for Sony's emulator: it takes CDRAM when a PSP game
/// starts a movie, and without it the game hangs for good (black screen, no
/// error). God of War's intro movie after "New Game" (2026-11-22): runs
/// with 2 MiB of CDRAM free, hangs with 1 MiB (our five 768 KiB off-screen
/// targets of its title had taken 3 of the 4 MiB free).
pub const CDRAM_RESERVE: u32 = 2 * 1024 * 1024;

/// Cached main memory the CPU fills and the GPU reads (see the module's head).
#[allow(dead_code)] // for the unswizzle quads and sprite corners, still C
pub fn gpu_input(name: &'static [u8], bytes: u32) -> Option<GpuBlock> {
    allocate(name, sys::SCE_KERNEL_MEMBLOCK_TYPE_USER_RW, bytes, gxm::MEMORY_ATTRIB_READ)
}
