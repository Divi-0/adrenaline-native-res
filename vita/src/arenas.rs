//! The GPU memory the capture records into, made by the display thread once
//! its GPU programs exist, and handed over through capture.rs's pointers:
//!
//! - the raw arena (`nativepsp_core::raw_arena`): raw PSP vertices the GPU
//!   reads as they are (draws with `RAW_VERTICES`, nativeres_raw_v.cg),
//!   written once per frame by the recorder and kept for 6 recorded frames.
//!   MTX's race start copies ~450 KiB of raw runs per frame.
//! - the index ring: the capture writes draws' index runs there as triangle
//!   lists (`Frame::ring_indices`), so the draw loop passes a pointer
//!   instead of building them (~1 ms of display CPU per race frame,
//!   measured 2026-11-10). MTX's race start: ~97,000 corners (~195 KiB)
//!   per frame, kept for 6 frames.
//! - the vertex cache's arena (`nativepsp_core::vertex_cache`): decoded
//!   vertices kept across frames, drawn by the GPU straight from there.
//! - PSP RAM itself, mapped for the GPU to read (zero copy, `RAW_PSP_RAM`).
//!
//! All cached and GPU-mapped (the CPU writes, the GPU reads right after).
//! Nothing here is ever freed: frames point into it.

use core::ffi::c_void;
use core::sync::atomic::{AtomicPtr, AtomicU32, Ordering};

use nativepsp_core::frame::FrameVertex;

use crate::capture::{
    INDEX_RING, INDEX_RING_BYTES, RAW_ARENA, RAW_ARENA_BYTES, VERTEX_ARENA,
    VERTEX_ARENA_VERTICES,
};
use crate::{gxm, render_log, sys};

/// Phycont memory (physically contiguous main memory, a pool of its own).
const RAW_ARENA_SIZE: usize = 6 * 1024 * 1024;
#[no_mangle]
#[used]
#[allow(non_upper_case_globals)]
pub static nativepsp_plan_phycont_raw_arena: u32 = sys::pages(RAW_ARENA_SIZE);

/// 8 windows of 192 KiB. In phycont memory (blocks of whole MiB: 2 MiB)
/// since the plugin runs on stock Adrenaline, whose vita2d pool takes
/// 0.5 MiB more main memory than the fork's did (user, 2026-11-14: main
/// memory was 0.46 MiB short of its safety margin); main memory
/// only if phycont is short.
const INDEX_RING_SIZE: usize = 8 * 196_608;
const INDEX_RING_PHYCONT_BYTES: usize = 2 * 1024 * 1024;
const _: () = assert!(INDEX_RING_SIZE <= INDEX_RING_PHYCONT_BYTES);
#[no_mangle]
#[used]
#[allow(non_upper_case_globals)]
pub static nativepsp_plan_phycont_index_ring: u32 = sys::pages(INDEX_RING_PHYCONT_BYTES);

/// 32768 vertices of main memory (short). Not CDRAM:
/// with 2 MiB of CDRAM, Adrenaline crashed (C2-12828-1) twice, and both
/// times the next start hung at a black screen until the Vita was rebooted
/// (2026-11-05).
const VERTEX_ARENA_BYTES: usize = 1024 * 1024;
#[no_mangle]
#[used]
#[allow(non_upper_case_globals)]
pub static nativepsp_plan_optin_vertex_arena: u32 = sys::pages(VERTEX_ARENA_BYTES);
/// The default ("vertexcache 2"): phycont, ~23 MiB free in MTX; blocks come
/// in whole MiB. 4 MiB = two windows of 65536 vertices (2026-11-08; one
/// window was full at MTX's race start): a cached draw's 16-bit indices
/// count from its window's first vertex, which the capture puts in
/// `Draw::first_vertex`.
const VERTEX_ARENA_PHYCONT_BYTES: usize = 4 * 1024 * 1024;
#[no_mangle]
#[used]
#[allow(non_upper_case_globals)]
pub static nativepsp_plan_phycont_vertex_arena: u32 = sys::pages(VERTEX_ARENA_PHYCONT_BYTES);

/// PSP RAM as the GPU sees it (the emulator's 32 MiB block, PSP 0x08000000
/// up), null until mapped; read by the draw loop for zero-copy draws.
pub static PSP_RAM_FOR_GPU: AtomicPtr<u8> = AtomicPtr::new(core::ptr::null_mut());
pub static PSP_RAM_FOR_GPU_BYTES: AtomicU32 = AtomicU32::new(0);
const PSP_RAM_GPU_BASE: u32 = 0x0800_0000;

/// A block of `block_type` mapped for the GPU to read, or the error to log.
fn mapped_block(name: &'static [u8], block_type: u32, bytes: usize) -> Result<*mut u8, &'static str> {
    // Safe: plain system calls; the block is ours alone.
    unsafe {
        let block = sys::alloc_block(name.as_ptr(), block_type, bytes as u32, core::ptr::null_mut());
        if block < 0 {
            return Err("no memory");
        }
        let mut base: *mut c_void = core::ptr::null_mut();
        sys::sceKernelGetMemBlockBase(block, &mut base);
        if gxm::sceGxmMapMemory(base, bytes as u32, gxm::MEMORY_ATTRIB_READ) < 0 {
            sys::free_block(block);
            return Err("not mapped for the GPU");
        }
        Ok(base as *mut u8)
    }
}

/// Makes the raw arena and the index ring (once, before the vertex arena,
/// so the capture finds them when it makes the vertex cache). 1 if the raw
/// arena exists now. The caller checked that the raw vertex program exists.
pub fn make_raw_arena() -> i32 {
    match mapped_block(b"NativePspRawArena\0", sys::SCE_KERNEL_MEMBLOCK_TYPE_USER_MAIN_PHYCONT_RW, RAW_ARENA_SIZE) {
        Ok(base) => {
            // The size first: the capture thread reads the pointer first.
            RAW_ARENA_BYTES.store(RAW_ARENA_SIZE as u32, Ordering::Release);
            RAW_ARENA.store(base, Ordering::Release);
            render_log::line(format_args!("raw arena ready: {} KiB phycont memory\n", RAW_ARENA_SIZE / 1024));
        }
        Err(why) => {
            render_log::line(format_args!("raw arena: {}\n", why));
            return 0;
        }
    }
    // Phycont first, main memory if there is none (see INDEX_RING_SIZE).
    let ring = match mapped_block(b"NativePspIndexRing\0", sys::SCE_KERNEL_MEMBLOCK_TYPE_USER_MAIN_PHYCONT_RW, INDEX_RING_PHYCONT_BYTES) {
        Ok(ring) => Ok((ring, "phycont")),
        Err(_) => mapped_block(b"NativePspIndexRing\0", sys::SCE_KERNEL_MEMBLOCK_TYPE_USER_RW, INDEX_RING_SIZE).map(|ring| (ring, "main")),
    };
    match ring {
        Ok((ring, kind)) => {
            INDEX_RING_BYTES.store(INDEX_RING_SIZE as u32, Ordering::Release);
            INDEX_RING.store(ring, Ordering::Release);
            render_log::line(format_args!("index ring ready: {} KiB {} memory\n", INDEX_RING_SIZE / 1024, kind));
        }
        Err(why) => render_log::line(format_args!("index ring: {}\n", why)),
    }
    1
}

/// Makes the vertex cache's arena (phycont or, with 0, main memory); without
/// it the capture runs without a vertex cache.
pub fn make_vertex_arena(phycont: i32) {
    let (block_type, bytes, kind) = if phycont != 0 {
        (sys::SCE_KERNEL_MEMBLOCK_TYPE_USER_MAIN_PHYCONT_RW, VERTEX_ARENA_PHYCONT_BYTES, "phycont")
    } else {
        (sys::SCE_KERNEL_MEMBLOCK_TYPE_USER_RW, VERTEX_ARENA_BYTES, "main")
    };
    match mapped_block(b"NativePspVertexArena\0", block_type, bytes) {
        Ok(base) => {
            VERTEX_ARENA_VERTICES
                .store((bytes / core::mem::size_of::<FrameVertex>()) as u32, Ordering::Release);
            VERTEX_ARENA.store(base as *mut FrameVertex, Ordering::Release);
            render_log::line(format_args!("vertex arena ready: {} KiB {} memory\n", bytes / 1024, kind));
        }
        Err(why) => render_log::line(format_args!("vertex arena: {}\n", why)),
    }
}

/// Zero copy (raw_arena.rs ZERO_COPY): maps PSP RAM for the GPU to read
/// (once, and kept; checked 2026-11-09 with "gxmmaptest": GXM maps it). 1 if
/// mapped.
pub fn map_psp_ram_for_gpu() -> i32 {
    if !PSP_RAM_FOR_GPU.load(Ordering::Acquire).is_null() {
        return 1;
    }
    let expected = sys::psp_to_vita(PSP_RAM_GPU_BASE) as *mut c_void;
    let mut info = sys::SceKernelMemBlockInfo {
        size: core::mem::size_of::<sys::SceKernelMemBlockInfo>() as u32,
        mapped_base: core::ptr::null_mut(),
        mapped_size: 0,
        memory_type: 0,
        access: 0,
        block_type: 0,
    };
    // Safe: plain system calls with a sized struct; the block is the
    // emulator's, which lives as long as we do.
    unsafe {
        if sys::sceKernelGetMemBlockInfoByAddr(expected, &mut info) < 0 || info.mapped_base != expected {
            render_log::line(format_args!("zero copy: PSP RAM block not where expected\n"));
            return 0;
        }
        if gxm::sceGxmMapMemory(info.mapped_base, info.mapped_size, gxm::MEMORY_ATTRIB_READ) < 0 {
            render_log::line(format_args!("zero copy: PSP RAM not mapped for the GPU\n"));
            return 0;
        }
    }
    PSP_RAM_FOR_GPU_BYTES.store(info.mapped_size, Ordering::Release);
    PSP_RAM_FOR_GPU.store(info.mapped_base as *mut u8, Ordering::Release);
    render_log::line(format_args!("zero copy: PSP RAM mapped for the GPU ({} KiB)\n", info.mapped_size / 1024));
    1
}
