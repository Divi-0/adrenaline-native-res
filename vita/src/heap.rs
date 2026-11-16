//! Our heap (the bookkeeping is `nativeres_core::heap`, tested there): one
//! block of main memory, made on first use, for the texture cache's texture
//! objects and the shader patcher's host memory (shader_patcher.rs). It
//! replaces the C library heap (newlib's calloc/free) we used while linked
//! into Adrenaline: a plugin of our own has none.
//!
//! Any thread (a spin lock; allocations are rare).

use core::ffi::c_void;

use nativeres_core::heap::Heap;

use crate::blocks::SpinLocked;
use crate::{render_log, sys};

/// Texture objects (100 bytes + 8 header; MTX's race start has ~400
/// textures: ~43 KiB) and the shader patcher's host memory (a few hundred
/// bytes per patched program). Main memory is short: `report`'s peak
/// shows what is really used.
const HEAP_BYTES: usize = 128 * 1024;
#[no_mangle]
#[used]
#[allow(non_upper_case_globals)]
pub static nativeres_plan_fixed_heap: u32 = sys::pages(HEAP_BYTES);

struct State {
    /// The block's words; null until made.
    memory: *mut u32,
    /// The block couldn't be made: don't try again.
    failed: bool,
    heap: Heap,
}

static STATE: SpinLocked<State> = SpinLocked::new(State { memory: core::ptr::null_mut(), failed: false, heap: Heap::EMPTY });

/// The block's words.
///
/// # Safety
/// `memory` must be the block (not null), used only under the lock.
unsafe fn words<'a>(memory: *mut u32) -> &'a mut [u32] {
    core::slice::from_raw_parts_mut(memory, HEAP_BYTES / 4)
}

impl State {
    /// Makes the block on first use. False if there is none.
    fn ready(&mut self) -> bool {
        if !self.memory.is_null() {
            return true;
        }
        if self.failed {
            return false;
        }
        // Safe: plain system calls; the block is ours alone.
        unsafe {
            let block = sys::alloc_block(b"NativeResHeap\0".as_ptr(), sys::SCE_KERNEL_MEMBLOCK_TYPE_USER_RW, HEAP_BYTES as u32, core::ptr::null_mut());
            let mut base: *mut c_void = core::ptr::null_mut();
            if block < 0 || sys::sceKernelGetMemBlockBase(block, &mut base) < 0 {
                self.failed = true;
                render_log::line(format_args!("heap: no memory ({:08x})\n", block));
                return false;
            }
            self.memory = base as *mut u32;
        }
        // Safe: just made, under the lock.
        self.heap = Heap::new(unsafe { words(self.memory) });
        true
    }
}

/// `size` bytes, 8-byte aligned, not cleared; null if there is no room.
pub fn alloc(size: usize) -> *mut c_void {
    STATE.with(|state| {
        if size > u32::MAX as usize || !state.ready() {
            return core::ptr::null_mut();
        }
        // Safe: the block exists; we hold the lock.
        let offset = state.heap.allocate(unsafe { words(state.memory) }, size as u32);
        // Safe: the heap hands out offsets inside the block.
        offset.map_or(core::ptr::null_mut(), |offset| unsafe { (state.memory as *mut u8).add(offset as usize) } as *mut c_void)
    })
}

/// `size` bytes cleared to zero (calloc), or null.
pub fn alloc_zeroed(size: usize) -> *mut c_void {
    let memory = alloc(size);
    if !memory.is_null() {
        // Safe: `size` bytes of ours.
        unsafe { core::ptr::write_bytes(memory as *mut u8, 0, size) };
    }
    memory
}

/// Gives back what `alloc` returned (null is ignored).
pub fn free(memory: *mut c_void) {
    if memory.is_null() {
        return;
    }
    STATE.with(|state| {
        if state.memory.is_null() {
            return;
        }
        let offset = (memory as usize).wrapping_sub(state.memory as usize);
        if offset >= HEAP_BYTES {
            render_log::line(format_args!("heap: free of {:08x}, not ours\n", memory as usize));
            return;
        }
        // Safe: the block exists; we hold the lock.
        if !state.heap.free(unsafe { words(state.memory) }, offset as u32) {
            render_log::line(format_args!("heap: free of {:08x}, not in use\n", memory as usize));
        }
    })
}

/// "heap <used>/<bytes> peak=<bytes> allocations=<n> failed=<n> bad_frees=<n>"
/// (render statistics).
pub fn report(text: &mut impl core::fmt::Write) {
    STATE.with(|state| {
        let heap = &state.heap;
        let _ = write!(
            text,
            "heap={}/{} heap_peak={} heap_allocations={} heap_failed={} heap_bad_frees={}",
            heap.used(),
            heap.bytes(),
            heap.peak(),
            heap.allocations(),
            heap.failed(),
            heap.bad_frees()
        );
    })
}
