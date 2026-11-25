//! The frame pool: GPU-readable memory for what a frame's draws build
//! (indices, converted vertices). The ring's bookkeeping and the reason for
//! it are in `nativeres_core::frame_pool`; this side owns the memory block,
//! reads the GPU's notification word and waits when the ring is full.
//!
//! Display thread only.

use core::ffi::c_void;
use core::fmt;
use core::ptr::{read_volatile, write_volatile};
use core::sync::atomic::{AtomicU32, Ordering};

use nativeres_core::frame::{MAX_INDICES, MAX_VERTICES};
use nativeres_core::frame_pool::{segment_done, Allocation, FramePool, FrameFence};

use crate::{gxm, render_log, sys, SingleThreaded};

/// A frame's vertices as our programs read them (32 bytes each), its
/// indices, and room for what else a pass builds.
const FRAME_POOL_BYTES: usize = MAX_VERTICES * 32 + MAX_INDICES * 2 + 128 * 1024;
#[no_mangle]
#[used]
#[allow(non_upper_case_globals)]
pub static nativeres_plan_fixed_frame_pool: u32 = sys::pages(FRAME_POOL_BYTES);

/// GXM's notification region has 512 words for the whole process. vita2d
/// uses none; the PSP emulator's code shares the process, so a high slot,
/// away from the first ones.
const NOTIFICATION_SLOT: usize = 496;
/// Waiting longer for the GPU than this: give up on the allocation.
const WAIT_US: u64 = 200_000;
/// Waiting longer than this for an old frame (`fence_frame`): the GPU is
/// stuck, drain it. God of War's gameplay frames take up to ~1 s of GPU.
const FENCE_WAIT_US: u64 = 3_000_000;
/// How often each kind of trouble is logged (they would repeat every frame).
const MAX_LOGGED: u32 = 5;

struct State {
    /// The ring's memory, mapped for the GPU; null until `ready` made it.
    memory: *mut u8,
    /// The block couldn't be made: don't try again.
    failed: bool,
    /// The notification word: the newest segment the GPU has finished.
    done: *mut u32,
    pool: FramePool,
    /// What `close_segment` hands to sceGxmEndScene.
    notification: gxm::Notification,
    logged_slow_gpu: u32,
    logged_too_big: u32,
    /// Which segment each recent frame closed (`fence_frame`).
    fence: FrameFence,
    logged_fence: u32,
}

/// Times the display thread had to wait for the GPU (render statistics
/// pool_waits; read and cleared by `take_waits`).
static WAITS: AtomicU32 = AtomicU32::new(0);

static STATE: SingleThreaded<State> = SingleThreaded::new(State {
    memory: core::ptr::null_mut(),
    failed: false,
    done: core::ptr::null_mut(),
    pool: FramePool::new(FRAME_POOL_BYTES as u32),
    notification: gxm::Notification { address: core::ptr::null_mut(), value: 0 },
    logged_slow_gpu: 0,
    logged_too_big: 0,
    fence: FrameFence::new(),
    logged_fence: 0,
});

fn log(text: fmt::Arguments) {
    render_log::line(text);
}

impl State {
    /// The newest segment number the GPU has written.
    fn done(&self) -> u32 {
        // Safe: `done` points into GXM's notification region once `memory`
        // is set; the GPU writes it, so the read is volatile.
        unsafe { read_volatile(self.done) }
    }

    fn set_done(&mut self, value: u32) {
        // Safe: as `done`.
        unsafe { write_volatile(self.done, value) }
    }

    /// Makes the memory block on first use. False if there is none.
    fn ready(&mut self) -> bool {
        if !self.memory.is_null() {
            return true;
        }
        if self.failed {
            return false;
        }
        // Safe: plain system calls; the block is ours alone.
        unsafe {
            let block = sys::alloc_block(
                b"NativeResFramePool\0".as_ptr(),
                sys::SCE_KERNEL_MEMBLOCK_TYPE_USER_RW,
                FRAME_POOL_BYTES as u32,
                core::ptr::null_mut(),
            );
            let mut base: *mut c_void = core::ptr::null_mut();
            if block < 0
                || sys::sceKernelGetMemBlockBase(block, &mut base) < 0
                || gxm::sceGxmMapMemory(base, FRAME_POOL_BYTES as u32, gxm::MEMORY_ATTRIB_READ) < 0
            {
                if block >= 0 {
                    sys::free_block(block);
                }
                log(format_args!("frame pool: no memory ({:08x})\n", block));
                self.failed = true;
                return false;
            }
            self.done = gxm::sceGxmGetNotificationRegion().add(NOTIFICATION_SLOT);
            self.set_done(0);
            self.memory = base as *mut u8;
        }
        log(format_args!("frame pool ready: {} KiB\n", FRAME_POOL_BYTES / 1024));
        true
    }

    /// Waits until the GPU has finished segment `id`. False if it didn't
    /// within `WAIT_US`.
    fn wait_for(&mut self, id: u32) -> bool {
        let _hook = nativeres_core::debug_hooks::span(nativeres_core::debug_hooks::Point::PoolWait);
        let start = sys::now_us();
        WAITS.fetch_add(1, Ordering::Relaxed);
        while !segment_done(self.done(), id) {
            if sys::now_us() - start > WAIT_US {
                if self.logged_slow_gpu < MAX_LOGGED {
                    self.logged_slow_gpu += 1;
                    log(format_args!("frame pool: the GPU didn't finish segment {} (done {})\n", id, self.done()));
                }
                return false;
            }
            // Safe: a plain sleep.
            unsafe { sys::sceKernelDelayThread(200) };
        }
        true
    }

    /// The GPU stopped telling us (or a scene failed and never will): wait
    /// until it has finished everything, then count all closed segments as
    /// done.
    fn finish_everything(&mut self) {
        // Safe: display thread, the context is vita2d's.
        unsafe { gxm::sceGxmFinish(gxm::context()) };
        let all_closed = self.pool.open_segment().wrapping_sub(1);
        self.set_done(all_closed);
    }
}

/// `size` bytes of GPU-readable memory aligned to `align` (a power of two)
/// for the current frame, or null (no pool, the frame needs more than the
/// ring holds, or the GPU didn't finish in time).
pub fn alloc(size: u32, align: u32) -> *mut c_void {
    // Safe: display thread only (see the module's head).
    let state = unsafe { STATE.get() };
    if !state.ready() {
        return core::ptr::null_mut();
    }
    loop {
        match state.pool.allocate(size, align, state.done()) {
            // Safe: the pool only hands out offsets inside the block.
            Allocation::At(offset) => return unsafe { state.memory.add(offset as usize) } as *mut c_void,
            Allocation::WaitFor(segment) => {
                if !state.wait_for(segment) {
                    return core::ptr::null_mut();
                }
            }
            Allocation::TooBig => {
                if state.logged_too_big < MAX_LOGGED {
                    state.logged_too_big += 1;
                    log(format_args!("frame pool: {} bytes don't fit\n", size));
                }
                return core::ptr::null_mut();
            }
        }
    }
}

/// How many frames back `fence_frame` waits for (debug `fenceframes`;
/// at most `RETIRE_FRAMES`, for textures and palette tables): 2, so at most
/// the previous frame is still on the GPU when the next is taken. Its
/// vertex work then starts right after we submit it (the GPU's vertex
/// phase overlaps the previous scene's fragments). God of War's gameplay
/// (2026-11-24): 1 and 2 both clean, 2 ~345 ms per frame vs ~390 ms
/// (overlap kept). Waiting the same way after taking the frame showed
/// stale-vertex triangles at any depth.
pub static FENCE_FRAMES: AtomicU32 = AtomicU32::new(2);

/// Before display frame `now` (`render_count`) takes the newest recorded
/// frame: waits until the GPU has finished frame `now - FENCE_FRAMES`
/// (`FrameFence`). So the frame's zero-copy vertices are read soon after
/// it was recorded, and memory the texture cache and palette tables retired
/// `RETIRE_FRAMES` frames ago is no longer read. Nothing to wait for while
/// the GPU keeps up.
pub fn fence_frame(now: u32) {
    // Safe: display thread only.
    let state = unsafe { STATE.get() };
    if state.memory.is_null() {
        return;
    }
    let back = FENCE_FRAMES.load(Ordering::Relaxed).min(nativeres_core::texture_cache::RETIRE_FRAMES);
    let Some(segment) = state.fence.must_finish_back(now, back) else { return };
    if segment_done(state.done(), segment) {
        return;
    }
    let start = sys::now_us();
    while !segment_done(state.done(), segment) {
        if sys::now_us() - start > FENCE_WAIT_US {
            state.finish_everything();
            break;
        }
        // Safe: a plain sleep.
        unsafe { sys::sceKernelDelayThread(200) };
    }
    if state.logged_fence < MAX_LOGGED {
        state.logged_fence += 1;
        log(format_args!("frame {}: waited {} us for the GPU to finish frame segment {}
", now, sys::now_us() - start, segment));
    }
}

/// At the end of display frame `frame`'s last scene: closes the segment of
/// everything allocated since the last call and returns the notification
/// for sceGxmEndScene's fragment notification (null without a frame pool).
pub fn close_segment(frame: u32) -> *const gxm::Notification {
    // Safe: display thread only.
    let state = unsafe { STATE.get() };
    if state.memory.is_null() {
        return core::ptr::null();
    }
    if state.pool.segments_full() {
        let oldest = state.pool.oldest_segment().unwrap_or(0);
        if !state.wait_for(oldest) {
            state.finish_everything();
        }
        state.pool.release(state.done());
    }
    let id = state.pool.close_segment();
    state.fence.closed(frame, id);
    state.notification = gxm::Notification { address: state.done, value: id };
    &state.notification
}

/// sceGxmEndScene failed: its notification will never come.
pub fn scene_failed() {
    // Safe: display thread only.
    let state = unsafe { STATE.get() };
    if !state.memory.is_null() {
        state.finish_everything();
    }
}

/// Times the display thread waited for the GPU since the last call
/// (render statistics pool_waits).
pub fn take_waits() -> u32 {
    WAITS.swap(0, Ordering::Relaxed)
}
