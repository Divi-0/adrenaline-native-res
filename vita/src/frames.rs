//! The three recorded frames shared by the capture thread (fills them) and
//! the display thread (frame_loop.rs draws them), handed over with
//! `nativepsp_core::triple_buffer`: no locks, the newest frame wins.

use core::ffi::c_void;
use core::sync::atomic::{AtomicBool, AtomicI32, AtomicPtr, AtomicU32, Ordering};

use nativepsp_core::frame::Frame;
use nativepsp_core::triple_buffer::{Consumer, Producer, TripleBuffer, SLOTS};

use crate::sys;
use crate::SingleThreaded;

static BUFFER: TripleBuffer = TripleBuffer::new();
/// The frames (one memory block); null until allocated.
static FRAMES: AtomicPtr<Frame> = AtomicPtr::new(core::ptr::null_mut());

/// The frames' memory (`allocate`), for the memory plan.
#[no_mangle]
#[used]
#[allow(non_upper_case_globals)]
pub static nativepsp_plan_fixed_frames: u32 = sys::pages(core::mem::size_of::<Frame>() * SLOTS);
/// Semaphore (max 1), signalled for every published frame.
static PUBLISHED: AtomicI32 = AtomicI32::new(-1);
static ALLOCATED: AtomicBool = AtomicBool::new(false);

static PRODUCER: SingleThreaded<Producer> = SingleThreaded::new(TripleBuffer::sides().0);
static CONSUMER: SingleThreaded<Consumer> = SingleThreaded::new(TripleBuffer::sides().1);

/// When the shown frame / the newest frame was published (process time, low 32 bits).
pub static SHOWN_PUBLISHED_US: AtomicU32 = AtomicU32::new(0);
pub static LAST_PUBLISHED_US: AtomicU32 = AtomicU32::new(0);

fn frame(slot: usize) -> *mut Frame {
    // Safe: only called once FRAMES points to SLOTS frames.
    unsafe { FRAMES.load(Ordering::Acquire).add(slot) }
}

/// Capture thread, once: allocates the frames (~6.6 MiB). False if memory
/// is short; the capture then runs without recording.
pub fn allocate() -> bool {
    let size = (core::mem::size_of::<Frame>() * SLOTS + 0xFFF) & !0xFFF;
    // Safe: plain system calls; the block is ours alone until FRAMES is set.
    unsafe {
        let block = sys::alloc_block(
            b"NativePspFrames\0".as_ptr(),
            sys::SCE_KERNEL_MEMBLOCK_TYPE_USER_RW,
            size as u32,
            core::ptr::null_mut(),
        );
        if block < 0 {
            return false;
        }
        let mut base: *mut c_void = core::ptr::null_mut();
        sys::sceKernelGetMemBlockBase(block, &mut base);
        let frames = base as *mut Frame;
        for slot in 0..SLOTS {
            (*frames.add(slot)).clear();
        }
        PUBLISHED.store(
            sys::sceKernelCreateSema(b"NativePspFramePublished\0".as_ptr(), 0, 0, 1, core::ptr::null_mut()),
            Ordering::Release,
        );
        FRAMES.store(frames, Ordering::Release);
    }
    ALLOCATED.store(true, Ordering::Release);
    true
}

/// Capture thread: the frame being filled.
///
/// # Safety
/// Capture thread only, after `allocate` succeeded; the reference must not
/// outlive the next `publish`.
pub unsafe fn building() -> &'static mut Frame {
    &mut *frame(PRODUCER.get().building())
}

/// Capture thread: the building frame is complete.
pub fn publish() {
    let now = sys::now_us() as u32;
    LAST_PUBLISHED_US.store(now, Ordering::Relaxed);
    // Safe: capture thread only.
    let producer = unsafe { PRODUCER.get() };
    let next = BUFFER.publish(producer, now);
    // Fails harmlessly if already signalled.
    unsafe {
        sys::sceKernelSignalSema(PUBLISHED.load(Ordering::Acquire), 1);
        (*frame(next)).clear();
    }
}

/// Display thread: waits until a frame it hasn't taken yet is there, at
/// most `timeout_us`. Returns 1 if there is one.
pub fn wait_for_frame(timeout_us: u32) -> i32 {
    if !ALLOCATED.load(Ordering::Acquire) {
        return 0;
    }
    let deadline = sys::now_us() + timeout_us as u64;
    while !BUFFER.has_new() {
        let now = sys::now_us();
        if now >= deadline {
            return 0;
        }
        let mut left = (deadline - now) as u32;
        // A stale signal just loops once more.
        unsafe { sys::sceKernelWaitSema(PUBLISHED.load(Ordering::Acquire), 1, &mut left) };
    }
    1
}

/// Display thread: the newest complete frame, or null if there is none
/// yet. Stays valid until the next call. `*is_new`: 1 if it wasn't
/// returned before.
///
/// # Safety
/// Display thread only; `is_new` must be writable.
pub unsafe fn acquire_frame(is_new: *mut i32) -> *const Frame {
    *is_new = 0;
    if !ALLOCATED.load(Ordering::Acquire) {
        return core::ptr::null();
    }
    match BUFFER.take(CONSUMER.get()) {
        Some((slot, new)) => {
            if new {
                *is_new = 1;
                SHOWN_PUBLISHED_US.store(BUFFER.published_us(slot), Ordering::Relaxed);
            }
            frame(slot)
        }
        None => core::ptr::null(),
    }
}
