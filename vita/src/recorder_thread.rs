//! The thread that records the vertices of the capture's draws (the
//! consumer half of `nativepsp_core::draw_stream`), on another core than
//! the capture thread: the capture follows the display lists and works out
//! the draws' states, this thread decodes, transforms and stores their
//! vertices, both for the same frame at the same time.
//!
//! It serves one draw stream at a time: the live capture's, or another
//! capture's (`serve`). Without the thread
//! (it couldn't be created) the capture records the vertices itself.

use core::ffi::c_void;
use core::sync::atomic::{AtomicBool, AtomicI32, AtomicPtr, AtomicU32, Ordering};

use nativepsp_core::draw_stream::{DrawStream, StreamHooks};

use crate::sys;

/// Just below the display thread (AdrenalineDraw, 0xA0): it has slack
/// (busy ~21-25 of 33 ms), and the display's render must not wait for it
/// at publish time. 0x70 (the capture's) made ~26% of frames miss the
/// refresh after next (3/1 wobble); 0xA1 halves the webcam's 3/1 pairs at
/// no game-side cost.
const PRIORITY: i32 = 0xA1;
const STACK_SIZE: u32 = 0x10000;
/// Cores 0, 1 and 2 (SCE_KERNEL_CPU_MASK_USER_0 | _1 | _2). Was 1 and 2
/// (leaving core 0 to the capture thread); measured 2026-11-05 on MTX's
/// paused race (the same frame every time): cores 1+2 frames 60-68 ms,
/// recorder busy 55-66; cores 0-2 57-64 / 45-56; core 1 alone 70-74.
/// The capture thread waits for this one 10-19 ms per frame anyway, so
/// core 0 has time for it. Idle in the race: core 0 17%, 1 34%, 2 7%.
const CORES: i32 = 0x0001_0000 | 0x0002_0000 | 0x0004_0000;
/// How long the thread sleeps without being woken (it checks then anyway).
const IDLE_TIMEOUT_US: u32 = 2000;
/// How long the producer waits for room or for the end of the frame.
const PRODUCER_WAIT_TRY_US: u32 = 50;
/// How long the thread waits when every copied job is recorded but the
/// copier is still copying the next ones.
const COPIER_WAIT_US: u32 = 20;

static SEMA: AtomicI32 = AtomicI32::new(-1);
static THREAD: AtomicI32 = AtomicI32::new(-1);

/// The thread's CPU time so far (its run clock, not wall time: busy
/// times include time it was preempted).
pub fn cpu_us() -> u64 {
    let thread = THREAD.load(Ordering::Relaxed);
    if thread < 0 { 0 } else { crate::threads::run_us_of(thread) }
}
static ACTIVE: AtomicPtr<DrawStream> = AtomicPtr::new(core::ptr::null_mut());
/// The stream the thread is recording right now (null between).
static IN_USE: AtomicPtr<DrawStream> = AtomicPtr::new(core::ptr::null_mut());

/// Microseconds the producer spent waiting (for room in the stream or for
/// the consumer to finish) and the consumer spent recording, since the last
/// `take_times`: which half of the capture holds the other up.
static PRODUCER_WAIT_US: AtomicU32 = AtomicU32::new(0);
static CONSUMER_BUSY_US: AtomicU32 = AtomicU32::new(0);

/// The producer's waiting so far (since the last `take_times`), to tell
/// waits within a walk from those at the frame's end.
pub fn producer_waited_us() -> u32 {
    PRODUCER_WAIT_US.load(Ordering::Relaxed)
}

/// (producer waiting, consumer busy) in microseconds since the last call.
pub fn take_times() -> (u32, u32) {
    (PRODUCER_WAIT_US.swap(0, Ordering::Relaxed), CONSUMER_BUSY_US.swap(0, Ordering::Relaxed))
}

/// Whether the capture records the vertices on this thread, and on which cores. 0 = off,
/// 1 = the default cores (`CORES`), otherwise a mask of user cores
/// (2 = core 1, 4 = core 2, 6 = cores 1+2, 7 = cores 0-2, 8 = core 3,
/// 15 = all four; core 0 alone can't be chosen: 1 means the default).
/// Core 3 only where `threads::set_cores` can; refused, the thread goes
/// back to the default. On by default since 2026-11-05 (it doubles MTX's
/// shown frame rate).
pub static RECORDER_THREAD_WANTED: AtomicU32 = AtomicU32::new(1);

/// Whether the thread is wanted.
pub fn wanted() -> bool {
    RECORDER_THREAD_WANTED.load(Ordering::Relaxed) != 0
}

/// The affinity mask asked for (`CORES` when none was given).
fn wanted_cores() -> i32 {
    match RECORDER_THREAD_WANTED.load(Ordering::Relaxed) {
        0 | 1 => CORES,
        cores => ((cores & 0xF) as i32) << 16,
    }
}

/// The hooks the served stream's producer uses.
pub static HOOKS: StreamHooks = StreamHooks { wake_consumer: wake, wait };

/// Set by the recorder thread just before it sleeps (and after checking
/// once more that nothing waits): only then does `wake` signal it. The
/// producer calls `wake` every few jobs; a system call each time (~180 per
/// race frame) while the thread was busy anyway cost the capture thread.
/// Sequentially consistent on both sides (the thread: set, then check for
/// work; the producer: publish, then read this), so no wake-up is lost.
static SLEEPING: AtomicBool = AtomicBool::new(false);

/// Wake-ups that signalled the thread (a system call each), since the
/// session started (capture log `wakes=`).
pub static WAKES: AtomicU32 = AtomicU32::new(0);

fn wake() {
    // The jobs were published with a release store: a full fence so that
    // reading the flag can't move ahead of it (store-load ordering).
    core::sync::atomic::fence(Ordering::SeqCst);
    if !SLEEPING.load(Ordering::SeqCst) {
        return;
    }
    SLEEPING.store(false, Ordering::SeqCst);
    WAKES.fetch_add(1, Ordering::Relaxed);
    let sema = SEMA.load(Ordering::Relaxed);
    if sema >= 0 {
        // Safe: a plain system call.
        unsafe { sys::sceKernelSignalSema(sema, 1) };
    }
}

fn wait() {
    let start = sys::now_us();
    // Safe: a plain system call.
    unsafe { sys::sceKernelDelayThread(PRODUCER_WAIT_TRY_US) };
    PRODUCER_WAIT_US.fetch_add((sys::now_us() - start) as u32, Ordering::Relaxed);
}

/// Starts the thread (once). False if it couldn't be started.
pub fn start() -> bool {
    if SEMA.load(Ordering::Relaxed) >= 0 {
        return true;
    }
    // Safe: plain system calls with valid arguments.
    unsafe {
        let sema = sys::sceKernelCreateSema(b"NativePspRecorderWake\0".as_ptr(), 0, 0, 1, core::ptr::null_mut());
        if sema < 0 {
            return false;
        }
        SEMA.store(sema, Ordering::Relaxed);
        let thread = sys::sceKernelCreateThread(
            b"NativePspRecorder\0".as_ptr(),
            recorder_thread,
            PRIORITY,
            STACK_SIZE,
            0,
            CORES,
            core::ptr::null_mut(),
        );
        sys::note_thread(thread);
        THREAD.store(thread, Ordering::Relaxed);
        if thread < 0 || sys::sceKernelStartThread(thread, 0, core::ptr::null_mut()) < 0 {
            SEMA.store(-1, Ordering::Relaxed);
            return false;
        }
    }
    true
}

/// From now on the thread records `stream`'s jobs (null: none). Returns
/// once it no longer works on the stream it served before, so that one's
/// producer may record its jobs itself again.
///
/// (Sequentially consistent: this thread stores ACTIVE then reads IN_USE,
/// the recorder thread stores IN_USE then reads ACTIVE; with weaker
/// orderings both could miss the other's store.)
pub fn serve(stream: *const DrawStream) {
    let before = ACTIVE.swap(stream as *mut DrawStream, Ordering::SeqCst);
    while !before.is_null() && before != stream as *mut DrawStream && IN_USE.load(Ordering::SeqCst) == before {
        wait();
    }
}

/// The stream served now (null: none).
pub fn serving() -> *const DrawStream {
    ACTIVE.load(Ordering::SeqCst)
}

unsafe extern "C" fn recorder_thread(_args: u32, _argp: *mut c_void) -> i32 {
    let sema = SEMA.load(Ordering::Relaxed);
    let mut cores = CORES;
    loop {
        // The cores can be changed at runtime.
        let wanted = wanted_cores();
        if wanted != cores && wanted != 0 {
            // Core 3 included goes through adrenaline_kernel (`threads::set_cores`).
            if crate::threads::set_cores(wanted) >= 0 {
                cores = wanted;
            } else {
                RECORDER_THREAD_WANTED.store(1, Ordering::Relaxed);
            }
        }
        let stream = ACTIVE.load(Ordering::SeqCst);
        IN_USE.store(stream, Ordering::SeqCst);
        // Checked again: `serve` may have switched away in between.
        if !stream.is_null() && ACTIVE.load(Ordering::SeqCst) == stream {
            // Safe: the served streams live in statics; `DrawStream` is made
            // for one producer and this one consumer (`record_published`).
            // Until nothing waits: the producer only wakes a sleeping thread.
            while !(*stream).is_idle() && ACTIVE.load(Ordering::SeqCst) == stream {
                let start = sys::now_us();
                if (*stream).record_published() {
                    CONSUMER_BUSY_US.fetch_add((sys::now_us() - start) as u32, Ordering::Relaxed);
                } else {
                    // Published jobs the copier (copy_thread.rs) hasn't
                    // copied yet: it is microseconds behind the producer.
                    sys::sceKernelDelayThread(COPIER_WAIT_US);
                }
            }
        }
        IN_USE.store(core::ptr::null_mut(), Ordering::SeqCst);
        // Sleep, unless jobs came in meanwhile (`SLEEPING`).
        SLEEPING.store(true, Ordering::SeqCst);
        core::sync::atomic::fence(Ordering::SeqCst);
        let stream = ACTIVE.load(Ordering::SeqCst);
        if stream.is_null() || (*stream).is_idle() {
            let mut timeout = IDLE_TIMEOUT_US;
            sys::sceKernelWaitSema(sema, 1, &mut timeout);
        }
        SLEEPING.store(false, Ordering::SeqCst);
    }
}
