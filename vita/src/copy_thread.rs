//! The thread that copies the capture's draws' vertex memory out of PSP RAM
//! (the copier of `nativeres_core::draw_stream`), on by default, on cores 0-2.
//!
//! Why: reading PSP RAM is slow for the Vita's CPU (uncached, ~1 us per
//! 128-byte burst while the game runs), and MTX's race copies ~1 MiB per
//! frame in ~2,200 pieces: ~9.5 ms of the capture thread's ~36 ms
//! (2026-11-08). The recorder thread can't copy them instead: it runs up to
//! a frame behind, and the game has written new vertices there by then
//! (`deferredcopy 1`: torn riders). This thread copies each job right after
//! the capture queues it; when it falls behind, the capture thread copies
//! the oldest jobs itself (`draw_stream::COPIER_LAG_JOBS`). Reads from
//! another core add bandwidth: core 3 copied ~100 MB/s while the capture and
//! recorder threads lost only 0.2-0.6 ms.
//!
//! Live, MTX's paused race (2026-11-08): frames recorded 81% without the
//! thread, 99% with it on cores 0-2 (none over 45 ms), 97% on all four cores
//! and 97-98% on core 3 alone, both with frames of 60-100 ms: the system
//! takes core 3 away now and then.
//!
//! While a frame is walked the thread polls with short sleeps instead of
//! waiting for a wake-up: the capture thread would otherwise pay a system
//! call every few jobs to wake it (`recorder_thread::wake` cost ~11 us
//! each). Idle for `IDLE_POLLS` polls, it sleeps until woken.

use core::ffi::c_void;
use core::sync::atomic::{AtomicBool, AtomicI32, AtomicPtr, AtomicU32, Ordering};

use nativeres_core::draw_stream::{CopierHooks, DrawStream};

use crate::sys;

/// Just below the display thread (AdrenalineDraw, 0xA0): it has slack
/// (busy ~21-25 of 33 ms), and the display's render must not wait for it
/// at publish time. 0x70 (the capture's) made ~26% of frames miss the
/// refresh after next (3/1 wobble); 0xA1 halves the webcam's 3/1 pairs at
/// no game-side cost.
const PRIORITY: i32 = 0xA1;
const STACK_SIZE: u32 = 0x4000;
/// Cores 0-2: the default, and where the thread starts; it moves itself to
/// other cores asked for.
const CREATION_CORES: i32 = 0x0001_0000 | 0x0002_0000 | 0x0004_0000;
/// Between polls while a frame is walked.
const POLL_US: u32 = 50;
/// Polls without work before the thread sleeps until woken (~1 ms: the
/// capture queues jobs every few microseconds while it walks a list).
const IDLE_POLLS: u32 = 20;
/// How long it sleeps without being woken (it checks then anyway).
const IDLE_TIMEOUT_US: u32 = 2000;

/// 0 = off (the capture thread copies), 1 = on, cores 0-2 (the default),
/// otherwise a mask of user cores as for `RECORDER_THREAD_WANTED` (8 =
/// core 3, 15 = all four). Needs the recorder thread. Core 3 refused
/// (`threads::set_cores`), the thread stays on cores 0-2.
pub static COPY_THREAD_WANTED: AtomicU32 = AtomicU32::new(1);

/// Whether the copier thread is wanted.
pub fn wanted() -> bool {
    COPY_THREAD_WANTED.load(Ordering::Relaxed) != 0
}

fn wanted_cores() -> i32 {
    match COPY_THREAD_WANTED.load(Ordering::Relaxed) {
        0 | 1 => CREATION_CORES,
        cores => ((cores & 0xF) as i32) << 16,
    }
}

static SEMA: AtomicI32 = AtomicI32::new(-1);
static THREAD: AtomicI32 = AtomicI32::new(-1);
static ACTIVE: AtomicPtr<DrawStream> = AtomicPtr::new(core::ptr::null_mut());
/// The stream the thread is copying for right now (null between).
static IN_USE: AtomicPtr<DrawStream> = AtomicPtr::new(core::ptr::null_mut());

/// Microseconds spent copying since the last `take_busy_us` (capture log
/// `copier=`).
static BUSY_US: AtomicU32 = AtomicU32::new(0);

/// Wake-ups that signalled the thread (a system call each), since the
/// session started (capture log `copier=<busy>us/<wakes>/<jobs the capture
/// thread copied itself>/<reads that disagreed>`, `draw_stream::PRODUCER_COPIES`,
/// `draw_stream::UNSTABLE_READS`).
pub static WAKES: AtomicU32 = AtomicU32::new(0);

/// The thread's copying time since the last call.
pub fn take_busy_us() -> u32 {
    BUSY_US.swap(0, Ordering::Relaxed)
}

/// The hooks the served stream's producer uses.
pub static HOOKS: CopierHooks = CopierHooks { wake };

/// Set by the thread just before it sleeps (after checking once more that
/// nothing waits): only then does `wake` signal it. Sequentially consistent
/// on both sides, as `recorder_thread::SLEEPING`.
static SLEEPING: AtomicBool = AtomicBool::new(false);

fn wake() {
    // The job was published with a release store: a full fence so that
    // reading the flag can't move ahead of it.
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

/// Starts the thread (once). False if it couldn't be started.
pub fn start() -> bool {
    if SEMA.load(Ordering::Relaxed) >= 0 {
        return true;
    }
    // Safe: plain system calls with valid arguments.
    unsafe {
        let sema = sys::sceKernelCreateSema(b"NativeResCopierWake\0".as_ptr(), 0, 0, 1, core::ptr::null_mut());
        if sema < 0 {
            return false;
        }
        SEMA.store(sema, Ordering::Relaxed);
        let thread = sys::sceKernelCreateThread(
            b"NativeResCopier\0".as_ptr(),
            copy_thread,
            PRIORITY,
            STACK_SIZE,
            0,
            CREATION_CORES,
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

/// From now on the thread copies for `stream` (null: none). Returns once it
/// no longer works on the stream it served before. The stream must have
/// been told about the copier first (`DrawStream::set_copier`), and told
/// to stop using it before this switches away from it.
pub fn serve(stream: *const DrawStream) {
    let before = ACTIVE.swap(stream as *mut DrawStream, Ordering::SeqCst);
    while !before.is_null() && before != stream as *mut DrawStream && IN_USE.load(Ordering::SeqCst) == before {
        // Safe: a plain system call.
        unsafe { sys::sceKernelDelayThread(POLL_US) };
    }
}

unsafe extern "C" fn copy_thread(_args: u32, _argp: *mut c_void) -> i32 {
    let sema = SEMA.load(Ordering::Relaxed);
    let mut cores = CREATION_CORES;
    loop {
        // The cores can be changed at runtime.
        let wanted = wanted_cores();
        if wanted != cores && wanted != 0 {
            // Core 3 included goes through adrenaline_kernel (`threads::set_cores`).
            if crate::threads::set_cores(wanted) >= 0 {
                cores = wanted;
            }
        }
        let stream = ACTIVE.load(Ordering::SeqCst);
        IN_USE.store(stream, Ordering::SeqCst);
        // Checked again: `serve` may have switched away in between.
        if !stream.is_null() && ACTIVE.load(Ordering::SeqCst) == stream {
            let mut idle_polls = 0;
            while idle_polls < IDLE_POLLS && ACTIVE.load(Ordering::SeqCst) == stream {
                let start = sys::now_us();
                // Safe: the served streams live in statics; `DrawStream`
                // is made for one copier (`copy_published`).
                if (*stream).copy_published() {
                    BUSY_US.fetch_add((sys::now_us() - start) as u32, Ordering::Relaxed);
                    idle_polls = 0;
                } else {
                    idle_polls += 1;
                    sys::sceKernelDelayThread(POLL_US);
                }
            }
        }
        IN_USE.store(core::ptr::null_mut(), Ordering::SeqCst);
        // Sleep, unless jobs came in meanwhile (`SLEEPING`).
        SLEEPING.store(true, Ordering::SeqCst);
        core::sync::atomic::fence(Ordering::SeqCst);
        let stream = ACTIVE.load(Ordering::SeqCst);
        if stream.is_null() || (*stream).all_copied() {
            let mut timeout = IDLE_TIMEOUT_US;
            sys::sceKernelWaitSema(sema, 1, &mut timeout);
        }
        SLEEPING.store(false, Ordering::SeqCst);
    }
}
