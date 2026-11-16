//! The capture's stage B on a thread of its own (`nativepsp_core::
//! walk_stream`): the capture thread (stage A) follows the display lists,
//! this thread works out the draws from what it walked (draw state,
//! textures, palettes, queuing the vertices for the recorder thread). Off
//! by default (`STAGE_THREAD_WANTED`).
//!
//! It serves one capture at a time (`serve`), with its way to PSP memory:
//! the live capture's reads PSP memory itself (its own block cache, VRAM
//! only where the plugin's copy holds it, noted in `VRAM_READS` for the
//! capture thread); another capture brings its own (`Source::External`).

use core::cell::UnsafeCell;
use core::ffi::c_void;
use core::sync::atomic::{AtomicI32, AtomicPtr, AtomicU32, Ordering};

use nativepsp_core::block_cache::{BlockCache, BlockSource, BLOCK_SIZE, BLOCK_WORDS};
use nativepsp_core::capture::{Capture, StageBRun};
use nativepsp_core::memory::{RAM_END, RAM_START};
use nativepsp_core::vram_requests::{StageVramReads, VramRequests, VRAM_SIZE, VRAM_START};
use nativepsp_core::walk_stream::WalkHooks;

use crate::{platform, sys};

/// As the capture thread (capture.rs): the two stages are one job.
const PRIORITY: i32 = 0x70;
/// Two block caches (~17 KiB each) live on it.
const STACK_SIZE: u32 = 0x20000;
/// Cores 0-2 (as the recorder thread's default).
const CORES: i32 = 0x0001_0000 | 0x0002_0000 | 0x0004_0000;
const IDLE_TIMEOUT_US: u32 = 2000;
/// How long stage A waits for room in the walk stream or the frame's end.
const WAIT_TRY_US: u32 = 50;

static SEMA: AtomicI32 = AtomicI32::new(-1);

/// 0 = off (the capture in one stage), 1 = the default cores (`CORES`),
/// otherwise a mask of user cores as for `RECORDER_THREAD_WANTED` (8 =
/// core 3, only where `threads::set_cores` can). Off by default.
pub static STAGE_THREAD_WANTED: AtomicU32 = AtomicU32::new(0);

pub fn wanted() -> bool {
    STAGE_THREAD_WANTED.load(Ordering::Relaxed) != 0
}

fn wanted_cores() -> i32 {
    match STAGE_THREAD_WANTED.load(Ordering::Relaxed) {
        0 | 1 => CORES,
        cores => ((cores & 0xF) as i32) << 16,
    }
}

/// Microseconds stage A waited for stage B, and stage B was busy, since the
/// last `take_times`.
static WAITED_US: AtomicU32 = AtomicU32::new(0);
static BUSY_US: AtomicU32 = AtomicU32::new(0);

/// Stage B's CPU time (the thread's run clock, not wall time), added up at
/// frame ends, since the last `take_times`.
static CPU_US: AtomicU32 = AtomicU32::new(0);

/// (stage A waited, stage B busy wall time, stage B CPU time) in
/// microseconds since the last call.
pub fn take_times() -> (u32, u32, u32) {
    (WAITED_US.swap(0, Ordering::Relaxed), BUSY_US.swap(0, Ordering::Relaxed), CPU_US.swap(0, Ordering::Relaxed))
}

pub static HOOKS: WalkHooks = WalkHooks { wake_reader: wake, wait };

fn wake() {
    let sema = SEMA.load(Ordering::Relaxed);
    if sema >= 0 {
        // Safe: a plain system call.
        unsafe { sys::sceKernelSignalSema(sema, 1) };
    }
}

fn wait() {
    let start = sys::now_us();
    // Safe: a plain system call.
    unsafe { sys::sceKernelDelayThread(WAIT_TRY_US) };
    WAITED_US.fetch_add((sys::now_us() - start) as u32, Ordering::Relaxed);
}

/// Where the served capture's stage B reads PSP memory.
#[derive(Clone, Copy)]
pub enum Source {
    /// PSP memory; VRAM as `requests` says the plugin's copy holds it.
    Live { requests: *const VramRequests },
    /// Memory of the caller's own: `run` runs stage B of the capture once
    /// (`Capture::run_stage_b_thread` with that memory), on this thread;
    /// `fresh` is true on the first run of a serving.
    External { run: unsafe fn(capture: *mut Capture, fresh: bool) -> StageBRun },
}

/// What the thread serves; written only while it serves nothing else
/// (`serve` waits for it).
struct Served {
    source: UnsafeCell<Source>,
}

// Safe: `source` is written by `serve` only while the thread doesn't use it
// (ACTIVE null and IN_USE null).
unsafe impl Sync for Served {}

static SERVED: Served = Served { source: UnsafeCell::new(Source::Live { requests: core::ptr::null() }) };
static ACTIVE: AtomicPtr<Capture> = AtomicPtr::new(core::ptr::null_mut());
/// Counts `serve` calls: the thread's block caches belong to one serving.
static SERVINGS: AtomicU32 = AtomicU32::new(0);
static IN_USE: AtomicPtr<Capture> = AtomicPtr::new(core::ptr::null_mut());

/// The VRAM blocks the live stage B read since the capture thread last took
/// them (`take_vram_reads`, at a frame's end: stage B is waiting then).
struct VramReads(UnsafeCell<StageVramReads>);
// Safe: used by stage B while it works and by the capture thread at a
// frame's end, when stage B waits (`Capture::run_stage_b_thread`).
unsafe impl Sync for VramReads {}
static VRAM_READS: VramReads = VramReads(UnsafeCell::new(StageVramReads::new()));

/// Capture thread, at a frame's end (stage B waiting): the VRAM blocks
/// stage B read, for `VramRequests::add_stage_reads`.
///
/// # Safety
/// Only while stage B waits at a frame's end or serves nothing.
pub unsafe fn vram_reads() -> &'static mut StageVramReads {
    &mut *VRAM_READS.0.get()
}

/// Starts the thread (once). False if it couldn't be started.
pub fn start() -> bool {
    if SEMA.load(Ordering::Relaxed) >= 0 {
        return true;
    }
    // Safe: plain system calls with valid arguments.
    unsafe {
        let sema = sys::sceKernelCreateSema(b"NativePspStageWake\0".as_ptr(), 0, 0, 1, core::ptr::null_mut());
        if sema < 0 {
            return false;
        }
        SEMA.store(sema, Ordering::Relaxed);
        let thread = sys::sceKernelCreateThread(
            b"NativePspStageB\0".as_ptr(),
            stage_thread,
            PRIORITY,
            STACK_SIZE,
            0,
            CORES,
            core::ptr::null_mut(),
        );
        sys::note_thread(thread);
        if thread < 0 || sys::sceKernelStartThread(thread, 0, core::ptr::null_mut()) < 0 {
            SEMA.store(-1, Ordering::Relaxed);
            return false;
        }
    }
    true
}

/// From now on the thread runs stage B of `capture` (null: none), reading
/// memory from `source`. Returns once it no longer works on the capture it
/// served before. (Sequentially consistent, as `recorder_thread::serve`.)
pub fn serve(capture: *mut Capture, source: Source) {
    let before = ACTIVE.swap(core::ptr::null_mut(), Ordering::SeqCst);
    while !before.is_null() && IN_USE.load(Ordering::SeqCst) == before {
        wait();
    }
    // Safe: nothing serves now, so the thread doesn't read the source.
    unsafe { *SERVED.source.get() = source };
    SERVINGS.fetch_add(1, Ordering::SeqCst);
    ACTIVE.store(capture, Ordering::SeqCst);
    wake();
}

/// The capture served now (null: none) and its source.
pub fn serving() -> (*mut Capture, Source) {
    // Safe: read by the thread that calls `serve`.
    (ACTIVE.load(Ordering::SeqCst), unsafe { *SERVED.source.get() })
}

/// PSP memory for the live stage B: like the capture thread's `VitaBlocks`,
/// without dumps (the capture runs in one stage while a dump runs).
struct StageBlocks {
    requests: *const VramRequests,
}

impl StageBlocks {
    /// VRAM is read from the plugin's copy, which may not hold the block.
    fn vram_ready(&self, address: u32) -> bool {
        if !(VRAM_START..VRAM_START + VRAM_SIZE).contains(&address) {
            return true;
        }
        let block = address & !(BLOCK_SIZE - 1);
        // Safe: the capture thread changes the flags only at a frame's end
        // (`VramRequests::plan`), while stage B waits; `VRAM_READS` is this
        // thread's while it works.
        unsafe {
            let ready = VramRequests::ready_for_stage(self.requests, block);
            (*VRAM_READS.0.get()).note(block, ready);
            ready
        }
    }
}

impl BlockSource for StageBlocks {
    fn copy_block(&mut self, address: u32, out: &mut [u32; BLOCK_WORDS]) -> bool {
        if !self.vram_ready(address) {
            return false;
        }
        let source = platform::psp_memory(address, BLOCK_SIZE);
        if source.is_null() {
            return false;
        }
        // Safe: both hold BLOCK_SIZE bytes.
        unsafe { platform::psp_copy(out.as_mut_ptr() as *mut u8, source, BLOCK_SIZE) };
        true
    }

    /// As the capture thread's `VitaBlocks::defer`: main RAM, and VRAM
    /// blocks the plugin's copy holds (noted in `VRAM_READS`).
    fn defer(&mut self, address: u32, bytes: u32) -> bool {
        let Some(end) = address.checked_add(bytes) else {
            return false;
        };
        if address >= RAM_START && end <= RAM_END {
            return true;
        }
        if address < VRAM_START || end > VRAM_START + VRAM_SIZE || bytes == 0 {
            return false;
        }
        let mut block = address & !(BLOCK_SIZE - 1);
        while block < end {
            if !self.vram_ready(block) {
                return false;
            }
            block += BLOCK_SIZE;
        }
        true
    }

    fn copy_words(&mut self, address: u32, out: &mut [u32]) -> bool {
        if !self.vram_ready(address) {
            return false;
        }
        let bytes = 4 * out.len() as u32;
        let source = platform::psp_memory(address, bytes);
        if source.is_null() {
            return false;
        }
        // Safe: `out` holds `bytes` bytes, and so does `source`.
        unsafe { platform::psp_copy(out.as_mut_ptr() as *mut u8, source, bytes) };
        true
    }
}

unsafe extern "C" fn stage_thread(_args: u32, _argp: *mut c_void) -> i32 {
    let sema = SEMA.load(Ordering::Relaxed);
    let mut cores = CORES;
    // Stage B's block caches, kept from one wake-up to the next (making one
    // clears 16 KiB, and stage B is woken hundreds of times per frame);
    // emptied at every frame end and for every new serving.
    let mut live = BlockCache::new(StageBlocks { requests: core::ptr::null() });
    let mut serving = u32::MAX;
    let mut cpu_at_frame_end = crate::threads::run_us();
    loop {
        let wanted = wanted_cores();
        if wanted != cores && wanted != 0 {
            if crate::threads::set_cores(wanted) >= 0 {
                cores = wanted;
            } else {
                STAGE_THREAD_WANTED.store(1, Ordering::Relaxed);
            }
        }
        let mut timeout = IDLE_TIMEOUT_US;
        sys::sceKernelWaitSema(sema, 1, &mut timeout);
        let capture = ACTIVE.load(Ordering::SeqCst);
        IN_USE.store(capture, Ordering::SeqCst);
        if !capture.is_null() && ACTIVE.load(Ordering::SeqCst) == capture {
            // Safe: `serve` changes the source only while nothing is served.
            let source = *SERVED.source.get();
            let servings = SERVINGS.load(Ordering::SeqCst);
            let mut fresh = false;
            if servings != serving {
                serving = servings;
                fresh = true;
                live.forget();
                if let Source::Live { requests } = source {
                    live.source.requests = requests;
                }
            }
            // Until the stream is empty (or the frame's end reached): stage A
            // may have written more meanwhile without waking us again.
            loop {
                let start = sys::now_us();
                // Safe: the served captures live in statics; this is their
                // only stage B thread, and the source belongs to the served
                // capture.
                let run = match source {
                    Source::External { run } => {
                        let result = run(capture, fresh);
                        fresh = false;
                        result
                    }
                    Source::Live { .. } => {
                        let run = Capture::run_stage_b_thread(capture, &mut live);
                        if run == StageBRun::FrameEnded {
                            live.forget();
                        }
                        run
                    }
                };
                if run == StageBRun::FrameEnded {
                    // One system call per frame: the CPU time this frame took.
                    let now = crate::threads::run_us();
                    CPU_US.fetch_add(now.wrapping_sub(cpu_at_frame_end) as u32, Ordering::Relaxed);
                    cpu_at_frame_end = now;
                }
                if run == StageBRun::Nothing || ACTIVE.load(Ordering::SeqCst) != capture {
                    break;
                }
                BUSY_US.fetch_add((sys::now_us() - start) as u32, Ordering::Relaxed);
            }
        }
        IN_USE.store(core::ptr::null_mut(), Ordering::SeqCst);
    }
}
