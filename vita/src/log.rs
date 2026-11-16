//! The capture log: one line per frame, handed to whatever an extension
//! sends it to (`ext::log_send`). Nothing is written on the Vita (the
//! memory card would fill up: a log file had grown to 75 MB once). While
//! nothing takes the lines, they wait in the ring; when it is full, newer
//! lines are dropped and counted (the next session's first line says how
//! many).
//!
//! The capture thread only appends to the ring (`nativepsp_core::log_ring`);
//! a low-priority writer thread does the sending, which can block for
//! milliseconds (that made the capture miss the display loop once a second
//! when it was file I/O).
//!
//! Why no file (2026-11-09): during MTX race soaks ux0 file access wedged
//! for every process, twice, with this writer stuck inside a write (and
//! the emulator's ms0: bridge too); a soak without our file writes stayed
//! clean.

use core::ffi::c_void;
use core::fmt::Write;
use core::sync::atomic::{AtomicI32, AtomicU32, Ordering};

use nativepsp_core::log_ring::LogRing;
use nativepsp_core::report::TextBuffer;

use crate::sys;

/// Below the display (0xA0) and texture refresh (0xA8) threads.
const WRITER_PRIORITY: i32 = 0xB0;
/// The capture thread asks for a send once this much is waiting.
pub const FLUSH_THRESHOLD: usize = 16 * 1024;
/// The longest `drain` waits for a connected PC to take everything.
const DRAIN_TIMEOUT_US: u64 = 1_000_000;

static RING: LogRing<{ 64 * 1024 }> = LogRing::new();
/// Semaphore: there is something to send. -1 = no writer thread.
static WAKEUP: AtomicI32 = AtomicI32::new(-1);
/// Bytes sent to the PC (wrapping).
static STREAMED: AtomicU32 = AtomicU32::new(0);

/// The log's progress: (bytes sent to the PC, bytes
/// waiting, bytes dropped).
pub fn progress() -> (u32, u32, u32) {
    (STREAMED.load(Ordering::Relaxed), RING.used() as u32, RING.dropped())
}

unsafe extern "C" fn writer_thread(_args: u32, _argp: *mut c_void) -> i32 {
    loop {
        sys::sceKernelWaitSema(WAKEUP.load(Ordering::Relaxed), 1, core::ptr::null_mut());
        loop {
            let chunk = RING.pending();
            // Nothing waiting, or no PC: the lines stay in the ring until
            // one connects (log_stream wakes us up then).
            if chunk.is_empty() || !crate::ext::log_send(chunk) {
                break;
            }
            STREAMED.fetch_add(chunk.len() as u32, Ordering::Relaxed);
            RING.consume(chunk.len());
        }
    }
}

/// Starts the writer thread. Without it nothing gets sent (and nothing
/// waits). Lines from before a game runs (the capture thread's start-up,
/// failures) are kept from here on, so a start that hangs before the game
/// leaves a trace (2026-11-09).
pub fn start_writer() {
    append(b"# NativeRes started, no game yet\n");
    // Safe: plain system calls with valid arguments.
    unsafe {
        let sema = sys::sceKernelCreateSema(b"NativePspLogWakeup\0".as_ptr(), 0, 0, 1, core::ptr::null_mut());
        if sema < 0 {
            return;
        }
        let thread = sys::sceKernelCreateThread(
            b"NativePspLogWriter\0".as_ptr(),
            writer_thread,
            WRITER_PRIORITY,
            0x2000,
            0,
            0,
            core::ptr::null_mut(),
        );
        WAKEUP.store(sema, Ordering::Release);
        sys::note_thread(thread);
        if thread < 0 || sys::sceKernelStartThread(thread, 0, core::ptr::null_mut()) < 0 {
            WAKEUP.store(-1, Ordering::Release);
        }
    }
}

/// Appends text; dropped (and counted) if the ring is full. Never blocks.
pub fn append(text: &[u8]) {
    RING.append(text);
}

/// Bytes appended but not sent yet.
pub fn waiting() -> usize {
    RING.used()
}

/// Asks the writer thread to send what was appended so far. Never blocks.
pub fn flush() {
    let sema = WAKEUP.load(Ordering::Acquire);
    if sema >= 0 && RING.used() > 0 {
        // Fails harmlessly if already signalled.
        unsafe { sys::sceKernelSignalSema(sema, 1) };
    }
}

/// Waits (up to 1 s) until everything appended is sent (new session,
/// panic); at once if no PC is connected (the lines then wait in the ring).
pub fn drain() {
    flush();
    let start = sys::now_us();
    while WAKEUP.load(Ordering::Acquire) >= 0
        && RING.used() > 0
        && crate::ext::log_connected()
        && sys::now_us() - start < DRAIN_TIMEOUT_US
    {
        // Safe: just sleeps.
        unsafe { sys::sceKernelDelayThread(1000) };
    }
}

/// A game session starts: its first line.
pub fn open(title_id: &[u8], session: u32) {
    drain();
    let mut line = [0u8; 160];
    let mut text = TextBuffer::new(&mut line);
    let title = core::str::from_utf8(title_id).unwrap_or("?");
    let _ = writeln!(text, "# NativeRes capture, title {title}, session {session:08x}, {} log bytes dropped so far", RING.dropped());
    let length = text.len();
    append(&line[..length]);
    flush();
}
