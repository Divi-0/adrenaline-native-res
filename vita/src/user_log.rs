//! The start-up log for users: `ux0:data/nativeres/log.txt`, a few plain
//! lines per Adrenaline start that say whether the plugin loaded and which
//! step failed (Adrenaline not found or a different version, hooks, the PSP
//! side's install, the PSP side never starting). Rewritten at every start.
//!
//! No file at all after starting Adrenaline means taiHEN never loaded the
//! plugin (the config.txt line, the file's name or place, no reboot).
//!
//! Unlike the capture log (log.rs: streamed, never a file, since ux0 file
//! access wedged under constant writes), this one holds a few lines per
//! start and is written only at a handful of moments outside the frame
//! loop: `line` only keeps the text, `save` writes the file.

use core::fmt::{self, Write};
use core::hint::spin_loop;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use nativeres_core::report::TextBuffer;

use crate::{render_log, sys};

const PATH: &[u8] = b"ux0:data/nativeres/log.txt\0";
/// Room for the ~12 lines of a start (lines that don't fit are dropped).
const BYTES: usize = 2048;

struct Buffer(core::cell::UnsafeCell<[u8; BYTES]>);

// Safe: only touched while LOCK is held.
unsafe impl Sync for Buffer {}

static BUFFER: Buffer = Buffer(core::cell::UnsafeCell::new([0; BYTES]));
static LENGTH: AtomicUsize = AtomicUsize::new(0);
static LOCK: AtomicBool = AtomicBool::new(false);
/// Lines kept since the last `save`.
static DIRTY: AtomicBool = AtomicBool::new(false);
/// The game has started (`finish`): no more lines, and once they are
/// saved, no more file access.
static FINISHED: AtomicBool = AtomicBool::new(false);

/// Held only for a copy of at most BYTES. Sleeps when spinning doesn't get
/// it (the holder may be a lower-priority thread on this core).
fn lock() {
    let mut spins = 0;
    while LOCK.compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed).is_err() {
        spins += 1;
        if spins < 100 {
            spin_loop();
        } else {
            // Safe: just sleeps.
            unsafe { sys::sceKernelDelayThread(10) };
        }
    }
}

fn unlock() {
    LOCK.store(false, Ordering::Release);
}

/// SceDateTime (psp2/rtc.h).
#[repr(C)]
#[derive(Default)]
struct DateTime {
    year: u16,
    month: u16,
    day: u16,
    hour: u16,
    minute: u16,
    second: u16,
    microsecond: u32,
}

extern "C" {
    // SceRtc, imported weak: only called once CLOCK is set.
    fn sceRtcGetCurrentClockLocalTime(time: *mut DateTime) -> i32;
}

/// Whether the clock may be read: SceRtc is loaded once AdrenalineUser
/// is (it imports it).
static CLOCK: AtomicBool = AtomicBool::new(false);

/// Lets `line` stamp lines with the date and time (AdrenalineUser loaded).
pub fn clock_ready() {
    CLOCK.store(true, Ordering::Release);
}

/// "YYYY-MM-DD hh:mm:ss.mmm " (local time), or nothing before the clock
/// can be read.
fn stamp(line: &mut TextBuffer) {
    if !CLOCK.load(Ordering::Acquire) {
        return;
    }
    let mut now = DateTime::default();
    // Safe: SceRtc is loaded (CLOCK), `now` has SceDateTime's layout.
    if unsafe { sceRtcGetCurrentClockLocalTime(&mut now) } >= 0 {
        let _ = write!(
            line,
            "{:04}-{:02}-{:02} {:02}:{:02}:{:02}.{:03} ",
            now.year,
            now.month,
            now.day,
            now.hour,
            now.minute,
            now.second,
            now.microsecond / 1000
        );
    }
}

/// Keeps one line (at most 255 bytes) for the file, stamped with the date
/// and time, and puts it into the renderer's log too. Never touches the
/// file (`save` does).
pub fn line(text: fmt::Arguments) {
    let mut bytes = [0u8; 256];
    let mut line = TextBuffer::new(&mut bytes);
    stamp(&mut line);
    // Writing to a TextBuffer can't fail.
    let _ = line.write_fmt(text);
    let _ = line.write_str("\n");
    let length = line.len();
    let text = &bytes[..length];
    render_log::write(text);
    if FINISHED.load(Ordering::Acquire) {
        return;
    }
    lock();
    let used = LENGTH.load(Ordering::Relaxed);
    // Safe: the lock is held; the copy stays inside the buffer.
    unsafe {
        let buffer = &mut *BUFFER.0.get();
        if used + text.len() <= BYTES {
            buffer[used..used + text.len()].copy_from_slice(text);
            LENGTH.store(used + text.len(), Ordering::Release);
        }
    }
    DIRTY.store(true, Ordering::Relaxed);
    unlock();
}

/// The log's last line is in (the game started): later lines are left out,
/// and after the next `save` the file is never touched again.
pub fn finish() {
    FINISHED.store(true, Ordering::Release);
}

/// Held by the thread in `save`.
static SAVING: AtomicBool = AtomicBool::new(false);

/// Writes the file with every line kept so far, if there are new ones.
/// Takes milliseconds: never from the display or capture loop. If another
/// thread is saving, the lines wait for the next call.
///
/// Writes from the buffer without LOCK (so `line` never waits for the
/// file): lines are only appended, the bytes below LENGTH never change.
pub fn save() {
    if !DIRTY.load(Ordering::Relaxed) || SAVING.swap(true, Ordering::Acquire) {
        return;
    }
    DIRTY.store(false, Ordering::Relaxed);
    let length = LENGTH.load(Ordering::Acquire);
    // Safe: NUL-terminated paths; the buffer's first `length` bytes are
    // written once, before LENGTH covered them (Release in `line`).
    unsafe {
        sys::sceIoMkdir(b"ux0:data\0".as_ptr(), 0o777);
        sys::sceIoMkdir(b"ux0:data/nativeres\0".as_ptr(), 0o777);
        let fd = sys::sceIoOpen(PATH.as_ptr(), sys::SCE_O_WRONLY | sys::SCE_O_CREAT | sys::SCE_O_TRUNC, 0o777);
        if fd >= 0 {
            sys::sceIoWrite(fd, (BUFFER.0.get() as *const u8).cast(), length as u32);
            sys::sceIoClose(fd);
        }
    }
    SAVING.store(false, Ordering::Release);
}

/// Whether `path` (NUL-terminated) exists and can be opened.
pub fn file_exists(path: &[u8]) -> bool {
    // Safe: a NUL-terminated path; the file is closed again.
    unsafe {
        let fd = sys::sceIoOpen(path.as_ptr(), sys::SCE_O_RDONLY, 0);
        if fd < 0 {
            return false;
        }
        sys::sceIoClose(fd);
        true
    }
}

/// The plugin's version: the git tag it was built from (build-plugin.sh
/// sets NATIVERES_VERSION), else the crate's.
pub const VERSION: &str = match option_env!("NATIVERES_VERSION") {
    Some(version) => version,
    None => env!("CARGO_PKG_VERSION"),
};
