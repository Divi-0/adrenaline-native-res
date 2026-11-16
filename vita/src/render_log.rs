//! The renderer's log: a ring in memory holding the last 64 KiB of text
//! (`read`).
//!
//! It used to go to a file, but opening, writing and closing a file takes
//! ~10 ms, on the display thread: at Final Fantasy's battle start, logging
//! the ~30 decoded textures stalled the picture for a third of a second.
//!
//! Written by the display thread (and now and then by the texture worker:
//! lines may interleave, nothing worse); read by any thread.

use core::cell::UnsafeCell;
use core::fmt::{self, Write};
use core::sync::atomic::{AtomicU32, Ordering};

use nativeres_core::report::TextBuffer;

const RENDER_LOG_SIZE: usize = 64 * 1024;

struct Ring(UnsafeCell<[u8; RENDER_LOG_SIZE]>);

// Safe: bytes written and read one by one; a reader racing a writer sees
// old or new text, never anything invalid.
unsafe impl Sync for Ring {}

static RING: Ring = Ring(UnsafeCell::new([0; RENDER_LOG_SIZE]));
/// Bytes ever written; the newest is at `WRITTEN - 1` modulo the size.
static WRITTEN: AtomicU32 = AtomicU32::new(0);

/// Appends `text`.
pub fn write(text: &[u8]) {
    let written = WRITTEN.load(Ordering::Relaxed);
    let ring = RING.0.get() as *mut u8;
    for (index, &byte) in text.iter().enumerate() {
        let at = (written as usize).wrapping_add(index) % RENDER_LOG_SIZE;
        // Safe: `at` is inside the ring.
        unsafe { ring.add(at).write(byte) };
    }
    WRITTEN.store(written.wrapping_add(text.len() as u32), Ordering::Release);
}

/// Appends one formatted line (at most 255 bytes; the rest is cut off).
pub fn line(text: fmt::Arguments) {
    let mut bytes = [0u8; 256];
    let mut line = TextBuffer::new(&mut bytes);
    // Writing to a TextBuffer can't fail.
    let _ = line.write_fmt(text);
    let length = line.len();
    write(&bytes[..length]);
}

/// Copies the newest part of the log (at most `size` bytes,
/// oldest first) to `out`; returns the length.
///
/// # Safety
/// `out` must hold `size` bytes.
pub unsafe fn read(out: *mut u8, size: i32) -> i32 {
    if out.is_null() || size <= 0 {
        return 0;
    }
    let written = WRITTEN.load(Ordering::Acquire);
    let length = (written as usize).min(RENDER_LOG_SIZE).min(size as usize);
    let ring = RING.0.get() as *const u8;
    for index in 0..length {
        let at = (written as usize).wrapping_sub(length).wrapping_add(index) % RENDER_LOG_SIZE;
        out.add(index).write(ring.add(at).read());
    }
    length as i32
}
