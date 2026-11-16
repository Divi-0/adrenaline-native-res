//! A byte ring for log text: one thread appends, another writes it out.
//!
//! Writing to ux0 can block for many milliseconds, which made the capture
//! thread miss the display loop once a second (a frame shown twice, then
//! one skipped: a small stutter while walking). So the capture thread only
//! appends here, and a low-priority writer thread does the file I/O. Text
//! that doesn't fit is dropped and counted, never waited for.

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicU32, Ordering};

pub struct LogRing<const SIZE: usize> {
    bytes: UnsafeCell<[u8; SIZE]>,
    /// Bytes appended so far (only the appending thread writes it).
    head: AtomicU32,
    /// Bytes taken out so far (only the writing thread writes it).
    tail: AtomicU32,
    /// Bytes that didn't fit.
    dropped: AtomicU32,
}

// One appending and one consuming thread (see the method docs); the
// head/tail counters hand the bytes over between them.
unsafe impl<const SIZE: usize> Sync for LogRing<SIZE> {}

impl<const SIZE: usize> Default for LogRing<SIZE> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const SIZE: usize> LogRing<SIZE> {
    const SIZE_IS_A_POWER_OF_TWO: () = assert!(SIZE.is_power_of_two() && SIZE <= 1 << 31);

    pub const fn new() -> Self {
        let () = Self::SIZE_IS_A_POWER_OF_TWO;
        LogRing { bytes: UnsafeCell::new([0; SIZE]), head: AtomicU32::new(0), tail: AtomicU32::new(0), dropped: AtomicU32::new(0) }
    }

    /// Appending thread only. Drops `text` (and counts it) if it doesn't fit.
    pub fn append(&self, text: &[u8]) -> bool {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);
        let free = SIZE - head.wrapping_sub(tail) as usize;
        if text.len() > free {
            self.dropped.fetch_add(text.len() as u32, Ordering::Relaxed);
            return false;
        }
        let start = head as usize % SIZE;
        let first = text.len().min(SIZE - start);
        // Safe: these bytes are free, and only this thread writes free bytes.
        unsafe {
            let bytes = &mut *self.bytes.get();
            bytes[start..start + first].copy_from_slice(&text[..first]);
            bytes[..text.len() - first].copy_from_slice(&text[first..]);
        }
        self.head.store(head.wrapping_add(text.len() as u32), Ordering::Release);
        true
    }

    /// Consuming thread only: the oldest appended bytes not taken yet, up to
    /// the end of the ring (the rest comes with the next call). Pass the
    /// length used to [`LogRing::consume`] afterwards.
    pub fn pending(&self) -> &[u8] {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Relaxed);
        let length = head.wrapping_sub(tail) as usize;
        let start = tail as usize % SIZE;
        // Safe: bytes between tail and head are complete and only change
        // after `consume` hands them back.
        unsafe { &(&*self.bytes.get())[start..start + length.min(SIZE - start)] }
    }

    /// Consuming thread only: `length` bytes of `pending` are written out.
    pub fn consume(&self, length: usize) {
        let tail = self.tail.load(Ordering::Relaxed);
        self.tail.store(tail.wrapping_add(length as u32), Ordering::Release);
    }

    /// Bytes appended but not consumed yet.
    pub fn used(&self) -> usize {
        self.head.load(Ordering::Acquire).wrapping_sub(self.tail.load(Ordering::Acquire)) as usize
    }

    pub fn dropped(&self) -> u32 {
        self.dropped.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drain<const SIZE: usize>(ring: &LogRing<SIZE>) -> Vec<u8> {
        let mut out = Vec::new();
        loop {
            let chunk = ring.pending();
            if chunk.is_empty() {
                return out;
            }
            out.extend_from_slice(chunk);
            let length = chunk.len();
            ring.consume(length);
        }
    }

    #[test]
    fn text_comes_out_as_it_went_in_across_the_end_of_the_ring() {
        let ring = LogRing::<16>::new();
        assert!(ring.append(b"0123456789"));
        assert_eq!(drain(&ring), b"0123456789");
        assert!(ring.append(b"abcdefghij")); // wraps at byte 16
        assert_eq!(ring.pending(), b"abcdef", "up to the end first");
        assert_eq!(drain(&ring), b"abcdefghij");
        assert_eq!(ring.used(), 0);
    }

    #[test]
    fn text_that_does_not_fit_is_dropped_and_counted() {
        let ring = LogRing::<16>::new();
        assert!(ring.append(b"0123456789"));
        assert!(!ring.append(b"0123456789"));
        assert_eq!(ring.dropped(), 10);
        assert!(ring.append(b"012345"), "exactly full is fine");
        assert_eq!(ring.used(), 16);
        assert_eq!(drain(&ring), b"0123456789012345");
    }

    #[test]
    fn two_threads() {
        let ring = std::sync::Arc::new(LogRing::<64>::new());
        let writer = {
            let ring = ring.clone();
            std::thread::spawn(move || {
                let mut out = Vec::new();
                while out.len() < 1000 {
                    let chunk = ring.pending();
                    out.extend_from_slice(chunk);
                    let length = chunk.len();
                    ring.consume(length);
                }
                out
            })
        };
        let mut sent = Vec::new();
        let mut n = 0u8;
        while sent.len() < 1000 {
            let piece = [n; 5];
            if ring.append(&piece) {
                sent.extend_from_slice(&piece);
                n = n.wrapping_add(1);
            }
        }
        assert_eq!(writer.join().unwrap(), sent);
    }
}
