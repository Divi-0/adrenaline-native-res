//! Handing finished frames from the capture thread to the display thread.
//!
//! Three frame slots rotate between the threads:
//!   building - being filled by the capture thread,
//!   ready    - the newest complete frame, waiting for the display thread,
//!   shown    - being drawn by the display thread.
//! Publishing swaps building <-> ready, taking swaps ready <-> shown. Both
//! swaps are one atomic exchange of the "ready" slot index (plus a flag
//! saying it holds a frame the display thread hasn't taken), so neither
//! thread ever waits for the other. The newest frame always wins: a ready
//! frame not taken in time is overwritten by the next one, never queued
//! (a queue would add a frame of latency).

use core::sync::atomic::{AtomicU32, Ordering};

pub const SLOTS: usize = 3;
/// In `ready`: the slot holds a frame not taken yet.
const NEW: u32 = 4;

pub struct TripleBuffer {
    /// Index of the ready slot, | NEW.
    ready: AtomicU32,
    /// When each slot's frame was published.
    published_us: [AtomicU32; SLOTS],
}

/// The capture thread's side: which slot it fills.
pub struct Producer {
    building: usize,
}

/// The display thread's side: which slot it draws, and whether that slot
/// holds a complete frame yet.
pub struct Consumer {
    shown: usize,
    has_frame: bool,
}

impl Default for TripleBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl TripleBuffer {
    /// Slot 0 is building, 1 ready (empty), 2 shown (empty).
    pub const fn new() -> Self {
        TripleBuffer { ready: AtomicU32::new(1), published_us: [AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0)] }
    }

    /// The two sides; create them once.
    pub const fn sides() -> (Producer, Consumer) {
        (Producer { building: 0 }, Consumer { shown: 2, has_frame: false })
    }

    /// True if a frame is waiting that the display thread hasn't taken.
    pub fn has_new(&self) -> bool {
        self.ready.load(Ordering::Acquire) & NEW != 0
    }

    /// Capture thread: the building slot's frame is complete. Returns the
    /// slot to fill next (the old ready one; its frame is dropped if the
    /// display thread never took it).
    pub fn publish(&self, producer: &mut Producer, now_us: u32) -> usize {
        self.published_us[producer.building].store(now_us, Ordering::Relaxed);
        let old = self.ready.swap(producer.building as u32 | NEW, Ordering::AcqRel);
        producer.building = (old & !NEW) as usize;
        producer.building
    }

    /// Display thread: takes the newest frame if there is a new one.
    /// Returns (slot to draw, is it new), or None while no frame was ever
    /// published.
    pub fn take(&self, consumer: &mut Consumer) -> Option<(usize, bool)> {
        let mut is_new = false;
        if self.has_new() {
            let old = self.ready.swap(consumer.shown as u32, Ordering::AcqRel);
            consumer.shown = (old & !NEW) as usize;
            consumer.has_frame = true;
            is_new = true;
        }
        consumer.has_frame.then_some((consumer.shown, is_new))
    }

    /// When the frame in `slot` was published.
    pub fn published_us(&self, slot: usize) -> u32 {
        self.published_us[slot].load(Ordering::Relaxed)
    }
}

impl Producer {
    pub fn building(&self) -> usize {
        self.building
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_pass_through_and_slots_never_collide() {
        let buffer = TripleBuffer::new();
        let (mut producer, mut consumer) = TripleBuffer::sides();
        assert_eq!(buffer.take(&mut consumer), None, "nothing published yet");

        let first = producer.building();
        let next = buffer.publish(&mut producer, 10);
        assert_ne!(next, first);
        assert_eq!(buffer.take(&mut consumer), Some((first, true)));
        assert_eq!(buffer.take(&mut consumer), Some((first, false)), "same frame again, not new");
        assert_eq!(buffer.published_us(first), 10);
        assert_ne!(producer.building(), first, "the capture never fills the shown slot");
    }

    #[test]
    fn the_newest_frame_wins() {
        let buffer = TripleBuffer::new();
        let (mut producer, mut consumer) = TripleBuffer::sides();
        let a = producer.building();
        buffer.publish(&mut producer, 1);
        let b = producer.building();
        buffer.publish(&mut producer, 2);
        assert_eq!(producer.building(), a, "a was never taken: refilled");
        assert_eq!(buffer.take(&mut consumer), Some((b, true)));
    }

    #[test]
    fn two_threads_see_each_frame_number_in_order() {
        use std::sync::Arc;
        let buffer = Arc::new(TripleBuffer::new());
        let slots: Arc<[AtomicU32; SLOTS]> = Arc::new([AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0)]);
        let (mut producer, mut consumer) = TripleBuffer::sides();
        let display = {
            let (buffer, slots) = (buffer.clone(), slots.clone());
            std::thread::spawn(move || {
                let mut last = 0;
                while last < 10_000 {
                    if let Some((slot, true)) = buffer.take(&mut consumer) {
                        let number = slots[slot].load(Ordering::Relaxed);
                        assert!(number > last, "frame {number} after {last}");
                        last = number;
                    }
                }
            })
        };
        for number in 1..=10_000 {
            slots[producer.building()].store(number, Ordering::Relaxed);
            buffer.publish(&mut producer, number);
        }
        display.join().unwrap();
    }
}
