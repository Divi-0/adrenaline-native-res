//! The frame pool's bookkeeping: a ring of GPU-readable memory for what a
//! frame's draws build (indices, converted vertices), handed out again only
//! after the GPU has finished with it.
//!
//! Why: these bytes used to come from vita2d's pool, which the display loop
//! resets every pass while the GPU may still be drawing earlier passes'
//! scenes (our scene takes ~30 ms of GPU time against a 16.7 ms loop).
//! Overwritten indices make the GPU read outside mapped memory, which the
//! driver reports as "GPU Driver detects GPU Crash" (the crashes of
//! 2026-11-09/10).
//!
//! How: everything allocated between two scene ends is a *segment*. The
//! scene's end has the GPU write the segment's number into a notification
//! word once its fragments are done; a segment's bytes return to the ring
//! only after that. This file holds the ring arithmetic, with no memory and
//! no GPU of its own, so it is tested on the PC; the Vita side
//! (renderer/vita/src/frame_pool.rs) owns the memory block, reads the
//! notification word and does the waiting.

/// Segments closed but not yet seen finished; the GPU is at most a few
/// scenes behind.
pub const MAX_SEGMENTS: usize = 16;

/// Everything allocated between two scene ends.
#[derive(Clone, Copy)]
struct Segment {
    /// Its number, the value the GPU writes when it has finished with it.
    id: u32,
    /// The ring's head when the segment was closed.
    end: u32,
}

/// What one try to allocate gave.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Allocation {
    /// The bytes start at this offset of the ring's memory.
    At(u32),
    /// No room until the GPU has finished this segment: wait for it, then
    /// try again.
    WaitFor(u32),
    /// The open segment alone fills the ring: this frame needs more than the
    /// ring holds.
    TooBig,
}

/// True once the GPU has finished segment `id`, `done` being the newest
/// number it has written. Segment numbers wrap around.
pub fn segment_done(done: u32, id: u32) -> bool {
    done.wrapping_sub(id) as i32 >= 0
}

pub struct FramePool {
    /// The ring's size.
    bytes: u32,
    /// The bytes in use (segments the GPU may still read, and the open one)
    /// run from `tail` to `head` around the ring. One byte always stays free
    /// between head and tail, so head == tail means nothing is in use.
    head: u32,
    tail: u32,
    /// The number of the segment being filled.
    open_id: u32,
    /// Closed segments not yet seen finished, oldest first.
    segments: [Segment; MAX_SEGMENTS],
    segment_count: usize,
}

impl FramePool {
    pub const fn new(bytes: u32) -> Self {
        FramePool { bytes, head: 0, tail: 0, open_id: 1, segments: [Segment { id: 0, end: 0 }; MAX_SEGMENTS], segment_count: 0 }
    }

    /// The number of the segment being filled.
    pub fn open_segment(&self) -> u32 {
        self.open_id
    }

    /// The oldest closed segment the GPU hasn't been seen to finish.
    pub fn oldest_segment(&self) -> Option<u32> {
        (self.segment_count > 0).then(|| self.segments[0].id)
    }

    /// True if no further segment can be closed before one is released.
    pub fn segments_full(&self) -> bool {
        self.segment_count == MAX_SEGMENTS
    }

    /// Gives the segments the GPU has finished (`done`: the newest number it
    /// wrote) back to the ring.
    pub fn release(&mut self, done: u32) {
        let mut released = 0;
        while released < self.segment_count && segment_done(done, self.segments[released].id) {
            self.tail = self.segments[released].end;
            released += 1;
        }
        if released > 0 {
            self.segments.copy_within(released..self.segment_count, 0);
            self.segment_count -= released;
        }
        if self.tail == self.head {
            // Empty: start over, the most room in one piece.
            self.tail = 0;
            self.head = 0;
        }
    }

    /// Where `size` bytes aligned to `align` (a power of two) fit into the
    /// free part of the ring: [head, end) and [0, tail) when head >= tail,
    /// else [head, tail). Not fitting at the end, they go to the start; the
    /// end's rest then waits until the tail has passed it.
    fn fit(&self, size: u32, align: u32) -> Option<u32> {
        let at = (self.head + align - 1) & !(align - 1);
        if self.head >= self.tail {
            // Up to the end, unless that makes head wrap onto a tail at 0.
            if at + size < self.bytes || (at + size == self.bytes && self.tail > 0) {
                return Some(at);
            }
            return (size < self.tail).then_some(0);
        }
        (at + size < self.tail).then_some(at)
    }

    /// One try to take `size` bytes aligned to `align` (a power of two) for
    /// the open segment, after releasing what `done` allows.
    pub fn allocate(&mut self, size: u32, align: u32, done: u32) -> Allocation {
        if size >= self.bytes {
            return Allocation::TooBig;
        }
        let size = size.max(1);
        self.release(done);
        if let Some(at) = self.fit(size, align) {
            self.head = at + size;
            return Allocation::At(at);
        }
        match self.oldest_segment() {
            Some(id) => Allocation::WaitFor(id),
            None => Allocation::TooBig,
        }
    }

    /// At a scene's end: closes the segment of everything allocated since
    /// the last call and returns its number, the value the GPU is to write
    /// when it has finished the scene. Call `release` first while
    /// `segments_full`; if the table is full all the same, the newest closed
    /// segment is kept until this one is done (its bytes are released later
    /// than needed, never earlier).
    pub fn close_segment(&mut self) -> u32 {
        let id = self.open_id;
        let segment = Segment { id, end: self.head };
        if self.segment_count == MAX_SEGMENTS {
            self.segments[MAX_SEGMENTS - 1] = segment;
        } else {
            self.segments[self.segment_count] = segment;
            self.segment_count += 1;
        }
        self.open_id = self.open_id.wrapping_add(1);
        id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BYTES: u32 = 1000;

    /// Allocates or panics with what came instead.
    fn take(pool: &mut FramePool, size: u32, align: u32, done: u32) -> u32 {
        match pool.allocate(size, align, done) {
            Allocation::At(at) => at,
            other => panic!("expected room for {size} bytes, got {other:?}"),
        }
    }

    #[test]
    fn allocations_follow_each_other_at_their_alignment() {
        let mut pool = FramePool::new(BYTES);
        assert_eq!(take(&mut pool, 6, 2, 0), 0);
        assert_eq!(take(&mut pool, 32, 16, 0), 16); // 6 rounded up to 16
        assert_eq!(take(&mut pool, 2, 2, 0), 48);
    }

    #[test]
    fn an_empty_request_still_gets_an_address_of_its_own() {
        let mut pool = FramePool::new(BYTES);
        assert_eq!(take(&mut pool, 0, 1, 0), 0);
        assert_eq!(take(&mut pool, 4, 1, 0), 1);
    }

    #[test]
    fn a_closed_segment_is_not_reused_before_the_gpu_finished_it() {
        let mut pool = FramePool::new(BYTES);
        assert_eq!(take(&mut pool, 600, 4, 0), 0);
        let first = pool.close_segment();
        // 600 more don't fit behind it, and the start is still the GPU's.
        assert_eq!(pool.allocate(600, 4, 0), Allocation::WaitFor(first));
        // Still not after an older number.
        assert_eq!(pool.allocate(600, 4, first.wrapping_sub(1)), Allocation::WaitFor(first));
        // Once the GPU wrote the segment's number, the ring is empty again.
        assert_eq!(take(&mut pool, 600, 4, first), 0);
    }

    #[test]
    fn what_does_not_fit_at_the_end_goes_to_the_start_once_that_is_free() {
        let mut pool = FramePool::new(BYTES);
        take(&mut pool, 300, 4, 0);
        let first = pool.close_segment(); // [0, 300)
        take(&mut pool, 500, 4, 0);
        let second = pool.close_segment(); // [300, 800)
        // 250 fit neither in [800, 1000) nor before the first segment.
        assert_eq!(pool.allocate(250, 4, 0), Allocation::WaitFor(first));
        // The first one finished: [0, 300) is free, 250 < 300 fits there.
        assert_eq!(take(&mut pool, 250, 4, first), 0);
        // The rest up to the second segment's start: one byte stays free.
        assert_eq!(pool.allocate(50, 1, first), Allocation::WaitFor(second));
        assert_eq!(take(&mut pool, 49, 1, first), 250);
    }

    #[test]
    fn the_head_never_wraps_onto_a_tail_at_the_start() {
        let mut pool = FramePool::new(BYTES);
        take(&mut pool, 900, 4, 0);
        // Exactly up to the end would make head == tail == 0 ("empty").
        assert_eq!(pool.allocate(100, 4, 0), Allocation::TooBig);
        assert_eq!(take(&mut pool, 99, 1, 0), 900);
    }

    #[test]
    fn up_to_the_end_is_fine_when_the_tail_has_moved_on() {
        let mut pool = FramePool::new(BYTES);
        take(&mut pool, 300, 4, 0);
        let first = pool.close_segment();
        take(&mut pool, 600, 4, 0);
        pool.close_segment();
        // The first segment is done: tail is at 300, so [900, 1000) is whole.
        assert_eq!(take(&mut pool, 100, 4, first), 900);
        // And the next allocation starts over at the front.
        assert_eq!(take(&mut pool, 200, 4, first), 0);
    }

    #[test]
    fn a_frame_bigger_than_the_ring_is_refused_not_waited_for() {
        let mut pool = FramePool::new(BYTES);
        assert_eq!(pool.allocate(BYTES, 1, 0), Allocation::TooBig);
        take(&mut pool, 700, 4, 0);
        // Only the open segment is in the ring: nothing to wait for.
        assert_eq!(pool.allocate(400, 4, 0), Allocation::TooBig);
    }

    #[test]
    fn segments_are_released_oldest_first_and_only_the_finished_ones() {
        let mut pool = FramePool::new(BYTES);
        take(&mut pool, 100, 4, 0);
        let first = pool.close_segment();
        take(&mut pool, 100, 4, 0);
        let second = pool.close_segment();
        take(&mut pool, 100, 4, 0);
        let third = pool.close_segment();
        assert_eq!(pool.oldest_segment(), Some(first));
        pool.release(second);
        assert_eq!(pool.oldest_segment(), Some(third));
        pool.release(third);
        assert_eq!(pool.oldest_segment(), None);
        // Empty again: the next bytes start at the front.
        assert_eq!(take(&mut pool, 10, 1, third), 0);
    }

    #[test]
    fn segment_numbers_compare_across_the_wrap() {
        assert!(segment_done(5, 5));
        assert!(segment_done(6, 5));
        assert!(!segment_done(4, 5));
        assert!(segment_done(2, u32::MAX)); // the counter wrapped
        assert!(!segment_done(u32::MAX, 2));
    }

    #[test]
    fn a_full_segment_table_keeps_bytes_longer_never_shorter() {
        let mut pool = FramePool::new(BYTES);
        let mut last = 0;
        for _ in 0..MAX_SEGMENTS {
            take(&mut pool, 10, 1, 0);
            last = pool.close_segment();
        }
        assert!(pool.segments_full());
        // One more close without a release: it takes the newest entry's place.
        take(&mut pool, 10, 1, 0);
        let extra = pool.close_segment();
        assert!(pool.segments_full());
        // The replaced segment's bytes stay taken until the extra one is done.
        pool.release(last);
        assert_eq!(pool.oldest_segment(), Some(extra));
        pool.release(extra);
        assert_eq!(pool.oldest_segment(), None);
    }
}
