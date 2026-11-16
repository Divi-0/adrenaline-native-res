//! A small heap: `malloc`/`free` for the few things on the Vita that need
//! memory of a size known only at run time and give it back later — the
//! texture objects of the texture cache and the GPU shader patcher's own
//! bookkeeping (its host-memory callbacks).
//!
//! Why our own: as a plugin of our own there is no C library's heap to
//! borrow; Adrenaline's newlib heap
//! was what `calloc`/`free` used while we were linked into it.
//!
//! How: one block of memory, cut into pieces that each start with an 8-byte
//! header — word 0: the piece's size in bytes, bit 0 set while it is in use;
//! word 1: the size of the piece before it (0 for the first). Sizes are
//! multiples of 8, so every piece and every payload is 8-byte aligned, as
//! `malloc` guarantees. Allocating takes the first free piece big enough
//! (splitting off the rest); freeing merges a piece with free neighbours on
//! both sides, so free space never stays cut up into neighbours.
//!
//! Allocations are rare (a texture made, a GPU program patched), so the
//! search simply walks the pieces in order. The memory is handed in as a
//! slice of words for each call: this file has no memory of its own, so it
//! is tested on the PC; the Vita side (renderer/vita/src/heap.rs) owns the
//! block.

/// Bytes of a piece's header (size word, previous size word).
pub const HEADER_BYTES: u32 = 8;
/// The smallest piece: a header and 8 bytes of payload.
const MIN_PIECE_BYTES: u32 = 16;
/// Bit 0 of the size word: the piece is in use.
const USED: u32 = 1;

/// The heap's bookkeeping; the pieces' headers are in the memory itself.
pub struct Heap {
    /// Bytes of memory managed (a multiple of 8).
    bytes: u32,
    /// Payload bytes in use (headers included), now and at most so far.
    used: u32,
    peak: u32,
    /// Pieces in use.
    allocations: u32,
    /// Allocations that found no room, frees of something not allocated.
    failed: u32,
    bad_frees: u32,
}

fn round_up_8(value: u32) -> Option<u32> {
    value.checked_add(7).map(|value| value & !7)
}

impl Heap {
    /// A heap over no memory (every allocation fails), until `new` makes
    /// the real one.
    pub const EMPTY: Heap = Heap { bytes: 0, used: 0, peak: 0, allocations: 0, failed: 0, bad_frees: 0 };

    /// A heap over `memory`: one free piece covering all of it (rounded
    /// down to whole 8 bytes).
    pub fn new(memory: &mut [u32]) -> Heap {
        let bytes = ((memory.len() * 4) as u32) & !7;
        if bytes >= MIN_PIECE_BYTES {
            memory[0] = bytes;
            memory[1] = 0;
        }
        Heap { bytes: if bytes >= MIN_PIECE_BYTES { bytes } else { 0 }, used: 0, peak: 0, allocations: 0, failed: 0, bad_frees: 0 }
    }

    fn size(memory: &[u32], piece: u32) -> u32 {
        memory[(piece / 4) as usize] & !USED
    }

    fn in_use(memory: &[u32], piece: u32) -> bool {
        memory[(piece / 4) as usize] & USED != 0
    }

    fn set(memory: &mut [u32], piece: u32, size: u32, used: bool) {
        memory[(piece / 4) as usize] = size | if used { USED } else { 0 };
    }

    fn set_previous_size(&self, memory: &mut [u32], piece: u32, previous: u32) {
        if piece < self.bytes {
            memory[(piece / 4 + 1) as usize] = previous;
        }
    }

    /// `size` bytes, 8-byte aligned: the byte offset of the payload in the
    /// memory, or None if no free piece is big enough.
    pub fn allocate(&mut self, memory: &mut [u32], size: u32) -> Option<u32> {
        let Some(need) = round_up_8(size.max(1)).and_then(|payload| payload.checked_add(HEADER_BYTES)) else {
            self.failed += 1;
            return None;
        };
        let mut piece = 0;
        while piece < self.bytes {
            let piece_size = Self::size(memory, piece);
            if !Self::in_use(memory, piece) && piece_size >= need {
                let rest = piece_size - need;
                if rest >= MIN_PIECE_BYTES {
                    Self::set(memory, piece, need, true);
                    let split = piece + need;
                    Self::set(memory, split, rest, false);
                    self.set_previous_size(memory, split, need);
                    self.set_previous_size(memory, split + rest, rest);
                } else {
                    Self::set(memory, piece, piece_size, true);
                }
                let taken = Self::size(memory, piece);
                self.used += taken;
                self.peak = self.peak.max(self.used);
                self.allocations += 1;
                return Some(piece + HEADER_BYTES);
            }
            piece += piece_size;
        }
        self.failed += 1;
        None
    }

    /// Gives back what `allocate` returned at `payload`. False (nothing
    /// changed) if `payload` isn't an allocation in use: a bug of the caller,
    /// counted.
    pub fn free(&mut self, memory: &mut [u32], payload: u32) -> bool {
        let Some(piece) = payload.checked_sub(HEADER_BYTES) else {
            self.bad_frees += 1;
            return false;
        };
        if piece % 8 != 0 || piece >= self.bytes || !self.is_piece(memory, piece) || !Self::in_use(memory, piece) {
            self.bad_frees += 1;
            return false;
        }
        let mut start = piece;
        let mut size = Self::size(memory, piece);
        self.used -= size;
        self.allocations -= 1;
        // Merge with the next piece if it is free.
        let next = start + size;
        if next < self.bytes && !Self::in_use(memory, next) {
            size += Self::size(memory, next);
        }
        // Merge with the previous piece if it is free.
        let previous_size = memory[(start / 4 + 1) as usize];
        if previous_size != 0 && !Self::in_use(memory, start - previous_size) {
            start -= previous_size;
            size += previous_size;
        }
        Self::set(memory, start, size, false);
        self.set_previous_size(memory, start + size, size);
        true
    }

    /// True if a piece starts at byte `offset` (walks from the start: frees
    /// are rare, and a wrong pointer must not corrupt the heap).
    fn is_piece(&self, memory: &[u32], offset: u32) -> bool {
        let mut piece = 0;
        while piece < offset {
            let size = Self::size(memory, piece);
            if size == 0 {
                return false;
            }
            piece += size;
        }
        piece == offset
    }

    /// Bytes managed.
    pub fn bytes(&self) -> u32 {
        self.bytes
    }

    /// Bytes in use (headers included), now and at most so far.
    pub fn used(&self) -> u32 {
        self.used
    }

    pub fn peak(&self) -> u32 {
        self.peak
    }

    pub fn allocations(&self) -> u32 {
        self.allocations
    }

    /// Allocations that found no room.
    pub fn failed(&self) -> u32 {
        self.failed
    }

    /// Frees of something that wasn't an allocation in use.
    pub fn bad_frees(&self) -> u32 {
        self.bad_frees
    }

    /// Checks every header against its neighbours (tests, and checks at
    /// run time): sizes add up to the whole, each previous-size word is right,
    /// no two free pieces are neighbours, and the in-use total matches.
    pub fn consistent(&self, memory: &[u32]) -> bool {
        let mut piece = 0;
        let mut previous = 0;
        let mut previous_free = false;
        let mut used = 0;
        let mut allocations = 0;
        while piece < self.bytes {
            let size = Self::size(memory, piece);
            let free = !Self::in_use(memory, piece);
            if size < MIN_PIECE_BYTES || size % 8 != 0 || memory[(piece / 4 + 1) as usize] != previous || (free && previous_free) {
                return false;
            }
            if !free {
                used += size;
                allocations += 1;
            }
            previous = size;
            previous_free = free;
            piece += size;
        }
        piece == self.bytes && used == self.used && allocations == self.allocations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn heap(words: usize) -> (Heap, Vec<u32>) {
        let mut memory = vec![0u32; words];
        let heap = Heap::new(&mut memory);
        (heap, memory)
    }

    #[test]
    fn allocations_are_8_byte_aligned_and_apart() {
        let (mut heap, mut memory) = heap(256);
        let a = heap.allocate(&mut memory, 100).unwrap();
        let b = heap.allocate(&mut memory, 1).unwrap();
        let c = heap.allocate(&mut memory, 13).unwrap();
        for offset in [a, b, c] {
            assert_eq!(offset % 8, 0);
        }
        assert!(b >= a + 104 && c >= b + 8);
        assert!(heap.consistent(&memory));
        assert_eq!(heap.allocations(), 3);
    }

    #[test]
    fn writes_to_a_payload_stay_inside_it() {
        let (mut heap, mut memory) = heap(256);
        let a = heap.allocate(&mut memory, 100).unwrap();
        let b = heap.allocate(&mut memory, 100).unwrap();
        for word in a / 4..(a + 100) / 4 {
            memory[word as usize] = 0xAAAA_AAAA;
        }
        assert!(heap.consistent(&memory));
        assert!(heap.free(&mut memory, b));
        assert!(heap.free(&mut memory, a));
        assert!(heap.consistent(&memory));
    }

    #[test]
    fn freed_space_is_merged_back_into_one_piece() {
        let (mut heap, mut memory) = heap(256);
        let bytes = heap.bytes();
        let pieces: Vec<u32> = (0..6).map(|_| heap.allocate(&mut memory, 40).unwrap()).collect();
        // Free in an order that merges with the next, the previous, and both.
        for &index in &[1, 3, 2, 0, 5, 4] {
            assert!(heap.free(&mut memory, pieces[index]));
            assert!(heap.consistent(&memory));
        }
        assert_eq!(heap.used(), 0);
        // The whole heap is one piece again: the biggest allocation fits.
        assert_eq!(heap.allocate(&mut memory, bytes - HEADER_BYTES), Some(HEADER_BYTES));
    }

    #[test]
    fn full_heap_says_no_and_counts_it() {
        let (mut heap, mut memory) = heap(64); // 256 bytes
        assert!(heap.allocate(&mut memory, 200).is_some());
        assert_eq!(heap.allocate(&mut memory, 100), None);
        assert_eq!(heap.failed(), 1);
        assert_eq!(heap.allocate(&mut memory, u32::MAX), None);
        assert!(heap.consistent(&memory));
    }

    #[test]
    fn a_small_rest_is_not_split_off() {
        let (mut heap, mut memory) = heap(16); // 64 bytes
        // 48 + 8 header = 56: the 8 left over can't be a piece.
        let a = heap.allocate(&mut memory, 48).unwrap();
        assert_eq!(heap.used(), 64);
        assert!(heap.consistent(&memory));
        assert!(heap.free(&mut memory, a));
        assert_eq!(heap.used(), 0);
    }

    #[test]
    fn freeing_twice_or_a_wrong_pointer_changes_nothing() {
        let (mut heap, mut memory) = heap(256);
        let a = heap.allocate(&mut memory, 32).unwrap();
        let b = heap.allocate(&mut memory, 32).unwrap();
        assert!(heap.free(&mut memory, a));
        assert!(!heap.free(&mut memory, a)); // twice
        assert!(!heap.free(&mut memory, b + 8)); // inside a payload
        assert!(!heap.free(&mut memory, 4)); // misaligned
        assert!(!heap.free(&mut memory, 0)); // before the first payload
        assert!(!heap.free(&mut memory, 4096)); // outside
        assert_eq!(heap.bad_frees(), 5);
        assert!(heap.consistent(&memory));
        assert!(heap.free(&mut memory, b));
    }

    #[test]
    fn freed_room_is_reused_first() {
        let (mut heap, mut memory) = heap(256);
        let a = heap.allocate(&mut memory, 100).unwrap();
        let _b = heap.allocate(&mut memory, 100).unwrap();
        assert!(heap.free(&mut memory, a));
        assert_eq!(heap.allocate(&mut memory, 96), Some(a));
        assert!(heap.consistent(&memory));
    }

    #[test]
    fn peak_remembers_the_most_in_use() {
        let (mut heap, mut memory) = heap(256);
        let a = heap.allocate(&mut memory, 100).unwrap();
        let b = heap.allocate(&mut memory, 100).unwrap();
        let peak = heap.used();
        heap.free(&mut memory, a);
        heap.free(&mut memory, b);
        assert_eq!(heap.peak(), peak);
        assert_eq!(heap.used(), 0);
    }

    /// Many texture objects coming and going, as the texture cache does.
    #[test]
    fn churn_keeps_the_heap_consistent() {
        let (mut heap, mut memory) = heap(16 * 1024);
        let mut live: Vec<u32> = Vec::new();
        let mut seed = 12345u32;
        for _ in 0..5000 {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
            if seed % 3 != 0 || live.is_empty() {
                let size = [100, 24, 300, 8, 1000][(seed >> 8) as usize % 5];
                if let Some(offset) = heap.allocate(&mut memory, size) {
                    live.push(offset);
                }
            } else {
                let index = (seed >> 8) as usize % live.len();
                assert!(heap.free(&mut memory, live.swap_remove(index)));
            }
        }
        assert!(heap.consistent(&memory));
        for offset in live {
            assert!(heap.free(&mut memory, offset));
        }
        assert!(heap.consistent(&memory));
        assert_eq!(heap.used(), 0);
    }
}
