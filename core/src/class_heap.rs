//! A heap for the shader compiler app (renderer/shadercompiler): the
//! `malloc`/`free` handed to Sony's runtime Cg compiler
//! (sceShaccCgSetDefaultAllocator). The C app used newlib's heap; a Rust app
//! without a C library has none.
//!
//! Why not `heap.rs`: that one takes the first free piece that fits and
//! checks every free by walking the pieces from the start, fine for a few
//! hundred texture objects, but a compiler makes many thousands of small
//! allocations, which would make every call slow.
//!
//! How: sizes are rounded up to a class (16-byte steps up to 1 KiB, then
//! powers of two), each class keeps a list of freed pieces, and new pieces
//! are cut from the top of the unused memory. Allocating and freeing are
//! constant time; freed pieces are reused only by their own class (no
//! merging), which costs some memory but is plenty for a few compiles in a
//! 32 MiB block. Every piece starts with a 16-byte header — word 0: a tag
//! (`IN_USE` or `FREE` with the class), word 1: the requested size while in
//! use, the next free piece's offset while free — so payloads are 16-byte
//! aligned (more than `malloc`'s 8), and a free of something not handed out
//! (or freed twice) is noticed and ignored, not followed.
//!
//! Like `heap.rs` the memory is handed in as a slice of words for each call,
//! so this file has no memory of its own and is tested on the PC.

/// Bytes of a piece's header.
pub const HEADER_BYTES: u32 = 16;
/// Small classes: payloads of 16, 32, ... 1024 bytes.
const SMALL_STEP: u32 = 16;
const SMALL_CLASSES: u32 = 64;
const SMALL_MAX: u32 = SMALL_STEP * SMALL_CLASSES;
/// Large classes: payloads of 2 KiB, 4 KiB, ... 2 GiB (2^11 .. 2^31).
const LARGE_CLASSES: u32 = 21;
pub const CLASSES: usize = (SMALL_CLASSES + LARGE_CLASSES) as usize;
/// Tags in a header's word 0 (low byte: the class).
const IN_USE: u32 = 0x4E50_5500;
const FREE: u32 = 0x4E50_4600;
/// "No piece" in a free list.
const NONE: u32 = u32::MAX;

/// The class for a payload of `size` bytes (None: bigger than any class).
fn class_of(size: u32) -> Option<u32> {
    let size = size.max(1);
    if size <= SMALL_MAX {
        return Some((size - 1) / SMALL_STEP);
    }
    // The smallest power of two >= size, from 2^11 up.
    let power = 32 - (size - 1).leading_zeros();
    let class = SMALL_CLASSES + (power - 11);
    (class < SMALL_CLASSES + LARGE_CLASSES).then_some(class)
}

/// The payload bytes of `class`.
fn class_bytes(class: u32) -> u32 {
    if class < SMALL_CLASSES {
        (class + 1) * SMALL_STEP
    } else {
        1u32 << (class - SMALL_CLASSES + 11)
    }
}

/// The heap's bookkeeping; the pieces' headers are in the memory itself.
pub struct ClassHeap {
    /// Bytes of memory managed (a multiple of 16).
    bytes: u32,
    /// Start of the memory never handed out yet.
    top: u32,
    /// Per class: the offset of the first freed piece, or NONE.
    free_lists: [u32; CLASSES],
    /// Bytes of pieces in use (headers included), now and at most so far.
    used: u32,
    peak: u32,
    /// Pieces in use now, allocations ever made.
    allocations: u32,
    total_allocations: u32,
    /// Allocations that found no room, frees of something not allocated.
    failed: u32,
    bad_frees: u32,
}

impl ClassHeap {
    /// A heap over `memory_words` words of memory (rounded down to whole
    /// 16 bytes); nothing in the memory is touched.
    pub const fn new(memory_words: usize) -> ClassHeap {
        let words = if memory_words < u32::MAX as usize / 4 { memory_words } else { u32::MAX as usize / 4 };
        let bytes = (words as u32 * 4) & !15;
        ClassHeap {
            bytes,
            top: 0,
            free_lists: [NONE; CLASSES],
            used: 0,
            peak: 0,
            allocations: 0,
            total_allocations: 0,
            failed: 0,
            bad_frees: 0,
        }
    }

    /// `size` bytes, 16-byte aligned: the byte offset of the payload in the
    /// memory, or None if there is no room.
    pub fn allocate(&mut self, memory: &mut [u32], size: u32) -> Option<u32> {
        let Some(class) = class_of(size) else {
            self.failed += 1;
            return None;
        };
        let piece_bytes = HEADER_BYTES + class_bytes(class);
        let reused = self.free_lists[class as usize];
        let piece = if reused != NONE {
            self.free_lists[class as usize] = memory[(reused / 4 + 1) as usize];
            reused
        } else if piece_bytes <= self.bytes - self.top {
            let piece = self.top;
            self.top += piece_bytes;
            piece
        } else {
            self.failed += 1;
            return None;
        };
        memory[(piece / 4) as usize] = IN_USE | class;
        memory[(piece / 4 + 1) as usize] = size;
        self.used += piece_bytes;
        self.peak = self.peak.max(self.used);
        self.allocations += 1;
        self.total_allocations += 1;
        Some(piece + HEADER_BYTES)
    }

    /// Gives back what `allocate` returned at `payload`. False (nothing
    /// changed) if `payload` isn't an allocation in use: counted.
    pub fn free(&mut self, memory: &mut [u32], payload: u32) -> bool {
        let piece = payload.wrapping_sub(HEADER_BYTES);
        if payload < HEADER_BYTES || piece % 16 != 0 || piece >= self.top {
            self.bad_frees += 1;
            return false;
        }
        let tag = memory[(piece / 4) as usize];
        let class = tag & 0xFF;
        if tag & !0xFF != IN_USE || class as usize >= CLASSES || class_bytes(class) > self.top - piece - HEADER_BYTES {
            self.bad_frees += 1;
            return false;
        }
        memory[(piece / 4) as usize] = FREE | class;
        memory[(piece / 4 + 1) as usize] = self.free_lists[class as usize];
        self.free_lists[class as usize] = piece;
        self.used -= HEADER_BYTES + class_bytes(class);
        self.allocations -= 1;
        true
    }

    /// The size `allocate` was asked for at `payload` (None if it isn't in
    /// use): what a `realloc` would copy.
    pub fn requested_size(&self, memory: &[u32], payload: u32) -> Option<u32> {
        let piece = payload.checked_sub(HEADER_BYTES)?;
        if piece % 16 != 0 || piece >= self.top || memory[(piece / 4) as usize] & !0xFF != IN_USE {
            return None;
        }
        Some(memory[(piece / 4 + 1) as usize])
    }

    /// Bytes managed.
    pub fn bytes(&self) -> u32 {
        self.bytes
    }

    /// Bytes of pieces in use (headers included).
    pub fn used(&self) -> u32 {
        self.used
    }

    /// The most `used` has been.
    pub fn peak(&self) -> u32 {
        self.peak
    }

    /// Bytes ever cut from the memory (in use or on a free list).
    pub fn top(&self) -> u32 {
        self.top
    }

    /// Pieces in use now.
    pub fn allocations(&self) -> u32 {
        self.allocations
    }

    /// Allocations ever made.
    pub fn total_allocations(&self) -> u32 {
        self.total_allocations
    }

    /// Allocations that found no room.
    pub fn failed(&self) -> u32 {
        self.failed
    }

    /// Frees of something not in use (ignored).
    pub fn bad_frees(&self) -> u32 {
        self.bad_frees
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn heap(words: usize) -> (ClassHeap, Vec<u32>) {
        (ClassHeap::new(words), vec![0u32; words])
    }

    #[test]
    fn classes_round_up_and_cover_every_size() {
        assert_eq!(class_bytes(class_of(0).unwrap()), 16);
        assert_eq!(class_bytes(class_of(1).unwrap()), 16);
        assert_eq!(class_bytes(class_of(16).unwrap()), 16);
        assert_eq!(class_bytes(class_of(17).unwrap()), 32);
        assert_eq!(class_bytes(class_of(1024).unwrap()), 1024);
        assert_eq!(class_bytes(class_of(1025).unwrap()), 2048);
        assert_eq!(class_bytes(class_of(2048).unwrap()), 2048);
        assert_eq!(class_bytes(class_of(2049).unwrap()), 4096);
        assert_eq!(class_bytes(class_of(1 << 31).unwrap()), 1 << 31);
        assert_eq!(class_of((1 << 31) + 1), None);
        for size in 1..70_000u32 {
            let class = class_of(size).unwrap();
            assert!(class_bytes(class) >= size, "{size}");
            assert!(class == 0 || class_bytes(class - 1) < size, "{size}");
        }
    }

    #[test]
    fn allocations_are_16_byte_aligned_and_apart() {
        let (mut heap, mut memory) = heap(1024);
        let a = heap.allocate(&mut memory, 100).unwrap();
        let b = heap.allocate(&mut memory, 1).unwrap();
        let c = heap.allocate(&mut memory, 1500).unwrap();
        for offset in [a, b, c] {
            assert_eq!(offset % 16, 0);
        }
        assert!(b >= a + 112 && c >= b + 16 + HEADER_BYTES);
        assert_eq!(heap.allocations(), 3);
        assert_eq!(heap.requested_size(&memory, c), Some(1500));
        assert_eq!(heap.used(), 3 * HEADER_BYTES + 112 + 16 + 2048);
    }

    #[test]
    fn writes_to_a_payload_stay_inside_it() {
        let (mut heap, mut memory) = heap(1024);
        let a = heap.allocate(&mut memory, 100).unwrap();
        let b = heap.allocate(&mut memory, 100).unwrap();
        for word in a / 4..(a + 112) / 4 {
            memory[word as usize] = 0xAAAA_AAAA;
        }
        assert!(heap.free(&mut memory, b));
        assert!(heap.free(&mut memory, a));
        assert_eq!(heap.used(), 0);
        assert_eq!(heap.bad_frees(), 0);
    }

    #[test]
    fn a_freed_piece_is_reused_by_its_class_only() {
        let (mut heap, mut memory) = heap(1024);
        let a = heap.allocate(&mut memory, 40).unwrap();
        assert!(heap.free(&mut memory, a));
        let top = heap.top();
        // Another class: cut new memory.
        let b = heap.allocate(&mut memory, 200).unwrap();
        assert_ne!(b, a);
        assert!(heap.top() > top);
        // Same class (33..48 bytes): the freed piece again.
        assert_eq!(heap.allocate(&mut memory, 33), Some(a));
        assert_eq!(heap.allocations(), 2);
        assert_eq!(heap.total_allocations(), 3);
    }

    #[test]
    fn free_lists_hand_back_every_freed_piece() {
        let (mut heap, mut memory) = heap(4096);
        let pieces: Vec<u32> = (0..20).map(|_| heap.allocate(&mut memory, 24).unwrap()).collect();
        for &piece in &pieces {
            assert!(heap.free(&mut memory, piece));
        }
        let top = heap.top();
        let mut again: Vec<u32> = (0..20).map(|_| heap.allocate(&mut memory, 30).unwrap()).collect();
        assert_eq!(heap.top(), top);
        again.sort_unstable();
        assert_eq!(again, pieces);
    }

    #[test]
    fn bad_and_double_frees_are_ignored() {
        let (mut heap, mut memory) = heap(1024);
        let a = heap.allocate(&mut memory, 64).unwrap();
        let b = heap.allocate(&mut memory, 64).unwrap();
        assert!(!heap.free(&mut memory, 0));
        assert!(!heap.free(&mut memory, 8));
        assert!(!heap.free(&mut memory, a + 16));
        assert!(!heap.free(&mut memory, heap.top() + HEADER_BYTES));
        assert!(!heap.free(&mut memory, u32::MAX));
        assert!(heap.free(&mut memory, a));
        assert!(!heap.free(&mut memory, a));
        assert_eq!(heap.bad_frees(), 6);
        assert_eq!(heap.allocations(), 1);
        // The free list wasn't broken: a comes back once, then new memory.
        assert_eq!(heap.allocate(&mut memory, 64), Some(a));
        let c = heap.allocate(&mut memory, 64).unwrap();
        assert!(c != a && c != b);
    }

    #[test]
    fn a_payload_that_looks_like_a_header_is_not_a_piece() {
        let (mut heap, mut memory) = heap(1024);
        let a = heap.allocate(&mut memory, 256).unwrap();
        // The caller writes a fake in-use header inside its payload.
        memory[(a / 4 + 4) as usize] = IN_USE | 70;
        assert!(!heap.free(&mut memory, a + 32));
        assert_eq!(heap.bad_frees(), 1);
    }

    #[test]
    fn running_out_fails_without_damage() {
        let (mut heap, mut memory) = heap(64); // 256 bytes
        assert_eq!(heap.bytes(), 256);
        let a = heap.allocate(&mut memory, 200).unwrap(); // 16 + 208
        assert_eq!(heap.allocate(&mut memory, 100), None);
        assert_eq!(heap.allocate(&mut memory, u32::MAX), None);
        assert_eq!(heap.failed(), 2);
        let b = heap.allocate(&mut memory, 16).unwrap(); // 16 + 16 = the rest
        assert_eq!(heap.top(), 256);
        assert!(heap.free(&mut memory, a));
        assert!(heap.free(&mut memory, b));
        assert_eq!(heap.peak(), 256);
    }
}
