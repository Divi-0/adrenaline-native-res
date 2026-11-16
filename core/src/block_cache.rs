//! PSP memory read in blocks through a small cache.
//!
//! The Vita maps PSP RAM uncached, so reading it word by word is very slow
//! (the capture thread needed ~15 ms per frame for Final Fantasy's menu).
//! Instead each block is copied once into a small cache of normal memory
//! and read from there, a line (`LINE_SIZE`) at a time as it is needed:
//! copying from uncached PSP RAM costs per byte (~9 us per KiB on the Vita,
//! 2026-11-05), so how much a frame copies matters.
//!
//! The cache lives here rather than in the Vita crate so the PC replay tool
//! runs the very same code: `replay --bench` then measures the capture as
//! the Vita runs it and counts the block copies it would make.

use crate::memory::{self, PspMemory};

/// Bytes per block. Frame dumps record memory in blocks of this size too.
pub const BLOCK_SIZE: u32 = 1024;
pub const BLOCK_WORDS: usize = BLOCK_SIZE as usize / 4;

/// Enough for a display list, its vertex buffers and palettes at once.
const CACHED_BLOCKS: usize = 16;

/// A cached block is filled a line at a time, as it is read: copying from
/// uncached PSP RAM costs per byte (~9 us per KiB on the Vita), and of the
/// blocks MTX's race reads (lists, small sub-lists scattered over memory,
/// index buffers) at most 60% of the lines are read.
pub const LINE_SIZE: u32 = 256;
pub const LINE_WORDS: usize = LINE_SIZE as usize / 4;
const LINES_PER_BLOCK: usize = (BLOCK_SIZE / LINE_SIZE) as usize;
const ALL_LINES: u8 = (1 << LINES_PER_BLOCK) - 1;

/// `read_words` reads a piece of an uncached block of at least this many
/// bytes straight from the source, without caching the whole block:
/// palettes and vertex buffers are read once, and a whole 1 KiB block from
/// uncached PSP RAM costs ~8 us on the Vita whatever part of it is needed
/// (MTX's race: ~580 palette loads of ~360 bytes each copied whole blocks).
/// Smaller reads (FF1's rectangles, a few words each, many per block) still
/// go through the cache.
pub const DIRECT_READ_MIN_BYTES: usize = 256;

/// Where the cache gets blocks from.
pub trait BlockSource {
    /// Copies the block at physical, block-aligned `address` into `out`.
    /// False if that memory isn't mapped; `out` must then be left as it is
    /// (it may hold another cached block).
    fn copy_block(&mut self, address: u32, out: &mut [u32; BLOCK_WORDS]) -> bool;

    /// Copies `out.len()` words at physical, word-aligned `address` (all in
    /// one block) straight into `out`, if this source can (see
    /// `DIRECT_READ_MIN_BYTES`; also used for single lines). False: the
    /// whole block is copied with `copy_block` instead (e.g. while a dump
    /// runs: the dump records whole blocks).
    fn copy_words(&mut self, _address: u32, _out: &mut [u32]) -> bool {
        false
    }

    /// `PspMemory::defer` for this source. Default: false.
    fn defer(&mut self, _address: u32, _bytes: u32) -> bool {
        false
    }
    /// The words at physical, word-aligned `address + 4 * positions[i]`
    /// into `out[i]`, read straight from the source (`PspMemory::read_samples`;
    /// the palette store's samples), if this source can. False: they are
    /// read through the cache instead. Default: each word with `copy_words`
    /// (the dumps; the Vita reads them with single loads).
    fn read_samples(&mut self, address: u32, positions: &[usize], out: &mut [u32]) -> bool {
        positions
            .iter()
            .zip(out.iter_mut())
            .all(|(&position, word)| self.copy_words(address + 4 * position as u32, core::slice::from_mut(word)))
    }
}

/// PSP memory read through a cache of `CACHED_BLOCKS` blocks. A new cache
/// (empty) is used for every capture event, so data the game wrote in the
/// meantime is picked up.
///
/// Main RAM and VRAM are readable; everything else reads as `None`.
pub struct BlockCache<S: BlockSource> {
    addresses: [u32; CACHED_BLOCKS],
    blocks: [[u32; BLOCK_WORDS]; CACHED_BLOCKS],
    /// Which lines (`LINE_SIZE`) of each cached block are filled.
    lines: [u8; CACHED_BLOCKS],
    used: usize,
    /// The block read last, checked first (most reads hit the same block),
    /// and the one before it, checked next (draws alternate between the
    /// display list and their vertices).
    last: usize,
    previous: usize,
    /// When each slot was last used; the least recently used one is replaced.
    last_used: [u32; CACHED_BLOCKS],
    clock: u32,
    /// Whole blocks copied from the source so far.
    pub copies: u32,
    /// Single lines copied from the source so far.
    pub line_copies: u32,
    /// Fill whole blocks at once instead of lines (for comparing; debug
    /// "copylines 0" on the Vita).
    pub whole_blocks: bool,
    /// Pieces of blocks read straight from the source (`read_words`).
    pub direct_reads: u32,
    pub source: S,
}

impl<S: BlockSource> BlockCache<S> {
    pub fn new(source: S) -> Self {
        BlockCache {
            addresses: [0; CACHED_BLOCKS],
            blocks: [[0; BLOCK_WORDS]; CACHED_BLOCKS],
            lines: [0; CACHED_BLOCKS],
            used: 0,
            last: 0,
            previous: 0,
            last_used: [0; CACHED_BLOCKS],
            clock: 0,
            copies: 0,
            line_copies: 0,
            whole_blocks: false,
            direct_reads: 0,
            source,
        }
    }

    /// Empties the cache (for reading memory that may have changed) without
    /// clearing its 16 KiB of block data, as `new` does: for a cache kept
    /// across many short uses (the capture's stage B, `walk_stream`).
    pub fn forget(&mut self) {
        self.used = 0;
        self.lines = [0; CACHED_BLOCKS];
        self.last = 0;
        self.previous = 0;
        self.last_used = [0; CACHED_BLOCKS];
        self.clock = 0;
    }

    /// The cache slot holding the block at `block_address`, loading it if
    /// needed. `None` if the block isn't readable.
    #[inline]
    fn slot_for(&mut self, block_address: u32) -> Option<usize> {
        if self.used > 0 && self.addresses[self.last] == block_address {
            return Some(self.last);
        }
        self.slot_for_other(block_address)
    }

    /// The slot already holding the block at `block_address`, if any.
    fn cached_slot(&self, block_address: u32) -> Option<usize> {
        if self.used > 0 && self.addresses[self.last] == block_address {
            return Some(self.last);
        }
        self.addresses[..self.used].iter().position(|&address| address == block_address)
    }

    fn slot_for_other(&mut self, block_address: u32) -> Option<usize> {
        self.clock += 1;
        let slot = if self.used > 1 && self.addresses[self.previous] == block_address {
            Some(self.previous)
        } else {
            self.addresses[..self.used].iter().position(|&address| address == block_address)
        };
        if let Some(slot) = slot {
            self.previous = self.last;
            self.last = slot;
            self.last_used[slot] = self.clock;
            return Some(slot);
        }

        // RAM and VRAM start and end on block boundaries, so checking the
        // whole block here covers every word read from it.
        if !memory::is_readable(block_address, BLOCK_SIZE) {
            return None;
        }

        let slot = if self.used < CACHED_BLOCKS {
            self.used
        } else {
            (0..CACHED_BLOCKS).min_by_key(|&slot| self.last_used[slot]).unwrap_or(0)
        };
        // Filled line by line as it is read (`fill`).
        self.lines[slot] = 0;
        if slot == self.used {
            self.used += 1;
        }
        self.last_used[slot] = self.clock;
        self.addresses[slot] = block_address;
        self.previous = self.last;
        self.last = slot;
        Some(slot)
    }

    /// Makes sure the words `first..first + count` (count >= 1) of the block
    /// in `slot` are filled. `None` if they can't be read.
    #[inline]
    fn fill(&mut self, slot: usize, first: usize, count: usize) -> Option<()> {
        let first_line = first / LINE_WORDS;
        let last_line = (first + count.max(1) - 1) / LINE_WORDS;
        let needed = (ALL_LINES >> (LINES_PER_BLOCK - 1 - last_line)) & (ALL_LINES << first_line);
        if self.lines[slot] & needed == needed {
            return Some(());
        }
        self.fill_lines(slot, needed)
    }

    /// Copies the lines in `needed` that `slot` doesn't have yet, each run
    /// of neighbouring lines at once; the whole block if the source can't
    /// copy parts of it.
    fn fill_lines(&mut self, slot: usize, needed: u8) -> Option<()> {
        let previous = crate::phase::enter(crate::phase::COPY);
        let started = crate::cycles::start(crate::cycles::LINES);
        let filled = self.fill_lines_now(slot, needed);
        crate::cycles::add(crate::cycles::LINES, started);
        crate::phase::leave(previous);
        filled
    }

    fn fill_lines_now(&mut self, slot: usize, needed: u8) -> Option<()> {
        let block_address = self.addresses[slot];
        let missing = if self.whole_blocks { ALL_LINES & !self.lines[slot] } else { needed & !self.lines[slot] };
        let mut line = 0;
        while line < LINES_PER_BLOCK {
            if missing & (1 << line) == 0 {
                line += 1;
                continue;
            }
            let mut end = line + 1;
            while end < LINES_PER_BLOCK && missing & (1 << end) != 0 {
                end += 1;
            }
            let words = &mut self.blocks[slot][line * LINE_WORDS..end * LINE_WORDS];
            if !self.source.copy_words(block_address + line as u32 * LINE_SIZE, words) {
                if !self.source.copy_block(block_address, &mut self.blocks[slot]) {
                    return None;
                }
                self.copies += 1;
                self.lines[slot] = ALL_LINES;
                return Some(());
            }
            self.line_copies += 1;
            for filled in line..end {
                self.lines[slot] |= 1 << filled;
            }
            line = end;
        }
        Some(())
    }
}

/// The address the capture reads: cache bits stripped and VRAM mirrors
/// folded into VRAM (`memory::fold_vram_mirror`). MTX Mototrax's title menu
/// reads display lists and vertices through the mirrors (0x04452758,
/// 0x042d098a): taken as they were, they were unreadable (lists cut short,
/// draws left out of a frame that was still shown: flicker) and never
/// requested from the plugin's VRAM copy.
fn capture_address(address: u32) -> u32 {
    memory::fold_vram_mirror(memory::physical(address))
}

impl<S: BlockSource> PspMemory for BlockCache<S> {
    fn read_u32(&mut self, address: u32) -> Option<u32> {
        let address = capture_address(address);
        if address % 4 != 0 {
            return None;
        }
        let block_address = address & !(BLOCK_SIZE - 1);
        let slot = self.slot_for(block_address)?;
        let index = ((address - block_address) / 4) as usize;
        self.fill(slot, index, 1)?;
        Some(self.blocks[slot][index])
    }

    /// Lends the words straight from the cached block when they all lie in
    /// one block.
    fn borrow_words(&mut self, address: u32, count: usize) -> Option<&[u32]> {
        let address = capture_address(address);
        let block_address = address & !(BLOCK_SIZE - 1);
        let first = ((address - block_address) / 4) as usize;
        if address % 4 != 0 || first + count > BLOCK_WORDS {
            return None;
        }
        let slot = self.slot_for(block_address)?;
        if count > 0 {
            self.fill(slot, first, count)?;
        }
        Some(&self.blocks[slot][first..first + count])
    }

    fn defer(&mut self, address: u32, bytes: u32) -> bool {
        let address = capture_address(address);
        address % 4 == 0 && memory::is_readable(address, bytes) && self.source.defer(address, bytes)
    }

    /// Straight from the source where it can (single words from uncached
    /// PSP RAM, ~0.15 us each, where the cache would copy a 256-byte line
    /// per sample, ~1.75 us), else through the cache.
    fn read_samples(&mut self, address: u32, positions: &[usize], out: &mut [u32]) -> Option<()> {
        let address = capture_address(address);
        let last = positions.iter().copied().max().unwrap_or(0) as u32;
        if address % 4 == 0
            && memory::is_readable(address, 4 * (last + 1))
            && self.source.read_samples(address, positions, out)
        {
            return Some(());
        }
        for (word, &position) in out.iter_mut().zip(positions) {
            *word = self.read_u32(address.wrapping_add(4 * position as u32))?;
        }
        Some(())
    }

    /// One block lookup per block touched instead of one per word.
    fn read_words(&mut self, address: u32, out: &mut [u32]) -> Option<()> {
        let mut address = capture_address(address);
        if address % 4 != 0 || !memory::is_readable(address, 4 * out.len() as u32) {
            return None;
        }
        let mut done = 0;
        while done < out.len() {
            let block_address = address & !(BLOCK_SIZE - 1);
            let first = ((address - block_address) / 4) as usize;
            let count = (BLOCK_WORDS - first).min(out.len() - done);
            let piece = &mut out[done..done + count];
            let direct = 4 * count >= DIRECT_READ_MIN_BYTES && self.cached_slot(block_address).is_none();
            if direct && self.source.copy_words(address, piece) {
                self.direct_reads += 1;
            } else {
                let slot = self.slot_for(block_address)?;
                self.fill(slot, first, count)?;
                piece.copy_from_slice(&self.blocks[slot][first..first + count]);
            }
            done += count;
            address += 4 * count as u32;
        }
        Some(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every block holds its own address in every word; blocks at
    /// `unmapped` don't exist.
    struct FakeSource {
        unmapped: u32,
    }

    impl BlockSource for FakeSource {
        fn copy_block(&mut self, address: u32, out: &mut [u32; BLOCK_WORDS]) -> bool {
            if address == self.unmapped {
                return false;
            }
            out.fill(address);
            true
        }
    }

    const RAM: u32 = 0x0880_0000;

    #[test]
    fn reads_go_through_the_cache() {
        let mut cache = BlockCache::new(FakeSource { unmapped: 0 });
        assert_eq!(cache.read_u32(RAM + 8), Some(RAM));
        assert_eq!(cache.read_u32(0x4880_0010), Some(RAM)); // mirror of the same block
        assert_eq!(cache.copies, 1);
        assert_eq!(cache.read_u32(RAM + 2), None); // unaligned
        assert_eq!(cache.read_u32(0x0100_0000), None); // not RAM or VRAM
    }

    /// MTX's title menu reads a list at 0x04452758 and vertices at
    /// 0x042d098a: VRAM mirrors, read (and requested from the plugin's
    /// VRAM copy) as VRAM itself. They counted as unreadable: a list cut
    /// short (bad_addr) and 15 draws left out of a shown frame.
    #[test]
    fn vram_mirrors_are_read_as_vram() {
        let mut cache = BlockCache::new(DirectSource);
        assert_eq!(cache.read_u32(0x0445_2758), Some(0x0405_2758));
        let mut words = [0; 4];
        cache.read_words(0x042d_0988, &mut words).unwrap();
        assert_eq!(words[0], 0x040d_0988);
        assert_eq!(cache.borrow_words(0x0465_2758, 1).map(|words| words[0]), Some(0x0405_2758));
    }

    #[test]
    fn read_words_spans_blocks() {
        let mut cache = BlockCache::new(FakeSource { unmapped: 0 });
        let mut words = [0; 4];
        cache.read_words(RAM + BLOCK_SIZE - 8, &mut words).unwrap();
        assert_eq!(words, [RAM, RAM, RAM + BLOCK_SIZE, RAM + BLOCK_SIZE]);
        assert_eq!(cache.copies, 2);
    }

    /// A source that can also copy parts of blocks (words hold their own
    /// address).
    struct DirectSource;

    impl BlockSource for DirectSource {
        fn copy_block(&mut self, address: u32, out: &mut [u32; BLOCK_WORDS]) -> bool {
            self.copy_words(address, out)
        }

        fn copy_words(&mut self, address: u32, out: &mut [u32]) -> bool {
            for (index, word) in out.iter_mut().enumerate() {
                *word = address + 4 * index as u32;
            }
            true
        }
    }

    #[test]
    fn big_reads_of_uncached_blocks_go_straight_to_the_source() {
        let mut cache = BlockCache::new(DirectSource);
        // 128 words from 256 bytes before a block end into the next block:
        // both pieces (256 bytes each) are read directly.
        let start = RAM + BLOCK_SIZE - 256;
        let mut words = vec![0; 128];
        cache.read_words(start, &mut words).unwrap();
        assert!(words.iter().enumerate().all(|(index, &word)| word == start + 4 * index as u32));
        assert_eq!((cache.copies, cache.direct_reads), (0, 2));
        // Small reads, and reads of cached blocks, use the cache (filled a
        // line at a time from a source that can copy parts of blocks).
        cache.read_words(RAM + 8, &mut [0; 4]).unwrap();
        assert_eq!((cache.copies, cache.line_copies, cache.direct_reads), (0, 1, 2));
        cache.read_words(RAM, &mut words).unwrap();
        assert_eq!(words[5], RAM + 20);
        assert_eq!(words[127], RAM + 4 * 127, "the second line, filled now");
        assert_eq!((cache.copies, cache.line_copies, cache.direct_reads), (0, 2, 2));
    }

    #[test]
    fn blocks_are_filled_a_line_at_a_time() {
        let mut cache = BlockCache::new(DirectSource);
        assert_eq!(cache.read_u32(RAM + 4), Some(RAM + 4));
        assert_eq!(cache.read_u32(RAM + 200), Some(RAM + 200), "same line");
        assert_eq!(cache.line_copies, 1);
        // Lines 2 and 3 (missing next to each other): one copy.
        assert_eq!(cache.borrow_words(RAM + 2 * LINE_SIZE + 8, 100).map(|words| words[0]), Some(RAM + 2 * LINE_SIZE + 8));
        assert_eq!(cache.line_copies, 2);
        // Line 1 between them, then everything is there.
        assert_eq!(cache.read_u32(RAM + LINE_SIZE), Some(RAM + LINE_SIZE));
        assert_eq!(cache.borrow_words(RAM, BLOCK_WORDS).map(|words| words[BLOCK_WORDS - 1]), Some(RAM + BLOCK_SIZE - 4));
        assert_eq!((cache.line_copies, cache.copies), (3, 0));
    }

    #[test]
    fn a_source_without_part_copies_fills_the_whole_block() {
        let mut cache = BlockCache::new(FakeSource { unmapped: 0 });
        assert_eq!(cache.read_u32(RAM + 3 * LINE_SIZE), Some(RAM));
        assert_eq!(cache.read_u32(RAM), Some(RAM));
        assert_eq!((cache.copies, cache.line_copies), (1, 0));
    }

    #[test]
    fn unmapped_blocks_are_unreadable_and_not_cached() {
        let mut cache = BlockCache::new(FakeSource { unmapped: RAM });
        assert_eq!(cache.read_u32(RAM), None);
        assert_eq!(cache.read_u32(RAM + BLOCK_SIZE), Some(RAM + BLOCK_SIZE));
        assert_eq!(cache.copies, 1);
    }

    #[test]
    fn alternating_blocks_stay_cached() {
        let mut cache = BlockCache::new(FakeSource { unmapped: 0 });
        for _ in 0..4 {
            assert_eq!(cache.read_u32(RAM), Some(RAM));
            assert_eq!(cache.read_u32(RAM + 5 * BLOCK_SIZE), Some(RAM + 5 * BLOCK_SIZE));
            assert_eq!(cache.read_u32(RAM + 9 * BLOCK_SIZE), Some(RAM + 9 * BLOCK_SIZE));
        }
        assert_eq!(cache.copies, 3);
    }

    #[test]
    fn least_recently_used_block_is_replaced() {
        let mut cache = BlockCache::new(FakeSource { unmapped: 0 });
        for block in 0..CACHED_BLOCKS as u32 {
            cache.read_u32(RAM + block * BLOCK_SIZE);
        }
        cache.read_u32(RAM); // block 0 is now the most recently used
        cache.read_u32(RAM + 100 * BLOCK_SIZE); // replaces block 1
        assert_eq!(cache.copies, CACHED_BLOCKS as u32 + 1);
        cache.read_u32(RAM);
        assert_eq!(cache.copies, CACHED_BLOCKS as u32 + 1);
        cache.read_u32(RAM + BLOCK_SIZE);
        assert_eq!(cache.copies, CACHED_BLOCKS as u32 + 2);
    }
}
