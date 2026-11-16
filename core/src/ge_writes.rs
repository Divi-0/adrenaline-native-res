//! Memory the GE itself writes while we follow its lists: block transfers.
//!
//! A display list can tell the GE to copy a rectangle of memory
//! (TRANSFER_START, see `display_list::block_transfer`), and later commands
//! of the same frame can read what it copied. MTX Mototrax does this for its
//! riders: before each rider a transfer copies the rider's bone matrices
//! (as BONE_MATRIX_DATA commands) from RAM into one fixed VRAM area, and the
//! rider's list then CALLs into that area. All riders share the area, so
//! the bones a rider gets depend on the transfer right before it.
//!
//! We follow the lists on our own, ahead of or behind the real GE, so PSP
//! memory doesn't hold what the GE would see at that point. Instead the
//! capture keeps its own copy of the blocks transfers wrote ([`GeWrites`])
//! and reads PSP memory through [`Overlaid`], which looks in those blocks
//! first. The copies are forgotten at the end of each frame: by then the
//! real GE has done the transfers too.

use crate::block_cache::{BLOCK_SIZE, BLOCK_WORDS};
use crate::memory::{physical, PspMemory};

/// How many 1 KiB blocks the GE may write per frame before we stop
/// following its writes. MTX's start gate: 4 transfers of ~1 KiB each.
pub const MAX_WRITTEN_BLOCKS: usize = 32;

/// The blocks written by block transfers in the current frame.
pub struct GeWrites {
    /// Physical, block-aligned addresses of the written blocks.
    addresses: [u32; MAX_WRITTEN_BLOCKS],
    blocks: [[u32; BLOCK_WORDS]; MAX_WRITTEN_BLOCKS],
    used: usize,
    /// The lowest and highest written block address: reads outside need no
    /// search (most reads: MTX writes VRAM, its lists are in RAM).
    low: u32,
    high: u32,
    /// Transfers done into the blocks this frame.
    pub transfers_done: u32,
    /// Transfers not done because they would need more blocks than are
    /// left (e.g. whole framebuffer copies) or had an unreadable source.
    pub transfers_skipped: u32,
}

impl GeWrites {
    pub const fn new() -> GeWrites {
        GeWrites {
            addresses: [0; MAX_WRITTEN_BLOCKS],
            blocks: [[0; BLOCK_WORDS]; MAX_WRITTEN_BLOCKS],
            used: 0,
            low: u32::MAX,
            high: 0,
            transfers_done: 0,
            transfers_skipped: 0,
        }
    }

    /// Forgets all writes and counts (frame end, reset).
    pub fn clear(&mut self) {
        self.used = 0;
        self.low = u32::MAX;
        self.high = 0;
        self.transfers_done = 0;
        self.transfers_skipped = 0;
    }

    pub fn is_empty(&self) -> bool {
        self.used == 0
    }

    /// Blocks not written yet.
    pub fn free_blocks(&self) -> usize {
        MAX_WRITTEN_BLOCKS - self.used
    }

    /// Whether the block at `block_address` was written.
    pub fn contains(&self, block_address: u32) -> bool {
        self.slot(block_address).is_some()
    }

    #[inline]
    fn slot(&self, block_address: u32) -> Option<usize> {
        if block_address < self.low || block_address > self.high {
            return None;
        }
        self.addresses[..self.used].iter().position(|&address| address == block_address)
    }

    /// Whether any written block lies in `bytes` bytes from `address`.
    #[inline]
    fn overlaps(&self, address: u32, bytes: u32) -> bool {
        if self.used == 0 || bytes == 0 {
            return false;
        }
        let first = address & !(BLOCK_SIZE - 1);
        let last = address.wrapping_add(bytes - 1) & !(BLOCK_SIZE - 1);
        if last < self.low || first > self.high {
            return false;
        }
        self.addresses[..self.used].iter().any(|&block| block >= first && block <= last)
    }
}

impl Default for GeWrites {
    fn default() -> Self {
        GeWrites::new()
    }
}

/// PSP memory as the GE sees it: the blocks in `writes` replace what
/// `inner` holds there. Writes go into `writes`.
pub struct Overlaid<'a, M: PspMemory> {
    pub writes: &'a mut GeWrites,
    pub inner: &'a mut M,
}

impl<M: PspMemory> Overlaid<'_, M> {
    /// The written block's slot for `block_address`, taking a new one (a
    /// copy of what `inner` holds there; zeros where it can't be read) on
    /// the first write. `None` when all blocks are taken.
    fn slot_for_write(&mut self, block_address: u32) -> Option<usize> {
        if let Some(slot) = self.writes.slot(block_address) {
            return Some(slot);
        }
        if self.writes.used == MAX_WRITTEN_BLOCKS {
            return None;
        }
        let slot = self.writes.used;
        let block = &mut self.writes.blocks[slot];
        if self.inner.read_words(block_address, block).is_none() {
            for (index, word) in block.iter_mut().enumerate() {
                *word = self.inner.read_u32(block_address + 4 * index as u32).unwrap_or(0);
            }
        }
        self.writes.addresses[slot] = block_address;
        self.writes.used += 1;
        self.writes.low = self.writes.low.min(block_address);
        self.writes.high = self.writes.high.max(block_address);
        Some(slot)
    }
}

impl<M: PspMemory> PspMemory for Overlaid<'_, M> {
    fn read_u32(&mut self, address: u32) -> Option<u32> {
        if !self.writes.is_empty() {
            let address = physical(address);
            let block_address = address & !(BLOCK_SIZE - 1);
            if address % 4 == 0 {
                if let Some(slot) = self.writes.slot(block_address) {
                    return Some(self.writes.blocks[slot][((address - block_address) / 4) as usize]);
                }
            }
        }
        self.inner.read_u32(address)
    }

    fn read_words(&mut self, address: u32, out: &mut [u32]) -> Option<()> {
        if !self.writes.overlaps(physical(address), 4 * out.len() as u32) {
            return self.inner.read_words(address, out);
        }
        // Rare (only reads near transferred data): word by word.
        for (index, word) in out.iter_mut().enumerate() {
            *word = self.read_u32(address.wrapping_add(4 * index as u32))?;
        }
        Some(())
    }

    /// Through the inner memory unless a transfer wrote near the samples.
    fn read_samples(&mut self, address: u32, positions: &[usize], out: &mut [u32]) -> Option<()> {
        let last = positions.iter().copied().max().unwrap_or(0) as u32;
        if !self.writes.overlaps(physical(address), 4 * (last + 1)) {
            return self.inner.read_samples(address, positions, out);
        }
        for (word, &position) in out.iter_mut().zip(positions) {
            *word = self.read_u32(address.wrapping_add(4 * position as u32))?;
        }
        Some(())
    }

    fn borrow_words(&mut self, address: u32, count: usize) -> Option<&[u32]> {
        let start = physical(address);
        if !self.writes.overlaps(start, 4 * count as u32) {
            return self.inner.borrow_words(address, count);
        }
        let block_address = start & !(BLOCK_SIZE - 1);
        let first = ((start - block_address) / 4) as usize;
        if start % 4 != 0 || first + count > BLOCK_WORDS {
            return None;
        }
        let slot = self.writes.slot(block_address)?;
        Some(&self.writes.blocks[slot][first..first + count])
    }

    /// Not where a transfer wrote: PSP memory doesn't hold that yet.
    fn defer(&mut self, address: u32, bytes: u32) -> bool {
        !self.writes.overlaps(physical(address), bytes) && self.inner.defer(address, bytes)
    }

    fn begin_write(&mut self, address: u32, bytes: u32) -> bool {
        let address = physical(address);
        let fits = bytes > 0 && {
            let first = address & !(BLOCK_SIZE - 1);
            let last = address.wrapping_add(bytes - 1) & !(BLOCK_SIZE - 1);
            let blocks = last.wrapping_sub(first) / BLOCK_SIZE + 1;
            // Blocks written before don't need a new slot.
            let new_blocks = || (0..blocks).filter(|&index| !self.writes.contains(first + index * BLOCK_SIZE)).count();
            last >= first && blocks as usize <= MAX_WRITTEN_BLOCKS && new_blocks() <= self.writes.free_blocks()
        };
        if fits {
            self.writes.transfers_done += 1;
        } else {
            self.writes.transfers_skipped += 1;
        }
        fits
    }

    fn write_u32(&mut self, address: u32, value: u32) -> bool {
        let address = physical(address);
        if address % 4 != 0 {
            return false;
        }
        let block_address = address & !(BLOCK_SIZE - 1);
        match self.slot_for_write(block_address) {
            Some(slot) => {
                self.writes.blocks[slot][((address - block_address) / 4) as usize] = value;
                true
            }
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::test_memory::TestMemory;

    const VRAM: u32 = 0x0415_4000;

    #[test]
    fn writes_are_read_back_and_the_rest_comes_from_inner() {
        let mut inner = TestMemory::default();
        inner.write(VRAM, &[1, 2, 3, 4]);
        let mut writes = GeWrites::new();
        let mut memory = Overlaid { writes: &mut writes, inner: &mut inner };

        assert!(memory.write_u32(VRAM + 4, 20));
        assert_eq!(memory.read_u32(VRAM), Some(1)); // copied from inner on the first write
        assert_eq!(memory.read_u32(VRAM + 4), Some(20));
        assert_eq!(memory.read_u32(0x4415_4004), Some(20)); // a mirror of the same address
        let mut words = [0; 3];
        memory.read_words(VRAM, &mut words).unwrap();
        assert_eq!(words, [1, 20, 3]);
        assert_eq!(memory.borrow_words(VRAM + 4, 2), Some(&[20, 3][..]));
        // Unreadable words of a written block read as zero.
        assert_eq!(memory.read_u32(VRAM + 64), Some(0));
        // Outside the written block: inner, unchanged.
        assert_eq!(inner.read_u32(VRAM + 4), Some(2));
    }

    #[test]
    fn writes_stop_when_all_blocks_are_taken() {
        let mut inner = TestMemory::default();
        let mut writes = GeWrites::new();
        let mut memory = Overlaid { writes: &mut writes, inner: &mut inner };
        for block in 0..MAX_WRITTEN_BLOCKS as u32 {
            assert!(memory.write_u32(VRAM + block * BLOCK_SIZE, block));
        }
        assert!(!memory.write_u32(VRAM + 100 * BLOCK_SIZE, 1));
        assert!(memory.write_u32(VRAM + 4, 1)); // an existing block still takes writes
        writes.clear();
        assert!(writes.is_empty());
    }
}
