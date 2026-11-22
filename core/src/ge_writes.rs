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
use crate::memory::{physical, PspMemory, RAM_END, RAM_START};

/// How many 1 KiB blocks the GE may write per frame before we stop
/// following its writes. MTX's start gate: 4 transfers of ~1 KiB each.
pub const MAX_WRITTEN_BLOCKS: usize = 32;

/// How many redirects (`Redirect`) a frame may hold at once. A new one
/// drops the older ones it covers whole (`redirect_write`), so uploads
/// over one area again and again take one each only while they live.
/// God of War: 84 transfers per frame at the title, 351 in the game
/// (2026-11-22), most of them into one scratch area its next CALL runs.
pub const MAX_REDIRECTS: usize = 384;

/// A transfer taken without copying: `bytes` bytes at `start` read as the
/// same bytes at `source` (main RAM, which the GE writes never touch).
#[derive(Clone, Copy)]
struct Redirect {
    start: u32,
    end: u32,
    source: u32,
    /// Its source read the same when uploaded in an earlier frame
    /// (`SourceSample`): model data, not rewritten every frame.
    steady: bool,
}

/// Upload sources remembered across frames (`GeWrites::sources`).
pub const SOURCE_SAMPLES: usize = 512;

/// An upload's source with a sample of its words when last uploaded. God of
/// War uploads ~340 sources per frame; over a second all but one read the
/// same (models), one 2.5 KiB source changed every frame (2026-11-22). Only
/// a steady source may be drawn from directly (`PspMemory::redirect_source`,
/// zero copy): the GPU draws a frame later, when the game may have written
/// a changing one again.
#[derive(Clone, Copy)]
struct SourceSample {
    source: u32,
    bytes: u32,
    sample: u32,
}

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
    /// Contiguous transfers from RAM, oldest first: where no written block
    /// lies, the newest redirect covering an address supplies it. Written
    /// blocks always hold the current view (a redirect is copied into the
    /// blocks it covers when it is made), so they come first.
    redirects: [Redirect; MAX_REDIRECTS],
    redirect_count: usize,
    /// Byte range all redirects lie in (reads outside need no search).
    redirect_low: u32,
    redirect_high: u32,
    /// Upload sources of earlier frames (kept over `clear`), placed by
    /// address (`sample_slot`).
    sources: [SourceSample; SOURCE_SAMPLES],
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
            redirects: [Redirect { start: 0, end: 0, source: 0, steady: false }; MAX_REDIRECTS],
            redirect_count: 0,
            redirect_low: u32::MAX,
            redirect_high: 0,
            sources: [SourceSample { source: 0, bytes: 0, sample: 0 }; SOURCE_SAMPLES],
        }
    }

    /// Notes `sample` for the upload source `source` of `bytes`; true if the
    /// same source was noted before with the same size and sample.
    fn note_source(&mut self, source: u32, bytes: u32, sample: u32) -> bool {
        const PROBES: usize = 8;
        let home = ((source >> 4) as usize).wrapping_mul(0x9E37_79B1) >> 7;
        let slot = (0..PROBES)
            .map(|probe| (home + probe) % SOURCE_SAMPLES)
            .find(|&slot| self.sources[slot].source == source || self.sources[slot].source == 0)
            .unwrap_or(home % SOURCE_SAMPLES);
        let known = self.sources[slot];
        self.sources[slot] = SourceSample { source, bytes, sample };
        known.source == source && known.bytes == bytes && known.sample == sample
    }

    /// Forgets all writes and counts (frame end, reset).
    pub fn clear(&mut self) {
        self.used = 0;
        self.low = u32::MAX;
        self.high = 0;
        self.transfers_done = 0;
        self.transfers_skipped = 0;
        self.redirect_count = 0;
        self.redirect_low = u32::MAX;
        self.redirect_high = 0;
    }

    pub fn is_empty(&self) -> bool {
        self.used == 0 && self.redirect_count == 0
    }

    /// Where the word at physical `address` comes from if a redirect (and
    /// no written block) supplies it: the newest redirect covering it.
    #[inline]
    fn redirected(&self, address: u32) -> Option<u32> {
        if address < self.redirect_low || address >= self.redirect_high {
            return None;
        }
        self.redirects[..self.redirect_count]
            .iter()
            .rev()
            .find(|redirect| address >= redirect.start && address < redirect.end)
            .map(|redirect| redirect.source + (address - redirect.start))
    }

    /// Whether `bytes` bytes from `address` lie in one redirect, the
    /// newest covering each of them, and no written block: then they read
    /// as the source's bytes in one go.
    fn redirected_run(&self, address: u32, bytes: u32) -> Option<u32> {
        self.redirected_run_at(address, bytes).map(|(_, source)| source)
    }

    /// `redirected_run` with the redirect's index.
    fn redirected_run_at(&self, address: u32, bytes: u32) -> Option<(usize, u32)> {
        let end = address.checked_add(bytes)?;
        let index = self.redirects[..self.redirect_count]
            .iter()
            .rposition(|redirect| address >= redirect.start && address < redirect.end)?;
        let redirect = self.redirects[index];
        let newer_overlap = self.redirects[index + 1..self.redirect_count]
            .iter()
            .any(|newer| newer.start < end && newer.end > address);
        (end <= redirect.end && !newer_overlap && !self.blocks_overlap(address, bytes))
            .then(|| (index, redirect.source + (address - redirect.start)))
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

    /// Whether any written block or redirect lies in `bytes` bytes from
    /// `address`.
    #[inline]
    fn overlaps(&self, address: u32, bytes: u32) -> bool {
        self.blocks_overlap(address, bytes) || self.redirects_overlap(address, bytes)
    }

    #[inline]
    fn redirects_overlap(&self, address: u32, bytes: u32) -> bool {
        if self.redirect_count == 0 || bytes == 0 {
            return false;
        }
        let end = address.saturating_add(bytes);
        end > self.redirect_low && address < self.redirect_high
    }

    /// Whether any written block lies in `bytes` bytes from `address`.
    #[inline]
    fn blocks_overlap(&self, address: u32, bytes: u32) -> bool {
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
        if self.writes.redirects_overlap(block_address, BLOCK_SIZE) {
            // The block as the GE sees it now: redirected words from their
            // source.
            for index in 0..BLOCK_WORDS {
                let address = block_address + 4 * index as u32;
                let from = self.writes.redirected(address).unwrap_or(address);
                self.writes.blocks[slot][index] = self.inner.read_u32(from).unwrap_or(0);
            }
        } else {
            let block = &mut self.writes.blocks[slot];
            if self.inner.read_words(block_address, block).is_none() {
                for (index, word) in block.iter_mut().enumerate() {
                    *word = self.inner.read_u32(block_address + 4 * index as u32).unwrap_or(0);
                }
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
            if let Some(source) = self.writes.redirected(address) {
                // Unreadable source words read as zero, as a copy makes them
                // (`display_list::copy_bytes`).
                return Some(self.inner.read_u32(source).unwrap_or(0));
            }
        }
        self.inner.read_u32(address)
    }

    fn read_words(&mut self, address: u32, out: &mut [u32]) -> Option<()> {
        let bytes = 4 * out.len() as u32;
        if !self.writes.overlaps(physical(address), bytes) {
            return self.inner.read_words(address, out);
        }
        if let Some(source) = self.writes.redirected_run(physical(address), bytes) {
            if self.inner.read_words(source, out).is_some() {
                return Some(());
            }
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
        if let Some(source) = self.writes.redirected_run(physical(address), 4 * (last + 1)) {
            if self.inner.read_samples(source, positions, out).is_some() {
                return Some(());
            }
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
        if let Some(source) = self.writes.redirected_run(start, 4 * count as u32) {
            return self.inner.borrow_words(source, count);
        }
        let block_address = start & !(BLOCK_SIZE - 1);
        let first = ((start - block_address) / 4) as usize;
        if start % 4 != 0 || first + count > BLOCK_WORDS {
            return None;
        }
        let slot = self.writes.slot(block_address)?;
        Some(&self.writes.blocks[slot][first..first + count])
    }

    /// Only from a steady source (`SourceSample`).
    fn redirect_source(&mut self, address: u32, bytes: u32) -> Option<u32> {
        if bytes == 0 {
            return None;
        }
        let (index, source) = self.writes.redirected_run_at(physical(address), bytes)?;
        self.writes.redirects[index].steady.then_some(source)
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

    /// Taken when the source is main RAM no GE write touched, both sides
    /// are word aligned and don't overlap, and a redirect is left: the
    /// blocks written before that the destination covers get its words
    /// now (blocks always hold the current view).
    fn redirect_write(&mut self, destination: u32, source: u32, bytes: u32) -> bool {
        let (destination, source) = (physical(destination), physical(source));
        let (Some(destination_end), Some(source_end)) = (destination.checked_add(bytes), source.checked_add(bytes)) else {
            return false;
        };
        let takes = bytes > 0
            && (destination | source | bytes) % 4 == 0
            && source >= RAM_START
            && source_end <= RAM_END
            && (source_end <= destination || destination_end <= source)
            && !self.writes.overlaps(source, bytes);
        if !takes {
            return false;
        }
        // Older redirects inside the destination are never read again (the
        // newest covering an address supplies it): their places are freed.
        let covered = |redirect: &Redirect| redirect.start >= destination && redirect.end <= destination_end;
        let live = &self.writes.redirects[..self.writes.redirect_count];
        if self.writes.redirect_count - live.iter().filter(|redirect| covered(redirect)).count() >= MAX_REDIRECTS {
            return false;
        }
        let mut kept = 0;
        for index in 0..self.writes.redirect_count {
            let redirect = self.writes.redirects[index];
            if !covered(&redirect) {
                self.writes.redirects[kept] = redirect;
                kept += 1;
            }
        }
        self.writes.redirect_count = kept;
        if self.writes.blocks_overlap(destination, bytes) {
            for slot in 0..self.writes.used {
                let block_address = self.writes.addresses[slot];
                let first = block_address.max(destination);
                let end = (block_address + BLOCK_SIZE).min(destination_end);
                let mut address = first;
                while address < end {
                    let word = self.inner.read_u32(source + (address - destination)).unwrap_or(0);
                    self.writes.blocks[slot][((address - block_address) / 4) as usize] = word;
                    address += 4;
                }
            }
        }
        // Eight words spread over the source, for `note_source`.
        let mut sample = 0x811C_9DC5u32;
        for k in 0..8 {
            let word = self.inner.read_u32(source + ((bytes / 4) * k / 8) * 4).unwrap_or(0);
            sample = (sample ^ word).wrapping_mul(0x0100_0193);
        }
        let writes = &mut *self.writes;
        let steady = writes.note_source(source, bytes, sample);
        writes.redirects[writes.redirect_count] = Redirect { start: destination, end: destination_end, source, steady };
        writes.redirect_count += 1;
        writes.redirect_low = writes.redirect_low.min(destination);
        writes.redirect_high = writes.redirect_high.max(destination_end);
        writes.transfers_done += 1;
        true
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
    const RAM_A: u32 = 0x0970_0000;
    const RAM_B: u32 = 0x0972_0000;

    fn ram(inner: &mut TestMemory) {
        inner.write(RAM_A, &(0..512).map(|word| 0xA000_0000 | word).collect::<Vec<_>>());
        inner.write(RAM_B, &(0..512).map(|word| 0xB000_0000 | word).collect::<Vec<_>>());
        inner.write(VRAM, &[7; 512]);
    }

    #[test]
    fn a_redirect_reads_its_source_and_the_newest_one_wins() {
        let mut inner = TestMemory::default();
        ram(&mut inner);
        let mut writes = GeWrites::new();
        let mut memory = Overlaid { writes: &mut writes, inner: &mut inner };
        assert!(memory.redirect_write(VRAM, RAM_A, 1024));
        assert!(memory.redirect_write(VRAM + 512, RAM_B, 1024));
        assert_eq!(memory.read_u32(VRAM), Some(0xA000_0000));
        assert_eq!(memory.read_u32(VRAM + 512), Some(0xB000_0000)); // newer
        assert_eq!(memory.read_u32(VRAM + 1536 - 4), Some(0xB000_00FF));
        assert_eq!(memory.read_u32(VRAM + 1536), Some(7)); // past both: inner
        assert_eq!(memory.read_u32(0x4415_4000 + 8), Some(0xA000_0002)); // uncached alias
        // One run inside one redirect, and one across both, read alike.
        let mut words = [0; 4];
        memory.read_words(VRAM + 16, &mut words).unwrap();
        assert_eq!(words, [0xA000_0004, 0xA000_0005, 0xA000_0006, 0xA000_0007]);
        memory.read_words(VRAM + 504, &mut words).unwrap();
        assert_eq!(words, [0xA000_007E, 0xA000_007F, 0xB000_0000, 0xB000_0001]);
        assert!(!memory.defer(VRAM, 4)); // not in PSP memory yet
        // Unreadable source words read as zero, as a copy makes them.
        assert!(memory.redirect_write(VRAM + 0x2000, 0x0990_0000, 8));
        assert_eq!(memory.read_u32(VRAM + 0x2000), Some(0));
        memory.read_words(VRAM + 0x2000, &mut words[..2]).unwrap();
        assert_eq!(words[..2], [0, 0]);
        assert_eq!(memory.writes.transfers_done, 3);
    }

    #[test]
    fn redirects_and_copied_blocks_keep_the_order_of_the_transfers() {
        let mut inner = TestMemory::default();
        ram(&mut inner);
        let mut writes = GeWrites::new();
        let mut memory = Overlaid { writes: &mut writes, inner: &mut inner };
        // A copy into the block, then a redirect over part of it: the
        // redirect's words replace the copy's.
        assert!(memory.write_u32(VRAM, 1));
        assert!(memory.write_u32(VRAM + 8, 3));
        assert!(memory.redirect_write(VRAM + 4, RAM_A, 4));
        assert_eq!(memory.read_u32(VRAM), Some(1));
        assert_eq!(memory.read_u32(VRAM + 4), Some(0xA000_0000));
        assert_eq!(memory.read_u32(VRAM + 8), Some(3));
        // A redirect, then a copy into part of it: the copy wins there, the
        // rest of the block still reads the redirect.
        let far = VRAM + 0x1000;
        assert!(memory.redirect_write(far, RAM_B, 64));
        assert!(memory.write_u32(far + 4, 99));
        assert_eq!(memory.read_u32(far), Some(0xB000_0000));
        assert_eq!(memory.read_u32(far + 4), Some(99));
        assert_eq!(memory.read_u32(far + 8), Some(0xB000_0002));
    }

    #[test]
    fn a_redirect_is_refused_when_its_source_isnt_plain_ram() {
        let mut inner = TestMemory::default();
        ram(&mut inner);
        let mut writes = GeWrites::new();
        let mut memory = Overlaid { writes: &mut writes, inner: &mut inner };
        assert!(!memory.redirect_write(VRAM, VRAM + 0x2000, 64)); // VRAM source
        assert!(!memory.redirect_write(VRAM + 2, RAM_A, 64)); // unaligned
        assert!(memory.write_u32(RAM_A, 5)); // a GE write into the source
        assert!(!memory.redirect_write(VRAM, RAM_A, 64));
        assert!(!memory.redirect_write(RAM_B, RAM_B + 32, 64)); // overlapping sides
        assert_eq!(memory.writes.transfers_done, 0);
    }

    #[test]
    fn only_steady_upload_sources_are_offered_for_direct_reads() {
        // God of War uploads its models' data every frame from the same
        // RAM; one source is rewritten every frame instead.
        let (model, changing) = (0x08F5_0000, 0x09E3_D1E0);
        let mut inner = TestMemory::default();
        inner.write(model, &[7; 64]);
        inner.write(changing, &[1; 64]);
        let mut writes = GeWrites::new();
        for frame in 0..3u32 {
            inner.write(changing + 4 * 32, &[frame]);
            let mut memory = Overlaid { writes: &mut writes, inner: &mut inner };
            assert!(memory.redirect_write(VRAM, model, 256));
            assert!(memory.redirect_write(VRAM + 0x1000, changing, 256));
            let first_time = frame == 0;
            assert_eq!(memory.redirect_source(VRAM + 16, 32), (!first_time).then_some(model + 16), "frame {frame}");
            assert_eq!(memory.redirect_source(VRAM + 0x1000, 32), None, "rewritten every frame");
            memory.writes.clear();
        }
    }
}
