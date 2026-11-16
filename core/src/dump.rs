//! Frame dump file format: everything needed to replay one PSP frame on a
//! PC (see `renderer/replay`).
//!
//! Layout (all numbers little-endian `u32`):
//!
//! ```text
//! header      magic "NPSPDUMP", version, snapshot_bytes, event_count,
//!             block_count, block_size, reserved          (32 bytes)
//! snapshot    Capture::save_state at the start of the frame
//! events      event_count x [type, arg0, arg1, arg2, arg3], as in the ring
//! blocks      block_count x [address, block_size bytes]: every block of
//!             PSP memory the frame read (lists, vertices, palettes,
//!             textures), copied when it was first read
//! ```
//!
//! Block addresses are physical PSP addresses; VRAM blocks keep their
//! 0x04xxxxxx address even though the Vita reads them from a copy.
//!
//! Optionally followed by the original picture, i.e. what Sony's PSP
//! emulator showed for this frame, for comparison:
//!
//! ```text
//! trailer     magic "NPSPORIG", width, height, stride (pixels),
//!             then stride x height RGBA pixels (0xAABBGGRR)
//! ```

use crate::block_cache::{BlockSource, BLOCK_WORDS};

pub const ORIGINAL_MAGIC: [u8; 8] = *b"NPSPORIG";

pub const MAGIC: [u8; 8] = *b"NPSPDUMP";
pub const VERSION: u32 = 1;
pub const HEADER_BYTES: usize = 32;
pub use crate::block_cache::BLOCK_SIZE;
/// Words per event: type plus four arguments.
pub const EVENT_WORDS: usize = 5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    pub snapshot_bytes: u32,
    pub event_count: u32,
    pub block_count: u32,
    pub block_size: u32,
}

impl Header {
    pub fn write(&self, out: &mut [u8; HEADER_BYTES]) {
        out[..8].copy_from_slice(&MAGIC);
        let words = [VERSION, self.snapshot_bytes, self.event_count, self.block_count, self.block_size, 0];
        for (index, word) in words.iter().enumerate() {
            out[8 + 4 * index..12 + 4 * index].copy_from_slice(&word.to_le_bytes());
        }
    }

    /// `None` if this isn't a dump of the supported version.
    pub fn read(bytes: &[u8]) -> Option<Header> {
        if bytes.len() < HEADER_BYTES || bytes[..8] != MAGIC {
            return None;
        }
        let word = |index: usize| u32::from_le_bytes(bytes[8 + 4 * index..12 + 4 * index].try_into().unwrap());
        if word(0) != VERSION {
            return None;
        }
        Some(Header { snapshot_bytes: word(1), event_count: word(2), block_count: word(3), block_size: word(4) })
    }
}

fn le_word(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

/// A dump in memory, read in place (no copies, no allocation), for running
/// its frame again (see `bench`).
pub struct DumpView<'a> {
    pub snapshot: &'a [u8],
    events: &'a [u8],
    blocks: &'a [u8],
}

impl<'a> DumpView<'a> {
    /// `None` if `bytes` isn't a complete dump of this version and block size.
    pub fn parse(bytes: &'a [u8]) -> Option<DumpView<'a>> {
        let header = Header::read(bytes)?;
        if header.block_size != BLOCK_SIZE {
            return None;
        }
        let snapshot_end = HEADER_BYTES.checked_add(header.snapshot_bytes as usize)?;
        let events_end = snapshot_end.checked_add(header.event_count as usize * EVENT_WORDS * 4)?;
        let blocks_end = events_end.checked_add(header.block_count as usize * BLOCK_RECORD_BYTES)?;
        Some(DumpView {
            snapshot: bytes.get(HEADER_BYTES..snapshot_end)?,
            events: bytes.get(snapshot_end..events_end)?,
            blocks: bytes.get(events_end..blocks_end)?,
        })
    }

    pub fn event_count(&self) -> usize {
        self.events.len() / (EVENT_WORDS * 4)
    }

    /// Event `index` as it was in the ring: type and four arguments.
    pub fn event(&self, index: usize) -> (u32, [u32; 4]) {
        let word = |n: usize| le_word(self.events, (index * EVENT_WORDS + n) * 4);
        (word(0), [word(1), word(2), word(3), word(4)])
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len() / BLOCK_RECORD_BYTES
    }

    fn block_address(&self, index: usize) -> u32 {
        le_word(self.blocks, index * BLOCK_RECORD_BYTES)
    }

    fn block_data(&self, index: usize) -> &'a [u8] {
        let start = index * BLOCK_RECORD_BYTES + 4;
        &self.blocks[start..start + BLOCK_SIZE as usize]
    }

    /// Fills `index` with the block numbers sorted by address, for
    /// `DumpBlocks`. Returns the filled part; `None` if `index` is too short.
    pub fn sort_blocks<'i>(&self, index: &'i mut [u32]) -> Option<&'i [u32]> {
        let index = index.get_mut(..self.block_count())?;
        for (number, slot) in index.iter_mut().enumerate() {
            *slot = number as u32;
        }
        index.sort_unstable_by_key(|&number| self.block_address(number as usize));
        Some(index)
    }
}

/// Address word + data of one block in the dump.
const BLOCK_RECORD_BYTES: usize = 4 + BLOCK_SIZE as usize;

/// A dump's memory as the source of a `BlockCache`: blocks are found by
/// binary search in an index from `DumpView::sort_blocks`.
pub struct DumpBlocks<'a> {
    pub view: &'a DumpView<'a>,
    pub sorted: &'a [u32],
}

impl DumpBlocks<'_> {
    /// Words from `address` on, across blocks (a deferred copy of a draw's
    /// memory); false if part of it isn't in the dump.
    pub fn read_range(&self, address: u32, out: &mut [u32]) -> bool {
        let mut copy = DumpBlocks { view: self.view, sorted: self.sorted };
        let mut address = address;
        let mut done = 0;
        while done < out.len() {
            let in_block = ((BLOCK_SIZE - address % BLOCK_SIZE) / 4) as usize;
            let count = in_block.min(out.len() - done);
            if !copy.copy_words(address, &mut out[done..done + count]) {
                return false;
            }
            done += count;
            address += 4 * count as u32;
        }
        true
    }
}

impl DumpBlocks<'_> {
    /// The words at `address + 4 * positions[i]` into `out[i]` (the vertex
    /// cache's samples, `draw_stream::DeferredReader::sample`): one block
    /// lookup per block touched; false if one isn't in the dump.
    pub fn read_samples(&self, address: u32, positions: &[usize], out: &mut [u32]) -> bool {
        let mut block: Option<(u32, &[u8])> = None;
        for (word, &position) in out.iter_mut().zip(positions) {
            let at = address.wrapping_add(4 * position as u32);
            let block_address = at & !(BLOCK_SIZE - 1);
            let data = match block {
                Some((known, data)) if known == block_address => data,
                _ => {
                    let Ok(found) =
                        self.sorted.binary_search_by_key(&block_address, |&number| self.view.block_address(number as usize))
                    else {
                        return false;
                    };
                    let data = self.view.block_data(self.sorted[found] as usize);
                    block = Some((block_address, data));
                    data
                }
            };
            let offset = (at - block_address) as usize;
            *word = u32::from_le_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]]);
        }
        true
    }
}

impl BlockSource for DumpBlocks<'_> {
    fn copy_block(&mut self, address: u32, out: &mut [u32; BLOCK_WORDS]) -> bool {
        let Ok(position) = self.sorted.binary_search_by_key(&address, |&number| self.view.block_address(number as usize))
        else {
            return false;
        };
        let data = self.view.block_data(self.sorted[position] as usize);
        // Whole words without per-byte bounds checks: this copy stands in for
        // the Vita's copy from PSP RAM, which the benchmark leaves out.
        for (word, bytes) in out.iter_mut().zip(data.chunks_exact(4)) {
            *word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        }
        true
    }

    /// As the Vita's live capture: main RAM and VRAM may be left to the
    /// recorder thread (`DrawStream::set_reader`; a benchmark on the Vita
    /// gives it `read_range` then, so it times what the live capture
    /// thread does). Both are in the dump as far as the frame read them.
    fn defer(&mut self, address: u32, bytes: u32) -> bool {
        crate::memory::is_readable(address, bytes)
    }

    /// As the Vita's sample loads: one block lookup per block touched.
    fn read_samples(&mut self, address: u32, positions: &[usize], out: &mut [u32]) -> bool {
        DumpBlocks::read_samples(self, address, positions, out)
    }

    /// As the Vita's direct reads: only the words asked for.
    fn copy_words(&mut self, address: u32, out: &mut [u32]) -> bool {
        let block_address = address & !(BLOCK_SIZE - 1);
        let Ok(position) =
            self.sorted.binary_search_by_key(&block_address, |&number| self.view.block_address(number as usize))
        else {
            return false;
        };
        let data = self.view.block_data(self.sorted[position] as usize);
        let first = (address - block_address) as usize;
        for (word, bytes) in out.iter_mut().zip(data[first..].chunks_exact(4)) {
            *word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_round_trip() {
        let header = Header { snapshot_bytes: 1234, event_count: 5, block_count: 77, block_size: BLOCK_SIZE };
        let mut bytes = [0u8; HEADER_BYTES];
        header.write(&mut bytes);
        assert_eq!(Header::read(&bytes), Some(header));
        bytes[0] = b'X';
        assert_eq!(Header::read(&bytes), None);
    }
}
