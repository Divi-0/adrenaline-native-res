//! The PSP plugin's decisions (renderer/psp), as plain functions so they
//! are tested here on the PC: which VRAM it copies and when, the GE sync
//! wait, the copy-method choice, the GE probe's lists, and the line in
//! Adrenaline's plugins.txt that loads it (written by the Vita side). The
//! plugin only adds the memory accesses (protocol: [`crate::psp_link`]).

use crate::vram_requests::{VramRange, MAX_RANGES, VRAM_SIZE};

/// The change check compares this many words of every block of this size
/// with the copy. Reads of VRAM and of the (uncached) copy are slow, ~3 us
/// per block, and the check of the recent ranges runs in the game's thread.
/// A reload rewrites whole blocks, so 2 words find it; the rolling refresh
/// catches the rest.
pub const CHECK_BLOCK_BYTES: u32 = 1024;
pub const CHECK_WORDS_PER_BLOCK: u32 = 2;
/// Reading VRAM is slow in the emulator (~30 us per KiB): per frame the copy
/// thread refreshes at most this many bytes, cycling through all ranges,
/// for changes the samples miss.
pub const REFRESH_BYTES_PER_FRAME: u32 = 4 * 1024;
/// At a list's enqueue the recent ranges are copied whole, not checked by
/// samples, while they fit this many bytes (the Vita side's default for
/// `VramRequestTable::whole_bytes`; 0 = samples only, as before 2026-11-17).
/// MTX keeps the vertices of its menu's 20 background tiles in VRAM and
/// animates their positions during the rider <-> bike wipe; the samples (2
/// words per KiB) hit the tiles' constant UVs and colours, so the copy kept
/// old positions and the Vita drew the tiles where the game had moved them
/// away from, over leftover texture memory (photo-like tiles for ~8 frames,
/// until the rolling refresh came by). The plugin's memcpy copies VRAM at
/// ~6.8 us per KiB (MTX menu: ~116 KiB, ~0.9 ms per enqueue).
pub const WHOLE_AT_ENQUEUE_BYTES: u32 = 256 * 1024;

/// What a VRAM copy of `size` bytes at `offset` really copies: the offset
/// aligned down to a word, the size rounded up to one; None when that is
/// empty or leaves VRAM.
pub fn copy_span(offset: u32, size: u32) -> Option<(u32, u32)> {
    let offset = offset & !3;
    let size = size.checked_add(3)? & !3;
    (offset < VRAM_SIZE && size != 0 && size <= VRAM_SIZE - offset).then_some((offset, size))
}

/// Copies are made in pieces of at most this many bytes, each with
/// interrupts off (`copy_pieces`). A copy the game's thread could interrupt
/// held the VRAM words its loop had just read: the game moved data in VRAM,
/// its enqueue-time copy wrote the new bytes, and when the copy thread went
/// on (the game waiting for the Vita's walk) it stored the old ones over
/// them. MHF (2026-11-20): one 64-byte line of the hunter's vertices with
/// old bytes until the next enqueue, a one-frame streak (the bytes at
/// 0x040ED680 were those the game had moved to 0x040EB680, or zeros).
/// ~7 us per KiB with memcpy: 4 KiB keeps interrupts off for ~30 us.
pub const COPY_PIECE_BYTES: u32 = 4 * 1024;

/// `size` bytes at `offset` as pieces (offset, size) of at most
/// [`COPY_PIECE_BYTES`], in order, covering them exactly.
pub fn copy_pieces(offset: u32, size: u32) -> impl Iterator<Item = (u32, u32)> {
    let end = offset.saturating_add(size);
    (offset..end).step_by(COPY_PIECE_BYTES as usize).map(move |at| (at, (end - at).min(COPY_PIECE_BYTES)))
}

/// The PSP's data cache line: cache maintenance works on whole lines.
pub const CACHE_LINE_BYTES: u32 = 64;

/// The whole cache lines covering `size` bytes at `address` (start and size
/// multiples of [`CACHE_LINE_BYTES`]), for the cached copy methods' cache
/// maintenance before they read VRAM through the cache.
///
/// sceKernelDcacheInvalidateRange refuses a range that is not line-aligned
/// (as PPSSPP has it: SCE_KERNEL_ERROR_CACHE_ALIGNMENT) and does nothing,
/// and the copies are only word-aligned (`copy_span`): lines of VRAM cached
/// by an earlier copy could stay and be read again after the game rewrote
/// VRAM. So the plugin writes the covering lines back and invalidates them
/// (sceKernelDcacheWritebackInvalidateRange; a plain invalidate of the
/// wider range would drop the game's own unwritten bytes next to ours), and
/// writes the copy's lines back whole. A precaution: suspected for MHF's
/// one-line streaks (2026-11-20), but an A/B (debug `copyverify 2`, 2 x 150
/// s each) showed no difference; their cause was `copy_pieces`.
pub fn cache_line_span(address: u32, size: u32) -> (u32, u32) {
    let start = address & !(CACHE_LINE_BYTES - 1);
    let end = address.saturating_add(size).saturating_add(CACHE_LINE_BYTES - 1) & !(CACHE_LINE_BYTES - 1);
    (start, end - start)
}

/// The word (index into the block) of the change check's `sample`: spread
/// over the block, at different columns (texture rows are often 64-512
/// bytes).
pub const fn sample_word(sample: u32) -> u32 {
    sample * (CHECK_BLOCK_BYTES / 4 / CHECK_WORDS_PER_BLOCK) + (sample * 9) % 32
}

/// The ranges the plugin has copied in full and keeps up to date.
#[derive(Clone, Copy, Debug)]
pub struct KnownRanges {
    pub ranges: [VramRange; MAX_RANGES],
    pub count: usize,
}

impl Default for KnownRanges {
    fn default() -> Self {
        Self::new()
    }
}

impl KnownRanges {
    pub const fn new() -> Self {
        KnownRanges { ranges: [VramRange { offset: 0, size: 0 }; MAX_RANGES], count: 0 }
    }

    pub fn ranges(&self) -> &[VramRange] {
        &self.ranges[..self.count.min(MAX_RANGES)]
    }

    /// Whether exactly this range is known (then it needn't be copied in
    /// full when the table changes).
    pub fn is_known(&self, offset: u32, size: u32) -> bool {
        self.ranges().iter().any(|range| range.offset == offset && range.size == size)
    }

    /// Whether the block at `block` overlaps one of the ranges in `mask`.
    pub fn in_ranges(&self, block: u32, mask: u32) -> bool {
        self.ranges().iter().enumerate().any(|(i, range)| {
            mask & (1 << i) != 0
                && u64::from(block) + u64::from(CHECK_BLOCK_BYTES) > u64::from(range.offset)
                && u64::from(block) < u64::from(range.offset) + u64::from(range.size)
        })
    }

    /// At a list's enqueue: the recent ranges (bit set in `recent_mask`)
    /// copied whole with `copy(offset, size)`, in table order, those that
    /// still fit `budget` bytes. Returns the mask of the ranges copied (the
    /// change check skips them).
    pub fn copy_recent_whole(&self, recent_mask: u32, budget: u32, mut copy: impl FnMut(u32, u32)) -> u32 {
        let (mut done, mut left) = (0u32, budget);
        for (i, range) in self.ranges().iter().enumerate() {
            if (recent_mask >> i) & 1 == 0 {
                continue;
            }
            let Some((offset, size)) = copy_span(range.offset, range.size) else {
                continue;
            };
            if size > left {
                continue;
            }
            copy(offset, size);
            left -= size;
            done |= 1 << i;
        }
        done
    }

    /// The change check over the recent ranges (`recent`, bit set in
    /// `recent_mask`) or the others; the others skip blocks a recent range
    /// also covers (those may hold what the frame on its way to the Vita was
    /// drawn with). `differs(block, word)`: VRAM and the copy differ at that
    /// word of the block; `copy(block)` copies the block again. Returns the
    /// blocks copied.
    pub fn changed_blocks(
        &self,
        recent_mask: u32,
        recent: bool,
        mut differs: impl FnMut(u32, u32) -> bool,
        mut copy: impl FnMut(u32),
    ) -> u32 {
        let mut changed = 0;
        for (i, range) in self.ranges().iter().enumerate() {
            if ((recent_mask >> i) & 1 != 0) != recent {
                continue;
            }
            let end = range.offset.saturating_add(range.size).min(VRAM_SIZE);
            let mut block = range.offset & !(CHECK_BLOCK_BYTES - 1);
            while block < end {
                if !(!recent && self.in_ranges(block, recent_mask))
                    && (0..CHECK_WORDS_PER_BLOCK).any(|sample| differs(block, sample_word(sample)))
                {
                    copy(block);
                    changed += 1;
                }
                block += CHECK_BLOCK_BYTES;
            }
        }
        changed
    }
}

/// Where the rolling refresh is.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Refresh {
    range: usize,
    offset: u32,
}

impl Refresh {
    pub const fn new() -> Self {
        Refresh { range: 0, offset: 0 }
    }

    /// The next `budget` bytes of the ranges, as `copy(offset, size)` calls,
    /// going on to the next range (and round) as each is done.
    pub fn step(&mut self, known: &KnownRanges, budget: u32, mut copy: impl FnMut(u32, u32)) {
        let ranges = known.ranges();
        if ranges.is_empty() {
            return;
        }
        let mut budget = budget;
        let mut steps = 0;
        while budget > 0 && steps < ranges.len() + 1 {
            if self.range >= ranges.len() {
                self.range = 0;
            }
            // `get`, not indexing: the PSP plugin links no panic path.
            let Some(&range) = ranges.get(self.range) else { return };
            let chunk = range.size.saturating_sub(self.offset).min(budget);
            if chunk > 0 {
                copy(range.offset.wrapping_add(self.offset), chunk);
            }
            budget -= chunk;
            self.offset += chunk;
            if self.offset >= range.size {
                self.range += 1;
                self.offset = 0;
            }
            steps += 1;
        }
    }
}

/// The GE sync wait's state: after a wait that timed out the plugin doesn't
/// wait again until the Vita side has handled the events written up to the
/// timeout, so a stuck Vita side can't stall the game. (Waiting only until
/// it was caught up at the moment of a sync never happened while it ran a
/// little behind: the sync then stayed off for good.)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SyncStall {
    stalled: bool,
    /// `write_count` when the last wait timed out.
    stalled_at: u32,
}

impl SyncStall {
    pub const fn new() -> Self {
        SyncStall { stalled: false, stalled_at: 0 }
    }

    /// Whether the game should wait now: the Vita side has `handled`
    /// events, `written` are in the ring (both counts wrap).
    pub fn should_wait(&mut self, handled: u32, written: u32) -> bool {
        if self.stalled && reached(handled, self.stalled_at) {
            self.stalled = false;
        }
        !self.stalled && !reached(handled, written)
    }

    /// The wait gave up with `written` events in the ring.
    pub fn timed_out(&mut self, written: u32) {
        self.stalled = true;
        self.stalled_at = written;
    }
}

/// Whether the count `handled` has reached `target` (wrapping counts).
pub fn reached(handled: u32, target: u32) -> bool {
    handled.wrapping_sub(target) as i32 >= 0
}

/// The copy method to use after the probe: the fastest of those that copied
/// correctly (`ok`: bit method - 1; `us[method - 1]`), not the VFPU one
/// (it works only in threads that have the VFPU), else word copies.
pub fn best_method(us: &[u32; 8], ok: u32) -> u32 {
    use crate::psp_link::copy_method::{METHODS, VFPU, WORDS};
    let mut best = WORDS;
    let mut best_us = u32::MAX;
    for method in 1..=METHODS {
        let index = (method - 1) as usize;
        if ok & (1 << index) != 0 && method != VFPU && us[index] < best_us {
            best_us = us[index];
            best = method;
        }
    }
    best
}

/// GE commands of the GE block-transfer probe's lists.
const GE_TRANSFER_SOURCE: u32 = 0xB2;
const GE_TRANSFER_SOURCE_WIDTH: u32 = 0xB3;
const GE_TRANSFER_DESTINATION: u32 = 0xB4;
const GE_TRANSFER_DESTINATION_WIDTH: u32 = 0xB5;
const GE_TRANSFER_SOURCE_POSITION: u32 = 0xEB;
const GE_TRANSFER_DESTINATION_POSITION: u32 = 0xEC;
const GE_TRANSFER_SIZE: u32 = 0xEE;
const GE_TRANSFER_START: u32 = 0xEA;
const GE_FINISH: u32 = 0x0F;
const GE_END: u32 = 0x0C;
/// Words of the longest probe list.
pub const GE_PROBE_LIST_WORDS: usize = 10;

/// The GE probe's list for `mode` (1 = an empty list, else a block transfer
/// of `bytes` from `from` to `to`, physical addresses, in rows of 256 32-bit
/// pixels): written into `list`, returns its length in words.
pub fn ge_probe_list(mode: u32, from: u32, to: u32, bytes: u32, list: &mut [u32; GE_PROBE_LIST_WORDS]) -> usize {
    let mut n = 0;
    let mut push = |word: u32| {
        list[n] = word;
        n += 1;
    };
    if mode != 1 {
        let rows = bytes / 1024;
        push(GE_TRANSFER_SOURCE << 24 | (from & 0xFF_FFF0));
        push(GE_TRANSFER_SOURCE_WIDTH << 24 | ((from >> 8) & 0xFF_0000) | 256);
        push(GE_TRANSFER_DESTINATION << 24 | (to & 0xFF_FFF0));
        push(GE_TRANSFER_DESTINATION_WIDTH << 24 | ((to >> 8) & 0xFF_0000) | 256);
        push(GE_TRANSFER_SOURCE_POSITION << 24);
        push(GE_TRANSFER_DESTINATION_POSITION << 24);
        push(GE_TRANSFER_SIZE << 24 | (rows.wrapping_sub(1) & 0x3FF) << 10 | (256 - 1));
        push(GE_TRANSFER_START << 24 | 1);
    }
    push(GE_FINISH << 24);
    push(GE_END << 24);
    n
}

/// The line in Adrenaline's plugins.txt (ms0:/seplugins/plugins.txt) that
/// loads the PSP plugin with every PSP game.
pub const PLUGINS_TXT_LINE: &[u8] = b"game, ms0:/seplugins/nativeres.prx, on";
const PLUGINS_TXT_COMMENT: &[u8] = b"# NativeRes capture plugin, every PSP game (added by nativeres.suprx)";
/// The plugin's file name, as found in any line about it.
const PLUGIN_FILE: &[u8] = b"nativeres.prx";

/// plugins.txt with our line added: None when it already has a line about
/// the plugin (on or off: the user's choice stays) or `out` is too small,
/// else the new file's length in `out`.
pub fn plugins_txt_with_ours(existing: &[u8], out: &mut [u8]) -> Option<usize> {
    let mentions = existing
        .split(|&byte| byte == b'\n')
        .any(|line| line.windows(PLUGIN_FILE.len()).any(|window| window.eq_ignore_ascii_case(PLUGIN_FILE)));
    if mentions {
        return None;
    }
    let mut length = 0;
    let mut push = |bytes: &[u8]| -> Option<()> {
        out.get_mut(length..length + bytes.len())?.copy_from_slice(bytes);
        length += bytes.len();
        Some(())
    };
    push(existing)?;
    let crlf = existing.windows(2).any(|pair| pair == b"\r\n");
    let newline: &[u8] = if crlf { b"\r\n" } else { b"\n" };
    if !existing.is_empty() && existing.last() != Some(&b'\n') {
        push(newline)?;
    }
    push(PLUGINS_TXT_COMMENT)?;
    push(newline)?;
    push(PLUGINS_TXT_LINE)?;
    push(newline)?;
    Some(length)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::psp_link::copy_method;

    /// MHF's streak (2026-11-20), modelled: a copy loop reads 64 bytes into
    /// registers, then stores them. `game_at` is the point where the game's
    /// thread runs: it moves the model data in VRAM and its enqueue-time copy
    /// writes the new bytes (whole, as `refresh_before_walk`). The game can
    /// only run between whole pieces when each runs with interrupts off
    /// (`copy_pieces`), but anywhere in one interruptible copy, including
    /// between a chunk's load and its store.
    fn copy_with_the_game_running(range: (u32, u32), game_at: (u32, bool), pieces: bool) -> (Vec<u8>, Vec<u8>) {
        const CHUNK: u32 = 64;
        let (offset, size) = range;
        let mut vram: Vec<u8> = (0..offset + size).map(|i| (i % 251) as u8).collect();
        let mut copy = vec![0u8; vram.len()];
        // The game: data moved down by 0x2000 (or anything new), then its own whole copy.
        let mut game_ran = false;
        let mut run_game = |vram: &mut Vec<u8>, copy: &mut Vec<u8>| {
            for byte in vram[offset as usize..].iter_mut() {
                *byte = byte.wrapping_add(0x55);
            }
            copy[offset as usize..].copy_from_slice(&vram[offset as usize..]);
        };
        let spans: Vec<(u32, u32)> = if pieces { copy_pieces(offset, size).collect() } else { vec![(offset, size)] };
        for (piece, piece_size) in spans {
            let mut at = piece;
            // In pieces the game can only run before a piece starts.
            if pieces && !game_ran && piece + piece_size > game_at.0 {
                run_game(&mut vram, &mut copy);
                game_ran = true;
            }
            while at < piece + piece_size {
                let held: Vec<u8> = vram[at as usize..(at + CHUNK) as usize].to_vec(); // load
                if !pieces && !game_ran && at >= game_at.0 && game_at.1 {
                    // Interruptible: between the load and the store.
                    run_game(&mut vram, &mut copy);
                    game_ran = true;
                }
                copy[at as usize..(at + CHUNK) as usize].copy_from_slice(&held); // store
                at += CHUNK;
                if !pieces && !game_ran && at > game_at.0 && !game_at.1 {
                    // Interruptible: between two chunks.
                    run_game(&mut vram, &mut copy);
                    game_ran = true;
                }
            }
        }
        (vram, copy)
    }

    #[test]
    fn the_game_never_runs_inside_a_copy_piece() {
        let range = (0x2000, 4 * COPY_PIECE_BYTES);
        let differing_lines = |(vram, copy): (Vec<u8>, Vec<u8>)| {
            (range.0..range.0 + range.1).step_by(64).filter(|&at| vram[at as usize..at as usize + 64] != copy[at as usize..at as usize + 64]).count()
        };
        // One interruptible copy, the game in the middle of a chunk: the
        // resumed copy stores one line of old bytes over the game's copy.
        let middle = range.0 + 0x1680;
        assert_eq!(differing_lines(copy_with_the_game_running(range, (middle, true), false)), 1, "the glitch: one stale line");
        assert_eq!(differing_lines(copy_with_the_game_running(range, (middle, false), false)), 0, "between chunks it is harmless");
        // In pieces with interrupts off the game runs only between pieces,
        // and every later piece reads the new bytes.
        for game_at in [range.0, middle, range.0 + COPY_PIECE_BYTES, range.0 + range.1 - 64] {
            for between_load_and_store in [true, false] {
                let result = copy_with_the_game_running(range, (game_at, between_load_and_store), true);
                assert_eq!(differing_lines(result), 0, "game at {game_at:#x}");
            }
        }
    }

    #[test]
    fn copies_go_in_pieces_that_cover_the_range_exactly() {
        // MHF's glitch: a range over the hunter's vertices, copied whole.
        let pieces: Vec<(u32, u32)> = copy_pieces(0x0C_C1CC, 0x2_1600).collect();
        assert_eq!(pieces.first(), Some(&(0x0C_C1CC, COPY_PIECE_BYTES)));
        let mut at = 0x0C_C1CC;
        for &(offset, size) in &pieces {
            assert_eq!(offset, at, "in order, no gap");
            assert!(size > 0 && size <= COPY_PIECE_BYTES);
            at += size;
        }
        assert_eq!(at, 0x0C_C1CC + 0x2_1600, "the whole range");
        assert_eq!(copy_pieces(0x100, 1024).collect::<Vec<_>>(), [(0x100, 1024)], "a block is one piece");
        assert_eq!(copy_pieces(0x100, 0).count(), 0);
        assert_eq!(copy_pieces(0, 2 * COPY_PIECE_BYTES).count(), 2);
    }

    #[test]
    fn cache_maintenance_covers_whole_lines() {
        // MHF's glitch: a word-aligned copy over the hunter's vertices; the
        // stale line 0x040ED680..0x040ED6C0 lies inside, the ends are partial.
        let (start, size) = cache_line_span(0x040E_D664, 0x9A);
        assert_eq!((start, size), (0x040E_D640, 0xC0));
        assert!(start <= 0x040E_D680 && 0x040E_D6C0 <= start + size);
        for (address, size) in [(0x0400_0000, 64), (0x0400_0004, 4), (0x0400_003C, 8), (0x0400_0040, 0x10000), (0x041F_FFFC, 4)] {
            let (start, covered) = cache_line_span(address, size);
            assert_eq!((start % CACHE_LINE_BYTES, covered % CACHE_LINE_BYTES), (0, 0), "{address:#x}+{size}");
            assert!(start <= address && address + size <= start + covered, "{address:#x}+{size}");
            assert!(covered < size + 2 * CACHE_LINE_BYTES, "{address:#x}+{size}: no more than the partial lines");
        }
        assert_eq!(cache_line_span(0x0400_0040, 0x10000), (0x0400_0040, 0x10000), "aligned stays as it is");
    }

    fn known(ranges: &[(u32, u32)]) -> KnownRanges {
        let mut known = KnownRanges::new();
        for (i, &(offset, size)) in ranges.iter().enumerate() {
            known.ranges[i] = VramRange { offset, size };
        }
        known.count = ranges.len();
        known
    }

    #[test]
    fn copy_span_rounds_and_stays_in_vram() {
        assert_eq!(copy_span(0x1002, 5), Some((0x1000, 8)));
        assert_eq!(copy_span(0, 0), None);
        assert_eq!(copy_span(VRAM_SIZE, 4), None);
        assert_eq!(copy_span(VRAM_SIZE - 4, 4), Some((VRAM_SIZE - 4, 4)));
        assert_eq!(copy_span(VRAM_SIZE - 4, 8), None);
        // Sizes near 4 GiB don't wrap into small ones (the Vita writes the
        // table; a bad entry must not copy anything).
        assert_eq!(copy_span(0, u32::MAX), None);
        assert_eq!(copy_span(0, u32::MAX - 3), None);
    }

    #[test]
    fn samples_are_the_c_plugins_words() {
        // main.c: k * (1024 / 4 / 2) + (k * 9) % 32
        assert_eq!(sample_word(0), 0);
        assert_eq!(sample_word(1), 128 + 9);
        assert!(sample_word(CHECK_WORDS_PER_BLOCK - 1) < CHECK_BLOCK_BYTES / 4);
    }

    #[test]
    fn known_ranges_and_overlap() {
        let known = known(&[(0x1000, 0x800), (0x4000, 0x400)]);
        assert!(known.is_known(0x1000, 0x800));
        assert!(!known.is_known(0x1000, 0x400));
        // Blocks touching range 0 (bit 0).
        assert!(!known.in_ranges(0x0C00, 1)); // 0x0C00..0x1000 ends where the range starts
        assert!(known.in_ranges(0x1000, 1));
        assert!(known.in_ranges(0x1400, 1));
        assert!(!known.in_ranges(0x1800, 1)); // range ends at 0x1800
        assert!(!known.in_ranges(0x1400, 2)); // not in mask
        // A range near the top of the address space doesn't wrap.
        let high = self::known(&[(u32::MAX - 0x100, 0x200)]);
        assert!(!high.in_ranges(0, 1));
    }

    #[test]
    fn changed_blocks_recent_and_others() {
        // Range 0 recent (0x0000..0x1000), range 1 not (0x0800..0x2000,
        // overlapping range 0's second half).
        let known = known(&[(0x0000, 0x1000), (0x0800, 0x1800)]);
        let all_differ = |_block: u32, _word: u32| true;
        let mut copied = [0u32; 16];
        let mut n = 0;
        let changed = known.changed_blocks(0b01, true, all_differ, |block| {
            copied[n] = block;
            n += 1;
        });
        assert_eq!(changed, 4);
        assert_eq!(&copied[..n], &[0x000, 0x400, 0x800, 0xC00]);
        // The others: range 1's blocks, minus those range 0 covers.
        n = 0;
        let changed = known.changed_blocks(0b01, false, all_differ, |block| {
            copied[n] = block;
            n += 1;
        });
        assert_eq!(changed, 4);
        assert_eq!(&copied[..n], &[0x1000, 0x1400, 0x1800, 0x1C00]);
        // Nothing differs: nothing copied.
        assert_eq!(known.changed_blocks(0b01, true, |_, _| false, |_| panic!()), 0);
        // Only the second sample of block 0x400 differs.
        let changed = known.changed_blocks(0b01, true, |block, word| block == 0x400 && word == sample_word(1), |block| {
            assert_eq!(block, 0x400)
        });
        assert_eq!(changed, 1);
    }

    /// MTX rider <-> bike select (2026-11-17, "graphics glitch again"): the
    /// 20 background tiles are triangle strips whose vertices MTX keeps in
    /// VRAM (0x041C29D4.., 4 vertices of 16 bytes per tile) and moves during
    /// the wipe. Only the positions change; the change check's two words in
    /// the block are a UV and a colour, so it found nothing and the copy kept
    /// the old positions (tiles drawn in a grid over leftover texture memory
    /// for ~8 frames; Sony's picture clean). Copied whole at the enqueue,
    /// the copy has the new positions.
    #[test]
    fn moved_vertices_the_samples_miss_are_copied_whole_at_the_enqueue() {
        const BLOCK_WORDS: usize = (CHECK_BLOCK_BYTES / 4) as usize;
        let block = 0x1C_2800u32;
        let known = known(&[(block, 2 * CHECK_BLOCK_BYTES)]);
        // vtype 0x13a, 16 bytes per vertex: u, v (16 bit); colour (16 bit),
        // normal x, y; normal z, pad, x; y, z (positions 16 bit). The wipe
        // moves the tiles sideways: only the x halfwords change.
        let tiles = |moved: i16| {
            let mut words = vec![0u32; 2 * BLOCK_WORDS];
            let first = (0x29D4 - 0x2800) / 4;
            for tile in 0..20 {
                for vertex in 0..4 {
                    let at = first + tile * 16 + vertex * 4;
                    let x = (96 * (tile % 5) as i16 + 96 * (vertex as i16 & 1)).wrapping_add(moved);
                    let (y, z) = (97 * (tile / 5) as i16 + 97 * (vertex as i16 >> 1), 300i16);
                    words[at] = 0x0080_0000 | vertex as u32;
                    words[at + 1] = 0x7F00_FFFF;
                    words[at + 2] = (x as u16 as u32) << 16 | 0x7F;
                    words[at + 3] = (z as u16 as u32) << 16 | y as u16 as u32;
                }
            }
            words
        };
        let vram = tiles(-37); // the game moved every tile
        let mut copy = tiles(0);
        let differs = |b: u32, word: u32| {
            let index = ((b - block) / 4 + word) as usize;
            vram[index] != copy[index]
        };
        assert_eq!(known.changed_blocks(1, true, differs, |_| {}), 0, "the samples see nothing (the bug)");

        let whole = known.copy_recent_whole(1, WHOLE_AT_ENQUEUE_BYTES, |offset, size| {
            let (from, count) = (((offset - block) / 4) as usize, (size / 4) as usize);
            copy[from..from + count].copy_from_slice(&vram[from..from + count]);
        });
        assert_eq!(whole, 1);
        assert_eq!(copy, vram, "the new positions");
    }

    #[test]
    fn whole_copies_stay_within_the_budget_and_skip_the_others() {
        // Range 0 recent and small, 1 not recent, 2 recent but too big for
        // what is left, 3 recent and fits.
        let known = known(&[(0x0000, 0x400), (0x1000, 0x400), (0x2000, 0x8000), (0xA000, 0x400)]);
        let mut copied = Vec::new();
        let whole = known.copy_recent_whole(0b1101, 0x1000, |offset, size| copied.push((offset, size)));
        assert_eq!(copied, [(0x0000, 0x400), (0xA000, 0x400)]);
        assert_eq!(whole, 0b1001, "range 2 is left to the samples");
        assert_eq!(known.copy_recent_whole(0b1101, 0, |_, _| panic!("budget 0: samples only")), 0);
    }

    #[test]
    fn changed_blocks_stop_at_vram_end() {
        let known = known(&[(VRAM_SIZE - 0x400, 0x10000)]);
        let mut blocks = 0;
        known.changed_blocks(1, true, |_, _| true, |block| {
            assert!(block < VRAM_SIZE);
            blocks += 1;
        });
        assert_eq!(blocks, 1);
    }

    #[test]
    fn refresh_cycles_through_all_ranges() {
        let known = known(&[(0x0000, 0x1800), (0x8000, 0x400)]);
        let mut refresh = Refresh::new();
        let mut copies = [(0u32, 0u32); 8];
        let mut n = 0;
        let mut record = |offset, size| {
            copies[n] = (offset, size);
            n += 1;
        };
        refresh.step(&known, 0x1000, &mut record);
        refresh.step(&known, 0x1000, &mut record);
        // Round 1: 0x1000 of range 0; round 2: its last 0x800, range 1's
        // 0x400, then range 0 again from its start.
        assert_eq!(&copies[..n], &[(0x0000, 0x1000), (0x1000, 0x800), (0x8000, 0x400), (0x0000, 0x400)]);
        // No ranges: nothing (the C plugin read a stale entry here).
        let mut empty = Refresh::new();
        empty.step(&KnownRanges::new(), 0x1000, |_, _| panic!());
    }

    #[test]
    fn refresh_skips_empty_ranges() {
        let known = known(&[(0x0000, 0), (0x4000, 0x400)]);
        let mut refresh = Refresh::new();
        let mut copied = 0;
        refresh.step(&known, 0x1000, |offset, size| {
            assert_eq!((offset, size), (0x4000, 0x400));
            copied += 1;
        });
        assert!(copied >= 1);
    }

    #[test]
    fn sync_stall_waits_again_once_caught_up() {
        let mut stall = SyncStall::new();
        assert!(!stall.should_wait(10, 10), "caught up: no wait");
        assert!(stall.should_wait(9, 10));
        stall.timed_out(12);
        assert!(!stall.should_wait(11, 14), "stalled until the events up to the timeout are handled");
        assert!(stall.should_wait(12, 14), "handled up to the timeout: waits again");
        // Wrapping counts.
        let mut stall = SyncStall::new();
        stall.timed_out(u32::MAX);
        assert!(!stall.should_wait(u32::MAX - 1, 3));
        assert!(stall.should_wait(0, 3));
        assert!(reached(1, u32::MAX));
    }

    #[test]
    fn best_method_is_the_fastest_correct_one_but_not_vfpu() {
        // The C plugin's live probe (MTX, 2026-11-14): words 2843, cached
        // 687, unrolled 1068, vfpu 421, memcpy 358; both DMA ways failed (ok
        // mask 0x76: words, cached, unrolled, vfpu, memcpy).
        let us = [0, 2843, 687, 0, 1068, 421, 358, 0];
        assert_eq!(best_method(&us, 0x76), copy_method::MEMCPY);
        // VFPU alone fastest and correct: not chosen.
        assert_eq!(best_method(&[9, 9, 9, 9, 9, 1, 9, 0], 1 << 5 | 1 << 2), copy_method::CACHED);
        // Nothing correct: words.
        assert_eq!(best_method(&us, 0), copy_method::WORDS);
    }

    #[test]
    fn ge_probe_lists_are_the_c_plugins() {
        let mut list = [0; GE_PROBE_LIST_WORDS];
        assert_eq!(ge_probe_list(1, 0, 0, 0, &mut list), 2);
        assert_eq!(&list[..2], &[0x0F00_0000, 0x0C00_0000]);
        // Mode 4: VRAM -> the copy area, 64 KiB.
        let n = ge_probe_list(4, 0x041F_0000, 0x0BFF_0000, 0x10000, &mut list);
        assert_eq!(n, 10);
        assert_eq!(
            &list[..n],
            &[
                0xB21F_0000,
                0xB304_0100,
                0xB4FF_0000,
                0xB50B_0100,
                0xEB00_0000,
                0xEC00_0000,
                0xEE00_FCFF, // 64 rows: (63 << 10) | 255
                0xEA00_0001,
                0x0F00_0000,
                0x0C00_0000
            ]
        );
    }

    #[test]
    fn plugins_txt_gets_our_line_once() {
        let mut out = [0u8; 512];
        let n = plugins_txt_with_ours(b"", &mut out).unwrap();
        let text = &out[..n];
        assert!(text.ends_with(b"game, ms0:/seplugins/nativeres.prx, on\n"));
        assert!(text.starts_with(b"# NativeRes"));
        // Again: already there.
        let mut again = [0u8; 512];
        assert_eq!(plugins_txt_with_ours(text, &mut again), None);
        // Turned off by the user: stays off.
        assert_eq!(plugins_txt_with_ours(b"game, ms0:/seplugins/NativeRes.prx, off\n", &mut again), None);
    }

    #[test]
    fn plugins_txt_keeps_the_users_lines_and_newlines() {
        let mut out = [0u8; 512];
        let n = plugins_txt_with_ours(b"vsh, ms0:/seplugins/a.prx, on\r\ngame, ms0:/b.prx, on", &mut out).unwrap();
        assert_eq!(
            &out[..n],
            &b"vsh, ms0:/seplugins/a.prx, on\r\ngame, ms0:/b.prx, on\r\n# NativeRes capture plugin, every PSP game (added by nativeres.suprx)\r\ngame, ms0:/seplugins/nativeres.prx, on\r\n"[..]
        );
        // Too small a buffer: nothing.
        let mut small = [0u8; 16];
        assert_eq!(plugins_txt_with_ours(b"vsh, ms0:/x.prx, on\n", &mut small), None);
    }
}
