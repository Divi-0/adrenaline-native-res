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

/// What a VRAM copy of `size` bytes at `offset` really copies: the offset
/// aligned down to a word, the size rounded up to one; None when that is
/// empty or leaves VRAM.
pub fn copy_span(offset: u32, size: u32) -> Option<(u32, u32)> {
    let offset = offset & !3;
    let size = size.checked_add(3)? & !3;
    (offset < VRAM_SIZE && size != 0 && size <= VRAM_SIZE - offset).then_some((offset, size))
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
pub const PLUGINS_TXT_LINE: &[u8] = b"game, ms0:/seplugins/nativepsp.prx, on";
const PLUGINS_TXT_COMMENT: &[u8] = b"# NativePSP capture plugin, every PSP game (added by nativepsp.suprx)";
/// The plugin's file name, as found in any line about it.
const PLUGIN_FILE: &[u8] = b"nativepsp.prx";

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
        assert!(text.ends_with(b"game, ms0:/seplugins/nativepsp.prx, on\n"));
        assert!(text.starts_with(b"# NativePSP"));
        // Again: already there.
        let mut again = [0u8; 512];
        assert_eq!(plugins_txt_with_ours(text, &mut again), None);
        // Turned off by the user: stays off.
        assert_eq!(plugins_txt_with_ours(b"game, ms0:/seplugins/NativePSP.prx, off\n", &mut again), None);
    }

    #[test]
    fn plugins_txt_keeps_the_users_lines_and_newlines() {
        let mut out = [0u8; 512];
        let n = plugins_txt_with_ours(b"vsh, ms0:/seplugins/a.prx, on\r\ngame, ms0:/b.prx, on", &mut out).unwrap();
        assert_eq!(
            &out[..n],
            &b"vsh, ms0:/seplugins/a.prx, on\r\ngame, ms0:/b.prx, on\r\n# NativePSP capture plugin, every PSP game (added by nativepsp.suprx)\r\ngame, ms0:/seplugins/nativepsp.prx, on\r\n"[..]
        );
        // Too small a buffer: nothing.
        let mut small = [0u8; 16];
        assert_eq!(plugins_txt_with_ours(b"vsh, ms0:/x.prx, on\n", &mut small), None);
    }
}
