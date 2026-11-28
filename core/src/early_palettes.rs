//! Palettes the game rewrites every frame, read at the walk's start.
//!
//! A palette's words are read with the first draw that uses it
//! (`recorder::LoadedPalette`): by the copier, when the walk gets to that
//! draw. The GE reads them when it executes the list, right after the game
//! enqueued it. A game that writes the next frame's palette while its list is
//! still being walked gives us the next frame's colours: Kenka Bancho's cloud
//! wipe slides a white alpha ramp one entry a frame, and the game writes the
//! next ramp 2-4.5 ms after enqueueing a list our walk takes ~10 ms over (the
//! mask is its second-to-last draw); every picture of the wipe was one alpha
//! step (16/255) whiter on the ramp than Sony's (2026-11-28).
//!
//! So palettes whose words changed at the same address since the last frame
//! (`note_frame`, at a frame's end) are read again when the next frame's
//! first list is enqueued (`read_now`), before the walk, and a draw that
//! places one of them takes those words (`words`) instead of a later copy.
//! A few at most (`MAX_EARLY`, 1 KiB each): only palettes that change.

use core::sync::atomic::{AtomicBool, Ordering};

use crate::frame::FramePalette;
use crate::memory::PspMemory;

/// Whether changing palettes are read at the walk's start (on by default;
/// off: with their draw's job only, as before 2026-11-28).
pub static EARLY_PALETTES: AtomicBool = AtomicBool::new(true);

/// Palettes read at a walk's start at most.
pub const MAX_EARLY: usize = 4;
/// Words of one palette at most (CLUT_LOAD: 64 blocks of 8 words).
const MAX_WORDS: usize = 512;
/// Slots of the table of palettes seen (a power of two).
const SEEN_SLOTS: usize = 512;
/// Slots probed for one palette.
const PROBES: usize = 8;
/// A palette is read early at this many walks after its last change.
const KEEP_FRAMES: u32 = 8;

#[derive(Clone, Copy)]
struct Seen {
    address: u32,
    word_count: u32,
    hash: u32,
    /// The frame it was last seen in (0: empty slot).
    frame: u32,
}

#[derive(Clone, Copy)]
struct Early {
    /// 0: a free entry.
    address: u32,
    word_count: u32,
    /// The frame its words last changed in.
    changed: u32,
    /// `words` were read for the frame being recorded.
    read: bool,
    words: [u32; MAX_WORDS],
}

pub struct EarlyPalettes {
    seen: [Seen; SEEN_SLOTS],
    early: [Early; MAX_EARLY],
    /// Frames noted (`note_frame`); 0 before the first.
    frame: u32,
}

impl Default for EarlyPalettes {
    fn default() -> Self {
        EarlyPalettes::new()
    }
}

impl EarlyPalettes {
    pub const fn new() -> EarlyPalettes {
        const NO_SEEN: Seen = Seen { address: 0, word_count: 0, hash: 0, frame: 0 };
        const NO_EARLY: Early = Early { address: 0, word_count: 0, changed: 0, read: false, words: [0; MAX_WORDS] };
        EarlyPalettes { seen: [NO_SEEN; SEEN_SLOTS], early: [NO_EARLY; MAX_EARLY], frame: 0 }
    }

    /// A frame's end: its palettes (`Frame::palettes`) and their hashes
    /// (`Frame::palette_hashes`, resolved). A palette whose hash differs
    /// from the one it had at the same address in the frame before is read
    /// at the next walks' starts. The words read for this frame are spent.
    pub fn note_frame(&mut self, palettes: &[FramePalette], hashes: &[u32]) {
        self.frame = self.frame.wrapping_add(1).max(1);
        let frame = self.frame;
        for (palette, &hash) in palettes.iter().zip(hashes) {
            if palette.address == 0 {
                continue; // words from an old saved state
            }
            let Some(slot) = self.slot(palette.address, palette.word_count) else {
                continue;
            };
            let seen = &mut self.seen[slot];
            let changed = seen.address == palette.address && seen.frame == frame.wrapping_sub(1) && seen.hash != hash;
            *seen = Seen { address: palette.address, word_count: palette.word_count, hash, frame };
            if changed {
                self.mark(palette.address, palette.word_count);
            }
        }
        for early in self.early.iter_mut() {
            early.read = false;
            if early.address != 0 && frame.wrapping_sub(early.changed) >= KEEP_FRAMES {
                early.address = 0;
            }
        }
    }

    /// The seen-table slot of a palette: its own, else an empty one or one
    /// not seen in the last frame; `None` if the probed slots are all taken.
    fn slot(&self, address: u32, word_count: u32) -> Option<usize> {
        let start = ((address ^ word_count.rotate_left(16)).wrapping_mul(0x9E37_79B1) >> 23) as usize % SEEN_SLOTS;
        let mut free = None;
        for probe in 0..PROBES {
            let slot = (start + probe) % SEEN_SLOTS;
            let seen = &self.seen[slot];
            if seen.address == address && seen.word_count == word_count {
                return Some(slot);
            }
            let stale = seen.frame == 0 || self.frame.wrapping_sub(seen.frame) > 1;
            if free.is_none() && stale {
                free = Some(slot);
            }
        }
        free
    }

    /// Reads `address` at the walk's start from now on (a free entry, or
    /// the one that changed longest ago; all changed lately: not added).
    fn mark(&mut self, address: u32, word_count: u32) {
        let frame = self.frame;
        if let Some(early) = self.early.iter_mut().find(|early| early.address == address && early.word_count == word_count) {
            early.changed = frame;
            return;
        }
        let oldest = (0..MAX_EARLY).min_by_key(|&i| if self.early[i].address == 0 { 0 } else { self.early[i].changed.max(1) }).unwrap_or(0);
        let entry = &mut self.early[oldest];
        if entry.address == 0 || frame.wrapping_sub(entry.changed) >= KEEP_FRAMES {
            *entry = Early { address, word_count, changed: frame, read: false, words: [0; MAX_WORDS] };
        }
    }

    /// The walk's start (the frame's first list enqueued): reads the
    /// changing palettes' words now. Returns how many were read.
    pub fn read_now(&mut self, memory: &mut impl PspMemory) -> u32 {
        if !EARLY_PALETTES.load(Ordering::Relaxed) {
            return 0;
        }
        let mut read = 0;
        for early in self.early.iter_mut().filter(|early| early.address != 0) {
            let count = (early.word_count as usize).min(MAX_WORDS);
            early.read = memory.read_words(early.address, &mut early.words[..count]).is_some();
            read += early.read as u32;
        }
        read
    }

    /// The words of the palette at `address` read at this walk's start, if
    /// it is one of the changing ones.
    pub fn words(&self, address: u32, word_count: u32) -> Option<&[u32]> {
        let early = self.early.iter().find(|early| early.read && early.address == address && early.word_count == word_count)?;
        Some(&early.words[..(word_count as usize).min(MAX_WORDS)])
    }

    /// Forgets everything (a new game).
    pub fn clear(&mut self) {
        *self = EarlyPalettes::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::test_memory::TestMemory;

    const ADDRESS: u32 = 0x08E8_7AD0;

    fn palette(address: u32) -> FramePalette {
        FramePalette { address, word_count: 8, offset: 0 }
    }

    #[test]
    fn a_palette_that_changes_at_its_address_is_read_at_the_next_walks_start() {
        let mut early = Box::new(EarlyPalettes::new());
        let mut memory = TestMemory::default();
        early.note_frame(&[palette(ADDRESS)], &[1]);
        memory.write(ADDRESS, &[0x11; 8]);
        assert_eq!(early.read_now(&mut memory), 0, "seen once: nothing to read early");
        early.note_frame(&[palette(ADDRESS)], &[2]);
        memory.write(ADDRESS, &[0x22; 8]);
        assert_eq!(early.read_now(&mut memory), 1);
        memory.write(ADDRESS, &[0x33; 8]); // the game writes the next one during the walk
        assert_eq!(early.words(ADDRESS, 8), Some(&[0x22u32; 8][..]), "the words of the walk's start");
        assert_eq!(early.words(ADDRESS, 16), None, "another length is another palette");
        early.note_frame(&[palette(ADDRESS)], &[3]);
        assert_eq!(early.words(ADDRESS, 8), None, "spent at the frame's end");
    }

    #[test]
    fn a_palette_that_stays_the_same_is_never_read_early() {
        let mut early = Box::new(EarlyPalettes::new());
        let mut memory = TestMemory::default();
        memory.write(ADDRESS, &[0x11; 8]);
        for _ in 0..5 {
            early.note_frame(&[palette(ADDRESS)], &[7]);
            assert_eq!(early.read_now(&mut memory), 0);
        }
    }

    #[test]
    fn a_change_after_a_frame_without_the_palette_does_not_count() {
        let mut early = Box::new(EarlyPalettes::new());
        let mut memory = TestMemory::default();
        memory.write(ADDRESS, &[0x11; 8]);
        early.note_frame(&[palette(ADDRESS)], &[1]);
        early.note_frame(&[], &[]);
        early.note_frame(&[palette(ADDRESS)], &[2]);
        assert_eq!(early.read_now(&mut memory), 0, "the palette was gone a frame: maybe another texture's");
    }

    #[test]
    fn a_palette_that_stopped_changing_is_dropped_after_a_while() {
        let mut early = Box::new(EarlyPalettes::new());
        let mut memory = TestMemory::default();
        memory.write(ADDRESS, &[0x11; 8]);
        early.note_frame(&[palette(ADDRESS)], &[1]);
        early.note_frame(&[palette(ADDRESS)], &[2]);
        for _ in 0..KEEP_FRAMES {
            assert_eq!(early.read_now(&mut memory), 1);
            early.note_frame(&[palette(ADDRESS)], &[2]);
        }
        assert_eq!(early.read_now(&mut memory), 0);
    }

    #[test]
    fn at_most_a_few_are_read_early() {
        let mut early = Box::new(EarlyPalettes::new());
        let mut memory = TestMemory::default();
        let palettes: Vec<FramePalette> = (0..10).map(|i| palette(ADDRESS + 0x400 * i)).collect();
        for i in 0..10 {
            memory.write(ADDRESS + 0x400 * i, &[i; 8]);
        }
        early.note_frame(&palettes, &[1; 10]);
        early.note_frame(&palettes, &[2; 10]);
        assert_eq!(early.read_now(&mut memory), MAX_EARLY as u32);
    }
}
