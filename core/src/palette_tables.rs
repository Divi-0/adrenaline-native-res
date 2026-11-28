//! The display's palette tables: which 256-colour table holds which palette.
//!
//! Index textures (CLUT4/CLUT8) don't carry their colours: each frame
//! texture gets a 256-colour table, set on the draw's copy of the GPU
//! texture. Final Fantasy's world map animates its water by switching
//! between four palettes every 8 frames; with the palette part of the
//! texture, ~180 map tiles counted as new each time (12 ms of decoding, a
//! frame shown late and one from the game's own picture). With tables that
//! costs one 1 KiB table.
//!
//! A table is filled once per palette (`PaletteKey`) and only given to
//! another palette once no frame still on its way through the GPU uses it
//! (`RETIRE_FRAMES`). This file is the bookkeeping, tested on the PC; the
//! Vita side (renderer/vita/src/palettes.rs) owns the tables' GPU memory and
//! fills them.

use crate::frame::{texture_flags, TextureDescriptor};
use crate::texture_cache::RETIRE_FRAMES;

/// MTX Mototrax keeps a palette beside nearly every texture: its LA race
/// frame needs 342 different tables (replay --palettes). With 256 the oldest
/// was taken even when an earlier draw of the same frame still used it, and
/// those draws showed another texture's colours (noise on the Vita only).
pub const TABLES: usize = 1024;
/// Bytes of one table (256 RGBA colours).
pub const TABLE_BYTES: usize = 256 * 4;
/// Hash buckets for finding a palette's table (a power of two).
const BUCKETS: usize = 1024;
/// End of a bucket's chain.
const NONE: i16 = -1;

/// What makes two palette tables the same: the CLUT format, the palette's
/// words (their hash and number) and whether the texture's alpha is ignored
/// (`texture_flags::IGNORE_ALPHA`: the table's alpha is then 0xFF).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PaletteKey {
    pub clut_format: u32,
    pub word_count: u32,
    pub hash: u32,
    pub ignore_alpha: bool,
}

impl PaletteKey {
    pub fn of(texture: &TextureDescriptor) -> PaletteKey {
        PaletteKey {
            clut_format: texture.clut_format,
            word_count: texture.palette_word_count,
            hash: texture.palette_hash,
            ignore_alpha: texture.flags & texture_flags::IGNORE_ALPHA != 0,
        }
    }

    fn bucket(&self) -> usize {
        let key = self.hash ^ self.clut_format.wrapping_mul(0x9E37_79B1) ^ (self.word_count << 20);
        ((key ^ (key >> 15)) as usize) & (BUCKETS - 1)
    }
}

/// What `PaletteTables::find` gave.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PaletteLookup {
    /// Table `n` holds the palette already.
    Found(usize),
    /// Table `n` is the palette's now: fill it.
    Fill(usize),
    /// Every table is used by this frame or one the GPU may still draw:
    /// the caller has to do without (the texture's own table).
    Full,
}

#[derive(Clone, Copy)]
struct Table {
    key: PaletteKey,
    /// The frame that used it last (`find`'s `now`); 0 = never filled.
    last_used: u32,
    /// The next table in the same bucket, or NONE.
    next: i16,
    /// The bucket it is in, or NONE.
    bucket: i16,
}

pub struct PaletteTables {
    tables: [Table; TABLES],
    buckets: [i16; BUCKETS],
    /// Where the search for a table to reuse goes on (round robin).
    hand: usize,
}

impl Default for PaletteTables {
    fn default() -> Self {
        PaletteTables::new()
    }
}

impl PaletteTables {
    pub const fn new() -> PaletteTables {
        const EMPTY: Table = Table {
            key: PaletteKey { clut_format: 0, word_count: 0, hash: 0, ignore_alpha: false },
            last_used: 0,
            next: NONE,
            bucket: NONE,
        };
        PaletteTables { tables: [EMPTY; TABLES], buckets: [NONE; BUCKETS], hand: 0 }
    }

    /// The table for `key` in frame `now` (the display's frame count, 0 is
    /// taken as 1), marked as used by it.
    pub fn find(&mut self, key: PaletteKey, now: u32) -> PaletteLookup {
        let now = now.max(1);
        let bucket = key.bucket();
        let mut at = self.buckets[bucket];
        while at != NONE {
            let table = &mut self.tables[at as usize];
            if table.last_used != 0 && table.key == key {
                table.last_used = now;
                return PaletteLookup::Found(at as usize);
            }
            at = table.next;
        }
        // A table no frame on its way through the GPU uses any more.
        let mut found = None;
        for _ in 0..TABLES {
            let candidate = self.hand;
            self.hand = (self.hand + 1) % TABLES;
            let last_used = self.tables[candidate].last_used;
            if last_used == 0 || now.wrapping_sub(last_used) >= RETIRE_FRAMES {
                found = Some(candidate);
                break;
            }
        }
        let Some(found) = found else {
            return PaletteLookup::Full;
        };
        self.unlink(found);
        self.tables[found] = Table { key, last_used: now, next: self.buckets[bucket], bucket: bucket as i16 };
        self.buckets[bucket] = found as i16;
        PaletteLookup::Fill(found)
    }

    /// Takes table `index` out of its bucket's chain.
    fn unlink(&mut self, index: usize) {
        let bucket = self.tables[index].bucket;
        if bucket == NONE {
            return;
        }
        let next = self.tables[index].next;
        if self.buckets[bucket as usize] == index as i16 {
            self.buckets[bucket as usize] = next;
        } else {
            let mut at = self.buckets[bucket as usize];
            while at != NONE {
                if self.tables[at as usize].next == index as i16 {
                    self.tables[at as usize].next = next;
                    break;
                }
                at = self.tables[at as usize].next;
            }
        }
        self.tables[index].bucket = NONE;
        self.tables[index].next = NONE;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(hash: u32) -> PaletteKey {
        PaletteKey { clut_format: 0x00FF_0003, word_count: 256, hash, ignore_alpha: false }
    }

    fn tables() -> Box<PaletteTables> {
        Box::new(PaletteTables::new())
    }

    #[test]
    fn a_palette_is_filled_once_and_then_found() {
        let mut tables = tables();
        let PaletteLookup::Fill(index) = tables.find(key(7), 1) else { panic!() };
        assert_eq!(tables.find(key(7), 1), PaletteLookup::Found(index));
        assert_eq!(tables.find(key(7), 50), PaletteLookup::Found(index));
    }

    #[test]
    fn every_part_of_the_key_counts() {
        let mut tables = tables();
        let PaletteLookup::Fill(first) = tables.find(key(7), 1) else { panic!() };
        let others = [
            PaletteKey { clut_format: 0x00FF_0002, ..key(7) },
            PaletteKey { word_count: 16, ..key(7) },
            PaletteKey { ignore_alpha: true, ..key(7) },
            key(8),
        ];
        for other in others {
            let PaletteLookup::Fill(index) = tables.find(other, 1) else { panic!("{other:?} found") };
            assert_ne!(index, first);
        }
    }

    #[test]
    fn frame_zero_counts_as_one() {
        // last_used 0 means "never filled": a table filled at frame 0 must
        // still be found.
        let mut tables = tables();
        let PaletteLookup::Fill(index) = tables.find(key(1), 0) else { panic!() };
        assert_eq!(tables.find(key(1), 0), PaletteLookup::Found(index));
    }

    #[test]
    fn tables_in_use_by_frames_on_the_gpu_are_not_taken() {
        let mut tables = tables();
        // Frame 10 uses every table.
        for hash in 0..TABLES as u32 {
            assert!(matches!(tables.find(key(hash), 10), PaletteLookup::Fill(_)));
        }
        // Frames 11 and 12 may still be drawn with them: no table is free.
        assert_eq!(tables.find(key(5000), 11), PaletteLookup::Full);
        assert_eq!(tables.find(key(5000), 12), PaletteLookup::Full);
        // The old palettes are all still there.
        let PaletteLookup::Found(three) = tables.find(key(3), 12) else { panic!() };
        // Frame 13: frame 10 is done, its tables can be taken, but not
        // palette 3's (frame 12 used it), even where the search starts.
        tables.hand = three;
        let PaletteLookup::Fill(index) = tables.find(key(5000), 13) else { panic!() };
        assert_ne!(index, three);
        assert_eq!(tables.find(key(3), 13), PaletteLookup::Found(three));
    }

    #[test]
    fn a_reused_table_leaves_its_old_bucket() {
        let mut tables = tables();
        for hash in 0..TABLES as u32 {
            tables.find(key(hash), 1);
        }
        // Every table is retired by frame 4: new palettes take them all over.
        for hash in 0..TABLES as u32 {
            assert!(matches!(tables.find(key(10_000 + hash), 4), PaletteLookup::Fill(_)));
        }
        // None of the old palettes is found any more (their tables hold
        // other colours now): filled again.
        for hash in 0..8 {
            assert!(!matches!(tables.find(key(hash), 8), PaletteLookup::Found(_)));
        }
        // And all the new ones that kept their table are found.
        assert!(matches!(tables.find(key(10_000 + 100), 8), PaletteLookup::Found(_)));
    }

    #[test]
    fn chains_stay_intact_when_a_middle_table_is_taken() {
        let mut tables = tables();
        // Keys in one bucket: the same bucket() value.
        let first = key(1);
        let same_bucket: Vec<PaletteKey> =
            (2..200_000u32).map(key).filter(|other| other.bucket() == first.bucket()).take(2).collect();
        let (second, third) = (same_bucket[0], same_bucket[1]);
        let PaletteLookup::Fill(a) = tables.find(first, 1) else { panic!() };
        let PaletteLookup::Fill(b) = tables.find(second, 1) else { panic!() };
        let PaletteLookup::Fill(c) = tables.find(third, 1) else { panic!() };
        // Keep the first and third in use, so only the second's table (and
        // the never-used ones) can be taken; walk the hand around to it.
        tables.hand = b;
        tables.find(first, 5);
        tables.find(third, 5);
        let PaletteLookup::Fill(taken) = tables.find(key(999_999), 5) else { panic!() };
        assert_eq!(taken, b);
        assert_eq!(tables.find(first, 6), PaletteLookup::Found(a));
        assert_eq!(tables.find(third, 6), PaletteLookup::Found(c));
        assert!(matches!(tables.find(second, 6), PaletteLookup::Fill(_)));
    }

    /// Kenka Bancho's cloud wipe: a new palette every frame that differs from
    /// the earlier ones only in its alpha ramp. Each must get a table of its
    /// own: with an earlier palette's table the frame drew an earlier wipe (the
    /// hole went back, 2-3 times per wipe, the same frames in every run).
    #[test]
    fn the_cloud_wipes_palettes_never_find_an_older_frames_table() {
        let mut tables = tables();
        for (frame, start) in (-15..=32).rev().enumerate() {
            let words = crate::texture::kenka_bancho_wipe_palette(start);
            let wipe = PaletteKey { hash: crate::texture::hash_words(&words), ..key(0) };
            let found = tables.find(wipe, frame as u32 + 1);
            assert!(matches!(found, PaletteLookup::Fill(_)), "frame {frame} (ramp at {start}): {found:?}, an older palette's table");
        }
    }
}
