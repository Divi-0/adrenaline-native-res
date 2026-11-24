//! Measuring aid (`ON`, off by default): does the game
//! rewrite the memory we copy for a frame before that frame is finished?
//!
//! Every range the draw stream copies (`DrawStream::copy`, `copy_one`) is
//! noted with three of its words (first, middle, last) as copied and the
//! time; at the end of the frame the capture reads those words again from
//! PSP memory (`check`). A changed word means the game wrote the range after
//! we copied it, so a copy made a little later, or while it was writing,
//! would have read other (or torn) data: the suspected cause of the broken
//! HUD text at MTX's race start (2026-11-08).
//!
//! Written by the copier and the producer (an index taken with fetch_add),
//! read by the capture thread between frames, when no copy runs.

use core::sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, Ordering};

/// Whether ranges are noted.
pub static ON: AtomicBool = AtomicBool::new(false);

/// Ranges noted per frame at most (MTX's race copies ~2,200).
pub const ENTRIES: usize = 4096;

#[derive(Clone, Copy, Default)]
struct Entry {
    /// Physical address of the range's first word.
    address: u32,
    word_count: u32,
    /// The words at positions 0, word_count / 2 and word_count - 1.
    values: [u32; 3],
    /// Clock (`set_clock`) when copied.
    copied_at: u32,
}

/// The table: ENTRIES entries, made by the Vita side when the watch is first
/// switched on (`set_table`; 96 KiB that only this measuring aid needs, kept
/// out of the always-needed memory since 2026-11-24). Null: nothing noted.
static TABLE: AtomicPtr<Entry> = AtomicPtr::new(core::ptr::null_mut());

/// Bytes `set_table` needs.
pub const TABLE_BYTES: usize = ENTRIES * core::mem::size_of::<Entry>();

/// Hands over the table's memory (TABLE_BYTES, word-aligned, never freed:
/// a copier may still be noting when the watch is switched off).
///
/// # Safety
/// `memory` must be TABLE_BYTES of writable memory, aligned for `u32`, that
/// lives as long as the process.
pub unsafe fn set_table(memory: *mut u8) {
    TABLE.store(memory as *mut Entry, Ordering::Release);
}

/// Whether the table is there (`set_table`).
pub fn has_table() -> bool {
    !TABLE.load(Ordering::Acquire).is_null()
}

static COUNT: AtomicU32 = AtomicU32::new(0);
static CLOCK: AtomicPtr<()> = AtomicPtr::new(core::ptr::null_mut());

/// Registers the clock, in microseconds (`None`: none).
pub fn set_clock(clock: Option<fn() -> u32>) {
    CLOCK.store(clock.map_or(core::ptr::null_mut(), |clock| clock as *mut ()), Ordering::Release);
}

fn now() -> u32 {
    let clock = CLOCK.load(Ordering::Acquire);
    if clock.is_null() {
        return 0;
    }
    // Safe: only `set_clock` stores here, from a `fn() -> u32`.
    let clock: fn() -> u32 = unsafe { core::mem::transmute(clock) };
    clock()
}

/// Notes a range just copied: `words` as copied from physical `address`.
#[inline]
pub fn note(address: u32, words: &[u32]) {
    // Main RAM only: VRAM is read from the plugin's copy, which changes
    // whenever the plugin copies it again.
    if !ON.load(Ordering::Relaxed) || words.is_empty() || crate::memory::physical(address) < crate::memory::RAM_START {
        return;
    }
    let table = TABLE.load(Ordering::Acquire);
    if table.is_null() {
        return;
    }
    let index = COUNT.fetch_add(1, Ordering::Relaxed) as usize;
    if index >= ENTRIES {
        return;
    }
    let last = words.len() - 1;
    let entry = Entry {
        address,
        word_count: words.len() as u32,
        values: [words[0], words[last / 2], words[last]],
        copied_at: now(),
    };
    // Safe: this index is ours alone (fetch_add), inside the table.
    unsafe { table.add(index).write(entry) };
}

/// What `check` found for one frame.
#[derive(Clone, Copy, Default, Debug)]
pub struct Rewrites {
    pub checked: u32,
    pub changed: u32,
    /// The first changed range: address, words, and microseconds from its
    /// copy to the check.
    pub first_address: u32,
    pub first_words: u32,
    pub first_age_us: u32,
    /// The youngest changed range's age: how soon after our copy the game
    /// had written it at the latest.
    pub youngest_age_us: u32,
}

/// Capture thread, between frames: reads the noted words again with
/// `sample` (as `DeferredReader::sample`) and forgets the notes.
pub fn check(sample: fn(address: u32, positions: &[usize], out: &mut [u32]) -> bool) -> Rewrites {
    let count = (COUNT.swap(0, Ordering::Acquire) as usize).min(ENTRIES);
    let table = TABLE.load(Ordering::Acquire);
    let now = now();
    if table.is_null() {
        return Rewrites::default();
    }
    let mut result = Rewrites { youngest_age_us: u32::MAX, ..Rewrites::default() };
    for index in 0..count {
        // Safe: between frames no copy writes the table.
        let entry = unsafe { table.add(index).read() };
        let last = entry.word_count as usize - 1;
        let positions = [0, last / 2, last];
        let mut now_values = [0u32; 3];
        if !sample(entry.address, &positions, &mut now_values) {
            continue;
        }
        result.checked += 1;
        if now_values != entry.values {
            let age = now.wrapping_sub(entry.copied_at);
            if result.changed == 0 {
                result.first_address = entry.address;
                result.first_words = entry.word_count;
                result.first_age_us = age;
            }
            result.changed += 1;
            result.youngest_age_us = result.youngest_age_us.min(age);
        }
    }
    if result.changed == 0 {
        result.youngest_age_us = 0;
    }
    result
}
