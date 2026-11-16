//! Decoded vertices kept across frames, in memory the GPU draws from.
//!
//! Most of a 3D game's vertex data is the same every frame: models are
//! stored once and drawn with new matrices (MTX Mototrax: 97% of the memory
//! a frame reads is unchanged in the next one). With the GPU transform
//! (`frame::GpuTransform`) a run's decoded vertices are in model space and
//! so don't depend on the matrices either. This cache keeps them: a run
//! seen before is not decoded again, and its vertices aren't even copied
//! into the frame: the draw points at the cache (`draw_flags::CACHED_VERTICES`)
//! and the GPU reads them there. On the Vita the arena is 1 MiB of main
//! memory mapped for the GPU (not CDRAM, see nativepsp_render.c), written
//! once by the CPU per new run.
//!
//! An entry is found by what decoding depends on (`RunKey`: address,
//! vertex count, format, texture coordinate scale and offset, colour
//! function) and checked against a hash of the run's bytes, so changed data
//! is decoded again. As in older PPSSPP versions' vertex cache
//! (VertexArrayInfo, ComputeMiniHash; current PPSSPP has no vertex cache),
//! a hit is mostly decided by a mini hash of a few sample words
//! (`find_quick`, `sample_hash`), so the run's bytes needn't be read at all;
//! all of them are hashed again every `FULL_CHECK_FRAMES` recorded frames
//! (`find`). Data that changes without touching the samples can show stale
//! for up to that many frames. It also keeps the run's bounding box, for the
//! near-plane test of a hit (the CPU would read the arena slowly on the Vita).
//!
//! The arena is a ring: new runs go after the newest, overwriting the
//! oldest. Frames still being shown use their runs' vertices, so an entry
//! used by one of the last `PROTECTED_FRAMES` recorded frames is never
//! overwritten; when the oldest one is, the new run simply isn't cached.

use crate::frame::FrameVertex;
use crate::raw_arena::RawArena;

/// What 16-bit indices reach: a cached draw's indices count from the start
/// of its window of the arena (`Draw::first_vertex`), and no run's vertices
/// cross a window's end (`allocate`). 2026-11-08: the arena was one window
/// (2 MiB); MTX's race start found it full for ~69 runs per frame, decoded
/// again every frame.
pub const WINDOW_VERTICES: u32 = 65536;

/// The window holding vertex `start` of the arena: its first vertex.
pub fn window_of(start: u32) -> u32 {
    start & !(WINDOW_VERTICES - 1)
}

/// Vertices in the PC replay's arena (the Vita's comes from the display
/// side, `nativepsp_render.c`): two windows, 4 MiB of vertices.
pub const ARENA_VERTICES: usize = 2 * WINDOW_VERTICES as usize;
/// Entries, in sets of `WAYS` (a key's set is chosen by its hash; within
/// the set the least recently used entry makes room).
const ENTRIES: usize = 8192;
const WAYS: usize = 4;
/// How long a note about a run is believed (recorded frames): "only the
/// CPU can record it" (a run partly behind the near plane may come into
/// view) and "its bytes change" (animated data, not worth the arena).
const NOTE_FRAMES: u32 = 30;
/// Allocations remembered for the ring (oldest first).
const ALLOCATIONS: usize = 8192;
/// Recorded frames whose runs stay in the arena: the frame being recorded,
/// the newest finished one, the one shown and the one before (the GPU may
/// still read it), with room to spare.
pub const PROTECTED_FRAMES: u32 = 6;
/// Allocations one store may pass over (still in use) looking for room.
const MAX_PASSES: usize = 64;
/// A cached run's bytes are hashed in full again after this many recorded
/// frames; in between its mini hash decides (`find_quick`).
pub const FULL_CHECK_FRAMES: u32 = 8;
/// Words a mini hash samples (`sample_hash`), at most.
pub const SAMPLE_WORDS: usize = 16;
/// How a mini hash samples now (debug "minisamples"): 1..=8 words spread
/// over the run, or `SAMPLE_BURSTS`: the first and the last 8 words (two
/// bursts). 4 by default (user's choice, 2026-11-07): each sample is an
/// uncached load of PSP RAM, ~0.75 ms of recorder time per frame in MTX's
/// race (8 samples: recorder busy ~27 ms, 4: ~24); a change the samples
/// miss shows for up to `FULL_CHECK_FRAMES` frames.
pub static SAMPLE_COUNT: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(DEFAULT_SAMPLES);
pub const DEFAULT_SAMPLES: usize = 4;
/// `SAMPLE_COUNT` for two bursts of 8 words.
pub const SAMPLE_BURSTS: usize = 16;

/// Whether runs are looked up by their mini hash at all (`find_quick`;
/// debug "quickhits 0|1"). Off (the default since 2026-11-08), every run is
/// read whole and checked by its full hash, every frame, as PPSSPP hashes
/// whole vertex data (DrawEngineCommon::ComputeMiniHash samples draw calls,
/// not words). MTX reuses vertex buffers for other content of the same
/// size (dust, HUD glyphs) where a few sampled words often match: with
/// quick hits its race start showed stale runs (long lines, smears, grey
/// triangles) in 4 of 6 filmed starts, without them in none of 4. With the
/// copier thread the words are copied anyway, so the full hash costs the
/// recorder only ~1-2 ms per race frame (user's choice, 2026-11-08).
pub static QUICK_HITS: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// The word positions (from the run's first word) `sample_hash` reads for
/// a run of `word_count` words, in increasing order (see `SAMPLE_COUNT`):
/// evenly spread, first and last included; or the first and the last 8
/// words (all of them for runs of up to 16).
pub fn sample_positions(word_count: usize) -> impl Iterator<Item = usize> {
    positions_for(SAMPLE_COUNT.load(core::sync::atomic::Ordering::Relaxed), word_count)
}

/// `sample_positions` for sampling `mode` (`SAMPLE_COUNT`).
fn positions_for(mode: usize, word_count: usize) -> impl Iterator<Item = usize> {
    let bursts = mode == SAMPLE_BURSTS;
    let samples = if bursts { word_count.min(SAMPLE_WORDS) } else { word_count.min(mode) };
    (0..samples).map(move |index| {
        if bursts {
            let half = SAMPLE_WORDS / 2;
            if index < half || word_count <= SAMPLE_WORDS { index } else { word_count - SAMPLE_WORDS + index }
        } else if samples < 2 {
            0
        } else {
            index * (word_count - 1) / (samples - 1)
        }
    })
}

/// A run's mini hash from its sample words (`sample_positions`, in order).

/// What a run's decoded vertices depend on besides its bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RunKey {
    pub address: u32,
    pub vertex_count: u32,
    pub format: u32,
    /// Texture coordinate scale and offset (u, v), as bits.
    pub texcoords: [u32; 4],
    /// The colour function (`recorder::Prepared` and the material colour).
    pub colors: u32,
}

impl RunKey {
    fn hash(&self) -> u32 {
        let mut hash = 0x811C_9DC5u32;
        for word in [self.address, self.vertex_count, self.format, self.colors].into_iter().chain(self.texcoords) {
            hash = (hash ^ word).wrapping_mul(0x0100_0193);
        }
        hash ^ (hash >> 15)
    }
}

pub fn sample_hash(word_count: usize, samples: impl Iterator<Item = u32>) -> u32 {
    let mut hash = (0x811C_9DC5u32 ^ word_count as u32).wrapping_mul(0x0100_0193);
    for word in samples {
        hash = (hash ^ word).wrapping_mul(0x0100_0193);
    }
    hash
}

/// A hash of a run's bytes (two 32-bit lanes; the arena is only used when
/// both match).
pub fn data_hash(words: &[u32]) -> u64 {
    let (mut a, mut b) = (0x811C_9DC5u32, 0x9E37_79B9u32);
    for &word in words {
        a = (a ^ word).wrapping_mul(0x0100_0193);
        b = b.rotate_left(5).wrapping_add(word ^ (b >> 7));
    }
    (a as u64) << 32 | b as u64
}

/// Asks the CPU to load `values` into its data cache (ARM `pld`, one per
/// 32-byte line, the Cortex-A9's line size); nothing elsewhere.
#[inline(always)]
fn prefetch<T>(values: &[T]) {
    #[cfg(target_arch = "arm")]
    {
        let start = values.as_ptr() as usize;
        let end = start + core::mem::size_of_val(values);
        let mut line = start & !31;
        while line < end {
            // Safe: `pld` only hints; it never faults or changes memory.
            unsafe { core::arch::asm!("pld [{0}]", in(reg) line, options(nostack, preserves_flags, readonly)) };
            line += 32;
        }
    }
    #[cfg(not(target_arch = "arm"))]
    let _ = values;
}

/// A model-space bounding box: min x, y, z, max x, y, z.
pub type Bounds = [f32; 6];

/// A key and the allocation holding its vertices, or the note that the
/// run is for the CPU's path. An entry that makes room for another key
/// leaves its allocation be: frames being shown may still read it.
#[derive(Clone, Copy, Debug, Default)]
struct Entry {
    key: RunKey,
    data_hash: u64,
    /// The mini hash of the bytes `data_hash` was checked with (`None`:
    /// none, the samples couldn't be read), the frame of that check, and
    /// the weight the run's vertices share (single-weight formats; its rows
    /// fold it in, `Transform::weighted_rows`).
    mini_hash: Option<u32>,
    checked: u32,
    weight: Option<f32>,
    /// Its allocation: `allocations[allocation]`, if that still has `serial`.
    allocation: u32,
    serial: u32,
    /// The entry hit right after this one the last time (+ 1; 0: none) and
    /// its allocation: prefetched at this one's hit (`prefetch_next`).
    next_entry: u32,
    next_allocation: u32,
    /// What the entry is: vertices in the arena or a note.
    kind: Kind,
    /// The recorded frame that found or stored it last.
    last_used: u32,
    valid: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Kind {
    /// The run's vertices are in `allocation`.
    #[default]
    Cached,
    /// The run couldn't go to the GPU (a vertex skinned on the CPU or
    /// behind the near plane): no allocation.
    CpuOnly,
    /// The run's bytes changed since it was cached: drawn from the frame
    /// without storing it again (data that changes every frame would fill
    /// the arena with runs never found again, and block the ring).
    Changing,
}

/// What `VertexCache::find` knows about a run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Lookup {
    /// Decoded before: its vertices are in the arena.
    Cached(Cached),
    /// Recently found to be one for the CPU's path (`mark_cpu_only`).
    CpuOnly,
    /// Its bytes change (from frame to frame): not worth storing.
    Changing,
    /// Not known (or its bytes changed).
    Unknown,
}

/// A run's place in the arena (the ring holds them oldest first).
#[derive(Clone, Copy, Debug, Default)]
struct Allocation {
    start: u32,
    count: u32,
    bounds: Bounds,
    /// The recorded frame (`VertexCache::frame`) that used it last.
    last_used: u32,
    /// Numbers allocations, so a reused ring slot is told apart.
    serial: u32,
    /// The entry that got it (it follows the record when that moves).
    entry: u32,
}

/// A cached run: where its vertices are, its bounds, and the weight its
/// vertices share (single-weight formats, as stored: the same bytes have
/// the same weight, so a hit needn't scan them for it).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cached {
    pub start: u32,
    pub count: u32,
    pub bounds: Bounds,
    pub weight: Option<f32>,
}

/// Lookups and stores of the current recorded frame, for the log.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub hits: u32,
    pub misses: u32,
    pub stored: u32,
    /// New runs not stored: the oldest allocation was still in use.
    pub full: u32,
    /// Runs known to go the CPU's path (`Lookup::CpuOnly`).
    pub cpu_only: u32,
    /// Runs whose bytes change (`Lookup::Changing`).
    pub changing: u32,
    /// Of `hits`: decided by the mini hash alone (`find_quick`).
    pub quick: u32,
    /// Of `hits`: runs already used earlier in the same frame (the same
    /// vertices drawn again).
    pub repeated: u32,
    /// Raw runs (`raw_arena::RawStats`): copied, their bytes, not fitting.
    pub raw_runs: u32,
    pub raw_bytes: u32,
    pub raw_full: u32,
    pub raw_lit: u32,
    /// Raw runs left in PSP RAM (zero copy) and their bytes.
    pub raw_psp_ram: u32,
    pub raw_psp_ram_bytes: u32,
}

pub struct VertexCache {
    /// The decoded vertices; read by the GPU (and `Frame::draw_corners`).
    arena: &'static mut [FrameVertex],
    entries: [Entry; ENTRIES],
    /// The ring's allocations: `allocations[(first + i) % ALLOCATIONS]`,
    /// i < `allocation_count`, oldest first.
    allocations: [Allocation; ALLOCATIONS],
    first: usize,
    allocation_count: usize,
    /// Where the next allocation goes.
    head: u32,
    /// Counts recorded frames (`next_frame`).
    frame: u32,
    next_serial: u32,
    /// A store found no room this frame (see `store`).
    full_this_frame: bool,
    /// The entry of the last hit or store (+ 1; 0: none), whose
    /// `next_entry` the next one becomes.
    last_hit: u32,
    pub stats: CacheStats,
    /// GPU memory for raw PSP vertices (`raw_arena`), if the display made
    /// it: runs drawn without decoding. It lives here because the vertex
    /// cache already goes wherever runs are recorded and counts the frames
    /// the GPU may still read; both are the GPU's vertex memory.
    pub raw: Option<RawArena>,
    /// The index ring (`Frame::ring_indices`), protected per frame as `raw`.
    pub index_ring: Option<RawArena>,
}

impl VertexCache {
    /// An empty cache over `arena` (at most `ARENA_VERTICES` are used).
    pub fn new(arena: &'static mut [FrameVertex]) -> VertexCache {
        let length = arena.len().min(ARENA_VERTICES);
        VertexCache {
            arena: &mut arena[..length],
            entries: [Entry::default(); ENTRIES],
            allocations: [Allocation::default(); ALLOCATIONS],
            first: 0,
            allocation_count: 0,
            head: 0,
            frame: PROTECTED_FRAMES,
            next_serial: 1,
            full_this_frame: false,
            last_hit: 0,
            stats: CacheStats::default(),
            raw: None,
            index_ring: None,
        }
    }

    /// Gives the cache the raw arena's memory (GPU memory on the Vita).
    pub fn set_raw_arena(&mut self, bytes: &'static mut [u8]) {
        self.raw = Some(RawArena::new(bytes));
    }

    /// Gives the cache the index ring's memory (GPU memory on the Vita).
    pub fn set_index_ring(&mut self, bytes: &'static mut [u8]) {
        self.index_ring = Some(RawArena::new(bytes));
    }

    /// Notes the hit (or store) of entry `index` as the successor of the
    /// last one, and prefetches the entry that followed it last time with
    /// its set and allocation: MTX draws its runs in about the same order
    /// every frame, and the tables (~1 MiB) miss the CPU's caches: a hit's
    /// lookup cost ~1,300 cycles, ~510 of them stalled on memory (Vita,
    /// 2026-11-08). Only a hint to the CPU: nothing changes if it guesses
    /// wrong. Live A/B (paired): MTX paused race recorder 30.5-30.6 ->
    /// 29.5-29.9 ms; race start 24.9 -> 26.3 and 22.4 -> 24.1 fps.
    fn prefetch_next(&mut self, index: usize) {
        if let Some(last) = (self.last_hit as usize).checked_sub(1) {
            self.entries[last].next_entry = index as u32 + 1;
            self.entries[last].next_allocation = self.entries[index].allocation;
        }
        self.last_hit = index as u32 + 1;
        let entry = &self.entries[index];
        if let Some(next) = (entry.next_entry as usize).checked_sub(1) {
            let set = next / WAYS * WAYS;
            if set + WAYS <= ENTRIES {
                prefetch(&self.entries[set..set + WAYS]);
            }
            if let Some(allocation) = self.allocations.get(entry.next_allocation as usize) {
                prefetch(core::slice::from_ref(allocation));
            }
        }
    }

    /// Makes an empty cache at `place` without building it on the stack
    /// first (it is ~400 KiB; the Vita's capture thread has 64 KiB).
    ///
    /// # Safety
    /// `place` must be writable memory for a `VertexCache`, all zero bytes
    /// (every table entry then reads as empty).
    pub unsafe fn init_zeroed(place: *mut VertexCache, arena: &'static mut [FrameVertex]) {
        use core::ptr::addr_of_mut;
        let length = arena.len().min(ARENA_VERTICES);
        addr_of_mut!((*place).arena).write(&mut arena[..length]);
        addr_of_mut!((*place).frame).write(PROTECTED_FRAMES);
        addr_of_mut!((*place).next_serial).write(1);
        addr_of_mut!((*place).raw).write(None);
        addr_of_mut!((*place).index_ring).write(None);
    }

    /// The arena, for the frame (`Frame::vertex_cache`).
    pub fn arena(&self) -> &[FrameVertex] {
        self.arena
    }

    /// A recorded frame is finished: its runs are protected for the next
    /// `PROTECTED_FRAMES - 1` frames. Returns the frame's statistics.
    pub fn next_frame(&mut self) -> CacheStats {
        self.frame = self.frame.wrapping_add(1);
        self.full_this_frame = false;
        if let Some(ring) = self.index_ring.as_mut() {
            ring.next_frame();
        }
        if let Some(raw) = self.raw.as_mut() {
            let stats = raw.next_frame();
            self.stats.raw_runs = stats.runs;
            self.stats.raw_bytes = stats.bytes;
            self.stats.raw_full = stats.full;
            self.stats.raw_lit = stats.lit;
            self.stats.raw_psp_ram = stats.psp_ram;
            self.stats.raw_psp_ram_bytes = stats.psp_ram_bytes;
        }
        core::mem::take(&mut self.stats)
    }

    /// The first entry of `key`'s set.
    fn set(key: &RunKey) -> usize {
        key.hash() as usize % (ENTRIES / WAYS) * WAYS
    }

    /// True if `entry` still means something: a CPU-only note not too old,
    /// or an allocation that wasn't overwritten.
    fn is_live(&self, entry: &Entry) -> bool {
        entry.valid
            && match entry.kind {
                Kind::Cached => self.allocations[entry.allocation as usize].serial == entry.serial,
                Kind::CpuOnly | Kind::Changing => self.frame.wrapping_sub(entry.last_used) < NOTE_FRAMES,
            }
    }

    /// A hit decided without the run's bytes: `key` cached with the same
    /// mini hash, its bytes hashed in full fewer than `FULL_CHECK_FRAMES`
    /// frames ago. Returns the cached run and the weight its vertices share;
    /// `None`: ask `find` with the full hash. Marks the run used.
    pub fn find_quick(&mut self, key: &RunKey, mini_hash: u32) -> Option<(Cached, Option<f32>)> {
        let set = Self::set(key);
        for index in set..set + WAYS {
            let entry = self.entries[index];
            if entry.key != *key || entry.kind != Kind::Cached || !self.is_live(&entry) {
                continue;
            }
            if entry.mini_hash != Some(mini_hash) || self.frame.wrapping_sub(entry.checked) >= FULL_CHECK_FRAMES {
                return None;
            }
            self.stats.repeated += (entry.last_used == self.frame) as u32;
            self.entries[index].last_used = self.frame;
            let allocation = &mut self.allocations[entry.allocation as usize];
            allocation.last_used = self.frame;
            self.stats.hits += 1;
            self.stats.quick += 1;
            let cached = Cached { start: allocation.start, count: allocation.count, bounds: allocation.bounds, weight: entry.weight };
            self.prefetch_next(index);
            return Some((cached, entry.weight));
        }
        None
    }

    /// What is known about the run `key` with these bytes; a cached run is
    /// marked as used by this frame (protecting its vertices) and checked
    /// now (`mini_hash` and `weight` noted for `find_quick`).
    pub fn find(&mut self, key: &RunKey, data_hash: u64, mini_hash: Option<u32>) -> Lookup {
        let set = Self::set(key);
        for index in set..set + WAYS {
            let entry = self.entries[index];
            if !(entry.key == *key && self.is_live(&entry)) {
                continue;
            }
            match entry.kind {
                Kind::Changing => {
                    self.stats.changing += 1;
                    return Lookup::Changing;
                }
                _ if entry.data_hash != data_hash => {
                    // Cached (or noted) with other bytes: the data changes.
                    self.entries[index] = Entry { kind: Kind::Changing, last_used: self.frame, data_hash, ..entry };
                    self.stats.changing += 1;
                    return Lookup::Changing;
                }
                Kind::CpuOnly => {
                    self.stats.cpu_only += 1;
                    return Lookup::CpuOnly;
                }
                Kind::Cached => {}
            }
            self.stats.repeated += (entry.last_used == self.frame) as u32;
            self.entries[index].last_used = self.frame;
            self.entries[index].mini_hash = mini_hash;
            self.entries[index].checked = self.frame;
            let allocation = &mut self.allocations[entry.allocation as usize];
            allocation.last_used = self.frame;
            self.stats.hits += 1;
            let cached = (allocation.start, allocation.count, allocation.bounds);
            self.prefetch_next(index);
            let (start, count, bounds) = cached;
            return Lookup::Cached(Cached { start, count, bounds, weight: entry.weight });
        }
        self.stats.misses += 1;
        Lookup::Unknown
    }

    /// Notes that the run `key` with these bytes goes the CPU's path, so
    /// the next frames don't try the GPU's first.
    pub fn mark_cpu_only(&mut self, key: &RunKey, data_hash: u64) {
        let index = self.entry_for(key);
        self.entries[index] =
            Entry { key: *key, data_hash, allocation: 0, serial: 0, kind: Kind::CpuOnly, last_used: self.frame, valid: true, ..Entry::default() };
    }

    /// The entry to (re)use for `key`: its own, else a dead one of its set,
    /// else the least recently used.
    fn entry_for(&self, key: &RunKey) -> usize {
        let set = Self::set(key);
        let ways = set..set + WAYS;
        if let Some(own) = ways.clone().find(|&index| self.entries[index].valid && self.entries[index].key == *key) {
            return own;
        }
        if let Some(dead) = ways.clone().find(|&index| !self.is_live(&self.entries[index])) {
            return dead;
        }
        ways.max_by_key(|&index| self.frame.wrapping_sub(self.entries[index].last_used)).unwrap_or(set)
    }

    /// Stores a run's decoded vertices; `None` if there is no room right
    /// now (the oldest allocation is still used by a frame being shown).
    #[allow(clippy::too_many_arguments)]
    pub fn store(
        &mut self,
        key: &RunKey,
        data_hash: u64,
        mini_hash: Option<u32>,
        weight: Option<f32>,
        vertices: &[FrameVertex],
        bounds: Bounds,
    ) -> Option<Cached> {
        let count = vertices.len() as u32;
        if count == 0 || count as usize > self.arena.len() || self.full_this_frame {
            self.stats.full += 1;
            return None;
        }
        // No record left: free one, passing over those still in use (as
        // `allocate` does with the arena).
        let mut passes = self.allocation_count.min(MAX_PASSES);
        let mut room = true;
        while room && self.allocation_count == ALLOCATIONS {
            room = self.retire_front(&mut passes).is_some();
        }
        let start = if room { self.allocate(count) } else { None };
        let Some(start) = start else {
            // Full of runs this frame uses: no use trying again this frame
            // (searching the whole ring for every new run cost ~250 ms per
            // frame in MTX's race with a 1 MiB arena).
            self.full_this_frame = true;
            self.stats.full += 1;
            return None;
        };
        self.arena[start as usize..(start + count) as usize].copy_from_slice(vertices);
        let serial = self.next_serial;
        self.next_serial = self.next_serial.wrapping_add(1).max(1);
        let index = (self.first + self.allocation_count) % ALLOCATIONS;
        let entry = self.entry_for(key);
        self.allocations[index] = Allocation { start, count, bounds, last_used: self.frame, serial, entry: entry as u32 };
        self.allocation_count += 1;
        self.entries[entry] = Entry {
            key: *key,
            data_hash,
            mini_hash,
            checked: self.frame,
            weight,
            allocation: index as u32,
            serial,
            kind: Kind::Cached,
            last_used: self.frame,
            valid: true,
            next_entry: self.entries[entry].next_entry,
            next_allocation: self.entries[entry].next_allocation,
        };
        self.stats.stored += 1;
        self.prefetch_next(entry);
        Some(Cached { start, count, bounds, weight })
    }

    /// Room for `count` vertices at the head (wrapping to the arena's
    /// start if they don't fit before its end). The allocations in the way
    /// (the oldest; the ring keeps them in address order) are evicted, or,
    /// if a recent frame still uses them, passed over: the head moves past
    /// them and their record to the back, so runs used every frame don't
    /// block the ring ("second chance"). `None` after `MAX_PASSES`.
    fn allocate(&mut self, count: u32) -> Option<u32> {
        let length = self.arena.len() as u32;
        if count > WINDOW_VERTICES {
            return None;
        }
        let mut passes = self.allocation_count.min(MAX_PASSES);
        loop {
            // A run stays inside one window (16-bit indices from its start)
            // and inside the arena: past the end of either, the head moves on
            // to the next window, or back to the arena's start.
            let limit = (window_of(self.head) + WINDOW_VERTICES).min(length);
            if self.head + count > limit {
                // What lies from the head to that end goes first.
                if self.allocation_count > 0 {
                    let front = self.allocations[self.first];
                    if front.start >= self.head && front.start < limit {
                        self.retire_front(&mut passes)?;
                        continue;
                    }
                }
                self.head = if limit >= length { 0 } else { limit };
                continue;
            }
            if self.allocation_count == 0 {
                break;
            }
            let front = self.allocations[self.first];
            if front.start < self.head || front.start >= self.head + count {
                break;
            }
            self.retire_front(&mut passes)?;
        }
        let start = self.head;
        self.head += count;
        Some(start)
    }

    /// The front allocation (the next one at or after the head): evicted,
    /// or passed over if a recent frame uses it (see `allocate`). `None`
    /// once `passes` are used up.
    fn retire_front(&mut self, passes: &mut usize) -> Option<()> {
        if self.evict_oldest() {
            return Some(());
        }
        if *passes == 0 {
            return None;
        }
        *passes -= 1;
        let front = self.allocations[self.first];
        self.head = front.start + front.count;
        self.first = (self.first + 1) % ALLOCATIONS;
        let back = (self.first + self.allocation_count - 1) % ALLOCATIONS;
        self.allocations[back] = front;
        let entry = &mut self.entries[front.entry as usize];
        if entry.valid && entry.kind == Kind::Cached && entry.serial == front.serial {
            entry.allocation = back as u32;
        }
        Some(())
    }

    /// Drops the oldest allocation (its entry, if any, no longer finds it:
    /// the slot's serial changes when reused). False if a recent frame
    /// still uses it.
    fn evict_oldest(&mut self) -> bool {
        let oldest = &mut self.allocations[self.first];
        if self.frame.wrapping_sub(oldest.last_used) < PROTECTED_FRAMES {
            return false;
        }
        oldest.serial = 0;
        self.first = (self.first + 1) % ALLOCATIONS;
        self.allocation_count -= 1;
        true
    }

    /// Forgets every entry and allocation (e.g. when a new game starts).
    /// The arena keeps its contents; frames still shown may read them.
    pub fn clear(&mut self) {
        // Allocations can't be reused before frames that read them are gone:
        // simplest is to keep the ring and only drop the entries' validity.
        for entry in self.entries.iter_mut() {
            entry.valid = false;
        }
    }
}

/// The bounds of model-space vertices (their x, y, z).
pub fn bounds_of(vertices: &[FrameVertex]) -> Bounds {
    let mut bounds = [f32::MAX, f32::MAX, f32::MAX, f32::MIN, f32::MIN, f32::MIN];
    for vertex in vertices {
        bounds[0] = bounds[0].min(vertex.x);
        bounds[1] = bounds[1].min(vertex.y);
        bounds[2] = bounds[2].min(vertex.z);
        bounds[3] = bounds[3].max(vertex.x);
        bounds[4] = bounds[4].max(vertex.y);
        bounds[5] = bounds[5].max(vertex.z);
    }
    bounds
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache(vertices: usize) -> Box<VertexCache> {
        let arena: &'static mut [FrameVertex] = Box::leak(vec![FrameVertex::default(); vertices].into_boxed_slice());
        Box::new(VertexCache::new(arena))
    }

    fn key(address: u32) -> RunKey {
        RunKey { address, vertex_count: 4, format: 0x11A, ..RunKey::default() }
    }

    fn vertices(n: usize, x: f32) -> Vec<FrameVertex> {
        (0..n).map(|i| FrameVertex { x: x + i as f32, ..FrameVertex::default() }).collect()
    }

    #[test]
    fn the_mini_hash_finds_a_run_until_the_next_full_check() {
        let mut cache = cache(256);
        let stored = cache.store(&key(0x100), 7, Some(42), Some(2.0), &vertices(4, 1.0), [0.0; 6]).unwrap();
        // Same samples: found without the bytes, with the run's weight.
        assert_eq!(cache.find_quick(&key(0x100), 42), Some((stored, Some(2.0))));
        // Other samples: the bytes must be checked.
        assert_eq!(cache.find_quick(&key(0x100), 43), None);
        // After FULL_CHECK_FRAMES frames the bytes are checked again, which
        // renews the mini hash.
        for _ in 0..FULL_CHECK_FRAMES {
            cache.next_frame();
        }
        assert_eq!(cache.find_quick(&key(0x100), 42), None);
        assert_eq!(cache.find(&key(0x100), 7, Some(42)), Lookup::Cached(stored));
        assert_eq!(cache.find_quick(&key(0x100), 42), Some((stored, Some(2.0))));
        // No mini hash (samples unreadable): never decided by it.
        cache.store(&key(0x200), 8, None, None, &vertices(4, 1.0), [0.0; 6]).unwrap();
        assert_eq!(cache.find_quick(&key(0x200), 0), None);
    }

    #[test]
    fn samples_are_spread_over_the_run() {
        assert_eq!(positions_for(8, 1).collect::<Vec<_>>(), [0]);
        assert_eq!(positions_for(8, 3).collect::<Vec<_>>(), [0, 1, 2]);
        assert_eq!(positions_for(8, 15).collect::<Vec<_>>(), [0, 2, 4, 6, 8, 10, 12, 14]);
        assert_eq!(positions_for(DEFAULT_SAMPLES, 16).collect::<Vec<_>>(), [0, 5, 10, 15]);
    }

    #[test]
    fn bursts_sample_the_first_and_last_words() {
        let positions = |words: usize| positions_for(SAMPLE_BURSTS, words).collect::<Vec<_>>();
        assert_eq!(positions(5), [0, 1, 2, 3, 4]);
        assert_eq!(positions(16), (0..16).collect::<Vec<_>>());
        assert_eq!(positions(40), [0, 1, 2, 3, 4, 5, 6, 7, 32, 33, 34, 35, 36, 37, 38, 39]);
    }

    #[test]
    fn a_cache_made_in_zeroed_memory_is_an_empty_one() {
        let arena: &'static mut [FrameVertex] = Box::leak(vec![FrameVertex::default(); 64].into_boxed_slice());
        let layout = std::alloc::Layout::new::<VertexCache>();
        // Safe: zeroed memory of the right layout, as init_zeroed asks.
        let cache = unsafe {
            let place = std::alloc::alloc_zeroed(layout) as *mut VertexCache;
            VertexCache::init_zeroed(place, arena);
            &mut *place
        };
        assert_eq!(cache.find(&key(0x100), 7, None), Lookup::Unknown);
        let stored = cache.store(&key(0x100), 7, None, None, &vertices(4, 1.0), [0.0; 6]).unwrap();
        assert_eq!(cache.find(&key(0x100), 7, None), Lookup::Cached(stored));
    }

    /// 16-bit indices count from a run's window: no run may cross a
    /// window's end (the head moves on to the next window instead).
    #[test]
    fn runs_never_cross_a_window() {
        let mut cache = cache(2 * WINDOW_VERTICES as usize);
        let mut second_window = 0;
        for n in 0..6000u32 {
            let count = 7 + n % 57;
            if let Some(stored) = cache.store(&key(0x1000 + 16 * n), n as u64, None, None, &vertices(count as usize, 1.0), [0.0; 6]) {
                assert_eq!(window_of(stored.start), window_of(stored.start + count - 1), "run {n} at {}", stored.start);
                assert!(stored.start + count <= 2 * WINDOW_VERTICES);
                second_window += (stored.start >= WINDOW_VERTICES) as u32;
            }
            if n % 200 == 199 {
                for _ in 0..PROTECTED_FRAMES + 1 {
                    cache.next_frame();
                }
            }
        }
        assert!(second_window > 0, "the second window was used");
    }

    #[test]
    fn a_stored_run_is_found_with_the_same_bytes_only() {
        let mut cache = cache(64);
        assert_eq!(cache.find(&key(0x100), 7, None), Lookup::Unknown);
        let stored = cache.store(&key(0x100), 7, None, None, &vertices(4, 1.0), [0.0; 6]).unwrap();
        assert_eq!(cache.arena()[stored.start as usize + 2].x, 3.0);
        assert_eq!(cache.find(&key(0x100), 7, None), Lookup::Cached(stored));
        assert_eq!(cache.find(&RunKey { colors: 1, ..key(0x100) }, 7, None), Lookup::Unknown, "decoded another way");
        assert_eq!(cache.find(&key(0x100), 8, None), Lookup::Changing, "bytes changed");
        assert_eq!((cache.stats.hits, cache.stats.misses, cache.stats.stored, cache.stats.changing), (1, 2, 1, 1));
    }

    #[test]
    fn the_ring_overwrites_old_runs_but_not_recent_ones() {
        let mut cache = cache(16);
        // Two runs of 8 fill the arena; a third needs the oldest's place,
        // which this frame still uses.
        cache.store(&key(0x100), 1, None, None, &vertices(8, 0.0), [0.0; 6]).unwrap();
        cache.store(&key(0x200), 2, None, None, &vertices(8, 0.0), [0.0; 6]).unwrap();
        assert!(cache.store(&key(0x300), 3, None, None, &vertices(8, 0.0), [0.0; 6]).is_none());
        assert_eq!(cache.stats.full, 1);
        // Frames later, run 0x200 still used, 0x100 not: 0x100's place is taken.
        for _ in 0..PROTECTED_FRAMES {
            cache.next_frame();
            assert!(matches!(cache.find(&key(0x200), 2, None), Lookup::Cached(_)));
        }
        let third = cache.store(&key(0x300), 3, None, None, &vertices(8, 9.0), [0.0; 6]).unwrap();
        assert_eq!(third.start, 0);
        assert_eq!(cache.find(&key(0x100), 1, None), Lookup::Unknown);
        assert!(matches!(cache.find(&key(0x200), 2, None), Lookup::Cached(_)));
        // A run that doesn't fit before the end wraps around.
        for _ in 0..PROTECTED_FRAMES {
            cache.next_frame();
        }
        let fourth = cache.store(&key(0x400), 4, None, None, &vertices(12, 0.0), [0.0; 6]).unwrap();
        assert_eq!(fourth.start, 0);
        assert_eq!(cache.find(&key(0x300), 3, None), Lookup::Unknown);
        assert_eq!(cache.find(&key(0x200), 2, None), Lookup::Unknown);
    }

    #[test]
    fn many_keys_of_one_set_keep_the_recent_ones_and_cpu_notes_expire() {
        let mut cache = cache(1024);
        // More keys than ways, all found again every frame: none is lost
        // while there are fewer than WAYS per set (keys spread over sets).
        for n in 0..64 {
            cache.store(&key(0x1000 + 16 * n), n as u64, None, None, &vertices(4, 0.0), [0.0; 6]).unwrap();
        }
        let found = (0..64).filter(|&n| matches!(cache.find(&key(0x1000 + 16 * n), n as u64, None), Lookup::Cached(_))).count();
        assert_eq!(found, 64);
        cache.mark_cpu_only(&key(0x9000), 1);
        assert_eq!(cache.find(&key(0x9000), 1, None), Lookup::CpuOnly);
        for _ in 0..NOTE_FRAMES {
            cache.next_frame();
        }
        assert_eq!(cache.find(&key(0x9000), 1, None), Lookup::Unknown, "tried again later");
    }

    #[test]
    fn runs_used_every_frame_are_passed_over_not_in_the_way() {
        let mut cache = cache(32);
        // A run used every frame at the arena's start, then runs used once.
        let hot = cache.store(&key(0x100), 1, None, None, &vertices(8, 0.0), [0.0; 6]).unwrap();
        for n in 0..3 {
            cache.store(&key(0x200 + 16 * n), 2, None, None, &vertices(8, 0.0), [0.0; 6]).unwrap();
        }
        for round in 0..4u32 {
            for _ in 0..PROTECTED_FRAMES {
                cache.next_frame();
                assert_eq!(cache.find(&key(0x100), 1, None), Lookup::Cached(hot), "round {round}");
            }
            // New runs keep finding room around the hot one.
            let new = cache.store(&key(0x1000 + round), 3, None, None, &vertices(8, 0.0), [0.0; 6]).expect("room");
            assert!(new.start >= 8, "never over the hot run");
        }
    }

    #[test]
    fn with_every_record_taken_an_unused_one_is_freed() {
        let mut cache = cache(ALLOCATIONS * 2);
        let hot = cache.store(&key(0x100), 1, None, None, &vertices(1, 0.0), [0.0; 6]).unwrap();
        for n in 1..ALLOCATIONS as u32 {
            cache.store(&key(0x100 + 16 * n), 2, None, None, &vertices(1, 0.0), [0.0; 6]).unwrap();
        }
        for _ in 0..PROTECTED_FRAMES {
            cache.next_frame();
            assert!(matches!(cache.find(&key(0x100), 1, None), Lookup::Cached(_)));
        }
        // The oldest record is the hot run's; the next one is free to go.
        assert!(cache.store(&key(0x9_0000), 3, None, None, &vertices(1, 0.0), [0.0; 6]).is_some());
        assert_eq!(cache.find(&key(0x100), 1, None), Lookup::Cached(hot));
    }

    #[test]
    fn a_run_whose_bytes_change_is_not_stored_again() {
        let mut cache = cache(1024);
        cache.store(&key(0x100), 1, None, None, &vertices(4, 0.0), [0.0; 6]).unwrap();
        assert_eq!(cache.find(&key(0x100), 2, None), Lookup::Changing);
        assert_eq!(cache.find(&key(0x100), 3, None), Lookup::Changing, "whatever the bytes, for a while");
        for _ in 0..NOTE_FRAMES {
            cache.next_frame();
        }
        assert_eq!(cache.find(&key(0x100), 3, None), Lookup::Unknown);
    }

    #[test]
    fn data_hash_sees_order_and_values() {
        assert_ne!(data_hash(&[1, 2]), data_hash(&[2, 1]));
        assert_ne!(data_hash(&[1, 2]), data_hash(&[1, 3]));
        assert_eq!(data_hash(&[5, 6, 7]), data_hash(&[5, 6, 7]));
    }
}
