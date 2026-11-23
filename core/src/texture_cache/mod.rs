//! The display thread's texture cache: the decoded GPU texture for each
//! texture a frame uses, and when a texture has to be decoded again.
//!
//! A PSP texture is pixel data in PSP memory plus a description
//! ([`TextureDescriptor`]); the game can change the pixels at any time
//! without telling anyone. Decoding every texture every frame is far too
//! slow, so decoded textures are kept, keyed by their description, and their
//! data is re-checked now and then:
//!
//! - New textures are decoded at once (unless this frame's texture work is
//!   past its deadline, or their VRAM copy hasn't arrived: then the frame
//!   waits, see [`Lookup::Waiting`]).
//! - A cached texture is re-checked (copied out of PSP memory and hashed) at
//!   most every [`VERIFY_INTERVAL`] frames, and at most
//!   [`VERIFY_BYTES_PER_FRAME`] bytes are re-checked per frame. A changed
//!   texture therefore shows up a few frames late.
//! - Textures that changed at their last re-check are re-checked at every
//!   use; ones that changed at several re-checks in a row are *volatile* and
//!   decoded again without hashing first (see `look_up`).
//! - Big direct-colour textures are compared by a sample of their data at
//!   every use ([`staging::sample_hash`]) and re-checked by a worker thread
//!   ([`refresh`]); all others (in RAM) by a few words the capture sampled
//!   while the GE drew the frame (`Frame::texture_samples`), compared with
//!   the same words of the data they were decoded from
//!   ([`staging::quick_hash`]): a mismatch re-checks them at once.
//!
//! The pieces:
//! - [`gpu`]: GPU texture handles, formats and sizes.
//! - `memory`: the memory budget, retired textures, evicting old ones.
//! - [`arena`]: the texture arena most textures' memory comes from.
//! - [`staging`]: copying texture data out of PSP memory and hashing it.
//! - [`refresh`]: re-checking big textures on a worker thread.
//!
//! Everything that touches the Vita (GPU textures, PSP memory, the clock,
//! the worker thread, the log) goes through [`Host`], so all of this runs
//! in PC tests (`tests.rs` replays bugs that were fixed before).

pub mod arena;
pub mod gpu;
mod memory;
pub mod refresh;
pub mod staging;
#[cfg(test)]
mod tests;

use core::fmt;
use core::sync::atomic::{AtomicBool, Ordering};

use crate::frame::{Frame, TextureDescriptor, MAX_TEXTURES};
use crate::texture::{self, format, DecodeError};
use crate::vram_requests::{VRAM_SIZE, VRAM_START};

use gpu::{can_unswizzle_on_gpu, GpuFormat, GpuTexture, TextureShape};
use arena::PageArena;
use memory::Retired;
use refresh::{PendingRefresh, RefreshJob, RefreshResult};
use staging::{quick_hash, sample_hash, stage, staged_quick_hash, FrameHash, StagedLayout};

pub use memory::{ARENA_IDLE_FRAMES, ARENA_SLACK, BUDGET_BYTES, MEMORY_MINIMUM, MEMORY_RESERVE, RETIRE_FRAMES, SPARE_FRAMES};

/// Final Fantasy's world map uses ~160 textures per frame (32x32 windows
/// into its tile set); with 128 entries every frame decoded them all again.
pub const CACHE_SIZE: usize = 512;
/// A cached texture is re-checked at most this often (frames).
pub const VERIFY_INTERVAL: u32 = 8;
/// Changed at this many re-checks in a row: volatile.
pub const VOLATILE_CHANGES: u8 = 2;
/// Volatile textures are hashed only every this many decodes.
pub const HASH_VOLATILE_EVERY: u32 = 4;
/// Buckets of the lookup hint table (see `find`): a power of two, several
/// times `CACHE_SIZE`, so that few textures share a bucket.
const HINT_BUCKETS: usize = 4096;
/// Whether cached textures are compared with the frame's capture-time
/// samples (`Frame::texture_samples`; on by default).
pub static QUICK_CHECKS: AtomicBool = AtomicBool::new(true);
/// Re-check budget per frame.
pub const VERIFY_BYTES_PER_FRAME: u32 = 128 * 1024;
/// Textures this big (and direct colour) are sampled at every use and
/// re-checked by the worker; smaller ones take < 1 ms on the display thread.
pub const BACKGROUND_MIN_BYTES: u32 = 64 * 1024;
/// Texture work per frame: once this long has passed since the display loop
/// started, new textures wait for the next frame and re-checks are skipped.
/// Final Fantasy's battle start brings ~25 new textures at once (a 50 ms
/// frame); instead our last picture is shown for a frame while they are
/// decoded. A deadline rather than a fixed 8 ms of texture work: frames
/// usually arrive as the loop starts, and drawing them takes 1-3 ms more,
/// so 12 ms still makes the refresh at 16.7 ms. With 8 ms the battle's
/// appearance fell back to the original picture for one frame (3 textures
/// left).
pub const TEXTURE_DEADLINE_US: u32 = 12_000;
/// At most this many frames in a row wait for textures past the deadline,
/// so a game that always needs more never falls back for good.
pub const MAX_DEFERRED_FRAMES: u32 = 3;
const FRAME_HASHES: usize = 64;
/// `FrameStart::trace_address` for tracing all VRAM textures ("tracetex 1").
pub const TRACE_VRAM: u32 = 1;
/// Textures that failed to decode are remembered so we don't retry (and
/// log) them every frame; retried when their pixel data changes.
const FAILED_TEXTURES: usize = 32;
/// A texture that failed for lack of memory (or room in the cache) is tried
/// again after this many frames (with the same data; changed data at once).
/// Remembered for good, MTX's HUD stayed missing for the rest of a race
/// after memory was short once (2026-11-09: one 64x64 CLUT8 texture, 40 draws).
const RETRY_FAILED_FRAMES: u32 = 30;

/// Everything the cache needs from the Vita (or from a test).
///
/// The worker thread uses one too (in [`refresh::run_refresh`]): only
/// `psp_memory`, `copy`, `decode`, `captured_frame_number` and `now_us`.
pub trait Host {
    /// Bytes of the texture arena (see [`arena`]), which the host makes on
    /// the first call; 0 if there is none (then every texture gets memory
    /// of its own). Asked once.
    fn texture_arena_bytes(&mut self) -> u32;
    /// A new GPU texture of this shape, or None if it can't be made. Its
    /// memory: `shape.bytes()` at byte `arena_offset` of the texture arena,
    /// or with None a memory block of its own.
    fn create_texture(&mut self, shape: &TextureShape, arena_offset: Option<u32>) -> Option<GpuTexture>;
    fn free_texture(&mut self, texture: GpuTexture);
    /// Decodes `data` (staged as texture `as_texture`) into `texture`, which
    /// has format `format`: palette indices, 16-bit texels, or RGBA with
    /// `palette` (the texture's own palette words) and, with `smooth`,
    /// transparent texels bled (see `texture::bleed_transparent_texels`).
    /// `level`: the mip level `as_texture` is (0, or a level of a texture
    /// with levels, `TextureShape::levels`): written at its place in the
    /// texture (`texture::mip_level_offset`).
    fn decode(
        &mut self,
        texture: GpuTexture,
        format: GpuFormat,
        as_texture: &TextureDescriptor,
        data: &[u8],
        palette: &[u32],
        smooth: bool,
        level: u8,
    ) -> Result<(), DecodeError>;
    /// True if the GPU unswizzle can run for textures of this size now
    /// (its vertices, program and depth surface exist or could be made).
    fn gpu_unswizzle_ready(&mut self, width: u16, height: u16) -> bool;
    /// Copies the swizzled data of `wanted` into `raw` (a render-target
    /// texture of `target`'s shape) and has the GPU unswizzle it into
    /// `target`, in a scene of its own before the frame's. False if nothing
    /// could be submitted.
    fn unswizzle_on_gpu(&mut self, wanted: &TextureDescriptor, target: GpuTexture, raw: GpuTexture) -> bool;
    /// Waits until the GPU has finished everything submitted.
    fn wait_for_gpu(&mut self);
    /// Free main memory for the app, in bytes.
    fn free_memory(&self) -> u32;

    /// PSP memory at physical `address` (VRAM: the plugin's copy of it), or
    /// None if it isn't mapped. The game may write it while we read.
    fn psp_memory(&self, address: u32, size: u32) -> Option<&[u8]>;
    /// The same as words (`address` is word-aligned).
    fn psp_words(&self, address: u32, size: u32) -> Option<&[u32]>;
    /// Copies out of PSP memory, fast (DMA for big copies on the Vita).
    fn copy(&self, target: &mut [u8], source: &[u8]);
    /// The shared VRAM copy request table.
    fn vram(&self) -> VramState;
    /// The frame number of the last frame the capture completed.
    fn captured_frame_number(&self) -> u32;

    /// True if there is a worker thread for [`refresh`].
    fn has_refresh_worker(&self) -> bool;
    /// Hands the (idle) worker a job.
    fn start_refresh(&mut self, job: &RefreshJob);
    /// The worker's result once it is done with the job, until `end_refresh`.
    fn finished_refresh(&self) -> Option<RefreshResult>;
    /// The result has been used: the worker is idle again.
    fn end_refresh(&mut self);

    /// Microseconds since some fixed point.
    fn now_us(&self) -> u64;
    /// Appends a line to the render log.
    fn log(&mut self, text: fmt::Arguments);
}

/// The PSP plugin's VRAM copies (`psp_link::VramRequestTable`, see
/// `vram_requests`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VramState {
    /// The table we last asked for, and the one the plugin last copied.
    pub generation: u32,
    pub copied_generation: u32,
    /// VRAM blocks the plugin found changed so far.
    pub changes: u32,
}

impl VramState {
    /// True while the plugin hasn't yet copied the VRAM ranges we last asked
    /// for: VRAM textures must not be decoded (or re-checked) then, the copy
    /// area may still hold old or no data. Takes one frame flip.
    pub fn copy_pending(&self) -> bool {
        self.copied_generation != self.generation
    }
}

/// What a frame gets for one of its textures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lookup {
    Ready(GpuTexture),
    /// Not for now: its VRAM copy hasn't arrived yet, or this frame's
    /// texture work is past its deadline. The frame should wait rather than
    /// be drawn with holes.
    Waiting,
    /// Can't be decoded (unsupported, out of memory, cache full).
    Unavailable,
    /// A PSP framebuffer we draw ourselves (one of `FrameStart::render_targets`):
    /// the display side uses its render target instead (render to texture,
    /// e.g. MTX Mototrax's screen effects; decoding the 1 MiB framebuffer
    /// from VRAM took ~20 ms every frame and gave the PSP's resolution).
    RenderTarget,
}

/// How many render targets `FrameStart::render_targets` can name: the
/// display side's two screen targets and six off-screen ones (8 since
/// 2026-11-22, was 4: God of War's title names six, and the two left out
/// were decoded from VRAM every frame).
pub const MAX_RENDER_TARGETS: usize = 8;

/// True if `texture` is a framebuffer the display side draws itself, or a
/// tile inside one (`offscreen::placement`: the display reads its part of
/// that target). Index formats aren't: a framebuffer read as palette
/// indices means something else than its colours.
fn is_render_target(start: &FrameStart, textures: &[TextureDescriptor], texture: &TextureDescriptor) -> bool {
    if texture.address == 0 || texture::is_palette_format(texture.format) {
        return false;
    }
    if start.render_targets.contains(&texture.address) {
        return true;
    }
    let (base, x) = crate::offscreen::placement(textures, texture.address);
    x > 0 && start.render_targets.contains(&base)
}

/// Set once per frame by the C side.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameStart {
    /// The PSP frame being drawn (`Frame::frame_number`).
    pub frame_number: u32,
    /// Counts drawn frames; the ages in this module are in these.
    pub render_count: u32,
    /// Menu option "NativeRes Texture Filter": 1 = smooth.
    pub smooth: bool,
    /// Log every lookup of the
    /// textures at this address. 0 = off, [`TRACE_VRAM`]: waits and decodes
    /// of all VRAM textures, and frames drawn again.
    pub trace_address: u32,
    /// When the display loop started this frame (low 32 bits of `now_us`),
    /// for [`TEXTURE_DEADLINE_US`].
    pub loop_start_us: u32,
    /// PSP framebuffers (VRAM addresses) the display side has drawn into
    /// render targets of its own, 0 = unused. Textures there are
    /// [`Lookup::RenderTarget`]. Not the framebuffer the frame draws into.
    pub render_targets: [u32; MAX_RENDER_TARGETS],
}

/// Counters of the texture cache's work.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TextureStats {
    pub decode_us: u64,
    /// Worker time for the refreshes.
    pub refresh_us: u64,
    pub hashed_bytes: u64,
    pub decodes: u32,
    /// Textures refreshed by the worker thread.
    pub refreshes: u32,
    /// Textures that couldn't be allocated.
    pub create_failures: u32,
    /// Bytes of GPU textures now (cached + retired); not a counter.
    pub texture_bytes: u32,
    /// Cached textures retired to make room or memory (not replaced ones).
    pub evictions: u32,
    /// Bytes of the texture arena in use now; not a counter.
    pub arena_bytes: u32,
    /// `find` calls, and of them those whose hint missed (a scan of all
    /// entries).
    pub finds: u32,
    pub scans: u32,
    /// Decodes whose data no longer matched the frame's capture-time sample.
    pub moved_on: u32,
}

/// Where this frame's texture time went, for the slow-frame log line
/// (mirrored in C as NativeResTextureTimes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameTimes {
    pub trim_us: u64,
    pub hash_us: u64,
    pub allocate_us: u64,
    pub decode_us: u64,
}

/// One cached texture.
#[derive(Clone, Copy)]
struct Entry {
    /// `palette_offset` is ignored when comparing, and index textures'
    /// whole palette (see `same_texture`).
    key: TextureDescriptor,
    /// Hash of its data when it was last hashed.
    data_hash: u32,
    /// None: a free slot.
    texture: Option<GpuTexture>,
    shape: TextureShape,
    /// Render count of the last frame that used it.
    last_used: u32,
    /// Render count when its data was last re-checked.
    verified_at: u32,
    /// Decoded for the smooth filter.
    smooth: bool,
    /// Re-checks in a row that found new pixels.
    changes: u8,
    /// Decodes in a row without hashing.
    unhashed: u8,
    /// Big direct colour: `sample_hash` of the data it was decoded from.
    sample_hash: u32,
    /// `quick_hash` of the data it was decoded from (0 = unknown: swapped
    /// in by the worker), compared with the frame's capture-time sample
    /// (`lookup_sample`).
    quick: u32,
    /// Rows from the top whose decoded texels all have alpha 255
    /// (`texture::opaque_rows`; with mip levels `ALL_ROWS_OPAQUE` or 0):
    /// `TextureCache::opaque_rows`.
    opaque_rows: u16,
}

const NO_DESCRIPTOR: TextureDescriptor = TextureDescriptor {
    address: 0,
    buffer_width: 0,
    width: 0,
    height: 0,
    format: 0,
    swizzled: 0,
    flags: 0,
    clut_format: 0,
    palette_offset: 0,
    palette_word_count: 0,
    palette_hash: 0,
};

const NO_SHAPE: TextureShape = TextureShape { width: 0, height: 0, format: GpuFormat::Rgba8888, render_target: false, levels: 1 };

const FREE_ENTRY: Entry = Entry {
    key: NO_DESCRIPTOR,
    data_hash: 0,
    texture: None,
    shape: NO_SHAPE,
    last_used: 0,
    verified_at: 0,
    smooth: false,
    changes: 0,
    unhashed: 0,
    sample_hash: 0,
    quick: 0,
    opaque_rows: 0,
};

const FREE_RETIRED: Retired = Retired { texture: None, shape: NO_SHAPE, ready_at: 0 };

#[derive(Clone, Copy)]
struct FailedTexture {
    key: TextureDescriptor,
    data_hash: u32,
    /// `render_count` from which it may be tried again; `u32::MAX`: never
    /// (it can't be decoded).
    retry_at: u32,
}

pub struct TextureCache {
    entries: [Entry; CACHE_SIZE],
    retired: [Retired; CACHE_SIZE],
    /// Bytes of GPU textures, cached and retired.
    bytes: u32,
    /// Where most textures' memory comes from (see [`arena`]).
    arena: PageArena,
    /// True once the host was asked for the arena.
    arena_asked: bool,
    /// Where a texture was found last time, by a hash of what makes it the
    /// same texture (see `find`): entry slots, only hints. Two per bucket:
    /// with one, two textures of a frame sharing a bucket took each other's
    /// hint every frame and both scanned all entries (MTX: 16-47 scans per
    /// frame of ~300-400 lookups).
    hints: [[u16; 2]; HINT_BUCKETS],
    // This frame (see `FrameStart`):
    frame_number: u32,
    render_count: u32,
    smooth: bool,
    trace_address: u32,
    loop_start_us: u32,
    /// Re-check bytes left this frame.
    verify_budget: u32,
    frame_hashes: [FrameHash; FRAME_HASHES],
    frame_hash_count: usize,
    times: FrameTimes,
    /// Frames in a row that waited for textures.
    deferred_frames: u32,
    /// `VramState::changes` when we last looked.
    seen_vram_changes: u32,
    failed: [Option<FailedTexture>; FAILED_TEXTURES],
    /// The mip levels of the texture being looked up (`Frame::texture_levels`),
    /// used when its flags say it has levels.
    lookup_levels: crate::frame::TextureLevels,
    /// The frame's capture-time sample of the texture being looked up
    /// (`Frame::texture_samples`; 0 = none), set like `lookup_levels`.
    pub lookup_sample: u32,
    /// See `moved_on`.
    moved_on: u32,
    /// Per texture of the last `textures_for_frame`: `opaque_rows`.
    frame_opaque_rows: [u16; MAX_TEXTURES],
    /// The `opaque_rows` of the entry the last `look_up` returned
    /// `Lookup::Ready` from (`ready`).
    last_opaque_rows: u16,
    failed_next: usize,
    refresh: Option<PendingRefresh>,
    stats: TextureStats,
}

impl Default for TextureCache {
    fn default() -> Self {
        Self::new()
    }
}

impl TextureCache {
    pub const fn new() -> Self {
        TextureCache {
            entries: [FREE_ENTRY; CACHE_SIZE],
            retired: [FREE_RETIRED; CACHE_SIZE],
            bytes: 0,
            arena: PageArena::new(),
            arena_asked: false,
            hints: [[0; 2]; HINT_BUCKETS],
            frame_number: 0,
            render_count: 0,
            smooth: false,
            trace_address: 0,
            loop_start_us: 0,
            verify_budget: VERIFY_BYTES_PER_FRAME,
            frame_hashes: [FrameHash { address: 0, size: 0, data_size: 0, hash: 0, quick: 0 }; FRAME_HASHES],
            frame_hash_count: 0,
            times: FrameTimes { trim_us: 0, hash_us: 0, allocate_us: 0, decode_us: 0 },
            deferred_frames: 0,
            seen_vram_changes: 0,
            failed: [None; FAILED_TEXTURES],
            lookup_levels: crate::frame::TextureLevels { address: [0; 8], buffer_width: [0; 8], log2_size: [0; 8] },
            lookup_sample: 0,
            moved_on: 0,
            frame_opaque_rows: [0; MAX_TEXTURES],
            last_opaque_rows: 0,
            failed_next: 0,
            refresh: None,
            stats: TextureStats {
                decode_us: 0,
                refresh_us: 0,
                hashed_bytes: 0,
                decodes: 0,
                refreshes: 0,
                create_failures: 0,
                texture_bytes: 0,
                evictions: 0,
                arena_bytes: 0,
                finds: 0,
                scans: 0,
                moved_on: 0,
            },
        }
    }

    /// Looks up (and decodes where needed) every texture of `frame` once,
    /// into `out` (indexed like `frame.textures`). Starts at a different
    /// texture each frame, so the re-check budget is shared fairly.
    /// `staging` is a buffer of `staging::STAGING_SIZE` bytes.
    pub fn textures_for_frame<H: Host>(
        &mut self,
        host: &mut H,
        staging: &mut [u8],
        frame: &Frame,
        start: FrameStart,
        out: &mut [Lookup; MAX_TEXTURES],
    ) {
        self.begin_frame(host, start);
        self.moved_on = 0;
        let textures = frame.textures();
        let palette_words = &frame.palette_words[..(frame.palette_word_count as usize).min(frame.palette_words.len())];
        let count = textures.len();
        for n in 0..count {
            let index = (n + self.render_count as usize) % count;
            let texture = &textures[index];
            self.last_opaque_rows = 0;
            out[index] = if is_render_target(&start, textures, texture) {
                Lookup::RenderTarget
            } else {
                // A texture with mip levels: their table for the lookup.
                self.lookup_levels = *frame.texture_levels(index);
                self.lookup_sample = frame.texture_sample(index);
                let lookup = self.texture_for(host, staging, palette_words, texture);
                if matches!(lookup, Lookup::Ready(_)) && self.shows_other_pixels(texture) {
                    self.moved_on += 1;
                    self.stats.moved_on += 1;
                }
                lookup
            };
            self.frame_opaque_rows[index] = if matches!(out[index], Lookup::Ready(_)) { self.last_opaque_rows } else { 0 };
        }
    }

    /// How many rows from the top of the last `textures_for_frame`'s
    /// texture `index` (ready) have only texels of alpha 255
    /// (`texture::opaque_rows`, `ALL_ROWS_OPAQUE`: all; 0 if not ready): a
    /// draw sampling only those rows sees an opaque texture.
    pub fn opaque_rows(&self, index: usize) -> u16 {
        self.frame_opaque_rows.get(index).copied().unwrap_or(0)
    }

    /// `Lookup::Ready` with entry `slot`'s texture, noting whether it is
    /// opaque (`opaque`).
    fn ready(&mut self, slot: usize) -> Lookup {
        let entry = &self.entries[slot];
        self.last_opaque_rows = entry.opaque_rows;
        Lookup::Ready(entry.texture.unwrap())
    }

    /// Textures of the last `textures_for_frame` whose pixels the game had
    /// rewritten before we read them (`shows_other_pixels`): the frame can't
    /// be drawn as the GE drew it, and isn't shown (nativeres_render.c).
    pub fn moved_on(&self) -> u32 {
        self.moved_on
    }

    /// True if the texture we have for `wanted` (just looked up) was decoded
    /// from other data than the capture sampled for this frame
    /// (`lookup_sample`): the game wrote new pixels after the GE drew the
    /// frame (MTX's menus load the next screen's textures while the last
    /// frame of the old one is on its way; decoded then, they showed as
    /// noise or wrong tiles; the level previews flashed another track's).
    /// Not for VRAM (no capture sample), textures whose sample we don't know
    /// (swapped in by the worker) nor volatile ones (rewritten every frame:
    /// always newer than the frame, and that's fine).
    fn shows_other_pixels(&mut self, wanted: &TextureDescriptor) -> bool {
        if self.lookup_sample == 0 || !QUICK_CHECKS.load(Ordering::Relaxed) {
            return false;
        }
        match self.find(wanted) {
            Some(slot) => {
                let entry = &self.entries[slot];
                entry.quick != 0 && entry.quick != self.lookup_sample && entry.changes < VOLATILE_CHANGES
            }
            None => false,
        }
    }

    /// Once per frame, before its textures are looked up.
    pub fn begin_frame<H: Host>(&mut self, host: &mut H, start: FrameStart) {
        let begin_us = host.now_us();
        if start.trace_address == TRACE_VRAM && start.frame_number == self.frame_number {
            let vram = host.vram();
            host.log(format_args!(
                "trace: frame {} again (captured {}) render {} vram gen {}/{} changes {}\n",
                start.frame_number,
                host.captured_frame_number(),
                start.render_count,
                vram.copied_generation,
                vram.generation,
                vram.changes
            ));
        }
        // Game frames went by without ours (renderer off, frames the capture
        // skipped, a CPU-drawn loading screen), or the numbers started again
        // (a savestate load): PSP memory may hold anything new by now, but
        // `render_count`, which the re-checks count in, stood still. So every
        // texture is due for a re-check now, past the byte budget. MTX's
        // level-select previews came back as noise otherwise: the last
        // track's indices with the new track's palettes.
        let skipped = start.frame_number.wrapping_sub(self.frame_number);
        let after_gap = self.frame_number != 0 && skipped > VERIFY_INTERVAL;
        self.frame_number = start.frame_number;
        self.render_count = start.render_count;
        if after_gap {
            let due = start.render_count.wrapping_sub(VERIFY_INTERVAL);
            for entry in self.entries.iter_mut().filter(|entry| entry.texture.is_some()) {
                entry.verified_at = due;
            }
        }
        self.smooth = start.smooth;
        self.trace_address = start.trace_address;
        self.loop_start_us = start.loop_start_us;
        self.times = FrameTimes::default();
        if !self.arena_asked {
            self.arena_asked = true;
            self.arena.set_size(host.texture_arena_bytes());
        }
        self.finish_refresh(host, start.frame_number);
        self.recheck_vram_textures_if_changed(host);
        self.trim(host);
        self.verify_budget = if after_gap { u32::MAX } else { VERIFY_BYTES_PER_FRAME };
        self.frame_hash_count = 0;
        self.times.trim_us = host.now_us() - begin_us;
    }

    /// After the frame: whether it waited for textures (not drawn).
    pub fn end_frame(&mut self, waited: bool) {
        self.deferred_frames = if waited { self.deferred_frames + 1 } else { 0 };
    }

    /// The counters since the last call; resets them.
    pub fn take_stats(&mut self) -> TextureStats {
        let stats = TextureStats { texture_bytes: self.bytes, arena_bytes: self.arena.used_bytes(), ..self.stats };
        self.stats = TextureStats::default();
        stats
    }

    pub fn frame_times(&self) -> FrameTimes {
        self.times
    }

    /// Bytes of GPU textures, cached and retired.
    pub fn bytes(&self) -> u32 {
        self.bytes
    }

    /// The GPU texture for frame texture `wanted` (`palette_words`: the
    /// frame's), decoding it if needed.
    pub fn texture_for<H: Host>(
        &mut self,
        host: &mut H,
        staging: &mut [u8],
        palette_words: &[u32],
        wanted: &TextureDescriptor,
    ) -> Lookup {
        let traced = match self.trace_address {
            0 => false,
            TRACE_VRAM => false, // traced inside look_up, see trace_vram
            address => wanted.address & 0x0FFF_FFFF == address & 0x0FFF_FFFF,
        };
        if !traced {
            return self.look_up(host, staging, palette_words, wanted);
        }
        let cached = self.find(wanted).is_some();
        let decodes = self.stats.decodes;
        let lookup = self.look_up(host, staging, palette_words, wanted);
        let (data_hash, verified_at) = match self.find(wanted) {
            Some(slot) => (self.entries[slot].data_hash, self.entries[slot].verified_at),
            None => (0, 0),
        };
        let vram = host.vram();
        if self.trace_address == TRACE_VRAM && lookup != Lookup::Waiting && self.stats.decodes == decodes {
            return lookup; // only waits and decodes
        }
        host.log(format_args!(
            "trace {:08x} {}x{} fmt {} frame {} (captured {}) render {}: {}{}{}{} data {:08x} verified {} vram gen {}/{} changes {}\n",
            wanted.address,
            wanted.width,
            wanted.height,
            wanted.format,
            self.frame_number,
            host.captured_frame_number(),
            self.render_count,
            if cached { "cached" } else { "new" },
            if self.stats.decodes != decodes { " decoded" } else { "" },
            if matches!(lookup, Lookup::Ready(_)) { "" } else { " NULL" },
            if lookup == Lookup::Waiting { " waiting" } else { "" },
            data_hash,
            verified_at,
            vram.copied_generation,
            vram.generation,
            vram.changes
        ));
        lookup
    }

    /// 4- and 8-bit palette textures are kept as raw indices
    /// (`GpuFormat::Palette8`); their colours come from a palette table
    /// chosen per draw, so the palette is no part of what makes them the
    /// same. Otherwise Final Fantasy's world map, which animates its water
    /// by switching palettes, decoded ~180 tiles again every 8 frames.
    fn is_index_texture(&self, texture: &TextureDescriptor) -> bool {
        !self.smooth && texture::is_vita_palette_format(texture.format)
    }

    fn same_texture(&self, a: &TextureDescriptor, b: &TextureDescriptor) -> bool {
        let same_data = a.address == b.address
            && a.buffer_width == b.buffer_width
            && a.width == b.width
            && a.height == b.height
            && a.format == b.format
            && a.swizzled == b.swizzled
            && a.flags == b.flags;
        if !same_data || self.is_index_texture(a) {
            return same_data;
        }
        a.clut_format == b.clut_format && a.palette_word_count == b.palette_word_count && a.palette_hash == b.palette_hash
    }

    /// The hint bucket for `texture`: a hash of what `same_texture`
    /// compares (the palette only where it counts).
    fn hint_bucket(&self, texture: &TextureDescriptor) -> usize {
        let mix = |hash: u32, value: u32| (hash ^ value).wrapping_mul(0x0100_0193);
        let mut hash = mix(0x811C_9DC5, texture.address);
        hash = mix(hash, texture.buffer_width);
        hash = mix(hash, texture.width as u32 | (texture.height as u32) << 16);
        hash = mix(hash, texture.format as u32 | (texture.swizzled as u32) << 8 | (texture.flags as u32) << 16);
        if !self.is_index_texture(texture) {
            hash = mix(hash, texture.clut_format);
            hash = mix(hash, texture.palette_word_count);
            hash = mix(hash, texture.palette_hash);
        }
        (hash ^ hash >> 16) as usize % HINT_BUCKETS
    }

    /// The cache slot holding `wanted`, if any. A race frame in MTX looks up
    /// ~393 textures, and scanning the 512 entries (~46 KiB, more than the
    /// CPU's 32 KiB L1 cache) for each took milliseconds per frame. So the
    /// slot where a texture was found is remembered in `hints` (the last two
    /// per bucket); a hint is checked before use (entries change without
    /// updating it), and only without a good hint are the entries scanned.
    fn find(&mut self, wanted: &TextureDescriptor) -> Option<usize> {
        let bucket = self.hint_bucket(wanted);
        self.stats.finds += 1;
        for hinted in self.hints[bucket] {
            let entry = &self.entries[hinted as usize];
            if entry.texture.is_some() && self.same_texture(&entry.key, wanted) {
                return Some(hinted as usize);
            }
        }
        self.stats.scans += 1;
        let found = self.entries.iter().position(|entry| entry.texture.is_some() && self.same_texture(&entry.key, wanted));
        if let Some(slot) = found {
            // Newest first; the older hint stays as the second.
            let hints = &mut self.hints[bucket];
            *hints = [slot as u16, hints[0]];
        }
        found
    }

    /// "tracetex 1": a VRAM texture waits or is decoded (only then, so the
    /// trace costs nothing on the common path).
    fn trace_vram<H: Host>(&self, host: &mut H, wanted: &TextureDescriptor, what: &str) {
        if self.trace_address != TRACE_VRAM || !is_in_vram(wanted) {
            return;
        }
        let vram = host.vram();
        host.log(format_args!(
            "vram {:08x} {}x{}: {} frame {} (captured {}) render {} gen {}/{} changes {}
",
            wanted.address,
            wanted.width,
            wanted.height,
            what,
            self.frame_number,
            host.captured_frame_number(),
            self.render_count,
            vram.copied_generation,
            vram.generation,
            vram.changes
        ));
    }

    fn past_deadline<H: Host>(&self, host: &H) -> bool {
        (host.now_us() as u32).wrapping_sub(self.loop_start_us) > TEXTURE_DEADLINE_US
    }

    /// The PSP plugin found VRAM blocks changed (e.g. the game loaded the
    /// field's tiles where the menu's graphics were): our VRAM textures are
    /// re-checked at their next use rather than at their next regular
    /// check, so old pixels don't stay on screen for long.
    fn recheck_vram_textures_if_changed<H: Host>(&mut self, host: &mut H) {
        let changes = host.vram().changes;
        if changes == self.seen_vram_changes {
            return;
        }
        self.seen_vram_changes = changes;
        let mut marked = 0;
        for entry in self.entries.iter_mut() {
            if entry.texture.is_some() && is_in_vram(&entry.key) {
                entry.verified_at = self.render_count.wrapping_sub(VERIFY_INTERVAL);
                marked += 1;
            }
        }
        // MTX changes VRAM every frame with no texture there: a line per
        // frame filled the render log.
        if marked > 0 {
            host.log(format_args!("vram changed ({} blocks so far): re-checking {} textures\n", changes, marked));
        }
    }

    fn has_failed_before(&self, wanted: &TextureDescriptor, data_hash: u32) -> bool {
        let render_count = self.render_count;
        self.failed.iter().flatten().any(|failed| {
            self.same_texture(&failed.key, wanted)
                && failed.data_hash == data_hash
                && (failed.retry_at == u32::MAX || (failed.retry_at.wrapping_sub(render_count) as i32) > 0)
        })
    }

    /// `transient`: it failed for lack of memory or room, not because it
    /// can't be decoded; then it is tried again after RETRY_FAILED_FRAMES.
    fn remember_failure<H: Host>(&mut self, host: &mut H, wanted: &TextureDescriptor, data_hash: u32, reason: &str, result: i32, transient: bool) {
        log_texture(host, reason, wanted, result);
        let retry_at = if transient { self.render_count.wrapping_add(RETRY_FAILED_FRAMES) } else { u32::MAX };
        self.failed[self.failed_next] = Some(FailedTexture { key: *wanted, data_hash, retry_at });
        self.failed_next = (self.failed_next + 1) % FAILED_TEXTURES;
    }

    /// The lookup itself, see the module description.
    fn look_up<H: Host>(
        &mut self,
        host: &mut H,
        staging: &mut [u8],
        palette_words: &[u32],
        wanted: &TextureDescriptor,
    ) -> Lookup {
        let render_count = self.render_count;
        let mut found = self.find(wanted);
        // Decoded for the other filter mode: decode again.
        if let Some(slot) = found {
            if self.entries[slot].smooth != self.smooth {
                self.evict(slot);
                found = None;
            }
        }

        let vram_pending = is_in_vram(wanted) && host.vram().copy_pending();
        let big = is_big_direct_colour(wanted);
        let mut new_sample_hash = 0;
        // The samples showed a change: decode without a full hash.
        let mut sampled = false;

        if let Some(slot) = found {
            if vram_pending && render_count.wrapping_sub(self.entries[slot].last_used) > VERIFY_INTERVAL {
                // A VRAM texture not used for a while: its range may have
                // been dropped from the copy requests meanwhile (after
                // vram_requests::KEEP_FRAMES unused frames) and asked for
                // again now, so the copy we decoded from may be outdated, and
                // the new copy hasn't arrived. Wait for it like for a new
                // texture; its old verified_at makes the next frame re-check
                // it. Final Fantasy's prologue draws a lightning bolt from
                // VRAM for ~3 frames now and then: its first frame showed the
                // old pixels of that VRAM area in the bolt's blue palette.
                self.trace_vram(host, wanted, "waiting (unused, copy pending)");
                return Lookup::Waiting;
            }
            let entry = &mut self.entries[slot];
            entry.last_used = render_count;
            // Changed at its last re-check: re-checked now, past the budgets.
            let mut changing = entry.changes > 0 && !vram_pending;
            if !vram_pending && big {
                new_sample_hash = sample_hash(host, wanted);
                let entry = &mut self.entries[slot];
                if new_sample_hash != entry.sample_hash {
                    sampled = true;
                    changing = true;
                    if entry.changes == 0 {
                        entry.changes = 1;
                    }
                } else if changing {
                    entry.changes = 0; // same samples: not changing (any more)
                    changing = false;
                }
            }
            let quick = self.entries[slot].quick;
            if !vram_pending && self.lookup_sample != 0 && quick != 0 && self.lookup_sample != quick && QUICK_CHECKS.load(Ordering::Relaxed) {
                // The capture's sample of the data the frame was drawn with
                // isn't the cached pixels': re-checked now (hashed and
                // decoded if it changed; big ones decoded without the hash).
                // The same: kept, even if the game has written other pixels
                // there since (the next screen's, loading: decoded now, with
                // this frame's palette, they were noise).
                changing = true;
                if big && !sampled {
                    sampled = true;
                    new_sample_hash = sample_hash(host, wanted);
                    let entry = &mut self.entries[slot];
                    if entry.changes == 0 {
                        entry.changes = 1;
                    }
                }
            }
            if !changing {
                if render_count.wrapping_sub(self.entries[slot].verified_at) < VERIFY_INTERVAL || vram_pending {
                    return self.ready(slot);
                }
                // Big: the worker re-checks it, off this thread's budgets.
                if self.can_refresh_in_background(host, wanted) {
                    if self.start_refresh(host, slot, wanted) {
                        self.entries[slot].verified_at = render_count;
                    }
                    return self.ready(slot);
                }
                let size = StagedLayout::of(wanted).size;
                // One texture per frame may exceed the budget, otherwise
                // textures bigger than the budget would never be re-checked.
                let fits_budget = size <= self.verify_budget || self.verify_budget == VERIFY_BYTES_PER_FRAME;
                if !fits_budget || self.past_deadline(host) {
                    return self.ready(slot);
                }
                self.verify_budget = self.verify_budget.saturating_sub(size);
            }
            self.entries[slot].verified_at = render_count;
        } else if vram_pending || (self.past_deadline(host) && self.deferred_frames < MAX_DEFERRED_FRAMES) {
            self.trace_vram(host, wanted, if vram_pending { "waiting (new, copy pending)" } else { "waiting (deadline)" });
            return Lookup::Waiting; // drawn from a following frame on
        }

        // Textures that changed at every re-check lately (Final Fantasy's
        // animated battle intro rays, 512 KiB each) are decoded again without
        // hashing first: the hash would only confirm the change and costs
        // about as much as the decode, together too much for one frame.
        // Every HASH_VOLATILE_EVERY-th time they are hashed, to notice when
        // they stop changing. An unhashed decode keeps the last hash we
        // computed, so that check compares against real data (it used to
        // compare against 0, and such a texture never stopped counting as
        // volatile).
        let layout = StagedLayout::of(wanted);
        let mut staged_size = 0;
        let mut data_hash = 0;
        // The quick sample of the data hashed (`staged_quick_hash`).
        let mut hashed_quick = 0;
        let mut changes = found.map_or(0, |slot| self.entries[slot].changes);
        let mut unhashed = 0u8;
        let volatile_unhashed = |entry: &Entry| {
            entry.changes >= VOLATILE_CHANGES && (entry.unhashed as u32) + 1 < HASH_VOLATILE_EVERY
        };
        match found {
            Some(slot) if sampled || volatile_unhashed(&self.entries[slot]) => {
                unhashed = self.entries[slot].unhashed.saturating_add(1);
                data_hash = self.entries[slot].data_hash;
            }
            // New and big: its samples find changes, no need for the full
            // hash (~3 ms for 512 KiB, three at once when Final Fantasy's
            // battle-start rays begin). Hash 0: the first regular re-check
            // hashes it.
            None if big => unhashed = 1,
            _ => {
                let hash_start = host.now_us();
                data_hash = self.data_hash(host, staging, wanted, &layout, &mut staged_size, &mut hashed_quick);
                self.times.hash_us += host.now_us() - hash_start;
                if let Some(slot) = found {
                    let entry = &mut self.entries[slot];
                    if entry.data_hash == data_hash {
                        // Same pixels as at the last hash. If it was decoded
                        // without hashing since then, it may show other
                        // pixels: decode once more. Either way it no longer
                        // counts as changing.
                        let decoded_unhashed = entry.unhashed > 0;
                        entry.changes = 0;
                        entry.unhashed = 0;
                        // The sample of the data just hashed, not of the
                        // memory now: the game may have written new pixels
                        // since (staged_quick_hash).
                        entry.quick = hashed_quick;
                        if !decoded_unhashed {
                            return self.ready(slot);
                        }
                        changes = 0;
                    } else {
                        changes = changes.saturating_add(1);
                    }
                }
            }
        }
        if found.is_none() && self.has_failed_before(wanted, data_hash) {
            return Lookup::Unavailable;
        }
        if let Some(slot) = found {
            // Same texture, new pixels: decode into a fresh texture, because
            // the GPU may still be drawing the previous frame with the old one.
            self.evict(slot);
        }

        let format = GpuFormat::for_texture(wanted, self.smooth);
        let allocate_start = host.now_us();
        let Some(slot) = self.make_room(host, TextureShape::new(wanted, format, false).bytes()) else {
            self.times.allocate_us += host.now_us() - allocate_start;
            self.remember_failure(host, wanted, data_hash, "texture cache full", -1, true);
            return Lookup::Unavailable;
        };
        let mut on_gpu = can_unswizzle_on_gpu(wanted, format);
        let mut texture = self.acquire(host, TextureShape::new(wanted, format, on_gpu));
        if texture.is_none() && on_gpu {
            on_gpu = false; // no render target: the CPU unswizzles
            texture = self.acquire(host, TextureShape::new(wanted, format, false));
        }
        self.times.allocate_us += host.now_us() - allocate_start;
        let Some(texture) = texture else {
            self.remember_failure(host, wanted, data_hash, "out of texture memory", -1, true);
            return Lookup::Unavailable;
        };
        let shape = TextureShape::new(wanted, format, on_gpu);

        if !sampled && big {
            // What the next use compares with.
            new_sample_hash = sample_hash(host, wanted);
        }
        // The quick sample of the pixels decoded: of the staged bytes on the
        // CPU path (set below), of the memory before the GPU reads it on the
        // GPU path (16-bit textures only).
        let mut quick = if on_gpu { quick_hash(host, wanted) } else { 0 };
        let decode_start = host.now_us();
        let mut result = Ok(());
        // Unswizzled on the GPU: not looked at (16-bit with alpha: rarely opaque).
        let mut opaque_rows = 0;
        if !on_gpu || !self.unswizzle_on_gpu(host, wanted, texture, shape) {
            if staged_size == 0 {
                // The hash came from an earlier copy this frame (or there was none).
                staged_size = stage(host, wanted, &layout, staging);
                quick = staged_quick_hash(host, wanted, &layout, &staging[..staged_size as usize]);
            } else {
                quick = hashed_quick; // the bytes hashed are the ones decoded
            }
            let palette = texture_palette(wanted, palette_words);
            let data = &staging[..staged_size as usize];
            // Palette textures: their colours come per frame (palette tables).
            result = host.decode(texture, format, &layout.as_texture, data, palette, self.smooth, 0);
            opaque_rows = texture::opaque_rows(data, &layout.as_texture);
            // Mip levels 1.. (menu "NativeRes Distant Textures"): each staged
            // from its own address and decoded into its place. Only level 0
            // is hashed: a game rewrites a texture's levels with it.
            let levels = crate::frame::texture_level_count(wanted) as usize;
            for level in 1..levels {
                if result.is_err() {
                    break;
                }
                let level_texture = texture::level_descriptor(wanted, &self.lookup_levels, level);
                let level_layout = StagedLayout::of(&level_texture);
                let size = stage(host, &level_texture, &level_layout, staging) as usize;
                result = if size == 0 {
                    Err(DecodeError::BadAddress)
                } else {
                    // Levels: any level may be sampled, so all or nothing.
                    if opaque_rows != texture::ALL_ROWS_OPAQUE
                        || texture::opaque_rows(&staging[..size], &level_layout.as_texture) != texture::ALL_ROWS_OPAQUE
                    {
                        opaque_rows = 0;
                    }
                    host.decode(texture, format, &level_layout.as_texture, &staging[..size], palette, self.smooth, level as u8)
                };
            }
        }
        let decode_us = host.now_us() - decode_start;
        if !big {
            new_sample_hash = quick;
        }
        if self.lookup_sample != 0 && quick != self.lookup_sample {
            // The game wrote other pixels there after the GE drew the frame:
            // what we decoded isn't what the frame showed.
            host.log(format_args!(
                "moved on: {:08x} {}x{} fmt {} sampled {:08x} at capture, {:08x} decoded ({})\n",
                wanted.address,
                wanted.width,
                wanted.height,
                wanted.format,
                self.lookup_sample,
                quick,
                if found.is_some() { "changed" } else { "new" }
            ));
        }
        self.stats.decode_us += decode_us;
        self.times.decode_us += decode_us;
        self.stats.decodes += 1;
        if let Err(error) = result {
            self.remember_failure(host, wanted, data_hash, "decode failed", error.code() as i32, false);
            self.free_now(host, texture, shape); // never used by the GPU
            return Lookup::Unavailable;
        }
        log_texture(host, "decoded", wanted, 0);
        self.trace_vram(host, wanted, if found.is_some() { "decoded again" } else { "decoded new" });
        host.log(format_args!("  in {} us, data hash {:08x}\n", decode_us, data_hash));

        let bucket = self.hint_bucket(wanted);
        let hints = &mut self.hints[bucket];
        *hints = [slot as u16, hints[0]];
        self.entries[slot] = Entry {
            key: *wanted,
            data_hash,
            texture: Some(texture),
            shape,
            last_used: render_count,
            verified_at: render_count,
            smooth: self.smooth,
            changes,
            unhashed,
            sample_hash: new_sample_hash,
            quick,
            opaque_rows,
        };
        self.ready(slot)
    }

    /// Has the GPU unswizzle `wanted` into `target` (see
    /// `gpu::can_unswizzle_on_gpu`). The swizzled data goes into a texture
    /// of its own for the GPU to read, a render target too (GPU-mapped),
    /// retired afterwards like any replaced texture. False, having
    /// submitted nothing, if it can't.
    fn unswizzle_on_gpu<H: Host>(
        &mut self,
        host: &mut H,
        wanted: &TextureDescriptor,
        target: GpuTexture,
        shape: TextureShape,
    ) -> bool {
        let bytes = wanted.width as u32 * wanted.height as u32 * 2;
        if host.psp_memory(wanted.address & 0x0FFF_FFFF, bytes).is_none()
            || !host.gpu_unswizzle_ready(wanted.width, wanted.height)
        {
            return false;
        }
        let Some(raw) = self.acquire(host, shape) else {
            return false;
        };
        let submitted = host.unswizzle_on_gpu(wanted, target, raw);
        if !self.retire(raw, shape) {
            host.wait_for_gpu(); // retired list full: free it once the GPU is done
            self.free_now(host, raw, shape);
        }
        submitted
    }
}

/// True if `texture` lies in PSP VRAM (any mirror's address).
fn is_in_vram(texture: &TextureDescriptor) -> bool {
    let address = texture.address & 0x0FFF_FFFF;
    (VRAM_START..VRAM_START + VRAM_SIZE).contains(&address)
}

fn is_big_direct_colour(texture: &TextureDescriptor) -> bool {
    texture.format < format::CLUT4 && texture::data_size(texture).unwrap_or(0) >= BACKGROUND_MIN_BYTES
}

/// The texture's own part of the frame's palette words.
pub fn texture_palette<'a>(texture: &TextureDescriptor, palette_words: &'a [u32]) -> &'a [u32] {
    let start = (texture.palette_offset as usize).min(palette_words.len());
    let end = (start + texture.palette_word_count as usize).min(palette_words.len());
    &palette_words[start..end]
}

fn log_texture<H: Host>(host: &mut H, what: &str, texture: &TextureDescriptor, result: i32) {
    host.log(format_args!(
        "{}: addr={:08x} bufw={} {}x{} fmt={} swz={} clut={:06x} pal={}+{} hash={:08x} result={}\n",
        what,
        texture.address,
        texture.buffer_width,
        texture.width,
        texture.height,
        texture.format,
        texture.swizzled,
        texture.clut_format,
        texture.palette_offset,
        texture.palette_word_count,
        texture.palette_hash,
        result
    ));
}
