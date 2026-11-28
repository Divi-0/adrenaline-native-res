//! Turns GE draws into a [`Frame`] description for the Vita GPU.
//!
//! For every PRIM command we read the vertices, convert the primitive to
//! plain triangles, and record the render state that matters for 2D:
//! texture, colours, blending, filtering, wrapping and scissor. Through
//! mode (screen-space 2D) vertices are used as they are; 3D vertices are
//! transformed to the screen first (`transform.rs`; Final Fantasy's world
//! map and intro).
//!
//! What is simplified for now (counted in the stats as "approx"):
//! - Texture functions other than modulate and replace are drawn as modulate.
//! - Blending: see `blend.rs` (fixed colours, doubled factors, alpha test).
//! - 3D: no lighting (vertex or material colours instead), texture
//!   coordinates only from the vertices, no depth test (draws land in list
//!   order), rectangles reaching behind the camera left out.

use crate::vertex_cache::{self, bounds_of, data_hash, Lookup, RunKey, VertexCache};
use crate::blend::{self, scale_color};
use crate::color::opaque;
use crate::command::{op, primitive, Command};
use crate::display_list::{CommandVisitor, MAX_PRIM_BATCH};
use crate::draw_stream::{DrawStream, Job, JobKind, QueuedDraw};
use crate::frame_builder::FrameBuilder;
use crate::frame::{draw_flags, test_flags, Catalog, Draw, Frame, FrameVertex, Geometry, PalettePlace, TextureDescriptor};
use crate::ge_state::GeState;
use crate::lighting::{pack_normal, pack_s8_normal, pack_scaled_normal};
use crate::transform::{TransformBuilder, TransformInputs, VertexLighting};
use crate::memory::{physical, PspMemory};
use crate::snapshot::{Reader, Writer};
use crate::stats::FrameStats;
use crate::texture::{describe_current_texture, is_palette_format};
use crate::transform::{clip_triangle, RowTransform, Texgen, Transform, TransformCache};
use crate::vertex::{CommonLayout, IndexFormat, Raw3d, Vertex, VertexBytes, VertexFormat, VertexLayout, MAX_WEIGHTS};

/// Palettes are up to 1 KiB; allow 2 KiB like PPSSPP for odd games.
const MAX_PALETTE_WORDS: usize = 512;

/// Texture functions (TEXTURE_FUNCTION bits 0..3).
/// Why a draw is drawn approximately (`Prepared::approximations`, a mask of
/// these): counted per reason in `FrameStats::approximated_by_reason`.
pub mod approx {
    /// Partly masked colour channels (COLOR_MASK): drawn unmasked.
    pub const MASKED_COLOR: u32 = 1 << 0;
    /// A blend factor the Vita can't do (`blend::translate`).
    pub const BLEND: u32 = 1 << 1;
    /// Texture function other than modulate/replace: drawn as modulate.
    pub const TEXTURE_FUNCTION: u32 = 1 << 2;
    /// Fog on an untextured draw (only textured draws are fogged so far).
    pub const UNTEXTURED_FOG: u32 = 1 << 3;
    /// LIGHTMODE 1: the separate specular colour added before texturing.
    pub const SEPARATE_SPECULAR: u32 = 1 << 4;
    /// Generated texture coordinates (TEXTURE_MAP_MODE 1/2).
    pub const TEXGEN: u32 = 1 << 5;
    /// Lit with lights while `lighting::LIGHTS_ON` left them out.
    pub const LIGHTS_LEFT_OUT: u32 = 1 << 6;
    /// Alpha test with blending off: normal blending stands in for it
    /// (`blend::translate`; transparent texels still write depth).
    pub const ALPHA_TEST: u32 = 1 << 7;
    /// Stencil ops write: the target's alpha gets the fragment's alpha, not
    /// the stencil value (`blend::alpha_write`).
    pub const STENCIL_ALPHA: u32 = 1 << 8;
    pub const NAMES: [&str; 9] = [
        "masked colour",
        "blend factor",
        "texture function",
        "untextured fog",
        "separate specular",
        "texgen",
        "lights left out",
        "alpha test as blending",
        "stencil as fragment alpha",
    ];
}

/// Samples the data of the draw's texture at its first use in the frame
/// (`Frame::texture_samples`, `texture::quick_sample`): 8 words straight from
/// PSP RAM, while the GE sync holds the game at this frame. Not for VRAM
/// textures (read from the plugin's copy, which has its own update rules:
/// `vram_requests`).
///
/// The copier reads them (a sample task, `copy_tasks::Layout::Sample`)
/// where it can read the memory now: 8 uncached reads for each of ~400
/// textures per MTX race frame were ~1 ms of the capture thread's run, the
/// thread the game waits for. The GE sync and the frame's end wait for the
/// tasks, so the sample is still taken while the game is held.
fn sample_texture(catalog: &mut Catalog, stats: &mut FrameStats, memory: &mut impl PspMemory, stream: &mut DrawStream, prepared: &Prepared) {
    let started = crate::cycles::start(crate::cycles::SAMPLE);
    sample_texture_now(catalog, stats, memory, stream, prepared);
    crate::cycles::add(crate::cycles::SAMPLE, started);
}

fn sample_texture_now(catalog: &mut Catalog, stats: &mut FrameStats, memory: &mut impl PspMemory, stream: &mut DrawStream, prepared: &Prepared) {
    let index = prepared.draw.texture;
    if prepared.draw.flags & draw_flags::TEXTURED == 0
        || crate::bench::experiment(crate::bench::SKIP_TEXTURE_SAMPLES)
        || catalog.is_sampled(index)
        || !crate::texture_cache::QUICK_CHECKS.load(core::sync::atomic::Ordering::Relaxed)
    {
        return;
    }
    let Some(texture) = catalog.textures().get(index as usize).copied() else {
        return;
    };
    let address = physical(texture.address);
    if !(crate::memory::RAM_START..crate::memory::RAM_END).contains(&address) {
        return;
    }
    let Some(positions) = crate::texture::quick_sample_positions(&texture) else {
        return;
    };
    // (The copier's word loads need a word-aligned address.)
    if address % 4 == 0 && !crate::bench::experiment(crate::bench::NO_SAMPLE_TASKS) {
        // `quick_sample_positions` had data: `data_size` is some.
        let words = crate::texture::data_size(&texture).unwrap_or(0) / 4;
        let last = positions[crate::texture::QUICK_SAMPLES - 1] as u32;
        if let Some(place) = catalog.texture_sample_place(index) {
            let layout = crate::copy_tasks::Layout::Sample { words };
            let task = crate::copy_tasks::CopyTask { source: address, length: 4 * (last + 1), destination: place as *mut u8, layout };
            if stream.hand_over_task(memory, task) {
                stats.texture_sample_tasks += 1;
                return;
            }
            // Read here after all (below); a failed read leaves it unsampled.
            catalog.unmark_sampled(index);
        }
    }
    let mut words = [0u32; crate::texture::QUICK_SAMPLES];
    if memory.read_samples(address, &positions, &mut words).is_some() {
        catalog.set_texture_sample(prepared.draw.texture, crate::texture::quick_sample(&words));
        stats.texture_samples_taken += 1;
    }
}

/// Why a draw's vertices were decoded on the CPU instead of going to the
/// GPU raw (`raw_arena`): counted per reason in `FrameStats::fallback_prims`
/// / `fallback_vertices` (capture log and replay: `cpu=`), so the CPU path
/// can be driven to zero game by game. The first reason that applies, in
/// this order.
pub mod fallback {
    /// Through mode (screen-space 2D): no raw program for it yet.
    pub const THROUGH: usize = 0;
    /// Raw vertices switched off, or no raw arena.
    pub const RAW_OFF: usize = 1;
    /// Morphing (several vertex sets blended by the GE).
    pub const MORPH: usize = 2;
    /// No GPU transform for the draw (`Prepared::gpu_transform`: untextured
    /// 3D draws).
    pub const UNTEXTURED: usize = 3;
    /// Lit with lights the GPU path doesn't take (`may_go_raw`: not
    /// directional, a colour function other than the plain one, no normals).
    pub const LIGHTS: usize = 4;
    /// 3D rectangles the raw sprite program can't draw (lights, FIX A,
    /// weights, indexed, no whole sprite).
    pub const SPRITES: usize = 5;
    /// The raw path tried and refused for another reason than those below
    /// (too many vertices for a window, a PRIM without a whole triangle).
    pub const REFUSED: usize = 6;
    /// No room in the raw arena (recent frames still use it; `raw=` counts
    /// these runs too).
    pub const ARENA_FULL: usize = 7;
    /// A frame table full: transforms, light sets, raw colours, bones,
    /// indices.
    pub const TABLES_FULL: usize = 8;
    /// The vertices or indices couldn't be read.
    pub const UNREADABLE: usize = 9;
    /// Texture coordinates from the transform (`Texgen`): only the CPU
    /// works them out.
    pub const TEXGEN: usize = 10;
    pub const COUNT: usize = 11;
    pub const NAMES: [&str; COUNT] =
        ["through", "raw_off", "morph", "untextured", "lights", "sprites", "refused", "arena_full", "tables_full", "unreadable", "texgen"];
}

/// Why a PRIM drawn with this state goes to the CPU (`fallback`), asked
/// where the raw path has turned it down; `raw` the raw arena (`None`:
/// none), which knows why it turned the draw down last (`RawArena::refusal`),
/// or `refused`: why the capture thread's builder turned it down
/// (`frame_builder`, which holds the arena then).
fn fallback_reason(
    prepared: &Prepared,
    inputs: Option<&TransformInputs>,
    format: VertexFormat,
    primitive_type: u32,
    raw: Option<&crate::raw_arena::RawArena>,
    refused: Option<usize>,
) -> usize {
    let Some(inputs) = inputs.filter(|_| !format.is_through()) else {
        return fallback::THROUGH;
    };
    if !crate::raw_arena::ENABLED.load(core::sync::atomic::Ordering::Relaxed) {
        return fallback::RAW_OFF;
    }
    // The draw's own reasons first: the capture thread's builder turns
    // those down without naming one (NFS UR's menu: 375 draws counted as
    // raw_off).
    if format.morph_count() != 1 {
        fallback::MORPH
    } else if prepared.texgen.is_some() {
        fallback::TEXGEN
    } else if !prepared.gpu_transform {
        fallback::UNTEXTURED
    } else if !may_go_raw(prepared, inputs.lighting().map(|lighting| lighting.raw_ready(format)), format) {
        fallback::LIGHTS
    } else if primitive_type == primitive::RECTANGLES {
        fallback::SPRITES
    } else {
        // No arena (or a builder that didn't try): raw_off.
        refused.or(raw.map(|raw| raw.refusal)).unwrap_or(fallback::RAW_OFF)
    }
}

/// Notes why the raw path turns a draw down if it does so now
/// (`RawArena::refusal`): call before each step that can fail.
#[inline(always)]
fn refusing(raw: &mut crate::raw_arena::RawArena, reason: usize) {
    raw.refusal = reason;
}

/// Counts `prims` PRIMs with `vertices` PSP vertices as decoded on the CPU
/// for `reason` (`fallback`).
fn count_fallback(stats: &mut FrameStats, reason: usize, prims: u32, vertices: u32) {
    stats.fallback_prims[reason] += prims;
    stats.fallback_vertices[reason] += vertices;
}

/// Counts a draw's approximations (`approx` mask) `times` times.
fn count_approximations(stats: &mut FrameStats, mask: u32, times: u32) {
    if mask == 0 {
        return;
    }
    stats.approximated_draws += mask.count_ones() * times;
    for bit in 0..approx::NAMES.len() {
        if mask & (1 << bit) != 0 {
            stats.approximated_by_reason[bit] += times;
        }
    }
}

const TEXTURE_FUNCTION_MODULATE: u32 = 0;
const TEXTURE_FUNCTION_REPLACE: u32 = 3;

/// The palette last loaded into the GE with CLUT_LOAD. It stays loaded
/// across frames until the game loads another one.
///
/// A load only notes where the palette is (address and length): nothing is
/// read on the capture thread, the one the game waits for (GE sync). MTX
/// Mototrax loads a palette before nearly every draw (Los Angeles race
/// start: ~720 loads per frame from ~470 addresses, ~270 KB of uncached
/// reads); copying them as the list was walked cost the capture thread ~2.9
/// ms per frame (2026-11-09). Now the first draw of a frame that uses a
/// palette gives it a place in the frame (`Catalog::add_palette`, by address
/// and length; `unstored`), and that draw's job carries the copy
/// (`QueuedDraw::palette`): the copier reads the words with the job's
/// vertices, the recorder thread stores them and their hash
/// (`Geometry::fill_palette`), and at the frame's end the textures get the
/// hash (`Frame::resolve_palette_hashes`). A palette loaded again from the
/// same address in the same frame is read once.
///
/// When the words are read: the capture always read them when it walked the
/// list, not when the game's GE would have loaded them; now a little later
/// still, when the copier gets to the job (before the game is let go at the
/// GE sync, as for vertices).
pub struct LoadedPalette {
    /// Physical address of the loaded words; 0: `given`.
    address: u32,
    word_count: u32,
    /// Words restored from a saved state that held words, not an address
    /// (dumps made before 2026-11-09, `load_state`): used as they are.
    given: [u32; MAX_PALETTE_WORDS],
    /// Changes whenever another palette is loaded (not when the loaded one
    /// is loaded again), so a draw can tell cheaply that its palette is
    /// still the one it had (`DrawStateCache::fits`).
    generation: u32,
    /// A palette placed in the frame whose words no job carries yet: the
    /// next queued draw takes it (`take_unstored`).
    unstored: Option<PaletteFill>,
    /// Palettes that change every frame, read at the walk's start
    /// (`early_palettes`; the capture reads and notes them).
    pub early: crate::early_palettes::EarlyPalettes,
}

/// A palette placed in the frame (`Catalog::add_palette`) and where its
/// words are read from (physical address; 0: `LoadedPalette::given`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PaletteFill {
    pub place: PalettePlace,
    pub address: u32,
}

impl LoadedPalette {
    pub const fn new() -> Self {
        LoadedPalette {
            address: 0,
            word_count: 0,
            given: [0; MAX_PALETTE_WORDS],
            generation: 0,
            unstored: None,
            early: crate::early_palettes::EarlyPalettes::new(),
        }
    }

    /// What tells loaded palettes apart: address and length.
    fn key(&self) -> (u32, u32) {
        (self.address, self.word_count)
    }

    /// The words restored from a saved state (`given`), for a fill from
    /// address 0.
    pub(crate) fn given_words(&self) -> &[u32] {
        &self.given[..self.word_count as usize]
    }

    /// Back to no palette loaded (a new game).
    pub fn clear(&mut self) {
        self.address = 0;
        self.word_count = 0;
        self.unstored = None;
        self.generation = self.generation.wrapping_add(1);
        self.early.clear();
    }

    /// A new frame: a palette placed in the old one is no concern of it.
    /// The loaded palette stays loaded.
    pub fn next_frame(&mut self) {
        self.unstored = None;
    }

    /// The palette placed in the frame whose words no job carries yet, if
    /// any (the queued draw that takes it carries them).
    #[cfg(test)]
    pub(crate) fn take_unstored(&mut self) -> Option<PaletteFill> {
        self.unstored.take()
    }

    /// `take_unstored` for a draw of `frame`: a palette read at the walk's
    /// start (`early_palettes`: one the game rewrites every frame) is
    /// stored into the frame now, with those words, and none is left for
    /// the job to read (later, when the game may have written the next
    /// frame's).
    pub(crate) fn take_unstored_in(&mut self, frame: *mut Frame, stats: &mut FrameStats) -> Option<PaletteFill> {
        let fill = self.unstored.take()?;
        if fill.address == 0 || frame.is_null() {
            return Some(fill);
        }
        let Some(words) = self.early.words(fill.address, fill.place.word_count) else {
            return Some(fill);
        };
        // Safe: `frame` is the frame being recorded; the place is this
        // palette's own (nobody else writes it).
        let Some((stored, hash)) = (unsafe { Frame::palette_place_raw(frame, fill.place) }) else {
            return Some(fill);
        };
        let count = (fill.place.word_count as usize).min(MAX_PALETTE_WORDS);
        // Safe: `palette_place_raw` checked that `count` words fit there.
        let stored = unsafe { core::slice::from_raw_parts_mut(stored, count) };
        let readable = words.len().min(count);
        stored[..readable].copy_from_slice(&words[..readable]);
        stored[readable..].fill(0);
        // Safe: as above.
        unsafe { *hash = crate::texture::hash_words(stored) };
        stats.early_palettes += 1;
        None
    }

    /// Saves where the palette is (and the words restored from an older
    /// state, if those are loaded).
    pub fn save_state(&self, out: &mut Writer) {
        out.u32(self.address);
        out.u32(self.word_count);
        if self.address == 0 {
            for word in self.given_words() {
                out.u32(*word);
            }
        }
    }

    /// `version` of the snapshot: up to 3, the words were saved (a palette
    /// loaded before the dump started is then used as saved); since 4 the
    /// address and length (its words are read from the dump's memory).
    pub fn load_state(&mut self, input: &mut Reader, version: u32) -> Option<()> {
        let address = if version >= 4 { input.u32()? } else { 0 };
        let word_count = input.u32()? as usize;
        if word_count > MAX_PALETTE_WORDS {
            return None;
        }
        if address == 0 {
            for index in 0..word_count {
                self.given[index] = input.u32()?;
            }
        }
        self.address = address;
        self.word_count = word_count as u32;
        self.unstored = None;
        self.generation = self.generation.wrapping_add(1);
        Some(())
    }

    /// CLUT_LOAD: `blocks` x 32 bytes from the palette address. Only noted.
    fn load(&mut self, ge: &GeState, blocks: u32) {
        let address = physical(ge.clut_address());
        let word_count = (blocks as usize * 8).min(MAX_PALETTE_WORDS) as u32;
        if (address, word_count) != self.key() {
            self.address = address;
            self.word_count = word_count;
            self.generation = self.generation.wrapping_add(1);
        }
    }

    /// The loaded palette's place in the frame, for a draw that uses it;
    /// a new place is noted as `unstored`. `None` if the frame is full.
    fn place_in(&mut self, catalog: &mut Catalog) -> Option<PalettePlace> {
        let placed = catalog.palette_count();
        let place = catalog.add_palette(self.address, self.word_count)?;
        if catalog.palette_count() != placed {
            self.unstored = Some(PaletteFill { place, address: self.address });
        }
        Some(place)
    }
}

/// The opcodes `Recorder` visits (its `wants`, the same for every recorder).
pub fn recorder_wants(opcode: u8) -> bool {
    opcode == op::PRIM || opcode == op::CLUT_LOAD || FrameStats::new_wants(opcode)
}

/// Visits commands and records 2D draws into `frame` and statistics into
/// `stats`. Created for each walk; the palette outlives it.
pub struct Recorder<'a> {
    pub stats: &'a mut FrameStats,
    pub frame: Option<&'a mut Frame>,
    pub palette: &'a mut LoadedPalette,
    pub draw_states: &'a mut DrawStateCache,
    /// Where the draws wait for their vertices to be recorded.
    pub stream: &'a mut DrawStream,
    /// Writes the draws the GPU takes raw itself, on this thread
    /// (`frame_builder`), while recording into a frame; `None`: all go to
    /// the stream.
    pub builder: Option<&'a mut FrameBuilder>,
}

impl CommandVisitor for Recorder<'_> {
    /// Draws and palette loads, plus what the statistics count; everything
    /// else only sets registers.
    #[inline]
    fn wants(&self, opcode: u8) -> bool {
        recorder_wants(opcode)
    }

    /// PRIMs that follow each other directly come as one batch.
    #[inline]
    fn batches_prims(&self) -> bool {
        true
    }

    fn visit<M: PspMemory>(&mut self, command: Command, ge: &GeState, memory: &mut M) {
        let started = crate::cycles::start(crate::cycles::VISIT);
        self.visit_now(command, ge, memory);
        crate::cycles::add(crate::cycles::VISIT, started);
    }

    fn visit_prims<M: PspMemory>(&mut self, prims: &[Command], ge: &GeState, memory: &mut M) {
        crate::cycles::add(crate::cycles::CALIBRATION, crate::cycles::start(crate::cycles::CALIBRATION));
        let started = crate::cycles::start(crate::cycles::PRIMS);
        self.visit_prims_now(prims, ge, memory);
        crate::cycles::add(crate::cycles::PRIMS, started);
    }
}

impl Recorder<'_> {
    fn visit_now<M: PspMemory>(&mut self, command: Command, ge: &GeState, memory: &mut M) {
        crate::phase::enter(crate::phase::RECORD);
        match command.opcode() {
            op::PRIM => self.visit_prims(&[command], ge, memory),
            op::CLUT_LOAD => {
                self.stats.visit(command, ge, memory);
                if !crate::bench::experiment(crate::bench::SKIP_PALETTES) {
                    crate::phase::enter(crate::phase::PALETTE);
                    self.palette.load(ge, command.data() & 0x3F);
                }
            }
            _ => self.stats.visit(command, ge, memory),
        }
    }

    fn visit_prims_now<M: PspMemory>(&mut self, prims: &[Command], ge: &GeState, memory: &mut M) {
        crate::phase::enter(crate::phase::RECORD);
        self.stats.count_prims(prims, ge);
        if let Some(frame) = self.frame.as_deref_mut() {
            // The frame's catalog is the producer's; its geometry belongs to
            // whoever records the queued jobs (`draw_stream`).
            let frame: *mut Frame = frame;
            // Safe: only the producer (this thread) uses the catalog.
            let mut catalog = unsafe { Frame::catalog_raw(frame) };
            let builder = self.builder.as_deref_mut().filter(|builder| builder.is_active());
            let mut queues = Queues { stream: self.stream, builder };
            record_prims(frame, &mut catalog, self.stats, self.palette, self.draw_states, ge, memory, prims, &mut queues);
        }
    }
}

/// VERTEX_TYPE's index format bits (11-12).
const INDEX_FORMAT_BITS: u32 = 3 << 11;

/// One PRIM command and where its data is: each PRIM of a batch starts
/// where the one before ended (`GeState::advance_after_draw`).
#[derive(Clone, Copy)]
pub(crate) struct PrimCall {
    pub primitive_type: u32,
    pub count: u32,
    pub vertex_address: u32,
    pub index_address: u32,
}

impl PrimCall {
    pub(crate) const EMPTY: PrimCall = PrimCall { primitive_type: 0, count: 0, vertex_address: 0, index_address: 0 };
}

/// Records PRIM commands that follow each other directly in the list (one
/// or a batch from `walk`, see `CommandVisitor::visit_prims`). Nothing in
/// the GE state changes between them but the data addresses, so a batch of
/// 3D lists, strips and fans has its state worked out once and its
/// vertices read in one run, like PPSSPP's inner PRIM loop
/// (GPUCommonHW.cpp Execute_Prim) and ExtendNonIndexedPrim. MTX Mototrax:
/// ~88% of its ~10,000 PRIMs per frame follow another PRIM.
///
/// The state is worked out here; the vertices are recorded by `run_job`
/// from the draw stream (`draw_stream`), in the same order.
#[allow(clippy::too_many_arguments)]
fn record_prims(
    frame: *mut Frame,
    catalog: &mut Catalog,
    stats: &mut FrameStats,
    palette: &mut LoadedPalette,
    draw_states: &mut DrawStateCache,
    ge: &GeState,
    memory: &mut impl PspMemory,
    prims: &[Command],
    stream: &mut Queues,
) {
    let format = ge.vertex_format();
    let vertex_size = ge.vertex_layout().size;
    // Only the batch's own calls are written: clearing all 32 first was
    // 512 bytes of stores per batch for nothing.
    let mut calls = [core::mem::MaybeUninit::<PrimCall>::uninit(); MAX_PRIM_BATCH];
    if crate::bench::experiment(crate::bench::ZEROED_CALLS) {
        calls.fill(core::mem::MaybeUninit::new(PrimCall::EMPTY));
    }
    let (mut vertex_address, mut index_address) = (ge.vertex_address, ge.index_address);
    let len = prims.len().min(MAX_PRIM_BATCH);
    for (call, prim) in calls.iter_mut().zip(&prims[..len]) {
        let count = prim.data() & 0xFFFF;
        call.write(PrimCall { primitive_type: (prim.data() >> 16) & 7, count, vertex_address, index_address });
        match format.index_format() {
            IndexFormat::None => vertex_address = vertex_address.wrapping_add(count * vertex_size),
            index => index_address = index_address.wrapping_add(count * index.size()),
        }
    }
    // Safe: the loop wrote the first `len` calls.
    let calls = unsafe { core::slice::from_raw_parts(calls.as_ptr() as *const PrimCall, len) };
    if calls.len() > 1 && can_batch(ge, calls) {
        queue_batch(frame, catalog, stats, palette, draw_states, ge, memory, calls, stream);
    } else if calls.len() > 1 && can_batch_indexed(ge, calls) {
        queue_draw(frame, catalog, stats, palette, draw_states, ge, memory, calls, stream);
    } else {
        for call in calls {
            queue_draw(frame, catalog, stats, palette, draw_states, ge, memory, core::slice::from_ref(call), stream);
        }
    }
}

/// Where the producer's draws go: the stream's job queue, and, while it
/// writes the frame's raw draws itself, the capture thread's builder.
pub(crate) struct Queues<'a> {
    pub stream: &'a mut DrawStream,
    pub builder: Option<&'a mut FrameBuilder>,
}

impl Queues<'_> {
    /// Writes a draw's PRIMs as raw draws on this thread
    /// (`FrameBuilder::try_record`) and queues the palette it placed, if any,
    /// for the recorder to store. `Err(reason)`: queue it as a job
    /// (`fallback` reason, `None` if the builder didn't try).
    fn write_direct(
        &mut self,
        frame: *mut Frame,
        stats: &mut FrameStats,
        memory: &mut impl PspMemory,
        palette: &mut LoadedPalette,
        (prepared, ge, inputs): (&Prepared, &GeState, &TransformInputs),
        calls: &[PrimCall],
    ) -> Result<(), Option<usize>> {
        let Some(builder) = self.builder.as_deref_mut() else {
            return Err(None);
        };
        if crate::bench::experiment(crate::bench::SKIP_QUEUE) {
            return Ok(());
        }
        let format = ge.vertex_format();
        let previous = crate::phase::enter(crate::phase::DIRECT);
        let started = crate::cycles::start(crate::cycles::DIRECT);
        let written = builder.try_record(frame, self.stream, memory, format, ge.vertex_layout(), prepared, ge.material_color(), inputs, calls);
        crate::cycles::add(crate::cycles::DIRECT, started);
        crate::phase::leave(previous);
        written?;
        if let Some(fill) = palette.take_unstored_in(frame, stats) {
            // The copier stores it (and hashes it) straight into the frame,
            // as the recorder would have (`copy_tasks::Layout::Palette`):
            // one copy task instead of a job through the recorder. Words
            // restored from an old state (address 0) still go as a job.
            // Safe: `frame` is the frame being recorded.
            let place = (fill.address != 0 && !crate::bench::experiment(crate::bench::NO_PALETTE_TASKS))
                .then(|| unsafe { Frame::palette_place_raw(frame, fill.place) })
                .flatten();
            if let Some((words, hash)) = place {
                let count = (fill.place.word_count as usize).min(MAX_PALETTE_WORDS) as u32;
                let layout = crate::copy_tasks::Layout::Palette { words: fill.place.word_count, hash };
                let task = crate::copy_tasks::CopyTask { source: fill.address, length: 4 * count, destination: words as *mut u8, layout };
                // False only if copied at once and unreadable: stored as
                // far as readable either way.
                self.stream.copy_task(memory, task);
                return Ok(());
            }
            let draw = QueuedDraw {
                kind: JobKind::Palette,
                prepared: *prepared,
                material_color: 0,
                format: 0,
                calls: &[],
                transform: None,
                memory: [(0, 0), (0, 0)],
                hash_vertices: false,
                zero_copy_vertices: false,
                index_bounds: None,
                palette: Some(fill),
                given_palette: palette.given_words(),
                segment: 0,
                refused: None,
            };
            self.stream.queue(frame, stats, memory, draw);
        }
        Ok(())
    }
}

/// True if a batch of indexed PRIMs can be one job: triangles, strips and
/// fans from one vertex buffer (indexed PRIMs never move the vertex
/// address; their indices follow each other). The job copies the indices
/// once and the vertices once, over the span all of them use, and records
/// the PRIMs one by one as before. Monster Hunter Freedom draws its models
/// one triangle (a 3-index strip) per PRIM, ~1,400 per frame: one job and
/// one copy each, of vertices 0..highest index, was ~3.7 MiB of slow PSP
/// reads per frame (the copier busy ~37 ms).
fn can_batch_indexed(ge: &GeState, calls: &[PrimCall]) -> bool {
    ge.vertex_format().index_format() != IndexFormat::None
        && calls.iter().all(|call| {
            (primitive::TRIANGLES..=primitive::TRIANGLE_FAN).contains(&call.primitive_type)
                && call.vertex_address == calls[0].vertex_address
        })
}

/// True if a batch of these PRIMs can be one job: 3D lists, strips and
/// fans without an index buffer, with a state that is drawn (see
/// `queue_draw`'s first checks, which the PRIMs one by one would make).
fn can_batch(ge: &GeState, calls: &[PrimCall]) -> bool {
    let format = ge.vertex_format();
    !format.is_through()
        && format.index_format() == IndexFormat::None
        && calls.iter().all(|call| (primitive::TRIANGLES..=primitive::TRIANGLE_FAN).contains(&call.primitive_type))
        && !(ge.is_clear_mode() && !ge.clear_writes_color() && !ge.clear_writes_depth() && !ge.clear_writes_stencil())
        && !(writes_no_color(ge) && !writes_depth(ge))
}

/// Queues a batch of 3D PRIMs (see `can_batch`) as one job: the state
/// worked out once, the vertices of all of them (one stretch of memory)
/// copied. Recorded by `record_batch_vertices`.
#[allow(clippy::too_many_arguments)]
fn queue_batch(
    frame: *mut Frame,
    catalog: &mut Catalog,
    stats: &mut FrameStats,
    palette: &mut LoadedPalette,
    draw_states: &mut DrawStateCache,
    ge: &GeState,
    memory: &mut impl PspMemory,
    calls: &[PrimCall],
    queues: &mut Queues,
) {
    let Some(prepared) = prepare(catalog, stats, palette, draw_states, ge) else {
        // The frame is full (counted once); the other PRIMs find the same.
        for &call in &calls[1..] {
            queue_draw(frame, catalog, stats, palette, draw_states, ge, memory, core::slice::from_ref(&call), queues);
        }
        return;
    };
    count_approximations(stats, prepared.approximations, calls.len() as u32);
    sample_texture(catalog, stats, memory, &mut *queues.stream, &prepared);
    let format = ge.vertex_format();
    let vertices = calls.iter().map(|call| call.count).sum::<u32>() * ge.vertex_layout().size;
    let previous = crate::phase::enter(crate::phase::TRANSFORM);
    let transform = draw_states.transforms.get_with_generation(ge, format);
    crate::phase::leave(previous);
    let refused = match queues.write_direct(frame, stats, memory, palette, (&prepared, ge, transform.0), calls) {
        Ok(()) => return,
        Err(refused) => refused,
    };
    let stream = &mut *queues.stream;
    let raw = goes_raw(&prepared, transform.0.lighting(), format);
    let draw = QueuedDraw {
        kind: JobKind::Batch,
        prepared,
        material_color: ge.material_color(),
        format: format.0,
        calls,
        transform: Some(transform),
        memory: [(calls[0].vertex_address, vertices), (0, 0)],
        hash_vertices: !raw,
        zero_copy_vertices: raw && crate::raw_arena::zero_copy_wanted(calls[0].vertex_address, vertices, format),
        index_bounds: None,
        palette: palette.take_unstored_in(frame, stats),
        given_palette: palette.given_words(),
        segment: queues.builder.as_deref_mut().map_or(0, |builder| builder.fallback_segment()),
        refused,
    };
    if crate::bench::experiment(crate::bench::SKIP_QUEUE) {
        return;
    }
    let previous = crate::phase::enter(crate::phase::QUEUE);
    let queue_started = crate::cycles::start(crate::cycles::QUEUE);
    stream.queue(frame, stats, memory, draw);
    crate::cycles::add(crate::cycles::QUEUE, queue_started);
    crate::phase::leave(previous);
}

/// Queues one PRIM, or a batch of indexed ones (`can_batch_indexed`): the
/// state worked out once, the memory of the vertices (and indices) copied.
/// Recorded by `record_vertices`, PRIM by PRIM.
#[allow(clippy::too_many_arguments)]
fn queue_draw(
    frame: *mut Frame,
    catalog: &mut Catalog,
    stats: &mut FrameStats,
    palette: &mut LoadedPalette,
    draw_states: &mut DrawStateCache,
    ge: &GeState,
    memory: &mut impl PspMemory,
    calls: &[PrimCall],
    queues: &mut Queues,
) {
    let call = calls[0];
    if call.primitive_type < primitive::TRIANGLES {
        stats.skipped_point_line_draws += 1;
        return;
    }
    if ge.is_clear_mode() && !ge.clear_writes_color() && !ge.clear_writes_depth() && !ge.clear_writes_stencil() {
        return; // clears nothing
    }
    // Draws that change no visible colour only update alpha, stencil or
    // depth, typically to build a mask for later draws. Drawing them would
    // be wrong (Final Fantasy's load screen: white rectangles that only mark
    // the stencil buffer). Those that write depth are kept, drawn so that
    // they keep the colour (MTX Mototrax: depth-only shapes).
    if writes_no_color(ge) && !writes_depth(ge) {
        stats.color_masked_draws += 1;
        return;
    }
    let Some(prepared) = prepare(catalog, stats, palette, draw_states, ge) else {
        return;
    };
    count_approximations(stats, prepared.approximations, calls.len() as u32);
    sample_texture(catalog, stats, memory, &mut *queues.stream, &prepared);
    let mut refused = None;
    if !ge.vertex_format().is_through() && queues.builder.is_some() {
        let previous = crate::phase::enter(crate::phase::TRANSFORM);
        let (inputs, _) = draw_states.transforms.get_with_generation(ge, ge.vertex_format());
        crate::phase::leave(previous);
        match queues.write_direct(frame, stats, memory, palette, (&prepared, ge, inputs), calls) {
            Ok(()) => return,
            Err(reason) => refused = reason,
        }
    }
    let stream = &mut *queues.stream;

    // The memory the vertices are in: with an index buffer, the indices (of
    // all the calls: they follow each other) and the vertices from the
    // lowest to the highest index (as PPSSPP's GetIndexBounds; reads find
    // their words by address).
    let format = ge.vertex_format();
    let vertex_size = ge.vertex_layout().size;
    let index_format = format.index_format();
    let mut producer_bounds = None;
    let memory_ranges = if index_format == IndexFormat::None {
        [(call.vertex_address, call.count * vertex_size), (0, 0)]
    } else {
        let count = calls.iter().map(|call| call.count).sum::<u32>();
        let previous = crate::phase::enter(crate::phase::INDICES);
        let bounds = if crate::bench::experiment(crate::bench::SKIP_HIGHEST_INDEX) {
            Some((0, count.saturating_sub(1)))
        } else {
            index_bounds(memory, index_format, call.index_address, count)
        };
        crate::phase::leave(previous);
        producer_bounds = bounds;
        let vertices = bounds.map_or((call.vertex_address, 0), |(lowest, highest)| {
            (call.vertex_address.wrapping_add(lowest.saturating_mul(vertex_size)), (highest - lowest + 1).saturating_mul(vertex_size))
        });
        [(call.index_address, count * index_format.size()), vertices]
    };
    let previous = crate::phase::enter(crate::phase::TRANSFORM);
    let transform = (!format.is_through()).then(|| draw_states.transforms.get_with_generation(ge, format));
    crate::phase::leave(previous);
    if crate::bench::experiment(crate::bench::SKIP_QUEUE) {
        return;
    }
    let draw = QueuedDraw {
        kind: JobKind::Single,
        prepared,
        material_color: ge.material_color(),
        format: format.0,
        calls,
        transform,
        memory: memory_ranges,
        // Only runs read the vertex cache (`record_3d_run`: no index buffer),
        // and not those recorded raw.
        hash_vertices: transform.is_some_and(|(inputs, _)| {
            format.index_format() == crate::vertex::IndexFormat::None
                && (call.primitive_type == primitive::RECTANGLES || !goes_raw(&prepared, inputs.lighting(), format))
        }),
        // Single PRIMs are mostly small; their vertices are copied. An
        // indexed batch's vertex span can stay in PSP RAM where its indices
        // will be converted (`record_raw_converted`).
        zero_copy_vertices: transform.is_some_and(|(inputs, _)| {
            index_format != IndexFormat::None
                && crate::raw_arena::INDEXED_DRAWS.load(core::sync::atomic::Ordering::Relaxed)
                && goes_raw(&prepared, inputs.lighting(), format)
                && calls.iter().all(|call| (primitive::TRIANGLES..=primitive::TRIANGLE_FAN).contains(&call.primitive_type))
                && crate::raw_arena::zero_copy_wanted(memory_ranges[1].0, memory_ranges[1].1, format)
        }),
        index_bounds: producer_bounds,
        palette: palette.take_unstored_in(frame, stats),
        given_palette: palette.given_words(),
        segment: queues.builder.as_deref_mut().map_or(0, |builder| builder.fallback_segment()),
        refused,
    };
    let previous = crate::phase::enter(crate::phase::QUEUE);
    let queue_started = crate::cycles::start(crate::cycles::QUEUE);
    stream.queue(frame, stats, memory, draw);
    crate::cycles::add(crate::cycles::QUEUE, queue_started);
    crate::phase::leave(previous);
}

/// The lowest and highest of the `count` indices at `address` (`None`:
/// none readable; indices after the first unreadable one don't count), to
/// know which vertices an indexed draw reads.
///
/// Reads whole words, a chunk at a time (like PPSSPP's GetIndexBounds over
/// a pointer): MTX's race frames have ~100,000 indices, and reading each one
/// through `read_u16` was a noticeable part of the capture thread's time.
fn index_bounds(memory: &mut impl PspMemory, format: IndexFormat, address: u32, count: u32) -> Option<(u32, u32)> {
    const CHUNK_WORDS: usize = 64;
    let index_size = format.size();
    if index_size == 0 || count == 0 {
        return None;
    }
    if address % index_size != 0 {
        // Misaligned indices would straddle words: index by index.
        return bounds_one_by_one(memory, format, address, 0, count);
    }
    let end = address.wrapping_add(count * index_size);
    let mut bounds: Option<(u32, u32)> = None;
    let mut word_address = address & !3;
    let mut words = [0u32; CHUNK_WORDS];
    while word_address < end {
        let chunk = (((end - word_address + 3) / 4) as usize).min(CHUNK_WORDS);
        if memory.read_words(word_address, &mut words[..chunk]).is_none() {
            // Part of it is unreadable: index by index up to that point.
            let first = (word_address.max(address) - address) / index_size;
            let rest = bounds_one_by_one(memory, format, address, first, count);
            return match (bounds, rest) {
                (Some(a), Some(b)) => Some(widen(widen(a, b.0), b.1)),
                _ => bounds.or(rest),
            };
        }
        for (number, &word) in words[..chunk].iter().enumerate() {
            let at = word_address + 4 * number as u32;
            let mut offset = 0;
            while offset < 4 {
                let index_at = at + offset;
                if index_at >= address && index_at < end {
                    let value = (word >> (8 * offset)) & if index_size == 4 { u32::MAX } else { (1 << (8 * index_size)) - 1 };
                    bounds = Some(bounds.map_or((value, value), |bounds| widen(bounds, value)));
                }
                offset += index_size;
            }
        }
        word_address += 4 * chunk as u32;
    }
    bounds
}

pub(crate) fn widen((low, high): (u32, u32), value: u32) -> (u32, u32) {
    (low.min(value), high.max(value))
}

/// `index_bounds` index by index, from index `first`, up to the first
/// unreadable one.
fn bounds_one_by_one(memory: &mut impl PspMemory, format: IndexFormat, address: u32, first: u32, count: u32) -> Option<(u32, u32)> {
    let size = format.size();
    (first..count)
        .map_while(|n| read_index(memory, format, address + n * size))
        .fold(None, |bounds, value| Some(bounds.map_or((value, value), |bounds| widen(bounds, value))))
}

fn read_index(memory: &mut impl PspMemory, format: IndexFormat, at: u32) -> Option<u32> {
    match format {
        IndexFormat::U8 => memory.read_u8(at).map(u32::from),
        IndexFormat::U16 => memory.read_u16(at).map(u32::from),
        _ => memory.read_u32(at),
    }
}

/// Records a queued job's vertices into the geometry (the second half of
/// recording a draw, see `draw_stream`). `memory` holds the job's copied
/// memory (or, for a job too big to queue, PSP memory itself).
/// `run_job_now`, measured (`cycles::JOBS`).
/// Stores the words of a palette the job's draw placed in the frame
/// (`LoadedPalette`): copied with the job's memory (the copier read them),
/// or read now where the job reads PSP memory directly. Zeros if they
/// can't be read.
fn store_palette(geometry: &mut Geometry, fill: PaletteFill, memory: &mut impl PspMemory) {
    let count = (fill.place.word_count as usize).min(MAX_PALETTE_WORDS);
    if let Some(words) = memory.borrow_words(fill.address, count) {
        geometry.fill_palette(fill.place, words);
        return;
    }
    let mut words = [0u32; MAX_PALETTE_WORDS];
    let mut readable = count;
    if memory.read_words(fill.address, &mut words[..count]).is_none() {
        // Partly unreadable: what can be read, up to the first word that can't.
        readable = (0..count)
            .take_while(|&index| memory.read_u32(fill.address + 4 * index as u32).map(|word| words[index] = word).is_some())
            .count();
    }
    geometry.fill_palette(fill.place, &words[..readable]);
}

/// A 3D job's transform: its inputs, which is all a raw draw needs
/// (`raw_blocks`: the vertex program multiplies), and the CPU's `Transform`,
/// worked out from them only for draws decoded on the CPU (`built`).
pub(crate) struct JobTransform<'a> {
    pub inputs: &'a TransformInputs,
    builder: &'a mut TransformBuilder,
    /// Why the capture thread's builder turned the job's draw down
    /// (`QueuedDraw::refused`), for `fallback_reason`.
    pub refused: Option<usize>,
}

impl<'a> JobTransform<'a> {
    pub(crate) fn new(inputs: &'a TransformInputs, builder: &'a mut TransformBuilder, refused: Option<usize>) -> Self {
        JobTransform { inputs, builder, refused }
    }

    /// The CPU's transform (`TransformBuilder::build`: redone only where the
    /// inputs changed since the last one built).
    fn built(&mut self) -> &Transform {
        let building = crate::phase::recorder::enter(crate::phase::recorder::TRANSFORM);
        let transform = self.builder.build(self.inputs);
        crate::phase::recorder::leave(building);
        transform
    }
}

pub(crate) fn run_job(
    geometry: &mut Geometry,
    stats: &mut FrameStats,
    job: &Job,
    calls: &[PrimCall],
    transform: Option<JobTransform>,
    memory: &mut impl PspMemory,
    cache: Option<&mut VertexCache>,
) {
    let started = crate::cycles::start(crate::cycles::JOBS);
    run_job_now(geometry, stats, job, calls, transform, memory, cache);
    crate::cycles::add(crate::cycles::JOBS, started);
}

fn run_job_now(
    geometry: &mut Geometry,
    stats: &mut FrameStats,
    job: &Job,
    calls: &[PrimCall],
    mut transform: Option<JobTransform>,
    memory: &mut impl PspMemory,
    mut cache: Option<&mut VertexCache>,
) {
    if let Some(fill) = job.palette {
        let previous = crate::phase::recorder::enter(crate::phase::recorder::PALETTE);
        store_palette(geometry, fill, memory);
        crate::phase::recorder::leave(previous);
    }
    if job.kind == JobKind::Palette {
        return;
    }
    let format = VertexFormat(job.format);
    let layout = format.layout();
    match (job.kind, transform.as_mut()) {
        (JobKind::Batch, Some(transform)) => record_batch_vertices(
            geometry,
            stats,
            memory,
            format,
            &layout,
            transform,
            &job.prepared,
            job.material_color,
            calls,
            cache,
        ),
        (_, mut transform) => {
            if let Some(transform) = transform.as_deref() {
                let indexed = (&job.prepared, job.material_color, job.index_bounds);
                if record_raw_indexed(geometry, memory, format, &layout, transform.inputs, indexed, calls, cache.as_deref_mut()) {
                    return;
                }
            }
            for &call in calls {
                record_vertices(
                    geometry,
                    stats,
                    memory,
                    format,
                    &layout,
                    transform.as_deref_mut(),
                    &job.prepared,
                    job.material_color,
                    call,
                    cache.as_deref_mut(),
                );
            }
        }
    }
}

/// Indexed 3D PRIMs (a job of one PRIM or of several, `can_batch_indexed`)
/// as one raw draw for the GPU (`record_raw_run`). False (nothing recorded)
/// where they can't go raw: the CPU path then records them PRIM by PRIM.
/// Monster Hunter Freedom draws everything indexed: on the CPU path a
/// village frame (~41,000 vertices) kept the recorder busy ~150 ms.
#[allow(clippy::too_many_arguments)]
fn record_raw_indexed(
    geometry: &mut Geometry,
    memory: &mut impl PspMemory,
    format: VertexFormat,
    layout: &VertexLayout,
    inputs: &TransformInputs,
    (prepared, material_color, bounds): (&Prepared, u32, Option<(u32, u32)>),
    calls: &[PrimCall],
    cache: Option<&mut VertexCache>,
) -> bool {
    if format.index_format() == IndexFormat::None
        || format.is_through()
        || !calls.iter().all(|call| (primitive::TRIANGLES..=primitive::TRIANGLE_FAN).contains(&call.primitive_type))
    {
        return false;
    }
    let Some(fields) = raw_colors_if_enabled(prepared, material_color, inputs.lighting(), format) else {
        return false;
    };
    let draw = Draw { first_vertex: geometry.vertex_count(), first_index: geometry.index_count(), ..prepared.draw };
    record_raw_draw(geometry, memory, format, layout, inputs, (calls, bounds), &fields, cache, draw)
}

/// A batch's vertices: chunks of up to RUN_VERTICES vertices read, decoded
/// and transformed in one run each and drawn as one draw. A chunk
/// `record_3d_run` doesn't take (a vertex behind the near plane, unreadable
/// data, the geometry full) goes through `record_vertices` PRIM by PRIM, so
/// the outcome is what the PRIMs one by one would give.
#[allow(clippy::too_many_arguments)]
fn record_batch_vertices(
    geometry: &mut Geometry,
    stats: &mut FrameStats,
    memory: &mut impl PspMemory,
    format: VertexFormat,
    layout: &VertexLayout,
    job_transform: &mut JobTransform,
    prepared: &Prepared,
    material_color: u32,
    calls: &[PrimCall],
    mut cache: Option<&mut VertexCache>,
) {
    let color_of = RunColor { prepared, material_color };
    let colors = prepared.color_key(material_color);
    // Raw PSP vertices for the GPU (`raw_arena`), where the draw allows it.
    let inputs = job_transform.inputs;
    let raw_colors = raw_colors_if_enabled(prepared, material_color, inputs.lighting(), format);
    let mut start = 0;
    while start < calls.len() {
        let mut end = start + 1;
        let mut vertices = calls[start].count as usize;
        while end < calls.len() && vertices + calls[end].count as usize <= RUN_VERTICES {
            vertices += calls[end].count as usize;
            end += 1;
        }
        let chunk = &calls[start..end];
        let mut draw = Draw { first_vertex: geometry.vertex_count(), first_index: geometry.index_count(), ..prepared.draw };
        if let Some(fields) = raw_colors.as_ref() {
            if record_raw_draw(geometry, memory, format, layout, inputs, (chunk, None), fields, cache.as_deref_mut(), draw) {
                start = end;
                continue;
            }
        }
        let raw = cache.as_deref().and_then(|cache| cache.raw.as_ref());
        let reason = fallback_reason(prepared, Some(inputs), format, chunk[0].primitive_type, raw, job_transform.refused);
        let gpu = GpuRun { wanted: prepared.gpu_transform, cache: cache.as_deref_mut(), colors, lighting: prepared.gpu_lighting() };
        let transform = job_transform.built();
        let previous = crate::phase::recorder::enter(crate::phase::recorder::CPU_DECODE);
        // Texgen draws: per PRIM below (`record_vertices_now`'s corner path).
        let run = if prepared.texgen.is_some() {
            Run::NotHandled
        } else {
            record_3d_run(geometry, memory, format, layout, transform, chunk, draw.first_vertex, color_of, gpu)
        };
        crate::phase::recorder::leave(previous);
        match run {
            Run::Recorded { transform, cached } => {
                // (Chunks the run path turns down are counted PRIM by PRIM
                // in `record_vertices_now`.)
                count_fallback(stats, reason, chunk.len() as u32, chunk.iter().map(|call| call.count).sum());
                draw.transform = transform;
                draw.flags |= draw_flags::INDEX_RUNS;
                if let Some(window) = cached {
                    geometry.finish_cached_draw(draw, window);
                } else {
                    geometry.finish_draw(draw);
                }
            }
            Run::FrameFull | Run::NotHandled => {
                geometry.undo_draw(&draw);
                for &call in chunk {
                    record_vertices(
                        geometry,
                        stats,
                        memory,
                        format,
                        layout,
                        Some(&mut *job_transform),
                        prepared,
                        material_color,
                        call,
                        cache.as_deref_mut(),
                    );
                }
            }
        }
        start = end;
    }
}

/// What a draw's GE state says before its vertices are read (`prepare`).
#[derive(Clone, Copy)]
pub(crate) struct Prepared {
    /// The draw without its vertex and index ranges and the RECTANGLES flag.
    pub draw: Draw,
    /// Approximations to count for every PRIM drawn with it (`approx` mask).
    approximations: u32,
    /// Through mode: 1 / texture size (texel coordinates to 0..1); 3D: 1.
    texel_scale: (f32, f32),
    /// The vertex alpha counts (blending, or a clear writing the stencil):
    /// otherwise vertices are made opaque.
    blending: bool,
    /// Source factor "fixed colour": multiplied into the colours instead.
    source_scale: Option<u32>,
    /// Doubled source alpha factors: the vertex alpha is doubled instead.
    double_source_alpha: bool,
    /// TEXTURE_FUNCTION bits 0..3, and whether the texture alpha is used.
    function: u32,
    uses_texture_alpha: bool,
    /// The GPU transforms this draw's vertices where the fast path can
    /// (`DrawStateCache::gpu_transform`; textured 3D draws only, so far).
    gpu_transform: bool,
    /// Lit 3D draws without lights: their colours (`AmbientLighting`).
    ambient_lighting: Option<AmbientLighting>,
    /// Generated texture coordinates (TEXTURE_MAP_MODE 1, `Texgen`): the
    /// draw takes neither the GPU transform nor the run path, and the
    /// per-corner path sets each vertex's u, v from it.
    texgen: Option<Texgen>,
}

/// The colour of a lit 3D vertex when no light is on (MTX's whole world,
/// ~8,100 PRIMs per race frame): the GE's lighting reduces to the material
/// emission plus the global ambient light times the material's ambient
/// colour, which is the vertex colour when MATERIAL_UPDATE says so. Per
/// channel, alpha too, rounded as PPSSPP's software lighting does it
/// (GPU/Software/Lighting.cpp, `ProcessSIMD`: factors 2c + 1, product
/// >> 10, clamped).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AmbientLighting {
    /// AMBIENT_COLOR | AMBIENT_ALPHA << 24.
    ambient: u32,
    /// MATERIAL_EMISSIVE (RGB; adds nothing to alpha).
    emissive: u32,
    /// MATERIAL_UPDATE bit 0: the vertex colour is the material's ambient.
    vertex_ambient: bool,
    /// MATERIAL_AMBIENT | MATERIAL_ALPHA << 24, for vertices without a
    /// colour or when the vertex colour isn't used.
    material: u32,
}

impl AmbientLighting {
    /// For a 3D draw with the current GE state: `Some` if it is lit with
    /// no light on.
    fn of(ge: &GeState) -> Option<AmbientLighting> {
        let lit = ge.register(op::LIGHTING_ENABLE) & 1 != 0;
        let any_light = (0..4).any(|light| ge.register(op::LIGHT_ENABLE_0 + light) & 1 != 0);
        (lit && !any_light).then(|| AmbientLighting {
            ambient: (ge.register(op::AMBIENT_COLOR) & 0x00FF_FFFF) | (ge.register(op::AMBIENT_ALPHA) & 0xFF) << 24,
            emissive: ge.register(op::MATERIAL_EMISSIVE) & 0x00FF_FFFF,
            vertex_ambient: ge.register(op::MATERIAL_UPDATE) & 1 != 0,
            material: ge.material_color(),
        })
    }

    /// The lit colour of a vertex with colour `vertex` (`None`: none).
    fn color(&self, vertex: Option<u32>) -> u32 {
        let material = match vertex {
            Some(color) if self.vertex_ambient => color,
            _ => self.material,
        };
        let mut lit = 0;
        for shift in [0, 8, 16, 24] {
            let channel = |word: u32| (word >> shift) & 0xFF;
            let ambient = ((2 * channel(material) + 1) * (2 * channel(self.ambient) + 1)) >> 10;
            lit |= (ambient + channel(self.emissive)).min(255) << shift;
        }
        lit
    }
}

impl Prepared {
    /// A vertex's colour before `frame_color`: its own (`None`: none) or
    /// the material colour, lit if the draw is (`AmbientLighting`).
    #[inline]
    fn base_color(&self, vertex: Option<u32>, material_color: u32) -> u32 {
        match &self.ambient_lighting {
            Some(lighting) => lighting.color(vertex),
            None => vertex.unwrap_or(material_color),
        }
    }

    pub(crate) const EMPTY: Prepared = Prepared {
        draw: crate::draw_stream::EMPTY_DRAW,
        approximations: 0,
        texel_scale: (1.0, 1.0),
        blending: false,
        source_scale: None,
        double_source_alpha: false,
        function: 0,
        uses_texture_alpha: false,
        gpu_transform: false,
        ambient_lighting: None,
        texgen: None,
    };

    /// Whether the GPU may light this draw's lit vertices (nativeres_3d_v.cg):
    /// `Some(opaque)` if it gets the GPU transform and `frame_color` does
    /// nothing to their colour but, for unblended draws, make it opaque
    /// (`GpuLights::material_diffuse` a). Others are lit on the CPU.
    fn gpu_lighting(&self) -> Option<bool> {
        let textured = self.draw.flags & draw_flags::TEXTURED != 0;
        let plain = self.source_scale.is_none() && !self.double_source_alpha && (!textured || self.function == TEXTURE_FUNCTION_MODULATE);
        (self.gpu_transform && plain).then_some(!self.blending)
    }

    /// Identifies how `frame_color` turns a vertex colour (with this
    /// material colour) into the stored one (`vertex_cache::RunKey`).
    fn color_key(&self, material_color: u32) -> u32 {
        let textured = self.draw.flags & draw_flags::TEXTURED != 0;
        let lighting = self.ambient_lighting;
        let bits = [
            material_color,
            self.function,
            textured as u32 | (self.blending as u32) << 1 | (self.double_source_alpha as u32) << 2 | (self.uses_texture_alpha as u32) << 3,
            self.source_scale.map_or(0, |scale| scale | 1 << 31),
            lighting.map_or(0, |lighting| lighting.ambient),
            lighting.map_or(0, |lighting| lighting.emissive | (lighting.vertex_ambient as u32) << 31 | 1 << 30),
            lighting.map_or(0, |lighting| lighting.material),
        ];
        bits.iter().fold(0x811C_9DC5u32, |hash, &word| (hash ^ word).wrapping_mul(0x0100_0193))
    }

    /// The colour a vertex ends up with: for textured draws the colour the
    /// texture is multiplied with.
    #[inline]
    fn frame_color(&self, color: u32) -> u32 {
        let textured = self.draw.flags & draw_flags::TEXTURED != 0;
        let color = if self.blending { color } else { opaque(color) }; // unblended: written as is, alpha doesn't matter
        let color = if self.double_source_alpha { double_alpha(color) } else { color };
        let color = match self.function {
            // Replace: the texture colour as is. Its alpha too, unless the
            // texture alpha is switched off; then the vertex alpha is used.
            TEXTURE_FUNCTION_REPLACE if textured && self.uses_texture_alpha => 0xFFFF_FFFF,
            TEXTURE_FUNCTION_REPLACE if textured => 0x00FF_FFFF | (color & 0xFF00_0000),
            _ => color,
        };
        match self.source_scale {
            Some(scale) => scale_color(color, scale),
            None => color,
        }
    }
}

/// Works out the state part of a draw (reusing the last draw's when the
/// GE state is the same, see `DrawStateCache`). `None` if the draw can't be
/// recorded (the frame is full).
fn prepare(
    catalog: &mut Catalog,
    stats: &mut FrameStats,
    palette: &mut LoadedPalette,
    draw_states: &mut DrawStateCache,
    ge: &GeState,
) -> Option<Prepared> {
    let previous = crate::phase::enter(crate::phase::PREPARE);
    let started = crate::cycles::start(crate::cycles::PREPARE);
    let prepared = prepare_now(catalog, stats, palette, draw_states, ge);
    crate::cycles::add(crate::cycles::PREPARE, started);
    crate::phase::leave(previous);
    prepared
}

fn prepare_now(
    catalog: &mut Catalog,
    stats: &mut FrameStats,
    palette: &mut LoadedPalette,
    draw_states: &mut DrawStateCache,
    ge: &GeState,
) -> Option<Prepared> {
    // Most draws have the same state as the one before (FF1's menu: 94%).
    let keep = crate::bench::experiment(crate::bench::SKIP_STATE_WORK) && draw_states.state.is_some();
    if !keep && !draw_states.fits(ge, palette, catalog) {
        let Some(state) = work_out_draw_state(catalog, stats, palette, ge) else {
            draw_states.forget();
            return None;
        };
        draw_states.store(state, ge, palette);
    }
    let state = draw_states.state?;
    let mut draw = state.draw;
    let mut approximations = state.approximations;
    let textured = draw.flags & draw_flags::TEXTURED != 0;

    // Textured draws: how the texture is combined with the vertex colour.
    let texture_function = ge.register(op::TEXTURE_FUNCTION);
    let function = texture_function & 7;
    let uses_texture_alpha = texture_function & (1 << 8) != 0;
    if textured {
        if texture_function & (1 << 16) != 0 {
            draw.flags |= draw_flags::DOUBLE_COLOR;
        }
        if !(function == TEXTURE_FUNCTION_MODULATE || function == TEXTURE_FUNCTION_REPLACE) {
            approximations |= approx::TEXTURE_FUNCTION;
        }
    }

    // 3D: transformed to screen pixels, texture coordinates already
    // fractions of the texture (through mode has texels).
    let format = ge.vertex_format();
    let texel_scale = if format.is_through() { state.texel_scale } else { (1.0, 1.0) };
    let mut ambient_lighting = None;
    let texgen = if textured && !format.is_through() { Texgen::of(ge, format) } else { None };
    if !format.is_through() && !crate::bench::experiment(crate::bench::SKIP_TRANSFORM_LOOKUP) {
        let transform = draw_states.transforms.get(ge, format);
        if transform.has_fog() {
            draw.flags |= draw_flags::FOG;
            draw.fog_color = ge.register(op::FOG_COLOR) & 0x00FF_FFFF;
            if !textured {
                approximations |= approx::UNTEXTURED_FOG;
            }
        }
        // Lit without lights: `AmbientLighting`; with lights: the
        // transform's `Lights`, approximated only where the GE would add a
        // separate specular colour after texturing.
        // (Lit with lights while `LIGHTS_ON` switched them off: drawn
        // unlit, counted.)
        ambient_lighting = AmbientLighting::of(ge);
        let lit = ge.register(op::LIGHTING_ENABLE) & 1 != 0;
        // Separate specular: exact where the GPU lights the draw (its
        // shader adds the specular part after texturing) and for untextured
        // draws (clamped parts added, the same as clamping the sum).
        // Texgen draws stay on the CPU route (texgen_run.rs).
        let gpu_lit = draw_states.gpu_transform && textured && !format.is_through() && texgen.is_none();
        match transform.lighting() {
            Some(lights) if lights.approximated() && textured && !(gpu_lit && lights.raw_ready(format)) => {
                approximations |= approx::SEPARATE_SPECULAR
            }
            None if lit && ambient_lighting.is_none() => approximations |= approx::LIGHTS_LEFT_OUT,
            _ => {}
        }
        // Texgen: exact for mode 1 (`Texgen`) unless projective (divided
        // per vertex); mode 2 (shade mapping) not done yet.
        if textured && ge.register(op::TEXTURE_MAP_MODE) & 3 != 0 && texgen.is_none_or(|texgen| texgen.projective) {
            approximations |= approx::TEXGEN;
        }
    }

    Some(Prepared {
        draw,
        approximations,
        texel_scale,
        blending: state.blending,
        source_scale: state.source_scale,
        double_source_alpha: state.double_source_alpha,
        function,
        uses_texture_alpha,
        gpu_transform: draw_states.gpu_transform && textured && !format.is_through() && texgen.is_none(),
        ambient_lighting,
        texgen,
    })
}

/// `record_vertices_now`, measured whole (`cycles::GENERAL`).
#[allow(clippy::too_many_arguments)]
fn record_vertices(
    geometry: &mut Geometry,
    stats: &mut FrameStats,
    memory: &mut impl PspMemory,
    format: VertexFormat,
    layout: &VertexLayout,
    transform: Option<&mut JobTransform>,
    prepared: &Prepared,
    material_color: u32,
    call: PrimCall,
    cache: Option<&mut VertexCache>,
) {
    let started = crate::cycles::start(crate::cycles::GENERAL);
    record_vertices_now(geometry, stats, memory, format, layout, transform, prepared, material_color, call, cache);
    crate::cycles::add(crate::cycles::GENERAL, started);
}

/// Records one PRIM's vertices (its state worked out already, `queue_draw`).
#[allow(clippy::too_many_arguments)]
fn record_vertices_now(
    geometry: &mut Geometry,
    stats: &mut FrameStats,
    memory: &mut impl PspMemory,
    format: VertexFormat,
    layout: &VertexLayout,
    transform: Option<&mut JobTransform>,
    prepared: &Prepared,
    material_color: u32,
    call: PrimCall,
    mut cache: Option<&mut VertexCache>,
) {
    let PrimCall { primitive_type, count, .. } = call;
    let mut draw = Draw { first_vertex: geometry.vertex_count(), first_index: geometry.index_count(), ..prepared.draw };
    let inputs = transform.as_ref().map(|transform| transform.inputs);
    if primitive_type == primitive::RECTANGLES {
        // Rectangles are never culled (PPSSPP: no cull for GE_PRIM_RECTANGLES).
        draw.flags = (draw.flags | draw_flags::RECTANGLES) & !(draw_flags::CULL | draw_flags::CULL_CCW);
        // 3D: raw sprites for the GPU where the draw allows it.
        if let Some(inputs) = inputs {
            if record_raw_sprites(geometry, memory, format, layout, inputs, prepared, material_color, call, cache.as_deref_mut(), draw) {
                return;
            }
        }
    } else if let (Some(inputs), IndexFormat::None) = (inputs, format.index_format()) {
        // Raw PSP vertices for the GPU where the draw allows it, as a
        // batch's runs (MTX: ~440 single PRIMs per race frame).
        if let Some(fields) = raw_colors_if_enabled(prepared, material_color, inputs.lighting(), format) {
            if record_raw_draw(geometry, memory, format, layout, inputs, (&[call], None), &fields, cache.as_deref_mut(), draw) {
                return;
            }
        }
    }
    // Everything below decodes the vertices on the CPU.
    let raw = cache.as_deref().and_then(|cache| cache.raw.as_ref());
    let refused = transform.as_ref().and_then(|transform| transform.refused);
    count_fallback(stats, fallback_reason(prepared, inputs, format, primitive_type, raw, refused), 1, count);
    let transform = transform.map(|transform| transform.built());
    let decoding = crate::phase::recorder::enter(crate::phase::recorder::CPU_DECODE);
    record_on_cpu(geometry, stats, memory, format, layout, transform, prepared, material_color, call, cache, draw);
    crate::phase::recorder::leave(decoding);
}

/// `record_vertices_now`'s CPU half: the vertices decoded (and for 3D
/// transformed, or left to the GPU transform through the vertex cache)
/// into the frame's vertices.
#[allow(clippy::too_many_arguments)]
fn record_on_cpu(
    geometry: &mut Geometry,
    stats: &mut FrameStats,
    memory: &mut impl PspMemory,
    format: VertexFormat,
    layout: &VertexLayout,
    transform: Option<&Transform>,
    prepared: &Prepared,
    material_color: u32,
    call: PrimCall,
    cache: Option<&mut VertexCache>,
    mut draw: Draw,
) {
    let PrimCall { primitive_type, count, .. } = call;
    let texel_scale = prepared.texel_scale;
    let first_vertex = draw.first_vertex;

    let frame_color = |color: u32| prepared.frame_color(color);
    let to_frame = |vertex: &Vertex, color: u32, fog: f32, w: f32| FrameVertex {
        x: vertex.x,
        y: vertex.y,
        // The PSP clamps depth to 0..65535 (per pixel; per vertex here).
        z: vertex.z.clamp(0.0, 65535.0) / 65535.0,
        w,
        u: vertex.u * texel_scale.0,
        v: vertex.v * texel_scale.1,
        color: frame_color(color),
        fog,
    };
    let color_of = |vertex: &Vertex| prepared.base_color(vertex.color, material_color);

    // 3D strips, fans and triangle lists of consecutive vertices: read and
    // transform the whole run at once.
    if let Some(transform) = transform {
        if primitive_type != primitive::RECTANGLES && format.index_format() == IndexFormat::None && prepared.texgen.is_none() {
            let run_color = RunColor { prepared, material_color };
            let gpu = GpuRun { wanted: prepared.gpu_transform, cache, colors: prepared.color_key(material_color), lighting: prepared.gpu_lighting() };
            match record_3d_run(geometry, memory, format, layout, transform, &[call], first_vertex, run_color, gpu) {
                Run::Recorded { transform, cached } => {
                    draw.transform = transform;
                    draw.flags |= draw_flags::INDEX_RUNS;
                    if let Some(window) = cached {
                        geometry.finish_cached_draw(draw, window);
                    } else {
                        geometry.finish_draw(draw);
                    }
                    return;
                }
                Run::FrameFull => {
                    geometry.undo_draw(&draw);
                    stats.frame_overflows += 1;
                    return;
                }
                Run::NotHandled => {} // the general path below
            }
        }
    }

    // Lit 3D draws with lights: colours from the normals (`Lights`). Worked
    // out only here: most lit runs took the run path above.
    let lit = transform.and_then(Transform::vertex_lighting);

    // Unskinned texgen draws of consecutive vertices (lit or not, any
    // source): four vertices at a time (`record_texgen_run`),
    // the same vertices and triangles as below. Checking (TEXGEN_RUNS 2):
    // recorded both ways, the results compared (`texgen_check`).
    let mut texgen_check_hash = None;
    if let (Some(transform), Some(texgen)) = (transform, prepared.texgen.as_ref()) {
        let shape = primitive_type != primitive::RECTANGLES && format.index_format() == IndexFormat::None;
        let mode = TEXGEN_RUNS.load(core::sync::atomic::Ordering::Relaxed);
        if shape && mode != 0 {
            if let Some(kernel) = transform.texgen_kernel(texgen) {
                let run = TexgenRun::new(kernel, prepared, material_color, lit.as_ref(), *texgen, format, layout, call, first_vertex);
                let started = crate::cycles::start(crate::cycles::TEXGEN_RUN);
                let recorded = record_texgen_run(geometry, memory, run, transform, &color_of, &to_frame, call);
                crate::cycles::add(crate::cycles::TEXGEN_RUN, started);
                match recorded {
                    Some(recorded) if mode == 1 => return finish_on_cpu(geometry, stats, call, &draw, recorded),
                    Some(recorded) => {
                        texgen_check_hash = Some(recorded_hash(geometry, &draw, &recorded));
                        geometry.undo_draw(&draw);
                    }
                    None => geometry.undo_draw(&draw), // the general path below
                }
            }
        }
    }

    // Read the vertices.
    let general_phase = crate::phase::recorder::enter(match prepared.texgen {
        Some(_) => crate::phase::recorder::TEXGEN_GENERAL,
        None => crate::phase::recorder::CPU_DECODE,
    });
    let mut corners = VertexReader::new(format, layout, memory, call.vertex_address, call.index_address);
    let mut screen_corners = CornerCache::new();

    let recorded: Result<(), Failure> = match primitive_type {
        primitive::RECTANGLES => {
            // Each pair of vertices gives opposite corners; the second
            // vertex's colour is used for the whole rectangle.
            (0..count / 2).try_for_each(|pair| {
                let (Some(a), Some(b)) = (corners.get(pair * 2), corners.get(pair * 2 + 1)) else {
                    return Err(Failure::Unreadable);
                };
                let color = match &lit {
                    Some(lighting) => corners.lit_color(pair * 2 + 1, lighting).ok_or(Failure::Unreadable)?,
                    None => color_of(&b),
                };
                // 3D: both corners on the screen. A rectangle can't be cut
                // at the near plane; one reaching behind it is left out.
                // It is flat on the screen, so drawn without perspective (w = 1).
                let (a, b) = match transform.as_ref() {
                    None => ((a, 1.0), (b, 1.0)),
                    Some(transform) => match (transform.apply_with_fog(&a, color), transform.apply_with_fog(&b, color)) {
                        (Some(a), Some(b)) => (a, b),
                        _ => return Ok(()),
                    },
                };
                let (mut a, b) = (to_frame(&a.0, color, a.1, 1.0), to_frame(&b.0, color, b.1, 1.0));
                a.z = b.z; // like the colour, the depth of the second vertex
                let top_right = FrameVertex { x: b.x, u: b.u, ..a };
                let bottom_left = FrameVertex { x: a.x, u: a.u, ..b };
                // Six vertices in order (the display side keeps the filter
                // inside each sprite by looking at its six corners).
                for corner in [a, top_right, bottom_left, top_right, b, bottom_left] {
                    geometry.add_corner(first_vertex, corner).ok_or(Failure::FrameFull)?;
                }
                Ok(())
            })
        }
        _ => (2..count).try_for_each(|i| {
            let indices = match primitive_type {
                primitive::TRIANGLES if i % 3 == 2 => [i - 2, i - 1, i],
                primitive::TRIANGLES => return Ok(()),
                primitive::TRIANGLE_STRIP => crate::frame::strip_triangle(i - 2),
                _ => [0, i - 1, i], // TRIANGLE_FAN
            };
            match transform.as_ref() {
                None => {
                    let [Some(a), Some(b), Some(c)] = indices.map(|index| corners.get(index)) else {
                        return Err(Failure::Unreadable);
                    };
                    for vertex in [a, b, c] {
                        geometry.add_corner(first_vertex, to_frame(&vertex, color_of(&vertex), 1.0, 1.0)).ok_or(Failure::FrameFull)?;
                    }
                }
                Some(transform) => {
                    // 3D. Strips and fans share corners between triangles:
                    // each vertex is read, transformed and stored once, and
                    // the triangles refer to it by index.
                    let mut screen = [Corner::Unreadable; 3];
                    for (slot, index) in indices.into_iter().enumerate() {
                        screen[slot] = screen_corners.get(index, |index| {
                            // The fast path, straight from the vertex's fields.
                            let (raw, base) = match &lit {
                                None => match corners.get_3d(index) {
                                    Some(raw) => (raw, prepared.base_color(raw.color, material_color)),
                                    None => return Corner::Unreadable,
                                },
                                Some(lighting) => match corners.get_3d_lit(index) {
                                    Some((raw, normal)) => (raw, lighting.color(&raw, normal)),
                                    None => return Corner::Unreadable,
                                },
                            };
                            let color = frame_color(base);
                            match transform.frame_vertex(&raw, color) {
                                Some(mut vertex) => {
                                    if let Some(texgen) = &prepared.texgen {
                                        let normal = if texgen.needs_normal() { corners.get_3d_lit(index).and_then(|(_, normal)| normal) } else { None };
                                        (vertex.u, vertex.v) = texgen.uv(&raw, normal);
                                    }
                                    match geometry.add_vertex(first_vertex, vertex) {
                                        Some(number) => Corner::OnScreen(number),
                                        None => Corner::FrameFull,
                                    }
                                }
                                None => Corner::BehindNearPlane,
                            }
                        });
                    }
                    if let [Corner::OnScreen(a), Corner::OnScreen(b), Corner::OnScreen(c)] = screen {
                        // All in front of the near plane (the usual case).
                        for number in [a, b, c] {
                            geometry.add_index(number).ok_or(Failure::FrameFull)?;
                        }
                    } else if screen.contains(&Corner::Unreadable) {
                        return Err(Failure::Unreadable);
                    } else if screen.contains(&Corner::FrameFull) {
                        return Err(Failure::FrameFull);
                    } else {
                        cut_triangle(geometry, &mut corners, transform, lit.as_ref(), indices, &color_of, &to_frame, first_vertex)?;
                    }
                }
            }
            Ok(())
        }),
    };
    crate::phase::recorder::leave(general_phase);
    if let Some(fast) = texgen_check_hash {
        texgen_check(fast, recorded_hash(geometry, &draw, &recorded), geometry.vertex_count() - draw.first_vertex);
    }
    finish_on_cpu(geometry, stats, call, &draw, recorded);
}

/// What a draw recorded (before `finish_on_cpu`): its vertices' bits, its
/// indices and the outcome, hashed (FNV-1a), for `texgen_check`.
fn recorded_hash(geometry: &Geometry, draw: &Draw, recorded: &Result<(), Failure>) -> u32 {
    let outcome = match recorded {
        Ok(()) => 0,
        Err(Failure::Unreadable) => 1,
        Err(Failure::FrameFull) => 2,
    };
    let vertices = geometry.vertices_from(draw.first_vertex).iter().flat_map(|v| [v.x, v.y, v.z, v.w, v.u, v.v, f32::from_bits(v.color), v.fog].map(f32::to_bits));
    let indices = geometry.indices_from(draw.first_index).iter().map(|&index| index as u32);
    core::iter::once(outcome).chain(vertices).chain(indices).fold(0x811C_9DC5u32, |hash, word| (hash ^ word).wrapping_mul(0x0100_0193))
}

/// TEXGEN_RUNS 2: texgen draws recorded by `record_texgen_run` and by the
/// general path, and how many of them came out different (expected 0;
/// debug `texgencheck` reads and clears them): [checked draws, differing
/// draws, vertices of the checked draws].
pub static TEXGEN_CHECKS: [core::sync::atomic::AtomicU32; 3] = [const { core::sync::atomic::AtomicU32::new(0) }; 3];

fn texgen_check(fast: u32, general: u32, vertices: u32) {
    use core::sync::atomic::Ordering::Relaxed;
    TEXGEN_CHECKS[0].fetch_add(1, Relaxed);
    TEXGEN_CHECKS[1].fetch_add((fast != general) as u32, Relaxed);
    TEXGEN_CHECKS[2].fetch_add(vertices, Relaxed);
}

/// The end of `record_on_cpu`: the draw finished, or dropped for its failure.
fn finish_on_cpu(geometry: &mut Geometry, stats: &mut FrameStats, call: PrimCall, draw: &Draw, recorded: Result<(), Failure>) {
    match recorded {
        Ok(()) => geometry.finish_draw(*draw),
        Err(failure) => {
            geometry.undo_draw(draw);
            match failure {
                Failure::Unreadable => {
                    stats.unreadable_draws += 1;
                    stats.last_unreadable_address = call.vertex_address;
                }
                Failure::FrameFull => stats.frame_overflows += 1,
            }
        }
    }
}

/// `record_on_cpu`'s triangles with a corner behind the near plane: the
/// part in front of it, as a fan of new vertices (each used once). Rare:
/// the vertices are read and transformed again. Lit draws take their
/// colours from `lit`, others from `color_of`.
#[allow(clippy::too_many_arguments)]
fn cut_triangle<M: PspMemory>(
    geometry: &mut Geometry,
    corners: &mut VertexReader<M>,
    transform: &Transform,
    lit: Option<&VertexLighting>,
    indices: [u32; 3],
    color_of: &impl Fn(&Vertex) -> u32,
    to_frame: &impl Fn(&Vertex, u32, f32, f32) -> FrameVertex,
    first_vertex: u32,
) -> Result<(), Failure> {
    let [Some(a), Some(b), Some(c)] = indices.map(|index| corners.get(index)) else {
        return Err(Failure::Unreadable);
    };
    let mut colors = [a, b, c].map(|vertex| color_of(&vertex));
    if let Some(lighting) = lit {
        for (color, index) in colors.iter_mut().zip(indices) {
            *color = corners.lit_color(index, lighting).ok_or(Failure::Unreadable)?;
        }
    }
    let clipped = [(a, colors[0]), (b, colors[1]), (c, colors[2])].map(|(vertex, color)| transform.to_clip(&vertex, color));
    let mut polygon = [clipped[0]; 4];
    let corner_count = clip_triangle(&clipped, &mut polygon);
    for last in 2..corner_count {
        for corner in [polygon[0], polygon[last - 1], polygon[last]] {
            let vertex = to_frame(&transform.to_screen(&corner), corner.color, corner.fog, corner.w);
            geometry.add_corner(first_vertex, vertex).ok_or(Failure::FrameFull)?;
        }
    }
    Ok(())
}

/// Texgen draws four vertices at a time (`record_texgen_run`): 1
/// (default), or each vertex on its own as before (0, for A/Bs), or both
/// and compared (2, `TEXGEN_CHECKS`); debug `texgenruns`, replay
/// NATIVERES_TEXGEN_RUNS. Same frames either way.
pub static TEXGEN_RUNS: core::sync::atomic::AtomicU8 = core::sync::atomic::AtomicU8::new(1);

/// Vertices per chunk of `record_texgen_run`: a multiple of 3 (a list's
/// triangles never straddle two chunks) and of 4 (`TexgenKernel::transform4`).
const TEXGEN_CHUNK: u32 = 60;

/// Words a chunk can span (as RUN_WORDS).
const TEXGEN_WORDS: usize = TEXGEN_CHUNK as usize * 17 + 1;

/// A vertex of `record_texgen_run` as the general path's `CornerCache`
/// knows it.
#[derive(Clone, Copy, PartialEq)]
enum Numbered {
    /// Not used by a triangle yet.
    Unset,
    /// Stored as this vertex number of the draw.
    On(u16),
    /// Behind the near plane (not stored; its triangles are cut).
    Behind,
}

/// `record_texgen_run`'s draw and what it carries from chunk to chunk. Its
/// two steps per chunk (`pass`, `emit`) don't depend on the memory type,
/// so they are in the plugin once, not once per memory type the recorder
/// is built for.
struct TexgenRun<'a> {
    kernel: crate::texgen_run::TexgenKernel,
    prepared: &'a Prepared,
    material_color: u32,
    /// Lit draws: the lighting (`Transform::vertex_lighting`).
    lit: Option<&'a VertexLighting<'a>>,
    texgen: Texgen,
    format: VertexFormat,
    layout: &'a VertexLayout,
    primitive_type: u32,
    first_vertex: u32,
    /// Unskinned common layouts at even addresses: fields read directly
    /// (`VertexFormat::read_common_unskinned`; NFS UR's menu: 0x112).
    direct: Option<CommonLayout>,
    /// Vertex 0 (fans) and the two vertices before the chunk.
    first: Numbered,
    previous: [Numbered; 2],
    /// `pass` put the chunk's vertices into the frame's spare room (else
    /// into the scratch vertices).
    in_place: bool,
}

/// Room for a chunk's vertices where the frame has none or they are
/// numbered triangle by triangle (`TexgenRun::pass` and `emit`): filled
/// in only when first needed (no 2 KiB to clear for every draw).
type TexgenScratch = Option<[FrameVertex; TEXGEN_CHUNK as usize]>;

impl<'a> TexgenRun<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        kernel: crate::texgen_run::TexgenKernel,
        prepared: &'a Prepared,
        material_color: u32,
        lit: Option<&'a VertexLighting<'a>>,
        texgen: Texgen,
        format: VertexFormat,
        layout: &'a VertexLayout,
        call: PrimCall,
        first_vertex: u32,
    ) -> Self {
        let stride = layout.size;
        let direct = format.common_layout(layout).filter(|_| !format.has_weights() && call.vertex_address % 2 == 0 && stride % 2 == 0);
        TexgenRun {
            kernel,
            prepared,
            material_color,
            lit,
            texgen,
            format,
            layout,
            primitive_type: call.primitive_type,
            first_vertex,
            direct,
            first: Numbered::Unset,
            previous: [Numbered::Unset; 2],
            in_place: false,
        }
    }

    /// Pass 1 of a chunk: its `n` vertices from `words` (the first at byte
    /// `skew`), into the frame if it has room (`spare_vertices`), else into
    /// `scratch`. The vertices behind the near plane (bit k: vertex k);
    /// `None` if one can't be decoded.
    #[inline(never)]
    fn pass(&mut self, geometry: &mut Geometry, scratch: &mut TexgenScratch, words: &[u32], skew: u32, n: u32) -> Option<u64> {
        let spare = geometry.spare_vertices(self.first_vertex, n);
        self.in_place = spare.is_some();
        let out = match spare {
            Some(spare) => spare,
            None => &mut scratch.get_or_insert([FrameVertex::default(); TEXGEN_CHUNK as usize])[..n as usize],
        };
        let (format, layout, stride) = (self.format, self.layout, self.layout.size);
        let mut finish = TexgenFinish {
            prepared: self.prepared,
            material_color: self.material_color,
            lit: self.lit,
            texgen: self.texgen,
            last_color: None,
            last_finished: self.prepared.frame_color(self.prepared.base_color(None, self.material_color)),
        };
        // The normal where lighting or the texgen needs it, as the general
        // path reads it (`VertexReader::get_3d_lit`).
        let shaded = self.lit.is_some() || self.texgen.needs_normal();
        let normals = shaded && layout.normal_offset.is_some();
        match &self.direct {
            // Unlit, no normals (NFS UR's menu: ~85 % of the texgen
            // vertices): per vertex only its colour on the ARM side.
            Some(common) if !shaded => texgen_pass_direct(&self.kernel, words, skew, n, format, layout, common, &mut finish, out),
            _ => texgen_pass(&self.kernel, n, out, |k, count| {
                // Any format: each vertex decoded on its own.
                let mut lanes = [[0.0f32; 4]; 5];
                let (mut colors, mut sources) = ([0u32; 4], [[0.0f32; 4]; 3]);
                for lane in 0..4 {
                    let bytes = VertexBytes::new(words, skew + (k + lane.min(count - 1)) * stride);
                    let raw = format.decode_3d(&bytes, layout)?;
                    let normal = if normals { Some(format.decode_normal(&bytes, layout)?) } else { None };
                    let (color, source) = finish.finish(raw.color, || raw, normal);
                    let lane = lane as usize;
                    for (field, value) in lanes.iter_mut().zip([raw.position[0], raw.position[1], raw.position[2], raw.texcoord[0], raw.texcoord[1]]) {
                        field[lane] = value;
                    }
                    colors[lane] = color;
                    for (row, value) in sources.iter_mut().zip(source) {
                        row[lane] = value;
                    }
                }
                let [x, y, z, u, v] = lanes;
                Some((crate::texgen_run::Fields4::from_lanes(x, y, z, u, v), colors, sources))
            }),
        }
    }

    /// The triangles of the chunk of `n` vertices from vertex `chunk` on
    /// (those whose last vertex is in it), after its `pass` (`behind`), in
    /// the general path's order; `cut` cuts a triangle at the near plane
    /// (`cut_triangle`).
    #[inline(never)]
    fn emit(
        &mut self,
        geometry: &mut Geometry,
        scratch: &mut TexgenScratch,
        chunk: u32,
        n: u32,
        behind: u64,
        cut: &mut dyn FnMut(&mut Geometry, [u32; 3]) -> Result<(), Failure>,
    ) -> Result<(), Failure> {
        let (primitive_type, first_vertex) = (self.primitive_type, self.first_vertex);
        let (first, previous) = (self.first, self.previous);
        let triangles = match primitive_type {
            primitive::TRIANGLES => chunk / 3..(chunk + n) / 3,
            _ => chunk.saturating_sub(2)..chunk + n - 2,
        };
        let corners_of = |t: u32| match primitive_type {
            primitive::TRIANGLES => [3 * t, 3 * t + 1, 3 * t + 2],
            primitive::TRIANGLE_STRIP => crate::frame::strip_triangle(t),
            _ => [0, t + 1, t + 2], // TRIANGLE_FAN
        };
        // The vertices before the chunk its triangles use, all on the screen.
        let before_ok = match primitive_type {
            _ if chunk == 0 => true,
            primitive::TRIANGLES => true,
            primitive::TRIANGLE_STRIP => matches!(previous, [Numbered::On(_), Numbered::On(_)]),
            _ => matches!((first, previous[1]), (Numbered::On(_), Numbered::On(_))),
        };
        let earlier = |v: u32| match v {
            0 => first,
            _ if v + 1 == chunk => previous[1],
            _ => previous[0], // v + 2 == chunk
        };
        let mut numbered = [Numbered::Unset; TEXGEN_CHUNK as usize];
        if behind == 0 && self.in_place && before_ok {
            // All on the screen: the vertices as they are, numbered in order.
            let base = geometry.vertex_count() - first_vertex;
            geometry.commit_vertices(n);
            for (k, slot) in numbered[..n as usize].iter_mut().enumerate() {
                *slot = Numbered::On((base + k as u32) as u16);
            }
            let number = |v: u32| match if v >= chunk { numbered[(v - chunk) as usize] } else { earlier(v) } {
                Numbered::On(number) => number,
                _ => unreachable!("all on the screen"),
            };
            let indices = geometry.reserve_indices(3 * triangles.len()).ok_or(Failure::FrameFull)?;
            for (three, t) in indices.chunks_exact_mut(3).zip(triangles) {
                for (index, v) in three.iter_mut().zip(corners_of(t)) {
                    *index = number(v);
                }
            }
        } else {
            // Triangle by triangle, as the general path.
            crate::phase::recorder::enter(crate::phase::recorder::TEXGEN_SLOW);
            let scratch = scratch.get_or_insert([FrameVertex::default(); TEXGEN_CHUNK as usize]);
            if self.in_place {
                let spare = geometry.spare_vertices(first_vertex, n).expect("room seen by pass");
                scratch[..n as usize].copy_from_slice(spare);
            }
            for t in triangles {
                let corners = corners_of(t);
                let mut state = [Numbered::Unset; 3];
                for (slot, v) in state.iter_mut().zip(corners) {
                    if v >= chunk {
                        let k = (v - chunk) as usize;
                        if numbered[k] == Numbered::Unset {
                            numbered[k] = if behind & (1 << k) != 0 {
                                Numbered::Behind
                            } else {
                                Numbered::On(geometry.add_vertex(first_vertex, scratch[k]).ok_or(Failure::FrameFull)?)
                            };
                        }
                        *slot = numbered[k];
                    } else {
                        *slot = earlier(v);
                    }
                }
                if let [Numbered::On(a), Numbered::On(b), Numbered::On(c)] = state {
                    for number in [a, b, c] {
                        geometry.add_index(number).ok_or(Failure::FrameFull)?;
                    }
                } else {
                    cut(geometry, corners)?;
                }
            }
        }
        // What the next chunk's triangles need of this one.
        let state_of = |v: u32| if v >= chunk { numbered[(v - chunk) as usize] } else { earlier(v) };
        let last = chunk + n - 1;
        self.previous = [state_of(last - 1), state_of(last)];
        if chunk == 0 {
            self.first = numbered[0];
        }
        Ok(())
    }
}

/// `record_on_cpu` for an unskinned texgen draw, lit or not, any texgen
/// source (`Transform::texgen_kernel`, `TexgenFinish`), a non-indexed
/// list, strip or fan: per chunk of TEXGEN_CHUNK vertices, all of them
/// worked out four at a time (`TexgenRun::pass`, `TexgenKernel::transform4`,
/// straight into the frame where it has room), then the chunk's triangles
/// in the general path's order (`TexgenRun::emit`). The general path
/// numbers each vertex when a triangle first uses it, which for these
/// primitives is the vertices' order; so a chunk with nothing behind the
/// near plane is its vertices as they are, and the others are numbered
/// triangle by triangle as there, with its cut code (`cut_triangle`, with
/// `color_of` and `to_frame`). The frame comes out the same as the general
/// path's.
///
/// `None` (the draw to be undone and done by the general path) for a
/// vertex that can't be decoded or a chunk whose words can't be read.
fn record_texgen_run<M: PspMemory>(
    geometry: &mut Geometry,
    memory: &mut M,
    mut run: TexgenRun,
    transform: &Transform,
    color_of: &impl Fn(&Vertex) -> u32,
    to_frame: &impl Fn(&Vertex, u32, f32, f32) -> FrameVertex,
    call: PrimCall,
) -> Option<Result<(), Failure>> {
    let PrimCall { primitive_type, count, vertex_address, index_address, .. } = call;
    // The vertices the triangles use (a list's incomplete last triangle:
    // none of its vertices).
    let used = match primitive_type {
        primitive::TRIANGLES => count / 3 * 3,
        _ if count >= 3 => count,
        _ => 0,
    };
    let (format, layout, first_vertex, lit) = (run.format, run.layout, run.first_vertex, run.lit);
    let stride = layout.size;
    let bytes_used = format.bytes_used(layout);
    // Filled in only when first needed (as `TexgenScratch`).
    let mut copy: Option<[u32; TEXGEN_WORDS]> = None;
    let mut scratch: TexgenScratch = None;
    let mut chunk = 0;
    while chunk < used {
        let n = (used - chunk).min(TEXGEN_CHUNK);
        let address = vertex_address.wrapping_add(chunk * stride);
        let skew = address & 3;
        let word_count = (skew + stride * (n - 1) + bytes_used).div_ceil(4) as usize;
        let started = crate::cycles::start(crate::cycles::TEXGEN_PASS);
        let behind = match memory.borrow_words(address & !3, word_count) {
            Some(words) => run.pass(geometry, &mut scratch, words, skew, n)?,
            None => {
                let words = copy.get_or_insert([0; TEXGEN_WORDS]).get_mut(..word_count)?;
                memory.read_words(address & !3, words)?;
                run.pass(geometry, &mut scratch, words, skew, n)?
            }
        };
        crate::cycles::add(crate::cycles::TEXGEN_PASS, started);
        let mut cut = |geometry: &mut Geometry, corners: [u32; 3]| {
            let mut reader = VertexReader::new(format, layout, memory, vertex_address, index_address);
            cut_triangle(geometry, &mut reader, transform, lit, corners, color_of, to_frame, first_vertex)
        };
        let outer = crate::phase::recorder::enter(crate::phase::recorder::TEXGEN_EMIT);
        let emitted = run.emit(geometry, &mut scratch, chunk, n, behind, &mut cut);
        crate::phase::recorder::leave(outer);
        if let Err(failure) = emitted {
            return Some(Err(failure));
        }
        chunk += n;
    }
    Some(Ok(()))
}

/// Pass 1 of `record_texgen_run` for one chunk: `n` vertices into `out`,
/// four at a time; `read4(k, count)` gives vertices k.. k + count - 1 (the
/// lanes after them: the last again) as lanes: raw position and texture
/// coordinates, finished colours, texgen normal sources (`None`:
/// unreadable, then so is the chunk). The vertices behind the near plane
/// (bit k: vertex k), from the kernel's flags read once at the end.
#[inline(always)]
fn texgen_pass(
    kernel: &crate::texgen_run::TexgenKernel,
    n: u32,
    out: &mut [FrameVertex],
    mut read4: impl FnMut(u32, u32) -> Option<(crate::texgen_run::Fields4, [u32; 4], [[f32; 4]; 3])>,
) -> Option<u64> {
    let mut front = [0u16; TEXGEN_CHUNK as usize + 4];
    let mut k = 0;
    let outer = crate::phase::recorder::enter(crate::phase::recorder::TEXGEN_READ);
    while k < n {
        let count = (n - k).min(4);
        crate::phase::recorder::enter(crate::phase::recorder::TEXGEN_READ);
        let Some((fields, colors, sources)) = read4(k, count) else {
            crate::phase::recorder::leave(outer);
            return None;
        };
        let stored = &mut out[k as usize..(k + count) as usize];
        crate::phase::recorder::enter(crate::phase::recorder::TEXGEN_KERNEL);
        let flags: &mut [u16; 4] = (&mut front[k as usize..k as usize + 4]).try_into().expect("four flags");
        kernel.transform4(&fields, colors, &sources, stored, flags);
        k += 4;
    }
    crate::phase::recorder::leave(outer);
    let behind = front[..n as usize].iter().enumerate().fold(0u64, |mask, (k, &flag)| mask | ((flag == 0) as u64) << k);
    Some(behind)
}

/// How `TexgenRun::pass` finishes a decoded vertex: its colour (lit by
/// `lit`, else `Prepared::base_color`, worked out again only when the
/// vertex colour changes; then `frame_color`) and, for texgen from the
/// normal, the source vector (`Texgen::normal_source`), from the vertex
/// colour, the raw fields (asked for only to light) and the normal
/// (`None`: the format has none).
struct TexgenFinish<'a> {
    prepared: &'a Prepared,
    material_color: u32,
    lit: Option<&'a VertexLighting<'a>>,
    texgen: Texgen,
    last_color: Option<u32>,
    last_finished: u32,
}

impl TexgenFinish<'_> {
    /// `finish`'s colour of an unlit vertex.
    #[inline(always)]
    fn finish_unlit(&mut self, vertex_color: Option<u32>) -> u32 {
        if vertex_color != self.last_color {
            self.last_color = vertex_color;
            self.last_finished = self.finished(vertex_color);
        }
        self.last_finished
    }

    #[inline(never)]
    fn finished(&self, vertex_color: Option<u32>) -> u32 {
        self.prepared.frame_color(self.prepared.base_color(vertex_color, self.material_color))
    }

    #[inline(always)]
    fn finish(&mut self, vertex_color: Option<u32>, raw: impl FnOnce() -> Raw3d, normal: Option<[f32; 3]>) -> (u32, [f32; 3]) {
        let color = match self.lit {
            Some(lighting) => {
                let outer = crate::phase::recorder::enter(crate::phase::recorder::TEXGEN_LIGHT);
                let color = self.prepared.frame_color(lighting.color(&raw(), normal));
                crate::phase::recorder::leave(outer);
                color
            }
            None => {
                if vertex_color != self.last_color {
                    self.last_color = vertex_color;
                    self.last_finished = self.prepared.frame_color(self.prepared.base_color(vertex_color, self.material_color));
                }
                self.last_finished
            }
        };
        let source = if self.texgen.needs_normal() { self.texgen.normal_source(normal) } else { [0.0; 3] };
        (color, source)
    }
}

/// `texgen_pass` for an unlit draw without normals of an unskinned
/// `CommonLayout` at an even address: positions and texture coordinates
/// straight into lanes (`Fields4::gather`), the colour with one load per
/// vertex (`VertexFormat::read_common_color`, per colour format) and
/// finished only when it changes (`TexgenFinish::finish_unlit`).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn texgen_pass_direct(
    kernel: &crate::texgen_run::TexgenKernel,
    words: &[u32],
    skew: u32,
    n: u32,
    format: VertexFormat,
    layout: &VertexLayout,
    common: &CommonLayout,
    finish: &mut TexgenFinish,
    out: &mut [FrameVertex],
) -> Option<u64> {
    let start = words.as_ptr() as *const u8;
    let stride = layout.size;
    let (position_offset, texcoord_offset) = (common.position_offset as usize, common.texcoord_offset as usize);
    // Safe: `words` covers every field of the chunk's `n` vertices (its
    // span), the first starts at an even address and the stride is even.
    macro_rules! pass {
        ($color:literal) => {
            texgen_pass(kernel, n, out, |k, count| {
                let first = unsafe { start.add((skew + stride * k) as usize) };
                let fields = unsafe { crate::texgen_run::Fields4::gather(first, stride as usize, count as usize, position_offset, texcoord_offset) };
                let colors = [0u32, 1, 2, 3].map(|lane| {
                    let vertex = unsafe { first.add(lane.min(count - 1) as usize * stride as usize) };
                    finish.finish_unlit(unsafe { VertexFormat::read_common_color::<$color>(vertex, common) })
                });
                Some((fields, colors, [[0.0; 4]; 3]))
            })
        };
    }
    match format.color_format_bits() {
        4 => pass!(4),
        5 => pass!(5),
        6 => pass!(6),
        _ => pass!(0),
    }
}

/// Most vertices `record_3d_run` takes; longer draws take the general path.
const RUN_VERTICES: usize = 64;

/// Words a run can span: the biggest vertex (8 float weights, float texture
/// coordinates, colour, float normal and position: 68 bytes) times
/// RUN_VERTICES, plus one word for an unaligned start.
const RUN_WORDS: usize = RUN_VERTICES * 17 + 1;

/// What `record_3d_run` did.
enum Run {
    /// The vertices and triangles are in the geometry (none if there is no
    /// whole triangle); the draw is still to be finished, with this
    /// `Draw::transform` (0: screen space), and with `cached` the first
    /// vertex of the vertex cache's window it is in, if the vertices are
    /// there (`Geometry::finish_cached_draw`).
    Recorded { transform: u32, cached: Option<u32> },
    /// No room in the geometry; the draw's vertices must be dropped.
    FrameFull,
    /// Nothing done: a case for the general path.
    NotHandled,
}

/// Whole triangles of a list, strip or fan of `count` vertices.
pub(crate) fn triangle_count(primitive_type: u32, count: u32) -> u32 {
    match primitive_type {
        primitive::TRIANGLES => count / 3,
        _ => count.saturating_sub(2),
    }
}

/// Records 3D lists, strips and fans without an index buffer whose
/// vertices follow each other in memory (one PRIM, or a batch of PRIMs with
/// the same state, see `record_batch`), the way PPSSPP decodes a draw's
/// vertex range in one go (DrawEngineCommon: SubmitPrim queues the range,
/// DecodeVerts decodes it in a loop): the vertices are read in one piece,
/// decoded and transformed in a tight loop, stored once each, and the
/// triangles added as indices. Leaves the geometry unchanged and answers
/// `NotHandled` for what the general path does differently: a vertex
/// behind the near plane (its triangles are cut), unreadable data, more
/// than RUN_VERTICES vertices.
///
/// Gives the triangles the general path gives, in the same order. (In a
/// batch the vertices of a PRIM's incomplete last triangle are stored too,
/// unused: they lie between the PRIMs' vertices.)
/// Whether the last `record_3d_run_now` found its run by the mini hash
/// alone (for measuring, `cycles::RUNS_QUICK`).
static LAST_RUN_QUICK: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// The frame colour of a run's vertex from its PSP colour (`None`: the
/// material colour). One named type for every caller of `record_3d_run`,
/// instead of a closure per caller: closures defined in functions generic
/// over the PSP memory are a type of their own per memory type, and each
/// such type compiled the run decoders (`decode_run`, ~35 KiB) once more.
#[derive(Clone, Copy)]
struct RunColor<'a> {
    prepared: &'a Prepared,
    material_color: u32,
}

impl RunColor<'_> {
    #[inline(always)]
    fn of(&self, color: Option<u32>) -> u32 {
        self.prepared.frame_color(self.prepared.base_color(color, self.material_color))
    }
}

/// `record_3d_run_now`, measured by outcome (`cycles::RUNS_*`).
#[allow(clippy::too_many_arguments)]
fn record_3d_run(
    geometry: &mut Geometry,
    memory: &mut impl PspMemory,
    format: VertexFormat,
    layout: &VertexLayout,
    transform: &Transform,
    calls: &[PrimCall],
    first_vertex: u32,
    color: RunColor,
    gpu: GpuRun,
) -> Run {
    let started = crate::cycles::start(crate::cycles::RUNS_CACHED);
    let run = record_3d_run_now(geometry, memory, format, layout, transform, calls, first_vertex, color, gpu);
    let quick = LAST_RUN_QUICK.load(core::sync::atomic::Ordering::Relaxed);
    let part = match run {
        Run::Recorded { cached: Some(_), .. } if quick => crate::cycles::RUNS_QUICK,
        Run::Recorded { cached: Some(_), .. } => crate::cycles::RUNS_CACHED,
        Run::Recorded { .. } => crate::cycles::RUNS_DECODED,
        _ => crate::cycles::RUNS_FAILED,
    };
    crate::cycles::add(part, started);
    run
}

#[allow(clippy::too_many_arguments)]
fn record_3d_run_now(
    geometry: &mut Geometry,
    memory: &mut impl PspMemory,
    format: VertexFormat,
    layout: &VertexLayout,
    transform: &Transform,
    calls: &[PrimCall],
    first_vertex: u32,
    color: RunColor,
    gpu: GpuRun,
) -> Run {
    // All vertices up to the last PRIM's last whole triangle.
    let (last, before) = calls.split_last().expect("at least one PRIM");
    let last_used = match last.primitive_type {
        primitive::TRIANGLES => last.count / 3 * 3,
        _ if last.count >= 3 => last.count,
        _ => 0,
    };
    let used = (before.iter().map(|call| call.count).sum::<u32>() + last_used) as usize;
    if used == 0 {
        return Run::Recorded { transform: 0, cached: None };
    }
    if used > RUN_VERTICES {
        return Run::NotHandled;
    }
    let stride = layout.size;
    let address = calls[0].vertex_address;
    let skew = address & 3;
    let span = skew + stride * (used as u32 - 1) + format.bytes_used(layout);
    let word_count = span.div_ceil(4) as usize;
    if word_count > RUN_WORDS {
        return Run::NotHandled;
    }

    // Decoded and transformed straight into the geometry (as PPSSPP decodes
    // into its vertex buffer): vertex i of the run becomes number base + i
    // of the draw. Something the general path must do instead: the geometry
    // goes back to where the run started.
    let start = geometry.vertex_count();
    let common = format.common_layout(layout);
    // Lit draws with lights: colours from the normals, which only the
    // decoding loop reads (`store_lit_run`, with `Transform::vertex_lighting`
    // worked out there: GPU-lit runs don't need it); not cached (the colours
    // depend on the matrices and lights).
    let lit = transform.lights.is_some();
    // Lit runs the GPU lights (`Prepared::gpu_lighting`, directional
    // lights): their vertices keep the raw colour and the normal
    // (`store_gpu_lit_run`), so they go through the vertex cache like unlit
    // ones; the lights go with the draw's transform. `Some(opaque)`.
    let gpu_lit = match (transform.lights.as_ref(), gpu.lighting) {
        (Some(lights), Some(opaque)) if gpu.wanted && lights.gpu_ready() => Some(opaque),
        _ => None,
    };
    // Unskinned vertices of a common layout at even addresses (the GE's
    // rule anyway; MTX: ~70% of its vertices): fields read directly.
    // Single-weight formats likewise (MTX: most of its skinned vertices
    // hang on one bone with weight 1.0).
    let direct = common.filter(|_| (!format.has_weights() || format.weight_count() == 1) && skew % 2 == 0 && !lit);
    let decoding = RunDecode { used, skew, stride, first_vertex, format, layout, transform, direct, common, gpu_lit: gpu_lit.is_some(), color };
    let decode_as = |geometry: &mut Geometry, words: &[u32], gpu_transform: bool, rows: &RowTransform| -> Run {
        decode_run(geometry, &decoding, words, gpu_transform, rows)
    };
    // For the GPU if it can take the whole run (no vertex skinned on the
    // CPU, none behind the near plane), else the CPU's fast path as before.
    // Formats with several weights always skin on the CPU.
    // Several weights: skinned while decoding, for the GPU too (as PPSSPP);
    // not cached (the bones change from frame to frame).
    let several_bones = transform.skins_several_bones();
    let try_gpu = gpu.wanted;
    let rows = transform.rows();
    // Runs for the GPU go through the vertex cache (their decoded vertices
    // don't depend on the matrices), unless skinned.
    let mut cache = gpu.cache.filter(|_| try_gpu && !several_bones && (!lit || gpu_lit.is_some()));
    let colors = if gpu_lit.is_some() { GPU_LIT_COLORS } else { gpu.colors };
    let key = RunKey { address, vertex_count: used as u32, format: format.0, texcoords: rows.texcoord_bits(), colors };
    // The rows the GPU gets for this run: with the run's weight folded in
    // for single-weight runs whose vertices all share one weight; for
    // skinned runs those for skinned positions.
    let mut gpu_rows = if several_bones { transform.skinned_row_transform() } else { rows };
    // A run the cache holds is mostly found by its mini hash (a few sample
    // words read in one go, `vertex_cache::find_quick`): then its bytes
    // aren't read at all (on the Vita not even copied in).
    let first_word = address & !3;
    let quick_wanted = vertex_cache::QUICK_HITS.load(core::sync::atomic::Ordering::Relaxed);
    let mini = cache.as_ref().filter(|_| quick_wanted).and_then(|_| {
        let mut positions = [0usize; vertex_cache::SAMPLE_WORDS];
        let mut count = 0;
        for position in vertex_cache::sample_positions(word_count) {
            positions[count] = position;
            count += 1;
        }
        let mut samples = [0u32; vertex_cache::SAMPLE_WORDS];
        memory.read_samples(first_word, &positions[..count], &mut samples[..count])?;
        Some(vertex_cache::sample_hash(word_count, samples[..count].iter().copied()))
    });
    let mut quick_hit = None;
    if let (Some(cache), Some(mini)) = (cache.as_deref_mut(), mini) {
        if let Some((cached, weight)) = cache.find_quick(&key, mini) {
            let hit_rows = weight.map_or(gpu_rows, |weight| transform.weighted_rows(weight));
            // GPU-lit: a negative weight would turn the normals around.
            let weight_ok = gpu_lit.is_none() || weight.is_none_or(|weight| weight > 0.0);
            if weight_ok && hit_rows.in_front(&cached.bounds) {
                gpu_rows = hit_rows;
                quick_hit = Some(cached.start);
            }
        }
    }
    let caching = cache.is_some();
    // `words`: the run's words (decoded); `hashed`: what its hash covers,
    // the job's whole vertex range where it can be had (`hash_words`), as
    // the copier hashed it (`copied`, `draw_stream::CopiedRange::hash`).
    let mut place = |geometry: &mut Geometry, words: &[u32], hashed: &[u32], copied: Option<u64>| -> Result<Placed, Run> {
        let hash_started = crate::cycles::start(crate::cycles::HASH);
        let hash = cache.as_ref().map(|_| copied.unwrap_or_else(|| data_hash(hashed)));
        if hash.is_some() {
            crate::cycles::add(crate::cycles::HASH, hash_started);
        }
        let mut known = false;
        let mut cpu_only = false;
        // Looked up first: a hit needs neither the words (with the copier's
        // hash) nor the weight scan below (the same bytes have the weight
        // stored with them, `vertex_cache::Cached::weight`).
        if let (Some(cache), Some(hash)) = (cache.as_deref_mut(), hash) {
            let lookup_started = crate::cycles::start(crate::cycles::RUN_LOOKUP);
            let lookup = cache.find(&key, hash, mini);
            crate::cycles::add(crate::cycles::RUN_LOOKUP, lookup_started);
            match lookup {
                Lookup::Cached(cached) => {
                    let hit_rows = cached.weight.map_or(gpu_rows, |weight| transform.weighted_rows(weight));
                    // GPU-lit: a negative weight would turn the normals around
                    // (never stored that way; as `find_quick`'s hits).
                    let weight_ok = gpu_lit.is_none() || cached.weight.is_none_or(|weight| weight > 0.0);
                    if weight_ok && hit_rows.in_front(&cached.bounds) {
                        gpu_rows = hit_rows;
                        return Ok(Placed::Cache { start: cached.start });
                    }
                    known = true; // cached, but partly behind the near plane now
                }
                Lookup::CpuOnly => cpu_only = true,
                Lookup::Changing => known = true, // for the GPU, but not stored
                Lookup::Unknown => {}
            }
        }
        let mut weight = None;
        // Lit runs (no direct path) too: MTX's riders hang on one bone
        // with weight 248/128; unfolded, their GPU attempt failed at every
        // run, which was then decoded and lit a second time for the CPU.
        let byte_weight = direct.is_some() || (lit && format.has_byte_weights());
        if try_gpu && byte_weight && format.weight_count() == 1 {
            weight = shared_weight(words, skew, stride, used);
            if let Some(weight) = weight {
                gpu_rows = transform.weighted_rows(weight);
            }
        }
        // GPU-lit: a weight of 0 or less would turn the normals (the normal
        // rows hold world x bone without the weight): lit on the CPU.
        let gpu_possible = try_gpu && !cpu_only && (gpu_lit.is_none() || weight.is_none_or(|weight| weight > 0.0));
        let decoded = match gpu_possible {
            true => match decode_as(geometry, words, true, &gpu_rows) {
                Run::NotHandled => {
                    geometry.truncate_vertices(start);
                    if let (Some(cache), Some(hash)) = (cache.as_deref_mut(), hash) {
                        cache.mark_cpu_only(&key, hash);
                    }
                    decode_as(geometry, words, false, &rows)
                }
                done => done,
            },
            false => decode_as(geometry, words, false, &rows),
        };
        match decoded {
            Run::Recorded { transform: 0, .. } => Ok(Placed::Frame { model_space: false }),
            Run::Recorded { .. } => {
                if let (Some(cache), Some(hash), false) = (cache.as_deref_mut(), hash, known) {
                    let vertices = geometry.vertices_from(start);
                    if let Some(cached) = cache.store(&key, hash, mini, weight, vertices, bounds_of(vertices)) {
                        geometry.truncate_vertices(start);
                        return Ok(Placed::Cache { start: cached.start });
                    }
                }
                Ok(Placed::Frame { model_space: true })
            }
            Run::NotHandled => {
                geometry.truncate_vertices(start);
                Err(Run::NotHandled)
            }
            Run::FrameFull => Err(Run::FrameFull),
        }
    };
    LAST_RUN_QUICK.store(quick_hit.is_some(), core::sync::atomic::Ordering::Relaxed);
    // The hash covers every PRIM's vertices (the job's vertex range, which
    // the copier copied and hashed), the same whoever works it out.
    let total = calls.iter().map(|call| call.count).sum::<u32>();
    let hash_words = ((skew + stride * total).div_ceil(4) as usize).max(word_count);
    let placed = match quick_hit {
        Some(start) => Ok(Placed::Cache { start }),
        None => {
            let copied = if caching { memory.copied_hash(first_word, hash_words) } else { None };
            match memory.borrow_words(first_word, hash_words) {
                Some(all) => place(geometry, &all[..word_count], all, copied),
                // Not all of the range at hand: the run's own words (hashed
                // as they are; only where the range couldn't be read whole).
                None => match memory.borrow_words(first_word, word_count) {
                    Some(words) => place(geometry, words, words, None),
                    None => {
                        // Across two memory blocks: copied first.
                        let mut copy = [0u32; RUN_WORDS];
                        if memory.read_words(first_word, &mut copy[..word_count]).is_none() {
                            return Run::NotHandled;
                        }
                        place(geometry, &copy[..word_count], &copy[..word_count], None)
                    }
                },
            }
        }
    };
    let (base, model_space, cached) = match placed {
        Ok(Placed::Frame { model_space }) => ((start - first_vertex) as u16, model_space, None),
        // Indices count from the start of the run's window of the cache.
        Ok(Placed::Cache { start }) => ((start - vertex_cache::window_of(start)) as u16, true, Some(vertex_cache::window_of(start))),
        Err(run) => return run,
    };
    // The vertices are in model space: the draw's transform goes along,
    // GPU-lit ones with their lights (`GpuTransform::light_set`).
    let entry_started = crate::cycles::start(crate::cycles::RUN_ENTRY);
    let mut entry = gpu_rows.gpu_transform();
    if let (true, Some(opaque), Some(lights)) = (model_space, gpu_lit, transform.lights.as_ref()) {
        match geometry.add_light_set(&lights.gpu(format.has_color(), opaque)) {
            Some(number) => {
                entry.normal_rows = transform.gpu_normal_rows(lights);
                entry.light_set[0] = number as f32;
            }
            None => {
                // The table is full: the CPU lights this run after all.
                geometry.truncate_vertices(start);
                return Run::NotHandled;
            }
        }
    }
    let transform_number = if model_space {
        match geometry.add_transform(&entry) {
            Some(number) => number,
            None => {
                // The table is full: the CPU transforms this run after all.
                geometry.truncate_vertices(start);
                return Run::NotHandled;
            }
        }
    } else {
        0
    };
    crate::cycles::add(crate::cycles::RUN_ENTRY, entry_started);

    // The triangles as index runs (`draw_flags::INDEX_RUNS`): per PRIM its
    // primitive type, vertex count and first vertex; the display thread
    // writes the triangle corners (into the GPU's index buffer, which it
    // filled by copying them before). PRIMs without a whole triangle add
    // nothing, as before.
    let runs = calls.iter().filter(|call| triangle_count(call.primitive_type, call.count) > 0).count();
    let indices_started = crate::cycles::start(crate::cycles::INDICES);
    let Some(out) = geometry.reserve_indices(3 * runs) else {
        return Run::FrameFull;
    };
    let mut records = out.chunks_exact_mut(3);
    let mut offset = base;
    for call in calls {
        if triangle_count(call.primitive_type, call.count) > 0 {
            let record = records.next().expect("reserved for every run");
            record.copy_from_slice(&[call.primitive_type as u16, call.count as u16, offset]);
        }
        offset = offset.wrapping_add(call.count as u16);
    }
    crate::cycles::add(crate::cycles::INDICES, indices_started);
    Run::Recorded { transform: transform_number, cached }
}

/// The colour rules (`frame::RawColors`) for a draw's raw runs: what
/// `Prepared::base_color` and `frame_color` do, for the shader. `None` where
/// the draw can't go raw: no GPU transform, morphing, or lights the GPU
/// path doesn't take. A fixed source colour (FIX A) goes raw as
/// `RawColors::scale` (not with lights, not as raw sprites). Lit with
/// lights (`raw_vertex::lighting::LIGHTS`): what the GPU-lit path takes
/// (`Prepared::gpu_lighting`: directional lights, the plain colour
/// function), and only formats with normals. Skinned formats (any number of
/// weights) are skinned in the shader where the run's weights differ.
fn raw_color_fields(
    prepared: &Prepared,
    material_color: u32,
    lights: Option<&crate::lighting::LightingRegisters>,
    format: VertexFormat,
) -> Option<crate::frame::RawColors> {
    use crate::frame::RawColors;
    use crate::raw_vertex::{channels, lighting, replace};
    if !may_go_raw(prepared, lights.map(|lights| lights.raw_ready(format)), format) {
        return None;
    }
    if lights.is_some() {
        let opaque = prepared.gpu_lighting()?;
        // White without a vertex colour (the light set's materials are used
        // then); alpha forced only for unblended draws, as the light set does.
        return Some(RawColors {
            material: [255.0; 4],
            ambient: [0.0; 4],
            emissive: [0.0; 4],
            rules: [lighting::LIGHTS, if opaque { 1.0 } else { 0.0 }, replace::NONE, 0.0],
            scale: [0.0; 4],
        });
    }
    let textured = prepared.draw.flags & draw_flags::TEXTURED != 0;
    let replacing = match prepared.function {
        TEXTURE_FUNCTION_REPLACE if textured && prepared.uses_texture_alpha => replace::ALL,
        TEXTURE_FUNCTION_REPLACE if textured => replace::RGB,
        _ => replace::NONE,
    };
    let opaque = if prepared.blending { 0.0 } else { 1.0 };
    let (lit, material, ambient, emissive) = match &prepared.ambient_lighting {
        Some(light) => (
            if light.vertex_ambient { lighting::VERTEX_MATERIAL } else { lighting::MATERIAL },
            channels(light.material),
            channels(light.ambient),
            channels(light.emissive),
        ),
        None => (lighting::NONE, channels(material_color), [0.0; 4], [0.0; 4]),
    };
    let doubled = if prepared.double_source_alpha { 1.0 } else { 0.0 };
    let scale = prepared.source_scale.map_or([0.0; 4], |scale| {
        let [r, g, b, _] = channels(scale);
        [r, g, b, 1.0]
    });
    Some(RawColors { material, ambient, emissive, rules: [lit, opaque, replacing, doubled], scale })
}

/// Whether a draw's runs can go raw (`raw_color_fields` gives colour rules),
/// without working the rules out: the producer asks it for every batch so
/// the copier doesn't hash vertices nobody looks up
/// (`draw_stream::QueuedDraw::hash_vertices`).
///
/// `lights_gpu_ready`: `None` without lights, else whether the raw
/// programs can light with them (`LightingRegisters::raw_ready`).
fn may_go_raw(prepared: &Prepared, lights_gpu_ready: Option<bool>, format: VertexFormat) -> bool {
    // Texgen draws stay on the CPU route (texgen_run.rs).
    if format.morph_count() != 1 || prepared.texgen.is_some() {
        return false;
    }
    match lights_gpu_ready {
        Some(gpu_ready) => prepared.gpu_lighting().is_some() && gpu_ready && format.has_normals(),
        None => prepared.gpu_transform,
    }
}

/// Whether a batch queued now will be recorded raw (see `may_go_raw`): raw
/// vertices on (`raw_arena::ENABLED`) and a draw that allows them. Runs may
/// still fall back to the old path (no room, unreadable); they then hash
/// their words themselves.
fn goes_raw(prepared: &Prepared, lighting: Option<&crate::lighting::LightingRegisters>, format: VertexFormat) -> bool {
    crate::raw_arena::ENABLED.load(core::sync::atomic::Ordering::Relaxed)
        && may_go_raw(prepared, lighting.map(|registers| registers.raw_ready(format)), format)
}

/// `raw_color_fields` while raw vertices are on (`raw_arena::ENABLED`).
pub(crate) fn raw_colors_if_enabled(
    prepared: &Prepared,
    material_color: u32,
    lights: Option<&crate::lighting::LightingRegisters>,
    format: VertexFormat,
) -> Option<crate::frame::RawColors> {
    if !crate::raw_arena::ENABLED.load(core::sync::atomic::Ordering::Relaxed) {
        return None;
    }
    raw_color_fields(prepared, material_color, lights, format)
}

/// Records `calls` as one raw draw (`record_raw_run`) from `draw` (the
/// prepared draw at the geometry's current end); `bounds`: indexed PRIMs'
/// lowest and highest index as the producer found them (`Job::index_bounds`).
/// False if it can't (no raw arena, or the run can't go raw): nothing is
/// recorded then, and the caller records the PRIMs the old way.
#[allow(clippy::too_many_arguments)]
fn record_raw_draw(
    geometry: &mut Geometry,
    memory: &mut impl PspMemory,
    format: VertexFormat,
    layout: &VertexLayout,
    inputs: &TransformInputs,
    (calls, bounds): (&[PrimCall], Option<(u32, u32)>),
    colors: &crate::frame::RawColors,
    cache: Option<&mut VertexCache>,
    mut draw: Draw,
) -> bool {
    let Some(raw) = cache.and_then(|cache| cache.raw.as_mut()) else {
        return false;
    };
    match record_raw_run(geometry, memory, format, layout, inputs, (calls, bounds), colors, raw) {
        Some(placed) => {
            draw.transform = placed.transform;
            // The vertices as the GPU reads them: plain vertices (an indexed
            // run's indices were converted or its vertices gathered), so no
            // index bits (one program per vertex layout).
            draw.vertex_format = format.0 & !INDEX_FORMAT_BITS;
            draw.raw_colors = placed.colors;
            draw.bones = placed.bones;
            match (placed.window, placed.listed) {
                (RawPlace::Arena(window), false) => geometry.finish_raw_draw(draw, window),
                (RawPlace::PspRam(address), false) => geometry.finish_raw_psp_draw(draw, address, layout.size),
                (RawPlace::Arena(window), true) => geometry.finish_raw_listed_draw(draw, window, false),
                (RawPlace::PspRam(address), true) => geometry.finish_raw_listed_draw(draw, address, true),
            }
            true
        }
        None => {
            geometry.undo_draw(&draw);
            false
        }
    }
}

/// Records a 3D rectangles PRIM as raw sprites (`draw_flags::RAW_SPRITES`):
/// each sprite's vertex pair written four times into the raw arena, one
/// index run `[RECTANGLES, sprites, first corner]`. False (nothing recorded)
/// where it can't: raw vertices off or no arena, colour rules the sprite
/// shader doesn't do (lit with lights, skinning: it has neither), an odd
/// PRIM without a whole sprite, unreadable, or no room; the CPU path then
/// draws them.
#[allow(clippy::too_many_arguments)]
fn record_raw_sprites(
    geometry: &mut Geometry,
    memory: &mut impl PspMemory,
    format: VertexFormat,
    layout: &VertexLayout,
    inputs: &TransformInputs,
    prepared: &Prepared,
    material_color: u32,
    call: PrimCall,
    cache: Option<&mut VertexCache>,
    mut draw: Draw,
) -> bool {
    let sprites = call.count / 2;
    let drawable = crate::raw_arena::SPRITES.load(core::sync::atomic::Ordering::Relaxed);
    if !drawable || !sprites_suit_raw(format, sprites) {
        return false;
    }
    let Some(colors) = raw_colors_if_enabled(prepared, material_color, inputs.lighting(), format) else {
        return false;
    };
    // The sprite program has neither lights nor FIX A (nativeres_raw_sprite_v.cg).
    if colors.rules[0] == crate::raw_vertex::lighting::LIGHTS || colors.scale[3] != 0.0 {
        return false;
    }
    let Some(raw) = cache.and_then(|cache| cache.raw.as_mut()) else {
        return false;
    };
    let stride = layout.size;
    let address = call.vertex_address;
    let skew = address & 3;
    let length = stride * (2 * sprites - 1) + format.bytes_used(layout);
    refusing(raw, fallback::UNREADABLE);
    let Some(words) = memory.borrow_words(address & !3, (skew + length).div_ceil(4) as usize) else {
        return false;
    };
    refusing(raw, fallback::TABLES_FULL);
    // Safe to view as bytes: `words` covers the PRIM (borrowed for its span).
    let bytes = unsafe { core::slice::from_raw_parts(words.as_ptr() as *const u8, words.len() * 4) };
    let source = &bytes[skew as usize..];
    let pair = 2 * stride as usize;
    let Some(tables) = geometry.raw_tables() else {
        return false;
    };
    let Some(number) = crate::raw_blocks::raw_transform(tables, inputs, None) else {
        return false;
    };
    let Some(colors_number) = tables.add_raw_colors(&colors) else {
        return false;
    };
    // A block of four pairs per sprite; whole blocks into the window, so
    // the first corner of each sprite is a multiple of 4.
    refusing(raw, fallback::ARENA_FULL);
    let Some((placed, out)) = raw.reserve(4 * pair as u32 * sprites, 4 * pair as u32) else {
        return false;
    };
    let copying = crate::phase::recorder::enter(crate::phase::recorder::ARENA);
    for (sprite, block) in out.chunks_exact_mut(4 * pair).enumerate() {
        // The last vertex may end before a whole stride: what's there.
        let start = sprite * pair;
        let end = (start + pair).min(source.len());
        for corner in block.chunks_exact_mut(pair) {
            corner[..end - start].copy_from_slice(&source[start..end]);
        }
    }
    crate::phase::recorder::leave(copying);
    let first_corner = 4 * placed.first_vertex(4 * pair as u32);
    refusing(raw, fallback::TABLES_FULL);
    let Some(record) = geometry.reserve_indices(3) else {
        return false;
    };
    record.copy_from_slice(&[primitive::RECTANGLES as u16, sprites as u16, first_corner as u16]);
    draw.transform = number;
    draw.vertex_format = format.0;
    draw.raw_colors = colors_number;
    draw.bones = 0;
    draw.flags |= draw_flags::RAW_SPRITES;
    geometry.finish_raw_draw(draw, placed.window);
    true
}

/// Whether `sprites` rectangles of this format can be raw sprites: at least
/// one, few enough for 16-bit corner numbers, no weights (the sprite program
/// doesn't skin), and not indexed: indexed rectangles name their vertices by
/// index, and the raw sprites are copied from the vertices in a row.
pub(crate) fn sprites_suit_raw(format: VertexFormat, sprites: u32) -> bool {
    sprites > 0 && sprites <= 65_536 / 4 && !format.has_weights() && format.index_format() == IndexFormat::None
}

/// Records a run of PRIMs as raw PSP vertices (`draw_flags::RAW_VERTICES`):
/// their bytes copied into the raw arena as they are, the triangles as index
/// runs counted from the arena window, a transform (with `raw_uv`) and the
/// colour rules (`colors`, see `raw_color_fields`), and the bones if the
/// shader skins it. Returns what the draw needs (`RawRun`); `None` if the
/// run can't go raw
/// (unreadable, single weights that differ, no room): the caller records it
/// the old way.
#[allow(clippy::too_many_arguments)]
fn record_raw_run(
    geometry: &mut Geometry,
    memory: &mut impl PspMemory,
    format: VertexFormat,
    layout: &VertexLayout,
    inputs: &TransformInputs,
    (calls, bounds): (&[PrimCall], Option<(u32, u32)>),
    colors: &crate::frame::RawColors,
    raw: &mut crate::raw_arena::RawArena,
) -> Option<RawRun> {
    refusing(raw, fallback::REFUSED);
    if format.index_format() != IndexFormat::None {
        return record_raw_indexed_run(geometry, memory, format, layout, inputs, (calls, bounds), colors, raw);
    }
    // All vertices up to the last PRIM's last whole triangle (as `record_3d_run`).
    let (last, before) = calls.split_last()?;
    let last_used = match last.primitive_type {
        primitive::TRIANGLES => last.count / 3 * 3,
        _ if last.count >= 3 => last.count,
        _ => 0,
    };
    let used = before.iter().map(|call| call.count).sum::<u32>() + last_used;
    if used == 0 || used > 65_536 {
        return None;
    }
    let stride = layout.size;
    let address = calls[0].vertex_address;
    let skew = address & 3;
    let length = stride * (used - 1) + format.bytes_used(layout);
    // Zero copy: the GPU reads the vertices from PSP RAM; the CPU doesn't
    // look at them.
    let in_psp_ram = memory.zero_copy(address, length);
    refusing(raw, fallback::UNREADABLE);
    let words = if in_psp_ram { &[][..] } else { memory.borrow_words(address & !3, (skew + length).div_ceil(4) as usize)? };
    refusing(raw, fallback::TABLES_FULL);
    let (number, colors_number, bones) = raw_entry(geometry, format, inputs, colors)?;
    let runs = calls.iter().filter(|call| triangle_count(call.primitive_type, call.count) > 0).count();
    refusing(raw, fallback::ARENA_FULL);
    let (window, mut first) = if in_psp_ram {
        raw.stats.psp_ram += 1;
        raw.stats.psp_ram_bytes += length;
        (RawPlace::PspRam(address), 0)
    } else {
        // Safe to view as bytes: `words` covers the run (borrowed for its span).
        let bytes = unsafe { core::slice::from_raw_parts(words.as_ptr() as *const u8, words.len() * 4) };
        let copying = crate::phase::recorder::enter(crate::phase::recorder::ARENA);
        let placed = raw.put(&bytes[skew as usize..(skew + length) as usize], stride);
        crate::phase::recorder::leave(copying);
        let placed = placed?;
        (RawPlace::Arena(placed.window), placed.first_vertex(stride))
    };
    raw.stats.lit += (colors.rules[0] == crate::raw_vertex::lighting::LIGHTS) as u32;
    refusing(raw, fallback::TABLES_FULL);
    let out = geometry.reserve_indices(3 * runs)?;
    let mut records = out.chunks_exact_mut(3);
    for call in calls {
        if triangle_count(call.primitive_type, call.count) > 0 {
            let record = records.next().expect("reserved for every run");
            record.copy_from_slice(&[call.primitive_type as u16, call.count as u16, first as u16]);
        }
        first += call.count;
    }
    Some(RawRun { transform: number, colors: colors_number, bones, window, listed: false })
}

/// A raw draw's transform (`raw_blocks::raw_transform`: its object, camera
/// and, lit by the GPU, light set), raw colours and bones (skinned formats,
/// one weight too: the program skins): `(Draw::transform, Draw::raw_colors,
/// Draw::bones)`.
fn raw_entry(
    geometry: &mut Geometry,
    format: VertexFormat,
    inputs: &TransformInputs,
    colors: &crate::frame::RawColors,
) -> Option<(u32, u32, u32)> {
    let previous = crate::phase::recorder::enter(crate::phase::recorder::RAW_ENTRY);
    let entry = raw_entry_now(geometry, format, inputs, colors);
    crate::phase::recorder::leave(previous);
    entry
}

fn raw_entry_now(
    geometry: &mut Geometry,
    format: VertexFormat,
    inputs: &TransformInputs,
    colors: &crate::frame::RawColors,
) -> Option<(u32, u32, u32)> {
    raw_table_entries(geometry.raw_tables()?, format, inputs, colors)
}

/// `raw_entry`'s work on the raw tables (also the capture thread's,
/// `frame_builder`).
pub(crate) fn raw_table_entries(
    tables: &mut crate::frame::RawTables,
    format: VertexFormat,
    inputs: &TransformInputs,
    colors: &crate::frame::RawColors,
) -> Option<(u32, u32, u32)> {
    let lit = colors.rules[0] == crate::raw_vertex::lighting::LIGHTS;
    let bones = if format.has_weights() { tables.add_bones(inputs.bones())? + 1 } else { 0 };
    let lighting = lit.then(|| (format.has_color(), colors.rules[1] != 0.0));
    let number = crate::raw_blocks::raw_transform(tables, inputs, lighting)?;
    let colors_number = tables.add_raw_colors(colors)?;
    Some((number, colors_number, bones))
}

/// Indexed PRIMs as a raw run: each index's vertex copied, in index
/// order, into the raw arena, so the GPU draws them as the plain
/// triangles, strips and fans they then are (index runs as for
/// `record_raw_run`). PPSSPP decodes the index range and keeps the
/// indices; gathering keeps our raw draws on index runs. The vertices come
/// from the job's copy (lowest..highest index, `queue_draw`). Skinned
/// formats (Monster Hunter Freedom places its static models with one 16-bit
/// weight per vertex) are skinned by the program. `None` if anything is
/// unreadable or doesn't fit (arena room may be left used).
#[allow(clippy::too_many_arguments)]
fn record_raw_gathered(
    geometry: &mut Geometry,
    memory: &mut impl PspMemory,
    format: VertexFormat,
    layout: &VertexLayout,
    inputs: &TransformInputs,
    (calls, bounds): (&[PrimCall], Option<(u32, u32)>),
    colors: &crate::frame::RawColors,
    raw: &mut crate::raw_arena::RawArena,
) -> Option<RawRun> {
    // Whole triangles of each PRIM (as `record_raw_run`'s last one).
    let used_of = |call: &PrimCall| match call.primitive_type {
        primitive::TRIANGLES => call.count / 3 * 3,
        _ if call.count >= 3 => call.count,
        _ => 0,
    };
    let used: u32 = calls.iter().map(used_of).sum();
    if used == 0 || used > 65_536 {
        return None;
    }
    let index_format = format.index_format();
    let index_size = index_format.size();
    let stride = layout.size;
    // All indices (the calls' follow each other) and the vertices they
    // name must be readable before anything is placed.
    let index_start = calls[0].index_address;
    let count: u32 = calls.iter().map(|call| call.count).sum();
    let index_skew = index_start & 3;
    let index_words = (index_skew + count * index_size).div_ceil(4) as usize;
    refusing(raw, fallback::UNREADABLE);
    memory.borrow_words(index_start & !3, index_words)?;
    let (lowest, highest) = bounds_of_job(memory, index_format, index_start, count, bounds)?;
    let span_start = calls[0].vertex_address.wrapping_add(lowest * stride);
    let span_skew = span_start & 3;
    let span_length = (highest - lowest) * stride + format.bytes_used(layout);
    let span_words = (span_skew + span_length).div_ceil(4) as usize;
    memory.borrow_words(span_start & !3, span_words)?;

    refusing(raw, fallback::ARENA_FULL);
    let (placed, out) = raw.reserve(used * stride, stride)?;
    let gathering = crate::phase::recorder::enter(crate::phase::recorder::GATHER);
    let gathered = gather_vertices(memory, format, calls, used_of, index_start, index_words, span_start, span_words, lowest, stride, out);
    crate::phase::recorder::leave(gathering);
    gathered?;
    let lit = colors.rules[0] == crate::raw_vertex::lighting::LIGHTS;
    refusing(raw, fallback::TABLES_FULL);
    let (number, colors_number, bones) = raw_entry(geometry, format, inputs, colors)?;
    let runs = calls.iter().filter(|call| used_of(call) >= 3).count();
    let out_indices = geometry.reserve_indices(3 * runs)?;
    let mut first = placed.first_vertex(stride);
    for (record, call) in out_indices.chunks_exact_mut(3).zip(calls.iter().filter(|call| used_of(call) >= 3)) {
        let take = used_of(call);
        record.copy_from_slice(&[call.primitive_type as u16, take as u16, first as u16]);
        first += take;
    }
    raw.stats.lit += lit as u32;
    raw.stats.gathered += 1;
    Some(RawRun { transform: number, colors: colors_number, bones, window: RawPlace::Arena(placed.window), listed: false })
}

/// The lowest and highest index of a job's `count` indices at
/// `index_start`: the producer's (`Job::index_bounds`, the vertex range it
/// had copied), or scanned now where the job has none (a draw recorded at
/// once, a dump's state).
fn bounds_of_job(
    memory: &mut impl PspMemory,
    format: IndexFormat,
    index_start: u32,
    count: u32,
    bounds: Option<(u32, u32)>,
) -> Option<(u32, u32)> {
    if bounds.is_some() {
        return bounds;
    }
    let scanning = crate::phase::recorder::enter(crate::phase::recorder::INDICES);
    let bounds = index_bounds(memory, format, index_start, count);
    crate::phase::recorder::leave(scanning);
    bounds
}

/// A span this many times larger than the corners it serves is gathered
/// instead of copied whole (`record_raw_indexed_run`): copying it would move
/// more bytes than gathering the corners' vertices.
pub(crate) const SPARSE_SPAN: u32 = 4;

/// Indexed PRIMs as a raw draw: their indices converted
/// (`record_raw_converted`) where `raw_arena::INDEXED_DRAWS` is on and the
/// vertex span isn't much larger than what the PRIMs use; else (and where
/// the conversion can't) each corner's vertex gathered
/// (`record_raw_gathered`).
#[allow(clippy::too_many_arguments)]
fn record_raw_indexed_run(
    geometry: &mut Geometry,
    memory: &mut impl PspMemory,
    format: VertexFormat,
    layout: &VertexLayout,
    inputs: &TransformInputs,
    (calls, bounds): (&[PrimCall], Option<(u32, u32)>),
    colors: &crate::frame::RawColors,
    raw: &mut crate::raw_arena::RawArena,
) -> Option<RawRun> {
    if crate::raw_arena::INDEXED_DRAWS.load(core::sync::atomic::Ordering::Relaxed) {
        let count: u32 = calls.iter().map(|call| call.count).sum();
        let (lowest, highest) = bounds_of_job(memory, format.index_format(), calls[0].index_address, count, bounds)?;
        let corners: u32 = calls.iter().map(|call| 3 * triangle_count(call.primitive_type, call.count)).sum();
        let span = highest - lowest + 1;
        let span_start = calls[0].vertex_address.wrapping_add(lowest.wrapping_mul(layout.size));
        let span_length = (highest - lowest) * layout.size + format.bytes_used(layout);
        // Left in PSP RAM, a span costs no copy however sparse it is.
        let in_psp_ram = memory.zero_copy(span_start, span_length);
        if in_psp_ram || span <= SPARSE_SPAN * corners {
            let converting = crate::phase::recorder::enter(crate::phase::recorder::CONVERT);
            let run = record_raw_converted(geometry, memory, format, layout, inputs, calls, colors, raw, (lowest, highest), in_psp_ram);
            crate::phase::recorder::leave(converting);
            return run;
        }
    }
    record_raw_gathered(geometry, memory, format, layout, inputs, (calls, bounds), colors, raw)
}

/// Indexed PRIMs (triangles, strips, fans; `calls`) as a raw draw that keeps
/// their indices, as PPSSPP's hardware path keeps them (DrawEngineCommon:
/// the vertex range from the lowest to the highest index decoded once, the
/// indices rebased to it): the vertices from index `lowest` to `highest`
/// copied into the raw arena in one piece (or left in PSP RAM: `in_psp_ram`,
/// zero copy), and each PRIM's whole triangles written as a triangle list of
/// 16-bit vertex numbers, `first + index - lowest`, `first` being the span's
/// first vertex in its window (0 in PSP RAM). The corners come in
/// `frame::run_indices`' order (lists 3t, 3t+1, 3t+2; strips
/// `strip_triangle`; fans 0, t+1, t+2): the triangles gathering gives, in the
/// same order and winding (`draw_flags::CULL`).
///
/// The bounds are the producer's, the indices the copier's (copied a little
/// later): an index outside them means the game rewrote the list meanwhile,
/// and the draw is turned down as unreadable. `None` where it can't (nothing
/// recorded but arena room and table entries).
#[allow(clippy::too_many_arguments)]
fn record_raw_converted(
    geometry: &mut Geometry,
    memory: &mut impl PspMemory,
    format: VertexFormat,
    layout: &VertexLayout,
    inputs: &TransformInputs,
    calls: &[PrimCall],
    colors: &crate::frame::RawColors,
    raw: &mut crate::raw_arena::RawArena,
    (lowest, highest): (u32, u32),
    in_psp_ram: bool,
) -> Option<RawRun> {
    let corners: u32 = calls.iter().map(|call| 3 * triangle_count(call.primitive_type, call.count)).sum();
    if corners == 0 || highest - lowest >= 65_536 {
        return None;
    }
    let stride = layout.size;
    let index_format = format.index_format();
    let index_start = calls[0].index_address;
    let count: u32 = calls.iter().map(|call| call.count).sum();
    let index_skew = index_start & 3;
    let index_words = (index_skew + count * index_format.size()).div_ceil(4) as usize;
    let span_start = calls[0].vertex_address.wrapping_add(lowest.wrapping_mul(stride));
    let span_skew = span_start & 3;
    let span_length = (highest - lowest) * stride + format.bytes_used(layout);

    // The span: copied into the arena, or left where it is.
    let lit = colors.rules[0] == crate::raw_vertex::lighting::LIGHTS;
    let (place, first) = if in_psp_ram {
        raw.stats.psp_ram += 1;
        raw.stats.psp_ram_bytes += span_length;
        (RawPlace::PspRam(span_start), 0)
    } else {
        refusing(raw, fallback::UNREADABLE);
        let words = memory.borrow_words(span_start & !3, (span_skew + span_length).div_ceil(4) as usize)?;
        // Safe to view as bytes: `words` covers the span (borrowed for it).
        let bytes = unsafe { core::slice::from_raw_parts(words.as_ptr() as *const u8, words.len() * 4) };
        refusing(raw, fallback::ARENA_FULL);
        let copying = crate::phase::recorder::enter(crate::phase::recorder::ARENA);
        let placed = raw.put(&bytes[span_skew as usize..(span_skew + span_length) as usize], stride);
        crate::phase::recorder::leave(copying);
        let placed = placed?;
        (RawPlace::Arena(placed.window), placed.first_vertex(stride))
    };

    refusing(raw, fallback::TABLES_FULL);
    let out = geometry.reserve_indices(corners as usize)?;
    refusing(raw, fallback::UNREADABLE);
    let words = memory.borrow_words(index_start & !3, index_words)?;
    // Safe to view as bytes: `words` covers every index.
    let bytes = unsafe { core::slice::from_raw_parts(words.as_ptr() as *const u8, words.len() * 4) };
    let indices = &bytes[index_skew as usize..];
    let converted = match index_format {
        IndexFormat::U8 => convert_indices::<1>(indices, calls, (lowest, highest), first, out),
        IndexFormat::U16 => convert_indices::<2>(indices, calls, (lowest, highest), first, out),
        _ => convert_indices::<4>(indices, calls, (lowest, highest), first, out),
    };
    if !converted {
        return None;
    }
    let (number, colors_number, bones) = raw_entry(geometry, format, inputs, colors)?;
    raw.stats.lit += lit as u32;
    raw.stats.converted += 1;
    Some(RawRun { transform: number, colors: colors_number, bones, window: place, listed: true })
}

/// `record_raw_converted`'s indices: each PRIM's whole triangles from
/// `indices` (the PRIMs' indices one after the other, `SIZE` bytes each,
/// little-endian) as `first + index - lowest` into `out`, in
/// `frame::run_indices`' corner order. False if an index lies outside
/// `lowest..=highest` (`out` is then partly written).
///
/// Per PRIM, not per corner: one slice of its indices and one of its
/// corners, each index read once (strips and fans keep the last two in a
/// sliding window) and the range checked once per PRIM through the largest
/// value. The corner-by-corner version (a closure per corner, a bounds
/// check per index and per corner, the primitive matched per triangle) was
/// the capture thread's largest part in MHF's intro (~945 indexed draws,
/// 11 % of the capture thread, ~4.4 ms per frame; 2026-11-19).
pub(crate) fn convert_indices<const SIZE: usize>(
    indices: &[u8],
    calls: &[PrimCall],
    (lowest, highest): (u32, u32),
    first: u32,
    out: &mut [u16],
) -> bool {
    // An index's number counted from `lowest` (wrapping: below `lowest`
    // becomes huge and fails the range check).
    #[inline(always)]
    fn number<const SIZE: usize>(index: &[u8], lowest: u32) -> u32 {
        let value = match SIZE {
            1 => index[0] as u32,
            2 => u16::from_le_bytes([index[0], index[1]]) as u32,
            _ => u32::from_le_bytes([index[0], index[1], index[2], index[3]]),
        };
        value.wrapping_sub(lowest)
    }
    let range = highest - lowest;
    // Wrapping: an out-of-range value is written before the PRIM's check refuses it.
    let corner = |value: u32| first.wrapping_add(value) as u16;
    let mut written = 0;
    let mut start = 0; // the PRIM's first index, counted from the job's first
    for call in calls {
        let count = call.count as usize;
        let triangles = triangle_count(call.primitive_type, call.count) as usize;
        let Some(source) = indices.get(start * SIZE..(start + count) * SIZE) else {
            return false;
        };
        start += count;
        let Some(target) = out.get_mut(written..written + 3 * triangles) else {
            return false;
        };
        written += 3 * triangles;
        if triangles == 0 {
            continue;
        }
        let mut source = source.chunks_exact(SIZE).map(|index| number::<SIZE>(index, lowest));
        let mut largest = 0;
        if call.primitive_type == primitive::TRIANGLES {
            for (out, value) in target.iter_mut().zip(source) {
                largest = largest.max(value);
                *out = corner(value);
            }
        } else {
            // Strips: triangle t is (t, t + 1, t + 2), every second one's
            // first two swapped (`frame::strip_triangle`); fans: (0, t + 1,
            // t + 2).
            let strip = call.primitive_type == primitive::TRIANGLE_STRIP;
            let (Some(mut a), Some(mut b)) = (source.next(), source.next()) else {
                return false;
            };
            largest = a.max(b);
            for (t, (triangle, c)) in target.chunks_exact_mut(3).zip(source).enumerate() {
                largest = largest.max(c);
                let swap = strip && t % 2 == 1;
                triangle[0] = corner(if swap { b } else { a });
                triangle[1] = corner(if swap { a } else { b });
                triangle[2] = corner(c);
                if strip {
                    a = b;
                }
                b = c;
            }
        }
        if largest > range {
            return false;
        }
    }
    true
}

/// `convert_indices` for indices whose bounds are their own (`lowest` the
/// smallest of all of them, the largest at most `lowest + 65,535`:
/// `frame_builder`'s indexed draws take the bounds from these very
/// indices), so no index can lie outside and nothing is checked. Writes the
/// same corners: `first + index - lowest` in 16 bits is the index's low 16
/// bits plus one `delta`, so each corner is one load and one add. A loop
/// per primitive type: strips two triangles per step (`a b c`, `c b d`: the
/// odd one's first two swapped without a test), fans with their first
/// index held. The shared loop of `convert_indices` worked out the swap and
/// the largest value per triangle, ~22 instructions per triangle (MHF's
/// intro: ~7,600 five-index strips per frame, ~1.1M cycles). False only if
/// `indices` or `out` are too short for `calls`.
pub(crate) fn convert_indices_in_bounds<const SIZE: usize>(indices: &[u8], calls: &[PrimCall], lowest: u32, first: u32, out: &mut [u16]) -> bool {
    let delta = first.wrapping_sub(lowest) as u16;
    // The index's low 16 bits (all of a 1- or 2-byte one) plus `delta`.
    #[inline(always)]
    fn corner<const SIZE: usize>(index: &[u8], delta: u16) -> u16 {
        let low = if SIZE == 1 { index[0] as u16 } else { u16::from_le_bytes([index[0], index[1]]) };
        low.wrapping_add(delta)
    }
    let mut written = 0;
    let mut start = 0; // the PRIM's first index, counted from the job's first
    for call in calls {
        let count = call.count as usize;
        let triangles = triangle_count(call.primitive_type, call.count) as usize;
        let Some(source) = indices.get(start * SIZE..(start + count) * SIZE) else {
            return false;
        };
        start += count;
        let Some(target) = out.get_mut(written..written + 3 * triangles) else {
            return false;
        };
        written += 3 * triangles;
        if triangles == 0 {
            continue;
        }
        match call.primitive_type {
            primitive::TRIANGLES => {
                for (out, index) in target.iter_mut().zip(source.chunks_exact(SIZE)) {
                    *out = corner::<SIZE>(index, delta);
                }
            }
            primitive::TRIANGLE_STRIP => {
                // Triangles 2k and 2k + 1 of indices a b c d: a b c, c b d.
                let (head, rest) = source.split_at(2 * SIZE);
                let mut a = corner::<SIZE>(&head[..SIZE], delta);
                let mut b = corner::<SIZE>(&head[SIZE..], delta);
                let mut pairs = target.chunks_exact_mut(6);
                let mut next = rest.chunks_exact(2 * SIZE);
                for (six, cd) in (&mut pairs).zip(&mut next) {
                    let c = corner::<SIZE>(&cd[..SIZE], delta);
                    let d = corner::<SIZE>(&cd[SIZE..], delta);
                    six.copy_from_slice(&[a, b, c, c, b, d]);
                    a = c;
                    b = d;
                }
                // An odd count of triangles: the last one, even, a b c.
                let last = pairs.into_remainder();
                if !last.is_empty() {
                    let at = rest.len() - next.remainder().len();
                    last.copy_from_slice(&[a, b, corner::<SIZE>(&rest[at..at + SIZE], delta)]);
                }
            }
            _ => {
                // Fans: triangle t is (0, t + 1, t + 2).
                let (head, rest) = source.split_at(2 * SIZE);
                let a = corner::<SIZE>(&head[..SIZE], delta);
                let mut b = corner::<SIZE>(&head[SIZE..], delta);
                for (three, index) in target.chunks_exact_mut(3).zip(rest.chunks_exact(SIZE)) {
                    let c = corner::<SIZE>(index, delta);
                    three.copy_from_slice(&[a, b, c]);
                    b = c;
                }
            }
        }
    }
    true
}

/// `record_raw_gathered`'s copy: each index's vertex, in index order, into
/// `out` (one stride each, a short last vertex filled with zeros). The
/// number of bytes written; `None` if the indices or the span can't be read.
#[allow(clippy::too_many_arguments)]
fn gather_vertices(
    memory: &mut impl PspMemory,
    format: VertexFormat,
    calls: &[PrimCall],
    used_of: impl Fn(&PrimCall) -> u32,
    index_start: u32,
    index_words: usize,
    span_start: u32,
    span_words: usize,
    lowest: u32,
    stride: u32,
    out: &mut [u8],
) -> Option<usize> {
    let index_format = format.index_format();
    let index_size = index_format.size();
    let index_skew = index_start & 3;
    let span_skew = span_start & 3;
    let mut written = 0usize;
    let mut position = 0u32; // index number from `index_start`
    let mut indices = [0u32; 64];
    for call in calls {
        let take = used_of(call);
        let mut done = 0;
        while done < take {
            let chunk = ((take - done) as usize).min(indices.len());
            {
                let words = memory.borrow_words(index_start & !3, index_words)?;
                // Safe to view as bytes: `words` covers every index.
                let bytes = unsafe { core::slice::from_raw_parts(words.as_ptr() as *const u8, words.len() * 4) };
                for (n, index) in indices[..chunk].iter_mut().enumerate() {
                    let at = (index_skew + (position + done + n as u32) * index_size) as usize;
                    *index = match index_format {
                        IndexFormat::U8 => bytes[at] as u32,
                        IndexFormat::U16 => u16::from_le_bytes([bytes[at], bytes[at + 1]]) as u32,
                        _ => u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]),
                    };
                }
            }
            let words = memory.borrow_words(span_start & !3, span_words)?;
            // Safe to view as bytes: `words` covers the span.
            let bytes = unsafe { core::slice::from_raw_parts(words.as_ptr() as *const u8, words.len() * 4) };
            for &index in &indices[..chunk] {
                // Indices outside the span (the game rewrote them after the
                // producer found the bounds): turned down as unreadable.
                let from = (span_skew as usize).wrapping_add((index.wrapping_sub(lowest) as usize).wrapping_mul(stride as usize));
                if from >= bytes.len() {
                    return None;
                }
                // The span's last vertex may end after its last used byte.
                let length = (stride as usize).min(bytes.len() - from);
                out[written..written + length].copy_from_slice(&bytes[from..from + length]);
                out[written + length..written + stride as usize].fill(0);
                written += stride as usize;
            }
            done += chunk as u32;
        }
        position += call.count;
    }
    Some(written)
}

/// What `record_raw_run` gives the draw: its `Draw::transform`,
/// `Draw::raw_colors` and `Draw::bones`, where its indices count vertices
/// from (`Draw::first_vertex`), and whether they are a plain triangle list
/// (`listed`: converted indices) or index runs.
struct RawRun {
    transform: u32,
    colors: u32,
    bones: u32,
    window: RawPlace,
    listed: bool,
}

/// Where a raw run's vertices are.
enum RawPlace {
    /// Copied into the raw arena: the window's byte offset.
    Arena(u32),
    /// Left in PSP RAM (zero copy): the PSP address of its first vertex.
    PspRam(u32),
}

/// What `decode_run` needs of a run besides its memory and rows (worked
/// out by `record_3d_run_now`).
struct RunDecode<'a> {
    used: usize,
    skew: u32,
    stride: u32,
    first_vertex: u32,
    format: VertexFormat,
    layout: &'a VertexLayout,
    transform: &'a Transform,
    /// Fields read directly (`store_direct_run_as`).
    direct: Option<CommonLayout>,
    common: Option<CommonLayout>,
    /// Lit on the GPU (`store_gpu_lit_run`) when transformed there.
    gpu_lit: bool,
    color: RunColor<'a>,
}

/// `record_3d_run_now`'s decoding of a run's words into the geometry, for
/// the GPU (`gpu_transform`) or the CPU. Generic over nothing, and never
/// inlined: the decoders it inlines (~35 KiB) are compiled once, not once
/// per PSP memory type and caller (2026-11-25: 4 copies in the release
/// plugin, 6 in the debug one).
#[inline(never)]
fn decode_run(geometry: &mut Geometry, decoding: &RunDecode, words: &[u32], gpu_transform: bool, rows: &RowTransform) -> Run {
    let RunDecode { used, skew, stride, first_vertex, format, layout, transform, direct, common, gpu_lit, color } = *decoding;
    // The colour by a plain type, not a closure: in this one big function
    // LLVM outlined a colour closure, a call per vertex (GoW recorder +1 ms,
    // 2026-11-25); `RunColor::of` is always inlined.
    let run_color = &color;
    let run = RunInput { words, used, skew, stride, first_vertex };
    if let Some(common) = direct {
        return if gpu_transform {
            store_direct_run_as::<true>(geometry, &run, format, transform, rows, &common, run_color)
        } else {
            store_direct_run_as::<false>(geometry, &run, format, transform, rows, &common, run_color)
        };
    }
    if gpu_transform && gpu_lit {
        return store_gpu_lit_run(geometry, &run, format, layout, transform, rows);
    }
    if let Some(lighting) = &transform.vertex_lighting() {
        return if gpu_transform {
            store_lit_run::<true>(geometry, &run, format, layout, transform, rows, lighting, run_color)
        } else {
            store_lit_run::<false>(geometry, &run, format, layout, transform, rows, lighting, run_color)
        };
    }
    // One loop per decoder (each call is inlined and compiled on its own).
    if gpu_transform {
        store_decoded_run::<true>(geometry, &run, format, layout, common, transform, rows, run_color)
    } else {
        store_decoded_run::<false>(geometry, &run, format, layout, common, transform, rows, run_color)
    }
}

/// Where a run's vertices are (see `record_3d_run`).
struct RunInput<'a> {
    /// The run's memory, from the word holding its first byte.
    words: &'a [u32],
    used: usize,
    /// Where the first vertex starts in `words`, in bytes.
    skew: u32,
    stride: u32,
    first_vertex: u32,
}

/// `store_run` for unskinned (`SKINNED` false) and single-weight vertices
/// of a common layout: each field read with one load
/// (`VertexFormat::read_common_unskinned`), the transform in locals
/// (`RowTransform`, NEON on the Vita). A single-weight vertex whose weight
/// is the one folded into the rows (`RowTransform::weight`: 1.0, or the
/// run's for the GPU) uses the rows with the bone folded in, like
/// `Transform::frame_vertex`; others go through `frame_vertex`.
fn store_direct_run<const COLOR: u32, const SKINNED: bool, const GPU: bool>(
    geometry: &mut Geometry,
    run: &RunInput,
    transform: &Transform,
    rows: &RowTransform,
    common: &CommonLayout,
    run_color: &RunColor,
) -> Run {
    let start = run.words.as_ptr() as *const u8;
    for index in 0..run.used as u32 {
        // Safe: `record_3d_run` made `words` cover every field of the run's
        // `used` vertices (its span), and checked the first starts at an even
        // address; the stride of a format with 16-bit fields is even.
        let (vertex, (position, texcoord, color)) = unsafe {
            let vertex = start.add((run.skew + run.stride * index) as usize);
            (vertex, VertexFormat::read_common_unskinned::<COLOR>(vertex, common))
        };
        let color = run_color.of(color);
        let screen = if SKINNED {
            // Safe: as above.
            let weight = unsafe { VertexFormat::read_first_weight(vertex) } as f32 * (1.0 / 128.0);
            let mut weights = [0.0; MAX_WEIGHTS];
            weights[0] = weight;
            // The rows fold in the bone x their weight (1, or the run's
            // weight: `Transform::weighted_rows`).
            if weight == rows.weight {
                vertex_with_rows::<GPU>(rows, position, texcoord, color)
            } else if GPU {
                None // skinned on the CPU: the run goes the CPU path
            } else {
                transform.frame_vertex(&Raw3d { position, texcoord, color: None, weights }, color)
            }
        } else {
            vertex_with_rows::<GPU>(rows, position, texcoord, color)
        };
        let Some(screen) = screen else {
            return Run::NotHandled;
        };
        if geometry.add_vertex(run.first_vertex, screen).is_none() {
            return Run::FrameFull;
        }
    }
    recorded::<GPU>()
}

/// The weight all `used` vertices of a single-weight run share (8-bit
/// weights, the first byte of each vertex, as `store_direct_run` reads
/// them), or `None` if they differ. Reads one byte per vertex: cheap next
/// to decoding, so it's done for cache hits too. (The CPU path's runs; raw
/// draws are skinned by the program, `raw_blocks`.)
fn shared_weight(words: &[u32], skew: u32, stride: u32, used: usize) -> Option<f32> {
    // Safe to view as bytes: `words` covers the run's span (`record_3d_run`).
    let bytes = unsafe { core::slice::from_raw_parts(words.as_ptr() as *const u8, words.len() * 4) };
    let weight_at = |index: usize| bytes.get((skew + stride * index as u32) as usize).copied();
    let first = weight_at(0)?;
    (1..used).all(|index| weight_at(index) == Some(first)).then_some(first as f32 * (1.0 / 128.0))
}

/// A vertex from its raw fields with the fast path's rows: in screen space
/// (`RowTransform::apply`), or for the GPU in model space
/// (`RowTransform::model_vertex`). `None` behind the near plane.
#[inline(always)]
fn vertex_with_rows<const GPU: bool>(rows: &RowTransform, position: [f32; 3], texcoord: [f32; 2], color: u32) -> Option<FrameVertex> {
    if GPU {
        rows.model_vertex(position, texcoord, color)
    } else {
        rows.apply(position, texcoord, color)
    }
}

/// What the store loops below return for a recorded run: the vertices in
/// model space (`transform` 1, `record_3d_run` then adds the table entry)
/// or in screen space (0).
fn recorded<const GPU: bool>() -> Run {
    Run::Recorded { transform: GPU as u32, cached: None }
}

/// How `record_3d_run` may record a run for the GPU transform: `wanted`
/// (`Prepared::gpu_transform`), and with the vertex cache, if there is one
/// (`colors`: `Prepared::color_key`).
struct GpuRun<'a> {
    wanted: bool,
    cache: Option<&'a mut VertexCache>,
    colors: u32,
    /// `Prepared::gpu_lighting`.
    lighting: Option<bool>,
}

/// `vertex_cache::RunKey::colors` of GPU-lit runs: their vertices hold the
/// raw vertex colour (and the packed normal), whatever the draw state.
const GPU_LIT_COLORS: u32 = 0x4C49_5452;

/// Where `record_3d_run` put a run's vertices.
enum Placed {
    /// In the frame, in model space or not.
    Frame { model_space: bool },
    /// In the vertex cache, from this vertex on.
    Cache { start: u32 },
}

/// `store_direct_run` for the format's colour and weights.
fn store_direct_run_as<const GPU: bool>(
    geometry: &mut Geometry,
    run: &RunInput,
    format: VertexFormat,
    transform: &Transform,
    rows: &RowTransform,
    common: &CommonLayout,
    run_color: &RunColor,
) -> Run {
    match (format.has_weights(), format.color_format_bits()) {
        (false, 4) => store_direct_run::<4, false, GPU>(geometry, run, transform, rows, common, run_color),
        (false, 5) => store_direct_run::<5, false, GPU>(geometry, run, transform, rows, common, run_color),
        (false, 6) => store_direct_run::<6, false, GPU>(geometry, run, transform, rows, common, run_color),
        (false, _) => store_direct_run::<0, false, GPU>(geometry, run, transform, rows, common, run_color),
        (true, 4) => store_direct_run::<4, true, GPU>(geometry, run, transform, rows, common, run_color),
        (true, 5) => store_direct_run::<5, true, GPU>(geometry, run, transform, rows, common, run_color),
        (true, 6) => store_direct_run::<6, true, GPU>(geometry, run, transform, rows, common, run_color),
        (true, _) => store_direct_run::<0, true, GPU>(geometry, run, transform, rows, common, run_color),
    }
}

/// `store_run` with the decoder for the format (one loop per decoder).
fn store_decoded_run<const GPU: bool>(
    geometry: &mut Geometry,
    run: &RunInput,
    format: VertexFormat,
    layout: &VertexLayout,
    common: Option<CommonLayout>,
    transform: &Transform,
    rows: &RowTransform,
    run_color: &RunColor,
) -> Run {
    let color = |_: &VertexBytes, raw: &Raw3d| Some(run_color.of(raw.color));
    match (common, format.color_format_bits()) {
        (Some(common), 4) => store_run::<GPU>(geometry, run, transform, rows, color, |bytes| Some(VertexFormat::decode_common::<4>(bytes, &common))),
        (Some(common), 5) => store_run::<GPU>(geometry, run, transform, rows, color, |bytes| Some(VertexFormat::decode_common::<5>(bytes, &common))),
        (Some(common), 6) => store_run::<GPU>(geometry, run, transform, rows, color, |bytes| Some(VertexFormat::decode_common::<6>(bytes, &common))),
        (Some(common), _) => store_run::<GPU>(geometry, run, transform, rows, color, |bytes| Some(VertexFormat::decode_common::<0>(bytes, &common))),
        (None, _) => store_run::<GPU>(geometry, run, transform, rows, color, |bytes| format.decode_3d(bytes, layout)),
    }
}

/// The GPU-lit counterpart of `store_run::<true>`: vertices in model space
/// with their raw colour (white without one) and their normal packed into
/// `w` (`lighting::pack_normal`); nativeres_3d_v.cg lights them. Normals of
/// several-bone vertices are skinned here (as their positions), single-bone
/// and unskinned ones stay raw (`Transform::gpu_normal_rows`). A vertex the
/// GPU can't take (zero normal, a weight not folded into the rows, no
/// normal on a one-bone draw) leaves the run to the CPU.
fn store_gpu_lit_run(geometry: &mut Geometry, run: &RunInput, format: VertexFormat, layout: &VertexLayout, transform: &Transform, rows: &RowTransform) -> Run {
    let has_normals = layout.normal_offset.is_some();
    if !has_normals && format.has_weights() && !transform.skins_normals() {
        return Run::NotHandled; // (0, 0, 1) must not go through the bone
    }
    // MTX's lit formats (16-bit positions and texture coordinates, 8-bit
    // normals, its riders and bikes): one loop per colour format with the
    // fields at fixed offsets, as `store_decoded_run` (the general decoders
    // per vertex made lit runs ~55% dearer than unlit ones on the Vita).
    if let (Some(common), Some(normal_offset), true) = (format.common_layout(layout), layout.normal_offset, format.has_s8_normals()) {
        return match format.color_format_bits() {
            4 => store_gpu_lit_common::<4>(geometry, run, transform, rows, &common, normal_offset),
            5 => store_gpu_lit_common::<5>(geometry, run, transform, rows, &common, normal_offset),
            6 => store_gpu_lit_common::<6>(geometry, run, transform, rows, &common, normal_offset),
            _ => store_gpu_lit_common::<0>(geometry, run, transform, rows, &common, normal_offset),
        };
    }
    let decode = |bytes: &VertexBytes| format.decode_3d(bytes, layout);
    let normal = |bytes: &VertexBytes| if has_normals { format.decode_normal(bytes, layout) } else { Some([0.0, 0.0, 1.0]) };
    if transform.skins_normals() {
        store_gpu_lit_skinned(geometry, run, transform, decode, normal)
    } else {
        store_gpu_lit_unskinned(geometry, run, transform, rows, decode, |bytes| normal(bytes).map(pack_normal))
    }
}

/// `store_gpu_lit_run` for a `CommonLayout` with 8-bit normals, its colour
/// format (VERTEX_TYPE bits 2..4) fixed at compile time.
fn store_gpu_lit_common<const COLOR: u32>(
    geometry: &mut Geometry,
    run: &RunInput,
    transform: &Transform,
    rows: &RowTransform,
    common: &CommonLayout,
    normal_offset: u32,
) -> Run {
    let decode = |bytes: &VertexBytes| Some(VertexFormat::decode_common::<COLOR>(bytes, common));
    if transform.skins_normals() {
        let normal = |bytes: &VertexBytes| Some(VertexFormat::s8_normal(bytes, normal_offset).map(|byte| byte as i8 as f32 * (1.0 / 128.0)));
        store_gpu_lit_skinned(geometry, run, transform, decode, normal)
    } else {
        // Raw 8-bit normals pack from their bytes (`pack_s8_normal`).
        let packed = |bytes: &VertexBytes| Some(pack_s8_normal(VertexFormat::s8_normal(bytes, normal_offset)));
        store_gpu_lit_unskinned(geometry, run, transform, rows, decode, packed)
    }
}

/// The GPU-lit loop for vertices whose normals stay raw (unskinned or on
/// one bone: `Transform::gpu_normal_rows` turns them on the GPU). `packed`
/// gives a vertex's packed normal; a zero normal, or a weight the rows
/// don't fold in, leaves the run to the CPU.
#[inline(always)]
fn store_gpu_lit_unskinned(
    geometry: &mut Geometry,
    run: &RunInput,
    transform: &Transform,
    rows: &RowTransform,
    decode: impl Fn(&VertexBytes) -> Option<Raw3d>,
    packed: impl Fn(&VertexBytes) -> Option<u32>,
) -> Run {
    let rows = *rows;
    for index in 0..run.used as u32 {
        let bytes = VertexBytes::new(run.words, run.skew + run.stride * index);
        let (Some(raw), Some(packed)) = (decode(&bytes), packed(&bytes)) else {
            return Run::NotHandled;
        };
        if packed == 0 || !transform.uses_these_rows(&rows, &raw.weights) {
            return Run::NotHandled;
        }
        let Some(mut vertex) = rows.model_vertex(raw.position, raw.texcoord, raw.color.unwrap_or(0xFFFF_FFFF)) else {
            return Run::NotHandled;
        };
        vertex.w = f32::from_bits(packed);
        if geometry.add_vertex(run.first_vertex, vertex).is_none() {
            return Run::FrameFull;
        }
    }
    recorded::<true>()
}

/// The GPU-lit loop for several-bone vertices: position and normal skinned
/// here in one pass (`Transform::skin_position_and_normal`), the normal
/// packed by its direction (`pack_scaled_normal`).
#[inline(always)]
fn store_gpu_lit_skinned(
    geometry: &mut Geometry,
    run: &RunInput,
    transform: &Transform,
    decode: impl Fn(&VertexBytes) -> Option<Raw3d>,
    normal: impl Fn(&VertexBytes) -> Option<[f32; 3]>,
) -> Run {
    let skinned = transform.skinned_row_transform();
    for index in 0..run.used as u32 {
        let bytes = VertexBytes::new(run.words, run.skew + run.stride * index);
        let (Some(raw), Some(normal)) = (decode(&bytes), normal(&bytes)) else {
            return Run::NotHandled;
        };
        let (position, normal) = transform.skin_position_and_normal(&raw, normal);
        let packed = pack_scaled_normal(normal);
        if packed == 0 {
            return Run::NotHandled;
        }
        let Some(mut vertex) = skinned.model_vertex(position, raw.texcoord, raw.color.unwrap_or(0xFFFF_FFFF)) else {
            return Run::NotHandled;
        };
        vertex.w = f32::from_bits(packed);
        if geometry.add_vertex(run.first_vertex, vertex).is_none() {
            return Run::FrameFull;
        }
    }
    recorded::<true>()
}

/// `store_run` for lit draws with lights: each vertex's normal read too and
/// its colour lit (`Transform::lit_color`). MTX Mototrax: its riders and
/// bikes, ~2,200 PRIMs per race frame.
fn store_lit_run<const GPU: bool>(
    geometry: &mut Geometry,
    run: &RunInput,
    format: VertexFormat,
    layout: &VertexLayout,
    transform: &Transform,
    rows: &RowTransform,
    lighting: &VertexLighting,
    run_color: &RunColor,
) -> Run {
    let has_normals = layout.normal_offset.is_some();
    let color = |bytes: &VertexBytes, raw: &Raw3d| {
        let normal = if has_normals { Some(format.decode_normal(bytes, layout)?) } else { None };
        Some(run_color.of(Some(lighting.color(raw, normal))))
    };
    store_run::<GPU>(geometry, run, transform, rows, color, |bytes| format.decode_3d(bytes, layout))
}

/// Decodes (`decode` reads one vertex's fields), transforms and stores a
/// run's vertices, `color` giving each vertex's colour (`None`:
/// unreadable). `rows`: the run's (`Transform::rows`, or `weighted_rows`
/// for a run of one shared weight). Always inlined, so each decoder gets a
/// loop of its own with nothing but its fields in it.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn store_run<const GPU: bool>(
    geometry: &mut Geometry,
    run: &RunInput,
    transform: &Transform,
    rows: &RowTransform,
    color: impl Fn(&VertexBytes, &Raw3d) -> Option<u32>,
    decode: impl Fn(&VertexBytes) -> Option<Raw3d>,
) -> Run {
    // The unskinned case in locals (copied: kept in registers), the rest as before.
    let rows = *rows;
    // Several weights, for the GPU: skinned here, the rest on the GPU
    // (`Transform::skins_several_bones`).
    let skinned = (GPU && transform.skins_several_bones()).then(|| transform.skinned_row_transform());
    for index in 0..run.used as u32 {
        let bytes = VertexBytes::new(run.words, run.skew + run.stride * index);
        let Some(raw) = decode(&bytes) else {
            return Run::NotHandled;
        };
        let Some(color) = color(&bytes, &raw) else {
            return Run::NotHandled;
        };
        let screen = if transform.uses_these_rows(&rows, &raw.weights) {
            vertex_with_rows::<GPU>(&rows, raw.position, raw.texcoord, color)
        } else if let Some(skinned) = &skinned {
            vertex_with_rows::<true>(skinned, transform.skin_position(&raw), raw.texcoord, color)
        } else if GPU {
            None // one weight, not the run's: the run goes the CPU path
        } else {
            transform.frame_vertex(&raw, color)
        };
        let Some(screen) = screen else {
            return Run::NotHandled;
        };
        if geometry.add_vertex(run.first_vertex, screen).is_none() {
            return Run::FrameFull;
        }
    }
    recorded::<GPU>()
}

/// Why a draw couldn't be recorded.
enum Failure {
    /// Some of its vertices couldn't be read.
    Unreadable,
    /// The frame (or the draw's 16-bit index range) is full.
    FrameFull,
}

/// A 3D vertex after the transform.
#[derive(Clone, Copy, PartialEq)]
enum Corner {
    /// Its data couldn't be read.
    Unreadable,
    /// Behind the near plane: its triangles must be cut (rare, done anew).
    BehindNearPlane,
    /// On the screen, but the frame has no room left for it.
    FrameFull,
    /// Stored in the frame: its number within the draw (for the indices).
    OnScreen(u16),
}

/// The last few transformed vertices of a draw, by vertex number. A strip
/// uses vertices n-2, n-1, n for triangle n, a fan 0, n-1, n: keeping the
/// first vertex and the last three means every vertex is transformed once
/// (MTX Mototrax: ~3000 strips per frame).
struct CornerCache {
    first: Option<Corner>,
    recent: [(u32, Corner); 3],
}

impl CornerCache {
    fn new() -> Self {
        CornerCache { first: None, recent: [(u32::MAX, Corner::Unreadable); 3] }
    }

    #[inline]
    fn get(&mut self, index: u32, make: impl FnOnce(u32) -> Corner) -> Corner {
        if index == 0 {
            if let Some(first) = self.first {
                return first;
            }
        }
        let slot = &mut self.recent[index as usize % 3];
        if slot.0 != index {
            *slot = (index, make(index));
            if index == 0 {
                self.first = Some(slot.1);
            }
        }
        slot.1
    }
}

/// The parts of a draw that follow from the GE state alone, not from its
/// vertices: worked out once and reused while the state stays the same.
#[derive(Clone, Copy, Debug, PartialEq)]
struct DrawState {
    /// `first_vertex`, `vertex_count` and the RECTANGLES, DOUBLE_COLOR and
    /// FOG flags (with `fog_color`) are filled in per draw.
    draw: Draw,
    /// The frame's texture entry `draw.texture` (when textured), to check
    /// that it is still there when the state is reused.
    texture: TextureDescriptor,
    /// The frame's stencil state `draw.blend` names (`blend::stencil_index`),
    /// to check likewise.
    stencil: crate::blend::StencilState,
    /// 1 / texture size, to turn texel coordinates into 0..1. Sizes are
    /// powers of two, so multiplying by this is exact (the same as dividing).
    texel_scale: (f32, f32),
    /// The vertex alpha counts (blending, or a clear writing the stencil):
    /// otherwise vertices are made opaque.
    blending: bool,
    /// Source factor "fixed colour": multiplied into the colours instead.
    source_scale: Option<u32>,
    /// Doubled source alpha factors: the vertex alpha is doubled instead.
    double_source_alpha: bool,
    /// Approximations to count for every draw with this state (`approx` mask).
    approximations: u32,
    /// Textured with a palette format: the loaded palette is part of the state.
    uses_palette: bool,
}

/// The state of the last draw and what it was worked out from. Reusing it
/// gives exactly what working it out again would: same registers, same
/// palette and the same texture entry in the frame mean the same texture
/// index, palette copy and flags.
pub struct DrawStateCache {
    state: Option<DrawState>,
    state_changes: u32,
    /// The palette the state was worked out with (when it uses one): its
    /// load generation, and its address and length.
    palette_generation: u32,
    palette_key: (u32, u32),
    /// 3D draws: the last transform (see `TransformCache`).
    transforms: TransformCache,
    /// Whether textured 3D draws on the fast path are left to the GPU's
    /// transform (`frame::GpuTransform`; `Capture::set_gpu_transform`).
    /// The setting, not a cache: `forget` keeps it.
    pub gpu_transform: bool,
}

impl DrawStateCache {
    pub const fn new() -> Self {
        DrawStateCache {
            state: None,
            state_changes: 0,
            palette_generation: 0,
            palette_key: (0, 0),
            transforms: TransformCache::new(),
            gpu_transform: false,
        }
    }

    pub fn forget(&mut self) {
        self.state = None;
        self.transforms.forget();
    }

    /// True if the stored state is the one a draw with the current GE
    /// state, palette and frame has.
    fn fits(&mut self, ge: &GeState, palette: &mut LoadedPalette, catalog: &Catalog) -> bool {
        let Some(state) = &self.state else {
            return false;
        };
        if self.state_changes != ge.state_changes() {
            return false;
        }
        let texture_still_there = state.draw.flags & draw_flags::TEXTURED == 0
            || catalog.textures().get(state.draw.texture as usize) == Some(&state.texture);
        if !texture_still_there {
            return false;
        }
        // Its stencil state is in this frame's table (a new frame starts
        // an empty one).
        if let Some(index) = crate::blend::stencil_index(state.draw.blend) {
            if catalog.stencil_state(index) != Some(state.stencil) {
                return false;
            }
        }
        if !state.uses_palette || self.palette_generation == palette.generation {
            return true;
        }
        // Other palettes were loaded since: the same state if the one
        // loaded now is its palette again.
        if self.palette_key == palette.key() {
            self.palette_generation = palette.generation;
            return true;
        }
        false
    }

    fn store(&mut self, state: DrawState, ge: &GeState, palette: &mut LoadedPalette) {
        if state.uses_palette {
            self.palette_generation = palette.generation;
            self.palette_key = palette.key();
        }
        self.state = Some(state);
        self.state_changes = ge.state_changes();
    }
}

/// Works out a draw's state: render target, tests, scissor, texture (added
/// to the frame with its palette) and blending. `None` if the frame is full.
fn work_out_draw_state(catalog: &mut Catalog, stats: &mut FrameStats, palette: &mut LoadedPalette, ge: &GeState) -> Option<DrawState> {
    let mut approximations = 0;
    if ge.register(op::COLOR_MASK) & 0x00FF_FFFF != 0 {
        approximations |= approx::MASKED_COLOR;
    }

    let mut draw = Draw {
        framebuffer: ge.framebuffer_address(),
        tests: test_state(ge),
        ..Draw::default()
    };
    (draw.scissor_left, draw.scissor_top, draw.scissor_right, draw.scissor_bottom) = ge.scissor();
    // 16-bit targets: the display cuts colours as the GE does
    // (`draw_flags::TARGET_FORMAT_SHIFT`); dithering not in clear mode
    // (PPSSPP neither), nor with every colour bit masked (the mask keeps
    // the colour the dither would change).
    let target = match ge.register(op::FRAMEBUFFER_PIXEL_FORMAT) & 3 {
        0 => draw_flags::TARGET_565,
        1 => draw_flags::TARGET_5551,
        2 => draw_flags::TARGET_4444,
        _ => 0,
    };
    draw.flags |= target << draw_flags::TARGET_FORMAT_SHIFT;
    let colour_masked = ge.register(op::COLOR_MASK) & 0x00FF_FFFF == 0x00FF_FFFF;
    if target != 0 && ge.register(op::DITHER_ENABLE) & 1 != 0 && !ge.is_clear_mode() && !colour_masked {
        draw.flags |= draw_flags::DITHER;
        let row = |index: u8| (ge.register(op::DITHER_ROW_0 + index) & 0xFFFF) as u64;
        catalog.set_dither_matrix(row(0) | row(1) << 16 | row(2) << 32 | row(3) << 48);
    }
    // Back-face culling; never in clear mode (`draw_flags::CULL`).
    if ge.register(op::CULL_FACE_ENABLE) & 1 != 0 && !ge.is_clear_mode() {
        draw.flags |= draw_flags::CULL | if ge.register(op::CULL) & 1 != 0 { draw_flags::CULL_CCW } else { 0 };
    }

    // Texture state. Clear-mode draws are never textured.
    let mut texel_scale = (1.0, 1.0);
    let mut texture = TextureDescriptor::default();
    let mut uses_palette = false;
    if ge.is_texture_enabled() && !ge.is_clear_mode() {
        texture = describe_current_texture(ge);
        if is_palette_format(texture.format) {
            uses_palette = true;
            let Some(place) = palette.place_in(catalog) else {
                stats.frame_overflows += 1;
                return None;
            };
            texture.palette_offset = place.offset;
            texture.palette_word_count = place.word_count;
            // Its index for now (the words aren't read yet): the hash of the
            // words once the frame is finished.
            texture.palette_hash = place.index;
        }
        texel_scale = (1.0 / texture.width as f32, 1.0 / texture.height as f32);
        // Mip levels, if the menu option wants them (`texture::MIP_MODE`).
        let mips = crate::texture::describe_levels(ge, &texture);
        if let Some(mips) = &mips {
            texture.flags |= ((mips.count - 1) as u16) << crate::frame::texture_flags::LEVELS_SHIFT;
        }
        let Some(index) = catalog.add_texture(texture) else {
            stats.frame_overflows += 1;
            return None;
        };
        draw.texture = index;
        draw.flags |= draw_flags::TEXTURED;
        if let Some(mips) = &mips {
            catalog.set_texture_levels(index, &mips.levels);
            draw.flags |= draw_flags::MIPS
                | if mips.linear { draw_flags::MIP_LINEAR } else { 0 }
                | match mips.slope_base {
                    // Slope mode (setting `slopemip`): the level per pixel.
                    Some(base) => draw_flags::pack_slope_base(base),
                    None => (mips.bias_eighths as u8 as u32) << draw_flags::MIP_BIAS_SHIFT,
                };
        }

        let filter = ge.register(op::TEXTURE_FILTER);
        let wrap = ge.register(op::TEXTURE_WRAP);
        if filter & (1 << 8) != 0 {
            draw.flags |= draw_flags::LINEAR_FILTER;
        }
        if wrap & 1 != 0 {
            draw.flags |= draw_flags::CLAMP_U;
        }
        if wrap & (1 << 8) != 0 {
            draw.flags |= draw_flags::CLAMP_V;
        }
    }

    // The target's alpha is the PSP's stencil: kept unless the draw writes
    // the stencil (`blend::alpha_write`). A clear's vertex alpha is the
    // stencil value, so it isn't made opaque (`blending` below).
    let alpha_write = blend::alpha_write(ge);
    draw.blend = blend::alpha_write_bits(alpha_write, ge.clear_writes_color());
    // A stencil test and a constant written, done exactly on the target's
    // alpha (`blend::stencil`): named by its index in the frame's table.
    let exact_stencil = match blend::stencil(ge) {
        blend::Stencil::Exact(state) => catalog
            .add_stencil_state(state)
            .map(|index| draw.blend |= (index as u32 + 1) << blend::BLEND_STENCIL_SHIFT)
            .is_some(),
        _ => false,
    };
    if alpha_write == (blend::AlphaWrite::Fragment { approximated: true }) && !exact_stencil {
        approximations |= approx::STENCIL_ALPHA;
    }
    let blend = blend::translate(ge);
    if let Some(blend) = blend {
        draw.flags |= draw_flags::BLEND;
        draw.blend |= blend.pack();
        if blend.approximated {
            approximations |= if ge.is_blend_enabled() { approx::BLEND } else { approx::ALPHA_TEST };
            if ge.is_blend_enabled() {
                stats.approx_blend = [op::BLEND_MODE, op::BLEND_FIXED_A, op::BLEND_FIXED_B].map(|opcode| ge.register(opcode) & 0x00FF_FFFF);
                stats.approx_blend_states += 1;
            }
        }
    }

    let stencil = crate::blend::stencil_index(draw.blend).and_then(|index| catalog.stencil_state(index)).unwrap_or_default();
    Some(DrawState {
        draw,
        stencil,
        texture,
        texel_scale,
        blending: blend.is_some() || alpha_write == blend::AlphaWrite::StencilClear,
        source_scale: blend.and_then(|blend| blend.source_scale),
        double_source_alpha: blend.is_some_and(|blend| blend.double_source_alpha),
        approximations,
        uses_palette,
    })
}

/// `rgba` with its alpha doubled, at most 255.
fn double_alpha(rgba: u32) -> u32 {
    let alpha = ((rgba >> 24) * 2).min(255);
    (rgba & 0x00FF_FFFF) | alpha << 24
}

/// Blend factor "FIX": use the fixed colour from BLEND_FIXED_A / _B.
const BLEND_FACTOR_FIXED: u32 = 0xA;

/// True if the draw cannot change any visible colour: either all colour
/// bits are masked, or blending computes `source * 0 + destination * 1`
/// (fixed factors black and white), which keeps every pixel as it was.
fn writes_no_color(ge: &GeState) -> bool {
    if ge.register(op::COLOR_MASK) & 0x00FF_FFFF == 0x00FF_FFFF {
        return true;
    }
    if !ge.is_blend_enabled() || ge.is_clear_mode() {
        return false;
    }
    let blend = ge.register(op::BLEND_MODE);
    let (source, destination, equation) = (blend & 0xF, (blend >> 4) & 0xF, (blend >> 8) & 7);
    let add = 0;
    equation == add
        && source == BLEND_FACTOR_FIXED
        && ge.register(op::BLEND_FIXED_A) & 0x00FF_FFFF == 0
        && destination == BLEND_FACTOR_FIXED
        && ge.register(op::BLEND_FIXED_B) & 0x00FF_FFFF == 0x00FF_FFFF
}

/// True if a (non-clear) draw writes depth: the PSP writes depth only with
/// the depth test on.
fn writes_depth(ge: &GeState) -> bool {
    !ge.is_clear_mode() && ge.register(op::DEPTH_TEST_ENABLE) & 1 != 0 && ge.register(op::DEPTH_WRITE_DISABLE) & 1 == 0
}

/// Depth compare function "always" (ZTEST register value).
const DEPTH_ALWAYS: u32 = 1;

/// The per-pixel tests a draw runs with, packed into `Draw::tests`.
/// In clear mode the GE skips all tests and writes depth if the clear
/// asks for it: recorded as the depth test "always".
fn test_state(ge: &GeState) -> u32 {
    if ge.is_clear_mode() {
        return if ge.clear_writes_depth() { test_flags::DEPTH_TEST | DEPTH_ALWAYS << test_flags::DEPTH_FUNCTION_SHIFT } else { 0 };
    }
    let enabled = |opcode: u8| ge.register(opcode) & 1 != 0;
    let mut tests = 0;
    if enabled(op::ALPHA_TEST_ENABLE) {
        let alpha_test = ge.register(op::ALPHA_TEST);
        tests |= test_flags::ALPHA_TEST
            | (alpha_test & 7) << test_flags::ALPHA_FUNCTION_SHIFT
            | ((alpha_test >> 8) & 0xFF) << test_flags::ALPHA_REFERENCE_SHIFT
            | ((alpha_test >> 16) & 0xFF) << test_flags::ALPHA_MASK_SHIFT;
    }
    if enabled(op::DEPTH_TEST_ENABLE) {
        tests |= test_flags::DEPTH_TEST | (ge.register(op::DEPTH_TEST) & 7) << test_flags::DEPTH_FUNCTION_SHIFT;
    }
    if enabled(op::STENCIL_TEST_ENABLE) {
        tests |= test_flags::STENCIL_TEST;
    }
    if enabled(op::LOGIC_OP_ENABLE) {
        tests |= test_flags::LOGIC_OP;
    }
    if enabled(op::COLOR_TEST_ENABLE) {
        tests |= test_flags::COLOR_TEST;
    }
    if enabled(op::DEPTH_WRITE_DISABLE) {
        tests |= test_flags::DEPTH_WRITE_DISABLED;
    }
    tests
}

/// Reads the n-th vertex of the current draw, following the index buffer
/// if there is one.
///
/// Vertices are decoded straight from the memory's cache when it can lend
/// them (`PspMemory::borrow_words`), which saves copying them: most 2D draws
/// are a single rectangle of two vertices, so this runs for every draw.
struct VertexReader<'a, M: PspMemory> {
    format: VertexFormat,
    layout: &'a VertexLayout,
    memory: &'a mut M,
    /// Bytes of each vertex that are read (`VertexFormat::bytes_used`).
    used: u32,
    vertex_address: u32,
    index_address: u32,
}

impl<'a, M: PspMemory> VertexReader<'a, M> {
    /// The vertices at `vertex_address` (and their indices at
    /// `index_address`, for formats with an index buffer).
    fn new(format: VertexFormat, layout: &'a VertexLayout, memory: &'a mut M, vertex_address: u32, index_address: u32) -> Self {
        VertexReader {
            format,
            layout,
            memory,
            used: format.bytes_used(layout),
            vertex_address,
            index_address,
        }
    }

    /// The n-th vertex's fields for the 3D fast path (`VertexFormat::decode_3d`).
    #[inline]
    fn get_3d(&mut self, n: u32) -> Option<Raw3d> {
        let (format, layout) = (self.format, self.layout);
        self.decode_with(n, |bytes| format.decode_3d(bytes, layout))
    }

    /// `get_3d` and the vertex's normal (`None`: the format has none), for
    /// lit draws.
    fn get_3d_lit(&mut self, n: u32) -> Option<(Raw3d, Option<[f32; 3]>)> {
        let (format, layout) = (self.format, self.layout);
        self.decode_with(n, |bytes| {
            let raw = format.decode_3d(bytes, layout)?;
            let normal = if layout.normal_offset.is_some() { Some(format.decode_normal(bytes, layout)?) } else { None };
            Some((raw, normal))
        })
    }

    /// The n-th vertex's lit colour (before `frame_color`).
    fn lit_color(&mut self, n: u32, lighting: &VertexLighting) -> Option<u32> {
        let (raw, normal) = self.get_3d_lit(n)?;
        Some(lighting.color(&raw, normal))
    }

    /// `decode` applied to the n-th vertex's bytes.
    #[inline]
    fn decode_with<T>(&mut self, n: u32, decode: impl Fn(&VertexBytes) -> Option<T>) -> Option<T> {
        let address = self.address_of(n)?;
        let skew = address & 3;
        let words = ((skew + self.used + 3) / 4) as usize;
        if let Some(words) = self.memory.borrow_words(address & !3, words) {
            return decode(&VertexBytes::new(words, skew));
        }
        // Not lent by the memory (e.g. across two blocks): copied.
        let mut copy = [0u32; 20];
        self.memory.read_words(address & !3, copy.get_mut(..words)?)?;
        decode(&VertexBytes::new(&copy[..words], skew))
    }

    /// Where the n-th vertex of the draw is, following the index buffer.
    #[inline]
    fn address_of(&mut self, n: u32) -> Option<u32> {
        let index = match self.format.index_format() {
            IndexFormat::None => n,
            IndexFormat::U8 => self.memory.read_u8(self.index_address + n)? as u32,
            IndexFormat::U16 => self.memory.read_u16(self.index_address + 2 * n)? as u32,
            IndexFormat::U32 => self.memory.read_u32(self.index_address + 4 * n)?,
        };
        Some(self.vertex_address.wrapping_add(index.wrapping_mul(self.layout.size)))
    }

    fn get(&mut self, n: u32) -> Option<Vertex> {
        let address = self.address_of(n)?;
        let skew = address & 3;
        let words = ((skew + self.used + 3) / 4) as usize;
        if let Some(words) = self.memory.borrow_words(address & !3, words) {
            return self.format.decode(&VertexBytes::new(words, skew), self.layout);
        }
        self.format.read(self.memory, address, self.layout)
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn index_bounds_read_words_and_match_index_by_index() {
        use crate::memory::test_memory::TestMemory;
        let mut memory = TestMemory::default();
        let base = 0x0890_0000;
        // 300 u16 indices (more than one chunk) starting at a halfword
        // boundary, then unreadable memory.
        let words: Vec<u32> = (0..152u32).map(|n| ((n * 7 % 251) << 16) | (n * 13 % 509)).collect();
        memory.write(base, &words);
        for (format, address, count) in [
            (IndexFormat::U16, base + 2, 300),
            (IndexFormat::U8, base + 1, 500),
            (IndexFormat::U32, base + 4, 140),
            (IndexFormat::U16, base + 2, 400), // runs past the readable words
            (IndexFormat::U16, base + 600, 1),
        ] {
            let values: Vec<u32> = (0..count).map_while(|n| read_index(&mut memory, format, address + n * format.size())).collect();
            let expected = values.iter().min().copied().zip(values.iter().max().copied());
            assert_eq!(index_bounds(&mut memory, format, address, count), expected, "{format:?} {count}");
        }
        assert_eq!(index_bounds(&mut memory, IndexFormat::U16, 0x0A00_0000, 4), None);
    }

    use super::*;
    use crate::block_cache::{BlockCache, BlockSource, BLOCK_SIZE, BLOCK_WORDS};
    use crate::draw_stream::StreamHooks;
    use crate::frame::new_boxed_frame;
    use crate::memory::test_memory::TestMemory;

    const VERTICES: u32 = 0x0890_0000;
    /// Through mode, u16 texcoords, 8888 colour, s16 positions (16 bytes).
    const SPRITE_FORMAT: u32 = (1 << 23) | 2 | (7 << 2) | (2 << 7);

    fn cmd(opcode: u8, data: u32) -> Command {
        Command(((opcode as u32) << 24) | data)
    }

    /// Records one PRIM command with argument `prim_data`.
    fn record_prim_data(
        frame: &mut Frame,
        stats: &mut FrameStats,
        palette: &mut LoadedPalette,
        draw_states: &mut DrawStateCache,
        ge: &GeState,
        memory: &mut impl PspMemory,
        prim_data: u32,
    ) {
        record_now(frame, stats, palette, draw_states, ge, memory, &[cmd(op::PRIM, prim_data)]);
    }

    /// `record_prims` with the draw stream drained right after.
    fn record_now(
        frame: &mut Frame,
        stats: &mut FrameStats,
        palette: &mut LoadedPalette,
        draw_states: &mut DrawStateCache,
        ge: &GeState,
        memory: &mut impl PspMemory,
        prims: &[Command],
    ) {
        let mut stream = Box::new(DrawStream::new());
        let frame: *mut Frame = frame;
        // Safe: one thread; the catalog is only used here.
        let mut catalog = unsafe { Frame::catalog_raw(frame) };
        record_prims(frame, &mut catalog, stats, palette, draw_states, ge, memory, prims, &mut Queues { stream: &mut stream, builder: None });
        stream.finish(stats);
    }

    /// Writes a 16-byte sprite vertex.
    fn vertex(memory: &mut TestMemory, address: u32, x: i16, y: i16, u: u16, v: u16, color: u32) {
        memory.write(address, &[(v as u32) << 16 | u as u32, color, (y as u16 as u32) << 16 | x as u16 as u32, 0]);
    }

    /// `ge_state::DRAW_STATE_REGISTERS` is complete: for many GE states
    /// (random register values, so enables, texture formats with and
    /// without palettes, clear mode and blend modes all come up), changing
    /// any register not listed leaves the worked-out draw state the same.
    #[test]
    fn draws_into_16_bit_targets_say_so_with_their_dithering() {
        // MHF draws into a 5551 framebuffer with dithering on (guild hall,
        // 2026-11-17); FF1 and MTX into 8888 ones.
        let mut frame = new_boxed_frame();
        let mut stats = FrameStats::new();
        let mut palette = LoadedPalette::new();
        let frame_pointer: *mut Frame = &mut *frame;
        // Safe: one thread; the catalog is only used here.
        let mut catalog = unsafe { Frame::catalog_raw(frame_pointer) };
        let mut ge = GeState::new();
        ge.set_register(cmd(op::FRAMEBUFFER_PIXEL_FORMAT, 1));
        ge.set_register(cmd(op::DITHER_ENABLE, 1));
        for (row, value) in [0x000f0f, 0x00f0f1, 0x000f0f, 0x0000f1].into_iter().enumerate() {
            ge.set_register(cmd(op::DITHER_ROW_0 + row as u8, value));
        }
        let state = work_out_draw_state(&mut catalog, &mut stats, &mut palette, &ge).unwrap();
        assert_eq!(crate::frame::target_channel_bits(state.draw.flags), Some([5, 5, 5, 1]));
        assert_ne!(state.draw.flags & draw_flags::DITHER, 0);
        drop(catalog);
        assert_eq!(frame.dither_matrix, 0x000f0f | 0x00f0f1 << 16 | 0x000f0f << 32 | 0x0000f1 << 48);

        let frame_pointer: *mut Frame = &mut *frame;
        // Safe: as above.
        let mut catalog = unsafe { Frame::catalog_raw(frame_pointer) };
        let formats = [(0, Some([5, 6, 5, 0])), (2, Some([4, 4, 4, 4])), (3, None)];
        for (format, bits) in formats {
            ge.set_register(cmd(op::FRAMEBUFFER_PIXEL_FORMAT, format));
            let state = work_out_draw_state(&mut catalog, &mut stats, &mut palette, &ge).unwrap();
            assert_eq!(crate::frame::target_channel_bits(state.draw.flags), bits, "format {format}");
            assert_eq!(state.draw.flags & draw_flags::DITHER != 0, bits.is_some(), "format {format}");
        }
        // No dithering when it is off, nor in clear mode.
        ge.set_register(cmd(op::FRAMEBUFFER_PIXEL_FORMAT, 1));
        ge.set_register(cmd(op::DITHER_ENABLE, 0));
        let state = work_out_draw_state(&mut catalog, &mut stats, &mut palette, &ge).unwrap();
        assert_eq!(state.draw.flags & draw_flags::DITHER, 0);
        ge.set_register(cmd(op::DITHER_ENABLE, 1));
        ge.set_register(cmd(op::CLEAR_MODE, 1));
        let state = work_out_draw_state(&mut catalog, &mut stats, &mut palette, &ge).unwrap();
        assert_eq!(state.draw.flags & draw_flags::DITHER, 0);
        assert_ne!(state.draw.flags & draw_flags::TARGET_FORMAT_MASK, 0, "clears are still cut");
        // Nor with every colour bit masked (NFS UR's title, draw 13: a depth
        // mask drawn with dithering on): the GE adds the dither before the
        // write, and the mask keeps every colour bit (PPSSPP DrawPixel.cpp).
        ge.set_register(cmd(op::CLEAR_MODE, 0));
        ge.set_register(cmd(op::COLOR_MASK, 0xFF_FFFF));
        let state = work_out_draw_state(&mut catalog, &mut stats, &mut palette, &ge).unwrap();
        assert_eq!(state.draw.flags & draw_flags::DITHER, 0);
        // Part of the colour written: dithered.
        ge.set_register(cmd(op::COLOR_MASK, 0xFF_FF00));
        let state = work_out_draw_state(&mut catalog, &mut stats, &mut palette, &ge).unwrap();
        assert_ne!(state.draw.flags & draw_flags::DITHER, 0);
    }

    #[test]
    fn registers_left_out_of_the_draw_state_do_not_change_it() {
        use crate::ge_state::DRAW_STATE_REGISTERS;
        let mut frame = new_boxed_frame();
        let mut stats = FrameStats::new();
        let mut palette = LoadedPalette::new();
        let frame: *mut Frame = &mut *frame;
        // Safe: one thread; the catalog is only used here.
        let mut catalog = unsafe { Frame::catalog_raw(frame) };
        let mut seed = 0x1234_5678u32;
        let mut random = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed & 0xFF_FFFF
        };
        let (mut textured, mut paletted) = (0, 0);
        for _ in 0..300 {
            let mut ge = GeState::new();
            for opcode in DRAW_STATE_REGISTERS {
                ge.set_register(cmd(opcode, random()));
            }
            let expected = work_out_draw_state(&mut catalog, &mut stats, &mut palette, &ge).unwrap();
            textured += (expected.draw.flags & draw_flags::TEXTURED != 0) as u32;
            paletted += expected.uses_palette as u32;
            for opcode in (0..=255u8).filter(|opcode| !DRAW_STATE_REGISTERS.contains(opcode)) {
                let mut changed = ge.clone();
                changed.set_register(cmd(opcode, random()));
                let state = work_out_draw_state(&mut catalog, &mut stats, &mut palette, &changed).unwrap();
                assert_eq!(state, expected, "opcode {opcode:02x}");
            }
        }
        assert!(textured > 30 && paletted > 5, "textured {textured}, with palette {paletted}");
    }

    fn ge_for_sprites() -> GeState {
        let mut ge = GeState::new();
        ge.set_register(cmd(op::VERTEX_TYPE, SPRITE_FORMAT));
        ge.set_register(cmd(op::ALPHA_BLEND_ENABLE, 1));
        ge.vertex_address = VERTICES;
        ge
    }

    #[test]
    fn rectangle_becomes_two_triangles() {
        let mut memory = TestMemory::default();
        vertex(&mut memory, VERTICES, 10, 20, 0, 0, 0xFF00_0000);
        vertex(&mut memory, VERTICES + 16, 30, 40, 0, 0, 0xFFFF_FFFF);
        let ge = ge_for_sprites();
        let mut frame = new_boxed_frame();
        let mut stats = FrameStats::new();
        let mut palette = LoadedPalette::new();

        record_prim_data(&mut frame, &mut stats, &mut palette, &mut DrawStateCache::new(), &ge, &mut memory, (primitive::RECTANGLES << 16) | 2);

        assert_eq!(frame.vertices().len(), 6);
        assert_eq!(frame.draws().len(), 1);
        let xs: Vec<f32> = frame.vertices().iter().map(|v| v.x).collect();
        assert_eq!(xs, vec![10.0, 30.0, 10.0, 30.0, 30.0, 10.0]);
        // Second vertex's colour for the whole rectangle.
        assert!(frame.vertices().iter().all(|v| v.color == 0xFFFF_FFFF));
    }

    #[test]
    fn textured_rectangle_has_normalised_uvs() {
        let mut memory = TestMemory::default();
        vertex(&mut memory, VERTICES, 0, 0, 0, 0, 0xFFFF_FFFF);
        vertex(&mut memory, VERTICES + 16, 32, 16, 32, 16, 0xFFFF_FFFF);
        let mut ge = ge_for_sprites();
        ge.set_register(cmd(op::TEXTURE_MAP_ENABLE, 1));
        ge.set_register(cmd(op::TEXTURE_FORMAT, 3));
        ge.set_register(cmd(op::TEXTURE_SIZE_0, 6 | (5 << 8))); // 64 x 32
        ge.set_register(cmd(op::TEXTURE_ADDRESS_0, 0x95_0000));
        ge.set_register(cmd(op::TEXTURE_BUFFER_WIDTH_0, 0x08_0040));
        let mut frame = new_boxed_frame();
        let mut stats = FrameStats::new();
        let mut palette = LoadedPalette::new();

        record_prim_data(&mut frame, &mut stats, &mut palette, &mut DrawStateCache::new(), &ge, &mut memory, (primitive::RECTANGLES << 16) | 2);

        let texture = frame.textures()[0];
        assert_eq!((texture.width, texture.height, texture.address), (64, 32, 0x0895_0000));
        let corner = frame.vertices()[4];
        assert_eq!((corner.u, corner.v), (0.5, 0.5));
        assert!(frame.vertices().iter().all(|vertex| vertex.color == 0xFFFF_FFFF && vertex.w == 1.0));
        assert_eq!(stats.approximated_draws, 0);
    }

    #[test]
    fn texture_function_sets_the_vertex_colours() {
        let mut memory = TestMemory::default();
        vertex(&mut memory, VERTICES, 0, 0, 0, 0, 0x8012_3456);
        vertex(&mut memory, VERTICES + 16, 32, 16, 32, 16, 0x8012_3456);
        let mut ge = ge_for_sprites();
        ge.set_register(cmd(op::TEXTURE_MAP_ENABLE, 1));
        ge.set_register(cmd(op::TEXTURE_FORMAT, 3));
        ge.set_register(cmd(op::TEXTURE_SIZE_0, 6 | (5 << 8))); // 64 x 32
        ge.set_register(cmd(op::TEXTURE_ADDRESS_0, 0x95_0000));
        ge.set_register(cmd(op::TEXTURE_BUFFER_WIDTH_0, 0x08_0040));
        let mut colors = Vec::new();
        // Modulate; replace without and with the texture's alpha.
        for function in [0, 3, 3 | (1 << 8)] {
            ge.set_register(cmd(op::TEXTURE_FUNCTION, function));
            let mut frame = new_boxed_frame();
            let mut stats = FrameStats::new();
            record_prim_data(&mut frame, &mut stats, &mut LoadedPalette::new(), &mut DrawStateCache::new(), &ge, &mut memory, (primitive::RECTANGLES << 16) | 2);
            colors.push(frame.vertices()[0].color);
            assert_eq!(stats.approximated_draws, 0);
        }
        assert_eq!(colors, [0x8012_3456, 0x80FF_FFFF, 0xFFFF_FFFF]);
    }

    #[test]
    fn unblended_draws_become_opaque() {
        let mut memory = TestMemory::default();
        vertex(&mut memory, VERTICES, 0, 0, 0, 0, 0x0012_3456);
        vertex(&mut memory, VERTICES + 16, 1, 1, 0, 0, 0x0012_3456);
        let mut ge = ge_for_sprites();
        ge.set_register(cmd(op::ALPHA_BLEND_ENABLE, 0));
        let mut frame = new_boxed_frame();
        let mut stats = FrameStats::new();
        record_prim_data(&mut frame, &mut stats, &mut LoadedPalette::new(), &mut DrawStateCache::new(), &ge, &mut memory, (primitive::RECTANGLES << 16) | 2);
        assert_eq!(frame.vertices()[0].color, 0xFF12_3456);
    }

    #[test]
    fn draws_with_all_color_bits_masked_are_not_recorded() {
        let mut memory = TestMemory::default();
        vertex(&mut memory, VERTICES, 0, 0, 0, 0, 0xFFFF_FFFF);
        vertex(&mut memory, VERTICES + 16, 8, 8, 0, 0, 0xFFFF_FFFF);
        let mut ge = ge_for_sprites();
        ge.set_register(cmd(op::COLOR_MASK, 0xFF_FFFF));
        let mut frame = new_boxed_frame();
        let mut stats = FrameStats::new();
        record_prim_data(&mut frame, &mut stats, &mut LoadedPalette::new(), &mut DrawStateCache::new(), &ge, &mut memory, (primitive::RECTANGLES << 16) | 2);
        assert!(frame.draws().is_empty());
        assert_eq!(stats.color_masked_draws, 1);
    }

    #[test]
    fn stencil_only_draws_with_keep_blending_are_not_recorded() {
        let mut memory = TestMemory::default();
        vertex(&mut memory, VERTICES, 0, 0, 0, 0, 0xFFFF_FFFF);
        vertex(&mut memory, VERTICES + 16, 8, 8, 0, 0, 0xFFFF_FFFF);
        let mut ge = ge_for_sprites();
        // source * FIXA(black) + destination * FIXB(white)
        ge.set_register(cmd(op::BLEND_MODE, 0x0AA));
        ge.set_register(cmd(op::BLEND_FIXED_A, 0));
        ge.set_register(cmd(op::BLEND_FIXED_B, 0xFF_FFFF));
        let mut frame = new_boxed_frame();
        let mut stats = FrameStats::new();
        record_prim_data(&mut frame, &mut stats, &mut LoadedPalette::new(), &mut DrawStateCache::new(), &ge, &mut memory, (primitive::RECTANGLES << 16) | 2);
        assert!(frame.draws().is_empty());

        // A real fixed-colour blend is still drawn.
        ge.set_register(cmd(op::BLEND_FIXED_A, 0x80_8080));
        record_prim_data(&mut frame, &mut stats, &mut LoadedPalette::new(), &mut DrawStateCache::new(), &ge, &mut memory, (primitive::RECTANGLES << 16) | 2);
        assert_eq!(frame.draws().len(), 1);
    }

    /// The framebuffer's alpha is the PSP's stencil. MTX Mototrax clears it
    /// to 0 with each frame (CLEAR 701, material alpha 0) and writes it
    /// nowhere else; its blur quad weights the previous frame by it. We made
    /// every unblended draw (the clear too) opaque and wrote alpha with
    /// every draw: the blur kept 95 % of old frames (ghost sky, slot 8).
    #[test]
    fn only_stencil_clears_write_the_targets_alpha() {
        use crate::blend::{BLEND_ALPHA_ONLY, BLEND_WRITES_ALPHA};
        let mut memory = TestMemory::default();
        vertex(&mut memory, VERTICES, 0, 0, 0, 0, 0x0000_0000);
        vertex(&mut memory, VERTICES + 16, 8, 8, 0, 0, 0x0000_0000);
        let mut ge = ge_for_sprites();
        ge.set_register(cmd(op::ALPHA_BLEND_ENABLE, 0));
        ge.set_register(cmd(op::FRAMEBUFFER_PIXEL_FORMAT, 3)); // 8888
        let record = |ge: &GeState, memory: &mut TestMemory| {
            let mut frame = new_boxed_frame();
            let mut stats = FrameStats::new();
            record_prim_data(&mut frame, &mut stats, &mut LoadedPalette::new(), &mut DrawStateCache::new(), ge, memory, (primitive::RECTANGLES << 16) | 2);
            let draw = frame.draws().first().copied();
            (draw.map(|draw| draw.blend), frame.vertices().first().map(|vertex| vertex.color), stats.approximated_by_reason[8])
        };

        // The clear: colour, stencil, depth. Its alpha 0 is written as is.
        ge.set_register(cmd(op::CLEAR_MODE, 0x701));
        let (blend, color, _) = record(&ge, &mut memory);
        assert_eq!(blend.map(|blend| blend & (BLEND_WRITES_ALPHA | BLEND_ALPHA_ONLY)), Some(BLEND_WRITES_ALPHA));
        assert_eq!(color, Some(0x0000_0000));

        // The stencil alone: drawn (it was left out), only the alpha written.
        ge.set_register(cmd(op::CLEAR_MODE, 0x201));
        let (blend, _, _) = record(&ge, &mut memory);
        assert_eq!(blend.map(|blend| blend & (BLEND_WRITES_ALPHA | BLEND_ALPHA_ONLY)), Some(BLEND_WRITES_ALPHA | BLEND_ALPHA_ONLY));

        // A clear of colour and depth keeps it.
        ge.set_register(cmd(op::CLEAR_MODE, 0x501));
        let (blend, _, _) = record(&ge, &mut memory);
        assert_eq!(blend.map(|blend| blend & BLEND_WRITES_ALPHA), Some(0));

        // Draws keep it, unblended (made opaque, alpha unused) or blended.
        ge.set_register(cmd(op::CLEAR_MODE, 0));
        let (blend, color, _) = record(&ge, &mut memory);
        assert_eq!((blend, color), (Some(0), Some(0xFF00_0000)));
        ge.set_register(cmd(op::ALPHA_BLEND_ENABLE, 1));
        ge.set_register(cmd(op::BLEND_MODE, 0x032)); // src alpha, 1 - src alpha
        let (blend, _, _) = record(&ge, &mut memory);
        assert_eq!(blend.map(|blend| blend & BLEND_WRITES_ALPHA), Some(0));

        // Stencil ops that write a constant: exact, the frame's stencil
        // state named (`blend::stencil`). Others write the fragment alpha,
        // counted as approximated (not when ALPHA_MASK keeps every bit).
        ge.set_register(cmd(op::STENCIL_TEST_ENABLE, 1));
        ge.set_register(cmd(op::STENCIL_OP, 0x02_0000)); // REPLACE on pass
        let (blend, _, approximated) = record(&ge, &mut memory);
        assert_eq!((blend.map(|blend| blend & BLEND_WRITES_ALPHA), approximated), (Some(BLEND_WRITES_ALPHA), 0));
        assert_eq!(blend.and_then(crate::blend::stencil_index), Some(0));
        ge.set_register(cmd(op::STENCIL_OP, 0x04_0000)); // INCR on pass
        let (blend, _, approximated) = record(&ge, &mut memory);
        assert_eq!((blend.map(|blend| blend & BLEND_WRITES_ALPHA), approximated), (Some(BLEND_WRITES_ALPHA), 1));
        assert_eq!(blend.and_then(crate::blend::stencil_index), None);
        ge.set_register(cmd(op::ALPHA_MASK, 0xFF));
        let (blend, _, approximated) = record(&ge, &mut memory);
        assert_eq!((blend.map(|blend| blend & BLEND_WRITES_ALPHA), approximated), (Some(0), 0));

        // 565 framebuffers have no stencil: written as before.
        ge.set_register(cmd(op::FRAMEBUFFER_PIXEL_FORMAT, 0));
        ge.set_register(cmd(op::STENCIL_TEST_ENABLE, 0));
        let (blend, _, _) = record(&ge, &mut memory);
        assert_eq!(blend.map(|blend| blend & BLEND_WRITES_ALPHA), Some(BLEND_WRITES_ALPHA));
    }

    /// Blends still approximated (coloured FIX B, destination alpha) are
    /// counted with the GE's blend registers, so the capture log says which
    /// ones a scene uses (`approx_blend=mode/FIX A/FIX B`).
    #[test]
    fn approximated_blends_are_counted_with_their_registers() {
        let mut memory = TestMemory::default();
        vertex(&mut memory, VERTICES, 0, 0, 0, 0, 0xFFFF_FFFF);
        vertex(&mut memory, VERTICES + 16, 8, 8, 0, 0, 0xFFFF_FFFF);
        let mut ge = ge_for_sprites();
        // source * FIXA + destination * FIXB, FIX B coloured: approximated.
        ge.set_register(cmd(op::BLEND_MODE, 0x0AA));
        ge.set_register(cmd(op::BLEND_FIXED_A, 0x80_8080));
        ge.set_register(cmd(op::BLEND_FIXED_B, 0x20_4060));
        let mut frame = new_boxed_frame();
        let mut stats = FrameStats::new();
        record_prim_data(&mut frame, &mut stats, &mut LoadedPalette::new(), &mut DrawStateCache::new(), &ge, &mut memory, (primitive::RECTANGLES << 16) | 2);
        assert_eq!(frame.draws().len(), 1);
        assert_eq!(stats.approximated_by_reason[1], 1, "blend factor");
        assert_eq!((stats.approx_blend, stats.approx_blend_states), ([0x0AA, 0x80_8080, 0x20_4060], 1));
        let mut line = String::new();
        crate::report::write_frame(&mut line, &stats).unwrap();
        assert!(line.contains("approx=1[1:1] approx_blend=0000aa/808080/204060x1"), "{line}");
    }

    /// A sprite textured with a CLUT8 texture whose palette (`blocks` x 32
    /// bytes) is loaded from 0x0880_0000 (`load_palette`).
    fn ge_for_palette_sprites() -> GeState {
        let mut ge = ge_for_sprites();
        ge.set_register(cmd(op::TEXTURE_MAP_ENABLE, 1));
        ge.set_register(cmd(op::TEXTURE_FORMAT, 5)); // CLUT8
        ge.set_register(cmd(op::TEXTURE_SIZE_0, 4 | (4 << 8))); // 16 x 16
        ge.set_register(cmd(op::TEXTURE_ADDRESS_0, 0x95_0000));
        ge.set_register(cmd(op::TEXTURE_BUFFER_WIDTH_0, 0x08_0010));
        ge.set_register(cmd(op::CLUT_ADDRESS, 0x80_0000));
        ge.set_register(cmd(op::CLUT_ADDRESS_UPPER, 0x08_0000));
        ge
    }

    fn load_palette(palette: &mut LoadedPalette, ge: &GeState, blocks: u32) {
        let mut stats = FrameStats::new();
        let mut frame = new_boxed_frame();
        let mut stream = Box::new(DrawStream::new());
        let mut draw_states = DrawStateCache::new();
        let mut recorder = Recorder {
            stats: &mut stats,
            frame: Some(&mut frame),
            palette,
            draw_states: &mut draw_states,
            stream: &mut stream,
            builder: None,
        };
        recorder.visit(cmd(op::CLUT_LOAD, blocks), ge, &mut TestMemory::default());
    }

    /// CLUT_LOAD reads nothing (the capture thread's time, 2026-11-09): the
    /// first draw of the frame that uses the palette places it, the words
    /// are read with that draw's job, and the frame's texture carries their
    /// hash once the frame is finished. A second draw with the same palette
    /// in the frame uses the same place, even after the game rewrote it.
    #[test]
    fn a_palette_is_read_with_the_first_draw_that_uses_it() {
        let mut memory = TestMemory::default();
        vertex(&mut memory, VERTICES, 0, 0, 0, 0, 0xFFFF_FFFF);
        vertex(&mut memory, VERTICES + 16, 16, 16, 16, 16, 0xFFFF_FFFF);
        let mut ge = ge_for_palette_sprites();
        let mut palette = LoadedPalette::new();
        load_palette(&mut palette, &ge, 1);
        let generation = palette.generation;
        load_palette(&mut palette, &ge, 1);
        assert_eq!(palette.generation, generation, "loading the loaded palette again is no change");
        // Written after the load, before the draw: the draw's job reads it.
        memory.write(0x0880_0000, &[0x1111_1111; 8]);

        let mut frame = new_boxed_frame();
        let mut stats = FrameStats::new();
        let mut draw_states = DrawStateCache::new();
        let sprite = (primitive::RECTANGLES << 16) | 2;
        record_prim_data(&mut frame, &mut stats, &mut palette, &mut draw_states, &ge, &mut memory, sprite);
        memory.write(0x0880_0000, &[0x2222_2222; 8]);
        ge.vertex_address = VERTICES;
        ge.set_register(cmd(op::ALPHA_TEST_ENABLE, 1)); // another state: worked out again
        record_prim_data(&mut frame, &mut stats, &mut palette, &mut draw_states, &ge, &mut memory, sprite);
        assert_eq!(frame.draws().len(), 2);
        assert_eq!(frame.palettes(), &[crate::frame::FramePalette { address: 0x0880_0000, word_count: 8, offset: 0 }]);
        assert_eq!(frame.palette_words[..8], [0x1111_1111; 8], "read once in the frame");
        assert_eq!(frame.textures()[0].palette_hash, 0, "the palette's index while recording");
        frame.resolve_palette_hashes();
        assert_eq!(frame.textures()[0].palette_hash, crate::texture::hash_words(&[0x1111_1111; 8]));
        assert!(palette.take_unstored().is_none(), "the job took it");
    }

    /// A palette loaded when a dump starts comes back with the dump's state:
    /// by address since snapshot version 4, as words from older dumps.
    #[test]
    fn the_loaded_palette_is_saved_and_restored() {
        let ge = ge_for_palette_sprites();
        let mut palette = LoadedPalette::new();
        load_palette(&mut palette, &ge, 2);
        let mut bytes = [0u8; 64];
        let mut out = Writer::new(&mut bytes);
        palette.save_state(&mut out);
        assert_eq!(out.len(), 8, "address and length");
        let mut restored = LoadedPalette::new();
        restored.load_state(&mut Reader::new(&bytes), 4).unwrap();
        assert_eq!(restored.key(), (0x0880_0000, 16));

        // Version 3: the words themselves; a draw stores them as they are.
        let mut bytes = [0u8; 64];
        let mut old = Writer::new(&mut bytes);
        old.u32(8);
        (0..8).for_each(|word| old.u32(0x100 + word));
        let mut palette = LoadedPalette::new();
        palette.load_state(&mut Reader::new(&bytes), 3).unwrap();
        let mut memory = TestMemory::default();
        vertex(&mut memory, VERTICES, 0, 0, 0, 0, 0xFFFF_FFFF);
        vertex(&mut memory, VERTICES + 16, 16, 16, 16, 16, 0xFFFF_FFFF);
        let mut frame = new_boxed_frame();
        let mut stats = FrameStats::new();
        record_prim_data(&mut frame, &mut stats, &mut palette, &mut DrawStateCache::new(), &ge, &mut memory, (primitive::RECTANGLES << 16) | 2);
        assert_eq!(frame.palettes()[0].address, 0);
        assert_eq!(frame.palette_words[..8], [0x100, 0x101, 0x102, 0x103, 0x104, 0x105, 0x106, 0x107]);
    }

    /// 3D with float positions and identity matrices; the viewport maps
    /// clip (x, y) to screen (100 + 10 x, 100 + 10 y).
    fn ge_for_3d() -> GeState {
        let mut ge = GeState::new();
        let float = |value: f32| value.to_bits() >> 8;
        ge.set_register(cmd(op::VERTEX_TYPE, 3 << 7)); // float positions, 3D
        // Projection: w = 1, x and y as they are; viewport: screen = 100 + 10 * clip.
        ge.set_register(cmd(op::PROJECTION_MATRIX_NUMBER, 0));
        for value in [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0] {
            ge.set_register(cmd(op::PROJECTION_MATRIX_DATA, float(value)));
        }
        for (number, data) in [(op::WORLD_MATRIX_NUMBER, op::WORLD_MATRIX_DATA), (op::VIEW_MATRIX_NUMBER, op::VIEW_MATRIX_DATA)] {
            ge.set_register(cmd(number, 0));
            for value in [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0] {
                ge.set_register(cmd(data, float(value)));
            }
        }
        for (opcode, value) in [(op::VIEWPORT_X_SCALE, 10.0), (op::VIEWPORT_Y_SCALE, 10.0), (op::VIEWPORT_X_CENTER, 100.0), (op::VIEWPORT_Y_CENTER, 100.0)] {
            ge.set_register(cmd(opcode, float(value)));
        }
        ge
    }

    /// Writes float positions to `address` and points the GE at them.
    fn positions_3d(memory: &mut TestMemory, ge: &mut GeState, address: u32, positions: &[[f32; 3]]) {
        let words: Vec<u32> = positions.iter().flatten().map(|value| value.to_bits()).collect();
        memory.write(address, &words);
        ge.vertex_address = address;
    }

    /// PPSSPP's rounding (factors 2c + 1, >> 10): full channels stay full,
    /// half ambient halves, emission adds and clamps, alpha too.
    #[test]
    fn ambient_lighting_without_lights_scales_the_vertex_colour() {
        let lighting = AmbientLighting { ambient: 0xFF80_80FF, emissive: 0x0000_0010, vertex_ambient: true, material: 0x8000_00FF };
        // R: 255 x 255 + 16 -> 255 (clamped); G: 255 x 128 -> 128; B: 0 x 128 -> 0; A: 255 x 255.
        assert_eq!(lighting.color(Some(0xFF00_FFFF)), 0xFF00_80FF);
        // No vertex colour: the material's (R 255, A 128 x 255 = 128).
        assert_eq!(lighting.color(None), 0x8000_00FF);
        // MATERIAL_UPDATE off: the material's even with a vertex colour.
        let material_only = AmbientLighting { vertex_ambient: false, ..lighting };
        assert_eq!(material_only.color(Some(0xFFFF_FFFF)), 0x8000_00FF);
        let white = AmbientLighting { ambient: 0xFFFF_FFFF, emissive: 0, vertex_ambient: true, material: 0 };
        assert_eq!(white.color(Some(0x8040_2010)), 0x8040_2010);
    }

    #[test]
    fn a_lit_draw_without_lights_is_drawn_with_the_ambient_light() {
        let mut memory = TestMemory::default();
        let mut ge = ge_for_3d();
        // Float positions with 8888 colours (colour before position).
        ge.set_register(cmd(op::VERTEX_TYPE, (3 << 7) | (7 << 2)));
        let float = |value: f32| value.to_bits();
        let words: Vec<u32> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
            .iter()
            .flat_map(|&[x, y]| [0xFFFF_FFFF, float(x), float(y), float(0.0)])
            .collect();
        memory.write(VERTICES, &words);
        ge.vertex_address = VERTICES;
        ge.set_register(cmd(op::LIGHTING_ENABLE, 1));
        ge.set_register(cmd(op::MATERIAL_UPDATE, 1));
        ge.set_register(cmd(op::AMBIENT_COLOR, 0x40_8080));
        ge.set_register(cmd(op::AMBIENT_ALPHA, 0xFF));

        let mut frame = new_boxed_frame();
        let mut stats = FrameStats::new();
        record_prim_data(&mut frame, &mut stats, &mut LoadedPalette::new(), &mut DrawStateCache::new(), &ge, &mut memory, (primitive::TRIANGLES << 16) | 3);
        assert_eq!(frame.draws().len(), 1);
        assert!(frame.vertices[..3].iter().all(|vertex| vertex.color == 0xFF40_8080), "{:08x}", frame.vertices[0].color);
        assert_eq!(stats.approximated_draws, 0, "lit without lights is no approximation now");

        // With a light on (direction 0 = (0, 0, 1), along the missing
        // normal's (0, 0, 1)): its diffuse colour adds to the ambient light.
        ge.set_register(cmd(op::LIGHT_ENABLE_0, 1));
        ge.set_register(cmd(op::MATERIAL_UPDATE, 3));
        ge.set_register(cmd(op::LIGHT_AMBIENT_0 + 1, 0x20_2020));
        let mut frame = new_boxed_frame();
        let mut stats = FrameStats::new();
        record_prim_data(&mut frame, &mut stats, &mut LoadedPalette::new(), &mut DrawStateCache::new(), &ge, &mut memory, (primitive::TRIANGLES << 16) | 3);
        assert_eq!(frame.vertices[0].color, 0xFF60_A0A0);
        assert_eq!(stats.approximated_draws, 0);
    }

    /// Lit draws with a light: each vertex's colour from its normal, the
    /// same through the run path (no index buffer) and the general path
    /// (8-bit indices), skinned normals included.
    #[test]
    fn lit_draws_with_a_light_are_shaded_by_their_normals() {
        let float = |value: f32| value.to_bits();
        let float24 = |value: f32| value.to_bits() >> 8;
        // Two triangles: the first faces the light (normal +z), the second
        // away from it (-z). Float positions and normals, 8888 colours.
        let words: Vec<u32> = (0..6)
            .flat_map(|index| {
                let facing = if index < 3 { 1.0 } else { -1.0 };
                let (x, y) = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)][index % 3];
                [0xFF80_8080, float(0.0), float(0.0), float(facing), float(x), float(y), float(0.0)]
            })
            .collect();
        for (indexed, skinned) in [(false, false), (true, false), (false, true)] {
            let mut memory = TestMemory::default();
            let mut ge = ge_for_3d();
            // Colour 8888 (bits 2..4), float normals (5..6), float positions (7..8).
            let mut format = (7 << 2) | (3 << 5) | (3 << 7);
            let mut vertex_words = words.clone();
            if skinned {
                // One float weight of 1.0, before the other fields; the bone
                // turns normals around (z -> -z): the first triangle now
                // faces away and the second faces the light.
                format |= (3 << 9) | (0 << 14);
                vertex_words = words.chunks(7).flat_map(|vertex| [&[float(1.0)][..], vertex].concat()).collect();
                ge.set_register(cmd(op::BONE_MATRIX_NUMBER, 0));
                for value in [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0] {
                    ge.set_register(cmd(op::BONE_MATRIX_DATA, float24(value)));
                }
            }
            memory.write(VERTICES, &vertex_words);
            ge.vertex_address = VERTICES;
            if indexed {
                format |= 1 << 11; // 8-bit indices
                memory.write(VERTICES + 0x1000, &[0x0302_0100, 0x0000_0504]);
                ge.index_address = VERTICES + 0x1000;
            }
            ge.set_register(cmd(op::VERTEX_TYPE, format));
            ge.set_register(cmd(op::LIGHTING_ENABLE, 1));
            ge.set_register(cmd(op::LIGHT_ENABLE_0, 1));
            ge.set_register(cmd(op::LIGHT_X_0 + 2, float24(1.0))); // towards the light: +z
            ge.set_register(cmd(op::LIGHT_AMBIENT_0 + 1, 0xFF_FFFF)); // white diffuse
            ge.set_register(cmd(op::MATERIAL_UPDATE, 3));
            ge.set_register(cmd(op::AMBIENT_ALPHA, 0xFF));

            let mut frame = new_boxed_frame();
            let mut stats = FrameStats::new();
            record_prim_data(&mut frame, &mut stats, &mut LoadedPalette::new(), &mut DrawStateCache::new(), &ge, &mut memory, (primitive::TRIANGLES << 16) | 6);
            assert_eq!(frame.draws().len(), 1, "indexed {indexed} skinned {skinned}");
            let draw = frame.draws()[0];
            let colors: Vec<u32> = (0..6)
                .map(|corner| {
                    let number = if draw.flags & draw_flags::INDEX_RUNS != 0 { corner } else { frame.indices[draw.first_index as usize + corner] as usize };
                    frame.vertices[draw.first_vertex as usize + number].color
                })
                .collect();
            let (lit, dark) = (0xFF80_8080, 0xFF00_0000);
            let expected = if skinned { [dark, dark, dark, lit, lit, lit] } else { [lit, lit, lit, dark, dark, dark] };
            assert_eq!(colors, expected, "indexed {indexed} skinned {skinned}");
        }
    }

    /// CULL_FACE_ENABLE and CULL become the draw's cull flags; clear mode
    /// and rectangles are never culled; strips are stored with one winding
    /// (every second triangle's first two corners swapped).
    #[test]
    fn culling_goes_into_the_draw_flags_and_strips_keep_one_winding() {
        let mut memory = TestMemory::default();
        let mut ge = ge_for_3d();
        positions_3d(&mut memory, &mut ge, VERTICES, &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]);
        // Indexed (general path): the indices are stored as they are.
        ge.set_register(cmd(op::VERTEX_TYPE, (3 << 7) | (1 << 11)));
        memory.write(VERTICES + 0x100, &[0x0302_0100]);
        ge.index_address = VERTICES + 0x100;
        ge.set_register(cmd(op::CULL_FACE_ENABLE, 1));
        ge.set_register(cmd(op::CULL, 1));
        let record = |ge: &GeState, memory: &mut TestMemory, prim: u32| {
            let mut frame = new_boxed_frame();
            record_prim_data(&mut frame, &mut FrameStats::new(), &mut LoadedPalette::new(), &mut DrawStateCache::new(), ge, memory, prim);
            frame
        };
        let frame = record(&ge, &mut memory, (primitive::TRIANGLE_STRIP << 16) | 4);
        let draw = frame.draws()[0];
        assert_eq!(draw.flags & (draw_flags::CULL | draw_flags::CULL_CCW), draw_flags::CULL | draw_flags::CULL_CCW);
        let corners: Vec<(f32, f32)> = frame.screen_corners(&draw).map(|vertex| (vertex.x, vertex.y)).collect();
        // Strip 0 1 2, 1 2 3: stored as 0 1 2 and 2 1 3, both turning the same way.
        let turn = |c: &[(f32, f32)]| (c[1].0 - c[0].0) * (c[2].1 - c[0].1) - (c[1].1 - c[0].1) * (c[2].0 - c[0].0);
        assert_eq!(corners.len(), 6);
        assert!(turn(&corners[..3]) * turn(&corners[3..]) > 0.0, "{corners:?}");

        ge.set_register(cmd(op::CULL, 0));
        let draw = record(&ge, &mut memory, (primitive::TRIANGLES << 16) | 3).draws()[0];
        assert_eq!(draw.flags & (draw_flags::CULL | draw_flags::CULL_CCW), draw_flags::CULL);
        let draw = record(&ge, &mut memory, (primitive::RECTANGLES << 16) | 2).draws()[0];
        assert_eq!(draw.flags & draw_flags::CULL, 0, "rectangles aren't culled");
        ge.set_register(cmd(op::CLEAR_MODE, 1 | 1 << 8));
        let draw = record(&ge, &mut memory, (primitive::TRIANGLES << 16) | 3).draws()[0];
        assert_eq!(draw.flags & draw_flags::CULL, 0, "clears aren't culled");
    }

    #[test]
    fn three_d_triangles_are_transformed() {
        let mut memory = TestMemory::default();
        let mut ge = ge_for_3d();
        positions_3d(&mut memory, &mut ge, VERTICES, &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]);

        let mut frame = new_boxed_frame();
        let mut stats = FrameStats::new();
        record_prim_data(&mut frame, &mut stats, &mut LoadedPalette::new(), &mut DrawStateCache::new(), &ge, &mut memory, (primitive::TRIANGLES << 16) | 3);
        assert_eq!(frame.draws().len(), 1);
        let corners: Vec<(f32, f32)> = frame.vertices[..3].iter().map(|vertex| (vertex.x, vertex.y)).collect();
        assert_eq!(corners, [(100.0, 100.0), (110.0, 100.0), (100.0, 120.0)]);
        assert_eq!((stats.skipped_3d_draws, stats.approximated_draws), (0, 0));
    }

    #[test]
    fn the_gpu_transform_draws_what_the_cpu_draws() {
        // Textured 3D: a strip in front, then one with a vertex behind the
        // near plane (clip z < -w), which the CPU clips as before.
        let mut memory = TestMemory::default();
        let mut ge = ge_for_3d();
        ge.set_register(cmd(op::TEXTURE_MAP_ENABLE, 1));
        let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.5], [0.0, 1.0, 0.0], [1.0, 1.0, -0.5], [0.0, 2.0, 0.0], [1.0, 2.0, -3.0], [0.0, 3.0, 0.0]];
        positions_3d(&mut memory, &mut ge, VERTICES, &positions);
        let prims = [cmd(op::PRIM, primitive::TRIANGLE_STRIP << 16 | 4), cmd(op::PRIM, primitive::TRIANGLE_STRIP << 16 | 3)];
        let mut record = |gpu_transform: bool| {
            let mut frame = new_boxed_frame();
            let mut states = DrawStateCache::new();
            states.gpu_transform = gpu_transform;
            let mut ge = ge.clone();
            for prim in prims {
                record_now(&mut frame, &mut FrameStats::new(), &mut LoadedPalette::new(), &mut states, &ge, &mut memory, &[prim]);
                ge.advance_after_draw(prim.data() & 0xFFFF);
            }
            frame
        };
        let (cpu, gpu) = (record(false), record(true));
        let screen = |frame: &Frame| -> Vec<FrameVertex> { frame.draws().iter().flat_map(|draw| frame.screen_corners(draw)).collect() };
        assert_eq!(screen(&gpu), screen(&cpu));
        // (description_hash differs: draws with different transforms don't
        // merge, so the GPU frame has two draws where the CPU's has one.)
        assert_eq!(gpu.vertex_hash(), cpu.vertex_hash());
        let transforms: Vec<u32> = gpu.draws().iter().map(|draw| draw.transform).collect();
        assert_eq!(transforms, [1, 0], "the GPU transforms the first strip only");
        assert!(cpu.draws().iter().all(|draw| draw.transform == 0));
        // Model space as read.
        assert_eq!((gpu.vertices[1].x, gpu.vertices[1].z), (1.0, 0.5));
    }

    /// Lit draws the GPU lights (`store_gpu_lit_run`): the same picture as
    /// the CPU's lighting (colours within two steps, positions exact),
    /// unskinned and with one bone (MTX's riders: 8-bit weight 248/128); the
    /// draw's transform carries the lights, its vertices the raw colour.
    #[test]
    fn gpu_lit_draws_look_like_cpu_lit_ones() {
        let float = |value: f32| value.to_bits();
        let float24 = |value: f32| value.to_bits() >> 8;
        for skinned in [false, true] {
            let mut memory = TestMemory::default();
            let mut ge = ge_for_3d();
            ge.set_register(cmd(op::TEXTURE_MAP_ENABLE, 1));
            // [u8 weight, 3 pad] 8888 colour, float normal, float position.
            let mut format = (7 << 2) | (3 << 5) | (3 << 7);
            if skinned {
                format |= 1 << 9; // one u8 weight
                // The bone turns about z by 90 degrees.
                ge.set_register(cmd(op::BONE_MATRIX_NUMBER, 0));
                for value in [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0] {
                    ge.set_register(cmd(op::BONE_MATRIX_DATA, float24(value)));
                }
            }
            ge.set_register(cmd(op::VERTEX_TYPE, format));
            let mut bytes: Vec<u8> = Vec::new();
            for (index, (x, y)) in [(0.0f32, 0.0f32), (0.4, 0.0), (0.0, 0.4), (0.4, 0.4)].into_iter().enumerate() {
                if skinned {
                    bytes.extend([248, 0, 0, 0]);
                }
                let tilt = index as f32 * 0.3;
                for word in [0xFF60_A0C0, float(tilt), float(0.5), float(1.0), float(x), float(y), float(0.0)] {
                    bytes.extend(u32::to_le_bytes(word));
                }
            }
            let words: Vec<u32> = bytes.chunks(4).map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap())).collect();
            memory.write(VERTICES, &words);
            ge.vertex_address = VERTICES;
            ge.set_register(cmd(op::LIGHTING_ENABLE, 1));
            ge.set_register(cmd(op::LIGHT_ENABLE_0, 1));
            ge.set_register(cmd(op::LIGHT_TYPE_0, 1)); // diffuse and specular
            ge.set_register(cmd(op::LIGHT_X_0, float24(0.3)));
            ge.set_register(cmd(op::LIGHT_X_0 + 1, float24(0.2)));
            ge.set_register(cmd(op::LIGHT_X_0 + 2, float24(1.0)));
            ge.set_register(cmd(op::LIGHT_AMBIENT_0 + 1, 0xC0_C0C0));
            ge.set_register(cmd(op::LIGHT_AMBIENT_0 + 2, 0x80_8080));
            ge.set_register(cmd(op::MATERIAL_UPDATE, 3));
            ge.set_register(cmd(op::MATERIAL_SPECULAR, 0xFF_FFFF));
            ge.set_register(cmd(op::MATERIAL_SPECULAR_COEFFICIENT, float24(4.0)));
            ge.set_register(cmd(op::AMBIENT_COLOR, 0x20_2020));
            ge.set_register(cmd(op::AMBIENT_ALPHA, 0xFF));
            let prim = cmd(op::PRIM, primitive::TRIANGLE_STRIP << 16 | 4);
            let mut record = |gpu_transform: bool| {
                let mut frame = new_boxed_frame();
                let mut states = DrawStateCache::new();
                states.gpu_transform = gpu_transform;
                record_now(&mut frame, &mut FrameStats::new(), &mut LoadedPalette::new(), &mut states, &ge, &mut memory, &[prim]);
                frame
            };
            let (cpu, gpu) = (record(false), record(true));
            let screen = |frame: &Frame| -> Vec<FrameVertex> { frame.draws().iter().flat_map(|draw| frame.screen_corners(draw)).collect() };
            let (cpu_corners, gpu_corners) = (screen(&cpu), screen(&gpu));
            assert_eq!(gpu_corners.len(), 6, "skinned {skinned}");
            for (a, b) in cpu_corners.iter().zip(&gpu_corners) {
                assert!((a.x - b.x).abs() < 1e-3 && (a.y - b.y).abs() < 1e-3, "skinned {skinned}: {a:?} {b:?}");
                for shift in [0, 8, 16, 24] {
                    let difference = ((a.color >> shift) & 0xFF).abs_diff((b.color >> shift) & 0xFF);
                    assert!(difference <= 2, "skinned {skinned}: {:08x} {:08x}", a.color, b.color);
                }
            }
            let draw = gpu.draws()[0];
            let transform = gpu.transform_of(&draw).expect("model space");
            assert!(gpu.light_set_of(transform).is_some(), "skinned {skinned}: the lights go along");
            assert_eq!(gpu.vertices[0].color, 0xFF60_A0C0, "the raw colour is stored");
            assert_ne!(cpu_corners[0].color, 0xFF60_A0C0, "and it is lit");
        }
    }

    /// The GPU-lit loops for MTX's lit formats (`store_gpu_lit_common`:
    /// 16-bit positions and texture coordinates, 4444 colour, 8-bit normals)
    /// with no bone, one bone (weight 248/128, as MTX's riders) and three
    /// bones (skinned here, one weight zero): the CPU's lighting's picture.
    #[test]
    fn gpu_lit_runs_of_mtx_formats_look_like_cpu_lit_ones() {
        let float24 = |value: f32| value.to_bits() >> 8;
        for bones in [0usize, 1, 3] {
            let mut memory = TestMemory::default();
            let mut ge = ge_for_3d();
            ge.set_register(cmd(op::TEXTURE_MAP_ENABLE, 1));
            let mut format = 2 | (6 << 2) | (1 << 5) | (2 << 7);
            if bones > 0 {
                format |= (1 << 9) | ((bones as u32 - 1) << 14); // u8 weights
            }
            ge.set_register(cmd(op::VERTEX_TYPE, format));
            // Bone b: turned about z by 30 b degrees, scaled by 1 + b / 4,
            // moved by (0.1 b, 0, 0).
            ge.set_register(cmd(op::BONE_MATRIX_NUMBER, 0));
            for bone in 0..bones {
                let (sin, cos) = (bone as f32 * 0.5236).sin_cos();
                let k = 1.0 + bone as f32 / 4.0;
                for value in [cos * k, sin * k, 0.0, -sin * k, cos * k, 0.0, 0.0, 0.0, k, 0.1 * bone as f32, 0.0, 0.0] {
                    ge.set_register(cmd(op::BONE_MATRIX_DATA, float24(value)));
                }
            }
            let layout = VertexFormat(format).layout();
            assert!(VertexFormat(format).common_layout(&layout).is_some() && VertexFormat(format).has_s8_normals(), "the specialised loops' formats");
            let mut bytes: Vec<u8> = Vec::new();
            for (index, (x, y)) in [(0.0f32, 0.0f32), (0.4, 0.0), (0.0, 0.4), (0.4, 0.4)].into_iter().enumerate() {
                let mut vertex = vec![0u8; layout.size as usize];
                let weights: &[u8] = match bones {
                    1 => &[248],
                    3 => &[[64, 40, 24], [72, 40, 16], [80, 40, 8], [88, 40, 0]][index],
                    _ => &[],
                };
                vertex[..weights.len()].copy_from_slice(weights);
                let at = layout.texcoord_offset.unwrap() as usize;
                vertex[at..at + 4].copy_from_slice(&[0x00, 0x40, 0x00, 0x20]);
                let at = layout.color_offset.unwrap() as usize;
                vertex[at..at + 2].copy_from_slice(&0xF8ACu16.to_le_bytes());
                let at = layout.normal_offset.unwrap() as usize;
                vertex[at..at + 3].copy_from_slice(&[(index as i8 * 30) as u8, 64, 100]);
                let at = layout.position_offset as usize;
                for (axis, value) in [x, y, 0.0].into_iter().enumerate() {
                    vertex[at + 2 * axis..at + 2 * axis + 2].copy_from_slice(&((value * 32767.0) as i16).to_le_bytes());
                }
                bytes.extend(vertex);
            }
            bytes.resize(bytes.len().div_ceil(4) * 4, 0);
            let words: Vec<u32> = bytes.chunks(4).map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap())).collect();
            memory.write(VERTICES, &words);
            ge.vertex_address = VERTICES;
            ge.set_register(cmd(op::LIGHTING_ENABLE, 1));
            ge.set_register(cmd(op::LIGHT_ENABLE_0, 1));
            ge.set_register(cmd(op::LIGHT_TYPE_0, 1)); // diffuse and specular
            ge.set_register(cmd(op::LIGHT_X_0, float24(0.3)));
            ge.set_register(cmd(op::LIGHT_X_0 + 1, float24(0.2)));
            ge.set_register(cmd(op::LIGHT_X_0 + 2, float24(1.0)));
            ge.set_register(cmd(op::LIGHT_AMBIENT_0 + 1, 0xC0_C0C0));
            ge.set_register(cmd(op::LIGHT_AMBIENT_0 + 2, 0x80_8080));
            ge.set_register(cmd(op::MATERIAL_UPDATE, 3));
            ge.set_register(cmd(op::MATERIAL_SPECULAR, 0xFF_FFFF));
            ge.set_register(cmd(op::MATERIAL_SPECULAR_COEFFICIENT, float24(4.0)));
            ge.set_register(cmd(op::AMBIENT_COLOR, 0x20_2020));
            ge.set_register(cmd(op::AMBIENT_ALPHA, 0xFF));
            let prim = cmd(op::PRIM, primitive::TRIANGLE_STRIP << 16 | 4);
            let mut record = |gpu_transform: bool| {
                let mut frame = new_boxed_frame();
                let mut states = DrawStateCache::new();
                states.gpu_transform = gpu_transform;
                record_now(&mut frame, &mut FrameStats::new(), &mut LoadedPalette::new(), &mut states, &ge, &mut memory, &[prim]);
                frame
            };
            let (cpu, gpu) = (record(false), record(true));
            let screen = |frame: &Frame| -> Vec<FrameVertex> { frame.draws().iter().flat_map(|draw| frame.screen_corners(draw)).collect() };
            let (cpu_corners, gpu_corners) = (screen(&cpu), screen(&gpu));
            assert_eq!(gpu_corners.len(), 6, "{bones} bones");
            for (a, b) in cpu_corners.iter().zip(&gpu_corners) {
                assert!((a.x - b.x).abs() < 1e-3 && (a.y - b.y).abs() < 1e-3, "{bones} bones: {a:?} {b:?}");
                for shift in [0, 8, 16, 24] {
                    let difference = ((a.color >> shift) & 0xFF).abs_diff((b.color >> shift) & 0xFF);
                    assert!(difference <= 2, "{bones} bones: {:08x} {:08x}", a.color, b.color);
                }
            }
            let draw = gpu.draws()[0];
            let transform = gpu.transform_of(&draw).expect("model space: the GPU-lit loop took the run");
            assert!(gpu.light_set_of(transform).is_some(), "{bones} bones: the lights go along");
            assert_ne!(cpu_corners[0].color, gpu.vertices[0].color, "{bones} bones: lit");
        }
    }

    #[test]
    fn skinned_runs_with_several_bones_go_to_the_gpu_and_draw_what_the_cpu_draws() {
        // Two weights per vertex (float), two bones: skinned while decoding,
        // the rest on the GPU (as PPSSPP), and the same picture as the CPU.
        let mut memory = TestMemory::default();
        let mut ge = ge_for_3d();
        let float = |value: f32| value.to_bits() >> 8;
        ge.set_register(cmd(op::TEXTURE_MAP_ENABLE, 1));
        ge.set_register(cmd(op::VERTEX_TYPE, (3 << 9) | (1 << 14) | (3 << 7)));
        ge.set_register(cmd(op::BONE_MATRIX_NUMBER, 0));
        // Bone 0: identity; bone 1: moved by (1, 0.5, 0).
        for value in [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.5, 0.0] {
            ge.set_register(cmd(op::BONE_MATRIX_DATA, float(value)));
        }
        let vertices: [[f32; 5]; 4] =
            [[1.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 1.0, 0.0, 0.5], [0.25, 0.75, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, 1.0, -0.5]];
        let words: Vec<u32> = vertices.iter().flatten().map(|value| value.to_bits()).collect();
        memory.write(VERTICES, &words);
        ge.vertex_address = VERTICES;
        let prim = cmd(op::PRIM, primitive::TRIANGLE_STRIP << 16 | 4);
        let mut record = |gpu_transform: bool| {
            let mut frame = new_boxed_frame();
            let mut states = DrawStateCache::new();
            states.gpu_transform = gpu_transform;
            record_now(&mut frame, &mut FrameStats::new(), &mut LoadedPalette::new(), &mut states, &ge, &mut memory, &[prim]);
            frame
        };
        let (cpu, gpu) = (record(false), record(true));
        let screen = |frame: &Frame| -> Vec<FrameVertex> { frame.draws().iter().flat_map(|draw| frame.screen_corners(draw)).collect() };
        assert_eq!(screen(&gpu).len(), 6);
        assert_eq!(screen(&gpu), screen(&cpu));
        assert_eq!(gpu.draws()[0].transform, 1, "the GPU transforms the skinned strip");
        assert_eq!(cpu.draws()[0].transform, 0);
        // Model space after skinning: vertex 2 = 0.25 x p + 0.75 x (p + (1, 0.5, 0)).
        assert_eq!((gpu.vertices[2].x, gpu.vertices[2].y), (0.75, 1.375));
    }

    #[test]
    fn cached_runs_draw_what_decoded_runs_draw() {
        use crate::vertex_cache::VertexCache;
        // A textured strip in front and one partly behind the near plane.
        let mut memory = TestMemory::default();
        let mut ge = ge_for_3d();
        ge.set_register(cmd(op::TEXTURE_MAP_ENABLE, 1));
        let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.5], [0.0, 1.0, 0.0], [1.0, 1.0, -0.5], [0.0, 2.0, 0.0], [1.0, 2.0, -3.0], [0.0, 3.0, 0.0]];
        positions_3d(&mut memory, &mut ge, VERTICES, &positions);
        let prims = [cmd(op::PRIM, primitive::TRIANGLE_STRIP << 16 | 4), cmd(op::PRIM, primitive::TRIANGLE_STRIP << 16 | 3)];
        let arena: &'static mut [FrameVertex] = Box::leak(vec![FrameVertex::default(); 256].into_boxed_slice());
        let cache: &'static mut VertexCache = Box::leak(Box::new(VertexCache::new(arena)));
        let mut stream = Box::new(DrawStream::new());
        stream.set_vertex_cache(Some(cache));

        let mut record = |gpu_transform: bool, stream: &mut DrawStream| {
            let mut frame = new_boxed_frame();
            let mut states = DrawStateCache::new();
            states.gpu_transform = gpu_transform;
            let mut ge = ge.clone();
            let frame_pointer: *mut Frame = &mut *frame;
            for prim in prims {
                // Safe: one thread; the catalog is only used here.
                let mut catalog = unsafe { Frame::catalog_raw(frame_pointer) };
                let mut stats = FrameStats::new();
                let queues = &mut Queues { stream: &mut *stream, builder: None };
                record_prims(frame_pointer, &mut catalog, &mut stats, &mut LoadedPalette::new(), &mut states, &ge, &mut memory, &[prim], queues);
                stream.finish(&mut stats);
                ge.advance_after_draw(prim.data() & 0xFFFF);
            }
            if let Some(cache) = stream.vertex_cache_mut() {
                frame.vertex_cache = cache.arena().as_ptr();
                frame.vertex_cache_len = cache.arena().len() as u32;
            }
            frame
        };
        let screen = |frame: &Frame| -> Vec<FrameVertex> { frame.draws().iter().flat_map(|draw| frame.screen_corners(draw)).collect() };

        let cpu = record(false, &mut DrawStream::new());
        let first = record(true, &mut stream);
        let stats = stream.vertex_cache_mut().unwrap().next_frame();
        assert_eq!((stats.hits, stats.stored), (0, 1), "the strip in front is stored");
        let second = record(true, &mut stream);
        let stats = stream.vertex_cache_mut().unwrap().next_frame();
        assert_eq!((stats.hits, stats.stored), (1, 0), "and found again");

        assert_eq!(screen(&first), screen(&cpu));
        assert_eq!(screen(&second), screen(&cpu));
        let flags: Vec<bool> = second.draws().iter().map(|draw| draw.flags & draw_flags::CACHED_VERTICES != 0).collect();
        assert_eq!(flags, [true, false]);
        assert_eq!(second.vertices().len(), cpu.vertices().len() - 4, "the cached strip's vertices aren't in the frame");
    }

    #[test]
    fn vertex_reader_matches_reading_one_by_one() {
        let mut memory = TestMemory::default();
        // Small values in every word so floats compare equal (no NaNs). 400
        // words, then unreadable memory: late vertices of long draws are
        // only partly readable.
        let words: Vec<u32> = (0..400u32).map(|i| i.wrapping_mul(0x0103_0507) & 0x3F7F_7F7F).collect();
        memory.write(VERTICES, &words);
        let mut compared = 0;
        for texcoord in [0, 2, 3] {
            for color in [0, 5, 7] {
                for position in 1..4 {
                    for weights in [0, (3 << 9) | (7 << 14)] {
                        let mut ge = GeState::new();
                        let format = (1 << 23) | texcoord | (color << 2) | (position << 7) | weights;
                        ge.set_register(cmd(op::VERTEX_TYPE, format));
                        for skew in 0..4 {
                            ge.vertex_address = VERTICES + skew;
                            for count in [1, 2, 3, 7, 40] {
                                let mut reader = VertexReader::new(ge.vertex_format(), ge.vertex_layout(), &mut memory, ge.vertex_address, ge.index_address);
                                let windowed: Vec<_> = (0..count).map(|n| reader.get(n)).collect();
                                for (n, vertex) in windowed.into_iter().enumerate() {
                                    let address = ge.vertex_address + n as u32 * ge.vertex_layout().size;
                                    let alone = ge.vertex_format().read(&mut memory, address, ge.vertex_layout());
                                    assert_eq!(vertex, alone, "format {format:#x} skew {skew} count {count} vertex {n}");
                                    compared += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(compared, 3 * 3 * 3 * 2 * 4 * (1 + 2 + 3 + 7 + 40));
    }

    /// Blocks from a list of words starting at `VERTICES`.
    struct WordsSource(Vec<u32>);

    impl BlockSource for WordsSource {
        fn copy_block(&mut self, address: u32, out: &mut [u32; BLOCK_WORDS]) -> bool {
            let first = (address.wrapping_sub(VERTICES) / 4) as usize;
            let Some(words) = self.0.get(first..first + BLOCK_WORDS) else {
                return false;
            };
            out.copy_from_slice(words);
            true
        }
    }

    #[test]
    fn vertices_borrowed_from_the_block_cache_match_reading_one_by_one() {
        let words: Vec<u32> = (0..2 * BLOCK_WORDS as u32).map(|i| i.wrapping_mul(0x0103_0507) & 0x3F7F_7F7F).collect();
        let mut reference = TestMemory::default();
        reference.write(VERTICES, &words);
        let mut store = crate::block_cache::BlockStore::<{ crate::block_cache::SMALL_BLOCKS }>::new();
        let mut cache = BlockCache::new(&mut store, WordsSource(words));
        let mut compared = 0;
        for format in [SPRITE_FORMAT, (1 << 23) | 3 | (5 << 2) | (3 << 7), (1 << 23) | (3 << 7) | (3 << 9) | (7 << 14)] {
            let mut ge = GeState::new();
            ge.set_register(cmd(op::VERTEX_TYPE, format));
            // Around the block boundary: vertices inside the first block,
            // across the boundary (not lendable) and inside the second.
            for skew in 0..4 {
                ge.vertex_address = VERTICES + BLOCK_SIZE - 3 * ge.vertex_layout().size + skew;
                for n in 0..6 {
                    let borrowed = VertexReader::new(ge.vertex_format(), ge.vertex_layout(), &mut cache, ge.vertex_address, ge.index_address).get(n);
                    let alone = VertexReader::new(ge.vertex_format(), ge.vertex_layout(), &mut reference, ge.vertex_address, ge.index_address).get(n);
                    assert_eq!(borrowed, alone, "format {format:#x} skew {skew} vertex {n}");
                    assert!(borrowed.is_some() || skew % 4 != 0);
                    compared += 1;
                }
            }
        }
        assert_eq!(compared, 3 * 4 * 6);
    }

    #[test]
    fn a_3d_strip_stores_each_vertex_once() {
        let mut memory = TestMemory::default();
        let mut ge = ge_for_3d();
        positions_3d(&mut memory, &mut ge, VERTICES, &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]);
        let mut frame = new_boxed_frame();
        let mut stats = FrameStats::new();
        record_prim_data(&mut frame, &mut stats, &mut LoadedPalette::new(), &mut DrawStateCache::new(), &ge, &mut memory, (primitive::TRIANGLE_STRIP << 16) | 4);

        assert_eq!(frame.vertices().len(), 4);
        // Stored as one index run (strip, 4 vertices, from 0), drawn as two triangles.
        assert_eq!(frame.indices(), &[primitive::TRIANGLE_STRIP as u16, 4, 0]);
        // (The second triangle's first two corners swapped: one winding, `strip_triangle`.)
        assert_eq!(frame.draw_indices(&frame.draws()[0]).collect::<Vec<_>>(), [0, 1, 2, 2, 1, 3]);
        let corners: Vec<(f32, f32)> = frame.draw_corners(&frame.draws()[0]).map(|vertex| (vertex.x, vertex.y)).collect();
        assert_eq!(corners, [(100.0, 100.0), (110.0, 100.0), (100.0, 110.0), (100.0, 110.0), (110.0, 100.0), (110.0, 110.0)]);

        // A second strip with the same state joins the draw; its indices
        // count on from the first strip's vertices.
        record_prim_data(&mut frame, &mut stats, &mut LoadedPalette::new(), &mut DrawStateCache::new(), &ge, &mut memory, (primitive::TRIANGLE_STRIP << 16) | 4);
        assert_eq!(frame.draws().len(), 1);
        assert_eq!(&frame.indices()[3..], &[primitive::TRIANGLE_STRIP as u16, 4, 4], "the run's first vertex moved on");
        assert_eq!(frame.draw_indices(&frame.draws()[0]).collect::<Vec<_>>(), [0, 1, 2, 2, 1, 3, 4, 5, 6, 6, 5, 7]);
    }

    #[test]
    fn a_batch_of_prims_gives_what_they_give_one_by_one() {
        // A strip of 5, a list of 4 (one triangle, one vertex left over), a
        // strip of 2 (no triangle) and a fan of 4, one after the other.
        let mut memory = TestMemory::default();
        let mut ge = ge_for_3d();
        let positions: Vec<[f32; 3]> = (0..15).map(|i| [i as f32 * 0.25, (i % 3) as f32 * 0.5, 0.0]).collect();
        positions_3d(&mut memory, &mut ge, VERTICES, &positions);
        let prims = [
            cmd(op::PRIM, primitive::TRIANGLE_STRIP << 16 | 5),
            cmd(op::PRIM, primitive::TRIANGLES << 16 | 4),
            cmd(op::PRIM, primitive::TRIANGLE_STRIP << 16 | 2),
            cmd(op::PRIM, primitive::TRIANGLE_FAN << 16 | 4),
        ];
        let corners = |frame: &Frame| -> Vec<(f32, f32)> {
            frame.draws().iter().flat_map(|draw| frame.draw_corners(draw).map(|v| (v.x, v.y)).collect::<Vec<_>>()).collect()
        };

        let mut batched = new_boxed_frame();
        let mut stats = FrameStats::new();
        let mut states = DrawStateCache::new();
        record_now(&mut batched, &mut stats, &mut LoadedPalette::new(), &mut states, &ge, &mut memory, &prims);

        let mut alone = new_boxed_frame();
        let mut states = DrawStateCache::new();
        let mut one_by_one = ge.clone();
        for prim in prims {
            record_now(&mut alone, &mut FrameStats::new(), &mut LoadedPalette::new(), &mut states, &one_by_one, &mut memory, &[prim]);
            one_by_one.advance_after_draw(prim.data() & 0xFFFF);
        }

        assert_eq!(corners(&batched), corners(&alone));
        assert_eq!((3 + 1 + 0 + 2) * 3, batched.corner_count(), "all triangles");
        assert_eq!(batched.draws().len(), 1);
    }

    #[test]
    fn a_consumer_thread_records_what_the_producer_alone_records() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        // 2,000 strips of 3..8 vertices from one buffer, one PRIM each (a
        // job each): the rings wrap several times.
        let mut memory = TestMemory::default();
        let mut ge = ge_for_3d();
        let positions: Vec<[f32; 3]> = (0..20_000).map(|i| [(i % 97) as f32 * 0.01, (i % 89) as f32 * 0.01, 0.0]).collect();
        positions_3d(&mut memory, &mut ge, VERTICES, &positions);
        let prims: Vec<Command> = (0..2000).map(|i| cmd(op::PRIM, primitive::TRIANGLE_STRIP << 16 | (3 + i % 6))).collect();
        let corners = |frame: &Frame| -> Vec<(u32, u32)> {
            frame.draws().iter().flat_map(|draw| frame.draw_corners(draw).map(|v| (v.x.to_bits(), v.y.to_bits())).collect::<Vec<_>>()).collect()
        };
        let mut record = |frame: &mut Frame, stream: &mut DrawStream| {
            let mut ge = ge.clone();
            let mut states = DrawStateCache::new();
            let mut palette = LoadedPalette::new();
            let mut stats = FrameStats::new();
            let frame: *mut Frame = frame;
            for &prim in &prims {
                // Safe: the catalog is only used by this (the producer's) thread.
                let mut catalog = unsafe { Frame::catalog_raw(frame) };
                let queues = &mut Queues { stream: &mut *stream, builder: None };
                record_prims(frame, &mut catalog, &mut stats, &mut palette, &mut states, &ge, &mut memory, &[prim], queues);
                ge.advance_after_draw(prim.data() & 0xFFFF);
            }
            stream.finish(&mut stats);
        };

        let mut alone = new_boxed_frame();
        record(&mut alone, &mut DrawStream::new());

        static HOOKS: StreamHooks = StreamHooks { wake_consumer: || {}, wait: std::thread::yield_now };
        let stream: &'static mut DrawStream = Box::leak(Box::new(DrawStream::new()));
        stream.set_hooks(Some(&HOOKS));
        let shared = stream as *const DrawStream as usize;
        let done = Arc::new(AtomicBool::new(false));
        let consumer = {
            let done = done.clone();
            std::thread::spawn(move || {
                // Safe: `DrawStream` is Sync; the stream lives for ever.
                let stream = unsafe { &*(shared as *const DrawStream) };
                while !done.load(Ordering::Acquire) {
                    stream.record_published();
                    std::thread::yield_now();
                }
            })
        };
        let mut threaded = new_boxed_frame();
        record(&mut threaded, stream);
        done.store(true, Ordering::Release);
        consumer.join().unwrap();

        assert!(alone.corner_count() > 10_000);
        assert_eq!(corners(&threaded), corners(&alone));
        assert_eq!(threaded.draws().len(), alone.draws().len());
    }

    /// Test memory whose main RAM the copier reads (`PspMemory::defer`),
    /// through `copier_test_read` (a plain function, as on the Vita).
    struct DeferrableTestMemory(TestMemory);

    impl PspMemory for DeferrableTestMemory {
        fn read_u32(&mut self, address: u32) -> Option<u32> {
            self.0.read_u32(address)
        }
        fn defer(&mut self, address: u32, _bytes: u32) -> bool {
            address >= crate::memory::RAM_START
        }
    }

    static COPIER_TEST_MEMORY: std::sync::Mutex<Option<TestMemory>> = std::sync::Mutex::new(None);

    static COPIER_TEST_READS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    fn copier_test_read(address: u32, out: &mut [u32]) -> bool {
        COPIER_TEST_READS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut memory = COPIER_TEST_MEMORY.lock().unwrap();
        let memory = memory.as_mut().unwrap();
        for (index, word) in out.iter_mut().enumerate() {
            match memory.read_u32(address + 4 * index as u32) {
                Some(value) => *word = value,
                None => return false,
            }
        }
        true
    }

    fn copier_test_sample(address: u32, positions: &[usize], out: &mut [u32]) -> bool {
        positions.iter().zip(out.iter_mut()).all(|(&position, word)| copier_test_read(address + 4 * position as u32, core::slice::from_mut(word)))
    }

    #[test]
    fn a_copier_thread_and_a_consumer_thread_record_what_the_producer_alone_records() {
        use crate::draw_stream::{CopierHooks, DeferredReader};
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        // As above: 2,000 strips, a job each, the rings wrap several times.
        let mut memory = TestMemory::default();
        let mut ge = ge_for_3d();
        let positions: Vec<[f32; 3]> = (0..20_000).map(|i| [(i % 97) as f32 * 0.01, (i % 89) as f32 * 0.01, 0.0]).collect();
        positions_3d(&mut memory, &mut ge, VERTICES, &positions);
        *COPIER_TEST_MEMORY.lock().unwrap() = Some(memory.clone());
        let prims: Vec<Command> = (0..2000).map(|i| cmd(op::PRIM, primitive::TRIANGLE_STRIP << 16 | (3 + i % 6))).collect();
        let corners = |frame: &Frame| -> Vec<(u32, u32)> {
            frame.draws().iter().flat_map(|draw| frame.draw_corners(draw).map(|v| (v.x.to_bits(), v.y.to_bits())).collect::<Vec<_>>()).collect()
        };
        let mut memory = DeferrableTestMemory(memory);
        let mut record = |frame: &mut Frame, stream: &mut DrawStream| {
            let mut ge = ge.clone();
            let mut states = DrawStateCache::new();
            let mut palette = LoadedPalette::new();
            let mut stats = FrameStats::new();
            let frame: *mut Frame = frame;
            for &prim in &prims {
                // Safe: the catalog is only used by this (the producer's) thread.
                let mut catalog = unsafe { Frame::catalog_raw(frame) };
                let queues = &mut Queues { stream: &mut *stream, builder: None };
                record_prims(frame, &mut catalog, &mut stats, &mut palette, &mut states, &ge, &mut memory, &[prim], queues);
                ge.advance_after_draw(prim.data() & 0xFFFF);
            }
            stream.finish(&mut stats);
        };

        let mut alone = new_boxed_frame();
        record(&mut alone, &mut DrawStream::new());

        const READER: DeferredReader = DeferredReader { read: copier_test_read, sample: copier_test_sample, copy_bytes: |_, _| false };
        static HOOKS: StreamHooks = StreamHooks { wake_consumer: || {}, wait: std::thread::yield_now };
        static COPIER: CopierHooks = CopierHooks { wake: || {} };

        // Single thread first: no copier thread takes the jobs, so the
        // producer copies them, as for a copier that fell behind
        // (`COPIER_LAG_JOBS`), and before it drains.
        let mut drained = new_boxed_frame();
        let mut stream = Box::new(DrawStream::new());
        stream.set_reader(Some(READER));
        stream.set_copier(Some(&COPIER));
        let helped_before = crate::draw_stream::PRODUCER_COPIES.load(Ordering::Relaxed);
        record(&mut drained, &mut stream);
        assert_eq!(corners(&drained), corners(&alone));
        assert!(crate::draw_stream::PRODUCER_COPIES.load(Ordering::Relaxed) > helped_before, "the producer helped");
        assert!(COPIER_TEST_READS.load(Ordering::Relaxed) >= 2000, "every job's vertices went through the reader");

        let stream: &'static mut DrawStream = Box::leak(Box::new(DrawStream::new()));
        stream.set_hooks(Some(&HOOKS));
        stream.set_reader(Some(READER));
        stream.set_copier(Some(&COPIER));
        let shared = stream as *const DrawStream as usize;
        let done = Arc::new(AtomicBool::new(false));
        let spawn = |copier: bool| {
            let done = done.clone();
            std::thread::spawn(move || {
                // Safe: `DrawStream` is Sync; the stream lives for ever.
                let stream = unsafe { &*(shared as *const DrawStream) };
                while !done.load(Ordering::Acquire) {
                    if copier {
                        stream.copy_published();
                    } else {
                        stream.record_published();
                    }
                    std::thread::yield_now();
                }
            })
        };
        let threads = [spawn(true), spawn(false)];
        let mut threaded = new_boxed_frame();
        let reads_before = COPIER_TEST_READS.load(Ordering::Relaxed);
        record(&mut threaded, stream);
        assert!(stream.all_copied() && stream.is_idle());
        assert!(COPIER_TEST_READS.load(Ordering::Relaxed) - reads_before >= 2000, "the copier thread read them");
        done.store(true, Ordering::Release);
        for thread in threads {
            thread.join().unwrap();
        }

        assert!(alone.corner_count() > 10_000);
        assert_eq!(corners(&threaded), corners(&alone));
        assert_eq!(threaded.draws().len(), alone.draws().len());
    }

    static SAMPLE_TEST_MEMORY: std::sync::Mutex<Option<TestMemory>> = std::sync::Mutex::new(None);
    static SAMPLE_TEST_READS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    fn sample_test_bytes(address: u32, out: &mut [u8]) -> bool {
        SAMPLE_TEST_READS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut memory = SAMPLE_TEST_MEMORY.lock().unwrap();
        let memory = memory.as_mut().unwrap();
        out.iter_mut().enumerate().all(|(n, byte)| {
            let at = address + n as u32;
            memory.read_u32(at & !3).map(|word| *byte = (word >> (8 * (at & 3))) as u8).is_some()
        })
    }

    #[test]
    fn a_texture_sampled_by_the_copier_gets_the_sample_the_capture_thread_takes() {
        use crate::draw_stream::{CopierHooks, DeferredReader};
        use std::sync::atomic::Ordering;
        let mut memory = TestMemory::default();
        vertex(&mut memory, VERTICES, 0, 0, 0, 0, 0xFFFF_FFFF);
        vertex(&mut memory, VERTICES + 16, 32, 16, 32, 16, 0xFFFF_FFFF);
        // A 64 x 32 32-bit texture (2,048 words), every word different.
        let words: Vec<u32> = (0..2048u32).map(|n| n.wrapping_mul(0x9E37_79B9) | 1).collect();
        memory.write(0x0895_0000, &words);
        *SAMPLE_TEST_MEMORY.lock().unwrap() = Some(memory.clone());
        let mut ge = ge_for_sprites();
        ge.set_register(cmd(op::TEXTURE_MAP_ENABLE, 1));
        ge.set_register(cmd(op::TEXTURE_FORMAT, 3));
        ge.set_register(cmd(op::TEXTURE_SIZE_0, 6 | (5 << 8)));
        ge.set_register(cmd(op::TEXTURE_ADDRESS_0, 0x95_0000));
        ge.set_register(cmd(op::TEXTURE_BUFFER_WIDTH_0, 0x08_0040));
        let prim = cmd(op::PRIM, (primitive::RECTANGLES << 16) | 2);
        let mut memory = DeferrableTestMemory(memory);
        // Two draws with the texture: sampled at the first only.
        let mut record = |frame: &mut Frame, stream: &mut DrawStream| {
            let (mut states, mut palette, mut stats) = (DrawStateCache::new(), LoadedPalette::new(), FrameStats::new());
            let frame: *mut Frame = frame;
            for _ in 0..2 {
                // Safe: the catalog is only used by this (the producer's) thread.
                let mut catalog = unsafe { Frame::catalog_raw(frame) };
                let queues = &mut Queues { stream: &mut *stream, builder: None };
                record_prims(frame, &mut catalog, &mut stats, &mut palette, &mut states, &ge, &mut memory, &[prim], queues);
            }
            stream.finish(&mut stats);
        };

        let mut alone = new_boxed_frame();
        record(&mut alone, &mut DrawStream::new());
        let positions = crate::texture::quick_sample_positions_in(words.len());
        assert_eq!(alone.texture_sample(0), crate::texture::quick_sample(&positions.map(|position| words[position])));

        // The jobs' memory (the vertices) from this test's memory too.
        fn read(address: u32, out: &mut [u32]) -> bool {
            let mut memory = SAMPLE_TEST_MEMORY.lock().unwrap();
            let memory = memory.as_mut().unwrap();
            out.iter_mut().enumerate().all(|(n, word)| memory.read_u32(address + 4 * n as u32).map(|value| *word = value).is_some())
        }
        // The texture's sample words counted (the jobs sample their vertices
        // with it too).
        fn sample(address: u32, positions: &[usize], out: &mut [u32]) -> bool {
            if address == 0x0895_0000 {
                SAMPLE_TEST_READS.fetch_add(positions.len(), std::sync::atomic::Ordering::Relaxed);
            }
            positions.iter().zip(out.iter_mut()).all(|(&position, word)| read(address + 4 * position as u32, core::slice::from_mut(word)))
        }
        const READER: DeferredReader = DeferredReader { read, sample, copy_bytes: sample_test_bytes };
        static COPIER: CopierHooks = CopierHooks { wake: || {} };
        let mut stream = Box::new(DrawStream::new());
        stream.set_reader(Some(READER));
        stream.set_copier(Some(&COPIER));
        let mut tasked = new_boxed_frame();
        record(&mut tasked, &mut stream);
        assert_eq!(tasked.texture_sample(0), alone.texture_sample(0));
        assert_eq!(SAMPLE_TEST_READS.load(Ordering::Relaxed), 8, "the copier's reader sampled the 8 words, once");
    }

    #[test]
    fn strip_becomes_triangles() {
        let mut memory = TestMemory::default();
        for i in 0..4 {
            vertex(&mut memory, VERTICES + 16 * i, i as i16, 0, 0, 0, 0xFFFF_FFFF);
        }
        let ge = ge_for_sprites();
        let mut frame = new_boxed_frame();
        let mut stats = FrameStats::new();
        record_prim_data(&mut frame, &mut stats, &mut LoadedPalette::new(), &mut DrawStateCache::new(), &ge, &mut memory, (primitive::TRIANGLE_STRIP << 16) | 4);
        let xs: Vec<f32> = frame.vertices().iter().map(|v| v.x).collect();
        // The second triangle's first two corners swapped (`strip_triangle`).
        assert_eq!(xs, vec![0.0, 1.0, 2.0, 2.0, 1.0, 3.0]);
    }

    /// Memory that lends its words (as a job's copied memory does).
    struct Lending {
        base: u32,
        words: Vec<u32>,
    }

    impl PspMemory for Lending {
        fn read_u32(&mut self, address: u32) -> Option<u32> {
            self.words.get((address.checked_sub(self.base)? / 4) as usize).copied()
        }
        fn borrow_words(&mut self, address: u32, count: usize) -> Option<&[u32]> {
            let first = (address.checked_sub(self.base)? / 4) as usize;
            self.words.get(first..first + count)
        }
    }

    /// Monster Hunter Freedom draws everything indexed (one triangle per
    /// PRIM): the CPU path took ~150 ms per village frame on the Vita.
    /// Indexed PRIMs go raw: the arena holds the vertices the indices name,
    /// in index order, one index run per PRIM; the one 16-bit weight every
    /// vertex has is skinned by the program (its bone in the bone table).
    #[test]
    fn indexed_prims_gather_their_vertices_into_raw_runs() {
        let ge = ge_for_3d();
        // One 16-bit weight, float positions, 8-bit indices: 16-byte vertices.
        let format = VertexFormat((2 << 9) | (3 << 7) | (1 << 11));
        let layout = format.layout();
        assert_eq!(layout.size, 16);
        let inputs = crate::transform::TransformInputs::of(&ge, format);
        let mut words = Vec::new();
        for i in 0..10u32 {
            words.extend([0x8000, (i as f32).to_bits(), 0f32.to_bits(), 0f32.to_bits()]); // weight 1.0
        }
        words.extend([u32::from_le_bytes([7, 8, 9, 2]), u32::from_le_bytes([3, 4, 0, 0])]);
        let mut memory = Lending { base: VERTICES, words };
        let indices = VERTICES + 160;
        let strip = |index_address| PrimCall { primitive_type: primitive::TRIANGLE_STRIP, count: 3, vertex_address: VERTICES, index_address };
        let calls = [strip(indices), strip(indices + 3)];
        let colors = raw_color_fields(&Prepared { gpu_transform: true, ..Prepared::EMPTY }, 0xFFFF_FFFF, inputs.lighting(), format).expect("raw");
        let mut raw = crate::raw_arena::RawArena::new(Box::leak(vec![0u8; crate::raw_arena::WINDOW_BYTES as usize].into_boxed_slice()));
        let mut frame = new_boxed_frame();
        let first_index = frame.indices().len();
        let run = {
            let (mut geometry, _) = frame.split();
            record_raw_gathered(&mut geometry, &mut memory, format, &layout, &inputs, (&calls, None), &colors, &mut raw).expect("gathered")
        };
        assert_ne!(run.bones, 0, "one weight: skinned by the program (its bone in the table)");
        let RawPlace::Arena(window) = run.window else { panic!("in the arena") };
        let runs = &frame.indices()[first_index..];
        assert_eq!(runs.len(), 6);
        assert_eq!((runs[0], runs[1], runs[4]), (primitive::TRIANGLE_STRIP as u16, 3, 3));
        assert_eq!(runs[5], runs[2] + 3);
        let x_of = |vertex: u16| {
            let at = window as usize + 16 * vertex as usize + 4;
            f32::from_le_bytes(raw.bytes()[at..at + 4].try_into().unwrap())
        };
        let xs: Vec<f32> = (0..6).map(|n| x_of(runs[2] + n)).collect();
        assert_eq!(xs, vec![7.0, 8.0, 9.0, 2.0, 3.0, 4.0]);
    }

    /// Indexed rectangles were taken by the raw sprite path, which copies
    /// each sprite's vertex pair from the vertices in a row (found reading
    /// the code, 2026-11-16; no dump has indexed sprites): they must be left
    /// to the CPU path, which reads them by index.
    #[test]
    fn indexed_rectangles_are_not_raw_sprites() {
        let plain = VertexFormat((3 << 7) | (2 << 2));
        assert!(sprites_suit_raw(plain, 4));
        for index_bits in [1 << 11, 2 << 11, 3 << 11] {
            assert!(!sprites_suit_raw(VertexFormat(plain.0 | index_bits), 4), "index format {}", index_bits >> 11);
        }
        assert!(!sprites_suit_raw(plain, 0), "no whole sprite");
    }

    /// Indexed PRIMs as `record_raw_converted` and `record_raw_gathered`
    /// record them (a list, a strip and a fan using different parts of one
    /// buffer, for each index size): the corners name the same vertices in
    /// the same order. Float positions, x = the vertex's number.
    /// `convert_indices` as it was until 2026-11-19 (corner by corner): the
    /// reference the per-PRIM version must match.
    fn convert_indices_corner_by_corner<const SIZE: usize>(
        indices: &[u8],
        calls: &[PrimCall],
        (lowest, highest): (u32, u32),
        first: u32,
        out: &mut [u16],
    ) -> bool {
        let index = |position: u32| -> u32 {
            let at = position as usize * SIZE;
            match SIZE {
                1 => indices[at] as u32,
                2 => u16::from_le_bytes([indices[at], indices[at + 1]]) as u32,
                _ => u32::from_le_bytes([indices[at], indices[at + 1], indices[at + 2], indices[at + 3]]),
            }
        };
        let range = highest - lowest;
        let mut written = 0;
        let mut start = 0;
        for call in calls {
            let corner_of = |offset: u32| -> Option<u16> {
                let value = index(start + offset).wrapping_sub(lowest);
                (value <= range).then_some(first.wrapping_add(value) as u16)
            };
            for t in 0..triangle_count(call.primitive_type, call.count) {
                let corners = match call.primitive_type {
                    primitive::TRIANGLES => [3 * t, 3 * t + 1, 3 * t + 2],
                    primitive::TRIANGLE_STRIP => crate::frame::strip_triangle(t),
                    _ => [0, t + 1, t + 2],
                };
                for corner in corners {
                    let Some(number) = corner_of(corner) else {
                        return false;
                    };
                    out[written] = number;
                    written += 1;
                }
            }
            start += call.count;
        }
        true
    }

    /// The per-PRIM conversion writes exactly what the corner-by-corner one
    /// did: random mixes of lists, strips and fans (counts 0..40, short
    /// ones too), every index size, and out-of-range indices refused.
    #[test]
    fn converted_indices_match_the_corner_by_corner_conversion() {
        fn check<const SIZE: usize>(seed: &mut u32) {
            let mut next = |limit: u32| {
                *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (*seed >> 8) % limit
            };
            for round in 0..400 {
                let prims = 1 + next(6);
                let mut calls = Vec::new();
                let mut values = Vec::new();
                for _ in 0..prims {
                    let primitive_type = [primitive::TRIANGLES, primitive::TRIANGLE_STRIP, primitive::TRIANGLE_FAN][next(3) as usize];
                    let count = next(40);
                    calls.push(PrimCall { primitive_type, count, vertex_address: 0, index_address: 0 });
                    let base = 100 + next(50);
                    values.extend((0..count).map(|_| base + next(if SIZE == 1 { 100 } else { 3000 })));
                }
                let (mut lowest, mut highest) = (values.iter().copied().min().unwrap_or(0), values.iter().copied().max().unwrap_or(0));
                if round % 5 == 4 && !values.is_empty() {
                    // Bounds that leave some index out: both must refuse.
                    if round % 2 == 0 { lowest += 1 } else { highest -= 1 }
                    if lowest > highest { continue; }
                }
                let bytes: Vec<u8> = values.iter().flat_map(|value| value.to_le_bytes()[..SIZE].to_vec()).collect();
                let corners: u32 = calls.iter().map(|call| 3 * triangle_count(call.primitive_type, call.count)).sum();
                let first = next(1000);
                let mut wanted = vec![0u16; corners as usize];
                let mut got = vec![0u16; corners as usize];
                let ok_wanted = convert_indices_corner_by_corner::<SIZE>(&bytes, &calls, (lowest, highest), first, &mut wanted);
                let ok_got = convert_indices::<SIZE>(&bytes, &calls, (lowest, highest), first, &mut got);
                assert_eq!(ok_got, ok_wanted, "size {SIZE} round {round}");
                if ok_wanted {
                    assert_eq!(got, wanted, "size {SIZE} round {round}");
                }
            }
        }
        let mut seed = 7;
        check::<1>(&mut seed);
        check::<2>(&mut seed);
        check::<4>(&mut seed);
    }

    /// The unchecked conversion (`convert_indices_in_bounds`, bounds taken
    /// from the indices themselves) writes exactly what the checked one
    /// writes: random mixes of lists, strips and fans (counts 0..40, odd
    /// and even strips, short PRIMs without triangles), every index size,
    /// 4-byte indices above 65,535 too, `first` below and above `lowest`.
    #[test]
    fn in_bounds_conversion_matches_the_checked_one() {
        fn check<const SIZE: usize>(seed: &mut u32) {
            let mut next = |limit: u32| {
                *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (*seed >> 8) % limit
            };
            for round in 0..600 {
                let prims = 1 + next(8);
                let mut calls = Vec::new();
                let mut values = Vec::new();
                let base = match SIZE {
                    1 => next(100),
                    2 => next(60_000),
                    _ => next(200_000),
                };
                for _ in 0..prims {
                    let primitive_type = [primitive::TRIANGLES, primitive::TRIANGLE_STRIP, primitive::TRIANGLE_FAN][next(3) as usize];
                    let count = if next(3) == 0 { next(6) } else { next(40) };
                    calls.push(PrimCall { primitive_type, count, vertex_address: 0, index_address: 0 });
                    values.extend((0..count).map(|_| base + next(if SIZE == 1 { 150 } else { 5000 })));
                }
                if values.is_empty() {
                    continue;
                }
                let (lowest, highest) = (*values.iter().min().unwrap(), *values.iter().max().unwrap());
                let bytes: Vec<u8> = values.iter().flat_map(|value| value.to_le_bytes()[..SIZE].to_vec()).collect();
                let corners: u32 = calls.iter().map(|call| 3 * triangle_count(call.primitive_type, call.count)).sum();
                let first = next(70_000);
                let mut wanted = vec![0u16; corners as usize];
                let mut got = vec![0xAAAAu16; corners as usize];
                assert!(convert_indices::<SIZE>(&bytes, &calls, (lowest, highest), first, &mut wanted), "size {SIZE} round {round}");
                assert!(convert_indices_in_bounds::<SIZE>(&bytes, &calls, lowest, first, &mut got), "size {SIZE} round {round}");
                assert_eq!(got, wanted, "size {SIZE} round {round}");
                // Too little room: refused, not a panic.
                if corners > 0 {
                    let mut short = vec![0u16; corners as usize - 1];
                    assert!(!convert_indices_in_bounds::<SIZE>(&bytes, &calls, lowest, first, &mut short));
                    assert!(!convert_indices_in_bounds::<SIZE>(&bytes[..bytes.len() - SIZE], &calls, lowest, first, &mut got));
                }
            }
        }
        let mut seed = 5;
        check::<1>(&mut seed);
        check::<2>(&mut seed);
        check::<4>(&mut seed);
    }

    #[test]
    fn converted_indices_draw_the_triangles_gathering_draws() {
        let ge = ge_for_3d();
        let values: [u32; 15] = [3, 4, 5, 9, 8, 7, 10, 12, 11, 13, 14, 19, 15, 16, 17];
        // Lists 3t..; strips with every second triangle's first two swapped;
        // fans from their first vertex.
        let wanted: Vec<f32> =
            [3, 4, 5, 9, 8, 7, 10, 12, 11, 11, 12, 13, 11, 13, 14, 19, 15, 16, 19, 16, 17].iter().map(|&x| x as f32).collect();
        for (index_bits, size) in [(1u32, 1u32), (2, 2), (3, 4)] {
            let format = VertexFormat((3 << 7) | (index_bits << 11));
            let layout = format.layout();
            assert_eq!(layout.size, 12);
            let inputs = crate::transform::TransformInputs::of(&ge, format);
            let mut words = Vec::new();
            for i in 0..20u32 {
                words.extend([(i as f32).to_bits(), 0, 0]);
            }
            let mut bytes: Vec<u8> = values.iter().flat_map(|value| value.to_le_bytes()[..size as usize].to_vec()).collect();
            bytes.resize(bytes.len().div_ceil(4) * 4, 0);
            words.extend(bytes.chunks_exact(4).map(|word| u32::from_le_bytes(word.try_into().unwrap())));
            let indices = VERTICES + 20 * 12;
            let call = |primitive_type, count, at: u32| PrimCall { primitive_type, count, vertex_address: VERTICES, index_address: indices + at * size };
            let calls = [call(primitive::TRIANGLES, 6, 0), call(primitive::TRIANGLE_STRIP, 5, 6), call(primitive::TRIANGLE_FAN, 4, 11)];
            let colors = raw_color_fields(&Prepared { gpu_transform: true, ..Prepared::EMPTY }, 0xFFFF_FFFF, inputs.lighting(), format).expect("raw");
            let mut corners_x = Vec::new();
            for convert in [true, false] {
                let mut memory = Lending { base: VERTICES, words: words.clone() };
                let mut raw = crate::raw_arena::RawArena::new(Box::leak(vec![0u8; crate::raw_arena::WINDOW_BYTES as usize].into_boxed_slice()));
                raw.put(&[0; 36], 12).unwrap(); // not at the window's start
                let mut frame = new_boxed_frame();
                let run = {
                    let (mut geometry, _) = frame.split();
                    if convert {
                        record_raw_converted(&mut geometry, &mut memory, format, &layout, &inputs, &calls, &colors, &mut raw, (3, 19), false)
                    } else {
                        record_raw_gathered(&mut geometry, &mut memory, format, &layout, &inputs, (&calls, Some((3, 19))), &colors, &mut raw)
                    }
                    .expect("recorded")
                };
                assert_eq!(run.listed, convert);
                let RawPlace::Arena(window) = run.window else { panic!("in the arena") };
                let numbers: Vec<u16> =
                    if convert { frame.indices().to_vec() } else { frame.indices().chunks_exact(3).flat_map(crate::frame::run_indices).collect() };
                let x_of = |vertex: u16| {
                    let at = window as usize + 12 * vertex as usize;
                    f32::from_le_bytes(raw.bytes()[at..at + 4].try_into().unwrap())
                };
                corners_x.push(numbers.into_iter().map(x_of).collect::<Vec<f32>>());
            }
            assert_eq!(corners_x[0], wanted, "converted, {size}-byte indices");
            assert_eq!(corners_x[1], wanted, "gathered, {size}-byte indices");
        }
    }

    /// The producer found the bounds; the copier copied the indices later.
    /// An index outside the bounds (the game rewrote the list in between)
    /// must turn the conversion down, not index past the copied span.
    #[test]
    fn an_index_outside_the_producers_bounds_turns_the_conversion_down() {
        let ge = ge_for_3d();
        let format = VertexFormat((3 << 7) | (1 << 11));
        let layout = format.layout();
        let inputs = crate::transform::TransformInputs::of(&ge, format);
        let mut words = Vec::new();
        for i in 0..8u32 {
            words.extend([(i as f32).to_bits(), 0, 0]);
        }
        words.push(u32::from_le_bytes([2, 3, 7, 0])); // 7: past the bounds below
        let calls = [PrimCall { primitive_type: primitive::TRIANGLES, count: 3, vertex_address: VERTICES, index_address: VERTICES + 8 * 12 }];
        let colors = raw_color_fields(&Prepared { gpu_transform: true, ..Prepared::EMPTY }, 0xFFFF_FFFF, inputs.lighting(), format).expect("raw");
        let mut raw = crate::raw_arena::RawArena::new(Box::leak(vec![0u8; crate::raw_arena::WINDOW_BYTES as usize].into_boxed_slice()));
        let mut frame = new_boxed_frame();
        let (mut geometry, _) = frame.split();
        let mut memory = Lending { base: VERTICES, words: words.clone() };
        assert!(record_raw_converted(&mut geometry, &mut memory, format, &layout, &inputs, &calls, &colors, &mut raw, (2, 5), false).is_none());
        let mut memory = Lending { base: VERTICES, words };
        assert!(record_raw_gathered(&mut geometry, &mut memory, format, &layout, &inputs, (&calls, Some((2, 5))), &colors, &mut raw).is_none());
    }

    /// MTX's puddle squares (user's savestate 8, 2026-11-10): the water is
    /// drawn with blend factor FIX A over raw ground, with an equal-depth
    /// GEQUAL. FIX A kept the water off the raw path, the two paths' depths
    /// differed by a few ulps, and the GPU flipped the test per 32x32 tile.
    /// A FIX A draw must go raw, its scale in the colour rules.
    #[test]
    fn fix_a_draws_go_raw_like_the_ground_they_cover() {
        let ge = ge_for_3d();
        let format = VertexFormat(3 << 7);
        let inputs = crate::transform::TransformInputs::of(&ge, format);
        let water = Prepared { gpu_transform: true, blending: true, source_scale: Some(0x00C8_C8C8), ..Prepared::EMPTY };
        assert!(may_go_raw(&water, None, format), "FIX A no longer keeps a draw off the raw path");
        let colors = raw_color_fields(&water, 0xFFFF_FFFF, inputs.lighting(), format).expect("FIX A goes raw");
        assert_eq!(colors.scale, [200.0, 200.0, 200.0, 1.0]);
        assert_eq!(colors.color(Some(0x80FF_FFFF)), crate::blend::scale_color(0x80FF_FFFF, 0x00C8_C8C8));
        let ground = Prepared { source_scale: None, ..water };
        assert_eq!(raw_color_fields(&ground, 0xFFFF_FFFF, inputs.lighting(), format).expect("raw").scale, [0.0; 4]);
    }
}
