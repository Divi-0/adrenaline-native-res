//! The capture thread writes raw draws itself (performance refactor step 2,
//! docs/performance-refactor-plan.md): a draw the GPU can take from the
//! PSP's own vertices never enters the draw stream's job queue.
//!
//! Before, every draw was built twice: the capture thread worked out its
//! state and queued a job (with its memory copied into the word ring by the
//! copier), and the recorder thread built the draw again from the job,
//! copying the words a second time, into the raw arena. For a raw draw the
//! capture thread already knows everything: the state, where the vertices
//! are and how many there are, the transform inputs. So it writes the final
//! `Draw` (with its table entries, `raw_blocks`), reserves the vertices'
//! room in the raw arena and leaves the copy to the copier (`copy_tasks`):
//! PSP memory straight into the arena, nothing in between. Indexed draws'
//! indices are read once, here (the capture thread had to read them for the
//! vertex range anyway) and converted in the same pass
//! (`recorder::convert_indices`).
//!
//! Draws the GPU can't take (2D, untextured 3D, ... `recorder::fallback`)
//! still go through the job queue to the recorder thread, unchanged. Both
//! kinds must be drawn in list order, so they are written into two lists:
//! the recorder's into the frame as before, the capture thread's into lists
//! of its own (`DirectLists`), and an order list notes how they alternate:
//! "direct draws a..b", "the draws of fallback segment s" (where they start
//! in the frame the recorder notes as it starts the segment's first job,
//! `draw_stream::QueuedDraw::segment`). When the frame is finished (the
//! recorder idle, every copy done) `finish_frame` moves both into the frame
//! in that order. The frame then looks as it always did: the display, the
//! replay and the dumps read it unchanged.
//!
//! Who owns what while a frame is recorded this way (`begin_frame`): the
//! raw arena and the index ring (taken out of the vertex cache), and the
//! frame's raw tables (`Frame::raw_tables_raw`): the capture thread. The
//! recorder thread records everything it is given on the CPU path.

use core::sync::atomic::{AtomicBool, Ordering};

use crate::command::primitive;
use crate::copy_tasks::{CopyTask, Layout};
use crate::draw_stream::{DrawStream, MAX_SEGMENTS};
use crate::frame::{draw_flags, Draw, Frame, Geometry, ListCounts, RawColors, MAX_DRAWS, MAX_INDICES};
use crate::memory::PspMemory;
use crate::raw_arena::RawArena;
use crate::recorder::{fallback, PrimCall, Prepared};
use crate::transform::TransformInputs;
use crate::vertex::{IndexFormat, VertexFormat, VertexLayout};
use crate::vertex_cache::VertexCache;

/// Whether the capture thread writes raw draws itself (off: everything goes
/// through the job queue, as before; the replay's `--jobs`). Read when a
/// frame starts.
pub static ENABLED: AtomicBool = AtomicBool::new(true);

/// Words of indices one indexed draw may have to be written directly (MHF:
/// batches of up to 32 one-triangle PRIMs; a longer one goes through the
/// job queue). With `GATHER_ENTRIES` sized so the lists fit one MiB.
pub const INDEX_WORDS: usize = 16 * 1024;

/// Vertex numbers of gathered spans (`Layout::Gather`) per frame. The MHF
/// intro gathers ~20 sparse spans per frame.
pub const GATHER_ENTRIES: usize = 16 * 1024;

/// The order list's entries, at most: fallback segments with the direct
/// stretches between them.
const MAX_ORDER: usize = 2 * MAX_SEGMENTS + 1;

/// VERTEX_TYPE's index format bits (11-12): raw draws' vertices are read as
/// plain vertices (their indices converted).
const INDEX_FORMAT_BITS: u32 = 3 << 11;

/// Runs `work` in the profile phase `phase` (one of `DIRECT`'s parts), then
/// goes back to the phase before.
#[inline(always)]
fn in_phase<T>(phase: u8, work: impl FnOnce() -> T) -> T {
    let previous = crate::phase::enter(phase);
    let result = work();
    crate::phase::leave(previous);
    result
}

/// Runs `work` counted as cycle part `part` (`cycles`, when measured).
#[inline(always)]
fn timed<T>(part: usize, work: impl FnOnce() -> T) -> T {
    let started = crate::cycles::start(part);
    let result = work();
    crate::cycles::add(part, started);
    result
}

/// One stretch of the frame's draws in list order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stretch {
    /// The capture thread's draws `first..end` (`DirectLists::draws`).
    Direct { first: u32, end: u32 },
    /// The draws the recorder wrote for fallback segment `segment`.
    Fallback { segment: u32 },
}

/// The capture thread's draws of the frame being recorded, their indices,
/// the order list, and room to read an indexed draw's indices. Big (~1 MiB):
/// made once outside the capture (phycont memory on the Vita).
pub struct DirectLists {
    draws: [Draw; MAX_DRAWS],
    /// (Index runs, converted indices: as `Frame::indices`.)
    indices: [u16; MAX_INDICES],
    order: [Stretch; MAX_ORDER],
    index_words: [u32; INDEX_WORDS],
    /// The gather tasks' vertex lists (`Layout::Gather`), kept until the
    /// frame's end (every task is done then).
    gather: [u16; GATHER_ENTRIES],
}

// One MiB of phycont memory on the Vita (its blocks come in whole MiB).
const _: () = assert!(core::mem::size_of::<DirectLists>() <= 1024 * 1024);

impl DirectLists {
    /// Makes the lists in zeroed memory at `place` (their contents are
    /// written before they are read).
    ///
    /// # Safety
    /// `place` must point to `size_of::<DirectLists>()` zeroed, writable,
    /// aligned bytes, used by nobody else.
    pub unsafe fn init_zeroed(_place: *mut DirectLists) {}

    /// The lists on the heap (tests; the replay makes its own the same way).
    #[cfg(test)]
    pub fn boxed() -> &'static mut DirectLists {
        let layout = std::alloc::Layout::new::<DirectLists>();
        // Safe: all-zero bytes are valid lists (integers, and `Stretch`
        // zero is `Direct { 0, 0 }`); the layout is non-zero-sized.
        unsafe { &mut *(std::alloc::alloc_zeroed(layout) as *mut DirectLists) }
    }
}

/// The previous raw draw's table entries and what they were worked out
/// from. A draw with the same vertex format, the same transform inputs (by
/// their change numbers, `TransformInputs::changes`) and the same raw colours
/// gets the same entries: the tables' own lookups (`raw_blocks::raw_transform`
/// against the newest entry, `RawTables::add_bones`, `add_raw_colors`) would
/// find exactly these again, since nothing else writes the tables between two
/// raw draws of this thread (the sprite path forgets them). About half the
/// draws repeat the one before (MTX's race: ~1,250 calls make ~650 draws).
#[derive(Clone, Copy)]
struct LastEntries {
    format: u32,
    /// Base, bone and lighting change numbers (never 0: inputs without
    /// numbers aren't remembered).
    changes: (u32, u32, u32),
    colors: RawColors,
    /// Transform, raw colours, bones (`Draw::transform`, `raw_colors`, `bones`).
    entries: (u32, u32, u32),
}

/// Counted per frame for the capture log (`FrameStats::direct`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DirectStats {
    /// PRIMs written directly, and the draws they became (after merging).
    pub prims: u32,
    pub draws: u32,
    /// Copy tasks and their bytes.
    pub tasks: u32,
    pub task_bytes: u32,
    /// Indexed spans much larger than their corners (`recorder::SPARSE_SPAN`):
    /// gathered corner by corner, and copied whole for want of list room.
    pub gathered: u32,
    pub sparse: u32,
    /// Fallback segments (stretches of recorder draws).
    pub segments: u32,
    /// Draws that took the previous draw's table entries (`LastEntries`).
    pub repeated: u32,
    /// Calls that extended the previous draw in place (`LastRun`; also
    /// counted in `repeated`).
    pub joined: u32,
}

impl DirectStats {
    pub const EMPTY: DirectStats =
        DirectStats { prims: 0, draws: 0, tasks: 0, task_bytes: 0, gathered: 0, sparse: 0, segments: 0, repeated: 0, joined: 0 };
}

/// The previous call, if it was a plain run (`Writer::run`: no indices, in
/// the arena, not PSP RAM) whose last PRIM used all its vertices, and nothing
/// was written after it. MTX draws a model as many calls of one state, each
/// one's vertices right after the last one's (race start: 676 of 1,427
/// calls). The whole way would give such a call the same table entries
/// (`LastEntries`), reserve its vertices right behind the last call's
/// (runs start stride-aligned, and the last one ended less than a stride
/// before), write its index runs right behind the last ones, and
/// `Geometry::finish_raw_draw` would merge it into the newest draw. A call
/// that goes on like that (`FrameBuilder::join`) does exactly that, without
/// the rest: the frame comes out bit-identical.
#[derive(Clone, Copy)]
struct LastRun {
    /// The draw's state as the call was given it (`Prepared::draw`, before
    /// its entries are filled in): a joining call's must be the same
    /// (`Draw::same_state`; its entries are checked by `LastEntries`).
    state: Draw,
    /// Where the next call's vertices start in PSP memory to go on.
    source_end: u32,
    /// Where its run lands in the arena: the byte offset (`Placed::offset`),
    /// the window, and its first vertex's number in the window.
    next_offset: u32,
    window: u32,
    next_first: u32,
    /// The format's vertex size and the bytes its last vertex uses
    /// (`VertexLayout::size`, `VertexFormat::bytes_used`; a joining call has
    /// the same format).
    stride: u32,
    tail: u32,
}

pub struct FrameBuilder {
    lists: Option<&'static mut DirectLists>,
    counts: ListCounts,
    order_count: usize,
    fallback_segments: u32,
    gather_count: usize,
    /// Writing draws in the frame being recorded (`begin_frame`).
    active: bool,
    /// The raw arena and the index ring while active (taken out of the
    /// vertex cache, given back when switched off).
    raw: Option<RawArena>,
    ring: Option<RawArena>,
    /// The previous raw draw's entries this frame (`LastEntries`).
    last_entries: Option<LastEntries>,
    /// The previous call, while the next may extend its draw (`LastRun`).
    last_run: Option<LastRun>,
    /// The last light set's lights (kept across frames: it depends on
    /// registers only).
    lights: crate::lighting::GpuLightsCache,
    pub stats: DirectStats,
}

impl FrameBuilder {
    pub const fn new() -> FrameBuilder {
        FrameBuilder {
            lists: None,
            counts: ListCounts::EMPTY,
            order_count: 0,
            fallback_segments: 0,
            gather_count: 0,
            active: false,
            raw: None,
            ring: None,
            last_entries: None,
            last_run: None,
            lights: crate::lighting::GpuLightsCache::new(),
            stats: DirectStats::EMPTY,
        }
    }

    /// Gives the builder its lists (`None`: none; it stays inactive).
    pub fn set_lists(&mut self, lists: Option<&'static mut DirectLists>) {
        self.lists = lists;
    }

    pub fn is_active(&self) -> bool {
        self.active
    }

    /// The raw arena while active (the frame's `raw_vertices` points there).
    pub fn raw_arena(&self) -> Option<&RawArena> {
        self.raw.as_ref()
    }

    /// The index ring while active.
    pub fn ring_mut(&mut self) -> Option<&mut RawArena> {
        self.ring.as_mut()
    }

    /// Starts a frame (the stream idle): active if wanted (`ENABLED`),
    /// raw vertices on, the lists made and the vertex cache holds a raw
    /// arena; takes the arena and the ring from `cache` then, or gives them
    /// back when no longer active. Returns whether it is active: the caller
    /// tells the stream who owns the raw tables
    /// (`DrawStream::set_raw_tables_with_consumer`).
    pub fn begin_frame(&mut self, wanted: bool, cache: Option<&mut VertexCache>) -> bool {
        let wanted = wanted
            && ENABLED.load(Ordering::Relaxed)
            && crate::raw_arena::ENABLED.load(Ordering::Relaxed)
            && self.lists.is_some();
        if let Some(cache) = cache {
            if wanted && self.raw.is_none() {
                self.raw = cache.raw.take();
                self.ring = cache.index_ring.take();
            } else if !wanted && self.raw.is_some() {
                cache.raw = self.raw.take();
                cache.index_ring = self.ring.take();
            }
        }
        self.active = wanted && self.raw.is_some();
        self.counts = ListCounts::EMPTY;
        self.order_count = 0;
        self.fallback_segments = 0;
        self.gather_count = 0;
        // The frame's tables start empty.
        self.last_entries = None;
        self.last_run = None;
        self.stats = DirectStats::EMPTY;
        self.active
    }

    /// A recorded frame is finished: the arena's and the ring's runs of it
    /// stay put (`RawArena::next_frame`) while the builder holds them; the
    /// arena's statistics then.
    pub fn next_frame(&mut self) -> Option<crate::raw_arena::RawStats> {
        if let Some(ring) = self.ring.as_mut() {
            ring.next_frame();
        }
        self.raw.as_mut().map(|raw| raw.next_frame())
    }

    /// Before a job that makes draws is queued: the `QueuedDraw::segment`
    /// it gets (0: it goes on with the fallback segment before it).
    pub fn fallback_segment(&mut self) -> u16 {
        if !self.active {
            return 0;
        }
        // The recorder's draws will go between: nothing to go on with.
        self.last_run = None;
        let Some(lists) = self.lists.as_deref_mut() else {
            return 0;
        };
        match self.order_count.checked_sub(1).map(|last| lists.order[last]) {
            Some(Stretch::Fallback { .. }) => 0,
            last => {
                if let Some(Stretch::Direct { first, .. }) = last {
                    lists.order[self.order_count - 1] = Stretch::Direct { first, end: self.counts.draw_count };
                }
                // `try_record` leaves room for this one.
                let segment = self.fallback_segments;
                lists.order[self.order_count] = Stretch::Fallback { segment };
                self.order_count += 1;
                self.fallback_segments += 1;
                self.stats.segments += 1;
                segment as u16 + 1
            }
        }
    }

    /// Writes `calls` (one draw's PRIMs, its state `prepared`) as raw draws
    /// if the GPU can take them: 3D lists, strips and fans, indexed or not,
    /// and 3D rectangles as raw sprites. `Err(reason)` (`recorder::fallback`)
    /// if not: nothing is written then (but arena room or table entries
    /// that may stay unused) and the caller queues the draw as a job.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_record(
        &mut self,
        frame: *mut Frame,
        stream: &mut DrawStream,
        memory: &mut impl PspMemory,
        format: VertexFormat,
        prepared: &Prepared,
        material_color: u32,
        inputs: &TransformInputs,
        calls: &[PrimCall],
    ) -> Result<(), Option<usize>> {
        if !self.active || self.order_count + 2 >= MAX_ORDER {
            return Err(None);
        }
        // Morphing, untextured, lights the GPU path doesn't take: the
        // recorder names those reasons itself.
        let colors = in_phase(crate::phase::DIRECT_COLORS, || {
            timed(crate::cycles::DIRECT_COLORS, || crate::recorder::raw_colors_if_enabled(prepared, material_color, inputs.lighting(), format))
        })
        .ok_or(None)?;
        // Taken: whatever happens below, the previous call is no longer the
        // last thing written (`run` notes this one if it can be gone on with).
        if let Some(last) = self.last_run.take() {
            if let Some(written) = self.join(&last, stream, memory, format, prepared, inputs, calls, &colors) {
                return written;
            }
        }
        let (Some(lists), Some(raw)) = (self.lists.as_deref_mut(), self.raw.as_mut()) else {
            return Err(None);
        };
        // A new direct stretch after a fallback segment: its draws merge
        // with none before.
        let starts_stretch = !matches!(self.order_count.checked_sub(1).map(|last| lists.order[last]), Some(Stretch::Direct { .. }));
        if starts_stretch {
            self.counts.merge_floor = self.counts.draw_count;
        }
        let DirectLists { draws, indices, index_words, gather, .. } = lists;
        // Safe: while active the raw tables are this thread's
        // (`DrawStream::set_raw_tables_with_consumer`).
        let tables = unsafe { Frame::raw_tables_raw(frame) };
        let mut geometry = Geometry::of_lists(&mut self.counts, draws, indices, Some(tables));
        let layout = format.layout();
        let mut out = Writer {
            geometry: &mut geometry,
            raw,
            stream,
            memory,
            stats: &mut self.stats,
            gather,
            gather_count: &mut self.gather_count,
            last_entries: &mut self.last_entries,
            last_run: &mut self.last_run,
            lights: &mut self.lights,
        };
        let draw = Draw { first_vertex: 0, first_index: out.geometry.index_count(), ..prepared.draw };
        let all_triangles = calls.iter().all(|call| (primitive::TRIANGLES..=primitive::TRIANGLE_FAN).contains(&call.primitive_type));
        let written = if all_triangles && format.index_format() == IndexFormat::None {
            out.run(format, &layout, inputs, calls, &colors, draw)
        } else if all_triangles {
            out.indexed(format, &layout, inputs, calls, &colors, draw, index_words)
        } else if let [call] = calls {
            if call.primitive_type == primitive::RECTANGLES {
                out.sprites(format, &layout, inputs, *call, &colors, draw)
            } else {
                Err(fallback::REFUSED)
            }
        } else {
            Err(fallback::REFUSED)
        };
        drop(out);
        drop(geometry);
        match written {
            Ok(()) => {
                if starts_stretch {
                    let first = self.counts.merge_floor;
                    lists.order[self.order_count] = Stretch::Direct { first, end: first };
                    self.order_count += 1;
                }
                self.stats.prims += calls.len() as u32;
                Ok(())
            }
            Err(reason) => Err(Some(reason)),
        }
    }

    /// Extends the newest draw with `calls` if they go on where the previous
    /// call (`last`) stopped, as `LastRun` describes: `None` if they don't
    /// (nothing written: `try_record` takes the whole way), else what the
    /// whole way would have returned, with the same frame written.
    #[allow(clippy::too_many_arguments)]
    fn join(
        &mut self,
        last: &LastRun,
        stream: &mut DrawStream,
        memory: &mut impl PspMemory,
        format: VertexFormat,
        prepared: &Prepared,
        inputs: &TransformInputs,
        calls: &[PrimCall],
        colors: &RawColors,
    ) -> Option<Result<(), Option<usize>>> {
        if crate::bench::experiment(crate::bench::NO_JOINED_CALLS) {
            return None;
        }
        let entries = self.last_entries.as_ref()?;
        let (base, lighting) = inputs.changes();
        // The whole way's choices, cheapest first: a plain run (`run`) whose
        // vertices follow the last call's, with the same state and the same
        // entries (`Writer::entries` would repeat them).
        let address = calls[0].vertex_address;
        let goes_on = address == last.source_end
            && format.index_format() == IndexFormat::None
            && calls.iter().all(|call| (primitive::TRIANGLES..=primitive::TRIANGLE_FAN).contains(&call.primitive_type))
            && entries.format == format.0
            && entries.changes == (base, inputs.bone_change(), lighting)
            && entries.colors.words() == colors.words()
            && last.state.same_state(&prepared.draw);
        if !goes_on {
            return None;
        }
        let (used, whole) = run_vertices(calls)?;
        let stride = last.stride;
        let length = stride * (used - 1) + last.tail;
        if crate::raw_arena::zero_copy_wanted(address, length, format) {
            return None;
        }
        let runs = calls.iter().filter(|call| crate::recorder::triangle_count(call.primitive_type, call.count) > 0).count();
        let (Some(lists), Some(raw)) = (self.lists.as_deref_mut(), self.raw.as_mut()) else {
            return None;
        };
        let index_start = self.counts.index_count as usize;
        let newest = (self.counts.draw_count as usize).checked_sub(1)?;
        if runs == 0 || index_start + 3 * runs > lists.indices.len() {
            return None;
        }
        debug_assert!(
            lists.draws[newest].first_index + lists.draws[newest].index_count == index_start as u32
                && lists.draws[newest].first_vertex == last.window,
            "the previous call's draw is the newest and ends with the indices"
        );
        // From here on the whole way would have taken the same steps.
        self.stats.repeated += 1;
        let Some((placed, out)) = in_phase(crate::phase::DIRECT_ARENA, || timed(crate::cycles::DIRECT_RESERVE, || raw.reserve(length, stride)))
        else {
            return Some(Err(Some(fallback::ARENA_FULL)));
        };
        let destination = out.as_mut_ptr();
        if placed.window != last.window || placed.offset != last.next_offset {
            // The next window: a draw of its own.
            raw.undo_last();
            self.stats.repeated -= 1;
            return None;
        }
        let task = CopyTask { source: address, length, destination, layout: Layout::Plain };
        if !in_phase(crate::phase::DIRECT_COPY, || timed(crate::cycles::DIRECT_COPY, || stream.copy_task(memory, task))) {
            raw.undo_last();
            return Some(Err(Some(fallback::UNREADABLE)));
        }
        self.stats.tasks += 1;
        self.stats.task_bytes += length;
        let mut first = last.next_first;
        let mut records = lists.indices[index_start..index_start + 3 * runs].chunks_exact_mut(3);
        for call in calls {
            if crate::recorder::triangle_count(call.primitive_type, call.count) > 0 {
                let record = records.next().expect("room for every run");
                record.copy_from_slice(&[call.primitive_type as u16, call.count as u16, first as u16]);
            }
            first += call.count;
        }
        self.counts.index_count += 3 * runs as u32;
        lists.draws[newest].index_count += 3 * runs as u32;
        raw.stats.lit += (colors.rules[0] == crate::raw_vertex::lighting::LIGHTS) as u32;
        self.stats.prims += calls.len() as u32;
        self.stats.joined += 1;
        if whole {
            self.last_run = Some(LastRun {
                source_end: address.wrapping_add(stride * used),
                next_offset: placed.offset + stride * used,
                next_first: first,
                ..*last
            });
        }
        Some(Ok(()))
    }

    /// The frame's end (the recorder idle, every copy task done): the
    /// capture thread's draws and the recorder's moved into `frame` in list
    /// order, the direct draws' indices after the recorder's. A frame
    /// without room for both (more than `MAX_DRAWS` draws) keeps only the
    /// recorder's and is marked overflowed.
    pub fn finish_frame(&mut self, frame: &mut Frame, stream: &DrawStream) {
        if !self.active {
            return;
        }
        let Some(lists) = self.lists.as_deref_mut() else {
            return;
        };
        if let Some(last) = self.order_count.checked_sub(1) {
            if let Stretch::Direct { first, .. } = lists.order[last] {
                lists.order[last] = Stretch::Direct { first, end: self.counts.draw_count };
            }
        }
        if self.counts.overflowed != 0 {
            frame.geometry_overflowed = 1;
        }
        let direct_draws = self.counts.draw_count as usize;
        let direct_indices = self.counts.index_count as usize;
        self.stats.draws = direct_draws as u32;
        if direct_draws == 0 {
            return;
        }
        let fallback_draws = frame.draw_count as usize;
        let index_offset = frame.index_count as usize;
        if fallback_draws + direct_draws > MAX_DRAWS || index_offset + direct_indices > MAX_INDICES {
            frame.geometry_overflowed = 1;
            return;
        }
        frame.indices[index_offset..index_offset + direct_indices].copy_from_slice(&lists.indices[..direct_indices]);
        // Where each fallback segment ends: where the next one starts, the
        // last one at the end of the recorder's draws.
        let segment_end = |segment: u32| -> u32 {
            if segment + 1 < self.fallback_segments {
                stream.segment_start(segment as usize + 1)
            } else {
                fallback_draws as u32
            }
        };
        // From the back: the recorder's draws only move up (direct draws go
        // in before some of them), so none is overwritten before it moved.
        let mut end = fallback_draws + direct_draws;
        for stretch in lists.order[..self.order_count].iter().rev() {
            match *stretch {
                Stretch::Direct { first, end: last } => {
                    let count = (last - first) as usize;
                    end -= count;
                    for (to, from) in frame.draws[end..end + count].iter_mut().zip(&lists.draws[first as usize..last as usize]) {
                        *to = Draw { first_index: from.first_index + index_offset as u32, ..*from };
                    }
                }
                Stretch::Fallback { segment } => {
                    let start = stream.segment_start(segment as usize) as usize;
                    let count = segment_end(segment) as usize - start;
                    end -= count;
                    frame.draws.copy_within(start..start + count, end);
                }
            }
        }
        debug_assert_eq!(end, 0, "every draw placed");
        frame.draw_count = (fallback_draws + direct_draws) as u32;
        frame.index_count = (index_offset + direct_indices) as u32;
    }
}

impl Default for FrameBuilder {
    fn default() -> Self {
        FrameBuilder::new()
    }
}

/// What one `try_record` writes with.
struct Writer<'g, 'a, M: PspMemory> {
    geometry: &'g mut Geometry<'a>,
    raw: &'g mut RawArena,
    stream: &'g mut DrawStream,
    memory: &'g mut M,
    stats: &'g mut DirectStats,
    gather: &'g mut [u16; GATHER_ENTRIES],
    gather_count: &'g mut usize,
    last_entries: &'g mut Option<LastEntries>,
    last_run: &'g mut Option<LastRun>,
    lights: &'g mut crate::lighting::GpuLightsCache,
}

impl<M: PspMemory> Writer<'_, '_, M> {
    /// The draw's table entries: transform, colours, bones (as
    /// `recorder::raw_table_entries`, each part in its own profile phase).
    fn entries(&mut self, format: VertexFormat, inputs: &TransformInputs, colors: &RawColors) -> Result<(u32, u32, u32), usize> {
        let (base, lighting) = inputs.changes();
        let changes = (base, inputs.bone_change(), lighting);
        let remembered = base != 0 && !crate::bench::experiment(crate::bench::NO_REPEATED_ENTRIES);
        if remembered {
            if let Some(last) = self.last_entries.as_ref() {
                if last.format == format.0 && last.changes == changes && last.colors.words() == colors.words() {
                    self.stats.repeated += 1;
                    return Ok(last.entries);
                }
            }
        }
        // Forgotten first: a lookup that fails half-way has changed the tables.
        *self.last_entries = None;
        let entries = self.work_out_entries(format, inputs, colors)?;
        *self.last_entries = remembered.then_some(LastEntries { format: format.0, changes, colors: *colors, entries });
        Ok(entries)
    }

    /// The table entries looked up or added (`entries` without the repeat).
    fn work_out_entries(&mut self, format: VertexFormat, inputs: &TransformInputs, colors: &RawColors) -> Result<(u32, u32, u32), usize> {
        use crate::phase::{DIRECT_BONES, DIRECT_COLORS, DIRECT_TRANSFORM};
        let tables = self.geometry.raw_tables().ok_or(fallback::TABLES_FULL)?;
        let lit = colors.rules[0] == crate::raw_vertex::lighting::LIGHTS;
        let bones = if format.has_weights() {
            in_phase(DIRECT_BONES, || tables.add_bones(inputs.bones())).ok_or(fallback::TABLES_FULL)? + 1
        } else {
            0
        };
        let lighting = lit.then(|| (format.has_color(), colors.rules[1] != 0.0));
        let lights = (!crate::bench::experiment(crate::bench::NO_LIGHTS_CACHE)).then_some(&mut *self.lights);
        let started = crate::cycles::start(crate::cycles::DIRECT_TRANSFORM);
        let number = in_phase(DIRECT_TRANSFORM, || crate::raw_blocks::raw_transform_with(tables, inputs, lighting, lights));
        crate::cycles::add(crate::cycles::DIRECT_TRANSFORM, started);
        let number = number.ok_or(fallback::TABLES_FULL)?;
        let colors_number = in_phase(DIRECT_COLORS, || tables.add_raw_colors(colors)).ok_or(fallback::TABLES_FULL)?;
        Ok((number, colors_number, bones))
    }

    /// Copies `length` bytes at `source` into `destination` (the arena room
    /// reserved last), laid out as `layout`, or has the copier do it. Err
    /// if they were copied at once and couldn't be read: the room is given
    /// back then.
    fn copy(&mut self, source: u32, length: u32, destination: *mut u8, layout: Layout) -> Result<(), usize> {
        let task = CopyTask { source, length, destination, layout };
        if in_phase(crate::phase::DIRECT_COPY, || timed(crate::cycles::DIRECT_COPY, || self.stream.copy_task(self.memory, task))) {
            self.stats.tasks += 1;
            self.stats.task_bytes += task.written() as u32;
            Ok(())
        } else {
            self.raw.undo_last();
            Err(fallback::UNREADABLE)
        }
    }

    /// Whether `bytes` at `address` stay in PSP RAM for the GPU (zero copy,
    /// `raw_arena::zero_copy_wanted`): only where a copier could have read
    /// them now (as for jobs: not while a dump records them).
    fn zero_copy(&mut self, address: u32, bytes: u32, format: VertexFormat) -> bool {
        crate::raw_arena::zero_copy_wanted(address, bytes, format)
            && in_phase(crate::phase::DIRECT_COPY, || self.memory.defer(address & !3, (bytes + (address & 3)).next_multiple_of(4)))
    }

    /// 3D lists, strips and fans without indices as one raw draw with index
    /// runs (`recorder::record_raw_run`, the vertices copied by a task).
    fn run(
        &mut self,
        format: VertexFormat,
        layout: &VertexLayout,
        inputs: &TransformInputs,
        calls: &[PrimCall],
        colors: &RawColors,
        mut draw: Draw,
    ) -> Result<(), usize> {
        // All vertices up to the last PRIM's last whole triangle.
        let state = draw;
        let (used, whole) = run_vertices(calls).ok_or(fallback::REFUSED)?;
        let stride = layout.size;
        let address = calls[0].vertex_address;
        let length = stride * (used - 1) + format.bytes_used(layout);
        let (number, colors_number, bones) = self.entries(format, inputs, colors)?;
        let in_psp_ram = self.zero_copy(address, length, format);
        let (window, offset, mut first, copy) = if in_psp_ram {
            self.raw.stats.psp_ram += 1;
            self.raw.stats.psp_ram_bytes += length;
            (address, 0, 0, None)
        } else {
            let (placed, out) = in_phase(crate::phase::DIRECT_ARENA, || timed(crate::cycles::DIRECT_RESERVE, || self.raw.reserve(length, stride)))
                .ok_or(fallback::ARENA_FULL)?;
            (placed.window, placed.offset, placed.first_vertex(stride), Some(out.as_mut_ptr()))
        };
        let previous = crate::phase::enter(crate::phase::DIRECT_ARENA);
        let runs = calls.iter().filter(|call| crate::recorder::triangle_count(call.primitive_type, call.count) > 0).count();
        let Some(out) = self.geometry.reserve_indices(3 * runs) else {
            crate::phase::leave(previous);
            return Err(fallback::TABLES_FULL);
        };
        let mut records = out.chunks_exact_mut(3);
        for call in calls {
            if crate::recorder::triangle_count(call.primitive_type, call.count) > 0 {
                let record = records.next().expect("reserved for every run");
                record.copy_from_slice(&[call.primitive_type as u16, call.count as u16, first as u16]);
            }
            first += call.count;
        }
        crate::phase::leave(previous);
        if let Some(destination) = copy {
            if let Err(reason) = self.copy(address, length, destination, Layout::Plain) {
                self.geometry.undo_draw(&draw);
                return Err(reason);
            }
        }
        self.raw.stats.lit += (colors.rules[0] == crate::raw_vertex::lighting::LIGHTS) as u32;
        draw.transform = number;
        draw.vertex_format = format.0;
        draw.raw_colors = colors_number;
        draw.bones = bones;
        in_phase(crate::phase::DIRECT_ARENA, || {
            timed(crate::cycles::DIRECT_FINISH, || {
                if in_psp_ram {
                    self.geometry.finish_raw_psp_draw(draw, window, stride);
                } else {
                    self.geometry.finish_raw_draw(draw, window);
                }
            })
        });
        // The next call may go on from here (`LastRun`): the newest draw is
        // this one (or this one merged in), its entries remembered.
        if !in_psp_ram && whole && runs > 0 && !self.geometry.is_overflowed() && self.last_entries.is_some() {
            *self.last_run = Some(LastRun {
                state,
                source_end: address.wrapping_add(stride * used),
                next_offset: offset + stride * used,
                window,
                next_first: first,
                stride,
                tail: format.bytes_used(layout),
            });
        }
        Ok(())
    }

    /// Indexed lists, strips and fans of one vertex buffer as one raw draw
    /// with a triangle list (`recorder::record_raw_converted`): the indices
    /// read once into `words`, their bounds found and the corners converted
    /// from there; the vertex span copied by a task or left in PSP RAM.
    #[allow(clippy::too_many_arguments)]
    fn indexed(
        &mut self,
        format: VertexFormat,
        layout: &VertexLayout,
        inputs: &TransformInputs,
        calls: &[PrimCall],
        colors: &RawColors,
        mut draw: Draw,
        words: &mut [u32; INDEX_WORDS],
    ) -> Result<(), usize> {
        if !crate::raw_arena::INDEXED_DRAWS.load(Ordering::Relaxed) {
            return Err(fallback::REFUSED);
        }
        let index_format = format.index_format();
        let size = index_format.size();
        let count: u32 = calls.iter().map(|call| call.count).sum();
        let corners: u32 = calls.iter().map(|call| 3 * crate::recorder::triangle_count(call.primitive_type, call.count)).sum();
        let index_start = calls[0].index_address;
        let skew = (index_start & 3) as usize;
        let word_count = (skew + (count * size) as usize).div_ceil(4);
        if corners == 0 || word_count > INDEX_WORDS {
            return Err(fallback::REFUSED);
        }
        let words = &mut words[..word_count];
        let previous = crate::phase::enter(crate::phase::DIRECT_INDICES);
        if self.memory.read_words(index_start & !3, words).is_none() {
            crate::phase::leave(previous);
            return Err(fallback::UNREADABLE);
        }
        // Safe to view as bytes: `words` holds every index.
        let bytes = unsafe { core::slice::from_raw_parts(words.as_ptr() as *const u8, 4 * word_count) };
        let indices = &bytes[skew..skew + (count * size) as usize];
        let (lowest, highest) = match index_format {
            IndexFormat::U8 => bounds::<1>(indices),
            IndexFormat::U16 => bounds::<2>(indices),
            _ => bounds::<4>(indices),
        };
        crate::phase::leave(previous);
        if highest - lowest >= 65_536 {
            return Err(fallback::REFUSED);
        }
        let stride = layout.size;
        let span_start = calls[0].vertex_address.wrapping_add(lowest.wrapping_mul(stride));
        let span_length = (highest - lowest) * stride + format.bytes_used(layout);
        let (number, colors_number, bones) = self.entries(format, inputs, colors)?;
        let in_psp_ram = self.zero_copy(span_start, span_length, format);
        if !in_psp_ram && (highest - lowest + 1) > crate::recorder::SPARSE_SPAN * corners {
            // Copying the span would move more bytes than its corners need.
            let entries = (number, colors_number, bones);
            match self.gathered(format, layout, calls, indices, (lowest, span_start), entries, draw) {
                Err(None) => self.stats.sparse += 1, // no list room: copied whole below
                written => return written.map_err(|reason| reason.unwrap_or(fallback::REFUSED)),
            }
        }
        let (place, first, copy) = if in_psp_ram {
            self.raw.stats.psp_ram += 1;
            self.raw.stats.psp_ram_bytes += span_length;
            (span_start, 0, None)
        } else {
            let (placed, out) = self.raw.reserve(span_length, stride).ok_or(fallback::ARENA_FULL)?;
            (placed.window, placed.first_vertex(stride), Some(out.as_mut_ptr()))
        };
        let out = self.geometry.reserve_indices(corners as usize).ok_or(fallback::TABLES_FULL)?;
        let converted = in_phase(crate::phase::DIRECT_INDICES, || match index_format {
            IndexFormat::U8 => crate::recorder::convert_indices::<1>(indices, calls, (lowest, highest), first, out),
            IndexFormat::U16 => crate::recorder::convert_indices::<2>(indices, calls, (lowest, highest), first, out),
            _ => crate::recorder::convert_indices::<4>(indices, calls, (lowest, highest), first, out),
        });
        debug_assert!(converted, "the bounds are these indices' own");
        if let Some(destination) = copy {
            if let Err(reason) = self.copy(span_start, span_length, destination, Layout::Plain) {
                self.geometry.undo_draw(&draw);
                return Err(reason);
            }
        }
        self.raw.stats.lit += (colors.rules[0] == crate::raw_vertex::lighting::LIGHTS) as u32;
        self.raw.stats.converted += 1;
        draw.transform = number;
        draw.vertex_format = format.0 & !INDEX_FORMAT_BITS;
        draw.raw_colors = colors_number;
        draw.bones = bones;
        in_phase(crate::phase::DIRECT_ARENA, || self.geometry.finish_raw_listed_draw(draw, place, in_psp_ram));
        Ok(())
    }

    /// Indexed PRIMs whose vertex span is much larger than the corners they
    /// use, as `recorder::record_raw_gathered` records them: each corner's
    /// vertex (each PRIM's whole triangles, in index order) gathered into
    /// the arena by a gather task, drawn with index runs as plain lists,
    /// strips and fans. `Err(None)`: no room for the task's list (nothing
    /// written; the caller copies the span whole instead).
    #[allow(clippy::too_many_arguments)]
    fn gathered(
        &mut self,
        format: VertexFormat,
        layout: &VertexLayout,
        calls: &[PrimCall],
        indices: &[u8],
        (lowest, span_start): (u32, u32),
        (number, colors_number, bones): (u32, u32, u32),
        mut draw: Draw,
    ) -> Result<(), Option<usize>> {
        // Whole triangles of each PRIM.
        let used_of = |call: &PrimCall| match call.primitive_type {
            primitive::TRIANGLES => call.count / 3 * 3,
            _ if call.count >= 3 => call.count,
            _ => 0,
        };
        let used: u32 = calls.iter().map(used_of).sum();
        if used == 0 || used > 65_536 {
            return Err(Some(fallback::REFUSED));
        }
        let start = *self.gather_count;
        if start + used as usize > GATHER_ENTRIES {
            return Err(None);
        }
        let size = format.index_format().size() as usize;
        let list = &mut self.gather[start..start + used as usize];
        let mut written = 0;
        let mut position = 0; // index number from the first PRIM's first
        for call in calls {
            for n in 0..used_of(call) as usize {
                let at = (position + n) * size;
                let index = match size {
                    1 => indices[at] as u32,
                    2 => u16::from_le_bytes([indices[at], indices[at + 1]]) as u32,
                    _ => u32::from_le_bytes([indices[at], indices[at + 1], indices[at + 2], indices[at + 3]]),
                };
                list[written] = (index - lowest) as u16;
                written += 1;
            }
            position += call.count as usize;
        }
        let list = list.as_ptr();
        let stride = layout.size;
        let (placed, out) = self.raw.reserve(used * stride, stride).ok_or(Some(fallback::ARENA_FULL))?;
        let destination = out.as_mut_ptr();
        let runs = calls.iter().filter(|call| used_of(call) >= 3).count();
        let records = self.geometry.reserve_indices(3 * runs).ok_or(Some(fallback::TABLES_FULL))?;
        let mut first = placed.first_vertex(stride);
        for (record, call) in records.chunks_exact_mut(3).zip(calls.iter().filter(|call| used_of(call) >= 3)) {
            let take = used_of(call);
            record.copy_from_slice(&[call.primitive_type as u16, take as u16, first as u16]);
            first += take;
        }
        let gather = Layout::Gather { stride, used: format.bytes_used(layout), list };
        if let Err(reason) = self.copy(span_start, used, destination, gather) {
            self.geometry.undo_draw(&draw);
            return Err(Some(reason));
        }
        *self.gather_count = start + used as usize;
        self.raw.stats.gathered += 1;
        self.stats.gathered += 1;
        draw.transform = number;
        draw.vertex_format = format.0 & !INDEX_FORMAT_BITS;
        draw.raw_colors = colors_number;
        draw.bones = bones;
        self.geometry.finish_raw_draw(draw, placed.window);
        Ok(())
    }

    /// A 3D rectangles PRIM as raw sprites (`recorder::record_raw_sprites`):
    /// each sprite's vertex pair four times in the arena (a task with
    /// `Layout::Sprites`), one index run `[RECTANGLES, sprites, first corner]`.
    fn sprites(
        &mut self,
        format: VertexFormat,
        layout: &VertexLayout,
        inputs: &TransformInputs,
        call: PrimCall,
        colors: &RawColors,
        mut draw: Draw,
    ) -> Result<(), usize> {
        let sprites = call.count / 2;
        let drawable = crate::raw_arena::SPRITES.load(Ordering::Relaxed);
        // The sprite program has neither lights nor FIX A (nativeres_raw_sprite_v.cg).
        if !drawable
            || !crate::recorder::sprites_suit_raw(format, sprites)
            || colors.rules[0] == crate::raw_vertex::lighting::LIGHTS
            || colors.scale[3] != 0.0
        {
            return Err(fallback::SPRITES);
        }
        // Rectangles are never culled (PPSSPP: no cull for GE_PRIM_RECTANGLES).
        draw.flags = (draw.flags | draw_flags::RECTANGLES) & !(draw_flags::CULL | draw_flags::CULL_CCW);
        let stride = layout.size;
        let pair = 2 * stride;
        let length = stride * (2 * sprites - 1) + format.bytes_used(layout);
        // Entries of their own (no lights): the next draw can't repeat the
        // remembered ones without the tables' lookups (`LastEntries`).
        *self.last_entries = None;
        let tables = self.geometry.raw_tables().ok_or(fallback::TABLES_FULL)?;
        let number = crate::raw_blocks::raw_transform(tables, inputs, None).ok_or(fallback::TABLES_FULL)?;
        let colors_number = tables.add_raw_colors(colors).ok_or(fallback::TABLES_FULL)?;
        // A block of four pairs per sprite; whole blocks into the window, so
        // the first corner of each sprite is a multiple of 4.
        let (placed, out) = self.raw.reserve(4 * pair * sprites, 4 * pair).ok_or(fallback::ARENA_FULL)?;
        let destination = out.as_mut_ptr();
        let first_corner = 4 * placed.first_vertex(4 * pair);
        let record = self.geometry.reserve_indices(3).ok_or(fallback::TABLES_FULL)?;
        record.copy_from_slice(&[primitive::RECTANGLES as u16, sprites as u16, first_corner as u16]);
        if let Err(reason) = self.copy(call.vertex_address, length, destination, Layout::Sprites { pair }) {
            self.geometry.undo_draw(&draw);
            return Err(reason);
        }
        draw.transform = number;
        draw.vertex_format = format.0;
        draw.raw_colors = colors_number;
        draw.bones = 0;
        draw.flags |= draw_flags::RAW_SPRITES;
        self.geometry.finish_raw_draw(draw, placed.window);
        Ok(())
    }
}

/// A plain run's vertices (`Writer::run`): all of them up to the last PRIM's
/// last whole triangle, and whether that is all of the last PRIM's. `None`
/// if none or more than 16-bit indices reach.
fn run_vertices(calls: &[PrimCall]) -> Option<(u32, bool)> {
    let (last, before) = calls.split_last()?;
    let last_used = match last.primitive_type {
        primitive::TRIANGLES => last.count / 3 * 3,
        _ if last.count >= 3 => last.count,
        _ => 0,
    };
    let used = before.iter().map(|call| call.count).sum::<u32>() + last_used;
    (used > 0 && used <= 65_536).then_some((used, last_used == last.count))
}

/// The lowest and highest of `indices` (`SIZE` bytes each, little-endian;
/// at least one).
fn bounds<const SIZE: usize>(indices: &[u8]) -> (u32, u32) {
    let mut lowest = u32::MAX;
    let mut highest = 0;
    for index in indices.chunks_exact(SIZE) {
        let value = match SIZE {
            1 => index[0] as u32,
            2 => u16::from_le_bytes([index[0], index[1]]) as u32,
            _ => u32::from_le_bytes([index[0], index[1], index[2], index[3]]),
        };
        lowest = lowest.min(value);
        highest = highest.max(value);
    }
    (lowest, highest)
}
