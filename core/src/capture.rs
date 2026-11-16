//! Follows the game's GE calls (as reported by the PSP plugin) and walks the
//! submitted display lists in the order the GE would execute them.
//!
//! The GE runs one list at a time, in queue order. A list that has reached
//! its stall address blocks the lists behind it until the game moves the
//! stall address, so we only ever walk the list at the front of the queue.

use crate::command::{op, Command};
use crate::display_list::{walk, CommandVisitor, DisplayList, PlainActions, WalkResult};
use crate::draw_stream::DrawStream;
use crate::frame::Frame;
use crate::ge_context::ContextTable;
use crate::ge_state::GeState;
use crate::ge_writes::{GeWrites, Overlaid};
use crate::memory::{physical, PspMemory};
use crate::recorder::{DrawStateCache, LoadedPalette, Recorder};
use crate::snapshot::{Reader, Writer, SNAPSHOT_VERSION};
use crate::stats::{DisplayedFramebuffer, FrameStats};
use crate::walk_stream::{self, StreamWriter, WalkHooks, WalkStream, WriteFilter};

use core::sync::atomic::{AtomicBool, AtomicPtr, Ordering};

/// How many lists can be queued at once. The PSP allows 64.
const QUEUE_CAPACITY: usize = 64;

/// Safety limit for one walk, to survive endless loops in a list.
const MAX_COMMANDS_PER_WALK: u32 = 500_000;

/// What the game did, as recorded by the PSP plugin.
/// The ring's event types (`event_ring::EVENT_*`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Event {
    /// sceGeListEnQueue. `id` is negative if the call failed. `context`:
    /// where the GE saves its state for the list's run (0 = none).
    ListEnqueued { list: u32, stall: u32, id: i32, context: u32 },
    /// sceGeListEnQueueHead: like `ListEnqueued`, but runs first.
    ListEnqueuedAtHead { list: u32, stall: u32, id: i32, context: u32 },
    /// sceGeSaveContext / sceGeRestoreContext (`ge_context`); `ok`: the
    /// call succeeded (it fails while lists run).
    ContextSaved { address: u32, ok: bool },
    ContextRestored { address: u32, ok: bool },
    /// sceGeListUpdateStallAddr.
    StallUpdated { id: i32, stall: u32 },
    /// sceDisplaySetFrameBuf: the frame is finished.
    FramebufferShown { address: u32, stride: u32, pixel_format: u32 },
}

use crate::event_ring::{
    EVENT_FRAMEBUFFER_FLIP, EVENT_GE_RESTORE_CONTEXT, EVENT_GE_SAVE_CONTEXT, EVENT_LIST_ENQUEUE, EVENT_LIST_ENQUEUE_HEAD,
    EVENT_STALL_UPDATE,
};

impl Event {
    /// From an event as the PSP plugin writes it into the ring.
    pub fn from_ring(event_type: u32, args: [u32; 4]) -> Option<Event> {
        Some(match event_type {
            EVENT_LIST_ENQUEUE => Event::ListEnqueued { list: args[0], stall: args[1], id: args[2] as i32, context: args[3] },
            EVENT_LIST_ENQUEUE_HEAD => {
                Event::ListEnqueuedAtHead { list: args[0], stall: args[1], id: args[2] as i32, context: args[3] }
            }
            EVENT_GE_SAVE_CONTEXT => Event::ContextSaved { address: args[0], ok: args[1] as i32 >= 0 },
            EVENT_GE_RESTORE_CONTEXT => Event::ContextRestored { address: args[0], ok: args[1] as i32 >= 0 },
            EVENT_STALL_UPDATE => Event::StallUpdated { id: args[0] as i32, stall: args[1] },
            EVENT_FRAMEBUFFER_FLIP => {
                Event::FramebufferShown { address: args[0], stride: args[1], pixel_format: args[2] }
            }
            _ => return None,
        })
    }
}

pub struct Capture {
    /// Lists in execution order; `queue[0]` is the one the GE is running.
    queue: [DisplayList; QUEUE_CAPACITY],
    queue_len: usize,
    ge: GeState,
    palette: LoadedPalette,
    /// The last draw's state, for the next draw (see `recorder`).
    draw_states: DrawStateCache,
    /// Draws waiting for their vertices to be recorded (see `draw_stream`).
    stream: DrawStream,
    /// What block transfers wrote this frame; the walk reads memory
    /// through it (see `ge_writes`).
    ge_writes: GeWrites,
    current_frame: FrameStats,
    frames_completed: u32,
    /// Only follow the lists, without statistics or recording draws. For
    /// timing the walk on its own (`bench`); wrong for anything else.
    pub walk_only: bool,
    /// The capture in two stages (`walk_stream`): the walk above (stage A)
    /// only follows the lists and writes what it executed into
    /// `walk_stream`; stage B (the `stage_*` fields with `palette`,
    /// `draw_states` and `stream`) works out the draws from it. Off: one
    /// walk does both, as before.
    pipelined: bool,
    walk_stream: WalkStream,
    /// Stage B's GE state, block-transfer writes and statistics (draws,
    /// textures, ...; the walk's counts stay in `current_frame`).
    stage_ge: GeState,
    stage_writes: GeWrites,
    stage_stats: FrameStats,
    /// Stage B in another thread (`set_stage_hooks`): the frame it records
    /// into (null: none; set by stage A with each event, the same for a
    /// whole frame), and set by stage B when it has reached the frame's end
    /// (it then waits until stage A has finished the frame and clears it).
    stage_frame: AtomicPtr<Frame>,
    stage_frame_ended: AtomicBool,
    /// Stage B's `PlainActions` (the recorder's, worked out once).
    stage_actions: Option<PlainActions>,
    /// Stage A's filter for register writes that change nothing.
    write_filter: WriteFilter,
    /// GE states saved by lists with a context and sceGeSaveContext.
    contexts: ContextTable,
}

impl Capture {
    pub const fn new() -> Capture {
        Capture {
            queue: [DisplayList::EMPTY; QUEUE_CAPACITY],
            queue_len: 0,
            ge: GeState::new(),
            palette: LoadedPalette::new(),
            draw_states: DrawStateCache::new(),
            stream: DrawStream::new(),
            ge_writes: GeWrites::new(),
            current_frame: FrameStats::new(),
            frames_completed: 0,
            walk_only: false,
            pipelined: false,
            walk_stream: WalkStream::new(),
            stage_ge: GeState::new(),
            stage_writes: GeWrites::new(),
            stage_stats: FrameStats::new(),
            stage_frame: AtomicPtr::new(core::ptr::null_mut()),
            stage_frame_ended: AtomicBool::new(false),
            stage_actions: None,
            write_filter: WriteFilter::new(),
            contexts: ContextTable::new(),
        }
    }

    /// Forgets everything, e.g. when a new game starts.
    ///
    /// Done field by field on purpose: `*self = Capture::new()` would build a
    /// ~36 KiB temporary on the stack first, which overflowed the Vita
    /// capture thread's stack. Queue slots past `queue_len` are never read,
    /// so they don't need clearing.
    pub fn reset(&mut self) {
        self.walk_stream.wait_until_read();
        self.queue_len = 0;
        self.ge = GeState::new();
        self.palette.clear(); // in place: LoadedPalette holds 2 KiB of words
        self.draw_states.forget();
        self.stream.clear();
        self.ge_writes.clear();
        self.current_frame = FrameStats::new();
        self.frames_completed = 0;
        self.walk_stream.clear();
        self.stage_ge = GeState::new();
        self.stage_writes.clear();
        self.stage_stats = FrameStats::new();
        self.write_filter.start(&self.stage_ge, crate::recorder::recorder_wants);
        self.stage_actions = None;
        self.contexts.clear();
    }

    /// The PSP put back the GE state saved at `address` (`ge_context`).
    fn restore_context(&mut self, address: u32) {
        if !self.contexts.restore(address, &mut self.ge) {
            self.current_frame.contexts_missing += 1;
            return;
        }
        self.current_frame.contexts_restored += 1;
        if self.pipelined {
            // Stage B keeps its own copy of the registers, fed by the
            // walk's commands; a restore doesn't reach it (yet).
            self.current_frame.contexts_pipelined += 1;
        }
    }

    /// Makes a `Capture` in zeroed memory at `place`, without building one
    /// on the stack first (it is ~0.5 MiB; the Vita's `capturebench` makes
    /// its own in a block of phycont memory while it runs): what `reset`
    /// doesn't set is valid as zero bytes (counters, flags, null pointers as
    /// `None`; ring contents are written before they are read).
    ///
    /// # Safety
    /// `place` must point to `size_of::<Capture>()` zeroed, writable bytes,
    /// suitably aligned, used by nobody else.
    pub unsafe fn init_zeroed(place: *mut Capture) {
        (*place).reset();
    }

    /// Saves what carries over from one frame to the next (GE registers,
    /// loaded palette, queued lists). Statistics are not saved.
    pub fn save_state(&self, out: &mut Writer) {
        out.u32(SNAPSHOT_VERSION);
        self.ge.save(out);
        self.palette.save_state(out);
        out.u32(self.queue_len as u32);
        for list in &self.queue[..self.queue_len] {
            list.save(out);
        }
    }

    /// Restores a state saved by `save_state`. `None` if the data is
    /// incomplete or from another version.
    pub fn load_state(&mut self, input: &mut Reader) -> Option<()> {
        let version = input.u32()?;
        if !(1..=SNAPSHOT_VERSION).contains(&version) {
            return None;
        }
        self.reset();
        self.ge.load(input, version)?;
        self.palette.load_state(input, version)?;
        let queue_len = input.u32()? as usize;
        if queue_len > QUEUE_CAPACITY {
            return None;
        }
        for index in 0..queue_len {
            self.queue[index] = DisplayList::load(input, version, self.ge.offset)?;
        }
        self.queue_len = queue_len;
        self.stage_ge.clone_from(&self.ge);
        self.write_filter.start(&self.stage_ge, crate::recorder::recorder_wants);
        Some(())
    }

    /// Runs the capture in two stages (`walk_stream`) or in one. Only
    /// between frames (stage B must have caught up).
    pub fn set_pipelined(&mut self, on: bool) {
        self.walk_stream.wait_until_read();
        self.pipelined = on;
        self.stage_ge.clone_from(&self.ge);
        self.write_filter.start(&self.stage_ge, crate::recorder::recorder_wants);
    }

    /// Plain register writes stage A passed on / left out so far.
    pub fn write_filter_counts(&self) -> (u32, u32) {
        (self.write_filter.sent, self.write_filter.dropped)
    }

    pub fn pipelined(&self) -> bool {
        self.pipelined
    }

    /// Stage B runs in another thread from now on, which calls
    /// `run_stage_b_thread` when woken (`None`: stage A's thread runs it).
    /// Only between frames.
    pub fn set_stage_hooks(&mut self, hooks: Option<&'static WalkHooks>) {
        self.walk_stream.wait_until_read();
        self.walk_stream.set_hooks(hooks);
    }

    /// Stage B's thread: works out the draws the walk stream holds, up to
    /// its end or the frame's end.
    ///
    /// # Safety
    /// `capture` must point to a `Capture` whose stage A runs in another
    /// thread with `set_stage_hooks` set, and only one thread may call this.
    /// Stage B's fields (`stage_*`, `palette`, `draw_states`, `stream`) are
    /// this thread's while the stream holds records; stage A touches them
    /// only after `wait_until_read` or a frame's end.
    pub unsafe fn run_stage_b_thread(capture: *mut Capture, memory: &mut impl PspMemory) -> StageBRun {
        let stream = &*core::ptr::addr_of!((*capture).walk_stream);
        let ended = &*core::ptr::addr_of!((*capture).stage_frame_ended);
        if stream.is_empty() || ended.load(Ordering::Acquire) {
            return StageBRun::Nothing;
        }
        let frame = (*capture).stage_frame.load(Ordering::Acquire);
        let mut memory = Overlaid { writes: &mut *core::ptr::addr_of_mut!((*capture).stage_writes), inner: memory };
        let recorder = Recorder {
            stats: &mut *core::ptr::addr_of_mut!((*capture).stage_stats),
            frame: frame.as_mut(),
            palette: &mut *core::ptr::addr_of_mut!((*capture).palette),
            draw_states: &mut *core::ptr::addr_of_mut!((*capture).draw_states),
            stream: &mut *core::ptr::addr_of_mut!((*capture).stream),
        };
        let mut visitors = Both(recorder, NoTrace);
        let actions = &*(*core::ptr::addr_of_mut!((*capture).stage_actions)).get_or_insert_with(|| PlainActions::for_visitor(&visitors));
        let ge = &mut *core::ptr::addr_of_mut!((*capture).stage_ge);
        if walk_stream::replay(stream, ge, &mut memory, &mut visitors, actions).is_some() {
            ended.store(true, Ordering::Release);
            return StageBRun::FrameEnded;
        }
        StageBRun::Worked
    }

    /// Whether textured 3D draws are left to the GPU's transform where the
    /// fast path takes them (their vertices then stay in model space, see
    /// `frame::GpuTransform`). Off by default; the display side must
    /// support it.
    pub fn set_gpu_transform(&mut self, on: bool) {
        self.walk_stream.wait_until_read();
        self.draw_states.gpu_transform = on;
    }

    pub fn gpu_transform(&self) -> bool {
        self.draw_states.gpu_transform
    }

    /// Gives the capture a vertex cache (decoded 3D vertices kept across
    /// frames, see `vertex_cache`); `None`: none. Only runs for the GPU
    /// transform use it.
    pub fn set_vertex_cache(&mut self, cache: Option<&'static mut crate::vertex_cache::VertexCache>) {
        self.walk_stream.wait_until_read();
        self.stream.set_vertex_cache(cache);
    }


    /// The draw stream's consumer from now on runs in another thread
    /// (`DrawStream::set_hooks`); `None`: the capture records the vertices
    /// itself.
    pub fn set_stream_hooks(&mut self, hooks: Option<&'static crate::draw_stream::StreamHooks>) {
        self.walk_stream.wait_until_read();
        self.stream.set_hooks(hooks);
    }

    /// See `DrawStream::set_reader`: the consumer thread copies the vertex
    /// memory itself.
    pub fn set_stream_reader(&mut self, reader: Option<crate::draw_stream::DeferredReader>) {
        self.walk_stream.wait_until_read();
        self.stream.set_reader(reader);
    }

    /// See `DrawStream::set_copier`: a copier thread copies the vertex
    /// memory as the draws are queued (with a reader, `set_stream_reader`).
    pub fn set_stream_copier(&mut self, copier: Option<&'static crate::draw_stream::CopierHooks>) {
        self.walk_stream.wait_until_read();
        self.stream.set_copier(copier);
    }

    /// The draw stream, for its consumer thread.
    pub fn stream(&self) -> &DrawStream {
        &self.stream
    }

    /// The draw stream's address without a reference to the capture (for
    /// a thread other than the capture's).
    ///
    /// # Safety
    /// `capture` must point to a `Capture`.
    pub unsafe fn stream_of(capture: *const Capture) -> *const DrawStream {
        core::ptr::addr_of!((*capture).stream)
    }

    pub fn frames_completed(&self) -> u32 {
        self.frames_completed
    }

    /// Reports ring events that were lost before we could read them.
    pub fn note_dropped_events(&mut self, count: u32) {
        self.current_frame.dropped_events += count;
    }

    /// Processes one event. Returns the finished frame's statistics when the
    /// event ends a frame.
    ///
    /// When `frame` is given, 2D draws are recorded into it for the Vita
    /// renderer; when the event ends a frame, `frame` is complete and the
    /// caller should pass a cleared one next time.
    pub fn handle(&mut self, event: Event, memory: &mut impl PspMemory, frame: Option<&mut Frame>) -> Option<FrameStats> {
        self.handle_traced(event, memory, frame, &mut NoTrace)
    }

    /// Like `handle`, but `tracer` also sees every command the GE executes,
    /// with the GE state after it. Used by the PC replay to explain a frame.
    pub fn handle_traced(
        &mut self,
        event: Event,
        memory: &mut impl PspMemory,
        frame: Option<&mut Frame>,
        tracer: &mut impl CommandVisitor,
    ) -> Option<FrameStats> {
        if self.pipelined && self.walk_stream.threaded() {
            let frame_pointer = frame.as_deref().map_or(core::ptr::null_mut(), |frame| frame as *const Frame as *mut Frame);
            self.stage_frame.store(frame_pointer, Ordering::Release);
        }
        match event {
            Event::ListEnqueued { list, stall, id, context } => {
                if id >= 0 {
                    self.enqueue(DisplayList::with_context(id, list, stall, context), false);
                }
            }
            Event::ListEnqueuedAtHead { list, stall, id, context } => {
                if id >= 0 {
                    self.enqueue(DisplayList::with_context(id, list, stall, context), true);
                }
            }
            Event::ContextSaved { address, ok } => {
                if ok {
                    self.contexts.save(address, &self.ge);
                    self.current_frame.contexts_saved += 1;
                }
            }
            Event::ContextRestored { address, ok } => {
                if ok {
                    self.restore_context(address);
                }
            }
            Event::StallUpdated { id, stall } => {
                self.current_frame.stall_updates += 1;
                if let Some(list) = self.queue[..self.queue_len].iter_mut().find(|list| list.id == id) {
                    list.set_stall(stall);
                }
            }
            Event::FramebufferShown { address, stride, pixel_format } => {
                let displayed = DisplayedFramebuffer { address, stride, pixel_format };
                if self.pipelined {
                    return Some(self.end_frame_pipelined(displayed, memory, frame, tracer));
                }
                return Some(self.end_frame(displayed, frame));
            }
        }
        self.run_queue(memory, frame, tracer);
        None
    }

    fn enqueue(&mut self, list: DisplayList, at_head: bool) {
        self.current_frame.lists_enqueued += 1;

        // The GE reuses a queue id once its list is done. If we still hold a
        // list with this id, we missed its end; drop it.
        if let Some(index) = self.queue[..self.queue_len].iter().position(|queued| queued.id == list.id) {
            self.remove(index);
            self.current_frame.lists_abandoned += 1;
        }

        if self.queue_len == QUEUE_CAPACITY {
            self.current_frame.queue_overflows += 1;
            return;
        }

        let position = if at_head { 0 } else { self.queue_len };
        self.queue.copy_within(position..self.queue_len, position + 1);
        self.queue[position] = list;
        self.queue_len += 1;
    }

    fn remove(&mut self, index: usize) {
        self.queue.copy_within(index + 1..self.queue_len, index);
        self.queue_len -= 1;
    }

    /// Walks the front list as far as possible, then the next ones.
    fn run_queue(&mut self, memory: &mut impl PspMemory, mut frame: Option<&mut Frame>, tracer: &mut impl CommandVisitor) {
        while self.queue_len > 0 {
            // A list with a context: the GE saves its state when the list
            // starts and restores it when it finishes (`ge_context`).
            let context = self.queue[0].context;
            if context != 0 && !self.queue[0].context_saved {
                self.contexts.save(context, &self.ge);
                self.queue[0].context_saved = true;
                self.current_frame.contexts_saved += 1;
            }
            // The walk, the recorder and the draw stream's copies of vertices
            // all see what block transfers wrote this frame.
            let overlaid = &mut Overlaid { writes: &mut self.ge_writes, inner: &mut *memory };
            let mut commands = 0;
            let result = if self.walk_only {
                walk(&mut self.queue[0], &mut self.ge, overlaid, &mut NoTrace, MAX_COMMANDS_PER_WALK, &mut commands)
            } else if self.pipelined {
                let mut writer = StreamWriter::new(&self.walk_stream, &mut self.write_filter);
                walk(&mut self.queue[0], &mut self.ge, overlaid, &mut writer, MAX_COMMANDS_PER_WALK, &mut commands)
            } else {
                let recorder = Recorder {
                    stats: &mut self.current_frame,
                    frame: frame.as_deref_mut(),
                    palette: &mut self.palette,
                    draw_states: &mut self.draw_states,
                    stream: &mut self.stream,
                };
                let mut visitors = Both(recorder, &mut *tracer);
                walk(&mut self.queue[0], &mut self.ge, overlaid, &mut visitors, MAX_COMMANDS_PER_WALK, &mut commands)
            };
            crate::phase::enter(crate::phase::OTHER);
            self.current_frame.commands += commands;
            match result {
                WalkResult::Paused => {
                    // The stream is full: stage B catches up, then the walk
                    // goes on.
                    self.run_stage_b(&mut *memory, frame.as_deref_mut(), tracer);
                    continue;
                }
                WalkResult::Stalled => {
                    if self.pipelined {
                        self.walk_stream.wake();
                    }
                    return;
                }
                WalkResult::Finished => {
                    self.current_frame.lists_finished += 1;
                    if context != 0 {
                        self.restore_context(context);
                    }
                }
                WalkResult::TooManyCommands => self.current_frame.endless_lists += 1,
                WalkResult::BadAddress(address) => {
                    self.current_frame.bad_addresses += 1;
                    self.current_frame.last_bad_address = address;
                }
            }
            self.remove(0);
        }
    }

    /// Stage B (single thread): works out the draws the stream holds,
    /// up to its end or the end of a frame (returned).
    fn run_stage_b(
        &mut self,
        memory: &mut impl PspMemory,
        frame: Option<&mut Frame>,
        tracer: &mut impl CommandVisitor,
    ) -> Option<DisplayedFramebuffer> {
        let mut memory = Overlaid { writes: &mut self.stage_writes, inner: memory };
        let recorder = Recorder {
            stats: &mut self.stage_stats,
            frame,
            palette: &mut self.palette,
            draw_states: &mut self.draw_states,
            stream: &mut self.stream,
        };
        let mut visitors = Both(recorder, tracer);
        let actions = PlainActions::for_visitor(&visitors);
        let shown = walk_stream::replay(&self.walk_stream, &mut self.stage_ge, &mut memory, &mut visitors, &actions);
        crate::phase::enter(crate::phase::OTHER);
        shown
    }

    /// `end_frame` with two stages: stage B finishes the frame, then the
    /// walk's counts join its statistics.
    fn end_frame_pipelined(
        &mut self,
        displayed: DisplayedFramebuffer,
        memory: &mut impl PspMemory,
        mut frame: Option<&mut Frame>,
        tracer: &mut impl CommandVisitor,
    ) -> FrameStats {
        self.walk_stream.push_end_frame(displayed);
        if self.walk_stream.threaded() {
            // Stage B (another thread) finishes the frame; then it waits.
            while !self.stage_frame_ended.load(Ordering::Acquire) {
                self.walk_stream.wake_and_wait();
            }
        } else {
            let shown = self.run_stage_b(memory, frame.as_deref_mut(), tracer);
            debug_assert_eq!(shown, Some(displayed));
        }
        let mut finished = self.stage_stats;
        self.stage_stats = FrameStats::new();
        self.finish_recording(&mut finished, displayed, frame);
        self.stage_writes.clear();

        let walk = &self.current_frame;
        finished.lists_enqueued = walk.lists_enqueued;
        finished.stall_updates = walk.stall_updates;
        finished.lists_finished = walk.lists_finished;
        finished.commands = walk.commands;
        finished.bad_addresses = walk.bad_addresses;
        finished.last_bad_address = walk.last_bad_address;
        finished.endless_lists = walk.endless_lists;
        finished.lists_abandoned = walk.lists_abandoned;
        finished.queue_overflows = walk.queue_overflows;
        finished.dropped_events = walk.dropped_events;
        finished.contexts_saved = walk.contexts_saved;
        finished.contexts_restored = walk.contexts_restored;
        finished.contexts_missing = walk.contexts_missing;
        finished.contexts_pipelined = walk.contexts_pipelined;
        finished.transfers_skipped = self.ge_writes.transfers_skipped;
        self.ge_writes.clear();
        finished.displayed = Some(displayed);
        finished.lists_pending = self.queue_len as u32;
        self.current_frame = FrameStats::new();
        self.frames_completed += 1;
        // Stage B may go on with the next frame's records.
        self.stage_frame_ended.store(false, Ordering::Release);
        finished
    }

    fn end_frame(&mut self, displayed: DisplayedFramebuffer, frame: Option<&mut Frame>) -> FrameStats {
        let mut stats = self.current_frame;
        self.finish_recording(&mut stats, displayed, frame);
        self.current_frame = stats;
        // The real GE has done this frame's transfers by now.
        self.current_frame.transfers_skipped = self.ge_writes.transfers_skipped;
        self.ge_writes.clear();
        let mut finished = self.current_frame;
        finished.displayed = Some(displayed);
        finished.lists_pending = self.queue_len as u32;
        self.current_frame = FrameStats::new();
        self.frames_completed += 1;
        finished
    }

    /// The recording's part of a frame's end: the draws still waiting for
    /// their vertices, the frame's fields, what doesn't carry over.
    fn finish_recording(&mut self, stats: &mut FrameStats, displayed: DisplayedFramebuffer, mut frame: Option<&mut Frame>) {
        // The frame's draws still waiting for their vertices.
        match frame.as_deref_mut() {
            Some(frame) => {
                self.stream.finish(stats);
                frame.merge_overflow();
                // Every palette's words are stored now.
                frame.resolve_palette_hashes();
            }
            None => self.stream.clear(),
        }
        if let Some(frame) = frame {
            frame.frame_number = self.frames_completed;
            frame.shown_framebuffer = physical(displayed.address);
            // The recorded frame's cached draws read the cache's arena; its
            // runs stay there while this frame may be shown.
            if let Some(cache) = self.stream.vertex_cache_mut() {
                frame.vertex_cache = cache.arena().as_ptr();
                frame.vertex_cache_len = cache.arena().len() as u32;
                // Its raw draws read the raw arena (`raw_arena`) likewise.
                if let Some(raw) = cache.raw.as_ref() {
                    frame.raw_vertices = raw.bytes().as_ptr();
                    frame.raw_vertices_len = raw.bytes().len() as u32;
                }
                // Its index runs as triangle lists, so the display doesn't
                // build them (~1 ms of its CPU per race frame, redrawbench).
                if let Some(ring) = cache.index_ring.as_mut() {
                    frame.write_ring_indices(ring);
                }
                stats.cache = cache.next_frame();
            }
        }
        // The next frame starts empty; nothing in it to reuse.
        self.draw_states.forget();
        self.palette.next_frame();
    }
}

/// What `Capture::run_stage_b_thread` did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageBRun {
    /// Nothing to do (the stream was empty, or stage B waits at a frame end).
    Nothing,
    /// Worked out what the stream held.
    Worked,
    /// Reached a frame's end: it waits now until stage A has finished the
    /// frame (memory read from here on belongs to the next frame).
    FrameEnded,
}

/// A tracer that ignores everything (plain `Capture::handle`).
struct NoTrace;

impl CommandVisitor for NoTrace {
    fn visit<M: PspMemory>(&mut self, _command: Command, _ge: &GeState, _memory: &mut M) {}

    #[inline]
    fn wants(&self, _opcode: u8) -> bool {
        false
    }

    #[inline]
    fn batches_prims(&self) -> bool {
        true
    }
}

/// Hands every command to two visitors, the first one first.
struct Both<A, B>(A, B);

impl<A: CommandVisitor, B: CommandVisitor> CommandVisitor for Both<A, B> {
    fn visit<M: PspMemory>(&mut self, command: Command, ge: &GeState, memory: &mut M) {
        if self.0.wants(command.opcode()) {
            self.0.visit(command, ge, memory);
        }
        if self.1.wants(command.opcode()) {
            self.1.visit(command, ge, memory);
        }
    }

    #[inline]
    fn wants(&self, opcode: u8) -> bool {
        self.0.wants(opcode) || self.1.wants(opcode)
    }

    #[inline]
    fn batches_prims(&self) -> bool {
        self.0.batches_prims() && self.1.batches_prims()
    }

    fn visit_prims<M: PspMemory>(&mut self, prims: &[Command], ge: &GeState, memory: &mut M) {
        if self.0.wants(op::PRIM) {
            self.0.visit_prims(prims, ge, memory);
        }
        if self.1.wants(op::PRIM) {
            self.1.visit_prims(prims, ge, memory);
        }
    }

    fn plain_run(&mut self, words: &[u32]) {
        self.0.plain_run(words);
        self.1.plain_run(words);
    }

    #[inline]
    fn should_pause(&self) -> bool {
        self.0.should_pause() || self.1.should_pause()
    }
}

impl<T: CommandVisitor> CommandVisitor for &mut T {
    fn visit<M: PspMemory>(&mut self, command: Command, ge: &GeState, memory: &mut M) {
        (**self).visit(command, ge, memory);
    }

    #[inline]
    fn wants(&self, opcode: u8) -> bool {
        (**self).wants(opcode)
    }

    #[inline]
    fn batches_prims(&self) -> bool {
        (**self).batches_prims()
    }

    fn visit_prims<M: PspMemory>(&mut self, prims: &[Command], ge: &GeState, memory: &mut M) {
        (**self).visit_prims(prims, ge, memory);
    }

    fn plain_run(&mut self, words: &[u32]) {
        (**self).plain_run(words);
    }

    #[inline]
    fn should_pause(&self) -> bool {
        (**self).should_pause()
    }
}

impl Default for Capture {
    fn default() -> Self {
        Capture::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::op;
    use crate::memory::test_memory::TestMemory;

    fn cmd(opcode: u8, data: u32) -> u32 {
        ((opcode as u32) << 24) | data
    }

    fn shown() -> Event {
        Event::FramebufferShown { address: 0x0400_0000, stride: 512, pixel_format: 3 }
    }

    #[test]
    fn counts_draws_of_a_stalled_list_as_it_grows() {
        let mut memory = TestMemory::default();
        let list = 0x0890_0000;
        memory.write(list, &[cmd(op::PRIM, 6), cmd(op::PRIM, 4), cmd(op::FINISH, 0), cmd(op::END, 0)]);

        let mut capture = Capture::new();
        capture.handle(Event::ListEnqueued { list, stall: list + 4, id: 1, context: 0 }, &mut memory, None);
        capture.handle(Event::StallUpdated { id: 1, stall: list + 16 }, &mut memory, None);
        let stats = capture.handle(shown(), &mut memory, None).expect("frame ends on flip");

        assert_eq!(stats.draws, 2);
        assert_eq!(stats.vertices, 10);
        assert_eq!(stats.lists_finished, 1);
        assert_eq!(stats.lists_pending, 0);
        assert_eq!(stats.stall_updates, 1);
    }

    #[test]
    fn second_list_waits_for_the_first() {
        let mut memory = TestMemory::default();
        let first = 0x0890_0000;
        let second = 0x0891_0000;
        memory.write(first, &[cmd(op::PRIM, 3), cmd(op::FINISH, 0), cmd(op::END, 0)]);
        memory.write(second, &[cmd(op::PRIM, 3), cmd(op::FINISH, 0), cmd(op::END, 0)]);

        let mut capture = Capture::new();
        // First list is stalled at its start, so nothing may run yet.
        capture.handle(Event::ListEnqueued { list: first, stall: first, id: 1, context: 0 }, &mut memory, None);
        capture.handle(Event::ListEnqueued { list: second, stall: 0, id: 2, context: 0 }, &mut memory, None);
        assert_eq!(capture.current_frame.draws, 0);

        capture.handle(Event::StallUpdated { id: 1, stall: 0 }, &mut memory, None);
        assert_eq!(capture.current_frame.draws, 2);
        assert_eq!(capture.current_frame.lists_finished, 2);
    }

    #[test]
    fn records_a_2d_frame_from_a_display_list() {
        use crate::frame::new_boxed_frame;

        let mut memory = TestMemory::default();
        let list = 0x0890_0000;
        let vertices = 0x0891_0000;
        // Two rectangles from one vertex buffer: the second PRIM continues
        // where the first stopped (vertex address auto-advance).
        let through_s16_color = (1 << 23) | (7 << 2) | (2 << 7); // 12-byte vertices
        memory.write(
            list,
            &[
                cmd(op::BASE, 0x08_0000),
                cmd(op::VERTEX_TYPE, through_s16_color),
                cmd(op::VERTEX_ADDRESS, 0x91_0000),
                cmd(op::FRAMEBUFFER_POINTER, 0x08_8000),
                cmd(op::PRIM, (6 << 16) | 2),
                cmd(op::PRIM, (6 << 16) | 2),
                cmd(op::FINISH, 0),
                cmd(op::END, 0),
            ],
        );
        for i in 0..4u32 {
            memory.write(vertices + 12 * i, &[0xFFFF_FFFF, i, 0]);
        }

        let mut frame = new_boxed_frame();
        let mut capture = Capture::new();
        capture.handle(Event::ListEnqueued { list, stall: 0, id: 0, context: 0 }, &mut memory, Some(&mut frame));
        let stats = capture.handle(shown(), &mut memory, Some(&mut frame)).unwrap();

        assert_eq!(stats.draws, 2);
        assert_eq!(frame.vertices().len(), 12);
        // Same state, consecutive vertices: merged into one draw.
        assert_eq!(frame.draws().len(), 1);
        assert_eq!(frame.draws()[0].framebuffer, 0x0408_8000);
        // The second rectangle used vertices 2 and 3.
        assert_eq!(frame.vertices()[6].x, 2.0);
        assert_eq!(frame.shown_framebuffer, 0x0400_0000);
    }

    /// Monster Hunter Freedom: one triangle (a 3-index strip) per PRIM, all
    /// from one vertex buffer. The PRIMs that follow each other are one job
    /// that copies only the vertices they use (lowest..highest index), and
    /// draw what the PRIMs one by one draw. Vertices 0-4 aren't readable
    /// here: a copy from vertex 0 (as before) would lose every triangle.
    #[test]
    fn indexed_prims_of_one_buffer_are_one_job_over_the_used_vertices() {
        let through_s16_color_u8_index = (1 << 23) | (7 << 2) | (2 << 7) | (1 << 11); // 12-byte vertices
        let vertices = 0x0891_0000;
        let indices = 0x0892_0000;
        let mut memory = TestMemory::default();
        for i in 5..14u32 {
            memory.write(vertices + 12 * i, &[0xFF00_0000 | i, (i << 16) | (10 * i), 0]);
        }
        memory.write(indices, &[u32::from_le_bytes([5, 6, 7, 11]), u32::from_le_bytes([12, 13, 8, 9]), u32::from_le_bytes([10, 0, 0, 0])]);
        let record = |memory: &mut TestMemory, separated: bool| {
            let list = 0x0890_0000;
            let mut words = vec![
                cmd(op::BASE, 0x08_0000),
                cmd(op::VERTEX_TYPE, through_s16_color_u8_index),
                cmd(op::VERTEX_ADDRESS, 0x91_0000),
                cmd(op::INDEX_ADDRESS, 0x92_0000),
                cmd(op::FRAMEBUFFER_POINTER, 0x08_8000),
            ];
            for _ in 0..3 {
                words.push(cmd(op::PRIM, (4 << 16) | 3));
                if separated {
                    words.push(cmd(op::NOP, 0));
                }
            }
            words.extend([cmd(op::FINISH, 0), cmd(op::END, 0)]);
            memory.write(list, &words);
            let mut frame = crate::frame::new_boxed_frame();
            let mut capture = Capture::new();
            capture.handle(Event::ListEnqueued { list, stall: 0, id: 0, context: 0 }, memory, Some(&mut frame));
            let stats = capture.handle(shown(), memory, Some(&mut frame)).unwrap();
            (stats.draws, stats.unreadable_draws, frame.vertices().to_vec(), frame.draws().len())
        };
        let batched = record(&mut memory, false);
        let one_by_one = record(&mut memory, true);
        assert_eq!(batched.0, 3);
        assert_eq!(batched.1, 0, "no triangle lost");
        assert_eq!(batched.2.len(), 9);
        assert_eq!(batched.2, one_by_one.2);
        assert_eq!(batched.3, one_by_one.3);
        // The second triangle is vertices 11, 12, 13.
        assert_eq!(batched.2[3].x, 110.0);
    }

    #[test]
    fn state_survives_save_and_load() {
        let mut memory = TestMemory::default();
        let list = 0x0890_0000;
        memory.write(list, &[cmd(op::BASE, 0x08_0000), cmd(op::PRIM, 3)]);

        let mut capture = Capture::new();
        // Stalls after the PRIM, so the list stays queued with BASE set.
        capture.handle(Event::ListEnqueued { list, stall: list + 8, id: 3, context: 0 }, &mut memory, None);

        let mut bytes = vec![0u8; 64 * 1024];
        let mut writer = Writer::new(&mut bytes);
        capture.save_state(&mut writer);
        assert!(!writer.overflowed());
        let len = writer.len();

        let mut restored = Capture::new();
        restored.load_state(&mut Reader::new(&bytes[..len])).unwrap();
        assert_eq!(restored.queue_len, 1);
        assert_eq!(restored.queue[0].id, 3);
        assert_eq!(restored.queue[0].pc, list + 8);
        assert_eq!(restored.ge.register(op::BASE), 0x08_0000);
    }

    #[test]
    fn a_call_after_a_block_transfer_reads_the_transferred_commands() {
        // As MTX's riders: a transfer copies commands from RAM into VRAM,
        // then the list CALLs them there.
        let mut memory = TestMemory::default();
        let list = 0x0890_0000;
        let source = 0x08A0_0000;
        let destination = 0x0415_4480;
        memory.write(source, &[cmd(op::PRIM, 5), cmd(op::RET, 0), 0, 0]);
        // What VRAM held before: a different draw.
        memory.write(destination, &[cmd(op::PRIM, 1), cmd(op::RET, 0), 0, 0]);
        let stride = 16; // pixels
        memory.write(
            list,
            &[
                cmd(op::TRANSFER_SOURCE, source & 0xFF_FFF0),
                cmd(op::TRANSFER_SOURCE_WIDTH, ((source >> 8) & 0xFF_0000) | stride),
                cmd(op::TRANSFER_DESTINATION, destination & 0xFF_FFF0),
                cmd(op::TRANSFER_DESTINATION_WIDTH, ((destination >> 8) & 0xFF_0000) | stride),
                cmd(op::TRANSFER_SOURCE_POSITION, 0),
                cmd(op::TRANSFER_DESTINATION_POSITION, 0),
                cmd(op::TRANSFER_SIZE, 3), // 4 x 1 pixels
                cmd(op::TRANSFER_START, 1), // 32-bit pixels: 16 bytes
                cmd(op::BASE, 0x04_0000),
                cmd(op::CALL, destination & 0xFF_FFFF),
                cmd(op::FINISH, 0),
                cmd(op::END, 0),
            ],
        );

        let mut capture = Capture::new();
        capture.handle(Event::ListEnqueued { list, stall: 0, id: 0, context: 0 }, &mut memory, None);
        let stats = capture.handle(shown(), &mut memory, None).unwrap();
        assert_eq!((stats.block_transfers, stats.draws, stats.vertices), (1, 1, 5));
        assert_eq!(stats.transfers_skipped, 0);
        // PSP memory itself is untouched, and the frame end forgot the copy.
        assert_eq!(memory.read_u32(destination), Some(cmd(op::PRIM, 1)));
        assert!(capture.ge_writes.is_empty());
    }

    #[test]
    fn two_stages_give_what_one_gives() {
        // A transfer writes commands, a CALL runs them (setting BASE and the
        // vertex address), a stall splits the list: what stage B must get
        // from the stream.
        let mut memory = TestMemory::default();
        let list = 0x0890_0000;
        let source = 0x08A0_0000;
        let destination = 0x0415_4480;
        let vertices = 0x0891_0000;
        let through_s16_color = (1 << 23) | (7 << 2) | (2 << 7);
        memory.write(
            source,
            &[
                cmd(op::BASE, 0x08_0000),
                cmd(op::VERTEX_TYPE, through_s16_color),
                cmd(op::VERTEX_ADDRESS, 0x91_0000),
                cmd(op::PRIM, (6 << 16) | 2),
                cmd(op::RET, 0),
                0,
                0,
                0,
            ],
        );
        for i in 0..4u32 {
            memory.write(vertices + 12 * i, &[0xFFFF_FFFF, i, 0]);
        }
        let stride = 16;
        memory.write(
            list,
            &[
                cmd(op::FRAMEBUFFER_POINTER, 0x08_8000),
                cmd(op::TRANSFER_SOURCE, source & 0xFF_FFF0),
                cmd(op::TRANSFER_SOURCE_WIDTH, ((source >> 8) & 0xFF_0000) | stride),
                cmd(op::TRANSFER_DESTINATION, destination & 0xFF_FFF0),
                cmd(op::TRANSFER_DESTINATION_WIDTH, ((destination >> 8) & 0xFF_0000) | stride),
                cmd(op::TRANSFER_SOURCE_POSITION, 0),
                cmd(op::TRANSFER_DESTINATION_POSITION, 0),
                cmd(op::TRANSFER_SIZE, 7), // 8 x 1 pixels of 32 bits: 8 words
                cmd(op::TRANSFER_START, 1),
                cmd(op::BASE, 0x04_0000),
                cmd(op::CALL, destination & 0xFF_FFFF),
                cmd(op::PRIM, (6 << 16) | 2),
                cmd(op::FINISH, 0),
                cmd(op::END, 0),
            ],
        );

        let mut results = Vec::new();
        for pipelined in [false, true] {
            let mut frame = crate::frame::new_boxed_frame();
            let mut capture = Capture::new();
            capture.set_pipelined(pipelined);
            capture.handle(Event::ListEnqueued { list, stall: list + 24, id: 0, context: 0 }, &mut memory, Some(&mut frame));
            capture.handle(Event::StallUpdated { id: 0, stall: 0 }, &mut memory, Some(&mut frame));
            let stats = capture.handle(shown(), &mut memory, Some(&mut frame)).unwrap();
            let xs: Vec<f32> = frame.vertices().iter().map(|vertex| vertex.x).collect();
            results.push((stats.draws, stats.commands, stats.block_transfers, stats.stall_updates, xs));
        }
        assert_eq!(results[0].0, 2);
        assert_eq!(results[0].4.len(), 12);
        assert_eq!(results[0], results[1]);
    }

    #[test]
    fn a_capture_made_in_zeroed_memory_records_what_a_new_one_records() {
        use crate::frame::new_boxed_frame;
        let mut memory = TestMemory::default();
        let list = 0x0890_0000;
        let vertices = 0x0891_0000;
        let through_s16_color = (1 << 23) | (7 << 2) | (2 << 7);
        memory.write(
            list,
            &[
                cmd(op::BASE, 0x08_0000),
                cmd(op::VERTEX_TYPE, through_s16_color),
                cmd(op::VERTEX_ADDRESS, 0x91_0000),
                cmd(op::FRAMEBUFFER_POINTER, 0x08_8000),
                cmd(op::PRIM, (6 << 16) | 2),
                cmd(op::FINISH, 0),
                cmd(op::END, 0),
            ],
        );
        for i in 0..2u32 {
            memory.write(vertices + 12 * i, &[0xFFFF_FFFF, i, 0]);
        }
        let layout = std::alloc::Layout::new::<Capture>();
        let mut results = Vec::new();
        for zeroed in [false, true] {
            // Safe: a zeroed block of a Capture's layout, ours alone.
            let mut capture: Box<Capture> = if zeroed {
                unsafe {
                    let place = std::alloc::alloc_zeroed(layout) as *mut Capture;
                    Capture::init_zeroed(place);
                    Box::from_raw(place)
                }
            } else {
                Box::new(Capture::new())
            };
            let mut frame = new_boxed_frame();
            capture.handle(Event::ListEnqueued { list, stall: 0, id: 0, context: 0 }, &mut memory, Some(&mut frame));
            let stats = capture.handle(shown(), &mut memory, Some(&mut frame)).unwrap();
            results.push((stats.draws, frame.description_hash(), frame.vertex_hash()));
        }
        assert_eq!(results[0].0, 1);
        assert_eq!(results[0], results[1]);
    }

    #[test]
    fn a_transfer_in_a_sub_list_that_rewrites_the_caller_is_seen_after_ret() {
        // The walk keeps the caller's command window during a CALL; a block
        // transfer in the sub-list rewrites the caller's next commands, which
        // must be read again after the RET.
        let mut memory = TestMemory::default();
        let list = 0x0890_0000;
        let sub = 0x0891_0000;
        let source = 0x08A0_0000;
        let destination = list + 16;
        memory.write(source, &[cmd(op::PRIM, 5), cmd(op::FINISH, 0), cmd(op::END, 0), 0]);
        memory.write(
            list,
            &[
                cmd(op::BASE, 0x08_0000),
                cmd(op::CALL, sub & 0xFF_FFFF),
                cmd(op::NOP, 0),
                cmd(op::NOP, 0),
                cmd(op::NOP, 0), // list + 16: replaced by the transfer
                cmd(op::FINISH, 0),
                cmd(op::END, 0),
            ],
        );
        let stride = 16;
        memory.write(
            sub,
            &[
                cmd(op::TRANSFER_SOURCE, source & 0xFF_FFF0),
                cmd(op::TRANSFER_SOURCE_WIDTH, ((source >> 8) & 0xFF_0000) | stride),
                cmd(op::TRANSFER_DESTINATION, destination & 0xFF_FFF0),
                cmd(op::TRANSFER_DESTINATION_WIDTH, ((destination >> 8) & 0xFF_0000) | stride),
                cmd(op::TRANSFER_SOURCE_POSITION, 0),
                cmd(op::TRANSFER_DESTINATION_POSITION, 0),
                cmd(op::TRANSFER_SIZE, 3), // 4 x 1 pixels
                cmd(op::TRANSFER_START, 1), // 32-bit pixels: 16 bytes
                cmd(op::RET, 0),
            ],
        );
        let mut capture = Capture::new();
        capture.handle(Event::ListEnqueued { list, stall: 0, id: 0, context: 0 }, &mut memory, None);
        let stats = capture.handle(shown(), &mut memory, None).unwrap();
        assert_eq!((stats.block_transfers, stats.draws, stats.vertices), (1, 1, 5));
    }

    #[test]
    fn a_transfer_too_big_to_follow_is_skipped_whole() {
        let mut memory = TestMemory::default();
        let list = 0x0890_0000;
        memory.write(
            list,
            &[
                cmd(op::TRANSFER_SOURCE, 0x00_0000),
                cmd(op::TRANSFER_SOURCE_WIDTH, 0x09_0000 | 512),
                cmd(op::TRANSFER_DESTINATION, 0x00_0000),
                cmd(op::TRANSFER_DESTINATION_WIDTH, 0x04_0000 | 512),
                cmd(op::TRANSFER_SIZE, (271 << 10) | 479), // a whole 480x272 screen
                cmd(op::TRANSFER_START, 1),
                cmd(op::FINISH, 0),
                cmd(op::END, 0),
            ],
        );
        let mut capture = Capture::new();
        capture.handle(Event::ListEnqueued { list, stall: 0, id: 0, context: 0 }, &mut memory, None);
        assert!(capture.ge_writes.is_empty());
        let stats = capture.handle(shown(), &mut memory, None).unwrap();
        assert_eq!(stats.transfers_skipped, 1);
    }

    #[test]
    fn reused_id_replaces_unfinished_list() {
        let mut memory = TestMemory::default();
        let list = 0x0890_0000;
        memory.write(list, &[cmd(op::PRIM, 3), cmd(op::FINISH, 0), cmd(op::END, 0)]);

        let mut capture = Capture::new();
        capture.handle(Event::ListEnqueued { list, stall: list, id: 1, context: 0 }, &mut memory, None);
        capture.handle(Event::ListEnqueued { list, stall: 0, id: 1, context: 0 }, &mut memory, None);
        let stats = capture.handle(shown(), &mut memory, None).unwrap();
        assert_eq!(stats.lists_abandoned, 1);
        assert_eq!(stats.draws, 1);
    }

    /// Monster Hunter Freedom after the name keyboard: the game's fog
    /// parameters were set once; a list run with a context (or between
    /// sceGeSaveContext and sceGeRestoreContext) zeroed them, and the PSP
    /// put them back. Without that the hunter was fully fogged (black).
    #[test]
    fn a_list_with_a_context_leaves_the_state_as_it_was() {
        let mut memory = TestMemory::default();
        let setup = 0x0890_0000;
        let utility = 0x0891_0000;
        let game = 0x0892_0000;
        memory.write(setup, &[cmd(op::FOG_END, 0x44_8000), cmd(op::FOG_SCALE, 0x3A_1000), cmd(op::FINISH, 0), cmd(op::END, 0)]);
        memory.write(utility, &[cmd(op::FOG_END, 0), cmd(op::FOG_SCALE, 0), cmd(op::FINISH, 0), cmd(op::END, 0)]);
        memory.write(game, &[cmd(op::PRIM, 3), cmd(op::FINISH, 0), cmd(op::END, 0)]);
        let mut capture = Capture::new();
        capture.handle(Event::ListEnqueued { list: setup, stall: 0, id: 0, context: 0 }, &mut memory, None);
        capture.handle(Event::ListEnqueued { list: utility, stall: 0, id: 1, context: 0x0990_0000 }, &mut memory, None);
        assert_eq!(capture.ge.register(op::FOG_END), 0x44_8000);
        assert_eq!(capture.ge.register(op::FOG_SCALE), 0x3A_1000);

        // The same by sceGeSaveContext / sceGeRestoreContext.
        capture.handle(Event::ContextSaved { address: 0x0990_1000, ok: true }, &mut memory, None);
        capture.handle(Event::ListEnqueued { list: utility, stall: 0, id: 2, context: 0 }, &mut memory, None);
        assert_eq!(capture.ge.register(op::FOG_END), 0);
        capture.handle(Event::ContextRestored { address: 0x0990_1000, ok: true }, &mut memory, None);
        assert_eq!(capture.ge.register(op::FOG_END), 0x44_8000);

        capture.handle(Event::ListEnqueued { list: game, stall: 0, id: 3, context: 0 }, &mut memory, None);
        let stats = capture.handle(shown(), &mut memory, None).unwrap();
        assert_eq!((stats.contexts_saved, stats.contexts_restored, stats.contexts_missing), (2, 2, 0));
    }

    /// A list with a context that stalls keeps its changes until it
    /// finishes, and saves the state only once (at its first walk).
    #[test]
    fn a_stalled_list_with_a_context_restores_at_its_finish() {
        let mut memory = TestMemory::default();
        let list = 0x0891_0000;
        memory.write(list, &[cmd(op::FOG_END, 0x11_0000), cmd(op::FOG_SCALE, 0), cmd(op::FINISH, 0), cmd(op::END, 0)]);
        let mut capture = Capture::new();
        capture.ge.set_register(Command(((op::FOG_END as u32) << 24) | 0x44_8000));
        capture.handle(Event::ListEnqueued { list, stall: list + 4, id: 1, context: 0x0990_0000 }, &mut memory, None);
        assert_eq!(capture.ge.register(op::FOG_END), 0x11_0000);
        capture.handle(Event::StallUpdated { id: 1, stall: list + 8 }, &mut memory, None);
        capture.handle(Event::StallUpdated { id: 1, stall: 0 }, &mut memory, None);
        assert_eq!(capture.ge.register(op::FOG_END), 0x44_8000);
        let stats = capture.handle(shown(), &mut memory, None).unwrap();
        assert_eq!((stats.contexts_saved, stats.contexts_restored), (1, 1));
    }
}
