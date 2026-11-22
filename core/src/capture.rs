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
use crate::frame_builder::FrameBuilder;
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

/// A physical address in VRAM or one of its mirrors.
fn is_vram(address: u32) -> bool {
    (0x0400_0000..0x0480_0000).contains(&physical(address))
}

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
    /// The GE's own registers and matrices, read by the plugin because we
    /// asked (`event_ring::EVENT_GE_STATE`): the block's address; `ok`: all
    /// reads succeeded.
    GeStateTaken { address: u32, ok: bool, psp_us: u32 },
}

use crate::event_ring::{
    EVENT_FRAMEBUFFER_FLIP, EVENT_GE_RESTORE_CONTEXT, EVENT_GE_SAVE_CONTEXT, EVENT_GE_STATE, EVENT_LIST_ENQUEUE,
    EVENT_LIST_ENQUEUE_HEAD, EVENT_STALL_UPDATE, GE_STATE_WORDS,
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
            EVENT_GE_STATE => Event::GeStateTaken { address: args[0], ok: args[2] == 0, psp_us: args[3] },
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
    /// SCREEN_OFFSET_X at the last frame's end (`FrameStats::screen_offset_x_changed`).
    last_screen_offset_x: u32,
    /// A new session started and the GE's own state hasn't come yet
    /// (`start_session`, `Event::GeStateTaken`).
    waiting_for_ge_state: bool,
    /// A walk stopping at a VRAM address keeps its list queued there
    /// (`vram_wait`) instead of dropping it: the Vita side gets the block
    /// copied and resumes the walk (`resume_after_vram`) or gives up
    /// (`give_up_vram`). Off in the PC replay (nothing would copy).
    pub vram_waits: bool,
    /// The VRAM address the front list's walk stopped at, waiting.
    vram_wait: Option<u32>,
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
    /// Writes the draws the GPU takes raw on this thread (`frame_builder`).
    builder: FrameBuilder,
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
            last_screen_offset_x: 0,
            waiting_for_ge_state: false,
            vram_waits: false,
            vram_wait: None,
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
            builder: FrameBuilder::new(),
        }
    }

    /// Forgets everything, e.g. when a new game starts.
    ///
    /// Done field by field on purpose: `*self = Capture::new()` would build a
    /// ~36 KiB temporary on the stack first, which overflowed the Vita
    /// capture thread's stack. Queue slots past `queue_len` are never read,
    /// so they don't need clearing.
    pub fn reset(&mut self) {
        self.forget(true);
    }

    /// A new session of the PSP plugin: forgets the lists, frames and
    /// palettes as `reset` does, but keeps the GE registers and matrices.
    ///
    /// A new session is either a game (re)start or an Adrenaline savestate
    /// load: the state restores the plugin's RAM with it, so the session
    /// number jumps too. After a load, Sony's GE has the registers of the
    /// moment the state was made, and the game never sends registers it sets
    /// once at boot again: Monster Hunter Freedom's scissor came out as
    /// 0,0-0,0 after a load (GeState::new), every draw clipped away, a black
    /// picture (2026-11-18). The registers we had are the closest we know.
    /// A game that really starts sets all its state again anyway.
    ///
    /// The GE's own state is asked for meanwhile (the Vita side writes the
    /// request into the ring) and replaces ours when it comes
    /// (`Event::GeStateTaken`); until then `waiting_for_ge_state`.
    pub fn start_session(&mut self) {
        self.forget(false);
        self.waiting_for_ge_state = true;
    }

    /// Since `start_session` or `expect_ge_state`, the GE's own state
    /// hasn't come yet: frames walked meanwhile may use wrong registers.
    pub fn waiting_for_ge_state(&self) -> bool {
        self.waiting_for_ge_state
    }

    /// The Vita side asked the plugin for the GE's state (`repairs_needed`).

    pub fn expect_ge_state(&mut self) {
        self.waiting_for_ge_state = true;
    }

    /// Takes over the GE's own state from the plugin's block at `address`.
    fn take_ge_state(&mut self, address: u32, memory: &mut impl PspMemory) {
        self.waiting_for_ge_state = false;
        let mut block = [0u32; GE_STATE_WORDS];
        if memory.read_words(address, &mut block).is_none() {
            self.current_frame.bad_addresses += 1;
            return;
        }
        let (registers, matrices) = self.ge.take_hardware_state(&block);
        self.current_frame.ge_states += 1;
        self.current_frame.ge_state_register_differences += registers;
        self.current_frame.ge_state_matrix_differences += matrices;
        if self.pipelined {
            // As a context restore: stage B's registers don't get it (yet).
            self.current_frame.contexts_pipelined += 1;
        }
    }

    fn forget(&mut self, registers_too: bool) {
        self.walk_stream.wait_until_read();
        self.queue_len = 0;
        if registers_too {
            self.ge = GeState::new();
        }
        self.palette.clear(); // in place: LoadedPalette holds 2 KiB of words
        self.draw_states.forget();
        self.stream.clear();
        self.restart_builder(!self.pipelined);
        self.ge_writes.clear();
        self.current_frame = FrameStats::new();
        self.frames_completed = 0;
        self.walk_stream.clear();
        // Stage B starts from the walk's registers (as in `set_pipelined`).
        if registers_too {
            self.stage_ge = GeState::new();
        } else {
            self.stage_ge.clone_from(&self.ge);
        }
        self.stage_writes.clear();
        self.stage_stats = FrameStats::new();
        self.write_filter.start(&self.stage_ge, crate::recorder::recorder_wants);
        self.stage_actions = None;
        self.contexts.clear();
        self.waiting_for_ge_state = false;
        self.vram_wait = None;
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
    /// on the stack first (it is ~0.5 MiB; a second capture, e.g. for
    /// timing a dumped frame, lives in a memory block of its own): what `reset`
    /// doesn't set is valid as zero bytes (counters, flags, null pointers as
    /// `None`; ring contents are written before they are read).
    ///
    /// # Safety
    /// `place` must point to `size_of::<Capture>()` zeroed, writable bytes,
    /// suitably aligned, used by nobody else.
    pub unsafe fn init_zeroed(place: *mut Capture) {
        // Not all-zero valid: zero bytes needn't mean `None` for its raw
        // arena (the layout of `Option<RawArena>` is the compiler's choice;
        // 2026-11-16 a zeroed one read as an empty arena: every raw run
        // "didn't fit"). Small, so written whole.
        core::ptr::addr_of_mut!((*place).builder).write(FrameBuilder::new());
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
        self.restart_builder(!on);
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
            builder: None,
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
        // The builder gives the old cache's raw arena back first.
        self.restart_builder(false);
        self.stream.set_vertex_cache(cache);
        self.restart_builder(!self.pipelined);
    }

    /// Starts the capture thread's builder (`frame_builder`) anew, writing
    /// raw draws if `wanted` (and it can): between frames only.
    fn restart_builder(&mut self, wanted: bool) {
        let active = self.builder.begin_frame(wanted, self.stream.vertex_cache_mut());
        self.stream.set_raw_tables_with_consumer(!active);
    }

    /// Gives the capture thread's builder its lists (`frame_builder`;
    /// `None`: none, every draw goes through the job queue).
    pub fn set_direct_lists(&mut self, lists: Option<&'static mut crate::frame_builder::DirectLists>) {
        self.walk_stream.wait_until_read();
        self.restart_builder(false);
        self.builder.set_lists(lists);
        self.restart_builder(!self.pipelined);
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
            Event::GeStateTaken { address, ok, psp_us } => {
                self.current_frame.ge_state_psp_us += psp_us;
                if ok {
                    self.take_ge_state(address, memory);
                } else {
                    self.waiting_for_ge_state = false;
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
                    builder: Some(&mut self.builder),
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
                    if self.vram_waits && !self.pipelined && !self.walk_only && is_vram(address) {
                        // Maybe a block not copied yet: the list stays at
                        // its pc (the command not read), see `vram_wait`.
                        self.vram_wait = Some(address);
                        return;
                    }
                    self.current_frame.bad_addresses += 1;
                    self.current_frame.last_bad_address = address;
                }
            }
            self.remove(0);
        }
    }

    /// The front list's walk stopped at this VRAM address and waits there
    /// (`vram_waits`); no event may be handled until `resume_after_vram`
    /// or `give_up_vram`.
    pub fn waiting_for_vram(&self) -> Option<u32> {
        self.vram_wait
    }

    /// The block the walk waited for is copied: the walk goes on from where
    /// it stopped (it may stop at another one).
    pub fn resume_after_vram(&mut self, memory: &mut impl PspMemory, frame: Option<&mut Frame>, waited_us: u32) {
        self.vram_wait = None;
        self.current_frame.vram_waits += 1;
        self.current_frame.vram_wait_us += waited_us;
        self.run_queue(memory, frame, &mut NoTrace);
    }

    /// No copy came (or none was possible): the list ends at the address,
    /// as a bad address, and the next one is walked.
    pub fn give_up_vram(&mut self, memory: &mut impl PspMemory, frame: Option<&mut Frame>, waited_us: u32) {
        let Some(address) = self.vram_wait.take() else {
            return;
        };
        self.current_frame.vram_wait_us += waited_us;
        self.current_frame.bad_addresses += 1;
        self.current_frame.last_bad_address = address;
        self.remove(0);
        self.run_queue(memory, frame, &mut NoTrace);
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
            builder: None,
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
        finished.ge_states = walk.ge_states;
        finished.ge_state_register_differences = walk.ge_state_register_differences;
        finished.ge_state_matrix_differences = walk.ge_state_matrix_differences;
        finished.ge_state_psp_us = walk.ge_state_psp_us;
        finished.vram_waits = walk.vram_waits;
        finished.vram_wait_us = walk.vram_wait_us;
        self.note_screen_offset_x(&mut finished);
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
        self.note_screen_offset_x(&mut finished);
        self.current_frame = FrameStats::new();
        self.frames_completed += 1;
        finished
    }

    /// SCREEN_OFFSET_X into the frame's stats when it changed since the
    /// last frame's end (`FrameStats::screen_offset_x_changed`; once per
    /// frame, not in the walk).
    fn note_screen_offset_x(&mut self, finished: &mut FrameStats) {
        let offset_x = self.ge.register(op::SCREEN_OFFSET_X);
        if offset_x != self.last_screen_offset_x {
            finished.screen_offset_x_changed = offset_x.max(1);
            self.last_screen_offset_x = offset_x;
        }
    }

    /// The recording's part of a frame's end: the draws still waiting for
    /// their vertices, the frame's fields, what doesn't carry over.
    fn finish_recording(&mut self, stats: &mut FrameStats, displayed: DisplayedFramebuffer, mut frame: Option<&mut Frame>) {
        // The frame's draws still waiting for their vertices.
        match frame.as_deref_mut() {
            Some(frame) => {
                self.stream.finish(stats);
                // The capture thread's raw draws and the recorder's, in
                // list order (`frame_builder`).
                self.builder.finish_frame(frame, &self.stream);
                stats.direct = self.builder.stats;
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
                // Its raw draws read the raw arena (`raw_arena`) likewise:
                // the vertex cache's, or the builder's while it holds it.
                if let Some(raw) = cache.raw.as_ref().or(self.builder.raw_arena()) {
                    frame.raw_vertices = raw.bytes().as_ptr();
                    frame.raw_vertices_len = raw.bytes().len() as u32;
                }
                // Its index runs as triangle lists, so the display doesn't
                // build them (~1 ms of its CPU per race frame, measured).
                let ring = match cache.index_ring.as_mut() {
                    Some(ring) => Some(ring),
                    None => self.builder.ring_mut(),
                };
                if let Some(ring) = ring {
                    frame.write_ring_indices(ring);
                }
                stats.cache = cache.next_frame();
                if let Some(raw) = self.builder.next_frame() {
                    stats.cache.note_raw(raw);
                }
            }
        }
        // The next frame starts empty; nothing in it to reuse. The builder
        // writes the raw draws of the next frame unless the capture runs in
        // two stages (`walk_stream`).
        self.restart_builder(!self.pipelined);
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

    #[inline]
    fn plain_runs_by_line(&self) -> bool {
        self.0.plain_runs_by_line() || self.1.plain_runs_by_line()
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

    #[inline]
    fn plain_runs_by_line(&self) -> bool {
        (**self).plain_runs_by_line()
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
    fn a_new_session_keeps_registers_set_once() {
        // MHF sets its scissor once at boot. After an Adrenaline savestate
        // load (a new plugin session) the game never sends it again: with
        // the registers forgotten, every draw was clipped to 0,0-0,0 and our
        // picture went black (2026-11-18).
        let mut memory = TestMemory::default();
        let list = 0x0890_0000;
        let scissor_end = (271 << 10) | 479;
        memory.write(list, &[cmd(op::SCISSOR_1, 0), cmd(op::SCISSOR_2, scissor_end), cmd(op::PRIM, 3)]);
        let mut capture = Capture::new();
        capture.handle(Event::ListEnqueued { list, stall: list + 12, id: 1, context: 0 }, &mut memory, None);
        assert_eq!(capture.ge.register(op::SCISSOR_2), scissor_end);

        capture.start_session();
        assert_eq!(capture.queue_len, 0, "the lists are forgotten");
        assert_eq!(capture.ge.register(op::SCISSOR_2), scissor_end, "the registers are kept");
        assert_eq!(capture.stage_ge.register(op::SCISSOR_2), capture.ge.register(op::SCISSOR_2));

        capture.reset();
        assert_eq!(capture.ge.register(op::SCISSOR_2), 0, "reset still forgets everything");
    }

    #[test]
    fn the_ges_own_state_replaces_ours_after_a_new_session() {
        // After a savestate load the registers we kept are the pre-load
        // area's (MHF: the village's fog colour in the river field, the
        // distance hazed over, 2026-11-18). The plugin's block of the GE's
        // own registers and matrices replaces them.
        use crate::event_ring::{GE_STATE_PROJECTION, GE_STATE_REGISTERS, GE_STATE_WORLD};
        let mut memory = TestMemory::default();
        let list = 0x0890_0000;
        memory.write(list, &[cmd(op::FOG_COLOR, 0x7F604F), cmd(op::PRIM, 3)]);
        let mut capture = Capture::new();
        capture.handle(Event::ListEnqueued { list, stall: list + 8, id: 1, context: 0 }, &mut memory, None);
        capture.start_session();
        assert!(capture.waiting_for_ge_state());

        let block_address = 0x0880_4000; // the plugin's buffer (kernel RAM)
        let mut block = [0u32; GE_STATE_WORDS];
        // As sceGeGetCmd gives them: with the opcode in the top byte.
        block[GE_STATE_REGISTERS + op::FOG_COLOR as usize] = cmd(op::FOG_COLOR, 0x606050);
        block[GE_STATE_REGISTERS + op::SCISSOR_2 as usize] = cmd(op::SCISSOR_2, (271 << 10) | 479);
        block[GE_STATE_REGISTERS + op::PRIM as usize] = cmd(op::PRIM, 0x123); // acts: left alone
        for scale in [op::TEXTURE_SCALE_U, op::TEXTURE_SCALE_V] {
            block[GE_STATE_REGISTERS + scale as usize] = cmd(scale, 0x3F_8000); // 1.0, as ours
        }
        block[GE_STATE_WORLD] = 1.0f32.to_bits() >> 8; // 24-bit floats
        block[GE_STATE_PROJECTION + 15] = 2.0f32.to_bits() >> 8;
        memory.write(block_address, &block);
        let matrices_before = capture.ge.matrix_generation();
        capture.handle(Event::GeStateTaken { address: block_address, ok: true, psp_us: 0 }, &mut memory, None);

        assert!(!capture.waiting_for_ge_state());
        assert_eq!(capture.ge.register(op::FOG_COLOR), 0x606050);
        assert_eq!(capture.ge.register(op::SCISSOR_2), (271 << 10) | 479);
        assert_eq!(capture.ge.register(op::PRIM), 3, "a drawing command's register isn't taken over");
        assert_eq!(capture.ge.matrices.world[0], 1.0);
        assert_eq!(capture.ge.matrices.projection[15], 2.0);
        assert_ne!(capture.ge.matrix_generation(), matrices_before, "transforms are worked out again");
        let stats = capture.current_frame;
        assert_eq!(stats.ge_states, 1);
        assert_eq!(stats.ge_state_register_differences, 2, "fog colour and scissor differed");
        assert_eq!(stats.ge_state_matrix_differences, 2);
    }

    #[test]
    fn a_walk_stopped_early_misses_a_register_until_the_ges_state_comes() {
        // MHF's river field turned upside down for good (2026-11-18): the
        // walk stopped at a bad address (a VRAM sub-list not copied) and
        // missed the command that set the viewport back after a pass with a
        // flipped one; the game never sends it again. The frame says so
        // (`walk_stopped_early`) and the GE's own state puts it right.
        use crate::event_ring::GE_STATE_REGISTERS;
        let flipped = 0x42_4140; // +48.3 as a 24-bit float
        let normal = 0xC3_0800; // -136
        let mut memory = TestMemory::default();
        let list = 0x0890_0000;
        // The flipped pass, then a jump into memory we can't read: the
        // reset that follows there is the GE's alone.
        memory.write(list, &[cmd(op::VIEWPORT_Y_SCALE, flipped), cmd(op::JUMP, 0x30_0000)]);
        let mut capture = Capture::new();
        capture.handle(Event::ListEnqueued { list, stall: 0, id: 1, context: 0 }, &mut memory, None);
        let stats = capture.handle(shown(), &mut memory, None).expect("a frame");
        assert!(stats.walk_stopped_early());
        assert_eq!(capture.ge.register(op::VIEWPORT_Y_SCALE), flipped);

        capture.expect_ge_state();
        let block_address = 0x0880_4000;
        let mut block = [0u32; GE_STATE_WORDS];
        block[GE_STATE_REGISTERS + op::VIEWPORT_Y_SCALE as usize] = cmd(op::VIEWPORT_Y_SCALE, normal);
        for scale in [op::TEXTURE_SCALE_U, op::TEXTURE_SCALE_V] {
            block[GE_STATE_REGISTERS + scale as usize] = cmd(scale, 0x3F_8000);
        }
        memory.write(block_address, &block);
        capture.handle(Event::GeStateTaken { address: block_address, ok: true, psp_us: 40 }, &mut memory, None);
        assert!(!capture.waiting_for_ge_state());
        assert_eq!(capture.ge.register(op::VIEWPORT_Y_SCALE), normal);
        assert_eq!(capture.current_frame.ge_state_register_differences, 1);
        assert_eq!(capture.current_frame.ge_state_psp_us, 40);
    }

    /// A list that CALLs a sublist in VRAM (as MHF's models do), which
    /// sets FOG_COLOR; the sublist isn't in memory yet.
    fn list_calling_into_vram(memory: &mut TestMemory) -> (u32, u32) {
        let list = 0x0890_0000;
        let sublist = 0x0410_0000;
        memory.write(list, &[cmd(op::BASE, 0x04_0000), cmd(op::CALL, 0x10_0000), cmd(op::FINISH, 0), cmd(op::END, 0)]);
        (list, sublist)
    }

    #[test]
    fn a_walk_waits_at_a_vram_block_and_goes_on_once_it_is_copied() {
        // MHF's field: new content CALLs a sublist in a VRAM block never
        // requested; the walk stopped there (bad address) for this frame
        // and the next, and missed the rest of the list (2026-11-18).
        let mut memory = TestMemory::default();
        let (list, sublist) = list_calling_into_vram(&mut memory);
        let mut capture = Capture::new();
        capture.vram_waits = true;
        capture.handle(Event::ListEnqueued { list, stall: 0, id: 1, context: 0 }, &mut memory, None);
        assert_eq!(capture.waiting_for_vram(), Some(sublist));
        assert_eq!(capture.queue_len, 1, "the list stays queued where it stopped");

        // The plugin copied the block.
        memory.write(sublist, &[cmd(op::FOG_COLOR, 0x606050), cmd(op::RET, 0)]);
        capture.resume_after_vram(&mut memory, None, 1200);
        assert_eq!(capture.waiting_for_vram(), None);
        assert_eq!(capture.ge.register(op::FOG_COLOR), 0x606050);
        let stats = capture.handle(shown(), &mut memory, None).expect("a frame");
        assert_eq!((stats.bad_addresses, stats.lists_finished), (0, 1));
        assert_eq!((stats.vram_waits, stats.vram_wait_us), (1, 1200));
    }

    #[test]
    fn a_walk_that_waited_in_vain_ends_at_the_block_as_before() {
        let mut memory = TestMemory::default();
        let (list, sublist) = list_calling_into_vram(&mut memory);
        let mut capture = Capture::new();
        capture.vram_waits = true;
        capture.handle(Event::ListEnqueued { list, stall: 0, id: 1, context: 0 }, &mut memory, None);
        capture.give_up_vram(&mut memory, None, 40_000);
        assert_eq!(capture.queue_len, 0);
        let stats = capture.handle(shown(), &mut memory, None).expect("a frame");
        assert_eq!((stats.bad_addresses, stats.last_bad_address), (1, sublist));
        assert_eq!((stats.vram_waits, stats.vram_wait_us), (0, 40_000));
        assert!(stats.walk_stopped_early(), "the GE's state is asked for then");
    }

    #[test]
    fn without_vram_waits_a_walk_stops_at_the_block() {
        // The PC replay and plugins that can't copy at the GE sync.
        let mut memory = TestMemory::default();
        let (list, _) = list_calling_into_vram(&mut memory);
        let mut capture = Capture::new();
        capture.handle(Event::ListEnqueued { list, stall: 0, id: 1, context: 0 }, &mut memory, None);
        assert_eq!(capture.waiting_for_vram(), None);
        assert_eq!(capture.queue_len, 0);
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

    /// 2026-11-16 (performance refactor step 2, first Vita run): the Vita
    /// makes the capture in zeroed memory, where the builder's
    /// `Option<RawArena>` read as an empty arena instead of none. The builder
    /// kept it (it held "an arena" already) and every raw run of the MTX
    /// title went to the CPU as "arena full". A zeroed capture must hold no
    /// arena and take the vertex cache's real one.
    #[test]
    fn a_zeroed_capture_takes_the_vertex_caches_raw_arena() {
        let layout = std::alloc::Layout::new::<Capture>();
        // Safe: a zeroed block of a Capture's layout, ours alone.
        let mut capture: Box<Capture> = unsafe {
            let place = std::alloc::alloc_zeroed(layout) as *mut Capture;
            Capture::init_zeroed(place);
            Box::from_raw(place)
        };
        assert!(capture.builder.raw_arena().is_none(), "no arena before the vertex cache brings one");
        crate::raw_arena::ENABLED.store(true, Ordering::Relaxed);
        let arena = Box::leak(vec![crate::frame::FrameVertex::default(); 16].into_boxed_slice());
        let cache = Box::leak(Box::new(crate::vertex_cache::VertexCache::new(arena)));
        cache.set_raw_arena(Box::leak(vec![0u8; crate::raw_arena::WINDOW_BYTES as usize].into_boxed_slice()));
        capture.set_vertex_cache(Some(cache));
        capture.set_direct_lists(Some(crate::frame_builder::DirectLists::boxed()));
        let raw = capture.builder.raw_arena().expect("the builder holds the cache's arena");
        assert_eq!(raw.bytes().len(), crate::raw_arena::WINDOW_BYTES as usize);
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
    fn big_uploads_into_one_vram_area_are_followed_in_order() {
        // God of War: Chains of Olympus streams ~530 KB of sub-lists and
        // vertices per frame through one VRAM area (0x041CC95C): a block
        // transfer from RAM (512-byte rows, up to 467 of them), then a CALL
        // into the area, then the next upload over it (2026-11-22). Each CALL
        // must see the upload right before it, however big.
        let mut memory = TestMemory::default();
        let list = 0x0890_0000;
        let (first, second) = (0x0970_0000, 0x0972_0000);
        let area = 0x041C_C95C;
        let rows = 80; // 40 KiB, more than the 32 blocks copies may take
        memory.write(first, &[cmd(op::PRIM, (3 << 16) | 5), cmd(op::RET, 0)]);
        memory.write(first + 8, &vec![0; (rows * 512 / 4 - 2) as usize]);
        memory.write(second, &[cmd(op::PRIM, (3 << 16) | 7), cmd(op::RET, 0)]);
        memory.write(second + 8, &vec![0; (rows * 512 / 4 - 2) as usize]);
        let upload = |source: u32| {
            [
                cmd(op::TRANSFER_SOURCE, source & 0xFF_FFF0),
                cmd(op::TRANSFER_SOURCE_WIDTH, ((source >> 8) & 0xFF_0000) | 128),
                cmd(op::TRANSFER_DESTINATION, area & 0xFF_FFF0),
                cmd(op::TRANSFER_DESTINATION_WIDTH, ((area >> 8) & 0xFF_0000) | 128),
                cmd(op::TRANSFER_SOURCE_POSITION, 0),
                cmd(op::TRANSFER_DESTINATION_POSITION, (area & 0xF) / 4), // x in 32-bit pixels
                cmd(op::TRANSFER_SIZE, ((rows - 1) << 10) | 127), // 128 x rows pixels
                cmd(op::TRANSFER_START, 1),
            ]
        };
        let mut commands = vec![cmd(op::BASE, 0x04_0000)];
        commands.extend(upload(first));
        commands.push(cmd(op::CALL, area & 0xFF_FFFF));
        commands.extend(upload(second));
        commands.push(cmd(op::CALL, area & 0xFF_FFFF));
        commands.extend([cmd(op::FINISH, 0), cmd(op::END, 0)]);
        memory.write(list, &commands);

        let mut capture = Capture::new();
        capture.handle(Event::ListEnqueued { list, stall: 0, id: 0, context: 0 }, &mut memory, None);
        let stats = capture.handle(shown(), &mut memory, None).unwrap();
        assert_eq!(stats.transfers_skipped, 0);
        assert_eq!((stats.block_transfers, stats.draws, stats.vertices), (2, 2, 12));
    }

    #[test]
    fn gameplay_uploads_past_the_titles_count_are_followed() {
        // God of War in the game (2026-11-22): 351 uploads per frame, most of
        // them 512-byte rows into one scratch area (0x0418A000) the next CALL
        // runs, object after object. After the 192nd the redirects ran out,
        // the uploads were skipped and the CALLs walked stale VRAM: vertex
        // floats as commands, 1.1M "vertices", ~120 ms walks.
        let mut memory = TestMemory::default();
        let list = 0x0890_0000;
        let area = 0x0418_A000;
        let uploads = 400u32;
        // Row counts (512 bytes each) as in the frame's first uploads.
        let row_counts = [38u32, 8, 5, 17, 4, 2, 105, 1, 43, 127];
        let mut commands = vec![cmd(op::BASE, 0x04_0000)];
        let mut vertices = 0;
        let mut next_source = 0x0900_0000;
        for index in 0..uploads {
            // Each object's sub-list in RAM: one PRIM with its own count.
            let rows = row_counts[index as usize % row_counts.len()];
            let source = next_source;
            next_source += rows * 512;
            let count = 3 + index % 5;
            vertices += count;
            memory.write(source, &[cmd(op::PRIM, (3 << 16) | count), cmd(op::RET, 0)]);
            memory.write(source + 8, &vec![0; (rows * 512 / 4 - 2) as usize]);
            commands.extend([
                cmd(op::TRANSFER_SOURCE, source & 0xFF_FFF0),
                cmd(op::TRANSFER_SOURCE_WIDTH, ((source >> 8) & 0xFF_0000) | 128),
                cmd(op::TRANSFER_DESTINATION, area & 0xFF_FFF0),
                cmd(op::TRANSFER_DESTINATION_WIDTH, ((area >> 8) & 0xFF_0000) | 128),
                cmd(op::TRANSFER_SOURCE_POSITION, 0),
                cmd(op::TRANSFER_DESTINATION_POSITION, 0),
                cmd(op::TRANSFER_SIZE, ((rows - 1) << 10) | 127),
                cmd(op::TRANSFER_START, 1),
                cmd(op::CALL, area & 0xFF_FFFF),
            ]);
        }
        commands.extend([cmd(op::FINISH, 0), cmd(op::END, 0)]);
        memory.write(list, &commands);

        let mut capture = Capture::new();
        capture.handle(Event::ListEnqueued { list, stall: 0, id: 0, context: 0 }, &mut memory, None);
        let stats = capture.handle(shown(), &mut memory, None).unwrap();
        assert_eq!(stats.transfers_skipped, 0);
        assert_eq!((stats.block_transfers, stats.draws, stats.vertices), (uploads, uploads, vertices));
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
