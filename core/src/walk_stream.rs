//! The capture split in two stages that can run on two cores: stage A
//! follows the display lists, stage B works out the draws.
//!
//! Following a list (`display_list::walk`) is mostly reading commands and
//! setting GE registers; working out each draw (`recorder`: draw state,
//! textures, palettes, queuing its vertices) needs the GE state at that
//! draw. Stage A walks the lists with its own `GeState` and writes what it
//! executed into a [`WalkStream`]; stage B reads the stream, applies the
//! same commands to its own `GeState` and hands them to the recorder
//! exactly as the walk would have ([`replay`]). Stage B never needs the
//! control flow (JUMP, CALL, RET, signals) or the list addresses, so the
//! stream holds only the executed commands, in execution order, plus what
//! stage B can't work out on its own:
//!
//! - VERTEX_ADDRESS / INDEX_ADDRESS: the resolved address (it depends on
//!   BASE and the offset, which CALL/RET and ORIGIN change at list
//!   addresses only stage A knows);
//! - RET and END: the BASE register afterwards (a signal return restores it).
//!
//! Plain register writes that leave a register as it is are left out
//! ([`WriteFilter`]): stage B would only store the same value again, and
//! doing every write twice was most of stage B's extra work (2026-11-06).
//!
//! Block transfers (TRANSFER_START) are done by both stages, each into its
//! own `GeWrites`: stage A needs them to read commands a transfer wrote,
//! stage B to read the vertices and palettes it wrote, at the same point of
//! the frame (stage B may be a long way behind stage A).
//!
//! The stream is a ring of words written by one thread and read by another
//! (or by the same thread, see `WalkStream::needs_draining`), with the same
//! publish/consume handover as `draw_stream`. A record is a header word
//! (kind in the top byte, length in the rest) and its words; a record
//! never wraps around the ring's end (a `PAD` header skips the rest).

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicU32, Ordering};

use crate::command::{op, Command};
use crate::display_list::{block_transfer, CommandVisitor, PlainAction, PlainActions, PLAIN_COMMANDS};
use crate::ge_state::GeState;
use crate::memory::PspMemory;
use crate::stats::DisplayedFramebuffer;

/// Ring size in words (32 KiB). A frame of MTX's Los Angeles race is
/// ~100,000 commands; the stages hand them over in pieces.
pub const STREAM_WORDS: usize = 8 * 1024;

/// The longest record: a plain run of a whole cache line
/// (`block_cache::LINE_SIZE`, 64 commands) plus its header.
const MAX_RECORD_WORDS: usize = 1 + 64;

/// Stage A pauses (single thread) or waits (two threads) when less than
/// this is free: one step of the walk writes at most a plain run and a
/// special command or a PRIM batch, each maybe after a `PAD`.
const ROOM_FOR_ONE_STEP: usize = 4 * MAX_RECORD_WORDS;

// Record kinds (the header's top byte).
/// Plain commands: as `run_plain_commands` executed them. Length = words.
const RUN: u32 = 1;
/// A command `walk` handles itself: the command word and one extra word
/// (see the module documentation).
const SPECIAL: u32 = 2;
/// PRIMs that directly follow each other (`CommandVisitor::visit_prims`).
/// Length = PRIM count.
const PRIMS: u32 = 3;
/// The game showed a frame: address, stride, pixel format.
const END_FRAME: u32 = 4;
/// Nothing more up to the ring's end.
const PAD: u32 = 5;

fn header(kind: u32, len: usize) -> u32 {
    (kind << 24) | len as u32
}

/// How stage A reaches a stage B in another thread (as `StreamHooks`).
pub struct WalkHooks {
    /// Tells stage B that records were published.
    pub wake_reader: fn(),
    /// Waits a little (for stage B to make room).
    pub wait: fn(),
}

/// A counter on a cache line of its own (the Vita's are 32 bytes; 64 for
/// bigger CPUs): stage A's and stage B's counters side by side made every
/// store of one thread throw the other's copy out of its core's cache.
#[repr(align(64))]
struct OwnLine(AtomicU32);

impl core::ops::Deref for OwnLine {
    type Target = AtomicU32;
    fn deref(&self) -> &AtomicU32 {
        &self.0
    }
}

/// How often stage B tells stage A how far it has read (in words): after
/// every record made the counter's cache line go back and forth between the
/// cores for every record.
const READ_PUBLISH_WORDS: u32 = 256;

/// The ring between the stages (see the module documentation).
pub struct WalkStream {
    words: UnsafeCell<[u32; STREAM_WORDS]>,
    /// Words written by stage A (positions count up for ever, wrapping).
    /// Only stage A writes it.
    written: OwnLine,
    /// Words stage B has read and is done with (published every
    /// `READ_PUBLISH_WORDS` and when it stops). Only stage B writes it.
    read: OwnLine,
    /// Stage B in another thread; `None`: the walking thread drains the
    /// stream itself whenever it fills up (`needs_draining`).
    hooks: Option<&'static WalkHooks>,
}

// Safe: stage A writes only words stage B has read (`free`), stage B reads
// only published ones; the handover goes through `written` / `read`.
unsafe impl Sync for WalkStream {}

impl WalkStream {
    pub const fn new() -> WalkStream {
        WalkStream {
            words: UnsafeCell::new([0; STREAM_WORDS]),
            written: OwnLine(AtomicU32::new(0)),
            read: OwnLine(AtomicU32::new(0)),
            hooks: None,
        }
    }

    /// Stage B runs in another thread from now on (`None`: in the walking
    /// thread). Only while the stream is empty.
    pub fn set_hooks(&mut self, hooks: Option<&'static WalkHooks>) {
        debug_assert!(self.is_empty());
        self.hooks = hooks;
    }

    /// Forgets everything in the stream (no stage B may be reading).
    pub fn clear(&mut self) {
        let written = self.written.load(Ordering::Relaxed);
        self.read.store(written, Ordering::Relaxed);
    }

    /// Whether stage B runs in another thread (`set_hooks`).
    pub fn threaded(&self) -> bool {
        self.hooks.is_some()
    }

    /// Two threads: wakes stage B and waits a little (a spin could keep
    /// stage B from running on the same core).
    pub fn wake_and_wait(&self) {
        if let Some(hooks) = self.hooks {
            (hooks.wake_reader)();
            (hooks.wait)();
        }
    }

    /// Two threads: waits until stage B has read everything.
    pub fn wait_until_read(&self) {
        if let Some(hooks) = self.hooks {
            while !self.is_empty() {
                (hooks.wake_reader)();
                (hooks.wait)();
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.read.load(Ordering::Acquire) == self.written.load(Ordering::Acquire)
    }

    fn free(&self) -> usize {
        let used = self.written.load(Ordering::Relaxed).wrapping_sub(self.read.load(Ordering::Acquire));
        STREAM_WORDS - used as usize
    }

    /// Single thread: stage A should stop and let stage B empty the stream.
    pub fn needs_draining(&self) -> bool {
        self.hooks.is_none() && self.free() < ROOM_FOR_ONE_STEP
    }

    // ---- Stage A ----

    /// Writes one record: the header, then `parts` one after the other.
    fn push(&self, kind: u32, len: usize, parts: &[&[u32]]) {
        let size = 1 + parts.iter().map(|part| part.len()).sum::<usize>();
        debug_assert!(size <= MAX_RECORD_WORDS);
        // Two threads: wait for room (single thread: `needs_draining` made
        // sure there is room before this step of the walk).
        if let Some(hooks) = self.hooks {
            while self.free() < 2 * MAX_RECORD_WORDS {
                (hooks.wake_reader)();
                (hooks.wait)();
            }
        }
        let mut position = self.written.load(Ordering::Relaxed);
        // Safe: the words from `written` on are free (`free`); stage B
        // doesn't read them until they are published below.
        let words = unsafe { &mut *self.words.get() };
        let index = position as usize % STREAM_WORDS;
        if index + size > STREAM_WORDS {
            words[index] = header(PAD, 0);
            position = position.wrapping_add((STREAM_WORDS - index) as u32);
        }
        let mut index = position as usize % STREAM_WORDS;
        words[index] = header(kind, len);
        index += 1;
        for part in parts {
            words[index..index + part.len()].copy_from_slice(part);
            index += part.len();
        }
        self.written.store(position.wrapping_add(size as u32), Ordering::Release);
    }

    /// The game showed a frame: stage B finishes it when it gets here.
    pub fn push_end_frame(&self, displayed: DisplayedFramebuffer) {
        self.push(END_FRAME, 0, &[&[displayed.address, displayed.stride, displayed.pixel_format]]);
        self.wake();
    }

    /// Tells stage B (another thread) that there is something to read.
    pub fn wake(&self) {
        if let Some(hooks) = self.hooks {
            (hooks.wake_reader)();
        }
    }
}

impl Default for WalkStream {
    fn default() -> Self {
        WalkStream::new()
    }
}

/// Which plain register writes stage A may leave out, and the register
/// values stage B has (as far as those writes go).
pub struct WriteFilter {
    /// Opcodes stage B only stores (not wanted by the recorder, not matrix
    /// uploads): a write of the value it already has changes nothing.
    droppable: [bool; 256],
    /// Stage B's registers: its `GeState` at the point the stream has
    /// reached. Kept for the droppable opcodes and BASE (which RET and END
    /// restore through their extra word).
    registers: [u32; 256],
    /// Plain writes passed on and left out (for measuring).
    pub sent: u32,
    pub dropped: u32,
}

impl WriteFilter {
    pub const fn new() -> Self {
        WriteFilter { droppable: [false; 256], registers: [0; 256], sent: 0, dropped: 0 }
    }

    /// Stage B starts from `ge` (its state equals stage A's now); `wants`:
    /// the opcodes stage B's visitor wants.
    pub fn start(&mut self, ge: &GeState, wants: impl Fn(u8) -> bool) {
        for opcode in 0..=255u8 {
            self.droppable[opcode as usize] =
                PLAIN_COMMANDS[opcode as usize] && GeState::same_value_is_no_change(opcode) && !wants(opcode);
        }
        self.registers = *ge.registers();
    }
}

impl Default for WriteFilter {
    fn default() -> Self {
        WriteFilter::new()
    }
}

/// Stage A's visitor: writes what the walk executes into the stream.
pub struct StreamWriter<'a> {
    pub stream: &'a WalkStream,
    filter: &'a mut WriteFilter,
    /// Records since stage B (another thread) was last woken.
    unwoken: u32,
}

impl<'a> StreamWriter<'a> {
    pub fn new(stream: &'a WalkStream, filter: &'a mut WriteFilter) -> Self {
        StreamWriter { stream, filter, unwoken: 0 }
    }

    fn written(&mut self) {
        // A wake-up is a system call; stage B goes on by itself while the
        // stream has records, and is woken at stalls and frame ends anyway.
        self.unwoken += 1;
        if self.unwoken == 256 {
            self.unwoken = 0;
            self.stream.wake();
        }
    }
}

impl CommandVisitor for StreamWriter<'_> {
    /// Everything `walk` doesn't treat as a plain command (those come
    /// through `plain_run`).
    #[inline]
    fn wants(&self, opcode: u8) -> bool {
        !PLAIN_COMMANDS[opcode as usize]
    }

    #[inline]
    fn batches_prims(&self) -> bool {
        true
    }

    fn visit<M: PspMemory>(&mut self, command: Command, ge: &GeState, _memory: &mut M) {
        let extra = match command.opcode() {
            op::VERTEX_ADDRESS => ge.vertex_address,
            op::INDEX_ADDRESS => ge.index_address,
            op::RET | op::END => {
                // Stage B takes this BASE (`apply_special`).
                self.filter.registers[op::BASE as usize] = ge.register(op::BASE);
                ge.register(op::BASE)
            }
            _ => 0,
        };
        self.stream.push(SPECIAL, 0, &[&[command.0, extra]]);
        self.written();
    }

    fn visit_prims<M: PspMemory>(&mut self, prims: &[Command], _ge: &GeState, _memory: &mut M) {
        // Safe: `Command` is a `u32` (repr(transparent)).
        let words = unsafe { core::slice::from_raw_parts(prims.as_ptr() as *const u32, prims.len()) };
        self.stream.push(PRIMS, prims.len(), &[words]);
        self.written();
    }

    fn plain_run(&mut self, words: &[u32]) {
        if crate::bench::experiment(crate::bench::NO_WRITE_FILTER) {
            self.stream.push(RUN, words.len(), &[words]);
            self.written();
            return;
        }
        // The writes that change something (a run is at most a cache line).
        let mut kept = [0u32; MAX_RECORD_WORDS];
        let mut count = 0;
        let filter = &mut *self.filter;
        for &word in words {
            let opcode = (word >> 24) as usize;
            let data = word & 0xFF_FFFF;
            if filter.droppable[opcode] {
                if filter.registers[opcode] == data {
                    continue;
                }
                filter.registers[opcode] = data;
            }
            kept[count] = word;
            count += 1;
        }
        filter.sent += count as u32;
        filter.dropped += (words.len() - count) as u32;
        if count > 0 {
            self.stream.push(RUN, count, &[&kept[..count]]);
            self.written();
        }
    }

    #[inline]
    fn should_pause(&self) -> bool {
        self.stream.needs_draining()
    }

    /// A run record holds at most a line (`MAX_RECORD_WORDS`).
    #[inline]
    fn plain_runs_by_line(&self) -> bool {
        true
    }
}

// ---- Stage B ----

/// Calls of `replay` so far (for measuring how often stage B is woken).
pub static REPLAYS: AtomicU32 = AtomicU32::new(0);

/// Stage B: applies the stream's commands to `ge` and hands them to
/// `visitor` as `walk` would, reading memory (vertices, palettes, block
/// transfer sources) from `memory`. Stops at the end of the stream or after
/// a frame's end; returns that frame's displayed framebuffer. `actions`:
/// `PlainActions::for_visitor(visitor)` (made once by the caller: stage B
/// runs this hundreds of times per frame).
pub fn replay<M: PspMemory, V: CommandVisitor>(
    stream: &WalkStream,
    ge: &mut GeState,
    memory: &mut M,
    visitor: &mut V,
    actions: &PlainActions,
) -> Option<DisplayedFramebuffer> {
    REPLAYS.fetch_add(1, Ordering::Relaxed);
    let written = stream.written.load(Ordering::Acquire);
    let mut position = stream.read.load(Ordering::Relaxed);
    // Safe: the words between `read` and `written` are published and stage
    // A doesn't touch them until `read` moves past them.
    let words = unsafe { &*stream.words.get() };
    let mut shown = None;
    let mut published = position;
    while position != written && shown.is_none() {
        let index = position as usize % STREAM_WORDS;
        let kind = words[index] >> 24;
        let len = (words[index] & 0xFF_FFFF) as usize;
        let body = &words[index + 1..];
        let size = match kind {
            PAD => STREAM_WORDS - index,
            RUN => {
                apply_run(&body[..len], ge, memory, visitor, actions);
                1 + len
            }
            SPECIAL => {
                apply_special(Command(body[0]), body[1], ge, memory, visitor);
                3
            }
            PRIMS => {
                // Safe: `Command` is a `u32` (repr(transparent)).
                let prims = unsafe { core::slice::from_raw_parts(body.as_ptr() as *const Command, len) };
                ge.set_register(prims[len - 1]);
                visitor.visit_prims(prims, ge, memory);
                for prim in prims {
                    ge.advance_after_draw(prim.data() & 0xFFFF);
                }
                1 + len
            }
            END_FRAME => {
                shown = Some(DisplayedFramebuffer { address: body[0], stride: body[1], pixel_format: body[2] });
                4
            }
            _ => unreachable!("walk stream record kind {kind}"),
        };
        position = position.wrapping_add(size as u32);
        if position.wrapping_sub(published) >= READ_PUBLISH_WORDS {
            stream.read.store(position, Ordering::Release);
            published = position;
        }
    }
    stream.read.store(position, Ordering::Release);
    shown
}

/// Plain commands, as `run_plain_commands` runs them, except those the
/// visitor wants, which go to it as in `walk`'s main loop.
fn apply_run<M: PspMemory, V: CommandVisitor>(
    words: &[u32],
    ge: &mut GeState,
    memory: &mut M,
    visitor: &mut V,
    actions: &PlainActions,
) {
    let mut done = 0;
    while done < words.len() {
        let command = Command(words[done]);
        match actions.0[command.opcode() as usize] {
            PlainAction::SetRegister => {
                if !crate::bench::experiment(crate::bench::SKIP_STAGE_B_PLAIN) {
                    ge.set_plain_register(command);
                }
                done += 1;
            }
            PlainAction::MatrixRun => done += ge.upload_matrix_run(&words[done..]),
            // Not in stage A's runs (`StreamWriter` wants them as specials);
            // as `run_plain_commands` runs them all the same.
            PlainAction::Address => {
                ge.apply_address(command);
                ge.set_register(command);
                done += 1;
            }
            // CALL / RET aren't in stage A's runs either (specials).
            PlainAction::Stop | PlainAction::Call | PlainAction::Ret | PlainAction::Visit | PlainAction::Prims => {
                ge.set_register(command);
                visitor.visit(command, ge, memory);
                done += 1;
            }
        }
    }
}

/// A command `walk` handles in its main loop (control flow, addresses,
/// block transfers, ...), minus the control flow.
fn apply_special<M: PspMemory, V: CommandVisitor>(
    command: Command,
    extra: u32,
    ge: &mut GeState,
    memory: &mut M,
    visitor: &mut V,
) {
    match command.opcode() {
        op::VERTEX_ADDRESS => ge.vertex_address = extra,
        op::INDEX_ADDRESS => ge.index_address = extra,
        op::RET | op::END => {
            if ge.register(op::BASE) != extra {
                ge.set_register(Command(((op::BASE as u32) << 24) | extra));
            }
        }
        op::TRANSFER_START => block_transfer(ge, memory, command),
        _ => {}
    }
    ge.set_register(command);
    if visitor.wants(command.opcode()) {
        visitor.visit(command, ge, memory);
    }
    if command.opcode() == op::PRIM {
        ge.advance_after_draw(command.data() & 0xFFFF);
    }
}
