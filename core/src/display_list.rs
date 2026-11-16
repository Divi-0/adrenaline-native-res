//! Walking a display list the way the GE executes it.
//!
//! [`walk`] follows control flow (JUMP, CALL/RET, SIGNAL+END variants) and
//! hands every executed command to a [`CommandVisitor`]. It stops at the
//! list's stall address, at FINISH+END, or on an error. Because it keeps its
//! position in [`DisplayList`], walking can resume later when the game moves
//! the stall address forward.
//!
//! Control-flow semantics follow PPSSPP's `GPUCommon::Execute_*` functions.

use crate::block_cache::LINE_SIZE;
use crate::command::{op, Command, SignalBehaviour, TargetKind};
use crate::ge_state::GeState;
use crate::memory::{is_valid_address, physical, PspMemory};
use crate::snapshot::{Reader, Writer};

/// Maximum nesting of CALLs and signal calls (same as PPSSPP).
const STACK_DEPTH: usize = 32;

/// Real hardware only nests plain CALLs two levels deep; a third CALL is
/// ignored. Some games rely on that (PPSSPP issue #7395).
const HARDWARE_CALL_DEPTH: usize = 2;

#[derive(Clone, Copy, Debug)]
struct ReturnPoint {
    /// Where execution continues after RET.
    address: u32,
    /// The offset to restore.
    offset: u32,
    /// The BASE register to restore. Only signal calls save it; plain CALLs
    /// leave it as `None`, which is also how hardware call depth is counted.
    base: Option<u32>,
}

/// One list handed to the GE, with our current position in it.
#[derive(Clone, Copy, Debug)]
pub struct DisplayList {
    /// Queue id returned by sceGeListEnQueue.
    pub id: i32,
    /// Address of the next command to execute.
    pub pc: u32,
    /// Execution waits when `pc` reaches this address. `None` = no stall.
    pub stall: Option<u32>,
    /// The list's OFFSET_ADDRESS / ORIGIN value. Each list has its own,
    /// 0 when enqueued, like PPSSPP's `DisplayList::offsetAddr`: Monster
    /// Hunter Freedom's lists jump with no OFFSET of their own after some
    /// other list ran ORIGIN, and a GE-wide offset sent its whole frame to
    /// 0x01b4xxxx.
    pub offset: u32,
    /// Where the GE saves its state for this list's run (`ge_context`,
    /// 0 = none), and whether we saved ours there yet (at the first walk).
    pub context: u32,
    pub context_saved: bool,
    stack: [ReturnPoint; STACK_DEPTH],
    stack_len: usize,
}

/// The piece of a list `walk` reads its commands from: the rest of one
/// cache line (`block_cache::LINE_SIZE`), copied out of memory once. MTX's
/// main list stops the plain-command run every one or two commands (PRIM,
/// VADDR, ...); going through the memory layers (GE writes, block cache,
/// line fill) for each of them was ~30% of the walk.
struct ListWindow {
    words: [u32; LINE_WORDS],
    /// Physical address of `words[0]`; `len` words are valid (0 = none).
    start: u32,
    len: usize,
}

const LINE_WORDS: usize = LINE_SIZE as usize / 4;

/// `walk` keeps a window per call depth (the deepest one shared by all
/// deeper calls): after a RET the caller's window still holds its line.
/// With one window every CALL and every RET refilled it from the block
/// cache (~1,800 CALLs in a frame of MTX's race, ~180 cycles a refill).
const KEPT_WINDOWS: usize = 4;

impl ListWindow {
    fn new() -> Self {
        ListWindow { words: [0; LINE_WORDS], start: 0, len: 0 }
    }

    /// The words from `pc` to the end of its line, at most `limit` of them
    /// (the stall address), copied in if `pc` isn't in the window. `None`
    /// if they can't be read.
    ///
    /// Inlined: nearly every call finds `pc` in the window (a race frame of
    /// MTX: ~17,500 calls, ~1,100 refills), and as a call of its own the
    /// check cost more in saving and restoring registers than in checking.
    #[inline(always)]
    fn words_from(&mut self, memory: &mut impl PspMemory, pc: u32, limit: usize) -> Option<&[u32]> {
        let pc = physical(pc);
        let offset = pc.wrapping_sub(self.start) as usize / 4;
        if pc % 4 != 0 || pc < self.start || offset >= self.len {
            return self.refill(memory, pc, limit);
        }
        Some(&self.words[offset..self.len.min(offset + limit)])
    }

    /// `words_from` for a `pc` outside the window: copies its line in.
    #[inline(never)]
    fn refill(&mut self, memory: &mut impl PspMemory, pc: u32, limit: usize) -> Option<&[u32]> {
        let started = crate::cycles::start(crate::cycles::REFILL);
        let filled = self.refill_now(memory, pc);
        crate::cycles::add(crate::cycles::REFILL, started);
        filled?;
        Some(&self.words[..self.len.min(limit)])
    }

    /// `refill` without the measuring: the window's length, if any.
    fn refill_now(&mut self, memory: &mut impl PspMemory, pc: u32) -> Option<()> {
        let to_line_end = ((LINE_SIZE - pc % LINE_SIZE) / 4) as usize;
        self.len = 0;
        if pc % 4 != 0 {
            return None;
        }
        let mut len = to_line_end;
        // Through the block cache (not `read_words`, which reads a whole
        // line straight from the source each time: after every RET the
        // caller's line would be read from PSP RAM again).
        if let Some(words) = memory.borrow_words(pc, to_line_end) {
            self.words[..to_line_end].copy_from_slice(words);
        } else if memory.read_words(pc, &mut self.words[..to_line_end]).is_none() {
            // Part of the line can't be read: the words up to there.
            len = 0;
            while len < to_line_end {
                match memory.read_u32(pc + 4 * len as u32) {
                    Some(word) => self.words[len] = word,
                    None => break,
                }
                len += 1;
            }
            if len == 0 {
                return None;
            }
        }
        self.start = pc;
        self.len = len;
        Some(())
    }

    /// One command word.
    #[inline]
    fn word(&mut self, memory: &mut impl PspMemory, pc: u32) -> Option<u32> {
        self.words_from(memory, pc, 1)?.first().copied()
    }

    /// Memory may have changed (a block transfer): read it again.
    fn forget(&mut self) {
        self.len = 0;
    }
}

/// Why [`walk`] returned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WalkResult {
    /// Reached the stall address; call `walk` again after the stall moves.
    Stalled,
    /// FINISH + END: the list is complete.
    Finished,
    /// Executed `max_commands` without finishing (likely an endless loop).
    TooManyCommands,
    /// A command or jump target was outside readable memory.
    BadAddress(u32),
    /// The visitor asked to stop for now (`CommandVisitor::should_pause`);
    /// call `walk` again to go on from the same command.
    Paused,
}

/// Receives the commands the GE would execute, after the GE state has been
/// updated by them. `memory` lets it read vertex or palette data.
pub trait CommandVisitor {
    fn visit<M: PspMemory>(&mut self, command: Command, ge: &GeState, memory: &mut M);

    /// Whether `visit` wants commands with this opcode. Most commands only
    /// set a register, which `walk` does itself; a visitor that needs only
    /// a few opcodes saves a call per command (like PPSSPP's FastRunLoop,
    /// which only calls a command's handler when it has one).
    #[inline]
    fn wants(&self, _opcode: u8) -> bool {
        true
    }

    /// Whether `walk` may hand PRIM commands that directly follow each
    /// other over as one batch (`visit_prims`), like PPSSPP's inner PRIM
    /// loop. Only for visitors that don't need the GE state per PRIM.
    #[inline]
    fn batches_prims(&self) -> bool {
        false
    }

    /// PRIM commands that directly follow each other, with nothing between
    /// them. `ge` is as for the first; each PRIM's vertices (or indices)
    /// start where the one before ended (`GeState::advance_after_draw`).
    /// By default visited one by one (with the first PRIM's addresses).
    fn visit_prims<M: PspMemory>(&mut self, prims: &[Command], ge: &GeState, memory: &mut M) {
        for &prim in prims {
            self.visit(prim, ge, memory);
        }
    }

    /// Plain commands (`PLAIN_COMMANDS`, not wanted by `wants`) that `walk`
    /// just executed, in order: the GE state already has them. Only the
    /// capture's stage A wants them (`walk_stream`), to pass them on.
    #[inline]
    fn plain_run(&mut self, _words: &[u32]) {}

    /// Checked before every command: true makes `walk` return
    /// `WalkResult::Paused` (stage A when its stream is full and the same
    /// thread has to empty it first, see `walk_stream`).
    #[inline]
    fn should_pause(&self) -> bool {
        false
    }
}

/// Most PRIM commands `walk` hands over at once.
pub const MAX_PRIM_BATCH: usize = 32;

impl DisplayList {
    /// Placeholder for unused queue slots. All zeros, so a static array of
    /// them costs no space in the module file.
    pub const EMPTY: DisplayList = DisplayList {
        id: 0,
        pc: 0,
        stall: None,
        offset: 0,
        context: 0,
        context_saved: false,
        stack: [ReturnPoint { address: 0, offset: 0, base: None }; STACK_DEPTH],
        stack_len: 0,
    };

    /// A list as passed to sceGeListEnQueue. A stall address of 0 means
    /// "no stall".
    pub fn new(id: i32, start: u32, stall: u32) -> DisplayList {
        let mut list = DisplayList::EMPTY;
        list.id = id;
        list.pc = physical(start);
        list.set_stall(stall);
        list
    }

    /// A list enqueued with a context (`PspGeListArgs::context`).
    pub fn with_context(id: i32, start: u32, stall: u32, context: u32) -> DisplayList {
        let mut list = DisplayList::new(id, start, stall);
        list.context = context;
        list
    }

    pub fn set_stall(&mut self, stall: u32) {
        self.stall = if stall == 0 { None } else { Some(physical(stall)) };
    }

    fn push(&mut self, point: ReturnPoint) -> bool {
        if self.stack_len == STACK_DEPTH {
            return false;
        }
        self.stack[self.stack_len] = point;
        self.stack_len += 1;
        true
    }

    fn pop(&mut self) -> Option<ReturnPoint> {
        if self.stack_len == 0 {
            return None;
        }
        self.stack_len -= 1;
        Some(self.stack[self.stack_len])
    }

    pub fn save(&self, out: &mut Writer) {
        out.u32(self.id as u32);
        out.u32(self.pc);
        out.u32(self.stall.unwrap_or(0));
        out.u32(self.offset);
        out.u32(self.context);
        out.bool(self.context_saved);
        out.u32(self.stack_len as u32);
        for point in &self.stack[..self.stack_len] {
            out.u32(point.address);
            out.u32(point.offset);
            out.bool(point.base.is_some());
            out.u32(point.base.unwrap_or(0));
        }
    }

    /// `ge_offset`: the GE-wide offset older snapshots (version < 5)
    /// kept instead of one per list.
    pub fn load(input: &mut Reader, version: u32, ge_offset: u32) -> Option<DisplayList> {
        let mut list = DisplayList::EMPTY;
        list.id = input.u32()? as i32;
        list.pc = input.u32()?;
        list.set_stall(input.u32()?);
        list.offset = if version >= 5 { input.u32()? } else { ge_offset };
        if version >= 6 {
            list.context = input.u32()?;
            list.context_saved = input.bool()?;
        }
        let stack_len = input.u32()? as usize;
        if stack_len > STACK_DEPTH {
            return None;
        }
        for _ in 0..stack_len {
            let address = input.u32()?;
            let offset = input.u32()?;
            let has_base = input.bool()?;
            let base = input.u32()?;
            list.push(ReturnPoint { address, offset, base: has_base.then_some(base) });
        }
        Some(list)
    }

    /// Number of plain CALLs on top of the stack (see HARDWARE_CALL_DEPTH).
    fn hardware_call_depth(&self) -> usize {
        self.stack[..self.stack_len].iter().rev().take_while(|point| point.base.is_none()).count()
    }
}

/// Executes `list` from its current position until it stalls, finishes,
/// fails, or `max_commands` commands have run.
/// `commands` counts the commands executed.
pub fn walk(
    list: &mut DisplayList,
    ge: &mut GeState,
    memory: &mut impl PspMemory,
    visitor: &mut impl CommandVisitor,
    max_commands: u32,
    commands: &mut u32,
) -> WalkResult {
    let started = crate::cycles::start(crate::cycles::WALK);
    // The offset belongs to the list (PPSSPP: gstate_c.offsetAddr =
    // list.offsetAddr when a list starts or resumes, saved back after).
    ge.offset = list.offset;
    let result = walk_list(list, ge, memory, visitor, max_commands, commands);
    list.offset = ge.offset;
    crate::cycles::add(crate::cycles::WALK, started);
    result
}

fn walk_list(
    list: &mut DisplayList,
    ge: &mut GeState,
    memory: &mut impl PspMemory,
    visitor: &mut impl CommandVisitor,
    max_commands: u32,
    commands: &mut u32,
) -> WalkResult {
    let mut executed = 0;
    let mut windows: [ListWindow; KEPT_WINDOWS] = core::array::from_fn(|_| ListWindow::new());
    let actions = PlainActions::for_visitor(visitor);
    while executed < max_commands {
        if visitor.should_pause() {
            return WalkResult::Paused;
        }
        // CALL and RET below change the depth for the next round.
        let window = &mut windows[list.stack_len.min(KEPT_WINDOWS - 1)];
        let mut transferred = false;
        let plain_started = crate::cycles::start(crate::cycles::PLAIN);
        let (ran, stopped_at) =
            run_plain_commands(list, ge, window, memory, &actions, visitor, max_commands - executed, commands);
        crate::cycles::add(crate::cycles::PLAIN, plain_started);
        crate::phase::enter(crate::phase::WALK);
        executed += ran;
        if executed >= max_commands {
            break;
        }
        executed += 1;
        if list.stall == Some(list.pc) {
            return WalkResult::Stalled;
        }

        let pc = list.pc;
        // The command the plain run stopped at, if it saw it, else read.
        let Some(word) = stopped_at.or_else(|| window.word(memory, pc)) else {
            return WalkResult::BadAddress(pc);
        };
        *commands += 1;
        let command = Command(word);

        // A PRIM and the PRIMs right after it go to the visitor together.
        if command.opcode() == op::PRIM && visitor.batches_prims() && visitor.wants(op::PRIM) {
            let mut batch = [command; MAX_PRIM_BATCH];
            let mut len = 1;
            let mut next = pc.wrapping_add(4);
            // The PRIMs after it, a piece of the window at a time (one look
            // at the window per piece, not per PRIM).
            loop {
                let room = (MAX_PRIM_BATCH - len).min((max_commands - executed) as usize);
                let room = commands_before_stall(list.stall, next, room);
                if room == 0 {
                    break;
                }
                let Some(words) = window.words_from(memory, next, room) else {
                    break;
                };
                let prims = words.iter().take_while(|&&word| Command(word).opcode() == op::PRIM).count();
                for (slot, &word) in batch[len..len + prims].iter_mut().zip(words) {
                    *slot = Command(word);
                }
                len += prims;
                executed += prims as u32;
                next = next.wrapping_add(4 * prims as u32);
                // A command that isn't a PRIM; else the piece ended at the
                // line's end (go on in the next line) or at `room`.
                if prims < words.len() {
                    break;
                }
            }
            *commands += len as u32 - 1;
            ge.set_register(batch[len - 1]);
            visitor.visit_prims(&batch[..len], ge, memory);
            crate::phase::enter(crate::phase::WALK);
            for prim in &batch[..len] {
                ge.advance_after_draw(prim.data() & 0xFFFF);
            }
            list.pc = next;
            continue;
        }
        let mut next_pc = pc.wrapping_add(4);
        let mut finished = false;

        match command.opcode() {
            op::JUMP => next_pc = ge.relative_address(command.data() & !3),

            // Taken only if the preceding BOUNDING_BOX found nothing on
            // screen. We don't evaluate bounding boxes yet, so we assume
            // "visible" and fall through, which visits a superset of draws.
            op::BOUNDING_BOX_JUMP => {}

            op::CALL => {
                let target = ge.relative_address(command.data() & !3);
                // A CALL to an address that isn't memory is skipped and the
                // list goes on, as PPSSPP does (GPUCommon::Execute_Call with
                // its default bIgnoreBadMemAccess) and as Sony's emulator
                // evidently does: MTX's attract demo calls 0x050040C4 (BASE 5
                // with offset 0) before drawing its logo and menu overlay.
                if is_valid_address(target) && list.hardware_call_depth() < HARDWARE_CALL_DEPTH {
                    let saved = ReturnPoint { address: next_pc, offset: ge.offset, base: None };
                    if list.push(saved) {
                        next_pc = target;
                    }
                }
            }

            op::RET => {
                // RET with an empty stack is ignored, like on hardware.
                if let Some(point) = list.pop() {
                    ge.offset = point.offset;
                    if let Some(base) = point.base {
                        ge.set_register(Command(((op::BASE as u32) << 24) | base));
                    }
                    next_pc = physical(point.address);
                }
            }

            op::END => {
                let previous = window.word(memory, pc.wrapping_sub(4)).map(Command);
                match previous {
                    Some(previous) if previous.opcode() == op::FINISH => finished = true,
                    Some(previous) if previous.opcode() == op::SIGNAL => {
                        if let Some(target) = signal_target(list, ge, previous, command, pc) {
                            next_pc = target;
                        }
                    }
                    // END without FINISH/SIGNAL: PPSSPP just continues.
                    _ => {}
                }
            }

            op::TRANSFER_START => {
                block_transfer(ge, memory, command);
                // The transfer may have written the lists themselves.
                transferred = true;
            }

            op::ORIGIN => ge.offset = pc,
            op::OFFSET_ADDRESS => ge.offset = command.data() << 8,
            op::VERTEX_ADDRESS => ge.vertex_address = ge.relative_address(command.data()),
            op::INDEX_ADDRESS => ge.index_address = ge.relative_address(command.data()),
            _ => {}
        }

        ge.set_register(command);
        if visitor.wants(command.opcode()) {
            visitor.visit(command, ge, memory);
            crate::phase::enter(crate::phase::WALK);
        } else if PLAIN_COMMANDS[command.opcode() as usize] {
            // Normally the plain run takes these; this one it couldn't.
            visitor.plain_run(&[word]);
        }
        if command.opcode() == op::PRIM {
            ge.advance_after_draw(command.data() & 0xFFFF);
        }
        list.pc = next_pc;
        if transferred {
            for window in windows.iter_mut() {
                window.forget();
            }
        }

        if finished {
            return WalkResult::Finished;
        }
    }
    WalkResult::TooManyCommands
}

/// Opcodes `walk` does nothing for but set the register (and hand to the
/// visitor if it wants them): everything but control flow, the address
/// commands and the draws.
pub(crate) const PLAIN_COMMANDS: [bool; 256] = {
    let mut plain = [true; 256];
    let special = [
        op::JUMP,
        op::BOUNDING_BOX_JUMP,
        op::CALL,
        op::RET,
        op::END,
        op::ORIGIN,
        op::OFFSET_ADDRESS,
        op::VERTEX_ADDRESS,
        op::INDEX_ADDRESS,
        op::PRIM,
        op::TRANSFER_START,
    ];
    let mut index = 0;
    while index < special.len() {
        plain[special[index] as usize] = false;
        index += 1;
    }
    plain
};

/// Runs the plain commands at the list's position that the visitor doesn't
/// want, straight from the memory block they are in (one lookup for the
/// run, like PPSSPP's FastRunLoop reading the list through a pointer):
/// most commands of a list only set a register. Stops at the first other
/// command, at the stall address, at the cache line's end or after `limit`
/// commands; returns how many it ran, and the command it stopped at if that
/// was a command it doesn't run (so `walk` needn't read it again).
///
/// A run of matrix DATA commands (12 or 16 in a row for every matrix the
/// game uploads) is handed to `GeState::upload_matrix_run` at once, like
/// PPSSPP's fast paths for them (GPUCommonHW.cpp Execute_BoneMtxData).
#[allow(clippy::too_many_arguments)]
fn run_plain_commands(
    list: &mut DisplayList,
    ge: &mut GeState,
    window: &mut ListWindow,
    memory: &mut impl PspMemory,
    actions: &PlainActions,
    visitor: &mut impl CommandVisitor,
    limit: u32,
    commands: &mut u32,
) -> (u32, Option<u32>) {
    let pc = list.pc;
    if pc % 4 != 0 {
        return (0, None);
    }
    // Up to the end of the cache line: a block is filled a line at a time
    // (`block_cache::LINE_SIZE`), so borrowing further would copy lines the
    // list may never reach (a CALLed sub-list is often a few dozen words).
    let count = ((LINE_SIZE - pc % LINE_SIZE) / 4).min(limit);
    let count = commands_before_stall(list.stall, pc, count as usize) as u32;
    if count == 0 {
        return (0, None);
    }
    crate::phase::enter(crate::phase::PLAIN);
    let Some(words) = window.words_from(memory, pc, count as usize) else {
        return (0, None);
    };
    let mut done = 0;
    let mut stopped_at = None;
    while done < words.len() {
        let command = Command(words[done]);
        match actions.0[command.opcode() as usize] {
            PlainAction::SetRegister => {
                ge.set_plain_register(command);
                done += 1;
            }
            PlainAction::MatrixRun => {
                crate::phase::enter(crate::phase::MATRIX);
                let started = crate::cycles::start(crate::cycles::MATRIX);
                done += ge.upload_matrix_run(&words[done..]);
                crate::cycles::add(crate::cycles::MATRIX, started);
                crate::phase::enter(crate::phase::PLAIN);
            }
            PlainAction::Stop => {
                stopped_at = Some(command.0);
                break;
            }
        }
    }
    if done > 0 {
        visitor.plain_run(&words[..done]);
    }
    let done = done as u32;
    list.pc = pc.wrapping_add(4 * done);
    *commands += done;
    (done, stopped_at)
}

/// `count`, or fewer if the stall address comes sooner: how many of the
/// `count` commands from `pc` on may run.
#[inline(always)]
fn commands_before_stall(stall: Option<u32>, pc: u32, count: usize) -> usize {
    match stall {
        Some(stall) if stall >= pc => count.min((stall.wrapping_sub(pc) / 4) as usize),
        _ => count,
    }
}

/// What `run_plain_commands` does with a command, by opcode.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PlainAction {
    SetRegister,
    /// A matrix DATA command: the run of them at once.
    MatrixRun,
    /// Not plain, or the visitor wants it: `walk` handles it.
    Stop,
}

/// `PlainAction` for every opcode, worked out once per `walk` (the
/// visitor's `wants` and `PLAIN_COMMANDS` folded together), so the run's
/// loop makes one table lookup per command instead of a chain of tests
/// whose outcome changes from command to command (mispredicted branches
/// are expensive on the Vita).
#[derive(Clone, Copy)]
pub struct PlainActions(pub(crate) [PlainAction; 256]);

impl PlainActions {
    pub(crate) fn for_visitor(visitor: &impl CommandVisitor) -> Self {
        PlainActions(core::array::from_fn(|opcode| {
            let opcode = opcode as u8;
            if !PLAIN_COMMANDS[opcode as usize] || visitor.wants(opcode) {
                PlainAction::Stop
            } else if is_matrix_data(opcode) {
                PlainAction::MatrixRun
            } else {
                PlainAction::SetRegister
            }
        }))
    }
}

fn is_matrix_data(opcode: u8) -> bool {
    matches!(opcode, op::WORLD_MATRIX_DATA | op::VIEW_MATRIX_DATA | op::PROJECTION_MATRIX_DATA | op::BONE_MATRIX_DATA)
}

/// One side of a block transfer, from its address, width and position
/// registers (layout as in PPSSPP's GPUgstate::getTransfer*).
struct TransferSide {
    /// Physical address of the rectangle's top-left pixel.
    start: u32,
    /// Bytes from one row to the next.
    row_bytes: u32,
}

fn transfer_side(ge: &GeState, address_op: u8, width_op: u8, position_op: u8, bytes_per_pixel: u32) -> TransferSide {
    let width_register = ge.register(width_op);
    let mut base = (ge.register(address_op) & 0xFF_FFF0) | ((width_register & 0xFF_0000) << 8);
    // VRAM mirrors at 0x04800000 wrap back (PPSSPP DoBlockTransfer).
    if base & 0x0480_0000 == 0x0480_0000 {
        base &= !0x0080_0000;
    }
    // Stride in pixels: bits 3..11; above 0x400 the GE uses 0.
    let stride = match width_register & 0x7F8 {
        stride if stride > 0x400 => 0,
        stride => stride,
    };
    let position = ge.register(position_op);
    let (x, y) = (position & 0x3FF, (position >> 10) & 0x3FF);
    TransferSide { start: base.wrapping_add((y * stride + x) * bytes_per_pixel), row_bytes: stride * bytes_per_pixel }
}

/// TRANSFER_START: the GE copies a rectangle of pixels from the transfer
/// source to the destination (PPSSPP GPUCommon::DoBlockTransfer). The copy
/// goes into `memory`'s writes (`ge_writes`), so the rest of the frame
/// reads the copied data; memories that take no writes skip it.
pub(crate) fn block_transfer(ge: &GeState, memory: &mut impl PspMemory, start: Command) {
    let bytes_per_pixel = if start.data() & 1 != 0 { 4 } else { 2 };
    let source = transfer_side(ge, op::TRANSFER_SOURCE, op::TRANSFER_SOURCE_WIDTH, op::TRANSFER_SOURCE_POSITION, bytes_per_pixel);
    let destination = transfer_side(
        ge,
        op::TRANSFER_DESTINATION,
        op::TRANSFER_DESTINATION_WIDTH,
        op::TRANSFER_DESTINATION_POSITION,
        bytes_per_pixel,
    );
    let size = ge.register(op::TRANSFER_SIZE);
    let width_bytes = ((size & 0x3FF) + 1) * bytes_per_pixel;
    let height = ((size >> 10) & 0x3FF) + 1;

    let destination_bytes = (height - 1) * destination.row_bytes + width_bytes;
    if !memory.begin_write(destination.start, destination_bytes) {
        return;
    }
    for row in 0..height {
        let from = source.start.wrapping_add(row * source.row_bytes);
        let to = destination.start.wrapping_add(row * destination.row_bytes);
        copy_bytes(memory, from, to, width_bytes);
    }
}

/// Copies `bytes` (a multiple of 2) within `memory`, word by word when both
/// sides are word aligned, else halfword by halfword. Unreadable source
/// words copy as zero.
fn copy_bytes(memory: &mut impl PspMemory, from: u32, to: u32, bytes: u32) {
    if from % 4 == 0 && to % 4 == 0 && bytes % 4 == 0 {
        for offset in (0..bytes).step_by(4) {
            let word = memory.read_u32(from + offset).unwrap_or(0);
            memory.write_u32(to + offset, word);
        }
        return;
    }
    for offset in (0..bytes).step_by(2) {
        let half = memory.read_u16(from + offset).unwrap_or(0) as u32;
        let target = to + offset;
        let word_address = target & !3;
        let shift = (target & 2) * 8;
        let word = memory.read_u32(word_address).unwrap_or(0);
        memory.write_u32(word_address, (word & !(0xFFFF << shift)) | (half << shift));
    }
}

/// Handles the jump/call/return forms of SIGNAL + END. Returns the new pc if
/// execution moves; `end_pc` is the address of the END command.
fn signal_target(list: &mut DisplayList, ge: &mut GeState, signal: Command, end: Command, end_pc: u32) -> Option<u32> {
    // The target address is split across the two commands:
    // high 16 bits in SIGNAL, low 16 bits in END.
    let raw_target = ((signal.data() & 0xFFFF) << 16 | (end.data() & 0xFFFF)) & !3;
    let signal_pc = end_pc.wrapping_sub(4);
    let resolve = |kind: TargetKind, ge: &GeState| match kind {
        TargetKind::Absolute => physical(raw_target),
        TargetKind::RelativeToSignal => physical(raw_target.wrapping_add(signal_pc)),
        TargetKind::RelativeToOrigin => ge.relative_address(raw_target),
    };

    match SignalBehaviour::from_signal_command(signal) {
        SignalBehaviour::Jump(kind) => Some(resolve(kind, ge)),
        SignalBehaviour::Call(kind) => {
            let target = resolve(kind, ge);
            let saved = ReturnPoint {
                address: end_pc.wrapping_add(4),
                offset: ge.offset,
                base: Some(ge.register(op::BASE)),
            };
            list.push(saved).then_some(target)
        }
        SignalBehaviour::Return => {
            let point = list.pop()?;
            ge.offset = point.offset;
            if let Some(base) = point.base {
                ge.set_register(Command(((op::BASE as u32) << 24) | base));
            }
            Some(physical(point.address))
        }
        SignalBehaviour::Handler | SignalBehaviour::Sync | SignalBehaviour::Unknown(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::test_memory::TestMemory;

    /// Records the opcodes it sees.
    #[derive(Default)]
    struct Recorder {
        opcodes: Vec<u8>,
    }

    impl CommandVisitor for Recorder {
        fn visit<M: PspMemory>(&mut self, command: Command, _ge: &GeState, _memory: &mut M) {
            self.opcodes.push(command.opcode());
        }
    }

    fn cmd(opcode: u8, data: u32) -> u32 {
        ((opcode as u32) << 24) | data
    }

    const LIST: u32 = 0x0890_0000;

    fn run(memory: &mut TestMemory, stall: u32) -> (WalkResult, Vec<u8>, DisplayList) {
        let mut list = DisplayList::new(0, LIST, stall);
        let mut ge = GeState::new();
        let mut recorder = Recorder::default();
        let result = walk(&mut list, &mut ge, memory, &mut recorder, 1000, &mut 0);
        (result, recorder.opcodes, list)
    }

    #[test]
    fn simple_list_finishes() {
        let mut memory = TestMemory::default();
        memory.write(LIST, &[cmd(op::PRIM, 3), cmd(op::FINISH, 0), cmd(op::END, 0)]);
        let (result, opcodes, _) = run(&mut memory, 0);
        assert_eq!(result, WalkResult::Finished);
        assert_eq!(opcodes, vec![op::PRIM, op::FINISH, op::END]);
    }

    #[test]
    fn commands_are_counted_across_a_block_end_and_a_jump_back() {
        let mut memory = TestMemory::default();
        // Starts 8 bytes before a block end; a JUMP goes back to the second
        // command (BASE 0x08 for the jump target).
        let start = LIST + crate::block_cache::BLOCK_SIZE - 8;
        let jump_back = cmd(op::JUMP, (start + 4) & 0x00FF_FFFF);
        memory.write(start, &[cmd(op::BASE, 0x08_0000), cmd(op::PRIM, 3), cmd(op::PRIM, 4), jump_back]);
        let mut list = DisplayList::new(0, start, 0);
        let mut ge = GeState::new();
        let mut recorder = Recorder::default();
        let mut commands = 0;
        // Endless (jumps back for ever): stops after 10 commands.
        let result = walk(&mut list, &mut ge, &mut memory, &mut recorder, 10, &mut commands);
        assert_eq!(result, WalkResult::TooManyCommands);
        assert_eq!(commands, 10);
        let expected = [op::BASE, op::PRIM, op::PRIM, op::JUMP, op::PRIM, op::PRIM, op::JUMP, op::PRIM, op::PRIM, op::JUMP];
        assert_eq!(recorder.opcodes, expected);
    }

    /// Records only the PRIMs it sees (so `walk` runs the rest itself).
    #[derive(Default)]
    struct PrimsOnly {
        prims: Vec<u32>,
    }

    impl CommandVisitor for PrimsOnly {
        fn visit<M: PspMemory>(&mut self, command: Command, _ge: &GeState, _memory: &mut M) {
            self.prims.push(command.data());
        }

        fn wants(&self, opcode: u8) -> bool {
            opcode == op::PRIM
        }
    }

    /// Records the PRIM batches it is handed, with the vertex address of
    /// each batch (to check the addresses advance across batches).
    #[derive(Default)]
    struct PrimBatches {
        batches: Vec<Vec<u32>>,
        vertex_addresses: Vec<u32>,
    }

    impl CommandVisitor for PrimBatches {
        fn visit<M: PspMemory>(&mut self, _command: Command, _ge: &GeState, _memory: &mut M) {}

        fn wants(&self, opcode: u8) -> bool {
            opcode == op::PRIM
        }

        fn batches_prims(&self) -> bool {
            true
        }

        fn visit_prims<M: PspMemory>(&mut self, prims: &[Command], ge: &GeState, _memory: &mut M) {
            self.batches.push(prims.iter().map(|prim| prim.data()).collect());
            self.vertex_addresses.push(ge.vertex_address);
        }
    }

    #[test]
    fn prim_batches_cross_line_ends_and_stop_at_the_stall_and_the_batch_limit() {
        let mut memory = TestMemory::default();
        // A VADDR, then 40 PRIMs whose run crosses a cache line end, a fog
        // colour, 2 more PRIMs, FINISH + END. The stall is at PRIM 38.
        // Vertices: float positions only, 12 bytes each.
        let start = LIST + LINE_SIZE - 4 * 5;
        let mut words = vec![cmd(op::BASE, 0x08_0000), cmd(op::VERTEX_TYPE, 3 << 7), cmd(op::VERTEX_ADDRESS, 0x90_0000)];
        words.extend((1..=40).map(|n| cmd(op::PRIM, n)));
        words.extend([cmd(op::FOG_COLOR, 1), cmd(op::PRIM, 41), cmd(op::PRIM, 42), cmd(op::FINISH, 0), cmd(op::END, 0)]);
        memory.write(start, &words);
        let mut list = DisplayList::new(0, start, start + 4 * (3 + 37));
        let mut ge = GeState::new();
        let mut visitor = PrimBatches::default();
        let mut commands = 0;

        assert_eq!(walk(&mut list, &mut ge, &mut memory, &mut visitor, 1000, &mut commands), WalkResult::Stalled);
        assert_eq!(visitor.batches, [(1..=32).collect::<Vec<u32>>(), (33..=37).collect()]);
        assert_eq!(commands, 3 + 37);

        list.set_stall(0);
        assert_eq!(walk(&mut list, &mut ge, &mut memory, &mut visitor, 1000, &mut commands), WalkResult::Finished);
        assert_eq!(&visitor.batches[2..], [vec![38, 39, 40], vec![41, 42]]);
        assert_eq!(commands, words.len() as u32);
        // Each batch starts where the PRIMs before it ended (PRIM n draws n
        // vertices).
        let after = |prims: u32| 0x0890_0000 + 12 * (1..=prims).sum::<u32>();
        assert_eq!(visitor.vertex_addresses, [after(0), after(32), after(37), after(40)]);
    }

    #[test]
    fn plain_commands_run_up_to_the_stall_and_on_after_it() {
        let mut memory = TestMemory::default();
        let mut words = vec![cmd(op::FOG_COLOR, 0x11), cmd(op::FOG_COLOR, 0x22), cmd(op::PRIM, 3), cmd(op::FOG_COLOR, 0x33)];
        words.extend([cmd(op::FOG_COLOR, 0x44), cmd(op::PRIM, 4), cmd(op::FINISH, 0), cmd(op::END, 0)]);
        memory.write(LIST, &words);
        // Stalls in the middle of the register run after the first PRIM.
        let mut list = DisplayList::new(0, LIST, LIST + 16);
        let mut ge = GeState::new();
        let mut visitor = PrimsOnly::default();
        let mut commands = 0;

        assert_eq!(walk(&mut list, &mut ge, &mut memory, &mut visitor, 1000, &mut commands), WalkResult::Stalled);
        assert_eq!((list.pc, commands, ge.register(op::FOG_COLOR)), (LIST + 16, 4, 0x33));
        assert_eq!(visitor.prims, [3]);

        list.set_stall(0);
        assert_eq!(walk(&mut list, &mut ge, &mut memory, &mut visitor, 1000, &mut commands), WalkResult::Finished);
        assert_eq!((commands, ge.register(op::FOG_COLOR)), (8, 0x44));
        assert_eq!(visitor.prims, [3, 4]);
    }

    #[test]
    fn a_run_of_matrix_data_sets_what_the_commands_one_by_one_set() {
        let mut memory = TestMemory::default();
        let mut words = vec![cmd(op::BONE_MATRIX_NUMBER, 12)];
        words.extend((0..14).map(|n| cmd(op::BONE_MATRIX_DATA, 0x3F_8000 + n * 0x100)));
        words.extend([cmd(op::WORLD_MATRIX_NUMBER, 0), cmd(op::WORLD_MATRIX_DATA, 0x40_0000), cmd(op::WORLD_MATRIX_DATA, 0x40_0000)]);
        words.extend([cmd(op::FOG_COLOR, 7), cmd(op::FINISH, 0), cmd(op::END, 0)]);
        memory.write(LIST, &words);

        let mut list = DisplayList::new(0, LIST, 0);
        let mut walked = GeState::new();
        let mut commands = 0;
        let result = walk(&mut list, &mut walked, &mut memory, &mut PrimsOnly::default(), 1000, &mut commands);
        assert_eq!((result, commands), (WalkResult::Finished, words.len() as u32));

        let mut one_by_one = GeState::new();
        for &word in &words {
            one_by_one.set_register(Command(word));
        }
        assert_eq!(walked.matrices.bones, one_by_one.matrices.bones);
        assert_eq!(walked.matrices.world, one_by_one.matrices.world);
        assert_eq!(walked.register(op::BONE_MATRIX_DATA), one_by_one.register(op::BONE_MATRIX_DATA));
        assert_eq!(walked.matrix_generation(), one_by_one.matrix_generation());
        assert_eq!(walked.bone_generation(), one_by_one.bone_generation());
    }

    #[test]
    fn stops_at_stall_and_resumes() {
        let mut memory = TestMemory::default();
        memory.write(LIST, &[cmd(op::PRIM, 3), cmd(op::PRIM, 3), cmd(op::FINISH, 0), cmd(op::END, 0)]);
        let mut list = DisplayList::new(0, LIST, LIST + 4);
        let mut ge = GeState::new();
        let mut recorder = Recorder::default();

        assert_eq!(walk(&mut list, &mut ge, &mut memory, &mut recorder, 1000, &mut 0), WalkResult::Stalled);
        assert_eq!(recorder.opcodes.len(), 1);

        list.set_stall(0);
        assert_eq!(walk(&mut list, &mut ge, &mut memory, &mut recorder, 1000, &mut 0), WalkResult::Finished);
        assert_eq!(recorder.opcodes.len(), 4);
    }

    #[test]
    fn call_and_return_use_base_and_offset() {
        let mut memory = TestMemory::default();
        // BASE selects address bits 24..28 = 0x08; CALL 0x910000 -> 0x08910000.
        memory.write(LIST, &[cmd(op::BASE, 0x08_0000), cmd(op::CALL, 0x91_0000), cmd(op::FINISH, 0), cmd(op::END, 0)]);
        memory.write(0x0891_0000, &[cmd(op::PRIM, 3), cmd(op::RET, 0)]);
        let (result, opcodes, _) = run(&mut memory, 0);
        assert_eq!(result, WalkResult::Finished);
        assert_eq!(opcodes, vec![op::BASE, op::CALL, op::PRIM, op::RET, op::FINISH, op::END]);
    }

    /// Monster Hunter Freedom, character creation: an earlier list ran
    /// ORIGIN (offset = its pc, 0x091661d4 on the Vita), then the frame's
    /// list `BASE 8; JUMP 0x9a5b30` chains through buffers with no OFFSET
    /// of its own. With one GE-wide offset the jump went to 0x01b4xxxx and
    /// the whole frame was lost (bad_addr); each list starts at offset 0.
    #[test]
    fn a_new_list_starts_with_offset_zero() {
        let mut memory = TestMemory::default();
        const ORIGIN_LIST: u32 = 0x0916_61d4;
        memory.write(ORIGIN_LIST, &[cmd(op::ORIGIN, 0), cmd(op::FINISH, 0), cmd(op::END, 0)]);
        memory.write(LIST, &[cmd(op::BASE, 0x08_0000), cmd(op::JUMP, 0x9a_5b30)]);
        memory.write(0x089a_5b30, &[cmd(op::PRIM, 3), cmd(op::FINISH, 0), cmd(op::END, 0)]);
        let mut ge = GeState::new();
        let mut first = DisplayList::new(0, ORIGIN_LIST, 0);
        assert_eq!(walk(&mut first, &mut ge, &mut memory, &mut Recorder::default(), 1000, &mut 0), WalkResult::Finished);
        assert_eq!(ge.offset, ORIGIN_LIST, "ORIGIN sets the offset of the list that runs it");
        let mut list = DisplayList::new(1, LIST, 0);
        let mut recorder = Recorder::default();
        let result = walk(&mut list, &mut ge, &mut memory, &mut recorder, 1000, &mut 0);
        assert_eq!(result, WalkResult::Finished);
        assert_eq!(recorder.opcodes, vec![op::BASE, op::JUMP, op::PRIM, op::FINISH, op::END]);
    }

    /// A list that stalls keeps its offset for when it goes on, even if
    /// another list ran in between.
    #[test]
    fn a_stalled_list_keeps_its_offset() {
        let mut memory = TestMemory::default();
        memory.write(LIST, &[cmd(op::OFFSET_ADDRESS, 0x00_0100), cmd(op::BASE, 0x08_0000), cmd(op::JUMP, 0x90_0000)]);
        memory.write(0x0891_0000, &[cmd(op::PRIM, 3), cmd(op::FINISH, 0), cmd(op::END, 0)]); // 0x08900000 + offset 0x10000
        let mut ge = GeState::new();
        let mut list = DisplayList::new(0, LIST, LIST + 8);
        assert_eq!(walk(&mut list, &mut ge, &mut memory, &mut Recorder::default(), 1000, &mut 0), WalkResult::Stalled);
        ge.offset = 0x1234_5600; // another list ran meanwhile
        list.set_stall(0);
        let mut recorder = Recorder::default();
        assert_eq!(walk(&mut list, &mut ge, &mut memory, &mut recorder, 1000, &mut 0), WalkResult::Finished);
        assert_eq!(recorder.opcodes, vec![op::JUMP, op::PRIM, op::FINISH, op::END]);
    }

    /// MTX's attract demo: a CALL to 0x050040C4, which isn't memory, is
    /// skipped and the rest of the list still runs (PPSSPP ignores it).
    #[test]
    fn a_call_to_no_memory_is_skipped() {
        let mut memory = TestMemory::default();
        memory.write(LIST, &[cmd(op::BASE, 0x05_0000), cmd(op::CALL, 0x00_40C4), cmd(op::PRIM, 3), cmd(op::FINISH, 0), cmd(op::END, 0)]);
        let (result, opcodes, _) = run(&mut memory, 0);
        assert_eq!(result, WalkResult::Finished);
        assert_eq!(opcodes, vec![op::BASE, op::CALL, op::PRIM, op::FINISH, op::END]);
    }

    #[test]
    fn third_nested_call_is_ignored() {
        let mut memory = TestMemory::default();
        memory.write(LIST, &[cmd(op::BASE, 0x08_0000), cmd(op::CALL, 0x91_0000), cmd(op::FINISH, 0), cmd(op::END, 0)]);
        memory.write(0x0891_0000, &[cmd(op::CALL, 0x92_0000), cmd(op::RET, 0)]);
        memory.write(0x0892_0000, &[cmd(op::CALL, 0x93_0000), cmd(op::RET, 0)]);
        memory.write(0x0893_0000, &[cmd(op::PRIM, 1), cmd(op::RET, 0)]);
        let (result, opcodes, _) = run(&mut memory, 0);
        assert_eq!(result, WalkResult::Finished);
        assert!(!opcodes.contains(&op::PRIM), "third-level CALL must be skipped");
    }

    #[test]
    fn signal_jump_uses_target_from_signal_and_end() {
        let mut memory = TestMemory::default();
        // Absolute signal jump (behaviour 0x10) to 0x08920000.
        memory.write(LIST, &[cmd(op::SIGNAL, 0x10_0892), cmd(op::END, 0x0000)]);
        memory.write(0x0892_0000, &[cmd(op::PRIM, 3), cmd(op::FINISH, 0), cmd(op::END, 0)]);
        let (result, opcodes, _) = run(&mut memory, 0);
        assert_eq!(result, WalkResult::Finished);
        assert_eq!(opcodes, vec![op::SIGNAL, op::END, op::PRIM, op::FINISH, op::END]);
    }

    #[test]
    fn block_transfer_copies_a_rectangle_row_by_row() {
        use crate::ge_writes::{GeWrites, Overlaid};
        let mut inner = TestMemory::default();
        // Source: 16-bit pixels, rows of 8 pixels (16 bytes) at 0x08A00000;
        // pixel (x, y) holds y * 0x100 + x.
        for y in 0..4u32 {
            let row: Vec<u32> = (0..4).map(|pair| (y << 8 | (2 * pair + 1)) << 16 | (y << 8 | 2 * pair)).collect();
            inner.write(0x08A0_0000 + 16 * y, &row);
        }
        inner.write(0x0410_0000, &[0xAAAA_AAAA; 16]);
        memory_list(&mut inner, &[
            cmd(op::TRANSFER_SOURCE, 0xA0_0000),
            cmd(op::TRANSFER_SOURCE_WIDTH, 0x08_0000 | 8),
            cmd(op::TRANSFER_DESTINATION, 0x10_0000),
            cmd(op::TRANSFER_DESTINATION_WIDTH, 0x04_0000 | 8),
            cmd(op::TRANSFER_SOURCE_POSITION, (1 << 10) | 1), // from (1, 1)
            cmd(op::TRANSFER_DESTINATION_POSITION, 2),        // to (2, 0)
            cmd(op::TRANSFER_SIZE, (1 << 10) | 2),            // 3 x 2 pixels
            cmd(op::TRANSFER_START, 0),                       // 16-bit
            cmd(op::FINISH, 0),
            cmd(op::END, 0),
        ]);
        let mut writes = GeWrites::new();
        let mut memory = Overlaid { writes: &mut writes, inner: &mut inner };
        let mut list = DisplayList::new(0, LIST, 0);
        let result = walk(&mut list, &mut GeState::new(), &mut memory, &mut Recorder::default(), 1000, &mut 0);
        assert_eq!(result, WalkResult::Finished);
        let pixel = |memory: &mut Overlaid<TestMemory>, x: u32, y: u32| memory.read_u16(0x0410_0000 + 16 * y + 2 * x).unwrap();
        let row0: Vec<u16> = (0..8).map(|x| pixel(&mut memory, x, 0)).collect();
        assert_eq!(row0, [0xAAAA, 0xAAAA, 0x101, 0x102, 0x103, 0xAAAA, 0xAAAA, 0xAAAA]);
        let row1: Vec<u16> = (0..8).map(|x| pixel(&mut memory, x, 1)).collect();
        assert_eq!(row1, [0xAAAA, 0xAAAA, 0x201, 0x202, 0x203, 0xAAAA, 0xAAAA, 0xAAAA]);
    }

    fn memory_list(memory: &mut TestMemory, words: &[u32]) {
        memory.write(LIST, words);
    }

    #[test]
    fn endless_loop_is_cut_off() {
        let mut memory = TestMemory::default();
        memory.write(LIST, &[cmd(op::BASE, 0x08_0000), cmd(op::JUMP, 0x90_0000)]);
        let (result, _, _) = run(&mut memory, 0);
        assert_eq!(result, WalkResult::TooManyCommands);
    }

    #[test]
    fn unreadable_memory_is_reported() {
        let mut memory = TestMemory::default();
        memory.write(LIST, &[cmd(op::NOP, 0)]);
        let (result, _, _) = run(&mut memory, 0);
        assert_eq!(result, WalkResult::BadAddress(LIST + 4));
    }
}
