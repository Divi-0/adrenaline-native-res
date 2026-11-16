//! Where the capture thread is, for a sampling profile on the Vita.
//!
//! PC instruction counts don't predict Vita time well, and the Vita gives
//! user code no profiler. So the capture thread notes which part of its
//! work it is in (one store per change, [`enter`] / [`leave`]), and a
//! sampler thread on another core reads [`CURRENT`] every few microseconds
//! while `capturebench` runs (debug "capturebench <runs> profile"): the
//! samples per phase split the time up. Only the capture thread (the draw
//! stream's producer) notes phases.
//!
//! Only while a profile runs ([`ENABLED`]): with the capture in two stages
//! (`walk_stream`) both threads pass here, and two cores storing to one
//! cache line all the time slowed both down.

use core::sync::atomic::{AtomicBool, AtomicU8, Ordering};

/// The phase the capture thread is in now.
pub static CURRENT: AtomicU8 = AtomicU8::new(OTHER);

/// Phases are noted only while this is set (the Vita's profile sampler).
pub static ENABLED: AtomicBool = AtomicBool::new(false);

pub const OTHER: u8 = 0;
/// `display_list::walk`'s loop for commands that aren't plain (PRIM, CALL,
/// RET, VADDR, ...), without what it calls below.
pub const WALK: u8 = 1;
/// Plain commands in runs (`run_plain_commands`), without matrix uploads.
pub const PLAIN: u8 = 2;
/// Matrix DATA runs (`GeState::upload_matrix_run`).
pub const MATRIX: u8 = 3;
/// The recorder's per-PRIM work not listed below (`record_prims`, batching,
/// statistics, building the job).
pub const RECORD: u8 = 4;
/// `prepare`: checking and working out the draw state.
pub const PREPARE: u8 = 5;
/// Palette loads (CLUT_LOAD).
pub const PALETTE: u8 = 6;
/// `DrawStream::queue` (copies into the stream, waiting for room).
pub const QUEUE: u8 = 7;
/// Transform lookups (`TransformCache`).
pub const TRANSFORM: u8 = 8;
/// Copying PSP memory into the block cache (`BlockCache::fill_lines`).
pub const COPY: u8 = 9;
/// Scanning indices (`index_bounds`).
pub const INDICES: u8 = 10;
/// The live capture thread sleeping until the next poll (debug
/// "liveprofile"): nothing to do.
pub const IDLE: u8 = 11;
/// The producer waiting for the recorder thread (room in the draw stream,
/// the frame's end) or for the copier (GE sync handshake).
pub const WAIT: u8 = 12;
/// The producer copying jobs' memory a lagging copier hasn't taken
/// (`DrawStream::help_a_lagging_copier`, `copy_untaken`).
pub const HELP: u8 = 13;

pub const NAMES: [&str; 14] = [
    "other", "walk", "plain", "matrix", "record", "prepare", "palette", "queue", "transform", "copy", "indices", "idle",
    "wait", "help",
];

/// Enters `phase`; returns the phase to go back to (`leave`).
#[inline(always)]
pub fn enter(phase: u8) -> u8 {
    if !ENABLED.load(Ordering::Relaxed) {
        return OTHER;
    }
    let previous = CURRENT.load(Ordering::Relaxed);
    CURRENT.store(phase, Ordering::Relaxed);
    previous
}

/// Goes back to `previous` (what `enter` returned).
#[inline(always)]
pub fn leave(previous: u8) {
    if ENABLED.load(Ordering::Relaxed) {
        CURRENT.store(previous, Ordering::Relaxed);
    }
}
