//! Where the capture and recorder threads are, for a sampling profile on
//! the Vita.
//!
//! PC instruction counts don't predict Vita time well, and the Vita gives
//! user code no profiler. So the capture thread notes which part of its
//! work it is in (one store per change, [`enter`] / [`leave`]), and a
//! sampler thread on another core reads [`CURRENT`] every few microseconds
//! while the capture runs: the samples per phase split the time up. The
//! recorder thread (the draw stream's consumer) notes its own phases in
//! [`RECORDER`] the same way ([`recorder::enter`] / [`recorder::leave`]).
//!
//! Only while a profile runs ([`ENABLED`]): with the capture in two stages
//! (`walk_stream`) both threads passed here, and two cores storing to one
//! cache line all the time slowed both down. For the same reason the two
//! words lie on cache lines of their own ([`PhaseWord`]).

use core::sync::atomic::{AtomicBool, AtomicU8, Ordering};

/// One thread's phase, alone on its cache line (the Vita's lines are 64
/// bytes): the capture thread and the recorder thread write theirs on
/// different cores.
#[repr(align(64))]
pub struct PhaseWord(AtomicU8);

impl PhaseWord {
    const fn new(phase: u8) -> PhaseWord {
        PhaseWord(AtomicU8::new(phase))
    }

    /// The phase noted last.
    #[inline(always)]
    pub fn load(&self) -> u8 {
        self.0.load(Ordering::Relaxed)
    }

    /// Notes `phase`, returning the one noted before.
    #[inline(always)]
    fn switch(&self, phase: u8) -> u8 {
        let previous = self.0.load(Ordering::Relaxed);
        self.0.store(phase, Ordering::Relaxed);
        previous
    }
}

/// The phase the capture thread is in now.
pub static CURRENT: PhaseWord = PhaseWord::new(OTHER);

/// The phase the recorder thread is in now (`recorder`'s phases).
pub static RECORDER: PhaseWord = PhaseWord::new(recorder::IDLE);

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
/// The live capture thread sleeping until the next poll: nothing to do.
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
    CURRENT.switch(phase)
}

/// Goes back to `previous` (what `enter` returned).
#[inline(always)]
pub fn leave(previous: u8) {
    if ENABLED.load(Ordering::Relaxed) {
        CURRENT.0.store(previous, Ordering::Relaxed);
    }
}

/// The recorder thread's phases (`RECORDER`): where the time of the draw
/// stream's consumer goes (`DrawStream::record_published` and below).
///
/// Draws too big for the stream are recorded by the capture thread itself
/// (`FrameStats::inline_draws`, rare); their phases land in `RECORDER` too.
pub mod recorder {
    use super::{Ordering, ENABLED, RECORDER};

    /// Recording jobs, in nothing listed below (taking the job, finishing
    /// draws, index runs, statistics).
    pub const OTHER: u8 = 0;
    /// Working a job's transform out of its inputs (`TransformBuilder::build`).
    pub const TRANSFORM: u8 = 1;
    /// Building a raw draw's table entries (`recorder::raw_entry`: the GPU
    /// transform, light set, raw colours, bones).
    pub const RAW_ENTRY: u8 = 2;
    /// Copying an indexed draw's vertices one by one in index order
    /// (`recorder::record_raw_gathered`).
    pub const GATHER: u8 = 3;
    /// Converting indexed PRIMs' indices into a triangle list
    /// (`recorder::record_raw_converted`).
    pub const CONVERT: u8 = 4;
    /// Scanning indices for their lowest and highest (`index_bounds`).
    pub const INDICES: u8 = 5;
    /// Copying raw vertex bytes into the raw arena (`RawArena::put`, raw
    /// sprites' corners).
    pub const ARENA: u8 = 6;
    /// Decoding vertices on the CPU (the fallback: `record_3d_run_now`,
    /// `record_vertices_now`).
    pub const CPU_DECODE: u8 = 7;
    /// Storing a palette's words (`recorder::store_palette`).
    pub const PALETTE: u8 = 8;
    /// Reading a job's memory itself (a range no copier copied).
    pub const COPY: u8 = 9;
    /// Waiting for the copier to finish the next job.
    pub const WAIT: u8 = 10;
    /// Nothing to record: asleep until the capture thread wakes it.
    pub const IDLE: u8 = 11;

    pub const NAMES: [&str; 12] = [
        "other", "transform", "raw_entry", "gather", "convert", "indices", "arena", "cpu_decode", "palette", "copy", "wait",
        "idle",
    ];

    /// Enters `phase`; returns the phase to go back to (`leave`).
    #[inline(always)]
    pub fn enter(phase: u8) -> u8 {
        if !ENABLED.load(Ordering::Relaxed) {
            return OTHER;
        }
        RECORDER.switch(phase)
    }

    /// Goes back to `previous` (what `enter` returned).
    #[inline(always)]
    pub fn leave(previous: u8) {
        if ENABLED.load(Ordering::Relaxed) {
            RECORDER.0.store(previous, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_two_phase_words_lie_on_different_cache_lines() {
        let capture = &CURRENT as *const PhaseWord as usize;
        let recorder = &RECORDER as *const PhaseWord as usize;
        assert_eq!(capture % 64, 0);
        assert_eq!(recorder % 64, 0);
        assert_ne!(capture, recorder);
    }
}
