//! CPU cycles (or a PMU event) per part of the capture's work, for
//! measuring on the Vita.
//!
//! The clock is a function registered from outside (`set_clock`), e.g. a
//! system call that reads the core's cycle counter in kernel mode (reading
//! it in user mode crashed). One read costs a system call (~1 us), so
//! parts are whole jobs and runs, and [`CALIBRATION`] measures what an
//! empty part costs, to subtract per entry. The measured thread must stay
//! on one core (every core has its own counter). Only the parts in
//! [`MEASURED`] are read: live, reading all of them (~13,000 capture-thread
//! parts per race frame) slowed the capture until it skipped frames, so
//! live, measure one thread's parts. Not measuring, a part costs
//! one load of that mask.

use core::sync::atomic::{AtomicPtr, AtomicU32, Ordering};

/// The parts measured, one bit per part (`1 << part`); 0: none.
pub static MEASURED: AtomicU32 = AtomicU32::new(0);
static CLOCK: AtomicPtr<()> = AtomicPtr::new(core::ptr::null_mut());

/// A whole `recorder::run_job`.
pub const JOBS: usize = 0;
/// Copying a job's deferred memory (`DrawStream::record_published`).
pub const COPIES: usize = 1;
/// `record_3d_run`, by outcome: from the vertex cache, decoded, or handed
/// to the general path (frame full / not handled).
pub const RUNS_CACHED: usize = 2;
pub const RUNS_DECODED: usize = 3;
pub const RUNS_FAILED: usize = 4;
/// `record_vertices` as a whole (its fast-path try is in `RUNS_*` too).
pub const GENERAL: usize = 5;
/// `vertex_cache::data_hash` of a run.
pub const HASH: usize = 6;
/// Writing a run's triangle indices.
pub const INDICES: usize = 7;
/// An empty part: the cost of measuring one entry.
pub const CALIBRATION: usize = 8;
// The capture thread (measured in `bench::Mode::Produce`, where the
// recorder thread only discards the jobs):
/// A whole `display_list::walk`.
pub const WALK: usize = 9;
/// `run_plain_commands` (inside WALK).
pub const PLAIN: usize = 10;
/// `Recorder::visit_prims` (inside WALK): a batch of PRIMs worked out and
/// queued, with PREPARE and QUEUE in it.
pub const PRIMS: usize = 11;
pub const PREPARE: usize = 12;
pub const QUEUE: usize = 13;
/// `Recorder::visit` for other commands (palette loads, statistics).
pub const VISIT: usize = 14;
/// Copying PSP memory into the block cache (`BlockCache::fill_lines`).
pub const LINES: usize = 15;
/// Refilling the walk's command window (`ListWindow::refill`, with the
/// LINES it causes), and matrix DATA runs (`GeState::upload_matrix_run`);
/// both inside PLAIN or WALK.
pub const REFILL: usize = 16;
pub const MATRIX: usize = 17;
/// Of RUNS_CACHED: the runs found by their mini hash alone
/// (`vertex_cache::find_quick`; measured instead of RUNS_CACHED).
pub const RUNS_QUICK: usize = 18;
/// Inside a run (`record_3d_run`): the vertex cache lookup (`find` and the
/// near-plane check of a hit), and the GPU transform entry with its light
/// set (`gpu_transform`, `Lights::gpu`, `add_light_set`, `add_transform`).
pub const RUN_LOOKUP: usize = 19;
pub const RUN_ENTRY: usize = 20;
/// A raw draw written on the capture thread (`frame_builder::FrameBuilder::try_record`).
pub const DIRECT: usize = 21;
/// Inside DIRECT: a draw's transform entry looked up or added
/// (`raw_blocks::raw_transform_with`; repeated entries don't get here).
pub const DIRECT_TRANSFORM: usize = 22;
/// Inside DIRECT: the draw's raw colours (`recorder::raw_colors_if_enabled`),
/// its copy handed over (`Writer::copy`), arena room reserved
/// (`RawArena::reserve`), the draw finished or merged (`Geometry::finish_raw_*`).
pub const DIRECT_COLORS: usize = 23;
pub const DIRECT_COPY: usize = 24;
pub const DIRECT_RESERVE: usize = 25;
pub const DIRECT_FINISH: usize = 26;
/// Inside PRIMS: a draw's texture sampled at its first use in the frame
/// (`recorder::sample_texture`).
pub const SAMPLE: usize = 27;
/// Inside DIRECT_TRANSFORM: the camera worked out and looked up
/// (`TransformInputs::raw_camera`, `RawTables::add_raw_camera`), the light
/// set (`gpu_lights_cached`, `add_light_set`).
pub const DIRECT_CAMERA: usize = 28;
pub const DIRECT_LIGHTS: usize = 29;
pub const COUNT: usize = 30;
pub const NAMES: [&str; COUNT] = [
    "jobs",
    "copies",
    "runs_cached",
    "runs_decoded",
    "runs_failed",
    "general",
    "hash",
    "indices",
    "calibration",
    "walk",
    "plain",
    "prims",
    "prepare",
    "queue",
    "visit",
    "lines",
    "refill",
    "matrix",
    "runs_quick",
    "run_lookup",
    "run_entry",
    "direct",
    "d_transform",
    "d_colors",
    "d_copy",
    "d_reserve",
    "d_finish",
    "sample",
    "d_camera",
    "d_lights",
];

/// Readings left out because they came out implausibly large (from 2^30
/// cycles or events on, ~2.4 s at 444 MHz): the counters were reset during
/// the part (something else on the core does that now and then: one part
/// per measurement read ~2^32 too much), or the thread changed cores.
pub static DROPPED: AtomicU32 = AtomicU32::new(0);
const IMPLAUSIBLE: u32 = 1 << 30;

/// Cycles per part and entries (the measuring thread alone writes them).
pub static TOTALS: [AtomicU32; COUNT] = [const { AtomicU32::new(0) }; COUNT];
pub static ENTRIES: [AtomicU32; COUNT] = [const { AtomicU32::new(0) }; COUNT];

/// Every part (`MEASURED`).
pub const ALL_PARTS: u32 = (1 << COUNT) - 1;
/// The recorder thread's parts, and the capture thread's (with the
/// calibration in both).
pub const RECORDER_PARTS: u32 = 1 << JOBS
    | 1 << COPIES
    | 1 << RUNS_CACHED
    | 1 << RUNS_DECODED
    | 1 << RUNS_FAILED
    | 1 << RUNS_QUICK
    | 1 << RUN_LOOKUP
    | 1 << RUN_ENTRY
    | 1 << GENERAL
    | 1 << HASH
    | 1 << INDICES
    | 1 << CALIBRATION;
pub const CAPTURE_PARTS: u32 = (ALL_PARTS & !RECORDER_PARTS) | 1 << CALIBRATION;

/// Registers the clock (`None`: none).
pub fn set_clock(clock: Option<fn() -> u32>) {
    CLOCK.store(clock.map_or(core::ptr::null_mut(), |clock| clock as *mut ()), Ordering::Release);
}

#[inline]
fn now() -> u32 {
    let clock = CLOCK.load(Ordering::Acquire);
    if clock.is_null() {
        return 0;
    }
    // Safe: only `set_clock` stores here, from a `fn() -> u32`.
    let clock: fn() -> u32 = unsafe { core::mem::transmute(clock) };
    clock()
}

/// Whether `part` is being measured.
#[inline(always)]
fn measured(part: usize) -> bool {
    MEASURED.load(Ordering::Relaxed) & (1 << part) != 0
}

/// The counter now if `part` is measured, else 0. (Parts measured by
/// outcome, `recorder::record_3d_run`, start with one of theirs.)
#[inline(always)]
pub fn start(part: usize) -> u32 {
    if measured(part) {
        now()
    } else {
        0
    }
}

/// Adds the cycles since `started` to `part`.
#[inline(always)]
pub fn add(part: usize, started: u32) {
    if measured(part) {
        let cycles = now().wrapping_sub(started);
        if cycles >= IMPLAUSIBLE {
            DROPPED.store(DROPPED.load(Ordering::Relaxed).wrapping_add(1), Ordering::Relaxed);
            return;
        }
        TOTALS[part].store(TOTALS[part].load(Ordering::Relaxed).wrapping_add(cycles), Ordering::Relaxed);
        ENTRIES[part].store(ENTRIES[part].load(Ordering::Relaxed).wrapping_add(1), Ordering::Relaxed);
    }
}

/// Zeroes the totals.
pub fn clear() {
    for index in 0..COUNT {
        TOTALS[index].store(0, Ordering::Relaxed);
        ENTRIES[index].store(0, Ordering::Relaxed);
    }
    DROPPED.store(0, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The crash of 2026-11-06: the counter
    /// read in user mode trapped. Now nothing reads it unless measuring was
    /// started and a clock (the kernel call) registered.
    #[test]
    fn crash_20261106_nothing_is_read_unless_started() {
        assert_eq!(MEASURED.load(Ordering::Relaxed), 0);
        assert_eq!(start(JOBS), 0);
        add(JOBS, 0);
        assert_eq!(ENTRIES[JOBS].load(Ordering::Relaxed), 0);
    }

    #[test]
    fn the_thread_masks_cover_every_part_once_but_the_calibration() {
        assert_eq!(RECORDER_PARTS | CAPTURE_PARTS, ALL_PARTS);
        assert_eq!(RECORDER_PARTS & CAPTURE_PARTS, 1 << CALIBRATION);
        assert!(COUNT <= 32);
    }
}
