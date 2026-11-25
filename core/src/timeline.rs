//! Timeline spans for the profiling build: which thread did what when, as
//! complete spans (zone, start, end in us) handed to a sink registered from
//! outside (the debug plugin's ring, `timeline start|stop`, shown on the PC
//! in Perfetto: tools/timeline.py).
//!
//! Two kinds of zones:
//! - the measuring parts of [`crate::cycles`] (zones 0..32): a part whose bit
//!   is set in [`TIMELINE`] becomes a span at its existing `cycles::start` /
//!   `cycles::add` points (no new call sites in the hot code; `TIMELINE`
//!   wins over `cycles::MEASURED` for a part in both);
//! - the zones below (from [`FIRST_ZONE`] on), on threads the parts don't
//!   cover (display, copier, stage B, texture worker), on while [`ZONES`] is.
//!
//! A span costs two clock reads (`sceKernelGetProcessTimeWide`, ~2.4 us
//! each) plus the sink: keep zones coarse (whole jobs, frames, waits).
//! Without the `instrument` feature every check is a constant and all of
//! this compiles to nothing; with it but not recording, one load each.

use core::sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, Ordering};

/// The `cycles` parts recorded as spans (`1 << part`); 0: none.
pub static TIMELINE: AtomicU32 = AtomicU32::new(0);
/// Whether the zones below are recorded.
pub static ZONES: AtomicBool = AtomicBool::new(false);
static CLOCK: AtomicPtr<()> = AtomicPtr::new(core::ptr::null_mut());
static SINK: AtomicPtr<()> = AtomicPtr::new(core::ptr::null_mut());

/// Receives a finished span: zone, start and end (us, the clock's).
pub type Sink = fn(zone: u32, start_us: u32, end_us: u32);

/// The display thread drawing a frame (`frame_loop`, render start to end).
pub const FIRST_ZONE: u32 = crate::cycles::COUNT as u32;
pub const RENDER: u32 = FIRST_ZONE;
/// The display thread waiting in the frame pool for the GPU to finish a
/// segment (`frame_pool`): a GPU-bound frame shows up here.
pub const POOL_WAIT: u32 = FIRST_ZONE + 1;
/// The copier copying a job's deferred memory (`copy_thread`).
pub const COPY: u32 = FIRST_ZONE + 2;
/// Stage B working through what stage A queued (`stage_thread`).
pub const STAGE_B: u32 = FIRST_ZONE + 3;
/// The texture worker refreshing a texture (`textures`).
pub const TEXTURE: u32 = FIRST_ZONE + 4;
/// Markers (start = end) the debug plugin sets: the PSP game showing a
/// frame (`sceDisplaySetFrameBuf`) and the display loop queuing a buffer.
pub const PSP_FLIP: u32 = FIRST_ZONE + 5;
pub const SWAP: u32 = FIRST_ZONE + 6;
pub const ZONE_NAMES: [&str; 7] = ["render", "pool_wait", "copy", "stage_b", "texture", "psp_flip", "swap"];

/// The name of a zone (a `cycles` part's, or one of the zones above).
pub fn name(zone: u32) -> &'static str {
    let zone = zone as usize;
    if zone < crate::cycles::COUNT {
        crate::cycles::NAMES[zone]
    } else {
        ZONE_NAMES.get(zone - crate::cycles::COUNT).copied().unwrap_or("?")
    }
}

/// Registers the clock (us) and the sink (`None`: none; then nothing is
/// recorded even with bits set).
pub fn set(clock: Option<fn() -> u32>, sink: Option<Sink>) {
    CLOCK.store(clock.map_or(core::ptr::null_mut(), |clock| clock as *mut ()), Ordering::Release);
    SINK.store(sink.map_or(core::ptr::null_mut(), |sink| sink as *mut ()), Ordering::Release);
}

/// The clock now (0 without one).
pub fn now_us() -> u32 {
    let clock = CLOCK.load(Ordering::Acquire);
    if clock.is_null() {
        return 0;
    }
    // Safe: only `set` stores here, from a `fn() -> u32`.
    let clock: fn() -> u32 = unsafe { core::mem::transmute(clock) };
    clock()
}

/// Hands a span to the sink (if one is registered).
#[cold]
#[inline(never)]
pub fn emit(zone: u32, start_us: u32, end_us: u32) {
    let sink = SINK.load(Ordering::Acquire);
    if sink.is_null() {
        return;
    }
    // Safe: only `set` stores here, from a `Sink`.
    let sink: Sink = unsafe { core::mem::transmute(sink) };
    sink(zone, start_us, end_us);
}

/// Whether the zones above are recorded.
#[inline(always)]
pub fn on() -> bool {
    crate::bench::INSTRUMENTED && ZONES.load(Ordering::Relaxed)
}

/// Whether `cycles` part `part` is recorded as a span.
#[inline(always)]
pub fn part_on(part: usize) -> bool {
    crate::bench::INSTRUMENTED && TIMELINE.load(Ordering::Relaxed) & (1 << part) != 0
}

/// A span with times the caller already has (e.g. its own busy timers).
#[inline(always)]
pub fn span(zone: u32, start_us: u32, end_us: u32) {
    if on() {
        emit(zone, start_us, end_us);
    }
}

/// A span from now until the guard is dropped.
#[inline(always)]
pub fn zone(zone: u32) -> Zone {
    if on() {
        Zone { zone, start_us: now_us() }
    } else {
        Zone { zone: OFF, start_us: 0 }
    }
}

const OFF: u32 = u32::MAX;

/// See [`zone`].
pub struct Zone {
    zone: u32,
    start_us: u32,
}

impl Drop for Zone {
    #[inline(always)]
    fn drop(&mut self) {
        if self.zone != OFF {
            emit(self.zone, self.start_us, now_us());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_zone_has_a_name() {
        assert_eq!(name(0), "jobs");
        assert_eq!(name(RENDER), "render");
        assert_eq!(name(SWAP), "swap");
        assert_eq!(FIRST_ZONE as usize + ZONE_NAMES.len() - 1, SWAP as usize);
    }

    /// The tests below change the statics: one test, run in order.
    #[cfg(feature = "instrument")]
    #[test]
    fn spans_reach_the_sink_only_while_recording() {
        use crate::cycles;
        use std::sync::Mutex;
        static SPANS: Mutex<Vec<(u32, u32, u32)>> = Mutex::new(Vec::new());
        static TIME: AtomicU32 = AtomicU32::new(100);
        fn clock() -> u32 {
            TIME.fetch_add(10, Ordering::Relaxed)
        }
        fn sink(zone: u32, start: u32, end: u32) {
            SPANS.lock().unwrap().push((zone, start, end));
        }
        let taken = || core::mem::take(&mut *SPANS.lock().unwrap());

        // Nothing without the switches, even with a sink.
        set(Some(clock), Some(sink));
        drop(zone(RENDER));
        span(COPY, 1, 2);
        cycles::add(cycles::JOBS, cycles::start(cycles::JOBS));
        assert!(taken().is_empty());

        // Zones: a guard spans its life, a span keeps the given times.
        ZONES.store(true, Ordering::Relaxed);
        drop(zone(RENDER));
        span(COPY, 1, 2);
        assert_eq!(taken(), vec![(RENDER, 100, 110), (COPY, 1, 2)]);

        // A cycles part on the timeline: a span between start and add, not
        // counted as cycles.
        TIMELINE.store(1 << cycles::JOBS, Ordering::Relaxed);
        let started = cycles::start(cycles::JOBS);
        cycles::add(cycles::JOBS, started);
        assert_eq!(taken(), vec![(cycles::JOBS as u32, 120, 130)]);
        assert_eq!(cycles::ENTRIES[cycles::JOBS].load(Ordering::Relaxed), 0);
        // Other parts stay off.
        cycles::add(cycles::WALK, cycles::start(cycles::WALK));
        assert!(taken().is_empty());
        // A part that started before recording did (started = 0) ends
        // without a span (2026-11-25, GoW: recorder jobs of "122 s").
        TIMELINE.store(0, Ordering::Relaxed);
        let before = cycles::start(cycles::JOBS);
        TIMELINE.store(1 << cycles::JOBS, Ordering::Relaxed);
        cycles::add(cycles::JOBS, before);
        assert!(taken().is_empty());

        // No sink: nothing, and no crash.
        set(None, None);
        drop(zone(RENDER));
        cycles::add(cycles::JOBS, cycles::start(cycles::JOBS));
        assert!(taken().is_empty());
        ZONES.store(false, Ordering::Relaxed);
        TIMELINE.store(0, Ordering::Relaxed);
    }
}
