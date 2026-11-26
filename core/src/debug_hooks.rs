//! Debug hooks: named points in the renderer where a debugging tool built on
//! top of it (the development plugin) can run its own code, like an event
//! handler: it registers a function for a point (`set`), and the renderer
//! calls it whenever it reaches that point (`call`, `span`).
//!
//! Only with the `debug-hooks` feature. Without it (the release plugin)
//! `call`/`span` are empty inline functions: nothing of this is in the
//! binary, not even a check. With it, a point costs one load while nothing is
//! registered: keep points coarse (per job, copy, frame, wait).
//!
//! What `a` and `b` mean, per point:
//!
//! | Point | where | a | b | returns |
//! |---|---|---|---|---|
//! | `Render` | display thread, a frame drawn (`frame_loop`) | start us | end us | - |
//! | `Copy` | copier, a job's memory copied (`copy_thread`) | start us | end us | - |
//! | `PoolWait` | display thread waits for the GPU (`frame_pool`) | `BEGIN`/`END` | 0 / begin's value | begin: a token |
//! | `Texture` | texture worker refreshes a texture (`textures`) | `BEGIN`/`END` | 0 / begin's value | begin: a token |
//! | `PartStart` | a measuring part of `cycles` starts | part | - | the value `PartEnd` gets |
//! | `PartEnd` | that part ends | part | `PartStart`'s value (0: started before the hook) | - |
//! | `Draw` | display thread, a textured frame draw just issued (`draw.rs`, after its `sceGxmDraw`) | index address | index count \| primitive type (bits 24..31) | - |
//! | `FragmentMade` | a fragment program was made (`basic_programs::fragment_program`) | the program | address of three words: its patcher id, its blend info (or null), the vertex program's GXP it is linked to | - |
//! | `FragmentSet` | display thread, a frame draw is about to set its fragment program (`draw.rs`) | the program | 0 | 0, or another fragment program to set in its place |
//! | `VertexMade` | a vertex program was made (`basic_programs::vertex_program`) | the program | address of five words: its patcher id, its attributes (address, count), its streams (address, count) | - |
//! | `VertexSet` | display thread, a frame draw is about to set its vertex program (`draw.rs`) | the program | 0 | 0, or another vertex program to set in its place |
//! | `VertexUniforms` | display thread, a default uniform buffer was reserved for the vertex program that is set (`draw.rs`); the renderer writes all of that program's uniforms right after | the buffer | 0 | - |
//!
//! Times are the process time in us (low 32 bits). A hook on `PartStart` /
//! `PartEnd` takes over the `cycles` parts that are measured
//! (`cycles::MEASURED`) from the cycle counting.

/// The points (see the table above).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Point {
    Render,
    Copy,
    PoolWait,
    Texture,
    PartStart,
    PartEnd,
    Draw,
    FragmentMade,
    FragmentSet,
    VertexMade,
    VertexSet,
    VertexUniforms,
}

pub const POINTS: usize = 12;
/// `a` of a `span` point at its begin and end.
pub const BEGIN: u32 = 0;
pub const END: u32 = 1;

/// A registered hook: the point, `a`, `b` (see the table) -> a value.
pub type Hook = fn(point: Point, a: u32, b: u32) -> u32;

/// Whether this build has the hooks at all.
pub const ENABLED: bool = cfg!(feature = "debug-hooks");

#[cfg(feature = "debug-hooks")]
static HOOKS: [core::sync::atomic::AtomicPtr<()>; POINTS] = [const { core::sync::atomic::AtomicPtr::new(core::ptr::null_mut()) }; POINTS];

/// Registers `hook` for `point` (`None`: removes it). Without the feature
/// nothing is ever called.
pub fn set(point: Point, hook: Option<Hook>) {
    #[cfg(feature = "debug-hooks")]
    HOOKS[point as usize].store(hook.map_or(core::ptr::null_mut(), |hook| hook as *mut ()), core::sync::atomic::Ordering::Release);
    #[cfg(not(feature = "debug-hooks"))]
    let _ = (point, hook);
}

/// Whether a hook is registered for `point`.
#[inline(always)]
pub fn hooked(point: Point) -> bool {
    #[cfg(feature = "debug-hooks")]
    return !HOOKS[point as usize].load(core::sync::atomic::Ordering::Relaxed).is_null();
    #[cfg(not(feature = "debug-hooks"))]
    {
        let _ = point;
        false
    }
}

/// Reaching `point`: calls its hook if one is registered (else 0).
#[inline(always)]
pub fn call(point: Point, a: u32, b: u32) -> u32 {
    #[cfg(feature = "debug-hooks")]
    {
        let hook = HOOKS[point as usize].load(core::sync::atomic::Ordering::Acquire);
        if !hook.is_null() {
            return call_registered(hook, point, a, b);
        }
    }
    #[cfg(not(feature = "debug-hooks"))]
    let _ = (point, a, b);
    0
}

/// The call itself, kept out of the inlined check.
#[cfg(feature = "debug-hooks")]
#[cold]
#[inline(never)]
fn call_registered(hook: *mut (), point: Point, a: u32, b: u32) -> u32 {
    // Safe: only `set` stores here, from a `Hook`.
    let hook: Hook = unsafe { core::mem::transmute(hook) };
    hook(point, a, b)
}

/// A `span` point: its hook is called with `BEGIN` now and with `END` (and
/// the begin's value) when the guard is dropped.
#[inline(always)]
pub fn span(point: Point) -> Span {
    #[cfg(feature = "debug-hooks")]
    if hooked(point) {
        return Span { point, token: call(point, BEGIN, 0), on: true };
    }
    Span { point, token: 0, on: false }
}

/// See [`span`].
pub struct Span {
    point: Point,
    token: u32,
    on: bool,
}

impl Drop for Span {
    #[inline(always)]
    fn drop(&mut self) {
        if ENABLED && self.on {
            call(self.point, END, self.token);
        }
    }
}

#[cfg(all(test, feature = "debug-hooks", feature = "instrument"))]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static CALLS: Mutex<Vec<(Point, u32, u32)>> = Mutex::new(Vec::new());

    fn record(point: Point, a: u32, b: u32) -> u32 {
        CALLS.lock().unwrap().push((point, a, b));
        if point == Point::PartStart {
            return 500 + a;
        }
        if a == BEGIN { 77 } else { 0 }
    }

    fn taken() -> Vec<(Point, u32, u32)> {
        core::mem::take(&mut *CALLS.lock().unwrap())
    }

    /// One test: the hooks are global.
    #[test]
    fn registered_hooks_run_where_the_renderer_reaches_their_point() {
        use crate::cycles;
        use core::sync::atomic::Ordering;
        // Nothing registered: nothing called, spans inert.
        call(Point::Copy, 1, 2);
        drop(span(Point::PoolWait));
        assert!(taken().is_empty());

        // A plain point gets its two values; a span gets BEGIN, then END
        // with the begin's value.
        set(Point::Copy, Some(record));
        set(Point::PoolWait, Some(record));
        call(Point::Copy, 1, 2);
        drop(span(Point::PoolWait));
        assert_eq!(taken(), vec![(Point::Copy, 1, 2), (Point::PoolWait, BEGIN, 0), (Point::PoolWait, END, 77)]);
        // Other points stay quiet.
        call(Point::Render, 1, 2);
        assert!(taken().is_empty());

        // Hooks on the parts take over measured cycles parts: PartEnd gets
        // PartStart's value, and nothing is counted as cycles.
        cycles::MEASURED.store(1 << cycles::JOBS, Ordering::Relaxed);
        set(Point::PartStart, Some(record));
        set(Point::PartEnd, Some(record));
        cycles::add(cycles::JOBS, cycles::start(cycles::JOBS));
        let jobs = cycles::JOBS as u32;
        assert_eq!(taken(), vec![(Point::PartStart, jobs, 0), (Point::PartEnd, jobs, 500 + jobs)]);
        assert_eq!(cycles::ENTRIES[cycles::JOBS].load(Ordering::Relaxed), 0);
        // A part that started before the hook: PartEnd with 0 (the hook
        // decides; the timeline drops it).
        set(Point::PartStart, None);
        let before = cycles::start(cycles::JOBS);
        set(Point::PartStart, Some(record));
        cycles::add(cycles::JOBS, before);
        assert_eq!(taken(), vec![(Point::PartEnd, jobs, before)]);
        cycles::MEASURED.store(0, Ordering::Relaxed);
        cycles::clear();

        for point in [Point::Copy, Point::PoolWait, Point::PartStart, Point::PartEnd] {
            set(point, None);
        }
        call(Point::Copy, 1, 2);
        assert!(taken().is_empty());
    }
}
