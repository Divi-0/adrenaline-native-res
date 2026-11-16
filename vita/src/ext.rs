//! Extension points: where a plugin built on top of this crate (a
//! development build with its own tools) can join in.
//!
//! The plugin as built here (plugin/) registers nothing: every function below then
//! does nothing or gives the default. A crate building its own plugin on
//! top of this one writes its own `module_start`, which calls
//! [`crate::plugin::start`] with its `Extensions`.
//!
//! All hooks are called on the renderer's own threads, in places where they
//! must not block for long: the display thread (`after_swap`,
//! `before_render`, `before_draw`, `frame_draw_started`, `frame_drawn`), the
//! capture thread (`dump_*`), the log writer (`log_send`).

use core::sync::atomic::{AtomicPtr, Ordering};

use nativeres_core::frame::Frame;

use crate::capture::CaptureState;
use crate::frame_loop::FrameLoop;
use crate::sys::SceUID;
use crate::targets::Targets;

/// The hooks of an extension. Every one is required; one that has nothing
/// to do returns at once.
pub struct Extensions {
    /// The plugin's start thread, before any hook is set: `skip` is
    /// `startskip=` (`nativeres_core::options::start_skip`).
    pub before_hooks: fn(skip: u32),
    /// The plugin's start thread, after the capture started.
    pub started: fn(skip: u32),
    /// A thread was made (the emulator's, Adrenaline's or ours).
    pub note_thread: fn(thread: SceUID),
    /// Display thread, right after the display loop queued its buffer.
    pub after_swap: fn(),
    /// PSP buttons to add to what the game reads (0: none).
    pub injected_buttons: fn() -> u32,
    /// Display thread, at the start of `render_frame`. True: the call drew
    /// something else; the frame isn't drawn this time.
    pub before_render: fn(state: &mut FrameLoop, targets: &mut Targets) -> bool,
    /// Display thread, before a frame is drawn. True: the call drew it
    /// itself; the frame loop shows what is there.
    pub before_draw: fn(state: &mut FrameLoop, targets: &mut Targets, frame: &Frame) -> bool,
    /// Display thread, when a frame starts being drawn.
    pub frame_draw_started: fn(frame: &Frame),
    /// Display thread, once a frame's render target is chosen.
    pub frame_drawn: fn(frame: &Frame),
    /// Capture thread: whether the next frame must be recorded whole.
    pub dump_requested: fn() -> bool,
    /// Capture thread, right after a frame ended (`recording`: the next
    /// frame is recorded).
    pub dump_start: fn(state: &mut CaptureState, recording: bool),
    /// Capture thread, when a recorded frame is complete.
    pub dump_finish: fn(state: &mut CaptureState, frame: &Frame),
    /// Log writer: sends capture log text somewhere. False: not sent (it
    /// waits in the log's ring).
    pub log_send: fn(bytes: &[u8]) -> bool,
    /// Whether the log has somewhere to go right now.
    pub log_connected: fn() -> bool,
    /// The calling thread onto the cores in `mask` when it includes core 3,
    /// which the system refuses to apps (< 0: refused).
    pub set_cores_with_core_3: fn(mask: i32) -> i32,
}

static REGISTERED: AtomicPtr<Extensions> = AtomicPtr::new(core::ptr::null_mut());

/// Registers `extensions` (`plugin::start`, before anything else runs).
pub fn register(extensions: &'static Extensions) {
    REGISTERED.store(extensions as *const Extensions as *mut Extensions, Ordering::Release);
}

/// The registered extension, if any.
#[inline]
pub fn get() -> Option<&'static Extensions> {
    let extensions = REGISTERED.load(Ordering::Acquire);
    // Safe: only ever set from a `&'static Extensions`.
    unsafe { extensions.as_ref() }
}

pub fn before_hooks(skip: u32) {
    if let Some(ext) = get() {
        (ext.before_hooks)(skip);
    }
}

pub fn started(skip: u32) {
    if let Some(ext) = get() {
        (ext.started)(skip);
    }
}

pub fn note_thread(thread: SceUID) {
    if let Some(ext) = get() {
        (ext.note_thread)(thread);
    }
}

pub fn after_swap() {
    if let Some(ext) = get() {
        (ext.after_swap)();
    }
}

pub fn injected_buttons() -> u32 {
    get().map_or(0, |ext| (ext.injected_buttons)())
}

pub fn before_render(state: &mut FrameLoop, targets: &mut Targets) -> bool {
    get().is_some_and(|ext| (ext.before_render)(state, targets))
}

pub fn before_draw(state: &mut FrameLoop, targets: &mut Targets, frame: &Frame) -> bool {
    get().is_some_and(|ext| (ext.before_draw)(state, targets, frame))
}

pub fn frame_draw_started(frame: &Frame) {
    if let Some(ext) = get() {
        (ext.frame_draw_started)(frame);
    }
}

pub fn frame_drawn(frame: &Frame) {
    if let Some(ext) = get() {
        (ext.frame_drawn)(frame);
    }
}

pub fn dump_requested() -> bool {
    get().is_some_and(|ext| (ext.dump_requested)())
}

pub fn dump_start(state: &mut CaptureState, recording: bool) {
    if let Some(ext) = get() {
        (ext.dump_start)(state, recording);
    }
}

pub fn dump_finish(state: &mut CaptureState, frame: &Frame) {
    if let Some(ext) = get() {
        (ext.dump_finish)(state, frame);
    }
}

pub fn log_send(bytes: &[u8]) -> bool {
    get().is_some_and(|ext| (ext.log_send)(bytes))
}

pub fn log_connected() -> bool {
    get().is_some_and(|ext| (ext.log_connected)())
}

/// `None` without an extension that can do it.
pub fn set_cores_with_core_3(mask: i32) -> Option<i32> {
    get().map(|ext| (ext.set_cores_with_core_3)(mask))
}
