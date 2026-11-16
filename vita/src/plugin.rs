//! The plugin's start (adrenaline-native-res.suprx, under `*PSPEMUCFW` in
//! ur0:tai/config.txt): taiHEN loads it into Adrenaline's process
//! (ScePspemu) when the process starts, AdrenalineUser already loaded.
//!
//! `module_start` (plugin/src/lib.rs) calls `start`, which only starts a
//! thread, which:
//! 1. waits for AdrenalineUser (Adrenaline's user module);
//! 2. hooks its imports (display_hooks.rs, io_hooks.rs) and ScePspemu's,
//!    installs the PSP plugin it carries (psp_plugin.rs);
//! 3. waits until Adrenaline starts its own threads (InitAdrenaline, when
//!    the PSP side boots: the moment the fork called our start), seen by
//!    the hook on AdrenalineUser's sceKernelCreateThread; if that was
//!    before our hook (never seen so far), after `START_WAIT_US` anyway;
//! 4. starts ours: capture (with the log writer and the options thread:
//!    ARM clock, menu entry), then the extension's part, if there is one
//!    (`ext`).
//!
//! `startskip=` in ux0:data/nativeres/config.txt (read first) leaves parts
//! out, to bisect a crash at Adrenaline's start (C1-2858-3 hunt).

use core::ffi::c_void;
use core::fmt::Write;
use core::sync::atomic::{AtomicU64, Ordering::Relaxed};

use nativeres_core::options::start_skip;
use nativeres_core::report::TextBuffer;

use crate::{capture, display_hooks, ext, io_hooks, log, options, psp_plugin, render_log, sys};

/// Process time (us) of module_start (the start's timeline goes into the
/// capture log, which the PC keeps even when the start crashes).
static MODULE_START_US: AtomicU64 = AtomicU64::new(0);

/// How long to wait for Adrenaline's threads before starting anyway.
const START_WAIT_US: u64 = 10_000_000;

/// tai_module_info_t (taihen.h).
#[repr(C)]
struct TaiModuleInfo {
    size: u32,
    modid: sys::SceUID,
    module_nid: u32,
    name: [u8; 27],
    exports_start: usize,
    exports_end: usize,
    imports_start: usize,
    imports_end: usize,
}

extern "C" {
    fn taiGetModuleInfo(module: *const u8, info: *mut TaiModuleInfo) -> i32;
}

fn loaded(name: &[u8]) -> bool {
    // Safe: all zeroes is a valid TaiModuleInfo.
    let mut info: TaiModuleInfo = unsafe { core::mem::zeroed() };
    info.size = core::mem::size_of::<TaiModuleInfo>() as u32;
    // Safe: `name` is NUL-terminated, `info` the size taiHEN expects.
    unsafe { taiGetModuleInfo(name.as_ptr(), &mut info) >= 0 }
}

unsafe extern "C" fn start_thread(_args: u32, _argp: *mut c_void) -> i32 {
    while !loaded(b"AdrenalineUser\0") {
        sys::sceKernelDelayThread(10 * 1000);
    }
    let skip = options::read_start_skip();
    let hooked_us = sys::now_us();
    ext::before_hooks(skip);
    if skip & start_skip::DISPLAY_HOOKS == 0 {
        display_hooks::install();
    }
    if skip & start_skip::IO_HOOKS == 0 {
        io_hooks::install();
    }
    let hooks_done_us = sys::now_us();
    // Before the PSP side boots (when Adrenaline starts its threads).
    if skip & start_skip::PSP_PLUGIN == 0 {
        psp_plugin::install();
    }
    let waited_from = sys::now_us();
    while !display_hooks::adrenaline_started() && sys::now_us() - waited_from < START_WAIT_US {
        sys::sceKernelDelayThread(10 * 1000);
    }
    let seen = display_hooks::adrenaline_started();
    render_log::line(format_args!(
        "plugin: hooked at {} us, Adrenaline's threads {} at {} us, startskip={}\n",
        hooked_us,
        if seen { "started" } else { "not seen, starting anyway" },
        sys::now_us(),
        skip
    ));
    if skip & start_skip::CAPTURE == 0 {
        capture::start();
    }
    let mut bytes = [0u8; 192];
    let mut text = TextBuffer::new(&mut bytes);
    let _ = write!(
        text,
        "plugin: module_start {} us, AdrenalineUser there {} us, hooks set {} us, Adrenaline's threads {} {} us, startskip={}\n",
        MODULE_START_US.load(Relaxed),
        hooked_us,
        hooks_done_us,
        if seen { "at" } else { "not seen, start at" },
        sys::now_us(),
        skip
    );
    let length = text.len();
    log::append(&bytes[..length]);
    ext::started(skip);
    sys::sceKernelExitDeleteThread(0)
}

/// The plugin's start, for its `module_start` (plugin/src/lib.rs, or a
/// crate building its own plugin): registers `extensions` (none for
/// the plugin built here, see `ext`) and starts the thread that does the rest.
/// Returns what `module_start` returns.
pub fn start(extensions: Option<&'static ext::Extensions>) -> i32 {
    MODULE_START_US.store(sys::now_us(), Relaxed);
    if let Some(extensions) = extensions {
        ext::register(extensions);
    }
    // Safe: plain system calls; `start_thread` matches `ThreadEntry`.
    unsafe {
        let thread = sys::sceKernelCreateThread(b"NativeResStart\0".as_ptr(), start_thread, 0x1000_0100, 0x2000, 0, 0, core::ptr::null_mut());
        if thread >= 0 {
            sys::sceKernelStartThread(thread, 0, core::ptr::null_mut());
        }
    }
    0 // SCE_KERNEL_START_SUCCESS
}
