//! Our own settings: distant textures, texture filter, the game's fps
//! label and the ARM clock. Read once from
//! ux0:data/nativepsp/config.txt (nativepsp_core::options; we never write
//! it), changed later only by the debug server; nothing is saved.
//!
//! The renderer's on/off switch is Adrenaline's setting "NativePSP
//! Renderer" (below Graphics Filtering; saved with Adrenaline's settings).
//! With it on, the display loop draws whatever Graphics Filtering is
//! selected (menu.c); the selected filter is never changed (user,
//! 2026-11-14).

use core::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, AtomicU8, Ordering::Relaxed};

use nativepsp_core::options::{start_skip, Options};

use crate::{adrenaline, frame_loop, menu, render_log, sys};

/// Where we read our settings from.
const PATH: &[u8] = b"ux0:data/nativepsp/config.txt\0";
/// The file is a few lines; anything past this is ignored.
const MAX_BYTES: usize = 1024;

/// The renderer paused while Adrenaline saves or loads a state
/// (debug/state_ops.rs), whatever the switch says.
static PAUSED: AtomicBool = AtomicBool::new(false);

/// Distant textures (`mip=`, debug "mipmode", the fork's menu entry): 0
/// sharp, 1 blurry like the PSP, 2 mipmapped.
pub static MIP_MODE: AtomicU8 = AtomicU8::new(0);

/// `startskip=` (nativepsp_core::options::start_skip), read by
/// `read_start_skip` before the hooks are set.
static START_SKIP: AtomicU32 = AtomicU32::new(0);

/// Reads `startskip=` from the settings file (the plugin's start thread,
/// before anything else): the parts to leave out.
pub fn read_start_skip() -> u32 {
    let mut buffer = [0u8; MAX_BYTES];
    let skip = Options::parse(read_file(&mut buffer)).start_skip;
    START_SKIP.store(skip, Relaxed);
    skip
}

/// True if `part` (a start_skip bit) was left out.
pub fn skipped(part: u32) -> bool {
    START_SKIP.load(Relaxed) & part != 0
}

/// The ARM clock our thread keeps (`armclock=`, debug "armclock").
static ARM_CLOCK: AtomicI32 = AtomicI32::new(444);

/// 1 if the renderer is on: Adrenaline's "NativePSP Renderer" (on until
/// its settings are found) and not paused.
pub fn renderer_enabled() -> i32 {
    (!PAUSED.load(Relaxed) && adrenaline::renderer_switch().unwrap_or(true)) as i32
}

/// Sets "NativePSP Renderer" (debug "renderer"; not saved).
pub fn set_renderer(on: bool) {
    adrenaline::set_renderer_switch(on);
}

/// Pauses the renderer (savestates) or lets it go on.
pub fn set_paused(paused: bool) {
    PAUSED.store(paused, Relaxed);
}

pub fn mip_mode() -> u8 {
    MIP_MODE.load(Relaxed)
}

pub fn set_mip_mode(mode: u8) {
    MIP_MODE.store(mode, Relaxed);
}

/// The ARM clock (MHz) kept from now on; set at once.
pub fn keep_arm_clock(mhz: i32) -> i32 {
    ARM_CLOCK.store(mhz, Relaxed);
    // Safe: a plain system call.
    unsafe { sys::scePowerSetArmClockFrequency(mhz) }
}

/// The settings file's text, or nothing if there is none.
fn read_file(buffer: &mut [u8; MAX_BYTES]) -> &[u8] {
    // Safe: a NUL-terminated path; reads into our buffer only.
    unsafe {
        let fd = sys::sceIoOpen(PATH.as_ptr(), sys::SCE_O_RDONLY, 0);
        if fd < 0 {
            return &[];
        }
        let read = sys::sceIoRead(fd, buffer.as_mut_ptr().cast(), MAX_BYTES as u32);
        sys::sceIoClose(fd);
        &buffer[..read.clamp(0, MAX_BYTES as i32) as usize]
    }
}

/// Reads the settings file and applies it (at the start, before the first
/// frame is drawn).
fn load() {
    let mut buffer = [0u8; MAX_BYTES];
    let text = read_file(&mut buffer);
    let options = Options::parse(text);
    render_log::line(format_args!(
        "options: {} ({} bytes): mip={} smooth={} gamefps={} armclock={} startskip={}\n",
        if text.is_empty() { "defaults, no ux0:data/nativepsp/config.txt" } else { "config.txt" },
        text.len(),
        options.mip,
        options.smooth as i32,
        options.game_fps as i32,
        options.arm_clock,
        options.start_skip
    ));
    set_mip_mode(options.mip);
    frame_loop::SMOOTH_TEXTURES.store(options.smooth as u8, Relaxed);
    crate::display_hooks::timings::set_game_fps_label(options.game_fps);
    keep_arm_clock(options.arm_clock);
}

/// Our settings thread: applies the file, puts "NativePSP Renderer" into
/// Adrenaline's menu and keeps Graphics Filtering in line (menu.rs, every
/// 50 ms), and keeps the ARM clock (the system can lower it again, e.g.
/// after standby; checked every 10 s).
unsafe extern "C" fn options_thread(_args: u32, _argp: *mut core::ffi::c_void) -> i32 {
    load();
    let mut menu_installed = false;
    let mut tick = 0u32;
    loop {
        if skipped(start_skip::MENU) {
            // Left out (`startskip=`).
        } else if menu_installed {
            menu::sync();
        } else {
            menu_installed = menu::try_install();
        }
        tick = tick.wrapping_add(1);
        crate::blocks::note_free_user_memory();
        if tick % 200 == 0 {
            let mhz = ARM_CLOCK.load(Relaxed);
            if sys::scePowerGetArmClockFrequency() != mhz {
                sys::scePowerSetArmClockFrequency(mhz);
            }
        }
        sys::sceKernelDelayThread(50 * 1000);
    }
}

/// Starts the settings thread (with the capture, at Adrenaline's start).
pub fn start() {
    // Safe: plain system calls; the entry function matches `ThreadEntry`.
    unsafe {
        let thread = sys::sceKernelCreateThread(
            b"NativePspOptions\0".as_ptr(),
            options_thread,
            0x1000_0100,
            0x2000,
            0,
            0,
            core::ptr::null_mut(),
        );
        sys::note_thread(thread);
        if thread >= 0 {
            sys::sceKernelStartThread(thread, 0, core::ptr::null_mut());
        }
    }
}
