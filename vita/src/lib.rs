//! The Vita side of NativeRes: the taiHEN plugin (plugin/), which sits
//! in stock Adrenaline's process (scripts/build-plugin.sh).
//!
//! - `plugin`: module_start; waits for AdrenalineUser, sets the hooks,
//!   starts our threads when Adrenaline starts its own.
//! - `display_hooks`: Adrenaline's display loop seen through hooks on
//!   AdrenalineUser's imports (our frame, our picture instead of Sony's,
//!   the loop's timings, the GAME fps label), the PSP's buttons, threads.
//! - `io_hooks`: AdrenalineUser's file calls on PSP memory through a buffer
//!   of ours.
//! - `psp_plugin`: the PSP side (nativeres.prx, renderer/psp) carried in
//!   this plugin, installed into Adrenaline's seplugins at the start.
//! - `capture`: the capture thread (event ring -> core `Capture` -> log
//!   line, recorded frames, VRAM copy requests).
//! - `frame_loop`: the display thread's part (`render_frame`): takes
//!   the newest frame, its textures, and draws it into its render target
//!   (`targets`, which also shows it: `present_frame`).
//! - `unswizzle`: the GPU unswizzle of swizzled textures.
//! - `frame_pool`: GPU memory for what a frame's draws build (the ring is in
//!   core, tested there).
//! - `gxm`: the GPU library's functions we call from Rust.
//! - `gpu_memory`: memory blocks mapped for the GPU.
//! - `gpu_textures`: the texture cache's GPU textures and the texture arena.
//! - `arenas`: GPU memory the capture records into (raw arena, index ring,
//!   vertex arena, PSP RAM mapped for the GPU).
//! - `draw`, `programs`, `render_stats`: drawing a frame's draws with GXM
//!   (our programs, the cached GXM state), the display's statistics.
//! - `basic_programs`: our shader patcher, colour and texture-tint programs,
//!   the 960x544 projection.
//! - `heap`: our malloc/free (texture objects, the shader patcher).
//! - `palettes`: the palette tables' GPU memory (index textures' colours).
//! - `frames`: hands recorded frames to the display thread
//!   (`acquire_frame`, `wait_for_frame`).
//! - `log`: the capture log (ring + writer thread; sent wherever an
//!   extension sends it).
//! - `dump`: recording a whole frame for the PC replay (`renderer/replay`),
//!   when an extension asks for it.
//! - `ext`: the extension points: where a development build of the plugin
//!   adds its own tools (none in the plugin as built from this crate).
//! - `platform`: where PSP memory is mapped, fast copies out of it.
//! - `textures`: the display thread's texture cache (decoding, re-checks,
//!   the worker thread for big textures).
//! - `sys`: the Vita system functions we call.
//! - `adrenaline`: the few things that need Adrenaline's own state.
//! - `blocks`: our memory blocks and the table of them ("memory").
//! - `render_log`: the renderer's log ring.
//! - `shaders`: our GPU programs (compiled GXP), embedded.
//! - `threads`: threads' CPU time and cores.
//! - `taihen`: hooks on other modules' imports.
//! - below: a stateless function for the display thread (palettes).
//!
//! Threads: the capture thread owns all capture state, the display thread
//! the texture cache (`SingleThreaded`); the functions below keep no state.

#![no_std]

pub mod adrenaline;
pub mod arenas;
pub mod basic_programs;
pub mod blocks;
pub mod capture;
pub mod copy_thread;
pub mod display_hooks;
pub mod draw;
pub mod dump;
pub mod ext;
pub mod frame_loop;
pub mod frame_pool;
pub mod frames;
pub mod gpu_memory;
pub mod gpu_textures;
pub mod gxm;
pub mod heap;
pub mod io_hooks;
pub mod log;
pub mod mem;
pub mod menu;
pub mod options;
pub mod palettes;
pub mod platform;
pub mod plugin;
pub mod programs;
pub mod psp_plugin;
pub mod recorder_thread;
pub mod render_log;
pub mod render_stats;
pub mod shaders;
pub mod stage_thread;
pub mod sys;
pub mod taihen;
pub mod targets;
pub mod textures;
pub mod threads;
pub mod unswizzle;
pub mod user_log;

use core::cell::UnsafeCell;
use core::fmt::Write;
use core::panic::PanicInfo;

use nativeres_core::frame::TextureDescriptor;
use nativeres_core::report::TextBuffer;
use nativeres_core::texture;

/// State only one thread uses (the capture thread's, or the display
/// thread's side of the frame handover). Transparent: the value's own
/// layout, nothing added.
#[repr(transparent)]
pub struct SingleThreaded<T>(UnsafeCell<T>);

// Safe because each value is used by one thread only (see `get`).
unsafe impl<T> Sync for SingleThreaded<T> {}

impl<T> SingleThreaded<T> {
    pub const fn new(value: T) -> Self {
        SingleThreaded(UnsafeCell::new(value))
    }

    /// # Safety
    /// Only from the one thread that owns the value, and no two references
    /// at the same time.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn get(&self) -> &mut T {
        &mut *self.0.get()
    }
}

/// The 256-entry RGBA palette of a CLUT4/CLUT8 texture into its own table,
/// when no palette table is free (frame_loop.rs). `all_palettes`: the
/// frame's palette words.
pub fn texture_palette(descriptor: &TextureDescriptor, all_palettes: &[u32], out: &mut [u32; 256]) {
    let start = (descriptor.palette_offset as usize).min(all_palettes.len());
    let end = (start + descriptor.palette_word_count as usize).min(all_palettes.len());
    texture::palette_rgba(descriptor, &all_palettes[start..end], out);
}

/// A bug in our code: log it and stop the thread it happened on (the
/// capture thread, normally). There is no unwinding on the Vita.
#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    let mut bytes = [0u8; 256];
    let mut text = TextBuffer::new(&mut bytes);
    let _ = write!(text, "PANIC: {}
", info);
    let len = text.len();
    log::append(&bytes[..len]);
    log::drain();
    loop {
        // Safe: ends this thread.
        unsafe { sys::sceKernelExitDeleteThread(0) };
    }
}
