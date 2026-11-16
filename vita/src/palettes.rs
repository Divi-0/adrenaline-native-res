//! The palette tables' GPU memory (which table holds which palette:
//! `nativepsp_core::palette_tables`, the reason for them there).
//!
//! Uncached GPU memory: 1 KiB of writes per new palette, nothing to flush.
//! Display thread only.

use core::ffi::c_void;
use core::sync::atomic::{AtomicU32, Ordering};

use nativepsp_core::frame::{Frame, TextureDescriptor};
use nativepsp_core::palette_tables::{PaletteKey, PaletteLookup, PaletteTables, TABLES, TABLE_BYTES};
use nativepsp_core::texture;

use crate::{gpu_memory, render_log, sys, SingleThreaded};

#[no_mangle]
#[used]
#[allow(non_upper_case_globals)]
pub static nativepsp_plan_fixed_palette_tables: u32 = sys::pages(TABLES * TABLE_BYTES);

/// "palette tables full" is logged this often at most.
const MAX_LOGGED: u32 = 5;

struct State {
    tables: PaletteTables,
    /// The tables' memory, null until made.
    memory: *mut u32,
    /// It couldn't be made: don't try again.
    failed: bool,
    logged_full: u32,
}

static STATE: SingleThreaded<State> =
    SingleThreaded::new(State { tables: PaletteTables::new(), memory: core::ptr::null_mut(), failed: false, logged_full: 0 });

/// Tables filled since the last `take_filled` (renderstats
/// palettes=).
static FILLED: AtomicU32 = AtomicU32::new(0);

/// The 256-colour table for index texture `texture` of `frame` in display
/// frame `now`, or null if there is no table memory or every table is in
/// use by this frame or one still on its way through the GPU; the caller
/// then fills the texture's own table.
///
/// Display thread only.
pub fn palette_for(frame: &Frame, texture: &TextureDescriptor, now: u32) -> *const c_void {
    // Safe: display thread only; the reference ends in this function.
    let state = unsafe { STATE.get() };
    if state.memory.is_null() {
        if state.failed {
            return core::ptr::null();
        }
        match gpu_memory::for_gpu(b"NativePspPalettes\0", (TABLES * TABLE_BYTES) as u32) {
            Some(block) => state.memory = block.base as *mut u32,
            None => {
                state.failed = true;
                return core::ptr::null();
            }
        }
    }
    let index = match state.tables.find(PaletteKey::of(texture), now) {
        PaletteLookup::Found(index) => index,
        PaletteLookup::Fill(index) => {
            let words = &frame.palette_words[..(frame.palette_word_count as usize).min(frame.palette_words.len())];
            let start = (texture.palette_offset as usize).min(words.len());
            let end = (start + texture.palette_word_count as usize).min(words.len());
            // Safe: table `index` of the tables' memory, ours to fill now.
            let table = unsafe { &mut *(state.memory.add(index * TABLE_BYTES / 4) as *mut [u32; 256]) };
            texture::palette_rgba(texture, &words[start..end], table);
            FILLED.fetch_add(1, Ordering::Relaxed);
            index
        }
        PaletteLookup::Full => {
            if state.logged_full < MAX_LOGGED {
                state.logged_full += 1;
                render_log::line(format_args!("palette tables full ({} in use)\n", TABLES));
            }
            return core::ptr::null();
        }
    };
    // Safe: inside the tables' memory.
    unsafe { state.memory.add(index * TABLE_BYTES / 4) as *const c_void }
}

/// Tables filled since the last call (renderstats palettes=).
pub fn take_filled() -> u32 {
    FILLED.swap(0, Ordering::Relaxed)
}
