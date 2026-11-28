//! The palette tables' GPU memory (which table holds which palette:
//! `nativeres_core::palette_tables`, the reason for them there).
//!
//! Uncached GPU memory: 1 KiB of writes per new palette, nothing to flush.
//! Display thread only.

use core::ffi::c_void;
use core::sync::atomic::{AtomicU32, Ordering};

use nativeres_core::frame::{Frame, TextureDescriptor};
use nativeres_core::opaque_source::{opaque_table_entries, table_lowest_alphas};
use nativeres_core::palette_tables::{PaletteKey, PaletteLookup, PaletteTables, TABLES, TABLE_BYTES};

use crate::{gpu_memory, render_log, sys, SingleThreaded};

#[no_mangle]
#[used]
#[allow(non_upper_case_globals)]
pub static nativeres_plan_fixed_palette_tables: u32 = sys::pages(TABLES * TABLE_BYTES);

/// "palette tables full" is logged this often at most.
const MAX_LOGGED: u32 = 5;

struct State {
    tables: PaletteTables,
    /// The tables' memory, null until made.
    memory: *mut u32,
    /// It couldn't be made: don't try again.
    failed: bool,
    logged_full: u32,
    /// Per table: which of its entries are opaque (`opaque_table_entries`)
    /// and their lowest alphas (`table_lowest_alphas`), worked out when it
    /// was filled.
    opaque: [u8; TABLES],
    lowest: [[u8; 2]; TABLES],
}

static STATE: SingleThreaded<State> = SingleThreaded::new(State {
    tables: PaletteTables::new(),
    memory: core::ptr::null_mut(),
    failed: false,
    logged_full: 0,
    opaque: [0; TABLES],
    lowest: [[0; 2]; TABLES],
});

/// Fills `table` (GPU memory: written once, never read back) with the 256
/// colours of index texture `texture` of `frame`; returns which entries are
/// opaque (`opaque_table_entries`) and their lowest alphas
/// (`table_lowest_alphas`).
pub fn fill_table(frame: &Frame, texture: &TextureDescriptor, table: &mut [u32; 256]) -> (u8, [u8; 2]) {
    let words = &frame.palette_words[..(frame.palette_word_count as usize).min(frame.palette_words.len())];
    let mut colors = [0u32; 256];
    crate::texture_palette(texture, words, &mut colors);
    *table = colors;
    (opaque_table_entries(&colors), table_lowest_alphas(&colors))
}

/// Tables filled since the last `take_filled` (render statistics
/// palettes=).
static FILLED: AtomicU32 = AtomicU32::new(0);

/// The 256-colour table for index texture `texture` of `frame` in display
/// frame `now`, which of its entries are opaque (`opaque_table_entries`)
/// and their lowest alphas (`table_lowest_alphas`),
/// or null if there is no table memory or every table is in use by this
/// frame or one still on its way through the GPU; the caller then fills the
/// texture's own table (`fill_table`).
///
/// Display thread only.
pub fn palette_for(frame: &Frame, texture: &TextureDescriptor, now: u32) -> (*const c_void, u8, [u8; 2]) {
    // Safe: display thread only; the reference ends in this function.
    let state = unsafe { STATE.get() };
    if state.memory.is_null() {
        if state.failed {
            return (core::ptr::null(), 0, [0; 2]);
        }
        match gpu_memory::for_gpu(b"NativeResPalettes\0", (TABLES * TABLE_BYTES) as u32) {
            Some(block) => state.memory = block.base as *mut u32,
            None => {
                state.failed = true;
                return (core::ptr::null(), 0, [0; 2]);
            }
        }
    }
    let index = match state.tables.find(PaletteKey::of(texture), now) {
        PaletteLookup::Found(index) => index,
        PaletteLookup::Fill(index) => {
            // Safe: table `index` of the tables' memory, ours to fill now.
            let table = unsafe { &mut *(state.memory.add(index * TABLE_BYTES / 4) as *mut [u32; 256]) };
            (state.opaque[index], state.lowest[index]) = fill_table(frame, texture, table);
            FILLED.fetch_add(1, Ordering::Relaxed);
            index
        }
        PaletteLookup::Full => {
            if state.logged_full < MAX_LOGGED {
                state.logged_full += 1;
                render_log::line(format_args!("palette tables full ({} in use)\n", TABLES));
            }
            return (core::ptr::null(), 0, [0; 2]);
        }
    };
    // Safe: inside the tables' memory.
    (unsafe { state.memory.add(index * TABLE_BYTES / 4) as *const c_void }, state.opaque[index], state.lowest[index])
}

/// Tables filled since the last call (render statistics palettes=).
pub fn take_filled() -> u32 {
    FILLED.swap(0, Ordering::Relaxed)
}
