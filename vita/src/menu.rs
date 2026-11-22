//! "NativeRes Renderer" in Adrenaline's own menu (Settings, right below
//! Graphics Filtering), added from outside as a plugin on stock Adrenaline
//! can: no C of ours in Adrenaline (user, 2026-11-14).
//!
//! - Adrenaline's Settings entries (menu.c `settings_entries`, in its
//!   data) get our switch right below Graphics Filtering
//!   (nativeres_core::menu_layout). Their count is a constant of the tab
//!   table (`tab_entries`, which ended up read-only with the code). With a
//!   blank line below the filter (Adrenaline built from master) the switch
//!   goes in place: the entries after it move one down and the blank line
//!   before "Memory Stick Location" goes. Without one (the v8.0.2 release,
//!   issue #1) the entries are copied into `GROWN`, one longer, and taiHEN
//!   points the Settings tab there. Adrenaline's own code only uses entry 0
//!   by index (and by name, so `sync` mirrors it into the copy), which
//!   stays. The switch is a byte of Adrenaline's saved settings
//!   (`nativeres_renderer_off`, padding; 0 = on), so Adrenaline saves it
//!   with the rest.
//! - Graphics Filtering stays the user's choice: its entry edits
//!   `USER_FILTER` instead. With the renderer on in a PSP game, the filter
//!   Adrenaline's display loop reads is made one that draws with the GPU
//!   (Original and the direct sharp scale draw nothing we could draw into:
//!   Bilinear instead), only in memory; the settings file always gets the
//!   user's choice (`config_write`, from io_hooks.rs' sceIoWrite hook).

use core::ffi::c_void;
use core::ptr::{read_volatile, write_volatile};
use core::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering::Relaxed};

use nativeres_core::menu_layout::{self, Layout, Line};
use nativeres_core::span;

use crate::adrenaline::{self, CONFIG_BYTES, FILTER_BILINEAR, GRAPHICS_FILTERING_OFFSET, NO_SMOOTH_GRAPHICS_OFFSET, RENDERER_OFF_OFFSET};
use crate::{render_log, sys};

/// menu.h `MenuEntry` (28 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
struct MenuEntry {
    name: *const u8,
    kind: i32,
    color: u32,
    callback: usize,
    value: *mut u8,
    options: *const Text,
    n_options: i32,
}

const _: () = assert!(core::mem::size_of::<MenuEntry>() == 28);

/// `MENU_ENTRY_TYPE_OPTION`: `value` indexes `options`.
const MENU_ENTRY_TYPE_OPTION: i32 = 1;

/// A NUL-terminated string in our module (for the menu's tables).
#[repr(transparent)]
struct Text(*const u8);
// Safe: points at a constant string.
unsafe impl Sync for Text {}

static RENDERER_NAME: Text = Text(b"NativeRes Renderer\0".as_ptr());
/// 0 = On (`nativeres_renderer_off` 0), 1 = Off.
static RENDERER_OPTIONS: [Text; 2] = [Text(b"On\0".as_ptr()), Text(b"Off\0".as_ptr())];

/// More Settings entries than this: not Adrenaline's table.
const MAX_ENTRIES: usize = 48;
/// `MENU_ENTRY_TYPE_TEXT`: a line of text; with an empty name a blank line
/// the cursor skips.
const MENU_ENTRY_TYPE_TEXT: i32 = 0;

/// Graphics Filtering as the user chose it (the menu's entry edits this
/// byte once ours).
static USER_FILTER: AtomicU8 = AtomicU8::new(0);
/// The filter we last put into Adrenaline's settings (NONE: nothing yet):
/// a different value there means someone else set it (Reset Adrenaline
/// Settings, `set_user_filter`), which becomes the user's choice.
static LAST_SET: AtomicU8 = AtomicU8::new(NONE);
const NONE: u8 = 0xFF;

/// AdrenalineUser's code and data: (start, end) of each.
fn module_ranges() -> Option<[(usize, usize); 2]> {
    let (text, text_bytes) = adrenaline::module_segment(b"AdrenalineUser\0", 0)?;
    let (data, data_bytes) = adrenaline::module_segment(b"AdrenalineUser\0", 1)?;
    Some([(text, text + text_bytes), (data, data + data_bytes)])
}

/// True if `address` points at the NUL-terminated `text` inside the module.
fn is_text(ranges: &[(usize, usize); 2], address: usize, text: &[u8]) -> bool {
    // Words are any numbers: a range check that can't wrap (a word
    // 0xFFFFFFFF crashed us, 2026-11-14).
    if !ranges.iter().any(|&(start, end)| span::inside(address as u32, text.len() as u32, start as u32, end as u32)) {
        return false;
    }
    // Safe: inside one of the module's mapped segments.
    text.iter().enumerate().all(|(index, &byte)| unsafe { read_volatile((address + index) as *const u8) } == byte)
}

/// The name of `entry` (a pointer, maybe to nothing).
fn name_of(entry: *const MenuEntry) -> usize {
    // Safe: callers pass entries inside the data segment.
    unsafe { read_volatile(core::ptr::addr_of!((*entry).name)) as usize }
}

/// Adrenaline's Settings tab: its entries (in its data, writable), how many,
/// and where the tab's `entries` word is (`n_entries` follows it): segment
/// index and offset in it, for `taiInjectData`.
struct SettingsTab {
    entries: *mut MenuEntry,
    count: usize,
    segment: i32,
    offset: u32,
}

/// Adrenaline's Settings tab: from the `TabEntry` named "Settings" right
/// after "States", in either segment (the compiler put the tab table, never
/// written, with the constants in the code segment). None unless its
/// entries hold "Graphics Filtering".
fn find_settings_entries(ranges: &[(usize, usize); 2]) -> Option<SettingsTab> {
    let (data, data_end) = ranges[1];
    for (segment, &(start, end)) in ranges.iter().enumerate() {
        let words = (end - start) / 4;
        // Safe: words inside the segment.
        let word = |index: usize| unsafe { read_volatile((start as *const u32).add(index)) } as usize;
        for index in 4..words.saturating_sub(4) {
            if !is_text(ranges, word(index), b"Settings\0") || !is_text(ranges, word(index - 4), b"States\0") {
                continue;
            }
            let (entries, count) = (word(index + 1), word(index + 2));
            if count == 0
                || count > MAX_ENTRIES
                || !span::inside(entries as u32, (count * core::mem::size_of::<MenuEntry>()) as u32, data as u32, data_end as u32)
            {
                continue;
            }
            let entries = entries as *mut MenuEntry;
            // Safe: inside the data segment (checked above).
            if (0..count).any(|entry| is_text(ranges, name_of(unsafe { entries.add(entry) }), b"Graphics Filtering\0")) {
                return Some(SettingsTab { entries, count, segment: segment as i32, offset: ((index + 1) * 4) as u32 });
            }
        }
    }
    None
}

/// Our copy of the Settings entries, one longer, when Adrenaline's have no
/// blank line to give up (`Layout::Grown`): the Settings tab points here.
static mut GROWN: [MenuEntry; MAX_ENTRIES + 1] = [MenuEntry {
    name: core::ptr::null(),
    kind: MENU_ENTRY_TYPE_TEXT,
    color: 0,
    callback: 0,
    value: core::ptr::null_mut(),
    options: core::ptr::null(),
    n_options: 0,
}; MAX_ENTRIES + 1];
/// Adrenaline's own Settings entries while the tab points at `GROWN` (0:
/// in place), and how many: Adrenaline still writes them by name (the CFW
/// entry's `n_options` at every menu open), `sync` mirrors that.
static ORIGINAL: AtomicUsize = AtomicUsize::new(0);
static ORIGINAL_COUNT: AtomicUsize = AtomicUsize::new(0);
/// Where the switch is in `GROWN` (entries after it are one down).
static GROWN_SWITCH: AtomicUsize = AtomicUsize::new(0);

/// tai_offset_args_t (taihen.h).
#[repr(C)]
struct TaiOffsetArgs {
    size: u32,
    modid: sys::SceUID,
    segidx: i32,
    offset: u32,
    thumb: i32,
    source: *const c_void,
    source_size: u32,
}

extern "C" {
    fn taiInjectDataForUser(args: *const TaiOffsetArgs) -> sys::SceUID;
}

/// taihen.h's `taiInjectData` (a header function over this export).
unsafe fn tai_inject_data(modid: sys::SceUID, segment: i32, offset: u32, data: *const c_void, size: u32) -> sys::SceUID {
    let args = TaiOffsetArgs {
        size: core::mem::size_of::<TaiOffsetArgs>() as u32,
        modid,
        segidx: segment,
        offset,
        thumb: 0,
        source: data,
        source_size: size,
    };
    taiInjectDataForUser(&args)
}

/// Puts our switch into Adrenaline's Settings, right below Graphics
/// Filtering (`menu_layout::plan`): in place over a blank line, or in
/// `GROWN` with the Settings tab pointed there. None if the entries weren't
/// found (tried again later); Some(false) if they can't take it (logged).
fn install() -> Option<bool> {
    let ranges = module_ranges()?;
    let config = adrenaline::config_address()?;
    let tab = find_settings_entries(&ranges)?;
    let (entries, count) = (tab.entries, tab.count);
    let mut lines = [Line::Other; MAX_ENTRIES];
    // Safe: Adrenaline's entries (found above), `count` long. The menu is
    // closed at Adrenaline's start, so nobody reads them meanwhile.
    unsafe {
        for (index, line) in lines[..count].iter_mut().enumerate() {
            let entry = read_volatile(entries.add(index));
            if is_text(&ranges, entry.name as usize, b"Graphics Filtering\0") {
                *line = Line::Filter;
            } else if entry.kind == MENU_ENTRY_TYPE_TEXT && is_text(&ranges, entry.name as usize, b"\0") {
                *line = Line::Blank;
            }
        }
        let switch = MenuEntry {
            name: RENDERER_NAME.0,
            kind: MENU_ENTRY_TYPE_OPTION,
            color: 0,
            callback: 0,
            value: config.add(RENDERER_OFF_OFFSET),
            options: RENDERER_OPTIONS.as_ptr(),
            n_options: RENDERER_OPTIONS.len() as i32,
        };
        USER_FILTER.store(read_volatile(config.add(GRAPHICS_FILTERING_OFFSET)), Relaxed);
        match menu_layout::plan(&lines[..count]) {
            None => {
                render_log::line(format_args!("menu: no Graphics Filtering for \"NativeRes Renderer\" ({} entries)\n", count));
                return Some(false);
            }
            Some(Layout::InPlace { filter, blank }) => {
                for index in (filter + 1..blank).rev() {
                    write_volatile(entries.add(index + 1), read_volatile(entries.add(index)));
                }
                write_volatile(core::ptr::addr_of_mut!((*entries.add(filter)).value), USER_FILTER.as_ptr());
                write_volatile(entries.add(filter + 1), switch);
                render_log::line(format_args!(
                    "menu: \"NativeRes Renderer\" at entry {} of {} (entries {:08x}, blank line {} gone), filter {}\n",
                    filter + 1,
                    count,
                    entries as usize,
                    blank,
                    USER_FILTER.load(Relaxed)
                ));
            }
            Some(Layout::Grown { filter }) => {
                let grown = core::ptr::addr_of_mut!(GROWN) as *mut MenuEntry;
                for index in 0..count {
                    let to = if index > filter { index + 1 } else { index };
                    write_volatile(grown.add(to), read_volatile(entries.add(index)));
                }
                write_volatile(core::ptr::addr_of_mut!((*grown.add(filter)).value), USER_FILTER.as_ptr());
                write_volatile(grown.add(filter + 1), switch);
                ORIGINAL.store(entries as usize, Relaxed);
                ORIGINAL_COUNT.store(count, Relaxed);
                GROWN_SWITCH.store(filter + 1, Relaxed);
                // The tab table may be in the read-only code segment:
                // taiHEN patches it (its `entries` and `n_entries` words).
                let words = [grown as u32, (count + 1) as u32];
                let injected = match adrenaline::module_id(b"AdrenalineUser\0") {
                    Some(module) => tai_inject_data(module, tab.segment, tab.offset, words.as_ptr().cast(), 8),
                    None => -1,
                };
                if injected < 0 {
                    ORIGINAL.store(0, Relaxed);
                    render_log::line(format_args!("menu: Settings tab not patched ({:08x})\n", injected));
                    return Some(false);
                }
                render_log::line(format_args!(
                    "menu: \"NativeRes Renderer\" at entry {} of {} (copy {:08x} of {:08x}, tab in segment {} +{:x}), filter {}\n",
                    filter + 1,
                    count + 1,
                    grown as usize,
                    entries as usize,
                    tab.segment,
                    tab.offset,
                    USER_FILTER.load(Relaxed)
                ));
            }
        }
    }
    Some(true)
}

/// Adrenaline's writes to its own Settings entries (the CFW entry's
/// `n_options`, at every menu open) carried into `GROWN`.
fn mirror_original() {
    let original = ORIGINAL.load(Relaxed) as *const MenuEntry;
    if original.is_null() {
        return;
    }
    let switch = GROWN_SWITCH.load(Relaxed);
    let grown = core::ptr::addr_of_mut!(GROWN) as *mut MenuEntry;
    for index in 0..ORIGINAL_COUNT.load(Relaxed) {
        let to = if index >= switch { index + 1 } else { index };
        // Safe: Adrenaline's entries and our copy, both in range (install).
        unsafe {
            let options = read_volatile(core::ptr::addr_of!((*original.add(index)).n_options));
            write_volatile(core::ptr::addr_of_mut!((*grown.add(to)).n_options), options);
        }
    }
}

/// The filter Adrenaline's display loop should use for the user's
/// `filter`: one that draws with the GPU while our renderer is on for a
/// PSP game, else the user's.
fn drawn_filter(config: *const u8, filter: u8) -> u8 {
    // Safe: Adrenaline's settings and its shared words (menu.c reads both
    // the same way, through CONVERT_ADDRESS).
    let (renderer_on, no_smooth, pops, native) = unsafe {
        (
            read_volatile(config.add(RENDERER_OFF_OFFSET)) == 0,
            read_volatile(config.add(NO_SMOOTH_GRAPHICS_OFFSET)) != 0,
            read_volatile(sys::psp_to_vita(adrenaline::POPS_MODE_ADDRESS) as *const i32) != 0,
            read_volatile(sys::psp_to_vita(adrenaline::DRAW_NATIVE_ADDRESS) as *const u32) != 0,
        )
    };
    let original = filter == 0;
    // "Sharp bilinear without scanlines" without smoothing: Adrenaline's
    // direct sharp scale (the display loop draws nothing).
    let direct_sharp = filter == 5 && no_smooth;
    if renderer_on && !pops && !native && (original || direct_sharp) {
        FILTER_BILINEAR
    } else {
        filter
    }
}

/// Brings Adrenaline's Graphics Filtering in line with the user's choice
/// and our renderer (every 50 ms, and before Adrenaline saves).
pub fn sync() {
    // Without our entry the menu edits the filter Adrenaline draws with:
    // leave it alone (issue #1: Original was turned into Bilinear at once).
    if !INSTALLED.load(Relaxed) {
        return;
    }
    mirror_original();
    let Some(config) = adrenaline::config_address() else { return };
    // Safe: Adrenaline's settings, bytes; races with the menu thread only
    // delay a change to the next sync.
    unsafe {
        let current = read_volatile(config.add(GRAPHICS_FILTERING_OFFSET));
        if current != LAST_SET.load(Relaxed) {
            USER_FILTER.store(current, Relaxed);
        }
        let wanted = drawn_filter(config, USER_FILTER.load(Relaxed));
        if wanted != current {
            write_volatile(config.add(GRAPHICS_FILTERING_OFFSET), wanted);
        }
        LAST_SET.store(wanted, Relaxed);
    }
}

/// Graphics Filtering as the user chose it.
pub fn user_filter() -> u8 {
    USER_FILTER.load(Relaxed)
}

/// Sets the user's Graphics Filtering (not saved).
pub fn set_user_filter(filter: u8) {
    if let Some(config) = adrenaline::config_address() {
        // Safe: a byte of Adrenaline's settings; `sync` takes it as the
        // user's choice.
        unsafe { write_volatile(config.add(GRAPHICS_FILTERING_OFFSET), filter) };
        LAST_SET.store(NONE, Relaxed);
        sync();
    }
}

/// Set once the menu is ours (`try_install`): from then on `USER_FILTER`
/// holds the user's choice.
static INSTALLED: AtomicBool = AtomicBool::new(false);

/// AdrenalineUser's sceIoWrite (io_hooks.rs): when it writes its settings
/// (`config` itself, 36 bytes), the copy to write instead, with the user's
/// Graphics Filtering rather than the one we draw with.
pub fn config_write(data: *const c_void, size: u32) -> Option<[u8; CONFIG_BYTES]> {
    if size as usize != CONFIG_BYTES || !INSTALLED.load(Relaxed) {
        return None;
    }
    let config = adrenaline::config_address()?;
    if data != config as *const c_void {
        return None;
    }
    sync();
    let mut copy = [0u8; CONFIG_BYTES];
    // Safe: Adrenaline's settings, CONFIG_BYTES long.
    unsafe { core::ptr::copy_nonoverlapping(config, copy.as_mut_ptr(), CONFIG_BYTES) };
    copy[GRAPHICS_FILTERING_OFFSET] = USER_FILTER.load(Relaxed);
    Some(copy)
}

/// Our settings thread: adds the switch to the menu and keeps the filter
/// in line (`sync`). Returns once the menu is ours.
pub fn try_install() -> bool {
    match install() {
        None => false,
        Some(false) => true,
        Some(true) => {
            INSTALLED.store(true, Relaxed);
            sync();
            true
        }
    }
}
