//! "NativeRes Renderer" in Adrenaline's own menu (Settings, right below
//! Graphics Filtering), added from outside as a plugin on stock Adrenaline
//! can: no C of ours in Adrenaline (user, 2026-11-14).
//!
//! - Adrenaline's Settings entries (menu.c `settings_entries`, in its
//!   data) get our switch right below Graphics Filtering, in place: their
//!   count is a constant (the tab table, `tab_entries`, ended up read-only
//!   with the code), so the entries after it move one down and the blank
//!   line before "Memory Stick Location" goes. Adrenaline's own code only
//!   uses entries 0 and 2 (Graphics Filtering) by index, which stay. The
//!   switch is a byte of Adrenaline's saved settings
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
use core::sync::atomic::{AtomicBool, AtomicU8, Ordering::Relaxed};

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

/// Adrenaline's Settings entries (in its data, writable) and how many: from
/// the `TabEntry` named "Settings" right after "States", in either segment
/// (the compiler put the tab table, never written, with the constants in
/// the code segment). None unless they hold "Graphics Filtering".
fn find_settings_entries(ranges: &[(usize, usize); 2]) -> Option<(*mut MenuEntry, usize)> {
    let (data, data_end) = ranges[1];
    for &(start, end) in ranges {
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
                return Some((entries, count));
            }
        }
    }
    None
}

/// Puts our switch into Adrenaline's Settings entries, in place (the entry
/// count is a constant): right below Graphics Filtering, the entries after
/// it one down, the first blank line after it gone. False if they weren't
/// found (tried again later); true also when they can't take it (logged).
fn install() -> bool {
    let (Some(ranges), Some(config)) = (module_ranges(), adrenaline::config_address()) else { return false };
    let Some((entries, count)) = find_settings_entries(&ranges) else { return false };
    // Safe: Adrenaline's entries (found above), `count` long. The menu is
    // closed at Adrenaline's start, so nobody reads them meanwhile.
    unsafe {
        let filter = (0..count).find(|&index| is_text(&ranges, name_of(entries.add(index)), b"Graphics Filtering\0"));
        let blank = filter.and_then(|filter| {
            (filter + 1..count).find(|&index| {
                let entry = read_volatile(entries.add(index));
                entry.kind == MENU_ENTRY_TYPE_TEXT && is_text(&ranges, entry.name as usize, b"\0")
            })
        });
        let (Some(filter), Some(blank)) = (filter, blank) else {
            render_log::line(format_args!("menu: no room for \"NativeRes Renderer\" ({} entries)\n", count));
            return true;
        };
        for index in (filter + 1..blank).rev() {
            write_volatile(entries.add(index + 1), read_volatile(entries.add(index)));
        }
        USER_FILTER.store(read_volatile(config.add(GRAPHICS_FILTERING_OFFSET)), Relaxed);
        write_volatile(core::ptr::addr_of_mut!((*entries.add(filter)).value), USER_FILTER.as_ptr());
        write_volatile(
            entries.add(filter + 1),
            MenuEntry {
                name: RENDERER_NAME.0,
                kind: MENU_ENTRY_TYPE_OPTION,
                color: 0,
                callback: 0,
                value: config.add(RENDERER_OFF_OFFSET),
                options: RENDERER_OPTIONS.as_ptr(),
                n_options: RENDERER_OPTIONS.len() as i32,
            },
        );
        render_log::line(format_args!(
            "menu: \"NativeRes Renderer\" at entry {} of {} (entries {:08x}, blank line {} gone), filter {}\n",
            filter + 1,
            count,
            entries as usize,
            blank,
            USER_FILTER.load(Relaxed)
        ));
    }
    true
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
    if !install() {
        return false;
    }
    sync();
    INSTALLED.store(true, Relaxed);
    true
}
