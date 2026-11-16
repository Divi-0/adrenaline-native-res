//! The few things we need from Adrenaline itself: its saved settings, its
//! shared area in PSP memory, and functions inside Sony's emulator.
//!
//! This is the seam between our code and Adrenaline. Everything here is
//! found the way a plugin for an unchanged Adrenaline finds it: the emulator's functions by their offsets in
//! ScePspemu's code, Adrenaline's settings by their magic words in
//! AdrenalineUser's data, savestates through the requests Adrenaline's menu
//! sends. The display loop itself is seen through hooks (display_hooks.rs).

use core::ffi::c_void;
use core::ptr::{addr_of, addr_of_mut, read_volatile, write_volatile};
use core::sync::atomic::{AtomicUsize, Ordering};

use crate::sys::{self, SceUID};

/// Adrenaline's saved settings (`AdrenalineConfig`, Adrenaline/user/main.h),
/// 36 bytes, in AdrenalineUser's data. Only the fields we read are named.
#[repr(C)]
struct AdrenalineConfig {
    magic: [u32; 2],
    /// Settings > Graphics Filtering (`GraphicFilterMode`).
    graphics_filtering: u8,
    no_smooth_graphics: u8,
    flux_mode: u8,
    /// Settings > Memory Stick location (`MemoryStickLocation`).
    ms_location: u8,
    rest: [u8; 23],
    /// Settings > "NativeRes Renderer" (the fork's menu, below Graphics
    /// Filtering): 0 = on (default; older files have 0 in this former
    /// padding), 1 = off. Stock Adrenaline keeps the byte when it saves.
    nativeres_renderer_off: u8,
}

const _: () = assert!(core::mem::size_of::<AdrenalineConfig>() == CONFIG_BYTES);

/// The settings struct's size, and offsets of the bytes menu.rs uses.
pub const CONFIG_BYTES: usize = 36;
pub const GRAPHICS_FILTERING_OFFSET: usize = 8;
pub const NO_SMOOTH_GRAPHICS_OFFSET: usize = 9;
pub const RENDERER_OFF_OFFSET: usize = 35;

/// `GraphicFilterMode::FILTER_BILINEAR`: Adrenaline's own smooth picture.
pub const FILTER_BILINEAR: u8 = 1;

/// ADRENALINE_CFG_MAGIC_1/2: the settings struct's first two words.
const CONFIG_MAGIC: [u32; 2] = [0x3148_3943, 0x8451_860B];

/// The highest Graphics Filtering stock Adrenaline offers (FILTER_LCD3X ..
/// 8); the fork once saved 9 for "NativeRes".
pub const FILTER_LAST: u8 = 8;

/// `SceAdrenaline::psp_cmd` requests (adrenaline_vita.h,
/// ADRENALINE_PSP_CMD_*).
const PSP_CMD_SAVESTATE: i32 = 2;
const PSP_CMD_LOADSTATE: i32 = 3;
/// The PSP side's game stops / goes on, as when Adrenaline's menu opens
/// and closes.
pub const PSP_CMD_PAUSE_WORLD: i32 = 4;
pub const PSP_CMD_RESUME_WORLD: i32 = 5;

/// Adrenaline's area in PSP memory (`SceAdrenaline` at ADRENALINE_ADDRESS,
/// Adrenaline/adrenaline_vita.h), which both sides read and write.
const ADRENALINE_ADDRESS: u32 = 0xABCD_E000;
const ADRENALINE_SIZE: u32 = 0x2000;
/// `SceAdrenaline` offsets: savestate_mode (the PSP side's SAVESTATE_MODE_*,
/// not NONE from taking a save/load until its resume after it), num (the
/// slot), titleid (12 bytes), psp_cmd (the request).
const SAVESTATE_MODE_OFFSET: usize = 0;
const NUM_OFFSET: usize = 4;
const TITLE_ID_OFFSET: usize = 6 * 4 + 128;
const TITLE_ID_BYTES: usize = 12;
/// `SceAdrenaline::pops_mode` (a PS1 game runs) as a PSP address, read
/// through CONVERT_ADDRESS as menu.c does.
pub const POPS_MODE_ADDRESS: u32 = ADRENALINE_ADDRESS + 16;
/// DRAW_NATIVE (adrenaline_vita.h): homebrew drawing at the Vita's
/// resolution itself.
pub const DRAW_NATIVE_ADDRESS: u32 = 0xABCD_EF00;
const PSP_CMD_OFFSET: usize = 420;
/// KERMIT_VIRTUAL_INTR_IMPOSE_CH1: tells the PSP side a request is waiting.
const KERMIT_REQUEST_INTERRUPT: i32 = 9;

/// The emulator makes the PSP's data cache state match before we read.
pub const KERMIT_INPUT_MODE: i32 = 1;
/// The emulator lets us write (Adrenaline's own writes into PSP memory).
pub const KERMIT_OUTPUT_MODE: i32 = 2;

/// Offsets of ScePspemuConvertAddress / ScePspemuWritebackCache in
/// ScePspemu's code (Thumb), the same on every firmware (main.c
/// GetFunctions).
const CONVERT_ADDRESS_OFFSET: usize = 0x6364;
const WRITEBACK_CACHE_OFFSET: usize = 0x6490;

/// Sony's picture of the PSP screen (SCE_PSPEMU_FRAMEBUFFER): 480x272
/// RGBA, 512 pixels per row.
const SONY_PICTURE: usize = 0x7400_0000;
const SONY_PICTURE_LINE: usize = 512;
/// A savestate file's header (AdrenalineStateHeader, zeroed: the PSP side
/// fills it) and thumbnail (every 2nd pixel of Sony's picture).
const STATE_HEADER_BYTES: usize = 0x100;
const THUMBNAIL_WIDTH: usize = 240;
const THUMBNAIL_HEIGHT: usize = 136;

/// tai_module_info_t (taihen.h).
#[repr(C)]
struct TaiModuleInfo {
    size: u32,
    modid: SceUID,
    module_nid: u32,
    name: [u8; 27],
    exports_start: usize,
    exports_end: usize,
    imports_start: usize,
    imports_end: usize,
}

/// SceKernelModuleInfo (0x1B8 bytes): four segments of 0x18 bytes from
/// 0x154 (size, perms, vaddr, memsz, filesz, res).
#[repr(C, align(4))]
struct ModuleInfo([u8; MODULE_INFO_BYTES]);

const MODULE_INFO_BYTES: usize = 0x1B8;
const MODULE_INFO_SEGMENTS: usize = 0x154;
const SEGMENT_BYTES: usize = 0x18;

type ConvertAddress = unsafe extern "C" fn(address: u32, mode: i32, cache_size: u32) -> i32;
type WritebackCache = unsafe extern "C" fn(address: *mut c_void, size: i32) -> i32;

extern "C" {
    fn taiGetModuleInfo(module: *const u8, info: *mut TaiModuleInfo) -> i32;
    fn sceKernelGetModuleInfo(module: SceUID, info: *mut ModuleInfo) -> i32;
    fn sceCompatInterrupt(interrupt: i32) -> i32;
}

/// Where loaded module `name` (NUL-terminated) has segment `index` (0 =
/// code, 1 = data): (Vita address, bytes).
pub fn module_segment(name: &[u8], index: usize) -> Option<(usize, usize)> {
    let mut tai = TaiModuleInfo {
        size: core::mem::size_of::<TaiModuleInfo>() as u32,
        modid: -1,
        module_nid: 0,
        name: [0; 27],
        exports_start: 0,
        exports_end: 0,
        imports_start: 0,
        imports_end: 0,
    };
    let mut info = ModuleInfo([0; MODULE_INFO_BYTES]);
    info.0[..4].copy_from_slice(&(MODULE_INFO_BYTES as u32).to_le_bytes());
    // Safe: both structs have the sizes taiHEN and the kernel expect, `name`
    // is NUL-terminated.
    unsafe {
        if taiGetModuleInfo(name.as_ptr(), &mut tai) < 0 || sceKernelGetModuleInfo(tai.modid, &mut info) < 0 {
            return None;
        }
    }
    let word = |offset: usize| u32::from_le_bytes([info.0[offset], info.0[offset + 1], info.0[offset + 2], info.0[offset + 3]]) as usize;
    let segment = MODULE_INFO_SEGMENTS + index * SEGMENT_BYTES;
    let (address, bytes) = (word(segment + 8), word(segment + 12));
    (address != 0).then_some((address, bytes))
}

/// ScePspemuConvertAddress and ScePspemuWritebackCache (Thumb addresses),
/// 0 until found.
static CONVERT_ADDRESS: AtomicUsize = AtomicUsize::new(0);
static WRITEBACK_CACHE: AtomicUsize = AtomicUsize::new(0);

/// The emulator's two functions, found in ScePspemu's code once.
fn emulator() -> Option<(ConvertAddress, WritebackCache)> {
    let mut convert = CONVERT_ADDRESS.load(Ordering::Acquire);
    if convert == 0 {
        let (text, _) = module_segment(b"ScePspemu\0", 0)?;
        WRITEBACK_CACHE.store((text + WRITEBACK_CACHE_OFFSET) | 1, Ordering::Relaxed);
        convert = (text + CONVERT_ADDRESS_OFFSET) | 1;
        CONVERT_ADDRESS.store(convert, Ordering::Release);
    }
    let writeback = WRITEBACK_CACHE.load(Ordering::Relaxed);
    // Safe: the emulator's functions at their fixed offsets, with these
    // signatures (Adrenaline/user/main.h).
    unsafe {
        Some((
            core::mem::transmute::<usize, ConvertAddress>(convert),
            core::mem::transmute::<usize, WritebackCache>(writeback),
        ))
    }
}

/// Adrenaline's settings (`AdrenalineConfig`), 0 until found.
static CONFIG: AtomicUsize = AtomicUsize::new(0);

/// Adrenaline's settings, found in AdrenalineUser's data by their magic
/// words (Adrenaline loads them before it starts anything of ours).
fn config() -> Option<*mut AdrenalineConfig> {
    let found = CONFIG.load(Ordering::Acquire);
    if found != 0 {
        return Some(found as *mut AdrenalineConfig);
    }
    let (data, bytes) = module_segment(b"AdrenalineUser\0", 1)?;
    let words = bytes / 4;
    let size_words = core::mem::size_of::<AdrenalineConfig>() / 4;
    // Safe: AdrenalineUser's data segment, mapped and readable; the struct
    // is word-aligned.
    unsafe {
        let data = data as *const u32;
        for index in 0..words.saturating_sub(size_words) {
            if read_volatile(data.add(index)) == CONFIG_MAGIC[0] && read_volatile(data.add(index + 1)) == CONFIG_MAGIC[1] {
                let found = data.add(index) as *mut AdrenalineConfig;
                CONFIG.store(found as usize, Ordering::Release);
                crate::render_log::line(format_args!(
                    "adrenaline: settings at {:08x} (data {:08x}, {} bytes): filter={} ms_location={}\n",
                    found as usize,
                    data as usize,
                    bytes,
                    read_volatile(addr_of!((*found).graphics_filtering)),
                    read_volatile(addr_of!((*found).ms_location))
                ));
                return Some(found);
            }
        }
    }
    None
}

/// Adrenaline's settings as bytes (`CONFIG_BYTES`), or None before they
/// are found.
pub fn config_address() -> Option<*mut u8> {
    config().map(|config| config as *mut u8)
}

/// Graphics Filtering as Adrenaline's display loop uses it now (menu.rs
/// may draw with another than the user's), or None before the settings
/// are found.
pub fn graphics_filtering() -> Option<u8> {
    // Safe: a byte the menu may change; any value read is fine.
    config().map(|config| unsafe { read_volatile(addr_of!((*config).graphics_filtering)) })
}

/// Settings > "NativeRes Renderer", or None before Adrenaline's settings
/// are found.
pub fn renderer_switch() -> Option<bool> {
    // Safe: a byte the menu may change; any value read is fine.
    config().map(|config| unsafe { read_volatile(addr_of!((*config).nativeres_renderer_off)) } == 0)
}

/// Sets "NativeRes Renderer" without saving it (Adrenaline saves it with
/// its other settings the next time its menu changes one).
pub fn set_renderer_switch(on: bool) {
    if let Some(config) = config() {
        // Safe: a byte; the menu writes it the same way.
        unsafe { write_volatile(addr_of_mut!((*config).nativeres_renderer_off), (!on) as u8) }
    }
}

/// Settings > Memory Stick location: where ms0: is (utils.c
/// getPspemuMemoryStickLocation).
pub fn memory_stick() -> &'static [u8] {
    // Safe: a byte; any value read is fine.
    match config().map(|config| unsafe { read_volatile(addr_of!((*config).ms_location)) }) {
        Some(1) => b"ur0:pspemu",
        Some(2) => b"imc0:pspemu",
        Some(3) => b"xmc0:pspemu",
        Some(4) => b"uma0:pspemu",
        Some(5) => b"uma0:",
        _ => b"ux0:pspemu",
    }
}

/// Adrenaline's area for reading (`KERMIT_INPUT_MODE`) or writing, or null
/// before the emulator runs.
fn area(mode: i32) -> *mut u8 {
    match emulator() {
        // Safe: the emulator's function, with plain numbers.
        Some((convert, _)) => unsafe { convert(ADRENALINE_ADDRESS, mode, ADRENALINE_SIZE) as u32 as usize as *mut u8 },
        None => core::ptr::null_mut(),
    }
}

/// Writes `value` at `offset` of Adrenaline's area and makes it visible to
/// the PSP side.
fn write_area(offset: usize, value: i32) -> bool {
    let area = area(KERMIT_OUTPUT_MODE);
    if area.is_null() {
        return false;
    }
    // Safe: inside SceAdrenaline, mapped while the emulator runs.
    unsafe {
        write_volatile(area.add(offset) as *mut i32, value);
        writeback(area.cast(), ADRENALINE_SIZE);
    }
    true
}

/// Asks the PSP side to do `command` (`PSP_CMD_*`), as Adrenaline's
/// SendAdrenalineRequest does.
pub fn send_request(command: i32) {
    if write_area(PSP_CMD_OFFSET, command) {
        // Safe: a plain system call.
        unsafe { sceCompatInterrupt(KERMIT_REQUEST_INTERRUPT) };
    }
}

/// True from the PSP side taking a savestate or load until it runs again
/// after it.
pub fn state_operation_running() -> bool {
    let area = area(KERMIT_INPUT_MODE);
    // Safe: as `write_area`.
    !area.is_null() && unsafe { read_volatile(area.add(SAVESTATE_MODE_OFFSET) as *const i32) } != 0
}

/// "<ms>/PSP/SAVESTATE" (+ "/STATE<slot>.BIN" with `slot`), NUL-terminated.
fn state_path(path: &mut [u8; 64], slot: Option<i32>) -> *const u8 {
    let mut length = 0;
    let mut push = |bytes: &[u8]| {
        for &byte in bytes {
            if length < path.len() - 1 {
                path[length] = byte;
                length += 1;
            }
        }
    };
    push(memory_stick());
    push(b"/PSP/SAVESTATE");
    if let Some(slot) = slot {
        push(b"/STATE");
        push(&[b'0' + (slot / 10) as u8, b'0' + (slot % 10) as u8]);
        push(b".BIN");
    }
    path[length] = 0;
    path.as_ptr()
}

/// Creates savestate `slot`'s file with a zeroed header and a thumbnail of
/// Sony's picture (states.c saveState): Adrenaline's side of the save only
/// opens it for writing, without creating it.
fn create_state_file(slot: i32) -> i32 {
    let mut path = [0u8; 64];
    // Safe: NUL-terminated paths; reads Sony's picture (mapped while the
    // emulator runs) into our buffer only.
    unsafe {
        sys::sceIoMkdir(state_path(&mut path, None), 0o777);
        let fd = sys::sceIoOpen(state_path(&mut path, Some(slot)), sys::SCE_O_WRONLY | sys::SCE_O_CREAT | sys::SCE_O_TRUNC, 0o777);
        if fd < 0 {
            return fd;
        }
        let header = [0u8; STATE_HEADER_BYTES];
        sys::sceIoWrite(fd, header.as_ptr().cast(), STATE_HEADER_BYTES as u32);
        // Two thumbnail rows per write.
        let mut rows = [0u32; THUMBNAIL_WIDTH * 2];
        for pair in 0..THUMBNAIL_HEIGHT / 2 {
            for (row, pixels) in rows.chunks_exact_mut(THUMBNAIL_WIDTH).enumerate() {
                let line = (SONY_PICTURE as *const u32).add((pair * 2 + row) * 2 * SONY_PICTURE_LINE);
                for (x, pixel) in pixels.iter_mut().enumerate() {
                    *pixel = read_volatile(line.add(x * 2));
                }
            }
            sys::sceIoWrite(fd, rows.as_ptr().cast(), (rows.len() * 4) as u32);
        }
        sys::sceIoClose(fd)
    }
}

/// Starts saving (`load` false) or loading Adrenaline savestate `slot`, as
/// Adrenaline's menu does (without stopping USB); `state_operation_running`
/// tells when it is done.
pub fn start_state_operation(load: bool, slot: i32) {
    if !load {
        let created = create_state_file(slot);
        if created < 0 {
            crate::render_log::line(format_args!("savestate {}: file not created ({:08x})\n", slot, created));
        }
    }
    write_area(NUM_OFFSET, slot);
    send_request(if load { PSP_CMD_LOADSTATE } else { PSP_CMD_SAVESTATE });
}

/// What the stock display loop reads every pass (menu.c, through
/// CONVERT_ADDRESS: the mapped address as it is, no cache maintenance): a
/// PS1 game runs.
pub fn pops_mode() -> bool {
    // Safe: Adrenaline's shared area, mapped while the emulator runs (the
    // display loop runs only then).
    unsafe { read_volatile(sys::psp_to_vita(POPS_MODE_ADDRESS) as *const i32) != 0 }
}

/// A savestate is being taken or loaded (`savestate_mode`, read as the
/// loop does: it draws "Please wait..." then).
pub fn savestate_screen() -> bool {
    // Safe: as `pops_mode`.
    unsafe { read_volatile(sys::psp_to_vita(ADRENALINE_ADDRESS + SAVESTATE_MODE_OFFSET as u32) as *const i32) != 0 }
}

/// Homebrew draws at the Vita's resolution itself (DRAW_NATIVE).
pub fn draw_native() -> bool {
    // Safe: as `pops_mode`.
    unsafe { read_volatile(sys::psp_to_vita(DRAW_NATIVE_ADDRESS) as *const u32) != 0 }
}

/// Where the emulator maps PSP address `address` on the Vita, and (with
/// `bytes` > 0) makes that range current for reading or writing (`mode`:
/// `KERMIT_INPUT_MODE` / `KERMIT_OUTPUT_MODE`). 0 before the emulator runs.
pub fn convert_address(address: u32, mode: i32, bytes: u32) -> usize {
    match emulator() {
        // Safe: the emulator's own function, with plain numbers.
        Some((convert, _)) => unsafe { convert(address, mode, bytes) as u32 as usize },
        None => 0,
    }
}

/// The running game's title ID (e.g. "ULES00581"), NUL-terminated.
pub fn title_id() -> [u8; TITLE_ID_BYTES] {
    let mut title = [0u8; TITLE_ID_BYTES];
    let area = area(KERMIT_INPUT_MODE);
    if !area.is_null() {
        // Safe: Adrenaline's area, mapped as long as the emulator runs.
        unsafe { core::ptr::copy_nonoverlapping(area.add(TITLE_ID_OFFSET), title.as_mut_ptr(), TITLE_ID_BYTES) };
    }
    title[TITLE_ID_BYTES - 1] = 0;
    title
}

/// Makes our writes to PSP memory at `address` (a Vita address) visible to
/// the PSP side.
///
/// # Safety
/// `address` must point at `size` bytes of mapped PSP memory.
pub unsafe fn writeback(address: *mut c_void, size: u32) {
    if let Some((_, writeback_cache)) = emulator() {
        writeback_cache(address, size as i32);
    }
}
