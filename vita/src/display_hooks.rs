//! Adrenaline's display loop (Adrenaline/user/menu.c `AdrenalineDraw`, stock)
//! as the plugin sees it: hooks on AdrenalineUser's imports, which the loop
//! calls through vita2d (statically linked into AdrenalineUser, so its GXM
//! calls are AdrenalineUser's). Our own module's GXM calls are not hooked.
//!
//! One pass of the stock loop, with Graphics Filtering other than Original
//! (menu.rs keeps it so while the renderer is on), and what we do there:
//!
//! - `sceGxmBeginScene` (vita2d_start_drawing, the display's scene): note
//!   the GXM context and the display thread, draw the game's new frame into
//!   its render target in a scene of our own (`frame_loop::render_frame`),
//!   then let the display's scene begin;
//! - `sceDmacMemcpy(psp_data, 0x74000000, ...)`: the copy of Sony's picture
//!   into the texture the loop draws (`psp_data`, learnt here): skipped while
//!   our picture is shown, or when it was copied right after the LCD sync;
//! - `sceGxmSetFragmentTexture` with that texture, then `sceGxmDraw`: the
//!   quad of Sony's picture; while ours is shown, we draw our target and the
//!   GAME fps label instead (`targets::present_frame`);
//! - `sceGxmEndScene`, `sceGxmDisplayQueueAddEntry` (vita2d_swap_buffers):
//!   the buffer queued (its callback data's first word), then an
//!   extension's after-swap work (`ext::after_swap`);
//! - `sceCompatLCDCSync`: sleep until shortly before the PSP's next LCD
//!   refresh instead of spinning, then the early copy of Sony's picture.
//!
//! The draw profile (`timings`) is taken at these points. Calls from
//! other threads (states.c's LCD sync for a savestate picture) pass through.
//!
//! On ScePspemu: `sceCompatWriteSharedCtrl` (the PSP's buttons: an
//! extension's injected ones are added) and `sceKernelCreateThread` (threads
//! noted, `ext::note_thread`). On AdrenalineUser: `sceKernelCreateThread`, which tells
//! when Adrenaline starts its threads (InitAdrenaline: the PSP side boots),
//! our moment to start (plugin.rs).

use core::ffi::c_void;
use core::ptr::null_mut;
use core::sync::atomic::{AtomicBool, AtomicI32, AtomicPtr, AtomicU32, AtomicU64, AtomicUsize, Ordering::Relaxed};

use nativeres_core::game_fps::{self, GameFps, LABEL_HEIGHT, LABEL_WIDTH};

use crate::basic_programs::TextureVertex;
use crate::capture::COMPLETED_FRAME_COUNT;
use crate::gxm::{self, Context};
use crate::sys::{self, SceUID};
use crate::taihen::Hook;
use crate::{adrenaline, ext, frame_loop, gpu_memory, render_log, targets, SingleThreaded};

// Library and function NIDs: vitasdk's db/360.
const SCE_GXM_NID: u32 = 0xF76B_66BD;
const BEGIN_SCENE_NID: u32 = 0x8734_FF4E;
const END_SCENE_NID: u32 = 0xFE30_0E2F;
const DRAW_NID: u32 = 0xBC05_9AFC;
const SET_FRAGMENT_TEXTURE_NID: u32 = 0x29C3_4DF5;
const DISPLAY_QUEUE_ADD_ENTRY_NID: u32 = 0xEC5C_26B5;
const SCE_DMACMGR_NID: u32 = 0xA9E5_B2F0;
const DMAC_MEMCPY_NID: u32 = 0x9B2F_F739;
const SCE_COMPAT_NID: u32 = 0x0F35_909D;
const LCDC_SYNC_NID: u32 = 0xBD17_7787;
const WRITE_SHARED_CTRL_NID: u32 = 0x2306_FFED;
const SCE_LIBKERNEL_NID: u32 = 0xCAE9_ACE6;
const CREATE_THREAD_NID: u32 = 0xC5C1_1EE7;

const ADRENALINE_USER: &[u8] = b"AdrenalineUser\0";
const PSPEMU: &[u8] = b"ScePspemu\0";

static BEGIN_SCENE: Hook = Hook::new();
static END_SCENE: Hook = Hook::new();
static DRAW: Hook = Hook::new();
static SET_FRAGMENT_TEXTURE: Hook = Hook::new();
static DISPLAY_QUEUE_ADD_ENTRY: Hook = Hook::new();
static DMAC_MEMCPY: Hook = Hook::new();
static LCDC_SYNC: Hook = Hook::new();
static WRITE_SHARED_CTRL: Hook = Hook::new();
static EMULATOR_CREATE_THREAD: Hook = Hook::new();
static USER_CREATE_THREAD: Hook = Hook::new();

/// Sony's picture of the PSP screen (SCE_PSPEMU_FRAMEBUFFER) and the size
/// the loop copies (SCE_PSPEMU_FRAMEBUFFER_SIZE: 512 x 272 RGBA).
const SONY_PICTURE: usize = 0x7400_0000;
const SONY_PICTURE_BYTES: u32 = 512 * 272 * 4;

/// Stock vita2d's pool for per-frame GPU data (vita2d_init: 1 MiB), made
/// by Adrenaline's display thread. Not ours, but the memory budget was
/// measured with the fork, where it was ours (0.5 MiB): the memory plan
/// counts it so the budget stays comparable.
#[no_mangle]
#[used]
pub static nativeres_plan_library_vita2d_pool: u32 = 1024 * 1024;

/// The display thread's GXM context (vita2d's), from its first
/// sceGxmBeginScene; null before.
static CONTEXT: AtomicPtr<Context> = AtomicPtr::new(null_mut());
/// The display thread (AdrenalineDraw), from the same call; 0 before.
static DISPLAY_THREAD: AtomicI32 = AtomicI32::new(0);
/// The texture data Sony's picture is copied into (menu.c `psp_data`),
/// from the loop's first copy; 0 before.
static PSP_DATA: AtomicUsize = AtomicUsize::new(0);
/// This pass shows our picture (`render_frame` said so at BeginScene).
static SHOWING_OURS: AtomicBool = AtomicBool::new(false);
/// The last pass showed Sony's picture (decides the early copy).
static SHOWED_SONY: AtomicBool = AtomicBool::new(true);
/// Sony's picture was copied right after the last LCD sync.
static COPIED_EARLY: AtomicBool = AtomicBool::new(false);
/// The texture set last on unit 0 is Sony's picture (`psp_data`).
static SONY_TEXTURE_SET: AtomicBool = AtomicBool::new(false);
/// The display buffer queued last, and the queued count.
static QUEUED_BUFFER: AtomicU32 = AtomicU32::new(0);
static SWAP_COUNT: AtomicU32 = AtomicU32::new(0);
/// AdrenalineUser created its display thread (InitAdrenaline ran).
static ADRENALINE_STARTED: AtomicBool = AtomicBool::new(false);

/// The display thread's GXM context (null before the loop's first scene).
pub fn context() -> *mut Context {
    CONTEXT.load(Relaxed)
}

/// The display buffer queued last (1024 pixels per row, ABGR; 0 before).
pub fn queued_buffer() -> *const u32 {
    QUEUED_BUFFER.load(Relaxed) as *const u32
}

/// Display buffers queued so far.
pub fn swap_count() -> u32 {
    SWAP_COUNT.load(Relaxed)
}

/// True once Adrenaline has created its display thread.
pub fn adrenaline_started() -> bool {
    ADRENALINE_STARTED.load(Relaxed)
}

fn now() -> u64 {
    sys::now_us()
}

fn on_display_thread() -> bool {
    // Safe: a plain system call.
    let thread = unsafe { sys::sceKernelGetThreadId() };
    thread == DISPLAY_THREAD.load(Relaxed)
}

/// A PS1 game, homebrew drawing at the Vita's resolution, or a savestate
/// being taken or loaded (the loop draws "Please wait..." then): none of it
/// is ours to draw.
fn not_ours_to_draw() -> bool {
    adrenaline::pops_mode() || adrenaline::draw_native() || adrenaline::savestate_screen()
}

// ------------------------------------------------------------ the loop's timings

/// The loop's switches and timings (were menu.c's): the GAME fps label, the
/// early copy of Sony's picture, when Sony's picture is copied, the draw
/// profile.
pub mod timings {
    use super::*;

    /// Buckets of 1 ms after the LCD sync.
    pub const SONY_TIMING_BUCKETS: usize = 20;
    /// Parts of the loop the draw profile times: our frame, the rest of the
    /// drawing, end scene + swap, the LCD sync.
    pub const DRAW_PROFILE_PARTS: usize = 4;

    pub(super) static GAME_FPS_LABEL: AtomicBool = AtomicBool::new(true);
    pub(super) static SONY_COPY_EARLY: AtomicBool = AtomicBool::new(true);
    pub(super) static LAST_LCD_SYNC_US: AtomicU64 = AtomicU64::new(0);
    pub(super) static SONY_COPY_AFTER_SYNC: [AtomicU32; SONY_TIMING_BUCKETS] = [const { AtomicU32::new(0) }; SONY_TIMING_BUCKETS];
    pub(super) static RENDER_END_AFTER_SYNC: [AtomicU32; SONY_TIMING_BUCKETS] = [const { AtomicU32::new(0) }; SONY_TIMING_BUCKETS];
    pub(super) static DRAW_PROFILE_US: [AtomicU64; DRAW_PROFILE_PARTS] = [const { AtomicU64::new(0) }; DRAW_PROFILE_PARTS];
    pub(super) static DRAW_PROFILE_LOOPS: AtomicU32 = AtomicU32::new(0);

    pub fn set_game_fps_label(on: bool) {
        GAME_FPS_LABEL.store(on, Relaxed);
    }

    pub fn set_sony_copy_early(on: bool) {
        SONY_COPY_EARLY.store(on, Relaxed);
    }

    /// Process time of the loop's last LCD sync.
    pub fn last_lcd_sync_us() -> u64 {
        LAST_LCD_SYNC_US.load(Relaxed)
    }

    /// Clears the Sony-picture copy and render-end counts.
    pub fn clear_sony_timings() {
        for bucket in 0..SONY_TIMING_BUCKETS {
            SONY_COPY_AFTER_SYNC[bucket].store(0, Relaxed);
            RENDER_END_AFTER_SYNC[bucket].store(0, Relaxed);
        }
    }

    /// (copies, render ends) counted in `bucket` since `clear_sony_timings`.
    pub fn sony_timing(bucket: usize) -> (u32, u32) {
        (SONY_COPY_AFTER_SYNC[bucket].load(Relaxed), RENDER_END_AFTER_SYNC[bucket].load(Relaxed))
    }

    /// Clears the part times.
    pub fn clear_draw_profile() {
        for part in &DRAW_PROFILE_US {
            part.store(0, Relaxed);
        }
        DRAW_PROFILE_LOOPS.store(0, Relaxed);
    }

    /// Loops since `clear_draw_profile`, and the parts' summed times (us).
    pub fn draw_profile() -> (u32, [u64; DRAW_PROFILE_PARTS]) {
        (DRAW_PROFILE_LOOPS.load(Relaxed), core::array::from_fn(|part| DRAW_PROFILE_US[part].load(Relaxed)))
    }

    /// Counts "now" into the 1 ms bucket after the last LCD sync.
    pub(super) fn note_after_sync(histogram: &[AtomicU32; SONY_TIMING_BUCKETS]) {
        let after = now().wrapping_sub(LAST_LCD_SYNC_US.load(Relaxed));
        let bucket = ((after / 1000) as usize).min(SONY_TIMING_BUCKETS - 1);
        histogram[bucket].fetch_add(1, Relaxed);
    }
}

use timings::*;

/// Process times of this pass's points (display thread only): BeginScene
/// entered, our frame drawn, the display's scene ended, the swap queued.
struct Profile {
    start: u64,
    rendered: u64,
    drawn: u64,
    swapped: u64,
}

static PROFILE: SingleThreaded<Profile> = SingleThreaded::new(Profile { start: 0, rendered: 0, drawn: 0, swapped: 0 });

fn profile() -> &'static mut Profile {
    // Safe: only the display thread's hooks use it (each checks the thread
    // or is called by the loop only).
    unsafe { PROFILE.get() }
}

// ----------------------------------------------------------------- the hooks

type BeginScene = unsafe extern "C" fn(
    *mut Context,
    u32,
    *const gxm::RenderTarget,
    *const gxm::ValidRegion,
    *mut gxm::SyncObject,
    *mut gxm::SyncObject,
    *const gxm::ColorSurface,
    *const gxm::DepthStencilSurface,
) -> i32;

/// The display's scene begins: our frame first (in a scene of its own).
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn begin_scene_hooked(
    context: *mut Context,
    flags: u32,
    target: *const gxm::RenderTarget,
    valid_region: *const gxm::ValidRegion,
    vertex_sync: *mut gxm::SyncObject,
    fragment_sync: *mut gxm::SyncObject,
    color_surface: *const gxm::ColorSurface,
    depth_stencil: *const gxm::DepthStencilSurface,
) -> i32 {
    let start = now();
    CONTEXT.store(context, Relaxed);
    let thread = sys::sceKernelGetThreadId();
    if DISPLAY_THREAD.swap(thread, Relaxed) != thread {
        ext::note_thread(thread);
    }
    let ours = if not_ours_to_draw() {
        false
    } else {
        let ours = frame_loop::render_frame();
        if !ours {
            note_after_sync(&RENDER_END_AFTER_SYNC);
        }
        ours
    };
    SHOWING_OURS.store(ours, Relaxed);
    SHOWED_SONY.store(!ours, Relaxed);
    SONY_TEXTURE_SET.store(false, Relaxed);
    let profile = profile();
    profile.start = start;
    profile.rendered = now();
    let next: BeginScene = core::mem::transmute(BEGIN_SCENE.next());
    next(context, flags, target, valid_region, vertex_sync, fragment_sync, color_surface, depth_stencil)
}

type DmacMemcpy = unsafe extern "C" fn(*mut c_void, *const c_void, u32) -> i32;

/// The loop's copy of Sony's picture.
unsafe extern "C" fn dmac_memcpy_hooked(target: *mut c_void, source: *const c_void, size: u32) -> i32 {
    if source as usize == SONY_PICTURE && size == SONY_PICTURE_BYTES && on_display_thread() {
        PSP_DATA.store(target as usize, Relaxed);
        if SHOWING_OURS.load(Relaxed) || COPIED_EARLY.swap(false, Relaxed) {
            return 0;
        }
        note_after_sync(&SONY_COPY_AFTER_SYNC);
    }
    let next: DmacMemcpy = core::mem::transmute(DMAC_MEMCPY.next());
    next(target, source, size)
}

type SetFragmentTexture = unsafe extern "C" fn(*mut Context, u32, *const gxm::Texture) -> i32;

unsafe extern "C" fn set_fragment_texture_hooked(context: *mut Context, unit: u32, texture: *const gxm::Texture) -> i32 {
    if unit == 0 {
        let data = PSP_DATA.load(Relaxed);
        SONY_TEXTURE_SET.store(data != 0 && !texture.is_null() && gxm::sceGxmTextureGetData(texture) as usize == data, Relaxed);
    }
    let next: SetFragmentTexture = core::mem::transmute(SET_FRAGMENT_TEXTURE.next());
    next(context, unit, texture)
}

type Draw = unsafe extern "C" fn(*mut Context, u32, u32, *const c_void, u32) -> i32;

/// The quad of Sony's picture becomes ours while ours is shown.
unsafe extern "C" fn draw_hooked(context: *mut Context, primitive: u32, index_format: u32, indices: *const c_void, count: u32) -> i32 {
    if SONY_TEXTURE_SET.swap(false, Relaxed) && SHOWING_OURS.load(Relaxed) {
        targets::present_frame();
        if GAME_FPS_LABEL.load(Relaxed) {
            draw_game_fps();
        }
        return 0;
    }
    let next: Draw = core::mem::transmute(DRAW.next());
    next(context, primitive, index_format, indices, count)
}

type EndScene = unsafe extern "C" fn(*mut Context, *const gxm::Notification, *const gxm::Notification) -> i32;

unsafe extern "C" fn end_scene_hooked(context: *mut Context, vertex: *const gxm::Notification, fragment: *const gxm::Notification) -> i32 {
    if context == CONTEXT.load(Relaxed) && on_display_thread() {
        profile().drawn = now();
    }
    let next: EndScene = core::mem::transmute(END_SCENE.next());
    next(context, vertex, fragment)
}

type DisplayQueueAddEntry = unsafe extern "C" fn(*mut gxm::SyncObject, *mut gxm::SyncObject, *const c_void) -> i32;

/// vita2d queues the display buffer just drawn (its callback data's first
/// word is the buffer). GXM belongs to ScePspemu, so the entry goes to
/// Sony's display callback (ScePspemu 0x810734EC, 2026-11-14): it reads only
/// that first word (sceDisplaySetFrameBuf with it, pitch 1024, 960x544,
/// sync 1), so the stack bytes after vita2d's 4-byte data are harmless.
unsafe extern "C" fn display_queue_add_entry_hooked(old: *mut gxm::SyncObject, new: *mut gxm::SyncObject, data: *const c_void) -> i32 {
    let display = on_display_thread();
    if display && !data.is_null() {
        QUEUED_BUFFER.store(*(data as *const u32), Relaxed);
        SWAP_COUNT.fetch_add(1, Relaxed);
    }
    let next: DisplayQueueAddEntry = core::mem::transmute(DISPLAY_QUEUE_ADD_ENTRY.next());
    let result = next(old, new, data);
    if display {
        ext::after_swap();
        profile().swapped = now();
    }
    result
}

/// The PSP's LCD refreshes at 59.94 Hz; sleep until this long before the
/// next one is due, then let sceCompatLCDCSync wait the rest.
const PSP_LCD_PERIOD_US: u64 = 16_683;
const LCD_SYNC_MARGIN_US: u64 = 2_500;

type LcdcSync = unsafe extern "C" fn() -> i32;

/// sceCompatLCDCSync waits for the PSP side's LCD refresh by polling, so
/// the display thread was busy for the whole wait: ~15 ms of every 16.7 ms
/// with Sony's picture, ~10 ms with ours (2026-11-04). Sleeping until
/// shortly before the refresh is due returns at the same moment with the
/// CPU time free for other threads. A late pass (refresh already due)
/// doesn't sleep. Then, while Sony's picture is shown (movies, loading
/// screens), it is copied right away, as the stock loop's copy right after
/// the sync (a copy after our frame caught the game writing the next movie
/// frame: MTX's intro split across the top).
unsafe extern "C" fn lcdc_sync_hooked() -> i32 {
    let next: LcdcSync = core::mem::transmute(LCDC_SYNC.next());
    if !on_display_thread() {
        return next();
    }
    COPIED_EARLY.store(false, Relaxed);
    let last = LAST_LCD_SYNC_US.load(Relaxed);
    let before = now();
    let due = last + PSP_LCD_PERIOD_US;
    if last != 0 && due > before + LCD_SYNC_MARGIN_US {
        sys::sceKernelDelayThread((due - before - LCD_SYNC_MARGIN_US) as u32);
    }
    let result = next();
    let synced = now();
    LAST_LCD_SYNC_US.store(synced, Relaxed);

    let profile = profile();
    if profile.start != 0 && profile.swapped >= profile.drawn && profile.drawn >= profile.rendered {
        DRAW_PROFILE_US[0].fetch_add(profile.rendered - profile.start, Relaxed);
        DRAW_PROFILE_US[1].fetch_add(profile.drawn - profile.rendered, Relaxed);
        DRAW_PROFILE_US[2].fetch_add(profile.swapped - profile.drawn, Relaxed);
        DRAW_PROFILE_US[3].fetch_add(synced.saturating_sub(profile.swapped), Relaxed);
        DRAW_PROFILE_LOOPS.fetch_add(1, Relaxed);
    }
    profile.start = 0;

    let data = PSP_DATA.load(Relaxed);
    if data != 0
        && SONY_COPY_EARLY.load(Relaxed)
        && crate::textures::copy_sony_picture_early(SHOWED_SONY.load(Relaxed))
        && !not_ours_to_draw()
    {
        note_after_sync(&SONY_COPY_AFTER_SYNC);
        sys::sceDmacMemcpy(data as *mut c_void, SONY_PICTURE as *const c_void, SONY_PICTURE_BYTES);
        COPIED_EARLY.store(true, Relaxed);
    }
    result
}

/// SceCtrlDataPsp (16 bytes): only the buttons are touched.
#[repr(C)]
struct CtrlDataPsp {
    time_stamp: u32,
    buttons: u32,
    rest: [u8; 8],
}

type WriteSharedCtrl = unsafe extern "C" fn(*mut CtrlDataPsp) -> i32;

/// The PSP's buttons on their way to the game: the ones pressed from the PC
/// are added. Adrenaline's own hook (main.c) runs after ours: it keeps them,
/// except while its menu is open (then the game sees only PS).
unsafe extern "C" fn write_shared_ctrl_hooked(pad: *mut CtrlDataPsp) -> i32 {
    if !pad.is_null() {
        (*pad).buttons |= ext::injected_buttons();
    }
    let next: WriteSharedCtrl = core::mem::transmute(WRITE_SHARED_CTRL.next());
    next(pad)
}

type CreateThread = unsafe extern "C" fn(*const u8, sys::ThreadEntry, i32, u32, u32, i32, *mut c_void) -> SceUID;

unsafe extern "C" fn emulator_create_thread_hooked(
    name: *const u8,
    entry: sys::ThreadEntry,
    priority: i32,
    stack: u32,
    attributes: u32,
    cores: i32,
    options: *mut c_void,
) -> SceUID {
    let next: CreateThread = core::mem::transmute(EMULATOR_CREATE_THREAD.next());
    let thread = next(name, entry, priority, stack, attributes, cores, options);
    ext::note_thread(thread);
    thread
}

/// The name of AdrenalineUser's display thread (InitAdrenaline creates it
/// last, after AdrenalinePowerTick and AdrenalineCompat).
const DISPLAY_THREAD_NAME: &[u8] = b"AdrenalineDraw";

unsafe extern "C" fn user_create_thread_hooked(
    name: *const u8,
    entry: sys::ThreadEntry,
    priority: i32,
    stack: u32,
    attributes: u32,
    cores: i32,
    options: *mut c_void,
) -> SceUID {
    let next: CreateThread = core::mem::transmute(USER_CREATE_THREAD.next());
    let thread = next(name, entry, priority, stack, attributes, cores, options);
    ext::note_thread(thread);
    if !name.is_null() && core::ffi::CStr::from_ptr(name as *const core::ffi::c_char).to_bytes() == DISPLAY_THREAD_NAME {
        ADRENALINE_STARTED.store(true, Relaxed);
    }
    thread
}

/// Sets all hooks (once; AdrenalineUser must be loaded) and logs their ids.
pub fn install() {
    // Safe: each hook function has the hooked function's signature and
    // passes the call on through its Hook.
    let ids = unsafe {
        [
            ("sceGxmBeginScene", BEGIN_SCENE.hook_import(ADRENALINE_USER, SCE_GXM_NID, BEGIN_SCENE_NID, begin_scene_hooked as *const c_void)),
            ("sceGxmEndScene", END_SCENE.hook_import(ADRENALINE_USER, SCE_GXM_NID, END_SCENE_NID, end_scene_hooked as *const c_void)),
            (
                "sceGxmSetFragmentTexture",
                SET_FRAGMENT_TEXTURE.hook_import(ADRENALINE_USER, SCE_GXM_NID, SET_FRAGMENT_TEXTURE_NID, set_fragment_texture_hooked as *const c_void),
            ),
            ("sceGxmDraw", DRAW.hook_import(ADRENALINE_USER, SCE_GXM_NID, DRAW_NID, draw_hooked as *const c_void)),
            (
                "sceGxmDisplayQueueAddEntry",
                DISPLAY_QUEUE_ADD_ENTRY.hook_import(
                    ADRENALINE_USER,
                    SCE_GXM_NID,
                    DISPLAY_QUEUE_ADD_ENTRY_NID,
                    display_queue_add_entry_hooked as *const c_void,
                ),
            ),
            ("sceDmacMemcpy", DMAC_MEMCPY.hook_import(ADRENALINE_USER, SCE_DMACMGR_NID, DMAC_MEMCPY_NID, dmac_memcpy_hooked as *const c_void)),
            ("sceCompatLCDCSync", LCDC_SYNC.hook_import(ADRENALINE_USER, SCE_COMPAT_NID, LCDC_SYNC_NID, lcdc_sync_hooked as *const c_void)),
            (
                "sceKernelCreateThread",
                USER_CREATE_THREAD.hook_import(ADRENALINE_USER, SCE_LIBKERNEL_NID, CREATE_THREAD_NID, user_create_thread_hooked as *const c_void),
            ),
            (
                "ScePspemu sceCompatWriteSharedCtrl",
                WRITE_SHARED_CTRL.hook_import(PSPEMU, SCE_COMPAT_NID, WRITE_SHARED_CTRL_NID, write_shared_ctrl_hooked as *const c_void),
            ),
            (
                "ScePspemu sceKernelCreateThread",
                EMULATOR_CREATE_THREAD.hook_import(PSPEMU, SCE_LIBKERNEL_NID, CREATE_THREAD_NID, emulator_create_thread_hooked as *const c_void),
            ),
        ]
    };
    for (name, id) in ids {
        render_log::line(format_args!("hook {}: {:08x}\n", name, id));
    }
}

// ------------------------------------------------------------ the GAME fps label

/// Two label pictures and their quads, used in turn: the one written is
/// never the one the GPU may still read from the last pass.
const LABEL_TEXTURE_BYTES: usize = LABEL_WIDTH * LABEL_HEIGHT * 4;
const LABEL_QUAD_BYTES: usize = 4 * core::mem::size_of::<TextureVertex>();
const LABEL_BLOCK_BYTES: usize = 2 * (LABEL_TEXTURE_BYTES + LABEL_QUAD_BYTES);
#[no_mangle]
#[used]
pub static nativeres_plan_fixed_game_fps_label: u32 = sys::pages(LABEL_BLOCK_BYTES);

/// Right of VitaCam's "60.0 FPS  RENDER 30" box, as menu.c drew it: at
/// (252, 4), 2x, white on black.
const LABEL_X: f32 = 252.0;
const LABEL_Y: f32 = 4.0;
const LABEL_SCALE: f32 = 2.0;

struct Label {
    tried: bool,
    base: *mut u8,
    textures: [gxm::Texture; 2],
    /// The picture/quad in use (0 or 1), None before the first value.
    current: Option<usize>,
    fps: GameFps,
}

static LABEL: SingleThreaded<Label> = SingleThreaded::new(Label {
    tried: false,
    base: null_mut(),
    textures: [gxm::Texture { control_words: [0; 4] }; 2],
    current: None,
    fps: GameFps::new(),
});

impl Label {
    fn texture_pixels(&self, index: usize) -> *mut u32 {
        // Safe: inside the block (see LABEL_BLOCK_BYTES).
        unsafe { self.base.add(index * LABEL_TEXTURE_BYTES) as *mut u32 }
    }

    fn quad(&self, index: usize) -> *mut TextureVertex {
        // Safe: as above, after both pictures.
        unsafe { self.base.add(2 * LABEL_TEXTURE_BYTES + index * LABEL_QUAD_BYTES) as *mut TextureVertex }
    }

    fn make(&mut self) -> bool {
        if !self.tried {
            self.tried = true;
            if let Some(block) = gpu_memory::for_gpu(b"NativeResGameFps\0", sys::pages(LABEL_BLOCK_BYTES)) {
                self.base = block.base;
                for index in 0..2 {
                    // Safe: a linear RGBA texture over our block's picture.
                    unsafe {
                        gxm::sceGxmTextureInitLinear(
                            &mut self.textures[index],
                            self.texture_pixels(index) as *const c_void,
                            gxm::TEXTURE_FORMAT_U8U8U8U8_ABGR,
                            LABEL_WIDTH as u32,
                            LABEL_HEIGHT as u32,
                            0,
                        )
                    };
                }
            }
        }
        !self.base.is_null()
    }

    /// Draws the value into the other picture and quad, then uses those.
    fn write(&mut self, fps: u32) {
        let index = self.current.map_or(0, |current| current ^ 1);
        let mut pixels = [0u32; LABEL_WIDTH * LABEL_HEIGHT];
        let used = game_fps::draw_label(fps, &mut pixels);
        // Safe: our picture and quad, which the GPU doesn't read now (the
        // other pair is the one in use).
        unsafe {
            core::ptr::copy_nonoverlapping(pixels.as_ptr(), self.texture_pixels(index), pixels.len());
            let right = used as f32 / LABEL_WIDTH as f32;
            let width = used as f32 * LABEL_SCALE;
            let height = LABEL_HEIGHT as f32 * LABEL_SCALE;
            let corners = [(0.0, 0.0, 0.0, 0.0), (width, 0.0, right, 0.0), (0.0, height, 0.0, 1.0), (width, height, right, 1.0)];
            let quad = self.quad(index);
            for (i, (x, y, u, v)) in corners.into_iter().enumerate() {
                quad.add(i).write(TextureVertex { x: LABEL_X + x, y: LABEL_Y + y, z: 0.5, w: 1.0, u, v });
            }
        }
        self.current = Some(index);
    }
}

/// Display thread, inside the display's scene, right after our picture:
/// "GAME <fps>" (the value measured once a second; drawn again only when it
/// changed).
fn draw_game_fps() {
    // Safe: display thread only.
    let label = unsafe { LABEL.get() };
    if !label.make() {
        return;
    }
    if label.fps.update(now(), COMPLETED_FRAME_COUNT.load(Relaxed)) {
        if let Some(fps) = label.fps.fps() {
            label.write(fps);
        }
    }
    let Some(current) = label.current else {
        return;
    };
    // Safe: inside the display's scene; our texture and quad.
    unsafe { targets::draw_quad(&label.textures[current], label.quad(current)) };
}
