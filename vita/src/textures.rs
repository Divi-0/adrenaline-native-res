//! The display thread's texture cache (`nativeres_core::texture_cache`) on
//! the Vita: its host (GPU textures from gpu_textures.rs,
//! PSP memory, the clock, the render log), the staging buffer, and the
//! worker thread that re-checks big textures.
//!
//! The frame loop (frame_loop.rs) calls, per frame it draws:
//! - `ready` (once there is a frame to draw),
//! - `for_frame` for all of the frame's textures,
//! - `frame_done` once it knows whether the frame waits.
//!
//! Threads: the cache belongs to the display thread. The worker only sees
//! the job it is handed (`WORKER`), and other threads only the counters
//! (`STATS`).

use core::cell::UnsafeCell;
use core::ffi::c_void;
use core::fmt;
use core::ptr::null_mut;
use core::sync::atomic::{AtomicBool, AtomicI32, AtomicPtr, AtomicU32, Ordering};

use nativeres_core::frame::{Frame, TextureDescriptor, MAX_TEXTURES};
use nativeres_core::texture::{self, DecodeError};
use nativeres_core::texture_cache::gpu::{GpuFormat, GpuTexture, TextureShape};
use nativeres_core::texture_cache::refresh::{run_refresh, RefreshJob, RefreshResult};
use nativeres_core::texture_cache::staging::STAGING_SIZE;
use nativeres_core::texture_cache::{FrameStart, FrameTimes, Host, Lookup, TextureCache, TextureStats, VramState, MAX_RENDER_TARGETS};

use crate::capture::{self, CAPTURED_FRAME_NUMBER};
use crate::platform::{psp_copy, psp_memory};
use crate::{gpu_textures, gxm, sys};
use crate::SingleThreaded;

/// The worker runs a little below the display thread (0xA0).
const WORKER_PRIORITY: i32 = 0xA8;
/// The C worker had 16 KiB; decoding needs little stack, this leaves room.
const WORKER_STACK_SIZE: u32 = 0x8000;

/// The display thread's part.
pub struct DisplayState {
    pub cache: TextureCache,
    /// The staging buffer (`STAGING_SIZE` bytes), null until allocated.
    pub staging: *mut u8,
    worker_tried: bool,
    pub lookups: [Lookup; MAX_TEXTURES],
    /// The last frame's start (for another lookup pass over that frame).
    pub last_start: Option<FrameStart>,
}

/// The display thread's texture cache and staging buffer.
///
/// # Safety
/// Display thread only, no other reference to it alive.
pub unsafe fn display_state() -> &'static mut DisplayState {
    DISPLAY.get()
}

static DISPLAY: SingleThreaded<DisplayState> = SingleThreaded::new(DisplayState {
    cache: TextureCache::new(),
    staging: null_mut(),
    worker_tried: false,
    lookups: [Lookup::Unavailable; MAX_TEXTURES],
    last_start: None,
});

/// Two staging buffers (the display thread's and the refresh worker's),
/// for the memory plan.
#[no_mangle]
#[used]
#[allow(non_upper_case_globals)]
pub static nativeres_plan_fixed_staging: u32 = 2 * sys::pages(STAGING_SIZE);

/// Allocates `STAGING_SIZE` bytes of cached main memory, or null.
fn allocate_staging(name: &[u8]) -> *mut u8 {
    // Safe: plain system calls; the block is ours alone.
    unsafe {
        let block = sys::alloc_block(
            name.as_ptr(),
            sys::SCE_KERNEL_MEMBLOCK_TYPE_USER_RW,
            STAGING_SIZE as u32,
            null_mut(),
        );
        if block < 0 {
            return null_mut();
        }
        let mut base: *mut c_void = null_mut();
        sys::sceKernelGetMemBlockBase(block, &mut base);
        base as *mut u8
    }
}

/// Display thread, before drawing a frame: makes the staging buffer and,
/// once, starts the worker (without one, big textures are re-checked on the
/// display thread). False if there is no memory for the staging buffer.
pub fn ready() -> bool {
    // Safe: only the display thread calls this.
    let display = unsafe { DISPLAY.get() };
    if display.staging.is_null() {
        display.staging = allocate_staging(b"NativeResStaging\0");
        if display.staging.is_null() {
            return false;
        }
    }
    if !display.worker_tried {
        display.worker_tried = true;
        start_worker();
    }
    true
}

/// What the frame loop tells the cache about the frame it draws.
pub struct FrameInfo {
    /// The frame loop's frame count (the cache counts ages in it).
    pub render_count: u32,
    /// Menu "NativeRes Texture Filter": always bilinear.
    pub smooth: bool,
    /// Palette and 8888 textures as U16x4 for the GE_WIDE programs (core
    /// `FrameStart::wide`).
    pub wide: bool,
    /// The texture address whose lookups are logged (0: none).
    pub trace_address: u32,
    /// When the display loop began (the texture work deadline counts from it).
    pub loop_start_us: u32,
    /// Framebuffers the frame loop has render targets for (0: none): textures
    /// there come back null, the frame loop uses its targets.
    pub render_targets: [u32; MAX_RENDER_TARGETS],
}

/// Display thread, once `ready` said true: looks up (decoding where needed)
/// every texture of `frame`. Writes the GPU texture for each to `textures`,
/// or null; `waiting` is 1 where it is null only for now (the texture's VRAM
/// copy hasn't arrived, or the texture work deadline passed).
pub fn for_frame(
    frame: &Frame,
    info: &FrameInfo,
    textures: &mut [*const gpu_textures::TextureObject; MAX_TEXTURES],
    waiting: &mut [u8; MAX_TEXTURES],
) {
    // Safe: only the display thread calls this; the reference ends here.
    let display = unsafe { DISPLAY.get() };
    if display.staging.is_null() {
        return;
    }
    // Safe: the staging buffer holds STAGING_SIZE bytes, used only here.
    let staging = unsafe { core::slice::from_raw_parts_mut(display.staging, STAGING_SIZE) };
    let start = FrameStart {
        frame_number: frame.frame_number,
        render_count: info.render_count,
        smooth: info.smooth,
        wide: info.wide,
        trace_address: info.trace_address,
        loop_start_us: info.loop_start_us,
        render_targets: info.render_targets,
    };
    display.cache.textures_for_frame(&mut VitaHost, staging, frame, start, &mut display.lookups);
    display.last_start = Some(start);
    for (index, lookup) in display.lookups.iter().enumerate().take(frame.textures().len()) {
        textures[index] = match lookup {
            Lookup::Ready(texture) => texture.handle as *const gpu_textures::TextureObject,
            _ => core::ptr::null(),
        };
        waiting[index] = (*lookup == Lookup::Waiting) as u8;
    }
    STATS.add(&display.cache.take_stats());
}

/// Display thread, after `for_frame`: how many rows from the top of the
/// frame's texture `index` (from the cache) have only texels of alpha 255
/// (`TextureCache::opaque_rows`; `texture::ALL_ROWS_OPAQUE`: all).
pub fn opaque_rows(index: usize) -> u16 {
    // Safe: only the display thread calls this.
    unsafe { DISPLAY.get() }.cache.opaque_rows(index)
}

/// Display thread, after `for_frame`: where the frame's texture `index`
/// (from the cache) has texels that show (`TextureCache::texel_mask`), for
/// the zero-alpha cull. None if it has no mask. Valid until the next
/// `for_frame`.
pub fn texel_mask(index: usize) -> Option<&'static nativeres_core::texture::TexelMask> {
    // Safe: only the display thread calls this, and only it changes the cache.
    unsafe { DISPLAY.get() }.cache.texel_mask(index)
}

/// Display thread, after `for_frame`: whether the frame waits for textures
/// (isn't drawn).
pub fn frame_done(waited: bool) {
    // Safe: only the display thread calls this.
    unsafe { DISPLAY.get() }.cache.end_frame(waited);
}

/// Display thread, after `for_frame`: how many of the frame's textures the
/// game rewrote before we read them (`TextureCache::moved_on`; the frame
/// isn't shown then).
pub fn moved_on() -> u32 {
    // Safe: only the display thread calls this.
    unsafe { DISPLAY.get() }.cache.moved_on()
}

/// Display thread: `bytes` of the texture arena for good
/// (`TextureCache::reserve_arena`), GPU-readable and writable; None if the
/// arena has no free run that long now, or none at all.
pub fn reserve_arena(bytes: u32) -> Option<*mut u8> {
    let base = crate::gpu_textures::arena_base();
    if base.is_null() {
        return None;
    }
    // Safe: only the display thread calls this.
    let offset = unsafe { DISPLAY.get() }.cache.reserve_arena(bytes)?;
    // Safe: inside the arena (the cache hands out its pages only).
    Some(unsafe { base.add(offset as usize) })
}

/// Display thread: where the last frame's texture time went.
pub fn frame_times() -> FrameTimes {
    // Safe: only the display thread calls this.
    unsafe { DISPLAY.get() }.cache.frame_times()
}

/// The texture counters for the render statistics, shared with the thread
/// that reads them (`take_counters`).
pub struct SharedTextureStats {
    decode_us: AtomicU32,
    refresh_us: AtomicU32,
    hashed_bytes: AtomicU32,
    decodes: AtomicU32,
    refreshes: AtomicU32,
    create_failures: AtomicU32,
    texture_bytes: AtomicU32,
    evictions: AtomicU32,
    arena_bytes: AtomicU32,
    finds: AtomicU32,
    scans: AtomicU32,
}

pub static STATS: SharedTextureStats = SharedTextureStats {
    decode_us: AtomicU32::new(0),
    refresh_us: AtomicU32::new(0),
    hashed_bytes: AtomicU32::new(0),
    decodes: AtomicU32::new(0),
    refreshes: AtomicU32::new(0),
    create_failures: AtomicU32::new(0),
    texture_bytes: AtomicU32::new(0),
    evictions: AtomicU32::new(0),
    arena_bytes: AtomicU32::new(0),
    finds: AtomicU32::new(0),
    scans: AtomicU32::new(0),
};

impl SharedTextureStats {
    pub fn add(&self, stats: &TextureStats) {
        let add = |counter: &AtomicU32, value: u64| counter.fetch_add(value as u32, Ordering::Relaxed);
        add(&self.decode_us, stats.decode_us);
        add(&self.refresh_us, stats.refresh_us);
        add(&self.hashed_bytes, stats.hashed_bytes);
        add(&self.decodes, stats.decodes as u64);
        add(&self.refreshes, stats.refreshes as u64);
        add(&self.create_failures, stats.create_failures as u64);
        add(&self.evictions, stats.evictions as u64);
        add(&self.finds, stats.finds as u64);
        add(&self.scans, stats.scans as u64);
        self.texture_bytes.store(stats.texture_bytes, Ordering::Relaxed);
        self.arena_bytes.store(stats.arena_bytes, Ordering::Relaxed);
    }
}

/// The texture cache's counters for the render statistics.
pub struct Counters {
    pub decode_us: u32,
    pub refresh_us: u32,
    pub hashed_bytes: u32,
    pub decodes: u32,
    pub refreshes: u32,
    pub create_failures: u32,
    /// Now, not since the last call.
    pub texture_bytes: u32,
    pub evictions: u32,
    /// Now, not since the last call.
    pub arena_bytes: u32,
    /// `find()` calls, and of them hint misses (scans of all entries).
    pub finds: u32,
    pub scans: u32,
}

/// The counters since the last call (any thread).
pub fn take_counters() -> Counters {
    let take = |counter: &AtomicU32| counter.swap(0, Ordering::Relaxed);
    Counters {
        decode_us: take(&STATS.decode_us),
        refresh_us: take(&STATS.refresh_us),
        hashed_bytes: take(&STATS.hashed_bytes),
        decodes: take(&STATS.decodes),
        refreshes: take(&STATS.refreshes),
        create_failures: take(&STATS.create_failures),
        texture_bytes: STATS.texture_bytes.load(Ordering::Relaxed),
        evictions: take(&STATS.evictions),
        arena_bytes: STATS.arena_bytes.load(Ordering::Relaxed),
        finds: take(&STATS.finds),
        scans: take(&STATS.scans),
    }
}

// ------------------------------------------------------------------------
// The worker thread (see `nativeres_core::texture_cache::refresh`)
// ------------------------------------------------------------------------

const IDLE: u32 = 0;
const RUNNING: u32 = 1;
const DONE: u32 = 2;

/// The job handed between the display thread and the worker. `state` says
/// who owns `job` and `result`: the display thread while IDLE or DONE, the
/// worker while RUNNING. Whoever hands them over stores `state` last
/// (Release), the other side loads it first (Acquire).
struct Worker {
    state: AtomicU32,
    /// Semaphore the worker waits on; < 0: no worker.
    wakeup: AtomicI32,
    staging: AtomicPtr<u8>,
    job: UnsafeCell<Option<RefreshJob>>,
    result: UnsafeCell<Option<RefreshResult>>,
}

// Safe: `job` and `result` are only used by the side `state` names.
unsafe impl Sync for Worker {}

static WORKER: Worker = Worker {
    state: AtomicU32::new(IDLE),
    wakeup: AtomicI32::new(-1),
    staging: AtomicPtr::new(null_mut()),
    job: UnsafeCell::new(None),
    result: UnsafeCell::new(None),
};

static WORKER_RUNNING: AtomicBool = AtomicBool::new(false);

/// Starts the worker and its own staging buffer. On failure big textures
/// are re-checked on the display thread.
fn start_worker() {
    let staging = allocate_staging(b"NativeResRefreshStaging\0");
    if staging.is_null() {
        return;
    }
    WORKER.staging.store(staging, Ordering::Release);
    // Safe: plain system calls.
    unsafe {
        let wakeup = sys::sceKernelCreateSema(b"NativeResRefreshWakeup\0".as_ptr(), 0, 0, 1, null_mut());
        if wakeup < 0 {
            log_line(format_args!("background refresh unavailable: sema {:08x}\n", wakeup));
            return;
        }
        WORKER.wakeup.store(wakeup, Ordering::Release);
        let thread = sys::sceKernelCreateThread(
            b"NativeResRefresh\0".as_ptr(),
            worker_thread,
            WORKER_PRIORITY,
            WORKER_STACK_SIZE,
            0,
            0,
            null_mut(),
        );
        sys::note_thread(thread);
        if thread < 0 || sys::sceKernelStartThread(thread, 0, null_mut()) < 0 {
            log_line(format_args!("background refresh unavailable: thread {:08x}\n", thread));
            return;
        }
    }
    WORKER_RUNNING.store(true, Ordering::Release);
}

unsafe extern "C" fn worker_thread(_args: u32, _argp: *mut c_void) -> i32 {
    let wakeup = WORKER.wakeup.load(Ordering::Acquire);
    let staging = core::slice::from_raw_parts_mut(WORKER.staging.load(Ordering::Acquire), STAGING_SIZE);
    while sys::sceKernelWaitSema(wakeup, 1, null_mut()) >= 0 {
        if WORKER.state.load(Ordering::Acquire) != RUNNING {
            continue;
        }
        // Safe: RUNNING, so the job is ours.
        let Some(job) = *WORKER.job.get() else { continue };
        let hook = nativeres_core::debug_hooks::span(nativeres_core::debug_hooks::Point::Texture);
        *WORKER.result.get() = Some(run_refresh(&mut VitaHost, &job, staging));
        drop(hook);
        WORKER.state.store(DONE, Ordering::Release); // the texels and result before the state
    }
    sys::sceKernelExitDeleteThread(0)
}

// ------------------------------------------------------------------------
// The host
// ------------------------------------------------------------------------

/// Appends a line to the render log.
fn log_line(text: fmt::Arguments) {
    crate::render_log::line(text);
}

/// Holds no state: what it reaches is the Vita's, or `WORKER`'s.
pub struct VitaHost;


fn as_object(texture: GpuTexture) -> *mut gpu_textures::TextureObject {
    texture.handle as *mut gpu_textures::TextureObject
}

impl Host for VitaHost {
    fn texture_arena_bytes(&mut self) -> u32 {
        gpu_textures::texture_arena_bytes()
    }

    fn create_texture(&mut self, shape: &TextureShape, arena_offset: Option<u32>) -> Option<GpuTexture> {
        // An arena offset comes from the cache's page allocator, so the
        // pages are this texture's alone.
        let texture = gpu_textures::create(shape, arena_offset);
        (!texture.is_null()).then_some(GpuTexture { handle: texture as usize, arena_offset })
    }

    fn free_texture(&mut self, texture: GpuTexture) {
        // Safe: the cache frees each texture it made once.
        unsafe { gpu_textures::free_texture(as_object(texture)) };
    }

    fn decode(
        &mut self,
        texture: GpuTexture,
        format: GpuFormat,
        as_texture: &TextureDescriptor,
        data: &[u8],
        palette: &[u32],
        smooth: bool,
        level: u8,
    ) -> Result<(), DecodeError> {
        // Safe: a texture made for `as_texture`'s size (level 0's size, or
        // this level of a texture with levels), which nobody else writes now
        // (fresh or retired long enough).
        let base = unsafe { gpu_textures::data(as_object(texture)) };
        // Every level's rows: its own width rounded up to 8 texels (GXM's
        // layout); a mip level at its place (texture::mip_level_offset).
        let texel_bytes = gpu_textures::texel_bytes(format);
        let stride = (as_texture.width as usize).div_ceil(8) * 8 * texel_bytes;
        let texels = if level == 0 {
            base
        } else {
            let (width, height) = ((as_texture.width as u32) << level, (as_texture.height as u32) << level);
            let offset = texture::mip_level_offset(width, height, texel_bytes as u32, level as u32) as usize;
            // Safe: the texture was made with all its levels (TextureShape::bytes).
            unsafe { base.add(offset) }
        };
        let height = as_texture.height as usize;
        let bytes = unsafe { core::slice::from_raw_parts_mut(texels, stride * height) };
        match format {
            GpuFormat::Palette8 => texture::decode_palette_indices(data, as_texture, bytes, stride),
            format if format.is_16_bit() => texture::decode_16_bit(data, as_texture, bytes, stride / 2),
            GpuFormat::Wide16 => {
                // RGBA into the second half (rows of stride / 8 texels), then
                // widened in place from the front: texel i's 8 bytes end at
                // 8 i + 8 <= 4 n + 4 i + 4 (n texels), never past a texel not
                // yet read. Safe: word-aligned (page-aligned memory).
                let texel_count = stride / 8 * height;
                let pixels = unsafe { core::slice::from_raw_parts_mut(texels.add(texel_count * 4) as *mut u32, texel_count) };
                texture::decode_texture(data, as_texture, palette, pixels, stride / 8)?;
                // Safe: the whole level, 8 bytes a texel, u16-aligned.
                let wide = unsafe { core::slice::from_raw_parts_mut(texels as *mut u16, texel_count * 4) };
                texture::widen_rgba(wide, texel_count);
                Ok(())
            }
            _ => {
                // Safe: GPU texture memory is page-aligned, so word-aligned.
                let pixels = unsafe { core::slice::from_raw_parts_mut(texels as *mut u32, stride / 4 * height) };
                texture::decode_texture(data, as_texture, palette, pixels, stride / 4)?;
                if smooth {
                    texture::bleed_transparent_texels(pixels, as_texture.width as usize, height, stride / 4);
                }
                Ok(())
            }
        }
    }

    fn gpu_unswizzle_ready(&mut self, width: u16, height: u16) -> bool {
        crate::unswizzle::ready(width as u32, height as u32)
    }

    fn unswizzle_on_gpu(&mut self, wanted: &TextureDescriptor, target: GpuTexture, raw: GpuTexture) -> bool {
        // Safe: render-target textures of the right shape, made by us and
        // not freed.
        unsafe { crate::unswizzle::unswizzle_on_gpu(wanted, &*as_object(target), &*as_object(raw)) }
    }

    fn wait_for_gpu(&mut self) {
        // Safe: plain call on the display thread.
        unsafe { gxm::sceGxmFinish(gxm::context()) };
    }

    fn free_memory(&self) -> u32 {
        let mut info = sys::SceKernelFreeMemorySizeInfo {
            size: core::mem::size_of::<sys::SceKernelFreeMemorySizeInfo>() as i32,
            size_user: 0,
            size_cdram: 0,
            size_phycont: 0,
        };
        // Safe: a valid, sized struct.
        unsafe { sys::sceKernelGetFreeMemorySize(&mut info) };
        info.size_user.max(0) as u32
    }

    fn psp_memory(&self, address: u32, size: u32) -> Option<&[u8]> {
        let data = psp_memory(address, size);
        // Safe: mapped PSP memory (the game may write it meanwhile; we only
        // copy and hash it).
        (!data.is_null()).then(|| unsafe { core::slice::from_raw_parts(data, size as usize) })
    }

    fn psp_words(&self, address: u32, size: u32) -> Option<&[u32]> {
        let data = psp_memory(address, size);
        // Safe: as `psp_memory`; texture addresses are 16-byte aligned.
        (!data.is_null() && data as usize % 4 == 0)
            .then(|| unsafe { core::slice::from_raw_parts(data as *const u32, size as usize / 4) })
    }

    fn copy(&self, target: &mut [u8], source: &[u8]) {
        let size = target.len().min(source.len()) as u32;
        // Safe: both hold `size` bytes and don't overlap (staging vs. PSP memory).
        unsafe { psp_copy(target.as_mut_ptr(), source.as_ptr(), size) };
    }

    fn vram(&self) -> VramState {
        capture::vram_state()
    }

    fn captured_frame_number(&self) -> u32 {
        CAPTURED_FRAME_NUMBER.load(Ordering::Acquire)
    }

    fn has_refresh_worker(&self) -> bool {
        WORKER_RUNNING.load(Ordering::Acquire)
    }

    fn start_refresh(&mut self, job: &RefreshJob) {
        // Safe: the cache only starts a job while the worker is IDLE, so
        // the job is ours.
        unsafe {
            *WORKER.job.get() = Some(*job);
            *WORKER.result.get() = None;
        }
        WORKER.state.store(RUNNING, Ordering::Release);
        // Safe: plain call.
        unsafe { sys::sceKernelSignalSema(WORKER.wakeup.load(Ordering::Acquire), 1) };
    }

    fn finished_refresh(&self) -> Option<RefreshResult> {
        if WORKER.state.load(Ordering::Acquire) != DONE {
            return None;
        }
        // Safe: DONE, so the result is ours.
        unsafe { *WORKER.result.get() }
    }

    fn end_refresh(&mut self) {
        WORKER.state.store(IDLE, Ordering::Release);
    }

    fn now_us(&self) -> u64 {
        sys::now_us()
    }

    fn log(&mut self, text: fmt::Arguments) {
        log_line(text);
    }
}

/// Display loop (menu.c): 1 if Sony's picture is to be copied right after
/// the LCD sync (`nativeres_core::display_loop::sony_copy_point`), 0 if only
/// once `NativeResRenderFrame` falls back to it.
pub fn copy_sony_picture_early(showed_sony_picture_last: bool) -> bool {
    use nativeres_core::display_loop::{sony_copy_point, CopyPoint};
    sony_copy_point(showed_sony_picture_last) == CopyPoint::AfterSync
}
