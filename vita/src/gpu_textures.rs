//! GPU textures for the texture cache (textures.rs): made, freed, and the
//! texture arena most of them live in.
//!
//! Our textures live in main memory (USER_RW), not vita2d's default CDRAM:
//! the PSP emulator uses nearly all CDRAM, and the first test ran out of it
//! after about a dozen textures. The GPU reads main memory too.
//!
//! They are not made with vita2d_create_empty_texture_format: Adrenaline's
//! vita2d (with render-target support) gives every texture a depth/stencil
//! buffer of 4 bytes per texel, uncached and GPU-mapped, and
//! vita2d_free_texture never frees it. So a 512x512 texture cost 1 MiB extra
//! and 25-50 ms to create, and every replaced texture leaked that (free
//! memory in Final Fantasy's battles fell from 24 to 11 MiB). Here one
//! memory block holds the texels and, for palette textures, the palette in
//! its last page (or the texture arena does).
//!
//! A `TextureObject` still has vita2d_texture's layout (from when C read
//! it); the draw path takes its `gxm`, the GPU unswizzle (unswizzle.rs) its
//! `render_target` and `color_surface`.
//!
//! Render-target textures (`TextureShape::render_target`, the GPU
//! unswizzle draws them): GPU-writable memory, a render target (shared by
//! all textures of that size) and a colour surface. Only reused as such,
//! and the CPU never writes them. Cached memory like the others: uncached
//! took 9-13 ms to allocate per 512 KiB, and the GPU's results read back
//! exact, so no dirty cache lines get written back
//! over them.
//!
//! Display thread only (the texture cache's).

use core::ffi::c_void;
use core::fmt;

use nativeres_core::texture_cache::gpu::{GpuFormat, TextureShape};

use crate::{gxm, heap, render_log, sys, SingleThreaded};

/// vita2d_texture (100 bytes), the fields we use named.
#[repr(C)]
pub struct TextureObject {
    /// gxm_tex
    pub gxm: gxm::Texture,
    /// data_UID: the texels' own block, or -1 in the texture arena.
    pub data_uid: sys::SceUID,
    palette_uid: sys::SceUID,
    /// gxm_rtgt: render-target textures only.
    pub render_target: *mut gxm::RenderTarget,
    /// gxm_sfc: render-target textures only.
    pub color_surface: gxm::ColorSurface,
    depth_surface: gxm::DepthStencilSurface,
    depth_uid: sys::SceUID,
}

const _: () = assert!(core::mem::size_of::<TextureObject>() == 100);

impl TextureObject {
    /// An object for a texture that isn't ours to free (a render target's
    /// picture, targets.rs): only `gxm` is used.
    pub const fn view(gxm: gxm::Texture) -> TextureObject {
        TextureObject {
            gxm,
            data_uid: -1,
            palette_uid: -1,
            render_target: core::ptr::null_mut(),
            color_surface: gxm::ColorSurface { words: [0; 12] },
            depth_surface: gxm::DepthStencilSurface { words: [0; 5] },
            depth_uid: -1,
        }
    }
}

/// GXM's texture format for the cache's `GpuFormat`.
pub fn gxm_format(format: GpuFormat) -> u32 {
    match format {
        GpuFormat::Rgba8888 => gxm::TEXTURE_FORMAT_U8U8U8U8_ABGR,
        GpuFormat::Palette8 => gxm::TEXTURE_FORMAT_P8_ABGR,
        GpuFormat::Bgr565 => gxm::TEXTURE_FORMAT_U5U6U5_BGR,
        GpuFormat::Abgr1555 => gxm::TEXTURE_FORMAT_U1U5U5U5_ABGR,
        GpuFormat::Abgr4444 => gxm::TEXTURE_FORMAT_U4U4U4U4_ABGR,
        GpuFormat::Wide16 => gxm::TEXTURE_FORMAT_U16U16U16U16_ABGR,
    }
}

/// The colour format a render-target texture is drawn in (the GPU unswizzle:
/// 16-bit as they are; 8888 for its device test, unswizzle::expand_test).
fn color_format(format: GpuFormat) -> Option<u32> {
    match format {
        GpuFormat::Rgba8888 => Some(gxm::COLOR_FORMAT_A8B8G8R8),
        GpuFormat::Bgr565 => Some(gxm::COLOR_FORMAT_U5U6U5_BGR),
        GpuFormat::Abgr1555 => Some(gxm::COLOR_FORMAT_U1U5U5U5_ABGR),
        GpuFormat::Abgr4444 => Some(gxm::COLOR_FORMAT_U4U4U4U4_ABGR),
        _ => None,
    }
}

/// Bytes per texel in GPU memory.
pub fn texel_bytes(format: GpuFormat) -> usize {
    format.texel_bytes() as usize
}

/// The texture arena (texture_cache/arena.rs): one phycont block, mapped for
/// the GPU once, that the texture cache hands out in 4 KiB pages; most
/// textures live there instead of in blocks of their own. Main memory is
/// the short kind (~3 MiB free in MTX's race), so the cache kept dropping
/// textures the last frame didn't use and decoded them again at every
/// camera cut; phycont had ~10 MiB free (2026-11-10). Made on first use and
/// kept.
const TEXTURE_ARENA_BYTES: usize = 6 * 1024 * 1024;
#[no_mangle]
#[used]
#[allow(non_upper_case_globals)]
pub static nativeres_plan_phycont_texture_arena: u32 = sys::pages(TEXTURE_ARENA_BYTES);

/// Render targets by texture size (a render target only describes the
/// size; the colour surface says where the texels go).
const MAX_RENDER_TARGETS: usize = 4;

struct State {
    /// The texture arena, null until made.
    arena: *mut u8,
    render_targets: [(u32, u32, *mut gxm::RenderTarget); MAX_RENDER_TARGETS],
}

static STATE: SingleThreaded<State> =
    SingleThreaded::new(State { arena: core::ptr::null_mut(), render_targets: [(0, 0, core::ptr::null_mut()); MAX_RENDER_TARGETS] });

fn log(text: fmt::Arguments) {
    render_log::line(text);
}

/// The texture arena's size, made on the first call; 0 if there is none
/// (textures then get blocks of their own).
pub fn texture_arena_bytes() -> u32 {
    // Safe: display thread only.
    let state = unsafe { STATE.get() };
    if !state.arena.is_null() {
        return TEXTURE_ARENA_BYTES as u32;
    }
    // Safe: plain system calls; the block is ours alone.
    unsafe {
        let block = sys::alloc_block(
            b"NativeResTextureArena\0".as_ptr(),
            sys::SCE_KERNEL_MEMBLOCK_TYPE_USER_MAIN_PHYCONT_RW,
            TEXTURE_ARENA_BYTES as u32,
            core::ptr::null_mut(),
        );
        if block < 0 {
            log(format_args!("texture arena: no memory ({:08x}), textures get blocks of their own\n", block));
            return 0;
        }
        let mut base: *mut c_void = core::ptr::null_mut();
        sys::sceKernelGetMemBlockBase(block, &mut base);
        if gxm::sceGxmMapMemory(base, TEXTURE_ARENA_BYTES as u32, gxm::MEMORY_ATTRIB_READ) < 0 {
            log(format_args!("texture arena: not mapped for the GPU\n"));
            sys::free_block(block);
            return 0;
        }
        state.arena = base as *mut u8;
    }
    log(format_args!("texture arena ready: {} KiB phycont memory\n", TEXTURE_ARENA_BYTES / 1024));
    TEXTURE_ARENA_BYTES as u32
}

/// The render target for textures of this size, made on first use; null if
/// it can't be.
fn render_target_for(width: u32, height: u32) -> *mut gxm::RenderTarget {
    // Safe: display thread only.
    let state = unsafe { STATE.get() };
    if let Some(&(_, _, target)) =
        state.render_targets.iter().find(|&&(w, h, target)| !target.is_null() && w == width && h == height)
    {
        return target;
    }
    let Some(slot) = state.render_targets.iter_mut().find(|(_, _, target)| target.is_null()) else {
        return core::ptr::null_mut();
    };
    let params = gxm::RenderTargetParams {
        flags: 0,
        width: width as u16,
        height: height as u16,
        scenes_per_frame: 1,
        multisample_mode: gxm::MULTISAMPLE_NONE,
        multisample_locations: 0,
        driver_mem_block: -1,
    };
    let mut target: *mut gxm::RenderTarget = core::ptr::null_mut();
    // Safe: valid parameters and out pointer.
    let result = unsafe { gxm::sceGxmCreateRenderTarget(&params, &mut target) };
    if result < 0 {
        log(format_args!("unswizzle: render target {}x{} failed: {:08x}\n", width, height, result));
        return core::ptr::null_mut();
    }
    *slot = (width, height, target);
    target
}

/// A texture of `shape`, in the arena at `arena_offset` (pages the cache
/// allocated there) or in a block of its own; null if it can't be made.
pub fn create(shape: &TextureShape, arena_offset: Option<u32>) -> *mut TextureObject {
    // Safe: display thread only.
    let state = unsafe { STATE.get() };
    let render_target = shape.render_target;
    let levels = if render_target { 1 } else { shape.levels.max(1) };
    let bytes = TextureShape { levels, ..*shape }.bytes();
    let (width, height) = (shape.width as u32, shape.height as u32);
    let format = gxm_format(shape.format);
    // Safe: plain system calls on memory that is ours: a fresh block, or
    // arena pages the cache handed to this texture alone.
    unsafe {
        let (data, block) = match arena_offset {
            Some(offset) => {
                // Already mapped for the GPU (read only: never a render target).
                if state.arena.is_null() || render_target || offset as usize + bytes as usize > TEXTURE_ARENA_BYTES {
                    return core::ptr::null_mut();
                }
                (state.arena.add(offset as usize) as *mut c_void, -1)
            }
            None => {
                // Render targets of whole MiB (512x512 8888: the
                // unswizzle tests) in phycont memory, which
                // hands out whole MiB and has more room than main memory
                // (~3 MiB left in MTX's race); the rest, and phycont
                // failures, in main memory.
                let mut block = -1;
                if render_target && bytes % (1024 * 1024) == 0 {
                    block = sys::alloc_block(
                        b"NativeResTexture\0".as_ptr(),
                        sys::SCE_KERNEL_MEMBLOCK_TYPE_USER_MAIN_PHYCONT_RW,
                        bytes,
                        core::ptr::null_mut(),
                    );
                }
                if block < 0 {
                    block = sys::alloc_block(
                        b"NativeResTexture\0".as_ptr(),
                        sys::SCE_KERNEL_MEMBLOCK_TYPE_USER_RW,
                        bytes,
                        core::ptr::null_mut(),
                    );
                }
                if block < 0 {
                    return core::ptr::null_mut();
                }
                let mut data: *mut c_void = core::ptr::null_mut();
                sys::sceKernelGetMemBlockBase(block, &mut data);
                (data, block)
            }
        };
        let access = if render_target { gxm::MEMORY_ATTRIB_READ | gxm::MEMORY_ATTRIB_WRITE } else { gxm::MEMORY_ATTRIB_READ };
        let texture = heap::alloc_zeroed(core::mem::size_of::<TextureObject>()) as *mut TextureObject;
        if texture.is_null() || (block >= 0 && gxm::sceGxmMapMemory(data, bytes, access) < 0) {
            heap::free(texture as *mut c_void);
            if block >= 0 {
                sys::free_block(block);
            }
            return core::ptr::null_mut();
        }
        let object = &mut *texture;
        if render_target {
            object.render_target = render_target_for(width, height);
            let made = !object.render_target.is_null()
                && color_format(shape.format).is_some_and(|color_format| {
                    gxm::sceGxmColorSurfaceInit(
                        &mut object.color_surface,
                        color_format,
                        gxm::COLOR_SURFACE_LINEAR,
                        gxm::COLOR_SURFACE_SCALE_NONE,
                        gxm::OUTPUT_REGISTER_SIZE_32BIT,
                        width,
                        height,
                        width,
                        data,
                    ) >= 0
                });
            if !made {
                // Render targets never live in the arena: `block` is ours.
                gxm::sceGxmUnmapMemory(data);
                heap::free(texture as *mut c_void);
                sys::free_block(block);
                return core::ptr::null_mut();
            }
        }
        object.data_uid = block;
        gxm::sceGxmTextureInitLinear(&mut object.gxm, data, format, width, height, if levels > 1 { levels as u32 } else { 0 });
        if shape.format == GpuFormat::Palette8 {
            // The palette in the last page.
            gxm::sceGxmTextureSetPalette(&mut object.gxm, (data as *const u8).add(bytes as usize - 0x1000) as *const c_void);
        }
        texture
    }
}

/// Frees a texture `create` made. An arena texture (data_uid -1) only frees
/// its object: the cache gives its pages back.
///
/// # Safety
/// The GPU must be done with it (the cache retires textures first), and it
/// must not be used again.
pub unsafe fn free_texture(texture: *mut TextureObject) {
    let object = &*texture;
    if object.data_uid >= 0 {
        gxm::sceGxmUnmapMemory(gxm::sceGxmTextureGetData(&object.gxm));
        sys::free_block(object.data_uid);
    }
    heap::free(texture as *mut c_void);
}

/// The texels of a texture `create` made.
///
/// # Safety
/// `texture` must be one `create` made and not freed.
pub unsafe fn data(texture: *const TextureObject) -> *mut u8 {
    gxm::sceGxmTextureGetData(&(*texture).gxm) as *mut u8
}
