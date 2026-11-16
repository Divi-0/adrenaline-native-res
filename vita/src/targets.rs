//! Render targets: one per PSP framebuffer (display thread only).
//!
//! A PSP framebuffer keeps its contents from one frame to the next, and
//! games rely on it: closing Final Fantasy's menu, the game draws only 29 of
//! the menu's ~800 draws per frame on top of what is already there. Drawn
//! straight onto the (cleared) Vita screen, the rest of the menu was missing
//! (black portraits, broken text).
//!
//! So each PSP framebuffer address gets a 960x544 render target that keeps
//! its contents too. A frame is drawn into the target of the framebuffer it
//! shows, in a GXM scene of its own (frame_loop.rs), and the display's scene
//! then copies that target to the screen (`NativePspPresentFrame`). A frame
//! is drawn into its target only once: drawing it again would apply
//! additive draws twice.
//!
//! A scene without a depth/stencil surface drew nothing at all (the target
//! stayed untouched, no error). All targets share one: games clear the depth
//! buffer before 3D drawing, and a frame is drawn in one scene, so only its
//! contents during that scene matter (GXM doesn't keep them).

use core::ptr::{null, null_mut};
use core::sync::atomic::{AtomicI32, AtomicU32, Ordering};

use crate::gpu_memory::{self, GpuBlock};
use crate::gpu_textures::TextureObject;
use crate::render_stats::stats;
use crate::basic_programs::{self, TextureVertex};
use crate::{draw, gxm, render_log, sys, SingleThreaded};

/// Double buffering. Was 3 ("plus one for games that use more"); 2 since
/// 2026-11-05 to save 2 MiB of main memory. A game that draws into more
/// framebuffers makes us reuse a target it used just before: counted in
/// the render statistics (target_evictions) and logged.
pub const TARGET_COUNT: usize = 2;
pub const TARGET_WIDTH: u32 = 960;
pub const TARGET_HEIGHT: u32 = 544;
const TARGET_BYTES: u32 = sys::pages((TARGET_WIDTH * TARGET_HEIGHT * 4) as usize);

/// The targets and their shared depth buffer.
#[no_mangle]
#[used]
#[allow(non_upper_case_globals)]
pub static nativepsp_plan_fixed_render_targets: u32 = (TARGET_COUNT as u32 + 1) * TARGET_BYTES;

/// Indices 0, 1, 2, ... for sceGxmDraw, which always wants an index buffer.
const LINEAR_INDICES: usize = nativepsp_core::frame::MAX_VERTICES;
#[no_mangle]
#[used]
#[allow(non_upper_case_globals)]
pub static nativepsp_plan_fixed_linear_indices: u32 = sys::pages(LINEAR_INDICES * 2);

/// Set to 1 by `NativePspPresentFrame`; whoever watches the shown frames
/// reads and clears it after each swap.
pub static FRAME_DRAWN: AtomicI32 = AtomicI32::new(0);
/// The PSP frame number (`Frame::frame_number`) `NativePspPresentFrame`
/// showed last.
pub static SHOWN_FRAME_NUMBER: AtomicU32 = AtomicU32::new(0);

/// The framebuffers we have targets for, for the capture (capture.rs
/// `display_targets`): textures drawn from a target need no VRAM copy from
/// the PSP plugin. Updated every drawn frame.
pub static DISPLAY_TARGETS: [AtomicU32; 4] = [AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0)];

/// Sony's picture of the PSP screen (SCE_PSPEMU_FRAMEBUFFER, as dump_file.rs
/// ORIGINAL_PICTURE): 512 pixels per row, 272 rows, RGBA.
pub const SONY_PICTURE: *const u32 = 0x7400_0000 as *const u32;
pub const SONY_PICTURE_STRIDE: usize = 512;

/// The four corners of a quad as a triangle strip.
const STRIP_CORNERS: [[f32; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
/// The full-screen strip's size.
const VERTEX_BYTES: u32 = core::mem::size_of::<TextureVertex>() as u32;
const QUAD_BYTES: u32 = 4 * VERTEX_BYTES;

pub struct Target {
    /// The PSP framebuffer; 0 = not in use.
    pub psp_address: u32,
    /// `render_count` (frame_loop.rs) when last drawn into or sampled.
    pub last_used: u32,
    /// Of the last frame drawn into it.
    pub frame_number: u32,
    /// New: cleared before the first frame.
    pub needs_clear: bool,
    /// Its 960x544 RGBA pixels (uncached); null until made.
    pixels: *mut u32,
    surface: gxm::ColorSurface,
    render_target: *mut gxm::RenderTarget,
    /// Only `gxm` is used: for drawing with it.
    pub texture: TextureObject,
    /// The same picture sampled with alpha 1 (format X8U8U8U8_1BGR), for
    /// draws that ignore the texture's alpha (`texture_flags::IGNORE_ALPHA`,
    /// texture function "RGB"): our alpha holds the PSP's stencil values,
    /// which such a draw doesn't see.
    pub opaque_view: TextureObject,
}

const NO_TEXTURE: gxm::Texture = gxm::Texture { control_words: [0; 4] };

const UNUSED_TARGET: Target = Target {
    psp_address: 0,
    last_used: 0,
    frame_number: 0,
    needs_clear: false,
    pixels: null_mut(),
    surface: gxm::ColorSurface { words: [0; 12] },
    render_target: null_mut(),
    texture: TextureObject::view(NO_TEXTURE),
    opaque_view: TextureObject::view(NO_TEXTURE),
};

impl Target {
    /// Its GPU memory and render target exist.
    pub fn exists(&self) -> bool {
        !self.render_target.is_null()
    }

    /// Makes its memory, colour surface and render target; false if it
    /// can't be made.
    fn create(&mut self) -> bool {
        if shared_depth().is_null() {
            return false;
        }
        let Some(GpuBlock { uid, base }) = gpu_memory::for_gpu(b"NativePspTarget\0", TARGET_BYTES) else {
            return false;
        };
        let params = gxm::RenderTargetParams {
            flags: 0,
            width: TARGET_WIDTH as u16,
            height: TARGET_HEIGHT as u16,
            scenes_per_frame: 1,
            multisample_mode: gxm::MULTISAMPLE_NONE,
            multisample_locations: 0,
            driver_mem_block: -1,
        };
        let mut result = 0;
        // Safe: our fresh block, valid parameters and out pointers.
        let made = unsafe {
            gxm::sceGxmColorSurfaceInit(
                &mut self.surface,
                gxm::COLOR_FORMAT_A8B8G8R8,
                gxm::COLOR_SURFACE_LINEAR,
                gxm::COLOR_SURFACE_SCALE_NONE,
                gxm::OUTPUT_REGISTER_SIZE_32BIT,
                TARGET_WIDTH,
                TARGET_HEIGHT,
                TARGET_WIDTH,
                base as *mut _,
            ) >= 0
                && {
                    result = gxm::sceGxmCreateRenderTarget(&params, &mut self.render_target);
                    result >= 0
                }
        };
        if !made {
            render_log::line(format_args!("render target: creation failed: {:08x}\n", result));
            self.render_target = null_mut();
            // Safe: our block, mapped by for_gpu, used by nothing.
            unsafe {
                gxm::sceGxmUnmapMemory(base as *mut _);
                sys::free_block(uid);
            }
            return false;
        }
        self.texture = TextureObject::view(NO_TEXTURE);
        // Safe: our block of 960x544 texels.
        unsafe {
            gxm::sceGxmTextureInitLinear(&mut self.texture.gxm, base as *const _, gxm::TEXTURE_FORMAT_U8U8U8U8_ABGR, TARGET_WIDTH, TARGET_HEIGHT, 0)
        };
        self.pixels = base as *mut u32;
        true
    }

    /// Its pixel at Vita screen position (x, y); 0 if it has no memory.
    pub fn pixel(&self, x: usize, y: usize) -> u32 {
        if self.pixels.is_null() || x >= TARGET_WIDTH as usize || y >= TARGET_HEIGHT as usize {
            return 0;
        }
        // Safe: inside the target's memory (uncached, GPU-written: read as is).
        unsafe { core::ptr::read_volatile(self.pixels.add(y * TARGET_WIDTH as usize + x)) }
    }
}

pub struct Targets {
    pub targets: [Target; TARGET_COUNT],
    /// What `NativePspPresentFrame` shows.
    pub shown: Option<usize>,
    /// Log lines left: targets reused too early, failed scenes.
    eviction_logs: u32,
    begin_logs: u32,
}

static TARGETS: SingleThreaded<Targets> =
    SingleThreaded::new(Targets { targets: [UNUSED_TARGET; TARGET_COUNT], shown: None, eviction_logs: 10, begin_logs: 5 });

/// The display thread's targets.
///
/// # Safety
/// Display thread only, and only one reference at a time (each entry point
/// takes it once and passes it on).
pub unsafe fn targets() -> &'static mut Targets {
    TARGETS.get()
}

/// The depth/stencil surface all targets (and the GPU unswizzle) share,
/// made on first use; null if it can't be.
static SHARED_DEPTH: SingleThreaded<Option<gxm::DepthStencilSurface>> = SingleThreaded::new(None);

pub fn shared_depth() -> *const gxm::DepthStencilSurface {
    // Safe: display thread only; the reference ends in this function.
    let depth = unsafe { SHARED_DEPTH.get() };
    if depth.is_none() {
        // 4 bytes per texel too.
        let Some(GpuBlock { uid, base }) = gpu_memory::for_gpu(b"NativePspTargetDepth\0", TARGET_BYTES) else {
            return null();
        };
        let mut surface = gxm::DepthStencilSurface { words: [0; 5] };
        // Safe: our fresh block; 960x544 are whole tiles, so the stride is
        // the width.
        let made = unsafe {
            gxm::sceGxmDepthStencilSurfaceInit(
                &mut surface,
                gxm::DEPTH_STENCIL_FORMAT_S8D24,
                gxm::DEPTH_STENCIL_SURFACE_TILED,
                TARGET_WIDTH,
                base as *mut _,
                null_mut(),
            ) >= 0
        };
        if !made {
            // Safe: our block, used by nothing.
            unsafe {
                gxm::sceGxmUnmapMemory(base as *mut _);
                sys::free_block(uid);
            }
            return null();
        }
        *depth = Some(surface);
    }
    // The static never moves; the surface is never replaced.
    depth.as_ref().map_or(null(), |surface| surface as *const _)
}

/// Indices 0, 1, 2, ... (`LINEAR_INDICES` of them), made on first use; null
/// if there is no memory.
static LINEAR: SingleThreaded<*mut u16> = SingleThreaded::new(null_mut());

pub fn linear_indices() -> *const u16 {
    // Safe: display thread only.
    let indices = unsafe { LINEAR.get() };
    if indices.is_null() {
        let Some(block) = gpu_memory::gpu_input(b"NativePspIndices\0", sys::pages(LINEAR_INDICES * 2)) else {
            return null();
        };
        let words = block.base as *mut u16;
        for i in 0..LINEAR_INDICES {
            // Safe: inside the block.
            unsafe { words.add(i).write(i as u16) };
        }
        *indices = words;
    }
    *indices
}

/// The full-screen strip (4 vertices, texture coordinates 0..1): always
/// the same, so made once in GPU memory; null if there is no memory.
static FULL_QUAD: SingleThreaded<*mut TextureVertex> = SingleThreaded::new(null_mut());

fn full_quad() -> *const TextureVertex {
    // Safe: display thread only.
    let quad = unsafe { FULL_QUAD.get() };
    if quad.is_null() {
        let Some(block) = gpu_memory::gpu_input(b"NativePspFullQuad\0", sys::pages(QUAD_BYTES as usize)) else {
            return null();
        };
        let vertices = block.base as *mut TextureVertex;
        for (i, corner) in STRIP_CORNERS.iter().enumerate() {
            // Safe: inside the block.
            unsafe {
                vertices.add(i).write(TextureVertex {
                    x: corner[0] * TARGET_WIDTH as f32,
                    y: corner[1] * TARGET_HEIGHT as f32,
                    z: 0.5,
                    w: 1.0,
                    u: corner[0],
                    v: corner[1],
                })
            };
        }
        *quad = vertices;
    }
    *quad
}

/// Draws `texture` over the whole scene, 1:1 with point filtering, with
/// the basic texture programs and `fragment` (our copy program writes all
/// four channels as they are). False if nothing could be drawn.
///
/// # Safety
/// Display thread, inside a scene.
unsafe fn draw_full_quad(texture: &gxm::Texture, fragment: *const gxm::FragmentProgram) -> bool {
    let (quad, indices) = (full_quad(), linear_indices());
    if quad.is_null() || indices.is_null() || fragment.is_null() || !basic_programs::use_texture_programs(fragment) {
        return false;
    }
    let context = gxm::context();
    gxm::sceGxmSetFragmentTexture(context, 0, texture);
    gxm::sceGxmSetVertexStream(context, 0, quad as *const _);
    gxm::sceGxmDraw(context, gxm::PRIMITIVE_TRIANGLE_STRIP, gxm::INDEX_FORMAT_U16, indices as *const _, 4);
    true
}

/// Draws `texture` on the 4-vertex strip `quad` (GPU memory, in screen
/// pixels) with the basic texture programs (alpha blending): the GAME fps
/// label.
///
/// # Safety
/// Display thread, inside the display's scene; `quad` mapped for the GPU
/// and not written while the GPU may read it.
pub unsafe fn draw_quad(texture: &gxm::Texture, quad: *const TextureVertex) {
    let indices = linear_indices();
    let Some(basic) = basic_programs::get() else {
        return;
    };
    if indices.is_null() || !basic_programs::use_texture_programs(basic.texture_tint_fragment) {
        return;
    }
    let context = gxm::context();
    gxm::sceGxmSetFragmentTexture(context, 0, texture);
    gxm::sceGxmSetVertexStream(context, 0, quad as *const _);
    gxm::sceGxmDraw(context, gxm::PRIMITIVE_TRIANGLE_STRIP, gxm::INDEX_FORMAT_U16, indices as *const _, 4);
}

impl Targets {
    /// The target that holds PSP framebuffer `psp_address`, if any.
    pub fn find(&self, psp_address: u32) -> Option<usize> {
        self.targets.iter().position(|target| target.exists() && target.psp_address == psp_address)
    }

    /// The target for a PSP framebuffer: its own, or the least recently used
    /// one (cleared first; a target not made yet before any other). None if
    /// none can be made. `render_count`: the frame being drawn.
    pub fn target_for(&mut self, psp_address: u32, render_count: u32) -> Option<usize> {
        if let Some(index) = self.find(psp_address) {
            return Some(index);
        }
        let mut oldest = 0;
        for (index, target) in self.targets.iter().enumerate().skip(1) {
            let current = &self.targets[oldest];
            if !target.exists() || (current.exists() && target.last_used < current.last_used) {
                oldest = index;
            }
        }
        let target = &mut self.targets[oldest];
        if !target.exists() && !target.create() {
            return None;
        }
        if target.psp_address != 0 && target.last_used.wrapping_add(2) >= render_count {
            stats().target_evictions += 1;
            if self.eviction_logs > 0 {
                self.eviction_logs -= 1;
                render_log::line(format_args!(
                    "render target for {:08x} reused for {:08x}: the game uses more than {} framebuffers\n",
                    target.psp_address, psp_address, TARGET_COUNT
                ));
            }
        }
        target.psp_address = psp_address;
        target.needs_clear = true;
        Some(oldest)
    }

    /// The renderer was off: what the targets hold is out of date.
    pub fn forget(&mut self) {
        for target in self.targets.iter_mut() {
            target.psp_address = 0;
        }
        self.shown = None;
    }

    /// Starts a scene drawing into target `index` (cleared first if new).
    pub fn begin_scene(&mut self, index: usize) {
        let target = &mut self.targets[index];
        // Safe: the display thread's context; a target that exists has the
        // shared depth surface.
        let result = unsafe {
            gxm::sceGxmBeginScene(gxm::context(), 0, target.render_target, null(), null_mut(), null_mut(), &target.surface, shared_depth())
        };
        if result < 0 && self.begin_logs > 0 {
            self.begin_logs -= 1;
            render_log::line(format_args!("sceGxmBeginScene (target {:08x}) failed: {:08x}\n", target.psp_address, result));
        }
        if target.needs_clear {
            // Safe: inside the scene.
            unsafe { basic_programs::clear_screen() };
            target.needs_clear = false;
        }
        draw::forget_gxm_state();
    }

    /// Copies what target `from` holds into target `to` (same size and
    /// format, alpha included: the game's alpha is part of the picture it
    /// samples), in a scene of its own before the frame's. True if submitted.
    pub fn copy(&mut self, from: usize, to: usize) -> bool {
        let copy_program = draw::copy_program();
        let depth = shared_depth();
        if copy_program.is_null() || !self.targets[from].exists() || !self.targets[to].exists() || depth.is_null() {
            return false;
        }
        let (source, target) = (&self.targets[from], &self.targets[to]);
        // Safe: both targets exist.
        unsafe {
            let context = gxm::context();
            if gxm::sceGxmBeginScene(context, 0, target.render_target, null(), null_mut(), null_mut(), &target.surface, depth) < 0 {
                return false;
            }
            draw_full_quad(&source.texture.gxm, copy_program);
            gxm::sceGxmEndScene(context, null(), null());
        }
        draw::forget_gxm_state();
        let frame_number = self.targets[from].frame_number;
        let target = &mut self.targets[to];
        target.frame_number = frame_number;
        target.needs_clear = false;
        true
    }

    /// Fills target `index` with Sony's picture of the PSP screen, 2x,
    /// instead of black. For a target a frame samples (MTX's motion blur
    /// keeps up to 95 % of the other framebuffer every frame) that holds
    /// nothing of the game's now: new (renderer just switched on: our
    /// blurred frames faded in from black over seconds) or from before a
    /// savestate load (the old place's picture showed through as a ghost for
    /// seconds). Sony's picture is the frame the PSP shows, the one before
    /// the frame being drawn. CPU writes (once per such event, ~2 MiB
    /// uncached), the GPU drained first.
    pub fn seed_from_sony(&mut self, index: usize, frame_number: u32) {
        let target = &mut self.targets[index];
        if target.pixels.is_null() {
            return;
        }
        // Safe: the display thread's context.
        unsafe { gxm::sceGxmFinish(gxm::context()) };
        // Safe: display thread only.
        let row = unsafe { SEED_ROW.get() };
        let width = TARGET_WIDTH as usize;
        for y in 0..TARGET_HEIGHT as usize / 2 {
            for x in 0..width / 2 {
                // Safe: inside Sony's picture (512 x 272 words, mapped).
                let pixel = unsafe { core::ptr::read_volatile(SONY_PICTURE.add(y * SONY_PICTURE_STRIDE + x)) };
                row[2 * x] = pixel;
                row[2 * x + 1] = pixel;
            }
            // Safe: two rows inside the target's memory.
            unsafe {
                core::ptr::copy_nonoverlapping(row.as_ptr(), target.pixels.add(2 * y * width), width);
                core::ptr::copy_nonoverlapping(row.as_ptr(), target.pixels.add((2 * y + 1) * width), width);
            }
        }
        target.needs_clear = false;
        target.frame_number = frame_number;
        stats().seeded_targets += 1;
    }

    /// Tells the capture which PSP framebuffers have targets
    /// (`DISPLAY_TARGETS`).
    pub fn publish_addresses(&self) {
        for (k, published) in DISPLAY_TARGETS.iter().enumerate() {
            let address = self.targets.get(k).filter(|target| target.exists()).map_or(0, |target| target.psp_address);
            published.store(address, Ordering::Relaxed);
        }
    }
}

/// One doubled row of Sony's picture (`seed_from_sony`).
static SEED_ROW: SingleThreaded<[u32; TARGET_WIDTH as usize]> = SingleThreaded::new([0; TARGET_WIDTH as usize]);

/// Display thread, inside the display's scene: shows the target of the last
/// frame drawn full-screen. Opaque: the target's alpha is the game's, not
/// meant for the screen.
pub fn present_frame() {
    // Safe: the display thread's entry point.
    let targets = unsafe { targets() };
    let Some(shown) = targets.shown else {
        return;
    };
    let target = &targets.targets[shown];
    let copy_program = draw::copy_program();
    let fragment = if copy_program.is_null() { basic_programs::get().map_or(null_mut(), |basic| basic.texture_tint_fragment) } else { copy_program };
    // Safe: inside the display's scene.
    if !unsafe { draw_full_quad(&target.texture.gxm, fragment) } {
        return;
    }
    FRAME_DRAWN.store(1, Ordering::Relaxed);
    SHOWN_FRAME_NUMBER.store(target.frame_number, Ordering::Relaxed);
}
