//! Unswizzling textures on the GPU (display thread only; the texture cache
//! asks for it, textures.rs).
//!
//! Final Fantasy's battle-start effect draws its previous frames, which the
//! game copies into 512x512 16-bit textures, a new one every few frames. The
//! CPU took ~4.5 ms to unswizzle one (and ~1.3 ms to copy it out of PSP
//! memory), so the rays ran at ~40 fps. Here the CPU only copies the data,
//! and the GPU puts the texels in order.
//!
//! A swizzled texture stores blocks of 16 bytes x 8 rows one after another:
//! for 16-bit texels, 8x8 squares, row of blocks after row of blocks. Read
//! as a texture 8 texels wide, the data is a column of those squares, each
//! in its right shape; drawing every square at its place in the texture
//! unswizzles it. The column is cut into slices of 4096 rows (the GPU's
//! largest texture height), 512 blocks each. One quad per block: 4096 for
//! 512x512, drawn with the basic texture program and our copy program, no
//! blending, point sampling (texel for texel).
//!
//! The quads are placed in the 960x544 coordinates (basic_programs::ORTHO), scaled to the
//! texture: the scene's viewport covers its render target, whatever its size.

use core::ptr::{null, null_mut};
use core::sync::atomic::{AtomicI32, Ordering};

use nativeres_core::frame::TextureDescriptor;

use crate::gpu_textures::TextureObject;
use crate::platform::{psp_copy, psp_memory};
use crate::targets::{shared_depth, TARGET_HEIGHT, TARGET_WIDTH};
use crate::basic_programs::{self, TextureVertex};
use crate::{draw, gpu_memory, gxm, render_log, sys, SingleThreaded};

const SLICE_ROWS: u32 = 4096;
const SLICE_BLOCKS: u32 = SLICE_ROWS / 8;
/// texture_cache::gpu::GPU_UNSWIZZLE_SLICE_BYTES.
const SLICE_BYTES: usize = 8 * SLICE_ROWS as usize * 2;

const MAX_LAYOUTS: usize = 4;
/// Quad indices for one slice: 6 per quad.
const QUAD_INDEX_BYTES: u32 = 0x2000;
const _: () = assert!(SLICE_BLOCKS as usize * 6 * 2 <= QUAD_INDEX_BYTES as usize);

/// At most: every layout for a 512x512 texture, and the quad indices.
#[no_mangle]
#[used]
#[allow(non_upper_case_globals)]
pub static nativeres_plan_fixed_unswizzle: u32 = sys::pages(
    MAX_LAYOUTS * sys::pages((512 / 8) * (512 / 8) * 4 * core::mem::size_of::<TextureVertex>()) as usize + QUAD_INDEX_BYTES as usize,
);

/// Compares a texture the GPU unswizzled with the data it came from (waits
/// for the GPU, ~100 ms): the next n.
/// First run, 2026-10-30: 3 of 3 Final Fantasy snapshots exact.
pub static UNSWIZZLE_CHECKS: AtomicI32 = AtomicI32::new(0);

/// The quads for one texture size (they only depend on it), GPU-readable.
#[derive(Clone, Copy)]
struct Layout {
    width: u32,
    height: u32,
    /// 4 per block, `SLICE_BLOCKS * 4` per slice; null: slot free.
    vertices: *const TextureVertex,
}

struct Layouts {
    layouts: [Layout; MAX_LAYOUTS],
    quad_indices: *const u16,
}

static LAYOUTS: SingleThreaded<Layouts> =
    SingleThreaded::new(Layouts { layouts: [Layout { width: 0, height: 0, vertices: null() }; MAX_LAYOUTS], quad_indices: null() });

/// The quads for textures of this size, made on first use; None if there is
/// no memory or no free slot.
fn layout_for(width: u32, height: u32) -> Option<Layout> {
    // Safe: display thread only; the reference ends in this function.
    let state = unsafe { LAYOUTS.get() };
    if let Some(layout) = state.layouts.iter().find(|l| !l.vertices.is_null() && l.width == width && l.height == height) {
        return Some(*layout);
    }
    if state.quad_indices.is_null() {
        let block = gpu_memory::gpu_input(b"NativeResQuadIndices\0", QUAD_INDEX_BYTES)?;
        let indices = block.base as *mut u16;
        const CORNERS: [u16; 6] = [0, 1, 2, 2, 1, 3];
        for quad in 0..SLICE_BLOCKS as usize {
            for (k, corner) in CORNERS.iter().enumerate() {
                // Safe: inside the block (asserted above).
                unsafe { indices.add(quad * 6 + k).write(quad as u16 * 4 + corner) };
            }
        }
        state.quad_indices = indices;
    }
    let slot = state.layouts.iter_mut().find(|l| l.vertices.is_null())?;
    let blocks = width / 8 * (height / 8);
    let bytes = sys::pages(blocks as usize * 4 * core::mem::size_of::<TextureVertex>());
    let block = gpu_memory::gpu_input(b"NativeResUnswizzle\0", bytes)?;
    let vertices = block.base as *mut TextureVertex;
    let (scale_x, scale_y) = (TARGET_WIDTH as f32 / width as f32, TARGET_HEIGHT as f32 / height as f32);
    const CORNERS: [[f32; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    for index in 0..blocks {
        // Where the block goes, and where it is in its slice.
        let x = (index % (width / 8) * 8) as f32;
        let y = (index / (width / 8) * 8) as f32;
        let v = (index % SLICE_BLOCKS * 8) as f32 / SLICE_ROWS as f32;
        for (k, corner) in CORNERS.iter().enumerate() {
            let vertex = TextureVertex {
                x: (x + corner[0] * 8.0) * scale_x,
                y: (y + corner[1] * 8.0) * scale_y,
                z: 0.5,
                w: 1.0,
                u: corner[0],
                v: v + corner[1] * 8.0 / SLICE_ROWS as f32,
            };
            // Safe: inside the block (4 vertices per block).
            unsafe { vertices.add(index as usize * 4 + k).write(vertex) };
        }
    }
    *slot = Layout { width, height, vertices };
    Some(*slot)
}

/// Where texel (x, y) of a swizzled 16-bit texture `width` texels wide is in
/// its data, counted in texels.
fn swizzled_texel(width: u32, x: u32, y: u32) -> usize {
    (((y / 8) * (width / 8) + x / 8) * 64 + (y % 8) * 8 + x % 8) as usize
}

/// Compares `target` (after the GPU) with `raw`.
fn check_unswizzled(texture: &TextureDescriptor, raw: *const u16, target: &TextureObject) {
    // Safe: the display thread's context; both textures hold width x height
    // 16-bit texels.
    unsafe {
        gxm::sceGxmFinish(gxm::context());
        let texels = gxm::sceGxmTextureGetData(&target.gxm) as *const u16;
        let (width, height) = (texture.width as u32, texture.height as u32);
        let (mut wrong, mut first_x, mut first_y) = (0u32, 0, 0);
        for y in 0..height {
            for x in 0..width {
                if *texels.add((y * width + x) as usize) != *raw.add(swizzled_texel(width, x, y)) {
                    if wrong == 0 {
                        (first_x, first_y) = (x, y);
                    }
                    wrong += 1;
                }
            }
        }
        render_log::line(format_args!(
            "unswizzle check {}x{} fmt={}: {} texels differ (first at {},{}: {:04x}, wanted {:04x})\n",
            width,
            height,
            texture.format,
            wrong,
            first_x,
            first_y,
            *texels.add((first_y * width + first_x) as usize),
            *raw.add(swizzled_texel(width, first_x, first_y))
        ));
    }
}

/// True if the GPU unswizzle can run for textures of this size now: its
/// vertices, program and the shared depth surface exist or could be made.
pub fn ready(width: u32, height: u32) -> bool {
    layout_for(width, height).is_some() && !draw::copy_program().is_null() && !shared_depth().is_null()
}

/// Copies the texture's swizzled data into `raw` (a render-target texture
/// of `target`'s size and format, from the cache, which retires it
/// afterwards: the GPU reads it) and has the GPU unswizzle it into `target`
/// (see texture_cache::gpu::can_unswizzle_on_gpu), in a scene of its own
/// before the frame's. False, having submitted nothing, if it can't.
pub fn unswizzle_on_gpu(texture: &TextureDescriptor, target: &TextureObject, raw: &TextureObject) -> bool {
    let (width, height) = (texture.width as u32, texture.height as u32);
    let bytes = width * height * 2;
    let source = psp_memory(texture.address & 0x0FFF_FFFF, bytes);
    let Some(layout) = layout_for(width, height) else {
        return false;
    };
    let copy_program = draw::copy_program();
    let depth = shared_depth();
    if source.is_null() || copy_program.is_null() || depth.is_null() || target.render_target.is_null() || raw.render_target.is_null() {
        return false;
    }
    // Safe: render-target textures we made, of this size; the source is
    // mapped PSP memory of `bytes`; `layout` and the quad indices are GPU
    // memory kept for good.
    unsafe {
        let format = gxm::sceGxmTextureGetFormat(&target.gxm);
        let raw_data = gxm::sceGxmTextureGetData(&raw.gxm) as *mut u8;
        psp_copy(raw_data, source, bytes);

        let context = gxm::context();
        let result =
            gxm::sceGxmBeginScene(context, 0, target.render_target, null(), null_mut(), null_mut(), &target.color_surface, depth);
        if result < 0 {
            render_log::line(format_args!("unswizzle: sceGxmBeginScene failed: {:08x}\n", result));
            return false;
        }
        if !basic_programs::use_texture_programs(copy_program) {
            gxm::sceGxmEndScene(context, null(), null());
            return false;
        }
        let quad_indices = LAYOUTS.get().quad_indices;
        let blocks = width / 8 * (height / 8);
        let mut first = 0;
        while first < blocks {
            // GXM copies texture state when it is set.
            let mut slice = gxm::Texture { control_words: [0; 4] };
            let data = raw_data.add((first / SLICE_BLOCKS) as usize * SLICE_BYTES);
            gxm::sceGxmTextureInitLinear(&mut slice, data as *const _, format, 8, SLICE_ROWS, 0);
            gxm::sceGxmSetFragmentTexture(context, 0, &slice);
            gxm::sceGxmSetVertexStream(context, 0, layout.vertices.add(first as usize * 4) as *const _);
            let quads = (blocks - first).min(SLICE_BLOCKS);
            gxm::sceGxmDraw(context, gxm::PRIMITIVE_TRIANGLES, gxm::INDEX_FORMAT_U16, quad_indices as *const _, quads * 6);
            first += SLICE_BLOCKS;
        }
        gxm::sceGxmEndScene(context, null(), null());
        draw::forget_gxm_state();

        if UNSWIZZLE_CHECKS.load(Ordering::Relaxed) > 0 {
            UNSWIZZLE_CHECKS.fetch_sub(1, Ordering::Relaxed);
            check_unswizzled(texture, raw_data as *const u16, target);
        }
    }
    true
}
