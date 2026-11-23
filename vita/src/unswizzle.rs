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
//! Device tests (debug "expandtest", "filtertest"): the GPU's widening of
//! 16-bit texels, single and linear-filtered, against the GE's.
//!
//! The quads are placed in the 960x544 coordinates (basic_programs::ORTHO), scaled to the
//! texture: the scene's viewport covers its render target, whatever its size.

use core::ptr::{null, null_mut};
use core::sync::atomic::{AtomicI32, Ordering};

use nativeres_core::frame::TextureDescriptor;

use crate::gpu_textures::{self, TextureObject};
use crate::platform::{psp_copy, psp_memory};
use crate::targets::{shared_depth, TARGET_HEIGHT, TARGET_WIDTH};
use nativeres_core::color;
use nativeres_core::texture::format as psp_format;
use nativeres_core::texture_cache::gpu::{GpuFormat, TextureShape};

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

/// What texel `raw` (PSP 16-bit, `format`) must become in a texture of
/// `target_format`: itself, or widened to 8888 as the GE does.
fn wanted_texel(raw: u16, format: u8, target_format: u32) -> u32 {
    if target_format != gxm::TEXTURE_FORMAT_U8U8U8U8_ABGR {
        return raw as u32;
    }
    match format {
        psp_format::BGR5650 => color::rgba_from_5650(raw),
        psp_format::ABGR5551 => color::rgba_from_5551(raw),
        _ => color::rgba_from_4444(raw),
    }
}

/// `raw` (PSP 16-bit, `format`) widened as k / n rounded to 8 bits per
/// channel (n = 31, 63, 15, 1): the model of a GPU that samples k / n. The
/// expand test counts against it too, to tell which one the GPU does.
fn linear_model_texel(raw: u16, format: u8) -> u32 {
    let widen = |value: u32, bits: u32| {
        let n = (1u32 << bits) - 1;
        ((value & n) * 255 * 2 + n) / (2 * n)
    };
    let raw = raw as u32;
    let (r, g, b, a) = match format {
        psp_format::BGR5650 => (widen(raw, 5), widen(raw >> 5, 6), widen(raw >> 11, 5), 255),
        psp_format::ABGR5551 => (widen(raw, 5), widen(raw >> 5, 5), widen(raw >> 10, 5), widen(raw >> 15, 1)),
        _ => (widen(raw, 4), widen(raw >> 4, 4), widen(raw >> 8, 4), widen(raw >> 12, 4)),
    };
    a << 24 | b << 16 | g << 8 | r
}

/// Compares `target` (after the GPU) with `raw`: texel for texel, or each
/// widened as the GE does (8888 targets). Returns the number that differ.
pub fn check_unswizzled(texture: &TextureDescriptor, raw: *const u16, target: &TextureObject) -> u32 {
    // Safe: the display thread's context; `raw` holds width x height 16-bit
    // texels, `target` as many of its format.
    unsafe {
        gxm::sceGxmFinish(gxm::context());
        let target_format = gxm::sceGxmTextureGetFormat(&target.gxm);
        let data = gxm::sceGxmTextureGetData(&target.gxm);
        let texel_at = |index: usize| {
            if target_format == gxm::TEXTURE_FORMAT_U8U8U8U8_ABGR {
                *(data as *const u32).add(index)
            } else {
                *(data as *const u16).add(index) as u32
            }
        };
        let (width, height) = (texture.width as u32, texture.height as u32);
        let wanted_at = |x: u32, y: u32| wanted_texel(*raw.add(swizzled_texel(width, x, y)), texture.format, target_format);
        let (mut wrong, mut first_x, mut first_y) = (0u32, 0, 0);
        let mut off_linear_model = 0u32;
        let wide = target_format == gxm::TEXTURE_FORMAT_U8U8U8U8_ABGR;
        for y in 0..height {
            for x in 0..width {
                let texel = texel_at((y * width + x) as usize);
                if texel != wanted_at(x, y) {
                    if wrong == 0 {
                        (first_x, first_y) = (x, y);
                    }
                    wrong += 1;
                }
                if wide && texel != linear_model_texel(*raw.add(swizzled_texel(width, x, y)), texture.format) {
                    off_linear_model += 1;
                }
            }
        }
        if wide {
            render_log::line(format_args!("unswizzle check: {} texels differ from k / n rounded\n", off_linear_model));
        }
        render_log::line(format_args!(
            "unswizzle check {}x{} fmt={}: {} texels differ (first at {},{}: {:08x}, wanted {:08x})\n",
            width,
            height,
            texture.format,
            wrong,
            first_x,
            first_y,
            texel_at((first_y * width + first_x) as usize),
            wanted_at(first_x, first_y)
        ));
        wrong
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
    if source.is_null() || !can_draw(width, height, target, raw) {
        return false;
    }
    // Safe: `raw` is a render-target texture we made of this size in a
    // 16-bit format; the source is mapped PSP memory of `bytes`.
    let raw_data = unsafe { gxm::sceGxmTextureGetData(&raw.gxm) as *mut u8 };
    unsafe { psp_copy(raw_data, source, bytes) };
    if !draw_unswizzled(width, height, target, raw) {
        return false;
    }
    if UNSWIZZLE_CHECKS.load(Ordering::Relaxed) > 0 {
        UNSWIZZLE_CHECKS.fetch_sub(1, Ordering::Relaxed);
        check_unswizzled(texture, raw_data as *const u16, target);
    }
    true
}

/// True if `draw_unswizzled` can run for these textures.
fn can_draw(width: u32, height: u32, target: &TextureObject, raw: &TextureObject) -> bool {
    layout_for(width, height).is_some()
        && !draw::copy_program().is_null()
        && !shared_depth().is_null()
        && !target.render_target.is_null()
        && !raw.render_target.is_null()
}

/// Draws the swizzled 16-bit data in `raw` into `target` in order with the
/// copy program (into a 16-bit texture of its format: as it is; into 8888:
/// widened by the texture unit, the expand test), in a scene of its own.
/// False, having submitted nothing, if it can't.
fn draw_unswizzled(width: u32, height: u32, target: &TextureObject, raw: &TextureObject) -> bool {
    let Some(layout) = layout_for(width, height) else {
        return false;
    };
    // Safe: a texture we made.
    let format = unsafe { gxm::sceGxmTextureGetFormat(&raw.gxm) };
    let program = draw::copy_program() as *mut gxm::FragmentProgram;
    let depth = shared_depth();
    if program.is_null() || depth.is_null() || target.render_target.is_null() || raw.render_target.is_null() {
        return false;
    }
    // Safe: render-target textures we made, of this size; `layout` and the
    // quad indices are GPU memory kept for good.
    unsafe {
        let raw_data = gxm::sceGxmTextureGetData(&raw.gxm) as *mut u8;
        let context = gxm::context();
        let result =
            gxm::sceGxmBeginScene(context, 0, target.render_target, null(), null_mut(), null_mut(), &target.color_surface, depth);
        if result < 0 {
            render_log::line(format_args!("unswizzle: sceGxmBeginScene failed: {:08x}\n", result));
            return false;
        }
        if !basic_programs::use_texture_programs(program) {
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
    }
    true
}

/// A device test of the GPU's linear filter on 16-bit textures (debug
/// "filtertest"): does a U5U6U5 / U1U5U5U5 / U4U4U4U4 texture filter to the
/// same values as its 8888 expansion (the GE's widening, which the texture
/// unit also does for single texels: `expand_test`)? A 256x256 texture of
/// pseudo-random texels and its 8888 copy, each drawn magnified ~7x (73
/// texels over 512 pixels: many filter weights) with the linear filter
/// through the copy program into a 512x512 8888 target, compared pixel for
/// pixel. `control`: the 8888 copy drawn point-filtered instead, which must
/// differ. (differing pixels, largest channel difference); None if it
/// couldn't run. Display thread, outside any scene. 2026-11-23: 0 pixels
/// for all three formats (control: ~262k, up to 192 levels).
pub fn filter_test(psp_format: u8, control: bool) -> Option<(u32, u32)> {
    const SIZE: u16 = 256;
    const OUT: u16 = 512;
    let texture = TextureDescriptor { width: SIZE, height: SIZE, buffer_width: SIZE as u32, format: psp_format, ..Default::default() };
    let raw_format = GpuFormat::raw_16_bit(&texture)?;
    let shape = |width, format, render_target| TextureShape { width, height: width, format, render_target, levels: 1 };
    let raw = gpu_textures::create(&shape(SIZE, raw_format, false), None);
    let wide = gpu_textures::create(&shape(SIZE, GpuFormat::Rgba8888, false), None);
    let out_raw = gpu_textures::create(&shape(OUT, GpuFormat::Rgba8888, true), None);
    let out_wide = gpu_textures::create(&shape(OUT, GpuFormat::Rgba8888, true), None);
    let quad = filter_quad();
    let made = [raw, wide, out_raw, out_wide].iter().all(|texture| !texture.is_null()) && !quad.is_null();
    let mut result = None;
    if made {
        // Safe: textures just made, of these sizes and formats.
        unsafe {
            let texels = gpu_textures::data(raw) as *mut u16;
            let expanded = gpu_textures::data(wide) as *mut u32;
            let mut seed = 0x1234_5678u32;
            for i in 0..SIZE as usize * SIZE as usize {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let value = (seed >> 16) as u16;
                texels.add(i).write(value);
                expanded.add(i).write(wanted_texel(value, psp_format, gxm::TEXTURE_FORMAT_U8U8U8U8_ABGR));
            }
            let wide_filter = if control { gxm::TEXTURE_FILTER_POINT } else { gxm::TEXTURE_FILTER_LINEAR };
            let drawn = draw_filtered(&*out_raw, &mut (*raw).gxm, gxm::TEXTURE_FILTER_LINEAR, quad)
                && draw_filtered(&*out_wide, &mut (*wide).gxm, wide_filter, quad);
            if drawn {
                gxm::sceGxmFinish(gxm::context());
                let a = gpu_textures::data(out_raw) as *const u32;
                let b = gpu_textures::data(out_wide) as *const u32;
                let (mut differ, mut largest, mut first) = (0u32, 0u32, None);
                for i in 0..OUT as usize * OUT as usize {
                    let (x, y) = (*a.add(i), *b.add(i));
                    if x != y {
                        differ += 1;
                        first.get_or_insert((i, x, y));
                        for shift in [0, 8, 16, 24] {
                            largest = largest.max(((x >> shift) & 0xFF).abs_diff((y >> shift) & 0xFF));
                        }
                    }
                }
                let (at, x, y) = first.unwrap_or((0, 0, 0));
                render_log::line(format_args!(
                    "filter test fmt={} control={}: {} of {} pixels differ, largest {} (first at {}: {:08x} vs {:08x})\n",
                    psp_format, control as u8, differ, OUT as u32 * OUT as u32, largest, at, x, y
                ));
                result = Some((differ, largest));
            }
        }
    }
    // Safe: made above; the GPU is done with them (finished above, or
    // nothing was submitted).
    unsafe {
        for texture in [raw, wide, out_raw, out_wide] {
            if !texture.is_null() {
                gpu_textures::free_texture(texture);
            }
        }
    }
    result
}

/// The filter test's quad: the whole target, texture coordinates 0..73/256
/// (made once, GPU-readable; null if there is no memory).
fn filter_quad() -> *const TextureVertex {
    static QUAD: SingleThreaded<*const TextureVertex> = SingleThreaded::new(null());
    // Safe: display thread only; the reference ends in this function.
    let quad = unsafe { QUAD.get() };
    if quad.is_null() {
        let Some(block) = gpu_memory::gpu_input(b"NativeResFilterQuad\0", 4096) else {
            return null();
        };
        let span = 73.0 / 256.0;
        let corners: [[f32; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        for (k, corner) in corners.iter().enumerate() {
            let vertex = TextureVertex {
                x: corner[0] * TARGET_WIDTH as f32,
                y: corner[1] * TARGET_HEIGHT as f32,
                z: 0.5,
                w: 1.0,
                u: 0.01 + corner[0] * span,
                v: 0.01 + corner[1] * span,
            };
            // Safe: 4 vertices fit the block.
            unsafe { (block.base as *mut TextureVertex).add(k).write(vertex) };
        }
        *quad = block.base as *const TextureVertex;
    }
    *quad
}

/// Draws `source` (filtered by `filter`, min and mag) over all of `target`
/// with the copy program, in a scene of its own.
///
/// # Safety
/// Display thread, outside any scene; `target` a render-target texture.
unsafe fn draw_filtered(target: &TextureObject, source: &mut gxm::Texture, filter: u32, quad: *const TextureVertex) -> bool {
    let program = draw::copy_program() as *mut gxm::FragmentProgram;
    let depth = shared_depth();
    let indices = crate::targets::linear_indices();
    if program.is_null() || depth.is_null() || indices.is_null() || target.render_target.is_null() {
        return false;
    }
    gxm::sceGxmTextureSetMinFilter(source, filter);
    gxm::sceGxmTextureSetMagFilter(source, filter);
    let context = gxm::context();
    if gxm::sceGxmBeginScene(context, 0, target.render_target, null(), null_mut(), null_mut(), &target.color_surface, depth) < 0 {
        return false;
    }
    if !basic_programs::use_texture_programs(program) {
        gxm::sceGxmEndScene(context, null(), null());
        return false;
    }
    gxm::sceGxmSetFragmentTexture(context, 0, source);
    gxm::sceGxmSetVertexStream(context, 0, quad as *const _);
    gxm::sceGxmDraw(context, gxm::PRIMITIVE_TRIANGLE_STRIP, gxm::INDEX_FORMAT_U16, indices as *const _, 4);
    gxm::sceGxmEndScene(context, null(), null());
    draw::forget_gxm_state();
    true
}

/// A device test of the texture unit's widening (debug "expandtest"):
/// every 16-bit value of the PSP format `psp_format` (0 5650, 1 5551, 2
/// 4444) unswizzled as the cache's textures are (512x512, the size of FF1's
/// frame copies: each value 4 times) but into an 8888 texture, compared
/// with the GE's widening (`check_unswizzled`, render log "unswizzle check
/// ..."; it also counts the texels off a k / n model: the proof that the
/// check sees a different widening, ~35 % of them for 5650 / 5551). The
/// differing texels; None if it couldn't run. Display thread, outside any
/// scene. 2026-11-23: 0 for all three formats.
pub fn expand_test(psp_format: u8) -> Option<u32> {
    let texture = &TextureDescriptor { width: 512, height: 512, buffer_width: 512, format: psp_format, swizzled: 1, ..Default::default() };
    let (width, height) = (texture.width as u32, texture.height as u32);
    let target_format = GpuFormat::Rgba8888;
    let raw_format = GpuFormat::raw_16_bit(texture)?;
    let shape = |format| TextureShape { width: texture.width, height: texture.height, format, render_target: true, levels: 1 };
    let target = gpu_textures::create(&shape(target_format), None);
    let raw = gpu_textures::create(&shape(raw_format), None);
    let mut wrong = None;
    if !target.is_null() && !raw.is_null() {
        // Safe: textures just made, of this size.
        unsafe {
            let raw_data = gpu_textures::data(raw) as *mut u16;
            for y in 0..height {
                for x in 0..width {
                    raw_data.add(swizzled_texel(width, x, y)).write((y * width + x) as u16);
                }
            }
            if draw_unswizzled(width, height, &*target, &*raw) {
                // Waits for the GPU (sceGxmFinish) before comparing.
                wrong = Some(check_unswizzled(texture, raw_data, &*target));
            }
        }
    }
    // Safe: made above; the GPU is done with them (check_unswizzled
    // finished it, or nothing was submitted).
    unsafe {
        if !target.is_null() {
            gpu_textures::free_texture(target);
        }
        if !raw.is_null() {
            gpu_textures::free_texture(raw);
        }
    }
    wrong
}
