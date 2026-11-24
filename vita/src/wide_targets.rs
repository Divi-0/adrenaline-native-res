//! Wide copies of off-screen targets (display thread only; DESIGN
//! "sixty-sixth session").
//!
//! God of War's GE_WIDE programs filter a texture as the GE from two LINEAR
//! reads instead of four point samples, but only from a U16x4 texture
//! holding k x 257 per channel (texture cache `GpuFormat::Wide16`; the
//! texture unit's 8-bit lerp rounds, its 16-bit one is exact). The cache
//! makes those for the textures it decodes. Textures the game draws from
//! its own framebuffers come from our off-screen targets instead
//! (frame_loop.rs `use_targets_as_textures`), 8888 surfaces: point samples.
//! In GoW's gameplay those are 43 % of the textured pixels (the bloom
//! picture added over the screen, the six shadow tiles), magnified many
//! times, where the two LINEAR reads gain most.
//!
//! So the GPU copies such a texture's part of its target into an 8888
//! surface twice as wide: texel x becomes pixels 2x (r, r, g, g) and 2x + 1
//! (b, b, a, a) (nativeres_widecopy_f.cg), the bytes of a U16x4 texel with
//! k x 257, and draws onto the screen read that surface as a U16x4 texture
//! of the PSP texture's own size (same texels, so the same pixels; and the
//! texture's own edges, as the GE clamps or wraps them).
//!
//! Only for the textures draws into the shown framebuffer read (draws into
//! off-screen targets take programs without wide twins; told by the frame
//! loop as they are drawn, `note_read`, so a texture gets its copy from the
//! next frame on), of power-of-two sizes (exact LINEAR coordinates) inside
//! their target (1:1 pixels). Those draws read such textures with the
//! texture's own edges either way (frame_loop.rs `screen_view`: a view of
//! just its part, or the copy), so a frame with or without the copy shows
//! the same pixels (device: 0 differing pixels, a gameplay frame). The
//! copies are grouped by size into classes: one surface per size, its slots
//! stacked, one render target, so the copies a class needs are one scene.
//! A slot is copied again whenever its target was drawn into since
//! (`Target::generation`), right before a scene into the shown framebuffer
//! (`flush`): GoW's gameplay makes 2-3 copy scenes a frame (the tiles once,
//! the bloom picture before each of the two screen scenes that add it).
//! The surfaces live in pages of the texture arena the cache gives up for
//! good when first needed (`POOL_BYTES`; in God of War main memory, CDRAM
//! beyond the movie reserve and phycont are all taken, 2026-11-24); while
//! the arena has no free run that long, the textures are read through
//! views of their own part (the same pixels) and the next frame asks again.

use core::ptr::{null, null_mut};
use core::sync::atomic::AtomicU8;

use crate::basic_programs::{self, TextureVertex};
use crate::gpu_textures::TextureObject;
use crate::render_stats::stats;
use crate::targets::{self, Targets, TARGET_HEIGHT, TARGET_WIDTH};
use crate::{draw, gpu_memory, gxm, render_log, shaders, sys, textures, SingleThreaded};

/// Debug "widetargets 0|1|2", where wide textures are on (frame_loop.rs
/// `wide`): draws into the shown framebuffer read off-screen target
/// textures through views of the whole target from the texture on (0, as
/// before 2026-11-24: their clamp and wrap at the target's edges), of just
/// the texture's part (1), or that and wide copies for the GE_WIDE programs
/// (2, the default; the same pixels as 1).
pub static WIDE_TARGETS: AtomicU8 = AtomicU8::new(2);

/// The textures draws into the shown framebuffer read lately (`note_read`):
/// (key, `render_count` of the frame), oldest replaced.
const MAX_READS: usize = 16;
/// A copy is made only for textures read so within this many frames.
const READ_FRAMES: u32 = 4;

/// Sizes (classes) and slots per size.
const MAX_CLASSES: usize = 3;
const MAX_SLOTS: usize = 8;
/// A class's surface takes about this much: as many slots as fit, at least
/// one, at most `MAX_SLOTS` (GoW: eight 64x64 tiles in 256 KiB, one 256x256
/// bloom picture in 512 KiB).
const CLASS_BYTES: u32 = 256 * 1024;
/// The largest and smallest texture side copied. At least 16: surfaces of
/// whole 32-pixel rows (twice as wide), as our off-screen targets have.
pub const MAX_SIDE: u32 = 256;
pub const MIN_SIDE: u32 = 16;
/// U16x4: 8 bytes a texel.
const TEXEL_BYTES: u32 = 8;

/// All classes' surfaces: taken from the texture arena (GoW: 192 KiB of
/// tiles, 512 KiB of bloom picture).
const POOL_BYTES: u32 = 768 * 1024;
/// The slots' quads (4 vertices each), GPU-readable main memory.
const QUAD_VERTICES: usize = MAX_CLASSES * MAX_SLOTS * 4;
const QUAD_BYTES: u32 = sys::pages(QUAD_VERTICES * core::mem::size_of::<TextureVertex>());
#[no_mangle]
#[used]
#[allow(non_upper_case_globals)]
pub static nativeres_plan_fixed_wide_copy_quads: u32 = QUAD_BYTES;

/// What a slot holds: the part of off-screen target `address` from PSP x
/// `x` on, alpha as stored or 1 (`opaque`: textures that don't read it).
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Key {
    pub address: u32,
    pub x: u32,
    pub opaque: bool,
}

struct Slot {
    /// None: free.
    key: Option<Key>,
    /// `render_count` of the frame that last used it.
    last_used: u32,
    /// The target's `generation` it holds; 0: nothing yet.
    copied: u32,
    /// The target's size its quad's texture coordinates are for.
    source: (u32, u32),
    /// Its rows of the class surface as a U16x4 texture.
    texture: TextureObject,
}

const FREE_SLOT: Slot = Slot { key: None, last_used: 0, copied: 0, source: (0, 0), texture: TextureObject::view(NO_TEXTURE) };
const NO_TEXTURE: gxm::Texture = gxm::Texture { control_words: [0; 4] };

struct Class {
    /// Texels of each slot's texture; 0: class unused.
    width: u32,
    height: u32,
    slot_count: usize,
    surface: gxm::ColorSurface,
    render_target: *mut gxm::RenderTarget,
    slots: [Slot; MAX_SLOTS],
}

const UNUSED_CLASS: Class = Class {
    width: 0,
    height: 0,
    slot_count: 0,
    surface: gxm::ColorSurface { words: [0; 12] },
    render_target: null_mut(),
    slots: [FREE_SLOT; MAX_SLOTS],
};

struct WideCopies {
    /// `MAX_READS`; render_count 0: unused.
    reads: [(Key, u32); MAX_READS],
    /// The quads and the program were tried (made or not).
    tried: bool,
    /// The pool (null: none yet), and the frame that last asked for it.
    pool: *mut u8,
    pool_asked: u32,
    pool_used: u32,
    quads: *mut TextureVertex,
    /// nativeres_widecopy_f with the basic texture vertex program.
    program: *mut gxm::FragmentProgram,
    classes: [Class; MAX_CLASSES],
    /// Log lines left (no room, failures).
    logs: u32,
}

static WIDE: SingleThreaded<WideCopies> = SingleThreaded::new(WideCopies {
    reads: [(Key { address: 0, x: 0, opaque: false }, 0); MAX_READS],
    tried: false,
    pool: null_mut(),
    pool_asked: 0,
    pool_used: 0,
    quads: null_mut(),
    program: null_mut(),
    classes: [UNUSED_CLASS; MAX_CLASSES],
    logs: 8,
});

fn wide() -> &'static mut WideCopies {
    // Safe: display thread only; each entry point takes it once.
    unsafe { WIDE.get() }
}

fn log(copies: &mut WideCopies, text: core::fmt::Arguments) {
    if copies.logs > 0 {
        copies.logs -= 1;
        render_log::line(text);
    }
}

/// True if a texture of this size from a target `target_size` (width,
/// height) big, from PSP x `x` on, can have a wide copy.
pub fn fits(width: u32, height: u32, x: u32, target_size: (u32, u32)) -> bool {
    let side = |n: u32| n.is_power_of_two() && (MIN_SIDE..=MAX_SIDE).contains(&n);
    side(width) && side(height) && x + width <= target_size.0 && height <= target_size.1
}

/// Makes the quads' memory and the program (once) and takes the pool (at
/// most one try a frame, `render_count`); false if they aren't there.
fn ready(copies: &mut WideCopies, render_count: u32) -> bool {
    if !copies.tried {
        copies.tried = true;
        let Some(quads) = gpu_memory::gpu_input(b"NativeResWideQuads\0", QUAD_BYTES) else {
            log(copies, format_args!("wide copies: no memory for the quads\n"));
            return false;
        };
        copies.quads = quads.base as *mut TextureVertex;
        let program = basic_programs::gxp(&shaders::nativeres_widecopy_f);
        copies.program = match basic_programs::register(program) {
            Some(id) => basic_programs::fragment_program(id, null(), basic_programs::gxp(&shaders::nativeres_texture_v)),
            None => null_mut(),
        };
        if copies.program.is_null() {
            log(copies, format_args!("wide copies: program not made\n"));
        }
    }
    if copies.quads.is_null() || copies.program.is_null() {
        return false;
    }
    if copies.pool.is_null() && copies.pool_asked != render_count {
        copies.pool_asked = render_count;
        match textures::reserve_arena(POOL_BYTES) {
            Some(pool) => {
                copies.pool = pool;
                render_log::line(format_args!("wide copies ready: {} KiB of the texture arena\n", POOL_BYTES / 1024));
            }
            None => log(copies, format_args!("wide copies: no {} KiB free in the texture arena now\n", POOL_BYTES / 1024)),
        }
    }
    !copies.pool.is_null()
}

/// The class for `width` x `height` textures, made in the pool if new; None
/// if there is no room.
fn class_for(copies: &mut WideCopies, width: u32, height: u32) -> Option<usize> {
    if let Some(index) = copies.classes.iter().position(|class| class.width == width && class.height == height) {
        return Some(index);
    }
    let index = copies.classes.iter().position(|class| class.width == 0)?;
    let slot_bytes = width * height * TEXEL_BYTES;
    let slot_count = (CLASS_BYTES / slot_bytes).clamp(1, MAX_SLOTS as u32);
    let bytes = slot_bytes * slot_count;
    let start = copies.pool_used.next_multiple_of(4096);
    if start + bytes > POOL_BYTES {
        log(copies, format_args!("wide copies: no room for {}x{} ({} KiB used)\n", width, height, copies.pool_used / 1024));
        return None;
    }
    // Safe: inside the pool.
    let base = unsafe { copies.pool.add(start as usize) };
    let (surface_width, surface_height) = (2 * width, height * slot_count);
    let class = &mut copies.classes[index];
    let mut render_target = null_mut();
    // Safe: our pool's memory, valid parameters and out pointers.
    let made = unsafe {
        gxm::sceGxmColorSurfaceInit(
            &mut class.surface,
            gxm::COLOR_FORMAT_A8B8G8R8,
            gxm::COLOR_SURFACE_LINEAR,
            gxm::COLOR_SURFACE_SCALE_NONE,
            gxm::OUTPUT_REGISTER_SIZE_32BIT,
            surface_width,
            surface_height,
            surface_width,
            base as *mut _,
        )
    };
    let result = if made < 0 { made } else { targets::make_render_target(surface_width, surface_height, 1, &mut render_target) };
    if result < 0 {
        log(copies, format_args!("wide copies: {}x{} surface not made: {:08x}\n", surface_width, surface_height, result));
        return None;
    }
    let class = &mut copies.classes[index];
    class.width = width;
    class.height = height;
    class.slot_count = slot_count as usize;
    class.render_target = render_target;
    for (s, slot) in class.slots.iter_mut().enumerate().take(slot_count as usize) {
        *slot = FREE_SLOT;
        // Safe: slot `s`'s rows of the pool.
        unsafe {
            let data = base.add((s as u32 * slot_bytes) as usize);
            gxm::sceGxmTextureInitLinear(&mut slot.texture.gxm, data as *const _, gxm::TEXTURE_FORMAT_U16U16U16U16_ABGR, width, height, 0);
        }
    }
    copies.pool_used = start + bytes;
    render_log::line(format_args!("wide copies: class {}x{}, {} slots, {} KiB\n", width, height, slot_count, bytes / 1024));
    Some(index)
}

/// Writes slot `s` of class `c`'s quad: its rows of the surface (in our
/// 960x544 drawing space, the viewport set to the surface in `flush`),
/// texture coordinates over its part of a `source` (width, height) target.
/// The GPU is drained first if the slot was drawn before (an earlier copy
/// scene may still read the old quad).
fn write_quad(copies: &mut WideCopies, c: usize, s: usize, x: u32, source: (u32, u32), drain: bool) {
    let class = &copies.classes[c];
    let (width, height, count) = (class.width as f32, class.height as f32, class.slot_count as f32);
    let (top, bottom) = (TARGET_HEIGHT as f32 * s as f32 / count, TARGET_HEIGHT as f32 * (s as f32 + 1.0) / count);
    let (left, right) = (x as f32 / source.0 as f32, (x as f32 + width) / source.0 as f32);
    let v = height / source.1 as f32;
    let corners = [(0.0, top, left, 0.0), (TARGET_WIDTH as f32, top, right, 0.0), (0.0, bottom, left, v), (TARGET_WIDTH as f32, bottom, right, v)];
    if drain {
        // Safe: the display thread's context, outside any scene.
        unsafe { gxm::sceGxmFinish(gxm::context()) };
    }
    let first = (c * MAX_SLOTS + s) * 4;
    for (k, &(x, y, u, v)) in corners.iter().enumerate() {
        // Safe: inside the quads' memory (`QUAD_VERTICES`).
        unsafe { copies.quads.add(first + k).write(TextureVertex { x, y, z: 0.5, w: 1.0, u, v }) };
    }
}

/// A draw into the shown framebuffer read texture `key` in frame
/// `render_count` (once per texture and frame): it gets a copy from the
/// next frame on (`texture_for`).
pub fn note_read(key: Key, render_count: u32) {
    let reads = &mut wide().reads;
    let at = reads.iter().position(|&(k, _)| k == key).unwrap_or_else(|| (0..MAX_READS).min_by_key(|&k| reads[k].1).unwrap_or(0));
    reads[at] = (key, render_count.max(1));
}

/// The wide copy for a `width` x `height` texture `key` of a `source`
/// (width, height) target this frame (`render_count`), its slot taken or
/// reused (copied by the next `flush`); None if it can't have one (not read
/// into the shown framebuffer lately, `note_read`; no memory; all slots of
/// its size used this frame). Call `fits` first.
pub fn texture_for(key: Key, width: u32, height: u32, source: (u32, u32), render_count: u32) -> Option<*const TextureObject> {
    let copies = wide();
    let read_lately = copies.reads.iter().any(|&(k, frame)| k == key && frame != 0 && render_count.wrapping_sub(frame) <= READ_FRAMES);
    if !read_lately || !ready(copies, render_count) {
        return None;
    }
    let c = class_for(copies, width, height)?;
    let count = copies.classes[c].slot_count;
    let slots = &copies.classes[c].slots[..count];
    let (s, reused) = match slots.iter().position(|slot| slot.key == Some(key)) {
        Some(s) => (s, true),
        None => {
            let s = slots
                .iter()
                .position(|slot| slot.key.is_none())
                .or_else(|| (0..count).filter(|&s| slots[s].last_used != render_count).min_by_key(|&s| slots[s].last_used))?;
            (s, false)
        }
    };
    let slot = &copies.classes[c].slots[s];
    if !reused || slot.source != source {
        let drain = slot.key.is_some();
        write_quad(copies, c, s, key.x, source, drain);
        let slot = &mut copies.classes[c].slots[s];
        slot.key = Some(key);
        slot.source = source;
        slot.copied = 0;
    }
    let slot = &mut copies.classes[c].slots[s];
    slot.last_used = render_count;
    Some(&slot.texture as *const TextureObject)
}

/// Copies every slot this frame (`render_count`) uses whose target was
/// drawn into since its last copy: one scene per class that has such.
/// Display thread, outside any scene; before each scene drawing into the
/// shown framebuffer.
pub fn flush(targets: &Targets, render_count: u32) {
    let copies = wide();
    if copies.pool.is_null() {
        return;
    }
    let depth = targets::shared_depth();
    let (Some(basic), false) = (basic_programs::get(), depth.is_null()) else {
        return;
    };
    for c in 0..MAX_CLASSES {
        let class = &copies.classes[c];
        if class.width == 0 {
            continue;
        }
        let stale = |slot: &Slot| {
            slot.last_used == render_count
                && slot.key.is_some_and(|key| {
                    targets.find_offscreen(key.address).is_some_and(|t| targets.offscreen[t].generation != slot.copied)
                })
        };
        if !class.slots[..class.slot_count].iter().any(stale) {
            continue;
        }
        let (surface_width, surface_height) = (2 * class.width, class.height * class.slot_count as u32);
        // Safe: the display thread's context; the class's surface and
        // render target, the shared depth surface (960x544 covers it).
        unsafe {
            let context = gxm::context();
            let result = gxm::sceGxmBeginScene(context, 0, class.render_target, null(), null_mut(), null_mut(), &class.surface, depth);
            if result < 0 {
                log(copies, format_args!("wide copies: sceGxmBeginScene failed: {:08x}\n", result));
                draw::forget_gxm_state();
                return;
            }
            let (half_width, half_height) = (surface_width as f32 / 2.0, surface_height as f32 / 2.0);
            gxm::sceGxmSetViewport(context, half_width, half_width, half_height, -half_height, 0.5, 0.5);
            gxm::sceGxmSetVertexProgram(context, basic.texture_vertex);
            gxm::sceGxmSetFragmentProgram(context, copies.program);
            let mut vertex_uniforms = null_mut();
            gxm::sceGxmReserveVertexDefaultUniformBuffer(context, &mut vertex_uniforms);
            gxm::sceGxmSetUniformDataF(vertex_uniforms, basic.texture_wvp, 0, 16, basic_programs::ORTHO.as_ptr());
            let indices = targets::linear_indices();
            for s in 0..class.slot_count {
                let slot = &copies.classes[c].slots[s];
                if !stale(slot) {
                    continue;
                }
                let Some(key) = slot.key else { continue };
                let Some(t) = targets.find_offscreen(key.address) else { continue };
                let target = &targets.offscreen[t];
                let mut source = target.texture.gxm;
                if key.opaque {
                    gxm::sceGxmTextureSetFormat(&mut source, gxm::TEXTURE_FORMAT_X8U8U8U8_1BGR);
                }
                gxm::sceGxmTextureSetMinFilter(&mut source, gxm::TEXTURE_FILTER_POINT);
                gxm::sceGxmTextureSetMagFilter(&mut source, gxm::TEXTURE_FILTER_POINT);
                gxm::sceGxmSetFragmentTexture(context, 0, &source);
                gxm::sceGxmSetVertexStream(context, 0, copies.quads.add((c * MAX_SLOTS + s) * 4) as *const _);
                gxm::sceGxmDraw(context, gxm::PRIMITIVE_TRIANGLE_STRIP, gxm::INDEX_FORMAT_U16, indices as *const _, 4);
                copies.classes[c].slots[s].copied = target.generation;
                stats().wide_copies += 1;
            }
            gxm::sceGxmEndScene(context, null(), null());
        }
        stats().wide_copy_scenes += 1;
        draw::forget_gxm_state();
        // Back to our targets' drawing space for whatever draws next.
        draw::set_target_size(TARGET_WIDTH, TARGET_HEIGHT, TARGET_WIDTH, TARGET_HEIGHT, 0);
    }
}
