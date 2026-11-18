//! Drawing a recorded frame's draws with GXM, on the display thread: each
//! PSP draw becomes one sceGxmDraw of plain triangles into the current
//! target scene (the scene and the targets: frame_loop.rs and targets.rs,
//! which call `draw_state` and `draw_textured` / `draw_colored` per draw).
//!
//! - textured draws: our programs (programs.rs), from the frame's vertices
//!   (copied into the frame pool), the vertex cache, the raw arena or PSP
//!   RAM itself (zero copy);
//! - untextured draws: the basic colour program, per-vertex colour.
//!
//! PSP coordinates are simply doubled: the PSP screen is 480x272.
//!
//! GXM state is cached (`GxmState`): setting programs, matrix, texture and
//! scissor again for every draw was most of the cost of Final Fantasy's
//! menu (~830 draws, mostly glyphs that differ only in their tint). GXM
//! keeps programs and textures set, and a reserved default uniform buffer
//! is used by all following draws until the next reservation. Forgotten
//! (`forget_gxm_state`) when a scene begins, since other code draws
//! in between.

use core::ffi::c_void;
use core::ptr::{null, null_mut};
use core::sync::atomic::{AtomicI32, AtomicU32, AtomicU8, Ordering};

use nativeres_core::blend::{BLEND_ALPHA_ONLY, BLEND_ENABLED, BLEND_FIX_B_ALPHA, BLEND_WRITES_ALPHA};
use nativeres_core::frame::{
    draw_flags, read_variant, test_flags, Draw, Frame, FrameVertex, GpuLights, GpuTransform, RawCamera, RawColors, RawTransform, ReadVariant,
    ReadWay, NO_RING_INDICES,
};
use nativeres_core::texture;
use nativeres_core::vertex::VertexFormat;

use crate::arenas;
use crate::capture::{RAW_ARENA, RAW_ARENA_BYTES, VERTEX_ARENA, VERTEX_ARENA_VERTICES};
use crate::frame_pool::alloc as frame_pool_alloc;
use crate::gpu_textures::TextureObject;
use crate::gxm::{self, ProgramParameter};
use crate::programs::{Kind, Programs, RawUniforms, RawVariant, SPRITE_CORNERS};
use crate::render_stats::stats;
use crate::{basic_programs, render_log, sys, SingleThreaded};

/// PSP screen pixels to Vita screen pixels.
const SCALE: f32 = 2.0;
/// Our own passes (unswizzle, target copies, presenting): no blending, all
/// four channels written as they are.
const COPY_BLEND: u32 = BLEND_WRITES_ALPHA;
/// PSP RAM's first address as the GPU sees it (zero copy).
const PSP_RAM_GPU_BASE: u32 = 0x0800_0000;
/// The PSP's primitive types in index runs (frame.rs run_indices).
const PRIM_TRIANGLES: u16 = 3;
const PRIM_TRIANGLE_STRIP: u16 = 4;
const PRIM_SPRITES: u16 = 6;

/// Parts of the per-draw work left out, to measure
/// each one's CPU cost. 0 = draw normally. Bits: 1 no vertex stream /
/// sceGxmDraw, 2 no programs / uniforms, 4 no texture state, 8 stop right
/// after the indices, 16 no draw function (scissor/depth/cull only), 32
/// nothing per draw (the last two in frame_loop.rs). Only set while the
/// bench runs (pictures are wrong then).
pub static REDRAW_SKIP: AtomicU32 = AtomicU32::new(0);
/// Frame draws GXM refused (sceGxmDraw < 0) and the last refusal's code
/// (debug "drawerrors"; the first few are also logged).
pub static DRAW_ERRORS: AtomicU32 = AtomicU32::new(0);
pub static LAST_DRAW_ERROR: AtomicU32 = AtomicU32::new(0);
/// Debug "nodepth 1": every draw without the depth test (diagnosis).
pub static NO_DEPTH: AtomicI32 = AtomicI32::new(0);

/// Counts and logs a refused frame draw.
fn note_draw_result(result: i32) {
    if result >= 0 {
        return;
    }
    let count = DRAW_ERRORS.fetch_add(1, Ordering::Relaxed);
    LAST_DRAW_ERROR.store(result as u32, Ordering::Relaxed);
    if count < 8 {
        render_log::line(format_args!("sceGxmDraw refused a draw: {:08x}
", result));
    }
}

/// Draw checks (off by default since 2026-11-10, after the frame pool
/// ended the GPU crashes: ~0.6-0.8 ms of display CPU per frame). Before each textured draw goes to the GPU, the bytes its
/// vertex streams would be read from (first vertex + highest index x
/// stride, plus one vertex) must lie in the memory mapped for that kind of
/// draw: PSP RAM (zero copy), the raw arena, the vertex cache arena, or the
/// vertices just copied into the frame pool; the sprite corner table too. A
/// draw outside is skipped and logged (render log "drawcheck: ..."). Made
/// after a GPU crash (2026-11-09): a GPU reading out of bounds was one
/// suspect.
pub static DRAW_CHECKS: AtomicI32 = AtomicI32::new(0);
pub static DRAW_CHECK_FAILURES: AtomicU32 = AtomicU32::new(0);

/// 1 (default); 0, for measuring only, draws
/// alpha-tested draws with the plain textured program (no discard; wrong
/// picture), to see what the discard costs the GPU.
pub static ALPHA_SHADER: AtomicI32 = AtomicI32::new(1);

/// Per-draw timing (render statistics convert_us / indices_us), off by
/// default: one sceKernelGetProcessTimeWide costs ~2.4 us on
/// the Vita, and the draw loop took four per textured draw (~5 ms per MTX
/// race frame went to measuring). Off, those two figures stay 0.
pub static DRAW_TIMING: AtomicI32 = AtomicI32::new(0);

/// Back-face culling off, on (default), or with the
/// windings swapped (to check which GXM winding is which).
pub static CULL_MODE: AtomicU8 = AtomicU8::new(1);

/// For the next drawn frame, the draws that still get the
/// discard program, counted by their tests and blend words (render log).
pub static ALPHA_LOG: AtomicI32 = AtomicI32::new(0);

/// Raw draws neither skinned nor lit with lights get the plain raw program:
/// the full program costs every vertex its skinning and lighting code,
/// switched off by uniforms or not (GPU time). 1 = on (default).
pub static RAW_PLAIN: AtomicI32 = AtomicI32::new(1);

/// Raw draws lit with lights or skinned get the smallest program that takes
/// them: one bone with lights, up to four bones with lights, else the full
/// program (nativeres_raw_v.cg's variants; 2026-11-17, MTX's race start with
/// ~290 small rider draws: GPU 24.8 -> 22.6 ms, picture identical). 0 = only
/// the plain and the full program, as before. 1 = on (default).
pub static RAW_VARIANTS: AtomicI32 = AtomicI32::new(1);

/// Lit one- and four-bone draws whose light set is flat (`flat_lights`) get
/// the flat programs (`RawVariant::Lit1Flat` / `Lit4Flat`: no pow, three
/// lights; the same picture). 1 = on (default), 0 = the general ones.
pub static RAW_FLAT_LIGHTS: AtomicI32 = AtomicI32::new(1);

/// Draws into the PSP's 16-bit framebuffers (`draw_flags::
/// TARGET_FORMAT_SHIFT`, Monster Hunter Freedom) get the frame-read
/// programs, which blend themselves and cut the colour to the target's bits
/// as the GE does (2026-11-17: MHF's guild hall was ~7 levels brighter and
/// greyer than Sony's: faint additive glow passes the PSP's 5 bits drop).
/// 1 = on (default: the fast programs where they fit, core frame.rs
/// `read_variant`), 0 = blended in 8 bits as before; 2 = measuring only:
/// the generic frame-read program writes the colour unblended and uncut;
/// 3 = the generic program for every such draw (to compare pictures);
/// 4 = normal alpha blends without the frame read (`ReadWay::Precut`);
/// measuring: 5 = only unblended draws cut, 6 = only blended ones.
pub static TARGET_CUT: AtomicI32 = AtomicI32::new(1);

/// Untextured normal alpha blends into 5551 whose vertex alphas are all 0
/// get the "invisible" frame-read program (core frame.rs
/// `ReadWay::Invisible`: the target dithered and cut again, no colour
/// maths; the same picture). 1 = on (default), 0 = the normal one.
pub static INVISIBLE_SOURCES: AtomicI32 = AtomicI32::new(1);

/// A GXM blend factor (core blend.rs `Factor` numbers) as the frame-read
/// program's coefficients: k0 + k1 s + k2 s.a + k3 d + k4 d.a, returned as
/// (k0, [k1, k2, k3, k4]).
fn factor_coefficients(factor: u32) -> (f32, [f32; 4]) {
    match factor {
        1 => (1.0, [0.0, 0.0, 0.0, 0.0]),
        2 => (0.0, [1.0, 0.0, 0.0, 0.0]),
        3 => (1.0, [-1.0, 0.0, 0.0, 0.0]),
        4 => (0.0, [0.0, 1.0, 0.0, 0.0]),
        5 => (1.0, [0.0, -1.0, 0.0, 0.0]),
        6 => (0.0, [0.0, 0.0, 1.0, 0.0]),
        7 => (1.0, [0.0, 0.0, -1.0, 0.0]),
        8 => (0.0, [0.0, 0.0, 0.0, 1.0]),
        9 => (1.0, [0.0, 0.0, 0.0, -1.0]),
        _ => (0.0, [0.0; 4]),
    }
}

/// The frame-read program's blend and cut uniforms (uSource, uDestination,
/// uBlend, uMinMax, uStep, uWiden, uWidenLow, uWrite, uPath; see
/// nativeres_textured_read_f.cg) for packed `blend` (`blending`: on, not
/// alpha only), the target's channel `bits`, which channels it writes and
/// whether it is dithered.
fn cut_uniforms(blend: u32, blending: bool, bits: [u32; 4], writes_color: bool, writes_alpha: bool, dithered: bool) -> [[f32; 4]; 9] {
    let (source, destination, equation) = if blending { (blend & 0xF, (blend >> 4) & 0xF, (blend >> 8) & 7) } else { (1, 0, 0) };
    let (source_constant, source_terms) = factor_coefficients(source);
    let (destination_constant, destination_terms) = factor_coefficients(destination);
    let (source_sign, destination_sign) = match equation {
        1 => (1.0, -1.0),
        2 => (-1.0, 1.0),
        _ => (1.0, 1.0),
    };
    let min_max = match equation {
        3 => [1.0, 0.0, 0.0, 0.0],
        4 => [1.0, 1.0, 0.0, 0.0],
        _ => [0.0; 4],
    };
    // Per channel: 1 / step, the step, the repeated top bits' factor.
    let step = bits.map(|bits| if bits == 0 { 1.0 } else { (1u32 << bits) as f32 / 256.0 });
    let widen = bits.map(|bits| match bits {
        0 => 1.0,
        1 => 255.0,
        _ => (256 >> bits) as f32,
    });
    let widen_low = bits.map(|bits| if bits <= 1 { 0.0 } else { (256 >> bits) as f32 / (1u32 << bits) as f32 });
    let color = if writes_color { 1.0 } else { 0.0 };
    let write = [color, color, color, if writes_alpha { 1.0 } else { 0.0 }];
    // The shader's short ways: normal alpha blending, additive, none.
    let way = match (blending, source, destination, equation) {
        (false, ..) => 2.0,
        (true, 4, 5, 0) => 0.0,
        (true, 4, 1, 0) => 1.0,
        _ => 3.0,
    };
    let measuring = if TARGET_CUT.load(Ordering::Relaxed) == 2 { 1.0 } else { 0.0 };
    let path = [way, if dithered { 1.0 } else { 0.0 }, measuring, 0.0];
    [
        source_terms,
        destination_terms,
        [source_constant, destination_constant, source_sign, destination_sign],
        min_max,
        step,
        widen,
        widen_low,
        write,
        path,
    ]
}

/// The alpha-test uniform that lets every fragment pass (the frame-read
/// program always runs its alpha test): 0..255 pass.
const ALPHA_ALL_PASS: [f32; 4] = alpha_test_uniform(0.0, 255.0, false);

/// The fragment programs' alpha-test uniform (uAlphaTest,
/// nativeres_textured_test_f.cg) for alpha values `lo..=hi` passing, or with
/// `outside` the ones outside: the bounds half a step out, z the sign that
/// makes a passing alpha's (alpha - x)(y - alpha) z positive.
const fn alpha_test_uniform(lo: f32, hi: f32, outside: bool) -> [f32; 4] {
    [lo - 0.5, hi + 0.5, if outside { -1.0 } else { 1.0 }, 0.0]
}

/// The frame-read program for `draw` if it goes to a 16-bit PSP target
/// whose colour is cut as the GE does (`TARGET_CUT`), else None (the 8-bit
/// way); `keep_target_alpha` as `Draws::keep_target_alpha`.
fn read_choice(frame: &Frame, draw: &Draw, keep_target_alpha: bool) -> Option<ReadVariant> {
    let mode = TARGET_CUT.load(Ordering::Relaxed);
    if draw.flags & draw_flags::TARGET_FORMAT_MASK == 0 || mode == 0 {
        return None;
    }
    // The fast programs leave the separate specular colour out.
    if mode == 2 || mode == 3 || separate_specular(frame, draw) {
        return Some(ReadVariant::Generic);
    }
    let variant = read_variant(draw.flags, draw.blend, keep_target_alpha, mode != 4);
    let off = matches!(variant, ReadVariant::Fast { way: ReadWay::Off, .. });
    match mode {
        5 if !off => None,
        6 if off => None,
        _ => Some(variant),
    }
}

/// The alpha values the PSP's alpha test `function` (its 3 bits) with
/// `reference` (0..255) lets pass: `lo..=hi`, or with `outside` the ones
/// outside that range; None for "always".
fn alpha_test_bounds(function: u32, reference: u32) -> Option<(f32, f32, bool)> {
    let reference = reference as f32;
    Some(match function & 7 {
        0 => (1.0, 0.0, false),               // never
        1 => return None,                     // always
        2 => (reference, reference, false),   // equal
        3 => (reference, reference, true),    // not equal
        4 => (0.0, reference - 1.0, false),   // less
        5 => (0.0, reference, false),         // less or equal
        6 => (reference + 1.0, 255.0, false), // greater
        _ => (reference, 255.0, false),       // greater or equal
    })
}

/// Whether `draw` is lit with a separate specular colour (LIGHTMODE 1):
/// its vertex program then hands the fragment program a specular part
/// (vSpecular), which the fast frame-read programs leave out (NO_SPECULAR,
/// nativeres_textured_read_f.cg). Draws the GPU doesn't light have none.
fn separate_specular(frame: &Frame, draw: &Draw) -> bool {
    let lights = if draw.flags & draw_flags::RAW_VERTICES != 0 {
        frame.raw_transform_of(draw).and_then(|(transform, _)| raw_light_set(frame, transform))
    } else if draw.transform != 0 && draw.transform <= frame.transform_count {
        frame.transforms.get(draw.transform as usize - 1).and_then(|transform| light_set(frame, transform))
    } else {
        None
    };
    lights.is_some_and(|lights| lights.material_specular[3] > 0.5)
}

/// Measuring only (0 = off): our scene draws only
/// its top <rows> pixel rows, to see how the GPU time scales with the area
/// drawn.
pub static GPU_CLIP_ROWS: AtomicI32 = AtomicI32::new(0);

/// The GPU tests over the next frames: see `draw_index_test` and
/// `draw_mip_test`.
pub static INDEX_TEST_FRAMES: AtomicI32 = AtomicI32::new(0);
pub static MIP_TEST_FRAMES: AtomicI32 = AtomicI32::new(0);
/// "shadetest <frames>": `draw_shade_test` after each of the next frames.
pub static SHADE_TEST_FRAMES: AtomicI32 = AtomicI32::new(0);

/// Which of our programs a shade-test quad goes through.
#[derive(Clone, Copy)]
enum ShadeProgram {
    /// The 8-bit ones (nativeres_textured_f.cg, with the test _test_f).
    EightBit,
    /// The fast 5551 "off" ones (nativeres_textured_read_off_f.cg, with the
    /// test _offtest_f): the colour cut to 5 bits per channel, widened.
    Cut5551,
}

/// The shade test's alpha-test cases: PSP compare function and reference,
/// over a quad of alpha 200 (`SHADE_ALPHA_CASE`). tools/shadetest.py knows
/// which pass: greater or equal 200 yes, greater 200 no, not equal 200
/// no, not equal 100 yes, equal 200 yes, less 200 no.
const SHADE_ALPHA_TESTS: [(u32, u32); 6] = [(7, 200), (6, 200), (3, 200), (3, 100), (2, 200), (4, 200)];
/// White at alpha 200: the texel's colour where it passes.
const SHADE_ALPHA_CASE: ShadeCase =
    ShadeCase { w: [1.0, 1.0], color: [0xC8FF_FFFF; 2], fog: 1.0, fog_and_scale: [0.0, 0.0, 0.0, 1.0], tested: true };

/// One quad of the shade test: its left and right corners' w and colour
/// (0xAABBGGRR), the fog factor at every corner, the fragment program's
/// fog colour and colour scale, and whether it goes through the alpha-test
/// program. tools/shadetest.py has the same table and the expected pixels.
struct ShadeCase {
    w: [f32; 2],
    color: [u32; 2],
    fog: f32,
    fog_and_scale: [f32; 4],
    tested: bool,
}

/// The shade test's texel: one colour over the whole texture, so filtering
/// can't change it (r 200, g 120, b 60, a 255).
const SHADE_TEST_TEXEL: u32 = 0xFF3C_78C8;
const SHADE_FOG: [f32; 4] = [127.0, 96.0, 79.0, 1.0];
const SHADE_TESTS: [ShadeCase; 7] = [
    // 2D (w 1), no fog.
    ShadeCase { w: [1.0, 1.0], color: [0xFF32_6496; 2], fog: 1.0, fog_and_scale: [0.0, 0.0, 0.0, 1.0], tested: false },
    // 3D, the same w at every corner.
    ShadeCase { w: [4.0, 4.0], color: [0xFF32_6496; 2], fog: 1.0, fog_and_scale: [0.0, 0.0, 0.0, 1.0], tested: false },
    // 3D, w 1 .. 8 across, half fog.
    ShadeCase { w: [1.0, 8.0], color: [0xFF32_6496; 2], fog: 0.5, fog_and_scale: SHADE_FOG, tested: false },
    // ... with colour doubling.
    ShadeCase { w: [1.0, 8.0], color: [0xFF32_6496; 2], fog: 0.25, fog_and_scale: [SHADE_FOG[0], SHADE_FOG[1], SHADE_FOG[2], 2.0], tested: false },
    // ... through the alpha-test program (alpha 200, all pass).
    ShadeCase { w: [1.0, 8.0], color: [0xC832_6496; 2], fog: 0.75, fog_and_scale: SHADE_FOG, tested: true },
    // Black to white across, w 1 .. 8: screen-linear, the middle is grey
    // (with perspective it would be near black).
    ShadeCase { w: [1.0, 8.0], color: [0xFF00_0000, 0xFFFF_FFFF], fog: 1.0, fog_and_scale: [0.0, 0.0, 0.0, 1.0], tested: false },
    // MHF's ground: fogged, w 2 .. 6, through the alpha-test program.
    ShadeCase { w: [2.0, 6.0], color: [0xFFAA_BBCC; 2], fog: 0.4, fog_and_scale: SHADE_FOG, tested: true },
];
/// The raw lod_bias field for the mip test (-1: as sceGxmTextureInitLinear
/// leaves it), to learn its scale.
pub static MIP_TEST_BIAS: AtomicI32 = AtomicI32::new(-1);
static MIP_TEST_DEFAULT_BIAS: AtomicU32 = AtomicU32::new(0);

fn smooth_textures() -> bool {
    crate::frame_loop::SMOOTH_TEXTURES.load(Ordering::Relaxed) != 0
}

fn skip(bit: u32) -> bool {
    REDRAW_SKIP.load(Ordering::Relaxed) & bit != 0
}

/// The clock for per-draw timing, or 0 when it is off.
fn draw_clock() -> u64 {
    if DRAW_TIMING.load(Ordering::Relaxed) != 0 {
        sys::now_us()
    } else {
        0
    }
}

/// GXM state as last set in the current scene.
struct GxmState {
    vertex_program: *const gxm::VertexProgram,
    fragment_program: *const gxm::FragmentProgram,
    /// The fragment colour uniform (`use_fragment_color`, as bits), for the
    /// parameter it was set for.
    tint: Option<(*const ProgramParameter, [u32; 4])>,
    texture: Option<gxm::Texture>,
    scissor: Option<[u32; 4]>,
    depth: Option<(u32, u32)>,
    cull: Option<u32>,
    /// The transform and texture coordinate scale the 3D / raw programs'
    /// uniforms were set for.
    transform: *const GpuTransform,
    uv_scale: [f32; 2],
    /// The object (with its camera) the raw programs' uniforms were set for.
    raw_transform: *const RawTransform,
    /// The colour rules the raw programs' uniforms were set for.
    raw_colors: *const RawColors,
    /// The bones the raw program's uniforms were set for (`Draw::bones`; 0:
    /// none, u32::MAX: unknown).
    bones: u32,
    /// The viewport's z offset and scale while raw draws set it; None: GXM's
    /// default.
    raw_viewport: Option<[f32; 2]>,
    /// The dither matrix whose texture is on texture unit 1
    /// (`use_dither_texture`).
    dither_matrix: Option<u64>,
}

const FORGOTTEN: GxmState = GxmState {
    vertex_program: null(),
    fragment_program: null(),
    tint: None,
    texture: None,
    scissor: None,
    depth: None,
    cull: None,
    transform: null(),
    uv_scale: [0.0; 2],
    raw_transform: null(),
    raw_colors: null(),
    bones: 0,
    raw_viewport: None,
    dither_matrix: None,
};

const ALPHA_LOG_KINDS: usize = 32;

/// The display thread's drawing state: our programs and the GXM state as
/// last set.
pub struct Draws {
    pub programs: Programs,
    state: GxmState,
    /// This frame's decision: draws keep the target's alpha (the PSP's
    /// stencil) unless they write it.
    pub keep_target_alpha: bool,
    /// The alpha-test program's range uniform as last set (bits).
    alpha_test_cache: [u32; 4],
    /// The alpha log (`ALPHA_LOG`): (tests, blend, count).
    alpha_log: [(u32, u32, u32); ALPHA_LOG_KINDS],
    alpha_log_kinds: usize,
    /// The 960x544 projection with the doubling and the depth folded in
    /// (`native_matrix`); None until made.
    native_matrix: Option<[f32; 16]>,
    /// The bones of the raw program's uniform (24 rows of 4).
    bones: [f32; 24 * 4],
    /// The mip/index test block (`test_block`); null until made.
    test_block: *mut u8,
    test_block_tried: bool,
}

static DRAWS: SingleThreaded<Draws> = SingleThreaded::new(Draws {
    programs: Programs::new(),
    state: FORGOTTEN,
    keep_target_alpha: false,
    alpha_test_cache: [0; 4],
    alpha_log: [(0, 0, 0); ALPHA_LOG_KINDS],
    alpha_log_kinds: 0,
    native_matrix: None,
    bones: [0.0; 24 * 4],
    test_block: null_mut(),
    test_block_tried: false,
});

/// The drawing state. Display thread only (every entry point below, and
/// whatever an extension draws on that thread); no reference kept across
/// calls into this module.
pub fn draws() -> &'static mut Draws {
    // Safe: display thread only, see above.
    unsafe { DRAWS.get() }
}

fn context() -> *mut gxm::Context {
    // Safe: the display thread.
    unsafe { gxm::context() }
}

/// Four words equal, compared one by one like C's memcmp. A plain `==` on
/// 16 bytes became a NEON compare whose result went back to ARM registers
/// lane by lane, which is slow on the Cortex-A9 (and floats compared as
/// floats need a VFP flag transfer each); this is per draw. Even written as
/// four compares LLVM merged them into NEON again: the cached side is read
/// with volatile reads, which it can't merge.
#[inline]
fn same_words(cached: &[u32; 4], b: &[u32; 4]) -> bool {
    // Safe: plain reads of a valid array.
    let word = |index: usize| unsafe { core::ptr::read_volatile(&cached[index]) };
    word(0) == b[0] && word(1) == b[1] && word(2) == b[2] && word(3) == b[3]
}

/// `values` as their bit patterns, for `same_words`.
#[inline]
fn bits(values: &[f32; 4]) -> [u32; 4] {
    [values[0].to_bits(), values[1].to_bits(), values[2].to_bits(), values[3].to_bits()]
}

/// The float components of `value` (a struct of f32s or arrays of them).
fn floats<T>(value: &T) -> *const f32 {
    value as *const T as *const f32
}

/// Sets `count` floats from `data` into uniform buffer `buffer`.
pub fn set_uniform(buffer: *mut c_void, parameter: *const ProgramParameter, count: u32, data: *const f32) {
    // Safe: a reserved buffer, a parameter of the program in use, `count`
    // floats readable at `data`.
    unsafe { gxm::sceGxmSetUniformDataF(buffer, parameter, 0, count, data) };
}

/// A fresh default vertex uniform buffer for the next draw.
pub fn reserve_vertex_uniforms() -> *mut c_void {
    let mut buffer = null_mut();
    // Safe: the display thread's context.
    unsafe { gxm::sceGxmReserveVertexDefaultUniformBuffer(context(), &mut buffer) };
    buffer
}

fn reserve_fragment_uniforms() -> *mut c_void {
    let mut buffer = null_mut();
    // Safe: the display thread's context.
    unsafe { gxm::sceGxmReserveFragmentDefaultUniformBuffer(context(), &mut buffer) };
    buffer
}

/// The program's colour scale and fog colour: rgb the fog colour as the
/// GE's 8-bit values (0..255, as `GeShade` computes with them), a colour
/// scale 2 with colour doubling (the shader clamps the result, like the PSP).
fn fog_and_scale(draw: &Draw) -> [f32; 4] {
    let mut values = [0.0, 0.0, 0.0, if draw.flags & draw_flags::DOUBLE_COLOR != 0 { 2.0 } else { 1.0 }];
    if draw.flags & draw_flags::FOG != 0 {
        for (channel, value) in values.iter_mut().take(3).enumerate() {
            *value = ((draw.fog_color >> (8 * channel)) & 0xFF) as f32;
        }
    }
    values
}

/// Whole triangles of an index run of `count` vertices (frame.rs run_triangles).
fn run_triangles(primitive: u16, count: u16) -> u32 {
    match primitive {
        PRIM_TRIANGLES => count as u32 / 3,
        PRIM_SPRITES => 2 * count as u32,
        _ => (count as u32).saturating_sub(2),
    }
}

/// The PSP's depth compare functions (ZTEST register) in its order: never,
/// always, equal, not equal, less, less or equal, greater, greater or equal.
const DEPTH_FUNCTIONS: [u32; 8] = [
    gxm::DEPTH_FUNC_NEVER,
    gxm::DEPTH_FUNC_ALWAYS,
    gxm::DEPTH_FUNC_EQUAL,
    gxm::DEPTH_FUNC_NOT_EQUAL,
    gxm::DEPTH_FUNC_LESS,
    gxm::DEPTH_FUNC_LESS_EQUAL,
    gxm::DEPTH_FUNC_GREATER,
    gxm::DEPTH_FUNC_GREATER_EQUAL,
];

/// True if a fragment with alpha 0 leaves the target as it was: no depth
/// (or stencil) write, and blending that weights the source by its alpha
/// and keeps the destination (factor 1 - source alpha or 1, added; alpha
/// blends the same way). A test that only rejects alpha 0 then needs no
/// discard: the fragments it throws away change nothing. Worth it: the
/// discard program costs the GPU (2026-11-10, MTX race, `ALPHA_SHADER` 0:
/// ~30 ms per frame with it, ~14-18 without); a discarding draw can't
/// use the PowerVR GPU's early hidden-surface removal.
fn alpha_zero_changes_nothing(draw: &Draw) -> bool {
    let writes_depth = draw.tests & test_flags::DEPTH_TEST != 0 && draw.tests & test_flags::DEPTH_WRITE_DISABLED == 0;
    let blend = draw.blend;
    let destination = (blend >> 4) & 0xF;
    !writes_depth
        && draw.tests & test_flags::STENCIL_TEST == 0
        && blend & BLEND_ENABLED != 0
        && (blend >> 8) & 7 == 0 // add
        && blend & 0xF == 4 // source alpha
        && (destination == 5 || destination == 1) // one minus source alpha, one
}

impl Draws {
    fn forget_state(&mut self) {
        self.state = FORGOTTEN;
    }

    /// The matrix of our textured program: the projection (ORTHO) with what
    /// the CPU used to do per vertex folded in, so a frame's vertices reach
    /// the GPU as they are: x, y in PSP pixels (times SCALE) and z = PSP
    /// depth d, turned into (1 - d) / 2. The matrix is row-major and
    /// applied as position x matrix (translation in the last row), so the
    /// scaling multiplies rows 0-2 and the depth's offset adds half of row 2
    /// to row 3.
    pub fn native_matrix(&mut self) -> *const f32 {
        let matrix = self.native_matrix.get_or_insert_with(|| {
            let ortho = &basic_programs::ORTHO;
            let mut matrix = [0.0f32; 16];
            for column in 0..4 {
                matrix[column] = SCALE * ortho[column];
                matrix[4 + column] = SCALE * ortho[4 + column];
                matrix[8 + column] = -0.5 * ortho[8 + column];
                matrix[12 + column] = 0.5 * ortho[8 + column] + ortho[12 + column];
            }
            matrix
        });
        matrix.as_ptr()
    }

    pub fn use_programs(&mut self, vertex: *const c_void, wvp: *const ProgramParameter, matrix: *const f32, fragment: *const c_void) {
        if vertex != self.state.vertex_program {
            // Safe: programs made by the patcher, the display's context.
            unsafe { gxm::sceGxmSetVertexProgram(context(), vertex) };
            set_uniform(reserve_vertex_uniforms(), wvp, 16, matrix);
            self.state.vertex_program = vertex;
        }
        if fragment != self.state.fragment_program {
            // Safe: as above.
            unsafe { gxm::sceGxmSetFragmentProgram(context(), fragment) };
            self.state.fragment_program = fragment;
            self.state.tint = None; // its uniforms are set anew for the new program
        }
    }

    pub fn set_depth(&mut self, function: u32, write: u32) {
        if self.state.depth == Some((function, write)) {
            return;
        }
        // Safe: the display's context.
        unsafe {
            gxm::sceGxmSetFrontDepthFunc(context(), function);
            gxm::sceGxmSetFrontDepthWriteEnable(context(), write);
        }
        self.state.depth = Some((function, write));
    }

    fn set_cull(&mut self, cull: u32) {
        if self.state.cull == Some(cull) {
            return;
        }
        // Safe: the display's context.
        unsafe { gxm::sceGxmSetCullMode(context(), cull) };
        self.state.cull = Some(cull);
    }

    /// A draw's back-face culling (`draw_flags::CULL`). CULL_CCW keeps the
    /// triangles turning clockwise on the screen (y down), so GXM culls the
    /// counter-clockwise ones then.
    fn use_cull(&mut self, draw: &Draw) {
        let mode = CULL_MODE.load(Ordering::Relaxed);
        if draw.flags & draw_flags::CULL == 0 || mode == 0 {
            self.set_cull(gxm::CULL_NONE);
            return;
        }
        let mut keeps_clockwise = draw.flags & draw_flags::CULL_CCW != 0;
        if mode == 2 {
            keeps_clockwise = !keeps_clockwise;
        }
        self.set_cull(if keeps_clockwise { gxm::CULL_CCW } else { gxm::CULL_CW });
    }

    /// A draw's depth test and depth writes. Like on the PSP, depth is
    /// written only by draws with the depth test on (clears: "always").
    fn use_depth(&mut self, draw: &Draw) {
        if draw.tests & test_flags::DEPTH_TEST == 0 || NO_DEPTH.load(Ordering::Relaxed) != 0 {
            self.set_depth(gxm::DEPTH_FUNC_ALWAYS, gxm::DEPTH_WRITE_DISABLED);
            return;
        }
        let write = draw.tests & test_flags::DEPTH_WRITE_DISABLED == 0;
        self.set_depth(
            DEPTH_FUNCTIONS[((draw.tests >> test_flags::DEPTH_FUNCTION_SHIFT) & 7) as usize],
            if write { gxm::DEPTH_WRITE_ENABLED } else { gxm::DEPTH_WRITE_DISABLED },
        );
    }

    /// Limits drawing to the PSP scissor rectangle (inclusive PSP pixels).
    fn set_scissor(&mut self, draw: &Draw) {
        let mut top = draw.scissor_top as u32 * 2;
        let left = draw.scissor_left as u32 * 2;
        let right = (draw.scissor_right as u32 * 2 + 1).min(959);
        let mut bottom = (draw.scissor_bottom as u32 * 2 + 1).min(543);
        let clip = GPU_CLIP_ROWS.load(Ordering::Relaxed);
        if clip > 0 && clip <= 544 && bottom > clip as u32 - 1 {
            bottom = clip as u32 - 1;
            top = top.min(bottom);
        }
        let scissor = [left, top, right, bottom];
        if self.state.scissor == Some(scissor) {
            return;
        }
        // Safe: the display thread's context.
        unsafe { gxm::sceGxmSetRegionClip(context(), gxm::REGION_CLIP_OUTSIDE, left, top, right, bottom) };
        self.state.scissor = Some(scissor);
    }

    /// Sets the fragment program's one colour uniform: the tint of the basic
    /// texture-tint program, or our program's fog colour and colour scale.
    pub fn use_fragment_color(&mut self, parameter: *const ProgramParameter, color: [f32; 4]) {
        let color_bits = bits(&color);
        if self.tint_is(parameter, &color_bits) {
            return;
        }
        set_uniform(reserve_fragment_uniforms(), parameter, 4, color.as_ptr());
        self.state.tint = Some((parameter, color_bits));
    }

    /// True if the fragment colour uniform holds `color_bits` for `parameter`.
    fn tint_is(&self, parameter: *const ProgramParameter, color_bits: &[u32; 4]) -> bool {
        matches!(&self.state.tint, Some((known, known_bits)) if *known == parameter && same_words(known_bits, color_bits))
    }

    fn use_texture(&mut self, texture: &gxm::Texture) {
        if self.state.texture.as_ref().is_some_and(|known| same_words(&known.control_words, &texture.control_words)) {
            return;
        }
        // Safe: GXM copies the texture state.
        unsafe { gxm::sceGxmSetFragmentTexture(context(), 0, texture) };
        self.state.texture = Some(*texture);
    }

    /// Puts the GE's dither matrix `matrix` (`Frame::dither_matrix`) on
    /// texture unit 1 for the fast dithered frame-read programs
    /// (READ_DITHER, uDitherTexture): an 8x8 texture, texel (x, y) = the
    /// matrix entry (x % 4, y % 4) + 8 in red, which they sample at WPOS / 16
    /// (two of ours per PSP pixel), repeated, unfiltered. 8x8 rather than
    /// 4x4: linear texture rows are at least 8 texels. From the frame pool,
    /// once per scene and matrix; if the pool is full the draw reads what
    /// the unit holds (the frame is late anyway).
    fn use_dither_texture(&mut self, matrix: u64) {
        if self.state.dither_matrix == Some(matrix) {
            return;
        }
        let texels = frame_pool_alloc(64 * 4, 64) as *mut u32;
        if texels.is_null() {
            return;
        }
        for y in 0..8 {
            for x in 0..8 {
                let value = (nativeres_core::frame::dither_value(matrix, x % 4, y % 4) + 8) as u32;
                // Safe: the pool gave room for 64 texels.
                unsafe { texels.add(y * 8 + x).write(0xFF00_0000 | value) };
            }
        }
        let mut texture = gxm::Texture { control_words: [0; 4] };
        // Safe: our texture struct over the pool's block; GXM copies the
        // texture state.
        unsafe {
            gxm::sceGxmTextureInitLinear(&mut texture, texels as *const c_void, gxm::TEXTURE_FORMAT_U8U8U8U8_ABGR, 8, 8, 0);
            gxm::sceGxmTextureSetMinFilter(&mut texture, gxm::TEXTURE_FILTER_POINT);
            gxm::sceGxmTextureSetMagFilter(&mut texture, gxm::TEXTURE_FILTER_POINT);
            gxm::sceGxmTextureSetUAddrMode(&mut texture, gxm::TEXTURE_ADDR_REPEAT);
            gxm::sceGxmTextureSetVAddrMode(&mut texture, gxm::TEXTURE_ADDR_REPEAT);
            gxm::sceGxmSetFragmentTexture(context(), 1, &texture);
        }
        self.state.dither_matrix = Some(matrix);
    }

    fn note_discard_draw(&mut self, draw: &Draw) {
        if let Some(kind) = self.alpha_log[..self.alpha_log_kinds].iter_mut().find(|kind| kind.0 == draw.tests && kind.1 == draw.blend) {
            kind.2 += 1;
        } else if self.alpha_log_kinds < ALPHA_LOG_KINDS {
            self.alpha_log[self.alpha_log_kinds] = (draw.tests, draw.blend, 1);
            self.alpha_log_kinds += 1;
        }
    }

    fn write_alpha_log(&mut self) {
        render_log::line(format_args!("alphalog: {} kinds of draws with the discard program\n", self.alpha_log_kinds));
        for &(tests, blend, count) in &self.alpha_log[..self.alpha_log_kinds] {
            render_log::line(format_args!("alphalog: {} draws tests={:08x} blend={:08x}\n", count, tests, blend));
        }
        self.alpha_log_kinds = 0;
    }

    /// The PSP's alpha test of a textured draw as the fragment programs'
    /// uniform (`alpha_test_uniform`: the alpha values that pass, 0..255). None if the draw needs no test
    /// program: no test, "always", a mask other than 0xFF (blending stands in
    /// for those, blend.rs), no test program, a test every alpha passes, or
    /// one that only rejects alpha 0 where that changes nothing.
    fn alpha_test_range(&mut self, draw: &Draw) -> Option<[f32; 4]> {
        if draw.tests & test_flags::ALPHA_TEST == 0
            || !self.programs.textured.have_test
            || (draw.tests >> test_flags::ALPHA_MASK_SHIFT) & 0xFF != 0xFF
        {
            return None;
        }
        let reference = (draw.tests >> test_flags::ALPHA_REFERENCE_SHIFT) & 0xFF;
        let (lo, hi, outside) = alpha_test_bounds(draw.tests >> test_flags::ALPHA_FUNCTION_SHIFT, reference)?;
        let stats = stats();
        stats.alpha_tested += 1;
        let passes_all = !outside && lo <= 0.0 && hi >= 255.0;
        let rejects_only_zero = (!outside && lo == 1.0 && hi >= 255.0) || (outside && lo == 0.0 && hi == 0.0);
        if passes_all || (rejects_only_zero && alpha_zero_changes_nothing(draw)) {
            stats.alpha_test_skipped += 1;
            return None;
        }
        if ALPHA_LOG.load(Ordering::Relaxed) != 0 {
            self.note_discard_draw(draw);
        }
        Some(alpha_test_uniform(lo, hi, outside))
    }

    /// Our textured fragment program's uniforms: fog colour and colour
    /// scale, for the alpha-test program its range, for the BLEND_FIX ones
    /// FIX B. Set into a fresh default uniform buffer only when something
    /// changed.
    fn textured_fragment_uniforms(&mut self, tested: bool, fog_and_scale: [f32; 4], alpha_range: [f32; 4], blend: u32) {
        let textured = &self.programs.textured;
        if blend & BLEND_ENABLED != 0 && blend & BLEND_FIX_B_ALPHA != 0 && textured.have_fix {
            // The BLEND_FIX variants (blend_program picked them): FIX B and
            // the source factor; rare, so no caching.
            let fix = [((blend >> 12) & 0xFF) as f32 / 255.0, ((blend >> 20) & 3) as f32, 0.0, 0.0];
            let buffer = reserve_fragment_uniforms();
            if tested {
                set_uniform(buffer, textured.fog_and_scale_test_fix, 4, fog_and_scale.as_ptr());
                set_uniform(buffer, textured.alpha_test_fix, 4, alpha_range.as_ptr());
                set_uniform(buffer, textured.blend_fix_test, 4, fix.as_ptr());
            } else {
                set_uniform(buffer, textured.fog_and_scale_fix, 4, fog_and_scale.as_ptr());
                set_uniform(buffer, textured.blend_fix, 4, fix.as_ptr());
            }
            self.state.tint = None; // the cached default buffer is no longer the one in use
            stats().fix_b_draws += 1;
            return;
        }
        if !tested {
            let parameter = textured.fog_and_scale;
            self.use_fragment_color(parameter, fog_and_scale);
            return;
        }
        let parameter = textured.fog_and_scale_test;
        let alpha_test = textured.alpha_test;
        let (fog_bits, range_bits) = (bits(&fog_and_scale), bits(&alpha_range));
        if self.tint_is(parameter, &fog_bits) && same_words(&self.alpha_test_cache, &range_bits) {
            return;
        }
        let buffer = reserve_fragment_uniforms();
        set_uniform(buffer, parameter, 4, fog_and_scale.as_ptr());
        set_uniform(buffer, alpha_test, 4, alpha_range.as_ptr());
        self.state.tint = Some((parameter, fog_bits));
        self.alpha_test_cache = range_bits;
    }

    /// The uniforms of frame-read program `variant` (`Programs::
    /// read_program`) for `draw`: for the generic one the blend GXM would do
    /// (`Programs::blend_program`), the target's channel bits and the GE's
    /// dither matrix; for the textured ones also fog, colour scale, alpha
    /// test (`ALPHA_ALL_PASS` without) and BLEND_FIX. The fast untextured
    /// ones have none.
    fn read_fragment_uniforms(&mut self, frame: &Frame, draw: &Draw, variant: ReadVariant, tested: bool, textured: Option<([f32; 4], [f32; 4])>) {
        let Some(bits) = nativeres_core::frame::target_channel_bits(draw.flags) else {
            return;
        };
        stats().cut_draws += 1;
        let generic = variant == ReadVariant::Generic;
        if let ReadVariant::Fast { way, dither, .. } = variant {
            stats().cut_fast_draws += 1;
            if matches!(way, ReadWay::Off | ReadWay::Precut) {
                stats().cut_off_draws += 1;
            }
            if dither {
                self.use_dither_texture(frame.dither_matrix);
            }
            if textured.is_none() {
                return;
            }
        }
        let program = self.programs.made_read_program(textured.is_some(), tested, variant);
        let mut blend = draw.blend;
        if !self.keep_target_alpha && blend & BLEND_ALPHA_ONLY == 0 {
            blend |= BLEND_WRITES_ALPHA; // as blend_program: nobody reads it
        }
        let enabled = blend & BLEND_ENABLED != 0;
        let writes_alpha = blend & BLEND_WRITES_ALPHA != 0;
        let alpha_only = writes_alpha && blend & BLEND_ALPHA_ONLY != 0;
        let dither_flag = draw.flags & draw_flags::DITHER != 0;
        let buffer = reserve_fragment_uniforms();
        if generic {
            let cut = cut_uniforms(blend, enabled && !alpha_only, bits, !alpha_only, writes_alpha, dither_flag);
            for (parameter, values) in program.cut.iter().zip(&cut) {
                set_uniform(buffer, *parameter, 4, values.as_ptr());
            }
            // The GE's matrix (zeros for an undithered draw).
            let rows: [[f32; 4]; 4] = core::array::from_fn(|y| {
                core::array::from_fn(|x| if dither_flag { nativeres_core::frame::dither_value(frame.dither_matrix, x, y) as f32 } else { 0.0 })
            });
            for (parameter, row) in program.dither.iter().zip(&rows) {
                set_uniform(buffer, *parameter, 4, row.as_ptr());
            }
        }
        if let Some((fog_and_scale, alpha_range)) = textured {
            // BLEND_FIX: rgb x (x + y * source alpha), alpha -> FIX B.
            let fix = if enabled && blend & BLEND_FIX_B_ALPHA != 0 {
                let fix_b = ((blend >> 12) & 0xFF) as f32 / 255.0;
                match (blend >> 20) & 3 {
                    1 => [0.0, 1.0, 1.0, fix_b],
                    2 => [1.0, -1.0, 1.0, fix_b],
                    _ => [1.0, 0.0, 1.0, fix_b],
                }
            } else {
                [1.0, 0.0, 0.0, 0.0]
            };
            set_uniform(buffer, program.fog_and_scale, 4, fog_and_scale.as_ptr());
            if !program.alpha_test.is_null() {
                set_uniform(buffer, program.alpha_test, 4, alpha_range.as_ptr());
            }
            if !program.blend_fix.is_null() {
                set_uniform(buffer, program.blend_fix, 4, fix.as_ptr());
            }
        }
        self.state.tint = None; // the cached default buffer is no longer the one in use
    }

    /// The 3D program with `transform` (a draw in model space) and
    /// `fragment`. Its uniforms go into a fresh default uniform buffer only
    /// when the transform changes or the program was just switched to:
    /// consecutive draws mostly share their transform (MTX's race start:
    /// 464 draws, 42 transforms).
    fn use_transform(&mut self, frame: &Frame, transform: &GpuTransform, uv_scale: [f32; 2], fragment: *const c_void) {
        let textured_vertex_3d = self.programs.textured.vertex_3d;
        let switched = textured_vertex_3d as *const c_void != self.state.vertex_program;
        let wvp_3d = self.programs.textured.wvp_3d;
        let matrix = self.native_matrix();
        self.use_programs(textured_vertex_3d, wvp_3d, matrix, fragment);
        if switched || !core::ptr::eq(transform, self.state.transform) || uv_scale != self.state.uv_scale {
            let textured = &self.programs.textured;
            let buffer = reserve_vertex_uniforms();
            set_uniform(buffer, textured.wvp_3d, 16, matrix);
            set_uniform(buffer, textured.rows_3d, 16, floats(&transform.rows));
            set_uniform(buffer, textured.fog_row_3d, 4, floats(&transform.fog_row));
            set_uniform(buffer, textured.viewport_3d, 4, floats(&transform.viewport));
            set_uniform(buffer, textured.depth_3d, 4, floats(&transform.depth));
            set_uniform(buffer, textured.uv_scale_3d, 2, uv_scale.as_ptr());
            // GPU lighting: the normal rows and the light set number always
            // (0 = the vertex colour as it is; 16 floats: the three rows and
            // `light_set`, which follows them), the lights only for lit draws.
            set_uniform(buffer, textured.normal_rows_3d, 16, floats(&transform.normal_rows));
            if let Some(lights) = light_set(frame, transform) {
                set_uniform(buffer, textured.lights_3d, LIGHT_FLOATS, floats(lights));
            }
            self.state.transform = transform;
            self.state.uv_scale = uv_scale;
        }
    }

    /// The viewport for raw draws seen by `camera` (nativeres_raw_v.cg sends
    /// z / w out as (z / w - 1) / 2, so the GPU clips at the PSP's near
    /// plane): depth = offset + scale * that = 0.5 + 0.5 * the PSP depth,
    /// what every other draw writes (measured on the Vita,
    /// 2026-11-08). x and y as GXM's default for our 960x544 targets.
    fn use_raw_viewport(&mut self, camera: &RawCamera) {
        let (center, scale) = (camera.depth[0], camera.depth[1]);
        let offset = 0.5 + 0.5 * (center + scale);
        if self.state.raw_viewport == Some([offset, scale]) {
            return;
        }
        // Safe: the display's context.
        unsafe { gxm::sceGxmSetViewport(context(), 480.0, 480.0, 272.0, -272.0, offset, scale) };
        self.state.raw_viewport = Some([offset, scale]);
    }

    /// GXM's default viewport again after raw draws (measured: x 480, 480,
    /// y 272, -272, z 0.5, 0.5).
    fn use_default_viewport(&mut self) {
        if self.state.raw_viewport.is_none() {
            return;
        }
        // Safe: the display's context.
        unsafe { gxm::sceGxmSetViewport(context(), 480.0, 480.0, 272.0, -272.0, 0.5, 0.5) };
        self.state.raw_viewport = None;
    }

    /// The raw (or sprite) program for `draw` with its object and camera
    /// (`RawTransform`, `RawCamera`), colour rules and `fragment`. Uniforms
    /// only when one of them, the program (the format), the bones or the
    /// texture coordinate scale changed; the lights only for draws lit by the
    /// GPU, bones only for the raw programs (the plain one has one at most).
    #[allow(clippy::too_many_arguments)]
    fn use_raw_transform(
        &mut self,
        frame: &Frame,
        draw: &Draw,
        set: RawSetChoice,
        program: *mut c_void,
        (transform, camera): (&RawTransform, &RawCamera),
        colors: &RawColors,
        uv_scale: [f32; 2],
        fragment: *const c_void,
    ) {
        let (uniforms, skinning) = match set {
            RawSetChoice::Raw(variant) => {
                let raw = self.programs.raw_set(variant);
                (raw.uniforms, Some(raw_skinning(raw)))
            }
            RawSetChoice::Sprites => (self.programs.sprites.uniforms, None),
        };
        let switched = program as *const c_void != self.state.vertex_program;
        let matrix = self.native_matrix();
        self.use_programs(program, uniforms.wvp, matrix, fragment);
        let bones_changed = set != RawSetChoice::Sprites && draw.bones != self.state.bones;
        if switched
            || !core::ptr::eq(transform, self.state.raw_transform)
            || !core::ptr::eq(colors, self.state.raw_colors)
            || bones_changed
            || uv_scale != self.state.uv_scale
        {
            let buffer = reserve_vertex_uniforms();
            let format = VertexFormat(draw.vertex_format);
            let format_uniform = [format.position_scale(), ((draw.vertex_format >> 2) & 7) as f32, 0.0, 0.0];
            let raw_uv = nativeres_core::raw_blocks::raw_uv(camera, format);
            set_raw_uniforms(buffer, &uniforms, matrix, (transform, camera), uv_scale, &format_uniform, &raw_uv, colors);
            if let Some(skinning) = skinning {
                set_uniform(buffer, skinning.scale, 4, floats(&colors.scale));
                if skinning.variant.lights() {
                    if let Some(lights) = raw_light_set(frame, transform) {
                        set_uniform(buffer, skinning.lights, LIGHT_FLOATS, floats(lights));
                    }
                }
                self.set_raw_bones(frame, draw, buffer, &skinning);
            }
            self.state.raw_transform = transform;
            self.state.raw_colors = colors;
            // Sprites have no bones: the raw programs set theirs again.
            self.state.bones = if set == RawSetChoice::Sprites { u32::MAX } else { draw.bones };
            self.state.uv_scale = uv_scale;
        }
        self.use_raw_viewport(camera);
    }

    /// The raw programs' bones and uSkin (nativeres_raw_v.cg): the programs
    /// with four or eight bones get as many (the format's, zeros for the
    /// rest) and skin where uSkin.x is 1; the one-bone programs get bone 0
    /// and add uSkin.z to the weight, so formats without weights get weight 1
    /// and the identity.
    fn set_raw_bones(&mut self, frame: &Frame, draw: &Draw, buffer: *mut c_void, skinning: &RawSkinning) {
        const WEIGHT_SCALES: [f32; 4] = [1.0, 1.0 / 128.0, 1.0 / 32768.0, 1.0];
        const IDENTITY_ROWS: [f32; 12] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let weight_type = (draw.vertex_format >> 9) & 3;
        let program_bones = skinning.variant.bones();
        let several = program_bones > 1;
        let bone_count = (((draw.vertex_format >> 14) & 7) + 1).min(program_bones);
        let first = draw.bones as usize;
        let wanted = 12 * bone_count as usize;
        let mut skin = [0.0f32; 4];
        let floats_used = 12 * program_bones as usize;
        if draw.bones != 0 && weight_type != 0 && first - 1 + wanted <= (frame.bone_float_count as usize).min(frame.bone_floats.len()) {
            self.bones[..floats_used].fill(0.0);
            self.bones[..wanted].copy_from_slice(&frame.bone_floats[first - 1..first - 1 + wanted]);
            skin[0] = 1.0;
            skin[1] = WEIGHT_SCALES[weight_type as usize];
        } else if several {
            self.bones[..floats_used].fill(0.0); // not skinned: uSkin.x 0
        } else {
            self.bones[..12].copy_from_slice(&IDENTITY_ROWS);
            skin[2] = 1.0;
        }
        set_uniform(buffer, skinning.bones, floats_used as u32, self.bones.as_ptr());
        set_uniform(buffer, skinning.skin, 4, skin.as_ptr());
    }

    /// The draw's triangle corners as an index buffer the GPU can read and
    /// their count: the index ring's list (written by the capture), or
    /// built in the frame pool (index runs written out in the order frame.rs
    /// run_indices gives; plain indices copied). None if the pool is full or
    /// the draw's indices lie outside the frame.
    fn draw_indices(&mut self, frame: &Frame, draw: &Draw, draw_index: usize) -> Option<(*const u16, u32)> {
        let start = draw_clock();
        let first = draw.first_index as usize;
        let stored = frame.indices.get(first..first.checked_add(draw.index_count as usize)?)?;
        let stats = stats();
        let runs = draw.flags & draw_flags::INDEX_RUNS != 0;
        let corners = if runs { run_corners(stored) } else { draw.index_count };
        // Written out by the capture already (the index ring).
        let at = if frame.ring_indices.is_null() { NO_RING_INDICES } else { *frame.draw_ring_index.get(draw_index)? };
        if at != NO_RING_INDICES && at as u64 + corners as u64 <= frame.ring_indices_len as u64 {
            stats.ring_draws += 1;
            stats.indices_us += draw_clock().wrapping_sub(start);
            // Safe: inside the ring (checked above).
            return Some((unsafe { frame.ring_indices.add(at as usize) }, corners));
        }
        let out = frame_pool_alloc(corners * 2, 2) as *mut u16;
        if !out.is_null() {
            // Safe: the pool gave `corners` u16s.
            let out = unsafe { core::slice::from_raw_parts_mut(out, corners as usize) };
            if runs {
                write_runs(stored, out);
            } else {
                out.copy_from_slice(stored);
            }
        }
        stats.indices_us += draw_clock().wrapping_sub(start);
        stats.pool_kb += (corners * 2 + draw.vertex_count * core::mem::size_of::<FrameVertex>() as u32) / 1024;
        (!out.is_null()).then_some((out as *const u16, corners))
    }

    /// The frame's vertices of `draw`, copied into the frame pool so the GPU
    /// can read them (our programs' matrix does the scaling and the depth);
    /// only textures from our render targets need their texture coordinates
    /// scaled. Draws without fog have fog 1 in every vertex (the core makes
    /// sure). None if the pool is full.
    fn textured_vertices(frame: &Frame, draw: &Draw, uv_scale: [f32; 2]) -> Option<&'static mut [FrameVertex]> {
        let first = draw.first_vertex as usize;
        let vertices = frame.vertices.get(first..first.checked_add(draw.vertex_count as usize)?)?;
        let out = frame_pool_alloc(draw.vertex_count * core::mem::size_of::<FrameVertex>() as u32, 16) as *mut FrameVertex;
        if out.is_null() {
            return None;
        }
        // Safe: the pool gave room for `vertex_count` vertices.
        let out = unsafe { core::slice::from_raw_parts_mut(out, vertices.len()) };
        out.copy_from_slice(vertices);
        if uv_scale != [1.0, 1.0] {
            for vertex in out.iter_mut() {
                vertex.u *= uv_scale[0];
                vertex.v *= uv_scale[1];
            }
        }
        Some(out)
    }

    /// Fallback without our programs: the basic texture-tint program, with
    /// the first vertex's colour for the whole draw. Sets the programs and
    /// returns the vertices, or null if the frame pool is full.
    fn use_basic_texture_tint(&mut self, frame: &Frame, draw: &Draw, uv_scale: [f32; 2]) -> *const c_void {
        let first = draw.first_vertex as usize;
        let Some(vertices) = first.checked_add(draw.vertex_count as usize).and_then(|end| frame.vertices.get(first..end)) else {
            return null();
        };
        let out = frame_pool_alloc(draw.vertex_count * core::mem::size_of::<basic_programs::TextureVertex>() as u32, 8)
            as *mut basic_programs::TextureVertex;
        if out.is_null() || vertices.is_empty() {
            return null();
        }
        // Safe: the pool gave room for them.
        let out_slice = unsafe { core::slice::from_raw_parts_mut(out, vertices.len()) };
        for (out, vertex) in out_slice.iter_mut().zip(vertices) {
            *out = basic_programs::TextureVertex {
                x: vertex.x * SCALE,
                y: vertex.y * SCALE,
                z: gpu_depth(vertex.z),
                w: 1.0,
                u: vertex.u * uv_scale[0],
                v: vertex.v * uv_scale[1],
            };
        }
        let mut tint = [0.0f32; 4];
        for (channel, value) in tint.iter_mut().enumerate() {
            *value = ((vertices[0].color >> (8 * channel)) & 0xFF) as f32 / 255.0;
        }
        if draw.flags & draw_flags::DOUBLE_COLOR != 0 {
            // The shader's output is clamped: values above 1 saturate, like on the PSP.
            for value in tint.iter_mut().take(3) {
                *value *= 2.0;
            }
        }
        let Some(basic) = basic_programs::get() else {
            return null();
        };
        let fragment = self.programs.blend_program(Kind::TextureTint, draw.blend, self.keep_target_alpha);
        let fragment = if fragment.is_null() { basic.texture_tint_fragment } else { fragment };
        self.use_programs(basic.texture_vertex, basic.texture_wvp, basic_programs::ORTHO.as_ptr(), fragment);
        self.use_fragment_color(basic.tint_color, tint);
        out as *const c_void
    }

    /// A textured draw. `palette`: the colour table of an index texture,
    /// else null. `uv_scale`: multiplies the texture coordinates (1, 1
    /// except for render targets used as textures).
    fn draw_textured(&mut self, frame: &Frame, draw: &Draw, draw_index: usize, texture: &TextureObject, palette: *const c_void, uv_scale: [f32; 2]) {
        let convert_start = draw_clock();
        let Some((indices, index_count)) = self.draw_indices(frame, draw, draw_index) else {
            return; // the frame pool is full
        };
        if skip(8) {
            return;
        }
        // The alpha test in the fragment program where the draw has one.
        let programs_ready = self.programs.textured_ready();
        let mut alpha_range = [0.0f32; 4];
        let mut tested = false;
        if programs_ready {
            if let Some(range) = self.alpha_test_range(draw) {
                alpha_range = range;
                tested = ALPHA_SHADER.load(Ordering::Relaxed) != 0;
            }
        }
        let keep = self.keep_target_alpha;
        // A 16-bit PSP target: the frame-read program (alpha test included).
        let choice = read_choice(frame, draw, keep);
        let read_variant = choice.unwrap_or(ReadVariant::Generic);
        let read_program = match choice {
            Some(variant) if programs_ready => self.programs.read_program(true, tested, variant).map(|program| program.program),
            _ => None,
        };
        let read = read_program.is_some();
        let read_alpha_range = if tested { alpha_range } else { ALPHA_ALL_PASS };
        let mut fragment = if let Some(program) = read_program {
            program
        } else if programs_ready {
            self.programs.blend_program(if tested { Kind::TexturedTest } else { Kind::Textured }, draw.blend, keep)
        } else {
            null_mut()
        };
        if tested && fragment.is_null() {
            // No program for this blend mode: without the test, as before.
            tested = false;
            fragment = self.programs.blend_program(Kind::Textured, draw.blend, keep);
        }
        // Vertices in model space: the 3D program transforms them (raw
        // draws: their own blocks, `raw_transform_of` below).
        let transform = if draw.transform != 0 && draw.flags & draw_flags::RAW_VERTICES == 0 {
            if fragment.is_null() || self.programs.textured.vertex_3d.is_null() || draw.transform > frame.transform_count {
                return; // can't happen: the capture only makes these once the program exists
            }
            match frame.transforms.get(draw.transform as usize - 1) {
                Some(transform) => Some(transform),
                None => return,
            }
        } else {
            None
        };
        let out: *const c_void;
        let mut sprites = false;
        // What the GPU may read stream 0 from: the
        // mapped region, and bytes per index step / for the last vertex.
        let mut check_region: *const u8 = null();
        let mut check_region_bytes = 0u32;
        let mut check_stride = 0u32;
        let mut check_tail = 0u32;
        let checks = DRAW_CHECKS.load(Ordering::Relaxed) != 0;
        let mut sprite_stride = 0u32;
        if !fragment.is_null() && draw.flags & draw_flags::RAW_VERTICES != 0 {
            // Raw PSP vertices in the raw arena or PSP RAM: drawn from there as they are.
            sprites = draw.flags & draw_flags::RAW_SPRITES != 0;
            let colors = match draw.raw_colors {
                0 => None,
                n if n <= frame.raw_colors_count => frame.raw_colors.get(n as usize - 1),
                _ => None,
            };
            // The smallest raw program that takes the draw's weights and lights.
            let set = if sprites {
                RawSetChoice::Sprites
            } else {
                let weights = if (draw.vertex_format >> 9) & 3 != 0 { ((draw.vertex_format >> 14) & 7) + 1 } else { 0 };
                let lit_with_lights = !colors.is_some_and(|colors| colors.rules[0] < 2.5);
                let plain = RAW_PLAIN.load(Ordering::Relaxed) != 0;
                let variants = RAW_VARIANTS.load(Ordering::Relaxed) != 0;
                let mut variant = raw_variant(weights, lit_with_lights, plain, variants);
                if variant != variant.flat() && RAW_FLAT_LIGHTS.load(Ordering::Relaxed) != 0 {
                    let flat = frame.raw_transform_of(draw).and_then(|(transform, _)| raw_light_set(frame, transform)).is_some_and(flat_lights);
                    if flat && self.programs.raw_set(variant.flat()).ready() {
                        variant = variant.flat();
                    }
                }
                if !self.programs.raw_set(variant).ready() {
                    variant = RawVariant::Full;
                }
                match variant {
                    RawVariant::Plain => stats().plain_draws += 1,
                    RawVariant::Lit1 | RawVariant::Lit4 => stats().lit_variant_draws += 1,
                    RawVariant::Lit1Flat | RawVariant::Lit4Flat => {
                        stats().lit_variant_draws += 1;
                        stats().flat_light_draws += 1;
                    }
                    RawVariant::Full => {}
                }
                RawSetChoice::Raw(variant)
            };
            let program = match set {
                RawSetChoice::Sprites => self.programs.sprites.program_for(draw.vertex_format),
                RawSetChoice::Raw(variant) => self.programs.raw_set(variant).program_for(draw.vertex_format),
            };
            let in_psp_ram = draw.flags & draw_flags::RAW_PSP_RAM != 0;
            let (Some(blocks), Some(colors)) = (frame.raw_transform_of(draw), colors) else {
                return; // can't happen: raw draws only come with the arena, the program, their blocks and colour rules
            };
            if program.is_null() {
                // Every program place taken by other formats (or no shaders):
                // the draw is missing from the picture. Counted (render statistics).
                stats().raw_no_program += 1;
                return;
            }
            if !in_psp_ram && (frame.raw_vertices.is_null() || draw.first_vertex >= frame.raw_vertices_len) {
                return; // can't happen (as above)
            }
            if in_psp_ram {
                // Zero copy: the game's own vertices in PSP RAM.
                let ram = arenas::PSP_RAM_FOR_GPU.load(Ordering::Acquire);
                let ram_bytes = arenas::PSP_RAM_FOR_GPU_BYTES.load(Ordering::Acquire);
                let offset = (draw.first_vertex & 0x1FFF_FFFF).wrapping_sub(PSP_RAM_GPU_BASE);
                if ram.is_null() || offset >= ram_bytes {
                    return; // can't happen: only recorded with zero copy on, after the mapping
                }
                // Safe: inside the mapped block (checked).
                out = unsafe { ram.add(offset as usize) } as *const c_void;
                check_region = ram;
                check_region_bytes = ram_bytes;
            } else {
                // Safe: inside the raw arena (checked against its length).
                out = unsafe { frame.raw_vertices.add(draw.first_vertex as usize) } as *const c_void;
                check_region = RAW_ARENA.load(Ordering::Relaxed);
                check_region_bytes = RAW_ARENA_BYTES.load(Ordering::Relaxed);
            }
            // The format's stride only where needed (layout() is not free).
            let stride = if sprites || checks { VertexFormat(draw.vertex_format).layout().size } else { 0 };
            if sprites {
                sprite_stride = stride;
            }
            if checks {
                // Sprites: streams 0 and 1 step two strides per corner
                // index; stream 1 starts one stride later.
                check_stride = if sprites { 2 * stride } else { stride };
                check_tail = check_stride;
            }
            stats().convert_us += draw_clock().wrapping_sub(convert_start);
            if !skip(2) {
                self.use_raw_transform(frame, draw, set, program, blocks, colors, uv_scale, fragment);
                if read {
                    self.read_fragment_uniforms(frame, draw, read_variant, tested, Some((fog_and_scale(draw), read_alpha_range)));
                } else {
                    self.textured_fragment_uniforms(tested, fog_and_scale(draw), alpha_range, draw.blend);
                }
            }
        } else if !fragment.is_null() {
            if draw.flags & draw_flags::CACHED_VERTICES != 0 {
                // In the vertex cache (GPU memory): drawn from there, no copy.
                if transform.is_none() || frame.vertex_cache.is_null() {
                    return; // can't happen: cached draws always have both
                }
                // Its indices count from its window of the cache.
                // Safe: the capture placed the draw inside the cache.
                out = unsafe { frame.vertex_cache.add(draw.first_vertex as usize) } as *const c_void;
                check_region = VERTEX_ARENA.load(Ordering::Relaxed) as *const u8;
                check_region_bytes =
                    VERTEX_ARENA_VERTICES.load(Ordering::Relaxed) * core::mem::size_of::<FrameVertex>() as u32;
                check_stride = core::mem::size_of::<FrameVertex>() as u32;
                check_tail = check_stride;
            } else {
                // The 3D program scales texture coordinates itself (uUvScale).
                let Some(vertices) = Self::textured_vertices(frame, draw, if transform.is_some() { [1.0, 1.0] } else { uv_scale })
                else {
                    return; // the frame pool is full
                };
                if smooth_textures() && draw.flags & draw_flags::RECTANGLES != 0 {
                    keep_filter_inside_sprites(vertices, &texture.gxm);
                }
                out = vertices.as_ptr() as *const c_void;
                check_region = out as *const u8;
                check_region_bytes = draw.vertex_count * core::mem::size_of::<FrameVertex>() as u32;
                check_stride = core::mem::size_of::<FrameVertex>() as u32;
                check_tail = check_stride;
            }
            stats().convert_us += draw_clock().wrapping_sub(convert_start);
            self.use_default_viewport();
            if !skip(2) {
                match transform {
                    Some(transform) => self.use_transform(frame, transform, uv_scale, fragment),
                    None => {
                        let (vertex, wvp) = (self.programs.textured.vertex, self.programs.textured.wvp);
                        let matrix = self.native_matrix();
                        self.use_programs(vertex, wvp, matrix, fragment);
                    }
                }
                if read {
                    self.read_fragment_uniforms(frame, draw, read_variant, tested, Some((fog_and_scale(draw), read_alpha_range)));
                } else {
                    self.textured_fragment_uniforms(tested, fog_and_scale(draw), alpha_range, draw.blend);
                }
            }
        } else {
            self.use_default_viewport();
            out = self.use_basic_texture_tint(frame, draw, uv_scale);
            if out.is_null() {
                return; // the frame pool is full
            }
        }

        // Filter and wrap mode are per draw on the PSP, so set them on a copy.
        if !skip(4) {
            let mut gxm_texture = texture.gxm;
            let linear = draw.flags & draw_flags::LINEAR_FILTER != 0 || smooth_textures();
            let filter = if linear { gxm::TEXTURE_FILTER_LINEAR } else { gxm::TEXTURE_FILTER_POINT };
            let address = |clamp: u32| if draw.flags & clamp != 0 { gxm::TEXTURE_ADDR_CLAMP } else { gxm::TEXTURE_ADDR_REPEAT };
            // Safe: our copy of the texture state.
            unsafe {
                gxm::sceGxmTextureSetMinFilter(&mut gxm_texture, filter);
                gxm::sceGxmTextureSetMagFilter(&mut gxm_texture, filter);
                gxm::sceGxmTextureSetUAddrMode(&mut gxm_texture, address(draw_flags::CLAMP_U));
                gxm::sceGxmTextureSetVAddrMode(&mut gxm_texture, address(draw_flags::CLAMP_V));
                // Mip levels (menu "NativeRes Distant Textures"): the GPU picks
                // the level; blended between two with the GE's "mip linear"
                // filters; the bias from the capture (lod bias field 31 + 8 x
                // levels, measured with `draw_mip_test`). Textures without
                // levels have a mip count of 0.
                if draw.flags & draw_flags::MIPS != 0 && gxm::sceGxmTextureGetMipmapCount(&gxm_texture) > 1 {
                    let mip_filter = if draw.flags & draw_flags::MIP_LINEAR != 0 {
                        gxm::TEXTURE_MIP_FILTER_ENABLED
                    } else {
                        gxm::TEXTURE_MIP_FILTER_DISABLED
                    };
                    gxm::sceGxmTextureSetMipFilter(&mut gxm_texture, mip_filter);
                    let bias = 31 + ((draw.flags & draw_flags::MIP_BIAS_MASK) >> draw_flags::MIP_BIAS_SHIFT) as u8 as i8 as i32;
                    gxm::sceGxmTextureSetLodBias(&mut gxm_texture, bias.clamp(0, 63) as u32);
                }
                if !palette.is_null() {
                    gxm::sceGxmTextureSetPalette(&mut gxm_texture, palette);
                }
            }
            self.use_texture(&gxm_texture);
        }

        if checks && !check_region.is_null() && check_stride != 0 {
            // Safe: `indices` holds `index_count` indices (draw_indices).
            let list = unsafe { core::slice::from_raw_parts(indices, index_count as usize) };
            let highest = list.iter().copied().max().unwrap_or(0) as u64;
            let start = out as usize as i64 - check_region as usize as i64;
            let reach = highest * check_stride as u64 + check_tail as u64;
            let outside = start < 0
                || start as u64 > check_region_bytes as u64
                || reach > check_region_bytes as u64 - start as u64
                || (sprites && highest >= SPRITE_CORNERS as u64);
            if outside {
                DRAW_CHECK_FAILURES.fetch_add(1, Ordering::Relaxed);
                render_log::line(format_args!(
                    "drawcheck: frame {} draw flags {:08x} format {:08x} first {:08x} highest index {} x {} (+{}) from {:+} in a {}-byte region: skipped\n",
                    frame.frame_number,
                    draw.flags,
                    draw.vertex_format,
                    draw.first_vertex,
                    highest,
                    check_stride,
                    check_tail,
                    start,
                    check_region_bytes
                ));
                return;
            }
        }
        if skip(1) {
            return;
        }
        // Safe: the streams and indices checked or built above, for the
        // programs set above.
        unsafe {
            gxm::sceGxmSetVertexStream(context(), 0, out);
            if sprites {
                // B one stride after A; the corner numbers (SpriteSet::program_for).
                gxm::sceGxmSetVertexStream(context(), 1, (out as *const u8).add(sprite_stride as usize) as *const c_void);
                gxm::sceGxmSetVertexStream(context(), 2, self.programs.sprites.corners as *const c_void);
            }
            note_draw_result(gxm::sceGxmDraw(context(), gxm::PRIMITIVE_TRIANGLES, gxm::INDEX_FORMAT_U16, indices as *const c_void, index_count));
        }
    }

    /// An untextured draw: the basic colour program, per-vertex colour.
    fn draw_colored(&mut self, frame: &Frame, draw: &Draw, draw_index: usize) {
        let convert_start = draw_clock();
        let indices = self.draw_indices(frame, draw, draw_index);
        if skip(8) {
            return;
        }
        let first = draw.first_vertex as usize;
        let Some(vertices) = first.checked_add(draw.vertex_count as usize).and_then(|end| frame.vertices.get(first..end)) else {
            return;
        };
        let out = frame_pool_alloc(draw.vertex_count * core::mem::size_of::<basic_programs::ColorVertex>() as u32, 16)
            as *mut basic_programs::ColorVertex;
        let Some((indices, index_count)) = indices else {
            return;
        };
        if out.is_null() {
            return;
        }
        // Safe: the pool gave room for them.
        let out_slice = unsafe { core::slice::from_raw_parts_mut(out, vertices.len()) };
        for (out, vertex) in out_slice.iter_mut().zip(vertices) {
            *out = basic_programs::ColorVertex { x: vertex.x * SCALE, y: vertex.y * SCALE, z: gpu_depth(vertex.z), color: vertex.color };
        }
        stats().convert_us += draw_clock().wrapping_sub(convert_start);
        let mut choice = read_choice(frame, draw, self.keep_target_alpha);
        // Alpha 0 at every vertex: a normal blend leaves the target as it
        // is (`ReadWay::Invisible`, MHF's full-screen fade quad).
        if INVISIBLE_SOURCES.load(Ordering::Relaxed) != 0 && vertices.iter().all(|vertex| vertex.color >> 24 == 0) {
            choice = choice.map(ReadVariant::for_invisible_source);
        }
        let read_variant = choice.unwrap_or(ReadVariant::Generic);
        let read_program = choice.and_then(|variant| self.programs.read_program(false, false, variant).map(|program| program.program));
        let read = read_program.is_some();
        let fragment = if let Some(program) = read_program {
            program
        } else {
            self.programs.blend_program(Kind::Color, draw.blend, self.keep_target_alpha)
        };
        self.use_default_viewport();
        let Some(basic) = basic_programs::get() else {
            return;
        };
        if !skip(2) {
            let fragment = if fragment.is_null() { basic.color_fragment } else { fragment };
            self.use_programs(basic.color_vertex, basic.color_wvp, basic_programs::ORTHO.as_ptr(), fragment);
            if read {
                self.read_fragment_uniforms(frame, draw, read_variant, false, None);
            }
        }
        if skip(1) {
            return;
        }
        // Safe: vertices and indices in the frame pool, the programs set.
        unsafe {
            gxm::sceGxmSetVertexStream(context(), 0, out as *const c_void);
            note_draw_result(gxm::sceGxmDraw(context(), gxm::PRIMITIVE_TRIANGLES, gxm::INDEX_FORMAT_U16, indices as *const c_void, index_count));
        }
    }

    /// The GPU-readable block of the GPU tests (index, mip): 1 MiB
    /// phycont, non-cached (no stale CPU cache lines for the GPU), made at
    /// the first test and kept. Null if there is no memory.
    fn test_block(&mut self) -> *mut u8 {
        if !self.test_block_tried {
            self.test_block_tried = true;
            // Safe: plain system calls; the block is ours alone.
            unsafe {
                let uid = sys::alloc_block(
                    b"NativeResTestBlock\0".as_ptr(),
                    sys::SCE_KERNEL_MEMBLOCK_TYPE_USER_MAIN_PHYCONT_NC_RW,
                    TEST_BLOCK_BYTES as u32,
                    null_mut(),
                );
                let mut base: *mut c_void = null_mut();
                if uid >= 0
                    && sys::sceKernelGetMemBlockBase(uid, &mut base) >= 0
                    && gxm::sceGxmMapMemory(base, TEST_BLOCK_BYTES as u32, gxm::MEMORY_ATTRIB_READ) >= 0
                {
                    self.test_block = base as *mut u8;
                } else {
                    render_log::line(format_args!("test block: no memory ({:08x})\n", uid));
                    if uid >= 0 {
                        sys::free_block(uid);
                    }
                }
            }
        }
        self.test_block
    }

    /// Are 16-bit index values from 64000 on
    /// drawn? gxm.h says SCE_GXM_INDEX_SOURCE_INDEX_16BIT values "must be
    /// lower than 64000", but the vertex cache's windows are 65536 vertices,
    /// so cached draws use 64000-65535 too. Five equal white squares in a
    /// row after the frame's draws, each from four vertices at index
    /// FIRST[k]..+3 of the test block (the basic colour program). Indices
    /// below 64000 are the control; a square missing or misplaced means
    /// those index values are not drawn as given.
    fn draw_index_test(&mut self) {
        const FIRST: [u32; 5] = [0, 63992, 64000, 65000, 65532];
        let block = self.test_block() as *mut basic_programs::ColorVertex;
        let indices = frame_pool_alloc(5 * 6 * 2, 2) as *mut u16;
        let Some(basic) = basic_programs::get() else {
            return;
        };
        if block.is_null() || indices.is_null() {
            return;
        }
        // Safe: the block holds 65536 colour vertices, the pool gave 30 indices.
        let (vertices, indices) =
            unsafe { (core::slice::from_raw_parts_mut(block, 65536), core::slice::from_raw_parts_mut(indices, 30)) };
        for (k, &first) in FIRST.iter().enumerate() {
            let (left, top, size) = (100.0 + 160.0 * k as f32, 220.0, 80.0);
            let corners = [(left, top), (left + size, top), (left, top + size), (left + size, top + size)];
            for (corner, &(x, y)) in corners.iter().enumerate() {
                vertices[first as usize + corner] = basic_programs::ColorVertex { x, y, z: 0.5, color: 0xFFFF_FFFF };
            }
            let first = first as u16;
            indices[6 * k..6 * k + 6].copy_from_slice(&[first, first + 1, first + 2, first + 1, first + 3, first + 2]);
        }
        self.prepare_test();
        self.use_programs(basic.color_vertex, basic.color_wvp, basic_programs::ORTHO.as_ptr(), basic.color_fragment);
        // Safe: the block is GPU-mapped.
        unsafe {
            gxm::sceGxmSetVertexStream(context(), 0, block as *const c_void);
            // One draw per square: a failing draw shows which one.
            for (k, first) in FIRST.iter().enumerate() {
                let result = gxm::sceGxmDraw(
                    context(),
                    gxm::PRIMITIVE_TRIANGLES,
                    gxm::INDEX_FORMAT_U16,
                    indices[6 * k..].as_ptr() as *const c_void,
                    6,
                );
                if result < 0 {
                    render_log::line(format_args!("indextest: draw from index {} failed: {:08x}\n", first, result));
                }
            }
        }
    }

    /// No scissor, no depth test, no culling, GXM's default viewport.
    fn prepare_test(&mut self) {
        // Safe: the display thread's context.
        unsafe { gxm::sceGxmSetRegionClip(context(), gxm::REGION_CLIP_NONE, 0, 0, 0, 0) };
        self.state.scissor = None;
        self.set_depth(gxm::DEPTH_FUNC_ALWAYS, gxm::DEPTH_WRITE_DISABLED);
        self.set_cull(gxm::CULL_NONE);
        self.use_default_viewport();
    }

    /// The shade test: 30x30 quads at PSP x 20 + 40 i through our textured
    /// programs exactly as a draw goes (vertex program, fragment program for
    /// no blending, uniforms), unblended over the frame:
    /// - y 100: `SHADE_TESTS` through the 8-bit programs;
    /// - y 140: the same through the 5551 "off" programs (cut to 5 bits);
    /// - y 180 (8-bit) and y 220 (5551): the alpha test, `SHADE_ALPHA_TESTS`
    ///   over a background quad (`SHADE_TESTS[0]`): where the test rejects,
    ///   the background shows.
    /// The PC tests never run the compiled programs; tools/shadetest.py
    /// checks these pixels against the GE's arithmetic (2026-11-18: a whole
    /// MHF scene came out black once while the PC replay was right).
    fn draw_shade_test(&mut self) {
        if !self.programs.textured_ready() {
            return;
        }
        // An 8x8 texture of one colour from the frame pool (GPU-mapped, kept
        // until the frame is drawn; not the 1 MiB test block, which a game
        // short of phycont memory like MHF doesn't leave room for).
        let texels = frame_pool_alloc(64 * 4, 64) as *mut u32;
        if texels.is_null() {
            return;
        }
        for i in 0..64 {
            // Safe: the pool gave room for 64 texels.
            unsafe { texels.add(i).write(SHADE_TEST_TEXEL) };
        }
        let mut texture = gxm::Texture { control_words: [0; 4] };
        // Safe: our texture struct over the block.
        unsafe {
            gxm::sceGxmTextureInitLinear(&mut texture, texels as *const c_void, gxm::TEXTURE_FORMAT_U8U8U8U8_ABGR, 8, 8, 0);
            gxm::sceGxmTextureSetMinFilter(&mut texture, gxm::TEXTURE_FILTER_POINT);
            gxm::sceGxmTextureSetMagFilter(&mut texture, gxm::TEXTURE_FILTER_POINT);
        }
        self.prepare_test();
        self.state.vertex_program = null(); // our matrix, set anew
        self.use_texture(&texture);
        for (i, case) in SHADE_TESTS.iter().enumerate() {
            let left = 20.0 + 40.0 * i as f32;
            self.shade_test_quad(left, 100.0, case, ShadeProgram::EightBit, ALPHA_ALL_PASS);
            self.shade_test_quad(left, 140.0, case, ShadeProgram::Cut5551, ALPHA_ALL_PASS);
        }
        for (i, &(function, reference)) in SHADE_ALPHA_TESTS.iter().enumerate() {
            let Some((lo, hi, outside)) = alpha_test_bounds(function, reference) else {
                continue;
            };
            let left = 20.0 + 40.0 * i as f32;
            for (top, program) in [(180.0, ShadeProgram::EightBit), (220.0, ShadeProgram::Cut5551)] {
                self.shade_test_quad(left, top, &SHADE_TESTS[0], ShadeProgram::EightBit, ALPHA_ALL_PASS);
                self.shade_test_quad(left, top, &SHADE_ALPHA_CASE, program, alpha_test_uniform(lo, hi, outside));
            }
        }
    }

    /// One shade-test quad, 30x30 PSP pixels at (`left`, `top`): `case`'s
    /// corners through `program`; `alpha_test` (`alpha_test_uniform`) for
    /// the alpha-test program where `case.tested`.
    fn shade_test_quad(&mut self, left: f32, top: f32, case: &ShadeCase, program: ShadeProgram, alpha_test: [f32; 4]) {
        let tested = case.tested && self.programs.textured.have_test;
        let off = ReadVariant::Fast { way: ReadWay::Off, keep_alpha: false, dither: false };
        let fragment = match program {
            ShadeProgram::EightBit => {
                let kind = if tested { Kind::TexturedTest } else { Kind::Textured };
                self.programs.blend_program(kind, 0, self.keep_target_alpha)
            }
            ShadeProgram::Cut5551 => match self.programs.read_program(true, tested, off) {
                Some(read) => read.program,
                None => return,
            },
        };
        if fragment.is_null() {
            return;
        }
        let vertex = self.programs.textured.vertex as *const c_void;
        let wvp = self.programs.textured.wvp;
        let matrix = self.native_matrix();
        self.use_programs(vertex, wvp, matrix, fragment as *const c_void);
        match program {
            ShadeProgram::EightBit => self.textured_fragment_uniforms(tested, case.fog_and_scale, alpha_test, 0),
            ShadeProgram::Cut5551 => {
                let read = self.programs.made_read_program(true, tested, off);
                let (fog_and_scale, alpha) = (read.fog_and_scale, read.alpha_test);
                let buffer = reserve_fragment_uniforms();
                set_uniform(buffer, fog_and_scale, 4, case.fog_and_scale.as_ptr());
                if !alpha.is_null() {
                    set_uniform(buffer, alpha, 4, alpha_test.as_ptr());
                }
                self.state.tint = None; // the cached default buffer is no longer the one in use
            }
        }
        let vertices = frame_pool_alloc(4 * core::mem::size_of::<FrameVertex>() as u32, 8) as *mut FrameVertex;
        let indices = frame_pool_alloc(6 * 2, 2) as *mut u16;
        if vertices.is_null() || indices.is_null() {
            return;
        }
        let size = 30.0;
        let corner = |x: f32, y: f32, side: usize, u: f32, v: f32| FrameVertex {
            x,
            y,
            z: 0.5,
            w: case.w[side],
            u,
            v,
            color: case.color[side],
            fog: case.fog,
        };
        let quad = [
            corner(left, top, 0, 0.0, 0.0),
            corner(left + size, top, 1, 1.0, 0.0),
            corner(left, top + size, 0, 0.0, 1.0),
            corner(left + size, top + size, 1, 1.0, 1.0),
        ];
        // Safe: the pool gave room for 4 vertices and 6 indices.
        unsafe {
            core::ptr::copy_nonoverlapping(quad.as_ptr(), vertices, 4);
            core::ptr::copy_nonoverlapping([0u16, 1, 2, 1, 3, 2].as_ptr(), indices, 6);
            gxm::sceGxmSetVertexStream(context(), 0, vertices as *const c_void);
            gxm::sceGxmDraw(context(), gxm::PRIMITIVE_TRIANGLES, gxm::INDEX_FORMAT_U16, indices as *const c_void, 6);
        }
    }

    /// The memory layout GXM wants for a linear
    /// texture's mip levels. A 64x64 texture with 7 levels, each level one
    /// colour (0 red, 1 green, 2 blue, 3 yellow, 4 cyan, 5 magenta, 6
    /// white), laid out as texture::mip_level_offset says. Row 1: ABGR8888;
    /// row 2: P8 with a palette whose entry n is level n's colour. Seven
    /// squares per row, the texture repeated 2, 4, ... 128 times across each,
    /// so the GPU (point mip filter) samples level 0, 1, ... 6. A right
    /// layout gives seven clean colours per row. In the test block.
    fn draw_mip_test(&mut self) {
        const COLORS: [u32; 7] = [0xFF00_00FF, 0xFF00_FF00, 0xFFFF_0000, 0xFF00_FFFF, 0xFFFF_FF00, 0xFFFF_00FF, 0xFFFF_FFFF];
        let block = self.test_block();
        if block.is_null() {
            return;
        }
        // 8888 texture at 0, P8 texture at 64 KiB, its palette at 96 KiB.
        // Safe: all inside the 1 MiB block.
        let (rgba, indexed, palette) = unsafe { (block, block.add(0x10000), block.add(0x18000) as *mut u32) };
        for (level, &color) in COLORS.iter().enumerate() {
            let (w, h) = ((64u32 >> level).max(1), (64u32 >> level).max(1));
            let texels = (w.div_ceil(8) * 8 * h) as usize;
            // Safe: inside the block (7 levels of 64x64 take < 64 KiB).
            unsafe {
                let level32 = rgba.add(texture::mip_level_offset(64, 64, 4, level as u32) as usize) as *mut u32;
                let level8 = indexed.add(texture::mip_level_offset(64, 64, 1, level as u32) as usize);
                for i in 0..texels {
                    level32.add(i).write(color);
                    level8.add(i).write(level as u8);
                }
                palette.add(level).write(color);
            }
        }
        let mut textures = [gxm::Texture { control_words: [0; 4] }; 2];
        // Safe: our texture structs over the block.
        unsafe {
            gxm::sceGxmTextureInitLinear(&mut textures[0], rgba as *const c_void, gxm::TEXTURE_FORMAT_U8U8U8U8_ABGR, 64, 64, 7);
            gxm::sceGxmTextureInitLinear(&mut textures[1], indexed as *const c_void, gxm::TEXTURE_FORMAT_P8_ABGR, 64, 64, 7);
            gxm::sceGxmTextureSetPalette(&mut textures[1], palette as *const c_void);
            MIP_TEST_DEFAULT_BIAS.store(gxm::sceGxmTextureGetLodBias(&textures[0]), Ordering::Relaxed);
            let bias = MIP_TEST_BIAS.load(Ordering::Relaxed);
            if bias >= 0 {
                gxm::sceGxmTextureSetLodBias(&mut textures[0], bias as u32);
                gxm::sceGxmTextureSetLodBias(&mut textures[1], bias as u32);
            }
        }
        self.prepare_test();
        // The test's own matrix (raw draws set theirs).
        self.state.vertex_program = null();
        let Some(basic) = basic_programs::get() else {
            return;
        };
        self.use_programs(basic.texture_vertex, basic.texture_wvp, basic_programs::ORTHO.as_ptr(), basic.texture_tint_fragment);
        self.use_fragment_color(basic.tint_color, [1.0; 4]);
        for (row, texture) in textures.iter().enumerate() {
            let mut texture = *texture;
            // Safe: our copy of the texture state.
            unsafe {
                gxm::sceGxmTextureSetMinFilter(&mut texture, gxm::TEXTURE_FILTER_POINT);
                gxm::sceGxmTextureSetMagFilter(&mut texture, gxm::TEXTURE_FILTER_POINT);
                gxm::sceGxmTextureSetMipFilter(&mut texture, gxm::TEXTURE_MIP_FILTER_DISABLED); // point between levels
                gxm::sceGxmTextureSetUAddrMode(&mut texture, gxm::TEXTURE_ADDR_REPEAT);
                gxm::sceGxmTextureSetVAddrMode(&mut texture, gxm::TEXTURE_ADDR_REPEAT);
            }
            self.use_texture(&texture);
            for k in 0..7 {
                let vertices = frame_pool_alloc(4 * core::mem::size_of::<basic_programs::TextureVertex>() as u32, 8)
                    as *mut basic_programs::TextureVertex;
                let indices = frame_pool_alloc(6 * 2, 2) as *mut u16;
                if vertices.is_null() || indices.is_null() {
                    return;
                }
                let (left, top, size) = (30.0 + 130.0 * k as f32, 120.0 + 160.0 * row as f32, 128.0);
                let span = (2u32 << k) as f32; // 64 * span texels over 128 pixels: level k
                let quad = [
                    basic_programs::TextureVertex { x: left, y: top, z: 0.5, w: 1.0, u: 0.0, v: 0.0 },
                    basic_programs::TextureVertex { x: left + size, y: top, z: 0.5, w: 1.0, u: span, v: 0.0 },
                    basic_programs::TextureVertex { x: left, y: top + size, z: 0.5, w: 1.0, u: 0.0, v: span },
                    basic_programs::TextureVertex { x: left + size, y: top + size, z: 0.5, w: 1.0, u: span, v: span },
                ];
                // Safe: the pool gave room for 4 vertices and 6 indices.
                unsafe {
                    core::ptr::copy_nonoverlapping(quad.as_ptr(), vertices, 4);
                    core::ptr::copy_nonoverlapping([0u16, 1, 2, 1, 3, 2].as_ptr(), indices, 6);
                    gxm::sceGxmSetVertexStream(context(), 0, vertices as *const c_void);
                    gxm::sceGxmDraw(context(), gxm::PRIMITIVE_TRIANGLES, gxm::INDEX_FORMAT_U16, indices as *const c_void, 6);
                }
            }
        }
    }
}

const TEST_BLOCK_BYTES: usize = 65536 * core::mem::size_of::<basic_programs::ColorVertex>();
#[no_mangle]
#[used]
#[allow(non_upper_case_globals)]
pub static nativeres_plan_phycont_test_block: u32 = sys::pages(TEST_BLOCK_BYTES);

/// Which raw program set a raw draw uses.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RawSetChoice {
    /// nativeres_raw_v.cg or one of its variants.
    Raw(RawVariant),
    /// nativeres_raw_sprite_v.cg.
    Sprites,
}

/// The raw program for a draw with `weights` skinning weights (0 without),
/// lit with directional lights or not: the smallest that takes it
/// (nativeres_raw_v.cg's variants). `variants` off (debug `rawvariants 0`):
/// only the plain and the full program, as before 2026-11-17.
fn raw_variant(weights: u32, lit_with_lights: bool, plain: bool, variants: bool) -> RawVariant {
    if plain && weights <= 1 && !lit_with_lights {
        RawVariant::Plain
    } else if variants && weights <= 1 {
        RawVariant::Lit1
    } else if variants && weights <= 4 {
        RawVariant::Lit4
    } else {
        RawVariant::Full
    }
}

/// Whether the flat programs (`RawVariant::Lit1Flat` / `Lit4Flat`,
/// nativeres_raw_v.cg RAW_FLAT_LIGHTS) light with `lights` exactly as the
/// general ones: the specular coefficient is 0 or less (the GE's pow is 1
/// for every light) and the fourth light is all zeros (direction, colours,
/// flags: it adds exactly 0; unused lights are zeros, `Lights::gpu`).
fn flat_lights(lights: &GpuLights) -> bool {
    let coefficient = lights.emissive[3];
    let zeros = |values: &[f32; 4]| values.iter().all(|value| value.to_bits() == 0);
    coefficient <= 0.0 && zeros(&lights.to_light[3]) && zeros(&lights.diffuse[3]) && zeros(&lights.specular[3])
}

/// The raw programs' uniforms beyond the ones the sprite program shares
/// (the plain program has no lights).
#[derive(Clone, Copy)]
struct RawSkinning {
    variant: RawVariant,
    scale: *const ProgramParameter,
    lights: *const ProgramParameter,
    skin: *const ProgramParameter,
    bones: *const ProgramParameter,
}

fn raw_skinning(set: &crate::programs::RawSet) -> RawSkinning {
    RawSkinning { variant: set.variant, scale: set.scale, lights: set.lights, skin: set.skin, bones: set.bones }
}

/// The uniforms all raw programs (and the sprite program) share.
#[allow(clippy::too_many_arguments)]
fn set_raw_uniforms(
    buffer: *mut c_void,
    uniforms: &RawUniforms,
    matrix: *const f32,
    (transform, camera): (&RawTransform, &RawCamera),
    uv_scale: [f32; 2],
    format: &[f32; 4],
    raw_uv: &[f32; 4],
    colors: &RawColors,
) {
    set_uniform(buffer, uniforms.wvp, 16, matrix);
    set_uniform(buffer, uniforms.world, 12, floats(&transform.world));
    set_uniform(buffer, uniforms.view, 12, floats(&camera.view));
    set_uniform(buffer, uniforms.projection, 16, floats(&camera.projection));
    set_uniform(buffer, uniforms.fog, 4, floats(&camera.fog));
    set_uniform(buffer, uniforms.viewport, 4, floats(&camera.viewport));
    set_uniform(buffer, uniforms.uv_scale, 2, uv_scale.as_ptr());
    set_uniform(buffer, uniforms.format, 4, format.as_ptr());
    set_uniform(buffer, uniforms.raw_uv, 4, raw_uv.as_ptr());
    set_uniform(buffer, uniforms.material, 4, floats(&colors.material));
    set_uniform(buffer, uniforms.ambient, 4, floats(&colors.ambient));
    set_uniform(buffer, uniforms.emissive, 4, floats(&colors.emissive));
    set_uniform(buffer, uniforms.colors, 4, floats(&colors.rules));
}

/// Floats of one light set (`GpuLights`), the lights uniform's size.
const LIGHT_FLOATS: u32 = (core::mem::size_of::<GpuLights>() / 4) as u32;

/// The lights of a GPU-lit transform (`GpuTransform::light_set`, 0: none).
fn light_set<'a>(frame: &'a Frame, transform: &GpuTransform) -> Option<&'a GpuLights> {
    light_set_number(frame, transform.light_set[0] as u32)
}

/// The lights of a raw draw lit by the GPU (`RawTransform::light_set`, 0: none).
fn raw_light_set<'a>(frame: &'a Frame, transform: &RawTransform) -> Option<&'a GpuLights> {
    light_set_number(frame, transform.light_set as u32)
}

fn light_set_number(frame: &Frame, set: u32) -> Option<&GpuLights> {
    if set == 0 || set > frame.light_set_count {
        return None;
    }
    frame.light_sets.get(set as usize - 1)
}

/// Depth: the PSP's depth value d (0..65535, `FrameVertex::z` = d / 65535)
/// must reach the GPU in the same order. The projection (ORTHO) turns a vertex
/// z into clip z = 1 - 2z (w = 1), so z = (1 - d) / 2 gives clip z / w = d.
fn gpu_depth(psp_depth: f32) -> f32 {
    (1.0 - psp_depth) * 0.5
}

/// The triangle corners of a draw's index runs (type, count, first vertex).
///
/// The run words are read with volatile reads: as an iterator sum, LLVM
/// vectorized this into NEON code that moves single 16-bit lanes between
/// NEON and ARM registers, which is slow on the Cortex-A9 (a draw has only a
/// few runs); with it the Rust draw loop took ~0.2 ms more per MTX race frame
/// than the C one (measured 2026-11-13).
fn run_corners(runs: &[u16]) -> u32 {
    let mut corners = 0;
    for run in runs.chunks_exact(3) {
        // Safe: plain reads of valid elements.
        let (primitive, count) = unsafe { (core::ptr::read_volatile(&run[0]), core::ptr::read_volatile(&run[1])) };
        corners += 3 * run_triangles(primitive, count);
    }
    corners
}

/// Writes index runs (type, count, first vertex) out as triangle lists, in
/// the order frame.rs run_indices gives.
fn write_runs(runs: &[u16], out: &mut [u16]) {
    let mut at = 0;
    for run in runs.chunks_exact(3) {
        let (primitive, count, base) = (run[0], run[1], run[2]);
        let triangles = run_triangles(primitive, count);
        match primitive {
            PRIM_TRIANGLES => {
                for t in 0..3 * triangles {
                    out[at] = base.wrapping_add(t as u16);
                    at += 1;
                }
            }
            PRIM_SPRITES => {
                // Corners 0 1 2, 1 3 2 of each sprite.
                for s in 0..count {
                    let c = base.wrapping_add(4 * s);
                    let corner = |k: u16| c.wrapping_add(k);
                    out[at..at + 6].copy_from_slice(&[c, corner(1), corner(2), corner(1), corner(3), corner(2)]);
                    at += 6;
                }
            }
            PRIM_TRIANGLE_STRIP => {
                // Every second triangle's first two corners swapped: one
                // winding for all (frame.rs strip_triangle, culling).
                for t in 0..triangles as u16 {
                    let odd = t & 1;
                    let first = base.wrapping_add(t);
                    out[at..at + 3].copy_from_slice(&[first.wrapping_add(odd), first.wrapping_add(1 - odd), first.wrapping_add(2)]);
                    at += 3;
                }
            }
            _ => {
                // fan
                for t in 0..triangles as u16 {
                    out[at..at + 3].copy_from_slice(&[base, base.wrapping_add(t + 1), base.wrapping_add(t + 2)]);
                    at += 3;
                }
            }
        }
    }
}

/// Smooth mode filters bilinearly. Sprites and tiles are cut from texture
/// atlases, so at a quad's edge the filter would blend in the neighbouring
/// sprite, which shows as seams between tiles. Pull each quad's texture
/// coordinates in by half a texel so the filter stays inside its sprite.
fn keep_filter_inside_sprites(vertices: &mut [FrameVertex], texture: &gxm::Texture) {
    // Safe: a texture we made.
    let (width, height) = unsafe { (gxm::sceGxmTextureGetWidth(texture), gxm::sceGxmTextureGetHeight(texture)) };
    let half_u = 0.5 / width as f32;
    let half_v = 0.5 / height as f32;
    for quad in vertices.chunks_exact_mut(6) {
        let (mut u_min, mut u_max, mut v_min, mut v_max) = (quad[0].u, quad[0].u, quad[0].v, quad[0].v);
        for vertex in quad.iter() {
            u_min = u_min.min(vertex.u);
            u_max = u_max.max(vertex.u);
            v_min = v_min.min(vertex.v);
            v_max = v_max.max(vertex.v);
        }
        for vertex in quad.iter_mut() {
            if u_max - u_min > 2.0 * half_u {
                vertex.u = vertex.u.max(u_min + half_u).min(u_max - half_u);
            }
            if v_max - v_min > 2.0 * half_v {
                vertex.v = vertex.v.max(v_min + half_v).min(v_max - half_v);
            }
        }
    }
}


// ------------------------------------------------------------------------
// For the frame loop, the targets and the benches (frame_loop.rs,
// targets.rs, unswizzle.rs, benches.rs), all on the display thread
// ------------------------------------------------------------------------

/// A scene began (other code drew in between): the GXM state is unknown.
pub fn forget_gxm_state() {
    draws().forget_state();
}

/// The basic texture-tint fragment program writing all four channels as
/// they are, for our own passes (unswizzle, target copies, presenting);
/// null if it can't be made.
pub fn copy_program() -> *mut c_void {
    draws().programs.blend_program(Kind::TextureTint, COPY_BLEND, true)
}

/// This frame's decision: draws keep the target's alpha (the PSP's
/// stencil), see frame_loop.rs.
pub fn set_keep_target_alpha(keep: bool) {
    draws().keep_target_alpha = keep;
}

/// A draw's scissor, depth test and culling.
pub fn draw_state(draw: &Draw) {
    let draws = draws();
    draws.set_scissor(draw);
    draws.use_depth(draw);
    draws.use_cull(draw);
}

/// Textured draw `draw_index` of `frame` with `texture` (one of our texture
/// objects or a render target's), the colour table `palette` (index
/// textures; else null) and `uv_scale`. Inside a target scene.
pub fn draw_textured(frame: &Frame, draw: &Draw, draw_index: usize, texture: &TextureObject, palette: *const c_void, uv_scale: [f32; 2]) {
    draws().draw_textured(frame, draw, draw_index, texture, palette, uv_scale);
}

/// Untextured draw `draw_index` of `frame`. Inside a target scene.
pub fn draw_colored(frame: &Frame, draw: &Draw, draw_index: usize) {
    draws().draw_colored(frame, draw, draw_index);
}

/// After a frame's draws: the GPU tests (index, mip) and the
/// alpha log.
pub fn draw_tests() {
    let draws = draws();
    if INDEX_TEST_FRAMES.load(Ordering::Relaxed) > 0 {
        INDEX_TEST_FRAMES.fetch_sub(1, Ordering::Relaxed);
        draws.draw_index_test();
    }
    if MIP_TEST_FRAMES.load(Ordering::Relaxed) > 0 {
        MIP_TEST_FRAMES.fetch_sub(1, Ordering::Relaxed);
        draws.draw_mip_test();
    }
    if SHADE_TEST_FRAMES.load(Ordering::Relaxed) > 0 {
        SHADE_TEST_FRAMES.fetch_sub(1, Ordering::Relaxed);
        draws.draw_shade_test();
    }
    if ALPHA_LOG.load(Ordering::Relaxed) != 0 {
        draws.write_alpha_log();
        ALPHA_LOG.store(0, Ordering::Relaxed);
    }
}

/// Before the target scene ends: GXM's defaults again, which Adrenaline's vita2d draws
/// with (all at z 0.5, no depth test, no culling, no clip).
pub fn end_draws() {
    let draws = draws();
    draws.use_default_viewport();
    // Safe: the display thread's context.
    unsafe { gxm::sceGxmSetRegionClip(context(), gxm::REGION_CLIP_NONE, 0, 0, 0, 0) };
    draws.state.scissor = None;
    draws.set_depth(gxm::DEPTH_FUNC_ALWAYS, gxm::DEPTH_WRITE_ENABLED);
    draws.set_cull(gxm::CULL_NONE);
}

/// The lod_bias field sceGxmTextureInitLinear left (the mip test reports it).
pub fn mip_test_default_bias() -> u32 {
    MIP_TEST_DEFAULT_BIAS.load(Ordering::Relaxed)
}

/// Makes the arenas the capture records into (once, at the first drawn
/// frame that wants them; "vertexcache 0|1|2": 1 main memory, 2 phycont
/// with the raw arena): the raw arena not without the raw vertex program;
/// 3D sprites go raw only once their program exists; zero copy only once
/// the GPU may read PSP RAM (all in arenas.rs and capture.rs).
pub fn make_arenas(phycont: bool) {
    let draws = draws();
    if phycont {
        if !draws.programs.raw.ready() {
            render_log::line(format_args!("raw arena: no raw vertex program\n"));
        } else if arenas::make_raw_arena() != 0 {
            crate::capture::set_raw_sprites(draws.programs.sprites.ready() as u32);
            crate::capture::set_zero_copy(arenas::map_psp_ram_for_gpu() as u32);
        }
    }
    arenas::make_vertex_arena(phycont as i32);
}
