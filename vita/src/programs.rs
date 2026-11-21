//! Our GPU programs: registered with our shader patcher (basic_programs.rs)
//! on first use, and the fragment programs per blend mode.
//!
//! The basic programs (basic_programs.rs) take a position and texture
//! coordinates, plus one tint for the whole draw. PSP draws need more per
//! vertex. Our programs (renderer/vita/shaders/<name>.cg, embedded by
//! shaders.rs) take per vertex:
//! position with w (perspective), texture coordinates, colour and the PSP
//! fog factor. They draw all textured draws; with the colour per vertex,
//! draws that differ only in their colour (Final Fantasy's menu glyphs) can
//! be joined.
//!
//! - the textured program (nativeres_textured_v/_f.cg), with the alpha-test
//!   and FIX B variants of the fragment program;
//! - the 3D program (nativeres_3d_v.cg): the same outputs for vertices in
//!   model space (the GPU transform and lighting);
//! - the raw programs (nativeres_raw_v.cg and its variants _raw_plain_v,
//!   _raw_lit1_v, _raw_lit4_v, _raw_lit1flat_v, _raw_lit4flat_v): raw PSP vertices, one patched program per
//!   PSP vertex format;
//! - the raw sprite programs (nativeres_raw_sprite_v.cg);
//! - blend programs: GXM bakes blending into the fragment program, so every
//!   blend mode a game uses needs its own; made on first use and kept.
//!
//! Display thread only (draw.rs owns the one `Programs`).

use core::ffi::c_void;
use core::mem::offset_of;
use core::ptr::{null, null_mut};

use nativeres_core::blend::{BLEND_ALPHA_ONLY, BLEND_ENABLED, BLEND_FIX_B_ALPHA, BLEND_WRITES_ALPHA};
use nativeres_core::frame::{FrameVertex, ReadVariant, ReadWay};
use nativeres_core::vertex::VertexFormat;

use crate::gxm::{self, PatcherId, ProgramParameter, VertexAttribute, VertexStream};
use crate::basic_programs::{self, gxp, parameter, register, resource};
use crate::{render_log, shaders};

/// The PSP's blend words' FIX B value and premultiply bits: uniforms of the
/// BLEND_FIX programs, not part of a program (NATIVERES_BLEND_FIX_B_VALUES).
pub const BLEND_FIX_B_VALUES: u32 = 0x3FF << 12;

/// Which fragment program: the basic colour and texture-tint programs, our
/// textured program without and with the PSP's alpha test, and both with
/// grey FIX B blending through the alpha (BLEND_FIX).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Color,
    TextureTint,
    Textured,
    /// `Textured` for draws without fog and without a separate specular
    /// colour (nativeres_textured_flat_f.cg): the same pixels, less work.
    TexturedFlat,
    TexturedTest,
    TexturedFix,
    TexturedTestFix,
}

/// Formats per program set (MTX uses a handful).
const RAW_PROGRAMS: usize = 32;
/// Blend programs kept.
const MAX_BLEND_PROGRAMS: usize = 96;

/// Where `format` is in `formats`. Looked up for every raw draw: as an
/// iterator search, LLVM vectorized it into NEON compares whose results went
/// back to ARM registers lane by lane (slow on the Cortex-A9; a set has a
/// handful of formats). Volatile reads keep it a plain loop.
fn find_format(formats: &[u32], format: u32) -> Option<usize> {
    // Safe: plain reads of valid elements.
    (0..formats.len()).find(|&index| unsafe { core::ptr::read_volatile(&formats[index]) } == format)
}

fn attribute(stream: u16, offset: u32, format: u8, count: u8, parameter: *const ProgramParameter) -> VertexAttribute {
    VertexAttribute { stream_index: stream, offset: offset as u16, format, component_count: count, register_index: resource(parameter) }
}

/// Makes a vertex program from registered program `id`; null if it fails
/// (logged with `what`).
fn vertex_program(id: PatcherId, attributes: &[VertexAttribute], streams: &[VertexStream], what: core::fmt::Arguments) -> *mut c_void {
    match basic_programs::vertex_program(id, attributes, streams) {
        Ok(program) => program,
        Err(result) => {
            render_log::line(format_args!("{}: vertex program {:08x}
", what, result));
            null_mut()
        }
    }
}

/// The textured, 3D and fragment variants' parameters.
pub struct Textured {
    tried: bool,
    vertex_id: PatcherId,
    fragment_id: PatcherId,
    pub vertex: *mut gxm::VertexProgram,
    pub wvp: *const ProgramParameter,
    pub fog_and_scale: *const ProgramParameter,
    /// nativeres_textured_flat_f.cg: without fog and specular.
    pub have_flat: bool,
    fragment_flat_id: PatcherId,
    pub fog_and_scale_flat: *const ProgramParameter,
    /// nativeres_textured_test_f.cg: the fragment program with the alpha test.
    pub have_test: bool,
    fragment_test_id: PatcherId,
    pub fog_and_scale_test: *const ProgramParameter,
    pub alpha_test: *const ProgramParameter,
    /// nativeres_textured_(test_)fix_f.cg: grey FIX B blending through the alpha.
    pub have_fix: bool,
    fragment_fix_id: PatcherId,
    fragment_test_fix_id: PatcherId,
    pub fog_and_scale_fix: *const ProgramParameter,
    pub blend_fix: *const ProgramParameter,
    pub fog_and_scale_test_fix: *const ProgramParameter,
    pub alpha_test_fix: *const ProgramParameter,
    pub blend_fix_test: *const ProgramParameter,
    /// nativeres_3d_v.cg; null if it couldn't be made (the capture then
    /// transforms on the CPU).
    pub vertex_3d: *mut gxm::VertexProgram,
    pub wvp_3d: *const ProgramParameter,
    pub rows_3d: *const ProgramParameter,
    pub fog_row_3d: *const ProgramParameter,
    pub viewport_3d: *const ProgramParameter,
    pub depth_3d: *const ProgramParameter,
    pub uv_scale_3d: *const ProgramParameter,
    pub normal_rows_3d: *const ProgramParameter,
    pub lights_3d: *const ProgramParameter,
}

/// Which raw program: nativeres_raw_v.cg or one of its variants (see the
/// shader's comment). Each takes the draws that need no more than it has;
/// the smaller the program, the less the GPU spends per draw and vertex.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum RawVariant {
    /// Up to eight bones, lights (nativeres_raw_v.cg).
    Full,
    /// One bone, no directional lights (nativeres_raw_plain_v.cg).
    Plain,
    /// Up to four bones, no directional lights (nativeres_raw_plain4_v.cg).
    Plain4,
    /// One bone, lights (nativeres_raw_lit1_v.cg). The lit variants take
    /// only draws lit with directional lights (`lights_only`).
    Lit1,
    /// Up to four bones, lights (nativeres_raw_lit4_v.cg).
    Lit4,
    /// `Lit1` / `Lit4` for flat light sets: specular coefficient <= 0 and
    /// at most three lights (nativeres_raw_lit1flat_v.cg, _lit4flat_v.cg;
    /// draw.rs `flat_lights`).
    Lit1Flat,
    Lit4Flat,
}

impl RawVariant {
    /// The most bones (skinning weights) the program takes.
    pub fn bones(self) -> u32 {
        match self {
            RawVariant::Full => 8,
            RawVariant::Plain4 | RawVariant::Lit4 | RawVariant::Lit4Flat => 4,
            RawVariant::Plain | RawVariant::Lit1 | RawVariant::Lit1Flat => 1,
        }
    }

    /// Whether the program lights with directional lights (uLights, aNormal).
    pub fn lights(self) -> bool {
        !matches!(self, RawVariant::Plain | RawVariant::Plain4)
    }

    /// Whether the program takes only draws lit with directional lights
    /// (nativeres_raw_v.cg RAW_LIGHTS_ONLY: without the unlit colour rules,
    /// so without uAmbient, uEmissive, uScale).
    pub fn lights_only(self) -> bool {
        self.lights() && self != RawVariant::Full
    }

    fn label(self) -> &'static str {
        match self {
            RawVariant::Full => "",
            RawVariant::Plain => " (plain)",
            RawVariant::Plain4 => " (plain, four bones)",
            RawVariant::Lit1 => " (lit, one bone)",
            RawVariant::Lit4 => " (lit, four bones)",
            RawVariant::Lit1Flat => " (flat lights, one bone)",
            RawVariant::Lit4Flat => " (flat lights, four bones)",
        }
    }

    /// The place among the variants with fixed-colour programs
    /// (`fixed_raw_sets`); None for the full program.
    fn fixed_index(self) -> Option<usize> {
        match self {
            RawVariant::Full => None,
            RawVariant::Plain => Some(0),
            RawVariant::Plain4 => Some(1),
            RawVariant::Lit1 => Some(2),
            RawVariant::Lit4 => Some(3),
            RawVariant::Lit1Flat => Some(4),
            RawVariant::Lit4Flat => Some(5),
        }
    }

    /// The same program for a flat light set (`Lit1Flat` / `Lit4Flat`);
    /// the others as they are.
    pub fn flat(self) -> RawVariant {
        match self {
            RawVariant::Lit1 => RawVariant::Lit1Flat,
            RawVariant::Lit4 => RawVariant::Lit4Flat,
            other => other,
        }
    }
}

/// A raw program's vertex colour format (VERTEX_TYPE bits 2-4): any (the
/// program decodes all, by uFormat.y) or one fixed (nativeres_raw_v.cg
/// RAW_COLOR_*: only that format's decode and attribute, 50-70 fewer
/// instructions per vertex, 2026-11-18). The full program is always `Any`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum RawColor {
    Any,
    /// No vertex colour (the material colour; also the unused field values 1-3).
    Material,
    Bits5650,
    Bits5551,
    Bits4444,
    Bits8888,
}

/// The fixed colour formats per variant (`Programs::raw_fixed`).
const FIXED_COLORS: usize = 5;
/// The variants with fixed-colour programs (all but the full one).
const FIXED_VARIANTS: usize = 6;

impl RawColor {
    /// The fixed colour format of VERTEX_TYPE `format`.
    pub fn of_format(format: u32) -> RawColor {
        match (format >> 2) & 7 {
            4 => RawColor::Bits5650,
            5 => RawColor::Bits5551,
            6 => RawColor::Bits4444,
            7 => RawColor::Bits8888,
            _ => RawColor::Material,
        }
    }

    /// Whether a program of `variant` with this colour format reads
    /// uMaterial: not a lit-only one for a vertex colour format (its
    /// lighting takes the vertex colour, and it has no unlit rules).
    fn uses_material(self, variant: RawVariant) -> bool {
        !variant.lights_only() || matches!(self, RawColor::Any | RawColor::Material)
    }

    /// Whether the program reads the two bytes of a 16-bit colour (aColorBytes).
    fn reads_bytes(self) -> bool {
        matches!(self, RawColor::Any | RawColor::Bits5650 | RawColor::Bits5551 | RawColor::Bits4444)
    }

    /// Whether the program reads an 8888 colour (aColor32).
    fn reads_32(self) -> bool {
        matches!(self, RawColor::Any | RawColor::Bits8888)
    }

    /// The place among the fixed formats (`FIXED_COLORS`); None for `Any`.
    fn fixed_index(self) -> Option<usize> {
        match self {
            RawColor::Any => None,
            RawColor::Material => Some(0),
            RawColor::Bits5650 => Some(1),
            RawColor::Bits5551 => Some(2),
            RawColor::Bits4444 => Some(3),
            RawColor::Bits8888 => Some(4),
        }
    }

    fn label(self) -> &'static str {
        match self {
            RawColor::Any => "",
            RawColor::Material => " [no colour]",
            RawColor::Bits5650 => " [5650]",
            RawColor::Bits5551 => " [5551]",
            RawColor::Bits4444 => " [4444]",
            RawColor::Bits8888 => " [8888]",
        }
    }
}

/// The compiled program of `variant` with colour format `color` (the full
/// program has only the generic one).
fn raw_gxp(variant: RawVariant, color: RawColor) -> *const gxm::Program {
    use RawColor::*;
    use RawVariant::*;
    match (variant, color) {
        (Full, _) => gxp(&shaders::nativeres_raw_v),
        (Plain, Any) => gxp(&shaders::nativeres_raw_plain_v),
        (Plain, Material) => gxp(&shaders::nativeres_raw_plain_none_v),
        (Plain, Bits5650) => gxp(&shaders::nativeres_raw_plain_5650_v),
        (Plain, Bits5551) => gxp(&shaders::nativeres_raw_plain_5551_v),
        (Plain, Bits4444) => gxp(&shaders::nativeres_raw_plain_4444_v),
        (Plain, Bits8888) => gxp(&shaders::nativeres_raw_plain_8888_v),
        (Plain4, Any) => gxp(&shaders::nativeres_raw_plain4_v),
        (Plain4, Material) => gxp(&shaders::nativeres_raw_plain4_none_v),
        (Plain4, Bits5650) => gxp(&shaders::nativeres_raw_plain4_5650_v),
        (Plain4, Bits5551) => gxp(&shaders::nativeres_raw_plain4_5551_v),
        (Plain4, Bits4444) => gxp(&shaders::nativeres_raw_plain4_4444_v),
        (Plain4, Bits8888) => gxp(&shaders::nativeres_raw_plain4_8888_v),
        (Lit1, Any) => gxp(&shaders::nativeres_raw_lit1_v),
        (Lit1, Material) => gxp(&shaders::nativeres_raw_lit1_none_v),
        (Lit1, Bits5650) => gxp(&shaders::nativeres_raw_lit1_5650_v),
        (Lit1, Bits5551) => gxp(&shaders::nativeres_raw_lit1_5551_v),
        (Lit1, Bits4444) => gxp(&shaders::nativeres_raw_lit1_4444_v),
        (Lit1, Bits8888) => gxp(&shaders::nativeres_raw_lit1_8888_v),
        (Lit4, Any) => gxp(&shaders::nativeres_raw_lit4_v),
        (Lit4, Material) => gxp(&shaders::nativeres_raw_lit4_none_v),
        (Lit4, Bits5650) => gxp(&shaders::nativeres_raw_lit4_5650_v),
        (Lit4, Bits5551) => gxp(&shaders::nativeres_raw_lit4_5551_v),
        (Lit4, Bits4444) => gxp(&shaders::nativeres_raw_lit4_4444_v),
        (Lit4, Bits8888) => gxp(&shaders::nativeres_raw_lit4_8888_v),
        (Lit1Flat, Any) => gxp(&shaders::nativeres_raw_lit1flat_v),
        (Lit1Flat, Material) => gxp(&shaders::nativeres_raw_lit1flat_none_v),
        (Lit1Flat, Bits5650) => gxp(&shaders::nativeres_raw_lit1flat_5650_v),
        (Lit1Flat, Bits5551) => gxp(&shaders::nativeres_raw_lit1flat_5551_v),
        (Lit1Flat, Bits4444) => gxp(&shaders::nativeres_raw_lit1flat_4444_v),
        (Lit1Flat, Bits8888) => gxp(&shaders::nativeres_raw_lit1flat_8888_v),
        (Lit4Flat, Any) => gxp(&shaders::nativeres_raw_lit4flat_v),
        (Lit4Flat, Material) => gxp(&shaders::nativeres_raw_lit4flat_none_v),
        (Lit4Flat, Bits5650) => gxp(&shaders::nativeres_raw_lit4flat_5650_v),
        (Lit4Flat, Bits5551) => gxp(&shaders::nativeres_raw_lit4flat_5551_v),
        (Lit4Flat, Bits4444) => gxp(&shaders::nativeres_raw_lit4flat_4444_v),
        (Lit4Flat, Bits8888) => gxp(&shaders::nativeres_raw_lit4flat_8888_v),
    }
}

/// The fixed-colour raw sets in `Programs::raw_fixed`'s order: per variant
/// (`RawVariant::fixed_index`) its five colour formats.
const fn fixed_raw_sets() -> [RawSet; FIXED_VARIANTS * FIXED_COLORS] {
    const VARIANTS: [RawVariant; FIXED_VARIANTS] = [
        RawVariant::Plain,
        RawVariant::Plain4,
        RawVariant::Lit1,
        RawVariant::Lit4,
        RawVariant::Lit1Flat,
        RawVariant::Lit4Flat,
    ];
    const COLORS: [RawColor; FIXED_COLORS] =
        [RawColor::Material, RawColor::Bits5650, RawColor::Bits5551, RawColor::Bits4444, RawColor::Bits8888];
    let mut sets = [const { RawSet::new(RawVariant::Full, RawColor::Any) }; FIXED_VARIANTS * FIXED_COLORS];
    let mut i = 0;
    while i < sets.len() {
        sets[i] = RawSet::new(VARIANTS[i / FIXED_COLORS], COLORS[i % FIXED_COLORS]);
        i += 1;
    }
    sets
}

/// A raw program's name in the log: variant and colour format.
struct Label(RawVariant, RawColor);

impl core::fmt::Display for Label {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.write_str(self.0.label())?;
        f.write_str(self.1.label())
    }
}

/// A raw program (nativeres_raw_v.cg or a variant) and its programs per PSP
/// vertex format.
pub struct RawSet {
    pub variant: RawVariant,
    pub color: RawColor,
    tried: bool,
    ready: bool,
    id: PatcherId,
    position: *const ProgramParameter,
    texcoord: *const ProgramParameter,
    color_bytes: *const ProgramParameter,
    color32: *const ProgramParameter,
    normal: *const ProgramParameter,
    weights0: *const ProgramParameter,
    weights1: *const ProgramParameter,
    /// The one-bone programs' first weight (aWeight).
    weight: *const ProgramParameter,
    pub uniforms: RawUniforms,
    pub lights: *const ProgramParameter,
    pub skin: *const ProgramParameter,
    pub bones: *const ProgramParameter,
    pub scale: *const ProgramParameter,
    formats: [u32; RAW_PROGRAMS],
    programs: [*mut gxm::VertexProgram; RAW_PROGRAMS],
    count: usize,
}

/// The uniforms the raw and sprite programs share (same names).
#[derive(Clone, Copy)]
pub struct RawUniforms {
    pub wvp: *const ProgramParameter,
    pub world: *const ProgramParameter,
    pub view: *const ProgramParameter,
    pub projection: *const ProgramParameter,
    pub fog: *const ProgramParameter,
    pub viewport: *const ProgramParameter,
    pub uv_scale: *const ProgramParameter,
    pub format: *const ProgramParameter,
    pub raw_uv: *const ProgramParameter,
    pub material: *const ProgramParameter,
    pub ambient: *const ProgramParameter,
    pub emissive: *const ProgramParameter,
    pub colors: *const ProgramParameter,
}

impl RawUniforms {
    const NONE: RawUniforms = RawUniforms {
        wvp: null(),
        world: null(),
        view: null(),
        projection: null(),
        fog: null(),
        viewport: null(),
        uv_scale: null(),
        format: null(),
        raw_uv: null(),
        material: null(),
        ambient: null(),
        emissive: null(),
        colors: null(),
    };

    fn find(program: *const gxm::Program) -> RawUniforms {
        RawUniforms {
            wvp: parameter(program, b"wvp\0"),
            world: parameter(program, b"uWorld\0"),
            view: parameter(program, b"uView\0"),
            projection: parameter(program, b"uProj\0"),
            fog: parameter(program, b"uFog\0"),
            viewport: parameter(program, b"uViewport\0"),
            uv_scale: parameter(program, b"uUvScale\0"),
            format: parameter(program, b"uFormat\0"),
            raw_uv: parameter(program, b"uRawUv\0"),
            material: parameter(program, b"uMaterial\0"),
            ambient: parameter(program, b"uAmbient\0"),
            emissive: parameter(program, b"uEmissive\0"),
            colors: parameter(program, b"uColors\0"),
        }
    }

    /// All found but the unlit colour rules' uAmbient and uEmissive (the
    /// lit-only raw programs have none: `unlit_rules_found`) and uMaterial
    /// (not in lit-only programs for a vertex colour format: `RawColor::uses_material`).
    fn all_found(&self) -> bool {
        [
            self.wvp,
            self.world,
            self.view,
            self.projection,
            self.fog,
            self.viewport,
            self.uv_scale,
            self.format,
            self.raw_uv,
            self.colors,
        ]
        .iter()
        .all(|parameter| !parameter.is_null())
    }

    /// uAmbient and uEmissive found (programs with the unlit colour rules).
    fn unlit_rules_found(&self) -> bool {
        !self.ambient.is_null() && !self.emissive.is_null()
    }
}

/// The raw sprite program (nativeres_raw_sprite_v.cg) and its programs per
/// format, plus the corner table.
pub struct SpriteSet {
    tried: bool,
    ready: bool,
    id: PatcherId,
    position: *const ProgramParameter,
    texcoord: *const ProgramParameter,
    position_b: *const ProgramParameter,
    texcoord_b: *const ProgramParameter,
    color_bytes_b: *const ProgramParameter,
    color32_b: *const ProgramParameter,
    corner: *const ProgramParameter,
    pub uniforms: RawUniforms,
    /// Vertex k's corner number, k % 4, for every vertex number a raw window
    /// can hold (192 KiB, at least 3 bytes per vertex, two per pair: <
    /// 32768). GPU-readable, filled once.
    pub corners: *mut u8,
    formats: [u32; RAW_PROGRAMS],
    programs: [*mut gxm::VertexProgram; RAW_PROGRAMS],
    count: usize,
}

pub const SPRITE_CORNERS: usize = 32768;
#[no_mangle]
#[used]
#[allow(non_upper_case_globals)]
pub static nativeres_plan_fixed_sprite_corners: u32 = crate::sys::pages(SPRITE_CORNERS);

/// A frame-read fragment program for draws into the PSP's 16-bit
/// framebuffers (nativeres_textured_read*_f.cg, the untextured
/// nativeres_color_read*_f.cg): it reads the pixel under it, blends itself
/// and cuts the result to the target's bits as the GE does. Made without
/// GXM blending. The generic one does every blend mode (uniforms); the fast
/// ones one blend each, 5551 only (core frame.rs `read_variant`); the fast
/// "off" ones (no blending) read no frame.
pub struct ReadProgram {
    tried: bool,
    pub program: *mut gxm::FragmentProgram,
    /// Textured only (null for the untextured ones).
    pub fog_and_scale: *const ProgramParameter,
    /// The generic one and the fast tested ones only (else null).
    pub alpha_test: *const ProgramParameter,
    /// The generic one only (else null).
    pub blend_fix: *const ProgramParameter,
    /// The generic one only (null for the fast ones): uSource,
    /// uDestination, uBlend, uMinMax, uStep, uWiden, uWidenLow, uWrite,
    /// uPath (the shader's comment), in this order.
    pub cut: [*const ProgramParameter; 9],
    /// The generic one only (else null; the fast dithered ones read the
    /// matrix from a texture, draw.rs `use_dither_texture`).
    pub dither: [*const ProgramParameter; 4],
}

/// Fast frame-read ways (`ReadWay`, all five).
const READ_WAYS: usize = 5;

/// Frame-read programs (`read_index`): untextured generic + the ways x
/// alpha kept or not x dithered or not, then the 565 ones (ways x dithered
/// or not x fogless or not (textured only); 565 keeps no alpha); textured the same, the fast ones also with
/// and without the alpha test (`ReadWay::Invisible` is untextured only, its
/// textured slots stay unused).
const READ_PROGRAMS: usize = READ_TEXTURED + 1 + 2 * READ_WAYS * 4 + 2 * READ_WAYS * 4;

/// The first textured one in `read_index` (the textured generic program).
const READ_TEXTURED: usize = 1 + READ_WAYS * 4 + READ_WAYS * 4;

/// The place of a frame-read program in `Programs::read`; `tested`: with
/// the alpha test (textured only; the generic one always has it).
fn read_index(textured: bool, tested: bool, variant: ReadVariant) -> usize {
    // 5551: 0..READ_WAYS * 4; 565 after them (never keeping the alpha).
    let fast = |way: ReadWay, keep_alpha: bool, dither: bool, green6: bool, fogless: bool| {
        if green6 {
            READ_WAYS * 4 + (2 * way as usize + dither as usize) * 2 + fogless as usize
        } else {
            (2 * way as usize + keep_alpha as usize) * 2 + dither as usize
        }
    };
    match (textured, variant) {
        (false, ReadVariant::Generic) => 0,
        (false, ReadVariant::Fast { way, keep_alpha, dither, green6, fogless }) => 1 + fast(way, keep_alpha, dither, green6, fogless),
        (true, ReadVariant::Generic) => READ_TEXTURED,
        (true, ReadVariant::Fast { way, keep_alpha, dither, green6, fogless }) => {
            READ_TEXTURED + 1 + 2 * fast(way, keep_alpha, dither, green6, fogless) + tested as usize
        }
    }
}

/// The compiled frame-read program of `variant` (as `read_index`). The
/// "off" ones keep the alpha by GXM's colour mask: one program for both,
/// and for `ReadWay::Precut` (GXM blends).
fn read_gxp(textured: bool, tested: bool, variant: ReadVariant) -> *const gxm::Program {
    use ReadWay::{Add, Invisible, Normal, Off, Precut};
    use shaders::*;
    let ReadVariant::Fast { way, keep_alpha, dither, green6, fogless } = variant else {
        return if textured { gxp(&nativeres_textured_read_f) } else { gxp(&nativeres_color_read_f) };
    };
    if green6 && fogless && textured {
        // Textured 565 draws without fog.
        return match (way, tested, dither) {
            (Normal | Invisible, false, false) => gxp(&nativeres_textured_read_normal565nofog_f),
            (Normal | Invisible, true, false) => gxp(&nativeres_textured_read_normal565nofogtest_f),
            (Normal | Invisible, false, true) => gxp(&nativeres_textured_read_normal565nofogdither_f),
            (Normal | Invisible, true, true) => gxp(&nativeres_textured_read_normal565nofogtestdither_f),
            (Add, false, false) => gxp(&nativeres_textured_read_add565nofog_f),
            (Add, true, false) => gxp(&nativeres_textured_read_add565nofogtest_f),
            (Add, false, true) => gxp(&nativeres_textured_read_add565nofogdither_f),
            (Add, true, true) => gxp(&nativeres_textured_read_add565nofogtestdither_f),
            (Off | Precut, false, false) => gxp(&nativeres_textured_read_off565nofog_f),
            (Off | Precut, true, false) => gxp(&nativeres_textured_read_off565nofogtest_f),
            (Off | Precut, false, true) => gxp(&nativeres_textured_read_off565nofogdither_f),
            (Off | Precut, true, true) => gxp(&nativeres_textured_read_off565nofogtestdither_f),
        };
    }
    if green6 {
        // 565 targets (no keep-alpha programs: `read_variant` never asks).
        return match (textured, way, tested, dither) {
            (false, Normal, _, false) => gxp(&nativeres_color_read_normal565_f),
            (false, Normal, _, true) => gxp(&nativeres_color_read_normal565dither_f),
            (false, Add, _, false) => gxp(&nativeres_color_read_add565_f),
            (false, Add, _, true) => gxp(&nativeres_color_read_add565dither_f),
            (false, Off | Precut, _, false) => gxp(&nativeres_color_read_off565_f),
            (false, Off | Precut, _, true) => gxp(&nativeres_color_read_off565dither_f),
            (false, Invisible, _, false) => gxp(&nativeres_color_read_invisible565_f),
            (false, Invisible, _, true) => gxp(&nativeres_color_read_invisible565dither_f),
            (true, Normal | Invisible, false, false) => gxp(&nativeres_textured_read_normal565_f),
            (true, Normal | Invisible, true, false) => gxp(&nativeres_textured_read_normal565test_f),
            (true, Normal | Invisible, false, true) => gxp(&nativeres_textured_read_normal565dither_f),
            (true, Normal | Invisible, true, true) => gxp(&nativeres_textured_read_normal565testdither_f),
            (true, Add, false, false) => gxp(&nativeres_textured_read_add565_f),
            (true, Add, true, false) => gxp(&nativeres_textured_read_add565test_f),
            (true, Add, false, true) => gxp(&nativeres_textured_read_add565dither_f),
            (true, Add, true, true) => gxp(&nativeres_textured_read_add565testdither_f),
            (true, Off | Precut, false, false) => gxp(&nativeres_textured_read_off565_f),
            (true, Off | Precut, true, false) => gxp(&nativeres_textured_read_off565test_f),
            (true, Off | Precut, false, true) => gxp(&nativeres_textured_read_off565dither_f),
            (true, Off | Precut, true, true) => gxp(&nativeres_textured_read_off565testdither_f),
        };
    }
    if !textured {
        return match (way, keep_alpha, dither) {
            (Normal, false, false) => gxp(&nativeres_color_read_normal_f),
            (Normal, true, false) => gxp(&nativeres_color_read_normalkeep_f),
            (Add, false, false) => gxp(&nativeres_color_read_add_f),
            (Add, true, false) => gxp(&nativeres_color_read_addkeep_f),
            (Off | Precut, _, false) => gxp(&nativeres_color_read_off_f),
            (Normal, false, true) => gxp(&nativeres_color_read_normaldither_f),
            (Normal, true, true) => gxp(&nativeres_color_read_normalkeepdither_f),
            (Add, false, true) => gxp(&nativeres_color_read_adddither_f),
            (Add, true, true) => gxp(&nativeres_color_read_addkeepdither_f),
            (Off | Precut, _, true) => gxp(&nativeres_color_read_offdither_f),
            (Invisible, false, false) => gxp(&nativeres_color_read_invisible_f),
            (Invisible, true, false) => gxp(&nativeres_color_read_invisiblekeep_f),
            (Invisible, false, true) => gxp(&nativeres_color_read_invisibledither_f),
            (Invisible, true, true) => gxp(&nativeres_color_read_invisiblekeepdither_f),
        };
    }
    // `ReadWay::Invisible` is untextured only (a textured draw never gets
    // it); should one come, the normal blend gives the same picture.
    match (way, keep_alpha, tested, dither) {
        (Normal | Invisible, false, false, false) => gxp(&nativeres_textured_read_normal_f),
        (Normal | Invisible, false, true, false) => gxp(&nativeres_textured_read_normaltest_f),
        (Normal | Invisible, true, false, false) => gxp(&nativeres_textured_read_normalkeep_f),
        (Normal | Invisible, true, true, false) => gxp(&nativeres_textured_read_normalkeeptest_f),
        (Add, false, false, false) => gxp(&nativeres_textured_read_add_f),
        (Add, false, true, false) => gxp(&nativeres_textured_read_addtest_f),
        (Add, true, false, false) => gxp(&nativeres_textured_read_addkeep_f),
        (Add, true, true, false) => gxp(&nativeres_textured_read_addkeeptest_f),
        (Off | Precut, _, false, false) => gxp(&nativeres_textured_read_off_f),
        (Off | Precut, _, true, false) => gxp(&nativeres_textured_read_offtest_f),
        (Normal | Invisible, false, false, true) => gxp(&nativeres_textured_read_normaldither_f),
        (Normal | Invisible, false, true, true) => gxp(&nativeres_textured_read_normaltestdither_f),
        (Normal | Invisible, true, false, true) => gxp(&nativeres_textured_read_normalkeepdither_f),
        (Normal | Invisible, true, true, true) => gxp(&nativeres_textured_read_normalkeeptestdither_f),
        (Add, false, false, true) => gxp(&nativeres_textured_read_adddither_f),
        (Add, false, true, true) => gxp(&nativeres_textured_read_addtestdither_f),
        (Add, true, false, true) => gxp(&nativeres_textured_read_addkeepdither_f),
        (Add, true, true, true) => gxp(&nativeres_textured_read_addkeeptestdither_f),
        (Off | Precut, _, false, true) => gxp(&nativeres_textured_read_offdither_f),
        (Off | Precut, _, true, true) => gxp(&nativeres_textured_read_offtestdither_f),
    }
}

impl ReadProgram {
    const fn new() -> ReadProgram {
        ReadProgram {
            tried: false,
            program: null_mut(),
            fog_and_scale: null(),
            alpha_test: null(),
            blend_fix: null(),
            cut: [null(); 9],
            dither: [null(); 4],
        }
    }

    /// Makes the program on first use: `textured` a textured one (linked
    /// to nativeres_textured_v), else an untextured one (nativeres_color_v).
    /// False if it can't be made (the draw then goes the 8-bit way).
    fn ready(&mut self, textured: bool, tested: bool, variant: ReadVariant) -> bool {
        if self.tried {
            return !self.program.is_null();
        }
        self.tried = true;
        let fragment_gxp = read_gxp(textured, tested, variant);
        let (vertex_gxp, label) = if textured {
            (gxp(&shaders::nativeres_textured_v), "textured")
        } else {
            (gxp(&shaders::nativeres_color_v), "untextured")
        };
        let Some(id) = register(fragment_gxp) else {
            render_log::line(format_args!("16-bit target program ({label}, {variant:?}): invalid GXP\n"));
            return false;
        };
        let generic = variant == ReadVariant::Generic;
        if generic {
            self.cut = [
                parameter(fragment_gxp, b"uSource\0"),
                parameter(fragment_gxp, b"uDestination\0"),
                parameter(fragment_gxp, b"uBlend\0"),
                parameter(fragment_gxp, b"uMinMax\0"),
                parameter(fragment_gxp, b"uStep\0"),
                parameter(fragment_gxp, b"uWiden\0"),
                parameter(fragment_gxp, b"uWidenLow\0"),
                parameter(fragment_gxp, b"uWrite\0"),
                parameter(fragment_gxp, b"uPath\0"),
            ];
            self.dither = [
                parameter(fragment_gxp, b"uDither0\0"),
                parameter(fragment_gxp, b"uDither1\0"),
                parameter(fragment_gxp, b"uDither2\0"),
                parameter(fragment_gxp, b"uDither3\0"),
            ];
        }
        let mut missing = generic && self.cut.iter().chain(&self.dither).any(|parameter| parameter.is_null());
        if textured {
            self.fog_and_scale = parameter(fragment_gxp, b"uFogAndScale\0");
            missing |= self.fog_and_scale.is_null();
            if generic || tested {
                self.alpha_test = parameter(fragment_gxp, b"uAlphaTest\0");
                missing |= self.alpha_test.is_null();
            }
            if generic {
                self.blend_fix = parameter(fragment_gxp, b"uBlendFix\0");
                missing |= self.blend_fix.is_null();
            }
        }
        if missing {
            render_log::line(format_args!("16-bit target program ({label}, {variant:?}): parameters missing\n"));
            return false;
        }
        // The "off" ones read no frame: GXM keeps the alpha by its colour
        // mask; the precut ones get GXM's normal alpha blending (the others
        // are made without GXM blending).
        let rgb = gxm::COLOR_MASK_R | gxm::COLOR_MASK_G | gxm::COLOR_MASK_B;
        let keep_rgb = gxm::BlendInfo::new(rgb, gxm::BLEND_FUNC_NONE, gxm::BLEND_FACTOR_ONE, gxm::BLEND_FACTOR_ZERO);
        // GXM's factors and functions as in `blend_program` (source alpha
        // 4, one minus source alpha 5, add 0 + 1).
        let normal = |mask| gxm::BlendInfo::new(mask, 1, 4, 5);
        let precut_keep = normal(rgb);
        let precut = normal(gxm::COLOR_MASK_ALL);
        let info = match variant {
            ReadVariant::Fast { way: ReadWay::Off, keep_alpha: true, .. } => &keep_rgb as *const gxm::BlendInfo,
            ReadVariant::Fast { way: ReadWay::Precut, keep_alpha: true, .. } => &precut_keep as *const gxm::BlendInfo,
            ReadVariant::Fast { way: ReadWay::Precut, keep_alpha: false, .. } => &precut as *const gxm::BlendInfo,
            _ => null(),
        };
        self.program = basic_programs::fragment_program(id, info, vertex_gxp);
        render_log::line(format_args!(
            "16-bit target program ({label}, {variant:?}) {}\n",
            if self.program.is_null() { "failed" } else { "ready" }
        ));
        !self.program.is_null()
    }
}

struct BlendProgram {
    kind: Kind,
    blend: u32,
    program: *mut gxm::FragmentProgram,
}

pub struct Programs {
    pub textured: Textured,
    pub raw: RawSet,
    pub raw_plain: RawSet,
    pub raw_plain4: RawSet,
    pub raw_lit1: RawSet,
    pub raw_lit4: RawSet,
    pub raw_lit1_flat: RawSet,
    pub raw_lit4_flat: RawSet,
    /// The variants with a fixed colour format (`fixed_raw_sets`).
    raw_fixed: [RawSet; FIXED_VARIANTS * FIXED_COLORS],
    pub sprites: SpriteSet,
    blend: [BlendProgram; MAX_BLEND_PROGRAMS],
    blend_count: usize,
    /// Draws into 16-bit PSP targets (`read_index`).
    read: [ReadProgram; READ_PROGRAMS],
}

const NO_BLEND_PROGRAM: BlendProgram = BlendProgram { kind: Kind::Color, blend: 0, program: null_mut() };

impl RawSet {
    const fn new(variant: RawVariant, color: RawColor) -> RawSet {
        RawSet {
            variant,
            color,
            tried: false,
            ready: false,
            id: null_mut(),
            position: null(),
            texcoord: null(),
            color_bytes: null(),
            color32: null(),
            normal: null(),
            weights0: null(),
            weights1: null(),
            weight: null(),
            uniforms: RawUniforms::NONE,
            lights: null(),
            skin: null(),
            bones: null(),
            scale: null(),
            formats: [0; RAW_PROGRAMS],
            programs: [null_mut(); RAW_PROGRAMS],
            count: 0,
        }
    }

    fn label(&self) -> Label {
        Label(self.variant, self.color)
    }

    /// Registers the program and finds its parameters (once).
    pub fn ready(&mut self) -> bool {
        if self.tried {
            return self.ready;
        }
        self.tried = true;
        let program = raw_gxp(self.variant, self.color);
        let Some(id) = register(program) else {
            render_log::line(format_args!("raw program{}: invalid or not registered\n", self.label()));
            return false;
        };
        self.id = id;
        self.position = parameter(program, b"aPosition\0");
        self.texcoord = parameter(program, b"aTexcoord\0");
        self.color_bytes = parameter(program, b"aColorBytes\0");
        self.color32 = parameter(program, b"aColor32\0");
        self.normal = parameter(program, b"aNormal\0");
        self.lights = parameter(program, b"uLights\0");
        self.weights0 = parameter(program, b"aWeights0\0");
        self.weights1 = parameter(program, b"aWeights1\0");
        self.weight = parameter(program, b"aWeight\0");
        self.skin = parameter(program, b"uSkin\0");
        self.bones = parameter(program, b"uBones\0");
        self.scale = parameter(program, b"uScale\0");
        self.uniforms = RawUniforms::find(program);
        let bones = self.variant.bones();
        // The lit-only programs have no unlit colour rules: no uAmbient,
        // uEmissive, uScale (left null; `set_uniform` skips null ones).
        let unlit_rules = !self.variant.lights_only();
        // A fixed colour format reads only its own attribute (none without colour).
        if self.position.is_null()
            || self.texcoord.is_null()
            || (self.color.reads_bytes() && self.color_bytes.is_null())
            || (self.color.reads_32() && self.color32.is_null())
            || (unlit_rules && (self.scale.is_null() || !self.uniforms.unlit_rules_found()))
            || self.skin.is_null()
            || self.bones.is_null()
            || !self.uniforms.all_found()
            || (self.color.uses_material(self.variant) && self.uniforms.material.is_null())
            || (bones == 1 && self.weight.is_null())
            || (bones > 1 && self.weights0.is_null())
            || (bones > 4 && self.weights1.is_null())
            || (self.variant.lights() && (self.normal.is_null() || self.lights.is_null()))
        {
            render_log::line(format_args!("raw program{}: parameters missing\n", self.label()));
            return false;
        }
        self.ready = true;
        render_log::line(format_args!("raw program{} ready\n", self.label()));
        true
    }

    /// The program patched for VERTEX_TYPE `format`, made at its first use;
    /// null if it can't be made. VERTEX_TYPE fields: texture coordinates
    /// bits 0-1 (none, u8, u16, float), colour bits 2-4 (4 5650, 5 5551,
    /// 6 4444, 7 8888), normal bits 5-6 and position bits 7-8 (s8, s16,
    /// float), weights bits 9-10 (u8, u16, float; always at byte 0, as many
    /// as bits 14-16 + 1). Fields a format lacks are read from byte 0 (the
    /// shader ignores them: texture scale 0, colour mode 0).
    pub fn program_for(&mut self, format: u32) -> *mut gxm::VertexProgram {
        if let Some(index) = find_format(&self.formats[..self.count], format) {
            return self.programs[index];
        }
        if !self.ready() || self.count == RAW_PROGRAMS {
            return null_mut();
        }
        let layout = VertexFormat(format).layout();
        let texcoord_offset = layout.texcoord_offset.unwrap_or(0);
        let color_offset = layout.color_offset.unwrap_or(0);
        let normal_offset = layout.normal_offset.unwrap_or(0);
        debug_assert!(self.color == RawColor::Any || self.color == RawColor::of_format(format));
        // Weights 1-4 and 5-8 (aWeights0, aWeights1); without (more) weights
        // one byte from byte 0, unused (the bones beyond the format's are zero).
        let weight_type = ((format >> 9) & 3) as usize;
        let weights = if weight_type != 0 { ((format >> 14) & 7) + 1 } else { 0 };
        let weights0 = if weights > 4 { 4 } else { weights.max(1) } as u8;
        let weights1 = if weights > 4 { weights - 4 } else { 1 } as u8;
        let weights1_offset = if weights > 4 { 4 * WEIGHT_SIZES[weight_type] } else { 0 };
        let weights1_format = if weights > 4 { WEIGHT_FORMATS[weight_type] } else { gxm::ATTRIBUTE_FORMAT_U8 };
        let position = attribute(0, layout.position_offset, POSITION_FORMATS[((format >> 7) & 3) as usize], 3, self.position);
        let mut attributes = [position; 7];
        attributes[1] = attribute(0, texcoord_offset, TEXCOORD_FORMATS[(format & 3) as usize], 2, self.texcoord);
        let mut used = 2;
        if self.color.reads_bytes() {
            attributes[used] = attribute(0, color_offset, gxm::ATTRIBUTE_FORMAT_U8, 2, self.color_bytes);
            used += 1;
        }
        if self.color.reads_32() {
            attributes[used] = attribute(0, color_offset, gxm::ATTRIBUTE_FORMAT_U8N, 4, self.color32);
            used += 1;
        }
        let bones = self.variant.bones();
        if bones == 1 {
            // The first weight (one byte from byte 0 without weights: the
            // program multiplies it by 0 then).
            attributes[used] = attribute(0, 0, WEIGHT_FORMATS[weight_type], 1, self.weight);
            used += 1;
        } else {
            attributes[used] = attribute(0, 0, WEIGHT_FORMATS[weight_type], weights0, self.weights0);
            used += 1;
            if bones > 4 {
                attributes[used] = attribute(0, weights1_offset, weights1_format, weights1, self.weights1);
                used += 1;
            }
        }
        if self.variant.lights() {
            attributes[used] = attribute(0, normal_offset, POSITION_FORMATS[((format >> 5) & 3) as usize], 3, self.normal);
            used += 1;
        }
        let stream = [VertexStream { stride: layout.size as u16, index_source: gxm::INDEX_SOURCE_INDEX_16BIT }];
        let program = vertex_program(
            self.id,
            &attributes[..used],
            &stream,
            format_args!("raw program{} for format {:08x}", self.label(), format),
        );
        if !program.is_null() {
            render_log::line(format_args!("raw program{} for format {:08x} (stride {})\n", self.label(), format, layout.size));
        }
        self.formats[self.count] = format;
        self.programs[self.count] = program;
        self.count += 1;
        program
    }
}

const TEXCOORD_FORMATS: [u8; 4] =
    [gxm::ATTRIBUTE_FORMAT_U8, gxm::ATTRIBUTE_FORMAT_U8, gxm::ATTRIBUTE_FORMAT_U16, gxm::ATTRIBUTE_FORMAT_F32];
const POSITION_FORMATS: [u8; 4] =
    [gxm::ATTRIBUTE_FORMAT_S8, gxm::ATTRIBUTE_FORMAT_S8, gxm::ATTRIBUTE_FORMAT_S16, gxm::ATTRIBUTE_FORMAT_F32];
const WEIGHT_FORMATS: [u8; 4] = TEXCOORD_FORMATS;
const WEIGHT_SIZES: [u32; 4] = [1, 1, 2, 4];

impl SpriteSet {
    const fn new() -> SpriteSet {
        SpriteSet {
            tried: false,
            ready: false,
            id: null_mut(),
            position: null(),
            texcoord: null(),
            position_b: null(),
            texcoord_b: null(),
            color_bytes_b: null(),
            color32_b: null(),
            corner: null(),
            uniforms: RawUniforms::NONE,
            corners: null_mut(),
            formats: [0; RAW_PROGRAMS],
            programs: [null_mut(); RAW_PROGRAMS],
            count: 0,
        }
    }

    /// Registers the program, finds its parameters and makes the corner
    /// table (once).
    pub fn ready(&mut self) -> bool {
        if self.tried {
            return self.ready;
        }
        self.tried = true;
        let program = gxp(&shaders::nativeres_raw_sprite_v);
        let Some(id) = register(program) else {
            render_log::line(format_args!("sprite program: invalid or not registered\n"));
            return false;
        };
        self.id = id;
        self.position = parameter(program, b"aPosition\0");
        self.texcoord = parameter(program, b"aTexcoord\0");
        self.position_b = parameter(program, b"aPositionB\0");
        self.texcoord_b = parameter(program, b"aTexcoordB\0");
        self.color_bytes_b = parameter(program, b"aColorBytesB\0");
        self.color32_b = parameter(program, b"aColor32B\0");
        self.corner = parameter(program, b"aCorner\0");
        self.uniforms = RawUniforms::find(program);
        let inputs = [self.position, self.texcoord, self.position_b, self.texcoord_b, self.color_bytes_b, self.color32_b, self.corner];
        if inputs.iter().any(|parameter| parameter.is_null()) || !self.uniforms.all_found() || self.uniforms.material.is_null() || !self.uniforms.unlit_rules_found() {
            render_log::line(format_args!("sprite program: parameters missing\n"));
            return false;
        }
        let Some(block) = crate::gpu_memory::gpu_input(b"NativeResSpriteCorners\0", SPRITE_CORNERS as u32) else {
            render_log::line(format_args!("sprite program: no corner table\n"));
            return false;
        };
        // Safe: the block holds SPRITE_CORNERS bytes and is ours.
        let corners = unsafe { core::slice::from_raw_parts_mut(block.base, SPRITE_CORNERS) };
        for (k, corner) in corners.iter_mut().enumerate() {
            *corner = (k & 3) as u8;
        }
        self.corners = block.base;
        self.ready = true;
        render_log::line(format_args!("sprite program ready\n"));
        true
    }

    /// The sprite program patched for VERTEX_TYPE `format` (as
    /// `RawSet::program_for`): each corner's vertex pair read as two streams
    /// over the same bytes (A at the pair's start, B one stride later, both
    /// a pair apart), the corner number from `corners`.
    pub fn program_for(&mut self, format: u32) -> *mut gxm::VertexProgram {
        if let Some(index) = find_format(&self.formats[..self.count], format) {
            return self.programs[index];
        }
        if !self.ready() || self.count == RAW_PROGRAMS {
            return null_mut();
        }
        let layout = VertexFormat(format).layout();
        let texcoord_offset = layout.texcoord_offset.unwrap_or(0);
        let color_offset = layout.color_offset.unwrap_or(0);
        let position_format = POSITION_FORMATS[((format >> 7) & 3) as usize];
        let texcoord_format = TEXCOORD_FORMATS[(format & 3) as usize];
        let attributes = [
            attribute(0, layout.position_offset, position_format, 3, self.position),
            attribute(0, texcoord_offset, texcoord_format, 2, self.texcoord),
            attribute(1, layout.position_offset, position_format, 3, self.position_b),
            attribute(1, texcoord_offset, texcoord_format, 2, self.texcoord_b),
            attribute(1, color_offset, gxm::ATTRIBUTE_FORMAT_U8, 2, self.color_bytes_b),
            attribute(1, color_offset, gxm::ATTRIBUTE_FORMAT_U8N, 4, self.color32_b),
            attribute(2, 0, gxm::ATTRIBUTE_FORMAT_U8, 1, self.corner),
        ];
        let pair = VertexStream { stride: (2 * layout.size) as u16, index_source: gxm::INDEX_SOURCE_INDEX_16BIT };
        let streams = [pair, pair, VertexStream { stride: 1, index_source: gxm::INDEX_SOURCE_INDEX_16BIT }];
        let program = vertex_program(self.id, &attributes, &streams, format_args!("sprite program for format {:08x}", format));
        if !program.is_null() {
            render_log::line(format_args!("sprite program for format {:08x} (stride {})\n", format, layout.size));
        }
        self.formats[self.count] = format;
        self.programs[self.count] = program;
        self.count += 1;
        program
    }
}

impl Programs {
    pub const fn new() -> Programs {
        Programs {
            textured: Textured {
                tried: false,
                vertex_id: null_mut(),
                fragment_id: null_mut(),
                vertex: null_mut(),
                wvp: null(),
                fog_and_scale: null(),
                have_flat: false,
                fragment_flat_id: null_mut(),
                fog_and_scale_flat: null(),
                have_test: false,
                fragment_test_id: null_mut(),
                fog_and_scale_test: null(),
                alpha_test: null(),
                have_fix: false,
                fragment_fix_id: null_mut(),
                fragment_test_fix_id: null_mut(),
                fog_and_scale_fix: null(),
                blend_fix: null(),
                fog_and_scale_test_fix: null(),
                alpha_test_fix: null(),
                blend_fix_test: null(),
                vertex_3d: null_mut(),
                wvp_3d: null(),
                rows_3d: null(),
                fog_row_3d: null(),
                viewport_3d: null(),
                depth_3d: null(),
                uv_scale_3d: null(),
                normal_rows_3d: null(),
                lights_3d: null(),
            },
            raw: RawSet::new(RawVariant::Full, RawColor::Any),
            raw_plain: RawSet::new(RawVariant::Plain, RawColor::Any),
            raw_plain4: RawSet::new(RawVariant::Plain4, RawColor::Any),
            raw_lit1: RawSet::new(RawVariant::Lit1, RawColor::Any),
            raw_lit4: RawSet::new(RawVariant::Lit4, RawColor::Any),
            raw_lit1_flat: RawSet::new(RawVariant::Lit1Flat, RawColor::Any),
            raw_lit4_flat: RawSet::new(RawVariant::Lit4Flat, RawColor::Any),
            raw_fixed: fixed_raw_sets(),
            sprites: SpriteSet::new(),
            blend: [NO_BLEND_PROGRAM; MAX_BLEND_PROGRAMS],
            blend_count: 0,
            read: [const { ReadProgram::new() }; READ_PROGRAMS],
        }
    }

    /// The frame-read program `variant` for a draw into a 16-bit PSP
    /// target (`textured` or not, `tested`: with the alpha test), or None
    /// if it can't be made.
    pub fn read_program(&mut self, textured: bool, tested: bool, variant: ReadVariant) -> Option<&ReadProgram> {
        let program = &mut self.read[read_index(textured, tested, variant)];
        program.ready(textured, tested, variant).then_some(&*program)
    }

    /// The frame-read program made before (`read_program`).
    pub fn made_read_program(&self, textured: bool, tested: bool, variant: ReadVariant) -> &ReadProgram {
        &self.read[read_index(textured, tested, variant)]
    }

    /// The raw program set of `variant` with colour format `color` (the
    /// full program only as `RawColor::Any`, whatever `color` says).
    pub fn raw_set(&mut self, variant: RawVariant, color: RawColor) -> &mut RawSet {
        if let (Some(place), Some(color)) = (variant.fixed_index(), color.fixed_index()) {
            return &mut self.raw_fixed[place * FIXED_COLORS + color];
        }
        match variant {
            RawVariant::Full => &mut self.raw,
            RawVariant::Plain => &mut self.raw_plain,
            RawVariant::Plain4 => &mut self.raw_plain4,
            RawVariant::Lit1 => &mut self.raw_lit1,
            RawVariant::Lit4 => &mut self.raw_lit4,
            RawVariant::Lit1Flat => &mut self.raw_lit1_flat,
            RawVariant::Lit4Flat => &mut self.raw_lit4_flat,
        }
    }

    /// Our textured programs, made on first use (with the alpha-test, FIX B
    /// and 3D variants). False if they can't be made (textured draws then use
    /// the basic texture-tint program: one colour per draw, no perspective,
    /// no fog).
    pub fn textured_ready(&mut self) -> bool {
        let textured = &mut self.textured;
        if textured.tried {
            return !textured.vertex.is_null();
        }
        textured.tried = true;
        let vertex_gxp = gxp(&shaders::nativeres_textured_v);
        let fragment_gxp = gxp(&shaders::nativeres_textured_f);
        let (Some(vertex_id), Some(fragment_id)) = (register(vertex_gxp), register(fragment_gxp)) else {
            render_log::line(format_args!("our programs: invalid GXP\n"));
            return false;
        };
        textured.vertex_id = vertex_id;
        textured.fragment_id = fragment_id;
        let position = parameter(vertex_gxp, b"aPosition\0");
        let texcoord = parameter(vertex_gxp, b"aTexcoord\0");
        let color = parameter(vertex_gxp, b"aColor\0");
        let fog = parameter(vertex_gxp, b"aFog\0");
        textured.wvp = parameter(vertex_gxp, b"wvp\0");
        textured.fog_and_scale = parameter(fragment_gxp, b"uFogAndScale\0");
        if [position, texcoord, color, fog, textured.wvp, textured.fog_and_scale].iter().any(|parameter| parameter.is_null()) {
            render_log::line(format_args!("our programs: parameters missing\n"));
            return false;
        }
        let attributes = [
            attribute(0, offset_of!(FrameVertex, x) as u32, gxm::ATTRIBUTE_FORMAT_F32, 4, position),
            attribute(0, offset_of!(FrameVertex, u) as u32, gxm::ATTRIBUTE_FORMAT_F32, 2, texcoord),
            attribute(0, offset_of!(FrameVertex, color) as u32, gxm::ATTRIBUTE_FORMAT_U8N, 4, color),
            attribute(0, offset_of!(FrameVertex, fog) as u32, gxm::ATTRIBUTE_FORMAT_F32, 1, fog),
        ];
        let stream = [VertexStream { stride: core::mem::size_of::<FrameVertex>() as u16, index_source: gxm::INDEX_SOURCE_INDEX_16BIT }];
        textured.vertex = vertex_program(vertex_id, &attributes, &stream, format_args!("our programs"));
        if textured.vertex.is_null() {
            return false;
        }
        render_log::line(format_args!("our programs ready\n"));
        self.make_flat_program();
        self.make_alpha_test_program();
        self.make_blend_fix_programs();
        self.make_3d_program();
        true
    }

    /// The variant without fog and specular (nativeres_textured_flat_f.cg).
    fn make_flat_program(&mut self) {
        let textured = &mut self.textured;
        let program = gxp(&shaders::nativeres_textured_flat_f);
        let Some(id) = register(program) else {
            render_log::line(format_args!("flat program: invalid GXP\n"));
            return;
        };
        textured.fragment_flat_id = id;
        textured.fog_and_scale_flat = parameter(program, b"uFogAndScale\0");
        if textured.fog_and_scale_flat.is_null() {
            render_log::line(format_args!("flat program: parameters missing\n"));
            return;
        }
        textured.have_flat = true;
    }

    /// The alpha-test variant (nativeres_textured_test_f.cg). Once it is
    /// registered, the capture leaves textured draws' alpha test to it
    /// instead of standing in with blending.
    fn make_alpha_test_program(&mut self) {
        let textured = &mut self.textured;
        let program = gxp(&shaders::nativeres_textured_test_f);
        let Some(id) = register(program) else {
            render_log::line(format_args!("alpha test program: invalid GXP\n"));
            return;
        };
        textured.fragment_test_id = id;
        textured.fog_and_scale_test = parameter(program, b"uFogAndScale\0");
        textured.alpha_test = parameter(program, b"uAlphaTest\0");
        if textured.fog_and_scale_test.is_null() || textured.alpha_test.is_null() {
            render_log::line(format_args!("alpha test program: parameters missing\n"));
            return;
        }
        textured.have_test = true;
        crate::capture::set_alpha_test_ready(1);
        render_log::line(format_args!("alpha test program ready\n"));
    }

    /// The BLEND_FIX variants (grey FIX B through the alpha, core blend.rs).
    fn make_blend_fix_programs(&mut self) {
        let textured = &mut self.textured;
        let program = gxp(&shaders::nativeres_textured_fix_f);
        let test_program = gxp(&shaders::nativeres_textured_test_fix_f);
        let (Some(id), Some(test_id)) = (register(program), register(test_program)) else {
            render_log::line(format_args!("blend fix programs: invalid GXP\n"));
            return;
        };
        textured.fragment_fix_id = id;
        textured.fragment_test_fix_id = test_id;
        textured.fog_and_scale_fix = parameter(program, b"uFogAndScale\0");
        textured.blend_fix = parameter(program, b"uBlendFix\0");
        textured.fog_and_scale_test_fix = parameter(test_program, b"uFogAndScale\0");
        textured.alpha_test_fix = parameter(test_program, b"uAlphaTest\0");
        textured.blend_fix_test = parameter(test_program, b"uBlendFix\0");
        let all = [
            textured.fog_and_scale_fix,
            textured.blend_fix,
            textured.fog_and_scale_test_fix,
            textured.alpha_test_fix,
            textured.blend_fix_test,
        ];
        if all.iter().any(|parameter| parameter.is_null()) {
            render_log::line(format_args!("blend fix programs: parameters missing\n"));
            return;
        }
        textured.have_fix = true;
        render_log::line(format_args!("blend fix programs ready\n"));
    }

    /// The 3D vertex program. Failing only leaves the GPU transform off.
    fn make_3d_program(&mut self) {
        let textured = &mut self.textured;
        let program = gxp(&shaders::nativeres_3d_v);
        let Some(id) = register(program) else {
            render_log::line(format_args!("3d program: invalid or not registered\n"));
            return;
        };
        let position = parameter(program, b"aPosition\0");
        let normal = parameter(program, b"aNormal\0");
        let texcoord = parameter(program, b"aTexcoord\0");
        let color = parameter(program, b"aColor\0");
        textured.normal_rows_3d = parameter(program, b"uNormalRows\0");
        textured.lights_3d = parameter(program, b"uLights\0");
        textured.wvp_3d = parameter(program, b"wvp\0");
        textured.rows_3d = parameter(program, b"uRows\0");
        textured.fog_row_3d = parameter(program, b"uFogRow\0");
        textured.viewport_3d = parameter(program, b"uViewport\0");
        textured.depth_3d = parameter(program, b"uDepth\0");
        textured.uv_scale_3d = parameter(program, b"uUvScale\0");
        let all = [
            position,
            normal,
            texcoord,
            color,
            textured.normal_rows_3d,
            textured.lights_3d,
            textured.wvp_3d,
            textured.rows_3d,
            textured.fog_row_3d,
            textured.viewport_3d,
            textured.depth_3d,
            textured.uv_scale_3d,
        ];
        if all.iter().any(|parameter| parameter.is_null()) {
            render_log::line(format_args!("3d program: parameters missing\n"));
            return;
        }
        // The frame's vertices as they are, x, y, z = model position; w holds
        // GPU-lit vertices' normal as four signed bytes (lighting.rs
        // pack_normal; read only for lit draws).
        let attributes = [
            attribute(0, offset_of!(FrameVertex, x) as u32, gxm::ATTRIBUTE_FORMAT_F32, 3, position),
            attribute(0, offset_of!(FrameVertex, w) as u32, gxm::ATTRIBUTE_FORMAT_S8N, 4, normal),
            attribute(0, offset_of!(FrameVertex, u) as u32, gxm::ATTRIBUTE_FORMAT_F32, 2, texcoord),
            attribute(0, offset_of!(FrameVertex, color) as u32, gxm::ATTRIBUTE_FORMAT_U8N, 4, color),
        ];
        let stream = [VertexStream { stride: core::mem::size_of::<FrameVertex>() as u16, index_source: gxm::INDEX_SOURCE_INDEX_16BIT }];
        textured.vertex_3d = vertex_program(id, &attributes, &stream, format_args!("3d program"));
        if textured.vertex_3d.is_null() {
            return;
        }
        crate::capture::GPU_TRANSFORM_READY.store(1, core::sync::atomic::Ordering::Release);
        render_log::line(format_args!("3d program ready\n"));
    }

    /// Our textured fragment program of `kind` with `blend_info` (null:
    /// blending off, all channels written); null if it can't be made.
    fn textured_fragment_program(&mut self, kind: Kind, blend_info: *const gxm::BlendInfo) -> *mut gxm::FragmentProgram {
        if !self.textured_ready() {
            return null_mut();
        }
        let textured = &self.textured;
        let id = match kind {
            Kind::TexturedTest if textured.have_test => textured.fragment_test_id,
            Kind::TexturedFix if textured.have_fix => textured.fragment_fix_id,
            Kind::TexturedTestFix if textured.have_test && textured.have_fix => textured.fragment_test_fix_id,
            Kind::Textured => textured.fragment_id,
            Kind::TexturedFlat if textured.have_flat => textured.fragment_flat_id,
            _ => return null_mut(),
        };
        basic_programs::fragment_program(id, blend_info, gxp(&shaders::nativeres_textured_v))
    }

    /// The fragment program of `kind` for a draw's blending (`Draw::blend`),
    /// or null if it can't be made (the caller then falls back to the basic
    /// normal alpha-blending program). `keep_target_alpha`: draws keep the
    /// target's alpha (the PSP's stencil) unless they write it.
    pub fn blend_program(&mut self, kind: Kind, blend: u32, keep_target_alpha: bool) -> *mut gxm::FragmentProgram {
        let mut kind = kind;
        let mut blend = blend;
        if !keep_target_alpha && blend & BLEND_ALPHA_ONLY == 0 {
            blend |= BLEND_WRITES_ALPHA; // nobody reads it: every draw writes alpha (cheaper)
        }
        // FIX B through the alpha: the BLEND_FIX variants; the value is a
        // uniform (draw.rs textured_fragment_uniforms), not part of the program.
        if blend & BLEND_ENABLED != 0 && blend & BLEND_FIX_B_ALPHA != 0 {
            kind = match kind {
                Kind::Textured | Kind::TexturedFlat => Kind::TexturedFix,
                Kind::TexturedTest => Kind::TexturedTestFix,
                other => other,
            };
            blend &= !BLEND_FIX_B_VALUES;
        }
        if let Some(known) = self.blend[..self.blend_count].iter().find(|known| known.kind == kind && known.blend == blend) {
            return known.program;
        }
        if self.blend_count == MAX_BLEND_PROGRAMS {
            return null_mut();
        }
        // The alpha channel is the PSP's stencil: kept unless the draw
        // writes it (WRITES_ALPHA; MTX's blur weights the sampled
        // framebuffer by it). Stencil-only clears write the alpha alone.
        let writes_alpha = blend & BLEND_WRITES_ALPHA != 0;
        let alpha_only = writes_alpha && blend & BLEND_ALPHA_ONLY != 0;
        let mask = if alpha_only {
            gxm::COLOR_MASK_A
        } else if writes_alpha {
            gxm::COLOR_MASK_ALL
        } else {
            gxm::COLOR_MASK_R | gxm::COLOR_MASK_G | gxm::COLOR_MASK_B
        };
        let info = if blend & BLEND_ENABLED != 0 && !alpha_only {
            let (source, destination, equation) = (blend & 0xF, (blend >> 4) & 0xF, (blend >> 8) & 7);
            if source > 9 || destination > 9 || equation > 4 {
                return null_mut();
            }
            // GXM's factors are the PSP's numbers; its functions the PSP's
            // equations + 1 (0 is NONE).
            Some(gxm::BlendInfo::new(mask, equation as u8 + 1, source as u8, destination as u8))
        } else if mask != gxm::COLOR_MASK_ALL {
            Some(gxm::BlendInfo::new(mask, gxm::BLEND_FUNC_NONE, gxm::BLEND_FACTOR_ONE, gxm::BLEND_FACTOR_ZERO))
        } else {
            None // blending off, all channels written
        };
        let info_pointer = info.as_ref().map_or(null(), |info| info as *const gxm::BlendInfo);
        let program = match kind {
            Kind::Color => basic_programs::blend_fragment_program(false, info_pointer),
            Kind::TextureTint => basic_programs::blend_fragment_program(true, info_pointer),
            _ => self.textured_fragment_program(kind, info_pointer),
        };
        if !program.is_null() {
            self.blend[self.blend_count] = BlendProgram { kind, blend, program };
            self.blend_count += 1;
        }
        program
    }
}
