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
//! - the raw programs (nativeres_raw_v.cg, _raw_plain_v.cg): raw PSP
//!   vertices, one patched program per PSP vertex format;
//! - the raw sprite programs (nativeres_raw_sprite_v.cg);
//! - blend programs: GXM bakes blending into the fragment program, so every
//!   blend mode a game uses needs its own; made on first use and kept.
//!
//! Display thread only (draw.rs owns the one `Programs`).

use core::ffi::c_void;
use core::mem::offset_of;
use core::ptr::{null, null_mut};

use nativepsp_core::blend::{BLEND_ALPHA_ONLY, BLEND_ENABLED, BLEND_FIX_B_ALPHA, BLEND_WRITES_ALPHA};
use nativepsp_core::frame::FrameVertex;
use nativepsp_core::vertex::VertexFormat;

use crate::gxm::{self, PatcherId, ProgramParameter, VertexAttribute, VertexStream};
use crate::basic_programs::{self, gxp, parameter, register, resource};
use crate::{render_log, shaders};

/// The PSP's blend words' FIX B value and premultiply bits: uniforms of the
/// BLEND_FIX programs, not part of a program (NATIVEPSP_BLEND_FIX_B_VALUES).
pub const BLEND_FIX_B_VALUES: u32 = 0x3FF << 12;

/// Which fragment program: the basic colour and texture-tint programs, our
/// textured program without and with the PSP's alpha test, and both with
/// grey FIX B blending through the alpha (BLEND_FIX).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Color,
    TextureTint,
    Textured,
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

/// A raw program (nativeres_raw_v.cg or the plain one) and its programs per
/// PSP vertex format.
pub struct RawSet {
    plain: bool,
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
    pub uniforms: RawUniforms,
    pub normal_rows: *const ProgramParameter,
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
    pub rows: *const ProgramParameter,
    pub fog_row: *const ProgramParameter,
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
        rows: null(),
        fog_row: null(),
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
            rows: parameter(program, b"uRows\0"),
            fog_row: parameter(program, b"uFogRow\0"),
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

    fn all_found(&self) -> bool {
        [
            self.wvp,
            self.rows,
            self.fog_row,
            self.viewport,
            self.uv_scale,
            self.format,
            self.raw_uv,
            self.material,
            self.ambient,
            self.emissive,
            self.colors,
        ]
        .iter()
        .all(|parameter| !parameter.is_null())
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
pub static nativepsp_plan_fixed_sprite_corners: u32 = crate::sys::pages(SPRITE_CORNERS);

struct BlendProgram {
    kind: Kind,
    blend: u32,
    program: *mut gxm::FragmentProgram,
}

pub struct Programs {
    pub textured: Textured,
    pub raw: RawSet,
    pub raw_plain: RawSet,
    pub sprites: SpriteSet,
    blend: [BlendProgram; MAX_BLEND_PROGRAMS],
    blend_count: usize,
}

const NO_BLEND_PROGRAM: BlendProgram = BlendProgram { kind: Kind::Color, blend: 0, program: null_mut() };

impl RawSet {
    const fn new(plain: bool) -> RawSet {
        RawSet {
            plain,
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
            uniforms: RawUniforms::NONE,
            normal_rows: null(),
            lights: null(),
            skin: null(),
            bones: null(),
            scale: null(),
            formats: [0; RAW_PROGRAMS],
            programs: [null_mut(); RAW_PROGRAMS],
            count: 0,
        }
    }

    fn label(&self) -> &'static str {
        if self.plain {
            " (plain)"
        } else {
            ""
        }
    }

    /// Registers the program and finds its parameters (once).
    pub fn ready(&mut self) -> bool {
        if self.tried {
            return self.ready;
        }
        self.tried = true;
        let program = if self.plain { gxp(&shaders::nativeres_raw_plain_v) } else { gxp(&shaders::nativeres_raw_v) };
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
        self.normal_rows = parameter(program, b"uNormalRows\0");
        self.lights = parameter(program, b"uLights\0");
        self.weights0 = parameter(program, b"aWeights0\0");
        self.weights1 = parameter(program, b"aWeights1\0");
        self.skin = parameter(program, b"uSkin\0");
        self.bones = parameter(program, b"uBones\0");
        self.scale = parameter(program, b"uScale\0");
        self.uniforms = RawUniforms::find(program);
        let skinned_lit = [self.normal, self.normal_rows, self.lights, self.weights0, self.weights1, self.skin, self.bones];
        if self.position.is_null()
            || self.texcoord.is_null()
            || self.color_bytes.is_null()
            || self.color32.is_null()
            || self.scale.is_null()
            || !self.uniforms.all_found()
            || (!self.plain && skinned_lit.iter().any(|parameter| parameter.is_null()))
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
        // Weights 1-4 and 5-8 (aWeights0, aWeights1); without (more) weights
        // one byte from byte 0, unused (the bones beyond the format's are zero).
        let weight_type = ((format >> 9) & 3) as usize;
        let weights = if weight_type != 0 { ((format >> 14) & 7) + 1 } else { 0 };
        let weights0 = if weights > 4 { 4 } else { weights.max(1) } as u8;
        let weights1 = if weights > 4 { weights - 4 } else { 1 } as u8;
        let weights1_offset = if weights > 4 { 4 * WEIGHT_SIZES[weight_type] } else { 0 };
        let weights1_format = if weights > 4 { WEIGHT_FORMATS[weight_type] } else { gxm::ATTRIBUTE_FORMAT_U8 };
        let attributes = [
            attribute(0, layout.position_offset, POSITION_FORMATS[((format >> 7) & 3) as usize], 3, self.position),
            attribute(0, texcoord_offset, TEXCOORD_FORMATS[(format & 3) as usize], 2, self.texcoord),
            attribute(0, color_offset, gxm::ATTRIBUTE_FORMAT_U8, 2, self.color_bytes),
            attribute(0, color_offset, gxm::ATTRIBUTE_FORMAT_U8N, 4, self.color32),
            attribute(0, normal_offset, POSITION_FORMATS[((format >> 5) & 3) as usize], 3, self.normal),
            attribute(0, 0, WEIGHT_FORMATS[weight_type], weights0, self.weights0),
            attribute(0, weights1_offset, weights1_format, weights1, self.weights1),
        ];
        let used = if self.plain { 4 } else { 7 };
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
        if inputs.iter().any(|parameter| parameter.is_null()) || !self.uniforms.all_found() {
            render_log::line(format_args!("sprite program: parameters missing\n"));
            return false;
        }
        let Some(block) = crate::gpu_memory::gpu_input(b"NativePspSpriteCorners\0", SPRITE_CORNERS as u32) else {
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
            raw: RawSet::new(false),
            raw_plain: RawSet::new(true),
            sprites: SpriteSet::new(),
            blend: [NO_BLEND_PROGRAM; MAX_BLEND_PROGRAMS],
            blend_count: 0,
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
        self.make_alpha_test_program();
        self.make_blend_fix_programs();
        self.make_3d_program();
        true
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
                Kind::Textured => Kind::TexturedFix,
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
