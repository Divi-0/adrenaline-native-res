//! 3D draws: from the game's coordinates to screen pixels, on the CPU.
//!
//! Without through mode the GE transforms every vertex (PPSSPP: `TransformUnit`,
//! `SoftwareTransform`):
//!
//! 1. world, view and projection matrices turn the position into clip
//!    coordinates (x, y, z, w);
//! 2. the viewport turns x / w and y / w into screen pixels:
//!    `screen = center + scale * clip / w`;
//! 3. the screen offset (OFFSET_X / OFFSET_Y, in 1/16 pixels) is subtracted,
//!    which places the result in the framebuffer.
//!
//! Before step 2, triangles are clipped against the near plane (z = -w):
//! a vertex just in front of the camera has w close to 0, and dividing by it
//! threw a corner of Final Fantasy's world-map haze 38000 pixels away, a
//! triangle covering the whole screen.
//!
//! We do the same and hand the Vita the result like a 2D draw, plus each
//! vertex's w. With a perspective projection (Final Fantasy's world map is a
//! tilted plane seen with a 60 degree field of view) texture coordinates
//! must not be interpolated linearly across the screen, but in proportion to
//! 1 / w: the Vita's vertex program multiplies the position by w again so the
//! GPU divides it out and interpolates that way. Colours and fog are the
//! exception: the PSP interpolates them linearly on the screen (ppsspp-ref
//! GPU/Software/Rasterizer.cpp, "Color interpolation is not perspective
//! corrected on the PSP"; fog "through the depth plane, like Gouraud color").
//! The vertex programs hand them over times w (w in vFog.y) and the fragment
//! programs divide by the interpolated w, which undoes the perspective.
//! Monster Hunter Freedom bakes its ground light and shadow edges into vertex
//! colours of large ground triangles: with perspective they came out up to
//! 4 levels brighter.
//!
//! Skinning (vertices with weights): before step 1, the position is
//! replaced by the sum of `weight[i] * bone[i] * position` over the vertex's
//! weights, the bones being 8 more 4x3 matrices (PPSSPP: the skinning in
//! `VertexDecoder` and `TransformUnit`). MTX Mototrax draws its whole
//! stadium and rider this way, mostly with a single weight of 1 (the bone
//! matrix then simply places that piece).
//!
//! Texture coordinates of 3D draws come out as fractions of the texture size
//! (after the texture scale and offset), not texels.
//!
//! Fog (FOG_ENABLE) is worked out per vertex as in PPSSPP: the fog factor
//! `(view-space z + FOG_END) * FOG_SCALE` is interpolated across the
//! triangle and clamped to 0..1 per pixel, where 1 keeps the colour and 0
//! gives the fog colour. Final Fantasy's world map fogs the far tiles white
//! (the haze along the top of the screen).

use crate::color::lerp_rgba;
use crate::command::op;
use crate::ge_state::{float24, GeState, Matrices, BONE_COUNT, TRANSFORM_REGISTERS};
use crate::frame::{FrameVertex, GpuLights, GpuTransform, RawCamera};
use crate::lighting::{LightingRegisters, Lights, ModelLights};
use crate::vertex::{Raw3d, Vertex, VertexFormat, MAX_WEIGHTS};

/// A vertex after the matrices, before the viewport: clip coordinates, and
/// texture coordinates as fractions of the texture.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClipVertex {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
    pub u: f32,
    pub v: f32,
    pub color: u32,
    /// Fog factor (see the module comment), 1 without fog.
    pub fog: f32,
}

impl ClipVertex {
    /// In front of the near plane (or on it).
    pub fn visible(&self) -> bool {
        self.z >= -self.w
    }

    /// The point `t` of the way from `self` to `other`.
    fn lerp(&self, other: &ClipVertex, t: f32) -> ClipVertex {
        let mix = |a: f32, b: f32| a + (b - a) * t;
        ClipVertex {
            x: mix(self.x, other.x),
            y: mix(self.y, other.y),
            z: mix(self.z, other.z),
            w: mix(self.w, other.w),
            u: mix(self.u, other.u),
            v: mix(self.v, other.v),
            color: lerp_rgba(self.color, other.color, t),
            fog: mix(self.fog, other.fog),
        }
    }
}

/// The part of a triangle in front of the near plane: 0, 3 or 4 corners
/// (a quad when one corner was behind), in the triangle's order.
pub fn clip_triangle(triangle: &[ClipVertex; 3], out: &mut [ClipVertex; 4]) -> usize {
    let mut count = 0;
    for i in 0..3 {
        let (a, b) = (&triangle[i], &triangle[(i + 1) % 3]);
        if a.visible() {
            out[count] = *a;
            count += 1;
        }
        if a.visible() != b.visible() {
            // Where the edge crosses z = -w: (z + w) goes from one sign to the other.
            let (da, db) = (a.z + a.w, b.z + b.w);
            out[count] = a.lerp(b, da / (da - db));
            count += 1;
        }
    }
    count
}

/// What turns one draw's vertices into screen space. Worked out once per
/// draw, or reused while nothing it depends on changed (`TransformCache`).
#[derive(Clone, Copy)]
pub struct Transform {
    /// projection x view x world, 4x4, column by column.
    matrix: [f32; 16],
    /// view x world: its third row gives the view-space z for fog.
    view_world: [f32; 16],
    /// FOG_END and FOG_SCALE when fog is on.
    fog: Option<(f32, f32)>,
    viewport_scale: (f32, f32),
    /// Viewport centre minus the screen offset.
    viewport_origin: (f32, f32),
    /// Depth = depth_center + depth_scale * z / w (0..65535).
    depth_scale: f32,
    depth_center: f32,
    position_scale: f32,
    texcoord_scale: f32,
    texture_scale: (f32, f32),
    texture_offset: (f32, f32),
    /// Skinned draws: the bone matrices (4x3, column by column) of the
    /// format's weights; `bone_count` 0 = no skinning.
    bones: [[f32; 12]; BONE_COUNT],
    bone_count: usize,
    /// Lit draws with lights on: the lighting (`lighting::Lights`), which
    /// gives the vertex colours from their normals. Kept here because it
    /// depends on the world and view matrices too.
    pub lights: Option<Lights>,
    /// The fast path (`frame_vertex`).
    fast: FastPath,
}

/// How one draw's vertices get their lit colours (`Transform::vertex_lighting`):
/// in model space where that works (`ModelLights`: MTX's riders and bikes),
/// else `Lights::color` per vertex.
pub struct VertexLighting<'a> {
    transform: &'a Transform,
    lights: &'a Lights,
    model: Option<ModelLights>,
}

impl VertexLighting<'_> {
    /// The lit colour of a vertex from its raw fields and its normal
    /// (`None`: the format has none).
    #[inline]
    pub fn color(&self, raw: &Raw3d, normal: Option<[f32; 3]>) -> u32 {
        if let (Some(model), Some(normal)) = (&self.model, normal) {
            // One bone: a weight above 0 keeps the normal's direction.
            if self.transform.bone_count == 0 || raw.weights[0] > 0.0 {
                if let Some(color) = model.color(raw.color, normal) {
                    return color;
                }
            }
        }
        self.transform.lit_color(self.lights, raw, normal)
    }
}

/// Everything per draw folded together so that a vertex goes from its raw
/// fields to the finished `FrameVertex` in one go (`Transform::frame_vertex`).
/// MTX Mototrax has ~20,000 vertices per frame; going through `Vertex`,
/// `ClipVertex` and `Vertex` again cost ~700 Vita cycles per vertex.
#[derive(Clone, Copy)]
struct FastPath {
    /// Unskinned vertices, and those of single-weight draws whose weight is
    /// exactly 1: clip = rows x (raw x, y, z, 1), with the position scale
    /// and the bone folded in. Rows as 4 numbers each (row by row, unlike
    /// the GE's matrices) for 4 dot products.
    rows: [[f32; 4]; 4],
    /// Fog factor = fog_row . (raw x, y, z, 1), with FOG_END and FOG_SCALE
    /// folded in; `None` without fog.
    fog_row: Option<[f32; 4]>,
    /// Other skinned vertices: the weighted sum of `bones` (each with the
    /// position scale folded in) applied to the raw position gives the
    /// model position, then `skinned_rows` / `skinned_fog_row` as above.
    bones: [[f32; 12]; BONE_COUNT],
    skinned_rows: [[f32; 4]; 4],
    skinned_fog_row: Option<[f32; 4]>,
    /// u = raw u * u_scale + u_offset (texture coordinate format, texture
    /// scale and offset folded together); v likewise.
    u_scale: f32,
    u_offset: f32,
    v_scale: f32,
    v_offset: f32,
    /// Frame depth = depth_center + depth_scale * z / w, already divided by 65535.
    depth_scale: f32,
    depth_center: f32,
}

/// `Transform::frame_vertex` for vertices that need no skinning, as a
/// small copy for a vertex loop (`Transform::rows`). Same numbers, same
/// order of operations, so the same results to the bit.
#[derive(Clone, Copy)]
pub struct RowTransform {
    rows: [[f32; 4]; 4],
    /// The same numbers column by column (`columns[c][r] = rows[r][c]`),
    /// for the NEON version of `clip`.
    #[cfg_attr(not(target_arch = "arm"), allow(dead_code))]
    columns: [[f32; 4]; 4],
    fog_row: Option<[f32; 4]>,
    viewport_origin: (f32, f32),
    viewport_scale: (f32, f32),
    depth_scale: f32,
    depth_center: f32,
    u_scale: f32,
    u_offset: f32,
    v_scale: f32,
    v_offset: f32,
    /// Single-weight skinned draws: the weight whose bone x weight is
    /// folded into `rows` (`Transform::weighted_rows`); vertices with
    /// another weight can't use these rows. 1.0 otherwise.
    pub weight: f32,
}

impl RowTransform {
    /// The clip-space position: rows x (x, y, z, 1), each row as
    /// ((r0 x + r1 y) + r2 z) + r3.
    #[cfg(not(target_arch = "arm"))]
    #[inline(always)]
    fn clip(&self, [x, y, z]: [f32; 3]) -> [f32; 4] {
        self.rows.map(|r| r[0] * x + r[1] * y + r[2] * z + r[3])
    }

    /// The same with NEON (PPSSPP's ARM vertex code does its matrix work
    /// with NEON too): all four rows at once as columns x x + columns x y +
    /// columns x z + column 3. ARMv7 NEON multiplies and adds with a
    /// rounding after each (VMLA is not fused), in the same order as above,
    /// so the results are the same to the bit (but for denormals, which
    /// NEON flushes to zero; the frame's hashes on the Vita and on the PC
    /// are compared).
    #[cfg(target_arch = "arm")]
    #[inline(always)]
    fn clip(&self, [x, y, z]: [f32; 3]) -> [f32; 4] {
        use core::arch::arm::{vaddq_f32, vgetq_lane_f32, vld1q_f32, vmlaq_n_f32, vmulq_n_f32};
        let c = &self.columns;
        // Safe: the Vita's Cortex-A9 has NEON (the target enables it); the
        // loads read the four floats of each column.
        unsafe {
            let mut clip = vmulq_n_f32(vld1q_f32(c[0].as_ptr()), x);
            clip = vmlaq_n_f32(clip, vld1q_f32(c[1].as_ptr()), y);
            clip = vmlaq_n_f32(clip, vld1q_f32(c[2].as_ptr()), z);
            clip = vaddq_f32(clip, vld1q_f32(c[3].as_ptr()));
            [vgetq_lane_f32(clip, 0), vgetq_lane_f32(clip, 1), vgetq_lane_f32(clip, 2), vgetq_lane_f32(clip, 3)]
        }
    }

    /// The finished vertex from the raw position and texture coordinates;
    /// `None` behind the near plane (see `Transform::frame_vertex`).
    #[inline(always)]
    pub fn apply(&self, position: [f32; 3], texcoord: [f32; 2], color: u32) -> Option<FrameVertex> {
        let [clip_x, clip_y, clip_z, clip_w] = self.clip(position);
        if !(clip_z >= -clip_w && clip_w > 0.0) {
            return None;
        }
        let [x, y, z] = position;
        let dot = |r: &[f32; 4]| r[0] * x + r[1] * y + r[2] * z + r[3];
        let over_w = 1.0 / clip_w;
        Some(FrameVertex {
            x: self.viewport_origin.0 + self.viewport_scale.0 * clip_x * over_w,
            y: self.viewport_origin.1 + self.viewport_scale.1 * clip_y * over_w,
            z: (self.depth_center + self.depth_scale * clip_z * over_w).clamp(0.0, 1.0),
            w: clip_w,
            u: texcoord[0] * self.u_scale + self.u_offset,
            v: texcoord[1] * self.v_scale + self.v_offset,
            color,
            fog: self.fog_row.as_ref().map_or(1.0, dot),
        })
    }
}

impl RowTransform {
    /// What the GPU needs to do `apply` itself (`GpuTransform`, the draw's
    /// entry in `Frame::transforms`).
    pub fn gpu_transform(&self) -> GpuTransform {
        GpuTransform {
            rows: self.rows,
            fog_row: self.fog_row.unwrap_or([0.0, 0.0, 0.0, 1.0]),
            viewport: [self.viewport_origin.0, self.viewport_origin.1, self.viewport_scale.0, self.viewport_scale.1],
            depth: [self.depth_center, self.depth_scale, 0.0, 0.0],
            normal_rows: [[0.0; 4]; 3],
            light_set: [0.0; 4],
            // Always set (raw draws use it): the same rows give the same
            // entry whichever path a run takes, so neighbouring raw and old
            // runs share one (`Geometry::add_transform` checks the last).
            raw_uv: self.raw_uv(),
        }
    }

    /// Texture coordinate scale u, v and offset u, v, for raw vertices
    /// (`GpuTransform::raw_uv`: raw texture coordinates x scale + offset).
    pub fn raw_uv(&self) -> [f32; 4] {
        [self.u_scale, self.v_scale, self.u_offset, self.v_offset]
    }

    /// The texture coordinate scale and offset (u, then v) as bits: part of
    /// what a decoded vertex depends on (`vertex_cache::RunKey`).
    pub fn texcoord_bits(&self) -> [u32; 4] {
        [self.u_scale.to_bits(), self.u_offset.to_bits(), self.v_scale.to_bits(), self.v_offset.to_bits()]
    }

    /// True if every point of the model-space box `bounds` is in front of
    /// the near plane (as `model_vertex` tests a vertex). The test is linear
    /// in the position, so the box's 8 corners decide it.
    pub fn in_front(&self, bounds: &[f32; 6]) -> bool {
        (0..8).all(|corner: usize| {
            let x = bounds[if corner & 1 != 0 { 3 } else { 0 }];
            let y = bounds[if corner & 2 != 0 { 4 } else { 1 }];
            let z = bounds[if corner & 4 != 0 { 5 } else { 2 }];
            let dot = |r: &[f32; 4]| r[0] * x + r[1] * y + r[2] * z + r[3];
            let (clip_z, clip_w) = (dot(&self.rows[2]), dot(&self.rows[3]));
            clip_z >= -clip_w && clip_w > 0.0
        })
    }

    /// The vertex for a draw the GPU transforms: the raw position as it is
    /// (model space; `gpu_transform` has the position scale), texture
    /// coordinates and colour as `apply` makes them. `None` where `apply`
    /// gives `None` (behind the near plane): only two of the four rows are
    /// worked out for that test.
    #[inline(always)]
    pub fn model_vertex(&self, position: [f32; 3], texcoord: [f32; 2], color: u32) -> Option<FrameVertex> {
        let [x, y, z] = position;
        let dot = |r: &[f32; 4]| r[0] * x + r[1] * y + r[2] * z + r[3];
        let (clip_z, clip_w) = (dot(&self.rows[2]), dot(&self.rows[3]));
        if !(clip_z >= -clip_w && clip_w > 0.0) {
            return None;
        }
        Some(FrameVertex {
            x,
            y,
            z,
            w: 1.0,
            u: texcoord[0] * self.u_scale + self.u_offset,
            v: texcoord[1] * self.v_scale + self.v_offset,
            color,
            fog: 1.0,
        })
    }
}

/// Row `r` of `m` x `bone` (4x4 times 4x3, both column by column, the bone
/// taken as 4x4 with (0, 0, 0, 1) as its last row): 12 multiplications.
fn folded_row(m: &[f32; 16], r: usize, bone: &[f32; 12]) -> [f32; 4] {
    let (m0, m1, m2) = (m[r], m[4 + r], m[8 + r]);
    let column = |c: usize| m0 * bone[c * 3] + m1 * bone[c * 3 + 1] + m2 * bone[c * 3 + 2];
    [column(0), column(1), column(2), column(3) + m[12 + r]]
}

impl FastPath {
    /// Filled in by `Transform::new`.
    const EMPTY: FastPath = FastPath {
        rows: [[0.0; 4]; 4],
        fog_row: None,
        bones: [[0.0; 12]; BONE_COUNT],
        skinned_rows: [[0.0; 4]; 4],
        skinned_fog_row: None,
        u_scale: 0.0,
        u_offset: 0.0,
        v_scale: 0.0,
        v_offset: 0.0,
        depth_scale: 0.0,
        depth_center: 0.0,
    };
}

/// Fog values can be infinite or NaN (24-bit floats with all exponent
/// bits); PPSSPP replaces them by +-65535.
fn finite(value: f32) -> f32 {
    if value.is_finite() {
        value
    } else if value.is_sign_negative() {
        -65535.0
    } else {
        65535.0
    }
}

/// A 4x3 matrix (4 columns of 3 rows, as the GE uploads it) row by row,
/// for dot products with (x, y, z, 1).
pub fn rows_of_4x3(m: &[f32; 12]) -> [[f32; 4]; 3] {
    core::array::from_fn(|r| [m[r], m[3 + r], m[6 + r], m[9 + r]])
}

/// A 4x3 matrix (4 columns of 3 rows) as 4x4, with (0, 0, 0, 1) as the last row.
fn widen(m: &[f32; 12]) -> [f32; 16] {
    [m[0], m[1], m[2], 0.0, m[3], m[4], m[5], 0.0, m[6], m[7], m[8], 0.0, m[9], m[10], m[11], 1.0]
}

/// a x b, a 4x4, b a 4x3 matrix (taken as 4x4 with (0, 0, 0, 1) as its last
/// row), both column by column. 48 multiplications instead of 64: this runs
/// for every skinned draw (MTX Mototrax: ~6000 per frame).
fn multiply_affine(a: &[f32; 16], b: &[f32; 12]) -> [f32; 16] {
    let mut out = [0.0; 16];
    for column in 0..4 {
        let (b0, b1, b2) = (b[column * 3], b[column * 3 + 1], b[column * 3 + 2]);
        for row in 0..4 {
            let mut value = a[row] * b0 + a[4 + row] * b1 + a[8 + row] * b2;
            if column == 3 {
                value += a[12 + row]; // the translation column times b's implicit 1
            }
            out[column * 4 + row] = value;
        }
    }
    out
}

/// What a draw's `Transform` is worked out from, copied out of the GE
/// state: the matrices, VERTEX_TYPE, the registers `TRANSFORM_REGISTERS`,
/// the format's bones and the lighting registers. The capture thread (the
/// one the game waits for in the GE sync) only copies these
/// (`TransformCache`); the recorder thread works the transform out of them
/// (`TransformBuilder`). Before 2026-11-09 the capture thread built every
/// transform itself (~1.8 ms of a ~33 ms MTX race frame, live).
#[derive(Clone, Copy)]
pub struct TransformInputs {
    format: u32,
    world: [f32; 12],
    view: [f32; 12],
    projection: [f32; 16],
    /// In the order of `TRANSFORM_REGISTERS`.
    registers: [u32; TRANSFORM_REGISTERS.len()],
    /// The format's bones (`bone_count` of them; the rest unused).
    bones: [[f32; 12]; BONE_COUNT],
    bone_count: usize,
    /// `None`: no lights (lighting off, or no light on).
    lighting: Option<LightingRegisters>,
    /// When each part last changed (`TransformCache` numbers every change):
    /// a `TransformBuilder` redoes only the parts that did.
    base_change: u32,
    bone_change: u32,
    lighting_change: u32,
    /// When the camera's part of the base last changed: the view and
    /// projection matrices and `TRANSFORM_REGISTERS`, all `raw_camera`
    /// reads. Kept when the base changed but these didn't (MTX: a new
    /// world matrix per object, the camera once per frame).
    camera_change: u32,
    /// When the lighting registers last changed (`lighting`): kept when the
    /// base changed but they didn't. With `camera_change` (the view) it
    /// names a draw's GPU light set (`Lights::gpu` reads nothing else).
    lights_change: u32,
    /// When the matrices or `TRANSFORM_REGISTERS` were last taken: the base
    /// without the format. Kept when only the format changed (MHF: once per
    /// model part); with `lights_change` it names a raw draw's transform
    /// (world, camera, light set: `raw_blocks::raw_transform`).
    matrices_change: u32,
}

impl TransformInputs {
    pub const EMPTY: TransformInputs = TransformInputs {
        format: 0,
        world: [0.0; 12],
        view: [0.0; 12],
        projection: [0.0; 16],
        registers: [0; TRANSFORM_REGISTERS.len()],
        bones: [[0.0; 12]; BONE_COUNT],
        bone_count: 0,
        lighting: None,
        base_change: 0,
        bone_change: 0,
        lighting_change: 0,
        camera_change: 0,
        lights_change: 0,
        matrices_change: 0,
    };

    /// The inputs of a draw with the current GE state in `format`, on their
    /// own (no change numbers).
    pub fn of(ge: &GeState, format: VertexFormat) -> TransformInputs {
        let mut inputs = TransformInputs::EMPTY;
        inputs.take_base(ge, format);
        inputs.take_bones(ge, format);
        inputs.lighting = LightingRegisters::of(ge);
        inputs
    }

    /// The matrices, VERTEX_TYPE and `TRANSFORM_REGISTERS`. True if the
    /// camera's part (view, projection, registers) changed, bit for bit.
    fn take_base(&mut self, ge: &GeState, format: VertexFormat) -> bool {
        let Matrices { world, view, projection, .. } = &ge.matrices;
        let same_bits = |old: &[f32], new: &[f32]| old.iter().zip(new).all(|(a, b)| a.to_bits() == b.to_bits());
        let mut camera_changed = !same_bits(&self.view, view) || !same_bits(&self.projection, projection);
        self.format = format.0;
        self.world = *world;
        self.view = *view;
        self.projection = *projection;
        for (value, &opcode) in self.registers.iter_mut().zip(&TRANSFORM_REGISTERS) {
            let register = ge.register(opcode);
            camera_changed |= *value != register;
            *value = register;
        }
        camera_changed
    }

    /// The bones the format's weights use (none without weights).
    fn take_bones(&mut self, ge: &GeState, format: VertexFormat) {
        self.bone_count = if format.has_weights() { (format.weight_count() as usize).min(BONE_COUNT) } else { 0 };
        for (index, bone) in self.bones[..self.bone_count].iter_mut().enumerate() {
            *bone = *ge.matrices.bone(index);
        }
    }

    /// Register `opcode`, one of `TRANSFORM_REGISTERS` (a constant opcode:
    /// the compiler finds its place).
    #[inline(always)]
    fn register(&self, opcode: u8) -> u32 {
        let slot = TRANSFORM_REGISTERS.iter().position(|&known| known == opcode).expect("one of TRANSFORM_REGISTERS");
        self.registers[slot]
    }

    /// Whether the draw is fogged (`Transform::has_fog`): FOG_ENABLE, never
    /// for clears.
    pub fn has_fog(&self) -> bool {
        self.register(op::FOG_ENABLE) & 1 != 0 && self.register(op::CLEAR_MODE) & 1 == 0
    }

    /// The lighting registers if lights are on (`Transform::lights` is
    /// `Some` exactly then).
    pub fn lighting(&self) -> Option<&LightingRegisters> {
        self.lighting.as_ref()
    }

    /// When the matrices, VERTEX_TYPE and registers last changed, and when
    /// the lighting did (numbers from `TransformCache`; 0 for inputs made
    /// with `of`): equal numbers, equal parts.
    pub fn changes(&self) -> (u32, u32) {
        (self.base_change, self.lighting_change)
    }

    /// When the matrices and registers were last taken and when the
    /// lighting registers last changed (`TransformCache`'s numbers; 0 for
    /// inputs made with `of`): equal numbers, equal `raw_world`,
    /// `raw_camera` and light sets. The format isn't part of them.
    pub fn raw_changes(&self) -> (u32, u32) {
        (self.matrices_change, self.lights_change)
    }

    /// When the camera's part of the base last changed (`TransformCache`'s
    /// number; 0 for inputs made with `of`): equal numbers, equal
    /// `raw_camera`.
    pub fn camera_change(&self) -> u32 {
        self.camera_change
    }

    /// When the lighting registers last changed (`TransformCache`'s number;
    /// 0 for inputs made with `of`): equal numbers, equal `lighting()`.
    pub fn lights_change(&self) -> u32 {
        self.lights_change
    }

    /// When the bones last changed (`TransformCache`'s number; 0 for inputs
    /// made with `of`).
    pub fn bone_change(&self) -> u32 {
        self.bone_change
    }

    /// The bones the format's weights use (4x3 each, column by column, as
    /// uploaded); empty without weights.
    pub fn bones(&self) -> &[[f32; 12]] {
        &self.bones[..self.bone_count]
    }

    /// The world matrix as `RawTransform::world`: its rows.
    pub fn raw_world(&self) -> [[f32; 4]; 3] {
        rows_of_4x3(&self.world)
    }

    /// The view and projection matrices and the transform registers as
    /// `RawCamera` (raw draws: the vertex program does the rest, as
    /// `Transform::rebuild` does it on the CPU).
    pub fn raw_camera(&self) -> RawCamera {
        let float = |opcode: u8| float24(self.register(opcode));
        let offset = |opcode: u8| (self.register(opcode) & 0xFFFF) as f32 / 16.0;
        let p = &self.projection;
        // Fog as `Transform::fog_row` folds it: (view z + end) x scale.
        let fog = if self.has_fog() {
            [1.0, finite(float(op::FOG_END)), finite(float(op::FOG_SCALE)), 0.0]
        } else {
            [0.0, 1.0, 1.0, 0.0]
        };
        RawCamera {
            view: rows_of_4x3(&self.view),
            projection: core::array::from_fn(|r| [p[r], p[4 + r], p[8 + r], p[12 + r]]),
            viewport: [
                float(op::VIEWPORT_X_CENTER) - offset(op::SCREEN_OFFSET_X),
                float(op::VIEWPORT_Y_CENTER) - offset(op::SCREEN_OFFSET_Y),
                float(op::VIEWPORT_X_SCALE),
                float(op::VIEWPORT_Y_SCALE),
            ],
            depth: [float(op::VIEWPORT_Z_CENTER) / 65535.0, float(op::VIEWPORT_Z_SCALE) / 65535.0, 0.0, 0.0],
            fog,
            texture: [float(op::TEXTURE_SCALE_U), float(op::TEXTURE_SCALE_V), float(op::TEXTURE_OFFSET_U), float(op::TEXTURE_OFFSET_V)],
        }
    }

    /// The light set of a draw the GPU lights with these inputs' lights
    /// (`Lights::gpu`; `None` without lights).
    pub fn gpu_lights(&self, has_vertex_color: bool, opaque: bool) -> Option<GpuLights> {
        let registers = self.lighting.as_ref()?;
        Some(Lights::from_registers(registers, &self.world, &self.view).gpu(has_vertex_color, opaque))
    }

    /// `gpu_lights` through `cache` (the same light set; the lights' part
    /// is reused while only the material changes).
    pub fn gpu_lights_cached(&self, has_vertex_color: bool, opaque: bool, cache: &mut crate::lighting::GpuLightsCache) -> Option<GpuLights> {
        let registers = self.lighting.as_ref()?;
        Some(cache.gpu_lights(registers, &self.world, &self.view, has_vertex_color, opaque))
    }

    /// Copies these inputs into `target` with only the bones they use (the
    /// draw stream's copy of each new transform, `draw_stream`). The unused
    /// bones of `target` keep what they held; nothing reads them.
    pub fn copy_to(&self, target: &mut TransformInputs) {
        // Every field is listed: a new field must be added here too (the
        // compiler checks it, see the destructuring).
        let TransformInputs {
            format,
            world,
            view,
            projection,
            registers,
            bones,
            bone_count,
            lighting,
            base_change,
            bone_change,
            lighting_change,
            camera_change,
            lights_change,
            matrices_change,
        } = self;
        target.format = *format;
        target.world = *world;
        target.view = *view;
        target.projection = *projection;
        target.registers = *registers;
        target.bones[..*bone_count].copy_from_slice(&bones[..*bone_count]);
        target.bone_count = *bone_count;
        // Most transforms have no lights: then only `None` is written.
        target.lighting = match lighting {
            Some(lighting) => Some(*lighting),
            None => None,
        };
        target.base_change = *base_change;
        target.bone_change = *bone_change;
        target.lighting_change = *lighting_change;
        target.camera_change = *camera_change;
        target.lights_change = *lights_change;
        target.matrices_change = *matrices_change;
    }
}

/// The last draw's transform inputs (capture thread). 3D games change the
/// matrices far less often than they draw (MTX: the bones change per draw,
/// the camera once per frame), so most draws reuse them and only take
/// their bones.
///
/// Whether anything changed is told by the GE state's change counters, so a
/// draw with nothing new costs a few compares (like PPSSPP's dirty flags).
pub struct TransformCache {
    /// Updated in place. Meaningful only while `valid`.
    inputs: TransformInputs,
    valid: bool,
    /// Changes whenever `inputs` do (see `generation`).
    generation: u32,
    transform_changes: u32,
    format: u32,
    matrix_generation: u32,
    bone_generation: u32,
    lighting_changes: u32,
    /// `lighting::LIGHTS_ON` when `inputs.lighting` was taken.
    lights_on: bool,
}

impl TransformCache {
    pub const fn new() -> Self {
        TransformCache {
            inputs: TransformInputs::EMPTY,
            valid: false,
            generation: 0,
            transform_changes: 0,
            format: 0,
            matrix_generation: 0,
            bone_generation: 0,
            lighting_changes: 0,
            lights_on: false,
        }
    }

    pub fn forget(&mut self) {
        self.valid = false;
    }

    /// Changes whenever the inputs `get` returns change, so the recorder's
    /// draw stream copies them only when they are new.
    pub fn generation(&self) -> u32 {
        self.generation
    }

    /// `get` and the generation of what it returned.
    pub fn get_with_generation(&mut self, ge: &GeState, format: VertexFormat) -> (&TransformInputs, u32) {
        self.get(ge, format);
        (&self.inputs, self.generation)
    }

    /// The transform inputs for a draw with the current GE state in `format`.
    pub fn get(&mut self, ge: &GeState, format: VertexFormat) -> &TransformInputs {
        let same_base = self.valid
            && self.transform_changes == ge.transform_changes()
            && self.format == format.0
            && self.matrix_generation == ge.matrix_generation();
        if !same_base {
            self.generation = self.generation.wrapping_add(1);
            let inputs = &mut self.inputs;
            // Only the format changed (MHF's intro: ~540 of ~650 redos a
            // frame): the matrices and registers are the ones taken.
            let same_matrices = self.valid
                && self.matrix_generation == ge.matrix_generation()
                && self.transform_changes == ge.transform_changes();
            let camera_changed = if same_matrices {
                inputs.format = format.0;
                false
            } else {
                inputs.matrices_change = self.generation;
                inputs.take_base(ge, format)
            };
            inputs.take_bones(ge, format);
            // The lighting registers are the ones taken last time if none
            // changed since: not copied and compared again (76 words; MHF's
            // intro changes the format ~600 times a frame, the lighting
            // ~120 times).
            let lights_on = crate::lighting::LIGHTS_ON.load(core::sync::atomic::Ordering::Relaxed);
            let same_lighting = self.valid
                && self.lighting_changes == ge.lighting_changes()
                && self.lights_on == lights_on
                && inputs.lights_change != 0;
            if !same_lighting {
                let lighting = LightingRegisters::of(ge);
                if inputs.lighting != lighting || inputs.lights_change == 0 {
                    inputs.lights_change = self.generation;
                }
                inputs.lighting = lighting;
            }
            self.lights_on = lights_on;
            inputs.base_change = self.generation;
            inputs.bone_change = self.generation;
            inputs.lighting_change = self.generation;
            // The camera's number stays while its parts are the same bits
            // (also across `forget`: the numbers only promise equal parts).
            if camera_changed || inputs.camera_change == 0 {
                inputs.camera_change = self.generation;
            }
            self.valid = true;
        } else {
            if format.has_weights() && self.bone_generation != ge.bone_generation() {
                self.generation = self.generation.wrapping_add(1);
                self.inputs.take_bones(ge, format);
                self.inputs.bone_change = self.generation;
            }
            // A lighting register changed (MTX: the material colours between
            // draws): the inputs are new only if the lighting registers of a
            // lit draw are (unlit draws: not).
            if self.lighting_changes != ge.lighting_changes() {
                let lighting = LightingRegisters::of(ge);
                self.lights_on = crate::lighting::LIGHTS_ON.load(core::sync::atomic::Ordering::Relaxed);
                if self.inputs.lighting != lighting {
                    self.generation = self.generation.wrapping_add(1);
                    self.inputs.lighting = lighting;
                    self.inputs.lighting_change = self.generation;
                    self.inputs.lights_change = self.generation;
                }
            }
        }
        self.lighting_changes = ge.lighting_changes();
        self.transform_changes = ge.transform_changes();
        self.format = format.0;
        self.matrix_generation = ge.matrix_generation();
        self.bone_generation = ge.bone_generation();
        &self.inputs
    }
}

/// Works out transforms from their inputs (recorder thread), redoing only
/// the parts whose inputs changed since the last one (`TransformInputs`'
/// change numbers): the bones per draw, the lights when a material colour
/// changed, everything when the matrices did.
pub struct TransformBuilder {
    /// Rebuilt in place (`Transform::rebuild`): a new value moved in cost
    /// ~350 instructions of copies per new transform (1.6 KB, twice).
    transform: Transform,
    valid: bool,
    base_change: u32,
    bone_change: u32,
    lighting_change: u32,
}

impl TransformBuilder {
    pub const fn new() -> Self {
        TransformBuilder { transform: Transform::EMPTY, valid: false, base_change: 0, bone_change: 0, lighting_change: 0 }
    }

    /// The transform `inputs` give. Inputs without change numbers
    /// (`TransformInputs::of`) must not be given: they all look alike.
    pub fn build(&mut self, inputs: &TransformInputs) -> &Transform {
        if !self.valid || self.base_change != inputs.base_change {
            self.transform.rebuild(inputs);
            self.valid = true;
        } else {
            if self.bone_change != inputs.bone_change {
                self.transform.set_bones(inputs);
            }
            if self.lighting_change != inputs.lighting_change {
                self.transform.set_lights(inputs);
            }
        }
        self.base_change = inputs.base_change;
        self.bone_change = inputs.bone_change;
        self.lighting_change = inputs.lighting_change;
        &self.transform
    }

    /// Builds the next transform from scratch (after a frame was dropped or
    /// the stream cleared).
    pub fn forget(&mut self) {
        self.valid = false;
    }
}

impl Default for TransformBuilder {
    fn default() -> Self {
        TransformBuilder::new()
    }
}

impl Transform {
    /// A placeholder (all zero) for arrays of transforms.
    pub const EMPTY: Transform = Transform {
        matrix: [0.0; 16],
        view_world: [0.0; 16],
        fog: None,
        viewport_scale: (0.0, 0.0),
        viewport_origin: (0.0, 0.0),
        depth_scale: 0.0,
        depth_center: 0.0,
        position_scale: 0.0,
        texcoord_scale: 0.0,
        texture_scale: (0.0, 0.0),
        texture_offset: (0.0, 0.0),
        bones: [[0.0; 12]; BONE_COUNT],
        bone_count: 0,
        lights: None,
        fast: FastPath::EMPTY,
    };

    pub fn new(ge: &GeState, format: VertexFormat) -> Transform {
        Transform::from_inputs(&TransformInputs::of(ge, format))
    }

    pub fn from_inputs(inputs: &TransformInputs) -> Transform {
        let mut transform = Transform::EMPTY;
        transform.rebuild(inputs);
        transform
    }

    /// Makes this the transform `inputs` give, in place (`TransformBuilder`).
    /// Field by field rather than a new value: on the Vita a new `Transform`
    /// was built and then copied (memcpy of its 1.6 KB, twice, and the bone
    /// tables cleared). The bone tables keep what they held beyond
    /// `bone_count`: nothing reads them.
    pub fn rebuild(&mut self, inputs: &TransformInputs) {
        self.view_world = multiply_affine(&widen(&inputs.view), &inputs.world);
        let matrix = multiply_affine(&inputs.projection, &inputs.view);
        self.matrix = multiply_affine(&matrix, &inputs.world);
        let format = VertexFormat(inputs.format);
        let float = |opcode: u8| float24(inputs.register(opcode));
        let offset = |opcode: u8| (inputs.register(opcode) & 0xFFFF) as f32 / 16.0;
        // Fog applies to 3D draws only, never to clears.
        let fog_on = inputs.has_fog();
        self.fog = fog_on.then(|| (finite(float(op::FOG_END)), finite(float(op::FOG_SCALE))));
        self.viewport_scale = (float(op::VIEWPORT_X_SCALE), float(op::VIEWPORT_Y_SCALE));
        self.viewport_origin =
            (float(op::VIEWPORT_X_CENTER) - offset(op::SCREEN_OFFSET_X), float(op::VIEWPORT_Y_CENTER) - offset(op::SCREEN_OFFSET_Y));
        self.depth_scale = float(op::VIEWPORT_Z_SCALE);
        self.depth_center = float(op::VIEWPORT_Z_CENTER);
        self.position_scale = format.position_scale();
        self.texcoord_scale = format.texcoord_scale();
        self.texture_scale = (float(op::TEXTURE_SCALE_U), float(op::TEXTURE_SCALE_V));
        self.texture_offset = (float(op::TEXTURE_OFFSET_U), float(op::TEXTURE_OFFSET_V));
        self.bone_count = 0;
        self.set_lights(inputs);
        self.set_base_fast_path();
        self.set_bones(inputs);
    }

    /// The lights `inputs` give (`None` without).
    fn set_lights(&mut self, inputs: &TransformInputs) {
        self.lights = inputs.lighting.as_ref().map(|registers| Lights::from_registers(registers, &inputs.world, &inputs.view));
    }

    /// The parts of the fast path that don't depend on the bones.
    fn set_base_fast_path(&mut self) {
        let scale = self.position_scale;
        // Row r of a column-by-column 4x4, with x, y, z multiplied by `scale`.
        let row = |m: &[f32; 16], r: usize, scale: f32| [m[r] * scale, m[4 + r] * scale, m[8 + r] * scale, m[12 + r]];
        let rows = |m: &[f32; 16], scale: f32| [row(m, 0, scale), row(m, 1, scale), row(m, 2, scale), row(m, 3, scale)];
        let fog_row = self.fog_row(row(&self.view_world, 2, scale));
        let skinned_fog_row = self.fog_row(row(&self.view_world, 2, 1.0));
        let fast = &mut self.fast;
        fast.rows = rows(&self.matrix, scale);
        fast.fog_row = fog_row;
        fast.skinned_rows = rows(&self.matrix, 1.0);
        fast.skinned_fog_row = skinned_fog_row;
        fast.u_scale = self.texcoord_scale * self.texture_scale.0;
        fast.u_offset = self.texture_offset.0;
        fast.v_scale = self.texcoord_scale * self.texture_scale.1;
        fast.v_offset = self.texture_offset.1;
        fast.depth_scale = self.depth_scale / 65535.0;
        fast.depth_center = self.depth_center / 65535.0;
    }

    /// The fog factor row from the view-space z row: FOG_END and FOG_SCALE folded in.
    fn fog_row(&self, view_z: [f32; 4]) -> Option<[f32; 4]> {
        self.fog.map(|(end, scale)| [view_z[0] * scale, view_z[1] * scale, view_z[2] * scale, (view_z[3] + end) * scale])
    }

    /// Takes the current bone matrices (skinned formats only). Runs for
    /// nearly every draw in MTX Mototrax (new bones before each draw, ~3000
    /// per frame), so it does only what the draw needs.
    fn set_bones(&mut self, inputs: &TransformInputs) {
        self.bone_count = inputs.bone_count;
        let scale = self.position_scale;
        for index in 0..self.bone_count {
            let bone = inputs.bones[index];
            self.bones[index] = bone;
            // The fast path's bones take the raw position: x, y, z columns scaled.
            let mut scaled = bone;
            for value in &mut scaled[..9] {
                *value *= scale;
            }
            self.fast.bones[index] = scaled;
        }
        if self.bone_count == 1 {
            // The usual case: the bone folded into the rows, for vertices
            // whose weight is exactly 1.
            let bone = self.fast.bones[0];
            self.fast.rows = [0, 1, 2, 3].map(|r| folded_row(&self.matrix, r, &bone));
            if self.fog.is_some() {
                self.fast.fog_row = self.fog_row(folded_row(&self.view_world, 2, &bone));
            }
        }
    }

    /// The fast path: a vertex from its raw fields to the frame, with
    /// `color` the colour it ends up with (texture function etc. applied).
    /// `None` if it is behind the near plane (its triangles must be cut,
    /// with `to_clip`). Gives what `to_clip`, `to_screen` and the recorder's
    /// conversion give, up to float rounding.
    #[inline(always)]
    pub fn frame_vertex(&self, raw: &Raw3d, color: u32) -> Option<FrameVertex> {
        let fast = &self.fast;
        let folded = self.bone_count == 0 || (self.bone_count == 1 && raw.weights[0] == 1.0);
        let ([x, y, z], rows, fog_row) = if folded {
            (raw.position, &fast.rows, &fast.fog_row)
        } else {
            (Self::skin_raw(fast, self.bone_count, raw), &fast.skinned_rows, &fast.skinned_fog_row)
        };
        let dot = |r: &[f32; 4]| r[0] * x + r[1] * y + r[2] * z + r[3];
        let clip_w = dot(&rows[3]);
        let clip_z = dot(&rows[2]);
        if !(clip_z >= -clip_w && clip_w > 0.0) {
            return None;
        }
        let over_w = 1.0 / clip_w;
        Some(FrameVertex {
            x: self.viewport_origin.0 + self.viewport_scale.0 * dot(&rows[0]) * over_w,
            y: self.viewport_origin.1 + self.viewport_scale.1 * dot(&rows[1]) * over_w,
            // The PSP clamps depth to 0..65535 (per pixel; per vertex here).
            z: (fast.depth_center + fast.depth_scale * clip_z * over_w).clamp(0.0, 1.0),
            w: clip_w,
            u: raw.texcoord[0] * fast.u_scale + fast.u_offset,
            v: raw.texcoord[1] * fast.v_scale + fast.v_offset,
            color,
            fog: fog_row.as_ref().map_or(1.0, dot),
        })
    }

    /// What `frame_vertex` does for a vertex that needs no skinning (no
    /// bones, or one bone with weight 1), copied out for a vertex loop
    /// (`recorder::record_3d_run`): a small value the compiler keeps in
    /// registers, where `frame_vertex` reads `self` again for every vertex.
    pub fn rows(&self) -> RowTransform {
        let fast = &self.fast;
        let r = &fast.rows;
        RowTransform {
            rows: fast.rows,
            columns: core::array::from_fn(|c| [r[0][c], r[1][c], r[2][c], r[3][c]]),
            fog_row: fast.fog_row,
            viewport_origin: self.viewport_origin,
            viewport_scale: self.viewport_scale,
            depth_scale: fast.depth_scale,
            depth_center: fast.depth_center,
            u_scale: fast.u_scale,
            u_offset: fast.u_offset,
            v_scale: fast.v_scale,
            v_offset: fast.v_offset,
            weight: 1.0,
        }
    }

    /// `rows()` for a run of single-weight vertices that all have `weight`:
    /// the GE's skinned position weight x (bone x position) is
    /// (weight x bone) x position, still one affine map, so weight x bone
    /// is folded into the rows like the bone of weight-1 vertices. MTX
    /// Mototrax's riders hang on one bone with weight 248/128; without this
    /// they went the CPU path (no GPU transform, no vertex cache). The
    /// rounding differs a little from skinning on the CPU (the weight is
    /// multiplied in before the matrices, not after).
    pub fn weighted_rows(&self, weight: f32) -> RowTransform {
        if self.bone_count != 1 || weight == 1.0 {
            return self.rows();
        }
        let bone = self.fast.bones[0].map(|value| value * weight);
        let rows = [0, 1, 2, 3].map(|r| folded_row(&self.matrix, r, &bone));
        let fog_row = if self.fog.is_some() { self.fog_row(folded_row(&self.view_world, 2, &bone)) } else { None };
        RowTransform {
            rows,
            columns: core::array::from_fn(|c| [rows[0][c], rows[1][c], rows[2][c], rows[3][c]]),
            fog_row,
            weight,
            ..self.rows()
        }
    }

    /// Draws whose vertices have several weights (bones): skinned on the CPU
    /// while decoding, the rest of the transform on the GPU, as PPSSPP does
    /// ("bones/skinning is now always handled in decode",
    /// VertexShaderGenerator.cpp; its NEON decoder skins, the GPU does world,
    /// view and projection). See `skin_position` and `skinned_row_transform`.
    pub fn skins_several_bones(&self) -> bool {
        self.bone_count > 1
    }

    /// The model-space position of a skinned vertex (from its raw fields):
    /// what `frame_vertex` transforms with the skinned rows.
    #[inline(always)]
    pub fn skin_position(&self, raw: &Raw3d) -> [f32; 3] {
        Self::skin_raw(&self.fast, self.bone_count, raw)
    }

    /// A lit vertex's normal in model space (for `Lights::color`): skinned
    /// like its position, with the bones' 3x3 parts (weights x bone x
    /// normal, summed; not normalized, `Lights` does that), as PPSSPP's
    /// decoder skins normals. Unskinned: as it is.
    #[inline]
    pub fn skin_normal(&self, weights: &[f32; MAX_WEIGHTS], normal: [f32; 3]) -> [f32; 3] {
        if self.bone_count == 0 {
            return normal;
        }
        let [x, y, z] = normal;
        let mut out = [0.0; 3];
        for (bone, &weight) in self.bones[..self.bone_count].iter().zip(weights) {
            if weight == 0.0 {
                continue;
            }
            for row in 0..3 {
                out[row] += weight * (bone[row] * x + bone[3 + row] * y + bone[6 + row] * z);
            }
        }
        out
    }

    /// A vertex's model-space position from its raw fields (position scale
    /// applied, skinned): what the world matrix takes, for point and spot
    /// lights (`Lights::color`).
    #[inline]
    pub fn model_position(&self, raw: &Raw3d) -> [f32; 3] {
        if self.bone_count == 0 {
            raw.position.map(|value| value * self.position_scale)
        } else {
            Self::skin_raw(&self.fast, self.bone_count, raw)
        }
    }

    /// How this draw's vertices get their lit colours (`VertexLighting`);
    /// `None` without lights. Worked out once per run or PRIM.
    pub fn vertex_lighting(&self) -> Option<VertexLighting<'_>> {
        let lights = self.lights.as_ref()?;
        let world = lights.world();
        // Normals reach world space through the world matrix, skinned ones
        // through the bone first (weight x bone x normal: for one bone the
        // weight only scales them). Several bones mix: no model space.
        let model = match self.bone_count {
            0 => Some(core::array::from_fn(|index| world[index])),
            1 => {
                let bone = &self.bones[0];
                // (world x bone) column c = world x (bone column c).
                Some(core::array::from_fn(|index| {
                    let (column, row) = (index / 3, index % 3);
                    (0..3).map(|k| world[3 * k + row] * bone[3 * column + k]).sum()
                }))
            }
            _ => None,
        };
        let model = model.and_then(|model| lights.in_model_space(&model));
        Some(VertexLighting { transform: self, lights, model })
    }

    /// GPU-lit draws: the matrix taking the normals the vertices carry to
    /// world space (`GpuTransform::normal_rows`). One bone: world x bone
    /// (the normals are the raw ones; a positive weight only scales them).
    /// Several bones: the world matrix (the normals are skinned on the CPU,
    /// `skin_normal`). Unskinned: the world matrix.
    pub fn gpu_normal_rows(&self, lights: &Lights) -> [[f32; 4]; 3] {
        lights.normal_rows((self.bone_count == 1).then(|| &self.bones[0]))
    }

    /// The bones of a skinned draw for the GPU (raw draws skinned in the
    /// shader, `Draw::bones`): the fast path's, which take the
    /// raw position (x, y, z columns times the position scale), 4x3 column
    /// by column; with `skinned_row_transform` after them. Empty unskinned.
    pub fn gpu_bones(&self) -> &[[f32; 12]] {
        &self.fast.bones[..self.bone_count]
    }

    /// Whether GPU-lit vertices of this draw skin their normals on the CPU
    /// (several bones; see `gpu_normal_rows`).
    pub fn skins_normals(&self) -> bool {
        self.bone_count > 1
    }

    /// The lit colour of a vertex (`Lights::color`) from its raw fields
    /// and its normal (`None`: the format has none, then (0, 0, 1), as
    /// PPSSPP). Only for transforms with lights.
    #[inline]
    pub fn lit_color(&self, lights: &Lights, raw: &Raw3d, normal: Option<[f32; 3]>) -> u32 {
        let normal = match normal {
            Some(normal) => self.skin_normal(&raw.weights, normal),
            None => [0.0, 0.0, 1.0],
        };
        let position = if lights.needs_position { self.model_position(raw) } else { [0.0; 3] };
        lights.color(raw.color, normal, position)
    }

    /// `rows()` for skinned positions (`skin_position`): world, view and
    /// projection without a bone, the position scale already applied.
    pub fn skinned_row_transform(&self) -> RowTransform {
        let rows = self.fast.skinned_rows;
        RowTransform {
            rows,
            columns: core::array::from_fn(|c| [rows[0][c], rows[1][c], rows[2][c], rows[3][c]]),
            fog_row: self.fast.skinned_fog_row,
            ..self.rows()
        }
    }

    /// True if a vertex with these weights goes through `rows` (`rows()`,
    /// or `weighted_rows` with the bone x the run's weight folded in)
    /// rather than being skinned.
    #[inline(always)]
    pub fn uses_these_rows(&self, rows: &RowTransform, weights: &[f32; MAX_WEIGHTS]) -> bool {
        self.bone_count == 0 || (self.bone_count == 1 && weights[0] == rows.weight)
    }

    /// True if `frame_vertex` uses the rows (`rows()`) for a vertex with
    /// these weights rather than skinning it.
    #[inline(always)]
    pub fn uses_rows(&self, weights: &[f32; MAX_WEIGHTS]) -> bool {
        self.bone_count == 0 || (self.bone_count == 1 && weights[0] == 1.0)
    }

    /// `skin_position` and a skinned normal in one pass over the bones (each
    /// weight looked at once, as PPSSPP's software skinning blends the bones
    /// once for both). The position is exactly `skin_position`'s; the normal
    /// goes through the fast path's bones, whose 3x3 parts carry the
    /// position scale: `skin_normal` times that one scale, the same
    /// direction (all `lighting::pack_scaled_normal` keeps). For GPU-lit
    /// several-bone runs (`recorder::store_gpu_lit_run`).
    #[inline(always)]
    pub fn skin_position_and_normal(&self, raw: &Raw3d, normal: [f32; 3]) -> ([f32; 3], [f32; 3]) {
        let [x, y, z] = raw.position;
        let [nx, ny, nz] = normal;
        let mut position = [0.0; 3];
        let mut skinned_normal = [0.0; 3];
        for (bone, &weight) in self.fast.bones[..self.bone_count].iter().zip(&raw.weights) {
            // `weight == 0.0` (both zeros) as an integer test: no transfer
            // of the VFP flags.
            if weight.to_bits() << 1 == 0 {
                continue;
            }
            for row in 0..3 {
                position[row] += weight * (bone[row] * x + bone[3 + row] * y + bone[6 + row] * z + bone[9 + row]);
                skinned_normal[row] += weight * (bone[row] * nx + bone[3 + row] * ny + bone[6 + row] * nz);
            }
        }
        (position, skinned_normal)
    }

    /// The model position of a skinned vertex from its raw position: the
    /// sum of weight x bone x position (the bones take the raw position).
    #[inline(always)]
    fn skin_raw(fast: &FastPath, bone_count: usize, raw: &Raw3d) -> [f32; 3] {
        let [x, y, z] = raw.position;
        let mut out = [0.0; 3];
        for (bone, &weight) in fast.bones[..bone_count].iter().zip(&raw.weights) {
            if weight == 0.0 {
                continue;
            }
            for row in 0..3 {
                out[row] += weight * (bone[row] * x + bone[3 + row] * y + bone[6 + row] * z + bone[9 + row]);
            }
        }
        out
    }

    /// The model-space position of a skinned vertex: the bone matrices
    /// applied to `position` (x, y, z, 1), weighted and summed.
    fn skin(&self, vertex: &Vertex, position: [f32; 4]) -> [f32; 4] {
        let mut skinned = [0.0, 0.0, 0.0, 1.0];
        for (bone, weight) in self.bones[..self.bone_count].iter().zip(&vertex.weights) {
            if *weight == 0.0 {
                continue;
            }
            for row in 0..3 {
                let moved = bone[row] * position[0] + bone[3 + row] * position[1] + bone[6 + row] * position[2] + bone[9 + row];
                skinned[row] += weight * moved;
            }
        }
        skinned
    }

    /// The vertex after the matrices, with `color` (the vertex's own or the
    /// material colour).
    pub fn to_clip(&self, vertex: &Vertex, color: u32) -> ClipVertex {
        let mut position = [
            vertex.x * self.position_scale,
            vertex.y * self.position_scale,
            vertex.z * self.position_scale,
            1.0,
        ];
        if self.bone_count > 0 {
            position = self.skin(vertex, position);
        }
        let (m, view_world) = (&self.matrix, &self.view_world);
        let clip = |row: usize| (0..4).map(|column| m[column * 4 + row] * position[column]).sum::<f32>();
        let fog = match self.fog {
            None => 1.0,
            Some((end, scale)) => {
                let view_z: f32 = (0..4).map(|column| view_world[column * 4 + 2] * position[column]).sum();
                (view_z + end) * scale
            }
        };
        ClipVertex {
            x: clip(0),
            y: clip(1),
            z: clip(2),
            w: clip(3),
            u: vertex.u * self.texcoord_scale * self.texture_scale.0 + self.texture_offset.0,
            v: vertex.v * self.texcoord_scale * self.texture_scale.1 + self.texture_offset.1,
            color,
            fog,
        }
    }

    /// A clipped vertex in screen pixels, `z` the PSP depth value (`w` must
    /// be > 0, see `clip_triangle`).
    pub fn to_screen(&self, clip: &ClipVertex) -> Vertex {
        // One division instead of three: divisions are slow on the Vita's CPU.
        let over_w = 1.0 / clip.w;
        Vertex {
            x: self.viewport_origin.0 + self.viewport_scale.0 * clip.x * over_w,
            y: self.viewport_origin.1 + self.viewport_scale.1 * clip.y * over_w,
            z: self.depth_center + self.depth_scale * clip.z * over_w,
            u: clip.u,
            v: clip.v,
            color: Some(clip.color),
            ..Vertex::default()
        }
    }

    /// True if this draw has fog.
    pub fn has_fog(&self) -> bool {
        self.fog.is_some()
    }

    /// One vertex in screen pixels, or `None` if it is behind the near
    /// plane. For rectangles, which can't be cut (both corners must be visible).
    pub fn apply(&self, vertex: &Vertex, color: u32) -> Option<Vertex> {
        self.apply_with_fog(vertex, color).map(|(screen, _)| screen)
    }

    /// `apply`, also returning the vertex's fog factor.
    pub fn apply_with_fog(&self, vertex: &Vertex, color: u32) -> Option<(Vertex, f32)> {
        let clip = self.to_clip(vertex, color);
        (clip.visible() && clip.w > 0.0).then(|| (self.to_screen(&clip), clip.fog))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::Command;
    use crate::vertex::Raw3d;

    fn set(ge: &mut GeState, opcode: u8, data: u32) {
        ge.set_register(Command(((opcode as u32) << 24) | data));
    }

    fn set_float(ge: &mut GeState, opcode: u8, value: f32) {
        set(ge, opcode, value.to_bits() >> 8);
    }

    /// Uploads a whole matrix, element by element like a game does.
    fn upload(ge: &mut GeState, number: u8, data: u8, values: &[f32]) {
        set(ge, number, 0);
        for value in values {
            set_float(ge, data, *value);
        }
    }

    const IDENTITY_43: [f32; 12] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];

    /// The usual sceGu setup for a 480x272 screen: viewport centred at
    /// 2048, 2048 and a screen offset of 2048 - 240, 2048 - 136.
    fn ge_with_viewport() -> GeState {
        let mut ge = GeState::new();
        set_float(&mut ge, op::VIEWPORT_X_SCALE, 240.0);
        set_float(&mut ge, op::VIEWPORT_Y_SCALE, -136.0);
        set_float(&mut ge, op::VIEWPORT_X_CENTER, 2048.0);
        set_float(&mut ge, op::VIEWPORT_Y_CENTER, 2048.0);
        set(&mut ge, op::SCREEN_OFFSET_X, (2048 - 240) << 4);
        set(&mut ge, op::SCREEN_OFFSET_Y, (2048 - 136) << 4);
        set_float(&mut ge, op::TEXTURE_SCALE_U, 1.0);
        set_float(&mut ge, op::TEXTURE_SCALE_V, 1.0);
        ge
    }

    fn vertex(x: f32, y: f32, z: f32) -> Vertex {
        Vertex { x, y, z, ..Vertex::default() }
    }

    #[test]
    fn orthographic_projection_maps_to_screen_pixels() {
        let mut ge = ge_with_viewport();
        upload(&mut ge, op::WORLD_MATRIX_NUMBER, op::WORLD_MATRIX_DATA, &IDENTITY_43);
        // View: move everything 10 units right.
        let mut view = IDENTITY_43;
        view[9] = 10.0;
        upload(&mut ge, op::VIEW_MATRIX_NUMBER, op::VIEW_MATRIX_DATA, &view);
        // Orthographic 0..480 x 0..272 (y down), like sceGumOrtho(0, 480, 272, 0, -1, 1).
        let projection = [
            2.0 / 480.0, 0.0, 0.0, 0.0,
            0.0, -2.0 / 272.0, 0.0, 0.0,
            0.0, 0.0, -1.0, 0.0,
            -1.0, 1.0, 0.0, 1.0,
        ];
        upload(&mut ge, op::PROJECTION_MATRIX_NUMBER, op::PROJECTION_MATRIX_DATA, &projection);

        let format = VertexFormat(3 << 7); // float positions
        let transform = Transform::new(&ge, format);
        let screen = transform.apply(&vertex(100.0, 50.0, 0.0), 0).unwrap();
        // 2/480 as a 24-bit float (16-bit mantissa) is off by ~1e-5, as on the GE.
        assert!((screen.x - 110.0).abs() < 0.01 && (screen.y - 50.0).abs() < 0.01, "{:?}", screen);
        assert_eq!(transform.to_clip(&vertex(100.0, 50.0, 0.0), 0).w, 1.0);
    }

    #[test]
    fn texture_coordinates_are_scaled_and_offset() {
        let mut ge = ge_with_viewport();
        set_float(&mut ge, op::TEXTURE_SCALE_U, 0.5);
        set_float(&mut ge, op::TEXTURE_OFFSET_V, 0.25);
        let format = VertexFormat(2 | (3 << 7)); // u16 texcoords, float positions
        let transform = Transform::new(&ge, format);
        // All-zero matrices: w = 0, nothing visible.
        assert!(transform.apply(&vertex(0.0, 0.0, 0.0), 0).is_none());

        let mut projection = [0.0; 16];
        projection[15] = 1.0; // w = 1
        upload(&mut ge, op::PROJECTION_MATRIX_NUMBER, op::PROJECTION_MATRIX_DATA, &projection);
        let transform = Transform::new(&ge, format);
        let screen = transform.apply(&Vertex { u: 32768.0, v: 16384.0, ..vertex(0.0, 0.0, 0.0) }, 0).unwrap();
        assert_eq!((screen.u, screen.v), (0.5, 0.75));
    }

    #[test]
    fn perspective_divides_by_w() {
        let mut ge = ge_with_viewport();
        upload(&mut ge, op::WORLD_MATRIX_NUMBER, op::WORLD_MATRIX_DATA, &IDENTITY_43);
        upload(&mut ge, op::VIEW_MATRIX_NUMBER, op::VIEW_MATRIX_DATA, &IDENTITY_43);
        // w = -z, z = -z - 2: the near plane 1 unit in front of the camera.
        let mut projection = [0.0; 16];
        projection[0] = 1.0;
        projection[5] = 1.0;
        projection[10] = -1.0;
        projection[11] = -1.0;
        projection[14] = -2.0;
        upload(&mut ge, op::PROJECTION_MATRIX_NUMBER, op::PROJECTION_MATRIX_DATA, &projection);
        let transform = Transform::new(&ge, VertexFormat(3 << 7));
        let screen = transform.apply(&vertex(1.0, 0.0, -2.0), 0).unwrap();
        assert!((screen.x - (240.0 + 240.0 * 0.5)).abs() < 1e-3);
        assert_eq!(transform.to_clip(&vertex(1.0, 0.0, -2.0), 0).w, 2.0);
        // Behind the camera, and between the camera and the near plane.
        assert!(transform.apply(&vertex(0.0, 0.0, 2.0), 0).is_none());
        assert!(transform.apply(&vertex(0.0, 0.0, -0.5), 0).is_none());
    }

    #[test]
    fn fog_factor_follows_view_space_depth() {
        let mut ge = ge_with_viewport();
        upload(&mut ge, op::WORLD_MATRIX_NUMBER, op::WORLD_MATRIX_DATA, &IDENTITY_43);
        // View: everything 50 units in front of the camera (z = -50).
        let mut view = IDENTITY_43;
        view[11] = -50.0;
        upload(&mut ge, op::VIEW_MATRIX_NUMBER, op::VIEW_MATRIX_DATA, &view);
        let mut projection = [0.0; 16];
        projection[15] = 1.0;
        upload(&mut ge, op::PROJECTION_MATRIX_NUMBER, op::PROJECTION_MATRIX_DATA, &projection);
        let format = VertexFormat(3 << 7);
        // Fog off: factor 1.
        assert_eq!(Transform::new(&ge, format).to_clip(&vertex(0.0, 0.0, 0.0), 0).fog, 1.0);
        // Final Fantasy's world map: end 65, scale 1/15.
        set(&mut ge, op::FOG_ENABLE, 1);
        set_float(&mut ge, op::FOG_END, 65.0);
        set_float(&mut ge, op::FOG_SCALE, 1.0 / 16.0);
        let transform = Transform::new(&ge, format);
        assert!(transform.has_fog());
        // z = -50: (−50 + 65) / 16; z = -65: all fog.
        assert_eq!(transform.to_clip(&vertex(0.0, 0.0, 0.0), 0).fog, 15.0 / 16.0);
        assert_eq!(transform.to_clip(&vertex(0.0, 0.0, -15.0), 0).fog, 0.0);
    }

    /// The capture thread takes inputs (`TransformCache`), the recorder
    /// builds from them redoing only what changed (`TransformBuilder`): each
    /// build must be what a transform worked out from scratch gives.
    #[test]
    fn transforms_built_part_by_part_are_the_ones_built_whole() {
        let mut ge = ge_with_viewport();
        upload(&mut ge, op::WORLD_MATRIX_NUMBER, op::WORLD_MATRIX_DATA, &IDENTITY_43);
        upload(&mut ge, op::VIEW_MATRIX_NUMBER, op::VIEW_MATRIX_DATA, &IDENTITY_43);
        let mut projection = [0.0; 16];
        projection[0] = 1.0;
        projection[5] = 1.0;
        projection[10] = 0.5;
        projection[15] = 1.0;
        upload(&mut ge, op::PROJECTION_MATRIX_NUMBER, op::PROJECTION_MATRIX_DATA, &projection);
        set(&mut ge, op::LIGHTING_ENABLE, 1);
        set(&mut ge, op::LIGHT_ENABLE_0, 1);
        set_float(&mut ge, op::LIGHT_X_0 + 2, 1.0);
        set(&mut ge, op::LIGHT_AMBIENT_0 + 1, 0xFF_FFFF);
        // Two float weights, float normals and positions.
        let format = VertexFormat((3 << 9) | (1 << 14) | (3 << 7) | (3 << 5));
        let raw = Raw3d { position: [1.0, 2.0, 0.5], texcoord: [0.0, 0.0], color: None, weights: [0.25, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] };
        let mut cache = TransformCache::new();
        let mut builder = TransformBuilder::new();
        let mut check = |ge: &GeState, what: &str| {
            let built = builder.build(cache.get(ge, format));
            let whole = Transform::new(ge, format);
            assert_eq!(built.frame_vertex(&raw, 0), whole.frame_vertex(&raw, 0), "{what}");
            assert_eq!(built.skinned_row_transform().gpu_transform(), whole.skinned_row_transform().gpu_transform(), "{what}");
            assert!(built.lights == whole.lights, "{what}: lights");
        };
        check(&ge, "first");
        let mut bones = [IDENTITY_43; 2];
        bones[0][9] = 10.0;
        bones[1][10] = 20.0;
        upload(&mut ge, op::BONE_MATRIX_NUMBER, op::BONE_MATRIX_DATA, &bones.concat());
        check(&ge, "new bones");
        set(&mut ge, op::MATERIAL_AMBIENT, 0x40_8020);
        check(&ge, "new material colour");
        set(&mut ge, op::LIGHT_ENABLE_0, 0);
        check(&ge, "lights off");
        upload(&mut ge, op::WORLD_MATRIX_NUMBER, op::WORLD_MATRIX_DATA, &[2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0]);
        check(&ge, "new world matrix");
        set(&mut ge, op::LIGHT_ENABLE_0, 1);
        bones[1][9] = -5.0;
        upload(&mut ge, op::BONE_MATRIX_NUMBER, op::BONE_MATRIX_DATA, &bones.concat());
        check(&ge, "lights on, new bones");
    }

    /// A new format redoes the inputs; their lighting registers are taken
    /// again only if one changed, and must be what taking them gives.
    #[test]
    fn a_new_format_keeps_the_lighting_unless_a_lighting_register_changed() {
        let mut ge = ge_with_viewport();
        upload(&mut ge, op::WORLD_MATRIX_NUMBER, op::WORLD_MATRIX_DATA, &IDENTITY_43);
        set(&mut ge, op::LIGHTING_ENABLE, 1);
        set(&mut ge, op::LIGHT_ENABLE_0, 1);
        set(&mut ge, op::MATERIAL_AMBIENT, 0x10_2030);
        let skinned = VertexFormat((3 << 9) | (1 << 14) | (3 << 7) | (3 << 5));
        let plain = VertexFormat((3 << 7) | (3 << 5));
        fn check(cache: &mut TransformCache, ge: &GeState, format: VertexFormat, what: &str) -> u32 {
            let inputs = cache.get(ge, format);
            assert_eq!(inputs.lighting(), TransformInputs::of(ge, format).lighting(), "{what}");
            let raw = Raw3d { position: [1.0, 2.0, 0.5], texcoord: [0.5, 0.25], color: None, weights: [1.0; 8] };
            let built = Transform::from_inputs(inputs);
            let whole = Transform::new(ge, format);
            assert_eq!(built.frame_vertex(&raw, 0), whole.frame_vertex(&raw, 0), "{what}");
            assert_eq!(built.rows().gpu_transform(), whole.rows().gpu_transform(), "{what}");
            assert!(built.lights == whole.lights, "{what}: lights");
            inputs.lights_change()
        }
        let mut cache = TransformCache::new();
        let first = check(&mut cache, &ge, skinned, "first");
        let camera = cache.get(&ge, skinned).camera_change();
        assert_eq!(check(&mut cache, &ge, plain, "new format"), first, "same lighting, same number");
        assert_eq!(cache.get(&ge, plain).camera_change(), camera, "same camera");
        set(&mut ge, op::VIEWPORT_X_SCALE, 0x43_0000);
        check(&mut cache, &ge, skinned, "new format and viewport");
        let mut view = IDENTITY_43;
        view[11] = -3.0;
        upload(&mut ge, op::VIEW_MATRIX_NUMBER, op::VIEW_MATRIX_DATA, &view);
        check(&mut cache, &ge, plain, "new format and view");
        assert_ne!(cache.get(&ge, plain).camera_change(), camera, "new camera");
        set(&mut ge, op::MATERIAL_AMBIENT, 0x40_5060);
        let second = check(&mut cache, &ge, skinned, "new format and material colour");
        assert_ne!(second, first);
        set(&mut ge, op::LIGHT_ENABLE_0, 0);
        check(&mut cache, &ge, plain, "new format, lights off");
        set(&mut ge, op::LIGHT_ENABLE_0, 1);
        upload(&mut ge, op::WORLD_MATRIX_NUMBER, op::WORLD_MATRIX_DATA, &IDENTITY_43);
        check(&mut cache, &ge, skinned, "new format and matrix, lights on again");
        cache.forget();
        set(&mut ge, op::MATERIAL_AMBIENT, 0x70_8090);
        check(&mut cache, &ge, skinned, "forgotten");
    }

    #[test]
    fn inputs_copied_with_their_used_bones_only_transform_the_same() {
        let mut ge = ge_with_viewport();
        upload(&mut ge, op::WORLD_MATRIX_NUMBER, op::WORLD_MATRIX_DATA, &IDENTITY_43);
        upload(&mut ge, op::VIEW_MATRIX_NUMBER, op::VIEW_MATRIX_DATA, &IDENTITY_43);
        let mut projection = [0.0; 16];
        projection[0] = 1.0;
        projection[5] = 1.0;
        projection[15] = 1.0;
        upload(&mut ge, op::PROJECTION_MATRIX_NUMBER, op::PROJECTION_MATRIX_DATA, &projection);
        let mut bones = [IDENTITY_43; 2];
        bones[0][9] = 10.0;
        bones[1][10] = 20.0;
        upload(&mut ge, op::BONE_MATRIX_NUMBER, op::BONE_MATRIX_DATA, &bones.concat());
        let format = VertexFormat((3 << 9) | (1 << 14) | (3 << 7));
        let original = Transform::new(&ge, format);
        // The ring slot held another draw's inputs (other bones, more of them).
        let mut other_ge = ge.clone();
        upload(&mut other_ge, op::BONE_MATRIX_NUMBER, op::BONE_MATRIX_DATA, &[[2.0; 12]; 8].concat());
        let mut inputs = TransformInputs::of(&other_ge, VertexFormat((3 << 9) | (7 << 14) | (3 << 7)));
        TransformInputs::of(&ge, format).copy_to(&mut inputs);
        let slot = Transform::from_inputs(&inputs);
        let mut skinned = vertex(1.0, 2.0, 0.0);
        skinned.weights[0] = 0.5;
        skinned.weights[1] = 0.5;
        let raw = Raw3d { position: [1.0, 2.0, 0.0], texcoord: [0.0, 0.0], color: None, weights: skinned.weights };
        assert_eq!(slot.to_clip(&skinned, 0), original.to_clip(&skinned, 0));
        assert_eq!(slot.frame_vertex(&raw, 0), original.frame_vertex(&raw, 0));
        assert_eq!(slot.skin_position(&raw), [6.0, 12.0, 0.0]);
        assert_eq!(slot.skinned_row_transform().gpu_transform(), original.skinned_row_transform().gpu_transform());
        assert_eq!(slot.rows().gpu_transform(), original.rows().gpu_transform());
    }

    #[test]
    fn skinned_positions_blend_the_bone_matrices() {
        let mut ge = ge_with_viewport();
        upload(&mut ge, op::WORLD_MATRIX_NUMBER, op::WORLD_MATRIX_DATA, &IDENTITY_43);
        upload(&mut ge, op::VIEW_MATRIX_NUMBER, op::VIEW_MATRIX_DATA, &IDENTITY_43);
        let mut projection = [0.0; 16];
        projection[0] = 1.0;
        projection[5] = 1.0;
        projection[15] = 1.0; // clip = (x, y, 0, 1)
        upload(&mut ge, op::PROJECTION_MATRIX_NUMBER, op::PROJECTION_MATRIX_DATA, &projection);
        // Bone 0 moves 10 right, bone 1 moves 20 up.
        let mut bones = [IDENTITY_43; 2];
        bones[0][9] = 10.0;
        bones[1][10] = 20.0;
        upload(&mut ge, op::BONE_MATRIX_NUMBER, op::BONE_MATRIX_DATA, &bones.concat());

        // Float weights (2 of them), float positions.
        let format = VertexFormat((3 << 9) | (1 << 14) | (3 << 7));
        let transform = Transform::new(&ge, format);
        let mut skinned = vertex(1.0, 2.0, 0.0);
        skinned.weights[0] = 1.0;
        let clip = transform.to_clip(&skinned, 0);
        assert_eq!((clip.x, clip.y), (11.0, 2.0), "one weight: that bone alone");
        skinned.weights[0] = 0.5;
        skinned.weights[1] = 0.5;
        let clip = transform.to_clip(&skinned, 0);
        assert_eq!((clip.x, clip.y), (6.0, 12.0), "half of each");

        // One weight: the bone folded into the matrix gives the same, and a
        // weight other than 1 still scales the bone's result.
        let single = Transform::new(&ge, VertexFormat((3 << 9) | (3 << 7)));
        let mut one = vertex(1.0, 2.0, 0.0);
        one.weights[0] = 1.0;
        assert_eq!(single.to_clip(&one, 0).x, 11.0);
        one.weights[0] = 0.5;
        assert_eq!(single.to_clip(&one, 0).x, 5.5);

        // The same vertex without weights in the format: bones ignored.
        let clip = Transform::new(&ge, VertexFormat(3 << 7)).to_clip(&skinned, 0);
        assert_eq!((clip.x, clip.y), (1.0, 2.0));
    }

    #[test]
    fn the_fast_path_gives_what_the_long_way_gives() {
        let mut ge = ge_with_viewport();
        let mut world = IDENTITY_43;
        world[9] = 3.0;
        upload(&mut ge, op::WORLD_MATRIX_NUMBER, op::WORLD_MATRIX_DATA, &world);
        let mut view = IDENTITY_43;
        view[11] = -20.0;
        upload(&mut ge, op::VIEW_MATRIX_NUMBER, op::VIEW_MATRIX_DATA, &view);
        // Perspective: w = -z, near plane at z = -1.
        let mut projection = [0.0; 16];
        projection[0] = 1.5;
        projection[5] = 2.0;
        projection[10] = -1.0;
        projection[11] = -1.0;
        projection[14] = -2.0;
        upload(&mut ge, op::PROJECTION_MATRIX_NUMBER, op::PROJECTION_MATRIX_DATA, &projection);
        let mut bone = IDENTITY_43;
        bone[0] = 0.5; // scales x
        bone[10] = 4.0; // moves up
        upload(&mut ge, op::BONE_MATRIX_NUMBER, op::BONE_MATRIX_DATA, &bone);
        set_float(&mut ge, op::VIEWPORT_Z_SCALE, -32767.5);
        set_float(&mut ge, op::VIEWPORT_Z_CENTER, 32767.5);
        set(&mut ge, op::FOG_ENABLE, 1);
        set_float(&mut ge, op::FOG_END, 30.0);
        set_float(&mut ge, op::FOG_SCALE, 0.05);
        set_float(&mut ge, op::TEXTURE_SCALE_U, 2.0);
        set_float(&mut ge, op::TEXTURE_OFFSET_V, 0.25);

        // u8 weight, u16 texcoords, s16 positions; and the same without weights.
        for format in [VertexFormat((1 << 9) | 2 | (2 << 7)), VertexFormat(2 | (2 << 7))] {
            let transform = Transform::new(&ge, format);
            for (x, y, z) in [(1000.0, -2000.0, 3000.0), (-16000.0, 500.0, -9000.0), (0.0, 0.0, 0.0)] {
                let vertex = Vertex { x, y, z, u: 16384.0, v: 8192.0, weights: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], ..Vertex::default() };
                let raw = Raw3d { position: [x, y, z], texcoord: [16384.0, 8192.0], color: None, weights: vertex.weights };
                let fast = transform.frame_vertex(&raw, 0xFF00_00FF).unwrap();
                let clip = transform.to_clip(&vertex, 0xFF00_00FF);
                let screen = transform.to_screen(&clip);
                let close = |a: f32, b: f32| (a - b).abs() <= 1e-3 * (1.0 + b.abs());
                assert!(close(fast.x, screen.x) && close(fast.y, screen.y), "{fast:?} {screen:?}");
                assert!(close(fast.z, screen.z.clamp(0.0, 65535.0) / 65535.0), "{fast:?} {screen:?}");
                assert!(close(fast.w, clip.w) && close(fast.fog, clip.fog), "{fast:?} {clip:?}");
                assert!(close(fast.u, screen.u) && close(fast.v, screen.v));
            }
        }
        // Skinned with weights other than 1: two bones, and one bone at 0.5.
        let mut second = IDENTITY_43;
        second[9] = -6.0;
        upload(&mut ge, op::BONE_MATRIX_NUMBER, op::BONE_MATRIX_DATA, &[bone, second].concat());
        let two_weights = VertexFormat((1 << 9) | (1 << 14) | 2 | (2 << 7));
        for (format, weights) in [(two_weights, [0.25, 0.75]), (VertexFormat((1 << 9) | 2 | (2 << 7)), [0.5, 0.0])] {
            let transform = Transform::new(&ge, format);
            let mut vertex = Vertex { x: 3000.0, y: -1000.0, z: -500.0, ..Vertex::default() };
            vertex.weights[..2].copy_from_slice(&weights);
            let raw = Raw3d { position: [vertex.x, vertex.y, vertex.z], texcoord: [0.0; 2], color: None, weights: vertex.weights };
            let fast = transform.frame_vertex(&raw, 0).unwrap();
            let screen = transform.to_screen(&transform.to_clip(&vertex, 0));
            assert!((fast.x - screen.x).abs() < 1e-3 && (fast.y - screen.y).abs() < 1e-3, "{fast:?} {screen:?}");
        }
    }

    #[test]
    fn the_gpu_transform_gives_what_the_cpu_gives() {
        let mut ge = ge_with_viewport();
        let mut view = IDENTITY_43;
        view[11] = -20.0;
        upload(&mut ge, op::VIEW_MATRIX_NUMBER, op::VIEW_MATRIX_DATA, &view);
        let mut projection = [0.0; 16];
        projection[0] = 1.5;
        projection[5] = 2.0;
        projection[10] = -1.0;
        projection[11] = -1.0;
        projection[14] = -2.0;
        upload(&mut ge, op::PROJECTION_MATRIX_NUMBER, op::PROJECTION_MATRIX_DATA, &projection);
        set_float(&mut ge, op::VIEWPORT_Z_SCALE, -32767.5);
        set_float(&mut ge, op::VIEWPORT_Z_CENTER, 32767.5);
        set_float(&mut ge, op::TEXTURE_SCALE_U, 2.0);
        for fog in [false, true] {
            set(&mut ge, op::FOG_ENABLE, fog as u32);
            set_float(&mut ge, op::FOG_END, 30.0);
            set_float(&mut ge, op::FOG_SCALE, 0.05);
            let rows = Transform::new(&ge, VertexFormat(2 | (2 << 7))).rows();
            let gpu = rows.gpu_transform();
            // In front, behind the near plane, behind the camera.
            for position in [[1000.0, -2000.0, 3000.0], [-16000.0, 500.0, -9000.0], [0.0, 0.0, 30000.0], [0.0, 0.0, 600000.0]] {
                let cpu = rows.apply(position, [16384.0, 8192.0], 0xFF00_00FF);
                let model = rows.model_vertex(position, [16384.0, 8192.0], 0xFF00_00FF);
                assert_eq!(cpu.is_some(), model.is_some(), "{position:?}");
                if let (Some(cpu), Some(model)) = (cpu, model) {
                    assert_eq!(gpu.to_screen(&model), cpu, "fog {fog}");
                }
            }
        }
    }

    #[test]
    fn a_runs_single_weight_folds_into_the_rows() {
        let mut ge = ge_with_viewport();
        let mut view = IDENTITY_43;
        view[11] = -20.0;
        upload(&mut ge, op::VIEW_MATRIX_NUMBER, op::VIEW_MATRIX_DATA, &view);
        let mut projection = [0.0; 16];
        projection[0] = 1.5;
        projection[5] = 2.0;
        projection[10] = -1.0;
        projection[11] = -1.0;
        projection[14] = -2.0;
        upload(&mut ge, op::PROJECTION_MATRIX_NUMBER, op::PROJECTION_MATRIX_DATA, &projection);
        // A bone that turns, scales and moves.
        let bone = [0.0, 1.5, 0.0, -1.5, 0.0, 0.0, 0.0, 0.0, 1.5, 4.0, -3.0, 2.0];
        upload(&mut ge, op::BONE_MATRIX_NUMBER, op::BONE_MATRIX_DATA, &bone);
        set(&mut ge, op::FOG_ENABLE, 1);
        set_float(&mut ge, op::FOG_END, 30.0);
        set_float(&mut ge, op::FOG_SCALE, 0.05);
        // 8-bit weights (one), 16-bit positions: MTX's riders.
        let transform = Transform::new(&ge, VertexFormat((1 << 9) | (2 << 7)));
        let weight = 248.0 / 128.0;
        let rows = transform.weighted_rows(weight);
        assert_eq!(rows.weight, weight);
        assert_eq!(transform.weighted_rows(1.0).weight, 1.0, "weight 1: the usual rows");
        let gpu = rows.gpu_transform();
        for position in [[1000.0, -2000.0, 3000.0], [-1600.0, 500.0, -900.0], [300.0, 200.0, 100.0]] {
            let mut weights = [0.0; MAX_WEIGHTS];
            weights[0] = weight;
            let raw = Raw3d { position, texcoord: [16384.0, 8192.0], color: None, weights };
            let cpu = transform.frame_vertex(&raw, 0xFF00_00FF);
            let model = rows.model_vertex(position, [16384.0, 8192.0], 0xFF00_00FF);
            assert_eq!(cpu.is_some(), model.is_some(), "{position:?}");
            if let (Some(cpu), Some(model)) = (cpu, model) {
                // Same up to rounding: the weight goes in before the matrices.
                let screen = gpu.to_screen(&model);
                for (a, b) in [(screen.x, cpu.x), (screen.y, cpu.y), (screen.z, cpu.z), (screen.w, cpu.w), (screen.fog, cpu.fog)] {
                    assert!((a - b).abs() <= 1e-4 * b.abs().max(1.0), "{position:?}: {a} vs {b}");
                }
            }
        }
    }

    fn clip_vertex(z: f32, w: f32, u: f32) -> ClipVertex {
        ClipVertex { x: 0.0, y: 0.0, z, w, u, v: 0.0, color: 0xFF00_00FF, fog: u }
    }

    #[test]
    fn triangles_are_cut_at_the_near_plane() {
        let mut out = [clip_vertex(0.0, 0.0, 0.0); 4];
        // All in front: unchanged.
        let triangle = [clip_vertex(0.0, 1.0, 0.0), clip_vertex(0.0, 1.0, 0.5), clip_vertex(0.0, 1.0, 1.0)];
        assert_eq!(clip_triangle(&triangle, &mut out), 3);
        assert_eq!(&out[..3], &triangle);
        // All behind: nothing.
        let behind = [clip_vertex(-2.0, 1.0, 0.0); 3];
        assert_eq!(clip_triangle(&behind, &mut out), 0);
        // One corner behind (z + w = -1, the others +1): a quad, cut halfway.
        let triangle = [clip_vertex(-2.0, 1.0, 0.0), clip_vertex(0.0, 1.0, 1.0), clip_vertex(0.0, 1.0, 1.0)];
        assert_eq!(clip_triangle(&triangle, &mut out), 4);
        assert_eq!(out[0].u, 0.5);
        assert_eq!(out[0].fog, 0.5, "fog is cut like the other values");
        assert_eq!(out[0].z + out[0].w, 0.0);
        assert_eq!((out[1].u, out[2].u), (1.0, 1.0));
        assert_eq!(out[3].u, 0.5);
    }
}
