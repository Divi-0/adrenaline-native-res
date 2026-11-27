//! The GE's per-vertex lighting with lights on: MTX Mototrax lights its
//! riders and bikes with three directional lights (diffuse and specular),
//! ~2,200 PRIMs per race frame. Done on the CPU while the vertices are
//! decoded, as PPSSPP's hardware-transform vertex shader does it
//! (GPU/Common/VertexShaderGenerator.cpp; same names where possible):
//!
//! ```text
//! colour0 = emissive + ambient x ambientColor
//!         + sum over lights of (light ambient x ambientColor + diffuse) x scale
//! diffuse = light diffuse x diffuseColor x max(N.L, 0)          (N.L powered for "powered diffuse")
//! colour1 = sum over specular lights of light specular x specularColor
//!           x pspPow(N.H, specular coefficient) x scale   (when N.L >= 0; H = normalize(L + view direction))
//! ```
//!
//! N is the vertex normal (skinned at decode, reversed with REVERSENORMAL)
//! in world space, normalized; L points to the light (directional: the
//! light's normalized direction; point and spot: from the vertex, with
//! distance attenuation, spots also by angle). ambientColor, diffuseColor
//! and specularColor are the vertex colour where MATERIAL_UPDATE says so,
//! else the material's.
//!
//! The GE adds colour1 (the specular part, LIGHTMODE 1) after texturing;
//! our vertices have one colour, so it is added to colour0 here (as LIGHTMODE
//! 0 does), counted as an approximation (`Lights::approximated`).

use core::sync::atomic::{AtomicBool, Ordering};

use crate::command::op;
use crate::frame::GpuLights;
use crate::ge_state::{float24, GeState};

/// Lit draws with lights drawn lit (true, default) or
/// with their vertex colours as they are (false, as before lighting step 2).
/// Read when a draw's transform is worked out.
pub static LIGHTS_ON: AtomicBool = AtomicBool::new(true);

/// Draws with point lights lit by the raw programs (true, default; the
/// Vita turns it off if its point-light programs aren't there) as PPSSPP's
/// vertex shader does (VertexShaderGenerator.cpp: toLight, inversesqrt,
/// attenuation clamp(1 / (a + b d + c d^2))), or on the CPU (false). User
/// decision 2026-11-22: PPSSPP's way, though not bit-exact with the CPU
/// path (God of War's ~700 point-lit draws a frame). Point lights with
/// specular and spot lights stay on the CPU.
pub static POINT_LIGHTS_GPU: AtomicBool = AtomicBool::new(true);

/// GPU-lit colours in the GE's integer arithmetic (`GpuLights::ge_color`),
/// as the vertex programs of God of War's program set (GE_LIGHT, core
/// game_quirks.rs `PSP_DEPTH_GAMES`; the Vita sets this with the set, the
/// replay with `--ge-light`). Off: float, rounded once (`float_color`), as
/// every other game's programs.
pub static GE_LIGHTING: AtomicBool = AtomicBool::new(false);

/// One enabled light, its registers decoded.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Light {
    /// LIGHTTYPE bits 8..9: 0 directional, 1 point, 2 (and 3) spot.
    kind: u32,
    /// LIGHTTYPE bits 0..1: 0 diffuse, 1 diffuse and specular, 2 powered
    /// diffuse.
    computation: u32,
    /// Directional: the direction to the light, normalized; else its
    /// position (world space).
    position: [f32; 3],
    /// Spots: the direction the light shines in, normalized.
    direction: [f32; 3],
    /// Constant, linear and quadratic attenuation.
    attenuation: [f32; 3],
    spot_cutoff: f32,
    spot_exponent: f32,
    ambient: [f32; 3],
    diffuse: [f32; 3],
    specular: [f32; 3],
}

const DIRECTIONAL: u32 = 0;
const POINT: u32 = 1;
const DIFFUSE_AND_SPECULAR: u32 = 1;
const POWERED_DIFFUSE: u32 = 2;

/// The lighting of a draw with lights on (`Lights::of`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Lights {
    lights: [Light; 4],
    count: usize,
    /// AMBIENT_COLOR / AMBIENT_ALPHA, 0..1.
    ambient: [f32; 4],
    emissive: [f32; 3],
    /// MATERIAL_AMBIENT / MATERIAL_ALPHA, MATERIAL_DIFFUSE, MATERIAL_SPECULAR.
    material_ambient: [f32; 4],
    material_diffuse: [f32; 3],
    material_specular: [f32; 3],
    specular_coefficient: f32,
    /// MATERIAL_UPDATE: bit 0 ambient, 1 diffuse, 2 specular from the vertex
    /// colour (for vertices that have one).
    material_update: u32,
    /// LIGHTMODE 1: the GE adds the specular part after texturing.
    pub separate_specular: bool,
    reverse_normal: bool,
    /// The world matrix (4x3, column by column) for normals and positions.
    world: [f32; 12],
    /// Towards the viewer for specular half vectors: the view matrix's z axis.
    view_direction: [f32; 3],
    /// Whether any light needs the vertex position (point and spot lights).
    pub needs_position: bool,
}

/// 0..1 colour channels of an RGB(A) register value (0xAABBGGRR).
fn channels(word: u32) -> [f32; 4] {
    bytes(word).map(|value| value * (1.0 / 255.0))
}

/// 0..255 colour channels of an RGBA value (0xAABBGGRR).
#[inline(always)]
fn bytes(word: u32) -> [f32; 4] {
    core::array::from_fn(|channel| ((word >> (8 * channel)) & 0xFF) as f32)
}

/// RGBA in 0..255 units, clamped and rounded, packed as 0xAABBGGRR.
///
/// Clamped after the conversion, in integers: `(v + 0.5) as i32` saturates
/// and turns NaN into 0, and `clamp(0, 255)` is one `usat`. The same bits as
/// clamping the float first (`(v.clamp(0.0, 255.0) + 0.5) as u32`, the form
/// before) for every input: in 0..=255 it is the same add; below 0 the sum is
/// under 0.5, so 0 either way; above 255 it is over 255.5, so 255 either way;
/// NaN is 0 both ways. The float clamp cost two compares (`vcmp` + `vmrs`)
/// per channel.
#[inline(always)]
fn pack(value: [f32; 4]) -> u32 {
    let mut color = 0u32;
    for (channel, value) in value.into_iter().enumerate() {
        color |= (((value + 0.5) as i32).clamp(0, 255) as u32) << (8 * channel);
    }
    color
}

fn rgb(word: u32) -> [f32; 3] {
    let [r, g, b, _] = channels(word);
    [r, g, b]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// 1 / sqrt(x) for x > 0: a first guess from the float's bits, then three
/// Newton steps (core has no square root).
fn inverse_sqrt(x: f32) -> f32 {
    let mut y = f32::from_bits(0x5F37_59DF - (x.to_bits() >> 1));
    for _ in 0..3 {
        y *= 1.5 - 0.5 * x * y * y;
    }
    y
}

/// `v` normalized; (0, 0, 1) for the zero vector (PPSSPP's normalizeOr001).
pub(crate) fn normalize_or_001(v: [f32; 3]) -> [f32; 3] {
    let length2 = dot(v, v);
    if !(length2 > 0.0) || !length2.is_finite() {
        return [0.0, 0.0, 1.0];
    }
    let scale = inverse_sqrt(length2);
    v.map(|value| value * scale)
}

/// The GE's pow for lighting (PPSSPP's pspPow): 1 for e <= 0, else 0 for
/// x <= 0, else exp2(e * log2(x)) with log2 and exp2 each a straight line
/// between powers of two, which reading a float's bits as an integer gives.
pub fn psp_pow(x: f32, e: f32) -> f32 {
    if e <= 0.0 {
        return 1.0;
    }
    if !(x > 0.0) {
        return 0.0;
    }
    const ONE: f32 = 1_065_353_216.0; // 1.0's bits
    let t = e * (x.max(1e-30).to_bits() as f32 - ONE) + ONE;
    f32::from_bits(t.max(0.0) as u32)
}

/// The lighting registers (`ge_state::is_lighting_register`, 76 of them) of
/// a draw with lights on, copied out of the GE state: the capture thread
/// takes them (`transform::TransformInputs`), the recorder thread works
/// `Lights` out of them (`Lights::from_registers`), so the capture thread,
/// the one the game waits for, doesn't.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LightingRegisters {
    words: [u32; LIGHTING_REGISTER_COUNT],
}

const LIGHTING_REGISTER_COUNT: usize = 76;

/// Where lighting register `opcode` is kept in `LightingRegisters::words`
/// (0x17..=0x1B, 0x51, 0x53..=0x58, 0x5B..=0x9A one after the other).
const fn lighting_slot(opcode: u8) -> usize {
    match opcode {
        0x17..=0x1B => (opcode - 0x17) as usize,
        0x51 => 5,
        0x53..=0x58 => 6 + (opcode - 0x53) as usize,
        0x5B..=0x9A => 12 + (opcode - 0x5B) as usize,
        _ => panic!("not a lighting register"),
    }
}

impl LightingRegisters {
    /// For a 3D draw with the current GE state: `Some` if lighting is on
    /// and at least one light is (when `Lights::of` gives lights).
    pub fn of(ge: &GeState) -> Option<LightingRegisters> {
        if !LIGHTS_ON.load(Ordering::Relaxed) {
            return None;
        }
        // The cheap checks first: this runs whenever a lighting register
        // changes (`TransformCache::get`), also for unlit draws.
        let any_light = (0..4u8).any(|number| ge.register(op::LIGHT_ENABLE_0 + number) & 1 != 0);
        if ge.register(op::LIGHTING_ENABLE) & 1 == 0 || !any_light {
            return None;
        }
        // Four runs of registers, each copied in one piece.
        let registers = ge.registers();
        let mut words = [0; LIGHTING_REGISTER_COUNT];
        let mut copy = |first: u8, last: u8| {
            let slot = lighting_slot(first);
            let count = (last - first) as usize + 1;
            words[slot..slot + count].copy_from_slice(&registers[first as usize..=last as usize]);
        };
        copy(0x17, 0x1B);
        copy(0x51, 0x51);
        copy(0x53, 0x58);
        copy(0x5B, 0x9A);
        Some(LightingRegisters { words })
    }

    #[inline]
    fn register(&self, opcode: u8) -> u32 {
        self.words[lighting_slot(opcode)]
    }

    /// The LIGHT_TYPE registers of the enabled lights.
    fn enabled_types(&self) -> impl Iterator<Item = u32> + '_ {
        (0..4u8)
            .filter(|&number| self.register(op::LIGHT_ENABLE_0 + number) & 1 != 0)
            .map(|number| self.register(op::LIGHT_TYPE_0 + number))
    }

    /// `Lights::approximated` of the lights these registers give.
    pub fn approximated(&self) -> bool {
        self.register(op::LIGHT_MODE) & 1 != 0 && self.enabled_types().any(|kind| kind & 3 == DIFFUSE_AND_SPECULAR)
    }

    /// `Lights::gpu_ready` of the lights these registers give.
    pub fn gpu_ready(&self) -> bool {
        self.enabled_types().all(|kind| (kind >> 8) & 3 == DIRECTIONAL)
    }

    /// Whether the raw programs can light draws of `format` with these
    /// lights: directional ones, and point lights without specular
    /// (`POINT_LIGHTS_GPU`) in draws of at most four bones (only the
    /// point-light variants of the one- and four-bone programs take them).
    pub fn raw_ready(&self, format: crate::vertex::VertexFormat) -> bool {
        if self.gpu_ready() {
            return true;
        }
        POINT_LIGHTS_GPU.load(Ordering::Relaxed)
            && (!format.has_weights() || format.weight_count() <= 4)
            && self.enabled_types().all(|kind| match (kind >> 8) & 3 {
                DIRECTIONAL => true,
                POINT => kind & 3 != DIFFUSE_AND_SPECULAR,
                _ => false,
            })
    }
}

/// The lighting registers `Lights::set_material` reads: the material's part
/// of a light set. Everything else in `LightingRegisters` describes the
/// lights themselves.
const MATERIAL_REGISTERS: [u8; 11] = [
    op::MATERIAL_UPDATE,
    op::MATERIAL_EMISSIVE,
    op::MATERIAL_AMBIENT,
    op::MATERIAL_DIFFUSE,
    op::MATERIAL_SPECULAR,
    op::MATERIAL_ALPHA,
    op::MATERIAL_SPECULAR_COEFFICIENT,
    op::AMBIENT_COLOR,
    op::AMBIENT_ALPHA,
    op::LIGHT_MODE,
    op::REVERSE_NORMAL,
];

/// Per `LightingRegisters` word: all ones for the lights' registers, zero
/// for the material's (`MATERIAL_REGISTERS`).
const LIGHT_PART_MASK: [u32; LIGHTING_REGISTER_COUNT] = {
    let mut mask = [u32::MAX; LIGHTING_REGISTER_COUNT];
    let mut index = 0;
    while index < MATERIAL_REGISTERS.len() {
        mask[lighting_slot(MATERIAL_REGISTERS[index])] = 0;
        index += 1;
    }
    mask
};

/// The last light set worked out (`gpu_lights_cached`): its lights and
/// their part of the GPU light set, and what they came from.
#[derive(Clone, Copy)]
struct CachedLights {
    /// The lighting registers without the material's (`LIGHT_PART_MASK`).
    light_registers: [u32; LIGHTING_REGISTER_COUNT],
    /// The view matrix's z axis (its bits): the specular half vectors.
    view_axis: [u32; 3],
    lights: Lights,
    light_part: GpuLights,
}

/// Remembers the lights of the last GPU light set, so a draw whose lights
/// stay and only the material changes (MTX's riders: another material
/// colour before nearly every part) doesn't work them out again: the
/// normalized directions and half vectors are taken over, only the
/// material's part is filled in anew. The result is the same, bit for bit,
/// as `Lights::from_registers(..).gpu(..)`.
pub struct GpuLightsCache {
    last: Option<CachedLights>,
}

impl GpuLightsCache {
    pub const fn new() -> GpuLightsCache {
        GpuLightsCache { last: None }
    }

    /// `Lights::from_registers(registers, world, view).gpu(has_vertex_color,
    /// opaque)`, the lights' part taken from the last call if the lights and
    /// the view direction are the same.
    pub fn gpu_lights(
        &mut self,
        registers: &LightingRegisters,
        world: &[f32; 12],
        view: &[f32; 12],
        has_vertex_color: bool,
        opaque: bool,
    ) -> GpuLights {
        let view_axis = [view[2].to_bits(), view[5].to_bits(), view[8].to_bits()];
        // Compared word by word under the mask (no masked copy on the stack,
        // no memcmp call; integers only, the A9's float compares stall).
        let same_lights = |last: &CachedLights| {
            last.view_axis == view_axis
                && registers.words.iter().zip(&last.light_registers).zip(&LIGHT_PART_MASK).all(|((&now, &then), &mask)| now & mask == then)
        };
        match self.last.as_mut() {
            Some(last) if same_lights(last) => {
                last.lights.set_material(registers);
                last.lights.with_material(last.light_part, has_vertex_color, opaque)
            }
            _ => {
                let light_registers: [u32; LIGHTING_REGISTER_COUNT] =
                    core::array::from_fn(|index| registers.words[index] & LIGHT_PART_MASK[index]);
                let lights = Lights::from_registers(registers, world, view);
                let light_part = lights.gpu_light_part();
                self.last = Some(CachedLights { light_registers, view_axis, lights, light_part });
                lights.with_material(light_part, has_vertex_color, opaque)
            }
        }
    }
}

impl Default for GpuLightsCache {
    fn default() -> Self {
        GpuLightsCache::new()
    }
}

impl Lights {
    /// For a 3D draw with the current GE state: `Some` if lighting is on
    /// and at least one light is.
    pub fn of(ge: &GeState) -> Option<Lights> {
        LightingRegisters::of(ge).map(|registers| Lights::from_registers(&registers, &ge.matrices.world, &ge.matrices.view))
    }

    /// The lights `registers` give with these world and view matrices (4x3,
    /// column by column).
    pub fn from_registers(registers: &LightingRegisters, world: &[f32; 12], view: &[f32; 12]) -> Lights {
        let register = |opcode: u8| registers.register(opcode);
        let mut lights = Lights { world: *world, ..Lights::default() };
        lights.set_material(registers);
        // The view matrix's third row (its z axis in world space).
        lights.view_direction = normalize_or_001([view[2], view[5], view[8]]);
        for number in 0..4u8 {
            if register(op::LIGHT_ENABLE_0 + number) & 1 == 0 {
                continue;
            }
            let kind_register = register(op::LIGHT_TYPE_0 + number);
            let vector = |first: u8| {
                let at = first + 3 * number;
                [float24(register(at)), float24(register(at + 1)), float24(register(at + 2))]
            };
            let colors = |which: u8| rgb(register(op::LIGHT_AMBIENT_0 + 3 * number + which));
            let kind = (kind_register >> 8) & 3;
            let position = vector(op::LIGHT_X_0);
            let light = Light {
                kind,
                computation: kind_register & 3,
                position: if kind == DIRECTIONAL { normalize_or_001(position) } else { position },
                direction: normalize_or_001(vector(op::LIGHT_DIRECTION_X_0)),
                attenuation: vector(op::LIGHT_ATTENUATION_A_0),
                spot_cutoff: float24(register(op::LIGHT_SPOT_CUTOFF_0 + number)),
                spot_exponent: float24(register(op::LIGHT_SPOT_EXPONENT_0 + number)),
                ambient: colors(0),
                diffuse: colors(1),
                specular: colors(2),
            };
            lights.needs_position |= kind != DIRECTIONAL;
            lights.lights[lights.count] = light;
            lights.count += 1;
        }
        lights
    }

    /// The material's part of `registers` (`MATERIAL_REGISTERS`): material
    /// colours, the global ambient light, LIGHTMODE, REVERSENORMAL. The rest
    /// (the lights, the view direction) stays as it is.
    fn set_material(&mut self, registers: &LightingRegisters) {
        let register = |opcode: u8| registers.register(opcode);
        let material_color = (register(op::MATERIAL_ALPHA) & 0xFF) << 24 | (register(op::MATERIAL_AMBIENT) & 0x00FF_FFFF);
        self.ambient = channels((register(op::AMBIENT_COLOR) & 0x00FF_FFFF) | (register(op::AMBIENT_ALPHA) & 0xFF) << 24);
        self.emissive = rgb(register(op::MATERIAL_EMISSIVE));
        self.material_ambient = channels(material_color);
        self.material_diffuse = rgb(register(op::MATERIAL_DIFFUSE));
        self.material_specular = rgb(register(op::MATERIAL_SPECULAR));
        self.specular_coefficient = float24(register(op::MATERIAL_SPECULAR_COEFFICIENT));
        self.material_update = register(op::MATERIAL_UPDATE) & 7;
        self.separate_specular = register(op::LIGHT_MODE) & 1 != 0;
        self.reverse_normal = register(op::REVERSE_NORMAL) & 1 != 0;
    }

    /// These lights for vertices whose normals reach world space through
    /// `model` (a 3x3 matrix, column by column: the world matrix's, times
    /// the bone's for skinned vertices), if the fast way works for them
    /// (`ModelLights`): only directional lights, and `model` a rotation
    /// with one scale for all axes (mirrored too). MTX Mototrax: its riders
    /// and bikes.
    pub fn in_model_space(&self, model: &[f32; 9]) -> Option<ModelLights> {
        let lights = &self.lights[..self.count];
        if lights.iter().any(|light| light.kind != DIRECTIONAL) {
            return None;
        }
        // The columns' lengths and angles: model^T x model = scale^2 x identity.
        let column = |c: usize| [model[3 * c], model[3 * c + 1], model[3 * c + 2]];
        let columns = [column(0), column(1), column(2)];
        let scale2 = (dot(columns[0], columns[0]) + dot(columns[1], columns[1]) + dot(columns[2], columns[2])) / 3.0;
        if !(scale2 > 1e-12) || !scale2.is_finite() {
            return None;
        }
        let tolerance = scale2 * SIMILARITY_TOLERANCE;
        for i in 0..3 {
            for j in i..3 {
                let expected = if i == j { scale2 } else { 0.0 };
                if (dot(columns[i], columns[j]) - expected).abs() > tolerance {
                    return None;
                }
            }
        }
        // A world vector v becomes model^T v / scale (REVERSENORMAL: negated):
        // its dot product with a model normal n is v . (model n) / scale,
        // and |model n| = scale |n|.
        let factor = inverse_sqrt(scale2) * if self.reverse_normal { -1.0 } else { 1.0 };
        let to_model = |v: [f32; 3]| columns.map(|column| dot(column, v) * factor);
        let mut model_lights = ModelLights {
            to_light: [[0.0; 3]; 4],
            half: [[0.0; 3]; 4],
            diffuse: [[0.0; 3]; 4],
            specular: [[0.0; 3]; 4],
            computation: [0; 4],
            count: self.count,
            emissive: self.emissive.map(|value| value * 255.0),
            ambient: [self.ambient[0], self.ambient[1], self.ambient[2], self.ambient[3]],
            material_ambient: self.material_ambient.map(|value| value * 255.0),
            material_diffuse: self.material_diffuse.map(|value| value * 255.0),
            material_specular: self.material_specular.map(|value| value * 255.0),
            material_update: self.material_update,
            specular_coefficient: self.specular_coefficient,
        };
        for (index, light) in lights.iter().enumerate() {
            model_lights.to_light[index] = to_model(light.position);
            let half: [f32; 3] = core::array::from_fn(|axis| light.position[axis] + self.view_direction[axis]);
            let half_length2 = dot(half, half);
            let half = if half_length2 > 0.0 { half.map(|value| value * inverse_sqrt(half_length2)) } else { [0.0; 3] };
            model_lights.half[index] = to_model(half);
            model_lights.diffuse[index] = light.diffuse;
            model_lights.specular[index] = light.specular;
            model_lights.computation[index] = light.computation;
            // Directional lights' ambient parts don't depend on the vertex:
            // summed into the global ambient light.
            for channel in 0..3 {
                model_lights.ambient[channel] += light.ambient[channel];
            }
        }
        Some(model_lights)
    }

    /// These lights for the GPU (`GpuLights`, nativeres_3d_v.cg): world
    /// space, directional lights only (`gpu_ready`). `has_vertex_color`:
    /// the draw's vertices have a colour (else the material's are used
    /// whatever MATERIAL_UPDATE says); `opaque`: the result's alpha is 1.
    pub fn gpu(&self, has_vertex_color: bool, opaque: bool) -> GpuLights {
        self.with_material(self.gpu_light_part(), has_vertex_color, opaque)
    }

    /// The lights' part of `gpu`: per light towards it, the half vector,
    /// its diffuse and specular colours (from the light registers and the
    /// view direction only; the rest zeros, `with_material` fills it in).
    fn gpu_light_part(&self) -> GpuLights {
        let flag = |on: bool| if on { 1.0 } else { 0.0 };
        let mut gpu = GpuLights::default();
        for (index, light) in self.lights[..self.count].iter().enumerate() {
            let [x, y, z] = light.position;
            if light.kind == POINT {
                // Position and attenuation (`GpuLights::to_light`, `half`).
                let [a, b, c] = light.attenuation;
                gpu.to_light[index] = [x, y, z, 1.0];
                gpu.half[index] = [a, b, c, 0.0];
                gpu.diffuse[index] = [light.diffuse[0], light.diffuse[1], light.diffuse[2], flag(light.computation == POWERED_DIFFUSE)];
                gpu.light_ambient[index] = [light.ambient[0], light.ambient[1], light.ambient[2], 0.0];
                continue;
            }
            gpu.to_light[index] = [x, y, z, 0.0];
            let half: [f32; 3] = core::array::from_fn(|axis| light.position[axis] + self.view_direction[axis]);
            let half_length2 = dot(half, half);
            if half_length2 > 0.0 {
                let [hx, hy, hz] = half.map(|value| value * inverse_sqrt(half_length2));
                gpu.half[index] = [hx, hy, hz, 0.0];
            }
            gpu.diffuse[index] = [light.diffuse[0], light.diffuse[1], light.diffuse[2], flag(light.computation == POWERED_DIFFUSE)];
            let specular = light.computation == DIFFUSE_AND_SPECULAR;
            gpu.specular[index] = [light.specular[0], light.specular[1], light.specular[2], flag(specular)];
        }
        gpu
    }

    /// `lights` (`gpu_light_part` of these lights) with the material's part
    /// of `gpu` filled in: the global ambient light plus each light's
    /// ambient colour (added in light order), the material colours, flags.
    fn with_material(&self, mut gpu: GpuLights, has_vertex_color: bool, opaque: bool) -> GpuLights {
        let flag = |on: bool| if on { 1.0 } else { 0.0 };
        gpu.ambient = self.ambient;
        // Point lights' ambient colours are attenuated (`light_ambient`).
        for light in self.lights[..self.count].iter().filter(|light| light.kind == DIRECTIONAL) {
            for channel in 0..3 {
                gpu.ambient[channel] += light.ambient[channel];
            }
        }
        gpu.emissive = [self.emissive[0], self.emissive[1], self.emissive[2], self.specular_coefficient];
        gpu.material_ambient = self.material_ambient;
        gpu.material_diffuse = [self.material_diffuse[0], self.material_diffuse[1], self.material_diffuse[2], flag(opaque)];
        gpu.material_specular =
            [self.material_specular[0], self.material_specular[1], self.material_specular[2], flag(self.separate_specular)];
        let update = if has_vertex_color { self.material_update } else { 0 };
        // w: REVERSE_NORMAL, for raw draws (the 3D program gets negated
        // normal rows instead, `normal_rows`).
        gpu.from_vertex = [flag(update & 1 != 0), flag(update & 2 != 0), flag(update & 4 != 0), flag(self.reverse_normal)];
        gpu
    }

    /// Whether the GPU can light with these lights (`gpu`): directional
    /// lights only, so far (MTX's riders and bikes).
    pub fn gpu_ready(&self) -> bool {
        self.lights[..self.count].iter().all(|light| light.kind == DIRECTIONAL)
    }

    /// The matrix taking a vertex's normal to world space for the GPU
    /// (`GpuTransform::normal_rows`): the world matrix's 3x3, times `bone`'s
    /// if given, negated for REVERSENORMAL, row by row.
    pub fn normal_rows(&self, bone: Option<&[f32; 12]>) -> [[f32; 4]; 3] {
        let w = &self.world;
        let sign = if self.reverse_normal { -1.0 } else { 1.0 };
        // Element (row, column) of world x bone; both column by column.
        let element = |row: usize, column: usize| match bone {
            None => w[3 * column + row],
            Some(bone) => (0..3).map(|k| w[3 * k + row] * bone[3 * column + k]).sum(),
        };
        core::array::from_fn(|row| [sign * element(row, 0), sign * element(row, 1), sign * element(row, 2), 0.0])
    }

    /// The world matrix the lights were worked out with (4x3, column by column).
    pub fn world(&self) -> &[f32; 12] {
        &self.world
    }

    /// Whether textured draws lit with these lights on the CPU are drawn
    /// approximately (the separate specular colour added before texturing:
    /// a CPU-lit vertex has one colour). The GPU's lighting adds it after
    /// texturing (`GpuLights::material_specular`, nativeres_raw_v.cg).
    pub fn approximated(&self) -> bool {
        self.separate_specular && self.lights[..self.count].iter().any(|light| light.computation == DIFFUSE_AND_SPECULAR)
    }

    /// The lit colour (RGBA) of a vertex with colour `vertex` (`None`:
    /// none), model-space `normal` (skinned) and, for point and spot
    /// lights, model-space `position` (skinned, scaled).
    pub fn color(&self, vertex: Option<u32>, normal: [f32; 3], position: [f32; 3]) -> u32 {
        let vertex = vertex.map(channels);
        let from_vertex = |bit: u32| if self.material_update & bit != 0 { vertex } else { None };
        let ambient_color = from_vertex(1).unwrap_or(self.material_ambient);
        let diffuse_color = from_vertex(2).map_or(self.material_diffuse, |[r, g, b, _]| [r, g, b]);
        let specular_color = from_vertex(4).map_or(self.material_specular, |[r, g, b, _]| [r, g, b]);

        let w = &self.world;
        let sign = if self.reverse_normal { -1.0 } else { 1.0 };
        let [nx, ny, nz] = normal.map(|value| value * sign);
        let world_normal = normalize_or_001([
            w[0] * nx + w[3] * ny + w[6] * nz,
            w[1] * nx + w[4] * ny + w[7] * nz,
            w[2] * nx + w[5] * ny + w[8] * nz,
        ]);
        let world_position = if self.needs_position {
            let [x, y, z] = position;
            [
                w[0] * x + w[3] * y + w[6] * z + w[9],
                w[1] * x + w[4] * y + w[7] * z + w[10],
                w[2] * x + w[5] * y + w[8] * z + w[11],
            ]
        } else {
            [0.0; 3]
        };

        let mut sum0 = [0.0f32; 4];
        for channel in 0..4 {
            sum0[channel] = self.ambient[channel] * ambient_color[channel] + if channel < 3 { self.emissive[channel] } else { 0.0 };
        }
        let mut sum1 = [0.0f32; 3];
        for light in &self.lights[..self.count] {
            let (to_light, scale) = if light.kind == DIRECTIONAL {
                (light.position, 1.0)
            } else {
                let towards: [f32; 3] = core::array::from_fn(|axis| light.position[axis] - world_position[axis]);
                let distance2 = dot(towards, towards);
                let inverse = if distance2 > 0.0 { inverse_sqrt(distance2) } else { 0.0 };
                let distance = distance2 * inverse;
                let to_light = towards.map(|value| value * inverse);
                let [a, b, c] = light.attenuation;
                let attenuation = (1.0 / (a + b * distance + c * distance2)).clamp(0.0, 1.0);
                let attenuation = if attenuation.is_nan() { 0.0 } else { attenuation };
                let scale = if light.kind == POINT {
                    attenuation
                } else {
                    let angle = dot(light.direction, to_light);
                    if angle >= light.spot_cutoff { attenuation * psp_pow(angle, light.spot_exponent) } else { 0.0 }
                };
                (to_light, scale)
            };
            let mut light_dot = dot(to_light, world_normal);
            if light.computation == POWERED_DIFFUSE {
                light_dot = psp_pow(light_dot, self.specular_coefficient);
            }
            let diffuse_factor = light_dot.max(0.0);
            if light.computation == DIFFUSE_AND_SPECULAR && light_dot >= 0.0 {
                let half: [f32; 3] = core::array::from_fn(|axis| to_light[axis] + self.view_direction[axis]);
                let half_length2 = dot(half, half);
                let half_dot = if half_length2 > 0.0 { dot(half, world_normal) * inverse_sqrt(half_length2) } else { 0.0 };
                let specular = psp_pow(half_dot, self.specular_coefficient);
                if specular > 0.0 {
                    for channel in 0..3 {
                        sum1[channel] += light.specular[channel] * specular_color[channel] * specular * scale;
                    }
                }
            }
            for channel in 0..3 {
                let diffuse = light.diffuse[channel] * diffuse_color[channel] * diffuse_factor;
                sum0[channel] += (light.ambient[channel] * ambient_color[channel] + diffuse) * scale;
            }
        }
        // One colour: the specular part added here (see the module comment).
        let value: [f32; 4] = core::array::from_fn(|channel| 255.0 * if channel < 3 { sum0[channel] + sum1[channel] } else { sum0[channel] });
        pack(value)
    }
}

/// A normal packed for the GPU into one word (`FrameVertex::w`'s bits):
/// x, y, z as signed bytes (S8N, -127..127) scaled so the largest
/// component is 127 (only the direction matters), byte 3 zero. A zero
/// normal packs as zero.
pub fn pack_normal(normal: [f32; 3]) -> u32 {
    let largest = normal.iter().fold(0.0f32, |largest, value| largest.max(value.abs()));
    if !(largest > 0.0) || !largest.is_finite() {
        return 0;
    }
    // 8-bit normals (fractions n / 128, MTX's) keep their bytes exactly.
    let as_bytes = normal.map(|value| value * 128.0);
    let exact = as_bytes.iter().all(|&byte| byte.abs() <= 128.0 && byte == (byte as i32) as f32);
    let scale = if exact { 128.0 } else { 127.0 / largest };
    let mut packed = 0u32;
    for (index, value) in normal.into_iter().enumerate() {
        let scaled = value * scale;
        // Rounded to the nearest (core has no `round`); within -127..127.
        let rounded = (scaled + if scaled < 0.0 { -0.5 } else { 0.5 }) as i32;
        let byte = rounded.clamp(-127, 127) as i8 as u8;
        packed |= (byte as u32) << (8 * index);
    }
    packed
}

/// `pack_normal` of an 8-bit normal (VERTEX_TYPE normal format 1: bytes n
/// meaning n / 128), from its bytes without floats: they are exact, so
/// `pack_normal` keeps each byte, -128 becoming -127. Zero for a zero normal.
#[inline(always)]
pub fn pack_s8_normal(bytes: [u8; 3]) -> u32 {
    let mut packed = 0u32;
    for (index, byte) in bytes.into_iter().enumerate() {
        let clamped = (byte as i8).max(-127) as u8;
        packed |= (clamped as u32) << (8 * index);
    }
    packed
}

/// `pack_normal` for normals that aren't fractions n / 128 (skinned ones):
/// the same result whenever `pack_normal` doesn't take its exact branch,
/// with fewer float compares (each costs a transfer of the VFP flags on
/// the Vita's Cortex-A9) and the rounding without a branch.
#[inline(always)]
pub fn pack_scaled_normal(normal: [f32; 3]) -> u32 {
    let [x, y, z] = normal.map(f32::abs);
    let largest = if x > y { x } else { y };
    let largest = if largest > z { largest } else { z };
    if !(largest > 0.0 && largest <= f32::MAX) {
        return 0;
    }
    let scale = 127.0 / largest;
    let mut packed = 0u32;
    for (index, value) in normal.into_iter().enumerate() {
        let scaled = value * scale;
        // Rounded to the nearest, halves away from zero (as `pack_normal`).
        let rounded = (scaled + 0.5f32.copysign(scaled)) as i32;
        let byte = rounded.clamp(-127, 127) as i8 as u8;
        packed |= (byte as u32) << (8 * index);
    }
    packed
}

/// `pack_normal` undone (as the GPU's S8N attribute reads it: byte / 127).
pub fn unpack_normal(packed: u32) -> [f32; 3] {
    core::array::from_fn(|index| ((packed >> (8 * index)) as u8 as i8) as f32 * (1.0 / 127.0))
}

impl GpuLights {
    /// The colour nativeres_3d_v.cg gives a GPU-lit vertex: `vertex` its
    /// colour as stored, `normal` as the shader reads it, `normal_rows` the
    /// transform's (`GpuTransform::normal_rows`). PPSSPP's hardware vertex
    /// shader (VertexShaderGenerator.cpp) for directional lights; the pow
    /// here is the GE's (`psp_pow`), the shader's a true pow (as PPSSPP's
    /// without bit operations: "close enough").
    ///
    /// Point lights (`to_light` w 1, the point-light raw programs) need the
    /// vertex's `world_position`: as PPSSPP's shader, towards the light
    /// times inversesqrt(distance^2), its diffuse and ambient parts times
    /// the attenuation clamp(1 / (a + b d + c d^2), 0, 1).
    ///
    /// `GE_LIGHTING` (God of War's program set): in the GE's integer
    /// arithmetic (`ge_color`), else in float (`float_color`).
    pub fn color(&self, vertex: u32, normal: [f32; 3], normal_rows: &[[f32; 4]; 3], world_position: [f32; 3]) -> u32 {
        if GE_LIGHTING.load(Ordering::Relaxed) {
            self.ge_color(vertex, normal, normal_rows, world_position)
        } else {
            self.float_color(vertex, normal, normal_rows, world_position)
        }
    }

    /// `color` in float, rounded once at the end (the normal programs).
    pub fn float_color(&self, vertex: u32, normal: [f32; 3], normal_rows: &[[f32; 4]; 3], world_position: [f32; 3]) -> u32 {
        let row = |r: usize| [normal_rows[r][0], normal_rows[r][1], normal_rows[r][2]];
        let world = normalize_or_001([dot(row(0), normal), dot(row(1), normal), dot(row(2), normal)]);
        let vertex = channels(vertex);
        let pick = |flag: f32, material: [f32; 4]| if flag > 0.5 { vertex } else { material };
        let ambient_color = pick(self.from_vertex[0], self.material_ambient);
        let diffuse_color = pick(self.from_vertex[1], self.material_diffuse);
        let specular_color = pick(self.from_vertex[2], self.material_specular);
        let coefficient = self.emissive[3];
        let mut diffuse = [0.0f32; 3];
        let mut specular = [0.0f32; 3];
        let mut point_ambient = [0.0f32; 3];
        for light in 0..4 {
            let xyz = |v: [f32; 4]| [v[0], v[1], v[2]];
            if self.to_light[light][3] > 0.5 {
                let towards: [f32; 3] = core::array::from_fn(|axis| self.to_light[light][axis] - world_position[axis]);
                let distance2 = dot(towards, towards);
                let inverse = inverse_sqrt(distance2);
                let distance = distance2 * inverse;
                let [a, b, c, _] = self.half[light];
                let scale = (1.0 / (a + b * distance + c * distance2)).clamp(0.0, 1.0);
                let scale = if scale.is_nan() { 0.0 } else { scale };
                let mut light_dot = dot(towards.map(|value| value * inverse), world);
                if self.diffuse[light][3] > 0.5 {
                    light_dot = psp_pow(light_dot, coefficient);
                }
                for channel in 0..3 {
                    diffuse[channel] += self.diffuse[light][channel] * light_dot.max(0.0) * scale;
                    point_ambient[channel] += self.light_ambient[light][channel] * scale;
                }
                continue;
            }
            let mut light_dot = dot(xyz(self.to_light[light]), world);
            if self.diffuse[light][3] > 0.5 {
                light_dot = psp_pow(light_dot, coefficient);
            }
            for channel in 0..3 {
                diffuse[channel] += self.diffuse[light][channel] * light_dot.max(0.0);
            }
            if self.specular[light][3] > 0.5 && light_dot >= 0.0 {
                let factor = psp_pow(dot(xyz(self.half[light]), world), coefficient);
                if factor > 0.0 {
                    for channel in 0..3 {
                        specular[channel] += self.specular[light][channel] * factor;
                    }
                }
            }
        }
        let mut value: [f32; 4] = core::array::from_fn(|channel| {
            if channel == 3 {
                return 255.0 * self.ambient[3] * ambient_color[3];
            }
            255.0
                * (self.emissive[channel]
                    + (self.ambient[channel] + point_ambient[channel]) * ambient_color[channel]
                    + diffuse[channel] * diffuse_color[channel]
                    + specular[channel] * specular_color[channel])
        });
        if self.material_diffuse[3] > 0.5 {
            value[3] = 255.0;
        }
        pack(value)
    }
}

impl GpuLights {
    /// `color` as the GE computes it (`GE_LIGHTING`; ppsspp-ref
    /// GPU/Software/Lighting.cpp ProcessSIMD, GEMath.h GELightColorProduct /
    /// GELightColorScale, probed on hardware), as the GE_LIGHT vertex
    /// programs (nativeres_3d_v.cg, nativeres_raw_v.cg): colours as 8-bit
    /// integers, each light colour x material colour ((2l + 1)(2m + 1)) >> 10,
    /// each factor (N.L, its power, the attenuation) s = floor(256 f) applied
    /// as ((2x + 1)(2s + 1)) >> 10, the terms summed and clamped; the alpha
    /// the ambient alpha x material alpha the same way. The float lighting
    /// was ~1 level brighter per light (God of War's title vs Sony's screen:
    /// 2.72 -> 2.52, DESIGN sixtieth session). Not exact: the summed ambient
    /// (global plus the directional lights') is one term (the GE truncates
    /// each light's own; God of War's lights have none), and N.L is ours in
    /// float, not the GE's float24.
    pub fn ge_color(&self, vertex: u32, normal: [f32; 3], normal_rows: &[[f32; 4]; 3], world_position: [f32; 3]) -> u32 {
        let row = |r: usize| [normal_rows[r][0], normal_rows[r][1], normal_rows[r][2]];
        let world = normalize_or_001([dot(row(0), normal), dot(row(1), normal), dot(row(2), normal)]);
        let to8 = |v: f32| ((v * 255.0 + 0.5) as i32).clamp(0, 255);
        let vertex8 = bytes(vertex).map(|v| v as i32);
        let material = |flag: f32, m: [f32; 4]| -> [i32; 4] { if flag > 0.5 { vertex8 } else { m.map(to8) } };
        let mac = material(self.from_vertex[0], self.material_ambient);
        let mdc = material(self.from_vertex[1], self.material_diffuse);
        let msc = material(self.from_vertex[2], self.material_specular);
        let product = |l: i32, m: i32| ((2 * l + 1) * (2 * m + 1)) >> 10;
        let scale = |x: i32, f: f32| {
            let s = ((256.0 * f) as i32).clamp(0, 256);
            ((2 * x + 1) * (2 * s + 1)) >> 10
        };
        let coefficient = self.emissive[3];
        let mut sum = [0i32; 3];
        let mut specular = [0i32; 3];
        for channel in 0..3 {
            sum[channel] = to8(self.emissive[channel]) + product(to8(self.ambient[channel]), mac[channel]);
        }
        for light in 0..4 {
            let xyz = |v: [f32; 4]| [v[0], v[1], v[2]];
            let point = self.to_light[light][3] > 0.5;
            let (l, att) = if point {
                let towards: [f32; 3] = core::array::from_fn(|axis| self.to_light[light][axis] - world_position[axis]);
                let distance2 = dot(towards, towards);
                let inverse = inverse_sqrt(distance2);
                let distance = distance2 * inverse;
                let [a, b, c, _] = self.half[light];
                let att = (1.0 / (a + b * distance + c * distance2)).clamp(0.0, 1.0);
                (towards.map(|value| value * inverse), if att.is_nan() { 0.0 } else { att })
            } else {
                (xyz(self.to_light[light]), 1.0)
            };
            let mut light_dot = dot(l, world);
            if self.diffuse[light][3] > 0.5 {
                light_dot = psp_pow(light_dot, coefficient);
            }
            for channel in 0..3 {
                if point {
                    let ambient = product(to8(self.light_ambient[light][channel]), mac[channel]);
                    sum[channel] += if att < 1.0 { scale(ambient, att) } else { ambient };
                }
                let mut diffuse = scale(product(to8(self.diffuse[light][channel]), mdc[channel]), light_dot);
                if att < 1.0 {
                    diffuse = scale(diffuse, att);
                }
                sum[channel] += diffuse;
            }
            if !point && self.specular[light][3] > 0.5 && light_dot >= 0.0 {
                let factor = psp_pow(dot(xyz(self.half[light]), world), coefficient);
                if factor > 0.0 {
                    for channel in 0..3 {
                        specular[channel] += scale(product(to8(self.specular[light][channel]), msc[channel]), factor);
                    }
                }
            }
        }
        // (The separate specular colour, LIGHTMODE 1, is added here as the
        // float `color` does: the programs send it apart.)
        let mut value: [f32; 4] = core::array::from_fn(|channel| {
            if channel == 3 {
                return product(to8(self.ambient[3]), mac[3]) as f32;
            }
            (sum[channel] + specular[channel]) as f32
        });
        if self.material_diffuse[3] > 0.5 {
            value[3] = 255.0;
        }
        pack(value)
    }
}

/// How far `Lights::in_model_space` lets a matrix be from a rotation with
/// one scale (relative to the scale squared): the normal's length is then
/// off by at most ~0.05%, a tenth of a colour step.
const SIMILARITY_TOLERANCE: f32 = 1e-3;

/// `Lights` turned into the model space of one draw (`Lights::in_model_space`):
/// a vertex's colour from its model-space normal with one inverse square
/// root and a few dot products, where `Lights::color` turns every normal
/// into world space first. Colours in 0..255 units. Gives what
/// `Lights::color` gives, up to float rounding (a test checks it).
#[derive(Clone, Copy, Debug)]
pub struct ModelLights {
    /// Per light: towards it, and the specular half vector, in model space
    /// divided by the model matrix's scale.
    to_light: [[f32; 3]; 4],
    half: [[f32; 3]; 4],
    /// Per light: its diffuse and specular colours (0..1).
    diffuse: [[f32; 3]; 4],
    specular: [[f32; 3]; 4],
    computation: [u32; 4],
    count: usize,
    /// MATERIAL_EMISSIVE, 0..255.
    emissive: [f32; 3],
    /// The global ambient light plus every light's ambient colour (RGB,
    /// 0..1), and AMBIENT_ALPHA (0..1).
    ambient: [f32; 4],
    /// The material colours, 0..255.
    material_ambient: [f32; 4],
    material_diffuse: [f32; 3],
    material_specular: [f32; 3],
    material_update: u32,
    specular_coefficient: f32,
}

impl ModelLights {
    /// The lit colour (RGBA) of a vertex with colour `vertex` (`None`:
    /// none) and model-space normal `normal` (`None` for a zero normal:
    /// `Lights::color` must do it).
    #[inline]
    pub fn color(&self, vertex: Option<u32>, normal: [f32; 3]) -> Option<u32> {
        let length2 = dot(normal, normal);
        if !(length2 > 0.0) {
            return None;
        }
        let inverse_length = inverse_sqrt(length2);
        let vertex = vertex.map(bytes);
        let from_vertex = |bit: u32| if self.material_update & bit != 0 { vertex } else { None };
        let ambient_color = from_vertex(1).unwrap_or(self.material_ambient);
        let diffuse_color = from_vertex(2).map_or(self.material_diffuse, |[r, g, b, _]| [r, g, b]);
        let specular_color = from_vertex(4).map_or(self.material_specular, |[r, g, b, _]| [r, g, b]);

        let mut diffuse = [0.0f32; 3];
        let mut specular = [0.0f32; 3];
        for index in 0..self.count {
            let computation = self.computation[index];
            let mut light_dot = dot(normal, self.to_light[index]) * inverse_length;
            if computation == POWERED_DIFFUSE {
                light_dot = psp_pow(light_dot, self.specular_coefficient);
            }
            let diffuse_factor = light_dot.max(0.0);
            for channel in 0..3 {
                diffuse[channel] += self.diffuse[index][channel] * diffuse_factor;
            }
            if computation == DIFFUSE_AND_SPECULAR && light_dot >= 0.0 {
                let factor = psp_pow(dot(normal, self.half[index]) * inverse_length, self.specular_coefficient);
                if factor > 0.0 {
                    for channel in 0..3 {
                        specular[channel] += self.specular[index][channel] * factor;
                    }
                }
            }
        }
        let value: [f32; 4] = core::array::from_fn(|channel| {
            if channel == 3 {
                return self.ambient[3] * ambient_color[3];
            }
            self.emissive[channel]
                + self.ambient[channel] * ambient_color[channel]
                + diffuse[channel] * diffuse_color[channel]
                + specular[channel] * specular_color[channel]
        });
        Some(pack(value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::Command;

    fn set(ge: &mut GeState, opcode: u8, data: u32) {
        ge.set_register(Command(((opcode as u32) << 24) | (data & 0xFF_FFFF)));
    }

    fn float(value: f32) -> u32 {
        value.to_bits() >> 8
    }

    /// `pack` as it was before the integer clamp: the reference.
    fn pack_float_clamp(value: [f32; 4]) -> u32 {
        let mut color = 0u32;
        for (channel, value) in value.into_iter().enumerate() {
            color |= ((value.clamp(0.0, 255.0) + 0.5) as u32) << (8 * channel);
        }
        color
    }

    #[test]
    fn pack_gives_the_float_clamps_bits_at_every_edge() {
        let below = |x: f32| f32::from_bits(x.to_bits() - 1); // next float towards 0 (x > 0)
        let edges = [
            f32::NAN, -f32::NAN, f32::from_bits(0x7F80_0001), f32::from_bits(0xFFC0_1234),
            f32::INFINITY, f32::NEG_INFINITY, f32::MAX, f32::MIN, 1e10, -1e10,
            0.0, -0.0, f32::MIN_POSITIVE, f32::from_bits(1), -f32::from_bits(1),
            0.5, below(0.5), -0.5, -below(0.5), 1.0, 1.5, below(1.5), 127.5, 128.0,
            254.5, below(254.5), 255.0, below(255.0), 255.5, below(255.5), 256.0, 256.5, 1000.0,
        ];
        for &x in &edges {
            for channel in 0..4 {
                let mut value = [37.25, 0.0, 255.0, 128.5];
                value[channel] = x;
                assert_eq!(pack(value), pack_float_clamp(value), "{x:?} ({:08X}) in channel {channel}", x.to_bits());
            }
        }
        assert_eq!(pack([0.0, 255.0, 254.5, f32::NAN]), 0x00FF_FF00);
        // 0.49999997 + 0.5 rounds to 1.0 in f32: 1, as before (the old form has the same add).
        assert_eq!(pack([-1.0, 300.0, below(0.5), 127.5]), 0x8001_FF00);
    }

    /// Every one of the 2^32 floats in every channel at once (~10 s in a
    /// release build): `cargo test --release -p nativeres-core pack_ -- --ignored`.
    #[test]
    #[ignore]
    fn pack_gives_the_float_clamps_bits_for_every_float() {
        for bits in 0..=u32::MAX {
            let value = [f32::from_bits(bits); 4];
            assert_eq!(pack(value), pack_float_clamp(value), "{bits:08X}");
        }
    }

    /// One white directional light along +z, identity world and view,
    /// ambient and diffuse from the vertex colour (MTX's setup).
    fn ge_with_one_light(computation: u32) -> GeState {
        let mut ge = GeState::new();
        set(&mut ge, op::LIGHTING_ENABLE, 1);
        set(&mut ge, op::LIGHT_ENABLE_0, 1);
        set(&mut ge, op::LIGHT_TYPE_0, computation);
        set(&mut ge, op::LIGHT_X_0 + 2, float(1.0)); // direction (0, 0, 1)
        set(&mut ge, op::LIGHT_AMBIENT_0 + 1, 0xFF_FFFF); // diffuse white
        set(&mut ge, op::LIGHT_AMBIENT_0 + 2, 0xFF_FFFF); // specular white
        set(&mut ge, op::MATERIAL_UPDATE, 3);
        set(&mut ge, op::MATERIAL_SPECULAR, 0xFF_FFFF);
        set(&mut ge, op::AMBIENT_ALPHA, 0xFF);
        set(&mut ge, op::MATERIAL_SPECULAR_COEFFICIENT, float(1.0));
        for (number, data) in [(op::WORLD_MATRIX_NUMBER, op::WORLD_MATRIX_DATA), (op::VIEW_MATRIX_NUMBER, op::VIEW_MATRIX_DATA)] {
            set(&mut ge, number, 0);
            for value in [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0] {
                set(&mut ge, data, float(value));
            }
        }
        ge
    }

    #[test]
    fn a_surface_facing_a_directional_light_gets_its_diffuse_colour() {
        let lights = Lights::of(&ge_with_one_light(0)).unwrap();
        let grey = 0xFF80_8080;
        // Facing the light: diffuse = vertex colour; alpha = ambient alpha x vertex alpha.
        assert_eq!(lights.color(Some(grey), [0.0, 0.0, 1.0], [0.0; 3]), 0xFF80_8080);
        // Turned away: no diffuse, no ambient (ambient colour 0): black.
        assert_eq!(lights.color(Some(grey), [0.0, 0.0, -1.0], [0.0; 3]) & 0x00FF_FFFF, 0);
        // At 60 degrees: half.
        let half = lights.color(Some(0xFFFF_FFFF), [0.0, 0.866_025_4, 0.5], [0.0; 3]) & 0xFF;
        assert!((127..=128).contains(&half), "{half}");
    }

    #[test]
    fn reversed_normals_light_the_other_side() {
        let mut ge = ge_with_one_light(0);
        set(&mut ge, op::REVERSE_NORMAL, 1);
        let lights = Lights::of(&ge).unwrap();
        assert_eq!(lights.color(Some(0xFFFF_FFFF), [0.0, 0.0, -1.0], [0.0; 3]), 0xFFFF_FFFF);
        assert_eq!(lights.color(Some(0xFFFF_FFFF), [0.0, 0.0, 1.0], [0.0; 3]) & 0x00FF_FFFF, 0);
    }

    #[test]
    fn specular_adds_where_the_half_vector_meets_the_normal() {
        let lights = Lights::of(&ge_with_one_light(DIFFUSE_AND_SPECULAR)).unwrap();
        // Normal along the light and the view: diffuse 0.25 grey + full specular: clamped white.
        assert_eq!(lights.color(Some(0xFF40_4040), [0.0, 0.0, 1.0], [0.0; 3]), 0xFFFF_FFFF);
        assert!(lights.approximated() == lights.separate_specular);
    }

    /// A rotation about an axis (unit length) by `angle` radians, times
    /// `scale`, column by column.
    fn rotation(axis: [f32; 3], angle: f32, scale: f32) -> [f32; 9] {
        let (sin, cos) = (angle.sin(), angle.cos());
        let [x, y, z] = axis;
        let t = 1.0 - cos;
        let rows = [
            [t * x * x + cos, t * x * y - sin * z, t * x * z + sin * y],
            [t * x * y + sin * z, t * y * y + cos, t * y * z - sin * x],
            [t * x * z - sin * y, t * y * z + sin * x, t * z * z + cos],
        ];
        core::array::from_fn(|index| rows[index % 3][index / 3] * scale)
    }

    /// MTX's riders: three directional lights (diffuse, one with specular),
    /// a rotated and scaled world matrix: the model-space way gives the
    /// colours `Lights::color` gives, within one step.
    #[test]
    fn model_space_lighting_matches_world_space_lighting() {
        let mut ge = ge_with_one_light(DIFFUSE_AND_SPECULAR);
        set(&mut ge, op::LIGHT_MODE, 1);
        set(&mut ge, op::REVERSE_NORMAL, 1);
        set(&mut ge, op::AMBIENT_COLOR, 0x30_2010);
        set(&mut ge, op::MATERIAL_SPECULAR_COEFFICIENT, float(8.0));
        for (number, direction, computation) in [(1u8, [0.6f32, 0.8, 0.0], 0u32), (2, [-0.3, 0.2, -0.9], POWERED_DIFFUSE)] {
            set(&mut ge, op::LIGHT_ENABLE_0 + number, 1);
            set(&mut ge, op::LIGHT_TYPE_0 + number, computation);
            for axis in 0..3 {
                set(&mut ge, op::LIGHT_X_0 + 3 * number + axis as u8, float(direction[axis]));
            }
            set(&mut ge, op::LIGHT_AMBIENT_0 + 3 * number, 0x08_1018);
            set(&mut ge, op::LIGHT_AMBIENT_0 + 3 * number + 1, 0x60_A0C0);
        }
        let world = rotation([0.48, 0.6, 0.64], 1.1, 2.5);
        set(&mut ge, op::WORLD_MATRIX_NUMBER, 0);
        for value in world.iter().chain(&[3.0, -4.0, 5.0]) {
            set(&mut ge, op::WORLD_MATRIX_DATA, float(*value));
        }
        let view = rotation([0.0, 0.6, 0.8], -0.7, 1.0);
        set(&mut ge, op::VIEW_MATRIX_NUMBER, 0);
        for value in view.iter().chain(&[0.0, 0.0, 0.0]) {
            set(&mut ge, op::VIEW_MATRIX_DATA, float(*value));
        }
        let lights = Lights::of(&ge).unwrap();
        // The matrices went through 24-bit floats: still within the tolerance.
        let model_lights = lights.in_model_space(&ge.matrices.world[..9].try_into().unwrap()).expect("a rotation and a scale");
        let mut worst = 0;
        for step in 0..500u32 {
            let angle = step as f32 * 0.37;
            let normal = [angle.cos() * (step as f32 * 0.11).sin(), angle.sin(), (step as f32 * 0.23).cos() * 0.7];
            let color = 0x8000_0000 | step.wrapping_mul(0x9E37_79B9) >> 8;
            let expected = lights.color(Some(color), normal, [0.0; 3]);
            let fast = model_lights.color(Some(color), normal).unwrap();
            for shift in [0, 8, 16, 24] {
                let difference = ((expected >> shift) & 0xFF).abs_diff((fast >> shift) & 0xFF);
                worst = worst.max(difference);
            }
        }
        assert!(worst <= 1, "worst channel difference {worst}");
    }

    /// The GPU's lighting (`GpuLights::color`, packed normals, normal
    /// rows with a bone) gives what `Lights::color` gives, within three
    /// steps (the packed normal's rounding).
    #[test]
    fn ge_lighting_truncates_each_term() {
        // One directional light, white, N.L = 0.3; material diffuse and
        // ambient 200, global ambient 64. The GE: ambient ((129 x 401) >> 10)
        // = 50, diffuse ((511 x 401) >> 10) = 200 scaled by floor(0.3 x 256)
        // = 76 as ((401 x 153) >> 10) = 59: 109. In float: 110.2 -> 110.
        let grey = |value: f32| [value / 255.0, value / 255.0, value / 255.0, 1.0];
        let mut lights = GpuLights::default();
        lights.to_light[0] = [0.0, 0.0, 1.0, 0.0];
        lights.diffuse[0] = [1.0, 1.0, 1.0, 0.0];
        lights.ambient = grey(64.0);
        lights.material_ambient = grey(200.0);
        lights.material_diffuse = [200.0 / 255.0, 200.0 / 255.0, 200.0 / 255.0, 0.0];
        let rows = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]];
        let normal = [(1.0f32 - 0.09).sqrt(), 0.0, 0.3];
        assert_eq!(lights.ge_color(0, normal, &rows, [0.0; 3]), 0xFF6D_6D6D);
        assert_eq!(lights.float_color(0, normal, &rows, [0.0; 3]), 0xFF6E_6E6E);
        // Alpha: ((2 x 255 + 1)(2 x 255 + 1)) >> 10 = 255.
    }

    #[test]
    fn gpu_lighting_matches_cpu_lighting() {
        let mut ge = ge_with_one_light(DIFFUSE_AND_SPECULAR);
        set(&mut ge, op::LIGHT_MODE, 1);
        set(&mut ge, op::REVERSE_NORMAL, 1);
        set(&mut ge, op::AMBIENT_COLOR, 0x30_2010);
        set(&mut ge, op::MATERIAL_SPECULAR_COEFFICIENT, float(8.0));
        set(&mut ge, op::LIGHT_ENABLE_0 + 1, 1);
        set(&mut ge, op::LIGHT_TYPE_0 + 1, POWERED_DIFFUSE);
        for (axis, value) in [0.6f32, 0.8, 0.0].into_iter().enumerate() {
            set(&mut ge, op::LIGHT_X_0 + 3 + axis as u8, float(value));
        }
        set(&mut ge, op::LIGHT_AMBIENT_0 + 4, 0x60_A0C0);
        let world = rotation([0.48, 0.6, 0.64], 1.1, 2.5);
        set(&mut ge, op::WORLD_MATRIX_NUMBER, 0);
        for value in world.iter().chain(&[3.0, -4.0, 5.0]) {
            set(&mut ge, op::WORLD_MATRIX_DATA, float(*value));
        }
        let lights = Lights::of(&ge).unwrap();
        assert!(lights.gpu_ready());
        let gpu = lights.gpu(true, false);
        // A bone that squashes y: not a rotation, the GPU normalizes anyway.
        let mut bone = [0.0f32; 12];
        let turned = rotation([0.0, 0.0, 1.0], 0.4, 1.0);
        bone[..9].copy_from_slice(&turned);
        bone[3..6].iter_mut().for_each(|value| *value *= 0.5);
        let rows = lights.normal_rows(Some(&bone));
        let mut worst = 0;
        for step in 0..300u32 {
            let angle = step as f32 * 0.37;
            let normal = [angle.cos() * (step as f32 * 0.11).sin(), angle.sin(), (step as f32 * 0.23).cos() * 0.7];
            let color = 0x8000_0000 | step.wrapping_mul(0x9E37_79B9) >> 8;
            // The CPU: the bone applied to the normal first, then the world.
            let skinned: [f32; 3] = core::array::from_fn(|row| (0..3).map(|k| bone[3 * k + row] * normal[k]).sum());
            let expected = lights.color(Some(color), skinned, [0.0; 3]);
            let packed = unpack_normal(pack_normal(normal));
            let fast = gpu.color(color, packed, &rows, [0.0; 3]);
            for shift in [0, 8, 16, 24] {
                worst = worst.max(((expected >> shift) & 0xFF).abs_diff((fast >> shift) & 0xFF));
            }
        }
        // Packing arbitrary float normals into bytes turns them by up to
        // ~0.3 degrees; specular power 8 makes that up to 3 steps.
        assert!(worst <= 3, "worst channel difference {worst}");
    }

    /// God of War's point lights: two point lights (diffuse, attenuated,
    /// with an ambient colour) and a directional one, ambient and diffuse
    /// from the vertex colour.
    fn ge_with_point_lights() -> GeState {
        let mut ge = GeState::new();
        set(&mut ge, op::LIGHTING_ENABLE, 1);
        set(&mut ge, op::MATERIAL_UPDATE, 3);
        set(&mut ge, op::AMBIENT_COLOR, 0x20_2020);
        set(&mut ge, op::AMBIENT_ALPHA, 0xFF);
        set(&mut ge, op::MATERIAL_EMISSIVE, 0x04_0810);
        let positions = [[2.0f32, 3.0, -1.0], [-4.0, 1.0, 6.0], [0.3, 0.9, 0.3]];
        for (light, position) in positions.into_iter().enumerate() {
            let number = light as u8;
            set(&mut ge, op::LIGHT_ENABLE_0 + number, 1);
            set(&mut ge, op::LIGHT_TYPE_0 + number, if light < 2 { POINT << 8 } else { 0 });
            for (axis, value) in position.into_iter().enumerate() {
                set(&mut ge, op::LIGHT_X_0 + 3 * number + axis as u8, float(value));
            }
            for (axis, value) in [0.5f32, 0.25, 0.05].into_iter().enumerate() {
                set(&mut ge, op::LIGHT_ATTENUATION_A_0 + 3 * number + axis as u8, float(value));
            }
            set(&mut ge, op::LIGHT_AMBIENT_0 + 3 * number, 0x10_3050 + 0x10 * light as u32);
            set(&mut ge, op::LIGHT_AMBIENT_0 + 3 * number + 1, 0xC0_A080 - 0x20 * light as u32);
        }
        let world = rotation([0.48, 0.6, 0.64], 0.7, 1.5);
        set(&mut ge, op::WORLD_MATRIX_NUMBER, 0);
        for value in world.iter().chain(&[1.0, -2.0, 0.5]) {
            set(&mut ge, op::WORLD_MATRIX_DATA, float(*value));
        }
        ge
    }

    /// The point-light raw programs' lighting (`GpuLights::color` with a
    /// world position) gives what `Lights::color` gives, within a step (the
    /// CPU rounds each vertex's colour, PPSSPP's shader doesn't).
    #[test]
    fn gpu_point_lighting_matches_cpu_lighting() {
        let ge = ge_with_point_lights();
        let registers = LightingRegisters::of(&ge).unwrap();
        assert!(!registers.gpu_ready());
        assert!(registers.raw_ready(crate::vertex::VertexFormat(0)));
        let lights = Lights::of(&ge).unwrap();
        assert!(!lights.gpu_ready(), "the 3D program has directional lights only");
        let gpu = lights.gpu(true, false);
        assert_eq!(gpu.to_light[0][3], 1.0);
        assert_eq!(gpu.to_light[2][3], 0.0);
        let rows = lights.normal_rows(None);
        let world = lights.world();
        let mut worst = 0;
        for step in 0..400u32 {
            let angle = step as f32 * 0.37;
            let normal = [angle.cos() * (step as f32 * 0.11).sin(), angle.sin(), (step as f32 * 0.23).cos() * 0.7];
            let position = [(step as f32 * 0.07).sin() * 3.0, (step as f32 * 0.13).cos() * 2.0, step as f32 * 0.01 - 2.0];
            let color = 0x8000_0000 | step.wrapping_mul(0x9E37_79B9) >> 8;
            let expected = lights.color(Some(color), normal, position);
            let world_position: [f32; 3] = core::array::from_fn(|row| {
                world[row] * position[0] + world[3 + row] * position[1] + world[6 + row] * position[2] + world[9 + row]
            });
            let fast = gpu.color(color, normal, &rows, world_position);
            for shift in [0, 8, 16, 24] {
                worst = worst.max(((expected >> shift) & 0xFF).abs_diff((fast >> shift) & 0xFF));
            }
        }
        assert!(worst <= 1, "worst channel difference {worst}");
    }

    #[test]
    fn point_lights_go_raw_only_without_specular_spots_or_many_bones() {
        let mut ge = ge_with_point_lights();
        let format = |weights: u32| crate::vertex::VertexFormat(if weights == 0 { 0 } else { (1 << 9) | ((weights - 1) << 14) });
        assert!(LightingRegisters::of(&ge).unwrap().raw_ready(format(4)));
        assert!(!LightingRegisters::of(&ge).unwrap().raw_ready(format(5)));
        set(&mut ge, op::LIGHT_TYPE_0 + 1, (POINT << 8) | DIFFUSE_AND_SPECULAR);
        assert!(!LightingRegisters::of(&ge).unwrap().raw_ready(format(0)));
        set(&mut ge, op::LIGHT_TYPE_0 + 1, 2 << 8);
        assert!(!LightingRegisters::of(&ge).unwrap().raw_ready(format(0)));
    }

    #[test]
    fn normals_pack_into_signed_bytes() {
        assert_eq!(pack_normal([0.0, 0.0, 0.0]), 0);
        assert_eq!(unpack_normal(pack_normal([0.0, 0.0, -3.0])), [0.0, 0.0, -1.0]);
        // 8-bit normals keep their bytes: 0, 127, -128 (clamped to -127).
        assert_eq!(pack_normal([0.0, 127.0 / 128.0, -1.0]), 0x0081_7F00);
        // Others are scaled to the bytes' range: only the direction counts.
        let [x, y, z] = unpack_normal(pack_normal([0.3, -0.15, 0.0]));
        assert_eq!((x, z), (1.0, 0.0));
        assert!((y + 0.5).abs() < 0.01);
    }

    #[test]
    fn eight_bit_normals_pack_from_their_bytes_as_pack_normal_packs_them() {
        // Every x and y byte, z over its edge cases and a few others.
        for z in [0x80u8, 0x81, 0xFF, 0, 1, 5, 0x7F] {
            for x in 0..=255u8 {
                for y in 0..=255u8 {
                    let normal = [x, y, z].map(|byte| byte as i8 as f32 * (1.0 / 128.0));
                    assert_eq!(pack_s8_normal([x, y, z]), pack_normal(normal), "bytes {x:02x} {y:02x} {z:02x}");
                }
            }
        }
    }

    #[test]
    fn scaled_normals_pack_as_pack_normal_unless_they_are_exact() {
        // Pseudo-random normals of all lengths and signs, and halves.
        let mut seed = 0x1234_5678u32;
        let mut next = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1 << 24) as f32 * 4.0 - 2.0
        };
        let mut compared = 0;
        for _ in 0..200_000 {
            let normal = [next(), next(), next()];
            let exact = normal.iter().all(|&value| (value * 128.0).abs() <= 128.0 && value * 128.0 == ((value * 128.0) as i32) as f32);
            if !exact {
                assert_eq!(pack_scaled_normal(normal), pack_normal(normal), "{normal:?}");
                compared += 1;
            }
        }
        assert!(compared > 190_000);
        for normal in [[0.0; 3], [0.0, -0.0, 0.0], [f32::NAN, 0.0, 0.0], [f32::INFINITY, 1.0, 0.0], [0.5, -0.25, 1.0 / 254.0]] {
            assert_eq!(pack_scaled_normal(normal), pack_normal(normal), "{normal:?}");
        }
    }

    #[test]
    fn only_rotations_with_one_scale_and_directional_lights_go_to_model_space() {
        let lights = Lights::of(&ge_with_one_light(0)).unwrap();
        let rotated = rotation([0.0, 0.0, 1.0], 0.5, 3.0);
        assert!(lights.in_model_space(&rotated).is_some());
        let mut squashed = rotated;
        squashed[0] *= 0.5; // x scaled differently
        squashed[1] *= 0.5;
        squashed[2] *= 0.5;
        assert!(lights.in_model_space(&squashed).is_none());
        assert!(lights.in_model_space(&[0.0; 9]).is_none());
        let mut ge = ge_with_one_light(0);
        set(&mut ge, op::LIGHT_TYPE_0, 1 << 8); // a point light
        assert!(Lights::of(&ge).unwrap().in_model_space(&rotated).is_none());
    }

    /// The cached light set is the uncached one, bit for bit: after a
    /// material-only change (taken over), after a light change and after a
    /// view change (worked out again), for both vertex-colour and opacity
    /// variants.
    #[test]
    fn cached_light_sets_equal_the_ones_worked_out_whole() {
        let words = |set: &GpuLights| -> [u32; 88] {
            // Safe: GpuLights is 88 f32s (repr(C)).
            unsafe { core::mem::transmute_copy(set) }
        };
        let mut ge = ge_with_one_light(DIFFUSE_AND_SPECULAR);
        set(&mut ge, op::LIGHT_ENABLE_0 + 1, 1);
        set(&mut ge, op::LIGHT_X_0 + 3, float(0.6));
        set(&mut ge, op::LIGHT_X_0 + 4, float(0.8));
        set(&mut ge, op::LIGHT_AMBIENT_0 + 3, 0x10_2030);
        set(&mut ge, op::LIGHT_AMBIENT_0, 0x05_0607);
        let mut cache = GpuLightsCache::new();
        let check = |ge: &GeState, cache: &mut GpuLightsCache| {
            let registers = LightingRegisters::of(ge).expect("lights on");
            for (has_color, opaque) in [(true, true), (false, false), (true, false)] {
                let whole = Lights::from_registers(&registers, &ge.matrices.world, &ge.matrices.view).gpu(has_color, opaque);
                let cached = cache.gpu_lights(&registers, &ge.matrices.world, &ge.matrices.view, has_color, opaque);
                assert_eq!(words(&cached), words(&whole));
            }
        };
        check(&ge, &mut cache);
        // Material only (MTX's riders between parts): the lights' part is kept.
        set(&mut ge, op::MATERIAL_AMBIENT, 0x33_4455);
        set(&mut ge, op::MATERIAL_DIFFUSE, 0x80_8080);
        set(&mut ge, op::AMBIENT_COLOR, 0x01_0203);
        set(&mut ge, op::LIGHT_MODE, 1);
        check(&ge, &mut cache);
        assert!(cache.last.is_some_and(|last| last.lights.material_ambient[0] > 0.3), "material taken over");
        // A light changes: worked out again.
        set(&mut ge, op::LIGHT_X_0 + 5, float(0.5));
        check(&ge, &mut cache);
        // The view turns: the half vectors change.
        set(&mut ge, op::VIEW_MATRIX_NUMBER, 0);
        for value in [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0] {
            set(&mut ge, op::VIEW_MATRIX_DATA, float(value));
        }
        check(&ge, &mut cache);
    }

    #[test]
    fn without_any_light_on_there_are_no_lights() {
        let mut ge = ge_with_one_light(0);
        set(&mut ge, op::LIGHT_ENABLE_0, 0);
        assert!(Lights::of(&ge).is_none());
    }

    #[test]
    fn the_ges_pow_is_a_straight_line_between_powers_of_two() {
        assert_eq!(psp_pow(0.5, 0.0), 1.0);
        assert_eq!(psp_pow(0.0, 2.0), 0.0);
        assert_eq!(psp_pow(-1.0, 2.0), 0.0);
        assert_eq!(psp_pow(0.5, 2.0), 0.25);
        assert_eq!(psp_pow(4.0, 0.5), 2.0);
        // Between powers of two it is the line, not the curve: 0.75^1 stays 0.75.
        assert_eq!(psp_pow(0.75, 1.0), 0.75);
    }

    #[test]
    fn inverse_sqrt_is_close() {
        for x in [0.25f32, 1.0, 2.0, 100.0, 1e-4] {
            let expected = 1.0 / (x as f64).sqrt();
            assert!(((inverse_sqrt(x) as f64 - expected) / expected).abs() < 1e-6, "{x}");
        }
    }
}
