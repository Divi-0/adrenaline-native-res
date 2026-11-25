//! The GE's own depth of raw vertices, worked out on the CPU once per vertex
//! for the GE_DEPTH vertex programs (God of War's PSP-depth twins,
//! nativeres_raw_v.cg `aGeDepth`).
//!
//! Those programs used to do the GE's integer vertex arithmetic themselves
//! (`GdScreenZ`, ~700 of their ~880 instructions), once per index: strips
//! shade a vertex up to three times and the GPU keeps no results between
//! them. In God of War's gameplay that was ~35 ms of the GPU's ~220 ms per
//! frame (DESIGN "sixty-ninth session"), while the ARM cores were mostly
//! idle. Here the same arithmetic (`ge_math::integer::screen_z`, bit for bit
//! the program's, device test "gedepthtest") runs once per vertex and the
//! program reads the result as an attribute.
//!
//! The one input that is not plain integer arithmetic is the model-space
//! position: the program skins the raw position with float dot products
//! (`weight x dot(bone row, (x, y, z, 1))`, summed over the bones), and the
//! GPU may round those differently from sequential f32 steps. `SkinRounding`
//! picks how the CPU rounds; the Vita's pixel A/B decides which one gives
//! the program's pictures.
//!
//! What a value in the stream means (the program's `aGeDepth`):
//! - 0..65535: the GE's screen depth of the vertex;
//! - `NO_DEPTH` (-1): the GE gives none there (w zero, denormal or huge):
//!   the program's float depth stands in, as `GdScreenZ`'s y = 0 did;
//! - `ON_GPU` (-2): work it out in the program (the old way; debug
//!   `gedepthcpu 0`).

use crate::frame::{RawCamera, RawTransform};
use crate::ge_math;
use crate::vertex::{VertexFormat, VertexLayout, MAX_WEIGHTS};

/// The stream's value where the GE gives no depth.
pub const NO_DEPTH: f32 = -1.0;
/// The stream's value that has the program do the GE's arithmetic itself.
pub const ON_GPU: f32 = -2.0;

/// How the CPU rounds the skinning's float steps (debug `gedepthcpu 1|2`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SkinRounding {
    /// Each product and sum rounded to f32 in turn, left to right
    /// (`raw_blocks::model_space`, the replay's CPU copy of the program).
    Sequential,
    /// Each bone row's dot product rounded once (summed exactly, as a
    /// fused dot product unit would), the weighted sum over the bones as
    /// fused multiply-adds.
    SingleRounding,
}

/// What a raw draw's GE depths need besides its vertices: the clip z and w
/// columns of world x view x projection as the GE combines them
/// (`ge_math::combined`; rows x, y, z, translation) and the viewport's depth
/// scale and centre as float24s. The program's uGeZ, uGeW, uGeDepth.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct GeDepthInputs {
    pub column_z: [f32; 4],
    pub column_w: [f32; 4],
    pub scale: f32,
    pub center: f32,
}

impl GeDepthInputs {
    /// A raw draw's inputs from its transform and camera (as core
    /// raw_vertex `exact_z` and the replay's `--ge-depth`).
    pub fn new(transform: &RawTransform, camera: &RawCamera) -> Self {
        let m = ge_math::combined(&transform.world, &camera.view, &camera.projection);
        let float24 = |value: f32| f32::from_bits((value.to_bits() + 0x80) & 0xFFFF_FF00);
        GeDepthInputs {
            column_z: [m[2], m[6], m[10], m[14]],
            column_w: [m[3], m[7], m[11], m[15]],
            scale: float24(camera.depth[1] * 65535.0),
            center: float24(camera.depth[0] * 65535.0),
        }
    }

    /// The integer arithmetic's per-transform part (`ge_math::integer::Prepared`).
    pub fn prepared(&self) -> ge_math::integer::Prepared {
        ge_math::integer::Prepared::new(self.column_z, self.column_w, self.scale, self.center)
    }

    /// As the program's uniforms: uGeZ, uGeW, uGeDepth.
    pub fn uniforms(&self) -> [[f32; 4]; 3] {
        [self.column_z, self.column_w, [self.scale, self.center, 0.0, 0.0]]
    }
}

/// One raw draw's vertices for `fill`: their bytes (vertex 0 at byte 0),
/// format and the bones the program skins them with (12 floats per bone;
/// empty: not skinned, as `Frame::bones_of` and the program's uSkin).
pub struct RawDrawVertices<'a> {
    pub bytes: &'a [u8],
    pub format: VertexFormat,
    pub layout: VertexLayout,
    pub bones: &'a [f32],
    /// Bytes of a vertex the format reads (`VertexFormat::bytes_used`).
    used: usize,
}

impl<'a> RawDrawVertices<'a> {
    /// `vertex_format` as the draw has it (`TEXCOORDS_FROM_POSITION` is
    /// not part of the layout).
    pub fn new(bytes: &'a [u8], vertex_format: u32, bones: &'a [f32]) -> Self {
        let format = VertexFormat(vertex_format & !crate::raw_blocks::TEXCOORDS_FROM_POSITION);
        let layout = format.layout();
        RawDrawVertices { bytes, format, used: format.bytes_used(&layout) as usize, layout, bones }
    }

    /// The model-space position of vertex `index` as the program skins it
    /// (with `rounding`); None if its bytes are not all there.
    pub fn model_position(&self, index: u32, rounding: SkinRounding) -> Option<[f32; 3]> {
        let offset = index as usize * self.layout.size as usize;
        let (position, weights) = self.position_and_weights(offset)?;
        Some(match rounding {
            SkinRounding::Sequential => skin_sequential(self.format, self.bones, position, &weights),
            SkinRounding::SingleRounding => skin_single_rounding(self.format, self.bones, position, &weights),
        })
    }

    /// The raw position and the scaled weights of the vertex at byte
    /// `offset`, as `decode_raw` reads them (`VertexFormat::decode_3d`: the
    /// same field offsets, 16-bit fields from their word half, floats only
    /// at word offsets), but nothing else of the vertex: the GE depth needs
    /// no more, and this runs once per vertex.
    fn position_and_weights(&self, offset: usize) -> Option<([f32; 3], [f32; MAX_WEIGHTS])> {
        let vertex = self.bytes.get(offset..offset + self.used)?;
        // Past the format's bytes decode_raw reads zeros (its word padding).
        let byte = |at: u32| vertex.get(at as usize).copied().unwrap_or(0);
        let half = |at: u32| {
            let at = ((at & !3) + (at & 2)) as usize; // decode_3d reads the word's half
            match vertex.get(at..at + 2) {
                Some(two) => u16::from_le_bytes([two[0], two[1]]),
                None => u16::from_le_bytes([byte(at as u32), byte(at as u32 + 1)]),
            }
        };
        let word = |at: u32| {
            (at % 4 == 0).then(|| match vertex.get(at as usize..at as usize + 4) {
                Some(four) => f32::from_le_bytes([four[0], four[1], four[2], four[3]]),
                None => f32::from_bits(u32::from_le_bytes([byte(at), byte(at + 1), byte(at + 2), byte(at + 3)])),
            })
        };
        let at = self.layout.position_offset;
        let position = match (self.format.0 >> 7) & 3 {
            0 | 1 => [byte(at) as i8 as f32, byte(at + 1) as i8 as f32, byte(at + 2) as i8 as f32],
            2 => [half(at) as i16 as f32, half(at + 2) as i16 as f32, half(at + 4) as i16 as f32],
            _ => [word(at)?, word(at + 4)?, word(at + 8)?],
        };
        let mut weights = [0.0; MAX_WEIGHTS];
        if self.format.has_weights() {
            let count = (self.format.weight_count() as usize).min(MAX_WEIGHTS);
            for (index, weight) in weights[..count].iter_mut().enumerate() {
                let index = index as u32;
                *weight = match self.format.weight_format() {
                    1 => byte(index) as f32 * (1.0 / 128.0),
                    2 => half(2 * index) as f32 * (1.0 / 32768.0),
                    _ => word(4 * index)?,
                };
            }
        }
        Some((position, weights))
    }

    /// The stream value of vertex `index` (see the module's head).
    pub fn ge_depth(&self, index: u32, inputs: &GeDepthInputs, rounding: SkinRounding) -> f32 {
        self.ge_depth_prepared(index, &inputs.prepared(), rounding)
    }

    #[inline]
    fn ge_depth_prepared(&self, index: u32, prepared: &ge_math::integer::Prepared, rounding: SkinRounding) -> f32 {
        let Some(model) = self.model_position(index, rounding) else {
            return NO_DEPTH;
        };
        prepared.screen_z(model).unwrap_or(NO_DEPTH)
    }

    /// `out[k]` = the stream value of vertex `first + k`.
    pub fn fill(&self, first: u32, out: &mut [f32], inputs: &GeDepthInputs, rounding: SkinRounding) {
        let prepared = inputs.prepared();
        for (index, value) in (first..).zip(out.iter_mut()) {
            *value = self.ge_depth_prepared(index, &prepared, rounding);
        }
    }
}

/// `raw_blocks::model_space`'s position (its normal left out), the same
/// steps: times the format's scale, then per bone and row
/// `model += weight x (((r0 x + r1 y) + r2 z) + r3)`.
#[inline]
fn skin_sequential(format: VertexFormat, bones: &[f32], position: [f32; 3], weights: &[f32]) -> [f32; 3] {
    let scale = format.position_scale();
    let [x, y, z] = position.map(|value| value * scale);
    if bones.is_empty() {
        return [x, y, z];
    }
    let mut model = [0.0f32; 3];
    for (bone, &weight) in bones.chunks_exact(12).zip(weights) {
        for (row, sum) in model.iter_mut().enumerate() {
            let r = &bone[4 * row..4 * row + 4];
            *sum += weight * (r[0] * x + r[1] * y + r[2] * z + r[3]);
        }
    }
    model
}

/// `raw_blocks::model_space`'s position with `SkinRounding::SingleRounding`.
fn skin_single_rounding(format: VertexFormat, bones: &[f32], position: [f32; 3], weights: &[f32]) -> [f32; 3] {
    let scale = format.position_scale();
    let scaled = position.map(|value| value * scale); // a power of two: exact
    if bones.is_empty() {
        return scaled;
    }
    let [x, y, z] = scaled.map(f64::from);
    let mut model = [0.0f32; 3];
    for (bone, &weight) in bones.chunks_exact(12).zip(weights) {
        for (row, sum) in model.iter_mut().enumerate() {
            let r = &bone[4 * row..4 * row + 4];
            // f32 x f32 products are exact in f64; their sum rounded once
            // (to f64, then f32: a double rounding, rare enough for a probe).
            let dot = (r[0] as f64 * x + r[1] as f64 * y + r[2] as f64 * z + r[3] as f64) as f32;
            *sum = (weight as f64 * dot as f64 + *sum as f64) as f32;
        }
    }
    model
}

/// The vertex ranges (first, end) a raw draw's index runs use
/// (`draw_flags::INDEX_RUNS`: [primitive, count, first vertex] each),
/// sorted and joined where they touch or overlap. Rectangles (raw sprites)
/// have none here: their programs have no GE depth.
pub fn run_ranges(runs: &[u16], out: &mut [(u32, u32)]) -> usize {
    let mut count = 0;
    for run in runs.chunks_exact(3) {
        if run[0] as u32 == crate::command::primitive::RECTANGLES || run[1] == 0 || count == out.len() {
            continue;
        }
        out[count] = (run[2] as u32, run[2] as u32 + run[1] as u32);
        count += 1;
    }
    let ranges = &mut out[..count];
    ranges.sort_unstable();
    let mut joined = 0;
    for k in 0..count {
        let (first, end) = ranges[k];
        if joined > 0 && first <= ranges[joined - 1].1 {
            ranges[joined - 1].1 = ranges[joined - 1].1.max(end);
        } else {
            ranges[joined] = (first, end);
            joined += 1;
        }
    }
    joined
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{RawCamera, RawTransform};
    use crate::raw_vertex::decode_raw;

    /// A small xorshift for repeatable random inputs.
    struct Random(u32);
    impl Random {
        fn next(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 17;
            self.0 ^= self.0 << 5;
            self.0
        }
        fn float(&mut self, range: f32) -> f32 {
            (self.next() as f32 / u32::MAX as f32 * 2.0 - 1.0) * range
        }
    }

    fn camera() -> (RawTransform, RawCamera) {
        let mut transform = RawTransform::default();
        transform.world = [[1.0, 0.0, 0.0, 3.0], [0.0, 1.0, 0.0, -2.0], [0.0, 0.0, 1.0, -60.0]];
        let mut camera = RawCamera::default();
        camera.view = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]];
        // A perspective projection (near 1, far 1000), rows as the GE has them.
        camera.projection = [[1.5, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, -1.002, -2.002], [0.0, 0.0, -1.0, 0.0]];
        camera.depth = [0.5, -0.5, 0.0, 0.0];
        (transform, camera)
    }

    /// Float positions (format: position f32, 8-bit weight x1 for skinned).
    fn vertices(random: &mut Random, count: usize, weights: bool) -> (Vec<u8>, u32) {
        // VERTEX_TYPE: position float (3 << 7); one 8-bit weight (1 << 9, count 1 = bits 14..16 = 0).
        let format = if weights { (3 << 7) | (1 << 9) } else { 3 << 7 };
        let layout = VertexFormat(format).layout();
        let mut bytes = vec![0u8; count * layout.size as usize];
        for k in 0..count {
            let base = k * layout.size as usize;
            if weights {
                bytes[base] = (random.next() % 129) as u8;
            }
            for c in 0..3 {
                let at = base + layout.position_offset as usize + 4 * c;
                bytes[at..at + 4].copy_from_slice(&random.float(40.0).to_le_bytes());
            }
        }
        (bytes, format)
    }

    #[test]
    fn sequential_is_the_cpu_copy_of_the_program() {
        let (transform, camera) = camera();
        let inputs = GeDepthInputs::new(&transform, &camera);
        let mut random = Random(0x1234_5678);
        for weights in [false, true] {
            let (bytes, format) = vertices(&mut random, 500, weights);
            let bones: Vec<f32> = if weights { (0..12).map(|_| random.float(2.0)).collect() } else { Vec::new() };
            let draw = RawDrawVertices::new(&bytes, format, &bones);
            let mut out = vec![0.0f32; 500];
            draw.fill(0, &mut out, &inputs, SkinRounding::Sequential);
            let with_depth = out.iter().filter(|&&value| value != NO_DEPTH).count();
            assert!(with_depth > 450, "only {with_depth} of 500 with a depth");
            let distinct: std::collections::BTreeSet<u32> = out.iter().map(|value| value.to_bits()).collect();
            assert!(distinct.len() > 100, "depths hardly differ");
            for (index, &value) in out.iter().enumerate() {
                let fields = decode_raw(&bytes, index * draw.layout.size as usize, draw.format, &draw.layout).unwrap();
                let (model, _) = crate::raw_blocks::model_space(draw.format, &bones, fields.position, &fields.weights, fields.normal);
                let expected = ge_math::integer::screen_z(model, inputs.column_z, inputs.column_w, inputs.scale, inputs.center).unwrap_or(NO_DEPTH);
                assert_eq!(value.to_bits(), expected.to_bits(), "vertex {index}");
                assert!(value == NO_DEPTH || (0.0..=65535.0).contains(&value) && value.fract() == 0.0);
            }
        }
    }

    /// The lean reader gives what `decode_raw` gives, for every position and
    /// weight format and weight count, with the other fields shifting the
    /// offsets around (texture coordinates, colours, normals).
    #[test]
    fn position_and_weights_read_as_decode_raw() {
        let mut random = Random(0xBEEF);
        let mut checked = 0;
        for position in 1..4u32 {
            for weights in 0..4u32 {
                for count in 0..8u32 {
                    for (texcoord, color, normal) in [(0, 0, 0), (1, 7, 1), (2, 4, 2), (3, 6, 3), (1, 5, 0)] {
                        let format = texcoord | color << 2 | normal << 5 | position << 7 | weights << 9 | count << 14;
                        let draw = RawDrawVertices::new(&[], format, &[]);
                        let size = draw.layout.size as usize;
                        let mut bytes = vec![0u8; 8 * size];
                        for byte in bytes.iter_mut() {
                            *byte = random.next() as u8;
                        }
                        let draw = RawDrawVertices::new(&bytes, format, &[]);
                        for index in 0..9 {
                            let expected = decode_raw(&bytes, index * size, draw.format, &draw.layout)
                                .map(|fields| (fields.position.map(f32::to_bits), fields.weights.map(f32::to_bits)));
                            let got = draw.position_and_weights(index * size).map(|(p, w)| (p.map(f32::to_bits), w.map(f32::to_bits)));
                            assert_eq!(got, expected, "format {format:08x} vertex {index}");
                            checked += 1;
                        }
                    }
                }
            }
        }
        assert!(checked > 2000);
    }

    /// `skin_sequential` is `raw_blocks::model_space`'s position, bit for bit.
    #[test]
    fn sequential_skin_is_model_space() {
        let mut random = Random(4242);
        for count in 0..8u32 {
            for position in 1..4u32 {
                let format = VertexFormat(position << 7 | 1 << 9 | count << 14);
                for _ in 0..2000 {
                    let bones: Vec<f32> = (0..12 * (count as usize + 1)).map(|_| random.float(3.0)).collect();
                    let weights: Vec<f32> = (0..MAX_WEIGHTS).map(|_| random.float(1.0)).collect();
                    let raw = [random.float(30000.0), random.float(30000.0), random.float(30000.0)];
                    let (expected, _) = crate::raw_blocks::model_space(format, &bones, raw, &weights, [0.0, 0.0, 1.0]);
                    assert_eq!(skin_sequential(format, &bones, raw, &weights).map(f32::to_bits), expected.map(f32::to_bits));
                    assert_eq!(skin_sequential(format, &[], raw, &weights).map(f32::to_bits), crate::raw_blocks::model_space(format, &[], raw, &weights, [0.0; 3]).0.map(f32::to_bits));
                }
            }
        }
    }

    #[test]
    fn single_rounding_equals_sequential_without_bones() {
        let (transform, camera) = camera();
        let inputs = GeDepthInputs::new(&transform, &camera);
        let mut random = Random(99);
        let (bytes, format) = vertices(&mut random, 300, false);
        let draw = RawDrawVertices::new(&bytes, format, &[]);
        for index in 0..300 {
            assert_eq!(draw.ge_depth(index, &inputs, SkinRounding::Sequential), draw.ge_depth(index, &inputs, SkinRounding::SingleRounding));
        }
    }

    #[test]
    fn uniforms_are_the_programs() {
        let (transform, camera) = camera();
        let inputs = GeDepthInputs::new(&transform, &camera);
        let m = ge_math::combined(&transform.world, &camera.view, &camera.projection);
        let u = inputs.uniforms();
        assert_eq!(u[0], [m[2], m[6], m[10], m[14]]);
        assert_eq!(u[1], [m[3], m[7], m[11], m[15]]);
        assert_eq!(u[2][0].to_bits() & 0xFF, 0); // float24s
        assert_eq!(u[2][1].to_bits() & 0xFF, 0);
    }

    #[test]
    fn missing_bytes_give_no_depth() {
        let (transform, camera) = camera();
        let inputs = GeDepthInputs::new(&transform, &camera);
        let mut random = Random(7);
        let (bytes, format) = vertices(&mut random, 4, false);
        let draw = RawDrawVertices::new(&bytes, format, &[]);
        assert_eq!(draw.ge_depth(4, &inputs, SkinRounding::Sequential), NO_DEPTH);
    }

    #[test]
    fn run_ranges_join_and_skip_sprites() {
        use crate::command::primitive::{RECTANGLES, TRIANGLE_STRIP};
        let strip = TRIANGLE_STRIP as u16;
        let runs = [strip, 5, 20, strip, 4, 0, RECTANGLES as u16, 3, 100, strip, 6, 22, strip, 0, 50, strip, 3, 4];
        let mut out = [(0u32, 0u32); 8];
        let count = run_ranges(&runs, &mut out);
        // (0,4) and (4,7) touch, (20,25) and (22,28) overlap; the sprite run and the empty one drop out.
        assert_eq!(&out[..count], &[(0, 7), (20, 28)]);
    }
}
