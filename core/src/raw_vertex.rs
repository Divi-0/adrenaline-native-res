//! What the GPU does with raw PSP vertices (`draw_flags::RAW_VERTICES`,
//! nativepsp_raw_v.cg), done on the CPU: for the PC replay, the frame
//! hashes and the tests. Keep both in step.
//!
//! A raw vertex is decoded from its bytes as the fast path decodes it
//! (`VertexFormat::decode_3d`: raw position, raw texture coordinates, the
//! colour widened to 8 bits per channel), then:
//! - position: clip = `rows` x (raw x, y, z, 1) (the rows hold the position
//!   scale and, for single-weight runs, the bone x weight); skinned in the
//!   shader (`Draw::bones`): the weighted sum of the bones applied
//!   to (raw x, y, z, 1) first, as `Transform::skin_raw`;
//! - texture coordinates: raw x (u, v scale) + (u, v offset) (`raw_uv`);
//! - colour: `RawColors::color` (and `lit_color`), what `recorder::Prepared`'s
//!   `base_color` and `frame_color` do;
//! - fog: `fog_row` . (raw x, y, z, 1).
//!
//! Triangles are cut at the PSP's near plane (clip z = -w) like the CPU path
//! cuts them (`transform::clip_triangle`); the GPU clips at the same plane
//! (its clip volume is -w <= z <= w, measured with debug "rawtest";
//! the shader puts its far plane out of reach).

use crate::frame::{Frame, FrameVertex, GpuLights, GpuTransform, RawColors};
use crate::transform::{clip_triangle, ClipVertex};
use crate::vertex::{VertexBytes, VertexFormat, VertexLayout};

/// `RawColors::rules[0]`: how a raw vertex is lit.
pub mod lighting {
    /// Not lit: its colour (or `raw_material` without one).
    pub const NONE: f32 = 0.0;
    /// Lit without lights (`recorder::AmbientLighting`), the vertex colour
    /// as material where it has one (MATERIAL_UPDATE bit 0).
    pub const VERTEX_MATERIAL: f32 = 1.0;
    /// Lit without lights, `raw_material` as material.
    pub const MATERIAL: f32 = 2.0;
    /// Lit with directional lights by the GPU: the transform's light set
    /// (`GpuTransform::light_set`, `normal_rows`) lights the vertex colour
    /// (white without one) with the vertex's normal, as nativepsp_3d_v.cg
    /// lights GPU-lit vertices (`GpuLights::color`).
    pub const LIGHTS: f32 = 3.0;
}

/// `RawColors::rules[2]`: the texture function "replace" (textured draws).
pub mod replace {
    pub const NONE: f32 = 0.0;
    /// RGB white, the vertex alpha kept (the texture alpha is switched off).
    pub const RGB: f32 = 1.0;
    /// All white (the texture's colour and alpha as they are).
    pub const ALL: f32 = 2.0;
}

/// The colour channels of 0xAABBGGRR as whole numbers 0..255 (R, G, B, A).
pub fn channels(rgba: u32) -> [f32; 4] {
    [0, 8, 16, 24].map(|shift| ((rgba >> shift) & 0xFF) as f32)
}

/// 0..255 channels (R, G, B, A) back to 0xAABBGGRR.
pub fn rgba(channels: [f32; 4]) -> u32 {
    channels.iter().enumerate().fold(0, |rgba, (index, &value)| rgba | (value as u32 & 0xFF) << (8 * index))
}

impl RawColors {
    /// The colour a raw vertex with colour `vertex` (`None`: none) ends up
    /// with, as the shader works it out (whole numbers, so exactly).
    pub fn color(&self, vertex: Option<u32>) -> u32 {
        let [lit, opaque, replacing, doubled] = self.rules;
        let mut color = match vertex {
            Some(color) => channels(color),
            None => self.material,
        };
        if lit == lighting::LIGHTS {
            // Lit by `lit_color` (needs the normal and the light set); here
            // only the base colour.
            return rgba(color);
        }
        if lit != lighting::NONE {
            // `AmbientLighting::color`: ((2m + 1)(2a + 1)) >> 10, plus the
            // emission, at most 255; m the vertex colour or the material.
            let material = if lit == lighting::VERTEX_MATERIAL && vertex.is_some() { color } else { self.material };
            for channel in 0..4 {
                let product = (2.0 * material[channel] + 1.0) * (2.0 * self.ambient[channel] + 1.0);
                // A whole number below 2^18: the shift is the shader's floor(product / 1024).
                color[channel] = (((product as u32) >> 10) as f32 + self.emissive[channel]).min(255.0);
            }
        }
        if opaque != 0.0 {
            color[3] = 255.0;
        }
        if doubled != 0.0 {
            color[3] = (2.0 * color[3]).min(255.0);
        }
        if replacing == replace::ALL {
            color = [255.0; 4];
        } else if replacing == replace::RGB {
            color[..3].fill(255.0);
        }
        if self.scale[3] != 0.0 {
            // FIX A: `blend::scale_color`, (c * s) / 255 rounded down; the
            // shader's floor((c * s + 0.5) / 255) is that for whole numbers.
            for channel in 0..3 {
                color[channel] = ((color[channel] as u32 * self.scale[channel] as u32) / 255) as f32;
            }
        }
        rgba(color)
    }

    /// The colour of a raw vertex lit by the GPU (`lighting::LIGHTS`): its
    /// base colour (`color`) lit with `lights`, the raw normal and the
    /// transform's normal rows (the normal's scale doesn't matter: the GPU
    /// normalizes), then made opaque if the draw is.
    pub fn lit_color(&self, lights: &GpuLights, normal_rows: &[[f32; 4]; 3], vertex: Option<u32>, normal: [f32; 3]) -> u32 {
        let lit = lights.color(self.color(vertex), normal, normal_rows);
        if self.rules[1] != 0.0 {
            lit | 0xFF00_0000
        } else {
            lit
        }
    }
}

impl GpuTransform {
    /// A raw vertex (raw position, texture coordinates, its colour from
    /// `colors`) in clip space, with its texture coordinates, colour and fog.
    pub fn raw_clip_vertex(&self, colors: &RawColors, position: [f32; 3], texcoord: [f32; 2], color: Option<u32>) -> ClipVertex {
        let [x, y, z] = position;
        let dot = |r: &[f32; 4]| r[0] * x + r[1] * y + r[2] * z + r[3];
        let [u_scale, v_scale, u_offset, v_offset] = self.raw_uv;
        ClipVertex {
            x: dot(&self.rows[0]),
            y: dot(&self.rows[1]),
            z: dot(&self.rows[2]),
            w: dot(&self.rows[3]),
            u: texcoord[0] * u_scale + u_offset,
            v: texcoord[1] * v_scale + v_offset,
            color: colors.color(color),
            fog: dot(&self.fog_row),
        }
    }

    /// A clip-space vertex in screen space, as `to_screen` turns one.
    pub fn clip_to_screen(&self, clip: &ClipVertex) -> FrameVertex {
        let over_w = 1.0 / clip.w;
        FrameVertex {
            x: self.viewport[0] + self.viewport[2] * clip.x * over_w,
            y: self.viewport[1] + self.viewport[3] * clip.y * over_w,
            z: (self.depth[0] + self.depth[1] * clip.z * over_w).clamp(0.0, 1.0),
            w: clip.w,
            u: clip.u,
            v: clip.v,
            color: clip.color,
            fog: clip.fog,
        }
    }
}

/// What `decode_raw` reads of a raw vertex.
pub struct RawFields {
    pub position: [f32; 3],
    pub texcoord: [f32; 2],
    pub color: Option<u32>,
    /// (0, 0, 1) without one.
    pub normal: [f32; 3],
    /// Scaled (128 = 1.0 for 8-bit weights); only the format's count set.
    pub weights: [f32; crate::vertex::MAX_WEIGHTS],
}

/// The raw fields of the vertex at byte `offset` of `bytes` (`format`'s
/// layout); `None` if it runs past the end.
pub fn decode_raw(bytes: &[u8], offset: usize, format: VertexFormat, layout: &VertexLayout) -> Option<RawFields> {
    let vertex = bytes.get(offset..offset + format.bytes_used(layout) as usize)?;
    // Into words for the decoders (the arena doesn't align vertices); a
    // vertex has at most 68 bytes.
    let mut words = [0u32; 18];
    for (word, chunk) in words.iter_mut().zip(vertex.chunks(4)) {
        let mut four = [0u8; 4];
        four[..chunk.len()].copy_from_slice(chunk);
        *word = u32::from_le_bytes(four);
    }
    let vertex_bytes = VertexBytes::new(&words, 0);
    let raw = format.decode_3d(&vertex_bytes, layout)?;
    let normal = if layout.normal_offset.is_some() { format.decode_normal(&vertex_bytes, layout)? } else { [0.0, 0.0, 1.0] };
    Some(RawFields { position: raw.position, texcoord: raw.texcoord, color: raw.color, normal, weights: raw.weights })
}

/// A skinned raw vertex's position and normal (`Draw::bones`):
/// the weighted sums of `bones` (12 floats each, as rows) applied to (raw x,
/// y, z, 1) and to the normal, as the shader adds them up.
pub fn skin(bones: &[f32], fields: &RawFields) -> ([f32; 3], [f32; 3]) {
    let [x, y, z] = fields.position;
    let [nx, ny, nz] = fields.normal;
    let mut position = [0.0; 3];
    let mut normal = [0.0; 3];
    for (bone, &weight) in bones.chunks_exact(12).zip(&fields.weights) {
        for row in 0..3 {
            let r = &bone[4 * row..4 * row + 4];
            position[row] += weight * (r[0] * x + r[1] * y + r[2] * z + r[3]);
            normal[row] += weight * (r[0] * nx + r[1] * ny + r[2] * nz);
        }
    }
    (position, normal)
}

/// A raw draw's triangle corners in screen space (three per triangle), its
/// triangles cut at the near plane (`Frame::screen_corners`).
pub struct RawCorners<'a, I: Iterator<Item = u16>> {
    indices: I,
    bytes: &'a [u8],
    window: usize,
    stride: usize,
    format: VertexFormat,
    layout: VertexLayout,
    transform: GpuTransform,
    colors: RawColors,
    /// The light set of a draw lit by the GPU (`lighting::LIGHTS`).
    lights: Option<GpuLights>,
    /// The bones of a draw skinned in the shader (12 floats each; empty: not).
    bones: &'a [f32],
    /// Raw sprites (`draw_flags::RAW_SPRITES`): index k is pair k, corner k % 4.
    sprites: bool,
    /// Corners of the last triangle (0, 3 or 6 after cutting), and the next.
    pending: [FrameVertex; 6],
    pending_count: usize,
    next: usize,
}

impl<'a, I: Iterator<Item = u16>> RawCorners<'a, I> {
    pub fn new(
        frame: &'a Frame,
        window: u32,
        vertex_format: u32,
        transform: GpuTransform,
        colors: RawColors,
        bones: &'a [f32],
        indices: I,
        sprites: bool,
    ) -> Self {
        let format = VertexFormat(vertex_format);
        let layout = format.layout();
        let lights = (colors.rules[0] == lighting::LIGHTS).then(|| frame.light_set_of(&transform).copied()).flatten();
        RawCorners {
            colors,
            lights,
            bones,
            indices,
            bytes: frame.raw_bytes(),
            window: window as usize,
            stride: layout.size as usize,
            format,
            layout,
            transform,
            sprites,
            pending: [FrameVertex::default(); 6],
            pending_count: 0,
            next: 0,
        }
    }

    fn clip_vertex(&self, index: u16) -> Option<ClipVertex> {
        let offset = self.window + index as usize * self.stride;
        let fields = decode_raw(self.bytes, offset, self.format, &self.layout)?;
        let (position, normal) = if self.bones.is_empty() { (fields.position, fields.normal) } else { skin(self.bones, &fields) };
        let mut vertex = self.transform.raw_clip_vertex(&self.colors, position, fields.texcoord, fields.color);
        if let Some(lights) = &self.lights {
            vertex.color = self.colors.lit_color(lights, &self.transform.normal_rows, fields.color, normal);
        }
        Some(vertex)
    }

    /// Corner `index % 4` of a raw sprite on the screen, as
    /// nativepsp_raw_sprite_v.cg makes it (and the CPU path,
    /// `recorder::record_vertices_now`): x and u from B for corners 1 and 3,
    /// y and v from B for corners 2 and 3, the fog of the corner's y vertex,
    /// colour and depth from B, flat (w 1). `None` if a corner is behind
    /// the near plane (the sprite is left out) or unreadable.
    fn sprite_corner(&self, index: u16) -> Option<FrameVertex> {
        let offset = self.window + 2 * index as usize * self.stride;
        let corner = |offset: usize| -> Option<FrameVertex> {
            let fields = decode_raw(self.bytes, offset, self.format, &self.layout)?;
            let clip = self.transform.raw_clip_vertex(&self.colors, fields.position, fields.texcoord, fields.color);
            (clip.z >= -clip.w && clip.w > 0.0).then(|| self.transform.clip_to_screen(&clip))
        };
        let (a, b) = (corner(offset)?, corner(offset + self.stride)?);
        let (x_from, y_from) = if index % 4 & 1 != 0 { (b, a) } else { (a, a) };
        let y_from = if index % 4 & 2 != 0 { b } else { y_from };
        Some(FrameVertex { x: x_from.x, u: x_from.u, y: y_from.y, v: y_from.v, fog: y_from.fog, z: b.z, w: 1.0, color: b.color })
    }
}

impl<I: Iterator<Item = u16>> Iterator for RawCorners<'_, I> {
    type Item = FrameVertex;

    fn next(&mut self) -> Option<FrameVertex> {
        while self.next == self.pending_count {
            let (a, b, c) = (self.indices.next()?, self.indices.next()?, self.indices.next()?);
            self.next = 0;
            self.pending_count = 0;
            if self.sprites {
                if let (Some(a), Some(b), Some(c)) = (self.sprite_corner(a), self.sprite_corner(b), self.sprite_corner(c)) {
                    self.pending[..3].copy_from_slice(&[a, b, c]);
                    self.pending_count = 3;
                }
                continue;
            }
            let (Some(a), Some(b), Some(c)) = (self.clip_vertex(a), self.clip_vertex(b), self.clip_vertex(c)) else {
                continue; // unreadable: left out
            };
            let mut cut = [a; 4];
            let count = clip_triangle(&[a, b, c], &mut cut);
            // 3 corners: one triangle; 4: a quad, two triangles (0 1 2, 0 2 3).
            let order: &[usize] = match count {
                3 => &[0, 1, 2],
                4 => &[0, 1, 2, 0, 2, 3],
                _ => &[],
            };
            for (slot, &corner) in order.iter().enumerate() {
                self.pending[slot] = self.transform.clip_to_screen(&cut[corner]);
            }
            self.pending_count = order.len();
        }
        let corner = self.pending[self.next];
        self.next += 1;
        Some(corner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::{rgba_from_4444, rgba_from_5551, rgba_from_5650};

    fn transform() -> RawColors {
        RawColors { material: [10.0, 20.0, 30.0, 40.0], ..RawColors::default() }
    }

    #[test]
    fn raw_colours_follow_the_recorder() {
        let mut t = transform();
        assert_eq!(t.color(Some(0x8040_2010)), 0x8040_2010);
        assert_eq!(t.color(None), rgba([10.0, 20.0, 30.0, 40.0]));
        t.rules = [0.0, 1.0, 0.0, 0.0];
        assert_eq!(t.color(Some(0x8040_2010)), 0xFF40_2010);
        t.rules = [0.0, 0.0, replace::RGB, 0.0];
        assert_eq!(t.color(Some(0x8040_2010)), 0x80FF_FFFF);
        t.rules = [0.0, 0.0, replace::ALL, 0.0];
        assert_eq!(t.color(Some(0x8020_4060)), 0xFFFF_FFFF);
        // Doubled source alpha: the alpha doubled, at most 255, before replace.
        t.rules = [0.0, 0.0, replace::NONE, 1.0];
        assert_eq!(t.color(Some(0x5020_4060)), 0xA020_4060);
        assert_eq!(t.color(Some(0x9020_4060)), 0xFF20_4060);
        t.rules = [0.0, 0.0, replace::RGB, 1.0];
        assert_eq!(t.color(Some(0x5020_4060)), 0xA0FF_FFFF);
        assert_eq!(t.color(Some(0x8040_2010)), 0xFFFF_FFFF);
    }

    #[test]
    fn ambient_lighting_matches_the_recorders_integer_formula() {
        // ((2m + 1)(2a + 1)) >> 10 + emissive, as AmbientLighting::color.
        let mut t = transform();
        t.ambient = [255.0, 128.0, 7.0, 200.0];
        t.emissive = [3.0, 250.0, 0.0, 0.0];
        t.rules = [lighting::VERTEX_MATERIAL, 0.0, 0.0, 0.0];
        let vertex = 0x9080_7060u32;
        let expected: [u32; 4] = core::array::from_fn(|channel| {
            let m = (vertex >> (8 * channel)) & 0xFF;
            let a = [255u32, 128, 7, 200][channel];
            let e = [3u32, 250, 0, 0][channel];
            ((((2 * m + 1) * (2 * a + 1)) >> 10) + e).min(255)
        });
        let expected = expected.iter().enumerate().fold(0, |out, (c, &v)| out | v << (8 * c));
        assert_eq!(t.color(Some(vertex)), expected);
        // MATERIAL: the vertex colour is ignored.
        t.rules[0] = lighting::MATERIAL;
        assert_eq!(t.color(Some(vertex)), t.color(None));
    }

    #[test]
    fn fix_a_scales_like_the_recorder_and_the_shader() {
        // RawColors::scale is blend::scale_color, last; the shader computes
        // floor((c * s + 0.5) * (1 / 255)) in floats: the same for all c, s.
        let mut t = transform();
        for scale in [0x00C8_C8C8u32, 0x0080_4020, 0x00FF_FFFF, 0x0001_00FE] {
            let [r, g, b, _] = channels(scale);
            t.scale = [r, g, b, 1.0];
            for vertex in [0x8040_2010u32, 0xFFFF_FFFF, 0x0000_0000, 0x7F01_FE80] {
                assert_eq!(t.color(Some(vertex)), crate::blend::scale_color(vertex, scale));
            }
        }
        for c in 0..=255u32 {
            for s in 0..=255u32 {
                let shader = (((c * s) as f32 + 0.5) * (1.0f32 / 255.0)).floor() as u32;
                assert_eq!(shader, c * s / 255, "c {c} s {s}");
            }
        }
        // After replace: RGB white times FIX A.
        t.scale = [200.0, 100.0, 0.0, 1.0];
        t.rules = [0.0, 0.0, replace::RGB, 0.0];
        assert_eq!(t.color(Some(0x8040_2010)), 0x8000_64C8);
    }

    #[test]
    fn sixteen_bit_colours_decode_like_the_fast_path() {
        // The decoders widen like color.rs; raw_color passes them on unchanged.
        let t = transform();
        for raw in [0x0000u16, 0xF00F, 0x1234, 0xFFFF, 0x8001] {
            assert_eq!(t.color(Some(rgba_from_4444(raw))), rgba_from_4444(raw));
            assert_eq!(t.color(Some(rgba_from_5551(raw))), rgba_from_5551(raw));
            assert_eq!(t.color(Some(rgba_from_5650(raw))), rgba_from_5650(raw));
        }
    }
}
