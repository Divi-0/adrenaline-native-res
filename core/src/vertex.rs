//! Reading vertices in the format described by the VERTEX_TYPE register.
//!
//! A PSP vertex is a packed struct whose fields depend on VERTEX_TYPE:
//! skinning weights, texture coordinates, colour, normal and position, in
//! that order, each aligned to its own size. The whole vertex is padded to
//! its largest alignment. Layout rules follow PPSSPP's
//! `VertexDecoder::SetVertexType`.

use crate::color::{rgba_from_4444, rgba_from_5551, rgba_from_5650};
use crate::memory::PspMemory;

/// The VERTEX_TYPE register, decoded on demand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VertexFormat(pub u32);

/// Index buffer element type (VERTEX_TYPE bits 11..13).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexFormat {
    None,
    U8,
    U16,
    U32,
}

impl IndexFormat {
    pub fn size(self) -> u32 {
        match self {
            IndexFormat::None => 0,
            IndexFormat::U8 => 1,
            IndexFormat::U16 => 2,
            IndexFormat::U32 => 4,
        }
    }
}

/// Where each field sits inside one vertex.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VertexLayout {
    pub size: u32,
    pub texcoord_offset: Option<u32>,
    pub color_offset: Option<u32>,
    pub normal_offset: Option<u32>,
    pub position_offset: u32,
}

/// One vertex as read from memory. In through mode (2D) positions are
/// screen pixels and texture coordinates are texels. Otherwise (3D) they are
/// the raw values; `VertexFormat::position_scale` and `texcoord_scale` turn
/// them into the GE's numbers.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vertex {
    pub x: f32,
    pub y: f32,
    /// Depth. Through mode (2D): the depth value itself, unsigned (0..65535
    /// for 16 bits); 3D: the raw signed value like x and y.
    pub z: f32,
    pub u: f32,
    pub v: f32,
    /// RGBA (0xAABBGGRR), if the format has a colour.
    pub color: Option<u32>,
    /// 3D only: skinning weights, one per bone matrix, already scaled
    /// (8 bits: 128 = 1.0, 16 bits: 32768 = 1.0); 0 for the bones the
    /// format has no weight for.
    pub weights: [f32; MAX_WEIGHTS],
}

/// A 3D vertex as the fast path reads it (`VertexFormat::decode_3d`): the
/// raw position and texture coordinates, not yet scaled.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Raw3d {
    pub position: [f32; 3],
    pub texcoord: [f32; 2],
    pub color: Option<u32>,
    /// Skinning weights, scaled; only the format's `weight_count` are set.
    pub weights: [f32; MAX_WEIGHTS],
}

/// Where the fields of a format `VertexFormat::decode_common` handles are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommonLayout {
    pub texcoord_offset: u32,
    /// Unused when the format has no colour.
    pub color_offset: u32,
    pub position_offset: u32,
    /// 0 = not skinned. The weights are always the first field.
    pub weight_count: u32,
}

/// The most skinning weights a vertex can have (one per bone matrix).
pub const MAX_WEIGHTS: usize = 8;

// Field sizes and alignments indexed by their 2- or 3-bit format value.
const TEXCOORD_SIZE: [u32; 4] = [0, 2, 4, 8];
const TEXCOORD_ALIGN: [u32; 4] = [0, 1, 2, 4];
const COLOR_SIZE: [u32; 8] = [0, 0, 0, 0, 2, 2, 2, 4];
const NORMAL_SIZE: [u32; 4] = [0, 3, 6, 12];
const NORMAL_ALIGN: [u32; 4] = [0, 1, 2, 4];
const POSITION_SIZE: [u32; 4] = [3, 3, 6, 12];
const POSITION_ALIGN: [u32; 4] = [1, 1, 2, 4];
const WEIGHT_SIZE: [u32; 4] = [0, 1, 2, 4];

/// The most words one vertex's fields can span: weights (32 bytes),
/// texture coordinates (8), colour (4), normal (12) and position (12) end
/// at byte 68 at most, plus up to 3 bytes because a vertex needn't start
/// on a word boundary.
const MAX_VERTEX_WORDS: usize = 18;

/// One vertex's bytes: words read from memory, and where the vertex starts
/// in them.
pub struct VertexBytes<'a> {
    words: &'a [u32],
    /// Where the vertex starts in `words`, in bytes (its address modulo 4
    /// when the words were read for this vertex alone).
    skew: u32,
}

impl<'a> VertexBytes<'a> {
    /// The vertex starting `skew` bytes into `words`.
    pub fn new(words: &'a [u32], skew: u32) -> Self {
        VertexBytes { words, skew }
    }

    /// True if the words reach `bytes` bytes past the vertex start. The
    /// readers below rely on it: every decoder checks it once for the
    /// furthest byte it reads (the position, the last field), instead of a
    /// bounds check per field (MTX Mototrax: ~35,000 vertices per frame).
    fn covers(&self, bytes: u32) -> bool {
        self.words.len() >= ((self.skew + bytes + 3) / 4) as usize
    }

    /// The word holding byte `at` (counted from the start of `words`).
    fn word(&self, at: u32) -> u32 {
        debug_assert!(((at / 4) as usize) < self.words.len());
        // Safe: only called with offsets below what `covers` was checked for.
        unsafe { *self.words.get_unchecked((at / 4) as usize) }
    }

    // The same results as PspMemory's read_u8/read_u16/read_u32 at the
    // vertex address + `offset`, including unaligned words being unreadable.
    fn u8(&self, offset: u32) -> u8 {
        let at = self.skew + offset;
        (self.word(at) >> ((at & 3) * 8)) as u8
    }

    fn u16(&self, offset: u32) -> u16 {
        let at = self.skew + offset;
        (self.word(at) >> ((at & 2) * 8)) as u16
    }

    fn u32(&self, offset: u32) -> Option<u32> {
        let at = self.skew + offset;
        (at % 4 == 0).then(|| self.word(at))
    }

    fn f32(&self, offset: u32) -> Option<f32> {
        self.u32(offset).map(f32::from_bits)
    }
}

fn align_up(value: u32, alignment: u32) -> u32 {
    if alignment <= 1 {
        value
    } else {
        (value + alignment - 1) & !(alignment - 1)
    }
}

impl VertexFormat {
    fn texcoord_format(self) -> usize {
        (self.0 & 3) as usize
    }

    fn color_format(self) -> usize {
        ((self.0 >> 2) & 7) as usize
    }

    fn normal_format(self) -> usize {
        ((self.0 >> 5) & 3) as usize
    }

    fn position_format(self) -> usize {
        ((self.0 >> 7) & 3) as usize
    }

    pub(crate) fn weight_format(self) -> usize {
        ((self.0 >> 9) & 3) as usize
    }

    /// Skinning weights are 8-bit (128 = 1.0), as `recorder::shared_weight` reads them.
    pub fn has_byte_weights(self) -> bool {
        self.has_weights() && self.weight_format() == 1
    }

    pub fn weight_count(self) -> u32 {
        ((self.0 >> 14) & 7) + 1
    }

    pub fn morph_count(self) -> u32 {
        ((self.0 >> 18) & 7) + 1
    }

    pub fn index_format(self) -> IndexFormat {
        match (self.0 >> 11) & 3 {
            0 => IndexFormat::None,
            1 => IndexFormat::U8,
            2 => IndexFormat::U16,
            _ => IndexFormat::U32,
        }
    }

    /// Through mode: positions are already screen coordinates (2D drawing).
    pub fn is_through(self) -> bool {
        self.0 & (1 << 23) != 0
    }

    /// Skinning weights: the position is blended from several bone matrices.
    pub fn has_weights(self) -> bool {
        self.weight_format() != 0
    }

    /// The format has normals (used by lighting and some texture mappings).
    pub fn has_normals(self) -> bool {
        self.normal_format() != 0
    }

    /// 3D draws: what raw normals are multiplied by (like positions).
    pub fn normal_scale(self) -> f32 {
        match self.normal_format() {
            1 => 1.0 / 128.0,
            2 => 1.0 / 32768.0,
            _ => 1.0,
        }
    }

    /// What raw weights are multiplied by (unsigned: 128 and 32768 are 1.0).
    pub fn weight_scale(self) -> f32 {
        match self.weight_format() {
            1 => 1.0 / 128.0,
            2 => 1.0 / 32768.0,
            _ => 1.0,
        }
    }

    /// 3D draws: what raw positions are multiplied by. 8- and 16-bit
    /// positions are fractions (-1..1), floats are taken as they are.
    pub fn position_scale(self) -> f32 {
        match self.position_format() {
            0 | 1 => 1.0 / 128.0,
            2 => 1.0 / 32768.0,
            _ => 1.0,
        }
    }

    /// 3D draws: what raw texture coordinates are multiplied by, before the
    /// texture scale and offset (8 and 16 bits: 128 and 32768 are 1.0).
    pub fn texcoord_scale(self) -> f32 {
        match self.texcoord_format() {
            1 => 1.0 / 128.0,
            2 => 1.0 / 32768.0,
            _ => 1.0,
        }
    }

    pub fn layout(self) -> VertexLayout {
        let mut size = 0;
        let mut biggest_alignment = 1;
        let mut place = |field_size: u32, alignment: u32| -> u32 {
            size = align_up(size, alignment);
            let offset = size;
            size += field_size;
            biggest_alignment = biggest_alignment.max(alignment);
            offset
        };

        let weights = self.weight_format();
        if weights != 0 {
            place(WEIGHT_SIZE[weights] * self.weight_count(), WEIGHT_SIZE[weights]);
        }

        let texcoord = self.texcoord_format();
        let texcoord_offset = (texcoord != 0).then(|| place(TEXCOORD_SIZE[texcoord], TEXCOORD_ALIGN[texcoord]));

        let color = self.color_format();
        let color_offset = (COLOR_SIZE[color] != 0).then(|| place(COLOR_SIZE[color], COLOR_SIZE[color]));

        let normal = self.normal_format();
        let normal_offset = (normal != 0).then(|| place(NORMAL_SIZE[normal], NORMAL_ALIGN[normal]));

        let position = self.position_format();
        let position_offset = place(POSITION_SIZE[position], POSITION_ALIGN[position]);

        // Morph targets are whole extra vertices stored back to back; we
        // only read the first one.
        let one_vertex = align_up(size, biggest_alignment);
        VertexLayout { size: one_vertex * self.morph_count(), texcoord_offset, color_offset, normal_offset, position_offset }
    }

    /// How many bytes of a vertex `read` uses: up to the end of the position,
    /// the last field (padding after it is never read).
    pub fn bytes_used(self, layout: &VertexLayout) -> u32 {
        layout.position_offset + POSITION_SIZE[self.position_format()]
    }

    /// Reads the vertex at `address`. Only the through-mode (2D)
    /// interpretation of positions and texture coordinates is implemented.
    pub fn read(self, memory: &mut impl PspMemory, address: u32, layout: &VertexLayout) -> Option<Vertex> {
        let skew = address & 3;
        let count = ((skew + self.bytes_used(layout) + 3) / 4) as usize;
        let mut words = [0; MAX_VERTEX_WORDS];
        memory.read_words(address & !3, words.get_mut(..count)?)?;
        self.decode(&VertexBytes::new(&words[..count], skew), layout)
    }

    /// Decodes a vertex from its bytes. `None` if a float or 32-bit colour
    /// isn't word-aligned (unreadable, as for `PspMemory::read_u32`) or the
    /// bytes end too early.
    pub fn decode(self, bytes: &VertexBytes, layout: &VertexLayout) -> Option<Vertex> {
        if !bytes.covers(self.bytes_used(layout)) {
            return None;
        }
        // The position is the last field.
        let position = layout.position_offset;
        let (x, y) = match self.position_format() {
            0 | 1 => (bytes.u8(position) as i8 as f32, bytes.u8(position + 1) as i8 as f32),
            2 => (bytes.u16(position) as i16 as f32, bytes.u16(position + 2) as i16 as f32),
            _ => (bytes.f32(position)?, bytes.f32(position + 4)?),
        };
        // Through mode: z is unsigned (PPSSPP's Step_PosS16Through).
        let through = self.is_through();
        let z = match self.position_format() {
            0 | 1 if through => bytes.u8(position + 2) as f32,
            0 | 1 => bytes.u8(position + 2) as i8 as f32,
            2 if through => bytes.u16(position + 4) as f32,
            2 => bytes.u16(position + 4) as i16 as f32,
            _ => bytes.f32(position + 8)?,
        };

        let [u, v] = self.read_texcoord(bytes, layout)?;
        let color = self.read_color(bytes, layout)?;
        let mut vertex = Vertex { x, y, z, u, v, color, ..Vertex::default() };
        if !self.is_through() {
            self.decode_3d_extras(bytes, layout, &mut vertex)?;
        }
        Some(vertex)
    }

    /// Raw texture coordinates (0, 0 if the format has none).
    #[inline(always)]
    fn read_texcoord(self, bytes: &VertexBytes, layout: &VertexLayout) -> Option<[f32; 2]> {
        Some(match layout.texcoord_offset {
            None => [0.0, 0.0],
            Some(at) => match self.texcoord_format() {
                1 => [bytes.u8(at) as f32, bytes.u8(at + 1) as f32],
                2 => [bytes.u16(at) as f32, bytes.u16(at + 2) as f32],
                _ => [bytes.f32(at)?, bytes.f32(at + 4)?],
            },
        })
    }

    /// The colour as RGBA, `Some(None)` if the format has none.
    #[inline(always)]
    fn read_color(self, bytes: &VertexBytes, layout: &VertexLayout) -> Option<Option<u32>> {
        Some(match layout.color_offset {
            None => None,
            Some(at) => Some(match self.color_format() {
                4 => rgba_from_5650(bytes.u16(at)),
                5 => rgba_from_5551(bytes.u16(at)),
                6 => rgba_from_4444(bytes.u16(at)),
                _ => bytes.u32(at)?,
            }),
        })
    }

    /// The fields a 3D vertex needs on the fast path (`Transform::frame_vertex`):
    /// position, texture coordinates and colour as stored, and the skinning
    /// weights. Only for formats without through mode.
    #[inline]
    pub fn decode_3d(self, bytes: &VertexBytes, layout: &VertexLayout) -> Option<Raw3d> {
        if !bytes.covers(self.bytes_used(layout)) {
            return None;
        }
        let at = layout.position_offset;
        let position = match self.position_format() {
            0 | 1 => [bytes.u8(at) as i8 as f32, bytes.u8(at + 1) as i8 as f32, bytes.u8(at + 2) as i8 as f32],
            2 => [bytes.u16(at) as i16 as f32, bytes.u16(at + 2) as i16 as f32, bytes.u16(at + 4) as i16 as f32],
            _ => [bytes.f32(at)?, bytes.f32(at + 4)?, bytes.f32(at + 8)?],
        };
        let mut weights = [0.0; MAX_WEIGHTS];
        if self.has_weights() {
            let count = (self.weight_count() as usize).min(MAX_WEIGHTS);
            for (index, weight) in weights[..count].iter_mut().enumerate() {
                let index = index as u32;
                *weight = match self.weight_format() {
                    1 => bytes.u8(index) as f32 * (1.0 / 128.0),
                    2 => bytes.u16(2 * index) as f32 * (1.0 / 32768.0),
                    _ => bytes.f32(4 * index)?,
                };
            }
        }
        Some(Raw3d {
            position,
            texcoord: self.read_texcoord(bytes, layout)?,
            color: self.read_color(bytes, layout)?,
            weights,
        })
    }

    /// The fields of formats `decode_common` handles, or `None` for the
    /// others: 16-bit positions, 16-bit texture coordinates, a 16-bit
    /// colour or none, 8-bit skinning weights or none (normals are skipped).
    /// (8888 colours stay with `decode_3d`, which rejects an unaligned one.) MTX
    /// Mototrax draws ~95% of its 3D vertices in such formats (0x11a and
    /// the 0x33a family).
    pub fn common_layout(self, layout: &VertexLayout) -> Option<CommonLayout> {
        let texcoord_offset = layout.texcoord_offset?;
        let usable = !self.is_through()
            && self.position_format() == 2
            && self.texcoord_format() == 2
            && matches!(self.color_format(), 0 | 4 | 5 | 6)
            && (!self.has_weights() || self.weight_format() == 1);
        usable.then(|| CommonLayout {
            texcoord_offset,
            color_offset: layout.color_offset.unwrap_or(0),
            position_offset: layout.position_offset,
            weight_count: if self.has_weights() { self.weight_count().min(MAX_WEIGHTS as u32) } else { 0 },
        })
    }

    /// `decode_3d` for a format with a `CommonLayout`, its colour format
    /// (VERTEX_TYPE bits 2..4) fixed at compile time: no per-field format
    /// checks, so the vertex loop compiles to straight code (PPSSPP JITs a
    /// decoder per vertex format for the same reason, VertexDecoderArm.cpp).
    /// `bytes` must cover the vertex (`covers` is not checked here).
    #[inline(always)]
    pub fn decode_common<const COLOR: u32>(bytes: &VertexBytes, layout: &CommonLayout) -> Raw3d {
        let at = layout.position_offset;
        let position = [bytes.u16(at) as i16 as f32, bytes.u16(at + 2) as i16 as f32, bytes.u16(at + 4) as i16 as f32];
        let at = layout.texcoord_offset;
        let texcoord = [bytes.u16(at) as f32, bytes.u16(at + 2) as f32];
        let at = layout.color_offset;
        let color = match COLOR {
            4 => Some(rgba_from_5650(bytes.u16(at))),
            5 => Some(rgba_from_5551(bytes.u16(at))),
            6 => Some(rgba_from_4444(bytes.u16(at))),
            _ => None,
        };
        let mut weights = [0.0; MAX_WEIGHTS];
        for (index, weight) in weights[..layout.weight_count as usize].iter_mut().enumerate() {
            *weight = bytes.u8(index as u32) as f32 * (1.0 / 128.0);
        }
        Raw3d { position, texcoord, color, weights }
    }

    /// Whether the format's normals are 8-bit (n / 128; MTX's lit riders),
    /// which `s8_normal` reads.
    pub fn has_s8_normals(self) -> bool {
        self.normal_format() == 1
    }

    /// The bytes of an 8-bit normal at `normal_offset` (`VertexLayout`), as
    /// `decode_common` reads its fields: `bytes` must cover the vertex.
    #[inline(always)]
    pub fn s8_normal(bytes: &VertexBytes, normal_offset: u32) -> [u8; 3] {
        [bytes.u8(normal_offset), bytes.u8(normal_offset + 1), bytes.u8(normal_offset + 2)]
    }

    /// Position, texture coordinates and colour of an unskinned vertex in
    /// a `CommonLayout`, each field read with one load (`decode_common`
    /// takes each 16-bit field out of its word with shifts; PPSSPP's JIT
    /// loads each field with one instruction too). Gives what
    /// `decode_common` gives for a vertex that starts at an even address.
    ///
    /// # Safety
    /// `vertex` must point to the vertex's first byte, at an even address,
    /// with all of its fields readable (up to `bytes_used`).
    #[inline(always)]
    pub unsafe fn read_common_unskinned<const COLOR: u32>(
        vertex: *const u8,
        layout: &CommonLayout,
    ) -> ([f32; 3], [f32; 2], Option<u32>) {
        // Little-endian like the PSP, so a 16-bit load is the PSP's halfword.
        let u16_at = |offset: u32| vertex.add(offset as usize).cast::<u16>().read_unaligned();
        let at = layout.position_offset;
        let position = [u16_at(at) as i16 as f32, u16_at(at + 2) as i16 as f32, u16_at(at + 4) as i16 as f32];
        let at = layout.texcoord_offset;
        let texcoord = [u16_at(at) as f32, u16_at(at + 2) as f32];
        let at = layout.color_offset;
        let color = match COLOR {
            4 => Some(rgba_from_5650(u16_at(at))),
            5 => Some(rgba_from_5551(u16_at(at))),
            6 => Some(rgba_from_4444(u16_at(at))),
            _ => None,
        };
        (position, texcoord, color)
    }

    /// The colour `read_common_unskinned` gives, with one load.
    ///
    /// # Safety
    /// As `read_common_unskinned`.
    #[inline(always)]
    pub unsafe fn read_common_color<const COLOR: u32>(vertex: *const u8, layout: &CommonLayout) -> Option<u32> {
        let at = || vertex.add(layout.color_offset as usize).cast::<u16>().read_unaligned();
        match COLOR {
            4 => Some(rgba_from_5650(at())),
            5 => Some(rgba_from_5551(at())),
            6 => Some(rgba_from_4444(at())),
            _ => None,
        }
    }

    /// The raw 8-bit skinning weight of a single-weight vertex (the first
    /// byte: weights are the first field); 128 = 1.0.
    ///
    /// # Safety
    /// As `read_common_unskinned`.
    #[inline(always)]
    pub unsafe fn read_first_weight(vertex: *const u8) -> u8 {
        vertex.read()
    }

    /// VERTEX_TYPE bits 2..4: 0 = no colour, 4 = 565, 5 = 5551, 6 = 4444, 7 = 8888.
    /// The vertices have a colour of their own.
    pub fn has_color(self) -> bool {
        self.color_format() >= 4
    }

    pub fn color_format_bits(self) -> u32 {
        self.color_format() as u32
    }

    /// Reads the field only 3D draws use: skinning weights (always the
    /// first field). Normals are only read where lighting needs them
    /// (`decode_normal`): MTX Mototrax has ~20,000 vertices per frame.
    fn decode_3d_extras(self, bytes: &VertexBytes, layout: &VertexLayout, vertex: &mut Vertex) -> Option<()> {
        let _ = layout;
        if self.has_weights() {
            let scale = self.weight_scale();
            let count = (self.weight_count() as usize).min(MAX_WEIGHTS);
            for (index, weight) in vertex.weights[..count].iter_mut().enumerate() {
                let raw = match self.weight_format() {
                    1 => bytes.u8(index as u32) as f32,
                    2 => bytes.u16(2 * index as u32) as f32,
                    _ => bytes.f32(4 * index as u32)?,
                };
                *weight = raw * scale;
            }
        }
        Some(())
    }

    /// The vertex's normal, scaled (8- and 16-bit normals are fractions like
    /// positions); `None` if the format has none or it isn't readable.
    pub fn decode_normal(self, bytes: &VertexBytes, layout: &VertexLayout) -> Option<[f32; 3]> {
        let at = layout.normal_offset?;
        if !bytes.covers(self.bytes_used(layout)) {
            return None;
        }
        let scale = self.normal_scale();
        let raw = match self.normal_format() {
            1 => [bytes.u8(at) as i8 as f32, bytes.u8(at + 1) as i8 as f32, bytes.u8(at + 2) as i8 as f32],
            2 => [bytes.u16(at) as i16 as f32, bytes.u16(at + 2) as i16 as f32, bytes.u16(at + 4) as i16 as f32],
            _ => [bytes.f32(at)?, bytes.f32(at + 4)?, bytes.f32(at + 8)?],
        };
        Some(raw.map(|value| value * scale))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::test_memory::TestMemory;

    #[test]
    fn decode_common_gives_what_decode_3d_gives() {
        // MTX's formats (0x11a, the 0x33a family with 1..8 weights), with
        // every 16-bit colour and none, at every byte offset of a word.
        let mut formats = vec![0x11a, 0x102];
        for weights in 0..8u32 {
            formats.push(0x33a | weights << 14);
        }
        for colour in [0u32, 4, 5, 6] {
            formats.push((0x11a & !(7 << 2)) | colour << 2);
        }
        // Small values in every byte: no NaNs, all patterns of signs.
        let words: Vec<u32> = (0..40u32).map(|i| i.wrapping_mul(0x9E37_79B1) ^ 0x8040_C020).collect();
        let mut checked = 0;
        for format in formats.into_iter().map(VertexFormat) {
            let layout = format.layout();
            let Some(common) = format.common_layout(&layout) else {
                continue;
            };
            for skew in [0, 2, 4, 6] {
                let bytes = VertexBytes::new(&words, skew);
                let expected = format.decode_3d(&bytes, &layout).expect("readable");
                let decoded = match format.color_format_bits() {
                    4 => VertexFormat::decode_common::<4>(&bytes, &common),
                    5 => VertexFormat::decode_common::<5>(&bytes, &common),
                    6 => VertexFormat::decode_common::<6>(&bytes, &common),
                    _ => VertexFormat::decode_common::<0>(&bytes, &common),
                };
                assert_eq!(decoded, expected, "format {:06x} skew {skew}", format.0);
                checked += 1;
            }
        }
        assert_eq!(checked, 4 * 14, "every format above is a common one");

        // The direct reads give the same for unskinned and single-weight
        // formats at even starts.
        let mut direct = 0;
        for format in [0x11au32, 0x102, 0x11a & !(7 << 2), 0x11a & !(7 << 2) | 4 << 2, 0x11a & !(7 << 2) | 5 << 2, 0x33a].map(VertexFormat) {
            let layout = format.layout();
            let common = format.common_layout(&layout).unwrap();
            for skew in [0u32, 2, 4, 6] {
                let bytes = VertexBytes::new(&words, skew);
                let expected = format.decode_3d(&bytes, &layout).unwrap();
                // Safe: `words` holds 160 bytes, the vertex at most 12 from `skew`.
                let vertex = unsafe { (words.as_ptr() as *const u8).add(skew as usize) };
                let read = unsafe {
                    match format.color_format_bits() {
                        4 => VertexFormat::read_common_unskinned::<4>(vertex, &common),
                        5 => VertexFormat::read_common_unskinned::<5>(vertex, &common),
                        6 => VertexFormat::read_common_unskinned::<6>(vertex, &common),
                        _ => VertexFormat::read_common_unskinned::<0>(vertex, &common),
                    }
                };
                assert_eq!(read, (expected.position, expected.texcoord, expected.color), "format {:06x} skew {skew}", format.0);
                if format.has_weights() {
                    let weight = unsafe { VertexFormat::read_first_weight(vertex) };
                    assert_eq!(weight as f32 * (1.0 / 128.0), expected.weights[0]);
                }
                direct += 1;
            }
        }
        assert_eq!(direct, 6 * 4);
    }

    /// Through mode, u16 texcoords, 8888 colour, s16 position: the typical
    /// 2D sprite vertex from sceGu.
    const SPRITE_FORMAT: u32 = (1 << 23) | 2 | (7 << 2) | (2 << 7);

    #[test]
    fn layout_of_a_typical_2d_vertex() {
        let layout = VertexFormat(SPRITE_FORMAT).layout();
        assert_eq!(layout.texcoord_offset, Some(0));
        assert_eq!(layout.color_offset, Some(4));
        assert_eq!(layout.position_offset, 8);
        // 4 + 4 + 6 = 14, padded to the 4-byte colour alignment.
        assert_eq!(layout.size, 16);
    }

    #[test]
    fn reads_a_typical_2d_vertex() {
        let mut memory = TestMemory::default();
        let address = 0x0890_0000;
        // u=16, v=32 | colour | x=100, y=-2 | z=0 + padding
        memory.write(address, &[(32 << 16) | 16, 0xFF11_2233, 0xFFFE_0064, 0]);
        let format = VertexFormat(SPRITE_FORMAT);
        let vertex = format.read(&mut memory, address, &format.layout()).unwrap();
        assert_eq!((vertex.u, vertex.v), (16.0, 32.0));
        assert_eq!(vertex.color, Some(0xFF11_2233));
        assert_eq!((vertex.x, vertex.y), (100.0, -2.0));
    }

    #[test]
    fn reads_a_3d_vertex() {
        let mut memory = TestMemory::default();
        let address = 0x0890_0000;
        // u16 texcoords, no colour, s16 position, not through mode:
        // u=0x4000, v=0x8000 | x=-0x4000, y=0x2000 | z=-1
        let format = VertexFormat(2 | (2 << 7));
        memory.write(address, &[0x8000_4000, 0x2000_C000, 0x0000_FFFF]);
        let vertex = format.read(&mut memory, address, &format.layout()).unwrap();
        assert_eq!((vertex.x, vertex.y, vertex.z), (-16384.0, 8192.0, -1.0));
        assert_eq!(vertex.x * format.position_scale(), -0.5);
        assert_eq!((vertex.u * format.texcoord_scale(), vertex.v * format.texcoord_scale()), (0.5, 1.0));
    }

    #[test]
    fn reads_weights_and_normal_of_a_skinned_vertex() {
        let mut memory = TestMemory::default();
        let address = 0x0890_0000;
        // 2 u8 weights, s8 normal, s16 position, not through mode (MTX's
        // 0x433a without texture coordinates and colour):
        // weights 128, 64 | normal 0, 127, -128 | (pad) x = 0x4000, y = 0, z = -0x4000
        let format = VertexFormat((1 << 9) | (1 << 14) | (1 << 5) | (2 << 7));
        let layout = format.layout();
        assert_eq!((layout.normal_offset, layout.position_offset, layout.size), (Some(2), 6, 12));
        memory.write(address, &[0x7F00_4080, 0x4000_0080, 0xC000_0000]);
        let vertex = format.read(&mut memory, address, &layout).unwrap();
        assert_eq!(&vertex.weights[..3], &[1.0, 0.5, 0.0]);
        let words = [0x7F00_4080, 0x4000_0080, 0xC000_0000];
        assert_eq!(format.decode_normal(&VertexBytes::new(&words, 0), &layout), Some([0.0, 127.0 / 128.0, -1.0]));
        assert_eq!((vertex.x, vertex.y, vertex.z), (16384.0, 0.0, -16384.0));
    }

    #[test]
    fn float_position_only() {
        let layout = VertexFormat(3 << 7).layout();
        assert_eq!(layout.size, 12);
        assert_eq!(layout.texcoord_offset, None);
    }

    /// The field-by-field reading `read` replaced (one memory read per
    /// field), kept as the reference.
    fn read_field_by_field(format: VertexFormat, memory: &mut TestMemory, address: u32, layout: &VertexLayout) -> Option<Vertex> {
        let position = address + layout.position_offset;
        let (x, y, z) = match format.position_format() {
            0 | 1 => (memory.read_u8(position)? as i8 as f32, memory.read_u8(position + 1)? as i8 as f32, memory.read_u8(position + 2)? as f32),
            2 => (memory.read_u16(position)? as i16 as f32, memory.read_u16(position + 2)? as i16 as f32, memory.read_u16(position + 4)? as f32),
            _ => (memory.read_f32(position)?, memory.read_f32(position + 4)?, memory.read_f32(position + 8)?),
        };
        let (u, v) = match layout.texcoord_offset {
            None => (0.0, 0.0),
            Some(offset) => match format.texcoord_format() {
                1 => (memory.read_u8(address + offset)? as f32, memory.read_u8(address + offset + 1)? as f32),
                2 => (memory.read_u16(address + offset)? as f32, memory.read_u16(address + offset + 2)? as f32),
                _ => (memory.read_f32(address + offset)?, memory.read_f32(address + offset + 4)?),
            },
        };
        let color = match layout.color_offset {
            None => None,
            Some(offset) => Some(match format.color_format() {
                4 => rgba_from_5650(memory.read_u16(address + offset)?),
                5 => rgba_from_5551(memory.read_u16(address + offset)?),
                6 => rgba_from_4444(memory.read_u16(address + offset)?),
                _ => memory.read_u32(address + offset)?,
            }),
        };
        Some(Vertex { x, y, z, u, v, color, ..Vertex::default() })
    }

    #[test]
    fn reading_whole_vertices_matches_field_by_field_reading() {
        let mut memory = TestMemory::default();
        let base = 0x0890_0000;
        // Varied bytes, but floats that compare equal (no NaNs): small values in every word.
        let words: Vec<u32> = (0..64u32).map(|i| (i * 0x0103_0507) & 0x3F7F_7F7F).collect();
        memory.write(base, &words);
        let mut compared = 0;
        for texcoord in 0..4 {
            for color in [0, 4, 5, 6, 7] {
                for position in 1..4 {
                    for weights in [0, 1 << 9, (2 << 9) | (3 << 14), (3 << 9) | (7 << 14)] {
                        let format = VertexFormat((1 << 23) | texcoord | (color << 2) | (position << 7) | weights);
                        let layout = format.layout();
                        for skew in 0..4 {
                            let address = base + 8 + skew;
                            let whole = format.read(&mut memory, address, &layout);
                            let reference = read_field_by_field(format, &mut memory, address, &layout);
                            assert_eq!(whole, reference, "format {:#x} at {:#x}", format.0, address);
                            compared += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(compared, 4 * 5 * 3 * 4 * 4);
    }
}
