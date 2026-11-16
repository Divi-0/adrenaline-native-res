//! PSP colour formats, converted to 32-bit RGBA.
//!
//! All results use the layout `0xAABBGGRR`: in memory the bytes are R, G, B,
//! A. That is the PSP's own 8888 layout and also what vita2d textures and
//! vertex colours expect, so no further swizzling is needed.

/// GE colour formats used by textures, palettes and vertices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorFormat {
    /// 16 bit: red 5, green 6, blue 5, no alpha.
    Bgr5650,
    /// 16 bit: red 5, green 5, blue 5, alpha 1.
    Abgr5551,
    /// 16 bit: 4 bits each.
    Abgr4444,
    /// 32 bit: 8 bits each.
    Abgr8888,
}

impl ColorFormat {
    /// From the 2-bit format field used by palettes and textures.
    pub fn from_bits(bits: u32) -> ColorFormat {
        match bits & 3 {
            0 => ColorFormat::Bgr5650,
            1 => ColorFormat::Abgr5551,
            2 => ColorFormat::Abgr4444,
            _ => ColorFormat::Abgr8888,
        }
    }

    pub fn bytes_per_pixel(self) -> u32 {
        match self {
            ColorFormat::Abgr8888 => 4,
            _ => 2,
        }
    }

    /// Converts one raw pixel value in this format to RGBA.
    pub fn to_rgba(self, raw: u32) -> u32 {
        match self {
            ColorFormat::Bgr5650 => rgba_from_5650(raw as u16),
            ColorFormat::Abgr5551 => rgba_from_5551(raw as u16),
            ColorFormat::Abgr4444 => rgba_from_4444(raw as u16),
            ColorFormat::Abgr8888 => raw,
        }
    }
}

/// Widens an n-bit channel to 8 bits so that the maximum maps to 255.
fn expand(value: u32, bits: u32) -> u32 {
    let value = value & ((1 << bits) - 1);
    (value << (8 - bits)) | (value >> (2 * bits - 8))
}

fn pack(r: u32, g: u32, b: u32, a: u32) -> u32 {
    (a << 24) | (b << 16) | (g << 8) | r
}

pub fn rgba_from_5650(raw: u16) -> u32 {
    let raw = raw as u32;
    pack(expand(raw, 5), expand(raw >> 5, 6), expand(raw >> 11, 5), 0xFF)
}

pub fn rgba_from_5551(raw: u16) -> u32 {
    let raw = raw as u32;
    let alpha = if raw & 0x8000 != 0 { 0xFF } else { 0 };
    pack(expand(raw, 5), expand(raw >> 5, 5), expand(raw >> 10, 5), alpha)
}

pub fn rgba_from_4444(raw: u16) -> u32 {
    let raw = raw as u32;
    let channel = |shift: u32| ((raw >> shift) & 0xF) * 0x11;
    pack(channel(0), channel(4), channel(8), channel(12))
}

/// Same colour with alpha set to fully opaque.
pub fn opaque(rgba: u32) -> u32 {
    rgba | 0xFF00_0000
}

/// The colour `t` (0..1) of the way from `a` to `b`, channel by channel.
pub fn lerp_rgba(a: u32, b: u32, t: f32) -> u32 {
    (0..4).fold(0, |out, channel| {
        let shift = channel * 8;
        let (from, to) = (((a >> shift) & 0xFF) as f32, ((b >> shift) & 0xFF) as f32);
        // + 0.5 and truncating: rounding (no_std has no f32::round).
        out | (((from + (to - from) * t).clamp(0.0, 255.0) + 0.5) as u32) << shift
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn white_and_black_convert_exactly() {
        assert_eq!(rgba_from_5650(0xFFFF), 0xFFFF_FFFF);
        assert_eq!(rgba_from_5650(0x0000), 0xFF00_0000);
        assert_eq!(rgba_from_5551(0xFFFF), 0xFFFF_FFFF);
        assert_eq!(rgba_from_5551(0x7FFF), 0x00FF_FFFF);
        assert_eq!(rgba_from_4444(0xFFFF), 0xFFFF_FFFF);
    }

    #[test]
    fn channels_land_in_the_right_bytes() {
        // Pure red in each 16-bit format.
        assert_eq!(rgba_from_5650(0x001F), 0xFF00_00FF);
        assert_eq!(rgba_from_5551(0x801F), 0xFF00_00FF);
        assert_eq!(rgba_from_4444(0xF00F), 0xFF00_00FF);
        // Pure blue.
        assert_eq!(rgba_from_5650(0xF800), 0xFFFF_0000);
    }
}
