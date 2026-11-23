//! GPU textures as the cache sees them: opaque handles made by the host,
//! the Vita formats we decode into, and what they cost in memory.

use crate::frame::{texture_flags, texture_level_count, TextureDescriptor};
use crate::texture::{self, format};

use super::BACKGROUND_MIN_BYTES;

/// A texture the host made with `Host::create_texture`. The cache only
/// stores and compares it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuTexture {
    /// The host's handle (on the Vita: a `vita2d_texture` pointer).
    pub handle: usize,
    /// Where its memory is: at this byte offset in the texture arena
    /// (`arena`), or None: a memory block of its own.
    pub arena_offset: Option<u32>,
}

/// The Vita texture formats PSP textures are decoded into. Mostly RGBA
/// (4 bytes per texel), except where the Vita GPU can show the PSP's data
/// more directly:
/// - 4- and 8-bit palette formats: an 8-bit palette texture plus a
///   256-colour table, a quarter of the memory, decoding only copies
///   indices (the table is chosen per draw, see the palette tables);
/// - 16-bit formats: the Vita has the same bit layouts, half the memory,
///   decoding only unswizzles. Exact: the Vita's texture unit widens 4-,
///   5- and 6-bit channels as the GE does (k << 3 | k >> 2 for 5 bits,
///   k << 2 | k >> 4 for 6), for single texels and through its linear
///   filter alike: on the Vita, 0 differences against the GE's widening and
///   against the same texture as 8888 (debug "expandtest", "filtertest",
///   2026-11-23; a model sampling k / 31 instead was ~35 % of texels off).
///
/// The smooth filter needs RGBA: it changes individual texels (see
/// `texture::bleed_transparent_texels`).
///
/// The numbers are what the C side's `NativeResCreateTexture` expects.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuFormat {
    /// SCE_GXM_TEXTURE_FORMAT_U8U8U8U8_ABGR
    Rgba8888 = 0,
    /// SCE_GXM_TEXTURE_FORMAT_P8_ABGR
    Palette8 = 1,
    /// SCE_GXM_TEXTURE_FORMAT_U5U6U5_BGR (PSP BGR5650)
    Bgr565 = 2,
    /// SCE_GXM_TEXTURE_FORMAT_U1U5U5U5_ABGR (PSP ABGR5551)
    Abgr1555 = 3,
    /// SCE_GXM_TEXTURE_FORMAT_U4U4U4U4_ABGR (PSP ABGR4444)
    Abgr4444 = 4,
}

impl GpuFormat {
    /// The format a PSP texture is decoded into, with or without the smooth
    /// texture filter.
    pub fn for_texture(texture: &TextureDescriptor, smooth: bool) -> GpuFormat {
        if smooth {
            return GpuFormat::Rgba8888;
        }
        match texture.format {
            format::CLUT4 | format::CLUT8 => GpuFormat::Palette8,
            format::BGR5650 => GpuFormat::Bgr565,
            format::ABGR5551 => GpuFormat::Abgr1555,
            format::ABGR4444 => GpuFormat::Abgr4444,
            _ => GpuFormat::Rgba8888,
        }
    }

    /// The Vita format with the same bit layout as a PSP 16-bit texture
    /// (its data as it is: what the GPU unswizzle reads, the same as
    /// `for_texture` without the smooth filter); None for the other formats.
    pub fn raw_16_bit(texture: &TextureDescriptor) -> Option<GpuFormat> {
        match texture.format {
            format::BGR5650 => Some(GpuFormat::Bgr565),
            format::ABGR5551 => Some(GpuFormat::Abgr1555),
            format::ABGR4444 => Some(GpuFormat::Abgr4444),
            _ => None,
        }
    }

    pub fn is_16_bit(self) -> bool {
        matches!(self, GpuFormat::Bgr565 | GpuFormat::Abgr1555 | GpuFormat::Abgr4444)
    }
}

/// What a GPU texture is made as. Retired textures are only reused for a
/// texture of exactly the same shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureShape {
    pub width: u16,
    pub height: u16,
    pub format: GpuFormat,
    /// Made for the GPU to draw into (see [`can_unswizzle_on_gpu`]): its
    /// memory is GPU-writable and it has a colour surface. The CPU never
    /// writes such textures.
    pub render_target: bool,
    /// Mip levels, level 0 included (1: none; `texture_flags::LEVELS_SHIFT`):
    /// laid out as `texture::mip_level_offset` says.
    pub levels: u8,
}

impl TextureShape {
    /// A texture for `texture`'s size in `format`.
    pub fn new(texture: &TextureDescriptor, format: GpuFormat, render_target: bool) -> TextureShape {
        TextureShape { width: texture.width, height: texture.height, format, render_target, levels: texture_level_count(texture) as u8 }
    }

    /// The memory the host allocates for it: rows padded to 8 texels, whole
    /// 4 KiB pages, plus a page for the colour table of palette textures.
    pub fn bytes(&self) -> u32 {
        let texel_bytes = match self.format {
            GpuFormat::Palette8 => 1,
            format if format.is_16_bit() => 2,
            _ => 4,
        };
        // All levels: the offset a level after the last would have.
        let texels = texture::mip_level_offset(self.width as u32, self.height as u32, texel_bytes, self.levels.max(1) as u32);
        let pages = (texels + 0xFFF) & !0xFFF;
        if self.format == GpuFormat::Palette8 {
            pages + 0x1000
        } else {
            pages
        }
    }
}

/// Size of the render targets for PSP framebuffers (the Vita screen). The
/// GPU unswizzle draws into textures no bigger than that, because it shares
/// their depth surface.
pub const TARGET_WIDTH: u32 = 960;
pub const TARGET_HEIGHT: u32 = 544;

/// The GPU unswizzle reads the swizzled data in slices of 8 x 4096 texels
/// (SLICE_BYTES in nativeres_render.c, which must match).
pub const GPU_UNSWIZZLE_SLICE_BYTES: u32 = 8 * 4096 * 2;

/// True if the GPU can unswizzle this texture into its texture (instead of
/// the CPU, ~6 ms for 512 KiB): swizzled 16-bit textures of at least
/// `BACKGROUND_MIN_BYTES` whose data is whole slices, with rows as wide as
/// the texture (no spare buffer width), that fit the shared depth surface.
/// Texels made opaque on decoding (`IGNORE_ALPHA`) and the smooth filter
/// (it changes texels) stay with the CPU.
pub fn can_unswizzle_on_gpu(texture: &TextureDescriptor, smooth: bool) -> bool {
    let (width, height) = (texture.width as u32, texture.height as u32);
    let bytes = width * height * 2;
    !smooth
        && GpuFormat::raw_16_bit(texture).is_some()
        && texture.swizzled != 0
        && texture.buffer_width == width
        && texture.flags & texture_flags::IGNORE_ALPHA == 0
        && width % 32 == 0
        && height % 8 == 0
        && width <= TARGET_WIDTH
        && height <= TARGET_HEIGHT
        && bytes >= BACKGROUND_MIN_BYTES
        && bytes % GPU_UNSWIZZLE_SLICE_BYTES == 0
        && texture::data_size(texture) == Some(bytes)
        && texture_level_count(texture) == 1
}
