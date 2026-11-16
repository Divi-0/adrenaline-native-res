//! PSP texture decoding to RGBA8888.
//!
//! Textures can be direct colour (16 or 32 bit per pixel) or palette
//! ("CLUT") formats where each pixel is a 4/8/16/32-bit index into a colour
//! table. They may also be *swizzled*: stored in blocks of 16 bytes x 8 rows
//! instead of row by row, which is faster for the PSP's texture cache.
//!
//! Decoding follows PPSSPP's `GPU/Common/TextureDecoder.cpp`.

use crate::color::ColorFormat;
use crate::frame::{texture_flags, TextureDescriptor, TextureLevels};
use crate::ge_state::GeState;
use crate::command::op;

/// TEXTURE_FORMAT values.
pub mod format {
    pub const BGR5650: u8 = 0;
    pub const ABGR5551: u8 = 1;
    pub const ABGR4444: u8 = 2;
    pub const ABGR8888: u8 = 3;
    pub const CLUT4: u8 = 4;
    pub const CLUT8: u8 = 5;
    pub const CLUT16: u8 = 6;
    pub const CLUT32: u8 = 7;
}

/// Largest texture the PSP supports.
pub const MAX_SIZE: u16 = 512;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeError {
    /// DXT compressed or invalid format.
    UnsupportedFormat,
    /// Larger than 512x512 or zero-sized.
    BadSize,
    /// Pixel data outside readable memory.
    BadAddress,
    /// The output buffer is too small.
    OutputTooSmall,
}

impl DecodeError {
    /// The number the C side and the render log use for it (0 is success).
    pub fn code(self) -> u32 {
        match self {
            DecodeError::UnsupportedFormat => 1,
            DecodeError::BadSize => 2,
            DecodeError::BadAddress => 3,
            DecodeError::OutputTooSmall => 4,
        }
    }
}

fn bits_per_pixel(format: u8) -> Option<u32> {
    match format {
        format::BGR5650 | format::ABGR5551 | format::ABGR4444 | format::CLUT16 => Some(16),
        format::ABGR8888 | format::CLUT32 => Some(32),
        format::CLUT4 => Some(4),
        format::CLUT8 => Some(8),
        _ => None,
    }
}

pub fn is_palette_format(format: u8) -> bool {
    (format::CLUT4..=format::CLUT32).contains(&format)
}

/// Describes the texture level 0 the GE would currently sample from.
/// Palette fields are left empty; the caller fills them in.
/// Menu option "NativeRes Distant Textures" (Adrenaline's config byte
/// `nativepsp_mip_mode`, set by the display every frame): 0 "Sharp" (level 0
/// only, the default), 1 "Blurry (like PSP)" (the game's mip levels, one
/// level blurrier than the GPU would pick at our 2x size: the PSP's choice),
/// 2 "Mipmapped" (the game's levels as the GPU picks them at our size).
pub static MIP_MODE: core::sync::atomic::AtomicU8 = core::sync::atomic::AtomicU8::new(0);

/// How a draw samples its texture's mip levels (`describe_levels`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MipLevels {
    pub levels: TextureLevels,
    /// Levels kept, level 0 included (2..=8).
    pub count: u32,
    /// The GE blends between levels ("mip linear").
    pub linear: bool,
    /// Bias in 1/8 levels (`draw_flags::MIP_BIAS_SHIFT`).
    pub bias_eighths: i8,
}

/// The current texture's mip levels, if the draw uses them and the menu
/// option wants them (`MIP_MODE`). As PPSSPP decides (TextureCacheCommon.cpp
/// sampler key): levels only with a mip minification filter (TEXTURE_FILTER
/// bits 0..3 >= 4) and a highest level above 0 (TEXTURE_MODE bits 16..19);
/// "auto" level mode (TEXLEVEL 0xC8 bits 0..1 = 0) biases by its offset
/// (bits 16..24, signed, 1/16 levels), "slope" (2) not, "const" (1) with an
/// offset of 0 or less means level 0 only. Each level must be half the size
/// of the one before (GPU mip chains); levels after one that isn't are left
/// out.
pub fn describe_levels(ge: &GeState, base: &TextureDescriptor) -> Option<MipLevels> {
    describe_levels_in_mode(ge, base, MIP_MODE.load(core::sync::atomic::Ordering::Relaxed))
}

/// `describe_levels` for menu option value `mode` (tests don't share the
/// global).
fn describe_levels_in_mode(ge: &GeState, base: &TextureDescriptor, mode: u8) -> Option<MipLevels> {
    let min_filter = ge.register(op::TEXTURE_FILTER) & 7;
    let max_level = (ge.register(op::TEXTURE_MODE) >> 16) & 7;
    if mode == 0 || min_filter & 4 == 0 || max_level == 0 {
        return None;
    }
    let level_register = ge.register(op::TEXTURE_LEVEL);
    let level_mode = level_register & 3;
    let offset = ((level_register >> 16) & 0xFF) as u8 as i8 as i32; // 1/16 levels
    if level_mode == 1 && offset <= 0 {
        return None;
    }
    let log2 = |size: u32| size.trailing_zeros().min(15) as u8;
    let mut levels = TextureLevels::default();
    levels.address[0] = base.address;
    levels.buffer_width[0] = base.buffer_width as u16;
    levels.log2_size[0] = log2(base.width as u32) | (log2(base.height as u32) << 4);
    let mut count = 1;
    for level in 1..=max_level as usize {
        let address_register = ge.register(op::TEXTURE_ADDRESS_0 + level as u8);
        let width_register = ge.register(op::TEXTURE_BUFFER_WIDTH_0 + level as u8);
        let size = ge.register(op::TEXTURE_SIZE_0 + level as u8);
        let (log2_width, log2_height) = ((size & 0xF) as u8, ((size >> 8) & 0xF) as u8);
        let previous = levels.log2_size[level - 1];
        let halves = log2_width == (previous & 0xF).saturating_sub(1) && log2_height == (previous >> 4).saturating_sub(1);
        if !halves {
            break;
        }
        levels.address[level] = crate::memory::physical((address_register & 0x00FF_FFF0) | ((width_register & 0x000F_0000) << 8));
        levels.buffer_width[level] = (width_register & 0x7FF) as u16;
        levels.log2_size[level] = log2_width | (log2_height << 4);
        count += 1;
    }
    if count < 2 {
        return None;
    }
    // Auto: the TEXLEVEL offset (1/16 -> 1/8 levels); const: the offset too
    // (as a bias, an approximation of a fixed level); slope: none (PPSSPP:
    // "not well understood"). Like the PSP: one level more at our 2x size.
    let game_bias = if level_mode == 2 { 0 } else { offset / 2 };
    let psp_like = if mode == 1 { 8 } else { 0 };
    Some(MipLevels { levels, count, linear: min_filter & 2 != 0, bias_eighths: (game_bias + psp_like).clamp(-31, 32) as i8 })
}

/// Level `level` of a texture with mip levels as a texture of its own: its
/// address, row stride and size (for staging and decoding it).
pub fn level_descriptor(base: &TextureDescriptor, levels: &TextureLevels, level: usize) -> TextureDescriptor {
    TextureDescriptor {
        address: levels.address[level],
        buffer_width: levels.buffer_width[level] as u32,
        width: 1 << (levels.log2_size[level] & 0xF),
        height: 1 << (levels.log2_size[level] >> 4),
        flags: base.flags & !texture_flags::LEVELS_MASK,
        ..*base
    }
}

/// Byte offset of mip level `level` in a GPU texture of `width` x `height`
/// texels of `texel_bytes` each: GXM's layout for linear textures (measured
/// on the Vita): the levels one
/// after another, each level's rows rounded up to 8 texels.
pub fn mip_level_offset(width: u32, height: u32, texel_bytes: u32, level: u32) -> u32 {
    let (mut w, mut h, mut offset) = (width, height, 0);
    for _ in 0..level {
        offset += w.div_ceil(8) * 8 * h * texel_bytes;
        w = (w / 2).max(1);
        h = (h / 2).max(1);
    }
    offset
}

pub fn describe_current_texture(ge: &GeState) -> TextureDescriptor {
    let size = ge.register(op::TEXTURE_SIZE_0);
    TextureDescriptor {
        address: ge.texture_address(),
        buffer_width: ge.register(op::TEXTURE_BUFFER_WIDTH_0) & 0x7FF,
        width: 1 << (size & 0xF).min(9),
        height: 1 << ((size >> 8) & 0xF).min(9),
        format: ge.texture_format() as u8,
        swizzled: (ge.register(op::TEXTURE_MODE) & 1) as u8,
        flags: if ge.register(op::TEXTURE_FUNCTION) & (1 << 8) == 0 { texture_flags::IGNORE_ALPHA } else { 0 },
        clut_format: if is_palette_format(ge.texture_format() as u8) { ge.register(op::CLUT_FORMAT) } else { 0 },
        palette_offset: 0,
        palette_word_count: 0,
        palette_hash: 0,
    }
}

/// Bytes per row of the texture as stored in memory. The GE always steps
/// rows by the buffer width, even when the texture is declared wider
/// (Final Fantasy's title logo: 512 pixels wide, buffer width 480).
/// Swizzled textures are stored in 16-byte blocks, so a row is at least 16.
fn row_bytes(texture: &TextureDescriptor, bits: u32) -> u32 {
    let stride = if texture.buffer_width != 0 { texture.buffer_width } else { texture.width as u32 };
    let bytes = stride * bits / 8;
    if texture.swizzled != 0 {
        bytes.max(16)
    } else {
        bytes
    }
}

/// Byte offset of byte `x_byte` in row `y`, honouring swizzling. Texel by
/// texel addressing, as the decoders did before `read_row`: kept as the
/// reference the tests compare against.
#[cfg(test)]
fn byte_offset(texture: &TextureDescriptor, row_bytes: u32, x_byte: u32, y: u32) -> u32 {
    if texture.swizzled == 0 {
        return y * row_bytes + x_byte;
    }
    // Swizzled: 16-byte-wide, 8-row-tall blocks, stored block row by block row.
    let blocks_per_row = row_bytes / 16;
    let block = (y / 8) * blocks_per_row + x_byte / 16;
    block * 128 + (y % 8) * 16 + x_byte % 16
}

/// Words of a texture's data read for its quick sample (`quick_sample`).
pub const QUICK_SAMPLES: usize = 8;

/// The word positions (from the texture's address) the quick sample reads:
/// the middle of each eighth of its data. `None` if it has no data.
pub fn quick_sample_positions(texture: &TextureDescriptor) -> Option<[usize; QUICK_SAMPLES]> {
    let words = (data_size(texture)? / 4) as usize;
    if words == 0 {
        return None;
    }
    Some(core::array::from_fn(|sample| (2 * sample + 1) * words / (2 * QUICK_SAMPLES)))
}

/// A few of a texture's data words folded into one value (never 0, which
/// means "not sampled"): the capture takes it while the GE draws the frame
/// (`Frame::texture_samples`), the texture cache from the data it decodes;
/// different values mean the frame's pixels are not the cached ones.
/// MTX's menus load the next screen's pictures into the same memory; the
/// display reads texture data later than the GE did, so only a sample taken
/// with the frame tells which data the frame drew with.
pub fn quick_sample(words: &[u32; QUICK_SAMPLES]) -> u32 {
    words.iter().fold(0x811C_9DC5u32, |hash, &word| (hash ^ word).wrapping_mul(0x0100_0193)) | 1
}

/// How many bytes the texture's pixels occupy, starting at its address.
/// `None` for unsupported formats.
pub fn data_size(texture: &TextureDescriptor) -> Option<u32> {
    let bits = bits_per_pixel(texture.format)?;
    let height = texture.height as u32;
    // Swizzled textures always occupy whole 8-row blocks.
    let rows = if texture.swizzled != 0 { (height + 7) & !7 } else { height };
    Some(row_bytes(texture, bits) * rows)
}

/// Where a texture's own texels lie when its buffer is wider than the
/// texture: `pieces` runs of `piece_bytes`, `source_step` bytes apart,
/// starting at the texture's address. Copied one after the other they form
/// the texture `compact` (same texels, buffer width = texture width).
///
/// Final Fantasy's world map draws each 32x32 tile as a window into a
/// 512-wide CLUT8 sheet: its data spans 32 whole sheet rows (16 KiB) but
/// only 1 KiB of that is the tile's. Copying and hashing just that 1 KiB is
/// 16x less work, and a tile no longer counts as changed when the animated
/// water elsewhere in the same rows changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompactLayout {
    pub pieces: u32,
    pub piece_bytes: u32,
    pub source_step: u32,
    pub compact: TextureDescriptor,
}

/// The compact layout of `texture`, or `None` if its data is already
/// contiguous (buffer width = width), its buffer is narrower than the
/// texture (rows overlap), or the format is unsupported.
pub fn compact_layout(texture: &TextureDescriptor) -> Option<CompactLayout> {
    let bits = bits_per_pixel(texture.format)?;
    let width_bits = texture.width as u32 * bits;
    if width_bits % 8 != 0 {
        return None;  // CLUT4 one texel wide: rows don't end on a byte
    }
    let row = row_bytes(texture, bits);
    let height = texture.height as u32;
    // Swizzled textures keep whole 16-byte x 8-row blocks.
    let compact_row = if texture.swizzled != 0 { (width_bits / 8).max(16).next_multiple_of(16) } else { width_bits / 8 };
    if compact_row >= row {
        return None;
    }
    let (pieces, piece_bytes, source_step) = if texture.swizzled != 0 {
        // One run per block row: its first blocks are the texture's.
        (height.div_ceil(8), compact_row * 8, row * 8)
    } else {
        (height, compact_row, row)
    };
    let compact = TextureDescriptor { buffer_width: compact_row * 8 / bits, ..*texture };
    Some(CompactLayout { pieces, piece_bytes, source_step, compact })
}

/// A hash of the texture's pixel data, to notice when it changes: FNV-1a in
/// four interleaved lanes (word i goes to lane i % 4), combined at the end.
/// One lane would make every step wait for the previous multiply; four
/// independent ones let the Vita's CPU overlap them (Final Fantasy's battle
/// intro re-checks 512 KiB textures every few frames).
pub fn hash_texture(data: &[u8]) -> u32 {
    const PRIME: u32 = 0x0100_0193;
    let mut lanes = [0x811C_9DC5u32; 4];
    let blocks = data.chunks_exact(16);
    let rest = blocks.remainder();
    for block in blocks {
        for (lane, word) in lanes.iter_mut().zip(block.chunks_exact(4)) {
            *lane = (*lane ^ u32::from_le_bytes([word[0], word[1], word[2], word[3]])).wrapping_mul(PRIME);
        }
    }
    for (lane, word) in lanes.iter_mut().zip(rest.chunks(4)) {
        let mut bytes = [0u8; 4];
        bytes[..word.len()].copy_from_slice(word);
        *lane = (*lane ^ u32::from_le_bytes(bytes)).wrapping_mul(PRIME);
    }
    lanes.iter().fold(0x811C_9DC5u32, |hash, lane| (hash ^ lane).wrapping_mul(PRIME))
}

/// Longest texel row in bytes: 512 texels of 32 bits.
const MAX_ROW_BYTES: usize = 512 * 4;

/// Copies texel row `y` into `out` (the row's first `out.len()` bytes), as
/// the GE addresses it: rows step by the buffer width (`row` bytes), and
/// swizzled textures are stored as 16-byte x 8-row blocks. Bytes past the
/// end of `data` read as 0 (Final Fantasy's Dragon Zombie is 128 texels
/// wide with a buffer width of 96: its last row runs past the data).
///
/// Copying whole 16-byte pieces is much faster than addressing each texel
/// (45 ms for a 512x512 texture on the Vita).
fn read_row(data: &[u8], texture: &TextureDescriptor, row: u32, y: u32, out: &mut [u8]) {
    if texture.swizzled == 0 {
        copy_clamped(data, (y * row) as usize, out);
        return;
    }
    let blocks_per_row = row / 16;
    let block_row_start = (y / 8) * blocks_per_row * 128 + (y % 8) * 16;
    let pieces = out.len().div_ceil(16);
    let last_piece_end = block_row_start as usize + (pieces - 1) * 128 + 16;
    if out.len() % 16 == 0 && last_piece_end <= data.len() {
        // Usual case, everything inside the data: fixed-size copies.
        let mut at = block_row_start as usize;
        for piece in out.chunks_exact_mut(16) {
            let source: &[u8; 16] = data[at..at + 16].try_into().unwrap();
            piece.copy_from_slice(source);
            at += 128;
        }
        return;
    }
    for (index, piece) in out.chunks_mut(16).enumerate() {
        copy_clamped(data, (block_row_start + index as u32 * 128) as usize, piece);
    }
}

fn copy_clamped(data: &[u8], start: usize, out: &mut [u8]) {
    let available = data.len().saturating_sub(start).min(out.len());
    if available > 0 {
        out[..available].copy_from_slice(&data[start..start + available]);
    }
    out[available..].fill(0);
}

/// Checks size, output and data length shared by all decoders; returns the
/// bits per texel and the row step in bytes.
fn check_decode(
    data: &[u8],
    texture: &TextureDescriptor,
    out_len: usize,
    out_stride: usize,
) -> Result<(u32, u32), DecodeError> {
    let bits = bits_per_pixel(texture.format).ok_or(DecodeError::UnsupportedFormat)?;
    let (width, height) = (texture.width as usize, texture.height as usize);
    if width == 0 || height == 0 || texture.width > MAX_SIZE || texture.height > MAX_SIZE {
        return Err(DecodeError::BadSize);
    }
    if out_stride < width || out_len < (height - 1) * out_stride + width {
        return Err(DecodeError::OutputTooSmall);
    }
    if (data.len() as u64) < data_size(texture).unwrap_or(u32::MAX) as u64 {
        return Err(DecodeError::BadAddress);
    }
    Ok((bits, row_bytes(texture, bits)))
}

/// Bytes holding `width` texels of `bits` bits.
fn texel_bytes(width: usize, bits: u32) -> usize {
    (width * bits as usize).div_ceil(8)
}

/// FNV-1a over palette words.
pub fn hash_words(words: &[u32]) -> u32 {
    words.iter().fold(0x811C_9DC5u32, |hash, word| (hash ^ word).wrapping_mul(0x0100_0193))
}

/// Decodes `texture` into RGBA8888 pixels (`0xAABBGGRR`), row by row with
/// `out_stride` pixels per row.
///
/// `data` is the texture's memory starting at its address, at least
/// `data_size(texture)` bytes. `palette` holds the raw palette words.
pub fn decode_texture(
    data: &[u8],
    texture: &TextureDescriptor,
    palette: &[u32],
    out: &mut [u32],
    out_stride: usize,
) -> Result<(), DecodeError> {
    let (bits, row) = check_decode(data, texture, out.len(), out_stride)?;
    let (width, height) = (texture.width as usize, texture.height as usize);
    let colors = PaletteColors::new(texture.clut_format, palette);
    let forced_alpha = if texture.flags & texture_flags::IGNORE_ALPHA != 0 { 0xFF00_0000 } else { 0 };
    let mut bytes = [0u8; MAX_ROW_BYTES];
    let bytes = &mut bytes[..texel_bytes(width, bits)];

    for y in 0..height {
        read_row(data, texture, row, y as u32, bytes);
        let out = &mut out[y * out_stride..y * out_stride + width];
        match texture.format {
            format::ABGR8888 => {
                for (pixel, raw) in out.iter_mut().zip(bytes.chunks_exact(4)) {
                    *pixel = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]) | forced_alpha;
                }
            }
            format::BGR5650 | format::ABGR5551 | format::ABGR4444 => {
                let color_format = match texture.format {
                    format::BGR5650 => ColorFormat::Bgr5650,
                    format::ABGR5551 => ColorFormat::Abgr5551,
                    _ => ColorFormat::Abgr4444,
                };
                for (pixel, raw) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                    *pixel = color_format.to_rgba(u16::from_le_bytes([raw[0], raw[1]]) as u32) | forced_alpha;
                }
            }
            format::CLUT4 => {
                for (pair, byte) in out.chunks_mut(2).zip(bytes.iter()) {
                    pair[0] = colors.lookup(*byte as u32 & 0xF) | forced_alpha;
                    if let Some(second) = pair.get_mut(1) {
                        *second = colors.lookup(*byte as u32 >> 4) | forced_alpha;
                    }
                }
            }
            format::CLUT8 => {
                for (pixel, byte) in out.iter_mut().zip(bytes.iter()) {
                    *pixel = colors.lookup(*byte as u32) | forced_alpha;
                }
            }
            format::CLUT16 => {
                for (pixel, raw) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                    *pixel = colors.lookup(u16::from_le_bytes([raw[0], raw[1]]) as u32) | forced_alpha;
                }
            }
            _ => {
                for (pixel, raw) in out.iter_mut().zip(bytes.chunks_exact(4)) {
                    *pixel = colors.lookup(u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]])) | forced_alpha;
                }
            }
        }
    }
    Ok(())
}

/// 16-bit direct colour textures (BGR5650, ABGR5551, ABGR4444) have the same
/// bit layout as the Vita GPU's U5U6U5_BGR, U1U5U5U5_ABGR and U4U4U4U4_ABGR
/// formats, so they can be shown as they are: half the memory of RGBA, and
/// "decoding" is copying rows. Final Fantasy's battle intro animates four
/// 512x512 ABGR4444 textures.
pub fn is_vita_16_bit_format(format: u8) -> bool {
    matches!(format, format::BGR5650 | format::ABGR5551 | format::ABGR4444)
}

/// The texels of a 16-bit direct colour texture, unswizzled, as
/// little-endian bytes with `out_stride` texels per row. Texels are made
/// opaque if the texture's alpha is ignored.
pub fn decode_16_bit(
    data: &[u8],
    texture: &TextureDescriptor,
    out: &mut [u8],
    out_stride: usize,
) -> Result<(), DecodeError> {
    if !is_vita_16_bit_format(texture.format) {
        return Err(DecodeError::UnsupportedFormat);
    }
    let (bits, row) = check_decode(data, texture, out.len() / 2, out_stride)?;
    let (width, height) = (texture.width as usize, texture.height as usize);
    // High byte of the alpha bits.
    let opaque = match texture.format {
        _ if texture.flags & texture_flags::IGNORE_ALPHA == 0 => 0,
        format::ABGR5551 => 0x80,
        format::ABGR4444 => 0xF0,
        _ => 0,
    };
    for y in 0..height {
        let start = y * out_stride * 2;
        let out = &mut out[start..start + texel_bytes(width, bits)];
        read_row(data, texture, row, y as u32, out);
        if opaque != 0 {
            for texel in out.chunks_exact_mut(2) {
                texel[1] |= opaque;
            }
        }
    }
    Ok(())
}

/// Palette textures (CLUT4, CLUT8) can be shown by the Vita GPU directly as
/// 8-bit palette textures: a quarter of the memory of RGBA and much less
/// decoding. `decode_palette_indices` writes one byte per texel (the raw
/// PSP index) and `palette_rgba` the matching 256-colour table, with the
/// PSP's index rules (shift, mask, start) already applied.
pub fn is_vita_palette_format(format: u8) -> bool {
    format == format::CLUT4 || format == format::CLUT8
}

/// Raw palette indices, one byte per texel, `out_stride` bytes per row.
pub fn decode_palette_indices(
    data: &[u8],
    texture: &TextureDescriptor,
    out: &mut [u8],
    out_stride: usize,
) -> Result<(), DecodeError> {
    if !is_vita_palette_format(texture.format) {
        return Err(DecodeError::UnsupportedFormat);
    }
    let (bits, row) = check_decode(data, texture, out.len(), out_stride)?;
    let (width, height) = (texture.width as usize, texture.height as usize);
    let mut bytes = [0u8; MAX_ROW_BYTES];
    let bytes = &mut bytes[..texel_bytes(width, bits)];
    for y in 0..height {
        let out = &mut out[y * out_stride..y * out_stride + width];
        if bits == 8 {
            read_row(data, texture, row, y as u32, out);
            continue;
        }
        read_row(data, texture, row, y as u32, bytes);
        for (pair, byte) in out.chunks_mut(2).zip(bytes.iter()) {
            pair[0] = byte & 0xF;  // low nibble first
            if let Some(second) = pair.get_mut(1) {
                *second = byte >> 4;
            }
        }
    }
    Ok(())
}

/// The 256 colours (RGBA) for `decode_palette_indices` output: entry i is
/// what the PSP shows for raw index i.
pub fn palette_rgba(texture: &TextureDescriptor, palette: &[u32], out: &mut [u32; 256]) {
    let colors = PaletteColors::new(texture.clut_format, palette);
    let forced_alpha = if texture.flags & texture_flags::IGNORE_ALPHA != 0 { 0xFF00_0000 } else { 0 };
    for (index, entry) in out.iter_mut().enumerate() {
        *entry = colors.lookup(index as u32) | forced_alpha;
    }
}

/// Gives fully transparent texels the average colour of their visible
/// neighbours (alpha stays 0), so bilinear filtering doesn't blend a
/// hidden colour into sprite edges (Final Fantasy's sprites have pink
/// transparent pixels: a pink fringe with smoothing). Only for smoothing:
/// with blending off, the colour of transparent texels is visible.
pub fn bleed_transparent_texels(pixels: &mut [u32], width: usize, height: usize, stride: usize) {
    let visible = |pixel: u32| pixel >> 24 != 0;
    for y in 0..height {
        for x in 0..width {
            if visible(pixels[y * stride + x]) {
                continue;
            }
            let mut sum = [0u32; 3];
            let mut count = 0;
            let neighbours = [(x.wrapping_sub(1), y), (x + 1, y), (x, y.wrapping_sub(1)), (x, y + 1)];
            for (nx, ny) in neighbours {
                if nx >= width || ny >= height {
                    continue;
                }
                let neighbour = pixels[ny * stride + nx];
                // Only originally visible neighbours: they are never changed here.
                if visible(neighbour) {
                    sum[0] += neighbour & 0xFF;
                    sum[1] += (neighbour >> 8) & 0xFF;
                    sum[2] += (neighbour >> 16) & 0xFF;
                    count += 1;
                }
            }
            if count > 0 {
                pixels[y * stride + x] = (sum[0] / count) | (sum[1] / count) << 8 | (sum[2] / count) << 16;
            }
        }
    }
}

/// The raw value of one texel of `bits` bits at byte `at` (4-bit texels:
/// `bit_shift` 0 = low nibble, 4 = high nibble). Test reference, see
/// `byte_offset`.
#[cfg(test)]
fn read_texel(data: &[u8], at: usize, bits: u32, bit_shift: u32) -> Option<u32> {
    Some(match bits {
        4 => (*data.get(at)? as u32 >> bit_shift) & 0xF,
        8 => *data.get(at)? as u32,
        16 => u16::from_le_bytes([*data.get(at)?, *data.get(at + 1)?]) as u32,
        _ => u32::from_le_bytes(data.get(at..at + 4)?.try_into().ok()?),
    })
}

/// A palette converted to RGBA once per texture, plus the CLUT_FORMAT rule
/// for turning a texel value into an entry number:
/// `((value >> shift) & mask) | start`.
struct PaletteColors {
    colors: [u32; 512],
    shift: u32,
    mask: u32,
    start: u32,
}

impl PaletteColors {
    fn new(clut_format: u32, words: &[u32]) -> Self {
        let entry_format = ColorFormat::from_bits(clut_format);
        // Entries outside the loaded palette stay transparent black.
        let mut colors = [0u32; 512];
        for (entry, color) in colors.iter_mut().enumerate() {
            let raw = if entry_format.bytes_per_pixel() == 4 {
                words.get(entry).copied()
            } else {
                words.get(entry / 2).map(|word| (word >> ((entry % 2) * 16)) & 0xFFFF)
            };
            if let Some(raw) = raw {
                *color = entry_format.to_rgba(raw);
            }
        }
        PaletteColors {
            colors,
            shift: (clut_format >> 2) & 0x1F,
            mask: (clut_format >> 8) & 0xFF,
            start: ((clut_format >> 16) & 0x1F) << 4,
        }
    }

    fn lookup(&self, value: u32) -> u32 {
        let entry = ((value >> self.shift) & self.mask) | self.start;
        self.colors[(entry & 0x1FF) as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ADDRESS: u32 = 0x0890_0000;

    /// Little-endian bytes of `words`, i.e. texture memory.
    fn bytes(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|word| word.to_le_bytes()).collect()
    }

    fn texture(format: u8, width: u16, height: u16) -> TextureDescriptor {
        TextureDescriptor { address: ADDRESS, buffer_width: width as u32, width, height, format, ..Default::default() }
    }

    #[test]
    fn decodes_8888() {
        let data = bytes(&[0xFF00_00FF, 0x8000_FF00]);
        let mut out = [0u32; 2];
        decode_texture(&data, &texture(format::ABGR8888, 2, 1), &[], &mut out, 2).unwrap();
        assert_eq!(out, [0xFF00_00FF, 0x8000_FF00]);
    }

    #[test]
    fn decodes_clut4_low_nibble_first() {
        // Pixels 0..8 = indices 1, 2, 0, 0, ...
        let data = bytes(&[0x0000_0021]);
        // Palette in 8888 format (CLUT_FORMAT entry format 3), mask 0xFF.
        let clut_format = 3 | (0xFF << 8);
        let palette = [0xFF00_0000, 0xFF00_00FF, 0xFF00_FF00];
        let mut descriptor = texture(format::CLUT4, 8, 1);
        descriptor.clut_format = clut_format;
        let mut out = [0u32; 8];
        decode_texture(&data, &descriptor, &palette, &mut out, 8).unwrap();
        assert_eq!(&out[..3], &[0xFF00_00FF, 0xFF00_FF00, 0xFF00_0000]);
    }

    #[test]
    fn decodes_clut8_with_16_bit_palette() {
        let data = bytes(&[0x0000_0100]); // indices 0, 1
        // 5551 entries: entry 0 = transparent, entry 1 = opaque red.
        let palette = [0x801F_0000];
        let mut descriptor = texture(format::CLUT8, 2, 1);
        descriptor.clut_format = 1 | (0xFF << 8);
        let mut out = [0u32; 2];
        decode_texture(&data, &descriptor, &palette, &mut out, 2).unwrap();
        assert_eq!(out, [0x0000_0000, 0xFF00_00FF]);
    }

    #[test]
    fn unswizzles_blocks() {
        // 8888, 8 pixels wide = 32 bytes per row = 2 blocks per block row.
        // In swizzled order the second block (bytes 128..256) holds
        // pixels 4..8 of rows 0..8.
        let mut words = vec![0u32; 64];
        words[32] = 0xAAAA_AAAA; // first word of block 1 = pixel (4, 0)
        words[4] = 0xBBBB_BBBB; // block 0, row 1 = pixel (0, 1)
        let data = bytes(&words);
        let mut descriptor = texture(format::ABGR8888, 8, 8);
        descriptor.swizzled = 1;
        let mut out = [0u32; 64];
        decode_texture(&data, &descriptor, &[], &mut out, 8).unwrap();
        assert_eq!(out[4], 0xAAAA_AAAA);
        assert_eq!(out[8], 0xBBBB_BBBB);
    }

    #[test]
    fn rows_step_by_buffer_width_not_texture_width() {
        // 8888, declared 4 pixels wide but stored with a stride of 2 pixels.
        let data = bytes(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let mut descriptor = texture(format::ABGR8888, 4, 2);
        descriptor.buffer_width = 2;
        let mut out = [0u32; 8];
        decode_texture(&data, &descriptor, &[], &mut out, 4).unwrap();
        // Row 1 starts at pixel 2 of memory, not pixel 4.
        assert_eq!(out[4], 3);
    }

    #[test]
    fn ignore_alpha_makes_texels_opaque() {
        let data = bytes(&[0x0011_2233]);
        let mut descriptor = texture(format::ABGR8888, 1, 1);
        descriptor.flags = texture_flags::IGNORE_ALPHA;
        let mut out = [0u32; 1];
        decode_texture(&data, &descriptor, &[], &mut out, 1).unwrap();
        assert_eq!(out[0], 0xFF11_2233);
    }

    #[test]
    fn transparent_texels_take_their_neighbours_colour() {
        // Opaque red, transparent pink, opaque blue in a row.
        let mut pixels = [0xFF00_00FF, 0x00FF_00FF, 0xFFFF_0000];
        bleed_transparent_texels(&mut pixels, 3, 1, 3);
        assert_eq!(pixels[1], 0x007F_007F); // average of red and blue, still transparent
        assert_eq!(pixels[0], 0xFF00_00FF);
    }

    #[test]
    fn too_little_data_is_an_error_not_a_crash() {
        let data = bytes(&[0; 3]);
        let mut out = [0u32; 16];
        assert_eq!(
            decode_texture(&data, &texture(format::ABGR8888, 4, 4), &[], &mut out, 4),
            Err(DecodeError::BadAddress)
        );
    }

    #[test]
    fn small_swizzled_texture_uses_whole_blocks() {
        // CLUT4, 8x2: rows are 4 bytes, but swizzled storage is one full
        // 16-byte x 8-row block.
        let mut descriptor = texture(format::CLUT4, 8, 2);
        descriptor.swizzled = 1;
        assert_eq!(data_size(&descriptor), Some(128));
    }

    #[test]
    fn palette_texture_as_indices_and_table() {
        // CLUT4 indices 1, 2; palette shifted by start position 16.
        let data = bytes(&[0x0000_0021]);
        let mut descriptor = texture(format::CLUT4, 8, 1);
        descriptor.clut_format = 3 | (0xFF << 8) | (1 << 16); // start = 16
        let mut indices = [0u8; 8];
        decode_palette_indices(&data, &descriptor, &mut indices, 8).unwrap();
        assert_eq!(&indices[..2], &[1, 2]);

        let mut words = vec![0u32; 32];
        words[17] = 0xFF00_00FF;
        let mut table = [0u32; 256];
        palette_rgba(&descriptor, &words, &mut table);
        assert_eq!(table[1], 0xFF00_00FF, "index 1 | start 16 = entry 17");
    }

    #[test]
    fn texels_beyond_the_buffer_read_as_zero() {
        // 8888, 4 wide but buffer width 2, 2 rows: only 16 bytes of data.
        let data = bytes(&[1, 2, 3, 4]);
        let mut descriptor = texture(format::ABGR8888, 4, 2);
        descriptor.buffer_width = 2;
        let mut out = [9u32; 8];
        decode_texture(&data, &descriptor, &[], &mut out, 4).unwrap();
        assert_eq!(out[4..8], [3, 4, 0, 0]);
    }

    /// Raw texel values the slow way, for comparing with the decoders.
    fn reference_texels(data: &[u8], texture: &TextureDescriptor) -> Vec<u32> {
        let bits = bits_per_pixel(texture.format).unwrap();
        let row = row_bytes(texture, bits);
        let mut texels = Vec::new();
        for y in 0..texture.height as u32 {
            for x in 0..texture.width as u32 {
                let x_bits = x * bits;
                let at = byte_offset(texture, row, x_bits / 8, y) as usize;
                texels.push(read_texel(data, at, bits, x_bits % 8).unwrap_or(0));
            }
        }
        texels
    }

    #[test]
    fn row_decoding_matches_texel_by_texel_addressing() {
        // Pseudo-random bytes; swizzled and linear; a buffer width narrower
        // than the texture (rows overlap, the last runs past the data).
        let data: Vec<u8> = (0..64 * 1024u32).map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8).collect();
        for swizzled in [0, 1] {
            // ABGR5551 at full buffer width, like Final Fantasy's battle-start snapshots.
            for (fmt, buffer_width) in [(format::CLUT4, 96), (format::CLUT8, 64), (format::ABGR4444, 32), (format::ABGR5551, 128), (format::ABGR8888, 64)] {
                let mut descriptor = texture(fmt, 128, 64);
                descriptor.swizzled = swizzled;
                descriptor.buffer_width = buffer_width;
                descriptor.clut_format = 3 | (0xFF << 8);
                let size = data_size(&descriptor).unwrap() as usize;
                let data = &data[..size];
                let expected = reference_texels(data, &descriptor);

                if is_vita_palette_format(fmt) {
                    let mut indices = vec![0u8; 128 * 64];
                    decode_palette_indices(data, &descriptor, &mut indices, 128).unwrap();
                    let got: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
                    assert_eq!(got, expected, "indices fmt {} swizzled {}", fmt, swizzled);
                } else if is_vita_16_bit_format(fmt) {
                    let mut texels = vec![0u8; 128 * 64 * 2];
                    decode_16_bit(data, &descriptor, &mut texels, 128).unwrap();
                    let got: Vec<u32> = texels.chunks_exact(2).map(|t| u16::from_le_bytes([t[0], t[1]]) as u32).collect();
                    assert_eq!(got, expected, "16-bit fmt {} swizzled {}", fmt, swizzled);
                } else {
                    let mut pixels = vec![0u32; 128 * 64];
                    decode_texture(data, &descriptor, &[], &mut pixels, 128).unwrap();
                    assert_eq!(pixels, expected, "8888 swizzled {}", swizzled);
                }
            }
        }
    }

    /// Copies the pieces of `layout` one after the other, like the Vita does.
    fn gather(data: &[u8], layout: &CompactLayout) -> Vec<u8> {
        let mut out = Vec::new();
        for piece in 0..layout.pieces {
            let start = (piece * layout.source_step) as usize;
            out.extend_from_slice(&data[start..start + layout.piece_bytes as usize]);
        }
        out
    }

    #[test]
    fn compact_layout_keeps_the_texels() {
        let data: Vec<u8> = (0..256 * 1024u32).map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8).collect();
        for swizzled in [0, 1] {
            for (fmt, width, height, buffer_width) in [
                (format::CLUT8, 32, 32, 512),  // Final Fantasy's world map tiles
                (format::CLUT4, 16, 8, 128),
                (format::CLUT4, 8, 4, 64),     // narrower than one swizzle block
                (format::ABGR5551, 64, 16, 256),
                (format::ABGR8888, 16, 8, 64),
            ] {
                let mut descriptor = texture(fmt, width, height);
                descriptor.swizzled = swizzled;
                descriptor.buffer_width = buffer_width;
                descriptor.clut_format = 3 | (0xFF << 8);
                let layout = compact_layout(&descriptor).expect("buffer wider than texture");
                let compact = gather(&data, &layout);
                assert_eq!(data_size(&layout.compact), Some(compact.len() as u32), "fmt {} swizzled {}", fmt, swizzled);
                assert!(compact.len() < data_size(&descriptor).unwrap() as usize);
                let full = &data[..data_size(&descriptor).unwrap() as usize];
                assert_eq!(
                    reference_texels(&compact, &layout.compact),
                    reference_texels(full, &descriptor),
                    "fmt {} swizzled {}",
                    fmt,
                    swizzled
                );
            }
        }
    }

    #[test]
    fn world_map_tile_is_four_runs_of_256_bytes() {
        let mut descriptor = texture(format::CLUT8, 32, 32);
        descriptor.swizzled = 1;
        descriptor.buffer_width = 512;
        let layout = compact_layout(&descriptor).unwrap();
        assert_eq!((layout.pieces, layout.piece_bytes, layout.source_step), (4, 256, 4096));
        assert_eq!(layout.compact.buffer_width, 32);
    }

    #[test]
    fn no_compact_layout_when_contiguous_or_overlapping() {
        assert_eq!(compact_layout(&texture(format::CLUT8, 32, 32)), None);
        let mut narrow = texture(format::ABGR8888, 128, 64);
        narrow.buffer_width = 96;
        assert_eq!(compact_layout(&narrow), None);
        // Swizzled CLUT4 8 wide in a 32-wide buffer: both are one 16-byte block.
        let mut small = texture(format::CLUT4, 8, 8);
        small.swizzled = 1;
        small.buffer_width = 32;
        assert_eq!(compact_layout(&small), None);
    }

    #[test]
    fn opaque_16_bit_texels_when_alpha_is_ignored() {
        let data = bytes(&[0x0FFF_0123, 0x7FFF_0000]);
        let mut descriptor = texture(format::ABGR4444, 4, 1);
        descriptor.flags = texture_flags::IGNORE_ALPHA;
        let mut texels = [0u8; 8];
        decode_16_bit(&data, &descriptor, &mut texels, 4).unwrap();
        assert_eq!(texels, [0x23, 0xF1, 0xFF, 0xFF, 0x00, 0xF0, 0xFF, 0xFF]);
    }

    #[test]
    fn rejects_compressed_formats() {
        let data = bytes(&[0; 8]);
        let mut out = [0u32; 16];
        assert_eq!(
            decode_texture(&data, &texture(8, 4, 4), &[], &mut out, 4),
            Err(DecodeError::UnsupportedFormat)
        );
    }

    /// GE state for a 64x64 texture at ADDRESS with levels 1..3 (32, 16, 8)
    /// right after it, min filter `filter`, level register `level`.
    fn ge_with_levels(filter: u32, level: u32) -> (GeState, TextureDescriptor) {
        use crate::command::Command;
        let mut ge = GeState::new();
        let mut set = |opcode: u8, data: u32| ge.set_register(Command(((opcode as u32) << 24) | data));
        set(op::TEXTURE_MODE, 3 << 16); // highest level 3
        set(op::TEXTURE_FILTER, filter);
        set(op::TEXTURE_LEVEL, level);
        let mut address = ADDRESS;
        for (n, log2) in [(0u8, 6u32), (1, 5), (2, 4), (3, 3)] {
            set(op::TEXTURE_ADDRESS_0 + n, address & 0x00FF_FFF0);
            set(op::TEXTURE_BUFFER_WIDTH_0 + n, ((address >> 8) & 0x000F_0000) | (1 << log2));
            set(op::TEXTURE_SIZE_0 + n, log2 | (log2 << 8));
            address += (1 << log2) * (1 << log2) * 4;
        }
        let base = TextureDescriptor { address: ADDRESS, buffer_width: 64, width: 64, height: 64, format: 3, ..Default::default() };
        (ge, base)
    }

    /// Mip levels only with the menu option, a mip filter and levels; the
    /// PSP-like mode adds a level of bias, "auto" adds the game's offset,
    /// "slope" doesn't; a level that isn't half the one before ends the chain.
    #[test]
    fn mip_levels_follow_the_ge_and_the_menu_option() {
        let (ge, base) = ge_with_levels(7, 0); // linear + mip linear, auto, offset 0
        assert_eq!(describe_levels_in_mode(&ge, &base, 0), None, "Sharp: no levels");
        let psp = describe_levels_in_mode(&ge, &base, 1).unwrap();
        assert_eq!((psp.count, psp.linear, psp.bias_eighths), (4, true, 8));
        assert_eq!(psp.levels.address[1], ADDRESS + 64 * 64 * 4);
        assert_eq!((psp.levels.buffer_width[2], psp.levels.log2_size[3]), (16, 3 | (3 << 4)));
        let native = describe_levels_in_mode(&ge, &base, 2).unwrap();
        assert_eq!(native.bias_eighths, 0, "Mipmapped: the GPU's choice at our size");

        let (ge, base) = ge_with_levels(5, 0x20 << 16); // mip nearest, auto, offset +2 levels (32/16)
        let auto = describe_levels_in_mode(&ge, &base, 2).unwrap();
        assert_eq!((auto.linear, auto.bias_eighths), (false, 16));
        let (ge, base) = ge_with_levels(7, 2 | (0x20 << 16)); // slope: offset ignored
        assert_eq!(describe_levels_in_mode(&ge, &base, 1).unwrap().bias_eighths, 8);
        let (ge, base) = ge_with_levels(7, 1); // const level 0
        assert_eq!(describe_levels_in_mode(&ge, &base, 1), None);
        let (ge, base) = ge_with_levels(1, 0); // no mip filter
        assert_eq!(describe_levels_in_mode(&ge, &base, 1), None);

        let (mut ge, base) = ge_with_levels(7, 0);
        ge.set_register(crate::command::Command(((op::TEXTURE_SIZE_0 as u32 + 2) << 24) | 5 | (5 << 8))); // level 2 not half
        assert_eq!(describe_levels_in_mode(&ge, &base, 1).unwrap().count, 2);
    }

    /// GXM's layout as measured on the Vita: levels back to back, rows
    /// rounded up to 8 texels; and a level as a texture of its own.
    #[test]
    fn mip_level_offsets_and_level_textures() {
        assert_eq!(mip_level_offset(64, 64, 4, 1), 64 * 64 * 4);
        assert_eq!(mip_level_offset(64, 64, 4, 4), (64 * 64 + 32 * 32 + 16 * 16 + 8 * 8) * 4);
        assert_eq!(mip_level_offset(64, 64, 4, 5), (64 * 64 + 32 * 32 + 16 * 16 + 8 * 8 + 8 * 4) * 4, "4 wide: rows of 8");
        assert_eq!(mip_level_offset(64, 64, 1, 7), 64 * 64 + 32 * 32 + 16 * 16 + 8 * 8 + 8 * 4 + 8 * 2 + 8);
        let (ge, base) = ge_with_levels(7, 0);
        let mips = describe_levels_in_mode(&ge, &base, 1).unwrap();
        let flagged = TextureDescriptor { flags: 3 << texture_flags::LEVELS_SHIFT, ..base };
        let level2 = level_descriptor(&flagged, &mips.levels, 2);
        assert_eq!((level2.width, level2.height, level2.buffer_width), (16, 16, 16));
        assert_eq!(level2.address, ADDRESS + (64 * 64 + 32 * 32) * 4);
        assert_eq!(level2.flags & texture_flags::LEVELS_MASK, 0, "a level alone has no levels");
    }
}
