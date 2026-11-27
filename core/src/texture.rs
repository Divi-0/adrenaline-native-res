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
    pub const DXT1: u8 = 8;
    pub const DXT3: u8 = 9;
    pub const DXT5: u8 = 10;
}

/// Largest texture the PSP supports.
pub const MAX_SIZE: u16 = 512;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeError {
    /// Invalid format (or one the called decoder doesn't do).
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
        format::CLUT8 | format::DXT3 | format::DXT5 => Some(8),
        format::DXT1 => Some(4),
        _ => None,
    }
}

/// DXT1/3/5: 4x4 texel blocks (PSP order: index rows before colours, see
/// `decode_dxt`).
pub fn is_dxt_format(format: u8) -> bool {
    (format::DXT1..=format::DXT5).contains(&format)
}

/// A DXT texture's blocks as PPSSPP lays them out (DecodeDXTBlocks): rows of
/// `per_row` blocks (the buffer width / 4, at least 1; a buffer width of 0
/// counts as 16 bytes, GetTextureBufw), `rows` block rows, `row_blocks`
/// decoded per row. A texture wider than its buffer runs on into the next
/// row's blocks. Swizzling doesn't apply.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DxtLayout {
    block_bytes: u32,
    per_row: u32,
    rows: u32,
    row_blocks: u32,
}

impl DxtLayout {
    fn of(texture: &TextureDescriptor) -> Option<DxtLayout> {
        let (block_bytes, bits) = match texture.format {
            format::DXT1 => (8, 4),
            format::DXT3 | format::DXT5 => (16, 8),
            _ => return None,
        };
        let buffer_width = if texture.buffer_width != 0 { texture.buffer_width } else { 8 * 16 / bits };
        Some(DxtLayout {
            block_bytes,
            per_row: (buffer_width / 4).max(1),
            rows: (texture.height as u32).div_ceil(4),
            row_blocks: (texture.width as u32).div_ceil(4),
        })
    }

    /// Bytes from the first block to the end of the last one decoded
    /// (PPSSPP's SourceExtent).
    fn bytes(&self) -> u32 {
        let extra = self.row_blocks.saturating_sub(self.per_row);
        (self.per_row * self.rows + extra) * self.block_bytes
    }
}

pub fn is_palette_format(format: u8) -> bool {
    (format::CLUT4..=format::CLUT32).contains(&format)
}

/// Describes the texture level 0 the GE would currently sample from.
/// Palette fields are left empty; the caller fills them in.
/// Menu option "NativeRes Distant Textures" (Adrenaline's config byte
/// `nativeres_mip_mode`, set by the display every frame): 0 "Sharp" (level 0
/// only, the default), 1 "Blurry (like PSP)" (the game's mip levels, one
/// level blurrier than the GPU would pick at our 2x size: the PSP's choice),
/// 2 "Mipmapped" (the game's levels as the GPU picks them at our size).
pub static MIP_MODE: core::sync::atomic::AtomicU8 = core::sync::atomic::AtomicU8::new(0);

/// Setting `slopemip=1` (options.rs; off by default): draws in the GE's
/// "slope" level mode (TEXLEVEL mode 2) sample their mip levels as the PSP
/// does, the level from each pixel's 1/w (`slope_level`), whatever
/// `MIP_MODE` says. Off: as before (the level the GPU picks, or level 0).
pub static SLOPE_MIP: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

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
    /// With `SLOPE_MIP`, a draw in slope mode: its `slope_base` (the level
    /// comes per pixel from `slope_level`; no bias then).
    pub slope_base: Option<i32>,
}

/// The float-bits log2 the GE's level selection uses, in 1/16 levels: the
/// exponent and the top 4 mantissa bits, a piecewise linear log2 floored to
/// 1/16 (ppsspp-ref GPU/Software/GEMath.h GELog16, from GPU probes).
pub fn ge_log16(value: f32) -> i32 {
    ((value.to_bits() >> 19) & 0x0FFF) as i32 - 127 * 16
}

/// The per-draw part of the slope mode's level, in 1/16 levels:
/// 16 + GELog16(slope) + the TEXLEVEL offset (ppsspp-ref GPU/Software/
/// Rasterizer.cpp CalculateSamplingParams). `slope_register`: TEXLODSLOPE's
/// 24 bits (a float24), `level_register`: TEXLEVEL's.
pub fn slope_base(slope_register: u32, level_register: u32) -> i32 {
    let slope = f32::from_bits((slope_register & 0x00FF_FFFF) << 8);
    let offset = ((level_register >> 16) & 0xFF) as u8 as i8 as i32;
    16 + ge_log16(slope) + offset
}

/// The GE's level in slope mode for a pixel whose 1/w is `q` (as the GE
/// interpolates it: linear on the screen): (level, fraction in 1/16 towards
/// the next level). detail = base - GELog16(q); none above 0 is level 0;
/// capped at the highest level; without blending between levels ("mip
/// nearest") rounded at half a level (ppsspp-ref Rasterizer.cpp
/// CalculateSamplingParams). The two levels then mix as
/// (next f + this (16 - f)) >> 4 per channel (Sampler.cpp).
pub fn slope_level(base: i32, q: f32, highest: u32, linear: bool) -> (u32, u32) {
    let detail = base - ge_log16(q);
    if detail <= 0 || highest == 0 {
        return (0, 0);
    }
    let mut level16 = detail.min(highest as i32 * 16);
    if !linear {
        level16 += 8;
    }
    ((level16 >> 4) as u32, if linear { (level16 & 15) as u32 } else { 0 })
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
    use core::sync::atomic::Ordering::Relaxed;
    describe_levels_with(ge, base, MIP_MODE.load(Relaxed), SLOPE_MIP.load(Relaxed))
}

/// `describe_levels` for menu option value `mode` (tests don't share the
/// globals).
#[cfg(test)]
fn describe_levels_in_mode(ge: &GeState, base: &TextureDescriptor, mode: u8) -> Option<MipLevels> {
    describe_levels_with(ge, base, mode, false)
}

/// `describe_levels` for menu option value `mode` and setting `slope`
/// (`SLOPE_MIP`): with it, slope-mode draws get their levels in any mode.
fn describe_levels_with(ge: &GeState, base: &TextureDescriptor, mode: u8, slope: bool) -> Option<MipLevels> {
    let min_filter = ge.register(op::TEXTURE_FILTER) & 7;
    let max_level = (ge.register(op::TEXTURE_MODE) >> 16) & 7;
    let level_register = ge.register(op::TEXTURE_LEVEL);
    let level_mode = level_register & 3;
    let slope = slope && level_mode == 2;
    if (mode == 0 && !slope) || min_filter & 4 == 0 || max_level == 0 {
        return None;
    }
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
    let linear = min_filter & 2 != 0;
    if slope {
        let base = slope_base(ge.register(op::TEXTURE_LOD_SLOPE), level_register);
        return Some(MipLevels { levels, count, linear, bias_eighths: 0, slope_base: Some(base) });
    }
    let game_bias = if level_mode == 2 { 0 } else { offset / 2 };
    let psp_like = if mode == 1 { 8 } else { 0 };
    Some(MipLevels { levels, count, linear, bias_eighths: (game_bias + psp_like).clamp(-31, 32) as i8, slope_base: None })
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
    Some(quick_sample_positions_in(words))
}

/// `quick_sample_positions` of data `words` words long (not 0): what the
/// copier reads for a sample task (`copy_tasks::Layout::Sample`).
pub fn quick_sample_positions_in(words: usize) -> [usize; QUICK_SAMPLES] {
    core::array::from_fn(|sample| (2 * sample + 1) * words / (2 * QUICK_SAMPLES))
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
    if let Some(layout) = DxtLayout::of(texture) {
        return Some(layout.bytes());
    }
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
    if is_dxt_format(texture.format) {
        return None; // block rows, not texel rows
    }
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
    let forced_alpha = if texture.flags & texture_flags::IGNORE_ALPHA != 0 { 0xFF00_0000 } else { 0 };
    if let Some(layout) = DxtLayout::of(texture) {
        decode_dxt(data, texture.format, &layout, out, out_stride, forced_alpha, (width, height));
        return Ok(());
    }
    let colors = PaletteColors::new(texture.clut_format, palette);
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

/// Decodes DXT blocks the way PPSSPP does (GPU/Common/TextureDecoder.cpp,
/// DXTDecoder). The PSP stores a block in the reverse order of the PC's:
/// bytes 0-3 the index rows (2 bits per texel, low bits first), 4-7 the two
/// RGB565 colours (red in the high bits); DXT3: bytes 8-15 one 16-bit alpha
/// row each (4 bits per texel); DXT5: bytes 8-13 the 48 bits of 3-bit alpha
/// indices (12 per row), 14 and 15 the two alphas. `data` holds at least
/// `layout.bytes()` (check_decode).
fn decode_dxt(
    data: &[u8],
    format: u8,
    layout: &DxtLayout,
    out: &mut [u32],
    out_stride: usize,
    forced_alpha: u32,
    (width, height): (usize, usize),
) {
    let block_bytes = layout.block_bytes as usize;
    for block_row in 0..layout.rows as usize {
        let y = block_row * 4;
        let block_height = (height - y).min(4);
        let mut index = block_row * layout.per_row as usize;
        for x in (0..width).step_by(4) {
            let block = &data[index * block_bytes..(index + 1) * block_bytes];
            index += 1;
            let block_width = (width - x).min(4);
            // DXT1's colours carry their alpha (255, transparent black);
            // DXT3/5's are 0 and get the block's own.
            let colors = dxt_colors(block, format == format::DXT1);
            for row in 0..block_height {
                let out = &mut out[(y + row) * out_stride + x..][..block_width];
                let mut indices = block[row] as u32;
                match format {
                    format::DXT3 => {
                        let mut alpha = u16::from_le_bytes([block[8 + 2 * row], block[9 + 2 * row]]) as u32;
                        for texel in out.iter_mut() {
                            *texel = colors[(indices & 3) as usize] | (alpha & 0xF) << 28 | forced_alpha;
                            indices >>= 2;
                            alpha >>= 4;
                        }
                    }
                    format::DXT5 => {
                        let alphas = dxt5_alphas(block[14], block[15]);
                        let all = u64::from(u32::from_le_bytes([block[8], block[9], block[10], block[11]]))
                            | u64::from(u16::from_le_bytes([block[12], block[13]])) << 32;
                        let mut alpha = (all >> (12 * row)) as u32;
                        for texel in out.iter_mut() {
                            *texel = colors[(indices & 3) as usize] | (alphas[(alpha & 7) as usize] as u32) << 24 | forced_alpha;
                            indices >>= 2;
                            alpha >>= 3;
                        }
                    }
                    _ => {
                        for texel in out.iter_mut() {
                            *texel = colors[(indices & 3) as usize] | forced_alpha;
                            indices >>= 2;
                        }
                    }
                }
            }
        }
    }
}

/// True if every texel `decode_texture` makes of `texture` from `data` has
/// alpha 255; false when unsure. Lets the display draw a normal alpha blend
/// of it unblended (core opaque_source.rs: no frame read on 16-bit
/// targets). Worked out from the source, not the decoded texels (those are
/// in uncached GPU memory on the Vita): 5650 and textures whose alpha is
/// ignored always; DXT1 when no decoded block uses its transparent colour
/// (3-colour mode, index 3); DXT5 when every alpha index used gives 255;
/// DXT3 never (PPSSPP's alpha is the nibble << 28: 0xF0 at most); others
/// not looked at (false).
pub fn all_texels_opaque(data: &[u8], texture: &TextureDescriptor) -> bool {
    opaque_rows(data, texture) == ALL_ROWS_OPAQUE
}

/// `opaque_rows`: every texel is opaque.
pub const ALL_ROWS_OPAQUE: u16 = u16::MAX;

/// How many rows of texels from the top `decode_texture` makes of `texture`
/// from `data` all have alpha 255 (`ALL_ROWS_OPAQUE`: the whole texture,
/// `all_texels_opaque`; 0 when unsure). A draw that samples only these rows
/// (core opaque_source `samples_rows_below`) sees an opaque texture: NFS
/// UR's title background is a 480x272 picture in a 512x512 DXT1 whose rows
/// from 280 on hold transparent blocks. Same rules as `all_texels_opaque`;
/// a DXT block row counts row by row, all four texels of each block.
pub fn opaque_rows(data: &[u8], texture: &TextureDescriptor) -> u16 {
    if texture.flags & texture_flags::IGNORE_ALPHA != 0 || texture.format == format::BGR5650 {
        return ALL_ROWS_OPAQUE;
    }
    let Some(layout) = DxtLayout::of(texture) else {
        return 0;
    };
    if texture.format == format::DXT3 || (data.len() as u64) < layout.bytes() as u64 {
        return 0;
    }
    let block_bytes = layout.block_bytes as usize;
    for row in 0..layout.rows as usize {
        let first = row * layout.per_row as usize;
        // Bit r: the block row's texel row r has a transparent texel.
        let mut transparent = 0u8;
        for index in first..first + layout.row_blocks as usize {
            let block = &data[index * block_bytes..(index + 1) * block_bytes];
            if texture.format == format::DXT1 {
                let (c1, c2) = (u16::from_le_bytes([block[4], block[5]]), u16::from_le_bytes([block[6], block[7]]));
                // 3-colour mode: index 3 is transparent black.
                if c1 <= c2 {
                    for (r, &line) in block[..4].iter().enumerate() {
                        if (0..4).any(|texel| (line >> (2 * texel)) & 3 == 3) {
                            transparent |= 1 << r;
                        }
                    }
                }
            } else {
                let alphas = dxt5_alphas(block[14], block[15]);
                let all = u64::from(u32::from_le_bytes([block[8], block[9], block[10], block[11]]))
                    | u64::from(u16::from_le_bytes([block[12], block[13]])) << 32;
                for r in 0..4 {
                    if (0..4).any(|texel| alphas[((all >> (3 * (4 * r + texel))) & 7) as usize] != 255) {
                        transparent |= 1 << r;
                    }
                }
            }
        }
        if transparent != 0 {
            let rows = 4 * row as u32 + transparent.trailing_zeros();
            if rows >= texture.height as u32 {
                break; // rows past the height aren't decoded
            }
            return rows as u16;
        }
    }
    ALL_ROWS_OPAQUE
}

/// Where a texture's texels may show: per cell of texels, which raw values
/// lie in it. A draw that keeps its target where the source alpha is 0
/// (core `zero_alpha`) can leave out a triangle whose fragments filter
/// from cells of transparent texels only (`zero_alpha::TexelTest`): God of
/// War's light streaks are 256x32 index textures transparent but for a few
/// columns, drawn over the whole screen (DESIGN "seventy-ninth session").
///
/// The texture's power-of-two size is cut into at most 32 columns and
/// `MASK_CELLS` cells of at least 8x8 texels (smaller textures: one cell
/// column or row). Made from the bytes a texture is decoded from
/// (`texel_mask`), so it is what the GPU samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TexelMask {
    /// log2 of a cell's width and height in texels.
    pub cell_shift: [u8; 2],
    /// log2 of the cells across and down.
    pub grid_shift: [u8; 2],
    /// Per cell, row by row: bit k set if a texel of raw value k lies in it
    /// (4-bit indices: the index; 4444 / 5551 / 8888: bit 0 for a texel
    /// whose alpha is not 0).
    pub used: [u16; MASK_CELLS],
}

/// The most cells a `TexelMask` has.
pub const MASK_CELLS: usize = 256;
/// The largest texture a mask is made of (texels): making one walks every
/// texel once, at decode (~10 ns each).
pub const MASK_MOST_TEXELS: u32 = 32 * 1024;

impl TexelMask {
    pub const EMPTY: TexelMask = TexelMask { cell_shift: [0; 2], grid_shift: [0; 2], used: [0; MASK_CELLS] };

    /// The raw values that are transparent in every texture of this kind
    /// whose texels are not palette indices (bit 0 means "alpha not 0").
    pub const DIRECT_TRANSPARENT: u16 = !1;

    /// The cells of the grid (`used` past them is 0 and means nothing).
    pub fn cells(&self) -> &[u16] {
        &self.used[..1 << (self.grid_shift[0] + self.grid_shift[1])]
    }

    /// True if every cell in columns `columns` and rows `rows` (inclusive,
    /// in cells, either may run past the grid's ends: they wrap) holds only
    /// raw values in `transparent`.
    pub fn transparent_cells(&self, columns: (i64, i64), rows: (i64, i64), transparent: u16) -> bool {
        let (column_mask, row_mask) = ((1i64 << self.grid_shift[0]) - 1, (1i64 << self.grid_shift[1]) - 1);
        let visible = !transparent;
        (rows.0..=rows.1).all(|row| {
            let first = ((row & row_mask) << self.grid_shift[0]) as usize;
            (columns.0..=columns.1).all(|column| self.used[first + (column & column_mask) as usize] & visible == 0)
        })
    }
}

/// The `TexelMask` of `texture` decoded from `data` (as `decode_texture`
/// reads it: swizzle, buffer width): 4-bit index textures and 4444 / 5551 /
/// 8888 ones whose alpha is used, power-of-two sizes of at most
/// `MASK_MOST_TEXELS` texels. None for other textures.
pub fn texel_mask(data: &[u8], texture: &TextureDescriptor) -> Option<TexelMask> {
    let (width, height) = (texture.width as u32, texture.height as u32);
    if texture.flags & texture_flags::IGNORE_ALPHA != 0
        || !width.is_power_of_two()
        || !height.is_power_of_two()
        || width * height > MASK_MOST_TEXELS
        || !matches!(texture.format, format::CLUT4 | format::ABGR4444 | format::ABGR5551 | format::ABGR8888)
    {
        return None;
    }
    let (bits, row) = check_decode(data, texture, width as usize * height as usize, width as usize).ok()?;
    let (width_log, height_log) = (width.trailing_zeros(), height.trailing_zeros());
    let cell_x = width_log.min(3).max(width_log.saturating_sub(5));
    let across = width_log - cell_x;
    let cell_y = height_log.min(3).max(height_log.saturating_sub(MASK_CELLS.trailing_zeros() - across));
    let mut mask = TexelMask { cell_shift: [cell_x as u8, cell_y as u8], grid_shift: [across as u8, (height_log - cell_y) as u8], used: [0; MASK_CELLS] };
    let mut bytes = [0u8; MAX_ROW_BYTES];
    let bytes = &mut bytes[..texel_bytes(width as usize, bits)];
    for y in 0..height {
        read_row(data, texture, row, y, bytes);
        let cells = &mut mask.used[((y >> cell_y) << across) as usize..][..1 << across];
        let texels_per_cell = 1usize << cell_x;
        let cell_bytes = (texels_per_cell * bits as usize).div_ceil(8);
        for (cell, chunk) in cells.iter_mut().zip(bytes.chunks(cell_bytes)) {
            *cell |= match texture.format {
                // Two texels a byte, the low nibble first; a 1-texel-wide texture uses only that one.
                format::CLUT4 if texels_per_cell == 1 => 1 << (chunk[0] & 0xF),
                format::CLUT4 => chunk.iter().fold(0u16, |used, &byte| used | 1 << (byte & 0xF) | 1 << (byte >> 4)),
                format::ABGR4444 => chunk.chunks_exact(2).any(|texel| texel[1] & 0xF0 != 0) as u16,
                format::ABGR5551 => chunk.chunks_exact(2).any(|texel| texel[1] & 0x80 != 0) as u16,
                _ => chunk.chunks_exact(4).any(|texel| texel[3] != 0) as u16,
            };
        }
    }
    Some(mask)
}

/// A DXT block's four colours (`0xAABBGGRR`), PPSSPP's DecodeColors.
fn dxt_colors(block: &[u8], dxt1: bool) -> [u32; 4] {
    let c1 = u16::from_le_bytes([block[4], block[5]]) as u32;
    let c2 = u16::from_le_bytes([block[6], block[7]]) as u32;
    let split = |c: u32| [(c >> 8) & 0xF8, (c >> 3) & 0xFC, (c << 3) & 0xF8];
    let (first, second) = (split(c1), split(c2));
    let alpha = if dxt1 { 255 } else { 0 };
    let make = |[r, g, b]: [u32; 3]| alpha << 24 | b << 16 | g << 8 | r;
    let mix = |a: [u32; 3], b: [u32; 3]| -> [u32; 3] { core::array::from_fn(|i| (a[i] + a[i] + b[i]) / 3) };
    if c1 > c2 {
        [make(first), make(second), make(mix(first, second)), make(mix(second, first))]
    } else {
        let average: [u32; 3] = core::array::from_fn(|i| (first[i] + second[i]) / 2);
        [make(first), make(second), make(average), 0]
    }
}

/// A DXT5 block's eight alphas, PPSSPP's DecodeAlphaDXT5 (8.8 fixed point
/// weights, + 31 before the shift).
fn dxt5_alphas(alpha1: u8, alpha2: u8) -> [u8; 8] {
    let (a1, a2) = (alpha1 as u32, alpha2 as u32);
    let lerp = |n: u32, steps: u32| ((a1 * ((steps - n) << 8) / steps + a2 * (n << 8) / steps + 31) >> 8) as u8;
    if alpha1 > alpha2 {
        [alpha1, alpha2, lerp(1, 7), lerp(2, 7), lerp(3, 7), lerp(4, 7), lerp(5, 7), lerp(6, 7)]
    } else {
        [alpha1, alpha2, lerp(1, 5), lerp(2, 5), lerp(3, 5), lerp(4, 5), 0, 255]
    }
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

/// Widens `count` RGBA texels (0xAABBGGRR words) stored as u16 pairs in the
/// second half of `texels` (from index 2 x `count`) into four u16 channels
/// each from the front, k -> k x 257 (= k / 255 when the GPU reads them:
/// `texture_cache::gpu::GpuFormat::Wide16`). In place: texel i is read
/// before its 4 channels are written, and those end at 4 i + 4 <= 2 count +
/// 2 i + 2, never past a texel not yet read.
pub fn widen_rgba(texels: &mut [u16], count: usize) {
    for i in 0..count {
        let at = 2 * count + 2 * i;
        let rgba = texels[at] as u32 | (texels[at + 1] as u32) << 16;
        for channel in 0..4 {
            texels[4 * i + channel] = ((rgba >> (8 * channel)) & 0xFF) as u16 * 257;
        }
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
    fn widens_rgba_in_place() {
        let words: Vec<u32> = (0..37u32).map(|i| i.wrapping_mul(0x9E37_79B9) ^ 0x00FF_0080).collect();
        let mut texels = vec![0u16; words.len() * 4];
        for (i, word) in words.iter().enumerate() {
            texels[2 * words.len() + 2 * i] = *word as u16;
            texels[2 * words.len() + 2 * i + 1] = (*word >> 16) as u16;
        }
        widen_rgba(&mut texels, words.len());
        for (i, word) in words.iter().enumerate() {
            for channel in 0..4 {
                assert_eq!(texels[4 * i + channel], ((word >> (8 * channel)) & 0xFF) as u16 * 257, "texel {i} channel {channel}");
            }
        }
    }

    /// The wide textures' premise (DESIGN "sixty-fifth session"): a palette
    /// texture decoded to RGBA with its palette has the colours the GPU
    /// shows for its indices through the palette table (`palette_rgba`),
    /// for every CLUT format (entry formats, shift, mask, start) and with
    /// IGNORE_ALPHA.
    #[test]
    fn rgba_decode_of_palette_textures_matches_the_palette_tables() {
        let mut seed = 0x1234_5678u32;
        let mut next = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            seed
        };
        for round in 0..400 {
            let format = if round % 2 == 0 { format::CLUT4 } else { format::CLUT8 };
            let (width, height) = (16u16 << (round % 3), 8u16 << (round % 2));
            let mut descriptor = texture(format, width, height);
            descriptor.swizzled = (round % 4 >= 2) as u8;
            descriptor.flags = if round % 5 == 0 { texture_flags::IGNORE_ALPHA } else { 0 };
            let (entry_format, shift, mask, start) = (next() % 4, next() % 8, next() & 0xFF, next() % 32);
            descriptor.clut_format = entry_format | shift << 2 | mask << 8 | start << 16;
            let palette: Vec<u32> = (0..(next() % 300)).map(|_| next()).collect();
            let data: Vec<u8> = (0..data_size(&descriptor).unwrap() as usize).map(|_| next() as u8).collect();
            let mut rgba = vec![0u32; width as usize * height as usize];
            decode_texture(&data, &descriptor, &palette, &mut rgba, width as usize).unwrap();
            let mut indices = vec![0u8; width as usize * height as usize];
            decode_palette_indices(&data, &descriptor, &mut indices, width as usize).unwrap();
            let mut table = [0u32; 256];
            palette_rgba(&descriptor, &palette, &mut table);
            for (i, (color, index)) in rgba.iter().zip(&indices).enumerate() {
                assert_eq!(*color, table[*index as usize], "round {round} texel {i}");
            }
        }
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
    fn rejects_invalid_formats() {
        // 8-10 are DXT (dxt_tests), 11-15 don't exist.
        let data = bytes(&[0; 64]);
        let mut out = [0u32; 16];
        for format in 11..16 {
            assert_eq!(
                decode_texture(&data, &texture(format, 4, 4), &[], &mut out, 4),
                Err(DecodeError::UnsupportedFormat)
            );
            assert_eq!(data_size(&texture(format, 4, 4)), None);
        }
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

    /// The GE's float-bits log2 and the slope mode's level, with Kenka
    /// Bancho's train numbers (TEXLODSLOPE 0x3e4ccc = 0.2, TEXLEVEL 0xc80002:
    /// slope, offset -56/16), worked out by hand from ppsspp-ref's rule.
    #[test]
    fn slope_mode_levels_as_the_ge_picks_them() {
        assert_eq!([ge_log16(1.0), ge_log16(2.0), ge_log16(0.5), ge_log16(1.5)], [0, 16, -16, 8]);
        // 0.2 = 1.6 x 2^-3: -48 + floor(0.6 x 16); 0.01 = 1.28 x 2^-7; 0.001 = 1.024 x 2^-10.
        assert_eq!([ge_log16(f32::from_bits(0x3e4ccc00)), ge_log16(0.01), ge_log16(0.001)], [-39, -108, -160]);
        let base = slope_base(0x3e4ccc, 0xc80002);
        assert_eq!(base, 16 - 39 - 56);
        // w 10: detail -24, level 0; w 100: 29 = level 1 and 13/16; w 1000: 81, capped at level 2.
        assert_eq!(slope_level(base, 0.1, 2, true), (0, 0));
        assert_eq!(slope_level(base, 0.01, 2, true), (1, 13));
        assert_eq!(slope_level(base, 0.001, 2, true), (2, 0));
        // Without blending between levels: rounded at half a level (29 + 8 = 37 -> 2).
        assert_eq!(slope_level(base, 0.01, 2, false), (2, 0));
        assert_eq!(slope_level(base, 0.001, 0, true), (0, 0), "no levels");
    }

    /// With `SLOPE_MIP`, slope-mode draws get their levels whatever the
    /// menu option, with their base and no bias; other modes as before.
    #[test]
    fn slope_mode_draws_take_their_levels_with_the_setting() {
        let (mut ge, base) = ge_with_levels(7, 0xc80002);
        ge.set_register(crate::command::Command(((op::TEXTURE_LOD_SLOPE as u32) << 24) | 0x3e4ccc));
        assert_eq!(describe_levels_with(&ge, &base, 0, false), None, "setting off, Sharp: none");
        let slope = describe_levels_with(&ge, &base, 0, true).unwrap();
        assert_eq!((slope.count, slope.linear, slope.bias_eighths, slope.slope_base), (4, true, 0, Some(-79)));
        assert_eq!(describe_levels_with(&ge, &base, 1, true).unwrap().slope_base, Some(-79), "any menu mode");
        let (ge, base) = ge_with_levels(7, 0); // auto: the setting changes nothing
        assert_eq!(describe_levels_with(&ge, &base, 0, true), None);
        assert_eq!(describe_levels_with(&ge, &base, 1, true).unwrap().slope_base, None);
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

/// DXT1/3/5 (Need for Speed Underground Rivals' title and cars): expected
/// values worked out by hand from PPSSPP's decoder (GPU/Common/
/// TextureDecoder.cpp, DXTDecoder; layout DecodeDXTBlocks).
#[cfg(test)]
mod dxt_tests {
    use super::*;

    fn dxt(format: u8, width: u16, height: u16, buffer_width: u32) -> TextureDescriptor {
        TextureDescriptor { address: 0x0900_0000, buffer_width, width, height, format, ..Default::default() }
    }

    /// A PSP DXT1 block: the four index rows first, then the two colours.
    fn block1(lines: [u8; 4], color1: u16, color2: u16) -> Vec<u8> {
        let mut block = lines.to_vec();
        block.extend_from_slice(&color1.to_le_bytes());
        block.extend_from_slice(&color2.to_le_bytes());
        block
    }

    fn decode(data: &[u8], texture: &TextureDescriptor) -> Vec<u32> {
        let (width, height) = (texture.width as usize, texture.height as usize);
        let mut out = vec![0u32; width * height];
        decode_texture(data, texture, &[], &mut out, width).unwrap();
        out
    }

    #[test]
    fn dxt1_four_colours_when_the_first_colour_is_larger() {
        // Row 0 uses indices 0, 1, 2, 3; red 0xF800 against blue 0x001F.
        let data = block1([0b11_10_01_00, 0, 0, 0], 0xF800, 0x001F);
        let out = decode(&data, &dxt(format::DXT1, 4, 4, 4));
        // (2 * 0xF8 + 0) / 3 = 0xA5, (0xF8 + 0) / 3 = 0x52.
        assert_eq!(&out[..4], &[0xFF00_00F8, 0xFFF8_0000, 0xFF52_00A5, 0xFFA5_0052]);
        assert!(out[4..].iter().all(|&texel| texel == 0xFF00_00F8));
    }

    #[test]
    fn dxt1_three_colours_and_transparent_black_otherwise() {
        let data = block1([0b11_10_01_00, 0, 0, 0], 0x001F, 0xF800);
        let out = decode(&data, &dxt(format::DXT1, 4, 4, 4));
        assert_eq!(&out[..4], &[0xFFF8_0000, 0xFF00_00F8, 0xFF7C_007C, 0x0000_0000]);
    }

    #[test]
    fn dxt1_ignored_alpha_makes_the_transparent_texel_opaque() {
        let data = block1([0b11_00_00_00, 0, 0, 0], 0x001F, 0xF800);
        let texture = TextureDescriptor { flags: texture_flags::IGNORE_ALPHA, ..dxt(format::DXT1, 4, 4, 4) };
        assert_eq!(decode(&data, &texture)[3], 0xFF00_0000);
    }

    #[test]
    fn dxt3_alpha_is_the_nibble_shifted_up() {
        let mut data = block1([0, 0, 0, 0], 0xF800, 0x001F);
        for line in [0xF0A5u16, 0, 0, 0] {
            data.extend_from_slice(&line.to_le_bytes());
        }
        let out = decode(&data, &dxt(format::DXT3, 4, 4, 4));
        // PPSSPP: alpha << 28, the nibble not repeated (0xF -> 0xF0).
        assert_eq!(&out[..4], &[0x5000_00F8, 0xA000_00F8, 0x0000_00F8, 0xF000_00F8]);
        assert_eq!(out[4], 0x0000_00F8);
    }

    fn block5(alpha1: u8, alpha2: u8, indices: u64) -> Vec<u8> {
        let mut data = block1([0, 0, 0, 0], 0xF800, 0x001F);
        data.extend_from_slice(&(indices as u32).to_le_bytes());
        data.extend_from_slice(&((indices >> 32) as u16).to_le_bytes());
        data.extend_from_slice(&[alpha1, alpha2]);
        data
    }

    #[test]
    fn dxt5_eight_alphas_when_the_first_alpha_is_larger() {
        // Row 0 alpha indices 0, 1, 2, 7.
        let data = block5(255, 0, 1 << 3 | 2 << 6 | 7 << 9);
        let out = decode(&data, &dxt(format::DXT5, 4, 4, 4));
        let alphas: Vec<u32> = out[..4].iter().map(|texel| texel >> 24).collect();
        // lerp8: (255 * (6 << 8) / 7 + 31) >> 8 = 218, n = 6: 36.
        assert_eq!(alphas, [255, 0, 218, 36]);
        assert!(out.iter().all(|texel| texel & 0x00FF_FFFF == 0x0000_00F8));
    }

    #[test]
    fn dxt5_six_alphas_plus_zero_and_full_otherwise() {
        let data = block5(0, 255, 1 << 3 | 2 << 6 | 6 << 9 | 7 << 12);
        let texture = dxt(format::DXT5, 8, 4, 8);
        let mut data = data;
        data.extend(block5(0, 255, 0));
        let out = decode(&data, &texture);
        let alphas: Vec<u32> = out[..5].iter().map(|texel| texel >> 24).collect();
        // lerp6: (255 * (1 << 8) / 5 + 31) >> 8 = 51; index 6 = 0, 7 = 255.
        assert_eq!(alphas, [0, 255, 51, 0, 0]);
        // Row 1 starts at bit 12: index 7.
        assert_eq!(out[8] >> 24, 255);
    }

    #[test]
    fn dxt_blocks_are_laid_out_in_rows_of_the_buffer_width() {
        // 8x8 DXT1, buffer 8 wide: blocks 0 1 / 2 3, one colour each.
        let colors = [0xF800u16, 0x07E0, 0x001F, 0xFFFF];
        let data: Vec<u8> = colors.iter().flat_map(|&color| block1([0; 4], color, 0)).collect();
        let texture = dxt(format::DXT1, 8, 8, 8);
        assert_eq!(data_size(&texture), Some(32));
        let out = decode(&data, &texture);
        assert_eq!([out[0], out[4], out[4 * 8], out[4 * 8 + 4]], [0xFF00_00F8, 0xFF00_FC00, 0xFFF8_0000, 0xFFF8_FCF8]);
        assert_eq!(out[3 * 8 + 3], 0xFF00_00F8);
    }

    #[test]
    fn dxt_rows_wider_than_the_buffer_run_on_into_the_next_row() {
        // 8 wide, buffer 4: one block per row, a row decodes two.
        let colors = [0xF800u16, 0x07E0, 0x001F];
        let data: Vec<u8> = colors.iter().flat_map(|&color| block1([0; 4], color, 0)).collect();
        let texture = dxt(format::DXT1, 8, 8, 4);
        assert_eq!(data_size(&texture), Some(24));
        let out = decode(&data, &texture);
        assert_eq!([out[0], out[4], out[4 * 8], out[4 * 8 + 4]], [0xFF00_00F8, 0xFF00_FC00, 0xFF00_FC00, 0xFFF8_0000]);
    }

    #[test]
    fn dxt_partial_blocks_and_ignored_swizzle() {
        // NFS UR: a 2x64 DXT5 strip, buffer 4.
        let data: Vec<u8> = (0..16).flat_map(|_| block5(255, 255, 0)).collect();
        let texture = TextureDescriptor { swizzled: 1, ..dxt(format::DXT5, 2, 64, 4) };
        assert_eq!(data_size(&texture), Some(16 * 16));
        let out = decode(&data, &texture);
        assert!(out.iter().all(|&texel| texel == 0xFF00_00F8));
    }

    #[test]
    fn opaque_texels_from_the_source() {
        // DXT1 in 4-colour mode, or 3-colour mode without index 3: opaque.
        let four = block1([0xFF; 4], 0xF800, 0x001F);
        let three_unused = block1([0b10_01_00_10; 4], 0x001F, 0xF800);
        let three_used = block1([0b11_00_00_00, 0, 0, 0], 0x001F, 0xF800);
        let texture = dxt(format::DXT1, 8, 4, 8);
        let both = |a: &[u8], b: &[u8]| [a, b].concat();
        assert!(all_texels_opaque(&both(&four, &three_unused), &texture));
        assert!(!all_texels_opaque(&both(&four, &three_used), &texture));
        // Each opaque claim matches the decoded texels.
        for data in [both(&four, &three_unused), both(&four, &three_used)] {
            assert_eq!(all_texels_opaque(&data, &texture), decode(&data, &texture).iter().all(|texel| texel >> 24 == 0xFF));
        }
        // DXT5 with alphas 255, 255 (the 6-alpha mode): indices 0..5 and 7
        // give 255, index 6 gives 0.
        let full = block5(255, 255, 7 | 7 << 3);
        assert!(all_texels_opaque(&full, &dxt(format::DXT5, 4, 4, 4)));
        let six = block5(255, 255, 6);
        assert!(!all_texels_opaque(&six, &dxt(format::DXT5, 4, 4, 4)));
        // DXT3 never; 5650 and ignored alpha always; others not looked at.
        assert!(!all_texels_opaque(&[0xFF; 16], &dxt(format::DXT3, 4, 4, 4)));
        assert!(all_texels_opaque(&[], &TextureDescriptor { format: format::BGR5650, ..dxt(format::DXT1, 4, 4, 4) }));
        assert!(all_texels_opaque(&three_used, &TextureDescriptor { flags: texture_flags::IGNORE_ALPHA, ..dxt(format::DXT1, 4, 4, 4) }));
        assert!(!all_texels_opaque(&[0xFF; 64], &TextureDescriptor { format: format::ABGR8888, ..dxt(format::DXT1, 4, 4, 4) }));
    }

    #[test]
    fn opaque_rows_from_the_top() {
        let four = block1([0xFF; 4], 0xF800, 0x001F);
        // Index 3 in 3-colour mode on the block's texel row 2.
        let row2 = block1([0, 0, 0b11 << 4, 0], 0x001F, 0xF800);
        // 8x12, buffer 8: three block rows of two blocks.
        let texture = dxt(format::DXT1, 8, 12, 8);
        let data = [&four[..], &four, &four, &row2, &four, &four].concat();
        assert_eq!(opaque_rows(&data, &texture), 4 + 2);
        let out = decode(&data, &texture);
        assert_eq!(out.iter().position(|texel| texel >> 24 != 0xFF).unwrap() / 8, 6);
        assert_eq!(opaque_rows(&four.repeat(6), &texture), ALL_ROWS_OPAQUE);
        // Run-on: an 8 wide row of a 4 buffer reads the next row's block too.
        let run_on = dxt(format::DXT1, 8, 8, 4);
        assert_eq!(opaque_rows(&[&four[..], &row2, &four].concat(), &run_on), 2);
        // DXT5: alpha index 6 (0) on texel row 3 of the second block row.
        let six_on_row3 = block5(255, 255, 6u64 << (3 * 12));
        let full = block5(255, 255, 0);
        let data = [&full[..], &six_on_row3].concat();
        assert_eq!(opaque_rows(&data, &dxt(format::DXT5, 4, 8, 4)), 4 + 3);
        // Rows past the height aren't decoded: a 6 high texture is opaque.
        assert_eq!(opaque_rows(&data, &dxt(format::DXT5, 4, 6, 4)), ALL_ROWS_OPAQUE);
        assert_eq!(opaque_rows(&[0xFF; 16], &dxt(format::DXT3, 4, 4, 4)), 0);
    }

    #[test]
    fn dxt_data_sizes() {
        // NFS UR's title: 512x512 DXT1 with a 480 buffer, 32x32 DXT5 with 24.
        assert_eq!(data_size(&dxt(format::DXT1, 512, 512, 480)), Some((120 * 128 + 8) * 8));
        assert_eq!(data_size(&dxt(format::DXT5, 32, 32, 24)), Some((6 * 8 + 2) * 16));
        assert_eq!(data_size(&dxt(format::DXT3, 4, 4, 4)), Some(16));
        // Buffer width 0 counts as 16 bytes: 32 DXT1 texels.
        assert_eq!(data_size(&dxt(format::DXT1, 32, 4, 0)), Some(8 * 8));
        assert_eq!(compact_layout(&dxt(format::DXT1, 32, 32, 64)), None);
    }
}
