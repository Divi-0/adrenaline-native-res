//! Reading texture data out of PSP memory.
//!
//! The Vita maps PSP RAM uncached, so the CPU reads it extremely slowly
//! (hashing 1 MiB of textures took ~190 ms per frame). DMA copies are fast,
//! so each texture's data is first copied into a normal, cached buffer (the
//! *staging* buffer) and hashed / decoded there.

use crate::frame::TextureDescriptor;
use crate::texture;

use super::{Host, TextureCache, FRAME_HASHES};

/// Size of a staging buffer: the largest texture, 512x512 x 4 bytes.
pub const STAGING_SIZE: usize = 1024 * 1024;

/// Big direct-colour textures are compared by one word per this many bytes
/// at every use (see [`sample_hash`]).
pub const SAMPLE_STEP: u32 = 1024;

/// What the staging buffer holds for a texture. A texture whose buffer is
/// wider than itself (Final Fantasy's world map tiles: 32x32 windows into a
/// 512-wide sheet) is staged as just its own texels, `pieces` runs of
/// `piece_bytes` that lie `source_step` apart in PSP memory; `as_texture`
/// then describes them (buffer width = width). Others are staged in one
/// piece as they are. See `texture::compact_layout`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StagedLayout {
    /// Decode / hash the staged data as this texture.
    pub as_texture: TextureDescriptor,
    /// Bytes staged; 0: unsupported format.
    pub size: u32,
    pub pieces: u32,
    pub piece_bytes: u32,
    pub source_step: u32,
}

impl StagedLayout {
    /// The compact layout where there is one, otherwise the whole data.
    pub fn of(texture: &TextureDescriptor) -> StagedLayout {
        match texture::compact_layout(texture) {
            Some(compact) => StagedLayout {
                as_texture: compact.compact,
                size: compact.pieces * compact.piece_bytes,
                pieces: compact.pieces,
                piece_bytes: compact.piece_bytes,
                source_step: compact.source_step,
            },
            None => StagedLayout::whole(texture),
        }
    }

    /// The texture's data in one piece, as it lies in PSP memory.
    pub fn whole(texture: &TextureDescriptor) -> StagedLayout {
        let size = texture::data_size(texture).unwrap_or(0);
        StagedLayout { as_texture: *texture, size, pieces: 1, piece_bytes: size, source_step: 0 }
    }

    /// Bytes of PSP memory the pieces span.
    fn span(&self) -> u32 {
        (self.pieces - 1) * self.source_step + self.piece_bytes
    }
}

/// Copies the data of `texture` into `staging` as `layout` describes.
/// Returns its size, or 0 if the texture is unsupported, too big or
/// outside PSP memory.
pub fn stage<H: Host>(host: &H, texture: &TextureDescriptor, layout: &StagedLayout, staging: &mut [u8]) -> u32 {
    if layout.size == 0 || layout.size as usize > staging.len() {
        return 0;
    }
    let Some(source) = host.psp_memory(texture.address & 0x0FFF_FFFF, layout.span()) else {
        return 0;
    };
    let piece_bytes = layout.piece_bytes as usize;
    for piece in 0..layout.pieces as usize {
        let from = piece * layout.source_step as usize;
        let to = piece * piece_bytes;
        host.copy(&mut staging[to..to + piece_bytes], &source[from..from + piece_bytes]);
    }
    layout.size
}

/// A cheap check of a big direct-colour texture for changes, done at every
/// use: one word per [`SAMPLE_STEP`] bytes, read straight from PSP memory,
/// at a different column in each step (~0.4 ms for 512 KiB). 0 if the data
/// can't be read.
///
/// Final Fantasy's battle-start rays draw the previous frames, which the
/// game copies into one of three 512 KiB textures every few frames; with
/// only the regular re-checks (every `VERIFY_INTERVAL` frames) we drew
/// snapshots several frames old and the effect looked completely
/// different. Changes the samples miss are still found by those re-checks.
pub fn sample_hash<H: Host>(host: &H, texture: &TextureDescriptor) -> u32 {
    let size = texture::data_size(texture).unwrap_or(0);
    let Some(words) = host.psp_words(texture.address & 0x0FFF_FFFF, size) else {
        return 0;
    };
    let mut hash = 0x811C_9DC5u32;
    let mut at = 0;
    let mut step = 0;
    while at + SAMPLE_STEP <= size {
        let offset = at + (step * 68) % SAMPLE_STEP;
        hash = (hash ^ words[(offset / 4) as usize]).wrapping_mul(0x0100_0193);
        at += SAMPLE_STEP;
        step += 1;
    }
    hash
}

/// `texture::quick_sample` of the texture's data as it is now (0 if it
/// can't be read): what a cached texture was decoded from, compared with
/// the frame's capture-time sample (`Frame::texture_samples`).
pub fn quick_hash<H: Host>(host: &H, texture: &TextureDescriptor) -> u32 {
    let Some(positions) = texture::quick_sample_positions(texture) else {
        return 0;
    };
    let size = texture::data_size(texture).unwrap_or(0);
    let Some(data) = host.psp_words(texture.address & 0x0FFF_FFFF, size) else {
        return 0;
    };
    if positions.iter().any(|&position| position >= data.len()) {
        return 0;
    }
    texture::quick_sample(&positions.map(|position| data[position]))
}

/// `quick_hash` of the data staged for `texture` (by [`stage`] with
/// `layout`, in `staged`): the sample of the very bytes hashed and decoded,
/// not of PSP memory a moment later. The game runs on while the display
/// copies its textures out; MTX loads the next screen's tiles into the same
/// RAM meanwhile, and a sample read after the copy described other pixels
/// than the cached ones (rider <-> bike select, 2026-11-17: half-loaded
/// tiles kept for VERIFY_INTERVAL frames). Words outside the staged pieces
/// (the gaps of a compact layout, not the texture's texels) are read from
/// PSP memory. 0 if one can't be read.
pub fn staged_quick_hash<H: Host>(host: &H, texture: &TextureDescriptor, layout: &StagedLayout, staged: &[u8]) -> u32 {
    let Some(positions) = texture::quick_sample_positions(texture) else {
        return 0;
    };
    let mut words = [0u32; texture::QUICK_SAMPLES];
    for (word, &position) in words.iter_mut().zip(positions.iter()) {
        let offset = 4 * position as u32;
        let (piece, within) =
            if layout.source_step == 0 { (0, offset) } else { (offset / layout.source_step, offset % layout.source_step) };
        let at = (piece * layout.piece_bytes + within) as usize;
        if piece < layout.pieces && within + 4 <= layout.piece_bytes && at + 4 <= staged.len() {
            *word = u32::from_le_bytes([staged[at], staged[at + 1], staged[at + 2], staged[at + 3]]);
        } else {
            match host.psp_words((texture.address & 0x0FFF_FFFF) + offset, 4) {
                Some(read) => *word = read[0],
                None => return 0,
            }
        }
    }
    texture::quick_sample(&words)
}

/// A hash of texture data already computed this frame: the same pixels are
/// often used several times (with different palettes, say).
#[derive(Clone, Copy)]
pub(super) struct FrameHash {
    pub address: u32,
    pub size: u32,
    /// `texture::data_size` of the texture hashed (the quick sample's
    /// positions depend on it).
    pub data_size: u32,
    pub hash: u32,
    /// `staged_quick_hash` of the same staged bytes.
    pub quick: u32,
}

impl TextureCache {
    /// Hash of `texture`'s data (staged as `layout` says), staging it if it
    /// isn't known yet this frame. Sets `*staged` to the data size when
    /// `staging` now holds this texture (0 if it doesn't), and `*quick` to
    /// the quick sample of the data hashed (`staged_quick_hash`; 0 with
    /// hash 0).
    pub(super) fn data_hash<H: Host>(
        &mut self,
        host: &H,
        staging: &mut [u8],
        texture: &TextureDescriptor,
        layout: &StagedLayout,
        staged: &mut u32,
        quick: &mut u32,
    ) -> u32 {
        *staged = 0;
        *quick = 0;
        let data_size = texture::data_size(texture).unwrap_or(0);
        let known = &self.frame_hashes[..self.frame_hash_count];
        if let Some(known) = known
            .iter()
            .find(|known| known.address == texture.address && known.size == layout.size && known.data_size == data_size)
        {
            *quick = known.quick;
            return known.hash;
        }
        let size = stage(host, texture, layout, staging);
        if size == 0 {
            return 0;
        }
        *staged = size;
        let data = &staging[..size as usize];
        let hash = texture::hash_texture(data);
        *quick = staged_quick_hash(host, texture, layout, data);
        self.stats.hashed_bytes += size as u64;
        if self.frame_hash_count < FRAME_HASHES {
            self.frame_hashes[self.frame_hash_count] =
                FrameHash { address: texture.address, size, data_size, hash, quick: *quick };
            self.frame_hash_count += 1;
        }
        hash
    }
}
