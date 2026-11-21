//! Re-checking big textures on a worker thread.
//!
//! Final Fantasy's battle intro rewrites three 512x512 ray textures all the
//! time. Copying one out of PSP memory and decoding it takes ~6.5 ms; done
//! on the display thread, every third frame took 8-11 ms to render and was
//! shown late (frames 22/11 ms apart instead of 17/17).
//!
//! So big textures (`BACKGROUND_MIN_BYTES`) are re-checked by a worker
//! thread on another core instead: the display thread hands it the texture
//! and a fresh GPU texture ([`RefreshJob`]) and keeps drawing with the old
//! pixels meanwhile. The worker ([`run_refresh`]) copies and hashes the
//! data; if it changed (or is known to keep changing, see
//! `VOLATILE_CHANGES`) it decodes it into the fresh texture, which the
//! display thread swaps in once the worker is done, a frame or two later
//! (`TextureCache::finish_refresh`). One job at a time.
//!
//! Only direct colour formats: palette textures need the frame's palette
//! words, which change with the frame.

use crate::frame::TextureDescriptor;
use crate::texture::{self, format, DecodeError};

use super::gpu::{GpuFormat, GpuTexture, TextureShape};
use super::staging::{stage, StagedLayout};
use super::{Host, TextureCache, BACKGROUND_MIN_BYTES, HASH_VOLATILE_EVERY, VOLATILE_CHANGES};

/// What the display thread hands the worker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RefreshJob {
    pub key: TextureDescriptor,
    pub format: GpuFormat,
    /// The fresh texture the worker decodes into.
    pub texture: GpuTexture,
    pub smooth: bool,
    /// Hash the data first; otherwise just decode (volatile textures).
    pub hash: bool,
    /// With `hash`: skip the decode if the data still has this hash ...
    pub old_hash: u32,
    /// ... unless this is set (the texture may show other pixels).
    pub decode_anyway: bool,
}

impl RefreshJob {
    fn shape(&self) -> TextureShape {
        TextureShape::new(&self.key, self.format, false)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefreshOutcome {
    /// The fresh texture holds the new pixels.
    Decoded,
    /// Hashed, same data as before: nothing decoded.
    Unchanged,
    /// The data is outside PSP memory (or too big to stage).
    Unreadable,
    DecodeFailed(DecodeError),
}

/// What the worker hands back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RefreshResult {
    pub outcome: RefreshOutcome,
    /// With `RefreshJob::hash`: the hash of the copied data.
    pub data_hash: u32,
    pub work_us: u32,
    /// `Host::captured_frame_number` right after the copy: the copied data
    /// may be newer than what this frame and earlier ones drew with.
    pub newer_than_frame: u32,
    /// Decoded: rows from the top whose texels all have alpha 255
    /// (`texture::opaque_rows`).
    pub opaque_rows: u16,
}

/// The job as the display thread remembers it.
#[derive(Clone, Copy)]
pub(super) struct PendingRefresh {
    /// The cache entry to update ...
    pub entry: usize,
    /// ... if it still holds this texture (it may have been evicted or
    /// reused for another texture meanwhile).
    pub old_texture: Option<GpuTexture>,
    pub job: RefreshJob,
}

/// The worker's part: copies the job's data into `staging` (the worker's
/// own buffer), hashes it if asked, and decodes it into the job's texture.
pub fn run_refresh<H: Host>(host: &mut H, job: &RefreshJob, staging: &mut [u8]) -> RefreshResult {
    let start = host.now_us();
    let mut result =
        RefreshResult { outcome: RefreshOutcome::Unreadable, data_hash: 0, work_us: 0, newer_than_frame: 0, opaque_rows: 0 };
    let layout = StagedLayout::whole(&job.key);
    let size = stage(host, &job.key, &layout, staging) as usize;
    if size == 0 {
        return result;
    }
    result.newer_than_frame = host.captured_frame_number();
    let data = &staging[..size];
    if job.hash {
        result.data_hash = texture::hash_texture(data);
        if result.data_hash == job.old_hash && !job.decode_anyway {
            result.outcome = RefreshOutcome::Unchanged;
            result.work_us = (host.now_us() - start) as u32;
            return result;
        }
    }
    result.opaque_rows = texture::opaque_rows(data, &job.key);
    result.outcome = match host.decode(job.texture, job.format, &job.key, data, &[], job.smooth, 0) {
        Ok(()) => RefreshOutcome::Decoded,
        Err(error) => RefreshOutcome::DecodeFailed(error),
    };
    result.work_us = (host.now_us() - start) as u32;
    result
}

impl TextureCache {
    pub(super) fn can_refresh_in_background<H: Host>(&self, host: &H, texture: &TextureDescriptor) -> bool {
        host.has_refresh_worker()
            && texture.format < format::CLUT4
            && crate::frame::texture_level_count(texture) == 1 // the worker decodes level 0 only
            && texture::data_size(texture).unwrap_or(0) >= BACKGROUND_MIN_BYTES
    }

    /// Hands cache entry `slot` (whose texture is `wanted`) to the worker.
    /// False if the worker is busy or there's no memory for the fresh
    /// texture; it is tried again next frame.
    pub(super) fn start_refresh<H: Host>(&mut self, host: &mut H, slot: usize, wanted: &TextureDescriptor) -> bool {
        if self.refresh.is_some() {
            return false;
        }
        let format = GpuFormat::for_texture(wanted, self.smooth);
        let Some(texture) = self.acquire(host, TextureShape::new(wanted, format, false)) else {
            return false;
        };
        let entry = &self.entries[slot];
        let job = RefreshJob {
            key: *wanted,
            format,
            texture,
            smooth: self.smooth,
            // Volatile textures are hashed only every HASH_VOLATILE_EVERY-th
            // time, as on the display thread (see `look_up`).
            hash: entry.changes < VOLATILE_CHANGES || entry.unhashed as u32 + 1 >= HASH_VOLATILE_EVERY,
            old_hash: entry.data_hash,
            decode_anyway: entry.unhashed > 0,
        };
        self.refresh = Some(PendingRefresh { entry: slot, old_texture: entry.texture, job });
        host.start_refresh(&job);
        true
    }

    /// Once per frame, before textures are looked up: puts a finished
    /// refresh into its cache entry, unless the entry changed meanwhile
    /// (evicted, reused for another texture, other filter mode).
    ///
    /// New pixels are only used for frames captured after the copy was
    /// made. An earlier frame drew before the game changed the texture:
    /// Final Fantasy reuses its font texture for the field right after the
    /// menu's last frame, and that frame showed garbage instead of text
    /// when a copy made after the change was swapped in for it.
    pub(super) fn finish_refresh<H: Host>(&mut self, host: &mut H, frame_number: u32) {
        let Some(pending) = self.refresh else { return };
        let Some(result) = host.finished_refresh() else { return };
        if result.outcome == RefreshOutcome::Decoded && (frame_number.wrapping_sub(result.newer_than_frame) as i32) <= 0 {
            return; // until a frame from after the copy is drawn
        }
        let job = pending.job;
        let entry = self.entries[pending.entry];
        let current = entry.texture.is_some()
            && entry.texture == pending.old_texture
            && self.same_texture(&entry.key, &job.key)
            && entry.smooth == job.smooth;
        if current {
            self.stats.refreshes += 1;
            self.stats.refresh_us += result.work_us as u64;
        }
        if current && result.outcome == RefreshOutcome::Unchanged {
            let entry = &mut self.entries[pending.entry];
            entry.changes = 0;
            entry.unhashed = 0;
        }
        let swapped =
            current && result.outcome == RefreshOutcome::Decoded && self.retire(entry.texture.unwrap(), entry.shape);
        if swapped {
            let entry = &mut self.entries[pending.entry];
            entry.texture = Some(job.texture);
            entry.shape = job.shape();
            entry.opaque_rows = result.opaque_rows;
            entry.quick = 0; // copied at some point: its sample isn't known
            if job.hash {
                // Same data as at the last hash: no longer volatile.
                let changed = result.data_hash != entry.data_hash;
                entry.changes = if changed { entry.changes.saturating_add(1) } else { 0 };
                entry.data_hash = result.data_hash;
                entry.unhashed = 0;
            } else {
                entry.unhashed = entry.unhashed.saturating_add(1);
            }
            host.log(format_args!(
                "refreshed: addr={:08x} {}x{} in {} us{}\n",
                job.key.address,
                job.key.width,
                job.key.height,
                result.work_us,
                if job.hash { ", hashed" } else { "" }
            ));
        } else if !self.retire(job.texture, job.shape()) {
            self.free_now(host, job.texture, job.shape()); // the GPU never saw it
        }
        self.refresh = None;
        host.end_refresh();
    }
}
