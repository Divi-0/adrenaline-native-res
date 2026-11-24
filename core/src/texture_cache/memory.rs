//! Memory for GPU textures: the budget, retired textures, and evicting the
//! least recently used.
//!
//! A texture taken out of the cache may still be read by the GPU for the
//! frame in flight, so it is *retired*: kept for `RETIRE_FRAMES` frames
//! before it may be reused or freed. Retired textures are reused for new
//! textures of the same shape: games that change a texture all the time
//! (Final Fantasy's menu redraws a 512x512 one) would otherwise make us
//! allocate and free megabytes per second, until memory runs out.
//!
//! Where a texture's memory comes from:
//! - the texture arena (see `arena`), first choice: pages of one big
//!   phycont block. When it gets nearly full (`ARENA_SLACK`), the least
//!   recently used textures leave it.
//! - a memory block of its own in main memory: render-target textures
//!   (their memory must be GPU-writable), and others when the arena is full
//!   or doesn't exist. The PSP emulator leaves the app little main memory
//!   (~29 MiB at the start, ~3 MiB in MTX's race), and a fixed texture
//!   budget ran it out of memory in battles (enemies invisible, flicker).
//!   So for these the cache follows the real free memory: keep
//!   `MEMORY_RESERVE` free for everything else (screenshots, dumps,
//!   vita2d), and never let a new texture take the last `MEMORY_MINIMUM`.
//!   Only textures of this kind are given up for main memory: arena
//!   textures don't use any.

use super::gpu::{GpuTexture, TextureShape};
use super::{Host, TextureCache, CACHE_SIZE};

/// Memory for decoded textures, cached and retired (RGBA8888: a 256x256
/// texture is 256 KiB). 24 MiB ran the PSP emulator out of memory in Final
/// Fantasy's menu.
pub const BUDGET_BYTES: u32 = 16 * 1024 * 1024;
/// A replaced texture may still be read by the GPU for the frame in flight,
/// so it is only reused or freed this many frames later.
pub const RETIRE_FRAMES: u32 = 3;
/// Retired textures nobody reused for this long are freed.
pub const SPARE_FRAMES: u32 = 120;
pub const MEMORY_RESERVE: u32 = 6 * 1024 * 1024;
pub const MEMORY_MINIMUM: u32 = 2 * 1024 * 1024;
/// Room kept free in the texture arena for new textures: below it, the
/// least recently used arena textures are retired (their pages are free
/// `RETIRE_FRAMES` later). MTX's camera cuts bring up to ~90 new textures,
/// mostly small.
pub const ARENA_SLACK: u32 = 1024 * 1024;
/// Textures leave a full arena only when no frame used them for this long:
/// fewer (or none) leave rather than ones that come back right away.
pub const ARENA_IDLE_FRAMES: u32 = 30;

/// A texture taken out of the cache, see the module description.
#[derive(Clone, Copy)]
pub(super) struct Retired {
    /// None: a free slot.
    pub texture: Option<GpuTexture>,
    pub shape: TextureShape,
    /// The render count from which the GPU is done with it.
    pub ready_at: u32,
}

/// Which textures a search is about, by where their memory is.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Memory {
    Any,
    /// In the texture arena.
    Arena,
    /// In a memory block of its own (main memory).
    OwnBlock,
}

impl Memory {
    fn holds(self, texture: GpuTexture) -> bool {
        match self {
            Memory::Any => true,
            Memory::Arena => texture.arena_offset.is_some(),
            Memory::OwnBlock => texture.arena_offset.is_none(),
        }
    }
}

impl TextureCache {
    fn is_ready(&self, spare: &Retired) -> bool {
        spare.texture.is_some() && self.render_count >= spare.ready_at
    }

    /// Gives a texture's memory back: its arena pages, or its block.
    fn release<H: Host>(&mut self, host: &mut H, texture: GpuTexture, shape: TextureShape) {
        host.free_texture(texture);
        if let Some(offset) = texture.arena_offset {
            self.arena.free(offset, shape.bytes());
        }
        self.bytes -= shape.bytes();
    }

    fn free_retired<H: Host>(&mut self, host: &mut H, slot: usize) {
        let spare = self.retired[slot];
        if let Some(texture) = spare.texture {
            self.retired[slot].texture = None;
            self.release(host, texture, spare.shape);
        }
    }

    /// Frees one retired texture in `memory` the GPU is done with. False if
    /// there is none.
    fn free_one_retired<H: Host>(&mut self, host: &mut H, memory: Memory) -> bool {
        let found = (0..CACHE_SIZE).find(|&slot| {
            let spare = &self.retired[slot];
            self.is_ready(spare) && memory.holds(spare.texture.unwrap())
        });
        match found {
            Some(slot) => {
                self.free_retired(host, slot);
                true
            }
            None => false,
        }
    }

    /// Puts a texture the GPU may still read into the retired list. False
    /// if the list is full.
    pub(super) fn retire(&mut self, texture: GpuTexture, shape: TextureShape) -> bool {
        let ready_at = self.render_count + RETIRE_FRAMES;
        match self.retired.iter_mut().find(|spare| spare.texture.is_none()) {
            Some(spare) => {
                *spare = Retired { texture: Some(texture), shape, ready_at };
                true
            }
            None => false,
        }
    }

    /// Frees a texture the GPU never saw (or is done with).
    pub(super) fn free_now<H: Host>(&mut self, host: &mut H, texture: GpuTexture, shape: TextureShape) {
        self.release(host, texture, shape);
    }

    /// Moves cache entry `slot` into the retired list. If the list is full
    /// the texture stays cached, rather than free one the GPU may still read.
    pub(super) fn evict(&mut self, slot: usize) {
        let entry = self.entries[slot];
        if let Some(texture) = entry.texture {
            if self.retire(texture, entry.shape) {
                self.entries[slot].texture = None;
            }
        }
    }

    /// `evict` for room or memory (counted in the stats). False if the
    /// texture stays cached (retired list full).
    fn evict_old(&mut self, slot: usize) -> bool {
        self.evict(slot);
        let evicted = self.entries[slot].texture.is_none();
        self.stats.evictions += evicted as u32;
        evicted
    }

    /// Retires the least recently used texture in `memory` that no frame
    /// used for `idle_frames` frames. False if there is none (or the
    /// retired list is full).
    fn evict_idle(&mut self, memory: Memory, idle_frames: u32) -> bool {
        let Some(oldest) = self.least_recently_used(memory) else {
            return false;
        };
        if self.entries[oldest].last_used + idle_frames >= self.render_count {
            return false;
        }
        self.evict_old(oldest)
    }

    /// Frees the least recently used texture in `memory` that no frame used
    /// for `idle_frames` frames (at least `RETIRE_FRAMES`: the GPU is done
    /// with it) right away, for room now (`evict_idle` retires it, for room
    /// a few frames on). False if there is none.
    fn free_idle<H: Host>(&mut self, host: &mut H, memory: Memory, idle_frames: u32) -> bool {
        let Some(oldest) = self.least_recently_used(memory) else {
            return false;
        };
        let entry = self.entries[oldest];
        let (Some(texture), true) = (entry.texture, entry.last_used + idle_frames.max(RETIRE_FRAMES) < self.render_count) else {
            return false;
        };
        self.entries[oldest].texture = None;
        self.release(host, texture, entry.shape);
        self.stats.evictions += 1;
        true
    }

    /// Arena bytes a new texture can have: free pages, retired arena
    /// textures' and those of textures no frame used for `ARENA_IDLE_FRAMES`
    /// (`acquire` frees those two when the arena is full).
    pub(super) fn arena_reclaimable(&self) -> u32 {
        let retired: u32 = self
            .retired
            .iter()
            .filter(|spare| spare.texture.is_some_and(|texture| texture.arena_offset.is_some()))
            .map(|spare| spare.shape.bytes())
            .sum();
        let idle: u32 = self
            .entries
            .iter()
            .filter(|entry| {
                entry.texture.is_some_and(|texture| texture.arena_offset.is_some())
                    && entry.last_used + ARENA_IDLE_FRAMES.max(RETIRE_FRAMES) < self.render_count
            })
            .map(|entry| entry.shape.bytes())
            .sum();
        self.arena.free_bytes() + retired + idle
    }

    /// True if the arena has less than `ARENA_SLACK` free, counting the
    /// pages of retired arena textures as free (they are, a few frames on).
    fn arena_short(&self) -> bool {
        if !self.arena.exists() || self.arena.free_bytes() >= ARENA_SLACK {
            return false;
        }
        let retired: u32 = self
            .retired
            .iter()
            .filter(|spare| spare.texture.is_some_and(|texture| texture.arena_offset.is_some()))
            .map(|spare| spare.shape.bytes())
            .sum();
        self.arena.free_bytes() + retired < ARENA_SLACK
    }

    /// Once per frame: free retired textures nobody reused for a while, and
    /// any ready ones while we're over budget or short of memory. When
    /// memory is still short, retire the least recently used textures
    /// (freed a few frames later, once the GPU is done with them).
    pub(super) fn trim<H: Host>(&mut self, host: &mut H) {
        for slot in 0..CACHE_SIZE {
            let spare = self.retired[slot];
            // Render-target textures are kept until memory is short (below):
            // making one (uncached memory) takes ~6 ms, and Final Fantasy's
            // battle start needs several at once, every battle.
            if self.is_ready(&spare)
                && !spare.shape.render_target
                && self.render_count >= spare.ready_at + SPARE_FRAMES
            {
                self.free_retired(host, slot);
            }
        }
        while self.bytes > BUDGET_BYTES && self.free_one_retired(host, Memory::Any) {}

        // Main memory short: textures in blocks of their own give theirs up.
        while host.free_memory() < MEMORY_RESERVE && self.free_one_retired(host, Memory::OwnBlock) {}
        for _ in 0..8 {
            // Only textures the last frame didn't use: the current one may need the rest.
            if host.free_memory() >= MEMORY_RESERVE || !self.evict_idle(Memory::OwnBlock, 1) {
                break;
            }
        }

        // The arena nearly full: retired arena textures free their pages,
        // then textures unused for a while leave it.
        while self.arena.free_bytes() < ARENA_SLACK && self.free_one_retired(host, Memory::Arena) {}
        for _ in 0..8 {
            if !self.arena_short() || !self.evict_idle(Memory::Arena, ARENA_IDLE_FRAMES) {
                break;
            }
        }
    }

    /// A texture of this shape: a retired one if the GPU is done with it,
    /// otherwise a new one, in the arena if there is room (render targets:
    /// never), else in a block of its own. None if there is no memory for
    /// it even after freeing all ready retired textures.
    pub(super) fn acquire<H: Host>(&mut self, host: &mut H, shape: TextureShape) -> Option<GpuTexture> {
        if let Some(slot) = (0..CACHE_SIZE).find(|&slot| {
            let spare = &self.retired[slot];
            self.is_ready(spare) && spare.shape == shape
        }) {
            // Its bytes stay counted, now for the new user.
            return self.retired[slot].texture.take();
        }
        let bytes = shape.bytes();
        if !shape.render_target && self.arena.exists() {
            loop {
                if let Some(offset) = self.arena.allocate(bytes) {
                    let texture = host.create_texture(&shape, Some(offset));
                    if texture.is_some() {
                        self.bytes += bytes;
                        return texture;
                    }
                    self.arena.free(offset, bytes);
                    break;
                }
                // No room: retired arena textures the GPU is done with,
                // then ones no frame used for a while (NFS UR's attract
                // movie found the arena full of the title's textures).
                if !self.free_one_retired(host, Memory::Arena) && !self.free_idle(host, Memory::Arena, ARENA_IDLE_FRAMES) {
                    break;
                }
            }
        }
        while host.free_memory() < bytes + MEMORY_MINIMUM && self.free_one_retired(host, Memory::OwnBlock) {}
        let texture = if host.free_memory() >= bytes + MEMORY_MINIMUM { host.create_texture(&shape, None) } else { None };
        match texture {
            Some(_) => self.bytes += bytes,
            None => self.stats.create_failures += 1,
        }
        texture
    }

    /// The least recently used cached texture in `memory` that this frame
    /// doesn't need.
    pub(super) fn least_recently_used(&self, memory: Memory) -> Option<usize> {
        let mut oldest: Option<usize> = None;
        for (slot, entry) in self.entries.iter().enumerate() {
            if entry.texture.is_some_and(|texture| memory.holds(texture))
                && entry.last_used != self.render_count
                && oldest.map_or(true, |oldest| entry.last_used < self.entries[oldest].last_used)
            {
                oldest = Some(slot);
            }
        }
        oldest
    }

    /// A free cache slot with room for `bytes` more, evicting old textures
    /// if needed. None if everything cached is in use by this frame.
    pub(super) fn make_room<H: Host>(&mut self, host: &mut H, bytes: u32) -> Option<usize> {
        // Over budget: free retired textures first, then retire old cached
        // ones (their memory is freed once the GPU is done, see `trim`).
        while self.bytes + bytes > BUDGET_BYTES {
            if self.free_one_retired(host, Memory::Any) {
                continue;
            }
            let Some(oldest) = self.least_recently_used(Memory::Any) else {
                break; // everything is needed this frame: go over budget
            };
            if !self.evict_old(oldest) {
                break; // retired list full
            }
        }
        if let Some(free) = self.entries.iter().position(|entry| entry.texture.is_none()) {
            return Some(free);
        }
        let oldest = self.least_recently_used(Memory::Any)?;
        self.evict_old(oldest).then_some(oldest)
    }
}
