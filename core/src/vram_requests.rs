//! Which parts of PSP VRAM the Vita side asks the PSP plugin to copy, and
//! which parts of that copy are already there.
//!
//! Sony's emulator keeps PSP VRAM where the Vita side can't read it, but
//! the PSP plugin can: it copies the ranges listed in a shared table to the
//! same offsets in a copy area the Vita reads instead (protocol in
//! [`crate::psp_link`], "VRAM copy requests"). Two kinds of users:
//!
//! - **Textures** in VRAM, read by the display thread. Their ranges stay
//!   requested until no frame used them for [`KEEP_FRAMES`] frames, so the
//!   table doesn't change whenever a texture is skipped for a frame (every
//!   change makes the plugin copy the new ranges in full). The display
//!   thread waits for `copied_generation` itself before decoding.
//! - **The capture** (display lists, vertices, palettes in VRAM). MTX
//!   Mototrax CALLs a list at 0x04155000: an uncopied area reads as zeros,
//!   i.e. NOP commands, and the walk ran through 684 KiB of them. So every
//!   capture read of a 1 KiB block asks [`VramRequests::block_ready`]
//!   first. Blocks read within [`KEEP_FRAMES`] frames are requested (runs
//!   with small gaps joined); a block counts as ready once the plugin
//!   reported a generation copied that covers it. Reading an unready block
//!   fails (the list walk stops at a bad address), requests the next
//!   [`LOOKAHEAD_BLOCKS`] too (lists are read in order), and the frame
//!   counts as incomplete ([`VramRequests::take_missing`]).
//!
//! Once per frame the capture thread calls [`VramRequests::plan`], writes
//! the returned table if it differs from the shared one (with a new
//! generation) and reports that with [`VramRequests::published`].

/// PSP VRAM, physical addresses.
pub const VRAM_START: u32 = 0x0400_0000;
pub const VRAM_SIZE: u32 = 0x20_0000;
/// Entries in the shared table (NATIVEPSP_VRAM_MAX_RANGES).
pub const MAX_RANGES: usize = 32;
pub const BLOCK_SIZE: u32 = 1024;
pub const BLOCKS: usize = (VRAM_SIZE / BLOCK_SIZE) as usize;
/// A range or block unused for longer than this is no longer requested.
pub const KEEP_FRAMES: u32 = 120;
/// Texture ranges used within this many frames are "recent": the plugin
/// checks them for changes in the game's thread at every GE sync, the
/// others only in its copy thread. Checking all of them in the game's
/// thread (up to 600 KiB, ~1.6 ms) made Final Fantasy's battle-start rays
/// miss every third vblank.
pub const RECENT_FRAMES: u32 = 2;
/// Blocks after an unready one that are requested with it.
pub const LOOKAHEAD_BLOCKS: usize = 64;
/// Unused blocks between used ones that are requested too, so the plugin
/// gets few ranges (doubled until the ranges fit into the table).
pub const GAP_BLOCKS: usize = 8;

/// One table entry: bytes from the start of VRAM (NativePspVramRange).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct VramRange {
    pub offset: u32,
    pub size: u32,
}

/// The table the plugin should have.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    pub ranges: [VramRange; MAX_RANGES],
    pub count: usize,
    /// Bit i: `ranges[i]` is recent (see [`RECENT_FRAMES`]).
    pub recent_mask: u32,
}

impl Plan {
    pub fn ranges(&self) -> &[VramRange] {
        &self.ranges[..self.count]
    }
}

pub struct VramRequests {
    /// Counts `plan` calls.
    frame: u32,
    texture_ranges: [VramRange; MAX_RANGES],
    texture_last_used: [u32; MAX_RANGES],
    texture_count: usize,
    /// `frame + 1` when the capture last read the block; 0 = never.
    block_last_used: [u32; BLOCKS],
    /// The plugin's copy holds the block (and keeps it up to date).
    ready: [bool; BLOCKS],
    /// Requested, copy not confirmed yet.
    pending: [bool; BLOCKS],
    any_pending: bool,
    pending_generation: u32,
    /// Whole blocks the last plan covers.
    covered: [bool; BLOCKS],
    /// Reads of unready blocks since the last `take_missing`.
    missing: u32,
    /// Blocks the last plan requested for the capture (for the log).
    capture_blocks: u32,
}

impl Default for VramRequests {
    fn default() -> Self {
        Self::new()
    }
}

/// The VRAM blocks the capture's stage B read (another thread; see
/// `VramRequests::ready_for_stage`).
pub struct StageVramReads {
    used: [u32; BLOCKS / 32],
    missing: u32,
}

impl StageVramReads {
    pub const fn new() -> Self {
        StageVramReads { used: [0; BLOCKS / 32], missing: 0 }
    }

    /// Stage B read the block at physical `address` (in VRAM); `ready`:
    /// it was in the plugin's copy.
    pub fn note(&mut self, address: u32, ready: bool) {
        if let Some(block) = address.checked_sub(VRAM_START).map(|offset| (offset / BLOCK_SIZE) as usize) {
            if block < BLOCKS {
                self.used[block / 32] |= 1 << (block % 32);
                if !ready {
                    self.missing += 1;
                }
            }
        }
    }
}

impl Default for StageVramReads {
    fn default() -> Self {
        Self::new()
    }
}

impl VramRequests {
    pub const fn new() -> Self {
        VramRequests {
            frame: 0,
            texture_ranges: [VramRange { offset: 0, size: 0 }; MAX_RANGES],
            texture_last_used: [0; MAX_RANGES],
            texture_count: 0,
            block_last_used: [0; BLOCKS],
            ready: [false; BLOCKS],
            pending: [false; BLOCKS],
            any_pending: false,
            pending_generation: 0,
            covered: [false; BLOCKS],
            missing: 0,
            capture_blocks: 0,
        }
    }

    /// Forgets everything (not recording; the next plan is empty).
    pub fn reset(&mut self) {
        let frame = self.frame;
        *self = Self::new();
        self.frame = frame;
    }

    /// The capture is about to read the 1 KiB block at physical `address`
    /// (in VRAM). True if the plugin's copy holds it; otherwise it gets
    /// requested, with the blocks after it.
    pub fn block_ready(&mut self, address: u32) -> bool {
        let Some(block) = address.checked_sub(VRAM_START).map(|offset| (offset / BLOCK_SIZE) as usize) else {
            return false;
        };
        if block >= BLOCKS {
            return false;
        }
        self.block_last_used[block] = self.frame + 1;
        if self.ready[block] {
            return true;
        }
        for next in block + 1..(block + LOOKAHEAD_BLOCKS).min(BLOCKS) {
            self.block_last_used[next] = self.frame + 1;
        }
        self.missing += 1;
        false
    }

    /// `block_ready` for the capture's stage B in another thread: only
    /// reads whether the block is in the plugin's copy; stage B notes its
    /// reads in its own [`StageVramReads`], added with `add_stage_reads`
    /// at the frame's end.
    ///
    /// # Safety
    /// `requests` must point to a `VramRequests` whose `ready` flags no
    /// thread changes meanwhile (only `plan`, at a frame's end, does).
    pub unsafe fn ready_for_stage(requests: *const VramRequests, address: u32) -> bool {
        let Some(block) = address.checked_sub(VRAM_START).map(|offset| (offset / BLOCK_SIZE) as usize) else {
            return false;
        };
        block < BLOCKS && core::ptr::read_volatile(core::ptr::addr_of!((*requests).ready[block]))
    }

    /// What stage B read this frame, as if it had called `block_ready`.
    pub fn add_stage_reads(&mut self, reads: &mut StageVramReads) {
        for (word_index, word) in reads.used.iter_mut().enumerate() {
            let mut bits = core::mem::take(word);
            while bits != 0 {
                let block = word_index * 32 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                self.block_last_used[block] = self.frame + 1;
                if !self.ready[block] {
                    for next in block + 1..(block + LOOKAHEAD_BLOCKS).min(BLOCKS) {
                        self.block_last_used[next] = self.frame + 1;
                    }
                }
            }
        }
        self.missing += core::mem::take(&mut reads.missing);
    }

    /// Reads of unready blocks since the last call (0 = the frame read
    /// nothing it lacked).
    pub fn take_missing(&mut self) -> u32 {
        core::mem::take(&mut self.missing)
    }

    /// Reads of unready blocks so far in this frame.
    pub fn missing(&self) -> u32 {
        self.missing
    }

    /// Blocks the last plan requested for the capture.
    pub fn capture_blocks(&self) -> u32 {
        self.capture_blocks
    }

    /// After each frame: `textures` are the VRAM byte ranges (offsets from
    /// the start of VRAM) the frame's textures use, `None` when not
    /// recording (then nothing is requested). `copied_generation` is the
    /// plugin's latest. Returns the table the plugin should have.
    pub fn plan(&mut self, textures: Option<&mut dyn Iterator<Item = VramRange>>, copied_generation: u32) -> Plan {
        self.frame = self.frame.wrapping_add(1);
        let Some(textures) = textures else {
            self.reset();
            return Plan { ranges: [VramRange::default(); MAX_RANGES], count: 0, recent_mask: 0 };
        };

        // A generation the plugin copied makes its blocks ready.
        if self.any_pending && copied_generation.wrapping_sub(self.pending_generation) as i32 >= 0 {
            for block in 0..BLOCKS {
                self.ready[block] |= self.pending[block];
                self.pending[block] = false;
            }
            self.any_pending = false;
        }

        for range in textures {
            if range.size > 0 && range.offset < VRAM_SIZE && range.size <= VRAM_SIZE - range.offset {
                self.add_texture_range(range);
            }
        }
        self.forget_old_texture_ranges();

        // Texture ranges first, then the capture's blocks.
        let mut plan = Plan { ranges: [VramRange::default(); MAX_RANGES], count: 0, recent_mask: 0 };
        for i in 0..self.texture_count {
            plan.ranges[i] = self.texture_ranges[i];
            if self.frame.wrapping_sub(self.texture_last_used[i]) < RECENT_FRAMES {
                plan.recent_mask |= 1 << i;
            }
        }
        plan.count = self.add_block_ranges(&mut plan.ranges, self.texture_count);
        for i in self.texture_count..plan.count {
            plan.recent_mask |= 1 << i; // lists are read in every frame that uses them
        }

        // Only whole blocks a range covers are kept up to date by the plugin
        // (a texture range can start or end inside a block).
        self.covered = [false; BLOCKS];
        for range in plan.ranges() {
            let first = range.offset.div_ceil(BLOCK_SIZE) as usize;
            let end = ((range.offset + range.size) / BLOCK_SIZE) as usize;
            for block in first..end.min(BLOCKS) {
                self.covered[block] = true;
            }
        }
        for block in 0..BLOCKS {
            self.ready[block] &= self.covered[block];
            self.pending[block] &= self.covered[block];
        }
        plan
    }

    /// The last plan was written to the shared table as `generation`: its
    /// blocks get ready once the plugin has copied that generation.
    pub fn published(&mut self, generation: u32) {
        for block in 0..BLOCKS {
            if self.covered[block] && !self.ready[block] {
                self.pending[block] = true;
                self.any_pending = true;
            }
        }
        self.pending_generation = generation;
    }

    fn add_texture_range(&mut self, range: VramRange) {
        for i in 0..self.texture_count {
            if self.texture_ranges[i] == range {
                self.texture_last_used[i] = self.frame;
                return;
            }
        }
        if self.texture_count < MAX_RANGES {
            self.texture_ranges[self.texture_count] = range;
            self.texture_last_used[self.texture_count] = self.frame;
            self.texture_count += 1;
        }
    }

    fn forget_old_texture_ranges(&mut self) {
        let mut kept = 0;
        for i in 0..self.texture_count {
            if self.frame.wrapping_sub(self.texture_last_used[i]) <= KEEP_FRAMES {
                self.texture_ranges[kept] = self.texture_ranges[i];
                self.texture_last_used[kept] = self.texture_last_used[i];
                kept += 1;
            }
        }
        self.texture_count = kept;
    }

    fn block_used(&self, block: usize) -> bool {
        let last = self.block_last_used[block];
        last != 0 && self.frame.wrapping_sub(last) < KEEP_FRAMES
    }

    /// Appends the runs of recently read blocks to `ranges[count..]`; if
    /// they don't fit, larger gaps are joined. Returns the new count.
    fn add_block_ranges(&mut self, ranges: &mut [VramRange; MAX_RANGES], count: usize) -> usize {
        let mut gap = GAP_BLOCKS;
        loop {
            let mut n = count;
            let mut blocks = 0;
            let mut fits = true;
            let mut block = 0;
            while block < BLOCKS {
                if !self.block_used(block) {
                    block += 1;
                    continue;
                }
                let mut end = block + 1; // one past the run's last used block
                let mut next = end;
                while next < BLOCKS && next <= end + gap {
                    if self.block_used(next) {
                        end = next + 1;
                    }
                    next += 1;
                }
                if n == MAX_RANGES {
                    fits = false;
                    break;
                }
                ranges[n] = VramRange { offset: block as u32 * BLOCK_SIZE, size: (end - block) as u32 * BLOCK_SIZE };
                blocks += end - block;
                n += 1;
                block = end;
            }
            if fits || gap >= BLOCKS {
                self.capture_blocks = blocks as u32;
                return n;
            }
            gap *= 2;
        }
    }
}

/// Whether `texture` needs the plugin's VRAM copy, given the framebuffers
/// the display has render targets for (`display_targets`, PSP addresses, 0 =
/// none; nativepsp_render.c publishes them) and the frame's shown framebuffer.
///
/// MTX samples framebuffers as textures while driving (2026-11-09): the VRAM
/// requests grew to 2.1-2.3 MB (nearly all of VRAM), and since the GE redraws
/// them every frame the plugin found ~800 changed blocks per frame and copied
/// them in the game's thread. The display doesn't use that copy where it has
/// a target at the texture's address: it samples the target (format < 4, not
/// the framebuffer the frame shows, as in nativepsp_render.c). Those textures
/// aren't requested. Others are: a first try that skipped every recently
/// drawn framebuffer lost MTX's HUD (drawn into a third buffer the display
/// keeps no target for).
pub fn texture_needs_copy(texture: &crate::frame::TextureDescriptor, display_targets: &[u32], shown_framebuffer: u32) -> bool {
    let fold = |address: u32| crate::memory::fold_vram_mirror(address & 0x0FFF_FFFF);
    let address = fold(texture.address);
    let from_target = texture.format < 4
        && address != fold(shown_framebuffer)
        && display_targets.iter().any(|&target| target != 0 && fold(target) == address);
    !from_target
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block_address(block: u32) -> u32 {
        VRAM_START + block * BLOCK_SIZE
    }

    fn no_textures() -> core::iter::Empty<VramRange> {
        core::iter::empty()
    }

    /// One frame: plan (nothing from textures), publish as `generation`.
    fn frame(requests: &mut VramRequests, copied: u32, generation: u32) -> Plan {
        let plan = requests.plan(Some(&mut no_textures()), copied);
        requests.published(generation);
        plan
    }

    #[test]
    fn a_block_is_ready_only_after_its_generation_was_copied() {
        let mut requests = Box::new(VramRequests::new());
        let list = block_address(0x155);
        assert!(!requests.block_ready(list), "nothing is copied at first");
        assert_eq!(requests.take_missing(), 1);

        // Requested with the blocks after it.
        let plan = frame(&mut requests, 0, 1);
        assert_eq!(plan.ranges(), &[VramRange { offset: 0x155 * BLOCK_SIZE, size: LOOKAHEAD_BLOCKS as u32 * BLOCK_SIZE }]);
        assert_eq!(plan.recent_mask, 1);
        assert!(!requests.block_ready(list), "generation 1 isn't copied yet");
        requests.take_missing();

        // The plugin copied generation 1: ready from the next plan on.
        frame(&mut requests, 1, 1);
        assert!(requests.block_ready(list));
        assert!(requests.block_ready(list + 63 * BLOCK_SIZE), "the look-ahead blocks too");
        assert!(!requests.block_ready(list + 64 * BLOCK_SIZE));
        assert_eq!(requests.take_missing(), 1);
    }

    #[test]
    fn blocks_unused_for_a_while_are_dropped_and_unready_again() {
        let mut requests = Box::new(VramRequests::new());
        requests.block_ready(block_address(10));
        frame(&mut requests, 0, 1);
        frame(&mut requests, 1, 1);
        assert!(requests.block_ready(block_address(10)));
        for _ in 0..KEEP_FRAMES + 1 {
            frame(&mut requests, 1, 1);
        }
        let plan = frame(&mut requests, 1, 1);
        assert_eq!(plan.count, 0);
        assert!(!requests.block_ready(block_address(10)));
    }

    #[test]
    fn nearby_runs_are_joined_and_distant_ones_kept_apart() {
        let mut requests = Box::new(VramRequests::new());
        // Mark single blocks as used without the look-ahead: make them ready first.
        for block in [0u32, 70, 200] {
            requests.block_ready(block_address(block));
        }
        let plan = frame(&mut requests, 0, 1);
        // 0..64 and 70..134 are 6 blocks apart: one run; 200.. separate.
        assert_eq!(
            plan.ranges(),
            &[
                VramRange { offset: 0, size: 134 * BLOCK_SIZE },
                VramRange { offset: 200 * BLOCK_SIZE, size: 64 * BLOCK_SIZE },
            ]
        );
    }

    #[test]
    fn too_many_runs_are_joined_until_they_fit() {
        let mut requests = Box::new(VramRequests::new());
        // 29 texture ranges leave 3 entries for the capture's 6 runs.
        let mut textures = (0..29u32).map(|i| VramRange { offset: i * 0x100, size: 0x100 });
        for run in 0..6u32 {
            requests.block_ready(block_address(100 + run * 300));
        }
        let plan = requests.plan(Some(&mut textures), 0);
        assert!(plan.count > 29 && plan.count <= MAX_RANGES, "29 texture ranges + joined runs, got {}", plan.count);
        for run in 0..6u32 {
            let offset = (100 + run * 300) * BLOCK_SIZE;
            let end = offset + LOOKAHEAD_BLOCKS as u32 * BLOCK_SIZE;
            assert!(
                plan.ranges()[29..].iter().any(|range| offset >= range.offset && end <= range.offset + range.size),
                "run {run} requested"
            );
        }
    }

    #[test]
    fn texture_ranges_come_first_and_are_recent_while_used() {
        let mut requests = Box::new(VramRequests::new());
        let texture = VramRange { offset: 0x174000, size: 0x40000 };
        let plan = requests.plan(Some(&mut core::iter::once(texture)), 0);
        assert_eq!(plan.ranges(), &[texture]);
        assert_eq!(plan.recent_mask, 1);
        let plan = requests.plan(Some(&mut no_textures()), 0);
        assert_eq!(plan.ranges(), &[texture], "kept while unused for a while");
        let plan = requests.plan(Some(&mut no_textures()), 0);
        assert_eq!(plan.recent_mask, 0, "no longer recent");
        for _ in 0..KEEP_FRAMES {
            requests.plan(Some(&mut no_textures()), 0);
        }
        assert_eq!(requests.plan(Some(&mut no_textures()), 0).count, 0);
    }

    #[test]
    fn a_texture_range_makes_its_whole_blocks_ready_for_the_capture() {
        let mut requests = Box::new(VramRequests::new());
        // Starts in the middle of block 1, ends at the end of block 3.
        let texture = VramRange { offset: 1536, size: 4096 - 1536 };
        requests.plan(Some(&mut core::iter::once(texture)), 0);
        requests.published(1);
        requests.plan(Some(&mut core::iter::once(texture)), 1);
        assert!(!requests.block_ready(block_address(1)), "partly covered");
        assert!(requests.block_ready(block_address(2)));
        assert!(requests.block_ready(block_address(3)));
    }

    #[test]
    fn not_recording_forgets_everything() {
        let mut requests = Box::new(VramRequests::new());
        requests.block_ready(block_address(5));
        frame(&mut requests, 0, 1);
        frame(&mut requests, 1, 1);
        assert!(requests.block_ready(block_address(5)));
        assert_eq!(requests.plan(None, 1).count, 0);
        assert!(!requests.block_ready(block_address(5)));
    }

    #[test]
    fn addresses_outside_vram_are_never_ready() {
        let mut requests = Box::new(VramRequests::new());
        assert!(!requests.block_ready(0x0880_0000));
        assert!(!requests.block_ready(VRAM_START - BLOCK_SIZE));
        assert_eq!(requests.take_missing(), 0, "not counted: not VRAM");
    }

    /// A framebuffer sampled as a texture is drawn by our display from its
    /// own target: no VRAM copy for it, except the one the frame shows. A
    /// framebuffer the display has no target for (MTX's HUD buffer) needs
    /// the copy, as do other textures.
    #[test]
    fn textures_the_display_draws_from_its_targets_need_no_vram_copy() {
        use crate::frame::TextureDescriptor;
        let targets = [0x0400_0000, 0x0408_8000, 0, 0];
        let picture = |address: u32| TextureDescriptor { address, format: 3, width: 512, height: 512, buffer_width: 512, ..Default::default() };
        assert!(!texture_needs_copy(&picture(0x0408_8000), &targets, 0x0400_0000), "the other buffer: our target");
        assert!(!texture_needs_copy(&picture(0x4408_8000), &targets, 0x0400_0000), "a VRAM mirror of it too");
        assert!(texture_needs_copy(&picture(0x0400_0000), &targets, 0x0400_0000), "the shown one: decoded from the copy");
        assert!(texture_needs_copy(&picture(0x0411_0000), &targets, 0x0400_0000), "a buffer without a target");
        let clut8 = TextureDescriptor { format: 5, ..picture(0x0408_8000) };
        assert!(texture_needs_copy(&clut8, &targets, 0x0400_0000), "palette formats are never targets");
    }
}
