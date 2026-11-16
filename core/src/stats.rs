//! Per-frame statistics about what the game asked the GE to do.
//!
//! Phase 1 only counts things; this tells us how much work a real renderer
//! would have to do per frame and which features (texture formats, render
//! targets, block transfers, ...) a game uses.

use crate::command::{op, Command};
use crate::display_list::CommandVisitor;
use crate::ge_state::GeState;
use crate::memory::PspMemory;

/// A small set of addresses without heap allocation. Addresses beyond the
/// capacity are only counted in `overflow`.
#[derive(Clone, Copy, Debug)]
pub struct AddressSet<const N: usize> {
    addresses: [u32; N],
    len: usize,
    /// Where the last inserted address was found or put, checked first:
    /// consecutive draws mostly use the same texture and target.
    last: usize,
    pub overflow: u32,
}

impl<const N: usize> AddressSet<N> {
    pub const fn new() -> Self {
        AddressSet { addresses: [0; N], len: 0, last: 0, overflow: 0 }
    }

    pub fn insert(&mut self, address: u32) {
        if self.last < self.len && self.addresses[self.last] == address {
            return;
        }
        if let Some(index) = self.as_slice().iter().position(|&known| known == address) {
            self.last = index;
            return;
        }
        if self.len < N {
            self.addresses[self.len] = address;
            self.last = self.len;
            self.len += 1;
        } else {
            self.overflow += 1;
        }
    }

    pub fn contains(&self, address: u32) -> bool {
        self.as_slice().contains(&address)
    }

    pub fn as_slice(&self) -> &[u32] {
        &self.addresses[..self.len]
    }

    /// Number of distinct addresses seen, including those that didn't fit.
    pub fn count(&self) -> u32 {
        self.len as u32 + self.overflow
    }
}

/// Counts distinct addresses: a small hash table (open addressing), since
/// a list scanned for every draw was too slow for MTX Mototrax (~1000
/// textures, ~3000 draws per frame). Past 3/4 full, new addresses are only
/// counted in `overflow`.
#[derive(Clone, Copy, Debug)]
pub struct AddressCounter<const N: usize> {
    /// `EMPTY` or an address. N must be a power of two.
    slots: [u32; N],
    len: usize,
    /// The address inserted last, checked first.
    last: u32,
    pub overflow: u32,
}

impl<const N: usize> AddressCounter<N> {
    const EMPTY: u32 = u32::MAX;

    pub const fn new() -> Self {
        AddressCounter { slots: [Self::EMPTY; N], len: 0, last: Self::EMPTY, overflow: 0 }
    }

    pub fn insert(&mut self, address: u32) {
        if address == self.last || address == Self::EMPTY {
            return;
        }
        self.last = address;
        // Fibonacci hashing: spreads nearby addresses over the table.
        let mut slot = (address.wrapping_mul(0x9E37_79B9) >> 16) as usize & (N - 1);
        loop {
            if self.slots[slot] == address {
                return;
            }
            if self.slots[slot] == Self::EMPTY {
                if self.len >= N / 4 * 3 {
                    self.overflow += 1;
                } else {
                    self.slots[slot] = address;
                    self.len += 1;
                }
                return;
            }
            slot = (slot + 1) & (N - 1);
        }
    }

    /// Number of distinct addresses seen, plus those that didn't fit.
    pub fn count(&self) -> u32 {
        self.len as u32 + self.overflow
    }
}

/// The framebuffer the game showed at the end of the frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DisplayedFramebuffer {
    pub address: u32,
    pub stride: u32,
    pub pixel_format: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct FrameStats {
    // Events from the game.
    pub lists_enqueued: u32,
    pub stall_updates: u32,

    // Results of walking the lists.
    pub lists_finished: u32,
    pub commands: u32,
    /// PRIM commands, i.e. draw calls.
    pub draws: u32,
    pub vertices: u32,
    /// Draws with screen-space (2D) vertices.
    pub through_mode_draws: u32,
    pub textured_draws: u32,
    /// Textured draws per texture format (see `command::TEXTURE_FORMAT_NAMES`).
    pub draws_by_texture_format: [u32; 16],
    /// BEZIER and SPLINE commands.
    pub curves: u32,
    pub block_transfers: u32,
    pub bounding_box_jumps: u32,
    pub signals: u32,
    /// Distinct texture addresses used by draws.
    pub textures: AddressCounter<1024>,
    /// Distinct framebuffer addresses that were drawn to.
    pub render_targets: AddressSet<8>,

    // Problems.
    pub bad_addresses: u32,
    /// The list address the last walk that stopped at a bad address couldn't read.
    pub last_bad_address: u32,
    pub endless_lists: u32,
    /// Lists replaced by a new list with the same id before we saw them finish.
    pub lists_abandoned: u32,
    /// Lists we could not track because our queue was full.
    pub queue_overflows: u32,
    /// Ring events lost because the reader fell behind.
    pub dropped_events: u32,
    /// GE contexts (`ge_context`): states saved, restored, restores of a
    /// context we never saw saved, and restores stage B didn't get
    /// (pipelined capture).
    pub contexts_saved: u32,
    pub contexts_restored: u32,
    pub contexts_missing: u32,
    pub contexts_pipelined: u32,
    /// Block transfers too big for `ge_writes` to follow (later reads of
    /// their destination see PSP memory as it is).
    pub transfers_skipped: u32,
    /// The vertex cache this frame (`vertex_cache::CacheStats`).
    pub cache: crate::vertex_cache::CacheStats,

    // Frame building (phase 2 renderer), only counted when it runs.
    /// 3D draws not rendered: skinned ones (bone weights).
    pub skipped_3d_draws: u32,
    /// Points and lines, not rendered yet.
    pub skipped_point_line_draws: u32,
    /// Draws whose vertex data couldn't be read.
    pub unreadable_draws: u32,
    /// Vertex address of the last draw counted in `unreadable_draws`.
    pub last_unreadable_address: u32,
    /// Draws rendered with a simplification, once per reason
    /// (`recorder::approx`).
    pub approximated_draws: u32,
    /// The same by reason: `approximated_by_reason[bit]` for
    /// `recorder::approx` bit `1 << bit` (the replay prints them).
    pub approximated_by_reason: [u32; 9],
    /// The GE's BLEND_MODE, BLEND_FIXED_A and BLEND_FIXED_B (24-bit data
    /// each) of the frame's last draw state whose blending was approximated
    /// (`recorder::approx::BLEND`), and how many such states there were:
    /// the capture log shows them (`approx_blend=`), to see which blends
    /// need doing exactly.
    pub approx_blend: [u32; 3],
    pub approx_blend_states: u32,
    /// The frame description ran out of space.
    pub frame_overflows: u32,
    /// Draws too big for the draw stream, recorded by the capture thread
    /// itself once the recorder thread was idle (the two halves don't run
    /// side by side meanwhile), and the PSP memory words they read.
    pub inline_draws: u32,
    pub inline_words: u32,
    /// Draws that write no colour (all RGB bits masked), not rendered.
    pub color_masked_draws: u32,
    /// PRIMs whose vertices were decoded on the CPU, and their PSP
    /// vertices, by reason (`recorder::fallback`; capture log `cpu=`).
    pub fallback_prims: [u32; crate::recorder::fallback::COUNT],
    pub fallback_vertices: [u32; crate::recorder::fallback::COUNT],

    /// Set by the frame-ending flip.
    pub displayed: Option<DisplayedFramebuffer>,
    /// Lists still being walked when the frame ended.
    pub lists_pending: u32,
}

impl FrameStats {
    pub const fn new() -> FrameStats {
        FrameStats {
            lists_enqueued: 0,
            stall_updates: 0,
            lists_finished: 0,
            commands: 0,
            draws: 0,
            vertices: 0,
            through_mode_draws: 0,
            textured_draws: 0,
            draws_by_texture_format: [0; 16],
            curves: 0,
            block_transfers: 0,
            bounding_box_jumps: 0,
            signals: 0,
            textures: AddressCounter::new(),
            render_targets: AddressSet::new(),
            bad_addresses: 0,
            last_bad_address: 0,
            endless_lists: 0,
            lists_abandoned: 0,
            queue_overflows: 0,
            dropped_events: 0,
            contexts_saved: 0,
            contexts_restored: 0,
            contexts_missing: 0,
            contexts_pipelined: 0,
            transfers_skipped: 0,
            cache: crate::vertex_cache::CacheStats { hits: 0, misses: 0, stored: 0, full: 0, cpu_only: 0, changing: 0, quick: 0, repeated: 0, raw_runs: 0, raw_bytes: 0, raw_full: 0, raw_lit: 0, raw_psp_ram: 0, raw_psp_ram_bytes: 0, raw_converted: 0, raw_gathered: 0 },
            skipped_3d_draws: 0,
            skipped_point_line_draws: 0,
            unreadable_draws: 0,
            last_unreadable_address: 0,
            approximated_draws: 0,
            approximated_by_reason: [0; 9],
            approx_blend: [0; 3],
            approx_blend_states: 0,
            frame_overflows: 0,
            inline_draws: 0,
            inline_words: 0,
            color_masked_draws: 0,
            fallback_prims: [0; crate::recorder::fallback::COUNT],
            fallback_vertices: [0; crate::recorder::fallback::COUNT],
            displayed: None,
            lists_pending: 0,
        }
    }
}

impl Default for FrameStats {
    fn default() -> Self {
        FrameStats::new()
    }
}

impl FrameStats {
    /// Counts PRIM commands that all draw with the GE state `ge` (one, or
    /// a batch from `walk`, `CommandVisitor::visit_prims`): the same counts
    /// as visiting them one by one, but the state is looked at once (per
    /// PRIM it was ~50 instructions, for ~10,500 PRIMs in a race frame of
    /// MTX). The address sets ignore an address they already hold.
    pub fn count_prims(&mut self, prims: &[Command], ge: &GeState) {
        let draws = prims.len() as u32;
        self.draws += draws;
        for prim in prims {
            self.vertices += prim.data() & 0xFFFF;
        }
        if ge.is_through_mode() {
            self.through_mode_draws += draws;
        }
        if ge.is_texture_enabled() {
            self.textured_draws += draws;
            self.draws_by_texture_format[ge.texture_format() as usize] += draws;
            self.textures.insert(ge.texture_address());
        }
        self.render_targets.insert(ge.framebuffer_address());
    }

    /// `wants` without a `FrameStats` (it is the same for all).
    pub fn new_wants(opcode: u8) -> bool {
        matches!(
            opcode,
            op::PRIM | op::BEZIER | op::SPLINE | op::TRANSFER_START | op::BOUNDING_BOX_JUMP | op::SIGNAL
        )
    }
}

impl CommandVisitor for FrameStats {
    /// The commands counted below.
    #[inline]
    fn wants(&self, opcode: u8) -> bool {
        FrameStats::new_wants(opcode)
    }

    fn visit<M: PspMemory>(&mut self, command: Command, ge: &GeState, _memory: &mut M) {
        // `commands` is counted by the walk (it only hands us `wants`).
        match command.opcode() {
            op::PRIM => self.count_prims(&[command], ge),
            op::BEZIER | op::SPLINE => self.curves += 1,
            op::TRANSFER_START => self.block_transfers += 1,
            op::BOUNDING_BOX_JUMP => self.bounding_box_jumps += 1,
            op::SIGNAL => self.signals += 1,
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_batch_of_prims_counts_what_the_prims_one_by_one_count() {
        let set = |ge: &mut GeState, opcode: u8, data: u32| ge.set_register(Command(((opcode as u32) << 24) | data));
        let prims: Vec<Command> = [3, 4, 300, 6].iter().map(|&count| Command(((op::PRIM as u32) << 24) | (4 << 16) | count)).collect();
        let mut batched = FrameStats::new();
        let mut one_by_one = FrameStats::new();
        let mut ge = GeState::new();
        // Untextured 3D, then textured through-mode draws to another target
        // and with other textures.
        for (texture, through, target) in [(None, false, 0x1000), (Some(0x10_0000), true, 0x8_8000), (Some(0x12_0000), true, 0x8_8000)] {
            set(&mut ge, op::TEXTURE_MAP_ENABLE, texture.is_some() as u32);
            set(&mut ge, op::TEXTURE_ADDRESS_0, texture.unwrap_or(0));
            set(&mut ge, op::TEXTURE_FORMAT, 3);
            set(&mut ge, op::VERTEX_TYPE, (through as u32) << 23);
            set(&mut ge, op::FRAMEBUFFER_POINTER, target);
            batched.count_prims(&prims, &ge);
            for &prim in &prims {
                one_by_one.visit(prim, &ge, &mut crate::memory::test_memory::TestMemory::default());
            }
        }
        let counts = |stats: &FrameStats| {
            let set = (stats.textures.count(), stats.render_targets.as_slice().to_vec());
            (stats.draws, stats.vertices, stats.through_mode_draws, stats.textured_draws, stats.draws_by_texture_format, set)
        };
        assert_eq!(counts(&batched), counts(&one_by_one));
        assert_eq!(counts(&batched).0, 12);
        assert_eq!(counts(&batched).5, (2, vec![0x0400_1000, 0x0408_8000]));
    }

    #[test]
    fn address_counter_counts_distinct_addresses() {
        let mut counter = AddressCounter::<16>::new();
        for address in [0x0890_0000, 0x0890_0010, 0x0890_0000, 0, 0x0890_0010] {
            counter.insert(address);
        }
        assert_eq!(counter.count(), 3);
        // 3/4 of 16 slots fit; the rest is still counted.
        for address in 1..=20 {
            counter.insert(address * 0x100);
        }
        assert_eq!(counter.count(), 3 + 20);
        assert_eq!(counter.overflow, 3 + 20 - 12);
    }
}
