//! What the PSP plugin (renderer/psp, kernel mode on the emulated PSP) and
//! the Vita side share in PSP memory: the event ring ([`crate::event_ring`])
//! and the VRAM copy request table ([`VramRequestTable`]). Both sides use
//! these definitions (until stage 4 they were the C header
//! shared/nativepsp_ring.h, which both sides had to match).
//!
//! Both live in the unused tail of Adrenaline's shared area (the
//! SceAdrenaline struct uses its first ~0x600 bytes, DRAW_NATIVE sits at
//! +0xF00): the ring at [`crate::event_ring::RING_ADDRESS`], the table after
//! it in the same 4 KiB page.
//!
//! **VRAM copy requests (Vita -> PSP).** Sony's emulator keeps the real PSP
//! VRAM where the Vita side can't read it; the region where PSP VRAM would
//! sit in RAM (PSP 0x0BE00000, the emulator's SCE_PSPEMU_VRAM) stays empty.
//! The PSP side, however, can read VRAM. So:
//! - the Vita side lists the VRAM ranges it needs (offsets from 0x04000000)
//!   and bumps `generation` (core [`crate::vram_requests`]);
//! - the plugin copies those ranges from VRAM to the same offsets at
//!   [`VRAM_COPY_ADDRESS`] (its copy thread, woken at every frame flip) and
//!   records what it did;
//! - once all ranges of a generation are copied, the plugin sets
//!   `copied_generation`; until then the Vita side doesn't decode new VRAM
//!   textures (it would read stale or empty data);
//! - games load new graphics where old ones were (Final Fantasy: the menu's
//!   crystals where the field's floor tiles go), so whenever the game has
//!   waited for the GE (sceGeDrawSync/ListSync; at the flip for games that
//!   don't) and at every list enqueue the plugin samples each 1 KiB block of
//!   the ranges the latest frames used (`recent_mask`), copies blocks that
//!   changed again and counts them in `changes` ([`crate::psp_copy`]). The
//!   other ranges are checked by the copy thread after the flip (in the
//!   game's idle time), except blocks a recent range also covers.
//!
//! **GE sync (2026-11-08).** When the game waits for the GE it may reuse the
//! memory of the lists it waited for (MTX fills its HUD buffer for the frame
//! after next). The Vita side reads that memory itself, often later than the
//! GE did, so while the Vita side asks for it (`vita_sync_wanted` in the
//! ring header) the plugin also waits until the Vita side has handled every
//! event written so far, at most [`crate::event_ring::SYNC_TIMEOUT_US`] per
//! sync ([`crate::psp_copy::SyncStall`]).

use crate::event_ring::{RawEvent, RingHeader, CAPACITY, RING_ADDRESS};
use crate::vram_requests::{VramRange, MAX_RANGES};

/// The ring and the table share this page (uncached PSP address).
pub const PAGE_SIZE: u32 = 0x1000;
/// The table (uncached kernel address, kseg1), after the ring.
pub const VRAM_REQUESTS_ADDRESS: u32 = 0xABCD_FD00;
/// The plugin's copy of VRAM: uncached PSP address of physical 0x0BE00000.
pub const VRAM_COPY_ADDRESS: u32 = 0xABE0_0000;
/// The same through the cached kernel address (kseg0).
pub const VRAM_COPY_CACHED_ADDRESS: u32 = VRAM_COPY_ADDRESS - 0xA000_0000;
/// PSP VRAM through its uncached (0x44000000) and cached (0x04000000)
/// addresses.
pub const VRAM_UNCACHED: u32 = 0x4400_0000;
pub const VRAM_CACHED: u32 = 0x0400_0000;

/// How the plugin copies VRAM (`copy_method`, `probe_chosen`; probe times
/// are at `probe_us[method - 1]`).
pub mod copy_method {
    /// Nothing copied yet.
    pub const NONE: u32 = 0;
    /// sceDmacMemcpy with the uncached addresses.
    pub const DMA: u32 = 1;
    /// Word by word, uncached VRAM (0x44000000).
    pub const WORDS: u32 = 2;
    /// Word by word through the cached addresses (cache invalidated before,
    /// written back after).
    pub const CACHED: u32 = 3;
    /// sceDmacMemcpy with the cached addresses.
    pub const DMA_CACHED: u32 = 4;
    /// Cached, 8 words per loop step.
    pub const UNROLLED: u32 = 5;
    /// Cached, VFPU quad loads/stores (threads with the VFPU only).
    pub const VFPU: u32 = 6;
    /// Cached, the kernel's memcpy (SysclibForKernel).
    pub const MEMCPY: u32 = 7;
    /// Methods 1..=METHODS are timed at the copy thread's start.
    pub const METHODS: u32 = 7;
    /// A GE block transfer: on request only (`ge_probe_request`).
    pub const GE: u32 = 8;

    /// The name in the capture log line.
    pub fn name(method: u32) -> &'static str {
        match method {
            DMA => "dma",
            WORDS => "words",
            CACHED => "cached",
            DMA_CACHED => "dmacached",
            UNROLLED => "unrolled",
            VFPU => "vfpu",
            MEMCPY => "memcpy",
            _ => "none",
        }
    }
}

/// The shared VRAM request table (was NativePspVramRequests), at
/// [`VRAM_REQUESTS_ADDRESS`]. "Vita" fields are written by the Vita side,
/// "plugin" fields by the PSP plugin.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct VramRequestTable {
    /// Vita: entries in `ranges`.
    pub range_count: u32,
    /// Vita: bumped when the ranges change.
    pub generation: u32,
    /// Plugin: the generation whose ranges are all copied.
    pub copied_generation: u32,
    /// Plugin: wall time the last copy took (the game may run in between).
    pub last_copy_us: u32,
    /// Plugin: [`copy_method`] of the last copy.
    pub copy_method: u32,
    /// Plugin: flips the copy thread has handled.
    pub copy_thread_runs: u32,
    /// Plugin: error code if the copy thread failed (0 = none).
    pub copy_thread_error: u32,
    /// Plugin: VRAM blocks found changed and copied again, so far.
    pub changes: u32,
    /// Plugin: change checks done at a GE sync, so far, and the last one's
    /// time.
    pub sync_checks: u32,
    pub last_sync_check_us: u32,
    /// Vita: bit i = `ranges[i]` was used by the latest frames.
    pub recent_mask: u32,
    /// Vita: the ranges to copy.
    pub ranges: [VramRange; MAX_RANGES],
    /// Plugin, once at the copy thread's start: the copy methods timed on
    /// the same VRAM bytes (`probe_us[method - 1]`), which of them copied
    /// correctly (bit method - 1), the one used from then on, and which DMA
    /// lookup found a function (1.., 0 = none).
    pub probe_bytes: u32,
    pub probe_us: [u32; 8],
    pub probe_ok: u32,
    pub probe_chosen: u32,
    pub probe_dma_lookup: u32,
    /// GE block-transfer probe, on request only:
    /// the Vita side sets `ge_probe_request`, the plugin runs it once and
    /// sets it back to 0, with the result (0 = copied correctly, 1 = copied
    /// wrongly, < 0 = error) and its time in `probe_us[copy_method::GE - 1]`.
    pub ge_probe_request: u32,
    pub ge_probe_result: i32,
    /// What sceGeListEnQueue returned, the last sceGeListSync(qid, 1)
    /// result, how often it was asked.
    pub ge_probe_qid: i32,
    pub ge_probe_state: i32,
    pub ge_probe_polls: u32,
    /// Plugin: change checks done at a list's enqueue, so far, and the last
    /// one's time.
    pub enqueue_checks: u32,
    pub last_enqueue_check_us: u32,
}

const _: () = assert!(core::mem::size_of::<VramRequestTable>() == 376);
// The ring fits before the table, the table into the page.
const _: () = assert!(
    core::mem::size_of::<RingHeader>() + CAPACITY as usize * core::mem::size_of::<RawEvent>()
        <= (VRAM_REQUESTS_ADDRESS - RING_ADDRESS) as usize
);
const _: () = assert!(
    (VRAM_REQUESTS_ADDRESS - RING_ADDRESS) as usize + core::mem::size_of::<VramRequestTable>() <= PAGE_SIZE as usize
);

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::offset_of;

    /// The offsets the C plugin (psp-plugin/main.c, nativepsp_ring.h) used:
    /// a Rust plugin and an old Vita side (or the other way round) must
    /// still understand each other.
    #[test]
    fn table_layout_is_the_c_one() {
        assert_eq!(offset_of!(VramRequestTable, recent_mask), 40);
        assert_eq!(offset_of!(VramRequestTable, ranges), 44);
        assert_eq!(offset_of!(VramRequestTable, probe_bytes), 44 + 32 * 8);
        assert_eq!(offset_of!(VramRequestTable, probe_us), 304);
        assert_eq!(offset_of!(VramRequestTable, probe_dma_lookup), 344);
        assert_eq!(offset_of!(VramRequestTable, ge_probe_request), 348);
        assert_eq!(offset_of!(VramRequestTable, last_enqueue_check_us), 372);
        assert_eq!(offset_of!(RingHeader, vita_sync_events), 40);
    }

    #[test]
    fn method_names() {
        assert_eq!(copy_method::name(copy_method::MEMCPY), "memcpy");
        assert_eq!(copy_method::name(copy_method::NONE), "none");
        assert_eq!(copy_method::name(copy_method::GE), "none");
    }
}
