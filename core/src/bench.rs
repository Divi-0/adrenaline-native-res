//! Runs a dumped frame through the capture again and again, to time it.
//!
//! Used by `replay --bench` on the PC, and on the Vita by whoever wants to
//! time the same work there. Memory comes from the
//! dump through the same `BlockCache` the Vita uses (a fresh one per event),
//! but the blocks are copied from normal memory: the time measured is the
//! capture's own work, without the slow reads of PSP RAM.
//!
//! Comparing the modes splits the time up: `WalkOnly` follows the lists,
//! `Stats` adds per-command statistics and palette loads, `Full` adds
//! recording the draws (vertices, textures, blending) into a `Frame`.
//! `Produce` is `Full` with the vertices thrown away instead of recorded:
//! the producer half of the draw stream alone (see `draw_stream`).

use core::sync::atomic::{AtomicU32, Ordering};

use crate::block_cache::BlockCache;

/// Measuring aid: each bit skips one part of the capture thread's per-draw work, so the
/// difference in time is what that part costs on the Vita (PC instruction
/// counts don't predict Vita time). The frames come out wrong meanwhile.
/// 0, the default, changes nothing. It is global: set it back to 0.
pub static EXPERIMENT: AtomicU32 = AtomicU32::new(0);
/// Palette loads (CLUT_LOAD) not read.
pub const SKIP_PALETTES: u32 = 1;
/// A draw keeps the last worked-out state instead of checking it.
pub const SKIP_STATE_WORK: u32 = 2;
/// `prepare` skips its transform lookup (fog flags).
pub const SKIP_TRANSFORM_LOOKUP: u32 = 4;
/// Draws are prepared but not queued (no copies into the stream).
pub const SKIP_QUEUE: u32 = 8;
/// Indexed draws don't scan their indices (vertex range = count).
pub const SKIP_HIGHEST_INDEX: u32 = 16;
/// The capture's stage B doesn't apply plain register writes (`walk_stream`).
pub const SKIP_STAGE_B_PLAIN: u32 = 32;
/// The capture's stage A passes every plain register write on, also those
/// that change nothing (`walk_stream::StreamWriter`); not frame-changing.
pub const NO_WRITE_FILTER: u32 = 64;
/// The capture thread's raw draws work their table entries out every time,
/// also when they repeat the previous draw's (`frame_builder::LastEntries`).
pub const NO_REPEATED_ENTRIES: u32 = 128;
/// The capture thread's raw draws work their light sets out whole every
/// time, also when only the material changed (`lighting::GpuLightsCache`).
pub const NO_LIGHTS_CACHE: u32 = 256;
/// The capture thread's raw draws never extend the previous draw in place
/// (`frame_builder::LastRun`): every call takes the whole way.
pub const NO_JOINED_CALLS: u32 = 512;
/// Copy tasks are published to the copier one by one, each with a wake
/// (`draw_stream::TASK_BATCH` off).
pub const NO_TASK_BATCHES: u32 = 1024;
/// Palettes of the capture thread's draws go to the recorder as jobs, as
/// before, instead of as copy tasks (`recorder::Queues::write_direct`).
pub const NO_PALETTE_TASKS: u32 = 2048;
/// Textures are not sampled at their first use in a frame
/// (`recorder::sample_texture`): timing only, the texture checks go wrong.
pub const SKIP_TEXTURE_SAMPLES: u32 = 4096;
/// Textures are sampled on the capture thread, as before, instead of by the
/// copier (`recorder::sample_texture`'s sample tasks).
pub const NO_SAMPLE_TASKS: u32 = 8192;
/// Raw draws build and look up their camera every time, also when its
/// change number is the newest lookup's (`RawTables::add_raw_camera_of`).
pub const NO_CAMERA_MEMO: u32 = 16384;
/// Raw draws work out and look up their light set every time, also when
/// its change numbers are the newest lookup's (`RawTables::add_light_set_of`).
pub const NO_LIGHTS_MEMO: u32 = 32768;
/// The walk's list windows take one line per refill, as before, also when
/// the walk ran off the window's end (`display_list::ListWindow`'s read-ahead).
pub const NO_READ_AHEAD: u32 = 65536;
/// VERTEX_ADDRESS, INDEX_ADDRESS and OFFSET_ADDRESS end the walk's plain
/// runs, as before (`display_list::PlainAction::Address`).
pub const NO_ADDRESS_RUNS: u32 = 131072;

/// True if experiment `bit` is on.
#[inline]
pub fn experiment(bit: u32) -> bool {
    EXPERIMENT.load(Ordering::Relaxed) & bit != 0
}
use crate::capture::{Capture, Event};
use crate::dump::{DumpBlocks, DumpView};
use crate::frame::Frame;
use crate::snapshot::Reader;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    WalkOnly,
    Stats,
    Full,
    Produce,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Totals {
    /// Block copies per run (on the Vita each copy from PSP RAM costs
    /// ~15 us, not counted in this benchmark).
    pub copies: u32,
    /// Draws and vertices recorded in the last run (`Full` only).
    pub draws: u32,
    pub vertices: u32,
    /// The last run's `Frame::description_hash` and `vertex_hash` (`Full`
    /// only), taken here while everything the frame points to exists: its
    /// cached draws point into the vertex cache's arena, which a caller may
    /// free when the call returns (hashing the frame later crashed on the
    /// Vita, 2026-11-06).
    pub description_hash: u64,
    pub vertex_hash: u64,
}

/// Runs the dump's events `runs` times from its saved start state. `None` if
/// the state can't be restored. `capture` and `frame` are only working
/// space; `sorted_blocks` comes from `DumpView::sort_blocks`.
pub fn run(
    dump: &DumpView,
    sorted_blocks: &[u32],
    mode: Mode,
    runs: u32,
    capture: &mut Capture,
    frame: &mut Frame,
) -> Option<Totals> {
    let mut totals = Totals::default();
    let mut copies = 0;
    for _ in 0..runs {
        capture.load_state(&mut Reader::new(dump.snapshot))?;
        capture.walk_only = mode == Mode::WalkOnly;
        capture.stream().discard_jobs.store(mode == Mode::Produce, core::sync::atomic::Ordering::Relaxed);
        frame.clear();
        for index in 0..dump.event_count() {
            let (event_type, args) = dump.event(index);
            let Some(event) = Event::from_ring(event_type, args) else {
                continue;
            };
            let mut memory = BlockCache::new(DumpBlocks { view: dump, sorted: sorted_blocks });
            let frame = if matches!(mode, Mode::Full | Mode::Produce) { Some(&mut *frame) } else { None };
            capture.handle(event, &mut memory, frame);
            copies += memory.copies + memory.line_copies;
        }
    }
    capture.walk_only = false;
    capture.stream().discard_jobs.store(false, core::sync::atomic::Ordering::Relaxed);
    totals.copies = copies / runs.max(1);
    totals.draws = frame.draw_count;
    totals.vertices = frame.vertex_count;
    if mode == Mode::Full {
        totals.description_hash = frame.description_hash();
        totals.vertex_hash = frame.vertex_hash();
    }
    Some(totals)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_cache::BLOCK_SIZE;
    use crate::command::op;
    use crate::dump::{Header, HEADER_BYTES};
    use crate::frame::new_boxed_frame;
    use crate::snapshot::Writer;

    const LIST: u32 = 0x0890_0000;

    fn command(opcode: u8, data: u32) -> u32 {
        ((opcode as u32) << 24) | data
    }

    /// A dump of one frame: one list drawing one rectangle, then the flip.
    /// Its memory block comes after an unrelated one, so finding it needs
    /// the sorted index.
    fn one_rectangle_dump() -> Vec<u8> {
        let mut block = vec![0u8; BLOCK_SIZE as usize];
        let words = [
            command(op::BASE, 0x08_0000),
            command(op::VERTEX_TYPE, (1 << 23) | 2 | (7 << 2) | (2 << 7)), // 2D sprite vertex
            command(op::VERTEX_ADDRESS, 0x90_0100),
            command(op::PRIM, (6 << 16) | 2), // one rectangle
            command(op::FINISH, 0),
            command(op::END, 0),
        ];
        for (index, word) in words.iter().enumerate() {
            block[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        // Two vertices at 0x100: (u, v), colour, (x, y), z.
        let vertices: [u32; 8] = [0, 0xFFFF_FFFF, 0, 0, (16 << 16) | 16, 0xFFFF_FFFF, (16 << 16) | 16, 0];
        for (index, word) in vertices.iter().enumerate() {
            block[0x100 + index * 4..0x104 + index * 4].copy_from_slice(&word.to_le_bytes());
        }

        let mut snapshot = vec![0u8; 4096];
        let mut writer = Writer::new(&mut snapshot);
        Capture::new().save_state(&mut writer);
        let snapshot_len = writer.len();
        snapshot.truncate(snapshot_len);

        let events: [[u32; 5]; 2] = [[1, LIST, 0, 0, 0], [4, 0x0400_0000, 512, 3, 0]];
        let header = Header { snapshot_bytes: snapshot_len as u32, event_count: 2, block_count: 2, block_size: BLOCK_SIZE };
        let mut bytes = vec![0u8; HEADER_BYTES];
        header.write((&mut bytes[..]).try_into().unwrap());
        bytes.extend_from_slice(&snapshot);
        for word in events.iter().flatten() {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        bytes.extend_from_slice(&0x0A00_0000u32.to_le_bytes());
        bytes.extend_from_slice(&[0xEE; BLOCK_SIZE as usize]);
        bytes.extend_from_slice(&LIST.to_le_bytes());
        bytes.extend_from_slice(&block);
        bytes
    }

    #[test]
    fn modes_do_their_share_of_the_work() {
        let bytes = one_rectangle_dump();
        let dump = DumpView::parse(&bytes).unwrap();
        let mut index = [0u32; 8];
        let sorted = dump.sort_blocks(&mut index).unwrap();
        let mut capture = Capture::new();
        let mut frame = new_boxed_frame();

        // The list and its vertices share one block but not one line
        // (`block_cache::LINE_SIZE`): two copies for the full capture, one
        // for the modes that don't read vertices.
        let full = run(&dump, sorted, Mode::Full, 3, &mut capture, &mut frame).unwrap();
        assert_eq!((full.copies, full.draws, full.vertices), (2, 1, 6));
        // The crash of 2026-11-06: the Vita
        // hashed the bench frame after its vertex cache was freed. The bench
        // reports the hashes itself.
        assert_eq!((full.description_hash, full.vertex_hash), (frame.description_hash(), frame.vertex_hash()));
        assert_ne!(full.description_hash, 0);
        for mode in [Mode::Stats, Mode::WalkOnly] {
            let totals = run(&dump, sorted, mode, 3, &mut capture, &mut frame).unwrap();
            assert_eq!(totals, Totals { copies: 1, ..Totals::default() }, "{mode:?}");
        }
        assert!(!capture.walk_only);
    }

    #[test]
    fn truncated_dumps_are_rejected() {
        let bytes = one_rectangle_dump();
        assert!(DumpView::parse(&bytes[..bytes.len() - 1]).is_none());
        let dump = DumpView::parse(&bytes).unwrap();
        assert!(dump.sort_blocks(&mut [0u32; 1]).is_none());
    }
}
