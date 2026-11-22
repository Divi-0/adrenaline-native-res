//! GPU memory for raw PSP vertices: runs the GPU reads in the PSP's own
//! layout.
//!
//! Instead of decoding a 3D run's vertices into `FrameVertex`es, the
//! recorder copies its bytes here as they are, and the display draws them
//! with a vertex program whose attribute formats read the PSP fields
//! (nativeres_raw_v.cg). Nothing is hashed, looked up or decoded on the CPU.
//!
//! The arena is a ring of bytes in windows of `WINDOW_BYTES`. A raw draw's
//! 16-bit indices (index runs) count vertices from the start of the window
//! its run lies in (`Draw::first_vertex` holds the window's byte offset), so:
//! - a run never crosses a window's end, and starts a whole number of
//!   strides after the window's start (padding of less than one vertex);
//! - a window holds at most 65,536 of the smallest PSP vertices (3 bytes),
//!   so no index reaches 65,536.
//!
//! Runs are written once per frame and never reused: a recorded frame's
//! runs stay untouched for the next `PROTECTED_FRAMES - 1` recorded frames
//! (frames waiting to be shown, shown, or still on the GPU), as the vertex
//! cache protects its runs. If the ring would overwrite them the run goes the
//! old way (counted in `RawStats::full`).

/// Whether 3D runs go to the GPU as raw PSP vertices. On by default since
/// 2026-11-09 (after the live checks: MTX
/// race start ~27.4 fps raw vs ~21.8 the old way, 5-minute soak clean).
/// Without a raw arena (no raw vertex program) runs go the old way anyway.
pub static ENABLED: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(true);

/// Zero copy (on on the Vita since 2026-11-09, after the race-start A/B
/// and a soak: the display switches it on once
/// it has mapped PSP RAM for the GPU; off here, so the replay never uses it):
/// big raw runs in main RAM aren't copied at all, neither by the copier nor
/// into this arena; the GPU reads them straight from PSP RAM, which the
/// display maps for it (`draw_flags::RAW_PSP_RAM`). The game is still let
/// go at the walk's end (GE sync), so it may rewrite such a run before the
/// GPU reads it: small ranges, which games rewrite (HUD text, copywatch
/// 2026-11-09: 10-30 words), are copied as before (`ZERO_COPY_MIN_BYTES`).
pub static ZERO_COPY: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// Whether indexed PRIMs keep their indices as raw draws
/// (`recorder::record_raw_converted`: the vertex span copied once, the
/// indices converted into a triangle list) instead of having every corner's
/// vertex gathered into the arena (`recorder::record_raw_gathered`, kept for
/// sparse spans). Monster Hunter Freedom draws everything indexed.
pub static INDEXED_DRAWS: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(true);

/// Texgen draws (TEXTURE_MAP_MODE 1) on the raw path where
/// `Texgen::raw_form` allows it (renderer fa3c40b). Off by default: the
/// GPU works their positions out with its own rounding, which no program
/// can make the CPU route's (measured on the Vita 2026-11-22: its multiply,
/// add and reciprocal aren't correctly rounded; a x b matched the ARM's
/// round-to-nearest in 71 % of the cases where rounding matters), and the
/// CPU route is the reference for these draws (NFS UR: 16 pixels on its
/// title, 20 on the menu). Debug `rawtexgen`, replay NATIVERES_RAW_TEXGEN=1.
pub static TEXGEN: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// Whether the display can draw raw sprites (`draw_flags::RAW_SPRITES`):
/// set once its sprite program exists (nativeres_render.c); without it 3D
/// rectangles stay on the CPU path.
pub static SPRITES: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// Runs smaller than this are copied even with zero copy on.
pub const ZERO_COPY_MIN_BYTES: u32 = 1024;

/// Whether a raw batch's vertices (`bytes` at `address`, in `format`) are
/// left in PSP RAM for the GPU (`ZERO_COPY`): main RAM (PSP 0x08000000 ..
/// 0x0A000000, not VRAM) and at least `ZERO_COPY_MIN_BYTES`. (Single-weight
/// formats were left out until 2026-11-16: the CPU read their weights to
/// fold a shared one into the matrix; the program skins them now,
/// `raw_blocks`.)
pub fn zero_copy_wanted(address: u32, bytes: u32, _format: crate::vertex::VertexFormat) -> bool {
    ZERO_COPY.load(core::sync::atomic::Ordering::Relaxed) && zero_copy_suits(address, bytes)
}

/// `zero_copy_wanted` without the switch.
fn zero_copy_suits(address: u32, bytes: u32) -> bool {
    let physical = address & 0x1FFF_FFFF;
    (0x0800_0000..0x0A00_0000).contains(&physical) && bytes >= ZERO_COPY_MIN_BYTES
}

/// Bytes per window: 65,536 vertices of the smallest PSP vertex (an 8-bit
/// position without anything else, 3 bytes).
pub const WINDOW_BYTES: u32 = 65_536 * 3;

/// Recorded frames whose runs stay put (the current one included). The
/// vertex cache keeps 6 (`vertex_cache::PROTECTED_FRAMES`): the frame being
/// recorded, the newest finished one, the one shown, the one before it on
/// the GPU, and room to spare. The same here.
pub const PROTECTED_FRAMES: usize = 6;

/// Raw runs of the current recorded frame, for the log.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RawStats {
    /// Runs copied into the arena.
    pub runs: u32,
    /// Their bytes (padding included).
    pub bytes: u32,
    /// Runs that didn't fit (recent frames still use the space).
    pub full: u32,
    /// Of all raw runs: lit by the GPU with lights (`raw_vertex::lighting::LIGHTS`).
    pub lit: u32,
    /// Runs left in PSP RAM (zero copy, not in `runs`) and their bytes.
    pub psp_ram: u32,
    pub psp_ram_bytes: u32,
    /// Indexed raw runs: with their indices converted
    /// (`recorder::record_raw_converted`), and gathered vertex by vertex
    /// (`record_raw_gathered`: the switch off, or a sparse span).
    pub converted: u32,
    pub gathered: u32,
    /// Where the arena stood when a run didn't fit: its length, head and
    /// oldest protected position (KiB, positions modulo 4 GiB) and frame.
    pub full_at: [u32; 4],
}

/// Where a run was put: its byte offset in the arena and the window it
/// lies in (byte offset of the window's start). Its first vertex is number
/// `(offset - window) / stride` of the window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Placed {
    pub offset: u32,
    pub window: u32,
}

impl Placed {
    /// The run's first vertex, counted from its window's start.
    pub fn first_vertex(&self, stride: u32) -> u32 {
        (self.offset - self.window) / stride
    }
}

pub struct RawArena {
    /// GPU memory on the Vita (read by the GPU, written only here).
    bytes: &'static mut [u8],
    /// Where the next run goes, as a position that only grows: the byte
    /// offset is `head % len` (the length is a whole number of windows, so
    /// windows start at the same positions in every lap).
    head: u64,
    /// The head at the start of each of the last `PROTECTED_FRAMES` recorded
    /// frames: `frame_starts[frame % PROTECTED_FRAMES]`.
    frame_starts: [u64; PROTECTED_FRAMES],
    /// Counts recorded frames (`next_frame`).
    frame: u64,
    pub stats: RawStats,
    /// Why the last raw draw the recorder turned down was turned down
    /// (`recorder::fallback`: arena or tables full, unreadable, other); set
    /// by the recorder as a draw passes each step, read when it falls back.
    pub refusal: usize,
    /// The head before the newest reservation and the bytes it added
    /// (`undo_last`; 0 bytes: nothing to undo). Plain numbers, not an
    /// `Option`: zeroed memory must read as "no arena" for an
    /// `Option<RawArena>` (`Capture::init_zeroed`), and a nested `Option`
    /// once gave the compiler another way to tell `None` apart.
    last_head: u64,
    last_added: u32,
    /// The window the head is in (its start position; the head may stand at
    /// its very end) and that start's byte offset in the arena: kept up to
    /// date as runs are reserved, so `reserve` needs no 64-bit division (the
    /// A9 has no divide instruction; `% len` and `/ WINDOW_BYTES` on u64 were
    /// two `__aeabi_uldivmod` calls per run, ~600 cycles with the rest).
    window_start: u64,
    window_offset: u32,
    /// The window state before the newest reservation (`undo_last`).
    last_window_start: u64,
    last_window_offset: u32,
    /// `protected_from`, worked out once per frame (`next_frame`).
    protected: u64,
}

impl RawArena {
    /// An empty arena over `bytes` (its length rounded down to whole
    /// windows; at least one window).
    pub fn new(bytes: &'static mut [u8]) -> RawArena {
        let length = bytes.len() / WINDOW_BYTES as usize * WINDOW_BYTES as usize;
        assert!(length > 0, "the raw arena needs at least one window ({WINDOW_BYTES} bytes)");
        RawArena {
            bytes: &mut bytes[..length],
            head: 0,
            frame_starts: [0; PROTECTED_FRAMES],
            frame: 0,
            stats: RawStats::default(),
            refusal: 0,
            last_head: 0,
            last_added: 0,
            window_start: 0,
            window_offset: 0,
            last_window_start: 0,
            last_window_offset: 0,
            protected: 0,
        }
    }

    /// Gives the newest reservation's room back (its run couldn't be read:
    /// `frame_builder`). Nothing if there was none since the last undo.
    pub fn undo_last(&mut self) {
        if self.last_added > 0 {
            self.head = self.last_head;
            self.window_start = self.last_window_start;
            self.window_offset = self.last_window_offset;
            self.stats.bytes -= self.last_added;
            self.stats.runs -= 1;
            self.last_added = 0;
        }
    }

    /// The arena's bytes, for the frame (`Frame::raw_vertices`).
    pub fn bytes(&self) -> &[u8] {
        self.bytes
    }

    /// A recorded frame is finished: its runs stay put for the next
    /// `PROTECTED_FRAMES - 1` frames. Returns its statistics.
    pub fn next_frame(&mut self) -> RawStats {
        self.last_added = 0;
        self.frame += 1;
        self.frame_starts[(self.frame % PROTECTED_FRAMES as u64) as usize] = self.head;
        self.protected = self.protected_from();
        core::mem::take(&mut self.stats)
    }

    /// The oldest position a run may not overwrite: the start of the oldest
    /// frame still protected. Before `PROTECTED_FRAMES` frames, 0.
    fn protected_from(&self) -> u64 {
        let oldest = self.frame.saturating_sub(PROTECTED_FRAMES as u64 - 1);
        if self.frame < PROTECTED_FRAMES as u64 - 1 {
            return 0;
        }
        self.frame_starts[(oldest % PROTECTED_FRAMES as u64) as usize]
    }

    /// Copies a run of `vertices` vertices of `stride` bytes (`bytes` holds
    /// them, the first one at its start) into the arena. `None` if it
    /// doesn't fit (too big for a window, or recent frames still use the
    /// space).
    pub fn put(&mut self, bytes: &[u8], stride: u32) -> Option<Placed> {
        let (placed, out) = self.reserve(bytes.len() as u32, stride)?;
        out.copy_from_slice(bytes);
        Some(placed)
    }

    /// Room for a run of `length` bytes, `stride` bytes per vertex, placed as
    /// `put` places it; the caller fills all of it. `None` as for `put`.
    pub fn reserve(&mut self, length: u32, stride: u32) -> Option<(Placed, &mut [u8])> {
        if length == 0 || stride == 0 || length > WINDOW_BYTES {
            return None;
        }
        let len = self.bytes.len() as u32;
        let (mut window, mut window_offset) = (self.window_start, self.window_offset);
        // The head lies in this window (or at its very end): a 32-bit
        // distance. The run starts a whole number of strides from the
        // window's start (one 32-bit division, the only one left).
        let into = (self.head - window) as u32;
        let remainder = into % stride;
        let mut start = if remainder != 0 { into + (stride - remainder) } else { into };
        if start as u64 + length as u64 > WINDOW_BYTES as u64 {
            // The next window (the arena is a whole number of them).
            window += WINDOW_BYTES as u64;
            window_offset += WINDOW_BYTES;
            if window_offset == len {
                window_offset = 0;
            }
            start = 0;
        }
        let position = window + start as u64;
        if position + length as u64 - self.protected > len as u64 {
            self.stats.full += 1;
            self.stats.full_at = [len / 1024, (self.head / 1024) as u32, (self.protected / 1024) as u32, self.frame as u32];
            return None;
        }
        let offset = window_offset + start;
        let added = (position + length as u64 - self.head) as u32;
        self.stats.bytes += added;
        self.stats.runs += 1;
        self.last_head = self.head;
        self.last_added = added;
        self.last_window_start = self.window_start;
        self.last_window_offset = self.window_offset;
        self.head = position + length as u64;
        self.window_start = window;
        self.window_offset = window_offset;
        let placed = Placed { offset, window: window_offset };
        Some((placed, &mut self.bytes[offset as usize..(offset + length) as usize]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_copy_suits_big_runs_in_main_ram() {
        assert!(zero_copy_suits(0x0890_0002, 4096));
        assert!(zero_copy_suits(0x4890_0000, 4096), "uncached address");
        assert!(!zero_copy_suits(0x0890_0000, 512), "small: copied");
        assert!(!zero_copy_suits(0x0410_0000, 4096), "VRAM");
    }

    fn arena(windows: usize) -> RawArena {
        let bytes: &'static mut [u8] = Box::leak(vec![0u8; windows * WINDOW_BYTES as usize + 100].into_boxed_slice());
        RawArena::new(bytes)
    }

    /// `reserve` without 64-bit divisions places every run exactly where
    /// the plain formula (head, window = head / WINDOW_BYTES, offset = position
    /// % length) placed it: over laps of the arena, with undos and frame ends
    /// mixed in, and strides 3..=60.
    #[test]
    fn reservations_match_the_plain_formula() {
        struct Model {
            head: u64,
            last: Option<u64>,
            frame_starts: [u64; PROTECTED_FRAMES],
            frame: u64,
        }
        impl Model {
            fn protected(&self) -> u64 {
                if self.frame < PROTECTED_FRAMES as u64 - 1 {
                    return 0;
                }
                self.frame_starts[((self.frame - (PROTECTED_FRAMES as u64 - 1)) % PROTECTED_FRAMES as u64) as usize]
            }
            fn reserve(&mut self, length: u32, stride: u32, len: u64) -> Option<Placed> {
                let (length, w) = (length as u64, WINDOW_BYTES as u64);
                let mut position = self.head;
                let mut window = position / w * w;
                let into = (position - window) % stride as u64;
                if into != 0 {
                    position += stride as u64 - into;
                }
                if position + length > window + w {
                    window += w;
                    position = window;
                }
                if position + length - self.protected() > len {
                    return None;
                }
                self.last = Some(self.head);
                self.head = position + length;
                Some(Placed { offset: (position % len) as u32, window: (window % len) as u32 })
            }
        }
        let mut arena = arena(3);
        let len = arena.bytes().len() as u64;
        let mut model = Model { head: 0, last: None, frame_starts: [0; PROTECTED_FRAMES], frame: 0 };
        let mut seed = 0x1234_5678u32;
        let mut next = |limit: u32| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed % limit
        };
        for step in 0..20_000 {
            match next(40) {
                0 => {
                    arena.next_frame();
                    model.frame += 1;
                    model.frame_starts[(model.frame % PROTECTED_FRAMES as u64) as usize] = model.head;
                    model.last = None;
                }
                1 => {
                    arena.undo_last();
                    if let Some(head) = model.last.take() {
                        model.head = head;
                    }
                }
                _ => {
                    let stride = 3 + next(58);
                    let limit = if next(10) == 0 { WINDOW_BYTES } else { 4000 };
                    let length = 1 + next(limit);
                    let got = arena.reserve(length, stride).map(|(placed, _)| placed);
                    assert_eq!(got, model.reserve(length, stride, len), "step {step}: {length} bytes, stride {stride}");
                }
            }
        }
        assert!(model.head > 4 * len, "went round the arena a few times");
    }

    #[test]
    fn runs_start_a_whole_number_of_strides_into_their_window() {
        let mut arena = arena(2);
        let a = arena.put(&[1; 12 * 5], 12).unwrap();
        assert_eq!(a, Placed { offset: 0, window: 0 });
        // A 16-byte format after 60 bytes: padded to 64 (4 strides of 16).
        let b = arena.put(&[2; 16 * 3], 16).unwrap();
        assert_eq!(b.offset, 64);
        assert_eq!(b.first_vertex(16), 4);
        assert_eq!(&arena.bytes()[64..112], &[2; 48][..]);
        // A 20-byte run after byte 112: next multiple of 20 is 120.
        let c = arena.put(&[3; 20], 20).unwrap();
        assert_eq!(c.offset, 120);
        assert_eq!(c.first_vertex(20), 6);
    }

    #[test]
    fn a_run_never_crosses_a_window_and_indices_stay_16_bit() {
        let mut arena = arena(2);
        // Fill the first window up to 6 bytes before its end.
        let fill = WINDOW_BYTES as usize - 6;
        arena.put(&vec![0; fill], 1).unwrap();
        let next = arena.put(&[9; 12], 12).unwrap();
        assert_eq!(next, Placed { offset: WINDOW_BYTES, window: WINDOW_BYTES });
        // The last vertex a window can hold, for the smallest stride, is number 65,535.
        assert_eq!((WINDOW_BYTES - 3) / 3, 65_535);
    }

    #[test]
    fn recent_frames_are_not_overwritten() {
        let mut arena = arena(2);
        let run = vec![7u8; WINDOW_BYTES as usize];
        // One window per frame: the arena holds two frames' runs.
        assert!(arena.put(&run, 4).is_some());
        arena.next_frame();
        assert!(arena.put(&run, 4).is_some());
        arena.next_frame();
        // The third would overwrite frame 0's run, still protected.
        assert!(arena.put(&run, 4).is_none());
        assert_eq!(arena.stats.full, 1);
        // Once frame 0 is older than PROTECTED_FRAMES frames, its space is free again.
        for _ in 0..PROTECTED_FRAMES {
            arena.next_frame();
        }
        assert!(arena.put(&run, 4).is_some());
    }

    #[test]
    fn small_frames_wrap_around_the_ring() {
        let mut arena = arena(2);
        for frame in 0..100u32 {
            let placed = arena.put(&[frame as u8; 30_000], 12).unwrap();
            assert_eq!((placed.offset - placed.window) % 12, 0);
            assert!(placed.offset + 30_000 <= placed.window + WINDOW_BYTES);
            arena.next_frame();
        }
    }
}
