//! Copies of PSP memory straight into the raw arena, for the draws the
//! capture thread writes itself (`frame_builder`, performance refactor
//! step 2).
//!
//! A task is "these bytes of PSP memory to this place in the arena", and
//! nothing else: the capture thread has already decided where the vertices
//! go and written the draw. The copier thread (vita copy_thread.rs) takes
//! tasks oldest first, as it takes the draw stream's jobs; the capture
//! thread copies the oldest ones itself when the copier falls behind, and
//! at the GE sync it copies whatever no one has taken yet, so a preempted
//! copier can't hold the game (`draw_stream::DrawStream::copy_untaken`).
//!
//! Who touches what: the producer (capture thread) writes a slot only after
//! the task that used it before is finished (`done`), then publishes it
//! (`published`, release). Tasks are taken by counting `claimed` up (compare
//! and swap; `published` read again for every try, see
//! `DrawStream::take_untaken_job` for the hang that rule fixed); whoever took
//! a task copies it, then sets its `done` flag and counts it `finished`.
//!
//! Small ranges are read until two reads agree, as the jobs' copies are
//! (`draw_stream::read_stable`: HUD text the game rewrites while we copy).

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicU32, Ordering};

use crate::memory::PspMemory;

/// Tasks published and not yet finished, at most. MTX's title: ~1,200 raw
/// runs per frame, most of them one task each.
pub const MAX_TASKS: usize = 2048;

/// Ranges up to this many bytes are read again until two reads agree
/// (`draw_stream::STABLE_COPIES`; the same limit as the jobs' 64 words).
const STABLE_CHECK_BYTES: usize = 256;
/// Reads after the first, at most, until two agree.
const STABLE_TRIES: usize = 3;

/// Tasks the producer copied itself, by cause (`producer_copied`), since
/// the session started (capture log `helped=`); never reset here.
pub static PRODUCER_TASKS: [AtomicU32; 5] = [AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0)];
/// Why the producer copied a task itself (`PRODUCER_TASKS`' index): the
/// copier more than `VRAM_LAG_TASKS` VRAM tasks behind, more than
/// `COPIER_LAG_JOBS` RAM tasks behind, at the GE sync (`copy_untaken`), at
/// the frame's end (`wait_for_tasks`), or a full ring (`CopyTasks::push`).
pub mod helped {
    pub const VRAM_LAG: usize = 0;
    pub const RAM_LAG: usize = 1;
    pub const SYNC: usize = 2;
    pub const FRAME_END: usize = 3;
    pub const RING_FULL: usize = 4;
}

/// Counts a task the producer copied itself for `cause` (`helped`).
#[inline]
pub fn producer_copied(cause: usize) {
    PRODUCER_TASKS[cause].fetch_add(1, Ordering::Relaxed);
}
/// Tasks whose source couldn't be read (their place is filled with zeros:
/// the draw is already in the frame), since the session started.
pub static UNREADABLE_TASKS: AtomicU32 = AtomicU32::new(0);

/// How a task's bytes are laid out at its destination.
#[derive(Clone, Copy, Debug)]
pub enum Layout {
    /// As they are: `CopyTask::length` bytes.
    Plain,
    /// Raw sprites (`frame_builder`): the `length` bytes are pieces of
    /// `pair` bytes (a sprite's two vertices; the last piece may be
    /// shorter), each written four times one after the other.
    Sprites { pair: u32 },
    /// Gathered vertices (`frame_builder`: an index span much larger than
    /// the corners it serves): `length` vertices, the k-th being vertex
    /// `list[k]` of the span at `source` (`stride` bytes apart, the first
    /// `used` bytes of each read; the rest of its stride zeros). The list
    /// lives in the builder's lists until the frame's end, when every task
    /// is done.
    Gather { stride: u32, used: u32, list: *const u16 },
    /// A palette's words (`frame_builder`'s draws: no job carries them):
    /// `length` bytes read into the first of `words` words at the
    /// destination (its place in `Frame::palette_words`), the rest zeros,
    /// and `texture::hash_words` of all `words` written to `hash` (its
    /// `Frame::palette_hashes` entry), as `Geometry::fill_palette` stores
    /// them. A source that can't be read whole: the words up to the first
    /// unreadable one (`recorder::store_palette`).
    Palette { words: u32, hash: *mut u32 },
    /// A texture's quick sample (`texture::quick_sample`, taken at the
    /// texture's first use in a frame: `recorder::sample_texture`): the
    /// words at `texture::quick_sample_positions_in(words)` of the data at
    /// `source` folded into one word, written to the destination (the
    /// texture's `Frame::texture_samples` entry); 0 ("not sampled") if one
    /// can't be read. `length` covers the words read (for `PspMemory::defer`).
    Sample { words: u32 },
}

/// One copy.
#[derive(Clone, Copy, Debug)]
pub struct CopyTask {
    /// PSP address of the first byte (any alignment).
    pub source: u32,
    /// Bytes to read from there (`Layout::Gather`: vertices).
    pub length: u32,
    /// Where they go (the raw arena, GPU memory on the Vita).
    pub destination: *mut u8,
    pub layout: Layout,
}

// Safe to send: the destination is memory the task's producer reserved for
// it (the raw arena; a palette's place and hash entry in the frame, given
// to this palette alone by `Catalog::add_palette`; a texture's sample entry,
// handed out once per frame by `Catalog::texture_sample_place`), touched by nobody else
// until the task is done, and a gather list stays unchanged until then.
unsafe impl Send for CopyTask {}

impl CopyTask {
    const EMPTY: CopyTask = CopyTask { source: 0, length: 0, destination: core::ptr::null_mut(), layout: Layout::Plain };

    /// Bytes written at `destination`.
    pub fn written(&self) -> usize {
        let length = self.length as usize;
        match self.layout {
            Layout::Plain => length,
            Layout::Sprites { pair } => 4 * pair as usize * length.div_ceil(pair as usize),
            Layout::Gather { stride, .. } => length * stride as usize,
            Layout::Palette { words, .. } => 4 * words as usize,
            Layout::Sample { .. } => 4,
        }
    }
}

/// Reads PSP memory for a task: `out.len()` bytes at `address` (any
/// alignment). False if they can't be read.
pub trait ByteSource {
    fn read_bytes(&mut self, address: u32, out: &mut [u8]) -> bool;

    /// The words at word-aligned `address + 4 * positions[i]` into `out[i]`
    /// (a sample task's). False if one can't be read. Default: word by word
    /// with `read_bytes`.
    fn read_samples(&mut self, address: u32, positions: &[usize], out: &mut [u32]) -> bool {
        positions.iter().zip(out.iter_mut()).all(|(&position, word)| {
            let mut bytes = [0u8; 4];
            let readable = self.read_bytes(address.wrapping_add(4 * position as u32), &mut bytes);
            *word = u32::from_le_bytes(bytes);
            readable
        })
    }
}

/// A copier's way to PSP memory (`DeferredReader::copy_bytes`, and
/// `DeferredReader::sample` for sample tasks: the address translated once,
/// a load per word; 4-byte `copy_bytes` reads went through a staging buffer).
pub struct ReaderSource {
    pub copy_bytes: fn(address: u32, out: &mut [u8]) -> bool,
    pub sample: fn(address: u32, positions: &[usize], out: &mut [u32]) -> bool,
}

impl ByteSource for ReaderSource {
    fn read_bytes(&mut self, address: u32, out: &mut [u8]) -> bool {
        (self.copy_bytes)(address, out)
    }

    fn read_samples(&mut self, address: u32, positions: &[usize], out: &mut [u32]) -> bool {
        (self.sample)(address, positions, out)
    }
}

/// The capture's own memory (`PspMemory`), for tasks it copies at once
/// (no copier, or memory a copier can't read now: `PspMemory::defer`).
pub struct MemorySource<'a, M: PspMemory>(pub &'a mut M);

impl<M: PspMemory> ByteSource for MemorySource<'_, M> {
    fn read_bytes(&mut self, address: u32, out: &mut [u8]) -> bool {
        const CHUNK_WORDS: usize = 256;
        let mut words = [0u32; CHUNK_WORDS];
        let mut done = 0;
        while done < out.len() {
            let at = address.wrapping_add(done as u32);
            let skew = (at & 3) as usize;
            let bytes = (out.len() - done).min(4 * CHUNK_WORDS - skew);
            let count = (skew + bytes).div_ceil(4);
            if self.0.read_words(at & !3, &mut words[..count]).is_none() {
                return false;
            }
            // Safe to view as bytes: `words` holds `count` read words.
            let read = unsafe { core::slice::from_raw_parts(words.as_ptr() as *const u8, 4 * count) };
            out[done..done + bytes].copy_from_slice(&read[skew..skew + bytes]);
            done += bytes;
        }
        true
    }
}

/// Copies `task` with `source`. A source that can't be read leaves zeros
/// (counted in `UNREADABLE_TASKS`); false then.
pub fn run(task: &CopyTask, source: &mut impl ByteSource, stable: bool) -> bool {
    // Zeros and the count of its own (only what can't be read).
    if let Layout::Palette { words, hash } = task.layout {
        return palette(task, (words, hash), source, stable);
    }
    if let Layout::Sample { words } = task.layout {
        return sample(task, words, source);
    }
    // Safe: the destination holds `written()` bytes reserved for this task
    // (see `CopyTask`).
    let out = unsafe { core::slice::from_raw_parts_mut(task.destination, task.written()) };
    let read = match task.layout {
        Layout::Plain => read_stable(source, task.source, out, stable),
        Layout::Sprites { pair } => copy_pieces(task, pair, source, out, stable),
        Layout::Gather { stride, used, list } => {
            // Safe: the list holds `length` numbers until the task is done.
            let list = unsafe { core::slice::from_raw_parts(list, task.length as usize) };
            gather(task.source, (stride, used), list, source, out)
        }
        Layout::Palette { .. } | Layout::Sample { .. } => unreachable!("handled above"),
    };
    if !read {
        out.fill(0);
        UNREADABLE_TASKS.fetch_add(1, Ordering::Relaxed);
    }
    read
}

/// A palette task (`Layout::Palette`): its words stored and hashed. False
/// if they couldn't all be read (counted in `UNREADABLE_TASKS`).
fn palette(task: &CopyTask, (words, hash): (u32, *mut u32), source: &mut impl ByteSource, stable: bool) -> bool {
    // Safe: the destination is the palette's place, `words` aligned words
    // reserved for it, and `hash` its own entry (see `CopyTask`).
    let (stored, hash) = unsafe { (core::slice::from_raw_parts_mut(task.destination as *mut u32, words as usize), &mut *hash) };
    let bytes = (task.length as usize).min(4 * stored.len()) / 4 * 4;
    // Safe to view the words as bytes (the Vita and the PC are little-endian).
    let out = unsafe { core::slice::from_raw_parts_mut(stored.as_mut_ptr() as *mut u8, bytes) };
    let whole = read_stable(source, task.source, out, stable);
    let readable = if whole {
        bytes / 4
    } else {
        UNREADABLE_TASKS.fetch_add(1, Ordering::Relaxed);
        (0..bytes / 4).take_while(|&n| source.read_bytes(task.source.wrapping_add(4 * n as u32), &mut out[4 * n..4 * n + 4])).count()
    };
    stored[readable..].fill(0);
    *hash = crate::texture::hash_words(stored);
    whole
}

/// A sample task (`Layout::Sample`): the texture's sample words read once
/// each (as the capture thread read them, `PspMemory::read_samples`) and
/// folded. False if one couldn't be read (counted in `UNREADABLE_TASKS`).
fn sample(task: &CopyTask, words: u32, source: &mut impl ByteSource) -> bool {
    let positions = crate::texture::quick_sample_positions_in(words as usize);
    let mut read = [0u32; crate::texture::QUICK_SAMPLES];
    let whole = source.read_samples(task.source, &positions, &mut read);
    let sample = if whole {
        crate::texture::quick_sample(&read)
    } else {
        UNREADABLE_TASKS.fetch_add(1, Ordering::Relaxed);
        0
    };
    // Safe: the destination is the texture's own sample entry (see `CopyTask`).
    unsafe { (task.destination as *mut u32).write(sample) };
    whole
}

/// Reads `out.len()` bytes at `address`; a small range again until two
/// reads agree (`draw_stream::read_stable`, for bytes).
fn read_stable(source: &mut impl ByteSource, address: u32, out: &mut [u8], stable: bool) -> bool {
    if !source.read_bytes(address, out) {
        return false;
    }
    let count = out.len();
    if !stable || count > STABLE_CHECK_BYTES {
        return true;
    }
    let mut again = [0u8; STABLE_CHECK_BYTES];
    for _ in 0..STABLE_TRIES {
        if !source.read_bytes(address, &mut again[..count]) || again[..count] == *out {
            return true;
        }
        crate::draw_stream::UNSTABLE_READS.fetch_add(1, Ordering::Relaxed);
        out.copy_from_slice(&again[..count]);
    }
    true
}

/// A gather task (`Layout::Gather`): each listed vertex read into its place;
/// vertices that follow each other in the span and in the list in one read.
/// (Not read twice: the old path read such spans whole, without a check.)
fn gather(span: u32, (stride, used): (u32, u32), list: &[u16], source: &mut impl ByteSource, out: &mut [u8]) -> bool {
    let stride_bytes = stride as usize;
    let mut k = 0;
    while k < list.len() {
        let mut end = k + 1;
        while end < list.len() && list[end] as u32 == list[end - 1] as u32 + 1 {
            end += 1;
        }
        let first = list[k] as u32;
        let bytes = (end - k - 1) * stride_bytes + used as usize;
        let at = k * stride_bytes;
        if !source.read_bytes(span.wrapping_add(first * stride), &mut out[at..at + bytes]) {
            return false;
        }
        out[at + bytes..end * stride_bytes].fill(0);
        k = end;
    }
    true
}

/// A sprite task: the source read a piece of whole units at a time, each
/// unit written four times.
fn copy_pieces(task: &CopyTask, unit: u32, source: &mut impl ByteSource, out: &mut [u8], stable: bool) -> bool {
    const PIECE_BYTES: usize = 1024;
    let unit = unit as usize;
    if unit == 0 || unit > PIECE_BYTES {
        return false;
    }
    let units_per_piece = PIECE_BYTES / unit;
    let mut buffer = [0u8; PIECE_BYTES];
    let length = task.length as usize;
    let mut done = 0;
    let mut written = 0;
    while done < length {
        let bytes = (length - done).min(units_per_piece * unit);
        if !read_stable(source, task.source.wrapping_add(done as u32), &mut buffer[..bytes], stable) {
            return false;
        }
        for piece in buffer[..bytes].chunks(unit) {
            for _ in 0..4 {
                out[written..written + piece.len()].copy_from_slice(piece);
                written += unit;
            }
        }
        done += bytes;
    }
    true
}

/// A counter alone on its cache line (the Vita's are 64 bytes): the
/// producer writes `published` while the copier swaps `claimed` and counts
/// `finished` up; on one line, every task moved the line between the cores.
#[repr(align(64))]
struct OwnLine(AtomicU32);

/// The ring of published tasks (see the module documentation).
///
/// Cheap for the producer (the capture thread, which the game waits for):
/// pushing a task normally touches nothing the copier writes. A slot is
/// known free when its old task is older than the last time every task
/// was complete (`known_done`, noted by `complete`, at least once per frame
/// end and GE sync); only otherwise does the producer read `claimed` and the
/// slot's `done` number. Measured on the Vita before (MTX race): ~650 cycles
/// per task, most of it barriers and cache lines coming from the copier's core.
///
/// `N` slots (`MAX_TASKS` by default; `draw_stream`'s VRAM ring is smaller).
pub struct CopyTasks<const N: usize = MAX_TASKS> {
    tasks: UnsafeCell<[CopyTask; N]>,
    /// Per slot: the number + 1 of the last task copied from it (written by
    /// whoever copied it, release; the producer never writes it).
    done: [AtomicU32; N],
    // Producer only.
    position: u32,
    /// Every task numbered below this is copied (`complete` saw it; a plain
    /// load and store on ARM, only the producer touches it).
    known_done: AtomicU32,
    // Shared.
    published: OwnLine,
    claimed: OwnLine,
    finished: OwnLine,
}

// Safe: the producer and the copiers touch a slot one after the other,
// handed over through `published`, `claimed` and `done` (module doc).
unsafe impl<const N: usize> Sync for CopyTasks<N> {}

impl<const N: usize> CopyTasks<N> {
    pub const fn new() -> CopyTasks<N> {
        CopyTasks {
            tasks: UnsafeCell::new([CopyTask::EMPTY; N]),
            done: [const { AtomicU32::new(0) }; N],
            position: 0,
            known_done: AtomicU32::new(0),
            published: OwnLine(AtomicU32::new(0)),
            claimed: OwnLine(AtomicU32::new(0)),
            finished: OwnLine(AtomicU32::new(0)),
        }
    }

    /// Producer: adds `task`, published with the next `publish`. Its slot's
    /// task before it must be done: known done (`known_done`), or else if
    /// nobody has taken it, the producer copies the oldest tasks itself
    /// (`copy`) until it is taken, then waits the moment its copier needs.
    /// (That task is published: fewer than `N` wait unpublished.)
    pub fn push(&mut self, task: CopyTask, mut copy: impl FnMut(&CopyTasks<N>) -> bool) {
        let slot = self.position as usize % N;
        let before = self.position.wrapping_sub(N as u32);
        let reused = self.position as usize >= N;
        if reused && before.wrapping_sub(self.known_done.load(Ordering::Relaxed)) as i32 >= 0 {
            // Whatever waits unpublished is newer than `before`: published,
            // the copiers (and `copy`) can reach it.
            self.publish();
            while self.claimed.0.load(Ordering::Acquire).wrapping_sub(before) as i32 <= 0 {
                if copy(self) {
                    producer_copied(helped::RING_FULL);
                }
            }
            while self.done[slot].load(Ordering::Acquire) != before.wrapping_add(1) {
                core::hint::spin_loop();
            }
        }
        // Safe: the slot's task before is done and nobody takes this one
        // before it is published below.
        unsafe { (*self.tasks.get())[slot] = task };
        self.position = self.position.wrapping_add(1);
    }

    /// Producer: publishes the tasks pushed so far (release: their slots are
    /// written before a copier can take them). False if there were none
    /// new. The counter is the producer's: reading it is cheap, the store is
    /// one barrier for however many tasks.
    pub fn publish(&self) -> bool {
        if self.published.0.load(Ordering::Relaxed) == self.position {
            return false;
        }
        self.published.0.store(self.position, Ordering::Release);
        true
    }

    /// Takes the oldest published task nobody has taken yet; `None` if
    /// there is none.
    fn take(&self) -> Option<u32> {
        let mut task = self.claimed.0.load(Ordering::Relaxed);
        loop {
            let published = self.published.0.load(Ordering::Acquire);
            if task == published {
                return None;
            }
            match self.claimed.0.compare_exchange_weak(task, task.wrapping_add(1), Ordering::Acquire, Ordering::Relaxed) {
                Ok(_) => return Some(task),
                Err(now) => task = now,
            }
        }
    }

    /// Copier (or the producer): copies the oldest task nobody has taken
    /// yet with `source`. False if there was none.
    pub fn copy_one(&self, source: &mut impl ByteSource, stable: bool) -> bool {
        let Some(number) = self.take() else {
            return false;
        };
        let slot = number as usize % N;
        // Safe: a published task this caller took (module documentation).
        let task = unsafe { (*self.tasks.get())[slot] };
        run(&task, source, stable);
        self.done[slot].store(number.wrapping_add(1), Ordering::Release);
        self.finished.0.fetch_add(1, Ordering::Release);
        true
    }

    /// True if every published task is taken (it may still be copied).
    pub fn all_taken(&self) -> bool {
        self.claimed.0.load(Ordering::Relaxed) == self.published.0.load(Ordering::Acquire)
    }

    /// Producer: true if every task it pushed is copied (noted for `push`:
    /// their slots are free). Never while some aren't published yet.
    pub fn complete(&self) -> bool {
        let complete = self.finished.0.load(Ordering::Acquire) == self.position;
        if complete {
            self.known_done.store(self.position, Ordering::Relaxed);
        }
        complete
    }

    /// Producer: pushed tasks nobody has taken yet (published or not).
    pub fn untaken(&self) -> u32 {
        self.position.wrapping_sub(self.claimed.0.load(Ordering::Relaxed))
    }

    /// Producer: tasks pushed so far (counts up for ever, wrapping).
    pub fn pushed(&self) -> u32 {
        self.position
    }
}

impl<const N: usize> Default for CopyTasks<N> {
    fn default() -> Self {
        CopyTasks::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// PSP memory where byte n of address a is (a + n) as u8.
    struct Counting {
        reads: u32,
        fail_at: Option<u32>,
    }

    impl ByteSource for Counting {
        fn read_bytes(&mut self, address: u32, out: &mut [u8]) -> bool {
            self.reads += 1;
            if self.fail_at.is_some_and(|fail| (address..address + out.len() as u32).contains(&fail)) {
                return false;
            }
            for (n, byte) in out.iter_mut().enumerate() {
                *byte = address.wrapping_add(n as u32) as u8;
            }
            true
        }
    }

    #[test]
    fn a_plain_task_copies_its_bytes_and_small_ones_are_read_twice() {
        let mut out = vec![0xEEu8; 20];
        let task = CopyTask { source: 0x0890_0003, length: 10, destination: out[5..].as_mut_ptr(), layout: Layout::Plain };
        let mut source = Counting { reads: 0, fail_at: None };
        assert!(run(&task, &mut source, true));
        assert_eq!(&out[5..15], &[3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        assert_eq!(&out[..5], &[0xEE; 5], "nothing before");
        assert_eq!(&out[15..], &[0xEE; 5], "nothing after");
        assert_eq!(source.reads, 2, "read again to check it");
    }

    #[test]
    fn a_sprite_task_writes_each_pair_four_times() {
        // Two sprites of 6-byte pairs, the last vertex 2 bytes short.
        let mut out = vec![0u8; 48];
        let task = CopyTask { source: 0x100, length: 10, destination: out.as_mut_ptr(), layout: Layout::Sprites { pair: 6 } };
        assert_eq!(task.written(), 48);
        assert!(run(&task, &mut Counting { reads: 0, fail_at: None }, false));
        for corner in 0..4 {
            assert_eq!(&out[6 * corner..6 * corner + 6], &[0, 1, 2, 3, 4, 5]);
            assert_eq!(&out[24 + 6 * corner..24 + 6 * corner + 4], &[6, 7, 8, 9]);
        }
    }

    /// A palette task stores its words, zeros after them, and the hash
    /// `Geometry::fill_palette` gives; partly unreadable, the words up to
    /// the first unreadable one (as `recorder::store_palette`).
    #[test]
    fn a_palette_task_stores_and_hashes_its_words() {
        let word = |n: u32| u32::from_le_bytes([n as u8, n as u8 + 1, n as u8 + 2, n as u8 + 3]);
        for (fail_at, readable) in [(None, 3), (Some(0x308), 2)] {
            let mut words = [0xEEEE_EEEEu32; 5];
            let mut hash = 0u32;
            let layout = Layout::Palette { words: 4, hash: &mut hash };
            let task = CopyTask { source: 0x300, length: 12, destination: words.as_mut_ptr() as *mut u8, layout };
            assert_eq!(task.written(), 16);
            let whole = run(&task, &mut Counting { reads: 0, fail_at }, true);
            assert_eq!(whole, fail_at.is_none());
            let expected: Vec<u32> = (0..4).map(|n| if n < readable { word(0x300 + 4 * n) } else { 0 }).collect();
            assert_eq!(&words[..4], &expected[..]);
            assert_eq!(words[4], 0xEEEE_EEEE, "nothing after its place");
            assert_eq!(hash, crate::texture::hash_words(&expected));
        }
    }

    #[test]
    fn a_sample_task_writes_the_quick_sample_of_its_words() {
        let word = |at: u32| u32::from_le_bytes([at as u8, at as u8 + 1, at as u8 + 2, at as u8 + 3]);
        // A 64x64 16-bit texture: 2,048 words, read at the middle of each eighth.
        let (source, words) = (0x1000u32, 2048u32);
        let positions = crate::texture::quick_sample_positions_in(words as usize);
        assert_eq!(positions, [128, 384, 640, 896, 1152, 1408, 1664, 1920]);
        let expected = crate::texture::quick_sample(&positions.map(|position| word(source + 4 * position as u32)));
        for (fail_at, wanted) in [(None, expected), (Some(source + 4 * 640), 0)] {
            let mut entries = [0xEEEE_EEEEu32; 2];
            let layout = Layout::Sample { words };
            let task = CopyTask { source, length: 4 * 1921, destination: entries.as_mut_ptr() as *mut u8, layout };
            assert_eq!(task.written(), 4);
            let mut memory = Counting { reads: 0, fail_at };
            assert_eq!(run(&task, &mut memory, true), fail_at.is_none());
            assert_eq!(entries, [wanted, 0xEEEE_EEEE], "its entry only");
            assert!(memory.reads <= 8, "each sample word read once, not the whole range");
        }
    }

    #[test]
    fn an_unreadable_task_leaves_zeros() {
        let mut out = vec![0xEEu8; 8];
        let task = CopyTask { source: 0x200, length: 8, destination: out.as_mut_ptr(), layout: Layout::Plain };
        assert!(!run(&task, &mut Counting { reads: 0, fail_at: Some(0x204) }, true));
        assert_eq!(out, vec![0; 8]);
    }

    #[test]
    fn the_ring_wraps_and_the_producer_copies_what_nobody_took() {
        let mut tasks = Box::new(CopyTasks::<MAX_TASKS>::new());
        let mut out = vec![0u8; MAX_TASKS * 3];
        let mut source = Counting { reads: 0, fail_at: None };
        for n in 0..(MAX_TASKS * 3) as u32 {
            let destination = unsafe { out.as_mut_ptr().add(n as usize) };
            let task = CopyTask { source: n, length: 1, destination, layout: Layout::Plain };
            // No copier: when the ring is full the producer copies the oldest.
            tasks.push(task, |ring| ring.copy_one(&mut source, false));
        }
        assert!(!tasks.complete());
        assert!(tasks.publish());
        while tasks.copy_one(&mut Counting { reads: 0, fail_at: None }, false) {}
        assert!(tasks.complete() && tasks.all_taken());
        assert!(out.iter().enumerate().all(|(n, &byte)| byte == n as u8));
    }

    /// Gathered: the listed vertices of the span in list order, 3 bytes used
    /// of 4, neighbours read together.
    #[test]
    fn a_gather_task_writes_the_listed_vertices() {
        let list = [5u16, 6, 0, 5];
        let mut out = vec![0xEEu8; 16];
        let task = CopyTask { source: 0x40, length: 4, destination: out.as_mut_ptr(), layout: Layout::Gather { stride: 4, used: 3, list: list.as_ptr() } };
        assert_eq!(task.written(), 16);
        let mut source = Counting { reads: 0, fail_at: None };
        assert!(run(&task, &mut source, true));
        assert_eq!(out, vec![0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0, 0x40, 0x41, 0x42, 0, 0x54, 0x55, 0x56, 0]);
        assert_eq!(source.reads, 3, "5 and 6 in one read");
    }

    /// The 2026-11-08 hang's rule: a copier never takes a task that isn't
    /// published (it would copy a slot still being written).
    #[test]
    fn nothing_unpublished_is_taken() {
        let tasks = CopyTasks::<MAX_TASKS>::new();
        assert!(tasks.take().is_none());
        assert!(!tasks.copy_one(&mut Counting { reads: 0, fail_at: None }, false));
        assert!(tasks.complete() && tasks.all_taken());
    }

    /// Pushed tasks wait for `publish` (published in batches): until then no
    /// copier takes them and the ring isn't complete, so a GE sync or frame
    /// end that publishes first can't miss them.
    #[test]
    fn pushed_tasks_wait_for_publish() {
        let mut tasks = CopyTasks::<MAX_TASKS>::new();
        let mut out = [0u8; 2];
        let mut source = Counting { reads: 0, fail_at: None };
        for n in 0..2 {
            let destination = unsafe { out.as_mut_ptr().add(n) };
            tasks.push(CopyTask { source: 0x10 + n as u32, length: 1, destination, layout: Layout::Plain }, |_| false);
        }
        assert!(!tasks.copy_one(&mut source, false), "not published yet");
        assert!(!tasks.complete() && tasks.all_taken() && tasks.untaken() == 2);
        assert!(tasks.publish());
        assert!(!tasks.publish(), "nothing new");
        while tasks.copy_one(&mut source, false) {}
        assert!(tasks.complete());
        assert_eq!(out, [0x10, 0x11]);
    }
}
