//! The recorder's draws, queued between working out their state and
//! recording their vertices.
//!
//! Recording a frame has two halves: following the display list and working
//! out each draw's state (textures, palettes, blending; `recorder::prepare`),
//! and reading, transforming and storing its vertices
//! (`recorder::run_job`). The first half (the producer) queues a [`Job`] per
//! draw (or per batch of PRIMs) here, with copies of the PSP memory its
//! vertices are in; the second half (the consumer) takes the jobs in order
//! and fills the frame's geometry. Both halves can run in one thread (the
//! queue is drained when it is full and at the end of a frame), or the
//! consumer on another core (PPSSPP's GPU thread splits its work the same
//! way): see [`StreamHooks`].
//!
//! The queue is four rings filled in the same order: jobs, their PRIM calls,
//! transform snapshots (only when the transform changed) and the copied
//! words. Every position counts up for ever (u32, wrapping); a job remembers
//! where its calls, words and transform start, so the oldest job not yet
//! recorded tells the producer what it may overwrite. Copying the vertex
//! bytes as the list is followed reads PSP memory at the same moment as
//! recording them on the spot would.
//!
//! Who touches what (the rules that make the two threads safe): the producer
//! writes ring slots that no unrecorded job uses and then publishes the job
//! (`published`, release); the consumer reads only published jobs and then
//! counts them recorded (`recorded`, release). The frame is split the same
//! way (`Frame::split`): the producer only works on its catalog (textures,
//! palettes), the consumer only on its geometry; the producer touches the
//! geometry only when the consumer is idle (`wait_until_idle`).
//!
//! A copier (`set_copier`, a third thread) can take the copying off the
//! producer: the producer then only reserves the words of deferrable memory
//! (as for `set_reader`), and the copier copies them for each published job
//! right away. Unlike the consumer copying them itself, the copier keeps up
//! with the producer, so the words are read about when the list is walked,
//! not a frame later (when the game may have written new vertices there).
//! Jobs are taken oldest first by counting `claimed` up (compare and swap),
//! so the producer can copy too when the copier falls behind (preempted:
//! `COPIER_LAG_JOBS`); whoever copied a job sets its `copy_done` flag
//! (release) and the consumer records a job only once its flag is set. A job
//! is touched by whoever claimed it until then: its ranges and reserved words.

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, Ordering};

use crate::frame::{Draw, Frame};
use crate::memory::{physical, PspMemory};
use crate::recorder::{self, PaletteFill, PrimCall, Prepared};
use crate::stats::FrameStats;
use crate::transform::{Transform, TransformBuilder, TransformInputs};
use crate::vertex_cache::VertexCache;

/// Ring sizes. (A frame of MTX Mototrax's Los Angeles race: ~3,000 jobs,
/// 10,600 PRIMs, ~500 transforms, ~700 KB of vertices.)
/// 2026-11-06: transforms 16 -> 64, calls 2048 -> 4096 and words 32K ->
/// 64K: the producer waited for room in them ~4.3, ~2.4 and ~0.8 times per
/// race frame (`WAITS`); MTX's paused race 27.9 -> 25.4 ms per recorded
/// frame with the first two.
/// 2026-11-08: jobs 512 -> 1024, calls 4096 -> 8192, transforms 64 ->
/// 192, with the live capture moved to phycont memory (main memory had
/// 0.1 MiB to spare): under the GE sync the game waits for the walk, and
/// the walk waited for the recorder whenever the transform ring was full
/// (2-4 times per frame; a faster walk only waited more). The limits in
/// use are `JOB_LIMIT` / `TRANSFORM_LIMIT`.
pub const MAX_JOBS: usize = 1024;
pub const MAX_CALLS: usize = 8192;
pub const MAX_TRANSFORMS: usize = 192;
pub const MAX_WORDS: usize = 64 * 1024;

/// Jobs and transforms the producer may have queued and not yet recorded
/// (at most `MAX_JOBS` / `MAX_TRANSFORMS`; lower values only for live A/Bs
/// against the old sizes).
pub static JOB_LIMIT: AtomicU32 = AtomicU32::new(MAX_JOBS as u32);
pub static TRANSFORM_LIMIT: AtomicU32 = AtomicU32::new(MAX_TRANSFORMS as u32);

/// The most words one job may copy; bigger draws are recorded at once
/// (after the jobs before them).
const MAX_JOB_WORDS: usize = MAX_WORDS / 4;

/// Published jobs no copier has taken yet beyond which the producer copies
/// the oldest ones itself, down to half as many (`set_copier`). The copier
/// polls every ~50 us and the producer queues a job every ~8 us (MTX's
/// race). Live A/B 2026-11-08 (MTX paused race,
/// copier on all cores): 32 -> the capture copied ~360 jobs per frame, 97-98%
/// of frames recorded; 64 -> ~240, 98-99%; 128 -> ~160, 95-98%; 256 -> ~20,
/// 90-93% (the recorder waits for the copier).
pub static COPIER_LAG_JOBS: AtomicU32 = AtomicU32::new(64);

/// Jobs the producer copied itself because the copier was behind, since the
/// session started (capture log `copier=`); never reset here.
pub static PRODUCER_COPIES: AtomicU32 = AtomicU32::new(0);

/// Whether copiers read small ranges until two reads agree (`read_stable`;
/// on by default).
pub static STABLE_COPIES: AtomicBool = AtomicBool::new(true);
/// Ranges up to this many words are read again to check them: MTX's
/// per-frame HUD text and particle quads (24-30 words) are what the game
/// rewrites while we copy (`copy_watch`, 2026-11-08); big static meshes
/// aren't, and reading them twice would cost the copier a lot.
const STABLE_CHECK_WORDS: usize = 64;
/// Reads after the first, at most, until two agree.
const STABLE_TRIES: usize = 3;
/// Word positions 0, 1, 2, ... for reading a range again as samples (one
/// burst: consecutive positions are read as one copy).
const CONSECUTIVE: [usize; STABLE_CHECK_WORDS] = {
    let mut positions = [0; STABLE_CHECK_WORDS];
    let mut index = 0;
    while index < STABLE_CHECK_WORDS {
        positions[index] = index;
        index += 1;
    }
    positions
};
/// Reads that disagreed with the one before (`read_stable`), since the
/// session started (capture log `copier=`); never reset here.
pub static UNSTABLE_READS: AtomicU32 = AtomicU32::new(0);

/// Reads `out.len()` words at `address` with `reader`; a small range again
/// until two reads agree (at most `STABLE_TRIES` more), so a copy that
/// overlapped the game writing the range for a later frame isn't used torn
/// (MTX's race start: HUD digits garbled, long thin lines from one vertex
/// of a glyph). If the game is still writing after the last try, the last
/// read is kept. False if the words can't be read.
fn read_stable(reader: DeferredReader, address: u32, out: &mut [u32]) -> bool {
    if !(reader.read)(address, out) {
        return false;
    }
    let count = out.len();
    if count > STABLE_CHECK_WORDS || !STABLE_COPIES.load(Ordering::Relaxed) {
        return true;
    }
    let mut again = [0u32; STABLE_CHECK_WORDS];
    for _ in 0..STABLE_TRIES {
        if !(reader.sample)(address, &CONSECUTIVE[..count], &mut again[..count]) || again[..count] == *out {
            return true;
        }
        UNSTABLE_READS.fetch_add(1, Ordering::Relaxed);
        out.copy_from_slice(&again[..count]);
    }
    true
}

/// How often the producer waited for the consumer (another thread), by
/// cause: the job, call, word or transform ring was full, or the consumer
/// had to become idle (frame end, inline draws, settings). Each count is one
/// `StreamHooks::wait`. For the capture log (`waits=`); never reset here.
pub static WAITS: [AtomicU32; 5] = [const { AtomicU32::new(0) }; 5];
pub const WAIT_JOBS: usize = 0;
pub const WAIT_CALLS: usize = 1;
pub const WAIT_WORDS: usize = 2;
pub const WAIT_TRANSFORMS: usize = 3;
pub const WAIT_IDLE: usize = 4;

/// Whether a job is one PRIM or a batch (`recorder::record_batch_vertices`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum JobKind {
    Single,
    Batch,
}

/// A stretch of PSP memory copied into the queue's words.
#[derive(Clone, Copy, Debug, Default)]
pub struct CopiedRange {
    /// Physical PSP address of the first copied word (word aligned).
    pub address: u32,
    /// Where it is in the word ring (an index, not a position).
    pub first_word: u32,
    /// Words copied; 0 = none. Fewer than asked for when the rest couldn't
    /// be read: the job then reads it as unreadable, as before.
    pub word_count: u32,
    /// Not copied yet: the consumer copies the words from PSP memory itself
    /// (`DrawStream::set_reader`) just before recording the job. The
    /// producer only reserved their place in the ring.
    pub deferred: bool,
    /// `vertex_cache::data_hash` of the copied words, if `hashed`: the
    /// copier hashes the vertex range of jobs that may be 3D runs as it
    /// copies them (the words are at hand then), so the recorder doesn't
    /// read them again for a vertex-cache hit (`CopiedMemory::copied_hash`).
    /// Live A/B 2026-11-08 (MTX paused race, 3 rounds): recorder busy 33.4-33.7
    /// -> 31.1-31.3 ms per frame, 27.2-27.6 -> 28.4-28.8 fps.
    pub hash: u64,
    pub hashed: bool,
    /// Left in PSP RAM for the GPU (`QueuedDraw::zero_copy_vertices`): the
    /// copier skips it; it stays `deferred`, so whoever must read it after
    /// all copies it then.
    pub zero_copy: bool,
}

impl CopiedRange {
    const NONE: CopiedRange =
        CopiedRange { address: 0, first_word: 0, word_count: 0, deferred: false, hash: 0, hashed: false, zero_copy: false };
}

/// Reads `out.len()` words of PSP memory at physical, word-aligned
/// `address` (main RAM or VRAM, see `PspMemory::defer`) from any thread.
/// False if they can't be read.
#[derive(Clone, Copy)]
pub struct DeferredReader {
    /// `out.len()` words at `address`; false if they can't be read.
    pub read: fn(address: u32, out: &mut [u32]) -> bool,
    /// The words at `address + 4 * positions[i]` into `out[i]`, in one call
    /// (the vertex cache's samples of a run: reading them one call each made
    /// ~10,000 calls per race frame, slower than copying everything).
    pub sample: fn(address: u32, positions: &[usize], out: &mut [u32]) -> bool,
}

/// One queued draw (or batch of PRIMs with the same state).
#[derive(Clone, Copy)]
pub(crate) struct Job {
    pub kind: JobKind,
    /// The draw's state, as `recorder::prepare` worked it out.
    pub prepared: Prepared,
    /// The colour of vertices without one (GeState::material_color).
    pub material_color: u32,
    /// VERTEX_TYPE.
    pub format: u32,
    /// Its transform's position in the transform ring (3D draws).
    transform: Option<u32>,
    /// See `QueuedDraw::hash_vertices`.
    hash_vertices: bool,
    /// See `QueuedDraw::index_bounds`.
    pub index_bounds: Option<(u32, u32)>,
    /// Its PRIM calls: ring index and count.
    first_call: u32,
    call_count: u32,
    /// The memory its vertices (and indices) are in, and its palette's
    /// (the third).
    ranges: [CopiedRange; 3],
    /// A palette its draw placed in the frame: the consumer stores its
    /// words (copied as the third range) before the draw's vertices
    /// (`recorder::run_job`).
    pub palette: Option<PaletteFill>,
    /// Ring positions when it was queued (before its own calls and words;
    /// the transform position it may still need): what the producer must
    /// not overwrite while it waits.
    call_position: u32,
    word_position: u32,
    transform_floor: u32,
}

impl Job {
    const EMPTY: Job = Job {
        kind: JobKind::Single,
        prepared: Prepared::EMPTY,
        material_color: 0,
        format: 0,
        transform: None,
        hash_vertices: false,
        index_bounds: None,
        first_call: 0,
        call_count: 0,
        ranges: [CopiedRange::NONE; 3],
        palette: None,
        call_position: 0,
        word_position: 0,
        transform_floor: 0,
    };
}

/// What a draw needs queued: the transform (with its
/// `TransformCache::generation`) for 3D draws, and the PSP memory to copy:
/// (address, bytes) for its vertices and its indices (0 bytes = none).
pub(crate) struct QueuedDraw<'a> {
    pub kind: JobKind,
    pub prepared: Prepared,
    pub material_color: u32,
    pub format: u32,
    pub calls: &'a [PrimCall],
    pub transform: Option<(&'a TransformInputs, u32)>,
    pub memory: [(u32, u32); 2],
    /// The copier hashes the vertex range (`CopiedRange::hash`) for the
    /// vertex cache: runs that aren't recorded raw (`recorder::goes_raw`).
    pub hash_vertices: bool,
    /// The vertex range (`memory[0]`, or `memory[1]` after an indexed
    /// draw's indices) is left in PSP RAM for the GPU
    /// (`raw_arena::zero_copy_wanted`): not copied by anyone.
    pub zero_copy_vertices: bool,
    /// Indexed draws: the lowest and highest index, as the producer found
    /// them (`recorder::index_bounds`; the vertex range copied is theirs),
    /// so the recorder needn't scan the indices again. The indices are
    /// copied a little later: the recorder checks each against these.
    pub index_bounds: Option<(u32, u32)>,
    /// A palette the draw placed in the frame (`recorder::LoadedPalette`),
    /// whose words are copied with the draw's memory and stored by the
    /// consumer.
    pub palette: Option<PaletteFill>,
    /// The words of a palette restored from a saved state (`PaletteFill`
    /// address 0).
    pub given_palette: &'a [u32],
}

/// How the producer reaches a consumer running in another thread. Without
/// hooks the producer drains the queue itself.
pub struct StreamHooks {
    /// Tells the consumer that jobs were published.
    pub wake_consumer: fn(),
    /// Waits a little (the consumer to make room or to finish).
    pub wait: fn(),
}

/// How the producer reaches a copier thread (`DrawStream::set_copier`).
pub struct CopierHooks {
    /// Tells the copier that a job was published (called for every job:
    /// it must be cheap while the copier is awake).
    pub wake: fn(),
}

/// The ring buffers; only reached through `DrawStream`'s methods.
struct Rings {
    jobs: [Job; MAX_JOBS],
    calls: [PrimCall; MAX_CALLS],
    /// What the jobs' transforms are worked out from (by the consumer,
    /// `TransformBuilder`).
    transforms: [TransformInputs; MAX_TRANSFORMS],
    words: [u32; MAX_WORDS],
}

/// The queue (see the module documentation).
pub struct DrawStream {
    rings: UnsafeCell<Rings>,
    // Producer only.
    job_position: u32,
    /// `recorded` as the producer last read it: room checks use it and read
    /// `recorded` again only when it says the rings are full (the consumer
    /// writes that counter all the time; reading it for every job made its
    /// cache line go back and forth between the cores). An older value only
    /// shows less room than there is.
    known_recorded: u32,
    call_position: u32,
    word_position: u32,
    transform_position: u32,
    /// `TransformCache::generation` of the transform at `transform_position - 1`.
    last_transform_generation: u32,
    hooks: Option<&'static StreamHooks>,
    /// Set: the consumer (or the copier) copies deferrable vertex memory.
    reader: Option<DeferredReader>,
    /// Set: a copier thread copies the deferred ranges of published jobs
    /// before the consumer may record them.
    copier: Option<&'static CopierHooks>,
    // Shared.
    /// Jobs queued (positions below are ready for the consumer, or for the
    /// copiers when there is a copier).
    published: AtomicU32,
    /// With a copier: the next job to copy (positions below are taken by
    /// the copier or the producer, see the module documentation).
    claimed: AtomicU32,
    /// With a copier: per job slot, whether the job's ranges are copied
    /// (set by whoever copied it, cleared by the producer before it
    /// publishes the slot's next job).
    copy_done: [AtomicBool; MAX_JOBS],
    /// With a copier: jobs whose copy is finished (each published job is
    /// copied exactly once, by the copier or the producer), so all are
    /// copied when this equals `published` (`copies_complete`).
    copies_finished: AtomicU32,
    /// Jobs recorded by the consumer.
    recorded: AtomicU32,
    /// The frame the jobs go into.
    frame: AtomicPtr<Frame>,
    /// The consumer's counts (unreadable draws, frame overflows), added to
    /// the frame's statistics by the producer when the consumer is idle.
    consumer_stats: UnsafeCell<FrameStats>,
    /// Benchmark only (`bench::Mode::Produce`): the consumer counts jobs
    /// recorded without recording them, to time the producer alone.
    pub discard_jobs: AtomicBool,
    /// The vertex cache, used by whoever records the jobs (the consumer,
    /// or the producer while the consumer is idle).
    vertex_cache: UnsafeCell<Option<&'static mut VertexCache>>,
    /// Works the jobs' transforms out of their inputs; used by whoever
    /// records the jobs (as `vertex_cache`).
    transform_builder: UnsafeCell<TransformBuilder>,
}

// Safe: the producer and the consumer touch disjoint parts (see the module
// documentation); the handover goes through `published` / `recorded`.
unsafe impl Sync for DrawStream {}

impl DrawStream {
    pub const fn new() -> DrawStream {
        DrawStream {
            rings: UnsafeCell::new(Rings {
                jobs: [Job::EMPTY; MAX_JOBS],
                calls: [PrimCall::EMPTY; MAX_CALLS],
                transforms: [TransformInputs::EMPTY; MAX_TRANSFORMS],
                words: [0; MAX_WORDS],
            }),
            job_position: 0,
            known_recorded: 0,
            call_position: 0,
            word_position: 0,
            transform_position: 0,
            last_transform_generation: 0,
            hooks: None,
            reader: None,
            copier: None,
            published: AtomicU32::new(0),
            claimed: AtomicU32::new(0),
            copy_done: [const { AtomicBool::new(false) }; MAX_JOBS],
            copies_finished: AtomicU32::new(0),
            recorded: AtomicU32::new(0),
            frame: AtomicPtr::new(core::ptr::null_mut()),
            consumer_stats: UnsafeCell::new(FrameStats::new()),
            discard_jobs: AtomicBool::new(false),
            vertex_cache: UnsafeCell::new(None),
            transform_builder: UnsafeCell::new(TransformBuilder::new()),
        }
    }

    /// From now on jobs are recorded by a consumer in another thread, which
    /// calls `record_published` when woken (`None`: by the producer itself).
    pub fn set_hooks(&mut self, hooks: Option<&'static StreamHooks>) {
        self.wait_until_idle();
        self.hooks = hooks;
    }

    /// From now on the consumer copies the PSP memory of draws itself where
    /// it can (`PspMemory::defer`), with `reader` (`None`: the producer
    /// copies everything, as before). Takes the copying off the producer,
    /// which is the slower half once the vertex cache saves the consumer
    /// most of its work (MTX's race, 2026-11-05). The consumer then reads
    /// the memory a little later than the producer would have.
    pub fn set_reader(&mut self, reader: Option<DeferredReader>) {
        self.wait_until_idle();
        self.reader = reader;
    }

    /// From now on a copier copies the deferred ranges of each published job
    /// (`copy_published`) before the consumer records it (`None`: the
    /// consumer copies them when it needs them). Only useful with a reader
    /// (`set_reader`): without one nothing is deferred. Without hooks for a
    /// consumer thread the producer copies them itself when it drains the
    /// queue (as a test of the order, on the PC).
    pub fn set_copier(&mut self, copier: Option<&'static CopierHooks>) {
        self.wait_until_idle();
        self.claimed.store(self.published.load(Ordering::Relaxed), Ordering::Release);
        self.copies_finished.store(self.published.load(Ordering::Relaxed), Ordering::Release);
        self.copier = copier;
    }

    /// True if the consumer may record `job` (published): its ranges are
    /// copied, or there is no copier.
    fn is_copied(&self, job: u32) -> bool {
        self.copier.is_none() || self.copy_done[job as usize % MAX_JOBS].load(Ordering::Acquire)
    }

    /// True if no published job waits for a copier to take it.
    pub fn all_copied(&self) -> bool {
        self.claimed.load(Ordering::Relaxed) == self.published.load(Ordering::Acquire)
    }

    /// Copier: copies the deferred ranges of the published jobs nobody has
    /// taken yet, oldest first (`copy_one`). False if there was none.
    pub fn copy_published(&self) -> bool {
        let mut any = false;
        while self.copy_one() {
            any = true;
        }
        any
    }

    /// Takes the oldest published job nobody has taken yet (`claimed`,
    /// compare and swap: the copier and the producer both take jobs) and
    /// copies its deferred ranges, then marks it copied. A range that can't
    /// be read is left with no words (the job reads it as unreadable, as
    /// when the consumer copies it). False if there was no job to take.
    fn copy_one(&self) -> bool {
        let Some(job) = self.take_untaken_job(|| {}) else {
            return false;
        };
        let rings = self.rings.get();
        // Safe: a published job this caller took: the producer doesn't
        // write it any more, the consumer waits for its flag, no other
        // copier takes it, so its ranges and reserved words are this
        // caller's (see the module documentation). Raw pointers only.
        unsafe {
            let slot = job as usize % MAX_JOBS;
            let ranges = core::ptr::addr_of_mut!((*rings).jobs[slot].ranges);
            let words = core::ptr::addr_of_mut!((*rings).words) as *mut u32;
            // The vertex range of a job that may be recorded as a 3D run
            // through the vertex cache is hashed here, where its words were
            // just read (not for raw runs: `QueuedDraw::hash_vertices`).
            let hash_vertices = (*rings).jobs[slot].hash_vertices;
            for (which, range) in (*ranges).iter_mut().enumerate().filter(|(_, range)| range.deferred && !range.zero_copy) {
                let out = core::slice::from_raw_parts_mut(words.add(range.first_word as usize), range.word_count as usize);
                let read = self.reader.is_some_and(|reader| read_stable(reader, range.address, out));
                range.deferred = false;
                if read {
                    crate::copy_watch::note(range.address, out);
                    if hash_vertices && which == 0 {
                        range.hash = crate::vertex_cache::data_hash(out);
                        range.hashed = true;
                    }
                } else {
                    range.word_count = 0;
                }
            }
        }
        self.copy_done[job as usize % MAX_JOBS].store(true, Ordering::Release);
        self.copies_finished.fetch_add(1, Ordering::Release);
        true
    }

    /// Takes the oldest published job nobody has taken yet (`claimed` one
    /// up, compare and swap), or `None` if there is none. `published` is read
    /// again for every try: with the value from before a failed swap, the
    /// copier once took a job the producer hadn't published yet (the
    /// producer had taken the ones in between meanwhile, `copy_untaken`),
    /// copied a slot still being written and flagged it copied; the producer
    /// then cleared the flag while publishing that job, nobody copied it any
    /// more, and the recorder and the capture waited for it for good (MTX's
    /// paused race, 2026-11-08: the capture stopped). `before_swap` runs
    /// between reading the counters and the swap (tests: another thread
    /// there).
    fn take_untaken_job(&self, mut before_swap: impl FnMut()) -> Option<u32> {
        let mut job = self.claimed.load(Ordering::Relaxed);
        loop {
            let published = self.published.load(Ordering::Acquire);
            // `claimed` never passes `published` (both only grow, and a job
            // is taken only below a fresh `published`).
            if job == published {
                return None;
            }
            before_swap();
            match self.claimed.compare_exchange_weak(job, job.wrapping_add(1), Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => return Some(job),
                Err(now) => job = now,
            }
        }
    }

    /// Producer: true if every published job's memory is copied (always
    /// without a copier: the producer copied it when it queued the job).
    /// The GE sync waits for this (capture.rs).
    pub fn copies_complete(&self) -> bool {
        self.copier.is_none() || self.copies_finished.load(Ordering::Acquire) == self.job_position
    }

    /// Producer, where it must not go on before every job is copied (the GE
    /// sync, capture.rs): copies the jobs no copier has taken yet itself
    /// instead of waiting for the copier to reach them. The copier may
    /// still be copying jobs it took (`copies_complete` tells).
    pub fn copy_untaken(&self) {
        if self.copier.is_some() {
            let previous = crate::phase::enter(crate::phase::HELP);
            while self.copy_one() {
                PRODUCER_COPIES.fetch_add(1, Ordering::Relaxed);
            }
            crate::phase::leave(previous);
        }
    }

    /// The producer, after publishing a job: if the copier has fallen
    /// `COPIER_LAG_JOBS` behind (preempted), copies the oldest jobs itself
    /// until it is half that, so the words are still read about when the
    /// list is walked and the consumer isn't held up.
    fn help_a_lagging_copier(&self) {
        let lag = |stream: &Self| stream.job_position.wrapping_sub(stream.claimed.load(Ordering::Relaxed));
        let limit = COPIER_LAG_JOBS.load(Ordering::Relaxed);
        if lag(self) <= limit {
            return;
        }
        let previous = crate::phase::enter(crate::phase::HELP);
        while lag(self) > limit / 2 && self.copy_one() {
            PRODUCER_COPIES.fetch_add(1, Ordering::Relaxed);
        }
        crate::phase::leave(previous);
    }

    /// Gives the stream a vertex cache (`None`: none). Waits until the
    /// consumer is idle.
    pub fn set_vertex_cache(&mut self, cache: Option<&'static mut VertexCache>) {
        self.wait_until_idle();
        *self.vertex_cache.get_mut() = cache;
    }

    /// The vertex cache, while the consumer is idle (between frames).
    pub fn vertex_cache_mut(&mut self) -> Option<&mut VertexCache> {
        self.wait_until_idle();
        self.vertex_cache.get_mut().as_deref_mut()
    }

    /// Forgets the queued jobs without recording them (waits for a consumer
    /// that is recording some).
    pub fn clear(&mut self) {
        self.wait_until_idle();
        let published = self.published.load(Ordering::Relaxed);
        // Without a consumer thread (nothing waited above) the forgotten
        // jobs count as taken too, so no copier writes their words later.
        self.claimed.store(published, Ordering::Release);
        self.copies_finished.store(published, Ordering::Release);
        self.recorded.store(published, Ordering::Release);
        self.job_position = published;
        self.known_recorded = published;
        // The last transform isn't reused after this.
        self.last_transform_generation = self.last_transform_generation.wrapping_add(0x8000_0000);
        // Safe: the consumer is idle (waited for above).
        unsafe { (*self.transform_builder.get()).forget() };
    }

    /// Queues a draw for `frame`, copying its memory: published for the
    /// consumer, or (without hooks) recorded when the queue is full. A draw
    /// too big for the queue is recorded at once from `memory`, after the
    /// jobs before it.
    pub(crate) fn queue(&mut self, frame: *mut Frame, stats: &mut FrameStats, memory: &mut impl PspMemory, mut draw: QueuedDraw) {
        self.frame.store(frame, Ordering::Relaxed);
        if let Some(fill) = draw.palette.filter(|fill| fill.address == 0) {
            // Words restored from a saved state (old dumps, once): stored
            // now, while the consumer is idle.
            self.wait_until_idle_or_drain(stats);
            // Safe: the consumer is idle, so the geometry is the producer's.
            unsafe { Frame::geometry_raw(frame) }.fill_palette(fill.place, draw.given_palette);
            draw.palette = None;
        }
        let palette_memory = draw.palette.map_or((0, 0), |fill| (fill.address, 4 * fill.place.word_count));
        let words: usize =
            draw.memory.iter().chain([&palette_memory]).map(|&(address, bytes)| words_for(address, bytes)).sum();
        if words > MAX_JOB_WORDS || draw.calls.len() > MAX_CALLS / 4 {
            self.wait_until_idle_or_drain(stats);
            stats.inline_draws += 1;
            stats.inline_words += words as u32;
            // Its palette is read from `memory` too (`recorder::run_job`).
            let job = Job {
                kind: draw.kind,
                prepared: draw.prepared,
                material_color: draw.material_color,
                format: draw.format,
                palette: draw.palette,
                index_bounds: draw.index_bounds,
                ..Job::EMPTY
            };
            // Safe: the consumer is idle, so the geometry and the vertex
            // cache are the producer's.
            let mut geometry = unsafe { Frame::geometry_raw(frame) };
            let cache = unsafe { (*self.vertex_cache.get()).as_deref_mut() };
            // Worked out on its own (rare): the consumer's builder stays as it is.
            let transform = draw.transform.map(|(inputs, _)| Transform::from_inputs(inputs));
            recorder::run_job(&mut geometry, stats, &job, draw.calls, transform.as_ref(), memory, cache);
            return;
        }
        let mut recorded = self.known_recorded;
        while let Some(full) = self.full_ring(recorded, &draw, words) {
            let fresh = self.recorded.load(Ordering::Acquire);
            if fresh != recorded {
                recorded = fresh;
                continue;
            }
            match self.hooks {
                Some(hooks) => {
                    WAITS[full].fetch_add(1, Ordering::Relaxed);
                    let previous = crate::phase::enter(crate::phase::WAIT);
                    (hooks.wake_consumer)();
                    (hooks.wait)();
                    crate::phase::leave(previous);
                }
                None => self.drain(),
            }
            recorded = self.recorded.load(Ordering::Acquire);
        }
        self.known_recorded = recorded;

        // Safe: the slots written below belong to no unrecorded job
        // (`full_ring`), so the consumer doesn't read them.
        let rings = unsafe { &mut *self.rings.get() };
        let call_position = self.call_position;
        let word_position = self.word_position;
        let transform_floor = self.transform_position.wrapping_sub(1);
        let transform = draw.transform.map(|(transform, generation)| {
            if !self.has_transform(generation) {
                transform.copy_to(&mut rings.transforms[self.transform_position as usize % MAX_TRANSFORMS]);
                self.transform_position = self.transform_position.wrapping_add(1);
                self.last_transform_generation = generation;
            }
            self.transform_position.wrapping_sub(1)
        });
        let first_call = self.take(Ring::Calls, draw.calls.len());
        rings.calls[first_call..first_call + draw.calls.len()].copy_from_slice(draw.calls);
        let mut ranges = [CopiedRange::NONE; 3];
        for (range, &(address, bytes)) in ranges.iter_mut().zip(draw.memory.iter().chain([&palette_memory])) {
            *range = self.copy(&mut rings.words, memory, address, bytes);
        }
        // The vertices are the second range after an indexed draw's indices.
        let vertex_range = if draw.memory[1].1 > 0 { 1 } else { 0 };
        ranges[vertex_range].zero_copy = draw.zero_copy_vertices && ranges[vertex_range].deferred;
        // Not copied yet (the slot's previous job is recorded, so nobody
        // reads the flag now; published below with release).
        self.copy_done[self.job_position as usize % MAX_JOBS].store(false, Ordering::Relaxed);
        rings.jobs[self.job_position as usize % MAX_JOBS] = Job {
            kind: draw.kind,
            prepared: draw.prepared,
            material_color: draw.material_color,
            format: draw.format,
            transform,
            hash_vertices: draw.hash_vertices,
            index_bounds: draw.index_bounds,
            first_call: first_call as u32,
            call_count: draw.calls.len() as u32,
            ranges,
            palette: draw.palette,
            call_position,
            word_position,
            transform_floor: transform.unwrap_or(transform_floor),
        };
        self.job_position = self.job_position.wrapping_add(1);
        self.published.store(self.job_position, Ordering::Release);
        if let Some(copier) = self.copier {
            (copier.wake)();
            self.help_a_lagging_copier();
        }
        if let Some(hooks) = self.hooks {
            if self.job_position % 8 == 0 {
                (hooks.wake_consumer)();
            }
        }
    }

    /// End of a frame (producer): every queued job recorded, the consumer's
    /// counts added to `stats`. Then the frame is the producer's alone again.
    pub fn finish(&mut self, stats: &mut FrameStats) {
        self.wait_until_idle_or_drain(stats);
        self.frame.store(core::ptr::null_mut(), Ordering::Relaxed);
    }

    /// The producer without a consumer thread: copies what a copier would
    /// (if one is set), then records the jobs itself.
    fn drain(&self) {
        if self.copier.is_some() {
            self.copy_published();
        }
        self.record_published();
    }

    /// Consumer: records the published jobs (with a copier: up to the first
    /// not copied yet), in order, into the frame. False if there was none
    /// to record.
    pub fn record_published(&self) -> bool {
        let published = self.published.load(Ordering::Acquire);
        let mut recorded = self.recorded.load(Ordering::Relaxed);
        if recorded == published || !self.is_copied(recorded) {
            return false;
        }
        let frame = self.frame.load(Ordering::Relaxed);
        // Safe: the producer publishes jobs only with `frame` set, works on
        // the catalog only meanwhile, and the consumer is the only one who
        // records (and touches the geometry and `consumer_stats`) while jobs
        // are published and not recorded.
        let rings = self.rings.get();
        let (mut geometry, stats, mut cache) =
            unsafe { (Frame::geometry_raw(frame), &mut *self.consumer_stats.get(), (*self.vertex_cache.get()).as_deref_mut()) };
        if self.discard_jobs.load(Ordering::Relaxed) {
            self.recorded.store(published, Ordering::Release);
            return true;
        }
        let recording = crate::phase::recorder::enter(crate::phase::recorder::OTHER);
        while recorded != published && self.is_copied(recorded) {
            // Safe: a published job, unchanged until it is counted recorded.
            let job = unsafe { &(*rings).jobs[recorded as usize % MAX_JOBS] };
            // Measuring (`cycles`): an empty part.
            crate::cycles::add(crate::cycles::CALIBRATION, crate::cycles::start(crate::cycles::CALIBRATION));
            // Deferred ranges are copied when a run needs them whole: a run the
            // vertex cache holds needs only its samples (`find_quick`).
            // Safe: the job's own words are reserved by the producer and not
            // touched by it until the job is recorded; the rest is only read.
            let mut memory = CopiedMemory {
                words: unsafe { core::ptr::addr_of_mut!((*rings).words) as *mut u32 },
                ranges: job.ranges,
                reader: self.reader,
                lifetime: core::marker::PhantomData,
            };
            let rings = unsafe { &*rings };
            let calls = &rings.calls[job.first_call as usize..][..job.call_count as usize];
            // Safe: the builder is the consumer's (as the geometry above).
            let builder = unsafe { &mut *self.transform_builder.get() };
            let building = crate::phase::recorder::enter(crate::phase::recorder::TRANSFORM);
            let transform = job.transform.map(|position| builder.build(&rings.transforms[position as usize % MAX_TRANSFORMS]));
            crate::phase::recorder::leave(building);
            recorder::run_job(&mut geometry, stats, job, calls, transform, &mut memory, cache.as_deref_mut());
            recorded = recorded.wrapping_add(1);
            self.recorded.store(recorded, Ordering::Release);
        }
        crate::phase::recorder::leave(recording);
        true
    }

    /// True if no published job waits for the consumer.
    pub fn is_idle(&self) -> bool {
        self.recorded.load(Ordering::Acquire) == self.published.load(Ordering::Relaxed)
    }

    fn wait_until_idle(&self) {
        if let Some(hooks) = self.hooks {
            let previous = crate::phase::enter(crate::phase::WAIT);
            while !self.is_idle() {
                WAITS[WAIT_IDLE].fetch_add(1, Ordering::Relaxed);
                if let Some(copier) = self.copier {
                    (copier.wake)();
                }
                (hooks.wake_consumer)();
                (hooks.wait)();
            }
            crate::phase::leave(previous);
        }
    }

    /// Waits for the consumer to record everything (or records it), then
    /// takes over its counts.
    fn wait_until_idle_or_drain(&mut self, stats: &mut FrameStats) {
        match self.hooks {
            Some(_) => self.wait_until_idle(),
            None => self.drain(),
        }
        // Safe: the consumer is idle.
        let consumer = unsafe { &mut *self.consumer_stats.get() };
        stats.unreadable_draws += consumer.unreadable_draws;
        if consumer.unreadable_draws > 0 {
            stats.last_unreadable_address = consumer.last_unreadable_address;
        }
        stats.frame_overflows += consumer.frame_overflows;
        consumer.unreadable_draws = 0;
        consumer.frame_overflows = 0;
        for reason in 0..crate::recorder::fallback::COUNT {
            stats.fallback_prims[reason] += core::mem::take(&mut consumer.fallback_prims[reason]);
            stats.fallback_vertices[reason] += core::mem::take(&mut consumer.fallback_vertices[reason]);
        }
    }

    fn has_transform(&self, generation: u32) -> bool {
        self.transform_position != 0 && self.last_transform_generation == generation
    }

    /// The ring (`WAIT_JOBS` .. `WAIT_TRANSFORMS`) without room for this
    /// draw, with the consumer at `recorded` (`known_recorded`); `None`:
    /// room in every ring.
    fn full_ring(&self, recorded: u32, draw: &QueuedDraw, words: usize) -> Option<usize> {
        let pending = self.job_position.wrapping_sub(recorded) as usize;
        if pending >= (JOB_LIMIT.load(Ordering::Relaxed) as usize).min(MAX_JOBS) {
            return Some(WAIT_JOBS);
        }
        // What the oldest unrecorded job still uses (nothing if none waits).
        let (calls_floor, words_floor, transforms_floor) = if pending == 0 {
            (self.call_position, self.word_position, self.transform_position.wrapping_sub(1))
        } else {
            // Safe: a published job; the producer wrote it, nobody changes it.
            let oldest = unsafe { &(*self.rings.get()).jobs[recorded as usize % MAX_JOBS] };
            (oldest.call_position, oldest.word_position, oldest.transform_floor)
        };
        let new_transform = draw.transform.is_some_and(|(_, generation)| !self.has_transform(generation));
        let transforms_used = self.transform_position.wrapping_sub(transforms_floor) as usize;
        if !fits(self.call_position.wrapping_sub(calls_floor), self.call_position, draw.calls.len(), MAX_CALLS) {
            Some(WAIT_CALLS)
        } else if !fits(self.word_position.wrapping_sub(words_floor), self.word_position, words, MAX_WORDS) {
            Some(WAIT_WORDS)
        } else if new_transform && transforms_used >= (TRANSFORM_LIMIT.load(Ordering::Relaxed) as usize).min(MAX_TRANSFORMS) {
            Some(WAIT_TRANSFORMS)
        } else {
            None
        }
    }

    /// Takes `count` contiguous slots of a ring (skipping its end if they
    /// don't fit there) and returns the index of the first.
    fn take(&mut self, ring: Ring, count: usize) -> usize {
        let (position, size) = match ring {
            Ring::Calls => (&mut self.call_position, MAX_CALLS),
            Ring::Words => (&mut self.word_position, MAX_WORDS),
        };
        let mut index = *position as usize % size;
        if index + count > size {
            *position = position.wrapping_add((size - index) as u32);
            index = 0;
        }
        *position = position.wrapping_add(count as u32);
        index
    }

    /// Copies the words holding [address, address + bytes) into the word
    /// ring: all of them, or those up to the first that can't be read.
    fn copy(&mut self, ring: &mut [u32; MAX_WORDS], memory: &mut impl PspMemory, address: u32, bytes: u32) -> CopiedRange {
        let count = words_for(address, bytes);
        if count == 0 {
            return CopiedRange::NONE;
        }
        let start = physical(address) & !3;
        let first_word = self.take(Ring::Words, count);
        if self.reader.is_some() && memory.defer(start, 4 * count as u32) {
            // The consumer copies them (`record_published`).
            return CopiedRange { address: start, first_word: first_word as u32, word_count: count as u32, deferred: true, ..CopiedRange::NONE };
        }
        let out = &mut ring[first_word..first_word + count];
        let mut copied = count;
        if memory.read_words(start, out).is_none() {
            copied = 0;
            while copied < count {
                match memory.read_u32(start + 4 * copied as u32) {
                    Some(word) => out[copied] = word,
                    None => break,
                }
                copied += 1;
            }
        }
        crate::copy_watch::note(start, &out[..copied]);
        CopiedRange { address: start, first_word: first_word as u32, word_count: copied as u32, ..CopiedRange::NONE }
    }
}

impl Default for DrawStream {
    fn default() -> Self {
        DrawStream::new()
    }
}

#[derive(Clone, Copy)]
enum Ring {
    Calls,
    Words,
}

/// True if `count` more contiguous slots fit into a ring of `size` slots of
/// which `used` are in use, the next free one at `position` (a run that
/// doesn't fit before the ring's end skips it, see `DrawStream::take`).
fn fits(used: u32, position: u32, count: usize, size: usize) -> bool {
    let index = position as usize % size;
    let skipped = if index + count > size { size - index } else { 0 };
    used as usize + skipped + count <= size
}

/// Words that hold the bytes [address, address + bytes).
fn words_for(address: u32, bytes: u32) -> usize {
    if bytes == 0 {
        return 0;
    }
    let start = physical(address) & !3;
    ((physical(address) + bytes - start).div_ceil(4)) as usize
}

/// A job's copied memory, read like PSP memory: what lies outside its
/// ranges is unreadable. A deferred range (`CopiedRange::deferred`) is
/// copied from PSP memory with `reader` the first time it is read as a
/// piece (`borrow_words`, `read_words`); single words and samples are read
/// straight from PSP memory until then.
pub(crate) struct CopiedMemory<'a> {
    /// The word ring (the job's ranges are its own; see `record_published`).
    words: *mut u32,
    ranges: [CopiedRange; 3],
    reader: Option<DeferredReader>,
    lifetime: core::marker::PhantomData<&'a mut [u32]>,
}

impl CopiedMemory<'_> {
    /// The range holding the `count` words at `address`, and their offset
    /// in it, if one range has them.
    fn find(&self, address: u32, count: usize) -> Option<(usize, usize)> {
        let address = physical(address);
        if address % 4 != 0 {
            return None;
        }
        self.ranges.iter().enumerate().find_map(|(which, range)| {
            let offset = address.checked_sub(range.address)? as usize / 4;
            (offset + count <= range.word_count as usize).then_some((which, offset))
        })
    }

    /// The range's words, copied in first if it is still deferred; `None`
    /// if they can't be read.
    fn words_of(&mut self, which: usize) -> Option<&[u32]> {
        let range = &mut self.ranges[which];
        // Safe: the job's own words (see the struct).
        let words = unsafe { core::slice::from_raw_parts_mut(self.words.add(range.first_word as usize), range.word_count as usize) };
        if range.deferred {
            range.deferred = false;
            let started = crate::cycles::start(crate::cycles::COPIES);
            let copying = crate::phase::recorder::enter(crate::phase::recorder::COPY);
            let copied = self.reader.is_some_and(|reader| (reader.read)(range.address, words));
            crate::phase::recorder::leave(copying);
            crate::cycles::add(crate::cycles::COPIES, started);
            if !copied {
                range.word_count = 0;
                return None;
            }
        }
        Some(words)
    }
}

impl PspMemory for CopiedMemory<'_> {
    fn read_u32(&mut self, address: u32) -> Option<u32> {
        let (which, offset) = self.find(address, 1)?;
        let range = self.ranges[which];
        if range.deferred {
            let mut word = [0u32];
            return self.reader.is_some_and(|reader| (reader.read)(physical(address), &mut word)).then_some(word[0]);
        }
        // Safe: the job's own words.
        Some(unsafe { *self.words.add(range.first_word as usize + offset) })
    }

    fn read_words(&mut self, address: u32, out: &mut [u32]) -> Option<()> {
        let (which, offset) = self.find(address, out.len())?;
        let words = self.words_of(which)?;
        out.copy_from_slice(&words[offset..offset + out.len()]);
        Some(())
    }

    fn borrow_words(&mut self, address: u32, count: usize) -> Option<&[u32]> {
        let (which, offset) = self.find(address, count)?;
        let words = self.words_of(which)?;
        Some(&words[offset..offset + count])
    }

    fn zero_copy(&self, address: u32, bytes: u32) -> bool {
        let start = physical(address) & !3;
        let end = physical(address) + bytes;
        self.ranges.iter().any(|range| range.zero_copy && range.address <= start && end <= range.address + 4 * range.word_count)
    }

    fn copied_hash(&mut self, address: u32, count: usize) -> Option<u64> {
        let address = physical(address);
        self.ranges
            .iter()
            .find(|range| range.hashed && !range.deferred && range.address == address && range.word_count as usize == count)
            .map(|range| range.hash)
    }

    /// One reader call for all samples while the range is still deferred.
    fn read_samples(&mut self, address: u32, positions: &[usize], out: &mut [u32]) -> Option<()> {
        let last = positions.iter().copied().max()?;
        let (which, offset) = self.find(address, last + 1)?;
        let range = self.ranges[which];
        if range.deferred {
            let reader = self.reader?;
            return (reader.sample)(physical(address), positions, out).then_some(());
        }
        for (word, &position) in out.iter_mut().zip(positions) {
            // Safe: the job's own words, inside the range (checked above).
            *word = unsafe { *self.words.add(range.first_word as usize + offset + position) };
        }
        Some(())
    }
}

/// An all-zero draw, for `Prepared::EMPTY`.
pub(crate) const EMPTY_DRAW: Draw = Draw {
    first_vertex: 0,
    vertex_count: 0,
    first_index: 0,
    index_count: 0,
    texture: 0,
    flags: 0,
    framebuffer: 0,
    blend: 0,
    tests: 0,
    scissor_left: 0,
    scissor_top: 0,
    scissor_right: 0,
    scissor_bottom: 0,
    fog_color: 0,
    transform: 0,
    vertex_format: 0,
    raw_colors: 0,
    bones: 0,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::test_memory::TestMemory;

    /// The hang of 2026-11-08 (MTX's paused race, the capture stopped):
    /// the copier read `published` = 2 and `claimed` = 0; before its swap
    /// the producer published job 2 and took jobs 0, 1 and 2 itself. The
    /// copier's swap failed with `claimed` = 3, which it compared with its
    /// old `published` (2) only, and took job 3, not published yet.
    #[test]
    fn hang_20261108_a_copier_never_takes_an_unpublished_job() {
        let stream = Box::new(DrawStream::new());
        stream.published.store(2, Ordering::Relaxed);
        let mut other_thread_ran = false;
        let taken = stream.take_untaken_job(|| {
            if !other_thread_ran {
                other_thread_ran = true;
                stream.published.store(3, Ordering::Relaxed);
                stream.claimed.store(3, Ordering::Relaxed);
            }
        });
        assert_eq!(taken, None, "job 3 isn't published");
        assert_eq!(stream.claimed.load(Ordering::Relaxed), 3);
        // Once published, it is taken.
        stream.published.store(4, Ordering::Relaxed);
        assert_eq!(stream.take_untaken_job(|| {}), Some(3));
        assert_eq!(stream.take_untaken_job(|| {}), None);
    }

    /// The copier's hash of a range stands for exactly that range's words
    /// (the recorder hashes them itself otherwise, the same way).
    #[test]
    fn a_copied_hash_is_given_for_exactly_its_range() {
        let mut ring = Box::new([0u32; MAX_WORDS]);
        ring[..4].copy_from_slice(&[1, 2, 3, 4]);
        let hash = crate::vertex_cache::data_hash(&ring[..4]);
        let range = CopiedRange { address: 0x0890_0000, word_count: 4, hash, hashed: true, ..CopiedRange::NONE };
        let mut copied =
            CopiedMemory { words: ring.as_mut_ptr(), ranges: [range, CopiedRange::NONE, CopiedRange::NONE], reader: None, lifetime: core::marker::PhantomData };
        assert_eq!(copied.copied_hash(0x4890_0000, 4), Some(hash), "any mirror of the address");
        assert_eq!(copied.copied_hash(0x0890_0000, 3), None, "fewer words");
        assert_eq!(copied.copied_hash(0x0890_0004, 3), None, "another start");
        copied.ranges[0].hashed = false;
        assert_eq!(copied.copied_hash(0x0890_0000, 4), None, "not hashed");
    }

    #[test]
    fn copied_memory_reads_what_was_copied_and_nothing_else() {
        let mut memory = TestMemory::default();
        memory.write(0x0890_0000, &[1, 2, 3, 4]);
        let mut stream = Box::new(DrawStream::new());
        let mut ring = Box::new([0u32; MAX_WORDS]);
        // 6 bytes from an unaligned start: the 2 words holding them.
        let range = stream.copy(&mut ring, &mut memory, 0x0890_0002, 6);
        assert_eq!((range.address, range.word_count), (0x0890_0000, 2));
        // Asked for more than can be read: what can be.
        let partial = stream.copy(&mut ring, &mut memory, 0x0890_0008, 16);
        assert_eq!(partial.word_count, 2);

        let mut copied =
            CopiedMemory { words: ring.as_mut_ptr(), ranges: [range, partial, CopiedRange::NONE], reader: None, lifetime: core::marker::PhantomData };
        assert_eq!(copied.read_u32(0x4890_0004), Some(2), "any mirror of the address");
        assert_eq!(copied.read_u32(0x0890_000C), Some(4));
        assert_eq!(copied.read_u32(0x0890_0010), None, "not copied");
        assert_eq!(copied.borrow_words(0x0890_0000, 2), Some(&[1, 2][..]));
        assert_eq!(copied.borrow_words(0x0890_0004, 2), None, "crosses into the other range");
    }

    fn fake_sample(address: u32, positions: &[usize], out: &mut [u32]) -> bool {
        for (word, &position) in out.iter_mut().zip(positions) {
            *word = address + 4 * position as u32;
        }
        true
    }

    /// Reads "memory" whose every word holds its own address.
    const FAKE_READER: DeferredReader = DeferredReader { read: fake_read, sample: fake_sample };

    #[test]
    fn a_deferred_range_is_copied_only_when_read_whole() {
        let mut memory = Deferrable(TestMemory::default());
        let mut stream = Box::new(DrawStream::new());
        let mut ring = Box::new([0u32; MAX_WORDS]);
        stream.reader = Some(FAKE_READER);
        let range = stream.copy(&mut ring, &mut memory, 0x0890_0000, 16);
        assert!(range.deferred);
        let mut copied = CopiedMemory {
            words: ring.as_mut_ptr(),
            ranges: [range, CopiedRange::NONE, CopiedRange::NONE],
            reader: Some(FAKE_READER),
            lifetime: core::marker::PhantomData,
        };
        // Samples (the vertex cache's) come from PSP memory in one call...
        let mut samples = [0u32; 2];
        assert_eq!(copied.read_samples(0x0890_0000, &[1, 3], &mut samples), Some(()));
        assert_eq!(samples, [0x0890_0004, 0x0890_000C]);
        assert!(copied.ranges[0].deferred, "...without copying the range");
        // ...the range whole is copied in on the first borrow.
        assert_eq!(copied.borrow_words(0x0890_0004, 2), Some(&[0x0890_0004, 0x0890_0008][..]));
        assert!(!copied.ranges[0].deferred);
        let first = range.first_word as usize;
        assert_eq!(ring[first..first + 4], [0x0890_0000, 0x0890_0004, 0x0890_0008, 0x0890_000C]);
        // Copied: samples from the ring.
        assert_eq!(copied.read_samples(0x0890_0000, &[0, 2], &mut samples), Some(()));
        assert_eq!(samples, [0x0890_0000, 0x0890_0008]);
    }

    /// Zero copy: the copier leaves a job's zero-copy range in PSP RAM (the
    /// GPU reads it there), copies its other ranges, and the recorder can
    /// still read the range (copied then).
    #[test]
    fn the_copier_leaves_zero_copy_ranges_alone() {
        let mut memory = Deferrable(TestMemory::default());
        let mut stream = Box::new(DrawStream::new());
        stream.reader = Some(FAKE_READER);
        let ring = unsafe { &mut (*stream.rings.get()).words };
        let mut vertices = stream.copy(ring, &mut memory, 0x0890_0000, 4096);
        vertices.zero_copy = true;
        let indices = stream.copy(ring, &mut memory, 0x0891_0000, 16);
        unsafe { (*stream.rings.get()).jobs[0].ranges = [vertices, indices, CopiedRange::NONE] };
        stream.published.store(1, Ordering::Release);
        assert!(stream.copy_one());
        let ranges = unsafe { (*stream.rings.get()).jobs[0].ranges };
        assert!(ranges[0].deferred, "the zero-copy range isn't copied");
        assert!(!ranges[1].deferred, "the other range is");
        let mut copied = CopiedMemory {
            words: unsafe { core::ptr::addr_of_mut!((*stream.rings.get()).words) as *mut u32 },
            ranges,
            reader: Some(FAKE_READER),
            lifetime: core::marker::PhantomData,
        };
        assert!(copied.zero_copy(0x0890_0002, 4000));
        assert!(!copied.zero_copy(0x0890_0000, 4100), "beyond the range");
        assert!(!copied.zero_copy(0x0891_0000, 16), "the index range");
        assert_eq!(copied.borrow_words(0x0890_0008, 1), Some(&[0x0890_0008][..]), "still readable");
    }

    /// Test memory whose main RAM may be read later by the consumer.
    struct Deferrable(TestMemory);

    impl PspMemory for Deferrable {
        fn read_u32(&mut self, address: u32) -> Option<u32> {
            self.0.read_u32(address)
        }
        fn defer(&mut self, address: u32, _bytes: u32) -> bool {
            address >= crate::memory::RAM_START
        }
    }

    fn fake_read(address: u32, out: &mut [u32]) -> bool {
        for (index, word) in out.iter_mut().enumerate() {
            *word = address + 4 * index as u32;
        }
        true
    }

    #[test]
    fn deferrable_memory_is_left_to_the_consumer() {
        let mut memory = Deferrable(TestMemory::default());
        memory.0.write(0x0890_0000, &[1, 2, 3, 4]);
        memory.0.write(0x0400_0000, &[5, 6]);
        let mut stream = Box::new(DrawStream::new());
        let mut ring = Box::new([0u32; MAX_WORDS]);
        // Without a reader: copied now, as before.
        let copied = stream.copy(&mut ring, &mut memory, 0x0890_0000, 16);
        assert!(!copied.deferred);
        assert_eq!(ring[copied.first_word as usize..][..4], [1, 2, 3, 4]);
        // With one: only its place reserved, the whole range promised.
        stream.reader = Some(FAKE_READER);
        let deferred = stream.copy(&mut ring, &mut memory, 0x4890_0004, 8);
        assert!(deferred.deferred);
        assert_eq!((deferred.address, deferred.word_count), (0x0890_0004, 2));
        assert_eq!(ring[deferred.first_word as usize..][..2], [0, 0], "not copied by the producer");
        // Memory that isn't deferrable (here VRAM) is still copied now.
        let vram = stream.copy(&mut ring, &mut memory, 0x0400_0000, 8);
        assert!(!vram.deferred);
        assert_eq!(ring[vram.first_word as usize..][..2], [5, 6]);
    }

    /// Memory the game is writing: each read (or sample) sees the next
    /// value, up to 2, then it stays.
    static CHANGING_READS: AtomicU32 = AtomicU32::new(0);

    fn changing_value() -> u32 {
        CHANGING_READS.fetch_add(1, Ordering::Relaxed).min(2)
    }

    fn changing_read(_address: u32, out: &mut [u32]) -> bool {
        out.fill(changing_value());
        true
    }

    fn changing_sample(_address: u32, _positions: &[usize], out: &mut [u32]) -> bool {
        out.fill(changing_value());
        true
    }

    #[test]
    fn small_ranges_are_read_until_two_reads_agree() {
        let reader = DeferredReader { read: changing_read, sample: changing_sample };
        let mut small = [9u32; 24];
        CHANGING_READS.store(0, Ordering::Relaxed);
        assert!(read_stable(reader, 0x0890_0000, &mut small));
        assert_eq!(small, [2; 24], "the settled words, not the first read's");
        assert_eq!(CHANGING_READS.load(Ordering::Relaxed), 4, "read, then 0 -> 1 -> 2 -> 2 agrees");
        // Big ranges are read once.
        let mut big = [9u32; STABLE_CHECK_WORDS + 1];
        CHANGING_READS.store(0, Ordering::Relaxed);
        assert!(read_stable(reader, 0x0890_0000, &mut big));
        assert_eq!(big, [0; STABLE_CHECK_WORDS + 1]);
        assert_eq!(CHANGING_READS.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn a_run_that_doesnt_fit_before_the_end_skips_it() {
        assert!(fits(0, 10, 6, 16));
        assert!(!fits(4, 10, 7, 16), "6 left before the end are skipped: 4 + 6 + 7 > 16");
        assert!(fits(0, 10, 7, 16), "nothing in use: skipping the end leaves room");
    }
}
