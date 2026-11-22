//! The capture thread: reads the event ring the nativeres.prx PSP plugin
//! writes (`nativeres_core::event_ring`), passes each event to the core's
//! `Capture`, logs one line of statistics per frame (`log`), and records
//! frames for the display thread (`frames`) while the renderer is on.
//!
//! It also keeps the PSP plugin's VRAM copy requests up to date
//! (`nativeres_core::vram_requests`) and records frame dumps for the PC
//! replay on request (`dump_file`).
//!
//! If the plugin isn't loaded the ring never gets its magic value and the
//! thread just sleeps.

use core::ffi::c_void;
use core::fmt::Write;
use core::ptr::{addr_of, addr_of_mut, read_volatile, write_volatile};
use core::sync::atomic::{AtomicBool, AtomicI32, AtomicPtr, AtomicU32, AtomicU64, Ordering};

use nativeres_core::block_cache::{BlockCache, BlockSource, BlockStore, BLOCK_SIZE, BLOCK_WORDS, CAPTURE_BLOCKS};
use nativeres_core::capture::{Capture, Event};
use nativeres_core::draw_stream;
use nativeres_core::event_ring::{self, Next, RawEvent, Ring, RingHeader, RingReader, RingSource};
use nativeres_core::frame::{Frame, FrameVertex};
use nativeres_core::memory::{RAM_END, RAM_START};
use nativeres_core::psp_link::{copy_method, VramRequestTable, VRAM_REQUESTS_ADDRESS};
use nativeres_core::frame_builder::DirectLists;
use nativeres_core::vertex_cache::VertexCache;
use nativeres_core::report::{self, TextBuffer};
use nativeres_core::stats::FrameStats;
use nativeres_core::texture_cache::VramState;
use nativeres_core::texture;
use nativeres_core::vram_requests::{Plan, VramRange, VramRequests, MAX_RANGES, VRAM_SIZE, VRAM_START};

use crate::dump::DumpRecorder;
use crate::{copy_thread, ext, frames, log, platform, recorder_thread, stage_thread, sys, SingleThreaded};

/// How often the ring is checked while a game with the plugin runs.
const POLL_INTERVAL_US: u32 = 1000;
/// How often we look for the plugin when it isn't running.
const IDLE_INTERVAL_US: u32 = 200 * 1000;
const PRIORITY: i32 = 0x70;
/// The core keeps its big state in statics, but per-frame statistics
/// (~1 KiB) are copied around on the stack.
const STACK_SIZE: u32 = 0x10000;

/// Asks the plugin's copy thread for one GE probe
/// of that mode (psp/, probe_ge_on_request; `vram_probe` reads the
/// result).
pub fn request_ge_probe(mode: u32) {
    let table = vram_table();
    // Safe: the table is mapped PSP memory the plugin also uses.
    unsafe {
        write_volatile(addr_of_mut!((*table).ge_probe_result), 0);
        write_volatile(addr_of_mut!((*table).ge_probe_request), mode);
        crate::adrenaline::writeback(table as *mut c_void, core::mem::size_of::<VramRequestTable>() as u32);
    }
}

/// The plugin's copy-method probe: bytes, microseconds
/// per method (`probe_us[method - 1]`, core psp_link), which copied
/// correctly (bits), the method chosen, the DMA lookup that found a function.
pub fn vram_probe(out: *mut u32) {
    let table = vram_table();
    // Safe: the table is mapped PSP memory; the caller passes 17 words.
    unsafe {
        *out = read_volatile(addr_of!((*table).probe_bytes));
        for method in 0..8 {
            *out.add(1 + method) = read_volatile(addr_of!((*table).probe_us[method]));
        }
        *out.add(9) = read_volatile(addr_of!((*table).probe_ok));
        *out.add(10) = read_volatile(addr_of!((*table).probe_chosen));
        *out.add(11) = read_volatile(addr_of!((*table).probe_dma_lookup));
        *out.add(12) = read_volatile(addr_of!((*table).ge_probe_request));
        *out.add(13) = read_volatile(addr_of!((*table).ge_probe_result)) as u32;
        *out.add(14) = read_volatile(addr_of!((*table).ge_probe_qid)) as u32;
        *out.add(15) = read_volatile(addr_of!((*table).ge_probe_state)) as u32;
        *out.add(16) = read_volatile(addr_of!((*table).ge_probe_polls));
    }
}

fn vram_table() -> *mut VramRequestTable {
    sys::psp_to_vita(VRAM_REQUESTS_ADDRESS) as *mut VramRequestTable
}

/// The VRAM copy state for the display thread's texture cache.
pub fn vram_state() -> VramState {
    let table = vram_table();
    // Safe: the table is mapped PSP memory the plugin also uses; all
    // accesses are volatile.
    unsafe {
        VramState {
            generation: read_volatile(addr_of!((*table).generation)),
            copied_generation: read_volatile(addr_of!((*table).copied_generation)),
            changes: read_volatile(addr_of!((*table).changes)),
        }
    }
}

/// Time the capture's block copies for the log:
/// the two clock reads per copy cost ~180 us per frame in FF1's menu.
pub static COPY_TIMING: AtomicI32 = AtomicI32::new(0);
/// The frame number of the last frame the capture completed. Texture data
/// copied after it was set may already be newer than what that frame drew.
pub static CAPTURED_FRAME_NUMBER: AtomicU32 = AtomicU32::new(0);
/// Frames the capture has seen end in this session, recorded or not (a
/// recorded frame's `frame_number` is the count before it ended). Equal to
/// `frame_number + 1` of a frame while the game hasn't shown another since.
pub static COMPLETED_FRAME_COUNT: AtomicU32 = AtomicU32::new(0);

/// Whether textured 3D draws are left to the
/// GPU's transform (`Capture::set_gpu_transform`; on by default). Only
/// used once the display side has its program for it
/// (`GPU_TRANSFORM_READY`).
pub static GPU_TRANSFORM_WANTED: AtomicU32 = AtomicU32::new(1);

/// Whether the capture's block cache copies PSP
/// memory a line (256 bytes) at a time as it is read (1, default) or whole
/// 1 KiB blocks (0, as before 2026-11-05; for comparing).
pub static COPY_LINES: AtomicU32 = AtomicU32::new(1);

/// Whether the recorder thread copies the draws'
/// vertex memory itself (`DrawStream::set_reader`), instead of the capture
/// thread. Only with the recorder thread. Off by default since 2026-11-07:
/// a recorder thread behind the game copied vertices the game had already
/// overwritten (MTX's title menu: torn riders, garbage over the menu).
pub static DEFERRED_COPY_WANTED: AtomicU32 = AtomicU32::new(0);

/// Whether the vertex cache finds runs by their mini
/// hash (`nativeres_core::vertex_cache::QUICK_HITS`).
pub fn set_quick_hits(on: u32) {
    nativeres_core::vertex_cache::QUICK_HITS.store(on != 0, Ordering::Relaxed);
}

/// Whether cached textures get a few-word check at
/// their first use in a frame (`nativeres_core::texture_cache::QUICK_CHECKS`).
pub fn set_quick_checks(on: u32) {
    nativeres_core::texture_cache::QUICK_CHECKS.store(on != 0, Ordering::Relaxed);
}

/// How far the copier thread may fall behind
/// before the capture thread copies the oldest jobs itself
/// (`nativeres_core::draw_stream::COPIER_LAG_JOBS`; at least 2).
pub fn set_copier_lag(jobs: u32) {
    let jobs = jobs.clamp(2, nativeres_core::draw_stream::MAX_JOBS as u32);
    nativeres_core::draw_stream::COPIER_LAG_JOBS.store(jobs, Ordering::Relaxed);
}

/// Note every copied range and check at each frame's
/// end whether the game rewrote it (`nativeres_core::copy_watch`; log
/// `rewritten=`). Off by default.
pub fn set_copy_watch(on: u32) {
    nativeres_core::copy_watch::ON.store(on != 0, Ordering::Relaxed);
}

/// Debug `readwatch`: at the end of every event (the game still held at its
/// GE sync) the lines the walk read through the block cache are read again
/// and compared (`BlockCache::changed_lines`); log `readwatch=changed/
/// checked@first`. Off by default (it reads everything twice).
pub static READ_WATCH: AtomicBool = AtomicBool::new(false);
static READ_CHANGED: AtomicU32 = AtomicU32::new(0);
static READ_CHECKED: AtomicU32 = AtomicU32::new(0);
static READ_FIRST: AtomicU32 = AtomicU32::new(0);

/// Debug `rawhash`: `Frame::raw_draw_hash` of every recorded frame when it
/// is complete (log `rawhash=<frame>:<hash>`) and of every frame the
/// display draws (`note_drawn_hash`; log `drawn=<frame>:<hash>,...`, those
/// drawn since the last line). The same PSP frame hashes the same in every
/// run of a deterministic scene; a one-frame glitch whose recorded hash is
/// off comes from the capture side, one whose drawn hash alone is off from
/// later on. Off by default (it reads the frame's raw data again).
pub static RAW_HASH: AtomicBool = AtomicBool::new(false);
static RECORDED_HASH: AtomicU64 = AtomicU64::new(0);
static RECORDED_HASH_FRAME: AtomicU32 = AtomicU32::new(0);
static RECORDED_PARTS: [AtomicU32; 4] = [const { AtomicU32::new(0) }; 4];
const DRAWN_HASHES: usize = 8;
static DRAWN_HASHES_WRITTEN: AtomicU32 = AtomicU32::new(0);
static DRAWN_HASHES_LOGGED: AtomicU32 = AtomicU32::new(0);
static DRAWN_HASH_FRAMES: [AtomicU32; DRAWN_HASHES] = [const { AtomicU32::new(0) }; DRAWN_HASHES];
static DRAWN_HASH_VALUES: [AtomicU64; DRAWN_HASHES] = [const { AtomicU64::new(0) }; DRAWN_HASHES];

/// Debug `skinwatch`: `Frame::longest_skinned_edge` of every recorded
/// frame when it is complete (log `skinwatch=<edge px>@<draw>/<skinned
/// triangles>`): MHF's one-frame streaks are huge triangles of skinned
/// characters. Off by default (it transforms every skinned vertex on the
/// CPU). `dump-glitch` (debug plugin) uses the same measure.
pub static SKIN_WATCH: AtomicBool = AtomicBool::new(false);
static SKIN_EDGE: AtomicU32 = AtomicU32::new(0);
static SKIN_DRAW: AtomicU32 = AtomicU32::new(0);
static SKIN_TRIANGLES: AtomicU32 = AtomicU32::new(0);

/// Display thread, debug `rawhash`: `frame` is about to be drawn.
pub fn note_drawn_hash(frame: &Frame) {
    if !RAW_HASH.load(Ordering::Relaxed) {
        return;
    }
    let slot = DRAWN_HASHES_WRITTEN.load(Ordering::Relaxed) as usize % DRAWN_HASHES;
    DRAWN_HASH_FRAMES[slot].store(frame.frame_number, Ordering::Relaxed);
    DRAWN_HASH_VALUES[slot].store(frame.raw_draw_hash(), Ordering::Relaxed);
    DRAWN_HASHES_WRITTEN.fetch_add(1, Ordering::Release);
}

/// Copiers read small ranges until two reads agree
/// (`nativeres_core::draw_stream::STABLE_COPIES`; on by default).
pub fn set_stable_copies(on: u32) {
    nativeres_core::draw_stream::STABLE_COPIES.store(on != 0, Ordering::Relaxed);
}

/// The clock `copy_watch` notes copies with (microseconds, low 32 bits).
fn clock_us() -> u32 {
    sys::now_us() as u32
}

/// Lit draws with lights drawn lit (1, default) or with
/// their vertex colours as they are (0) (`nativeres_core::lighting::LIGHTS_ON`).
pub fn set_lights(on: u32) {
    nativeres_core::lighting::LIGHTS_ON.store(on != 0, Ordering::Relaxed);
}

/// Words a vertex-cache mini hash samples
/// (`nativeres_core::vertex_cache::SAMPLE_COUNT`).
pub fn set_mini_samples(count: u32) {
    let count = (count as usize).clamp(1, nativeres_core::vertex_cache::SAMPLE_WORDS);
    nativeres_core::vertex_cache::SAMPLE_COUNT.store(count, Ordering::Relaxed);
}


/// The recorder thread's way to PSP memory (`DeferredReader`): plain main
/// RAM, copied as the capture's direct reads are (`VitaBlocks::copy_words`).
fn read_psp_ram(address: u32, out: &mut [u32]) -> bool {
    let bytes = 4 * out.len() as u32;
    let source = platform::psp_memory(address, bytes);
    if source.is_null() {
        return false;
    }
    // Safe: `out` holds `bytes` bytes; `source` is mapped PSP RAM
    // (`PspMemory::defer`: main RAM, and VRAM blocks the plugin's copy holds;
    // `psp_memory` leads VRAM to that copy).
    unsafe { platform::psp_copy(out.as_mut_ptr() as *mut u8, source, bytes) };
    true
}
/// The vertex cache's samples of a run (`DeferredReader::sample`): the
/// address translated once, then a load per sample.
fn sample_psp_ram(address: u32, positions: &[usize], out: &mut [u32]) -> bool {
    let Some(&last) = positions.iter().max() else {
        return true;
    };
    let source = platform::psp_memory(address, 4 * (last as u32 + 1)) as *const u32;
    if source.is_null() {
        return false;
    }
    // Consecutive positions (`vertex_cache::SAMPLE_BURSTS`) in one copy:
    // uncached PSP RAM costs per load, and a VLDM burst of up to 64 bytes
    // costs about what one word does.
    let mut index = 0;
    while index < positions.len() {
        let mut end = index + 1;
        while end < positions.len() && positions[end] == positions[end - 1] + 1 {
            end += 1;
        }
        if end - index == 1 {
            // Safe: inside the mapped range just asked for (the game may
            // write it meanwhile: volatile).
            out[index] = unsafe { core::ptr::read_volatile(source.add(positions[index])) };
        } else {
            // Safe: `out[index..end]` holds those words, inside the range.
            unsafe {
                platform::psp_copy(
                    out[index..end].as_mut_ptr() as *mut u8,
                    source.add(positions[index]) as *const u8,
                    4 * (end - index) as u32,
                )
            };
        }
        index = end;
    }
    true
}

/// The copier's copy tasks (`DeferredReader::copy_bytes`, core copy_tasks):
/// `out.len()` bytes at any address into any place of the raw arena. Both
/// word-aligned: straight there (`psp_copy`'s bursts); otherwise the words
/// holding them are read in bursts into a buffer on the stack and the bytes
/// copied out of it (cached memory), as the recorder copied the word ring's
/// words into the arena before.
fn copy_psp_bytes(address: u32, out: &mut [u8]) -> bool {
    const STAGING_BYTES: usize = 2048;
    let skew = (address & 3) as usize;
    let words_bytes = (skew + out.len()).next_multiple_of(4);
    let source = platform::psp_memory(address & !3, words_bytes as u32);
    if source.is_null() {
        return false;
    }
    if skew == 0 && out.as_ptr() as usize & 3 == 0 {
        // Safe: `out` holds its length; `source` is mapped PSP memory of
        // at least that length (the game may write it meanwhile).
        unsafe { platform::psp_copy(out.as_mut_ptr(), source, out.len() as u32) };
        return true;
    }
    // Not zeroed: only the bytes `psp_copy` wrote are read back (a zeroed
    // 2 KiB buffer cost every unaligned task a memset: ~1 in 4 of MTX's).
    let mut staging = core::mem::MaybeUninit::<[u32; STAGING_BYTES / 4]>::uninit();
    let mut done = 0;
    while done < out.len() {
        let from = skew + done; // bytes from the first word
        let lead = from & 3;
        let take = (out.len() - done).min(STAGING_BYTES - lead);
        let read = (lead + take).next_multiple_of(4);
        // Safe: `staging` holds `read` bytes; the words read lie inside the
        // `words_bytes` mapped above.
        unsafe { platform::psp_copy(staging.as_mut_ptr() as *mut u8, source.add(from - lead), read as u32) };
        // Safe to view as bytes: `read` bytes of `staging` were written just now.
        let bytes = unsafe { core::slice::from_raw_parts(staging.as_ptr() as *const u8, read) };
        out[done..done + take].copy_from_slice(&bytes[lead..lead + take]);
        done += take;
    }
    true
}

/// The recorder thread's (and the copier's) way to PSP memory.
const PSP_RAM_READER: nativeres_core::draw_stream::DeferredReader =
    nativeres_core::draw_stream::DeferredReader { read: read_psp_ram, sample: sample_psp_ram, copy_bytes: copy_psp_bytes };

/// Set by the display side (programs.rs) once its 3D vertex program
/// is made: without it, draws in model space couldn't be drawn.
pub static GPU_TRANSFORM_READY: AtomicU32 = AtomicU32::new(0);

/// The vertex cache's arena: main memory the display side
/// allocated and mapped for the GPU (arenas.rs); null until then
/// or if it couldn't.
pub static VERTEX_ARENA: AtomicPtr<FrameVertex> = AtomicPtr::new(core::ptr::null_mut());
pub static VERTEX_ARENA_VERTICES: AtomicU32 = AtomicU32::new(0);

/// The vertex cache's tables (`make_vertex_cache`), for the memory plan:
/// in phycont memory, whose blocks come in whole MiB, since the cache is on
/// by default (2026-11-06) and main memory is short.
const VERTEX_CACHE_TABLE_BYTES: usize = (core::mem::size_of::<VertexCache>() + 0xF_FFFF) & !0xF_FFFF;
#[no_mangle]
#[used]
#[allow(non_upper_case_globals)]
pub static nativeres_plan_phycont_vertex_cache_tables: u32 = VERTEX_CACHE_TABLE_BYTES as u32;


/// Capture thread: the vertex cache over the display's arena, made once the
/// arena exists (its tables in phycont memory). `None` if there is no arena
/// yet or the memory for the tables is missing.
fn make_vertex_cache() -> Option<&'static mut VertexCache> {
    let arena = VERTEX_ARENA.load(Ordering::Acquire);
    let vertices = VERTEX_ARENA_VERTICES.load(Ordering::Acquire) as usize;
    if arena.is_null() || vertices == 0 {
        return None;
    }
    let size = VERTEX_CACHE_TABLE_BYTES;
    // Safe: plain system calls; the block is zeroed before use, and the
    // arena is ours alone (the display only reads what frames point at).
    unsafe {
        let block = sys::alloc_block(
            b"NativeResVertexCache\0".as_ptr(),
            sys::SCE_KERNEL_MEMBLOCK_TYPE_USER_MAIN_PHYCONT_RW,
            size as u32,
            core::ptr::null_mut(),
        );
        if block < 0 {
            return None;
        }
        let mut base: *mut c_void = core::ptr::null_mut();
        sys::sceKernelGetMemBlockBase(block, &mut base);
        core::ptr::write_bytes(base as *mut u8, 0, size);
        let place = base as *mut VertexCache;
        VertexCache::init_zeroed(place, core::slice::from_raw_parts_mut(arena, vertices));
        // The raw arena, if the display made one (before the vertex arena,
        // so it is there now if at all).
        let raw = RAW_ARENA.load(Ordering::Acquire);
        let raw_bytes = RAW_ARENA_BYTES.load(Ordering::Acquire) as usize;
        if !raw.is_null() && raw_bytes >= nativeres_core::raw_arena::WINDOW_BYTES as usize {
            (*place).set_raw_arena(core::slice::from_raw_parts_mut(raw, raw_bytes));
        }
        // The index ring likewise (made with the raw arena).
        let ring = INDEX_RING.load(Ordering::Acquire);
        let ring_bytes = INDEX_RING_BYTES.load(Ordering::Acquire) as usize;
        if !ring.is_null() && ring_bytes >= nativeres_core::raw_arena::WINDOW_BYTES as usize {
            (*place).set_index_ring(core::slice::from_raw_parts_mut(ring, ring_bytes));
        }
        Some(&mut *place)
    }
}

/// The capture thread's lists for the raw draws it writes itself
/// (`nativeres_core::frame_builder::DirectLists`, ~1.1 MiB), in phycont
/// memory (blocks of whole MiB), made with the vertex cache.
const DIRECT_LISTS_BYTES: usize = (core::mem::size_of::<DirectLists>() + 0xF_FFFF) & !0xF_FFFF;
#[no_mangle]
#[used]
#[allow(non_upper_case_globals)]
pub static nativeres_plan_phycont_direct_lists: u32 = DIRECT_LISTS_BYTES as u32;

/// Capture thread: the builder's lists, or `None` if the memory is missing
/// (every draw then goes through the job queue, as before step 2).
fn make_direct_lists() -> Option<&'static mut DirectLists> {
    // Safe: plain system calls; the block is zeroed before use and ours alone.
    unsafe {
        let block = sys::alloc_block(
            b"NativeResDirectDraws\0".as_ptr(),
            sys::SCE_KERNEL_MEMBLOCK_TYPE_USER_MAIN_PHYCONT_RW,
            DIRECT_LISTS_BYTES as u32,
            core::ptr::null_mut(),
        );
        if block < 0 {
            return None;
        }
        let mut base: *mut c_void = core::ptr::null_mut();
        sys::sceKernelGetMemBlockBase(block, &mut base);
        core::ptr::write_bytes(base as *mut u8, 0, DIRECT_LISTS_BYTES);
        let place = base as *mut DirectLists;
        DirectLists::init_zeroed(place);
        Some(&mut *place)
    }
}

/// The display's raw arena (arenas.rs `make_raw_arena`): GPU memory for
/// raw PSP vertices (`nativeres_core::raw_arena`), made with the vertex
/// arena when the raw vertex program exists. Null: none.
pub static RAW_ARENA: AtomicPtr<u8> = AtomicPtr::new(core::ptr::null_mut());
pub static RAW_ARENA_BYTES: AtomicU32 = AtomicU32::new(0);

/// The display's index ring (arenas.rs `make_raw_arena`): GPU memory
/// where the capture writes run draws' triangle lists (`Frame::ring_indices`).
/// Null: none (the display builds the lists itself).
pub static INDEX_RING: AtomicPtr<u8> = AtomicPtr::new(core::ptr::null_mut());
pub static INDEX_RING_BYTES: AtomicU32 = AtomicU32::new(0);

/// 3D runs as raw PSP vertices for the GPU
/// (`nativeres_core::raw_arena::ENABLED`), where the draw allows it.
pub fn set_raw_vertices(on: u32) {
    nativeres_core::raw_arena::ENABLED.store(on != 0, Ordering::Relaxed);
}

/// The display's sprite program exists: 3D rectangles may go raw
/// (`nativeres_core::raw_arena::SPRITES`).
pub fn set_raw_sprites(on: u32) {
    nativeres_core::raw_arena::SPRITES.store(on != 0, Ordering::Relaxed);
}

/// The framebuffers the display has render targets for (PSP addresses, 0 =
/// none; targets.rs, updated every drawn frame).
fn display_targets() -> [u32; 4] {
    let published = &crate::targets::DISPLAY_TARGETS;
    [0, 1, 2, 3].map(|k| published[k].load(Ordering::Relaxed))
}

/// Whether the plugin is asked to copy VRAM ranges
/// (textures, display lists in VRAM) for us (1, default). 0 is a measuring
/// aid: the plugin's copies run in the game's own (emulated) CPU time; without
/// them VRAM lists and textures can't be read, so frames come out broken.
static VRAM_REQUESTS_WANTED: AtomicBool = AtomicBool::new(true);

pub fn set_vram_requests(on: u32) {
    VRAM_REQUESTS_WANTED.store(on != 0, Ordering::Relaxed);
}

/// Bytes of the recent VRAM ranges the plugin copies whole at a list's
/// enqueue (`nativeres_core::psp_copy::WHOLE_AT_ENQUEUE_BYTES`; 0: only
/// the change check's samples, as before 2026-11-17).
static VRAM_WHOLE_BYTES: AtomicU32 = AtomicU32::new(nativeres_core::psp_copy::WHOLE_AT_ENQUEUE_BYTES);

pub fn set_vram_whole_bytes(bytes: u32) {
    VRAM_WHOLE_BYTES.store(bytes, Ordering::Relaxed);
}

/// Debug `copyverify`: bit 0: the plugin checks every VRAM copy right after
/// making it (`VramRequestTable::verify_copies`; log `copyverify=bad/checked
/// @last bad offset`); bit 1: the old cache maintenance; bit 2: copies in one
/// interruptible piece (as before). Off by default.
static VRAM_VERIFY: AtomicU32 = AtomicU32::new(0);

pub fn set_copy_verify(on: u32) {
    VRAM_VERIFY.store(on & 7, Ordering::Relaxed);
}

/// Menu "NativeRes Distant Textures" (Adrenaline config byte
/// MIP_MODE, pushed by the display every frame): see
/// `nativeres_core::texture::MIP_MODE`.
pub fn set_mip_mode(mode: u32) {
    nativeres_core::texture::MIP_MODE.store(mode.min(2) as u8, Ordering::Relaxed);
}

/// The display's alpha-test program exists: textured draws' alpha test is
/// left to its fragment shader (`nativeres_core::blend::ALPHA_TEST_IN_SHADER`).
pub fn set_alpha_test_ready(ready: u32) {
    nativeres_core::blend::ALPHA_TEST_IN_SHADER.store(ready != 0, Ordering::Relaxed);
}

/// Big raw runs drawn by the GPU straight from PSP RAM
/// (`nativeres_core::raw_arena::ZERO_COPY`). Switched on by the display once
/// it has mapped PSP RAM for the GPU (arenas.rs `map_psp_ram_for_gpu`), the
/// default since 2026-11-09.
pub fn set_zero_copy(on: u32) {
    nativeres_core::raw_arena::ZERO_COPY.store(on != 0, Ordering::Relaxed);
}

/// Whether the capture should leave 3D draws to the GPU transform now.
pub fn gpu_transform_wanted() -> bool {
    GPU_TRANSFORM_WANTED.load(Ordering::Relaxed) != 0 && GPU_TRANSFORM_READY.load(Ordering::Relaxed) != 0
}

/// The capture's block copies since the last frame (for the log line).
#[derive(Default)]
struct CopyStats {
    us: u32,
    count: u32,
    /// Of `count`: pieces read straight into their destination
    /// (`copy_words`: vertices, palettes), and their bytes; the rest are
    /// whole 1 KiB blocks (display lists, small reads).
    direct: u32,
    direct_bytes: u32,
}

/// The live capture (~1 MiB with its draw stream rings) and its block
/// cache's store (~130 KiB, more than the capture thread's stack), in a
/// block of phycont memory of their own made when the capture thread starts
/// (main memory is short: the capture was in the module's data, 0.66 MiB of
/// it). Dereferences to the `Capture`; only the capture thread uses it, and
/// only after `allocate`.
pub struct CaptureBox {
    capture: *mut Capture,
    store: *mut BlockStore<CAPTURE_BLOCKS>,
}

/// Where the store lies in the block: after the capture, on a cache line.
const STORE_OFFSET: usize = (core::mem::size_of::<Capture>() + 63) & !63;
/// The block, whole MiB (phycont blocks come in MiB), for the memory plan.
const CAPTURE_BLOCK_BYTES: usize = (STORE_OFFSET + core::mem::size_of::<BlockStore<CAPTURE_BLOCKS>>() + 0xF_FFFF) & !0xF_FFFF;
#[no_mangle]
#[used]
#[allow(non_upper_case_globals)]
pub static nativeres_plan_phycont_capture: u32 = CAPTURE_BLOCK_BYTES as u32;

impl CaptureBox {
    const NONE: CaptureBox = CaptureBox { capture: core::ptr::null_mut(), store: core::ptr::null_mut() };

    /// Makes the capture in phycont memory (kept until Adrenaline ends).
    /// False if there was no memory for it.
    fn allocate(&mut self) -> bool {
        // Safe: plain system calls; the block is zeroed and ours alone,
        // `init_zeroed` is made for zeroed memory, and a zeroed store is a
        // store (`BlockStore`).
        unsafe {
            let uid = sys::alloc_block(
                b"NativeResCaptureState\0".as_ptr(),
                sys::SCE_KERNEL_MEMBLOCK_TYPE_USER_MAIN_PHYCONT_RW,
                CAPTURE_BLOCK_BYTES as u32,
                core::ptr::null_mut(),
            );
            if uid < 0 {
                return false;
            }
            let mut base: *mut c_void = core::ptr::null_mut();
            sys::sceKernelGetMemBlockBase(uid, &mut base);
            core::ptr::write_bytes(base as *mut u8, 0, CAPTURE_BLOCK_BYTES);
            Capture::init_zeroed(base as *mut Capture);
            self.capture = base as *mut Capture;
            self.store = (base as *mut u8).add(STORE_OFFSET) as *mut BlockStore<CAPTURE_BLOCKS>;
        }
        true
    }

    /// The capture and its block cache's store, both at once.
    fn with_store(&mut self) -> (&mut Capture, &mut BlockStore<CAPTURE_BLOCKS>) {
        debug_assert!(!self.capture.is_null() && !self.store.is_null());
        // Safe: set by `allocate` before the capture thread uses them; two
        // separate parts of the block; the capture thread alone uses them.
        unsafe { (&mut *self.capture, &mut *self.store) }
    }

    /// The block cache's store.
    fn store(&mut self) -> &mut BlockStore<CAPTURE_BLOCKS> {
        self.with_store().1
    }
}

impl core::ops::Deref for CaptureBox {
    type Target = Capture;
    fn deref(&self) -> &Capture {
        debug_assert!(!self.capture.is_null());
        // Safe: set by `allocate` before the capture thread uses it.
        unsafe { &*self.capture }
    }
}

impl core::ops::DerefMut for CaptureBox {
    fn deref_mut(&mut self) -> &mut Capture {
        debug_assert!(!self.capture.is_null());
        // Safe: as `deref`; the capture thread alone uses it.
        unsafe { &mut *self.capture }
    }
}

/// The draw stream's limits in
/// use (`draw_stream::TRANSFORM_LIMIT` / `JOB_LIMIT`, at most the ring
/// sizes), for A/Bs against the old 64 / 512.
pub fn set_stream_limits(transforms: u32, jobs: u32) {
    draw_stream::TRANSFORM_LIMIT.store(transforms.clamp(2, draw_stream::MAX_TRANSFORMS as u32), Ordering::Relaxed);
    draw_stream::JOB_LIMIT.store(jobs.clamp(2, draw_stream::MAX_JOBS as u32), Ordering::Relaxed);
}

/// Everything the capture thread owns.
pub struct CaptureState {
    pub capture: CaptureBox,
    pub last_frame: FrameStats,
    pub dump: DumpRecorder,
    vram: VramRequests,
    reader: RingReader,
    copies: CopyStats,
}

static STATE: SingleThreaded<CaptureState> = SingleThreaded::new(CaptureState {
    capture: CaptureBox::NONE,
    last_frame: FrameStats::new(),
    dump: DumpRecorder::new(),
    vram: VramRequests::new(),
    reader: RingReader::new(),
    copies: CopyStats { us: 0, count: 0, direct: 0, direct_bytes: 0 },
});

/// PSP memory as the capture reads it (see `nativeres_core::block_cache`).
/// While a dump runs, every block copied also goes into the dump.
pub struct VitaBlocks<'a> {
    dump: &'a mut DumpRecorder,
    vram: &'a mut VramRequests,
    copies: &'a mut CopyStats,
    /// Ask `vram` whether a VRAM block is in the plugin's copy (off for
    /// reading a dumped frame's textures: those aren't the capture's).
    check_vram: bool,
}

impl BlockSource for VitaBlocks<'_> {
    fn copy_block(&mut self, address: u32, out: &mut [u32; BLOCK_WORDS]) -> bool {
        // VRAM is read from the plugin's copy, which may not hold this block
        // yet: it would read as zeros (NOP commands, for a list in VRAM).
        let in_vram = (VRAM_START..VRAM_START + VRAM_SIZE).contains(&address);
        if in_vram && self.check_vram && !self.vram.block_ready(address) {
            return false;
        }
        let source = platform::psp_memory(address, BLOCK_SIZE);
        if source.is_null() {
            return false;
        }
        let target = out.as_mut_ptr() as *mut u8;
        let timed = COPY_TIMING.load(Ordering::Relaxed) != 0;
        let start = if timed { sys::now_us() } else { 0 };
        // Safe: both hold BLOCK_SIZE bytes.
        unsafe { platform::psp_copy(target, source, BLOCK_SIZE) };
        if timed {
            self.copies.us += (sys::now_us() - start) as u32;
        }
        self.copies.count += 1;
        if self.dump.is_active() {
            // Safe: `out` holds BLOCK_SIZE bytes.
            unsafe { self.dump.record_block(address, target) };
        }
        true
    }

    /// Main RAM, and VRAM whose blocks the plugin's copy holds (noted as
    /// needed, as reading them now would: `VramRequests::block_ready`); not
    /// while a dump runs (the dump must record the blocks).
    fn defer(&mut self, address: u32, bytes: u32) -> bool {
        let Some(end) = address.checked_add(bytes) else {
            return false;
        };
        if self.dump.is_active() {
            return false;
        }
        if address >= RAM_START && end <= RAM_END {
            return true;
        }
        if !self.check_vram || address < VRAM_START || end > VRAM_START + VRAM_SIZE || bytes == 0 {
            return false;
        }
        let mut block = address & !(BLOCK_SIZE - 1);
        while block < end {
            if !self.vram.block_ready(block) {
                return false;
            }
            block += BLOCK_SIZE;
        }
        true
    }

    /// The palette store's samples: single loads straight from PSP RAM or
    /// the VRAM copy (as `sample_psp_ram`), where the VRAM blocks are there
    /// (noted as needed, as for `defer`); not while a dump runs (the dump
    /// must record the blocks).
    fn read_samples(&mut self, address: u32, positions: &[usize], out: &mut [u32]) -> bool {
        if self.dump.is_active() {
            return false;
        }
        let last = positions.iter().copied().max().unwrap_or(0) as u32;
        let end = address + 4 * (last + 1);
        if self.check_vram && (VRAM_START..VRAM_START + VRAM_SIZE).contains(&address) {
            let mut block = address & !(BLOCK_SIZE - 1);
            while block < end {
                if !self.vram.block_ready(block) {
                    return false;
                }
                block += BLOCK_SIZE;
            }
        }
        sample_psp_ram(address, positions, out)
    }

    /// Straight from PSP RAM or the VRAM copy, not while a dump runs (the
    /// dump records whole blocks).
    fn copy_words(&mut self, address: u32, out: &mut [u32]) -> bool {
        if self.dump.is_active() {
            return false;
        }
        let in_vram = (VRAM_START..VRAM_START + VRAM_SIZE).contains(&address);
        if in_vram && self.check_vram && !self.vram.block_ready(address & !(BLOCK_SIZE - 1)) {
            return false;
        }
        let bytes = 4 * out.len() as u32;
        let source = platform::psp_memory(address, bytes);
        if source.is_null() {
            return false;
        }
        let timed = COPY_TIMING.load(Ordering::Relaxed) != 0;
        let start = if timed { sys::now_us() } else { 0 };
        // Safe: `out` holds `bytes` bytes, and so does `source` (one block).
        unsafe { platform::psp_copy(out.as_mut_ptr() as *mut u8, source, bytes) };
        if timed {
            self.copies.us += (sys::now_us() - start) as u32;
        }
        self.copies.count += 1;
        self.copies.direct += 1;
        self.copies.direct_bytes += bytes;
        true
    }
}

impl CaptureState {
    fn memory(&mut self, check_vram: bool) -> BlockCache<'_, VitaBlocks<'_>, CAPTURE_BLOCKS> {
        let source = VitaBlocks { dump: &mut self.dump, vram: &mut self.vram, copies: &mut self.copies, check_vram };
        BlockCache::new(self.capture.store(), source)
    }

    /// Handles one event; true if it completed a frame (then `last_frame`
    /// holds its statistics). Draws are recorded into `frame` if given.
    fn handle(&mut self, event: &RawEvent, frame: Option<&mut Frame>) -> bool {
        let Some(parsed) = Event::from_ring(event.event_type, event.args) else {
            return false;
        };
        if self.dump.is_active() {
            self.dump.record_event(event.event_type, &event.args);
        }
        let CaptureState { capture, last_frame, dump, vram, copies, .. } = self;
        let (capture, store) = capture.with_store();
        let mut memory = BlockCache::new(store, VitaBlocks { dump, vram, copies, check_vram: true });
        memory.whole_blocks = COPY_LINES.load(Ordering::Relaxed) == 0;
        let result = match capture.handle(parsed, &mut memory, frame) {
            Some(stats) => {
                *last_frame = stats;
                true
            }
            None => false,
        };
        if READ_WATCH.load(Ordering::Relaxed) {
            let (changed, checked, first) = memory.changed_lines();
            READ_CHANGED.fetch_add(changed, Ordering::Relaxed);
            READ_CHECKED.fetch_add(checked, Ordering::Relaxed);
            if changed > 0 && READ_FIRST.load(Ordering::Relaxed) == 0 {
                READ_FIRST.store(first, Ordering::Relaxed);
            }
        }
        result
    }

    /// After each frame: updates the plugin's VRAM copy requests for the
    /// recorded `frame` (None when not recording: nothing is requested).
    fn request_vram(&mut self, frame: Option<&Frame>) {
        let table = vram_table();
        // Safe: the table is mapped PSP memory the plugin also uses; all
        // accesses are volatile.
        let copied = unsafe { read_volatile(addr_of!((*table).copied_generation)) };
        let frame = frame.filter(|_| VRAM_REQUESTS_WANTED.load(Ordering::Relaxed));
        let plan = match frame {
            Some(frame) => {
                // Textures the display draws from its own targets need no copy.
                let targets = display_targets();
                let shown = frame.shown_framebuffer;
                let mut textures = frame
                    .textures()
                    .iter()
                    .filter(|texture| nativeres_core::vram_requests::texture_needs_copy(texture, &targets, shown))
                    .filter_map(|texture| {
                    // Textures in a VRAM mirror need the VRAM itself copied.
                    let address = nativeres_core::memory::fold_vram_mirror(texture.address & 0x0FFF_FFFF);
                    let offset = address.checked_sub(VRAM_START).filter(|&offset| offset < VRAM_SIZE)?;
                    let size = texture::data_size(texture).unwrap_or(0);
                    Some(VramRange { offset, size })
                });
                self.vram.plan(Some(&mut textures), copied)
            }
            None => self.vram.plan(None, copied),
        };
        self.send_vram_plan(&plan);
    }

    /// After a frame walked without recording while the renderer is on:
    /// no texture ranges added (the requested ones age out as usual), the
    /// blocks the walk read stay requested.
    fn request_vram_keeping_ranges(&mut self) {
        // Safe: as in request_vram.
        let copied = unsafe { read_volatile(addr_of!((*vram_table()).copied_generation)) };
        let plan = if VRAM_REQUESTS_WANTED.load(Ordering::Relaxed) {
            self.vram.plan(Some(&mut core::iter::empty()), copied)
        } else {
            self.vram.plan(None, copied)
        };
        self.send_vram_plan(&plan);
    }

    /// After an event whose walk stopped at a VRAM block
    /// (`Capture::waiting_for_vram`): requests the block's run right away
    /// and waits for the plugin's copy (the game waits in its GE sync for
    /// us meanwhile, where the plugin wakes its copy thread), then resumes
    /// the walk; gives up when nothing new can be requested or after
    /// `VRAM_WAIT_LIMIT_US` for the whole event, or as soon as the plugin
    /// stopped holding the game (its sync wait timed out: the game may be
    /// reusing the list memory, so the walk must not go on; resuming then
    /// read half-rewritten lists as commands and MHF's intro fly-over stayed
    /// black, 2026-11-18). `missing_before`: the frame's unready reads
    /// before the event.
    fn settle_vram_waits(&mut self, ring: &SharedRing, recording: bool, missing_before: u32) {
        let mut missing = missing_before;
        let first_stop = sys::now_us();
        let timeouts = ring.header().sync_timeouts;
        while self.capture.waiting_for_vram().is_some() {
            let at_stop = self.vram.missing();
            let started = sys::now_us();
            let copied = self.wait_for_new_vram(ring, first_stop, timeouts);
            let waited = (sys::now_us() - started) as u32;
            // Safe: the capture thread, recording (frames allocated); used
            // before the frame's publish.
            let frame = if recording { Some(unsafe { frames::building() }) } else { None };
            let CaptureState { capture, dump, vram, copies, .. } = self;
            let (capture, store) = capture.with_store();
            let mut memory = BlockCache::new(store, VitaBlocks { dump, vram, copies, check_vram: true });
            memory.whole_blocks = COPY_LINES.load(Ordering::Relaxed) == 0;
            if copied {
                capture.resume_after_vram(&mut memory, frame, waited);
                // The stop's failed read is made good; later ones count.
                missing += self.vram.missing() - at_stop;
                self.vram.set_missing(missing);
            } else {
                capture.give_up_vram(&mut memory, frame, waited);
                missing = self.vram.missing();
            }
        }
    }

    /// Publishes the table with the blocks the walk just missed and waits
    /// until the plugin has copied it. False if the table didn't change
    /// (the block was requested already and its copy is out) or the copy
    /// didn't come in time. A block distrusted after a pause changes it: it
    /// gets a range of its own, which the plugin copies in full
    /// (`VramRequests::block_ready`, `DISTRUST_FRAMES`).
    fn wait_for_new_vram(&mut self, ring: &SharedRing, first_stop: u64, timeouts: u32) -> bool {
        let table = vram_table();
        // Safe: as in request_vram.
        let copied = unsafe { read_volatile(addr_of!((*table).copied_generation)) };
        let plan = self.vram.plan_mid_frame(copied);
        // Safe: the table is mapped PSP memory the plugin also uses.
        let Some(generation) = (unsafe { publish_vram_plan(table, &plan) }) else {
            return false;
        };
        self.vram.published(generation);
        loop {
            // Safe: as above.
            let copied = unsafe { read_volatile(addr_of!((*table).copied_generation)) };
            let game_moved_on = ring.header().sync_timeouts != timeouts;
            if game_moved_on || sys::now_us() - first_stop >= VRAM_WAIT_LIMIT_US {
                return false;
            }
            if self.vram.note_copied(copied) {
                return true;
            }
            // Safe: a plain kernel call.
            unsafe { sys::sceKernelDelayThread(50) };
        }
    }

    fn send_vram_plan(&mut self, plan: &Plan) {
        let table = vram_table();
        // Safe: the table is mapped PSP memory the plugin also uses.
        if let Some(generation) = unsafe { publish_vram_plan(table, plan) } {
            self.vram.published(generation);
        }
    }
}

/// Writes `plan` to the shared table if it differs; returns the new
/// generation then. The list is hidden from the plugin (count 0) while it
/// is rewritten.
///
/// # Safety
/// `table` must point to the mapped request table.
unsafe fn publish_vram_plan(table: *mut VramRequestTable, plan: &Plan) -> Option<u32> {
    let whole = VRAM_WHOLE_BYTES.load(Ordering::Relaxed);
    if read_volatile(addr_of!((*table).whole_bytes)) != whole {
        write_volatile(addr_of_mut!((*table).whole_bytes), whole);
        crate::adrenaline::writeback(addr_of_mut!((*table).whole_bytes) as *mut c_void, 4);
    }
    let verify = VRAM_VERIFY.load(Ordering::Relaxed);
    if read_volatile(addr_of!((*table).verify_copies)) != verify {
        write_volatile(addr_of_mut!((*table).verify_copies), verify);
        crate::adrenaline::writeback(addr_of_mut!((*table).verify_copies) as *mut c_void, 4);
    }
    if read_volatile(addr_of!((*table).recent_mask)) != plan.recent_mask {
        write_volatile(addr_of_mut!((*table).recent_mask), plan.recent_mask);
        crate::adrenaline::writeback(addr_of_mut!((*table).recent_mask) as *mut c_void, 4);
    }
    let ranges = addr_of_mut!((*table).ranges) as *mut VramRange;
    let mut changed = read_volatile(addr_of!((*table).range_count)) != plan.count as u32;
    for (i, range) in plan.ranges().iter().enumerate() {
        changed |= read_volatile(ranges.add(i)) != *range;
    }
    if !changed {
        return None;
    }
    write_volatile(addr_of_mut!((*table).range_count), 0);
    for (i, range) in plan.ranges().iter().enumerate() {
        write_volatile(ranges.add(i), *range);
    }
    let generation = read_volatile(addr_of!((*table).generation)).wrapping_add(1);
    write_volatile(addr_of_mut!((*table).generation), generation);
    write_volatile(addr_of_mut!((*table).range_count), plan.count as u32);
    crate::adrenaline::writeback(table as *mut c_void, core::mem::size_of::<VramRequestTable>() as u32);
    Some(generation)
}

/// The ring in PSP memory, read with volatile loads (the plugin writes it
/// at any time).
struct SharedRing(*const Ring);

impl SharedRing {
    /// GE sync (core psp_link): the events read, walked and copied.
    fn set_handled(&self, count: u32) {
        // Safe: the ring is mapped PSP memory we may write (as the VRAM
        // request table next to it); the plugin only reads this field.
        unsafe { write_volatile(addr_of!((*self.0).header.vita_handled) as *mut u32, count) };
    }

    /// Asks the plugin for the GE's own state (`EVENT_GE_STATE`): a request
    /// number different from the last one it answered.
    fn request_ge_state(&self) {
        // Safe: as `set_handled`; the plugin only reads the request.
        unsafe {
            let answered = read_volatile(addr_of!((*self.0).header.state_answered));
            write_volatile(addr_of!((*self.0).header.vita_state_request) as *mut u32, answered.wrapping_add(1));
        }
    }

    /// GE sync: whether the plugin should wait for us.
    fn set_sync_wanted(&self, wanted: bool) {
        let value = if wanted { event_ring::SYNC_WANTED } else { 0 };
        // Safe: as `set_handled`.
        unsafe { write_volatile(addr_of!((*self.0).header.vita_sync_wanted) as *mut u32, value) };
    }

    /// Asks the plugin for GE_SYNC events (the log's `syncs=`).
    fn request_sync_events(&self) {
        // Safe: as `set_handled`.
        unsafe { write_volatile(addr_of!((*self.0).header.vita_sync_events) as *mut u32, event_ring::SYNC_WANTED) };
    }
}

/// Whether the plugin waits in the game's GE sync until
/// the capture has read the lists the game waited for (on by default, user's
/// choice 2026-11-08: MTX refilled its HUD buffer for a later frame before
/// our walk reached the HUD). Costs the game time whenever the capture is
/// behind; at most event_ring::SYNC_TIMEOUT_US per sync.
pub static GE_SYNC_WANTED: AtomicU32 = AtomicU32::new(1);

/// How long the capture thread waits for the copier before telling the
/// plugin an event is handled (the copier is microseconds behind).
const COPIES_WAIT_LIMIT_US: u64 = 5_000;

impl RingSource for SharedRing {
    fn header(&self) -> RingHeader {
        // Safe: the ring is mapped PSP memory.
        unsafe { read_volatile(addr_of!((*self.0).header)) }
    }
    fn event(&self, index: u32) -> RawEvent {
        // Safe: index < CAPACITY (RingReader takes it modulo).
        unsafe { read_volatile((addr_of!((*self.0).events) as *const RawEvent).add(index as usize)) }
    }
}

/// Per-frame numbers measured here rather than in the core.
#[derive(Default)]
struct FrameTiming {
    frame_number: u32,
    /// PSP clock at the previous flip.
    last_flip_psp_time: u32,
    /// Vita time spent handling this frame's events.
    vita_time_us: u64,
    /// Thread CPU time at the previous frame.
    last_run_us: u64,
    /// PSP time from this frame's flip to the newest event in the ring when
    /// the capture finished the frame: how far the game had moved on (its
    /// memory for the next frames being written) while we walked.
    behind_us: u32,
    /// The game's GE syncs in this frame (the plugin's GE_SYNC events).
    syncs: GeSyncs,
}

/// What the game's GE syncs in one frame cost it (log `syncs=`), from the
/// plugin's GE_SYNC events and the PSP times of the frame's events.
#[derive(Default)]
struct GeSyncs {
    count: u32,
    /// Waiting for the GE (Sony's sync), then for us (the GE sync handshake).
    ge_us: u32,
    waited_us: u32,
    /// PSP time of the frame's first list enqueue (`enqueued`).
    first_enqueue: u32,
    enqueued: bool,
    /// The last sync: from the first enqueue to its start, and its end.
    enqueue_to_sync_us: u32,
    last_end: u32,
    /// Vita wall time handling the frame's list events (walking, then
    /// waiting for the copier before telling the plugin), and that wait.
    walk_us: u32,
    copies_wait_us: u32,
    /// Of `walk_us`: waiting for the recorder (draw stream rings full).
    stream_wait_us: u32,
}

impl GeSyncs {
    /// Notes an event read from the ring (before it is handled).
    fn note(&mut self, event: &RawEvent) {
        match event.event_type {
            1 | 2 if !self.enqueued => {
                self.first_enqueue = event.psp_time_us;
                self.enqueued = true;
            }
            event_ring::EVENT_GE_SYNC => {
                let (ge_us, waited_us) = (event.args[2], event.args[3]);
                self.count += 1;
                self.ge_us = self.ge_us.wrapping_add(ge_us);
                self.waited_us = self.waited_us.wrapping_add(waited_us);
                let start = event.psp_time_us.wrapping_sub(ge_us).wrapping_sub(waited_us);
                self.enqueue_to_sync_us = if self.enqueued { start.wrapping_sub(self.first_enqueue).min(10_000_000) } else { 0 };
                self.last_end = event.psp_time_us;
            }
            _ => {}
        }
    }
}

/// Recorded frames held back after a new session while the GE's own state
/// hasn't come (`Capture::waiting_for_ge_state`): the plugin answers at the
/// game's next GE sync, normally within the first frame. A plugin that
/// never answers costs this many frames, then the kept registers are used.
const MAX_GE_STATE_FRAMES: u32 = 30;

/// Recorded frames held back while the GE's state asked for after a frame
/// whose walk stopped early (`FrameStats::walk_stopped_early`) hasn't come:
/// the plugin answers at the next GE sync, within the next frame.
const MAX_REPAIR_FRAMES: u32 = 3;

/// How long the capture waits, in all, for the plugin to copy the VRAM
/// blocks one event's walk stopped at (`CaptureState::settle_vram_waits`).
/// The plugin holds the game in its GE sync for at most
/// `event_ring::SYNC_TIMEOUT_US` (50 ms), counted from the sync, which comes
/// after the stop: 30 ms leaves the resumed walk time before the game may
/// reuse its list memory.
const VRAM_WAIT_LIMIT_US: u64 = 30_000;

/// Debug `vramwait 0|1`: walks wait for new VRAM blocks to be copied (1,
/// default) or stop there as before (0).
pub static VRAM_WAIT_WANTED: AtomicBool = AtomicBool::new(true);

/// Debug `gestate`: asks the plugin for the GE's state in a running game
/// (no new session). It replaces ours; the log line's `gestate_regdiff=` /
/// `gestate_mtxdiff=` say how far our walk was from it (0 / 0 = exact).
pub static GE_STATE_CHECK: AtomicBool = AtomicBool::new(false);

/// Recorded frames dropped because the plugin gave up waiting in the
/// game's GE sync while they were walked (core `frame_gate`; log `torn=`).
static TORN_FRAMES: AtomicU32 = AtomicU32::new(0);

/// One line per frame, e.g. `frame=12 dt=16.7ms cpu=... lists=1/1 ...`.
/// `sync`: the ring header, for the plugin's GE sync counts (`gesync=`).
fn log_frame(state: &mut CaptureState, timing: &mut FrameTiming, flip: &RawEvent, recorded: bool, sync: &RingHeader) {
    // Since the last line: the capture thread waiting for the recorder
    // thread (room in the draw stream, end of the frame), and the recorder
    // thread recording. Which half holds the other up, live.
    let (stream_waited_us, stream_busy_us) = recorder_thread::take_times();
    // The same for the walk stream: the capture thread (stage A) waiting
    // for stage B's thread, and stage B working (`stage_thread`).
    let (stage_waited_us, stage_busy_us, stage_cpu_us) = stage_thread::take_times();
    let mut line = [0u8; 1024];
    let mut text = TextBuffer::new(&mut line[..1023]);
    let frame_time = flip.psp_time_us.wrapping_sub(timing.last_flip_psp_time);
    let run_us = crate::threads::run_us();
    let table = vram_table();
    // Safe: volatile reads of the mapped table.
    let (count, last_copy_us, method, sync_checks, last_sync_us, changes, runs, error, mut bytes) = unsafe {
        (
            read_volatile(addr_of!((*table).range_count)),
            read_volatile(addr_of!((*table).last_copy_us)),
            read_volatile(addr_of!((*table).copy_method)),
            read_volatile(addr_of!((*table).sync_checks)),
            read_volatile(addr_of!((*table).last_sync_check_us)),
            read_volatile(addr_of!((*table).changes)),
            read_volatile(addr_of!((*table).copy_thread_runs)),
            read_volatile(addr_of!((*table).copy_thread_error)),
            0u32,
        )
    };
    // Safe: as above. The plugin's change checks at list enqueues.
    let (enqueue_checks, last_enqueue_us) =
        unsafe { (read_volatile(addr_of!((*table).enqueue_checks)), read_volatile(addr_of!((*table).last_enqueue_check_us))) };
    for i in 0..(count as usize).min(MAX_RANGES) {
        // Safe: as above.
        bytes += unsafe { read_volatile((addr_of!((*table).ranges) as *const VramRange).add(i)).size };
    }
    let method = copy_method::name(method);
    let _ = write!(
        text,
        "frame={} rec={} dt={}.{}ms cpu={}us run={}us copy={}us/{}x/{}direct/{}k stream={}waited/{}busy stage={}waited/{}busy/{}cpu waits={}/{}/{}/{}/{} badread={}@{:08x}+{:x} vram={}x/{}k/{}us/{} sync={}/{}us enq={}/{}us changes={} thread={}/{:08x} vramlist={}k/{}missing@{:08x} free={}k ours={}k wakes={} copier={}us/{}/{}/{} helped={}/{}/{}/{}/{} gesync={}/{}ms/{} behind={}.{}ms ",
        timing.frame_number,
        recorded as u8,
        frame_time / 1000,
        (frame_time % 1000) / 100,
        timing.vita_time_us,
        run_us.wrapping_sub(timing.last_run_us),
        state.copies.us,
        state.copies.count,
        state.copies.direct,
        state.copies.direct_bytes / 1024,
        stream_waited_us,
        stream_busy_us,
        stage_waited_us,
        stage_busy_us,
        stage_cpu_us,
        // Producer waits since the session started, by cause: rings full
        // (jobs/calls/words/transforms), then waiting for an idle consumer.
        draw_stream::WAITS[0].load(Ordering::Relaxed),
        draw_stream::WAITS[1].load(Ordering::Relaxed),
        draw_stream::WAITS[2].load(Ordering::Relaxed),
        draw_stream::WAITS[3].load(Ordering::Relaxed),
        draw_stream::WAITS[4].load(Ordering::Relaxed),
        platform::REFUSED_READS.load(Ordering::Relaxed),
        platform::LAST_REFUSED_ADDRESS.load(Ordering::Relaxed),
        platform::LAST_REFUSED_SIZE.load(Ordering::Relaxed),
        count,
        bytes / 1024,
        last_copy_us,
        method,
        sync_checks,
        last_sync_us,
        enqueue_checks,
        last_enqueue_us,
        changes,
        runs,
        error,
        state.vram.capture_blocks(),
        state.vram.missing(),
        state.vram.first_missing(),
        crate::blocks::free_user_memory().max(0) / 1024,
        crate::blocks::our_memory() / 1024,
        crate::recorder_thread::WAKES.load(Ordering::Relaxed),
        crate::copy_thread::take_busy_us(),
        crate::copy_thread::WAKES.load(Ordering::Relaxed),
        nativeres_core::draw_stream::PRODUCER_COPIES.load(Ordering::Relaxed),
        nativeres_core::draw_stream::UNSTABLE_READS.load(Ordering::Relaxed),
        // Copy tasks the capture thread copied itself, by cause
        // (`copy_tasks::helped`: VRAM lag, RAM lag, GE sync, frame end, ring full).
        nativeres_core::copy_tasks::PRODUCER_TASKS[0].load(Ordering::Relaxed),
        nativeres_core::copy_tasks::PRODUCER_TASKS[1].load(Ordering::Relaxed),
        nativeres_core::copy_tasks::PRODUCER_TASKS[2].load(Ordering::Relaxed),
        nativeres_core::copy_tasks::PRODUCER_TASKS[3].load(Ordering::Relaxed),
        nativeres_core::copy_tasks::PRODUCER_TASKS[4].load(Ordering::Relaxed),
        sync.sync_waits,
        sync.sync_wait_us / 1000,
        sync.sync_timeouts,
        timing.behind_us / 1000,
        (timing.behind_us % 1000) / 100,
    );
    // The game's GE syncs this frame: count / ms waiting for the GE / ms
    // waiting for us / ms from the first enqueue to the last sync's start /
    // ms from its end to the flip.
    let syncs = &timing.syncs;
    let sync_to_flip = if syncs.count > 0 { flip.psp_time_us.wrapping_sub(syncs.last_end).min(10_000_000) } else { 0 };
    let tenths = |us: u32| (us / 1000, (us % 1000) / 100);
    let (ge, waited, enqueue_to_sync, to_flip) =
        (tenths(syncs.ge_us), tenths(syncs.waited_us), tenths(syncs.enqueue_to_sync_us), tenths(sync_to_flip));
    let (walk, copies_wait, stream_wait) = (tenths(syncs.walk_us), tenths(syncs.copies_wait_us), tenths(syncs.stream_wait_us));
    let _ = write!(
        text,
        "syncs={}/{}.{}/{}.{}/{}.{}/{}.{}ms walk={}.{}/{}.{}/{}.{}ms ",
        syncs.count, ge.0, ge.1, waited.0, waited.1, enqueue_to_sync.0, enqueue_to_sync.1, to_flip.0, to_flip.1,
        walk.0, walk.1, copies_wait.0, copies_wait.1, stream_wait.0, stream_wait.1
    );
    let _ = report::write_frame(&mut text, &state.last_frame);
    // Copied ranges the game rewrote before the frame's
    // end (`nativeres_core::copy_watch`): changed/checked, the first one
    // (address+words, ms from its copy), the youngest changed one's age.
    if RAW_HASH.load(Ordering::Relaxed) {
        let _ = write!(text, " rawhash={}:{:016x} ", RECORDED_HASH_FRAME.load(Ordering::Relaxed), RECORDED_HASH.load(Ordering::Relaxed));
        let parts = RECORDED_PARTS.each_ref().map(|part| part.load(Ordering::Relaxed));
        let _ = write!(text, "parts={:08x}/{:08x}/{:08x}/{:08x} drawn=", parts[0], parts[1], parts[2], parts[3]);
        let written = DRAWN_HASHES_WRITTEN.load(Ordering::Acquire);
        let logged = DRAWN_HASHES_LOGGED.swap(written, Ordering::Relaxed);
        for n in written.saturating_sub(DRAWN_HASHES as u32).max(logged)..written {
            let slot = n as usize % DRAWN_HASHES;
            let _ = write!(text, "{}:{:016x},", DRAWN_HASH_FRAMES[slot].load(Ordering::Relaxed), DRAWN_HASH_VALUES[slot].load(Ordering::Relaxed));
        }
    }
    if VRAM_VERIFY.load(Ordering::Relaxed) & 1 != 0 {
        let table = vram_table();
        // Safe: the table is mapped PSP memory.
        let (checked, bad, last) = unsafe {
            (
                read_volatile(addr_of!((*table).verify_checked)),
                read_volatile(addr_of!((*table).verify_bad)),
                read_volatile(addr_of!((*table).verify_last_bad)),
            )
        };
        let _ = write!(text, " copyverify={bad}/{checked}@{last:06x}");
    }
    if SKIN_WATCH.load(Ordering::Relaxed) {
        let edge = f32::from_bits(SKIN_EDGE.load(Ordering::Relaxed));
        let _ = write!(text, " skinwatch={}@{}/{}", edge as u32, SKIN_DRAW.load(Ordering::Relaxed) as i32, SKIN_TRIANGLES.load(Ordering::Relaxed));
    }
    if READ_WATCH.load(Ordering::Relaxed) {
        let _ = write!(
            text,
            " readwatch={}/{}@{:08x}",
            READ_CHANGED.swap(0, Ordering::Relaxed),
            READ_CHECKED.swap(0, Ordering::Relaxed),
            READ_FIRST.swap(0, Ordering::Relaxed)
        );
    }
    if nativeres_core::copy_watch::ON.load(Ordering::Relaxed) {
        let r = nativeres_core::copy_watch::check(sample_psp_ram);
        let _ = write!(
            text,
            " rewritten={}/{}@{:08x}+{}w/{}.{}ms/{}.{}ms",
            r.changed,
            r.checked,
            r.first_address,
            r.first_words,
            r.first_age_us / 1000,
            (r.first_age_us % 1000) / 100,
            r.youngest_age_us / 1000,
            (r.youngest_age_us % 1000) / 100
        );
    }
    let _ = write!(text, " torn={}", TORN_FRAMES.load(Ordering::Relaxed));
    let length = text.len();
    line[length] = b'\n';
    log::append(&line[..=length]);

    timing.frame_number += 1;
    timing.last_flip_psp_time = flip.psp_time_us;
    timing.vita_time_us = 0;
    timing.last_run_us = run_us;
    timing.syncs = GeSyncs::default();
    state.copies = CopyStats::default();
    // Write to the file about once a second, so little is lost if the app is closed.
    if timing.frame_number % 60 == 0 {
        log::flush();
    }
}

fn renderer_enabled() -> bool {
    crate::options::renderer_enabled() != 0
}

unsafe extern "C" fn capture_thread(_args: u32, _argp: *mut c_void) -> i32 {
    // Safe: this thread is the only user of STATE (and of the building frame).
    let state = STATE.get();
    if !state.capture.allocate() {
        log::append(b"capture: no phycont memory for it, not capturing\n");
        log::flush();
        crate::user_log::line(format_args!("ERROR: not enough memory for the capture: NativeRes is off"));
        crate::user_log::save();
        return 0;
    }
    let have_frames = frames::allocate();
    log::append(if have_frames { b"capture: frames ready\n" } else { b"capture: no memory for frames, not recording\n" });
    log::append(b"capture: waiting for the PSP plugin's event ring\n");
    log::flush();
    if !have_frames {
        crate::user_log::line(format_args!("ERROR: not enough memory for frames: NativeRes draws nothing"));
    }
    crate::user_log::line(format_args!("OK: renderer ready, waiting for the PSP side (nativeres.prx) to start with a game"));
    crate::user_log::save();
    nativeres_core::copy_watch::set_clock(Some(clock_us));
    // Whether the draws' vertices are recorded on another core
    // (recorder_thread.rs; can be switched at run time).
    let mut threaded = false;
    // Whether the draw stream has a reader (the recorder or the copier
    // thread copies deferrable memory), and whether the copier thread copies.
    let mut deferred_copy = false;
    let mut copying = false;
    let ring = SharedRing(sys::psp_to_vita(event_ring::RING_ADDRESS) as *const Ring);
    let mut session = 0;
    let mut capturing = false;
    // Recording into the building frame since the last frame start.
    let mut recording = false;
    // Renderer on, but this frame is walked without recording (behind).
    let mut skipping = false;
    // Whether each recorded frame is shown (core frame_gate).
    let mut gate = nativeres_core::frame_gate::FrameGate::new();
    // The plugin's given-up GE sync waits when the frame being recorded
    // started (set where `recording` is).
    let mut timeouts_at_start = 0u32;
    // Frames completed since the new session while the GE's state hadn't
    // come (see MAX_GE_STATE_FRAMES).
    let mut ge_state_frames = 0u32;
    // How many such frames are held for the request waiting (session start
    // or repair).
    let mut ge_state_limit = MAX_GE_STATE_FRAMES;
    let mut timing = FrameTiming::default();
    let mut vertex_cache_tried = false;
    let mut ring_version_reported = false;

    // Whether stage B of the capture runs on its own thread
    // (stage_thread.rs; switched at frame ends).
    let mut staged = false;

    loop {
        if recorder_thread::wanted() != threaded {
            threaded = recorder_thread::wanted() && recorder_thread::start();
            if threaded {
                state.capture.set_stream_hooks(Some(&recorder_thread::HOOKS));
                recorder_thread::serve(state.capture.stream());
            } else {
                state.capture.set_stream_hooks(None);
                recorder_thread::serve(core::ptr::null());
            }
        }
        // Who copies the draws' vertex memory: the copier thread right after
        // the walk queues each draw (copy_thread.rs, the
        // default), this thread at walk time ("copythread 0"), or the
        // recorder thread when it records the draw ("copythread 0" and
        // "deferredcopy 1"). The copier and deferred copies need the recorder
        // thread and a reader; the copier is switched off before, and on
        // after, the reader.
        let copier = threaded && copy_thread::wanted() && copy_thread::start();
        if copying && !copier {
            // Waits until the copier has copied (and the recorder recorded)
            // every queued job, then lets the thread go.
            state.capture.set_stream_copier(None);
            copy_thread::serve(core::ptr::null());
            copying = false;
        }
        let reader = copier || (threaded && DEFERRED_COPY_WANTED.load(Ordering::Relaxed) != 0);
        if reader != deferred_copy {
            deferred_copy = reader;
            state.capture.set_stream_reader(if reader { Some(PSP_RAM_READER) } else { None });
        }
        if copier && !copying {
            state.capture.set_stream_copier(Some(&copy_thread::HOOKS));
            copy_thread::serve(state.capture.stream());
            copying = true;
        }

        // Switched any time: each draw is recorded whole one way or the other.
        let gpu_transform = gpu_transform_wanted();
        if gpu_transform != state.capture.gpu_transform() {
            state.capture.set_gpu_transform(gpu_transform);
        }
        // The vertex cache, once the display has made its arena (tried once).
        if gpu_transform && !vertex_cache_tried && !VERTEX_ARENA.load(Ordering::Acquire).is_null() {
            vertex_cache_tried = true;
            let cache = make_vertex_cache();
            let message: &[u8] = if cache.is_some() { b"vertex cache ready\n" } else { b"vertex cache: no memory for its tables\n" };
            log::append(message);
            state.capture.set_vertex_cache(cache);
            let lists = make_direct_lists();
            let message: &[u8] = if lists.is_some() { b"direct draws ready\n" } else { b"direct draws: no memory for their lists\n" };
            log::append(message);
            state.capture.set_direct_lists(lists);
        }
        let header = ring.header();
        if header.magic != event_ring::MAGIC || header.version != event_ring::VERSION {
            if header.magic == event_ring::MAGIC && !ring_version_reported {
                // An older or newer PSP plugin: never captured from.
                ring_version_reported = true;
                let mut line = [0u8; 96];
                let mut text = TextBuffer::new(&mut line);
                let _ = writeln!(text, "capture: PSP plugin's event ring is version {}, expected {}", header.version, event_ring::VERSION);
                let length = text.len();
                log::append(&line[..length]);
                log::flush();
                crate::user_log::line(format_args!(
                    "ERROR: the PSP side (nativeres.prx) is another version ({}, expected {}): an old copy in seplugins, or a savestate from an older version",
                    header.version,
                    event_ring::VERSION
                ));
            }
            if capturing {
                log::flush();
                capturing = false;
            }
            sys::sceKernelDelayThread(IDLE_INTERVAL_US);
            continue;
        }

        // A new session means the game (and the plugin) restarted.
        if !capturing || header.session != session {
            session = header.session;
            let title = crate::adrenaline::title_id();
            let length = title.iter().position(|&byte| byte == 0).unwrap_or(title.len());
            log::open(&title[..length], session);
            // The user log's last line: nothing is logged after the game's
            // start (saved by the settings thread, no file write here).
            // Adrenaline's title ID is often still empty here.
            if length == 0 {
                crate::user_log::line(format_args!("OK: PSP side running, a game started"));
            } else {
                crate::user_log::line(format_args!(
                    "OK: PSP side running, game {}",
                    core::str::from_utf8(&title[..length]).unwrap_or("?")
                ));
            }
            crate::user_log::finish();
            state.reader.restart();
            // Keeps the GE registers: a savestate load is a new session too
            // (Capture::start_session); the GE's own come when the plugin
            // answers.
            state.capture.start_session();
            ring.request_ge_state();
            ge_state_frames = 0;
            ge_state_limit = MAX_GE_STATE_FRAMES;
            timing = FrameTiming::default();
            if have_frames {
                frames::building().clear();
            }
            recording = false;
            skipping = false;
            capturing = true;
        }

        // Whether the plugin should wait for us in the game's GE sync
        // (`GE_SYNC_WANTED`): only while we record frames from it.
        let sync_wanted = GE_SYNC_WANTED.load(Ordering::Relaxed) != 0 && have_frames && renderer_enabled();
        ring.set_sync_wanted(sync_wanted);
        ring.request_sync_events();
        // Walks wait for new VRAM blocks only where the plugin copies them
        // while the game waits for us (not a savestate's older plugin).
        state.capture.vram_waits = sync_wanted
            && ring.header().plugin_features == event_ring::COPIES_AT_SYNC
            && VRAM_REQUESTS_WANTED.load(Ordering::Relaxed)
            && VRAM_WAIT_WANTED.load(Ordering::Relaxed);
        if GE_STATE_CHECK.swap(false, Ordering::Relaxed) {
            ring.request_ge_state();
        }
        let lost = state.reader.catch_up(header.write_count);
        if lost > 0 {
            state.capture.note_dropped_events(lost);
        }
        let mut start_time = sys::now_us();
        loop {
            let event = match state.reader.next(&ring) {
                Next::Empty => break,
                Next::Lost(count) => {
                    state.capture.note_dropped_events(count);
                    continue;
                }
                Next::Event(event) => event,
            };
            // Frames are only recorded while the renderer is on. Switching
            // happens at frame boundaries so no half-recorded frame is shown.
            let frame = if recording { Some(frames::building()) } else { None };
            timing.syncs.note(&event);
            let handle_start = sys::now_us();
            let stream_waited_before = recorder_thread::producer_waited_us();
            let missing_before = state.vram.missing();
            let completed = state.handle(&event, frame);
            if state.capture.waiting_for_vram().is_some() {
                state.settle_vram_waits(&ring, recording, missing_before);
            }
            // GE sync: the plugin may be waiting in the game's GE sync until
            // we have read (and copied) this far.
            if sync_wanted {
                let started = sys::now_us();
                state.capture.stream().copy_untaken();
                let previous = nativeres_core::phase::enter(nativeres_core::phase::WAIT);
                while !state.capture.stream().copies_complete() && sys::now_us() - started < COPIES_WAIT_LIMIT_US {
                    sys::sceKernelDelayThread(20);
                }
                nativeres_core::phase::leave(previous);
                timing.syncs.copies_wait_us += (sys::now_us() - started) as u32;
            }
            ring.set_handled(state.reader.consumed());
            if matches!(event.event_type, 1..=3) {
                timing.syncs.walk_us += (sys::now_us() - handle_start) as u32;
                timing.syncs.stream_wait_us += recorder_thread::producer_waited_us().wrapping_sub(stream_waited_before);
            }
            if !completed {
                continue;
            }

            if staged {
                // Stage B waits at the frame's end now.
                state.vram.add_stage_reads(stage_thread::vram_reads());
            }
            COMPLETED_FRAME_COUNT.store(state.capture.frames_completed(), Ordering::Relaxed);
            if state.capture.waiting_for_ge_state() {
                ge_state_frames += 1;
            }
            let now = sys::now_us();
            timing.vita_time_us += now - start_time;
            start_time = now;
            let newest = ring.event(ring.header().write_count.wrapping_sub(1) % event_ring::CAPACITY);
            timing.behind_us = newest.psp_time_us.wrapping_sub(event.psp_time_us).min(10_000_000);
            if recording && RAW_HASH.load(Ordering::Relaxed) {
                // Safe: the capture thread, the frame complete, before its publish.
                let frame = unsafe { frames::building() };
                RECORDED_HASH.store(frame.raw_draw_hash(), Ordering::Relaxed);
                let parts = frame.raw_draw_hash_parts();
                for (slot, part) in RECORDED_PARTS.iter().zip(parts) {
                    slot.store(part as u32, Ordering::Relaxed);
                }
                RECORDED_HASH_FRAME.store(frame.frame_number, Ordering::Relaxed);
            }
            if recording && SKIN_WATCH.load(Ordering::Relaxed) {
                // Safe: as above.
                let frame = unsafe { frames::building() };
                let (edge, draw, triangles) = frame.longest_skinned_edge();
                SKIN_EDGE.store(edge.to_bits(), Ordering::Relaxed);
                SKIN_DRAW.store(draw, Ordering::Relaxed);
                SKIN_TRIANGLES.store(triangles, Ordering::Relaxed);
            }
            log_frame(state, &mut timing, &event, recording, &ring.header());
            if recording {
                let frame = frames::building();
                CAPTURED_FRAME_NUMBER.store(frame.frame_number, Ordering::Relaxed);
                state.request_vram(Some(frame));
                ext::dump_finish(state, frame);
                // A frame that lost events can lack whole display lists (it
                // showed as a black flash), and one that read VRAM not
                // copied yet lacks what is there: keep showing the last one.
                let missing = state.vram.take_missing();
                // A walk that stopped at a bad address or ran away lacks the
                // rest of its list (MTX's attract demo: the logo and menu
                // overlay at the end; the list had changed under a late
                // walk): held back too, but at most frame_gate::MAX_HELD_BACK in
                // a row, so a game that always ends that way still shows.
                // A frame whose GE sync the plugin gave up waiting in (the
                // game rewrote what the walk read: torn) is never shown.
                let stopped_early = state.last_frame.bad_addresses > 0 || state.last_frame.endless_lists > 0;
                // After a savestate load: the registers of the time before
                // until the GE's own come (the village's fog in MHF's field).
                let without_ge_state = state.capture.waiting_for_ge_state() && ge_state_frames < ge_state_limit;
                let checks = nativeres_core::frame_gate::FrameChecks {
                    dropped_events: state.last_frame.dropped_events,
                    missing_vram: missing,
                    stopped_early,
                    without_ge_state,
                    sync_timeouts: (timeouts_at_start, ring.header().sync_timeouts),
                };
                if gate.should_show(&checks) {
                    frames::publish();
                } else {
                    if checks.sync_timeouts.0 != checks.sync_timeouts.1 {
                        TORN_FRAMES.fetch_add(1, Ordering::Relaxed);
                    }
                    frame.clear();
                }
            } else if skipping {
                // Walked, not recorded: keep the VRAM requests (forgetting
                // them made every recorded frame after a skipped one read
                // uncopied VRAM, and it was never shown).
                state.vram.take_missing();
                state.request_vram_keeping_ranges();
            } else {
                state.vram.take_missing();
                state.request_vram(None);
            }
            // The walk missed commands the GE ran: get the GE's registers.
            // (Again if an earlier request went unanswered past its limit.)
            let request_open = state.capture.waiting_for_ge_state() && ge_state_frames < ge_state_limit;
            if state.last_frame.walk_stopped_early() && !request_open {
                ring.request_ge_state();
                state.capture.expect_ge_state();
                ge_state_frames = 0;
                ge_state_limit = MAX_REPAIR_FRAMES;
            }
            // Behind the game (recording took longer than its frame): walk
            // the frames that are already over without recording them, and
            // record the newest (RingReader::newer_frame_waiting).
            // A requested dump is recorded anyway.
            let wanted = have_frames && renderer_enabled();
            recording = wanted && (ext::dump_requested() || !state.reader.newer_frame_waiting(&ring));
            timeouts_at_start = ring.header().sync_timeouts;
            skipping = wanted && !recording;
            // Stage B on its own thread, switched between frames. Not while
            // a dump runs: the dump records the blocks the capture thread
            // reads.
            let stage_wanted = stage_thread::wanted() && !ext::dump_requested() && !state.dump.is_active();
            if stage_wanted != staged {
                staged = stage_wanted && stage_thread::start();
                let capture: *mut Capture = &mut *state.capture;
                if staged {
                    state.capture.set_pipelined(true);
                    state.capture.set_stage_hooks(Some(&stage_thread::HOOKS));
                    let requests = core::ptr::addr_of!(state.vram);
                    stage_thread::serve(capture, stage_thread::Source::Live { requests });
                } else {
                    state.capture.set_stage_hooks(None);
                    stage_thread::serve(core::ptr::null_mut(), stage_thread::Source::Live { requests: core::ptr::null() });
                    state.capture.set_pipelined(false);
                }
                let message: &[u8] = if staged { b"stage B on its own thread
" } else { b"stage B off
" };
                log::append(message);
            }
            ext::dump_start(state, recording);
        }
        timing.vita_time_us += sys::now_us() - start_time;

        if log::waiting() >= log::FLUSH_THRESHOLD {
            log::flush();
        }
        nativeres_core::phase::enter(nativeres_core::phase::IDLE);
        sys::sceKernelDelayThread(POLL_INTERVAL_US);
    }
}

/// Starts the log writer, our options thread and the capture thread
/// (plugin.rs, when Adrenaline starts its own threads).
pub fn start() {
    log::start_writer();
    crate::options::start();
    // Safe: plain system calls with valid arguments.
    unsafe {
        let thread = sys::sceKernelCreateThread(
            b"NativeResCapture\0".as_ptr(),
            capture_thread,
            PRIORITY,
            STACK_SIZE,
            0,
            0,
            core::ptr::null_mut(),
        );
        sys::note_thread(thread);
        if thread >= 0 {
            sys::sceKernelStartThread(thread, 0, core::ptr::null_mut());
        }
    }
}

/// For `dump_file`: the memory view that reads a dumped frame's textures.
pub fn texture_memory(state: &mut CaptureState) -> BlockCache<'_, VitaBlocks<'_>, CAPTURE_BLOCKS> {
    state.memory(false)
}
