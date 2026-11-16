//! The VRAM copy (protocol: `nativepsp_core::psp_link`; decisions: core
//! `psp_copy`, tested there). Sony's emulator keeps PSP VRAM where the Vita
//! side can't read it; we copy the ranges it lists to the copy area and keep
//! them up to date:
//! - ranges new in the list in full (copy thread, after a flip);
//! - at every list enqueue and GE sync, in the game's thread, the recent
//!   ranges' blocks whose samples changed (games load new graphics where
//!   old ones were, MTX rewrites display lists it keeps in VRAM);
//! - in the copy thread the other ranges' changed blocks, and a rolling
//!   refresh for changes the samples miss.
//!
//! **Why the change check is where it is.** When the game has waited for
//! the GE (sceGeDrawSync / sceGeListSync), VRAM holds exactly what the frame
//! was drawn with, and only then may the game change it. Checking at the
//! flip instead was too late: Final Fantasy draws the menu's last frame,
//! loads the field's graphics into VRAM and only then flips, so the copy
//! already held the field's tiles and that menu frame showed them. At a
//! list's enqueue, before the Vita side hears of it: MTX keeps display lists
//! in VRAM (CALL 0x04155000) and rewrites parts of them while driving; the
//! Vita side walks the lists right after the enqueue (2026-11-09). Games
//! that never sync are checked at the flip (copy thread).
//!
//! **Copy methods (2026-11-09).** Reading VRAM through the uncached address
//! costs ~30 us per KiB in the emulator; while driving in MTX the change
//! check after the GE sync copied ~800 KiB in the game's thread (~25 ms). So
//! the methods are timed once against each other (`probe_copy_methods`) and
//! the fastest that copies correctly is used (core `psp_copy::best_method`).
//!
//! **The copy thread.** Woken at every frame flip; the copy used to run in
//! the flip hook (the game's thread), where the copy of new textures (3.3 ms
//! when Final Fantasy's battle appears) made that frame miss its vblank. The
//! PSP has one CPU: with its low priority the thread only gets the time the
//! game spends waiting (usually for the next vblank). If the game leaves it
//! none, the copy arrives late and the Vita side shows the game's own
//! picture meanwhile (it waits for `copied_generation`). `last_copy_us` is
//! wall time, including time the game took in between.

use core::ffi::c_void;
use core::ptr::{addr_of, read_volatile, write_volatile};
use core::sync::atomic::{AtomicU32, AtomicUsize, Ordering::Relaxed};

use nativepsp_core::psp_copy::{
    best_method, copy_span, ge_probe_list, KnownRanges, Refresh, CHECK_BLOCK_BYTES, GE_PROBE_LIST_WORDS,
    REFRESH_BYTES_PER_FRAME,
};
use nativepsp_core::psp_link::{copy_method, VRAM_CACHED, VRAM_COPY_ADDRESS, VRAM_COPY_CACHED_ADDRESS, VRAM_UNCACHED};
use nativepsp_core::vram_requests::{VramRange, MAX_RANGES, VRAM_SIZE};

use crate::sys::*;
use crate::{add, get, now_us, set, table, without_interrupts, Global, CHECK_AT_FLIP, COPY_WAKEUP, SYNCED_SINCE_FLIP};

/// The copy method in use.
static METHOD: AtomicU32 = AtomicU32::new(copy_method::WORDS);
/// sceDmacMemcpy (0 = not found).
static DMAC_MEMCPY: AtomicUsize = AtomicUsize::new(0);
type DmacMemcpyFn = unsafe extern "C" fn(to: *mut c_void, from: *const c_void, size: u32) -> i32;

fn dmac_memcpy() -> Option<DmacMemcpyFn> {
    let address = DMAC_MEMCPY.load(Relaxed);
    // Safe: the kernel function's address, of this type.
    (address != 0).then(|| unsafe { core::mem::transmute::<usize, DmacMemcpyFn>(address) })
}

fn uncached_vram(offset: u32) -> *const u32 {
    (VRAM_UNCACHED + offset) as *const u32
}

fn cached_vram(offset: u32) -> *const u32 {
    (VRAM_CACHED + offset) as *const u32
}

fn uncached_copy(offset: u32) -> *mut u32 {
    (VRAM_COPY_ADDRESS + offset) as *mut u32
}

fn cached_copy(offset: u32) -> *mut u32 {
    (VRAM_COPY_CACHED_ADDRESS + offset) as *mut u32
}

// The copies below take an offset and size inside VRAM, multiples of 4
// (`copy_vram` and the probe make sure). Volatile accesses keep each a loop
// of word loads and stores as written (the compiler would make the plain
// ones a memcpy call: that is method MEMCPY).

/// Word by word, both sides uncached: no cache maintenance needed.
fn copy_words(offset: u32, size: u32) {
    let (from, to) = (uncached_vram(offset), uncached_copy(offset));
    for word in 0..(size / 4) as usize {
        // Safe: inside VRAM and the copy area (see above).
        unsafe { write_volatile(to.add(word), read_volatile(from.add(word))) };
    }
}

/// Word by word through the cached addresses: the cache invalidated
/// before, written back after.
fn copy_cached(offset: u32, size: u32) {
    let (from, to) = (cached_vram(offset), cached_copy(offset));
    // Safe: inside VRAM and the copy area.
    unsafe {
        sceKernelDcacheInvalidateRange(from.cast(), size);
        for word in 0..(size / 4) as usize {
            write_volatile(to.add(word), read_volatile(from.add(word)));
        }
        sceKernelDcacheWritebackRange(to.cast(), size);
    }
}

/// Cached, 8 words per step: fewer emulated instructions per word (the
/// emulator's cost is per instruction as much as per memory access).
fn copy_unrolled(offset: u32, size: u32) {
    let (from, to) = (cached_vram(offset), cached_copy(offset));
    let words = (size / 4) as usize;
    // Safe: inside VRAM and the copy area.
    unsafe {
        sceKernelDcacheInvalidateRange(from.cast(), size);
        let mut word = 0;
        while word + 8 <= words {
            let mut eight = [0u32; 8];
            for (i, value) in eight.iter_mut().enumerate() {
                *value = read_volatile(from.add(word + i));
            }
            for (i, value) in eight.iter().enumerate() {
                write_volatile(to.add(word + i), *value);
            }
            word += 8;
        }
        while word < words {
            write_volatile(to.add(word), read_volatile(from.add(word)));
            word += 1;
        }
        sceKernelDcacheWritebackRange(to.cast(), size);
    }
}

/// VFPU quad loads and stores, 64 bytes per step, the rest word by word.
/// Only in threads whose VFPU state the kernel saves (`thread_has_vfpu`).
fn copy_vfpu(offset: u32, size: u32) {
    let (from, to) = (cached_vram(offset), cached_copy(offset));
    // Safe: inside VRAM and the copy area; nps_vfpu_copy64 needs 16-byte
    // alignment, checked.
    unsafe {
        sceKernelDcacheInvalidateRange(from.cast(), size);
        let mut done = 0;
        if (from as u32) & 15 == 0 && (to as u32) & 15 == 0 {
            nps_vfpu_copy64(from, to, size / 64);
            done = size / 64 * 64;
        }
        for word in (done / 4) as usize..(size / 4) as usize {
            write_volatile(to.add(word), read_volatile(from.add(word)));
        }
        sceKernelDcacheWritebackRange(to.cast(), size);
    }
}

/// Cached, the kernel's memcpy.
fn copy_memcpy(offset: u32, size: u32) {
    let (from, to) = (cached_vram(offset), cached_copy(offset));
    // Safe: inside VRAM and the copy area.
    unsafe {
        sceKernelDcacheInvalidateRange(from.cast(), size);
        memcpy(to.cast(), from.cast(), size as usize);
        sceKernelDcacheWritebackRange(to.cast(), size);
    }
}

/// The last thread asked whether it has the VFPU, and the answer (the
/// game's thread at its GE syncs, the copy thread).
static VFPU_THREAD: Global<(SceUID, bool)> = Global::new((-1, false));

fn thread_has_vfpu() -> bool {
    // Safe: plain kernel calls; the cache is two words, and a thread
    // switch in between only costs another lookup (as in the C plugin).
    unsafe {
        let thread = sceKernelGetThreadId();
        let known = VFPU_THREAD.get();
        if thread != known.0 {
            let mut info: ThreadInfo = core::mem::zeroed();
            info.size = core::mem::size_of::<ThreadInfo>() as u32;
            let vfpu = sceKernelReferThreadStatus(thread, &mut info) >= 0 && info.attr & THREAD_ATTR_VFPU != 0;
            *known = (thread, vfpu);
        }
        known.1
    }
}

/// Copies with `method`; false if it couldn't.
fn copy_with(method: u32, offset: u32, size: u32) -> bool {
    match method {
        copy_method::CACHED => copy_cached(offset, size),
        copy_method::UNROLLED => copy_unrolled(offset, size),
        copy_method::VFPU => {
            if !thread_has_vfpu() {
                return false;
            }
            copy_vfpu(offset, size)
        }
        copy_method::MEMCPY => copy_memcpy(offset, size),
        copy_method::DMA_CACHED => {
            let Some(dmac) = dmac_memcpy() else { return false };
            let (from, to) = (cached_vram(offset), cached_copy(offset));
            // Safe: inside VRAM and the copy area.
            return unsafe {
                sceKernelDcacheWritebackInvalidateRange(to.cast(), size);
                dmac(to.cast(), from.cast(), size) >= 0
            };
        }
        copy_method::DMA => {
            let Some(dmac) = dmac_memcpy() else { return false };
            // Safe: as above.
            return unsafe { dmac(uncached_copy(offset).cast(), uncached_vram(offset).cast(), size) >= 0 };
        }
        _ => copy_words(offset, size),
    }
    true
}

/// Whether the copy of `size` bytes at `offset` equals VRAM (both read
/// uncached, as the Vita side sees the copy).
fn copy_matches(offset: u32, size: u32) -> bool {
    let (vram, copy) = (uncached_vram(offset), uncached_copy(offset));
    // Safe: inside VRAM and the copy area.
    (0..(size / 4) as usize).all(|word| unsafe { read_volatile(vram.add(word)) == read_volatile(copy.add(word)) })
}

/// The probe copies the top 64 KiB, which games rarely draw into (VRAM
/// there may change while we copy: a method only counts as wrong if a
/// second try also differs).
const PROBE_OFFSET: u32 = VRAM_SIZE - 0x10000;
const PROBE_BYTES: u32 = 0x10000;
const NID_DMAC_MEMCPY: u32 = 0x617F_3FE6;
const NID_DMAC_TRY_MEMCPY: u32 = 0xD97F_94D8;

/// Times each copy method on the same 64 KiB of VRAM (written where its
/// copy belongs, so the copy area only gets fresher) and picks the fastest
/// that copies correctly. Copy thread, once at its start.
fn probe_copy_methods() {
    // sceDmacMemcpy / sceDmacTryMemcpy under the names the firmware may use.
    const LOOKUPS: [(&[u8], &[u8], u32); 6] = [
        (b"sceDMAManager\0", b"sceDmac\0", NID_DMAC_MEMCPY),
        (b"sceDMAManager\0", b"sceDmac\0", NID_DMAC_TRY_MEMCPY),
        (b"sceDMAManager\0", b"sceDmac_driver\0", NID_DMAC_MEMCPY),
        (b"sceDMAManager\0", b"sceDmac_driver\0", NID_DMAC_TRY_MEMCPY),
        (b"sceDmac\0", b"sceDmac\0", NID_DMAC_MEMCPY),
        (b"sceDMACManager\0", b"sceDmac\0", NID_DMAC_MEMCPY),
    ];
    DMAC_MEMCPY.store(0, Relaxed);
    set!(probe_dma_lookup, 0);
    for (i, (module, library, nid)) in LOOKUPS.iter().enumerate() {
        // Safe: NUL-terminated names.
        let found = unsafe { sctrlHENFindFunction(module.as_ptr(), library.as_ptr(), *nid) };
        if found != 0 {
            DMAC_MEMCPY.store(found as usize, Relaxed);
            set!(probe_dma_lookup, i as u32 + 1);
            break;
        }
    }
    set!(probe_bytes, PROBE_BYTES);
    set!(probe_ok, 0);
    let mut us = [0u32; 8];
    let mut ok = 0u32;
    for method in copy_method::DMA..=copy_method::METHODS {
        let index = (method - 1) as usize;
        let mut correct = false;
        let mut took = 0;
        for _attempt in 0..2 {
            let start = now_us();
            let done = copy_with(method, PROBE_OFFSET, PROBE_BYTES);
            took = now_us().wrapping_sub(start);
            correct = done && copy_matches(PROBE_OFFSET, PROBE_BYTES);
            if correct {
                break;
            }
        }
        us[index] = took;
        set_probe_us(index, took);
        if correct {
            ok |= 1 << index;
            set!(probe_ok, ok);
        }
    }
    let best = best_method(&us, ok);
    METHOD.store(best, Relaxed);
    set!(probe_chosen, best);
}

fn set_probe_us(index: usize, us: u32) {
    if index < 8 {
        // Safe: the table is fixed, mapped PSP memory; index checked.
        unsafe { write_volatile((core::ptr::addr_of_mut!((*table()).probe_us) as *mut u32).add(index), us) };
    }
}

/// Copies `size` bytes at `offset` (rounded to words, inside VRAM) with the
/// method in use, word by word where that can't.
fn copy_vram(offset: u32, size: u32) {
    let Some((offset, size)) = copy_span(offset, size) else { return };
    let method = METHOD.load(Relaxed);
    if copy_with(method, offset, size) {
        set!(copy_method, method);
    } else {
        copy_words(offset, size);
        set!(copy_method, copy_method::WORDS);
    }
}

/// The ranges copied in full and kept up to date. Written only by the copy
/// thread (or the flip hook when there is none), with interrupts off; the
/// game's threads read it at their GE syncs and enqueues, which run at a
/// higher priority than the copy thread, so a check never sees a half
/// update.
static KNOWN: Global<KnownRanges> = Global::new(KnownRanges::new());
/// The generation `KNOWN` is from.
static KNOWN_GENERATION: AtomicU32 = AtomicU32::new(u32::MAX);
/// The rolling refresh (copy thread only).
static REFRESH: Global<Refresh> = Global::new(Refresh::new());

fn known() -> &'static KnownRanges {
    // Safe: see `KNOWN`.
    unsafe { KNOWN.get() }
}

/// Copies blocks of the recent ranges (`recent`) or of the others whose
/// samples changed, and counts them in `changes`.
fn check_for_changes(recent: bool) {
    let (vram, copy) = (VRAM_UNCACHED as *const u32, VRAM_COPY_ADDRESS as *const u32);
    let changed = known().changed_blocks(
        get!(recent_mask),
        recent,
        // Safe: blocks inside VRAM and the copy area (changed_blocks stops
        // at VRAM's end), word inside the block.
        |block, word| unsafe {
            let index = (block / 4 + word) as usize;
            read_volatile(vram.add(index)) != read_volatile(copy.add(index))
        },
        |block| copy_vram(block, CHECK_BLOCK_BYTES),
    );
    if changed != 0 {
        add!(changes, changed);
    }
}

/// At a list's enqueue, before the Vita side hears of it: the recent
/// ranges' changed blocks are copied now (game's thread; not extra time,
/// only earlier than at the sync).
pub fn refresh_before_walk() {
    if known().count == 0 {
        return;
    }
    let start = now_us();
    check_for_changes(true);
    add!(enqueue_checks, 1);
    set!(last_enqueue_check_us, now_us().wrapping_sub(start));
}

/// At the game's GE sync (its thread): the recent ranges' changed blocks.
pub fn check_at_sync() {
    if known().count == 0 {
        return;
    }
    let start = now_us();
    check_for_changes(true);
    SYNCED_SINCE_FLIP.store(true, Relaxed);
    add!(sync_checks, 1);
    set!(last_sync_check_us, now_us().wrapping_sub(start));
}

/// Copies the VRAM ranges the Vita side asked for: ranges new in its list
/// in full, then the change checks and the rolling refresh.
pub fn copy_requested_vram() {
    let count = (get!(range_count) as usize).min(MAX_RANGES);
    if count == 0 {
        // Safe: see `KNOWN`.
        without_interrupts(|| unsafe { KNOWN.get().count = 0 });
        set!(copied_generation, get!(generation)); // nothing to copy
        return;
    }
    let start = now_us();
    let generation = get!(generation);
    if generation != KNOWN_GENERATION.load(Relaxed) {
        let mut ranges = [VramRange::default(); MAX_RANGES];
        for (i, range) in ranges.iter_mut().enumerate().take(count) {
            // Safe: the table is fixed, mapped PSP memory; i < MAX_RANGES.
            *range = unsafe { read_volatile((addr_of!((*table()).ranges) as *const VramRange).add(i)) };
            if !known().is_known(range.offset, range.size) {
                copy_vram(range.offset, range.size);
            }
        }
        // The game's threads may check for changes in between: they must
        // not see half-updated ranges.
        without_interrupts(|| {
            // Safe: see `KNOWN`.
            let known = unsafe { KNOWN.get() };
            known.ranges = ranges;
            known.count = count;
            KNOWN_GENERATION.store(generation, Relaxed);
            // Safe: copy thread only (or the flip hook without one).
            unsafe { *REFRESH.get() = Refresh::new() };
        });
        set!(copied_generation, generation);
    }
    if CHECK_AT_FLIP.load(Relaxed) {
        check_for_changes(true);
    }
    check_for_changes(false);
    // Safe: copy thread only (or the flip hook without one).
    unsafe { REFRESH.get() }.step(known(), REFRESH_BYTES_PER_FRAME, |offset, size| copy_vram(offset, size));
    set!(last_copy_us, now_us().wrapping_sub(start));
}

/// GE block transfers (TRXKICK) as a VRAM copy (user, 2026-11-09: "can we
/// do the memcpy there?"), measured only, on request
/// (`PspLink::ge_probe_request`), never at boot. Two tries crashed the emulator (C1-2858-3): a
/// list written to an invalid address (0x40000000 | kernel address), then a
/// correct kernel-memory list copying VRAM into the copy area (0x0BE00000,
/// which is also where the emulator keeps its own VRAM translation). So the
/// probe has modes that narrow it down, one per request, all lists in user
/// memory:
///   1 an empty list (FINISH, END)          - is our enqueue accepted at all?
///   2 RAM -> RAM transfer (user memory)    - does the block transfer work?
///   3 VRAM -> RAM transfer (user memory)   - can the GE read VRAM out?
///   4 VRAM -> the copy area (0x0BE00000)   - the emulator's VRAM region?
///   5 a 256 KiB GE copy right after the game's GE sync (MTX's requested
///     list range is ~259 KiB), when the GE is idle: the time a copy at
///     that moment really takes, in the game's thread.
/// The list goes straight to the original sceGeListEnQueue (so the Vita
/// side's capture never sees it), callback id -1 (none), and is waited for
/// by polling with a timeout.
const GE_COPY_TIMEOUT_US: u32 = 50_000;
/// sceGeListSync mode 1 (peek): PSP_GE_LIST_COMPLETED.
const GE_LIST_COMPLETED: i32 = 0;
/// Finished lists are released: asking again gives "invalid ID".
const GE_INVALID_ID: i32 = 0x8000_0100_u32 as i32;
const GE_SCRATCH_BYTES: u32 = 64 * 1024;
const GE_SYNC_PROBE_BYTES: u32 = 256 * 1024;

/// User memory for the GE probe: the list (256 bytes), then the scratch
/// (0 until allocated; copy thread and the game's sync, never at once: a
/// request is for one of them).
static GE_BLOCK_BASE: AtomicU32 = AtomicU32::new(0);

/// Physical address (what the GE takes) of any PSP address.
fn physical(address: u32) -> u32 {
    address & 0x1FFF_FFFF
}

fn ge_user_memory() -> Option<u32> {
    let base = GE_BLOCK_BASE.load(Relaxed);
    if base != 0 {
        return Some(base);
    }
    // Safe: plain kernel calls, a NUL-terminated name.
    let base = unsafe {
        let block = sceKernelAllocPartitionMemory(
            USER_PARTITION,
            b"NativePspGeProbe\0".as_ptr(),
            SMEM_HIGH,
            256 + GE_SCRATCH_BYTES,
            core::ptr::null_mut(),
        );
        if block < 0 {
            return None;
        }
        sceKernelGetBlockHeadAddr(block) as u32
    };
    GE_BLOCK_BASE.store(base, Relaxed);
    (base != 0).then_some(base)
}

/// Builds the list for `mode` in user memory (written uncached) and runs
/// it. 0 when the GE has finished it, else an error.
fn run_ge_probe_list(base: u32, mode: u32, from: u32, to: u32, bytes: u32) -> i32 {
    let (Some(enqueue), Some(sync)) = (crate::original_enqueue(), crate::original_list_sync()) else {
        return -1;
    };
    let mut words = [0u32; GE_PROBE_LIST_WORDS];
    let n = ge_probe_list(mode, from, to, bytes, &mut words);
    let list = (0x4000_0000 | physical(base)) as *mut u32; // user, uncached
    for (i, word) in words.iter().enumerate().take(n) {
        // Safe: our 256-byte list in user memory.
        unsafe { write_volatile(list.add(i), *word) };
    }
    // Safe: the original GE function with our list; no stall, no callback.
    let queue = unsafe { enqueue(0x0800_0000 | (physical(base) & 0x01FF_FFFF), 0, -1, 0) };
    set!(ge_probe_qid, queue);
    set!(ge_probe_state, 0);
    set!(ge_probe_polls, 0);
    if queue < 0 {
        return queue;
    }
    let start = now_us();
    loop {
        // Safe: as above.
        let state = unsafe { sync(queue, 1) };
        set!(ge_probe_state, state);
        add!(ge_probe_polls, 1);
        // "Invalid ID" after an accepted enqueue means done.
        if state == GE_LIST_COMPLETED || (state == GE_INVALID_ID && get!(ge_probe_polls) > 1) {
            return 0;
        }
        if state < 0 {
            return state;
        }
        if now_us().wrapping_sub(start) > GE_COPY_TIMEOUT_US {
            return -2;
        }
        // Safe: a plain kernel call.
        unsafe { sceKernelDelayThread(50) };
    }
}

/// Whether `bytes` at the physical addresses `a` and `b` are equal (read
/// uncached).
fn equal(a: u32, b: u32, bytes: u32) -> bool {
    let (a, b) = ((0x4000_0000 | a) as *const u32, (0x4000_0000 | b) as *const u32);
    // Safe: VRAM, our user memory or the copy area, as the probe chose.
    (0..(bytes / 4) as usize).all(|i| unsafe { read_volatile(a.add(i)) == read_volatile(b.add(i)) })
}

fn ge_probe_done(result: i32, took: u32) {
    set_probe_us((copy_method::GE - 1) as usize, took);
    set!(ge_probe_result, result);
    set!(ge_probe_request, 0);
}

/// The GE probe, mode 1-4, in the copy thread: result 0 = finished and
/// copied correctly, 1 = copied wrongly, < 0 = error (-10 no user memory,
/// -1 no GE functions, -2 timeout, else the GE's error code).
fn probe_ge_on_request() {
    let mode = get!(ge_probe_request);
    if mode == 0 || mode == 5 {
        return; // 5: the game's sync hook runs it
    }
    let mut took = 0;
    let result = if mode > 4 || crate::original_enqueue().is_none() || crate::original_list_sync().is_none() {
        -1
    } else if let Some(base) = ge_user_memory() {
        let scratch = physical(base) + 256;
        let (from, to, bytes) = match mode {
            2 => {
                let half = GE_SCRATCH_BYTES / 2;
                // A pattern to copy.
                let source = (0x4000_0000 | scratch) as *mut u32;
                for i in 0..half / 4 {
                    // Safe: our scratch in user memory.
                    unsafe { write_volatile(source.add(i as usize), 0x9E37_79B9_u32.wrapping_mul(i + 1)) };
                }
                (scratch, scratch + half, half)
            }
            3 => (VRAM_CACHED + PROBE_OFFSET, scratch, GE_SCRATCH_BYTES),
            4 => (VRAM_CACHED + PROBE_OFFSET, physical(VRAM_COPY_ADDRESS) + PROBE_OFFSET, PROBE_BYTES),
            _ => (0, 0, 0),
        };
        let start = now_us();
        let mut result = run_ge_probe_list(base, mode, from, to, bytes);
        took = now_us().wrapping_sub(start);
        if result == 0 && mode != 1 && !equal(from, to, bytes) {
            result = 1;
        }
        result
    } else {
        -10
    };
    ge_probe_done(result, took);
}

/// `geprobe 5`, right after the game's GE sync.
pub fn probe_ge_at_sync() {
    if get!(ge_probe_request) != 5 {
        return;
    }
    let mut took = 0;
    let result = match ge_user_memory() {
        Some(base) => {
            let offset = VRAM_SIZE - GE_SYNC_PROBE_BYTES;
            let start = now_us();
            let mut result =
                run_ge_probe_list(base, 4, VRAM_CACHED + offset, physical(VRAM_COPY_ADDRESS) + offset, GE_SYNC_PROBE_BYTES);
            took = now_us().wrapping_sub(start);
            if result == 0 && !copy_matches(offset, GE_SYNC_PROBE_BYTES) {
                result = 1;
            }
            result
        }
        None => -10,
    };
    ge_probe_done(result, took);
}

/// PSP: a higher number is a lower priority.
const COPY_THREAD_PRIORITY: i32 = 0x70;
const COPY_THREAD_STACK_SIZE: i32 = 0x1000;

unsafe extern "C" fn copy_thread(_args: u32, _argp: *mut c_void) -> i32 {
    probe_copy_methods();
    loop {
        let result = sceKernelWaitSema(COPY_WAKEUP.load(Relaxed), 1, core::ptr::null_mut());
        if result < 0 {
            set!(copy_thread_error, result as u32);
            COPY_WAKEUP.store(-1, Relaxed); // the flip hook copies from now on
            return 0;
        }
        copy_requested_vram();
        probe_ge_on_request();
        add!(copy_thread_runs, 1);
    }
}

/// Starts the copy thread; if that fails, the flip hook copies in the
/// game's thread (as before the thread existed).
pub fn start_thread() {
    // Safe: plain kernel calls, NUL-terminated names.
    unsafe {
        let wakeup = sceKernelCreateSema(b"NativePspCopyWakeup\0".as_ptr(), 0, 0, 1, core::ptr::null_mut());
        if wakeup < 0 {
            set!(copy_thread_error, wakeup as u32);
            return;
        }
        COPY_WAKEUP.store(wakeup, Relaxed);
        let thread = sceKernelCreateThread(
            b"NativePspVramCopy\0".as_ptr(),
            copy_thread,
            COPY_THREAD_PRIORITY,
            COPY_THREAD_STACK_SIZE,
            THREAD_ATTR_VFPU,
            core::ptr::null_mut(),
        );
        let result = if thread < 0 { thread } else { sceKernelStartThread(thread, 0, core::ptr::null_mut()) };
        if result < 0 {
            set!(copy_thread_error, result as u32);
            COPY_WAKEUP.store(-1, Relaxed);
            sceKernelDeleteSema(wakeup);
        }
    }
}
