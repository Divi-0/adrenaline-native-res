//! NativeRes's PSP side: the kernel module nativeres.prx that runs with every
//! PSP game (loaded by Adrenaline's plugins.txt; the Vita plugin
//! nativeres.suprx carries it and installs it).
//!
//! Observe only: it hooks the calls a game uses to hand display lists to the
//! PSP graphics engine (GE) and to show a finished frame, and records each
//! call as an event in the shared ring (`nativeres_core::event_ring`). The
//! Vita side reads the ring and walks the same display lists. Every hook
//! calls the original function unchanged, so the game renders exactly as
//! before. One exception (2026-11-08): after the game's GE sync it may wait
//! for the Vita side to read the lists (`wait_for_vita`), so the game can't
//! reuse their memory before we have read it.
//!
//! It also copies the VRAM the Vita side asks for (`copy`, protocol:
//! `nativeres_core::psp_link`).
//!
//! Hooking is done as GePatch (TheFloW) does it: the kernel function is
//! found by NID and its system call pointed at our function, so the game's
//! user-mode calls land there (in kernel mode).

#![no_std]
#![no_main]
#![feature(asm_experimental_arch)]
// get!/set! carry their own unsafe blocks, also inside unsafe fns.
#![allow(unused_unsafe)]

mod copy;
mod sys;

use core::ffi::c_void;
use core::ptr::{addr_of_mut, read_volatile, write_volatile};
use core::sync::atomic::{AtomicBool, AtomicI32, AtomicUsize, Ordering::Relaxed};

use nativeres_core::event_ring::{
    Ring, CAPACITY, EVENT_FRAMEBUFFER_FLIP, EVENT_GE_RESTORE_CONTEXT, EVENT_GE_SAVE_CONTEXT, EVENT_GE_STATE, EVENT_GE_SYNC,
    EVENT_LIST_ENQUEUE, EVENT_LIST_ENQUEUE_HEAD, EVENT_STALL_UPDATE, GE_STATE_BONES, GE_STATE_PROJECTION,
    GE_STATE_REGISTERS, GE_STATE_TEXGEN, GE_STATE_VIEW, GE_STATE_WORDS, GE_STATE_WORLD, MAGIC, RING_ADDRESS,
    SYNC_TIMEOUT_US, SYNC_WANTED, VERSION, COPIES_AT_SYNC,
};
use nativeres_core::psp_copy::SyncStall;
use nativeres_core::psp_link::{copy_method, VramRequestTable, VRAM_REQUESTS_ADDRESS};

use sys::*;

/// NIDs of the functions we hook (6.61 firmware).
const NID_GE_LIST_ENQUEUE: u32 = 0xAB49_E76A;
const NID_GE_LIST_ENQUEUE_HEAD: u32 = 0x1C0D_95A6;
const NID_GE_LIST_UPDATE_STALL_ADDR: u32 = 0xE0D6_8148;
const NID_DISPLAY_SET_FRAME_BUF: u32 = 0x289D_82FE;
const NID_GE_DRAW_SYNC: u32 = 0xB287_BD61;
const NID_GE_LIST_SYNC: u32 = 0x0344_4EB4;
const NID_GE_SAVE_CONTEXT: u32 = 0x438A_385A;
const NID_GE_RESTORE_CONTEXT: u32 = 0x0BF6_08FB;
/// Called, not hooked (the GE state block, `answer_state_request`).
const NID_GE_GET_CMD: u32 = 0xDC93_CFEF;
const NID_GE_GET_MTX: u32 = 0x57C8_945B;

type EnqueueFn = unsafe extern "C" fn(list: u32, stall: u32, callback: i32, args: u32) -> i32;
type StallFn = unsafe extern "C" fn(queue: i32, stall: u32) -> i32;
type FrameBufFn = unsafe extern "C" fn(address: u32, width: i32, format: i32, sync: i32) -> i32;
type DrawSyncFn = unsafe extern "C" fn(mode: i32) -> i32;
type ListSyncFn = unsafe extern "C" fn(queue: i32, mode: i32) -> i32;
type ContextFn = unsafe extern "C" fn(context: u32) -> i32;
type GetCmdFn = unsafe extern "C" fn(cmd: i32) -> u32;
type GetMtxFn = unsafe extern "C" fn(kind: i32, matrix: *mut u32) -> i32;

/// The original functions (0 until hooked; set before the system call is
/// patched, so a hook never sees 0).
static ORIGINAL_ENQUEUE: AtomicUsize = AtomicUsize::new(0);
static ORIGINAL_ENQUEUE_HEAD: AtomicUsize = AtomicUsize::new(0);
static ORIGINAL_UPDATE_STALL: AtomicUsize = AtomicUsize::new(0);
static ORIGINAL_SET_FRAME_BUF: AtomicUsize = AtomicUsize::new(0);
static ORIGINAL_DRAW_SYNC: AtomicUsize = AtomicUsize::new(0);
static ORIGINAL_LIST_SYNC: AtomicUsize = AtomicUsize::new(0);
static ORIGINAL_SAVE_CONTEXT: AtomicUsize = AtomicUsize::new(0);
static ORIGINAL_RESTORE_CONTEXT: AtomicUsize = AtomicUsize::new(0);
static GE_GET_CMD: AtomicUsize = AtomicUsize::new(0);
static GE_GET_MTX: AtomicUsize = AtomicUsize::new(0);

/// A mutable static for the plugin's single-CPU threads. Who touches what,
/// and with which care, is said at each one (as the C plugin's statics).
pub struct Global<T>(core::cell::UnsafeCell<T>);

// Safe: the PSP has one CPU; each user documents how its threads share it.
unsafe impl<T> Sync for Global<T> {}

impl<T> Global<T> {
    pub const fn new(value: T) -> Self {
        Global(core::cell::UnsafeCell::new(value))
    }

    /// Safety: no other reference to the value is used meanwhile.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn get(&self) -> &mut T {
        &mut *self.0.get()
    }
}

/// The shared ring (uncached, so writes go straight to memory).
pub fn ring() -> *mut Ring {
    RING_ADDRESS as *mut Ring
}

/// The shared VRAM request table (uncached).
pub fn table() -> *mut VramRequestTable {
    VRAM_REQUESTS_ADDRESS as *mut VramRequestTable
}

/// Reads a field of the table.
macro_rules! get {
    ($field:ident) => {
        // Safe: the table is fixed, mapped PSP memory.
        unsafe { ::core::ptr::read_volatile(::core::ptr::addr_of!((*$crate::table()).$field)) }
    };
}

/// Writes a field of the table.
macro_rules! set {
    ($field:ident, $value:expr) => {{
        let value = $value;
        // Safe: as `get!`.
        unsafe { ::core::ptr::write_volatile(::core::ptr::addr_of_mut!((*$crate::table()).$field), value) }
    }};
}

/// Adds to a counter in the table.
macro_rules! add {
    ($field:ident, $value:expr) => {{
        let value = $value;
        set!($field, get!($field).wrapping_add(value))
    }};
}

pub(crate) use {add, get, set};

/// Reads a field of the ring header.
fn header_get(field: fn(*mut Ring) -> *mut u32) -> u32 {
    // Safe: the ring is fixed, mapped PSP memory.
    unsafe { read_volatile(field(ring())) }
}

fn header_set(field: fn(*mut Ring) -> *mut u32, value: u32) {
    // Safe: as `header_get`.
    unsafe { write_volatile(field(ring()), value) }
}

macro_rules! header {
    ($field:ident) => {
        // Safe: only forms the field's address.
        |ring: *mut Ring| unsafe { addr_of_mut!((*ring).header.$field) }
    };
}

pub fn now_us() -> u32 {
    // Safe: a plain kernel call.
    unsafe { sceKernelGetSystemTimeLow() }
}

/// Interrupts off on the PSP's one CPU: nothing else runs meanwhile.
pub fn without_interrupts<R>(work: impl FnOnce() -> R) -> R {
    // Safe: plain kernel calls, paired.
    let flags = unsafe { sceKernelCpuSuspendIntr() };
    let result = work();
    unsafe { sceKernelCpuResumeIntr(flags) };
    result
}

/// Appends one event to the ring. Interrupts are off so two game threads
/// calling GE functions at the same time can't write the same slot.
fn push_event(event_type: u32, args: [u32; 4]) {
    without_interrupts(|| {
        let sequence = header_get(header!(write_count));
        // Safe: the ring is fixed, mapped PSP memory; the index is below
        // CAPACITY. The fields in the C plugin's order, the sequence last.
        unsafe {
            let event = addr_of_mut!((*ring()).events[(sequence % CAPACITY) as usize]);
            write_volatile(addr_of_mut!((*event).event_type), event_type);
            for (i, arg) in args.iter().enumerate() {
                write_volatile(addr_of_mut!((*event).args[i]), *arg);
            }
            write_volatile(addr_of_mut!((*event).psp_time_us), now_us());
            write_volatile(addr_of_mut!((*event).sequence), sequence);
        }
        header_set(header!(write_count), sequence.wrapping_add(1));
    });
}

fn original(slot: &AtomicUsize) -> usize {
    slot.load(Relaxed)
}

/// The context address in the game's `PspGeListArgs` (size, context,
/// stack count, stacks), 0 if none: the GE saves its state there when the
/// list starts and restores it when the list ends (core `ge_context`).
fn list_context(args: u32) -> u32 {
    // The game's pointer: user memory only (or its uncached mirror).
    let user = args & 0x3FFF_FFFF;
    if args % 4 != 0 || !(0x0880_0000..0x0A00_0000 - 8).contains(&user) {
        return 0;
    }
    // Safe: a word-aligned address inside user RAM.
    unsafe {
        let size = read_volatile(args as *const u32);
        if size < 8 {
            return 0;
        }
        read_volatile((args + 4) as *const u32)
    }
}

unsafe extern "C" fn ge_list_enqueue_hook(list: u32, stall: u32, callback: i32, args: u32) -> i32 {
    let original: EnqueueFn = core::mem::transmute(original(&ORIGINAL_ENQUEUE));
    let result = original(list, stall, callback, args);
    copy::refresh_before_walk();
    push_event(EVENT_LIST_ENQUEUE, [list, stall, result as u32, list_context(args)]);
    result
}

unsafe extern "C" fn ge_list_enqueue_head_hook(list: u32, stall: u32, callback: i32, args: u32) -> i32 {
    let original: EnqueueFn = core::mem::transmute(original(&ORIGINAL_ENQUEUE_HEAD));
    let result = original(list, stall, callback, args);
    copy::refresh_before_walk();
    push_event(EVENT_LIST_ENQUEUE_HEAD, [list, stall, result as u32, list_context(args)]);
    result
}

unsafe extern "C" fn ge_save_context_hook(context: u32) -> i32 {
    let original: ContextFn = core::mem::transmute(original(&ORIGINAL_SAVE_CONTEXT));
    let result = original(context);
    push_event(EVENT_GE_SAVE_CONTEXT, [context, result as u32, 0, 0]);
    result
}

unsafe extern "C" fn ge_restore_context_hook(context: u32) -> i32 {
    let original: ContextFn = core::mem::transmute(original(&ORIGINAL_RESTORE_CONTEXT));
    let result = original(context);
    push_event(EVENT_GE_RESTORE_CONTEXT, [context, result as u32, 0, 0]);
    result
}

unsafe extern "C" fn ge_list_update_stall_addr_hook(queue: i32, stall: u32) -> i32 {
    let original: StallFn = core::mem::transmute(original(&ORIGINAL_UPDATE_STALL));
    let result = original(queue, stall);
    push_event(EVENT_STALL_UPDATE, [queue as u32, stall, result as u32, 0]);
    result
}

/// The original sceGeListEnQueue (the GE probe's lists, which the Vita
/// side's capture never sees).
pub fn original_enqueue() -> Option<EnqueueFn> {
    let address = original(&ORIGINAL_ENQUEUE);
    // Safe: the kernel function's address, of this type.
    (address != 0).then(|| unsafe { core::mem::transmute::<usize, EnqueueFn>(address) })
}

/// The original sceGeListSync.
pub fn original_list_sync() -> Option<ListSyncFn> {
    let address = original(&ORIGINAL_LIST_SYNC);
    // Safe: as `original_enqueue`.
    (address != 0).then(|| unsafe { core::mem::transmute::<usize, ListSyncFn>(address) })
}

/// The GE sync wait's state (game threads only, at their GE syncs).
static SYNC_STALL: Global<SyncStall> = Global::new(SyncStall::new());

/// After the game waited for the GE: while the Vita side asks for it, wait
/// until it has handled every event written so far, at most
/// SYNC_TIMEOUT_US (core `psp_copy::SyncStall`). Returns the microseconds
/// waited (0 if it didn't).
/// How long the GE sync wait goes on after a VRAM copy the Vita side
/// requested was last outstanding (`wait_for_vita`): its walk resumes after
/// the copy (it waits at most 30 ms itself).
const VRAM_COPY_GRACE_US: u32 = 35_000;

fn wait_for_vita() -> u32 {
    if header_get(header!(vita_sync_wanted)) != SYNC_WANTED {
        return 0;
    }
    let target = header_get(header!(write_count));
    // Safe: game threads at their GE syncs only; none waits inside.
    let stall = unsafe { SYNC_STALL.get() };
    if !stall.should_wait(header_get(header!(vita_handled)), target) {
        return 0;
    }
    let start = now_us();
    // The Vita side may be waiting for VRAM blocks it just requested (its
    // walk stopped at new VRAM content). The copy thread copies them now, in
    // the time the game waits here anyway, and the wait goes on while such a
    // copy is outstanding or just done (the walk resumes after it): leaving
    // earlier let the game reuse its list memory under the resumed walk
    // (MHF's intro fly-over black, 2026-11-18). Bounded by VRAM_COPY_GRACE_US
    // after the last outstanding copy and by SYNC_TIMEOUT_US + that in all.
    let mut woken_for = get!(copied_generation);
    let mut copy_seen = 0u32; // when a copy was last outstanding (0: never)
    loop {
        let generation = get!(generation);
        if generation != get!(copied_generation) {
            copy_seen = now_us() | 1;
            if generation != woken_for {
                woken_for = generation;
                wake_copier();
            }
        }
        // Safe: a plain kernel call.
        unsafe { sceKernelDelayThread(100) };
        if nativeres_core::psp_copy::reached(header_get(header!(vita_handled)), target) {
            break;
        }
        if header_get(header!(vita_sync_wanted)) != SYNC_WANTED {
            break;
        }
        let now = now_us();
        let copying = copy_seen != 0 && now.wrapping_sub(copy_seen) < VRAM_COPY_GRACE_US;
        let waited = now.wrapping_sub(start);
        if (waited >= SYNC_TIMEOUT_US && !copying) || waited >= SYNC_TIMEOUT_US + VRAM_COPY_GRACE_US {
            header_set(header!(sync_timeouts), header_get(header!(sync_timeouts)).wrapping_add(1));
            stall.timed_out(header_get(header!(write_count)));
            break;
        }
    }
    let waited = now_us().wrapping_sub(start);
    header_set(header!(sync_waits), header_get(header!(sync_waits)).wrapping_add(1));
    header_set(header!(sync_wait_us), header_get(header!(sync_wait_us)).wrapping_add(waited));
    waited
}

/// The GE state block (`EVENT_GE_STATE`), cache-line aligned so the
/// write-back covers only it.
#[repr(C, align(64))]
struct GeStateBlock([u32; GE_STATE_WORDS]);

/// Written only by `answer_state_request` (game threads at their GE syncs,
/// one at a time: interrupts stay on, but a request is answered once).
static GE_STATE: Global<GeStateBlock> = Global::new(GeStateBlock([0; GE_STATE_WORDS]));

/// sceGeGetMtx's matrix numbers: bones 0..7, then world, view, projection,
/// texture generation (pspge.h PSP_GE_MTX_*), with where each goes in the
/// block.
const GE_MATRICES: [(i32, usize); 12] = [
    (0, GE_STATE_BONES),
    (1, GE_STATE_BONES + 12),
    (2, GE_STATE_BONES + 24),
    (3, GE_STATE_BONES + 36),
    (4, GE_STATE_BONES + 48),
    (5, GE_STATE_BONES + 60),
    (6, GE_STATE_BONES + 72),
    (7, GE_STATE_BONES + 84),
    (8, GE_STATE_WORLD),
    (9, GE_STATE_VIEW),
    (10, GE_STATE_PROJECTION),
    (11, GE_STATE_TEXGEN),
];

/// If the Vita side asked for the GE's state (`vita_state_request` differs
/// from `state_answered`): reads every register (sceGeGetCmd) and matrix
/// (sceGeGetMtx) into `GE_STATE` and sends `EVENT_GE_STATE`. Called after
/// the game's GE sync, so the GE is idle and its registers are those the
/// lists so far left. The Vita side asks after a new session: a loaded
/// savestate brings back the GE registers of its moment, which the game
/// never sends again.
fn answer_state_request() {
    let request = header_get(header!(vita_state_request));
    if request == header_get(header!(state_answered)) {
        return;
    }
    header_set(header!(state_answered), request);
    let start = now_us();
    let (get_cmd, get_mtx) = (original(&GE_GET_CMD), original(&GE_GET_MTX));
    // Safe: only this function writes the block, once per request.
    let block = unsafe { &mut GE_STATE.get().0 };
    let mut result = 0u32;
    if get_cmd == 0 || get_mtx == 0 {
        result = 0x8000_0001; // not found: the Vita side keeps its registers
    } else {
        // Safe: the kernel functions' addresses, of these types.
        let (get_cmd, get_mtx) =
            unsafe { (core::mem::transmute::<usize, GetCmdFn>(get_cmd), core::mem::transmute::<usize, GetMtxFn>(get_mtx)) };
        // In the game's system call k1 says "user-mode caller": the GE
        // driver would refuse our kernel buffer.
        let k1 = set_k1(0);
        // Raw pointers rather than slices: the plugin may link no panic
        // path (bounds checks).
        let words = block.as_mut_ptr();
        for cmd in 0..256 {
            // Safe: a plain kernel call with a valid register number; the
            // register part of the block has 256 words.
            unsafe { *words.add(GE_STATE_REGISTERS + cmd) = get_cmd(cmd as i32) };
        }
        for (kind, at) in GE_MATRICES {
            // Safe: the block has room at `at` for the matrix (16 words for
            // the projection, 12 for the others; PPSSPP sceGeGetMtx).
            let error = unsafe { get_mtx(kind, words.add(at)) };
            if error < 0 && result == 0 {
                result = error as u32;
            }
        }
        set_k1(k1);
    }
    // Safe: the block, in kernel RAM the Vita side reads directly.
    unsafe { sceKernelDcacheWritebackRange(block.as_ptr().cast(), (GE_STATE_WORDS * 4) as u32) };
    push_event(EVENT_GE_STATE, [block.as_ptr() as u32, request, result, now_us().wrapping_sub(start)]);
}

/// `kind` 1 = sceGeDrawSync, 2 = sceGeListSync (`queue`); `ge_us`: how
/// long Sony's sync took (the game waiting for the GE).
fn after_ge_sync(kind: u32, queue: i32, ge_us: u32) {
    answer_state_request();
    let waited = wait_for_vita();
    copy::probe_ge_at_sync();
    if header_get(header!(vita_sync_events)) == SYNC_WANTED {
        push_event(EVENT_GE_SYNC, [kind, queue as u32, ge_us, waited]);
    }
    copy::check_at_sync();
}

unsafe extern "C" fn ge_draw_sync_hook(mode: i32) -> i32 {
    let original: DrawSyncFn = core::mem::transmute(original(&ORIGINAL_DRAW_SYNC));
    let start = now_us();
    let result = original(mode);
    if mode == 0 {
        // Waited until the GE was done.
        after_ge_sync(1, 0, now_us().wrapping_sub(start));
    }
    result
}

unsafe extern "C" fn ge_list_sync_hook(queue: i32, mode: i32) -> i32 {
    let original: ListSyncFn = core::mem::transmute(original(&ORIGINAL_LIST_SYNC));
    let start = now_us();
    let result = original(queue, mode);
    if mode == 0 {
        after_ge_sync(2, queue, now_us().wrapping_sub(start));
    }
    result
}

/// The copy thread's wake-up semaphore (max count 1: flips while it copies
/// merge); -1 = no copy thread, the flip hook copies itself.
pub static COPY_WAKEUP: AtomicI32 = AtomicI32::new(-1);
/// The GE was synced since the last flip (games that never sync are
/// checked at the flip).
pub static SYNCED_SINCE_FLIP: AtomicBool = AtomicBool::new(false);
/// Latched by the flip hook for the copy thread.
pub static CHECK_AT_FLIP: AtomicBool = AtomicBool::new(false);

unsafe extern "C" fn display_set_frame_buf_hook(address: u32, width: i32, format: i32, sync: i32) -> i32 {
    let original: FrameBufFn = core::mem::transmute(original(&ORIGINAL_SET_FRAME_BUF));
    CHECK_AT_FLIP.store(!SYNCED_SINCE_FLIP.load(Relaxed), Relaxed);
    SYNCED_SINCE_FLIP.store(false, Relaxed);
    if !wake_copier() {
        copy::copy_requested_vram();
    }
    push_event(EVENT_FRAMEBUFFER_FLIP, [address, width as u32, format as u32, sync as u32]);
    original(address, width, format, sync)
}

/// Wakes the copy thread; false if there is none.
fn wake_copier() -> bool {
    let wakeup = COPY_WAKEUP.load(Relaxed);
    if wakeup < 0 {
        return false;
    }
    // We run in the game's system call, where k1 still says "user-mode
    // caller": the kernel would refuse our (kernel) semaphore.
    let k1 = set_k1(0);
    // Safe: our semaphore; fails harmlessly if already signalled.
    unsafe { sceKernelSignalSema(wakeup, 1) };
    set_k1(k1);
    true
}

/// Finds a kernel function by NID and routes its system call to `hook`;
/// the original goes to `slot` first.
fn hook_syscall(module: &[u8], library: &[u8], nid: u32, hook: usize, slot: &AtomicUsize) {
    // Safe: NUL-terminated names; patching a found function's system call
    // is what systemctrl offers this for.
    unsafe {
        let original = sctrlHENFindFunction(module.as_ptr(), library.as_ptr(), nid);
        if original != 0 {
            slot.store(original as usize, Relaxed);
            sctrlHENPatchSyscall(original, hook as *const c_void);
        }
    }
}

fn init_ring() {
    header_set(header!(magic), 0); // the reader ignores the ring meanwhile
    header_set(header!(version), VERSION);
    header_set(header!(capacity), CAPACITY);
    header_set(header!(session), now_us());
    header_set(header!(write_count), 0);
    header_set(header!(vita_sync_wanted), 0); // until the Vita side asks again
    header_set(header!(vita_sync_events), 0);
    header_set(header!(vita_state_request), 0);
    header_set(header!(state_answered), 0);
    header_set(header!(plugin_features), 0); // until the copy thread runs
    header_set(header!(sync_waits), 0);
    header_set(header!(sync_wait_us), 0);
    header_set(header!(sync_timeouts), 0);

    set!(range_count, 0);
    set!(copied_generation, get!(generation)); // nothing pending
    set!(last_copy_us, 0);
    set!(copy_method, copy_method::NONE);
    set!(copy_thread_runs, 0);
    set!(copy_thread_error, 0);
    set!(changes, 0);
    set!(sync_checks, 0);
    set!(last_sync_check_us, 0);

    header_set(header!(magic), MAGIC);
}

#[no_mangle]
pub unsafe extern "C" fn module_start(_args: u32, _argp: *mut c_void) -> i32 {
    init_ring();
    copy::start_thread();
    if COPY_WAKEUP.load(Relaxed) >= 0 {
        header_set(header!(plugin_features), COPIES_AT_SYNC);
    }

    const GE: &[u8] = b"sceGE_Manager\0";
    const GE_DRIVER: &[u8] = b"sceGe_driver\0";
    hook_syscall(GE, GE_DRIVER, NID_GE_LIST_ENQUEUE, ge_list_enqueue_hook as EnqueueFn as usize, &ORIGINAL_ENQUEUE);
    hook_syscall(GE, GE_DRIVER, NID_GE_LIST_ENQUEUE_HEAD, ge_list_enqueue_head_hook as EnqueueFn as usize, &ORIGINAL_ENQUEUE_HEAD);
    hook_syscall(GE, GE_DRIVER, NID_GE_LIST_UPDATE_STALL_ADDR, ge_list_update_stall_addr_hook as StallFn as usize, &ORIGINAL_UPDATE_STALL);
    hook_syscall(
        b"sceDisplay_Service\0",
        b"sceDisplay_driver\0",
        NID_DISPLAY_SET_FRAME_BUF,
        display_set_frame_buf_hook as FrameBufFn as usize,
        &ORIGINAL_SET_FRAME_BUF,
    );
    hook_syscall(GE, GE_DRIVER, NID_GE_DRAW_SYNC, ge_draw_sync_hook as DrawSyncFn as usize, &ORIGINAL_DRAW_SYNC);
    hook_syscall(GE, GE_DRIVER, NID_GE_LIST_SYNC, ge_list_sync_hook as ListSyncFn as usize, &ORIGINAL_LIST_SYNC);
    hook_syscall(GE, GE_DRIVER, NID_GE_SAVE_CONTEXT, ge_save_context_hook as ContextFn as usize, &ORIGINAL_SAVE_CONTEXT);
    hook_syscall(GE, GE_DRIVER, NID_GE_RESTORE_CONTEXT, ge_restore_context_hook as ContextFn as usize, &ORIGINAL_RESTORE_CONTEXT);
    GE_GET_CMD.store(sctrlHENFindFunction(GE.as_ptr(), GE_DRIVER.as_ptr(), NID_GE_GET_CMD) as usize, Relaxed);
    GE_GET_MTX.store(sctrlHENFindFunction(GE.as_ptr(), GE_DRIVER.as_ptr(), NID_GE_GET_MTX) as usize, Relaxed);

    sceKernelDcacheWritebackInvalidateAll();
    sceKernelIcacheClearAll();
    0
}

#[no_mangle]
pub unsafe extern "C" fn module_stop(_args: u32, _argp: *mut c_void) -> i32 {
    0
}

/// Nothing in the plugin may panic (scripts/build-psp-plugin.sh checks that
/// no panic path is linked); if one did, the game's thread stops here with
/// a mark the Vita side shows in the capture log (`thread=.../bad0ba1c`).
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    set!(copy_thread_error, 0xBAD0_BA1C);
    loop {
        // Safe: a plain kernel call.
        unsafe { sceKernelDelayThread(1_000_000) };
    }
}
