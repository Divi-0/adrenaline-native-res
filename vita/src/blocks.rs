//! Our memory blocks, all allocated in one place.
//!
//! Main memory is short inside Adrenaline: the PSP emulator and Adrenaline
//! itself (32 MiB for savestates alone) take most of the app's share, and
//! our frames, render targets and textures take most of the rest. When it
//! runs out, Sony's code may crash instead of failing politely (suspected
//! for C2-12828-1, 2026-11-05). So:
//!
//! - Every block we allocate goes through `alloc_block` /
//!   `free_block`, which keep a table of the live ones (`report`, with the
//!   free memory), so a live run shows exactly where the memory goes.
//! - Each allocation site states its size in the build as a constant named
//!   `nativeres_plan_<kind>_<name>` (a `#[no_mangle]` static, so it keeps
//!   that name in the ELF), so a tool can read them from the ELF and check
//!   that the build fits the memory measured on the Vita.

use core::cell::UnsafeCell;
use core::ffi::c_void;
use core::fmt::Write;
use core::sync::atomic::{AtomicBool, Ordering};

use nativeres_core::report::TextBuffer;

use crate::sys::{self, SceUID};

/// Enough for the texture cache (512 cached + 512 retired) and the rest.
const MAX_BLOCKS: usize = 2048;
/// Blocks with the same name are summed up in "memory".
const MAX_NAMES: usize = 48;

/// Which pool a block comes from; each has its own free memory.
#[derive(Clone, Copy, PartialEq)]
enum Pool {
    Main = 0,
    Cdram = 1,
    Phycont = 2,
}

fn pool_of(block_type: u32) -> Pool {
    match block_type {
        sys::SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RW => Pool::Cdram,
        sys::SCE_KERNEL_MEMBLOCK_TYPE_USER_MAIN_PHYCONT_RW | sys::SCE_KERNEL_MEMBLOCK_TYPE_USER_MAIN_PHYCONT_NC_RW => Pool::Phycont,
        _ => Pool::Main,
    }
}

/// One live block.
#[derive(Clone, Copy)]
struct Block {
    /// 0 = free slot.
    uid: SceUID,
    /// Rounded up to whole 4 KiB pages, as the kernel allocates.
    size: u32,
    /// Index into `Table::names`.
    name: u16,
    pool: Pool,
}

/// The blocks that share one name, summed up.
#[derive(Clone, Copy)]
struct Name {
    /// A NUL-terminated string that lives as long as the program (the
    /// callers pass literals).
    name: *const u8,
    count: u32,
    bytes: u32,
    peak_bytes: u32,
}

struct Table {
    blocks: [Block; MAX_BLOCKS],
    names: [Name; MAX_NAMES],
    name_count: usize,
    /// Bytes in live blocks per `Pool`.
    pool_bytes: [u32; 3],
    peak_main_bytes: u32,
    /// Allocations the kernel refused.
    failures: u32,
    last_failure_name: *const u8,
    last_failure_size: u32,
    /// Allocated, but the table was full.
    untracked: u32,
}

impl Table {
    const fn new() -> Self {
        Table {
            blocks: [Block { uid: 0, size: 0, name: 0, pool: Pool::Main }; MAX_BLOCKS],
            names: [Name { name: core::ptr::null(), count: 0, bytes: 0, peak_bytes: 0 }; MAX_NAMES],
            name_count: 0,
            pool_bytes: [0; 3],
            peak_main_bytes: 0,
            failures: 0,
            last_failure_name: core::ptr::null(),
            last_failure_size: 0,
            untracked: 0,
        }
    }

    /// The index of `name` in `names`, added if it is new. When the table
    /// of names is full, the last one collects the rest.
    ///
    /// # Safety
    /// `name` must be a NUL-terminated string that is never freed.
    unsafe fn name_index(&mut self, name: *const u8) -> usize {
        for index in 0..self.name_count {
            let known = self.names[index].name;
            if known == name || c_str(known) == c_str(name) {
                return index;
            }
        }
        if self.name_count == MAX_NAMES {
            return MAX_NAMES - 1;
        }
        self.names[self.name_count].name = name;
        self.name_count += 1;
        self.name_count - 1
    }
}

/// Allocations come from several threads (display, capture, others)
/// and are rare: a spin lock is enough.
pub struct SpinLocked<T> {
    locked: AtomicBool,
    value: UnsafeCell<T>,
}

// Safe: `with` lets one thread at a time reach the value.
unsafe impl<T> Sync for SpinLocked<T> {}

impl<T> SpinLocked<T> {
    pub const fn new(value: T) -> Self {
        SpinLocked { locked: AtomicBool::new(false), value: UnsafeCell::new(value) }
    }

    pub fn with<R>(&self, work: impl FnOnce(&mut T) -> R) -> R {
        while self.locked.swap(true, Ordering::Acquire) {
            core::hint::spin_loop();
        }
        // Safe: we hold the lock.
        let result = work(unsafe { &mut *self.value.get() });
        self.locked.store(false, Ordering::Release);
        result
    }
}

static TABLE: SpinLocked<Table> = SpinLocked::new(Table::new());

/// The text of a NUL-terminated string ("-" for null).
///
/// # Safety
/// `text` must be null or NUL-terminated and stay valid.
unsafe fn c_str<'a>(text: *const u8) -> &'a str {
    if text.is_null() {
        return "-";
    }
    let mut length = 0;
    while *text.add(length) != 0 {
        length += 1;
    }
    // The names are ASCII literals.
    core::str::from_utf8_unchecked(core::slice::from_raw_parts(text, length))
}

/// `sceKernelAllocMemBlock` that also notes the block in the table.
///
/// # Safety
/// `name` must be a NUL-terminated string that is never freed (a literal);
/// `options` as for sceKernelAllocMemBlock.
pub unsafe fn alloc_block(name: *const u8, block_type: u32, size: u32, options: *mut c_void) -> SceUID {
    let uid = sys::sceKernelAllocMemBlock(name, block_type, size, options);
    TABLE.with(|table| {
        if uid < 0 {
            table.failures += 1;
            table.last_failure_name = name;
            table.last_failure_size = size;
            return;
        }
        let rounded = sys::pages(size as usize);
        let Some(slot) = table.blocks.iter().position(|block| block.uid == 0) else {
            table.untracked += 1;
            return;
        };
        let name_index = table.name_index(name);
        let pool = pool_of(block_type);
        table.blocks[slot] = Block { uid, size: rounded, name: name_index as u16, pool };
        let entry = &mut table.names[name_index];
        entry.count += 1;
        entry.bytes += rounded;
        entry.peak_bytes = entry.peak_bytes.max(entry.bytes);
        table.pool_bytes[pool as usize] += rounded;
        table.peak_main_bytes = table.peak_main_bytes.max(table.pool_bytes[Pool::Main as usize]);
    });
    uid
}

/// `sceKernelFreeMemBlock` for a block made by `alloc_block`.
///
/// # Safety
/// Nothing may use the block's memory any more.
pub unsafe fn free_block(uid: SceUID) -> i32 {
    TABLE.with(|table| {
        if let Some(slot) = table.blocks.iter().position(|block| block.uid == uid) {
            let block = table.blocks[slot];
            let entry = &mut table.names[block.name as usize];
            entry.count -= 1;
            entry.bytes -= block.size;
            table.pool_bytes[block.pool as usize] -= block.size;
            table.blocks[slot].uid = 0;
        }
    });
    sys::sceKernelFreeMemBlock(uid)
}

fn free_memory() -> sys::SceKernelFreeMemorySizeInfo {
    let mut info = sys::SceKernelFreeMemorySizeInfo {
        size: core::mem::size_of::<sys::SceKernelFreeMemorySizeInfo>() as i32,
        size_user: 0,
        size_cdram: 0,
        size_phycont: 0,
    };
    // Safe: `info` is a valid, sized struct.
    unsafe { sys::sceKernelGetFreeMemorySize(&mut info) };
    info
}

/// One line: the totals (ours = our blocks in main memory, the pool the
/// memory budget is about; CDRAM and phycont separately), then
/// "block=<name>/<count>/<bytes>/<peak bytes>" per block name. Sizes in
/// bytes, for exact sums. NUL-terminated; returns
/// the text's length.
///
/// # Safety
/// `out` must hold `size` bytes.
pub unsafe fn memory_info(out: *mut u8, size: i32) -> i32 {
    if out.is_null() || size <= 0 {
        return 0;
    }
    let bytes = core::slice::from_raw_parts_mut(out, size as usize);
    let (text_bytes, terminator) = bytes.split_at_mut(size as usize - 1);
    let mut text = TextBuffer::new(text_bytes);
    let free = free_memory();
    TABLE.with(|table| {
        // Writing to a TextBuffer can't fail (it cuts off what doesn't fit).
        let _ = write!(
            text,
            "min_free_user={}@{} free_user={} free_cdram={} free_phycont={} ours={} ours_peak={} ours_cdram={} ours_phycont={} failures={} last_failure={}/{} untracked={}",
            MIN_FREE_USER.load(core::sync::atomic::Ordering::Relaxed),
            MIN_FREE_USER_MS.load(core::sync::atomic::Ordering::Relaxed),
            free.size_user,
            free.size_cdram,
            free.size_phycont,
            table.pool_bytes[Pool::Main as usize],
            table.peak_main_bytes,
            table.pool_bytes[Pool::Cdram as usize],
            table.pool_bytes[Pool::Phycont as usize],
            table.failures,
            c_str(table.last_failure_name),
            table.last_failure_size,
            table.untracked,
        );
        for entry in &table.names[..table.name_count] {
            let _ = write!(text, " block={}/{}/{}/{}", c_str(entry.name), entry.count, entry.bytes, entry.peak_bytes);
        }
    });
    let length = text.len();
    terminator[0] = 0;
    if length < text_bytes.len() {
        text_bytes[length] = 0;
    }
    length as i32
}

/// Free main memory in bytes (for the capture log).
pub fn free_user_memory() -> i32 {
    free_memory().size_user
}

/// The least free main memory seen (every 50 ms by the options thread,
/// from Adrenaline's start), and when (process time, ms): "memory" reports
/// it as min_free_user=<bytes>@<ms>. Diagnosis of the C1-2858-3 crashes at
/// Adrenaline's start (2026-11-14).
static MIN_FREE_USER: core::sync::atomic::AtomicI32 = core::sync::atomic::AtomicI32::new(i32::MAX);
static MIN_FREE_USER_MS: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);

pub fn note_free_user_memory() {
    use core::sync::atomic::Ordering::Relaxed;
    let free = free_user_memory();
    if free < MIN_FREE_USER.load(Relaxed) {
        MIN_FREE_USER.store(free, Relaxed);
        MIN_FREE_USER_MS.store((sys::now_us() / 1000) as u32, Relaxed);
    }
}

/// Bytes in our main-memory blocks.
pub fn our_memory() -> u32 {
    TABLE.with(|table| table.pool_bytes[Pool::Main as usize])
}
