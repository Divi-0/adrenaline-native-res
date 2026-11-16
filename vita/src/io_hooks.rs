//! AdrenalineUser's file calls on PSP memory go through a buffer of ours.
//!
//! The PSP's ms0: is served by AdrenalineUser's msfs.c (the
//! ScePspemuRemoteMsfs thread), which hands the kernel pointers straight
//! into PSP RAM (up to 1 MiB per call). Our renderer maps that same memory
//! for the GPU (zero copy, sceGxmMapMemory) while the GPU reads it, and on
//! 2026-11-09 ux0 file access wedged for every process, twice, during MTX
//! race soaks, with that thread stuck inside the file system. Since then no
//! file call reads or writes PSP memory itself: the data goes through
//! `BOUNCE`, copied by the CPU (the fork did this in msfs.c; the plugin does
//! it with hooks on AdrenalineUser's imports).
//!
//! - sceIoRead / sceIoWrite with a buffer inside the emulator's window: in
//!   pieces of `BOUNCE_BYTES` through the buffer;
//! - sceIoDread / sceIoGetstat / sceIoChstat with their struct inside it:
//!   the kernel fills (or reads) a copy of ours.
//!
//! sceIoWrite also writes Adrenaline's settings with the user's Graphics
//! Filtering (menu.rs `config_write`): one hook for both (a second hook of
//! ours on the same import was never tried).

use core::ffi::c_void;
use core::sync::atomic::{AtomicBool, Ordering};

use nativepsp_core::span;

use crate::sys::{self, SceUID};
use crate::taihen::Hook;
use crate::{menu, render_log, SingleThreaded};

const SCE_IOFILEMGR_NID: u32 = 0xF2FF_276E;
const SCE_IO_READ_NID: u32 = 0xFDB3_2293;
const SCE_IO_WRITE_NID: u32 = 0x34EF_D876;
const SCE_LIBKERNEL_NID: u32 = 0xCAE9_ACE6;
const SCE_IO_DREAD_NID: u32 = 0x9C8B_6624;
const SCE_IO_GETSTAT_NID: u32 = 0xBCA5_B623;
const SCE_IO_CHSTAT_NID: u32 = 0x2948_2F7F;

static IO_READ: Hook = Hook::new();
static IO_WRITE: Hook = Hook::new();
static IO_DREAD: Hook = Hook::new();
static IO_GETSTAT: Hook = Hook::new();
static IO_CHSTAT: Hook = Hook::new();

/// Where the emulator maps PSP memory (`sys::psp_to_vita`'s range).
const WINDOW_START: u32 = 0x6800_0000;
const WINDOW_END: u32 = 0x7800_0000;

/// True if any of `bytes` bytes at `address` lie in the window.
fn touches_window(address: *const c_void, bytes: u32) -> bool {
    span::touches(address as u32, bytes, WINDOW_START, WINDOW_END)
}

/// SceIoDirent is 0x160 bytes, SceIoStat 0x58: one buffer holds either.
const STRUCT_BYTES: usize = 0x160;
const BOUNCE_BYTES: usize = 32 * 1024;

#[repr(C, align(64))]
struct Buffers {
    data: [u8; BOUNCE_BYTES],
    record: [u8; STRUCT_BYTES],
}

static BUFFERS: SingleThreaded<Buffers> = SingleThreaded::new(Buffers { data: [0; BOUNCE_BYTES], record: [0; STRUCT_BYTES] });
static BUSY: AtomicBool = AtomicBool::new(false);

/// The buffers, for one caller at a time (normally only the msfs thread
/// comes here; another one waits).
struct Lock;

impl Lock {
    fn take() -> Lock {
        while BUSY.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_err() {
            // Safe: a plain system call.
            unsafe { sys::sceKernelDelayThread(100) };
        }
        Lock
    }

    fn buffers(&mut self) -> &mut Buffers {
        // Safe: only one Lock exists at a time.
        unsafe { BUFFERS.get() }
    }
}

impl Drop for Lock {
    fn drop(&mut self) {
        BUSY.store(false, Ordering::Release);
    }
}

type IoTransfer = unsafe extern "C" fn(SceUID, *mut c_void, u32) -> i32;

/// Reads or writes `size` bytes at `data` in pieces through the buffer, as
/// one call: the bytes done, or the first piece's error.
unsafe fn bounced(next: IoTransfer, fd: SceUID, data: *mut u8, size: u32, write: bool) -> i32 {
    let mut lock = Lock::take();
    let buffer = lock.buffers().data.as_mut_ptr();
    let mut done = 0u32;
    while done < size {
        let piece = (size - done).min(BOUNCE_BYTES as u32);
        if write {
            core::ptr::copy_nonoverlapping(data.add(done as usize), buffer, piece as usize);
        }
        let result = next(fd, buffer as *mut c_void, piece);
        if result < 0 {
            return if done == 0 { result } else { done as i32 };
        }
        let moved = (result as u32).min(piece);
        if !write {
            core::ptr::copy_nonoverlapping(buffer, data.add(done as usize), moved as usize);
        }
        done += moved;
        if moved < piece {
            break;
        }
    }
    done as i32
}

unsafe extern "C" fn io_read_hooked(fd: SceUID, data: *mut c_void, size: u32) -> i32 {
    let next: IoTransfer = core::mem::transmute(IO_READ.next());
    if touches_window(data, size) {
        return bounced(next, fd, data as *mut u8, size, false);
    }
    next(fd, data, size)
}

unsafe extern "C" fn io_write_hooked(fd: SceUID, data: *const c_void, size: u32) -> i32 {
    let next: IoTransfer = core::mem::transmute(IO_WRITE.next());
    if let Some(config) = menu::config_write(data, size) {
        return next(fd, config.as_ptr() as *mut c_void, size);
    }
    if touches_window(data, size) {
        return bounced(next, fd, data as *mut u8, size, true);
    }
    next(fd, data as *mut c_void, size)
}

/// Calls `call` with a copy of the `bytes` at `record` (copied back after).
unsafe fn with_record_copy(record: *mut c_void, bytes: usize, call: impl FnOnce(*mut c_void) -> i32) -> i32 {
    let mut lock = Lock::take();
    let copy = lock.buffers().record.as_mut_ptr();
    core::ptr::copy_nonoverlapping(record as *const u8, copy, bytes);
    let result = call(copy as *mut c_void);
    core::ptr::copy_nonoverlapping(copy, record as *mut u8, bytes);
    result
}

const DIRENT_BYTES: usize = 0x160;
const STAT_BYTES: usize = 0x58;

unsafe extern "C" fn io_dread_hooked(fd: SceUID, entry: *mut c_void) -> i32 {
    type IoDread = unsafe extern "C" fn(SceUID, *mut c_void) -> i32;
    let next: IoDread = core::mem::transmute(IO_DREAD.next());
    if !entry.is_null() && touches_window(entry, DIRENT_BYTES as u32) {
        return with_record_copy(entry, DIRENT_BYTES, |copy| next(fd, copy));
    }
    next(fd, entry)
}

unsafe extern "C" fn io_getstat_hooked(path: *const u8, stat: *mut c_void) -> i32 {
    type IoGetstat = unsafe extern "C" fn(*const u8, *mut c_void) -> i32;
    let next: IoGetstat = core::mem::transmute(IO_GETSTAT.next());
    if !stat.is_null() && touches_window(stat, STAT_BYTES as u32) {
        return with_record_copy(stat, STAT_BYTES, |copy| next(path, copy));
    }
    next(path, stat)
}

unsafe extern "C" fn io_chstat_hooked(path: *const u8, stat: *mut c_void, bits: u32) -> i32 {
    type IoChstat = unsafe extern "C" fn(*const u8, *mut c_void, u32) -> i32;
    let next: IoChstat = core::mem::transmute(IO_CHSTAT.next());
    if !stat.is_null() && touches_window(stat, STAT_BYTES as u32) {
        return with_record_copy(stat, STAT_BYTES, |copy| next(path, copy, bits));
    }
    next(path, stat, bits)
}

const _: () = assert!(DIRENT_BYTES <= STRUCT_BYTES && STAT_BYTES <= STRUCT_BYTES);

/// Sets the hooks (once; AdrenalineUser must be loaded) and logs their ids.
pub fn install() {
    const USER: &[u8] = b"AdrenalineUser\0";
    // Safe: each hook has the hooked function's signature and passes the
    // call on.
    let ids = unsafe {
        [
            ("sceIoRead", IO_READ.hook_import(USER, SCE_IOFILEMGR_NID, SCE_IO_READ_NID, io_read_hooked as *const c_void)),
            ("sceIoWrite", IO_WRITE.hook_import(USER, SCE_IOFILEMGR_NID, SCE_IO_WRITE_NID, io_write_hooked as *const c_void)),
            ("sceIoDread", IO_DREAD.hook_import(USER, SCE_LIBKERNEL_NID, SCE_IO_DREAD_NID, io_dread_hooked as *const c_void)),
            ("sceIoGetstat", IO_GETSTAT.hook_import(USER, SCE_LIBKERNEL_NID, SCE_IO_GETSTAT_NID, io_getstat_hooked as *const c_void)),
            ("sceIoChstat", IO_CHSTAT.hook_import(USER, SCE_LIBKERNEL_NID, SCE_IO_CHSTAT_NID, io_chstat_hooked as *const c_void)),
        ]
    };
    for (name, id) in ids {
        render_log::line(format_args!("hook {}: {:08x}\n", name, id));
    }
}
