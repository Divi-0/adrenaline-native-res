//! The Vita system functions we call, declared by hand (signatures from
//! vitasdk's psp2/ headers).

#![allow(non_snake_case, non_camel_case_types, dead_code)]

use core::ffi::c_void;

pub type SceUID = i32;
pub type ThreadEntry = unsafe extern "C" fn(args: u32, argp: *mut c_void) -> i32;

pub const SCE_KERNEL_MEMBLOCK_TYPE_USER_RW: u32 = 0x0C20_D060;
/// Main memory, uncached.
pub const SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE: u32 = 0x0C20_8060;
/// Graphics memory; the PSP emulator uses nearly all of it.
pub const SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RW: u32 = 0x0940_8060;
/// Physically contiguous main memory: a pool of its own (whole MiB).
pub const SCE_KERNEL_MEMBLOCK_TYPE_USER_MAIN_PHYCONT_RW: u32 = 0x0C80_D060;
/// The same, uncached.
pub const SCE_KERNEL_MEMBLOCK_TYPE_USER_MAIN_PHYCONT_NC_RW: u32 = 0x0D80_8060;
pub const SCE_O_WRONLY: i32 = 0x0002;
pub const SCE_O_CREAT: i32 = 0x0200;
pub const SCE_O_TRUNC: i32 = 0x0400;

/// SceKernelMemBlockInfo (0x18 bytes).
#[repr(C)]
pub struct SceKernelMemBlockInfo {
    pub size: u32,
    pub mapped_base: *mut c_void,
    pub mapped_size: u32,
    pub memory_type: i32,
    pub access: u32,
    pub block_type: u32,
}

const _: () = assert!(core::mem::size_of::<SceKernelMemBlockInfo>() == 0x18);

/// SceKernelFreeMemorySizeInfo (0x10 bytes).
#[repr(C)]
pub struct SceKernelFreeMemorySizeInfo {
    pub size: i32,
    pub size_user: i32,
    pub size_cdram: i32,
    pub size_phycont: i32,
}

/// SceKernelThreadInfo (0x80 bytes).
#[repr(C)]
pub struct SceKernelThreadInfo {
    pub size: u32,
    pub process_id: SceUID,
    pub name: [u8; 32],
    pub attr: u32,
    pub status: u32,
    pub entry: Option<ThreadEntry>,
    pub stack: *mut c_void,
    pub stack_size: i32,
    pub init_priority: i32,
    pub current_priority: i32,
    pub init_cpu_affinity_mask: i32,
    pub current_cpu_affinity_mask: i32,
    pub current_cpu_id: i32,
    pub last_executed_cpu_id: i32,
    pub wait_type: u32,
    pub wait_id: SceUID,
    pub exit_status: i32,
    /// CPU time the thread has run, in microseconds.
    pub run_clocks: u64,
    pub intr_preempt_count: u32,
    pub thread_preempt_count: u32,
    pub thread_release_count: u32,
    pub change_cpu_count: i32,
    pub f_notify_callback: i32,
    pub reserved: i32,
}

const _: () = assert!(core::mem::size_of::<SceKernelThreadInfo>() == 0x80);

pub const SCE_NET_AF_INET: i32 = 2;
pub const SCE_NET_SOCK_STREAM: i32 = 1;
pub const SCE_NET_SOL_SOCKET: i32 = 0xFFFF;
pub const SCE_NET_SO_SNDTIMEO: i32 = 0x1005;

/// SceNetSockaddrIn (16 bytes): an IPv4 address and port, both in network
/// byte order.
#[repr(C)]
pub struct SceNetSockaddrIn {
    pub sin_len: u8,
    pub sin_family: u8,
    pub sin_port: u16,
    pub sin_addr: u32,
    pub sin_vport: u16,
    pub sin_zero: [u8; 6],
}

const _: () = assert!(core::mem::size_of::<SceNetSockaddrIn>() == 16);

impl SceNetSockaddrIn {
    /// `port` on every address of the Vita (for a listening socket).
    pub fn any_address(port: u16) -> Self {
        SceNetSockaddrIn {
            sin_len: 0,
            sin_family: SCE_NET_AF_INET as u8,
            sin_port: port.to_be(),
            sin_addr: 0, // SCE_NET_INADDR_ANY
            sin_vport: 0,
            sin_zero: [0; 6],
        }
    }
}

pub const SCE_O_RDONLY: i32 = 0x0001;
pub const SCE_SEEK_SET: i32 = 0;
pub const SCE_SEEK_END: i32 = 2;

/// SceKernelSystemInfo (0x48 bytes): per core, the time it has idled.
#[repr(C)]
pub struct SceKernelSystemInfo {
    pub size: u32,
    pub active_cpu_mask: u32,
    pub cpu_info: [SceKernelCpuInfo; 4],
}

#[repr(C)]
pub struct SceKernelCpuInfo {
    /// Microseconds the core has idled.
    pub idle_clock: u64,
    pub comes_out_of_idle_count: u32,
    pub thread_switch_count: u32,
}

const _: () = assert!(core::mem::size_of::<SceKernelSystemInfo>() == 0x48);

/// SceNetInitParam (12 bytes): the network library's working memory.
#[repr(C)]
pub struct SceNetInitParam {
    pub memory: *mut c_void,
    pub size: i32,
    pub flags: i32,
}

/// SceSysmoduleModuleId of the network library.
pub const SCE_SYSMODULE_NET: u32 = 0x0001;

/// SceDisplayFrameBuf (0x18 bytes).
#[repr(C)]
pub struct SceDisplayFrameBuf {
    pub size: u32,
    pub base: *mut c_void,
    pub pitch: u32,
    pub pixel_format: u32,
    pub width: u32,
    pub height: u32,
}

const _: () = assert!(core::mem::size_of::<SceDisplayFrameBuf>() == 0x18);

// Ours, written in Rust; here so their callers can keep saying `sys::`.
pub use crate::blocks::{alloc_block, free_block};
pub use crate::ext::note_thread;

extern "C" {
    pub fn sceKernelAllocMemBlock(name: *const u8, block_type: u32, size: u32, options: *mut c_void) -> SceUID;
    pub fn sceKernelFreeMemBlock(uid: SceUID) -> i32;
    pub fn sceKernelGetMemBlockBase(uid: SceUID, base: *mut *mut c_void) -> i32;
    pub fn sceKernelGetMemBlockInfoByAddr(address: *mut c_void, info: *mut SceKernelMemBlockInfo) -> i32;
    pub fn sceKernelGetFreeMemorySize(info: *mut SceKernelFreeMemorySizeInfo) -> i32;

    pub fn sceKernelCreateSema(name: *const u8, attr: u32, initial: i32, max: i32, options: *mut c_void) -> SceUID;
    pub fn sceKernelWaitSema(sema: SceUID, signal: i32, timeout_us: *mut u32) -> i32;
    pub fn sceKernelSignalSema(sema: SceUID, signal: i32) -> i32;
    pub fn sceKernelDeleteSema(sema: SceUID) -> i32;

    pub fn sceKernelCreateThread(
        name: *const u8,
        entry: ThreadEntry,
        priority: i32,
        stack_size: u32,
        attr: u32,
        cpu_affinity_mask: i32,
        options: *mut c_void,
    ) -> SceUID;
    pub fn sceKernelStartThread(thread: SceUID, args: u32, argp: *mut c_void) -> i32;
    pub fn sceKernelGetThreadId() -> SceUID;
    pub fn sceKernelGetThreadInfo(thread: SceUID, info: *mut SceKernelThreadInfo) -> i32;
    pub fn sceKernelChangeThreadCpuAffinityMask(thread: SceUID, cpu_affinity_mask: i32) -> i32;
    pub fn sceKernelDelayThread(us: u32) -> i32;
    pub fn sceKernelExitDeleteThread(status: i32) -> i32;
    pub fn sceKernelWaitThreadEnd(thread: SceUID, status: *mut i32, timeout_us: *mut u32) -> i32;
    pub fn sceKernelExitThread(status: i32) -> i32;
    pub fn sceKernelDeleteThread(thread: SceUID) -> i32;
    pub fn sceKernelChangeThreadPriority(thread: SceUID, priority: i32) -> i32;
    pub fn sceKernelGetSystemInfo(info: *mut SceKernelSystemInfo) -> i32;
    pub fn sceKernelGetProcessTimeWide() -> u64;
    /// Resets the idle timers (0 = SCE_KERNEL_POWER_TICK_DEFAULT): no
    /// standby for now (debug "powertick").
    pub fn sceKernelPowerTick(tick_type: i32) -> i32;

    pub fn scePowerGetArmClockFrequency() -> i32;
    pub fn scePowerSetArmClockFrequency(mhz: i32) -> i32;
    pub fn scePowerGetBusClockFrequency() -> i32;
    pub fn scePowerGetGpuClockFrequency() -> i32;

    pub fn sceDisplayGetVcount() -> i32;

    pub fn sceIoOpen(path: *const u8, flags: i32, mode: i32) -> SceUID;
    pub fn sceIoRead(fd: SceUID, data: *mut c_void, size: u32) -> i32;
    pub fn sceIoLseek32(fd: SceUID, offset: i32, whence: i32) -> i32;
    pub fn sceIoWrite(fd: SceUID, data: *const c_void, size: u32) -> i32;
    pub fn sceIoClose(fd: SceUID) -> i32;
    pub fn sceIoMkdir(path: *const u8, mode: i32) -> i32;

    pub fn sceDmacMemcpy(target: *mut c_void, source: *const c_void, size: u32) -> i32;

    pub fn sceSysmoduleLoadModule(id: u32) -> i32;

    // SceNet (initialized by the debug server's thread)
    pub fn sceNetInit(param: *mut SceNetInitParam) -> i32;
    pub fn sceNetRecv(socket: i32, data: *mut c_void, size: u32, flags: i32) -> i32;
    pub fn sceNetSocket(name: *const u8, domain: i32, socket_type: i32, protocol: i32) -> i32;
    pub fn sceNetBind(socket: i32, address: *const SceNetSockaddrIn, address_size: u32) -> i32;
    pub fn sceNetListen(socket: i32, backlog: i32) -> i32;
    pub fn sceNetAccept(socket: i32, address: *mut SceNetSockaddrIn, address_size: *mut u32) -> i32;
    pub fn sceNetSend(socket: i32, data: *const c_void, size: u32, flags: i32) -> i32;
    pub fn sceNetSetsockopt(socket: i32, level: i32, option: i32, value: *const c_void, value_size: u32) -> i32;
    pub fn sceNetSocketClose(socket: i32) -> i32;
}

/// True once the loader has filled our (weak) imports of SceNet. The
/// plugin starts before anyone loads SceNet; an import stub not filled yet
/// is `sub r12, pc, #8; mov pc, #0` (a jump to 0), a filled one `movw/movt
/// r12; bx r12` (stage 3's proof, 2026-11-13).
pub fn net_imports_filled() -> bool {
    const UNFILLED_SECOND_WORD: u32 = 0xE3A0_F000; // mov pc, #0
    let stub = sceNetInit as *const u32;
    // Safe: the stub is three words of our own code segment (readable).
    unsafe { core::ptr::read_volatile(stub.add(1)) != UNFILLED_SECOND_WORD }
}

/// Microseconds since the process started.
pub fn now_us() -> u64 {
    // Safe: no arguments, no state.
    unsafe { sceKernelGetProcessTimeWide() }
}

/// Where PSP address `address` (any segment) is mapped on the Vita:
/// Adrenaline's CONVERT_ADDRESS.
pub fn psp_to_vita(address: u32) -> usize {
    0x6800_0000 + (address & 0x0FFF_FFFF) as usize
}

/// `bytes` rounded up to whole 4 KiB pages, as the kernel allocates them:
/// for the `nativepsp_plan_*` statics that tell tools/memcheck.py how much
/// memory an allocation site takes (see blocks.rs).
pub const fn pages(bytes: usize) -> u32 {
    ((bytes + 0xFFF) & !0xFFF) as u32
}
