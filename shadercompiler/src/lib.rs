//! NativePSP Shader Compiler: a small Vita app (development only), the Rust
//! version of the C app that was tools/shadercompiler/main.c (DESIGN.md
//! "All Rust, one plugin", stage 5).
//!
//! Compiles every .cg file in ux0:data/nativepsp/shaders/ with Sony's
//! runtime Cg compiler (ur0:data/libshacccg.suprx, extracted by tools like
//! ShaRKBR33D) and writes <name>.gxp plus <name>.gxp.txt (the compiler's
//! messages) next to it. Files ending in _v.cg are vertex programs, all
//! others fragment programs. When done it writes done.txt and quits.
//!
//! done.txt has one line per file, `<name>.cg: OK <bytes> (<ms> ms)` or
//! `<name>.cg: ERR <why>`, then a line starting with `#` (the heap's
//! numbers and the total time). tools/vita.py shaders uploads the sources,
//! starts this app over vitacompanion, waits for done.txt and downloads the
//! .gxp files into renderer/vita/shaders/, which the plugin embeds. Users of
//! the plugin never need libshacccg.
//!
//! It is a separate app because loading libshacccg inside Adrenaline's PSP
//! emulator process failed (0x8002D0F3, a missing preload) and once crashed
//! Adrenaline. An app gets the system's preloaded modules libshacccg needs.
//!
//! Built by tools/build-shadercompiler.sh (cargo, the SDK's linker and
//! tools, no C library): the compiler's malloc/free are ours
//! (`nativepsp_core::class_heap` over one 32 MiB block, as big as the C
//! app's newlib heap).

#![no_std]
#![allow(non_snake_case, non_upper_case_globals)]

use core::cell::UnsafeCell;
use core::ffi::c_void;
use core::fmt::Write;
use core::panic::PanicInfo;
use core::sync::atomic::{AtomicBool, AtomicI32, Ordering};

use nativepsp_core::class_heap::ClassHeap;

type SceUID = i32;

const SHADER_DIR: &str = "ux0:data/nativepsp/shaders";
const SHACCCG_PATH: &[u8] = b"ur0:data/libshacccg.suprx\0";
const MAX_SOURCE: usize = 60 * 1024;
/// The compiler's memory (the C app: `_newlib_heap_size_user`).
const HEAP_BYTES: u32 = 32 * 1024 * 1024;

const SCE_O_RDONLY: i32 = 0x0001;
const SCE_O_WRONLY: i32 = 0x0002;
const SCE_O_CREAT: i32 = 0x0200;
const SCE_O_TRUNC: i32 = 0x0400;
const SCE_SEEK_SET: i32 = 0;
const SCE_SEEK_END: i32 = 2;
const SCE_KERNEL_MEMBLOCK_TYPE_USER_RW: u32 = 0x0C20_D060;

/// The main thread's stack (read by vita-elf-create into the process
/// parameters). The C app had the default (256 KiB); the compiler is deep
/// code, and the room costs nothing here.
#[no_mangle]
#[used]
pub static sceUserMainThreadStackSize: u32 = 1024 * 1024;

// ---- Sony's runtime Cg compiler (psp2/shacccg.h) ----

const SCE_SHACCCG_PROFILE_VP: u32 = 0;
const SCE_SHACCCG_PROFILE_FP: u32 = 1;
const SCE_SHACCCG_TRIVIAL: u32 = 1;
const SCE_SHACCCG_ENGLISH: u32 = 0;
const SCE_SHACCCG_DIAGNOSTIC_LEVEL_WARNING: u32 = 1;
const SCE_SHACCCG_DIAGNOSTIC_LEVEL_ERROR: u32 = 2;

#[repr(C)]
struct SceShaccCgSourceFile {
    file_name: *const u8,
    text: *const u8,
    size: u32,
}

#[repr(C)]
struct SceShaccCgSourceLocation {
    file: *const SceShaccCgSourceFile,
    line_number: u32,
    column_number: u32,
}

type OpenFile = unsafe extern "C" fn(*const u8, *const SceShaccCgSourceLocation, *const SceShaccCgCompileOptions, *mut *const u8) -> *mut SceShaccCgSourceFile;
type ReleaseFile = unsafe extern "C" fn(*const SceShaccCgSourceFile, *const SceShaccCgCompileOptions);

#[repr(C)]
struct SceShaccCgCallbackList {
    open_file: Option<OpenFile>,
    release_file: Option<ReleaseFile>,
    locate_file: *const c_void,
    absolute_path: *const c_void,
    release_file_name: *const c_void,
    file_date: *const c_void,
}

#[repr(C)]
struct SceShaccCgCompileOptions {
    main_source_file: *const u8,
    target_profile: u32,
    entry_function_name: *const u8,
    search_path_count: u32,
    search_paths: *const c_void,
    macro_definition_count: u32,
    macro_definitions: *const c_void,
    include_file_count: u32,
    include_files: *const c_void,
    suppressed_warnings_count: u32,
    suppressed_warnings: *const c_void,
    locale: u32,
    use_fx: i32,
    no_stdlib: i32,
    optimization_level: i32,
    use_fastmath: i32,
    use_fastprecision: i32,
    use_fastint: i32,
    field_48: i32,
    warnings_as_errors: i32,
    performance_warnings: i32,
    warning_level: i32,
    pedantic: i32,
    pedantic_error: i32,
    field_60: i32,
    field_64: i32,
}

#[repr(C)]
struct SceShaccCgDiagnosticMessage {
    level: u32,
    code: u32,
    location: *const SceShaccCgSourceLocation,
    message: *const u8,
}

#[repr(C)]
struct SceShaccCgCompileOutput {
    program_data: *const u8,
    program_size: u32,
    diagnostic_count: i32,
    diagnostics: *const SceShaccCgDiagnosticMessage,
}

const _: () = assert!(core::mem::size_of::<SceShaccCgSourceFile>() == 0xC);
const _: () = assert!(core::mem::size_of::<SceShaccCgSourceLocation>() == 0xC);
const _: () = assert!(core::mem::size_of::<SceShaccCgCallbackList>() == 0x18);
const _: () = assert!(core::mem::size_of::<SceShaccCgCompileOptions>() == 0x68);
const _: () = assert!(core::mem::size_of::<SceShaccCgDiagnosticMessage>() == 0x10);
const _: () = assert!(core::mem::size_of::<SceShaccCgCompileOutput>() == 0x10);

/// SceIoDirent (0x160 bytes): SceIoStat (0x58), d_name[256], d_private, a
/// spare word.
#[repr(C)]
struct SceIoDirent {
    stat: [u8; 0x58],
    name: [u8; 256],
    private: *mut c_void,
    dummy: i32,
}

const _: () = assert!(core::mem::size_of::<SceIoDirent>() == 0x160);

extern "C" {
    fn sceShaccCgInitializeCompileOptions(options: *mut SceShaccCgCompileOptions) -> i32;
    fn sceShaccCgInitializeCallbackList(callbacks: *mut SceShaccCgCallbackList, defaults: u32);
    fn sceShaccCgCompileProgram(options: *const SceShaccCgCompileOptions, callbacks: *const SceShaccCgCallbackList, unknown: i32) -> *const SceShaccCgCompileOutput;
    fn sceShaccCgDestroyCompileOutput(output: *const SceShaccCgCompileOutput);
    fn sceShaccCgSetDefaultAllocator(malloc: unsafe extern "C" fn(u32) -> *mut c_void, free: unsafe extern "C" fn(*mut c_void)) -> i32;
    fn sceShaccCgReleaseCompiler();

    fn sceKernelLoadStartModule(path: *const u8, args: u32, argp: *mut c_void, flags: i32, option: *mut c_void, status: *mut i32) -> SceUID;
    fn sceKernelExitProcess(status: i32) -> i32;
    fn sceKernelAllocMemBlock(name: *const u8, block_type: u32, size: u32, options: *mut c_void) -> SceUID;
    fn sceKernelGetMemBlockBase(uid: SceUID, base: *mut *mut c_void) -> i32;
    fn sceKernelGetProcessTimeWide() -> u64;

    fn sceIoOpen(path: *const u8, flags: i32, mode: i32) -> SceUID;
    fn sceIoRead(fd: SceUID, data: *mut c_void, size: u32) -> i32;
    fn sceIoWrite(fd: SceUID, data: *const c_void, size: u32) -> i32;
    fn sceIoLseek32(fd: SceUID, offset: i32, whence: i32) -> i32;
    fn sceIoClose(fd: SceUID) -> i32;
    fn sceIoRemove(path: *const u8) -> i32;
    fn sceIoRename(old: *const u8, new: *const u8) -> i32;
    fn sceIoDopen(path: *const u8) -> SceUID;
    fn sceIoDread(fd: SceUID, entry: *mut SceIoDirent) -> i32;
    fn sceIoDclose(fd: SceUID) -> i32;
}

// ---- Small helpers ----

/// A static touched by one thread at a time (the app's main thread; the
/// heap under its own lock).
struct Shared<T>(UnsafeCell<T>);
// Safe: see above.
unsafe impl<T> Sync for Shared<T> {}

/// Text built in a fixed buffer, cut off when full; always NUL-terminated,
/// so it can be a path.
struct Text<const N: usize> {
    bytes: [u8; N],
    len: usize,
}

impl<const N: usize> Text<N> {
    fn new() -> Self {
        Text { bytes: [0; N], len: 0 }
    }

    fn of(args: core::fmt::Arguments) -> Self {
        let mut text = Self::new();
        let _ = text.write_fmt(args);
        text
    }

    fn as_bytes(&self) -> &[u8] {
        &self.bytes[..self.len]
    }

    /// The text with its NUL.
    fn c_str(&self) -> *const u8 {
        self.bytes.as_ptr()
    }
}

impl<const N: usize> Write for Text<N> {
    fn write_str(&mut self, text: &str) -> core::fmt::Result {
        let room = (N - 1).saturating_sub(self.len);
        let take = text.len().min(room);
        self.bytes[self.len..self.len + take].copy_from_slice(&text.as_bytes()[..take]);
        self.len += take;
        self.bytes[self.len] = 0;
        Ok(())
    }
}

/// The bytes before the first NUL.
fn until_nul(bytes: &[u8]) -> &[u8] {
    let end = bytes.iter().position(|&byte| byte == 0).unwrap_or(bytes.len());
    &bytes[..end]
}

/// A C string from the compiler as text ("?" for null).
///
/// # Safety
/// `text` must be null or point to a NUL-terminated string.
unsafe fn c_text<'a>(text: *const u8) -> &'a str {
    if text.is_null() {
        return "?";
    }
    let mut len = 0;
    while *text.add(len) != 0 {
        len += 1;
    }
    core::str::from_utf8(core::slice::from_raw_parts(text, len)).unwrap_or("(not UTF-8)")
}

fn write_all(fd: SceUID, bytes: &[u8]) -> bool {
    // Safe: `bytes` is readable for its length.
    fd >= 0 && unsafe { sceIoWrite(fd, bytes.as_ptr() as *const c_void, bytes.len() as u32) } == bytes.len() as i32
}

fn now_us() -> u64 {
    // Safe: no arguments.
    unsafe { sceKernelGetProcessTimeWide() }
}

/// done.tmp, written as we go (renamed to done.txt at the end).
static SUMMARY: AtomicI32 = AtomicI32::new(-1);

fn say(args: core::fmt::Arguments) {
    let text = Text::<512>::of(args);
    write_all(SUMMARY.load(Ordering::Relaxed), text.as_bytes());
}

// ---- The compiler's heap ----

struct HeapState {
    memory: *mut u32,
    heap: ClassHeap,
}

static HEAP: Shared<HeapState> = Shared(UnsafeCell::new(HeapState { memory: core::ptr::null_mut(), heap: ClassHeap::new(0) }));
static HEAP_LOCK: AtomicBool = AtomicBool::new(false);

/// Runs `body` on the heap under its lock (the compiler may call from any
/// thread it makes).
fn with_heap<R>(body: impl FnOnce(&mut HeapState) -> R) -> R {
    while HEAP_LOCK.compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed).is_err() {
        core::hint::spin_loop();
    }
    // Safe: we hold the lock.
    let result = body(unsafe { &mut *HEAP.0.get() });
    HEAP_LOCK.store(false, Ordering::Release);
    result
}

/// The block's words.
///
/// # Safety
/// `memory` must be the block of HEAP_BYTES (made before the compiler gets
/// the heap), used only under the lock.
unsafe fn words<'a>(memory: *mut u32) -> &'a mut [u32] {
    core::slice::from_raw_parts_mut(memory, (HEAP_BYTES / 4) as usize)
}

/// Makes the heap's block. False if there is no memory.
fn heap_init() -> bool {
    // Safe: plain system calls; the block is ours alone.
    let base = unsafe {
        let block = sceKernelAllocMemBlock(b"NativePspShaccHeap\0".as_ptr(), SCE_KERNEL_MEMBLOCK_TYPE_USER_RW, HEAP_BYTES, core::ptr::null_mut());
        let mut base: *mut c_void = core::ptr::null_mut();
        if block < 0 || sceKernelGetMemBlockBase(block, &mut base) < 0 {
            say(format_args!("ERR no memory for the compiler's heap: {:08x}\n", block as u32));
            return false;
        }
        base
    };
    with_heap(|state| {
        state.memory = base as *mut u32;
        state.heap = ClassHeap::new((HEAP_BYTES / 4) as usize);
    });
    true
}

unsafe extern "C" fn heap_malloc(size: u32) -> *mut c_void {
    with_heap(|state| {
        let memory = state.memory as *mut u8;
        // Safe: the block exists (the compiler gets the heap after it); we
        // hold the lock.
        match state.heap.allocate(unsafe { words(state.memory) }, size) {
            // Safe: the heap hands out offsets inside the block.
            Some(offset) => unsafe { memory.add(offset as usize) as *mut c_void },
            None => core::ptr::null_mut(),
        }
    })
}

unsafe extern "C" fn heap_free(pointer: *mut c_void) {
    if pointer.is_null() {
        return;
    }
    with_heap(|state| {
        let offset = (pointer as usize).wrapping_sub(state.memory as usize);
        if offset < HEAP_BYTES as usize {
            // Safe: as in heap_malloc.
            state.heap.free(unsafe { words(state.memory) }, offset as u32);
        }
    })
}

// ---- Compiling ----

static SOURCE_TEXT: Shared<[u8; MAX_SOURCE + 1]> = Shared(UnsafeCell::new([0; MAX_SOURCE + 1]));
static SOURCE_FILE: Shared<SceShaccCgSourceFile> = Shared(UnsafeCell::new(SceShaccCgSourceFile { file_name: core::ptr::null(), text: core::ptr::null(), size: 0 }));

/// The compiler asks for its one source file (no includes): the one read.
unsafe extern "C" fn open_source(
    _name: *const u8,
    _from: *const SceShaccCgSourceLocation,
    _options: *const SceShaccCgCompileOptions,
    _error: *mut *const u8,
) -> *mut SceShaccCgSourceFile {
    SOURCE_FILE.0.get()
}

unsafe extern "C" fn release_source(_file: *const SceShaccCgSourceFile, _options: *const SceShaccCgCompileOptions) {}

/// Reads `path` into SOURCE_TEXT (NUL after it): its size, or None if it
/// can't be read or is empty or bigger than MAX_SOURCE.
fn read_source(path: *const u8) -> Option<u32> {
    // Safe: plain file calls; SOURCE_TEXT is only touched by this thread
    // (the compiler reads it later, on this thread too).
    unsafe {
        let fd = sceIoOpen(path, SCE_O_RDONLY, 0);
        if fd < 0 {
            return None;
        }
        let size = sceIoLseek32(fd, 0, SCE_SEEK_END);
        sceIoLseek32(fd, 0, SCE_SEEK_SET);
        let text = &mut *SOURCE_TEXT.0.get();
        let ok = size > 0 && size as usize <= MAX_SOURCE && sceIoRead(fd, text.as_mut_ptr() as *mut c_void, size as u32) == size;
        sceIoClose(fd);
        if !ok {
            return None;
        }
        text[size as usize] = 0;
        Some(size as u32)
    }
}

/// True for a vertex program: the name without ".cg" ends in "_v".
fn is_vertex_program(name: &str) -> bool {
    name.strip_suffix(".cg").is_some_and(|stem| stem.ends_with("_v"))
}

/// Compiles one file; true on success.
fn compile(name: &str) -> bool {
    let started = now_us();
    let stem = &name[..name.len() - 3];
    let input = Text::<256>::of(format_args!("{SHADER_DIR}/{name}"));
    let output_path = Text::<256>::of(format_args!("{SHADER_DIR}/{stem}.gxp"));
    let log_path = Text::<256>::of(format_args!("{SHADER_DIR}/{stem}.gxp.txt"));

    let Some(size) = read_source(input.c_str()) else {
        say(format_args!("{name}: ERR cannot read (max {MAX_SOURCE} bytes)\n"));
        return false;
    };
    // Safe: the compiler reads these only during the call below.
    unsafe {
        *SOURCE_FILE.0.get() = SceShaccCgSourceFile { file_name: input.c_str(), text: (*SOURCE_TEXT.0.get()).as_ptr(), size };
    }

    // Safe: the structs are the header's layout (sizes asserted), filled
    // by the compiler's own initializers first; the output is read only
    // until it is destroyed.
    unsafe {
        let mut options: SceShaccCgCompileOptions = core::mem::zeroed();
        sceShaccCgInitializeCompileOptions(&mut options);
        options.main_source_file = input.c_str();
        options.target_profile = if is_vertex_program(name) { SCE_SHACCCG_PROFILE_VP } else { SCE_SHACCCG_PROFILE_FP };
        options.entry_function_name = b"main\0".as_ptr();
        options.locale = SCE_SHACCCG_ENGLISH;
        options.optimization_level = 3;
        options.use_fastmath = 1;
        options.use_fastprecision = 1;

        let mut callbacks: SceShaccCgCallbackList = core::mem::zeroed();
        sceShaccCgInitializeCallbackList(&mut callbacks, SCE_SHACCCG_TRIVIAL);
        callbacks.open_file = Some(open_source);
        callbacks.release_file = Some(release_source);

        let output = sceShaccCgCompileProgram(&options, &callbacks, 0);
        if output.is_null() {
            say(format_args!("{name}: ERR compiler returned nothing\n"));
            return false;
        }
        let output = &*output;
        let log = sceIoOpen(log_path.c_str(), SCE_O_WRONLY | SCE_O_CREAT | SCE_O_TRUNC, 0o666);
        let mut first_error: Option<(&str, u32)> = None;
        let count = output.diagnostic_count.max(0) as usize;
        let diagnostics: &[SceShaccCgDiagnosticMessage] =
            if output.diagnostics.is_null() || count == 0 { &[] } else { core::slice::from_raw_parts(output.diagnostics, count) };
        for diagnostic in diagnostics {
            let line_number = if diagnostic.location.is_null() { 0 } else { (*diagnostic.location).line_number };
            let message = c_text(diagnostic.message);
            let level = match diagnostic.level {
                SCE_SHACCCG_DIAGNOSTIC_LEVEL_ERROR => "error",
                SCE_SHACCCG_DIAGNOSTIC_LEVEL_WARNING => "warning",
                _ => "info",
            };
            let line = Text::<512>::of(format_args!("{level} line {line_number}: {message}\n"));
            write_all(log, line.as_bytes());
            if diagnostic.level == SCE_SHACCCG_DIAGNOSTIC_LEVEL_ERROR && first_error.is_none() {
                first_error = Some((message, line_number));
            }
        }
        if log >= 0 {
            sceIoClose(log);
        }
        let ok = if !output.program_data.is_null() && output.program_size != 0 {
            let program = core::slice::from_raw_parts(output.program_data, output.program_size as usize);
            let fd = sceIoOpen(output_path.c_str(), SCE_O_WRONLY | SCE_O_CREAT | SCE_O_TRUNC, 0o666);
            let written = write_all(fd, program);
            if fd >= 0 {
                sceIoClose(fd);
            }
            if written {
                let ms = (now_us() - started) / 1000;
                say(format_args!("{name}: OK {} ({ms} ms)\n", output.program_size));
            } else {
                say(format_args!("{name}: ERR cannot write the .gxp\n"));
            }
            written
        } else {
            let (message, line_number) = first_error.unwrap_or(("no program produced", 0));
            say(format_args!("{name}: ERR line {line_number}: {message}\n"));
            false
        };
        sceShaccCgDestroyCompileOutput(output);
        ok
    }
}

/// Compiles every .cg in SHADER_DIR.
fn compile_all() {
    let dir_path = Text::<64>::of(format_args!("{SHADER_DIR}"));
    // Safe: plain directory calls into our own entry.
    unsafe {
        let dir = sceIoDopen(dir_path.c_str());
        if dir < 0 {
            say(format_args!("ERR cannot open {SHADER_DIR}: {:08x}\n", dir as u32));
            return;
        }
        let mut entry: SceIoDirent = core::mem::zeroed();
        while sceIoDread(dir, &mut entry) > 0 {
            let name = until_nul(&entry.name);
            if let Ok(name) = core::str::from_utf8(name) {
                if name.len() > 3 && name.ends_with(".cg") {
                    compile(name);
                }
            }
        }
        sceIoDclose(dir);
    }
}

fn run() {
    let started = now_us();
    let done = Text::<64>::of(format_args!("{SHADER_DIR}/done.txt"));
    let temporary = Text::<64>::of(format_args!("{SHADER_DIR}/done.tmp"));
    // Safe: plain file calls.
    unsafe {
        sceIoRemove(done.c_str());
        SUMMARY.store(sceIoOpen(temporary.c_str(), SCE_O_WRONLY | SCE_O_CREAT | SCE_O_TRUNC, 0o666), Ordering::Relaxed);
    }

    // Safe: loads Sony's compiler; its imports of ours are bound then.
    let module = unsafe { sceKernelLoadStartModule(SHACCCG_PATH.as_ptr(), 0, core::ptr::null_mut(), 0, core::ptr::null_mut(), core::ptr::null_mut()) };
    if module < 0 {
        say(format_args!("ERR cannot load ur0:data/libshacccg.suprx: {:08x}\n", module as u32));
    } else if heap_init() {
        // Safe: the heap is ready; the compiler is loaded.
        unsafe { sceShaccCgSetDefaultAllocator(heap_malloc, heap_free) };
        compile_all();
        // Safe: the compiler is loaded.
        unsafe { sceShaccCgReleaseCompiler() };
        let (peak, top, total, live, failed, bad_frees) = with_heap(|state| {
            let heap = &state.heap;
            (heap.peak(), heap.top(), heap.total_allocations(), heap.allocations(), heap.failed(), heap.bad_frees())
        });
        say(format_args!(
            "# heap peak={peak} top={top} of {HEAP_BYTES} allocations={total} left={live} failed={failed} bad_frees={bad_frees}, {} ms\n",
            (now_us() - started) / 1000
        ));
    }
    finish();
}

/// Closes done.tmp and renames it, so done.txt only ever appears complete.
fn finish() {
    let done = Text::<64>::of(format_args!("{SHADER_DIR}/done.txt"));
    let temporary = Text::<64>::of(format_args!("{SHADER_DIR}/done.tmp"));
    let summary = SUMMARY.swap(-1, Ordering::Relaxed);
    // Safe: plain file calls.
    unsafe {
        if summary >= 0 {
            sceIoClose(summary);
        }
        sceIoRename(temporary.c_str(), done.c_str());
    }
}

/// The app's entry (the process's main thread).
#[no_mangle]
pub extern "C" fn module_start(_args: u32, _argp: *mut c_void) -> i32 {
    run();
    loop {
        // Safe: ends the app.
        unsafe { sceKernelExitProcess(0) };
    }
}

/// A bug of ours: say so in done.txt (tools/vita.py prints it and fails),
/// then quit.
#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    say(format_args!("PANIC: {info}\n"));
    finish();
    loop {
        // Safe: ends the app.
        unsafe { sceKernelExitProcess(1) };
    }
}
