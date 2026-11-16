//! The PSP side (psp/ -> nativepsp.prx, built by
//! scripts/build-psp-plugin.sh) is carried inside this plugin, so the user
//! installs one file. At Adrenaline's start, before its PSP side boots,
//! `install` puts it where Adrenaline's PSP side loads plugins from:
//! `<ms0>/seplugins/nativepsp.prx` (written only when it differs from the
//! file there) and its line in `<ms0>/seplugins/plugins.txt` (added only
//! when no line mentions the plugin: a line the user turned off stays off;
//! core `psp_copy::plugins_txt_with_ours`). <ms0> is Adrenaline's "Memory
//! Stick Location" setting.
//!
//! A new PSP plugin makes every savestate stale: they hold the PSP side's
//! memory, the old plugin's code included.

use core::ffi::c_void;

use nativepsp_core::psp_copy::plugins_txt_with_ours;

use crate::{adrenaline, render_log, sys, SingleThreaded};

static PRX: &[u8] = include_bytes!("../../target/nativepsp.prx");

/// plugins.txt is read and written whole; bigger ones are left alone.
const PLUGINS_TXT_MAX: usize = 8 * 1024;
const CHUNK: usize = 4 * 1024;

/// The start thread's buffers (its stack is 8 KiB): plugins.txt as read and
/// as written, a piece of the installed plugin.
struct Buffers {
    existing: [u8; PLUGINS_TXT_MAX],
    updated: [u8; PLUGINS_TXT_MAX + 256],
    chunk: [u8; CHUNK],
}

static BUFFERS: SingleThreaded<Buffers> =
    SingleThreaded::new(Buffers { existing: [0; PLUGINS_TXT_MAX], updated: [0; PLUGINS_TXT_MAX + 256], chunk: [0; CHUNK] });

/// "<ms0><suffix>", NUL-terminated.
fn path<'a>(buffer: &'a mut [u8; 64], suffix: &[u8]) -> &'a [u8] {
    let mut length = 0;
    for &byte in adrenaline::memory_stick().iter().chain(suffix) {
        if length < buffer.len() - 1 {
            buffer[length] = byte;
            length += 1;
        }
    }
    buffer[length] = 0;
    &buffer[..=length]
}

fn shown(path: &[u8]) -> &str {
    core::str::from_utf8(&path[..path.len() - 1]).unwrap_or("?")
}

/// What `read_file` found.
enum Read {
    /// The file's length (in the buffer).
    Bytes(usize),
    /// No such file (or it can't be read).
    Missing,
    /// Bigger than the buffer.
    TooBig,
}

/// Reads the whole file into `into`.
fn read_file(path: &[u8], into: &mut [u8]) -> Read {
    // Safe: a NUL-terminated path; reads into `into` within its length.
    unsafe {
        let fd = sys::sceIoOpen(path.as_ptr(), sys::SCE_O_RDONLY, 0);
        if fd < 0 {
            return Read::Missing;
        }
        let mut length = 0;
        let result = loop {
            if length == into.len() {
                // Full: the file is bigger if anything is left.
                let mut probe = [0u8; 1];
                break if sys::sceIoRead(fd, probe.as_mut_ptr().cast(), 1) == 0 { Read::Bytes(length) } else { Read::TooBig };
            }
            let read = sys::sceIoRead(fd, into[length..].as_mut_ptr().cast(), (into.len() - length) as u32);
            if read < 0 {
                break Read::Missing;
            }
            if read == 0 {
                break Read::Bytes(length);
            }
            length += read as usize;
        };
        sys::sceIoClose(fd);
        result
    }
}

/// Whether the file holds exactly `bytes`.
fn file_equals(path: &[u8], bytes: &[u8], chunk: &mut [u8; CHUNK]) -> bool {
    // Safe: a NUL-terminated path; reads into `chunk` within its length.
    unsafe {
        let fd = sys::sceIoOpen(path.as_ptr(), sys::SCE_O_RDONLY, 0);
        if fd < 0 {
            return false;
        }
        let mut offset = 0;
        let equal = loop {
            let read = sys::sceIoRead(fd, chunk.as_mut_ptr().cast(), CHUNK as u32);
            if read <= 0 {
                break read == 0 && offset == bytes.len();
            }
            let read = read as usize;
            if offset + read > bytes.len() || chunk[..read] != bytes[offset..offset + read] {
                break false;
            }
            offset += read;
        };
        sys::sceIoClose(fd);
        equal
    }
}

/// Writes `bytes` as the whole file; true when all were written.
fn write_file(path: &[u8], bytes: &[u8]) -> bool {
    // Safe: a NUL-terminated path; writes from `bytes` within its length.
    unsafe {
        let fd = sys::sceIoOpen(path.as_ptr(), sys::SCE_O_WRONLY | sys::SCE_O_CREAT | sys::SCE_O_TRUNC, 0o777);
        if fd < 0 {
            return false;
        }
        let written = sys::sceIoWrite(fd, bytes.as_ptr() as *const c_void, bytes.len() as u32);
        let closed = sys::sceIoClose(fd);
        written == bytes.len() as i32 && closed >= 0
    }
}

/// Installs the PSP plugin and its plugins.txt line (start thread, once,
/// before the PSP side boots).
pub fn install() {
    let start = sys::now_us();
    // Safe: the start thread is the only user.
    let buffers = unsafe { BUFFERS.get() };
    let (mut directory, mut prx, mut list) = ([0u8; 64], [0u8; 64], [0u8; 64]);
    let directory = path(&mut directory, b"/seplugins");
    let prx = path(&mut prx, b"/seplugins/nativepsp.prx");
    let list = path(&mut list, b"/seplugins/plugins.txt");
    // Safe: a NUL-terminated path; fails harmlessly when it exists.
    unsafe { sys::sceIoMkdir(directory.as_ptr(), 0o777) };

    let plugin = if file_equals(prx, PRX, &mut buffers.chunk) {
        "up to date"
    } else if write_file(prx, PRX) {
        "written"
    } else {
        "NOT written"
    };
    let existing = match read_file(list, &mut buffers.existing) {
        Read::Bytes(length) => Some(length),
        Read::Missing => Some(0),
        Read::TooBig => None,
    };
    let line = match existing.map(|length| plugins_txt_with_ours(&buffers.existing[..length], &mut buffers.updated)) {
        None => "plugins.txt too big, not changed",
        Some(None) => "line there",
        Some(Some(length)) if write_file(list, &buffers.updated[..length]) => "line added",
        Some(Some(_)) => "line NOT added",
    };
    render_log::line(format_args!(
        "psp plugin: {} {} bytes {}, {}, {} us\n",
        shown(prx),
        PRX.len(),
        plugin,
        line,
        sys::now_us() - start
    ));
}
