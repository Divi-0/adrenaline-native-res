//! The progress screen: a text terminal on the Vita's display while the
//! app compiles. The app's own frame buffer (2 MiB of CDRAM, shown with
//! sceDisplaySetFrameBuf, no GXM), the 8x8 font at twice its size: 80
//! columns, 30 rows. A title row with the time so far, a status row, a
//! progress bar with the count, the file being compiled, and a scrolling
//! log of the results (errors in red).
//!
//! Everything runs on the app's main thread, between compiles: the screen
//! stands still while the compiler works on one file. Only rows whose
//! text changed are drawn again.

use core::cell::UnsafeCell;
use core::ffi::c_void;
use core::fmt::Write;

use crate::font;

const WIDTH: usize = 960;
const HEIGHT: usize = 544;
/// The font's pixels drawn 2x2.
const SCALE: usize = 2;
/// The glyphs are 5 pixels wide: one column of space after them.
const CELL_WIDTH: usize = 6 * SCALE;
/// 16 pixels of glyph, one above and one below.
const ROW_HEIGHT: usize = 8 * SCALE + 2;
pub const COLUMNS: usize = WIDTH / CELL_WIDTH;
const ROWS: usize = HEIGHT / ROW_HEIGHT;
/// 960 * 544 * 4 bytes, rounded up to CDRAM's 256 KiB.
const BUFFER_BYTES: u32 = 2 * 1024 * 1024;
const SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RW: u32 = 0x0940_8060;
const SCE_DISPLAY_PIXELFORMAT_A8B8G8R8: u32 = 0;
const SCE_DISPLAY_SETBUF_NEXTFRAME: i32 = 1;

const ROW_TITLE: usize = 0;
const ROW_STATUS: usize = 1;
const ROW_BAR: usize = 3;
const ROW_CURRENT: usize = 4;
const LOG_FIRST: usize = 6;
const LOG_ROWS: usize = ROWS - LOG_FIRST;
/// The progress bar's width: the row's columns before the count.
const BAR_COLUMNS: usize = COLUMNS - 22;

/// A8B8G8R8 in a little-endian word.
const fn rgb(r: u32, g: u32, b: u32) -> u32 {
    0xFF00_0000 | (b << 16) | (g << 8) | r
}

const BACKGROUND: u32 = rgb(0x10, 0x12, 0x16);
const TITLE_BACKGROUND: u32 = rgb(0x1E, 0x3A, 0x6E);
const WHITE: u32 = rgb(0xF0, 0xF0, 0xF0);
pub const GREY: u32 = rgb(0xA8, 0xAC, 0xB4);
pub const GREEN: u32 = rgb(0x7C, 0xE0, 0x8C);
pub const RED: u32 = rgb(0xFF, 0x6A, 0x5E);
pub const YELLOW: u32 = rgb(0xF0, 0xD0, 0x60);
const BAR_DONE: u32 = rgb(0x4C, 0xB8, 0x5C);
const BAR_FAILED: u32 = rgb(0xD8, 0x50, 0x44);
const BAR_LEFT: u32 = rgb(0x30, 0x34, 0x3C);

extern "C" {
    fn sceKernelAllocMemBlock(name: *const u8, block_type: u32, size: u32, options: *mut c_void) -> i32;
    fn sceKernelGetMemBlockBase(uid: i32, base: *mut *mut c_void) -> i32;
    fn sceKernelGetProcessTimeWide() -> u64;
    fn sceKernelPowerTick(tick_type: i32) -> i32;
    fn sceKernelDelayThread(microseconds: u32) -> i32;
    fn sceDisplaySetFrameBuf(frame: *const SceDisplayFrameBuf, sync: i32) -> i32;
    fn sceCtrlPeekBufferPositive(port: i32, data: *mut SceCtrlData, count: i32) -> i32;
}

/// psp2/display.h SceDisplayFrameBuf.
#[repr(C)]
struct SceDisplayFrameBuf {
    size: u32,
    base: *mut c_void,
    pitch: u32,
    pixel_format: u32,
    width: u32,
    height: u32,
}
const _: () = assert!(core::mem::size_of::<SceDisplayFrameBuf>() == 0x18);

/// psp2/ctrl.h SceCtrlData (0x20 bytes; room to spare).
#[repr(C)]
struct SceCtrlData {
    time_stamp: u64,
    buttons: u32,
    rest: [u8; 52],
}

fn now_us() -> u64 {
    // Safe: no arguments.
    unsafe { sceKernelGetProcessTimeWide() }
}

/// One screen row: its text and colours, and (the bar row) the bar's
/// pixels.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Row {
    text: [u8; COLUMNS],
    len: u8,
    fg: u32,
    bg: u32,
    /// Pixels of the bar from the left edge: done, failed, all (0: none).
    bar: (u16, u16, u16),
}

impl Row {
    const BLANK: Row = Row { text: [b' '; COLUMNS], len: 0, fg: GREY, bg: BACKGROUND, bar: (0, 0, 0) };

    fn of(fg: u32, bg: u32, args: core::fmt::Arguments) -> Row {
        let mut row = Row { fg, bg, ..Row::BLANK };
        let _ = row.write_fmt(args);
        row
    }

    /// `text` right-aligned at the row's end (over what is there).
    fn right(&mut self, text: &[u8]) {
        let take = text.len().min(COLUMNS);
        let start = COLUMNS - take;
        if (self.len as usize) < start {
            self.text[self.len as usize..start].fill(b' ');
        }
        self.text[start..].copy_from_slice(&text[..take]);
        self.len = COLUMNS as u8;
    }
}

impl Write for Row {
    fn write_str(&mut self, text: &str) -> core::fmt::Result {
        for &byte in text.as_bytes() {
            if self.len as usize == COLUMNS {
                break;
            }
            self.text[self.len as usize] = byte;
            self.len += 1;
        }
        Ok(())
    }
}

/// Text cut off when full (a log line before it is split into rows).
struct Line {
    bytes: [u8; COLUMNS * 4],
    len: usize,
}

impl Write for Line {
    fn write_str(&mut self, text: &str) -> core::fmt::Result {
        let take = text.len().min(self.bytes.len() - self.len);
        self.bytes[self.len..self.len + take].copy_from_slice(&text.as_bytes()[..take]);
        self.len += take;
        Ok(())
    }
}

struct Console {
    pixels: *mut u32,
    started: u64,
    /// Time of the batches before this process (restore).
    elapsed_before: u64,
    status: Row,
    current: Row,
    total: u32,
    done: u32,
    failed: u32,
    /// The compiler's heap after the last source: bytes cut, of all.
    heap: (u32, u32),
    /// The log as a ring: `log_next` is the oldest row once it is full.
    log: [Row; LOG_ROWS],
    log_next: usize,
    /// What each screen row shows now.
    shown: [Row; ROWS],
    /// Time spent drawing (reported in done.txt).
    draw_us: u64,
    /// Rows drawn: a fresh buffer holds garbage, all rows go once.
    drawn_once: bool,
}

struct Shared(UnsafeCell<Console>);
// Safe: only the app's main thread touches the console.
unsafe impl Sync for Shared {}

static CONSOLE: Shared = Shared(UnsafeCell::new(Console {
    pixels: core::ptr::null_mut(),
    started: 0,
    elapsed_before: 0,
    status: Row::BLANK,
    current: Row::BLANK,
    total: 0,
    done: 0,
    failed: 0,
    heap: (0, 0),
    log: [Row::BLANK; LOG_ROWS],
    log_next: 0,
    shown: [Row::BLANK; ROWS],
    draw_us: 0,
    drawn_once: false,
}));

fn console() -> &'static mut Console {
    // Safe: main thread only (see Shared); no caller holds two at once.
    unsafe { &mut *CONSOLE.0.get() }
}

/// What the screen carries from one batch (process) to the next: the
/// counts and the log (plain bytes in the app's resume.bin).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Saved {
    done: u32,
    failed: u32,
    log_next: u32,
    log: [Row; LOG_ROWS],
}

impl Saved {
    pub const EMPTY: Saved = Saved { done: 0, failed: 0, log_next: 0, log: [Row::BLANK; LOG_ROWS] };
}

/// The counts and the log, for the next batch.
pub fn save() -> Saved {
    let console = console();
    Saved { done: console.done, failed: console.failed, log_next: console.log_next as u32, log: console.log }
}

/// Goes on from a batch before (before `init`): its counts, its log and
/// the time it took.
pub fn restore(saved: &Saved, elapsed_us: u64) {
    let console = console();
    console.done = saved.done;
    console.failed = saved.failed;
    console.log_next = saved.log_next as usize % LOG_ROWS;
    console.log = saved.log;
    console.elapsed_before = elapsed_us;
}

/// Makes the frame buffer and shows it; false without it (the app works
/// on, without a screen).
pub fn init() -> bool {
    let console = console();
    console.started = now_us();
    // Safe: plain system calls; the block is ours alone.
    let base = unsafe {
        let block = sceKernelAllocMemBlock(b"NativeResScreen\0".as_ptr(), SCE_KERNEL_MEMBLOCK_TYPE_USER_CDRAM_RW, BUFFER_BYTES, core::ptr::null_mut());
        let mut base: *mut c_void = core::ptr::null_mut();
        if block < 0 || sceKernelGetMemBlockBase(block, &mut base) < 0 {
            return false;
        }
        base
    };
    console.pixels = base as *mut u32;
    console.draw();
    let frame = SceDisplayFrameBuf {
        size: core::mem::size_of::<SceDisplayFrameBuf>() as u32,
        base,
        pitch: WIDTH as u32,
        pixel_format: SCE_DISPLAY_PIXELFORMAT_A8B8G8R8,
        width: WIDTH as u32,
        height: HEIGHT as u32,
    };
    // Safe: the buffer is 960x544 words, ours for the app's life.
    if unsafe { sceDisplaySetFrameBuf(&frame, SCE_DISPLAY_SETBUF_NEXTFRAME) } < 0 {
        console.pixels = core::ptr::null_mut();
        return false;
    }
    true
}

/// The status row (what the compiler loaded, what is switched on).
pub fn status(color: u32, args: core::fmt::Arguments) {
    let console = console();
    console.status = Row::of(color, BACKGROUND, args);
    console.draw();
}

/// How many sources this run compiles.
pub fn set_total(total: u32) {
    console().total = total;
}

/// The file now being compiled (shown until its result).
pub fn begin(name: &str, bytes: u32, vertex: bool) {
    let console = console();
    let kind = if vertex { "vertex" } else { "fragment" };
    console.current = Row::of(WHITE, BACKGROUND, format_args!("> {name}  ({kind}, {:.1} KiB)", bytes as f32 / 1024.0));
    console.draw();
}

/// One file's result: a log line, counted as done (and failed unless
/// `ok`).
pub fn result(ok: bool, args: core::fmt::Arguments) {
    let console = console();
    console.done += 1;
    if !ok {
        console.failed += 1;
    }
    console.current = Row::BLANK;
    console.push(if ok { GREY } else { RED }, args);
    console.draw();
}

/// A log line that is no file's result.
pub fn note(color: u32, args: core::fmt::Arguments) {
    let console = console();
    console.push(color, args);
    console.draw();
}

/// The compiler's heap after a source: bytes cut from its top, of all
/// (drawn with the next change).
pub fn heap(top: u32, all: u32) {
    console().heap = (top, all);
}

/// The last line: the run's sums.
pub fn summary(args: core::fmt::Arguments) {
    let console = console();
    let color = if console.failed == 0 { GREEN } else { RED };
    console.current = Row::of(color, BACKGROUND, args);
    console.draw();
}

/// Time spent drawing so far, in microseconds.
pub fn draw_us() -> u64 {
    console().draw_us
}

/// Keeps the last screen up for `seconds` or until a button is pressed
/// (one held at the start does not count). Returns at once without a
/// screen.
pub fn hold(seconds: u32) {
    let console = console();
    if console.pixels.is_null() {
        return;
    }
    let until = now_us() + seconds as u64 * 1_000_000;
    let mut armed = false;
    let mut shown_left = u64::MAX;
    loop {
        let now = now_us();
        if now >= until {
            return;
        }
        let mut pad = SceCtrlData { time_stamp: 0, buttons: 0, rest: [0; 52] };
        // Safe: our own struct, one sample.
        let read = unsafe { sceCtrlPeekBufferPositive(0, &mut pad, 1) };
        if read >= 0 {
            if pad.buttons == 0 {
                armed = true;
            } else if armed {
                return;
            }
        }
        let left = (until - now).div_ceil(1_000_000);
        if left != shown_left {
            shown_left = left;
            console.status = Row::of(YELLOW, BACKGROUND, format_args!("press any button to close ({left} s)"));
            console.draw();
        }
        // Safe: plain wait.
        unsafe { sceKernelDelayThread(50_000) };
    }
}

impl Console {
    /// Adds a log line, split into rows of COLUMNS at spaces where it can
    /// (continuations indented).
    fn push(&mut self, color: u32, args: core::fmt::Arguments) {
        let mut line = Line { bytes: [0; COLUMNS * 4], len: 0 };
        let _ = line.write_fmt(args);
        let mut rest = &line.bytes[..line.len];
        let mut first = true;
        while first || !rest.is_empty() {
            let indent: &[u8] = if first { b"" } else { b"     " };
            let room = COLUMNS - indent.len();
            let take = if rest.len() <= room {
                rest.len()
            } else {
                // The last space that fits (a word longer than a row is cut).
                rest[..=room].iter().rposition(|&byte| byte == b' ').filter(|&space| space > 0).unwrap_or(room)
            };
            let mut row = Row { fg: color, ..Row::BLANK };
            row.text[..indent.len()].copy_from_slice(indent);
            row.text[indent.len()..indent.len() + take].copy_from_slice(&rest[..take]);
            row.len = (indent.len() + take) as u8;
            self.log[self.log_next] = row;
            self.log_next = (self.log_next + 1) % LOG_ROWS;
            rest = &rest[take..];
            while let [b' ', tail @ ..] = rest {
                rest = tail;
            }
            first = false;
        }
    }

    /// The screen as it should be now.
    fn rows(&self) -> [Row; ROWS] {
        let mut rows = [Row::BLANK; ROWS];
        let seconds = (self.elapsed_before + now_us() - self.started) as f32 / 1e6;
        let mut title = Row::of(WHITE, TITLE_BACKGROUND, format_args!(" NativeRes Shader Compiler"));
        let mut clock = Line { bytes: [0; COLUMNS * 4], len: 0 };
        let _ = write!(clock, "{seconds:.1} s ");
        title.right(&clock.bytes[..clock.len]);
        rows[ROW_TITLE] = title;
        let mut status = self.status;
        if self.heap.1 > 0 {
            let mut heap = Line { bytes: [0; COLUMNS * 4], len: 0 };
            let mib = |bytes: u32| bytes as f32 / (1024.0 * 1024.0);
            let _ = write!(heap, "  heap {:.1}/{:.0} MiB ", mib(self.heap.0), mib(self.heap.1));
            status.len = status.len.min((COLUMNS - heap.len) as u8);
            status.right(&heap.bytes[..heap.len]);
        }
        rows[ROW_STATUS] = status;

        let total = self.total.max(1);
        let percent = self.done * 100 / total;
        let mut bar = Row::of(GREY, BACKGROUND, format_args!("{:width$}  {}/{}  {percent}%", "", self.done, self.total, width = BAR_COLUMNS));
        let all = BAR_COLUMNS * CELL_WIDTH;
        let done = all * self.done.min(total) as usize / total as usize;
        let failed = all * self.failed.min(total) as usize / total as usize;
        bar.bar = (done as u16, failed as u16, all as u16);
        rows[ROW_BAR] = bar;
        // The "current" row ends with the counts (a long name is cut).
        let mut sums = Line { bytes: [0; COLUMNS * 4], len: 0 };
        let _ = write!(sums, "  ok {}  errors {} ", self.done - self.failed, self.failed);
        let mut current = self.current;
        current.len = current.len.min((COLUMNS - sums.len) as u8);
        current.right(&sums.bytes[..sums.len]);
        rows[ROW_CURRENT] = current;

        // The log: the newest row at the bottom.
        for index in 0..LOG_ROWS {
            rows[LOG_FIRST + index] = self.log[(self.log_next + index) % LOG_ROWS];
        }
        rows
    }

    /// Draws the rows that changed (all, the first time).
    fn draw(&mut self) {
        // Safe: no arguments; keeps the Vita from dimming or sleeping
        // during a long run.
        unsafe { sceKernelPowerTick(0) };
        if self.pixels.is_null() {
            return;
        }
        let started = now_us();
        let rows = self.rows();
        // Safe: the buffer is WIDTH * HEIGHT words (init), ours alone.
        let pixels = unsafe { core::slice::from_raw_parts_mut(self.pixels, WIDTH * HEIGHT) };
        for (index, row) in rows.iter().enumerate() {
            if !self.drawn_once || *row != self.shown[index] {
                draw_row(pixels, index, row);
                self.shown[index] = *row;
            }
        }
        if !self.drawn_once {
            pixels[ROWS * ROW_HEIGHT * WIDTH..].fill(BACKGROUND);
            self.drawn_once = true;
        }
        self.draw_us += now_us() - started;
    }
}

fn glyph(byte: u8) -> &'static [u8; 8] {
    let byte = if (font::FIRST..=font::LAST).contains(&byte) { byte } else { b'?' };
    &font::GLYPHS[(byte - font::FIRST) as usize]
}

/// Draws one row's band of pixels whole (no clear first: nothing
/// flickers).
fn draw_row(pixels: &mut [u32], index: usize, row: &Row) {
    let (done, failed, all) = (row.bar.0 as usize, row.bar.1 as usize, row.bar.2 as usize);
    for y in 0..ROW_HEIGHT {
        let line = &mut pixels[(index * ROW_HEIGHT + y) * WIDTH..][..WIDTH];
        let glyph_row = y.checked_sub(1).map(|y| y / SCALE).filter(|&y| y < 8);
        let in_bar = (3..ROW_HEIGHT - 3).contains(&y);
        let mut x = 0;
        for &byte in &row.text[..row.len as usize] {
            let bits = glyph_row.map_or(0, |glyph_y| glyph(byte)[glyph_y]);
            for column in 0..CELL_WIDTH {
                let on = bits & (0x80 >> (column / SCALE)) != 0;
                line[x] = if on {
                    row.fg
                } else if in_bar && x < all {
                    if x < failed {
                        BAR_FAILED
                    } else if x < done {
                        BAR_DONE
                    } else {
                        BAR_LEFT
                    }
                } else {
                    row.bg
                };
                x += 1;
            }
        }
        line[x..].fill(row.bg);
    }
}
