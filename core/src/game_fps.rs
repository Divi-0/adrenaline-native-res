//! The diagnostics label (game fps, drawn/held/skipped pictures, timings):
//! measured once a second, its picture white 5x7 letters on black, drawn by
//! the GPU at 2x.

/// A gap longer than this (renderer off, game paused) starts over.
const RESTART_US: u64 = 5_000_000;
/// The measuring period.
const PERIOD_US: u64 = 1_000_000;

/// Black border around the text, in texels.
const BORDER: usize = 2;
const GLYPH_WIDTH: usize = 5;
const GLYPH_HEIGHT: usize = 7;
/// Texels from one letter's start to the next.
const ADVANCE: usize = GLYPH_WIDTH + 1;

/// Opaque white and black, RGBA in memory (ABGR words).
pub const WHITE: u32 = 0xFFFF_FFFF;
pub const BLACK: u32 = 0xFF00_0000;

/// 5x7 glyphs, one byte per row (bit 4 = leftmost column).
fn glyph(character: u8) -> [u8; GLYPH_HEIGHT] {
    match character {
        b'0' => [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E],
        b'1' => [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E],
        b'2' => [0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F],
        b'3' => [0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E],
        b'4' => [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
        b'5' => [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E],
        b'6' => [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E],
        b'7' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
        b'8' => [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E],
        b'9' => [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C],
        b'G' => [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F],
        b'A' => [0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        b'M' => [0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11],
        b'E' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F],
        b'D' => [0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E],
        b'R' => [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11],
        b'W' => [0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A],
        b'N' => [0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11],
        b'H' => [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        b'L' => [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F],
        b'S' => [0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E],
        b'K' => [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11],
        b'I' => [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
        b'P' => [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10],
        b'C' => [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E],
        b'U' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        b'X' => [0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11],
        b'O' => [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        b'.' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C],
        _ => [0; GLYPH_HEIGHT],
    }
}

/// The diagnostics label (config `gamefps=1`, debug `gamefps 1`): what the
/// display loop did, top right, three lines:
///
/// ```text
/// GAME 60 DRAWN 60 LOW 58   game frames / new pictures shown per second,
///                           the 1% low: new pictures per second at the
///                           worst 1% of gaps over the last 10 s
/// HELD 0 SKIP 0 GAP 17      over the last second: passes that showed the
///                           last picture again, game frames never shown,
///                           longest ms between two new pictures
/// CPU 1.3 MAX 3.4 SWAP 0.2  our frame's display-thread ms per pass
///                           (average, longest), longest wait to queue
///                           the picture (display queue full: GPU behind)
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DrawStatsValues {
    pub game: u32,
    pub drawn: u32,
    /// 1% low, new pictures per second.
    pub low: u32,
    pub held: u32,
    pub skipped: u32,
    pub gap_ms: u32,
    /// Tenths of a millisecond.
    pub cpu_avg: u32,
    pub cpu_max: u32,
    pub swap_max: u32,
}

/// Gaps between new pictures kept for the 1% low (10 s at 60 fps).
const LOW_GAPS: usize = 600;
/// One refresh of the Vita's screen (59.94 Hz).
const REFRESH_US: u64 = 16_683;

/// Counts `DrawStatsValues` pass by pass (display thread).
#[derive(Clone, Copy, Debug)]
pub struct DrawStats {
    since_us: u64,
    game_then: u32,
    passes: u32,
    drawn: u32,
    held: u32,
    skipped: u32,
    last_frame: Option<u32>,
    last_new_us: u64,
    gap_us: u64,
    cpu_sum_us: u64,
    cpu_max_us: u64,
    swap_max_us: u64,
    /// Gaps between new pictures, tenths of a ms, a ring.
    gaps: [u16; LOW_GAPS],
    gap_count: usize,
    gap_next: usize,
    values: Option<DrawStatsValues>,
}

impl Default for DrawStats {
    fn default() -> Self {
        DrawStats::new()
    }
}

impl DrawStats {
    pub const fn new() -> Self {
        DrawStats {
            since_us: 0,
            game_then: 0,
            passes: 0,
            drawn: 0,
            held: 0,
            skipped: 0,
            last_frame: None,
            last_new_us: 0,
            gap_us: 0,
            cpu_sum_us: 0,
            cpu_max_us: 0,
            swap_max_us: 0,
            gaps: [0; LOW_GAPS],
            gap_count: 0,
            gap_next: 0,
            values: None,
        }
    }

    /// The last second's values (None before the first second).
    pub fn values(&self) -> Option<DrawStatsValues> {
        self.values
    }

    /// Starts a second (keeps the gaps, the values and the last picture).
    fn new_second(&mut self, now_us: u64, game: u32) {
        self.since_us = now_us;
        self.game_then = game;
        self.passes = 0;
        self.drawn = 0;
        self.held = 0;
        self.skipped = 0;
        self.gap_us = 0;
        self.cpu_sum_us = 0;
        self.cpu_max_us = 0;
        self.swap_max_us = 0;
    }

    /// New pictures per second at the worst 1% of the kept gaps (at least
    /// one gap), 0 without gaps.
    fn one_percent_low(&self) -> u32 {
        let count = self.gap_count;
        if count == 0 {
            return 0;
        }
        let worst = (count / 100).max(1);
        // The `worst` largest gaps, by repeated selection (few: 6 of 600).
        let mut taken = [false; LOW_GAPS];
        let mut sum: u64 = 0;
        for _ in 0..worst {
            let mut best: Option<usize> = None;
            for index in 0..count {
                if !taken[index] && best.is_none_or(|b| self.gaps[index] > self.gaps[b]) {
                    best = Some(index);
                }
            }
            let Some(best) = best else { break };
            taken[best] = true;
            sum += self.gaps[best] as u64;
        }
        if sum == 0 {
            return 0;
        }
        // tenths of a ms -> per second: 10000 / mean.
        ((10_000 * worst as u64 + sum / 2) / sum) as u32
    }

    /// One display pass that showed our picture of PSP frame `shown_frame`
    /// at `now_us`: `cpu_us` our frame's time on the display thread,
    /// `swap_us` the previous pass's wait to queue its picture, `game` the
    /// capture's completed frame count. True when a new second's values
    /// are there (the label must be drawn again).
    pub fn pass(&mut self, now_us: u64, shown_frame: u32, cpu_us: u64, swap_us: u64, game: u32) -> bool {
        if self.since_us == 0 || now_us.wrapping_sub(self.since_us) > RESTART_US {
            // A first pass, or back after a long gap (renderer off, game
            // paused): start over, old gaps included.
            let values = self.values;
            *self = DrawStats::new();
            self.values = values;
            self.new_second(now_us, game);
            self.last_frame = Some(shown_frame);
            self.last_new_us = now_us;
            return false;
        }
        self.passes += 1;
        match self.last_frame {
            Some(last) if last == shown_frame => self.held += 1,
            _ => {
                self.drawn += 1;
                if let Some(last) = self.last_frame {
                    self.skipped += shown_frame.wrapping_sub(last).saturating_sub(1).min(1000);
                }
                // In whole refreshes: the pass's CPU time jitters by a few
                // ms, the screen changes only at a refresh (20 ms read as
                // 1 refresh, a held frame as 2).
                let refreshes = ((now_us - self.last_new_us + REFRESH_US / 2) / REFRESH_US).max(1);
                let gap = refreshes * REFRESH_US;
                self.gap_us = self.gap_us.max(gap);
                self.gaps[self.gap_next] = ((gap + 50) / 100).min(u16::MAX as u64) as u16;
                self.gap_next = (self.gap_next + 1) % LOW_GAPS;
                self.gap_count = (self.gap_count + 1).min(LOW_GAPS);
                self.last_new_us = now_us;
            }
        }
        self.last_frame = Some(shown_frame);
        self.cpu_sum_us += cpu_us;
        self.cpu_max_us = self.cpu_max_us.max(cpu_us);
        self.swap_max_us = self.swap_max_us.max(swap_us);
        let elapsed = now_us - self.since_us;
        if elapsed < PERIOD_US {
            return false;
        }
        let per_second = |count: u32| ((count as u64 * 1_000_000 + elapsed / 2) / elapsed) as u32;
        let tenths = |us: u64| ((us + 50) / 100).min(9999) as u32;
        self.values = Some(DrawStatsValues {
            game: per_second(game.wrapping_sub(self.game_then)),
            drawn: per_second(self.drawn),
            low: self.one_percent_low(),
            held: self.held,
            skipped: self.skipped,
            gap_ms: ((self.gap_us + 500) / 1000).min(999) as u32,
            cpu_avg: tenths(self.cpu_sum_us / self.passes.max(1) as u64),
            cpu_max: tenths(self.cpu_max_us),
            swap_max: tenths(self.swap_max_us),
        });
        self.new_second(now_us, game);
        true
    }
}

/// The diagnostics label's picture: STATS_WIDTH x STATS_HEIGHT texels.
pub const STATS_WIDTH: usize = 152;
pub const STATS_HEIGHT: usize = 2 * BORDER + 3 * (GLYPH_HEIGHT + LINE_GAP) - LINE_GAP;
const LINE_GAP: usize = 2;

/// One line of text: bytes and length.
pub type Line = ([u8; 32], usize);

fn push(line: &mut Line, bytes: &[u8]) {
    for &byte in bytes {
        if line.1 < line.0.len() {
            line.0[line.1] = byte;
            line.1 += 1;
        }
    }
}

/// Appends `value` in decimal.
fn push_number(line: &mut Line, value: u32) {
    let mut digits = [0u8; 10];
    let mut count = 0;
    let mut rest = value;
    loop {
        digits[count] = b'0' + (rest % 10) as u8;
        count += 1;
        rest /= 10;
        if rest == 0 {
            break;
        }
    }
    for index in (0..count).rev() {
        push(line, &[digits[index]]);
    }
}

/// Appends tenths as "<whole>.<tenth>".
fn push_tenths(line: &mut Line, tenths: u32) {
    push_number(line, tenths / 10);
    push(line, b".");
    push_number(line, tenths % 10);
}

/// The three lines of `values`.
pub fn stats_lines(values: &DrawStatsValues) -> [Line; 3] {
    let mut lines = [([0u8; 32], 0usize); 3];
    push(&mut lines[0], b"GAME ");
    push_number(&mut lines[0], values.game);
    push(&mut lines[0], b" DRAWN ");
    push_number(&mut lines[0], values.drawn);
    push(&mut lines[0], b" LOW ");
    push_number(&mut lines[0], values.low);
    push(&mut lines[1], b"HELD ");
    push_number(&mut lines[1], values.held);
    push(&mut lines[1], b" SKIP ");
    push_number(&mut lines[1], values.skipped);
    push(&mut lines[1], b" GAP ");
    push_number(&mut lines[1], values.gap_ms);
    push(&mut lines[2], b"CPU ");
    push_tenths(&mut lines[2], values.cpu_avg);
    push(&mut lines[2], b" MAX ");
    push_tenths(&mut lines[2], values.cpu_max);
    push(&mut lines[2], b" SWAP ");
    push_tenths(&mut lines[2], values.swap_max);
    lines
}

/// Writes `values` into `pixels` (STATS_WIDTH x STATS_HEIGHT, rows top
/// first): white letters on black, lines cut at the picture's width.
/// Returns the texels used from the left, the width to draw.
pub fn draw_stats_label(values: &DrawStatsValues, pixels: &mut [u32; STATS_WIDTH * STATS_HEIGHT]) -> usize {
    pixels.fill(BLACK);
    let mut used = 0;
    for (line, (text, length)) in stats_lines(values).iter().enumerate() {
        let top = BORDER + line * (GLYPH_HEIGHT + LINE_GAP);
        let fits = (*length).min((STATS_WIDTH - 2 * BORDER + 1) / ADVANCE);
        for (position, &character) in text[..fits].iter().enumerate() {
            let left = BORDER + position * ADVANCE;
            for (row, bits) in glyph(character).iter().enumerate() {
                for column in 0..GLYPH_WIDTH {
                    if bits & (0x10 >> column) != 0 {
                        pixels[(top + row) * STATS_WIDTH + left + column] = WHITE;
                    }
                }
            }
        }
        used = used.max(BORDER + fits * ADVANCE - 1 + BORDER);
    }
    used
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_count_new_held_and_skipped_frames_over_a_second() {
        let mut stats = DrawStats::new();
        // The first pass only starts counting.
        assert!(!stats.pass(10_000_000, 100, 1000, 0, 500));
        // 60 passes 1/60 s apart: frame steps 1, except pass 11 showing
        // the same frame again and pass 21 jumping two frames.
        let mut frame = 100;
        let mut changed = false;
        for pass in 1..=60u64 {
            if pass != 11 {
                frame += 1;
            }
            if pass == 21 {
                frame += 1;
            }
            let cpu = if pass == 30 { 3400 } else { 1000 };
            let swap = if pass == 5 { 200 } else { 0 };
            changed = stats.pass(10_000_000 + pass * 16_667, frame, cpu, swap, 500 + pass as u32);
        }
        assert!(changed);
        let values = stats.values().unwrap();
        assert_eq!(values.game, 60);
        assert_eq!(values.drawn, 59);
        assert_eq!(values.held, 1);
        assert_eq!(values.skipped, 1);
        // The held pass: 2 refreshes between new pictures.
        assert_eq!(values.gap_ms, 33);
        // 59 gaps, the worst 1% (one): 33.3 ms -> 30 per second.
        assert_eq!(values.low, 30);
        assert_eq!(values.cpu_max, 34);
        assert_eq!(values.cpu_avg, 10);
        assert_eq!(values.swap_max, 2);
    }

    #[test]
    fn the_one_percent_low_takes_the_worst_gaps_of_ten_seconds() {
        let mut stats = DrawStats::new();
        stats.pass(0 + 1, 0, 0, 0, 0);
        // 10 s of steady frames with 6 held ones (a 1% low of 30) ...
        let mut now = 1u64;
        let mut frame = 0;
        for pass in 1..=600u64 {
            now += 16_667;
            if pass % 100 != 50 {
                frame += 1;
            }
            stats.pass(now, frame, 0, 0, pass as u32);
        }
        assert_eq!(stats.values().unwrap().low, 30);
        // ... then 10 s without: the old gaps leave the ring, back to 60.
        for pass in 601..=1200u64 {
            now += 16_667;
            frame += 1;
            stats.pass(now, frame, 0, 0, pass as u32);
        }
        assert_eq!(stats.values().unwrap().low, 60);
    }

    #[test]
    fn pass_time_jitter_is_not_a_held_frame() {
        let mut stats = DrawStats::new();
        stats.pass(1, 0, 0, 0, 0);
        // New pictures every refresh, the passes' CPU times 13 .. 20 ms apart.
        let mut now = 1u64;
        for pass in 1..=120u64 {
            now += if pass % 2 == 0 { 20_000 } else { 13_366 };
            stats.pass(now, pass as u32, 0, 0, pass as u32);
        }
        let values = stats.values().unwrap();
        assert_eq!(values.gap_ms, 17);
        assert_eq!(values.low, 60);
    }

    #[test]
    fn stats_lines_read_as_meant() {
        let values = DrawStatsValues { game: 60, drawn: 59, low: 30, held: 1, skipped: 2, gap_ms: 33, cpu_avg: 13, cpu_max: 34, swap_max: 2 };
        let lines = stats_lines(&values);
        let text = |i: usize| core::str::from_utf8(&lines[i].0[..lines[i].1]).unwrap().to_string();
        assert_eq!(text(0), "GAME 60 DRAWN 59 LOW 30");
        assert_eq!(text(1), "HELD 1 SKIP 2 GAP 33");
        assert_eq!(text(2), "CPU 1.3 MAX 3.4 SWAP 0.2");
        let mut pixels = [0; STATS_WIDTH * STATS_HEIGHT];
        let used = draw_stats_label(&values, &mut pixels);
        assert_eq!(used, 2 + 24 * 6 - 1 + 2);
        assert!(used <= STATS_WIDTH);
        // Every letter has a glyph (only spaces are blank).
        for line in &lines {
            for &c in &line.0[..line.1] {
                assert!(c == b' ' || glyph(c) != [0; GLYPH_HEIGHT], "{}", c as char);
            }
        }
        // Big values are cut at the width, no panic.
        let big = DrawStatsValues { game: 999, drawn: 999, low: 999, held: 9999, skipped: 9999, gap_ms: 999, cpu_avg: 9999, cpu_max: 9999, swap_max: 9999 };
        assert!(draw_stats_label(&big, &mut pixels) <= STATS_WIDTH);
    }
}
