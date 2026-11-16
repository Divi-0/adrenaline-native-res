//! The "GAME <fps>" label next to VitaCam's box: the PSP game's own frame
//! rate (frames the capture saw end, one per GE frame the game showed),
//! measured once a second, and its picture (white 5x7 letters on black,
//! drawn by the GPU at 2x).
//!
//! Was menu.c's UpdateGameFps / DrawGameFps (vita2d's pgf text) until the
//! renderer became a plugin (DESIGN.md stage 2 step 4): the plugin can't use
//! Adrenaline's font, so the label is a small texture the CPU writes when
//! the value changes.

/// Measures the game's frame rate from the capture's completed frames.
#[derive(Clone, Copy, Debug, Default)]
pub struct GameFps {
    since_us: u64,
    frames_then: u32,
    fps: Option<u32>,
}

/// A gap longer than this (renderer off, game paused) starts over.
const RESTART_US: u64 = 5_000_000;
/// The measuring period.
const PERIOD_US: u64 = 1_000_000;

impl GameFps {
    pub const fn new() -> Self {
        GameFps { since_us: 0, frames_then: 0, fps: None }
    }

    /// The last value measured (None before the first second).
    pub fn fps(&self) -> Option<u32> {
        self.fps
    }

    /// Called whenever our picture is shown, with the process time and the
    /// capture's completed frame count. True when the value changed (the
    /// label must be drawn again).
    pub fn update(&mut self, now_us: u64, frames: u32) -> bool {
        if self.since_us == 0 || now_us.wrapping_sub(self.since_us) > RESTART_US {
            self.since_us = now_us;
            self.frames_then = frames;
            return false;
        }
        let elapsed = now_us - self.since_us;
        if elapsed < PERIOD_US {
            return false;
        }
        let counted = frames.wrapping_sub(self.frames_then) as u64;
        let fps = ((counted * 1_000_000 + elapsed / 2) / elapsed) as u32;
        self.since_us = now_us;
        self.frames_then = frames;
        if self.fps == Some(fps) {
            return false;
        }
        self.fps = Some(fps);
        true
    }
}

/// The label's picture: LABEL_WIDTH x LABEL_HEIGHT texels (room for
/// "GAME 999"), drawn 2x; `draw_label` says how much of its width is used.
pub const LABEL_WIDTH: usize = 56;
pub const LABEL_HEIGHT: usize = 11;
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
        _ => [0; GLYPH_HEIGHT],
    }
}

/// Writes "GAME <fps>" into `pixels` (LABEL_WIDTH x LABEL_HEIGHT, rows top
/// first): white letters on black. Values above 999 show as 999. Returns
/// the texels used from the left (text and border), the width to draw.
pub fn draw_label(fps: u32, pixels: &mut [u32; LABEL_WIDTH * LABEL_HEIGHT]) -> usize {
    pixels.fill(BLACK);
    let mut text = *b"GAME    ";
    let mut length = 5;
    let fps = fps.min(999);
    let digits = if fps >= 100 { 3 } else if fps >= 10 { 2 } else { 1 };
    let mut value = fps;
    for index in (0..digits).rev() {
        text[length + index] = b'0' + (value % 10) as u8;
        value /= 10;
    }
    length += digits;
    for (position, &character) in text[..length].iter().enumerate() {
        let left = BORDER + position * ADVANCE;
        for (row, bits) in glyph(character).iter().enumerate() {
            for column in 0..GLYPH_WIDTH {
                if bits & (0x10 >> column) != 0 {
                    pixels[(BORDER + row) * LABEL_WIDTH + left + column] = WHITE;
                }
            }
        }
    }
    BORDER + length * ADVANCE - 1 + BORDER
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_second_only_starts_counting() {
        let mut fps = GameFps::new();
        assert!(!fps.update(10_000_000, 100));
        assert!(!fps.update(10_500_000, 115));
        assert_eq!(fps.fps(), None);
    }

    #[test]
    fn thirty_frames_in_a_second_read_30() {
        let mut fps = GameFps::new();
        fps.update(10_000_000, 100);
        assert!(fps.update(11_000_000, 130));
        assert_eq!(fps.fps(), Some(30));
        // The same value again: nothing to redraw.
        assert!(!fps.update(12_000_000, 160));
        assert!(fps.update(13_000_000, 184));
        assert_eq!(fps.fps(), Some(24));
    }

    #[test]
    fn a_long_gap_starts_over() {
        let mut fps = GameFps::new();
        fps.update(10_000_000, 100);
        fps.update(11_000_000, 130);
        // 6 s without our picture (renderer off): not 1 frame in 6 s.
        assert!(!fps.update(17_000_000, 131));
        assert_eq!(fps.fps(), Some(30));
        assert!(fps.update(18_000_000, 191));
        assert_eq!(fps.fps(), Some(60));
    }

    fn white(pixels: &[u32; LABEL_WIDTH * LABEL_HEIGHT], column: usize, row: usize) -> bool {
        pixels[row * LABEL_WIDTH + column] == WHITE
    }

    #[test]
    fn the_label_shows_game_and_the_value() {
        let mut pixels = [0; LABEL_WIDTH * LABEL_HEIGHT];
        // "GAME 60": 7 letters of 5 texels, 1 apart, 2 of border each side.
        assert_eq!(draw_label(60, &mut pixels), 45);
        assert!(pixels.iter().all(|&p| p == WHITE || p == BLACK));
        // The border stays black.
        for column in 0..LABEL_WIDTH {
            for row in [0, 1, LABEL_HEIGHT - 2, LABEL_HEIGHT - 1] {
                assert!(!white(&pixels, column, row), "{column} {row}");
            }
        }
        for row in 0..LABEL_HEIGHT {
            for column in [0, 1, 43, 44, 45, LABEL_WIDTH - 1] {
                assert!(!white(&pixels, column, row), "{column} {row}");
            }
        }
        // M's top row (letter 3, columns 14-18): #...#
        let m: Vec<bool> = (14..19).map(|column| white(&pixels, column, 2)).collect();
        assert_eq!(m, [true, false, false, false, true]);
        // The blank (letter 5, columns 26-30) is all black.
        assert!((26..31).all(|column| (0..LABEL_HEIGHT).all(|row| !white(&pixels, column, row))));
        // '6' (letter 6, columns 32-36), its middle row (row 2 + 3): #### .
        let six: Vec<bool> = (32..37).map(|column| white(&pixels, column, 5)).collect();
        assert_eq!(six, [true, true, true, true, false]);
        // Every set bit of the 6 glyphs is a white texel, nothing else.
        let lit: u32 = b"GAME60".iter().map(|&c| glyph(c).iter().map(|row| row.count_ones()).sum::<u32>()).sum();
        assert_eq!(pixels.iter().filter(|&&p| p == WHITE).count() as u32, lit);
    }

    #[test]
    fn small_and_big_values_fit() {
        let mut pixels = [0; LABEL_WIDTH * LABEL_HEIGHT];
        assert_eq!(draw_label(7, &mut pixels), 39);
        // Above 999: "GAME 999", 8 letters, inside the picture.
        assert_eq!(draw_label(5000, &mut pixels), 51);
        assert!(51 <= LABEL_WIDTH);
        let nine: Vec<bool> = (44..49).map(|column| white(&pixels, column, 5)).collect();
        assert_eq!(nine, [false, true, true, true, true]);
    }
}
