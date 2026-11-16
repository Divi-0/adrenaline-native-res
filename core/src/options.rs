//! Our settings file (ux0:data/nativepsp/config.txt on the Vita): one
//! `name=value` per line, read once when the plugin starts. We never write
//! it; a missing file or line means the default, an unknown name or a bad
//! value is skipped.
//!
//! ```text
//! mip=0        # distant textures: 0 sharp (default), 1 like the PSP, 2 mipmapped
//! smooth=0     # texture filter: 0 the game's own (default), 1 always smooth
//! gamefps=1    # 1 (default): the game's own fps top left
//! armclock=444 # ARM clock in MHz (333 = Sony's default, 444 = ours)
//! startskip=0  # parts of the plugin left out at the start (`start_skip`)
//! ```
//!
//! The renderer's on/off switch is not here: it is Adrenaline's setting
//! "NativePSP Renderer" (renderer/vita/src/adrenaline.rs).

/// `startskip=` bits: parts of the plugin not started, to find which of
/// them a crash at Adrenaline's start needs (the C1-2858-3 hunt, DESIGN.md).
/// Read before the hooks are set. 0 (default): everything.
pub mod start_skip {
    /// AdrenalineUser's file calls (io_hooks.rs: the bounce for PSP memory).
    pub const IO_HOOKS: u32 = 1;
    /// Installing the PSP plugin into <ms0>/seplugins (psp_plugin.rs).
    pub const PSP_PLUGIN: u32 = 2;
    /// Our "NativePSP Renderer" entry in Adrenaline's menu (menu.rs).
    pub const MENU: u32 = 4;
    /// The debug server (TCP 7777; also what loads SceNet).
    pub const DEBUG_SERVER: u32 = 8;
    /// The log stream (TCP 7778).
    pub const LOG_STREAM: u32 = 16;
    /// The display loop's hooks (display_hooks.rs): no renderer at all.
    pub const DISPLAY_HOOKS: u32 = 32;
    /// The capture with its threads, the log writer and the options thread
    /// (ARM clock, menu entry).
    pub const CAPTURE: u32 = 64;
    /// The log-only hooks on ScePspemu (pspemu_watch.rs: its heartbeat and
    /// suspend handshake).
    pub const PSPEMU_WATCH: u32 = 128;
    /// Every bit there is.
    pub const ALL: u32 = 255;
}

/// The settings, with their defaults from `Options::default`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Options {
    pub mip: u8,
    pub smooth: bool,
    pub game_fps: bool,
    pub arm_clock: i32,
    /// `start_skip` bits.
    pub start_skip: u32,
}

impl Default for Options {
    fn default() -> Self {
        Options { mip: 0, smooth: false, game_fps: true, arm_clock: 444, start_skip: 0 }
    }
}

/// The ARM clocks the Vita offers (MHz), lowest to highest.
const ARM_CLOCKS: [i32; 6] = [111, 166, 222, 266, 333, 444];

impl Options {
    /// The settings in `text` (the file's bytes), defaults for the rest.
    pub fn parse(text: &[u8]) -> Options {
        let mut options = Options::default();
        for line in text.split(|&byte| byte == b'\n') {
            // A comment runs to the end of the line.
            let line = match line.iter().position(|&byte| byte == b'#') {
                Some(hash) => &line[..hash],
                None => line,
            };
            let Some(equals) = line.iter().position(|&byte| byte == b'=') else { continue };
            let name = trim(&line[..equals]);
            let Some(value) = number(trim(&line[equals + 1..])) else { continue };
            match name {
                b"mip" if value <= 2 => options.mip = value as u8,
                b"smooth" if value <= 1 => options.smooth = value == 1,
                b"gamefps" if value <= 1 => options.game_fps = value == 1,
                b"armclock" if ARM_CLOCKS.contains(&(value as i32)) => options.arm_clock = value as i32,
                b"startskip" if value <= start_skip::ALL => options.start_skip = value,
                _ => {}
            }
        }
        options
    }
}

/// `bytes` without spaces, tabs and CRs on either side.
fn trim(bytes: &[u8]) -> &[u8] {
    let blank = |byte: &u8| matches!(byte, b' ' | b'\t' | b'\r');
    let start = bytes.iter().position(|byte| !blank(byte)).unwrap_or(bytes.len());
    let end = bytes.iter().rposition(|byte| !blank(byte)).map_or(start, |last| last + 1);
    &bytes[start..end]
}

/// A decimal number of at most 4 digits (no sign).
fn number(bytes: &[u8]) -> Option<u32> {
    if bytes.is_empty() || bytes.len() > 4 || !bytes.iter().all(u8::is_ascii_digit) {
        return None;
    }
    Some(bytes.iter().fold(0, |value, digit| value * 10 + (digit - b'0') as u32))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_file_gives_the_defaults() {
        assert_eq!(Options::parse(b""), Options::default());
        let defaults = Options::default();
        assert!(defaults.game_fps && !defaults.smooth);
        assert_eq!((defaults.mip, defaults.arm_clock), (0, 444));
    }

    #[test]
    fn reads_every_setting() {
        let options = Options::parse(b"mip=2\nsmooth=1\ngamefps=0\narmclock=333\nstartskip=33\n");
        assert_eq!(options, Options { mip: 2, smooth: true, game_fps: false, arm_clock: 333, start_skip: 33 });
        assert_eq!(options.start_skip, start_skip::IO_HOOKS | start_skip::DISPLAY_HOOKS);
    }

    #[test]
    fn crlf_spaces_and_comments() {
        // Notepad's CRLF, spaces around '=', comments on their own and after a value.
        let options = Options::parse(b"# NativePSP\r\n  mip = 1  # blurry\r\n\tsmooth\t=\t1\r\n");
        assert_eq!((options.mip, options.smooth), (1, true));
    }

    #[test]
    fn bad_values_and_unknown_names_keep_the_default() {
        let options = Options::parse(b"mip=3\nsmooth=yes\nsmooth=-1\narmclock=500\ngamefps=\nfoo=1\nmip\n=1\ngamefps=00000\nrenderer=0\nstartskip=256\n");
        assert_eq!(options, Options::default());
    }

    #[test]
    fn the_last_line_wins_and_needs_no_newline() {
        assert_eq!(Options::parse(b"mip=1\nmip=2").mip, 2);
    }
}
