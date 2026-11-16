//! Translating the GE's blend state into a GPU blend state.
//!
//! The GE computes `equation(source * source_factor, destination * destination_factor)`
//! like OpenGL and the Vita's GXM do, with a few differences:
//! - the factor "FIX" uses a constant colour (BLEND_FIXED_A / _B). GXM has no
//!   blend constant. For the source side we get the same result by
//!   multiplying the constant into the draw's colours and using factor ONE;
//!   for the destination side only black (ZERO) and white (ONE) are exact.
//! - "doubled" factors (2 x alpha) don't exist on the Vita. When the draw
//!   uses only the doubled *source* alpha (2a and 1 - 2a, MTX Mototrax's
//!   menu panels), the recorder doubles the vertex alpha instead and plain
//!   source alpha is used: exact while the vertex alpha is at most 0x80
//!   (texture alpha x 2 x vertex alpha never needs clamping then). Other
//!   uses are approximated.
//! - destination alpha holds the PSP's stencil values, which the Vita
//!   framebuffer doesn't have: approximated.
//! - the framebuffer's alpha channel IS the stencil: blending, and every
//!   draw without stencil writes, leave it alone; clears with the stencil
//!   bit write the vertex alpha (`alpha_write`, PPSSPP GPUStateUtils.cpp
//!   `IsStencilTestOutputDisabled`). Our targets keep it the same way
//!   (colour mask RGB, `BLEND_WRITES_ALPHA`), so a game sampling its
//!   framebuffer's alpha (MTX's blur quad: weight = sampled alpha) gets the
//!   PSP's values.
//! - the alpha test: textured draws are alpha-tested in the Vita's fragment
//!   shader (nativeres_textured_test_f.cg, once the display has it:
//!   `ALPHA_TEST_IN_SHADER`). Where it can't (untextured, a mask other than
//!   0xFF), a draw with blending off and the alpha test on is drawn with
//!   normal blending instead: exact for fully transparent and opaque texels,
//!   but transparent texels still write depth. Approximated.
//!
//! Factor and equation names follow PPSSPP's `GEBlendSrcFactor`,
//! `GEBlendDstFactor` and `GEBlendMode`.

use core::sync::atomic::{AtomicBool, Ordering};

use crate::command::op;
use crate::ge_state::GeState;

/// The display can alpha-test textured draws in its fragment shader (set by
/// the Vita once its program exists; the replay always). Until then the
/// alpha test is stood in for by blending (`translate`).
pub static ALPHA_TEST_IN_SHADER: AtomicBool = AtomicBool::new(false);

/// Whether the current draw's alpha test is done by the display's fragment
/// shader: textured (not clear mode), all alpha bits compared (mask 0xFF).
pub fn alpha_test_in_shader(ge: &GeState) -> bool {
    shader_can_alpha_test(ge, ALPHA_TEST_IN_SHADER.load(Ordering::Relaxed))
}

/// `alpha_test_in_shader` with `ready` for `ALPHA_TEST_IN_SHADER`.
fn shader_can_alpha_test(ge: &GeState, ready: bool) -> bool {
    ge.register(op::ALPHA_TEST_ENABLE) & 1 != 0
        && ready
        && ge.is_texture_enabled()
        && !ge.is_clear_mode()
        && (ge.register(op::ALPHA_TEST) >> 16) & 0xFF == 0xFF
}

/// GPU blend factors (the C side maps them to `SceGxmBlendFactor`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Factor {
    Zero = 0,
    One = 1,
    SourceColor = 2,
    OneMinusSourceColor = 3,
    SourceAlpha = 4,
    OneMinusSourceAlpha = 5,
    DestinationColor = 6,
    OneMinusDestinationColor = 7,
    DestinationAlpha = 8,
    OneMinusDestinationAlpha = 9,
}

/// GPU blend equations (the C side maps them to `SceGxmBlendFunc`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Equation {
    Add = 0,
    Subtract = 1,
    ReverseSubtract = 2,
    Min = 3,
    Max = 4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Blend {
    pub equation: Equation,
    pub source: Factor,
    pub destination: Factor,
    /// RGB to multiply into the draw's colours (source factor FIX), if any.
    pub source_scale: Option<u32>,
    /// The vertex alpha is to be doubled (clamped): the factors are the
    /// GE's "2 x source alpha" ones, translated to plain source alpha.
    pub double_source_alpha: bool,
    /// Something had to be approximated.
    pub approximated: bool,
    /// The GE's fixed destination factor FIX B, grey, which GXM's blend
    /// factors can't take (no constant colour; PPSSPP uses CONSTANT_COLOR):
    /// the fragment program writes it as its alpha (`destination` is then
    /// `SourceAlpha`) and applies the source factor itself (`source` is
    /// then `One`, or what it was if it doesn't read the source alpha).
    pub fix_b: Option<FixB>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixB {
    /// FIX B (all three channels), 0..255.
    pub grey: u8,
    /// What the program multiplies the colour by: the GE's source factor.
    pub premultiply: Premultiply,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Premultiply {
    None = 0,
    SourceAlpha = 1,
    OneMinusSourceAlpha = 2,
}

/// In `Draw::blend`: FIX B as the program's alpha (`Blend::fix_b`), its
/// value in bits 12..20 and the premultiply in bits 20..22.
pub const BLEND_FIX_B_ALPHA: u32 = 1 << 30;
pub const BLEND_FIX_B_SHIFT: u32 = 12;
pub const BLEND_PREMULTIPLY_SHIFT: u32 = 20;

/// Packed into `Draw::blend`: bit 31 set = blending on; source factor in
/// bits 0..4, destination factor in bits 4..8, equation in bits 8..11.
pub const BLEND_ENABLED: u32 = 1 << 31;

/// In `Draw::blend`, also without `BLEND_ENABLED`: the draw writes the
/// target's alpha channel (`alpha_write`). Without it the alpha is kept
/// (GPU colour mask RGB).
pub const BLEND_WRITES_ALPHA: u32 = 1 << 29;
/// With `BLEND_WRITES_ALPHA`: only the alpha is written, as it is (no
/// blending; the factors are ignored): clears of the stencil alone.
pub const BLEND_ALPHA_ONLY: u32 = 1 << 28;

/// The channels a draw writes into its target, from its packed blend word
/// (the Vita's colour mask; the replay's rasterizer does the same).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChannelMask {
    pub rgb: bool,
    pub alpha: bool,
}

pub fn channel_mask(packed: u32) -> ChannelMask {
    let alpha = packed & BLEND_WRITES_ALPHA != 0;
    ChannelMask { rgb: !(alpha && packed & BLEND_ALPHA_ONLY != 0), alpha }
}

/// How a draw changes its target's alpha channel, which on the PSP is the
/// stencil buffer (`alpha_write`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AlphaWrite {
    /// Left as it is: no stencil write (blending never touches it).
    Keep,
    /// A clear with the stencil bit: the vertex alpha is the stencil value,
    /// written as is. Exact.
    StencilClear,
    /// The fragment's alpha, as before this rule existed: for 565
    /// framebuffers (no stencil; what reads them back sees alpha 1) and for
    /// draws whose stencil ops write (the value should be the op's result:
    /// approximated, see `approximated`).
    Fragment { approximated: bool },
}

/// How the current draw writes the alpha channel (the stencil), after
/// PPSSPP (GPU/Common/GPUStateUtils.cpp: `IsStencilTestOutputDisabled`,
/// clear mode's alpha mask): stencil ops only act with the stencil test
/// on, an op other than KEEP and not all bits masked (ALPHA_MASK, set bits
/// keep). MTX Mototrax's races write it only with the frame's clear
/// (CLEAR 701, alpha 0), and its blur quad weights the sampled previous
/// frame by that alpha: with every draw writing alpha 0xFF our blur kept
/// 95 % of old frames (ghost pictures in the sky, slot 8).
pub fn alpha_write(ge: &GeState) -> AlphaWrite {
    if ge.register(op::FRAMEBUFFER_PIXEL_FORMAT) & 3 == 0 {
        return AlphaWrite::Fragment { approximated: false }; // 565: no stencil
    }
    if ge.is_clear_mode() {
        return if ge.clear_writes_stencil() { AlphaWrite::StencilClear } else { AlphaWrite::Keep };
    }
    let ops = ge.register(op::STENCIL_OP);
    let writes = ge.register(op::STENCIL_TEST_ENABLE) & 1 != 0
        && ge.register(op::ALPHA_MASK) & 0xFF != 0xFF
        && [0, 8, 16].iter().any(|shift| (ops >> shift) & 0xFF != 0); // 0 = KEEP
    if writes {
        AlphaWrite::Fragment { approximated: true }
    } else {
        AlphaWrite::Keep
    }
}

/// Whether a draw sampling one of our render targets (a PSP framebuffer)
/// sees the target's alpha, the PSP's stencil values. Not with the texture
/// alpha switched off (texture function "RGB", `texture_flags::IGNORE_ALPHA`):
/// MTX's motion blur samples the other framebuffer that way, weighted by its
/// vertex alpha alone; we sampled our target's alpha with it, which isn't
/// the PSP's (ghost pictures in the sky, slot 8), and since the targets
/// keep the stencil's 0 the blur would vanish. The display samples such
/// targets through a view with alpha 1.
pub fn samples_target_alpha(texture_flags: u16) -> bool {
    texture_flags & crate::frame::texture_flags::IGNORE_ALPHA == 0
}

/// Whether a draw's packed blend reads the destination alpha (a factor
/// DESTINATION_ALPHA or ONE_MINUS_DESTINATION_ALPHA).
pub fn reads_destination_alpha(packed: u32) -> bool {
    let reads = |factor: u32| factor == Factor::DestinationAlpha as u32 || factor == Factor::OneMinusDestinationAlpha as u32;
    packed & BLEND_ENABLED != 0 && packed & BLEND_ALPHA_ONLY == 0 && (reads(packed & 0xF) || reads((packed >> 4) & 0xF))
}

/// Frames the display keeps the targets' alpha exact (`channel_mask`)
/// after the game last read it (`samples_target_alpha`,
/// `reads_destination_alpha`).
pub const KEEP_TARGET_ALPHA_FRAMES: u32 = 120;

/// Whether frame `render_count` draws with the colour mask that keeps the
/// targets' alpha (the PSP's stencil), given the last frame that read it
/// (`last_read`, 0 = none yet). The mask costs the GPU: MTX's LA race
/// frame 12.3 -> 14.6 ms (paused redraw), and MTX never reads the alpha.
/// Without it every draw writes alpha, as before. The first frame that
/// reads it sees alpha written without the mask.
pub fn keep_target_alpha(last_read: u32, render_count: u32) -> bool {
    last_read != 0 && render_count.wrapping_sub(last_read) <= KEEP_TARGET_ALPHA_FRAMES
}

/// The `Draw::blend` bits for `alpha_write`'s answer (`channel_mask`).
pub fn alpha_write_bits(write: AlphaWrite, clear_writes_color: bool) -> u32 {
    match write {
        AlphaWrite::Keep => 0,
        AlphaWrite::StencilClear if !clear_writes_color => BLEND_WRITES_ALPHA | BLEND_ALPHA_ONLY,
        AlphaWrite::StencilClear | AlphaWrite::Fragment { .. } => BLEND_WRITES_ALPHA,
    }
}

impl Blend {
    /// Normal transparency: source * alpha + destination * (1 - alpha).
    pub const ALPHA: Blend = Blend {
        equation: Equation::Add,
        source: Factor::SourceAlpha,
        destination: Factor::OneMinusSourceAlpha,
        source_scale: None,
        double_source_alpha: false,
        approximated: false,
        fix_b: None,
    };

    /// Keeps the colour as it is (source * 0 + destination * 1): for draws
    /// that only write depth, like clears of the depth buffer alone.
    pub const KEEP_COLOR: Blend = Blend {
        equation: Equation::Add,
        source: Factor::Zero,
        destination: Factor::One,
        source_scale: None,
        double_source_alpha: false,
        approximated: false,
        fix_b: None,
    };

    pub fn pack(&self) -> u32 {
        let fix_b = match self.fix_b {
            Some(fix) => {
                BLEND_FIX_B_ALPHA | (fix.grey as u32) << BLEND_FIX_B_SHIFT | (fix.premultiply as u32) << BLEND_PREMULTIPLY_SHIFT
            }
            None => 0,
        };
        BLEND_ENABLED | self.source as u32 | (self.destination as u32) << 4 | (self.equation as u32) << 8 | fix_b
    }
}

/// The blend to use for the current draw, or `None` to write the colour
/// as is (blending off).
pub fn translate(ge: &GeState) -> Option<Blend> {
    if ge.is_clear_mode() {
        return (!ge.clear_writes_color()).then_some(Blend::KEEP_COLOR);
    }
    if ge.register(op::COLOR_MASK) & 0x00FF_FFFF == 0x00FF_FFFF {
        return Some(Blend::KEEP_COLOR); // only depth (or stencil) written
    }
    if !ge.is_blend_enabled() {
        // With blending off the GE writes every pixel, also transparent
        // texels, unless the alpha test throws them away. Where the display
        // can't alpha-test the draw, normal blending stands in for it
        // (exact for fully transparent / opaque texels).
        let alpha_tested = ge.register(op::ALPHA_TEST_ENABLE) & 1 != 0;
        return (alpha_tested && !alpha_test_in_shader(ge)).then_some(Blend { approximated: true, ..Blend::ALPHA });
    }

    let mode = ge.register(op::BLEND_MODE);
    let mut approximated = false;

    let equation = match (mode >> 8) & 7 {
        0 => Equation::Add,
        1 => Equation::Subtract,
        2 => Equation::ReverseSubtract,
        3 => Equation::Min,
        4 => Equation::Max,
        _ => {
            approximated = true; // absolute difference
            Equation::Add
        }
    };
    // MIN and MAX ignore the factors (on the GE as in GL; PPSSPP passes them
    // through unused): One/One is exact on any GPU, also where the GE's
    // factors (destination alpha, FIX) would count as approximated. MTX's
    // menu: MAX with destination colour / alpha.
    if matches!(equation, Equation::Min | Equation::Max) {
        return Some(Blend {
            equation,
            source: Factor::One,
            destination: Factor::One,
            source_scale: None,
            double_source_alpha: false,
            approximated: false,
            fix_b: None,
        });
    }

    // Doubled source alpha (6: 2a, 7: 1 - 2a) on one side, and nothing on
    // the other side that needs the undoubled alpha (only colours, fixed
    // colours or the same doubled alpha): doubling the vertex alpha does it.
    const DOUBLED_SOURCE_ALPHA: [u32; 2] = [6, 7];
    const NO_SOURCE_ALPHA: [u32; 5] = [0, 1, 6, 7, 0xA];
    let (source_code, destination_code) = (mode & 0xF, (mode >> 4) & 0xF);
    let double_source_alpha = (DOUBLED_SOURCE_ALPHA.contains(&source_code) || DOUBLED_SOURCE_ALPHA.contains(&destination_code))
        && NO_SOURCE_ALPHA.contains(&source_code)
        && NO_SOURCE_ALPHA.contains(&destination_code);
    let doubled = |approximated: &mut bool, factor: Factor| {
        if double_source_alpha {
            factor
        } else {
            approximate(approximated, factor)
        }
    };

    let mut source_scale = None;
    let source_code = mode & 0xF;
    let mut source = match mode & 0xF {
        0 => Factor::DestinationColor,
        1 => Factor::OneMinusDestinationColor,
        2 => Factor::SourceAlpha,
        3 => Factor::OneMinusSourceAlpha,
        4 => approximate(&mut approximated, Factor::DestinationAlpha),
        5 => approximate(&mut approximated, Factor::OneMinusDestinationAlpha),
        6 => doubled(&mut approximated, Factor::SourceAlpha), // 2 x source alpha
        7 => doubled(&mut approximated, Factor::OneMinusSourceAlpha),
        8 => approximate(&mut approximated, Factor::DestinationAlpha),
        9 => approximate(&mut approximated, Factor::OneMinusDestinationAlpha),
        _ => match ge.register(op::BLEND_FIXED_A) & 0x00FF_FFFF {
            0 => Factor::Zero,
            0x00FF_FFFF => Factor::One,
            color => {
                source_scale = Some(color); // source * FIXA, exactly
                Factor::One
            }
        },
    };

    let mut fix_b_grey = None;
    let mut destination = match (mode >> 4) & 0xF {
        0 => Factor::SourceColor,
        1 => Factor::OneMinusSourceColor,
        2 => Factor::SourceAlpha,
        3 => Factor::OneMinusSourceAlpha,
        4 => approximate(&mut approximated, Factor::DestinationAlpha),
        5 => approximate(&mut approximated, Factor::OneMinusDestinationAlpha),
        6 => doubled(&mut approximated, Factor::SourceAlpha),
        7 => doubled(&mut approximated, Factor::OneMinusSourceAlpha),
        8 => approximate(&mut approximated, Factor::DestinationAlpha),
        9 => approximate(&mut approximated, Factor::OneMinusDestinationAlpha),
        _ => match ge.register(op::BLEND_FIXED_B) & 0x00FF_FFFF {
            0 => Factor::Zero,
            0x00FF_FFFF => Factor::One,
            color if ge.is_texture_enabled() && grey_of(color).is_some() => {
                // Exact through the program's alpha (`Blend::fix_b`), below.
                fix_b_grey = grey_of(color);
                Factor::SourceAlpha
            }
            color => {
                approximated = true;
                // The nearer of ZERO and ONE by average brightness.
                let sum = (color & 0xFF) + ((color >> 8) & 0xFF) + ((color >> 16) & 0xFF);
                if sum >= 3 * 128 { Factor::One } else { Factor::Zero }
            }
        },
    };

    // FIX B through the alpha: the source factor must not read the source
    // alpha on the GPU any more; the program applies it instead.
    let mut fix_b = None;
    if let Some(grey) = fix_b_grey {
        let premultiply = match source_code {
            2 => Some(Premultiply::SourceAlpha),
            3 => Some(Premultiply::OneMinusSourceAlpha),
            0 | 1 => Some(Premultiply::None), // destination colour: no alpha read
            code if code >= 0xA => Some(Premultiply::None), // FIX A: in the colour (source_scale) or 0 / 1
            _ => None, // destination alpha, doubled alpha: not this way
        };
        match premultiply {
            Some(premultiply) => {
                if premultiply != Premultiply::None {
                    source = Factor::One;
                }
                fix_b = Some(FixB { grey, premultiply });
            }
            None => {
                approximated = true;
                destination = if grey >= 128 { Factor::One } else { Factor::Zero };
            }
        }
    }

    Some(Blend { equation, source, destination, source_scale, double_source_alpha, approximated, fix_b })
}

/// The grey level of a fixed colour (RGB) whose channels are equal, up to a
/// difference of 2 (their average), or None.
fn grey_of(color: u32) -> Option<u8> {
    let (r, g, b) = (color & 0xFF, (color >> 8) & 0xFF, (color >> 16) & 0xFF);
    let (low, high) = (r.min(g).min(b), r.max(g).max(b));
    (high - low <= 2).then_some(((r + g + b + 1) / 3) as u8)
}

fn approximate(approximated: &mut bool, factor: Factor) -> Factor {
    *approximated = true;
    factor
}

/// `rgba` with its RGB multiplied by `scale` (RGB, 0..255 = 0..1).
pub fn scale_color(rgba: u32, scale: u32) -> u32 {
    let channel = |shift: u32| (((rgba >> shift) & 0xFF) * ((scale >> shift) & 0xFF) / 255) << shift;
    (rgba & 0xFF00_0000) | channel(0) | channel(8) | channel(16)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::Command;

    fn ge_with(registers: &[(u8, u32)]) -> GeState {
        let mut ge = GeState::new();
        for &(opcode, data) in registers {
            ge.set_register(Command(((opcode as u32) << 24) | data));
        }
        ge
    }

    /// MTX's Paradise Valley mountains: source + destination * FIX B (grey),
    /// which the GPU can't do with a fixed factor: FIX B goes into the alpha.
    #[test]
    fn grey_fixed_destination_goes_through_the_alpha() {
        let ge = ge_with(&[
            (op::ALPHA_BLEND_ENABLE, 1),
            (op::TEXTURE_MAP_ENABLE, 1),
            (op::BLEND_MODE, 0x0AA),
            (op::BLEND_FIXED_A, 0xFF_FFFF),
            (op::BLEND_FIXED_B, 0x60_6060),
        ]);
        let blend = translate(&ge).unwrap();
        assert_eq!((blend.source, blend.destination), (Factor::One, Factor::SourceAlpha));
        assert_eq!(blend.fix_b, Some(FixB { grey: 0x60, premultiply: Premultiply::None }));
        assert!(!blend.approximated);
        let packed = blend.pack();
        assert_eq!(packed & BLEND_FIX_B_ALPHA, BLEND_FIX_B_ALPHA);
        assert_eq!((packed >> BLEND_FIX_B_SHIFT) & 0xFF, 0x60);

        // source alpha as the source factor: applied by the program
        let ge = ge_with(&[
            (op::ALPHA_BLEND_ENABLE, 1),
            (op::TEXTURE_MAP_ENABLE, 1),
            (op::BLEND_MODE, 0x0A2),
            (op::BLEND_FIXED_B, 0x40_4141),
        ]);
        let blend = translate(&ge).unwrap();
        assert_eq!((blend.source, blend.destination), (Factor::One, Factor::SourceAlpha));
        assert_eq!(blend.fix_b.map(|f| f.premultiply), Some(Premultiply::SourceAlpha));

        // coloured FIX B, or no texture (no program of ours): approximated as before
        let ge = ge_with(&[
            (op::ALPHA_BLEND_ENABLE, 1),
            (op::TEXTURE_MAP_ENABLE, 1),
            (op::BLEND_MODE, 0x0AA),
            (op::BLEND_FIXED_B, 0x20_4080),
        ]);
        let blend = translate(&ge).unwrap();
        assert!(blend.fix_b.is_none() && blend.approximated);
    }

    #[test]
    fn normal_alpha_blending() {
        let ge = ge_with(&[(op::ALPHA_BLEND_ENABLE, 1), (op::BLEND_MODE, 0x032)]);
        assert_eq!(translate(&ge), Some(Blend::ALPHA));
    }

    #[test]
    fn additive_light_from_final_fantasy_battle_flash() {
        // source * source alpha + destination * FIXB(white)
        let ge = ge_with(&[(op::ALPHA_BLEND_ENABLE, 1), (op::BLEND_MODE, 0x0A2), (op::BLEND_FIXED_B, 0xFF_FFFF)]);
        let blend = translate(&ge).unwrap();
        assert_eq!((blend.source, blend.destination, blend.equation), (Factor::SourceAlpha, Factor::One, Equation::Add));
        assert!(!blend.approximated);
    }

    #[test]
    fn fixed_source_colour_is_multiplied_into_the_draw() {
        let ge = ge_with(&[
            (op::ALPHA_BLEND_ENABLE, 1),
            (op::BLEND_MODE, 0x03A),
            (op::BLEND_FIXED_A, 0x80_8080),
        ]);
        let blend = translate(&ge).unwrap();
        assert_eq!(blend.source, Factor::One);
        assert_eq!(blend.source_scale, Some(0x80_8080));
        assert_eq!(scale_color(0xFFFF_FFFF, 0x80_8080), 0xFF80_8080);
    }

    #[test]
    fn doubled_source_alpha_doubles_the_vertex_alpha() {
        // MTX Mototrax's menu panels: 2a * source + (1 - 2a) * destination.
        let ge = ge_with(&[(op::ALPHA_BLEND_ENABLE, 1), (op::BLEND_MODE, 0x076)]);
        let blend = translate(&ge).unwrap();
        assert_eq!((blend.source, blend.destination), (Factor::SourceAlpha, Factor::OneMinusSourceAlpha));
        assert!(blend.double_source_alpha && !blend.approximated);
        // Mixed with the plain source alpha: can't be done by doubling.
        let ge = ge_with(&[(op::ALPHA_BLEND_ENABLE, 1), (op::BLEND_MODE, 0x036)]);
        let blend = translate(&ge).unwrap();
        assert!(!blend.double_source_alpha && blend.approximated);
    }

    #[test]
    fn blending_off_writes_directly_unless_alpha_tested() {
        let ge = ge_with(&[(op::ALPHA_BLEND_ENABLE, 0)]);
        assert_eq!(translate(&ge), None);
        let ge = ge_with(&[(op::ALPHA_BLEND_ENABLE, 0), (op::ALPHA_TEST_ENABLE, 1)]);
        assert_eq!(translate(&ge).map(|blend| blend.source), Some(Factor::SourceAlpha));
    }

    /// Textured draws with all alpha bits compared are alpha-tested by the
    /// display's shader once it has the program; the rest keep blending as
    /// the stand-in.
    /// Slot 8's ghost sky: MTX's blur quad samples the other framebuffer
    /// with the texture alpha off (frame description flags=1): the PSP
    /// weights it by the vertex alpha alone, not by the stencil.
    #[test]
    fn targets_sampled_without_texture_alpha_are_seen_opaque() {
        use crate::frame::texture_flags::IGNORE_ALPHA;
        assert!(!samples_target_alpha(IGNORE_ALPHA));
        assert!(samples_target_alpha(0));
    }

    #[test]
    fn the_alpha_is_kept_exact_only_while_the_game_reads_it() {
        assert!(!keep_target_alpha(0, 500), "never read");
        assert!(keep_target_alpha(500, 500));
        assert!(keep_target_alpha(500, 500 + KEEP_TARGET_ALPHA_FRAMES));
        assert!(!keep_target_alpha(500, 501 + KEEP_TARGET_ALPHA_FRAMES));
        assert!(reads_destination_alpha(Blend { destination: Factor::OneMinusDestinationAlpha, ..Blend::ALPHA }.pack()));
        assert!(reads_destination_alpha(Blend { source: Factor::DestinationAlpha, ..Blend::ALPHA }.pack()));
        assert!(!reads_destination_alpha(Blend::ALPHA.pack()));
        assert!(!reads_destination_alpha(Blend { source: Factor::DestinationAlpha, ..Blend::ALPHA }.pack() & !BLEND_ENABLED));
    }

    #[test]
    fn the_shader_alpha_tests_textured_draws_with_a_full_mask() {
        let textured = ge_with(&[(op::ALPHA_TEST_ENABLE, 1), (op::ALPHA_TEST, 0xFF_00_06), (op::TEXTURE_MAP_ENABLE, 1)]);
        assert!(shader_can_alpha_test(&textured, true));
        assert!(!shader_can_alpha_test(&textured, false), "no program yet");
        let untextured = ge_with(&[(op::ALPHA_TEST_ENABLE, 1), (op::ALPHA_TEST, 0xFF_00_06)]);
        assert!(!shader_can_alpha_test(&untextured, true));
        let masked = ge_with(&[(op::ALPHA_TEST_ENABLE, 1), (op::ALPHA_TEST, 0xF0_00_06), (op::TEXTURE_MAP_ENABLE, 1)]);
        assert!(!shader_can_alpha_test(&masked, true));
        let untested = ge_with(&[(op::ALPHA_TEST, 0xFF_00_06), (op::TEXTURE_MAP_ENABLE, 1)]);
        assert!(!shader_can_alpha_test(&untested, true));
    }

    /// MTX's menu: MAX with destination colour and
    /// destination alpha factors counted as an approximated blend, though
    /// MIN/MAX ignore the factors. Exact as One/One.
    #[test]
    fn min_and_max_ignore_the_factors_and_are_exact() {
        for (mode, equation) in [(0xF34C40, Equation::Max), (0x0394, Equation::Min)] {
            let ge = ge_with(&[(op::ALPHA_BLEND_ENABLE, 1), (op::BLEND_MODE, mode), (op::BLEND_FIXED_A, 0x23_0B80)]);
            let blend = translate(&ge).expect("blended");
            assert_eq!((blend.equation, blend.source, blend.destination), (equation, Factor::One, Factor::One));
            assert!(!blend.approximated);
            assert_eq!((blend.source_scale, blend.fix_b), (None, None));
        }
    }
}
