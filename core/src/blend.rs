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

/// In `Draw::blend`: the draw's exact stencil state, as its index in
/// `Frame::stencil_states` + 1 in bits 22..28 (0: none, `stencil`).
pub const BLEND_STENCIL_SHIFT: u32 = 22;
pub const BLEND_STENCIL_MASK: u32 = 0x3F << BLEND_STENCIL_SHIFT;

/// The `Frame::stencil_states` index a packed blend word names, if any.
pub fn stencil_index(packed: u32) -> Option<usize> {
    (((packed & BLEND_STENCIL_MASK) >> BLEND_STENCIL_SHIFT) as usize).checked_sub(1)
}

/// A draw's stencil test and stencil write, done exactly on the target's
/// alpha (the PSP's stencil): the subset of the GE's stencil state that
/// comes down to "test the stored value, then write a constant or keep it"
/// (`stencil`). Packed: bits 0..8 the value written, bit 8 set if it is
/// written, bits 9..12 the compare function (PSP order, see `passes`),
/// bit 12 set if there is a test, bits 16..24 the reference (already
/// masked) and bits 24..32 the mask.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct StencilState(pub u32);

impl StencilState {
    const WRITES: u32 = 1 << 8;
    const TESTS: u32 = 1 << 12;

    pub fn new(test: Option<(u32, u8, u8)>, write: Option<u8>) -> StencilState {
        let mut packed = 0;
        if let Some(value) = write {
            packed |= value as u32 | Self::WRITES;
        }
        if let Some((function, reference, mask)) = test {
            packed |= Self::TESTS | (function & 7) << 9 | ((reference & mask) as u32) << 16 | (mask as u32) << 24;
        }
        StencilState(packed)
    }

    /// The value the draw writes where it draws (`None`: the alpha stays).
    pub fn write(self) -> Option<u8> {
        (self.0 & Self::WRITES != 0).then_some(self.0 as u8)
    }

    /// The test as (function, reference & mask, mask), if there is one.
    pub fn test(self) -> Option<(u32, u8, u8)> {
        (self.0 & Self::TESTS != 0).then_some(((self.0 >> 9) & 7, (self.0 >> 16) as u8, (self.0 >> 24) as u8))
    }

    /// Whether a pixel whose target alpha is `stored` passes the test: the
    /// GE compares the reference with the stored value, both masked
    /// (ppsspp-ref GPU/Software/DrawPixel.cpp `StencilTestPassed`: "ref <
    /// stencil" for LESS).
    pub fn passes(self, stored: u8) -> bool {
        let Some((function, reference, mask)) = self.test() else {
            return true;
        };
        let stored = stored & mask;
        match function {
            0 => false,
            1 => true,
            2 => reference == stored,
            3 => reference != stored,
            4 => reference < stored,
            5 => reference <= stored,
            6 => reference > stored,
            _ => reference >= stored,
        }
    }
}

/// What the GE's stencil state does to a draw (`stencil`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stencil {
    /// No stencil test (or no stencil: 565, clear mode): `alpha_write` alone.
    Off,
    /// Done exactly by testing and writing the target's alpha.
    Exact(StencilState),
    /// Ops or masks the alpha can't do (INCR, DECR, INVERT, a partly masked
    /// write, writes on a failed test): as before, the fragment's alpha is
    /// written and nothing is tested.
    Approximated,
}

/// The stencil state of the current (non-clear) draw, as the GE runs it
/// (ppsspp-ref GPU/Software/DrawPixel.cpp `DrawPixelInline`): with the
/// test on, a failed test applies the "stencil fail" op and stops, a failed
/// depth test the "depth fail" op, else the "pass" op; KEEP leaves the
/// stored value, ZERO writes 0, REPLACE the (unmasked) reference; bits set
/// in ALPHA_MASK aren't written. Exact when the draw's stencil comes down
/// to a test and a constant written where it draws: fail ops KEEP (or
/// never run: test ALWAYS, depth test off or ALWAYS) and the pass op KEEP,
/// ZERO or REPLACE with all bits written (or none). God of War marks its
/// picture with REPLACE (refs 0, 0x10, 0x20, 0x2C) and tests it for the
/// depth-of-field blur (EQUAL 0x10, mask 0x10) and the bloom's bright pass.
pub fn stencil(ge: &GeState) -> Stencil {
    if ge.register(op::FRAMEBUFFER_PIXEL_FORMAT) & 3 == 0 || ge.is_clear_mode() || ge.register(op::STENCIL_TEST_ENABLE) & 1 == 0 {
        return Stencil::Off;
    }
    const KEEP: u32 = 0;
    const ZERO: u32 = 1;
    const REPLACE: u32 = 2;
    let test = ge.register(op::STENCIL_TEST);
    let (function, reference, mask) = (test & 7, (test >> 8) as u8, (test >> 16) as u8);
    let ops = ge.register(op::STENCIL_OP);
    let (stencil_fail, depth_fail, pass) = (ops & 0xFF, (ops >> 8) & 0xFF, (ops >> 16) & 0xFF);
    let write_mask = ge.register(op::ALPHA_MASK) & 0xFF; // set bits: not written
    let depth_can_fail = ge.register(op::DEPTH_TEST_ENABLE) & 1 != 0 && ge.register(op::DEPTH_TEST) & 7 != 1;
    let writes = write_mask != 0xFF;
    let fails_written = writes && ((function != 1 && stencil_fail != KEEP) || (depth_can_fail && depth_fail != KEEP));
    let write = match pass {
        _ if !writes => None,
        KEEP => None,
        ZERO => Some(0),
        REPLACE => Some(reference),
        _ => return Stencil::Approximated,
    };
    if fails_written || (write.is_some() && write_mask != 0) {
        return Stencil::Approximated;
    }
    let test = (function != 1).then_some((function, reference, mask));
    if test.is_none() && write.is_none() {
        return Stencil::Off;
    }
    Stencil::Exact(StencilState::new(test, write))
}

/// How the display does a draw's exact stencil state (`stencil_way`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StencilWay {
    /// Nothing to do (no state).
    None,
    /// Alpha 0 written (REPLACE 0, ZERO): GXM's alpha blend factors ZERO,
    /// ZERO; the colour's program and blending as before. (By the BLEND_FIX
    /// program instead, keeping unblended draws opaque to the GPU: GoW's
    /// gameplay 64.4 -> 67.8 ms GPU, slower; 2026-11-22.)
    Zero,
    /// A constant alpha from a textured BLEND_FIX program (its uBlendFix
    /// with x = value / 255, y = 0: colour unchanged, alpha the value) and
    /// GXM's alpha factors ONE, ZERO; the colour blending as before, as it
    /// doesn't read the source alpha.
    FixProgram(u8),
    /// The generic frame-read program (nativeres_textured_read_f.cg
    /// uStencilTest, uStencilWrite): a stencil test, or a value whose draw's
    /// blending needs the source alpha (or has no BLEND_FIX program:
    /// untextured, FIX B already through the alpha). It blends in the
    /// program, so the alpha is free for the value.
    FrameRead(StencilState),
}

/// The way for a draw with stencil state `state` (`Frame::stencil_state_of`),
/// packed blend `blend` and textured or not. God of War writes its marks
/// mostly unblended (`FixProgram`, or `Zero`), tests them in a few
/// full-screen passes (`FrameRead`).
pub fn stencil_way(state: Option<StencilState>, blend: u32, textured: bool) -> StencilWay {
    let Some(state) = state else {
        return StencilWay::None;
    };
    if state.test().is_some() {
        return StencilWay::FrameRead(state);
    }
    let reads_source_alpha = |factor: u32| {
        [Factor::SourceAlpha, Factor::OneMinusSourceAlpha].iter().any(|known| *known as u32 == factor)
    };
    let blended = blend & BLEND_ENABLED != 0 && blend & BLEND_ALPHA_ONLY == 0;
    let needs_source_alpha = blended && (reads_source_alpha(blend & 0xF) || reads_source_alpha((blend >> 4) & 0xF) || blend & BLEND_FIX_B_ALPHA != 0);
    match state.write() {
        None => StencilWay::None,
        Some(0) => StencilWay::Zero,
        Some(_) if !textured || needs_source_alpha => StencilWay::FrameRead(state),
        Some(value) => StencilWay::FixProgram(value),
    }
}

/// The frame-read program's stencil uniforms (uStencilTest, uStencilWrite,
/// nativeres_textured_read_f.cg) for `state` (`None`: no test, alpha as
/// blended): the stored values that pass as a range half a step out, as the
/// alpha test's uniform.
pub fn stencil_uniforms(state: Option<StencilState>) -> [[f32; 4]; 2] {
    let state = state.unwrap_or_default();
    let test = match state.test() {
        None => [-1.0, 256.0, 1.0, 0.0],
        Some((function, reference, mask)) => {
            let reference = reference as f32;
            let mask = mask as f32;
            // The stored (masked) values that pass "reference <op> stored".
            let (lo, hi, outside) = match function {
                0 => (1.0, 0.0, false),                // never: an empty range
                2 => (reference, reference, false),    // equal
                3 => (reference, reference, true),     // not equal
                4 => (reference + 1.0, 255.0, false),  // reference < stored
                5 => (reference, 255.0, false),        // reference <= stored
                6 => (0.0, reference - 1.0, false),    // reference > stored
                7 => (0.0, reference, false),          // reference >= stored
                _ => (0.0, 255.0, false),              // always
            };
            [lo - 0.5, hi + 0.5, if outside { -1.0 } else { 1.0 }, mask]
        }
    };
    let write = match state.write() {
        Some(value) => [value as f32, 1.0, 0.0, 0.0],
        None => [0.0; 4],
    };
    [test, write]
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

    /// God of War's gameplay (2026-11-22): the world writes its stencil
    /// marks with REPLACE (test ALWAYS, KEEP on depth fail), the
    /// depth-of-field blur tests EQUAL 0x10 under mask 0x10.
    #[test]
    fn god_of_wars_stencil_marks_and_tests_are_exact() {
        let mark = |reference: u32| {
            ge_with(&[
                (op::FRAMEBUFFER_PIXEL_FORMAT, 3),
                (op::STENCIL_TEST_ENABLE, 1),
                (op::STENCIL_TEST, 0x3C_0001 | reference << 8),
                (op::STENCIL_OP, 0x02_0002),
                (op::DEPTH_TEST_ENABLE, 1),
                (op::DEPTH_TEST, 5),
            ])
        };
        assert_eq!(stencil(&mark(0x20)), Stencil::Exact(StencilState::new(None, Some(0x20))));
        assert_eq!(stencil(&mark(0)), Stencil::Exact(StencilState::new(None, Some(0))));
        let blur = ge_with(&[
            (op::FRAMEBUFFER_PIXEL_FORMAT, 3),
            (op::STENCIL_TEST_ENABLE, 1),
            (op::STENCIL_TEST, 0x10_1002),
            (op::STENCIL_OP, 0),
        ]);
        let Stencil::Exact(state) = stencil(&blur) else { panic!("{:?}", stencil(&blur)) };
        assert_eq!((state.test(), state.write()), (Some((2, 0x10, 0x10)), None));
        assert!(state.passes(0x10) && state.passes(0x3F) && !state.passes(0x2C) && !state.passes(0));
        // GREATER 0x10, REPLACE on pass: written only where 0x10 > stored.
        let over = ge_with(&[
            (op::FRAMEBUFFER_PIXEL_FORMAT, 3),
            (op::STENCIL_TEST_ENABLE, 1),
            (op::STENCIL_TEST, 0xFF_1006),
            (op::STENCIL_OP, 0x02_0000),
            (op::DEPTH_TEST_ENABLE, 1),
            (op::DEPTH_TEST, 5),
        ]);
        let Stencil::Exact(state) = stencil(&over) else { panic!() };
        assert_eq!(state.write(), Some(0x10));
        assert!(state.passes(0) && !state.passes(0x10) && !state.passes(0x20));
    }

    #[test]
    fn stencil_ops_the_alpha_cant_do_stay_approximated() {
        let base = [(op::FRAMEBUFFER_PIXEL_FORMAT, 3), (op::STENCIL_TEST_ENABLE, 1), (op::STENCIL_TEST, 0xFF_2001)];
        let with = |extra: &[(u8, u32)]| stencil(&ge_with(&[&base[..], extra].concat()));
        // INCR on pass; REPLACE on a depth fail that can happen; a partly
        // masked write; a write when an EQUAL test fails.
        assert_eq!(with(&[(op::STENCIL_OP, 0x04_0000)]), Stencil::Approximated);
        assert_eq!(with(&[(op::STENCIL_OP, 0x02_0200), (op::DEPTH_TEST_ENABLE, 1), (op::DEPTH_TEST, 4)]), Stencil::Approximated);
        assert_eq!(with(&[(op::STENCIL_OP, 0x02_0000), (op::ALPHA_MASK, 0x0F)]), Stencil::Approximated);
        assert_eq!(with(&[(op::STENCIL_TEST, 0xFF_2002), (op::STENCIL_OP, 0x02_0002)]), Stencil::Approximated);
        // A depth-fail op without a depth test never runs; nothing written,
        // nothing tested; 565 has no stencil; all bits masked: kept.
        assert!(matches!(with(&[(op::STENCIL_OP, 0x02_0200)]), Stencil::Exact(_)));
        assert_eq!(with(&[(op::STENCIL_OP, 0)]), Stencil::Off);
        assert_eq!(with(&[(op::STENCIL_OP, 0x02_0000), (op::FRAMEBUFFER_PIXEL_FORMAT, 0)]), Stencil::Off);
        assert_eq!(with(&[(op::STENCIL_OP, 0x02_0000), (op::ALPHA_MASK, 0xFF)]), Stencil::Off);
    }

    #[test]
    fn stencil_ways_follow_what_the_blend_needs_of_the_alpha() {
        let write = |value| Some(StencilState::new(None, Some(value)));
        let off = 0;
        let normal = Blend::ALPHA.pack();
        let fixed = BLEND_ENABLED | Factor::One as u32 | (Factor::One as u32) << 4; // additive (FIX A, FIX B white)
        assert_eq!(stencil_way(None, normal, true), StencilWay::None);
        assert_eq!(stencil_way(write(0), normal, true), StencilWay::Zero);
        assert_eq!(stencil_way(write(0x20), off, true), StencilWay::FixProgram(0x20));
        assert_eq!(stencil_way(write(0x20), fixed, true), StencilWay::FixProgram(0x20));
        // The blend takes the source alpha, FIX B rides in it, or no
        // BLEND_FIX program (untextured): the frame-read program.
        assert_eq!(stencil_way(write(0x10), normal, true), StencilWay::FrameRead(StencilState::new(None, Some(0x10))));
        assert!(matches!(stencil_way(write(0x10), fixed | BLEND_FIX_B_ALPHA, true), StencilWay::FrameRead(_)));
        assert!(matches!(stencil_way(write(0x10), off, false), StencilWay::FrameRead(_)));
        // A test always reads the target.
        let tested = StencilState::new(Some((2, 0x10, 0x10)), None);
        assert_eq!(stencil_way(Some(tested), fixed, true), StencilWay::FrameRead(tested));
    }

    /// The program's range test (uStencilTest) agrees with `passes` for
    /// every function, reference, mask sample and stored value.
    #[test]
    fn the_stencil_uniform_range_is_the_stencil_test() {
        for function in 0..8 {
            for (reference, mask) in [(0x10, 0x10), (0x20, 0xFF), (0, 0xFF), (0xFF, 0xFF), (0x2C, 0x3C)] {
                let state = StencilState::new(Some((function, reference, mask)), Some(7));
                let [test, write] = stencil_uniforms(Some(state));
                assert_eq!(write, [7.0, 1.0, 0.0, 0.0]);
                for stored in 0..=255u8 {
                    let masked = (stored & test[3] as u8) as f32;
                    let program_passes = (masked - test[0]) * (test[1] - masked) * test[2] >= 0.0;
                    assert_eq!(program_passes, state.passes(stored), "function {function} ref {reference:x} mask {mask:x} stored {stored:x}");
                }
            }
        }
        // No state: everything passes, alpha as blended.
        let [test, write] = stencil_uniforms(None);
        assert!((0..=255).all(|stored| ((stored & test[3] as u32) as f32 - test[0]) * (test[1] - (stored & test[3] as u32) as f32) * test[2] >= 0.0));
        assert_eq!(write, [0.0; 4]);
    }

    #[test]
    fn the_stencil_state_index_rides_in_the_blend_word() {
        assert_eq!(stencil_index(0), None);
        assert_eq!(stencil_index(BLEND_ENABLED | 1 << BLEND_STENCIL_SHIFT), Some(0));
        assert_eq!(stencil_index(63 << BLEND_STENCIL_SHIFT), Some(62));
        // Clear of the other fields.
        let used = BLEND_ENABLED | BLEND_WRITES_ALPHA | BLEND_ALPHA_ONLY | BLEND_FIX_B_ALPHA | 0xFF << BLEND_FIX_B_SHIFT | 3 << BLEND_PREMULTIPLY_SHIFT | 0x7FF;
        assert_eq!(used & BLEND_STENCIL_MASK, 0);
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
