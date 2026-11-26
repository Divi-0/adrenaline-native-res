//! Triangles whose source alpha is 0 everywhere, in draws that then keep
//! the target.
//!
//! The GE blends "source alpha, one minus source alpha, add" (normal) and
//! "source alpha, one, add" (additive) from terms ((2c + 1)(2f + 1)) >> 10
//! (`frame::ge_blend_term`). With a source alpha of 0 the source's term is
//! (2c + 1) >> 10 = 0 for every colour, and the target's term has factor 255
//! either way, which gives the target's colour itself: the fragment changes
//! nothing. GXM's own blend (s x 0 + d x 1) and the frame-read programs
//! (nativeres_textured_read_f.cg READ_NORMAL, READ_ADD) come out the same.
//!
//! A fragment's alpha is the texel's times the vertex's, (p + 1) t >> 8
//! (`GeShade`): 0 for p = 0 whatever the texel. The vertex alpha between
//! three corners of alpha 0 is 0 (0 b0 + 0 b1 + 0 b2 in any float order).
//! So a triangle whose three corners end up with alpha 0 draws nothing, as
//! long as the draw writes nothing but its blended colour: no depth, no
//! stencil. Such a triangle is left out; God of War's cloud layer is a dome
//! of ~100 triangles that fades to alpha 0 over most of the sky (62 % of the
//! layer's fragments in its first gameplay scene, ~0.9 screens a frame;
//! DESIGN "seventy-eighth session").
//!
//! - `keeps_target`: the draw's blend and state.
//! - `raw_zero_stays_zero`, `RawAlphas`: raw vertices (in PSP RAM or the
//!   raw arena), whose final alpha the vertex program works out from the
//!   vertex's own (`RawColors::color`).
//! - `cull_list`, `cull_strip_runs`: the draw's triangles without those.
//! - `Memo`: which draws in PSP RAM (not cached) are worth reading at all.

use crate::blend::{Equation, Factor, BLEND_ALPHA_ONLY, BLEND_ENABLED, BLEND_FIX_B_ALPHA, BLEND_WRITES_ALPHA};
use crate::frame::{draw_flags, test_flags, Draw, Frame, RawColors};
use crate::raw_vertex::lighting;
use crate::vertex::VertexFormat;

/// True if a fragment of `draw` whose source alpha is 0 leaves the target
/// exactly as it is: an 8-bit target (a 16-bit one is dithered and cut
/// again), blended "source alpha, one minus source alpha or one, add", and
/// nothing else written: no depth, no alpha (the PSP's stencil), no logic
/// op. Fog, colour doubling and a separate specular colour only change the
/// source's colour, which factor 0 drops. An alpha or stencil test can only
/// kill the fragment, which keeps the target too.
pub fn keeps_target(frame: &Frame, draw: &Draw) -> bool {
    let target = (draw.flags & draw_flags::TARGET_FORMAT_MASK) >> draw_flags::TARGET_FORMAT_SHIFT;
    let writes_depth = draw.tests & test_flags::DEPTH_TEST != 0 && draw.tests & test_flags::DEPTH_WRITE_DISABLED == 0;
    let (source, destination, equation) = (draw.blend & 0xF, (draw.blend >> 4) & 0xF, (draw.blend >> 8) & 7);
    draw.flags & draw_flags::BLEND != 0
        && draw.flags & draw_flags::DITHER == 0
        && target == 0
        && draw.blend & BLEND_ENABLED != 0
        && draw.blend & (BLEND_FIX_B_ALPHA | BLEND_WRITES_ALPHA | BLEND_ALPHA_ONLY) == 0
        && equation == Equation::Add as u32
        && source == Factor::SourceAlpha as u32
        && (destination == Factor::OneMinusSourceAlpha as u32 || destination == Factor::One as u32)
        && draw.tests & test_flags::LOGIC_OP == 0
        && !writes_depth
        && frame.stencil_state_of(draw).is_none_or(|state| state.write().is_none())
}

/// True if a raw vertex whose own alpha is 0 ends up with alpha 0 under
/// colour rules `colors` (`RawColors::color`: lit without lights
/// ((2 x 0 + 1)(2a + 1)) >> 10 = 0, doubled 0; not the texture function
/// "replace" with the texture's alpha, not a draw made opaque). The rules
/// take the alpha by itself, so one colour of alpha 0 answers for all. Lit
/// by the GPU's lights: no (their alpha is not looked at here).
pub fn raw_zero_stays_zero(colors: &RawColors) -> bool {
    colors.rules[0] != lighting::LIGHTS && colors.color(Some(0x00FF_FFFF)) >> 24 == 0
}

/// The alphas of raw vertices of one format, back to back in `bytes` (the
/// draw's window: vertex 0 at the start).
#[derive(Clone, Copy)]
pub struct RawAlphas<'a> {
    bytes: &'a [u8],
    stride: usize,
    /// The byte holding the alpha's bits, from the vertex's start, and its mask.
    at: usize,
    mask: u8,
}

impl<'a> RawAlphas<'a> {
    /// None where no vertex can have alpha 0 or it isn't the vertex's own:
    /// no colour (the material's), 565 (opaque), morphed vertices (their
    /// colours are mixed).
    pub fn new(format: VertexFormat, bytes: &'a [u8]) -> Option<Self> {
        if format.morph_count() != 1 {
            return None;
        }
        let layout = format.layout();
        let color = layout.color_offset? as usize;
        // The alpha's bits, little endian: 5551 bit 15, 4444 bits 12..15, 8888 the fourth byte.
        let (at, mask) = match format.color_format_bits() {
            5 => (color + 1, 0x80),
            6 => (color + 1, 0xF0),
            7 => (color + 3, 0xFF),
            _ => return None,
        };
        let stride = layout.size as usize;
        (stride > at).then_some(Self { bytes, stride, at, mask })
    }

    /// One vertex's size in bytes.
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// The same vertices' alphas in `bytes` (a copy of some of them: vertex
    /// 0 at its start).
    pub fn over<'b>(&self, bytes: &'b [u8]) -> RawAlphas<'b> {
        RawAlphas { bytes, stride: self.stride, at: self.at, mask: self.mask }
    }

    /// True if vertex number `vertex` has alpha 0 (a vertex outside the
    /// bytes: no).
    #[inline]
    pub fn is_zero(&self, vertex: u16) -> bool {
        self.bytes.get(vertex as usize * self.stride + self.at).is_some_and(|byte| byte & self.mask == 0)
    }
}

/// What a cull left: how many indices it wrote, and the triangles kept and
/// left out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Culled {
    pub indices: usize,
    pub kept: u32,
    pub culled: u32,
}

/// The triangles of triangle list `list` without those whose three corners
/// all are `zero`, written to `out` in their order. None if none goes (the
/// list stays as it is) or `out` is shorter than the list.
pub fn cull_list(list: &[u16], mut zero: impl FnMut(u16) -> bool, out: &mut [u16]) -> Option<Culled> {
    if out.len() < list.len() {
        return None;
    }
    let (mut indices, mut culled) = (0, 0);
    for triangle in list.chunks_exact(3) {
        if zero(triangle[0]) && zero(triangle[1]) && zero(triangle[2]) {
            culled += 1;
        } else {
            out[indices..indices + 3].copy_from_slice(triangle);
            indices += 3;
        }
    }
    (culled > 0).then_some(Culled { indices, kept: (indices / 3) as u32, culled })
}

/// Triangle-strip runs (`[type, count, first vertex]` each, all strips:
/// `frame::stitched_strip_count`) as one triangle strip, like
/// `frame::stitched_strip`, without the triangles whose three corners all
/// are `zero`: the others in their order, each with the winding it had.
///
/// A strip turns every second triangle around, so a stretch of kept
/// triangles that began at triangle k of its run must begin at a place of
/// k's parity. Stretches are joined as the runs are: the last index before
/// the join and the first after it repeated (the first once more where the
/// parity asks for it; a strip that begins with an odd stretch begins with
/// its first index twice). Every triangle across a join has a repeated
/// corner and draws nothing.
///
/// None if no triangle goes (the draw's strip stays as it is) or the result
/// doesn't fit `out` (single kept triangles between left-out ones take up
/// to six indices each: where that is longer than the strip was, the cull
/// isn't worth its vertices).
pub fn cull_strip_runs(runs: &[u16], mut zero: impl FnMut(u16) -> bool, out: &mut [u16]) -> Option<Culled> {
    let (mut length, mut kept, mut culled) = (0usize, 0u32, 0u32);
    // Appends the stretch of triangles `first..=last` of the run starting at vertex `base`.
    let mut append = |length: &mut usize, base: u16, first: u16, last: u16| -> Option<()> {
        let start = base.wrapping_add(first);
        let joins = match *length {
            0 => (first & 1) as usize,
            before => 2 + ((before + first as usize) & 1),
        };
        let count = (last - first) as usize + 3;
        let end = *length + joins + count;
        if end > out.len() {
            return None;
        }
        if *length > 0 {
            out[*length] = out[*length - 1];
            out[*length + 1..*length + joins].fill(start);
        } else {
            out[..joins].fill(start);
        }
        for (k, slot) in out[*length + joins..end].iter_mut().enumerate() {
            *slot = start.wrapping_add(k as u16);
        }
        *length = end;
        Some(())
    };
    for run in runs.chunks_exact(3).filter(|run| run[1] >= 3) {
        let (count, base) = (run[1], run[2]);
        // The corners of triangle k are vertices k, k + 1, k + 2 of the run.
        let (mut zero_0, mut zero_1) = (zero(base), zero(base.wrapping_add(1)));
        // The first triangle of the stretch of kept ones being gathered.
        let mut open = None;
        for k in 0..count - 2 {
            let zero_2 = zero(base.wrapping_add(k + 2));
            let gone = zero_0 && zero_1 && zero_2;
            (zero_0, zero_1) = (zero_1, zero_2);
            if gone {
                culled += 1;
                if let Some(first) = open.take() {
                    append(&mut length, base, first, k - 1)?;
                }
            } else {
                kept += 1;
                open.get_or_insert(k);
            }
        }
        if let Some(first) = open {
            append(&mut length, base, first, count - 3)?;
        }
    }
    (culled > 0).then_some(Culled { indices: length, kept, culled })
}

/// Which raw draws are worth reading for their alphas.
///
/// A raw draw's vertices are in PSP RAM (`draw_flags::RAW_PSP_RAM`), which
/// is not cached (a single load costs ~0.8 us under load), or in the raw
/// arena, cold on the display thread. Reading the alphas of God of War's
/// ~45 blended draws cost the display thread 3.1-3.5 ms a frame
/// (2026-11-26), and half of those draws have no such triangle at all. So a
/// draw (`key`: its vertices' PSP address, or its texture's for one in the
/// arena, and its index count) that lost next to none is asked again only
/// some frames later; one that lost some is read every frame, since its
/// alphas may change. The memo only decides whether a draw is read: what is
/// left out is always decided from the bytes of the frame drawn.
pub struct Memo {
    entries: [MemoEntry; Memo::SLOTS],
}

#[derive(Clone, Copy)]
struct MemoEntry {
    address: u32,
    indices: u32,
    /// Frames to let pass before the draw is read again.
    wait: u8,
}

impl Memo {
    /// Sets of `WAYS` entries (draws whose keys fall into one set); only
    /// waiting draws take an entry. (One entry per key threw God of War's
    /// ~45 draws out of each other's places: a third of them were read
    /// every frame.)
    const SETS: usize = 32;
    const WAYS: usize = 4;
    const SLOTS: usize = Self::SETS * Self::WAYS;
    /// A draw is worth reading every frame if at least one in `WORTH` of
    /// its triangles went. No more than that can be asked: the CPU cannot
    /// tell how large they are on the screen (two glow draws of God of
    /// War's title lose ~25 of ~300 triangles each, 2 ms of GPU time;
    /// counting only draws that lose many, its cloud layer's ~800, kept
    /// 7.2 of the title's 9.2 ms).
    const WORTH: u32 = 16;
    /// The frames a draw not worth it waits: `MIN_WAIT` and up to 31 more,
    /// by its address, so that a scene's draws don't all come back in the
    /// same frame.
    const MIN_WAIT: u8 = 24;

    pub const fn new() -> Self {
        Self { entries: [MemoEntry { address: 0, indices: 0, wait: 0 }; Self::SLOTS] }
    }

    fn hash(address: u32) -> u32 {
        address.wrapping_mul(0x9E37_79B1)
    }

    /// The entries of the set `address` falls into.
    fn set(&mut self, address: u32) -> &mut [MemoEntry] {
        let first = (Self::hash(address) >> 27) as usize * Self::WAYS; // the top 5 bits: `SETS`
        &mut self.entries[first..first + Self::WAYS]
    }

    /// True if the draw with key `address` and `indices` indices is to be
    /// read now; false (and one frame less to wait) while it waits.
    pub fn due(&mut self, address: u32, indices: u32) -> bool {
        match self.set(address).iter_mut().find(|entry| entry.wait > 0 && entry.address == address && entry.indices == indices) {
            Some(entry) => {
                entry.wait -= 1;
                false
            }
            None => true,
        }
    }

    /// What reading that draw gave: `culled` triangles left out, `kept` not.
    pub fn note(&mut self, address: u32, indices: u32, culled: u32, kept: u32) {
        let worth = culled.saturating_mul(Self::WORTH) >= culled.saturating_add(kept).max(1);
        let wait = if worth { 0 } else { Self::MIN_WAIT + ((Self::hash(address) >> 12) & 31) as u8 };
        let set = self.set(address);
        // Its own entry, else a free one (nobody waiting in it), else the one closest to its next read.
        let own = set.iter().position(|entry| entry.address == address && entry.indices == indices);
        let free = || set.iter().position(|entry| entry.wait == 0);
        let soonest = || (0..set.len()).min_by_key(|&k| set[k].wait).unwrap_or(0);
        match own {
            Some(place) => set[place].wait = wait,
            None if worth => {}
            None => {
                let place = free().unwrap_or_else(soonest);
                set[place] = MemoEntry { address, indices, wait };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blend::Blend;
    use crate::command::primitive::TRIANGLE_STRIP;
    use crate::frame::{ge_blend_term, stitched_strip, stitched_strip_count};
    use crate::raw_vertex::replace;

    #[test]
    fn a_source_of_alpha_0_keeps_the_target_in_the_ges_blend() {
        for source in 0..=255 {
            for target in 0..=255 {
                // Normal: source x alpha + target x (255 - alpha); additive: source x alpha + target x 255.
                assert_eq!(ge_blend_term(source, 0) + ge_blend_term(target, 255 - 0), target, "source {source}, target {target}");
                assert_eq!((ge_blend_term(source, 0) + ge_blend_term(target, 255)).min(255), target);
            }
        }
    }

    #[test]
    fn a_vertex_alpha_of_0_gives_a_fragment_alpha_of_0() {
        // GeShade: (p + 1) t >> 8 with the vertex alpha p and the texel's t.
        for texel in 0..=255u32 {
            assert_eq!(((0 + 1) * texel) >> 8, 0);
        }
    }

    fn blended(source: Factor, destination: Factor, equation: Equation) -> Draw {
        Draw {
            flags: draw_flags::TEXTURED | draw_flags::BLEND | draw_flags::FOG | draw_flags::DOUBLE_COLOR,
            blend: Blend { equation, source, destination, ..Blend::ALPHA }.pack(),
            tests: test_flags::DEPTH_TEST | test_flags::DEPTH_WRITE_DISABLED | test_flags::ALPHA_TEST,
            ..Default::default()
        }
    }

    #[test]
    fn normal_and_additive_blends_that_write_only_colour_keep_the_target() {
        let frame = crate::frame::new_boxed_frame();
        let normal = blended(Factor::SourceAlpha, Factor::OneMinusSourceAlpha, Equation::Add);
        assert!(keeps_target(&frame, &normal));
        assert!(keeps_target(&frame, &blended(Factor::SourceAlpha, Factor::One, Equation::Add)));
        // Other factors and equations: the source's term or the target's is not the identity.
        assert!(!keeps_target(&frame, &blended(Factor::One, Factor::OneMinusSourceAlpha, Equation::Add)));
        assert!(!keeps_target(&frame, &blended(Factor::SourceAlpha, Factor::Zero, Equation::Add)));
        assert!(!keeps_target(&frame, &blended(Factor::SourceAlpha, Factor::OneMinusSourceColor, Equation::Add)));
        assert!(!keeps_target(&frame, &blended(Factor::SourceAlpha, Factor::One, Equation::Subtract)));
        assert!(!keeps_target(&frame, &blended(Factor::SourceAlpha, Factor::One, Equation::Min)));
        // Not blended.
        assert!(!keeps_target(&frame, &Draw { flags: normal.flags & !draw_flags::BLEND, ..normal }));
        assert!(!keeps_target(&frame, &Draw { blend: normal.blend & !BLEND_ENABLED, ..normal }));
        // A 16-bit target (cut again, dithered) or dither.
        for format in [draw_flags::TARGET_565, draw_flags::TARGET_5551, draw_flags::TARGET_4444] {
            assert!(!keeps_target(&frame, &Draw { flags: normal.flags | format << draw_flags::TARGET_FORMAT_SHIFT, ..normal }));
        }
        assert!(!keeps_target(&frame, &Draw { flags: normal.flags | draw_flags::DITHER, ..normal }));
        // Depth written, the alpha written, FIX B, alpha only, a logic op.
        assert!(!keeps_target(&frame, &Draw { tests: test_flags::DEPTH_TEST, ..normal }));
        for bit in [BLEND_WRITES_ALPHA, BLEND_FIX_B_ALPHA, BLEND_ALPHA_ONLY] {
            assert!(!keeps_target(&frame, &Draw { blend: normal.blend | bit, ..normal }));
        }
        assert!(!keeps_target(&frame, &Draw { tests: normal.tests | test_flags::LOGIC_OP, ..normal }));
        // No depth test at all: nothing written either.
        assert!(keeps_target(&frame, &Draw { tests: 0, ..normal }));
    }

    #[test]
    fn raw_rules_that_keep_an_alpha_of_0() {
        let plain = RawColors { material: [255.0; 4], ..Default::default() };
        assert!(raw_zero_stays_zero(&plain));
        // Lit without lights: ((2 x 0 + 1)(2a + 1)) >> 10 = 0 for every ambient alpha.
        for lit in [lighting::VERTEX_MATERIAL] {
            let colors = RawColors { ambient: [255.0; 4], rules: [lit, 0.0, 0.0, 0.0], ..plain };
            assert!(raw_zero_stays_zero(&colors));
        }
        // The material's alpha instead of the vertex's, lights, opaque, replace with the texture's alpha.
        assert!(!raw_zero_stays_zero(&RawColors { ambient: [255.0; 4], rules: [lighting::MATERIAL, 0.0, 0.0, 0.0], ..plain }));
        assert!(!raw_zero_stays_zero(&RawColors { rules: [lighting::LIGHTS, 0.0, 0.0, 0.0], ..plain }));
        assert!(!raw_zero_stays_zero(&RawColors { rules: [0.0, 1.0, 0.0, 0.0], ..plain }));
        assert!(!raw_zero_stays_zero(&RawColors { rules: [0.0, 0.0, replace::ALL, 0.0], ..plain }));
        // Replace keeping the vertex alpha, doubled alpha, FIX A (colour only): 0 stays.
        assert!(raw_zero_stays_zero(&RawColors { rules: [0.0, 0.0, replace::RGB, 1.0], ..plain }));
        assert!(raw_zero_stays_zero(&RawColors { scale: [128.0, 128.0, 128.0, 1.0], ..plain }));
        // And whatever the rules say for alpha 0 they say for every colour with it.
        for rgb in [0x00_0000, 0x12_3456, 0xFF_FFFF] {
            assert_eq!(plain.color(Some(rgb)) >> 24, 0);
        }
    }

    /// Vertex format bits: texture coordinates 16-bit (2), colour `color` (bits 2..4), position float (3 << 7).
    fn format(color: u32) -> VertexFormat {
        VertexFormat(2 | color << 2 | 3 << 7)
    }

    #[test]
    fn raw_alphas_read_each_colour_format() {
        for (color, zero, some) in [(7u32, 0x00FF_FFFFu32, 0x0100_0000u32), (6, 0x0FFF, 0x1000), (5, 0x7FFF, 0x8000)] {
            let format = format(color);
            let layout = format.layout();
            let (stride, at) = (layout.size as usize, layout.color_offset.unwrap() as usize);
            let mut bytes = vec![0xAAu8; 3 * stride];
            for (vertex, value) in [zero, some, zero].into_iter().enumerate() {
                let field = &mut bytes[vertex * stride + at..];
                if color == 7 {
                    field[..4].copy_from_slice(&value.to_le_bytes());
                } else {
                    field[..2].copy_from_slice(&(value as u16).to_le_bytes());
                }
            }
            let alphas = RawAlphas::new(format, &bytes).unwrap();
            assert_eq!([0, 1, 2, 3].map(|vertex| alphas.is_zero(vertex)), [true, false, true, false], "colour format {color}");
            // A copy of vertices 1 and 2 alone.
            assert_eq!(alphas.stride(), stride);
            let copy = alphas.over(&bytes[stride..]);
            assert_eq!([0, 1, 2].map(|vertex| copy.is_zero(vertex)), [false, true, false]);
            // What the decoder makes of the same bytes.
            assert_eq!(crate::opaque_source::lowest_vertex_alpha(format, &bytes[..stride]), Some(0));
            assert_ne!(crate::opaque_source::lowest_vertex_alpha(format, &bytes[stride..2 * stride]), Some(0));
        }
        // No colour, 565 (no alpha), morphed vertices: nothing to tell.
        assert!(RawAlphas::new(format(0), &[0; 64]).is_none());
        assert!(RawAlphas::new(format(4), &[0; 64]).is_none());
        assert!(RawAlphas::new(VertexFormat(format(7).0 | 1 << 18), &[0; 64]).is_none());
    }

    #[test]
    fn a_list_loses_its_zero_triangles() {
        let zero = |vertex: u16| vertex < 4;
        let list = [0, 1, 2, 2, 3, 4, 1, 2, 3, 7, 8, 9];
        let mut out = [0u16; 12];
        assert_eq!(cull_list(&list, zero, &mut out), Some(Culled { indices: 6, kept: 2, culled: 2 }));
        assert_eq!(out[..6], [2, 3, 4, 7, 8, 9]);
        // Nothing to leave out; no room.
        assert_eq!(cull_list(&[2, 3, 4], zero, &mut out), None);
        assert_eq!(cull_list(&list, zero, &mut out[..11]), None);
        // All of them.
        assert_eq!(cull_list(&[0, 1, 2, 3, 2, 1], zero, &mut out), Some(Culled { indices: 0, kept: 0, culled: 2 }));
    }

    /// A triangle strip's triangles as the GPU draws them (every second one
    /// turned around), without those with a repeated corner.
    fn strip_triangles(strip: &[u16]) -> Vec<[u16; 3]> {
        strip
            .windows(3)
            .enumerate()
            .map(|(i, w)| if i % 2 == 0 { [w[0], w[1], w[2]] } else { [w[1], w[0], w[2]] })
            .filter(|t| t[0] != t[1] && t[1] != t[2] && t[0] != t[2])
            .collect()
    }

    /// Culls `runs` with `zero` (with room for any result) and checks the
    /// result against the stitched strip's triangles without the zero ones;
    /// returns it.
    fn check_strip(runs: &[u16], zero: impl Fn(u16) -> bool + Copy) -> Option<Culled> {
        let whole: Vec<u16> = stitched_strip(runs).collect();
        assert_eq!(stitched_strip_count(runs).unwrap_or(0) as usize, whole.len());
        let mut out = vec![0xFFFFu16; 6 * whole.len() + 8];
        let result = cull_strip_runs(runs, zero, &mut out);
        let all = strip_triangles(&whole);
        let wanted: Vec<[u16; 3]> = all.iter().copied().filter(|t| !t.iter().all(|&vertex| zero(vertex))).collect();
        if let Some(culled) = result {
            assert_eq!(strip_triangles(&out[..culled.indices]), wanted, "runs {runs:?}");
            assert_eq!(culled.kept as usize, wanted.len());
            assert_eq!((culled.kept + culled.culled) as usize, all.len());
            assert!(culled.culled > 0);
        }
        result
    }

    #[test]
    fn a_strip_loses_its_zero_triangles_and_keeps_the_windings() {
        let strip = |count: u16, base: u16| [TRIANGLE_STRIP as u16, count, base];
        // One run of 10 vertices (8 triangles); zero vertices 0..=4: triangles 0, 1, 2 go.
        let runs = strip(10, 0);
        assert_eq!(check_strip(&runs, |v| v <= 4), Some(Culled { indices: 1 + 7, kept: 5, culled: 3 }));
        // Triangles 0 and 1 go: the rest begins at an even triangle, no extra index.
        assert_eq!(check_strip(&runs, |v| v <= 3), Some(Culled { indices: 8, kept: 6, culled: 2 }));
        // The middle goes: two stretches and a join. Of two indices where the
        // second stretch's first triangle (5, odd) lands at an odd place by
        // itself (5 + 2), of three where it doesn't (triangle 5 after 6 + 2).
        assert_eq!(check_strip(&runs, |v| (3..=6).contains(&v)), Some(Culled { indices: 5 + 2 + 5, kept: 6, culled: 2 }));
        assert_eq!(check_strip(&runs, |v| (4..=7).contains(&v)), Some(Culled { indices: 6 + 2 + 4, kept: 6, culled: 2 }));
        assert_eq!(check_strip(&runs, |v| (4..=6).contains(&v)), Some(Culled { indices: 6 + 3 + 5, kept: 7, culled: 1 }));
        // The end goes; everything goes; nothing goes.
        assert_eq!(check_strip(&runs, |v| v >= 6), Some(Culled { indices: 8, kept: 6, culled: 2 }));
        assert_eq!(check_strip(&runs, |_| true), Some(Culled { indices: 0, kept: 0, culled: 8 }));
        assert_eq!(check_strip(&runs, |_| false), None);
        // Several runs (one without a triangle), every pattern of zero vertices in turn.
        let runs: Vec<u16> = [strip(5, 0), strip(2, 5), strip(4, 7), strip(7, 11), strip(3, 18)].concat();
        let mut culled_some = 0;
        for pattern in 0u32..1 << 21 {
            if pattern % 7 != 0 && pattern.count_ones() < 12 {
                continue; // a seventh of them, and all the dense ones
            }
            if check_strip(&runs, |v| pattern >> v & 1 != 0).is_some() {
                culled_some += 1;
            }
        }
        assert!(culled_some > 1000);
    }

    #[test]
    fn the_memo_lets_a_draw_without_zero_triangles_wait() {
        let mut memo = Memo::new();
        let (sky, effect) = (0x0912_3400, 0x09AB_CD00);
        // Nothing known: read.
        assert!(memo.due(sky, 114));
        assert!(memo.due(effect, 63));
        // The sky lost half its triangles: read every frame.
        memo.note(sky, 114, 400, 420);
        // The effect lost one of a hundred: it waits 24..55 frames, then is read again.
        memo.note(effect, 63, 1, 99);
        let mut waited = 0;
        while !memo.due(effect, 63) {
            waited += 1;
            assert!(memo.due(sky, 114));
        }
        assert!((24..=55).contains(&waited), "waited {waited}");
        // Read again and nothing still: waits again. Then it has some: every frame.
        memo.note(effect, 63, 0, 101);
        assert!(!memo.due(effect, 63));
        memo.note(effect, 63, 25, 275);
        assert!(memo.due(effect, 63) && memo.due(effect, 63));
        // One in twenty: not worth it. One in sixteen is.
        memo.note(effect, 63, 5, 95);
        assert!(!memo.due(effect, 63));
        memo.note(effect, 63, 10, 150);
        assert!(memo.due(effect, 63));
        // Another draw at the address (another index count): read at once.
        memo.note(effect, 63, 0, 101);
        assert!(memo.due(effect, 64));
        // Draws falling into one set wait side by side (four of them); a fifth takes
        // the place of one of the others, which is then read again.
        let set_of = |address: u32| Memo::hash(address) >> 27;
        let same: Vec<u32> = (0..u32::MAX).map(|k| 0x0800_0000 + 4 * k).filter(|&a| set_of(a) == set_of(effect)).take(5).collect();
        let mut memo = Memo::new();
        for &address in &same[..4] {
            memo.note(address, 30, 0, 10);
        }
        assert!(same[..4].iter().all(|&address| !memo.due(address, 30)));
        memo.note(same[4], 30, 0, 10);
        assert!(!memo.due(same[4], 30));
        assert_eq!(same[..4].iter().filter(|&&address| memo.due(address, 30)).count(), 1);
        // A draw worth reading takes no entry.
        let mut memo = Memo::new();
        memo.note(sky, 114, 400, 420);
        assert!(memo.entries.iter().all(|entry| entry.wait == 0 && entry.address == 0));
    }

    #[test]
    fn a_cull_that_needs_more_indices_than_the_strip_is_not_made() {
        // 13 vertices, 11 triangles; zero vertices 3 4 5 and 9 10 11: triangles 3 and 9 go,
        // and three stretches with two joins need more than 13 indices.
        let runs = [TRIANGLE_STRIP as u16, 13, 0];
        let zero = |v: u16| (3..=5).contains(&v) || (9..=11).contains(&v);
        let mut out = [0u16; 13];
        assert_eq!(cull_strip_runs(&runs, zero, &mut out), None);
        // With room it is right.
        let mut out = [0u16; 32];
        let culled = cull_strip_runs(&runs, zero, &mut out).unwrap();
        assert_eq!((culled.kept, culled.culled), (9, 2));
        let whole: Vec<u16> = stitched_strip(&runs).collect();
        let wanted: Vec<[u16; 3]> = strip_triangles(&whole).into_iter().filter(|t| !t.iter().all(|&v| zero(v))).collect();
        assert_eq!(strip_triangles(&out[..culled.indices]), wanted);
    }
}
