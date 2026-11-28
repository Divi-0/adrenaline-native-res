//! Normal alpha blends whose source is opaque everywhere.
//!
//! The GE blends a source of alpha 255 with "source alpha, one minus
//! source alpha, add" into exactly the source: its terms ((2c + 1)(2f + 1))
//! >> 10 give c for f = 255 and 0 for f = 0 (`frame::ge_blend_term`). So a
//! draw into a 16-bit target whose every fragment has alpha 255 can take
//! the frame-read program for "no blending" (`ReadWay::Off`: no frame read,
//! the same cut and dither) and draw the same picture. Monster Hunter
//! Freedom draws its whole ground that way (normal blending on, opaque
//! palettes, vertex alpha 255): the frame reads cost the river field
//! several ms of GPU time.
//!
//! A fragment's alpha is the texel's times the vertex's ((p + 1) t >> 8,
//! 255 only when both are 255), so this module answers two questions: is
//! every texel the GPU can sample opaque (`opaque_table_entries`, for our
//! palette textures), and is every vertex's final alpha 255
//! (`lowest_vertex_alpha`, `RawColors::alpha_always_255`). Interpolated
//! alphas between vertices of alpha 255 are 255, and so are filtered texels
//! between opaque ones.

use crate::frame::{draw_flags, Draw, Frame, RawColors, ReadVariant, ReadWay};
use crate::raw_vertex::lighting;
use crate::vertex::VertexFormat;

/// `opaque_table_entries` bit: the first 16 entries are opaque (enough for
/// 4-bit index textures, which sample only those).
pub const FIRST_16_OPAQUE: u8 = 1;
/// `opaque_table_entries` bit: all 256 entries are opaque.
pub const ALL_256_OPAQUE: u8 = 2;

/// Which parts of a 256-colour table (RGBA, `texture::palette_rgba`) are
/// opaque: `FIRST_16_OPAQUE`, `ALL_256_OPAQUE`.
pub fn opaque_table_entries(table: &[u32; 256]) -> u8 {
    let opaque = |entries: &[u32]| entries.iter().all(|entry| entry >> 24 == 0xFF);
    if !opaque(&table[..16]) {
        return 0;
    }
    if opaque(&table[16..]) {
        FIRST_16_OPAQUE | ALL_256_OPAQUE
    } else {
        FIRST_16_OPAQUE
    }
}

/// True if a palette texture of PSP format `format` (4 = 4-bit indices,
/// 5..7 = 8, 16, 32 bits) samples only opaque entries of a table with
/// `opaque_table_entries` result `entries`.
pub fn palette_texture_opaque(format: u8, entries: u8) -> bool {
    if format == 4 {
        entries & FIRST_16_OPAQUE != 0
    } else {
        entries & ALL_256_OPAQUE != 0
    }
}

/// The lowest alpha (widened to 8 bits, as `VertexFormat` decodes colours)
/// of the vertices in `vertices` (whole vertices of `format`, back to
/// back), or None if the format has no colour (the draw's material colour
/// counts then). A trailing part vertex is left out.
pub fn lowest_vertex_alpha(format: VertexFormat, vertices: &[u8]) -> Option<u8> {
    let layout = format.layout();
    let at = layout.color_offset? as usize;
    let stride = layout.size as usize;
    if stride == 0 {
        return Some(0);
    }
    let fields = vertices.chunks_exact(stride);
    let half = |vertex: &[u8]| u16::from_le_bytes([vertex[at], vertex[at + 1]]);
    Some(match format.color_format_bits() {
        // 565: no alpha (opaque).
        4 => 0xFF,
        // 5551: the top bit.
        5 => {
            if fields.into_iter().all(|vertex| half(vertex) & 0x8000 != 0) {
                0xFF
            } else {
                0
            }
        }
        // 4444: the top nibble, widened x 17.
        6 => fields.map(|vertex| (half(vertex) >> 12) as u8).min().unwrap_or(0xF) * 17,
        // 8888: the top byte.
        _ => fields.map(|vertex| vertex[at + 3]).min().unwrap_or(0xFF),
    })
}

impl RawColors {
    /// True if every vertex of a raw draw with these colour rules ends up
    /// with alpha 255 when the lowest vertex alpha is `lowest` (None: the
    /// vertices have no colour). Each rule (`RawColors::color`: ambient
    /// lighting, opaque, doubling, replace) keeps a higher alpha at least
    /// as high, so the lowest one decides. Lit by the GPU's lights: only
    /// when the draw is made opaque (the lights' alpha isn't looked at).
    pub fn alpha_always_255(&self, lowest: Option<u8>) -> bool {
        self.alpha_reaches(lowest, 0xFF)
    }

    /// True if every vertex of a raw draw with these colour rules ends up
    /// with an alpha of at least `target` when the lowest vertex alpha is
    /// `lowest` (None: no colour); `alpha_always_255` for 255. Lit by the
    /// GPU's lights: only when the draw is made opaque (or for target 0).
    pub fn alpha_reaches(&self, lowest: Option<u8>, target: u8) -> bool {
        if self.rules[0] == lighting::LIGHTS {
            return self.rules[1] != 0.0 || target == 0;
        }
        let vertex = lowest.map(|alpha| (alpha as u32) << 24 | 0x00FF_FFFF);
        self.color(vertex) >> 24 >= target as u32
    }

    /// The lowest vertex alpha (8 bits) that these rules turn into 255
    /// (`alpha_always_255`), or None if not even 255 does: a draw's vertices
    /// all end up opaque exactly when each has at least this alpha
    /// (`least_alpha_reaching(255)`).
    pub fn least_opaque_alpha(&self) -> Option<u8> {
        self.least_alpha_reaching(0xFF)
    }

    /// The lowest vertex alpha (8 bits) that these rules turn into at least
    /// `target` (`alpha_reaches`), or None if not even 255 does. Every rule
    /// keeps a higher alpha at least as high (ambient lighting's product and
    /// emission, doubling, the opaque and replace rules set it), so the
    /// answer is a threshold, found by halving 0..255.
    pub fn least_alpha_reaching(&self, target: u8) -> Option<u8> {
        if !self.alpha_reaches(Some(0xFF), target) {
            return None;
        }
        if self.alpha_reaches(Some(0), target) {
            return Some(0);
        }
        // 0 fails, 255 passes: the threshold lies in (low, high].
        let (mut low, mut high) = (0u8, 0xFFu8);
        while high - low > 1 {
            let middle = low + (high - low) / 2;
            if self.alpha_reaches(Some(middle), target) {
                high = middle;
            } else {
                low = middle;
            }
        }
        Some(high)
    }
}

/// True if every vertex a raw draw uses has an alpha (widened to 8 bits as
/// `lowest_vertex_alpha`) of at least `least`: the vertices of its index
/// runs (`runs`: `stored` as [primitive type, count, first vertex] triples)
/// or the ones its indices name, not the whole span `first..end` between
/// them (`raw_vertex_span`). The other vertices of the span give no
/// fragment: Kenka Bancho's train draws ~100 vertices each out of a model
/// buffer of ~1,800 that every draw spans, and on the Vita each vertex read
/// is an uncached load from PSP RAM (~0.2 us): 12.6 MB of spans a frame cost
/// the display thread 88 ms, the used vertices are 0.8 MB.
///
/// `window`: the draw's vertices from its vertex 0, back to back in
/// `format` (which has a colour), at least up to `end`. Each used vertex is
/// read once (`seen`: one bit per vertex of `first..end`, zero, at least
/// (end - first) / 32 words rounded up), and the scan stops at the first
/// one below `least`. False if a vertex lies outside `window`, `first..end`
/// or `seen`.
pub fn used_alphas_reach(
    format: VertexFormat,
    window: &[u8],
    stored: &[u16],
    runs: bool,
    (first, end): (u32, u32),
    least: u8,
    seen: &mut [u32],
) -> bool {
    let layout = format.layout();
    let Some(at) = layout.color_offset else {
        return false; // no colour: the caller asks the rules alone
    };
    let bits = format.color_format_bits();
    if least == 0 || bits == 4 {
        return true; // every alpha reaches 0; 565 has none: 255
    }
    let stride = layout.size as usize;
    // The byte that holds the alpha: the high byte of a 16-bit colour, the
    // top byte of 8888. One load per vertex (uncached on the Vita).
    let byte = at as usize + if bits == 7 { 3 } else { 1 };
    let alpha = |high: u8| -> u8 {
        match bits {
            5 => if high & 0x80 != 0 { 0xFF } else { 0 },
            6 => (high >> 4) * 17,
            _ => high,
        }
    };
    let mut check = |vertex: u32| -> bool {
        let Some(slot) = vertex.checked_sub(first).filter(|_| vertex < end) else {
            return false;
        };
        let (word, bit) = ((slot / 32) as usize, 1u32 << (slot % 32));
        let Some(seen_word) = seen.get_mut(word) else {
            return false;
        };
        if *seen_word & bit != 0 {
            return true;
        }
        *seen_word |= bit;
        window.get(vertex as usize * stride + byte).is_some_and(|&high| alpha(high) >= least)
    };
    if runs {
        for run in stored.chunks_exact(3) {
            // Sprites: two vertices per count (as `raw_vertex_span`).
            let count = if run[0] as u32 == crate::command::primitive::RECTANGLES { 2 * run[1] as u32 } else { run[1] as u32 };
            let start = run[2] as u32;
            if !(start..start + count).all(&mut check) {
                return false;
            }
        }
        true
    } else {
        stored.iter().all(|&index| check(index as u32))
    }
}

/// The variant for a draw whose every fragment has alpha 255: a normal
/// alpha blend (frame read or GXM's) becomes no blending, the same picture
/// (module comment); anything else stays.
pub fn for_opaque_source(variant: ReadVariant) -> ReadVariant {
    match variant {
        ReadVariant::Fast { way: ReadWay::Normal | ReadWay::Precut, keep_alpha, dither, green6, fogless, wide: false } => {
            ReadVariant::Fast { way: ReadWay::Off, keep_alpha, dither, green6, fogless, wide: false }
        }
        other => other,
    }
}

/// True if an alpha test with fragment-program uniform `uniform` (x, y, z:
/// a fragment passes when (alpha - x)(y - alpha) z > 0, draw.rs
/// `alpha_test_uniform`) lets alpha 255 through. A draw whose every
/// fragment has alpha 255 then passes it everywhere and can drop it: no
/// `discard` (the GPU pays for it even where it passes), the same picture
/// and depth.
pub fn alpha_test_lets_255_pass(uniform: [f32; 4]) -> bool {
    (255.0 - uniform[0]) * (uniform[1] - 255.0) * uniform[2] > 0.0
}

/// True if an alpha test with fragment-program uniform `uniform` (as
/// `alpha_test_lets_255_pass`) passes exactly the alphas from a lower bound
/// up (greater, greater or equal, equal 255; nearly every test games use):
/// then "discard where alpha < x" is the same test as the two-sided
/// (alpha - x)(y - alpha) z < 0, since y - alpha > 0 for every alpha
/// 0..255. The fast frame-read programs test that way (3 instructions
/// less, nativeres_textured_read_f.cg READ_TEST); the other tests take the
/// generic program.
pub fn alpha_test_is_one_sided(uniform: [f32; 4]) -> bool {
    uniform[2] > 0.0 && uniform[1] > 255.0
}

/// The lowest alpha `l` (0..255) such that an alpha test with uniform
/// `uniform` (as `alpha_test_lets_255_pass`) passes every alpha from `l` to
/// 255, whole or not: a draw whose fragments all have at least that alpha
/// passes it everywhere and can drop it, the same picture and depth. None
/// if 255 itself doesn't pass. The test passes strictly between x and y
/// (z > 0) or strictly outside them (z < 0); x and y lie half-way between
/// whole numbers (draw.rs `alpha_test_uniform`), the GE's integer test.
pub fn alpha_test_passes_from(uniform: [f32; 4]) -> Option<u8> {
    let (x, y, z) = (uniform[0], uniform[1], uniform[2]);
    let (low, high) = if x <= y { (x, y) } else { (y, x) };
    // The first whole number above `bound` (0 below 0), None past 255.
    let above = |bound: f32| -> Option<u8> {
        if bound < 0.0 {
            Some(0)
        } else if bound < 255.0 {
            Some(bound as u8 + 1)
        } else {
            None
        }
    };
    if z > 0.0 && low < high {
        // (low, high): all of [l, 255] when 255 < high and l > low.
        if high > 255.0 { above(low) } else { None }
    } else if z < 0.0 {
        // Outside [low, high]: below it entirely, or above it.
        if low > 255.0 { Some(0) } else { above(high) }
    } else {
        None
    }
}

/// How far below the GE's product the GPU's fragment alpha may fall
/// (`fragment_alpha_floor`): bilinear filtering between texels of at least
/// t gives at least t up to its float rounding, and the program's product
/// t x v rounds its own way. Two levels cover both by far.
pub const FRAGMENT_ALPHA_SLACK: u32 = 2;

/// A lower bound of a textured fragment's alpha when every texel the GPU
/// can sample has alpha at least `texel` and every vertex at least
/// `vertex` (both final, 8 bits): the GE's modulate ((v + 1) t) >> 8 less
/// `FRAGMENT_ALPHA_SLACK`. It is also at most each of t and v, so it bounds
/// the other texture functions too (replace: t, decal: v).
pub fn fragment_alpha_floor(texel: u8, vertex: u8) -> u8 {
    (((vertex as u32 + 1) * texel as u32) >> 8).saturating_sub(FRAGMENT_ALPHA_SLACK) as u8
}

/// The lowest vertex alpha that makes `fragment_alpha_floor(texel, v)` at
/// least `needed`, or None if not even 255 does (it grows with v).
pub fn vertex_alpha_needed(texel: u8, needed: u8) -> Option<u8> {
    if fragment_alpha_floor(texel, 0xFF) < needed {
        return None;
    }
    // Halving: the floor at `high` reaches `needed`.
    let (mut low, mut high) = (0u32, 0xFFu32);
    if fragment_alpha_floor(texel, 0) >= needed {
        return Some(0);
    }
    while high - low > 1 {
        let middle = (low + high) / 2;
        if fragment_alpha_floor(texel, middle as u8) >= needed {
            high = middle;
        } else {
            low = middle;
        }
    }
    Some(high as u8)
}

/// The lowest alpha among the first 16 entries and among all 256 of a
/// 256-colour table (RGBA, `texture::palette_rgba`): what a 4-bit and an
/// 8-bit index texture can sample (`palette_texture_lowest_alpha`).
pub fn table_lowest_alphas(table: &[u32; 256]) -> [u8; 2] {
    let lowest = |entries: &[u32]| entries.iter().map(|entry| (entry >> 24) as u8).min().unwrap_or(0xFF);
    let first = lowest(&table[..16]);
    [first, first.min(lowest(&table[16..]))]
}

/// The lowest alpha a palette texture of PSP format `format` (as
/// `palette_texture_opaque`) samples from a table with `table_lowest_alphas`
/// result `lowest`.
pub fn palette_texture_lowest_alpha(format: u8, lowest: [u8; 2]) -> u8 {
    if format == 4 { lowest[0] } else { lowest[1] }
}

/// How far past its own value a texture coordinate may be taken as
/// sampled, in texels: the GPU's rounding of interpolated coordinates
/// (inside the triangle they lie between the vertices' values) and of its
/// filter positions is far below this.
const ROW_SLACK: f32 = 1.0 / 16.0;

/// The last texel row (0..height-1) a draw with V clamped can sample when
/// its largest texture coordinate v (0..1 across the texture) is `v_max`:
/// nearest filtering takes row floor(t), bilinear rows floor(t - 0.5) and
/// the next (t = v x height), each counted `ROW_SLACK` further.
pub fn last_sampled_row(v_max: f32, height: u16, linear: bool) -> u32 {
    let t = v_max * height as f32 + ROW_SLACK;
    // floor (no_std): `as` truncates toward 0 and saturates.
    let floor = |x: f32| {
        let i = x as i32;
        if (i as f32) > x { i - 1 } else { i }
    };
    let last = if linear { floor(t - 0.5) + 1 } else { floor(t) };
    (last.max(0) as u32).min((height as u32).saturating_sub(1))
}

/// True if a draw with these `flags`, texture height and screen-space
/// `vertices` samples only the texture's first `rows` rows (the ones
/// `texture::opaque_rows` found opaque; `ALL_ROWS_OPAQUE`: all). Only for
/// V clamped (repeating wraps to the bottom rows) and without mip levels
/// (other levels aren't counted by rows). The fragments' coordinates lie
/// between the vertices', so the largest v bounds them.
pub fn samples_only_rows(vertices: &[crate::frame::FrameVertex], flags: u32, height: u16, rows: u16) -> bool {
    if rows == crate::texture::ALL_ROWS_OPAQUE {
        return true;
    }
    if rows == 0 || vertices.is_empty() || flags & draw_flags::CLAMP_V == 0 || flags & draw_flags::MIPS != 0 {
        return false;
    }
    let mut v_max = f32::MIN;
    for vertex in vertices {
        if !vertex.v.is_finite() {
            return false;
        }
        v_max = v_max.max(vertex.v);
    }
    last_sampled_row(v_max, height, flags & draw_flags::LINEAR_FILTER != 0) < rows as u32
}

/// The colour all of a draw's CPU `vertices` share (0xAABBGGRR), if they
/// do: such a draw is shaded by the constant-colour program
/// (nativeres_textured_const_f.cg, `const_tint`) instead of the flat one,
/// the same pixels in half the instructions (God of War's full-screen bloom
/// passes). None for no vertices.
pub fn constant_color(vertices: &[crate::frame::FrameVertex]) -> Option<u32> {
    let color = vertices.first()?.color;
    vertices.iter().all(|vertex| vertex.color == color).then_some(color)
}

/// The constant-colour program's uTint for vertex colour `color`
/// (0xAABBGGRR) and the draw's colour scale (1, or 2 with colour doubling:
/// RGB only): (p + 1) x scale / 256 per channel, what the flat program
/// multiplies the texel's whole value with ((p + 1) t >> 8 as the GE's
/// modulate, nativeres_textured_flat_f.cg). Exact in f32: p + 1 <= 256 and
/// the scale a power of two.
pub fn const_tint(color: u32, scale: f32) -> [f32; 4] {
    let mut tint = [0.0f32; 4];
    for (channel, value) in tint.iter_mut().enumerate() {
        let plus_one = (((color >> (8 * channel)) & 0xFF) + 1) as f32;
        *value = plus_one * if channel < 3 { scale } else { 1.0 } * (1.0 / 256.0);
    }
    tint
}

/// The vertex numbers a raw draw (`draw_flags::RAW_VERTICES`) uses, as
/// `first..end` counted from its window (`Draw::first_vertex`): from its
/// index runs or its triangle list. None without any.
pub fn raw_vertex_span(frame: &Frame, draw: &Draw) -> Option<(u32, u32)> {
    let start = draw.first_index as usize;
    let stored = frame.indices.get(start..start + draw.index_count as usize)?;
    let (mut first, mut end) = (u32::MAX, 0u32);
    if draw.flags & draw_flags::INDEX_RUNS != 0 {
        for run in stored.chunks_exact(3) {
            // [primitive type, count, first vertex]; sprites: two per count.
            let vertices = if run[0] as u32 == crate::command::primitive::RECTANGLES { 2 * run[1] as u32 } else { run[1] as u32 };
            first = first.min(run[2] as u32);
            end = end.max(run[2] as u32 + vertices);
        }
    } else {
        for &index in stored {
            first = first.min(index as u32);
            end = end.max(index as u32 + 1);
        }
    }
    (first < end).then_some((first, end))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::ge_blend_term;
    use crate::raw_vertex::replace;

    #[test]
    fn a_normal_blend_of_an_opaque_source_is_the_source() {
        for source in 0..=255 {
            for target in 0..=255 {
                assert_eq!(ge_blend_term(source, 255) + ge_blend_term(target, 0), source, "source {source}, target {target}");
            }
        }
    }

    fn vertex(color: u32) -> crate::frame::FrameVertex {
        crate::frame::FrameVertex { color, ..Default::default() }
    }

    #[test]
    fn a_draw_has_a_constant_colour_only_if_every_vertex_has_it() {
        assert_eq!(constant_color(&[]), None);
        assert_eq!(constant_color(&[vertex(0xFF33_4C65)]), Some(0xFF33_4C65));
        assert_eq!(constant_color(&[vertex(0xFFFF_FFFF); 6]), Some(0xFFFF_FFFF));
        let mut vertices = [vertex(0xFFFF_FFFF); 6];
        vertices[5].color = 0xFEFF_FFFF;
        assert_eq!(constant_color(&vertices), None);
    }

    /// The constant-colour program's product t x uTint - 127.5 / 256, taken to
    /// bytes as its floor saturated at 255 (the program's conversion), is the
    /// GE's modulate (p + 1) t >> 8 (x 2 with colour doubling, clamped) that
    /// the flat program works out from the vertex colour, for every texel
    /// value, colour byte and scale; every step is exact in f32.
    #[test]
    fn the_constant_tint_gives_the_flat_programs_bytes() {
        for scale in [1u32, 2] {
            for p in 0..=255u32 {
                let tint = const_tint(p | p << 8 | p << 16 | p << 24, scale as f32);
                for t in 0..=255u32 {
                    // RGB (scaled) and alpha (never scaled).
                    for (channel, factor) in [(0, scale), (3, 1)] {
                        let product = t as f32 * tint[channel];
                        assert_eq!(product as f64, (t * (p + 1) * factor) as f64 / 256.0, "exact product: t {t} p {p}");
                        let shifted = product - 127.5 / 256.0;
                        // The conversion to bytes rounds to nearest: + 0.5, floor, saturate.
                        let byte = ((shifted + 0.5) as u32).min(255);
                        let ge = ((t * (p + 1)) >> 8) * factor;
                        let ge = if factor == 2 { (t * (p + 1) * 2 >> 8).min(255) } else { ge };
                        assert_eq!(byte, ge, "t {t} p {p} scale {factor}");
                    }
                }
            }
        }
    }

    #[test]
    fn only_normal_blends_become_unblended() {
        let fast = |way, keep_alpha, dither| ReadVariant::Fast { way, keep_alpha, dither, green6: false, fogless: false, wide: false };
        for keep_alpha in [false, true] {
            for dither in [false, true] {
                for way in [ReadWay::Normal, ReadWay::Precut] {
                    assert_eq!(for_opaque_source(fast(way, keep_alpha, dither)), fast(ReadWay::Off, keep_alpha, dither));
                }
                for way in [ReadWay::Add, ReadWay::Off, ReadWay::Invisible] {
                    assert_eq!(for_opaque_source(fast(way, keep_alpha, dither)), fast(way, keep_alpha, dither));
                }
            }
        }
        assert_eq!(for_opaque_source(ReadVariant::Generic), ReadVariant::Generic);
    }

    #[test]
    fn alpha_tests_that_let_255_pass() {
        // As draw.rs `alpha_test_uniform(lo, hi, outside)` builds them.
        let uniform = |lo: f32, hi: f32, outside: bool| [lo - 0.5, hi + 0.5, if outside { -1.0 } else { 1.0 }, 0.0];
        for reference in 0..=255 {
            let r = reference as f32;
            assert!(alpha_test_lets_255_pass(uniform(r, 255.0, false)), "greater or equal {reference}");
            assert_eq!(alpha_test_lets_255_pass(uniform(r + 1.0, 255.0, false)), reference < 255, "greater {reference}");
            assert_eq!(alpha_test_lets_255_pass(uniform(r, r, false)), reference == 255, "equal {reference}");
            assert_eq!(alpha_test_lets_255_pass(uniform(r, r, true)), reference != 255, "not equal {reference}");
            assert_eq!(alpha_test_lets_255_pass(uniform(0.0, r, false)), reference == 255, "less or equal {reference}");
            assert!(!alpha_test_lets_255_pass(uniform(0.0, r - 1.0, false)), "less {reference}");
        }
        assert!(!alpha_test_lets_255_pass(uniform(1.0, 0.0, false)), "never");
    }

    #[test]
    fn alpha_tests_pass_everything_from_their_threshold() {
        // As draw.rs `alpha_test_uniform(lo, hi, outside)` builds them.
        let uniform = |lo: f32, hi: f32, outside: bool| [lo - 0.5, hi + 0.5, if outside { -1.0 } else { 1.0 }, 0.0];
        let passes = |u: [f32; 4], a: f32| (a - u[0]) * (u[1] - a) * u[2] > 0.0;
        let mut checked = 0;
        for outside in [false, true] {
            for lo in -1..=257 {
                for hi in -1..=257 {
                    let u = uniform(lo as f32, hi as f32, outside);
                    let from = alpha_test_passes_from(u);
                    match from {
                        Some(l) => {
                            // Every alpha from l, whole or between (filtered fragments).
                            for quarter in (4 * l as u32)..=(4 * 255) {
                                let a = quarter as f32 / 4.0;
                                assert!(passes(u, a), "lo {lo} hi {hi} outside {outside}: from {l}, {a} fails");
                            }
                            // The lowest such whole number: from l - 1 on, some
                            // alpha (whole or between) fails.
                            if l > 0 {
                                let below = (0..4).map(|q| (l - 1) as f32 + q as f32 / 4.0);
                                assert!(below.clone().any(|a| !passes(u, a)), "lo {lo} hi {hi} outside {outside}: all from {} pass", l - 1);
                            }
                            checked += 1;
                        }
                        None => assert!(!passes(u, 255.0), "lo {lo} hi {hi} outside {outside}: 255 passes"),
                    }
                }
            }
        }
        assert!(checked > 30_000);
        // Kenka Bancho's walls: GREATER 1 passes from 2.
        assert_eq!(alpha_test_passes_from(uniform(2.0, 255.0, false)), Some(2));
        assert_eq!(alpha_test_passes_from(uniform(1.0, 0.0, false)), None, "never");
        assert_eq!(alpha_test_passes_from([0.0, 300.0, 0.0, 0.0]), None, "z 0: never");
    }

    #[test]
    fn the_fragment_alpha_floor_bounds_every_texture_function() {
        for texel in 0..=255u8 {
            for vertex in 0..=255u8 {
                let floor = fragment_alpha_floor(texel, vertex);
                let modulate = ((vertex as u32 + 1) * texel as u32) >> 8;
                assert!(floor as u32 + FRAGMENT_ALPHA_SLACK <= modulate.max(FRAGMENT_ALPHA_SLACK), "t {texel} v {vertex}");
                assert!(floor <= texel && floor <= vertex, "t {texel} v {vertex}: replace and decal");
            }
            for needed in 0..=255u8 {
                let brute = (0..=255u8).find(|&v| fragment_alpha_floor(texel, v) >= needed);
                assert_eq!(vertex_alpha_needed(texel, needed), brute, "t {texel} needed {needed}");
            }
        }
        // The train: texels of 254, the test from 2: vertices of 4 or more.
        assert_eq!(vertex_alpha_needed(254, 2), Some(4));
        assert_eq!(fragment_alpha_floor(254, 255), 252);
    }

    #[test]
    fn the_lowest_alpha_of_a_table() {
        let mut table = [0xFF00_0000u32; 256];
        assert_eq!(table_lowest_alphas(&table), [255, 255]);
        table[200] = 0xFE12_3456;
        assert_eq!(table_lowest_alphas(&table), [255, 254]);
        assert_eq!(palette_texture_lowest_alpha(4, table_lowest_alphas(&table)), 255);
        assert_eq!(palette_texture_lowest_alpha(5, table_lowest_alphas(&table)), 254);
        table[3] = 0x1000_0000;
        assert_eq!(table_lowest_alphas(&table), [16, 16]);
    }

    #[test]
    fn rules_reach_a_target_from_a_threshold() {
        for colors in rule_sets() {
            for target in [0u8, 1, 2, 4, 128, 252, 254, 255] {
                let least = colors.least_alpha_reaching(target);
                for alpha in 0..=255u8 {
                    assert_eq!(colors.alpha_reaches(Some(alpha), target), least.is_some_and(|least| alpha >= least), "rules {:?} target {target} alpha {alpha}", colors.rules);
                }
            }
            assert_eq!(colors.least_alpha_reaching(255), colors.least_opaque_alpha());
        }
    }

    #[test]
    fn one_sided_alpha_tests_are_the_lower_bound_alone() {
        // As draw.rs `alpha_test_uniform(lo, hi, outside)` and
        // `alpha_test_bounds` (not equal 0 as greater or equal 1) build them.
        let uniform = |lo: f32, hi: f32, outside: bool| [lo - 0.5, hi + 0.5, if outside { -1.0 } else { 1.0 }, 0.0];
        let two_sided = |u: [f32; 4], a: f32| (a - u[0]) * (u[1] - a) * u[2] >= 0.0;
        let one_sided = |u: [f32; 4], a: f32| a >= u[0];
        for reference in 0..=255 {
            let r = reference as f32;
            let tests = [
                (uniform(r, 255.0, false), true),        // greater or equal
                (uniform(r + 1.0, 255.0, false), true),  // greater
                (uniform(r, r, false), reference == 255), // equal (255: from 255 up)
                (uniform(r, r, true), false),            // not equal
                (uniform(0.0, r, false), reference == 255), // less or equal (255: all)
                (uniform(0.0, r - 1.0, false), false),   // less
            ];
            for (i, (u, expected)) in tests.into_iter().enumerate() {
                assert_eq!(alpha_test_is_one_sided(u), expected, "test {i} reference {reference}");
                if expected {
                    for alpha in 0..=255 {
                        let a = alpha as f32;
                        assert_eq!(one_sided(u, a), two_sided(u, a), "test {i} reference {reference} alpha {alpha}");
                    }
                }
            }
        }
        // Never (draw.rs `alpha_test_bounds`: 1..0): the generic program.
        assert!(!alpha_test_is_one_sided(uniform(1.0, 0.0, false)), "never");
    }

    #[test]
    fn opaque_tables_by_part() {
        let mut table = [0xFF00_0000u32; 256];
        assert_eq!(opaque_table_entries(&table), FIRST_16_OPAQUE | ALL_256_OPAQUE);
        table[200] = 0x7F12_3456;
        assert_eq!(opaque_table_entries(&table), FIRST_16_OPAQUE);
        assert!(palette_texture_opaque(4, FIRST_16_OPAQUE));
        assert!(!palette_texture_opaque(5, FIRST_16_OPAQUE));
        table[3] = 0xFE00_0000;
        assert_eq!(opaque_table_entries(&table), 0);
        assert!(!palette_texture_opaque(4, 0));
    }

    /// VERTEX_TYPE: 16-bit texture coordinates and positions with colour
    /// format `color` (4..7), as MHF's ground (0x11a = 4444).
    fn format(color: u32) -> VertexFormat {
        VertexFormat(2 | color << 2 | 2 << 7)
    }

    /// Vertices of `format` with colours `colors` (the field's raw bits).
    fn vertices(format: VertexFormat, colors: &[u32]) -> Vec<u8> {
        let layout = format.layout();
        let mut bytes = vec![0u8; layout.size as usize * colors.len()];
        for (vertex, &color) in bytes.chunks_exact_mut(layout.size as usize).zip(colors) {
            let at = layout.color_offset.unwrap() as usize;
            let size = if format.color_format_bits() == 7 { 4 } else { 2 };
            vertex[at..at + size].copy_from_slice(&color.to_le_bytes()[..size]);
        }
        bytes
    }

    #[test]
    fn the_lowest_vertex_alpha_per_colour_format() {
        assert_eq!(format(6).0, 0x11a);
        let f4444 = format(6);
        assert_eq!(lowest_vertex_alpha(f4444, &vertices(f4444, &[0xF123, 0xFFFF])), Some(255));
        assert_eq!(lowest_vertex_alpha(f4444, &vertices(f4444, &[0xF123, 0xEFFF])), Some(238));
        let f5551 = format(5);
        assert_eq!(lowest_vertex_alpha(f5551, &vertices(f5551, &[0x8000, 0xFFFF])), Some(255));
        assert_eq!(lowest_vertex_alpha(f5551, &vertices(f5551, &[0x8000, 0x7FFF])), Some(0));
        let f8888 = format(7);
        assert_eq!(lowest_vertex_alpha(f8888, &vertices(f8888, &[0xFF00_0000, 0xFE00_0000])), Some(254));
        let f565 = format(4);
        assert_eq!(lowest_vertex_alpha(f565, &vertices(f565, &[0x0000])), Some(255));
        // No colour field: the material decides.
        assert_eq!(lowest_vertex_alpha(VertexFormat(2 | 2 << 7), &[0u8; 64]), None);
        // A part vertex at the end is left out.
        let mut bytes = vertices(f4444, &[0xFFFF]);
        bytes.extend_from_slice(&[0u8; 3]);
        assert_eq!(lowest_vertex_alpha(f4444, &bytes), Some(255));
    }

    #[test]
    fn the_colour_rules_keep_the_lowest_alpha_deciding() {
        let plain = RawColors { material: [255.0, 255.0, 255.0, 255.0], ..RawColors::default() };
        assert!(plain.alpha_always_255(Some(255)));
        assert!(!plain.alpha_always_255(Some(254)));
        // Without a vertex colour: the material's alpha.
        assert!(plain.alpha_always_255(None));
        assert!(!RawColors { material: [255.0, 255.0, 255.0, 200.0], ..plain }.alpha_always_255(None));
        // Made opaque, doubled (128 -> 255), replaced with white.
        assert!(RawColors { rules: [0.0, 1.0, 0.0, 0.0], ..plain }.alpha_always_255(Some(0)));
        let doubled = RawColors { rules: [0.0, 0.0, 0.0, 1.0], ..plain };
        assert!(doubled.alpha_always_255(Some(128)));
        assert!(!doubled.alpha_always_255(Some(127)));
        assert!(RawColors { rules: [0.0, 0.0, replace::ALL, 0.0], ..plain }.alpha_always_255(Some(0)));
        assert!(!RawColors { rules: [0.0, 0.0, replace::RGB, 0.0], ..plain }.alpha_always_255(Some(0)));
        // Ambient lighting of the vertex colour: alpha 255 x ambient 255 is 255.
        let ambient = RawColors { ambient: [255.0; 4], rules: [lighting::VERTEX_MATERIAL, 0.0, 0.0, 0.0], ..plain };
        assert!(ambient.alpha_always_255(Some(255)));
        assert!(!RawColors { ambient: [255.0, 255.0, 255.0, 254.0], ..ambient }.alpha_always_255(Some(255)));
        // The GPU's lights: only when made opaque.
        let lit = RawColors { rules: [lighting::LIGHTS, 0.0, 0.0, 0.0], ..plain };
        assert!(!lit.alpha_always_255(Some(255)));
        assert!(RawColors { rules: [lighting::LIGHTS, 1.0, 0.0, 0.0], ..plain }.alpha_always_255(Some(255)));
    }

    #[test]
    fn a_raw_draw_spans_its_runs_and_its_listed_vertices() {
        let mut frame = crate::frame::new_boxed_frame();
        let listed = [7u16, 3, 9, 3, 4, 5];
        let runs = [crate::command::primitive::TRIANGLE_STRIP as u16, 10, 20, crate::command::primitive::RECTANGLES as u16, 2, 4];
        frame.indices[..6].copy_from_slice(&listed);
        frame.indices[6..12].copy_from_slice(&runs);
        frame.index_count = 12;
        let draw = Draw { first_index: 0, index_count: 6, flags: draw_flags::RAW_VERTICES, ..Draw::default() };
        assert_eq!(raw_vertex_span(&frame, &draw), Some((3, 10)));
        let run_draw = Draw { first_index: 6, index_count: 6, flags: draw_flags::RAW_VERTICES | draw_flags::INDEX_RUNS, ..draw };
        // The strip: 20..30; the sprites: 4..8.
        assert_eq!(raw_vertex_span(&frame, &run_draw), Some((4, 30)));
        assert_eq!(raw_vertex_span(&frame, &Draw { index_count: 0, ..draw }), None);
    }

    /// Colour rules of every kind `alpha_always_255` knows.
    fn rule_sets() -> Vec<RawColors> {
        let plain = RawColors { material: [255.0, 255.0, 255.0, 255.0], ..RawColors::default() };
        let mut sets = Vec::new();
        for lit in [lighting::NONE, lighting::VERTEX_MATERIAL, lighting::MATERIAL, lighting::LIGHTS] {
            for opaque in [0.0, 1.0] {
                for replacing in [replace::NONE, replace::RGB, replace::ALL] {
                    for doubled in [0.0, 1.0] {
                        for (ambient, emissive) in [(255.0, 0.0), (254.0, 0.0), (128.0, 0.0), (100.0, 60.0), (0.0, 255.0), (0.0, 0.0)] {
                            for material in [255.0, 254.0, 200.0] {
                                sets.push(RawColors {
                                    material: [255.0, 255.0, 255.0, material],
                                    ambient: [255.0, 255.0, 255.0, ambient],
                                    emissive: [0.0, 0.0, 0.0, emissive],
                                    rules: [lit, opaque, replacing, doubled],
                                    ..plain
                                });
                            }
                        }
                    }
                }
            }
        }
        sets
    }

    #[test]
    fn the_least_opaque_alpha_is_the_threshold_of_the_rules() {
        for colors in rule_sets() {
            let least = colors.least_opaque_alpha();
            for alpha in 0..=255u8 {
                assert_eq!(
                    colors.alpha_always_255(Some(alpha)),
                    least.is_some_and(|least| alpha >= least),
                    "rules {:?} ambient {:?} emissive {:?} material {:?}, alpha {alpha}, least {least:?}",
                    colors.rules,
                    colors.ambient,
                    colors.emissive,
                    colors.material
                );
            }
        }
        let plain = RawColors { material: [255.0, 255.0, 255.0, 255.0], ..RawColors::default() };
        assert_eq!(plain.least_opaque_alpha(), Some(255));
        assert_eq!(RawColors { rules: [0.0, 0.0, 0.0, 1.0], ..plain }.least_opaque_alpha(), Some(128));
        assert_eq!(RawColors { rules: [0.0, 1.0, 0.0, 0.0], ..plain }.least_opaque_alpha(), Some(0));
        assert_eq!(RawColors { rules: [lighting::LIGHTS, 0.0, 0.0, 0.0], ..plain }.least_opaque_alpha(), None);
    }

    /// The reference for `used_alphas_reach`: the used vertices (runs or
    /// indices) gathered, and the old whole-slice `lowest_vertex_alpha`.
    fn reach_by_gathering(format: VertexFormat, window: &[u8], stored: &[u16], runs: bool, least: u8) -> bool {
        let stride = format.layout().size as usize;
        let mut used: Vec<u32> = if runs {
            stored
                .chunks_exact(3)
                .flat_map(|run| {
                    let count = if run[0] as u32 == crate::command::primitive::RECTANGLES { 2 * run[1] as u32 } else { run[1] as u32 };
                    run[2] as u32..run[2] as u32 + count
                })
                .collect()
        } else {
            stored.iter().map(|&index| index as u32).collect()
        };
        used.sort_unstable();
        used.dedup();
        let mut gathered = Vec::new();
        for vertex in used {
            gathered.extend_from_slice(&window[vertex as usize * stride..][..stride]);
        }
        lowest_vertex_alpha(format, &gathered).map_or(true, |lowest| lowest >= least)
    }

    #[test]
    fn only_the_used_vertices_decide() {
        use crate::command::primitive::{RECTANGLES, TRIANGLE_STRIP};
        // A model buffer of 200 vertices: a draw that names a few of them
        // is opaque even where vertices between them are not.
        let mut state = 0x1234_5678u32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state >> 8
        };
        for bits in [5, 6, 7] {
            let format = format(bits);
            for round in 0..200 {
                // Mostly opaque vertices, some not (more of them in later rounds).
                let colors: Vec<u32> = (0..200)
                    .map(|_| {
                        let random = next();
                        let opaque = random % 100 >= round / 4;
                        match bits {
                            5 => (if opaque { 0x8000 } else { 0 }) | (random & 0x7FFF),
                            6 => (if opaque { 0xF000 } else { (random >> 4) % 15 << 12 }) | (random & 0x0FFF),
                            _ => (if opaque { 0xFF00_0000 } else { (random % 255) << 24 }) | (random & 0xFF_FFFF),
                        }
                    })
                    .collect();
                let window = vertices(format, &colors);
                let least = [0xFF, 0x80, 0x11, 0][round as usize % 4];
                let listed: Vec<u16> = (0..1 + next() % 40).map(|_| (next() % 200) as u16).collect();
                let runs = [TRIANGLE_STRIP as u16, (1 + next() % 20) as u16, (next() % 150) as u16, RECTANGLES as u16, (1 + next() % 5) as u16, (next() % 150) as u16];
                for (stored, are_runs) in [(&listed[..], false), (&runs[..], true)] {
                    let expected = reach_by_gathering(format, &window, stored, are_runs, least);
                    let mut seen = [0u32; 7];
                    assert_eq!(
                        used_alphas_reach(format, &window, stored, are_runs, (0, 200), least, &mut seen),
                        expected,
                        "bits {bits} round {round} least {least} runs {are_runs}"
                    );
                    // A span that starts later (the bits count from it).
                    let first = stored.iter().copied().min().unwrap_or(0) as u32;
                    if !are_runs {
                        let mut seen = [0u32; 7];
                        assert_eq!(used_alphas_reach(format, &window, stored, false, (first, 200), least, &mut seen), expected);
                    }
                }
            }
        }
        // Where every vertex of the span is used: the old whole-span answer.
        let f4444 = format(6);
        let window = vertices(f4444, &[0xF000, 0xF000, 0xE000, 0xF000]);
        let all = [0u16, 1, 2, 3];
        assert_eq!(lowest_vertex_alpha(f4444, &window), Some(0xEE));
        assert!(!used_alphas_reach(f4444, &window, &all, false, (0, 4), 0xFF, &mut [0]));
        assert!(used_alphas_reach(f4444, &window, &all, false, (0, 4), 0xEE, &mut [0]));
        // Vertex 2 left out: opaque.
        assert!(used_alphas_reach(f4444, &window, &[0, 1, 3, 3, 1], false, (0, 4), 0xFF, &mut [0]));
        // 565: no alpha (255); least 0: anything.
        assert!(used_alphas_reach(format(4), &[], &all, false, (0, 4), 0xFF, &mut [0]));
        assert!(used_alphas_reach(f4444, &[], &all, false, (0, 4), 0, &mut [0]));
        // Outside the window, the span or the bits: false.
        assert!(!used_alphas_reach(f4444, &window[..window.len() - f4444.layout().size as usize], &all, false, (0, 4), 0xEE, &mut [0]));
        assert!(!used_alphas_reach(f4444, &window, &all, false, (1, 4), 0xEE, &mut [0]));
        assert!(!used_alphas_reach(f4444, &window, &all, false, (0, 3), 0xEE, &mut [0]));
        assert!(!used_alphas_reach(f4444, &window, &all, false, (0, 40), 0xEE, &mut []));
        // No colour field: the caller asks the rules.
        assert!(!used_alphas_reach(VertexFormat(2 | 2 << 7), &[0u8; 64], &all, false, (0, 4), 0xFF, &mut [0]));
    }

    #[test]
    fn rows_a_draw_samples() {
        use crate::frame::FrameVertex;
        use crate::texture::ALL_ROWS_OPAQUE;
        // NFS UR's title background: 512 high, v up to 271.493 texels,
        // bilinear: rows 0..271 (the picture) and, counted with the slack,
        // 272 (weight ~0); opaque rows 280.
        let v_max = 0.530260f32;
        assert_eq!(last_sampled_row(v_max, 512, true), 272);
        assert_eq!(last_sampled_row(v_max, 512, false), 271);
        // Bilinear: the row below a row's centre counts from the slack on.
        assert_eq!(last_sampled_row(10.4 / 16.0, 16, true), 10);
        assert_eq!(last_sampled_row(10.45 / 16.0, 16, true), 11);
        assert_eq!(last_sampled_row(10.99 / 16.0, 16, false), 11);
        // Clamped to the texture.
        assert_eq!(last_sampled_row(2.0, 16, true), 15);
        assert_eq!(last_sampled_row(-1.0, 16, true), 0);
        let at = |v: f32| FrameVertex { v, ..Default::default() };
        let vertices = [at(-0.000963), at(v_max), at(0.2)];
        let clamped = draw_flags::TEXTURED | draw_flags::LINEAR_FILTER | draw_flags::CLAMP_U | draw_flags::CLAMP_V;
        assert!(samples_only_rows(&vertices, clamped, 512, 280));
        assert!(samples_only_rows(&vertices, clamped, 512, 273));
        assert!(!samples_only_rows(&vertices, clamped, 512, 272));
        // Repeating V, mip levels, no rows, no vertices, NaN: no.
        assert!(!samples_only_rows(&vertices, clamped & !draw_flags::CLAMP_V, 512, 280));
        assert!(!samples_only_rows(&vertices, clamped | draw_flags::MIPS, 512, 280));
        assert!(!samples_only_rows(&vertices, clamped, 512, 0));
        assert!(!samples_only_rows(&[], clamped, 512, 280));
        assert!(!samples_only_rows(&[at(f32::NAN)], clamped, 512, 280));
        // A whole opaque texture: any draw.
        assert!(samples_only_rows(&[], 0, 512, ALL_ROWS_OPAQUE));
    }
}
