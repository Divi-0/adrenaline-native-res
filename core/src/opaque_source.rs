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
        if self.rules[0] == lighting::LIGHTS {
            return self.rules[1] != 0.0;
        }
        let vertex = lowest.map(|alpha| (alpha as u32) << 24 | 0x00FF_FFFF);
        self.color(vertex) >> 24 == 0xFF
    }
}

/// The variant for a draw whose every fragment has alpha 255: a normal
/// alpha blend (frame read or GXM's) becomes no blending, the same picture
/// (module comment); anything else stays.
pub fn for_opaque_source(variant: ReadVariant) -> ReadVariant {
    match variant {
        ReadVariant::Fast { way: ReadWay::Normal | ReadWay::Precut, keep_alpha, dither } => {
            ReadVariant::Fast { way: ReadWay::Off, keep_alpha, dither }
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

    #[test]
    fn only_normal_blends_become_unblended() {
        let fast = |way, keep_alpha, dither| ReadVariant::Fast { way, keep_alpha, dither };
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
}
