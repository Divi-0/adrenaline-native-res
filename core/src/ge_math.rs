//! The GE's own arithmetic for vertex positions, as measured on a PSP
//! (ppsspp-ref GPU/Software/GEMath.h, the gpu/probe experiments): float24
//! values (floats with the low 8 mantissa bits clear), products truncated
//! at a fixed bit weight, matrix rows summed exactly then truncated, its
//! own reciprocal (a table), an adder without guard bits.
//!
//! Used to find out how exact our depth can get (2026-11-23, God of War's
//! title fence: near-coplanar draws tie in the PSP's 16-bit depth): with
//! `GE_EXACT_DEPTH` set (the replay's `REPLAY_DEPTH_MODE` 8), raw corners
//! take their screen depth from `screen_z`, compared with Sony's own depth
//! buffer.

use core::sync::atomic::{AtomicU8, Ordering};

/// Raw corners take their screen depth from `screen_z` (debugging, the
/// replay): 0 off, 1 all of the GE's arithmetic, 2 float clip coordinates
/// with the GE's viewport steps, 3 the GE's clip coordinates with a float
/// viewport.
pub static GE_EXACT_DEPTH: AtomicU8 = AtomicU8::new(0);

/// `f` with the low 8 mantissa bits cleared.
pub fn truncate24(f: f32) -> f32 {
    f32::from_bits(f.to_bits() & 0xFFFF_FF00)
}

/// 2^e as a double (e from -1022 to 1023).
fn pow2(e: i32) -> f64 {
    f64::from_bits(((e + 1023) as u64) << 52)
}

/// floorf for |x| < 2^31.
pub fn floor(x: f32) -> f32 {
    let t = x as i32 as f32;
    if t > x {
        t - 1.0
    } else {
        t
    }
}

/// One term of a matrix row: `mantissa x 2^lsb`.
#[derive(Clone, Copy)]
struct Term {
    mantissa: i32,
    lsb: i32,
}

/// The exact product of two float24s truncated to 15 fraction bits below
/// the inputs' exponents combined (ppsspp-ref `GEProduct`); zero and
/// denormals give no term.
fn product(a: f32, b: f32) -> Option<Term> {
    let (ba, bb) = (a.to_bits(), b.to_bits());
    let (ea, eb) = (((ba >> 23) & 0xFF) as i32, ((bb >> 23) & 0xFF) as i32);
    if ea == 0 || eb == 0 {
        return None;
    }
    let ma = ((ba & 0x007F_FFFF) | 0x0080_0000) as u64;
    let mb = ((bb & 0x007F_FFFF) | 0x0080_0000) as u64;
    let m = ((ma * mb) >> 31) as i32;
    Some(Term { mantissa: if (ba ^ bb) & 0x8000_0000 != 0 { -m } else { m }, lsb: (ea - 127) + (eb - 127) - 15 })
}

/// A matrix row as the GE sums it (`GERowSum`): every term truncated to the
/// bit weight of the largest, added exactly, the sum a float24.
fn row_sum(terms: &[Option<Term>]) -> f32 {
    let Some(top) = terms.iter().flatten().map(|term| term.lsb).max() else {
        return 0.0;
    };
    let mut sum: i64 = 0;
    for term in terms.iter().flatten() {
        let shift = top - term.lsb;
        if shift < 32 {
            let m = term.mantissa as i64;
            sum += if m < 0 { -((-m) >> shift) } else { m >> shift };
        }
    }
    if sum == 0 {
        return 0.0;
    }
    truncate24(((sum as f32) as f64 * pow2(top)) as f32)
}

/// `a x b` of two 4x4 matrices in the GE's row-vector layout (entry r, c at
/// r x 4 + c), each entry summed like a row (`GECombineMatrices`).
pub fn combine(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for r in 0..4 {
        for c in 0..4 {
            let terms: [Option<Term>; 4] = core::array::from_fn(|k| product(a[r * 4 + k], b[k * 4 + c]));
            out[r * 4 + c] = row_sum(&terms);
        }
    }
    out
}

/// Clip coordinate `c` of position `v` through the combined matrix `m`
/// (`GEClipComponent`): the position truncated to float24s, the
/// translation a term of its own.
pub fn clip_component(v: [f32; 3], m: &[f32; 16], c: usize) -> f32 {
    row_sum(&[
        product(truncate24(v[0]), m[c]),
        product(truncate24(v[1]), m[4 + c]),
        product(truncate24(v[2]), m[8 + c]),
        product(1.0, m[12 + c]),
    ])
}

/// The GE's reciprocal segments (`geRecipSegments`): start in units of
/// 2^-17 and slope in units of 2^-23 per step of the low 8 mantissa bits.
pub const RECIP: [(i32, i32); 128] = [
    (131072, -254), (130055, -250), (129054, -246), (128071, -243),
    (127099, -239), (126143, -235), (125203, -232), (124274, -228),
    (123361, -225), (122461, -222), (121574, -219), (120700, -216),
    (119836, -212), (118986, -209), (118150, -207), (117323, -204),
    (116508, -201), (115704, -198), (114913, -196), (114130, -193),
    (113358, -190), (112599, -188), (111847, -185), (111107, -183),
    (110375, -180), (109654, -178), (108943, -176), (108239, -173),
    (107545, -171), (106861, -169), (106184, -167), (105517, -165),
    (104858, -163), (104206, -161), (103563, -159), (102927, -157),
    (102300, -155), (101680, -153), (101067, -151), (100463, -150),
    (99865, -148), (99273, -146), (98689, -144), (98113, -143),
    (97542, -141), (96977, -139), (96421, -138), (95869, -136),
    (95326, -135), (94786, -133), (94255, -132), (93727, -130),
    (93207, -129), (92691, -127), (92182, -126), (91680, -125),
    (91180, -123), (90688, -122), (90201, -121), (89717, -119),
    (89240, -118), (88769, -117), (88302, -116), (87838, -114),
    (87381, -113), (86928, -112), (86481, -111), (86037, -110),
    (85599, -109), (85164, -108), (84732, -106), (84307, -105),
    (83885, -104), (83468, -103), (83055, -102), (82646, -101),
    (82241, -100), (81839, -99), (81442, -98), (81049, -97),
    (80659, -96), (80275, -96), (79892, -95), (79513, -94),
    (79138, -93), (78766, -92), (78398, -91), (78033, -90),
    (77673, -90), (77315, -89), (76960, -88), (76608, -87),
    (76260, -86), (75914, -85), (75574, -85), (75234, -84),
    (74898, -83), (74564, -82), (74236, -82), (73908, -81),
    (73584, -80), (73264, -80), (72944, -79), (72628, -78),
    (72316, -78), (72005, -77), (71697, -76), (71393, -76),
    (71090, -75), (70789, -74), (70493, -74), (70197, -73),
    (69906, -73), (69615, -72), (69327, -71), (69042, -71),
    (68759, -70), (68479, -70), (68200, -69), (67925, -69),
    (67650, -68), (67378, -67), (67109, -67), (66841, -66),
    (66577, -66), (66313, -65), (66052, -65), (65793, -64),
];

/// The GE's reciprocal of a normal float24 (`GERecip`; w's exponent below 126).
pub fn recip(w: f32) -> f32 {
    let bits = w.to_bits();
    let i = (bits >> 8) & 0x7FFF;
    let e = ((bits >> 23) & 0xFF) as i32 - 127;
    let (b, m) = RECIP[(i >> 8) as usize];
    let q = (64 * b + 63 + m * (i & 255) as i32) >> 7;
    let r = (q as f64 * pow2(-16 - e)) as f32;
    if bits & 0x8000_0000 != 0 {
        -r
    } else {
        r
    }
}

/// A product formed exactly, then truncated to a float24 (`ProductToFloat24`).
fn product_to_f24(d: f64) -> f32 {
    let mut bits = d.to_bits();
    if (bits >> 52) & 0x7FF != 0x7FF {
        bits &= !((1u64 << (52 - 15)) - 1);
    }
    f64::from_bits(bits) as f32
}

/// The GE's adder (`GEAdd`): the smaller term truncated to the larger's precision.
pub fn add(a: f32, b: f32) -> f32 {
    let (mut ba, mut bb) = (a.to_bits(), b.to_bits());
    let (ea, eb) = (((ba >> 23) & 0xFF) as i32, ((bb >> 23) & 0xFF) as i32);
    if ea == 0 || eb == 0 {
        return match (ea, eb) {
            (0, 0) => 0.0,
            (0, _) => truncate24(b),
            _ => truncate24(a),
        };
    }
    let emax = ea.max(eb);
    let (na, nb) = (8 + emax - ea, 8 + emax - eb);
    ba = if na >= 24 { ba & 0x8000_0000 } else { ba & !((1u32 << na) - 1) };
    bb = if nb >= 24 { bb & 0x8000_0000 } else { bb & !((1u32 << nb) - 1) };
    f32::from_bits(ba) + f32::from_bits(bb)
}

/// A screen coordinate (`GEViewport`): component / w as component x the
/// reciprocal, truncated, then scaled and offset; `None` where the GE's
/// arithmetic isn't modelled (w zero, denormal, huge or not finite).
pub fn viewport(clip_c: f32, clip_w: f32, scale: f32, center: f32) -> Option<f32> {
    let w = truncate24(clip_w);
    if !w.is_finite() || !clip_c.is_finite() || w.abs() < f32::MIN_POSITIVE || ((w.to_bits() >> 23) & 0xFF) >= 253 {
        return None;
    }
    let ndc = product_to_f24(truncate24(clip_c) as f64 * recip(w) as f64);
    Some(add(product_to_f24(ndc as f64 * scale as f64), center))
}

/// The GE's screen depth (0..65535, floored and clamped as with depth clamp)
/// of model position `model` under world, view (4x3 in our layout: row r =
/// the coefficients of x, y, z and the translation of output r) and
/// projection (4x4, row r = clip coordinate r), with the viewport's depth
/// scale and centre (float24 values). `None` where `viewport` gives none.
pub fn screen_z(model: [f32; 3], world: &[[f32; 4]; 3], view: &[[f32; 4]; 3], projection: &[[f32; 4]; 4], scale: f32, center: f32) -> Option<f32> {
    let mode = GE_EXACT_DEPTH.load(Ordering::Relaxed);
    let (z, w) = if mode == 2 {
        // Float clip coordinates as our programs compute them (world, view, projection in turn).
        let row = |m: &[f32; 4], v: [f32; 3]| m[0] * v[0] + m[1] * v[1] + m[2] * v[2] + m[3];
        let world_position = world.map(|r| row(&r, model));
        let view_position = view.map(|r| row(&r, world_position));
        (row(&projection[2], view_position), row(&projection[3], view_position))
    } else {
        let matrix = combined(world, view, projection);
        (clip_component(model, &matrix, 2), clip_component(model, &matrix, 3))
    };
    let value = if mode == 3 { Some(center + scale * z / w) } else { viewport(z, w, scale, center) };
    value.map(|value| floor(value).clamp(0.0, 65535.0))
}

/// World x view x projection as the GE combines them (`GECombineMatrices`),
/// in the GE's row-vector layout (entry k, c = coefficient of input k in
/// clip coordinate c); layouts as `screen_z`'s.
pub fn combined(world: &[[f32; 4]; 3], view: &[[f32; 4]; 3], projection: &[[f32; 4]; 4]) -> [f32; 16] {
    // To the GE's row-vector layout: entry (k, c) = coefficient of input k in output c.
    let affine = |m: &[[f32; 4]; 3]| -> [f32; 16] {
        let mut out = [0.0f32; 16];
        for c in 0..3 {
            for k in 0..4 {
                out[k * 4 + c] = m[c][k];
            }
        }
        out[15] = 1.0;
        out
    };
    let mut proj = [0.0f32; 16];
    for c in 0..4 {
        for k in 0..4 {
            proj[k * 4 + c] = projection[c][k];
        }
    }
    combine(&combine(&affine(world), &affine(view)), &proj)
}

/// The GE's screen depth in 32-bit integer steps only, as a vertex program
/// can do it (nativeres_raw_v.cg GE_DEPTH does it step for step):
/// `column_z` / `column_w` are the combined matrix's (`combined`) clip z
/// and w columns (rows x, y, z, translation). Same results as `screen_z`
/// (test `the_integer_way_is_the_ges_way`). Every product stays below 2^31:
/// float24 mantissas have 16 significant bits, so the GE's truncated
/// product of two is 2^15 + fa + fb + (fa fb >> 15); a full 16 x 17-bit
/// product is formed as two partial products without its low 8 bits (it
/// has at least 31, only its top 16 are kept).
pub mod integer {
    use super::RECIP;

    fn bits(f: f32) -> i32 {
        f.to_bits() as i32
    }
    fn float(b: i32) -> f32 {
        f32::from_bits(b as u32)
    }
    /// 2^e for e in -126..=127.
    fn pow2(e: i32) -> f32 {
        float((e + 127) << 23)
    }
    fn truncate24(f: f32) -> f32 {
        float(bits(f) & -256)
    }
    /// No term: its weight far below any other's.
    const NONE: (i32, i32) = (0, -100_000);

    /// (mantissa, bit weight) of the GE's truncated product of two float24s.
    fn product(a: f32, b: f32) -> (i32, i32) {
        let (ba, bb) = (bits(a), bits(b));
        let (ea, eb) = ((ba >> 23) & 255, (bb >> 23) & 255);
        if ea == 0 || eb == 0 {
            return NONE;
        }
        let (fa, fb) = ((ba >> 8) & 32767, (bb >> 8) & 32767);
        let m = 32768 + fa + fb + ((fa * fb) >> 15);
        (if (ba ^ bb) < 0 { -m } else { m }, ea + eb - 269)
    }

    fn shifted(term: (i32, i32), top: i32) -> i32 {
        let shift = top - term.1;
        if term.0 == 0 || shift >= 32 {
            return 0;
        }
        let a = term.0.abs() >> shift;
        if term.0 < 0 {
            -a
        } else {
            a
        }
    }

    fn row_sum(t: [(i32, i32); 4]) -> f32 {
        let top = t[0].1.max(t[1].1).max(t[2].1.max(t[3].1));
        if top == NONE.1 {
            return 0.0;
        }
        let sum = shifted(t[0], top) + shifted(t[1], top) + shifted(t[2], top) + shifted(t[3], top);
        if sum == 0 {
            return 0.0;
        }
        float24_of(sum, top)
    }

    /// The float24 `sum` x 2^`weight` truncated to 16 significant bits, made
    /// from bits: the Vita's GPU converts ints of 17+ bits to floats one
    /// low now and then (a device test of the program, 2026-11-24).
    fn float24_of(sum: i32, weight: i32) -> f32 {
        let a = sum.abs();
        // The leading bit's place + 1 (1 for 0 and 1, at most 24), as the
        // program's 23 comparisons (`GdFloat24Of`) find it.
        let length = (32 - a.leading_zeros() as i32).clamp(1, 24);
        let m16 = if length > 16 { a >> (length - 16) } else { a << (16 - length) };
        let sign = if sum < 0 { i32::MIN } else { 0 };
        float(sign | ((weight + length - 1 + 127) << 23) | ((m16 & 32767) << 8))
    }

    /// `Prepared::screen_z` is `screen_z`, bit for bit: random columns and
    /// models over wide exponent ranges, with zeros, denormals and huge values.
    #[test]
    fn prepared_is_screen_z() {
        let mut state = 0x2468_ACE1u32;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let mut value = |next: &mut dyn FnMut() -> u32| -> f32 {
            match next() % 16 {
                0 => 0.0,
                1 => f32::from_bits(next() & 0x807F_FFFF), // denormal or zero
                2 => f32::from_bits((next() & 0x80FF_FFFF) | 0x7E00_0000), // huge
                _ => {
                    let exponent = 127 + (next() % 40) as i32 - 20;
                    f32::from_bits((next() & 0x807F_FFFF) | ((exponent as u32) << 23))
                }
            }
        };
        let mut with_depth = 0;
        for round in 0..200_000 {
            let column_z = [value(&mut next), value(&mut next), value(&mut next), value(&mut next)];
            let column_w = [value(&mut next), value(&mut next), value(&mut next), value(&mut next)];
            let scale = truncate24(value(&mut next));
            let center = truncate24(f32::from_bits(next() % 0x4780_0000));
            let prepared = Prepared::new(column_z, column_w, scale, center);
            for _ in 0..4 {
                let model = [value(&mut next), value(&mut next), value(&mut next)];
                let expected = screen_z(model, column_z, column_w, scale, center);
                let got = prepared.screen_z(model);
                assert_eq!(got.map(f32::to_bits), expected.map(f32::to_bits), "round {round}: {model:?} {column_z:?} {column_w:?} {scale} {center}");
                with_depth += expected.is_some() as u32;
            }
        }
        assert!(with_depth > 100_000, "{with_depth}");
    }

    /// `float24_of`'s leading-bit length equals the program's 23
    /// comparisons for every magnitude up to 2^25 (and some beyond).
    #[test]
    fn length_is_the_programs() {
        let by_comparisons = |a: i32| {
            let mut length = 1;
            for b in 1..24 {
                if a >= 1 << b {
                    length = b + 1;
                }
            }
            length
        };
        let samples = (0..1 << 25).step_by(7).chain([0, 1, 2, 3, 1 << 23, (1 << 24) - 1, 1 << 24, 1 << 30, i32::MAX]);
        for a in samples {
            assert_eq!((32 - a.leading_zeros() as i32).clamp(1, 24), by_comparisons(a), "{a}");
        }
    }

    /// The four terms of a clip row (for device tests of the programs).
    pub fn lab_terms(model: [f32; 3], column: [f32; 4]) -> [(i32, i32); 4] {
        [
            product(truncate24(model[0]), column[0]),
            product(truncate24(model[1]), column[1]),
            product(truncate24(model[2]), column[2]),
            product(1.0, column[3]),
        ]
    }

    fn clip(model: [f32; 3], column: [f32; 4]) -> f32 {
        row_sum([
            product(truncate24(model[0]), column[0]),
            product(truncate24(model[1]), column[1]),
            product(truncate24(model[2]), column[2]),
            product(1.0, column[3]),
        ])
    }

    /// a x b truncated to 16 significant bits, for a, b in 2^15..2^17 with
    /// a b >= 2^30: (mantissa, its bit weight).
    fn product16(a: i32, b: i32) -> (i32, i32) {
        let p8 = a * (b >> 8) + ((a * (b & 255)) >> 8);
        let length = if p8 >= 1 << 25 {
            26
        } else if p8 >= 1 << 24 {
            25
        } else if p8 >= 1 << 23 {
            24
        } else {
            23
        };
        (p8 >> (length - 16), length - 8)
    }

    /// The 16-bit mantissa (hidden bit included) of a normal float24 and its bit weight.
    fn mantissa(f: f32) -> (i32, i32) {
        let b = bits(f);
        (32768 + ((b >> 8) & 32767), ((b >> 23) & 255) - 142)
    }

    fn add(a: f32, b: f32) -> f32 {
        let (mut ba, mut bb) = (bits(a), bits(b));
        let (ea, eb) = ((ba >> 23) & 255, (bb >> 23) & 255);
        if ea == 0 {
            return if eb == 0 { 0.0 } else { truncate24(b) };
        }
        if eb == 0 {
            return truncate24(a);
        }
        let emax = ea.max(eb);
        let (na, nb) = (8 + emax - ea, 8 + emax - eb);
        ba = if na >= 24 { ba & i32::MIN } else { ba & -(1 << na) };
        bb = if nb >= 24 { bb & i32::MIN } else { bb & -(1 << nb) };
        float(ba) + float(bb)
    }

    /// `screen_z` for many vertices of one transform (core
    /// ge_depth_stream.rs, once per raw vertex): what the columns give to
    /// each product worked out once, and each model coordinate's bits taken
    /// apart once for both rows. The same arithmetic as `screen_z` step by
    /// step (test `prepared_is_screen_z`).
    #[derive(Clone, Copy, Debug)]
    pub struct Prepared {
        z: PreparedRow,
        w: PreparedRow,
        scale: f32,
        center: f32,
    }

    /// A clip row's columns as `product` takes them apart: per model
    /// coordinate (fraction bits, exponent, sign bit) with exponent 0 for
    /// "no term", and the translation's term (`product(1.0, column[3])`).
    #[derive(Clone, Copy, Debug)]
    struct PreparedRow {
        factors: [(i32, i32, i32); 3],
        translation: (i32, i32),
    }

    impl PreparedRow {
        fn new(column: [f32; 4]) -> Self {
            let factor = |value: f32| {
                let b = bits(value);
                ((b >> 8) & 32767, (b >> 23) & 255, b & i32::MIN)
            };
            PreparedRow { factors: [factor(column[0]), factor(column[1]), factor(column[2])], translation: product(1.0, column[3]) }
        }

        /// `clip(model, column)` from the model's parts (`model_parts`).
        fn clip(&self, parts: &[(i32, i32, i32); 3]) -> f32 {
            let term = |k: usize| {
                let (fa, ea, sa) = parts[k];
                let (fb, eb, sb) = self.factors[k];
                if ea == 0 || eb == 0 {
                    return NONE;
                }
                let m = 32768 + fa + fb + ((fa * fb) >> 15);
                (if (sa ^ sb) != 0 { -m } else { m }, ea + eb - 269)
            };
            row_sum([term(0), term(1), term(2), self.translation])
        }
    }

    impl Prepared {
        pub fn new(column_z: [f32; 4], column_w: [f32; 4], scale: f32, center: f32) -> Self {
            Prepared { z: PreparedRow::new(column_z), w: PreparedRow::new(column_w), scale, center }
        }

        /// `screen_z(model, column_z, column_w, scale, center)`.
        #[inline]
        pub fn screen_z(&self, model: [f32; 3]) -> Option<f32> {
            // Each coordinate truncated to a float24, taken apart as `product` does.
            let parts = model.map(|value| {
                let b = bits(truncate24(value));
                ((b >> 8) & 32767, (b >> 23) & 255, b & i32::MIN)
            });
            depth_of(truncate24(self.z.clip(&parts)), truncate24(self.w.clip(&parts)), self.scale, self.center).0
        }
    }

    /// `super::screen_z` in integer steps; `None` where it gives none.
    pub fn screen_z(model: [f32; 3], column_z: [f32; 4], column_w: [f32; 4], scale: f32, center: f32) -> Option<f32> {
        screen_z_steps(model, column_z, column_w, scale, center).0
    }

    /// `screen_z` and its intermediates (for device tests of the programs):
    /// clip z bits, clip w bits, q, scaled bits, sum bits.
    pub fn screen_z_steps(model: [f32; 3], column_z: [f32; 4], column_w: [f32; 4], scale: f32, center: f32) -> (Option<f32>, [i32; 5]) {
        let z = truncate24(clip(model, column_z));
        let w = truncate24(clip(model, column_w));
        depth_of(z, w, scale, center)
    }

    /// The screen depth from clip z and w (float24s): `screen_z_steps`
    /// from the clip rows on.
    fn depth_of(z: f32, w: f32, scale: f32, center: f32) -> (Option<f32>, [i32; 5]) {
        let mut steps = [0i32; 5];
        steps[0] = bits(z);
        steps[1] = bits(w);
        let ew = (bits(w) >> 23) & 255;
        if ew == 0 || ew >= 253 {
            return (None, steps);
        }
        // The reciprocal: q x 2^(-16 - e).
        let i = (bits(w) >> 8) & 32767;
        let (start, slope) = RECIP[(i >> 8) as usize];
        let q = (64 * start + 63 + slope * (i & 255)) >> 7;
        steps[2] = q;
        let negative = (bits(z) ^ bits(w)) < 0;
        let ez = (bits(z) >> 23) & 255;
        let es = (bits(scale) >> 23) & 255;
        let scaled = if ez == 0 || es == 0 {
            0.0
        } else {
            // ndc = z x reciprocal, truncated; then x scale, truncated.
            let (mz, wz) = mantissa(z);
            let (n, wn) = product16(mz, q);
            let ndc_weight = wz + (-16 - (ew - 127)) + wn;
            let (ms, ws) = mantissa(scale);
            let (p, wp) = product16(n, ms);
            let weight = ndc_weight + ws + wp;
            let half = weight / 2;
            let value = p as f32 * pow2(half) * pow2(weight - half);
            if negative != (bits(scale) < 0) {
                -value
            } else {
                value
            }
        };
        let value = add(scaled, center);
        steps[3] = bits(scaled);
        steps[4] = bits(value);
        (Some(super::floor(value).clamp(0.0, 65535.0)), steps)
    }
}

/// The GE's screen depth from float32 steps: plain adds and multiplies
/// rounded to nearest (ties to even, IEEE 754), a few bit masks, no
/// mantissa or exponent bookkeeping and no loops. Same results as
/// `integer::screen_z` (tests `float_steps_are_the_integer_way`,
/// `a_float_row_is_the_ges_row`) for inputs that are zero or normal floats
/// (no denormals: the integer way takes them as zero, here they would
/// stay tiny numbers) and wherever no product leaves the float range.
/// Written as the vertex program does it (nativeres_raw_v.cg GE_DEPTH):
/// no special cases, most steps four values wide.
///
/// The tools, from numerical analysis (error-free transformations: Dekker,
/// Veltkamp, Rump, Ogita, Oishi):
/// - A float24 has 16 significant bits. Split one factor into its top 8
///   bits and the rest: both partial products have at most 24 bits, so
///   float32 forms them exactly.
/// - `(x + c) - c` rounds x to the spacing of floats near c (a power of
///   two we choose through c); `x - rounded` is exact and tells which way
///   it went, so one step back turns "nearest" into "toward zero" (`cut`).
/// - Cutting each term at its own bit weight and then at the row's is the
///   same as cutting it once at the row's (both toward zero), so a row
///   needs one grid: the largest term's.
/// - A sum's rounding error is exact too (`low - (sum - high)`), which
///   says whether the nearest float lies beyond the true value.
pub mod float_steps {
    use super::RECIP;

    /// `f` with the low 8 mantissa bits cleared.
    fn truncate24(f: f32) -> f32 {
        f32::from_bits(f.to_bits() & 0xFFFF_FF00)
    }

    /// A float's sign and power of two: its bits without the mantissa.
    /// Zero for zero and denormals, which give the GE no term.
    fn unit(f: f32) -> f32 {
        f32::from_bits(f.to_bits() & 0xFF80_0000)
    }

    /// The top 8 of a float24's 16 significant bits, and the rest (exact).
    fn halves(f: f32) -> (f32, f32) {
        let high = f32::from_bits(f.to_bits() & 0xFFFF_0000);
        (high, f - high)
    }

    fn sign_of(f: f32) -> f32 {
        if f < 0.0 {
            -1.0
        } else {
            1.0
        }
    }

    /// `x` cut toward zero to a multiple of `step`'s size. `c` is
    /// 3 x 2^22 steps (floats of its size are one step apart, and x + c
    /// stays in that range for |x| below 2^22 steps); `c`, `step` and
    /// `sign` (+-1) carry x's sign.
    #[inline]
    fn cut(x: f32, c: f32, step: f32, sign: f32) -> f32 {
        let rounded = (x + c) - c;
        // Exact: at most half a step, on x's own grid.
        let error = x - rounded;
        if error * sign < 0.0 {
            rounded - step
        } else {
            rounded
        }
    }

    /// One clip row's column, per coordinate: the entry's two halves and
    /// its unit (a vertex program works these out once per draw).
    #[derive(Clone, Copy, Debug)]
    struct Row {
        high: [f32; 4],
        low: [f32; 4],
        unit: [f32; 4],
    }

    impl Row {
        fn new(column: [f32; 4]) -> Row {
            let mut row = Row { high: [0.0; 4], low: [0.0; 4], unit: [0.0; 4] };
            for (k, &entry) in column.iter().enumerate() {
                (row.high[k], row.low[k]) = halves(entry);
                row.unit[k] = unit(entry);
            }
            row
        }

        /// The GE's clip coordinate (`integer`'s `clip`): `v` = the model
        /// position as float24s and 1, `v_unit` their units.
        fn clip(&self, v: [f32; 4], v_unit: [f32; 4]) -> f32 {
            let mut terms = [0.0f32; 4];
            let mut weights = [0.0f32; 4];
            let mut top = 0.0f32;
            for k in 0..4 {
                // sign x 2^(both exponents): the term lies in [1, 4) of it.
                let weight = v_unit[k] * self.unit[k];
                // The product in two exact parts; the small one cut to the
                // large one's grid (2^-22 of the weight), so their sum is exact.
                let low = cut(v[k] * self.low[k], 3.0 * weight, weight * (1.0 / 4_194_304.0), sign_of(weight));
                terms[k] = v[k] * self.high[k] + low;
                weights[k] = weight;
                top = top.max(weight.abs());
            }
            // Every term cut at the largest term's 15th fraction bit, summed exactly.
            let grid = top * (1.0 / 32_768.0);
            let mut sum = 0.0f32;
            for k in 0..4 {
                let sign = sign_of(weights[k]);
                sum += cut(terms[k], sign * grid * 12_582_912.0, sign * grid, sign);
            }
            truncate24(sum)
        }
    }

    /// `a x b` cut to 16 significant bits toward zero, for float24s a and
    /// b = `b_high` + `b_low` (`halves`), none zero (`ProductToFloat24`).
    fn product24(a: f32, b_high: f32, b_low: f32) -> f32 {
        let (high, low) = (a * b_high, a * b_low);
        let sum = high + low;
        // Exact (|high| >= |low|): what the sum's rounding left out.
        let error = low - (sum - high);
        // Opposite signs: rounded away from zero, the true product lies
        // before `sum`. One float step back (a step of the sum's own size:
        // at a power of two that is two floats, still inside the 16-bit cell
        // before it) gives the right top 16 bits. A product so small that
        // `error * sum` underflows would miss the step: far below any depth.
        let back = if error * sum < 0.0 { unit(sum) * (1.0 / 8_388_608.0) } else { 0.0 };
        truncate24(sum - back)
    }

    /// The GE's adder (`GEAdd`) for float24s: both terms cut at the larger
    /// one's 15th fraction bit, then added (exact). A zero term leaves the
    /// other as it is.
    fn add(a: f32, b: f32) -> f32 {
        let grid = unit(a).abs().max(unit(b).abs()) * (1.0 / 32_768.0);
        let part = |x: f32| {
            let sign = sign_of(x);
            cut(x, sign * grid * 12_582_912.0, sign * grid, sign)
        };
        part(a) + part(b)
    }

    /// `integer::Prepared` in float steps: what the columns, the scale and
    /// the centre give, worked out once per transform.
    #[derive(Clone, Copy, Debug)]
    pub struct Prepared {
        z: Row,
        w: Row,
        scale_high: f32,
        scale_low: f32,
        scale_is_zero: bool,
        center: f32,
    }

    impl Prepared {
        pub fn new(column_z: [f32; 4], column_w: [f32; 4], scale: f32, center: f32) -> Self {
            let (scale_high, scale_low) = halves(scale);
            Prepared { z: Row::new(column_z), w: Row::new(column_w), scale_high, scale_low, scale_is_zero: unit(scale) == 0.0, center }
        }

        /// `integer::screen_z(model, column_z, column_w, scale, center)`.
        pub fn screen_z(&self, model: [f32; 3]) -> Option<f32> {
            let v = [truncate24(model[0]), truncate24(model[1]), truncate24(model[2]), 1.0];
            let v_unit = v.map(unit);
            self.depth_of(self.z.clip(v, v_unit), self.w.clip(v, v_unit))
        }

        /// The screen depth from clip z and w (float24s).
        fn depth_of(&self, z: f32, w: f32) -> Option<f32> {
            let w_bits = w.to_bits();
            let exponent = (w_bits >> 23) & 0xFF;
            if exponent == 0 || exponent >= 253 {
                return None;
            }
            // sign x 2^-e of w: its exponent mirrored.
            let inverse_unit = f32::from_bits((w_bits & 0x8000_0000) | (0x7F00_0000 - (w_bits & 0x7F80_0000)));
            // The reciprocal's table step, in whole numbers below 2^24:
            // i = w's 15 fraction bits, q = (64 start + 63 + slope (i & 255)) >> 7.
            let i = (w * inverse_unit - 1.0) * 32_768.0;
            let segment = super::floor(i * (1.0 / 256.0));
            let (start, slope) = RECIP[segment as usize];
            let q = super::floor((64.0 * start as f32 + 63.0 + slope as f32 * (i - 256.0 * segment)) * (1.0 / 128.0));
            // q x 2^(-16 - e), with w's sign.
            let reciprocal = q * (inverse_unit * (1.0 / 65_536.0));
            let scaled = if unit(z) == 0.0 || self.scale_is_zero {
                0.0
            } else {
                // ndc = z x reciprocal, cut; then x scale, cut.
                let (reciprocal_high, reciprocal_low) = halves(reciprocal);
                product24(product24(z, reciprocal_high, reciprocal_low), self.scale_high, self.scale_low)
            };
            Some(super::floor(add(scaled, self.center)).clamp(0.0, 65535.0))
        }
    }

    /// `integer::screen_z` in float steps; `None` where it gives none.
    pub fn screen_z(model: [f32; 3], column_z: [f32; 4], column_w: [f32; 4], scale: f32, center: f32) -> Option<f32> {
        Prepared::new(column_z, column_w, scale, center).screen_z(model)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn generator(seed: u32) -> impl FnMut() -> u32 {
            let mut state = seed;
            move || {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state
            }
        }

        /// Zero (either sign) or a float24 within `range` powers of two of 1;
        /// now and then a mantissa of all zeros, all ones or a clear low
        /// byte. No denormals: the float steps take none (see the module).
        fn float24(next: &mut impl FnMut() -> u32, range: u32) -> f32 {
            match next() % 16 {
                0 => 0.0,
                1 => -0.0,
                kind => {
                    let exponent = 127 + (next() % (2 * range + 1)) as i32 - range as i32;
                    let mantissa = match kind {
                        2 => 0,
                        3 => 0x007F_FF00,
                        4 => next() & 0x007F_0000,
                        _ => next() & 0x007F_FF00,
                    };
                    f32::from_bits((next() & 0x8000_0000) | ((exponent as u32) << 23) | mantissa)
                }
            }
        }

        /// A clip row from float steps is the GE's row, bit for bit: random
        /// float24 columns and positions up to 30 powers of two from 1.
        #[test]
        fn a_float_row_is_the_ges_row() {
            let mut next = generator(0x1357_9BDF);
            for range in [3, 8, 20, 30] {
                for round in 0..400_000 {
                    let column = [float24(&mut next, range), float24(&mut next, range), float24(&mut next, range), float24(&mut next, range)];
                    // Positions with all 24 bits: the row truncates them itself.
                    let model = [0, 1, 2].map(|_| f32::from_bits(float24(&mut next, range).to_bits() | (next() & 0xFF)));
                    let mut matrix = [0.0f32; 16];
                    for k in 0..4 {
                        matrix[4 * k + 2] = column[k];
                    }
                    let expected = crate::ge_math::clip_component(model, &matrix, 2);
                    let v = [truncate24(model[0]), truncate24(model[1]), truncate24(model[2]), 1.0];
                    let got = Row::new(column).clip(v, v.map(unit));
                    assert!(got == expected, "range {range} round {round}: {model:?} {column:?}: {got} ({:08x}), the GE's {expected} ({:08x})", got.to_bits(), expected.to_bits());
                }
            }
        }

        /// The whole depth from float steps is `integer::screen_z`: random
        /// columns, positions, scales and centres, with zeros and denormals.
        #[test]
        fn float_steps_are_the_integer_way() {
            let mut next = generator(0x2468_ACE1);
            let mut with_depth = 0;
            for round in 0..300_000 {
                let range = [4, 10, 20][round % 3];
                let column_z = [float24(&mut next, range), float24(&mut next, range), float24(&mut next, range), float24(&mut next, range)];
                let column_w = [float24(&mut next, range), float24(&mut next, range), float24(&mut next, range), float24(&mut next, range)];
                let scale = float24(&mut next, 16);
                let center = truncate24(f32::from_bits(next() % 0x4780_0000));
                let prepared = Prepared::new(column_z, column_w, scale, center);
                for _ in 0..4 {
                    let model = [0, 1, 2].map(|_| f32::from_bits(float24(&mut next, range).to_bits() | (next() & 0xFF)));
                    let expected = crate::ge_math::integer::screen_z(model, column_z, column_w, scale, center);
                    let got = prepared.screen_z(model);
                    assert!(got == expected, "round {round}: {model:?} z {column_z:?} w {column_w:?} scale {scale} center {center}: {got:?}, the integer way {expected:?}");
                    with_depth += expected.is_some() as u32;
                }
            }
            assert!(with_depth > 600_000, "{with_depth}");
        }

        /// God of War's title camera, as `the_integer_way_is_the_ges_way`.
        #[test]
        fn float_steps_are_the_ges_way_for_a_game_camera() {
            let view = [[-0.9155426, 0.0, 0.40221405, 0.0], [0.0, 1.0, 0.0, 0.0], [-0.40221405, 0.0, -0.9155426, 0.0]];
            let projection = [[3.569519, 0.0, 0.0, 0.0], [0.0, 6.2991943, 0.0, 0.0], [0.0, 0.0, -1.0039978, -0.40079498], [0.0, 0.0, -1.0, 0.0]];
            let float24 = |value: f32| f32::from_bits((value.to_bits() + 0x80) & 0xFFFF_FF00);
            let (scale, center) = (float24(32767.5), float24(32767.5));
            let mut seed = 0x1234_5678u32;
            let mut random = move |range: f32| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((seed >> 8) as f32 / 16_777_216.0 - 0.5) * 2.0 * range
            };
            for set in 0..256 {
                let s = 1.0 + random(0.5);
                let world = [[s, random(0.2), 0.0, random(20.0)], [0.0, s, random(0.2), random(10.0)], [random(0.2), 0.0, s, 40.0 + random(60.0) + 5.0 * set as f32]];
                let matrix = crate::ge_math::combined(&world, &view, &projection);
                let column = |c: usize| [matrix[c], matrix[4 + c], matrix[8 + c], matrix[12 + c]];
                let prepared = Prepared::new(column(2), column(3), scale, center);
                for _ in 0..4000 {
                    let model = [random(60.0), random(60.0), random(60.0)];
                    let exact = crate::ge_math::screen_z(model, &world, &view, &projection, scale, center);
                    assert_eq!(prepared.screen_z(model), exact, "model {model:?} set {set}");
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_reciprocal_of_powers_of_two_is_exact() {
        assert_eq!(recip(1.0), 1.0);
        assert_eq!(recip(4.0), 0.25);
        assert_eq!(recip(-2.0), -0.5);
    }

    #[test]
    fn the_adder_drops_the_smaller_terms_low_bits() {
        // 2048 + 1/256: below the larger term's 16 significant bits.
        assert_eq!(add(2048.0, 1.0 / 256.0), 2048.0);
        assert_eq!(add(2048.0, 1.0), 2049.0);
    }

    #[test]
    fn an_identity_transform_keeps_the_depth() {
        let identity = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]];
        let projection = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
        // z 0.5, w 1: 32768 + 32767 x 0.5 = 49151.5 -> 49151.
        assert_eq!(screen_z([0.0, 0.0, 0.5], &identity, &identity, &projection, 32767.0, 32768.0), Some(49151.0));
    }

    #[test]
    fn the_integer_way_is_the_ges_way() {
        // God of War's title camera (trace prim 198), worlds moved around
        // and random model positions.
        let view = [[-0.9155426, 0.0, 0.40221405, 0.0], [0.0, 1.0, 0.0, 0.0], [-0.40221405, 0.0, -0.9155426, 0.0]];
        let projection = [[3.569519, 0.0, 0.0, 0.0], [0.0, 6.2991943, 0.0, 0.0], [0.0, 0.0, -1.0039978, -0.40079498], [0.0, 0.0, -1.0, 0.0]];
        let float24 = |value: f32| f32::from_bits((value.to_bits() + 0x80) & 0xFFFF_FF00);
        let (scale, center) = (float24(32767.5), float24(32767.5));
        let mut seed = 0x1234_5678u32;
        let mut random = move |range: f32| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((seed >> 8) as f32 / 16_777_216.0 - 0.5) * 2.0 * range
        };
        let (mut tested, mut none) = (0u32, 0u32);
        for set in 0..64 {
            let s = 1.0 + random(0.5);
            let world = [[s, random(0.2), 0.0, random(20.0)], [0.0, s, random(0.2), random(10.0)], [random(0.2), 0.0, s, 40.0 + random(60.0) + set as f32]];
            let matrix = combined(&world, &view, &projection);
            let column = |c: usize| [matrix[c], matrix[4 + c], matrix[8 + c], matrix[12 + c]];
            for _ in 0..2000 {
                let model = [random(60.0), random(60.0), random(60.0)];
                let exact = screen_z(model, &world, &view, &projection, scale, center);
                let integer = integer::screen_z(model, column(2), column(3), scale, center);
                assert_eq!(exact, integer, "model {model:?} set {set}");
                tested += 1;
                none += exact.is_none() as u32;
            }
        }
        assert!(none < tested / 10, "{none} of {tested} without a depth");
    }
}
