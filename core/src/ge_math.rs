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
const RECIP: [(i32, i32); 128] = [
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
    let mode = GE_EXACT_DEPTH.load(Ordering::Relaxed);
    let (z, w) = if mode == 2 {
        // Float clip coordinates as our programs compute them (world, view, projection in turn).
        let row = |m: &[f32; 4], v: [f32; 3]| m[0] * v[0] + m[1] * v[1] + m[2] * v[2] + m[3];
        let world_position = world.map(|r| row(&r, model));
        let view_position = view.map(|r| row(&r, world_position));
        (row(&projection[2], view_position), row(&projection[3], view_position))
    } else {
        let matrix = combine(&combine(&affine(world), &affine(view)), &proj);
        (clip_component(model, &matrix, 2), clip_component(model, &matrix, 3))
    };
    let value = if mode == 3 { Some(center + scale * z / w) } else { viewport(z, w, scale, center) };
    value.map(|value| floor(value).clamp(0.0, 65535.0))
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
}
