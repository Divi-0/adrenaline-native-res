//! Texgen draws on the CPU route (`raw_arena::TEXGEN` off, the default:
//! the Vita's GPU doesn't round its float maths like the ARM, so raw
//! texgen draws can't give the CPU route's positions), four vertices at a
//! time: `TexgenKernel::transform4` does what `Transform::frame_vertex`
//! and `Texgen::uv` do for one vertex, with the same operations in the same
//! order, so the vertices come out the same to the bit
//! (`recorder::record_texgen_run` uses it; the frame hashes are the
//! general path's).
//!
//! On the Vita with NEON (PPSSPP's ARM vertex decoder does its matrix work
//! with NEON too): ARMv7 NEON rounds after each multiply and add (VMLA is
//! not fused) and always to nearest, as the VFP does here; it flushes
//! denormals to zero, which the VFP does too only with FPSCR.FZ (debug
//! command `fpscr`; `RowTransform::clip` relies on the same). 1 / w and the
//! projective texgen's divides are VFP divides per vertex (NEON has no
//! exact divide). On the PC the same steps lane by lane.
//!
//! NFS Underground Rivals' main menu draws ~32,000 such vertices per frame
//! (strips of ~160 vertices, s16 positions, u16 texture coordinates, 565
//! colours, texgen from the texture coordinates); the per-vertex general
//! path took ~1,500 Vita cycles per vertex.

use crate::frame::FrameVertex;

/// `Transform::frame_vertex` with `Texgen::uv`'s texture coordinates, for
/// an unskinned draw whose texgen comes from the position or the texture
/// coordinates (`Transform::texgen_kernel`).
#[derive(Clone, Copy, Debug)]
pub struct TexgenKernel {
    /// Clip = rows x (raw x, y, z, 1), row by row (`FastPath::rows`).
    pub(crate) rows: [[f32; 4]; 4],
    pub(crate) fog_row: Option<[f32; 4]>,
    pub(crate) viewport_origin: (f32, f32),
    pub(crate) viewport_scale: (f32, f32),
    pub(crate) depth_center: f32,
    pub(crate) depth_scale: f32,
    /// Texgen source: the texture coordinates (true) or the position.
    pub(crate) from_texcoords: bool,
    /// The source's format scale (texture coordinates or position).
    pub(crate) source_scale: f32,
    /// The texgen matrix, 4x3, column by column (`Texgen::matrix`).
    pub(crate) matrix: [f32; 12],
    pub(crate) projective: bool,
}

/// Four vertices' raw fields lane by lane, as `VertexFormat::decode_3d`
/// gives them, and their finished colours (`frame_color`).
#[derive(Clone, Copy, Default)]
pub struct Lanes4 {
    pub x: [f32; 4],
    pub y: [f32; 4],
    pub z: [f32; 4],
    pub u: [f32; 4],
    pub v: [f32; 4],
    pub color: [u32; 4],
}

impl TexgenKernel {
    /// Four vertices to the frame: `out[n]` for lanes n < out.len() (1..=4;
    /// the other lanes are worked out but not stored). Returns the lanes
    /// behind the near plane (bit n: lane n), where `frame_vertex` gives
    /// `None`; their `out` entries are garbage.
    #[inline(always)]
    pub fn transform4(&self, input: &Lanes4, out: &mut [FrameVertex]) -> u32 {
        let (x, y, z) = (F4::load(&input.x), F4::load(&input.y), F4::load(&input.z));
        // ((r0 x + r1 y) + r2 z) + r3 per row, as `frame_vertex`'s dot.
        let dot = |r: &[f32; 4]| x.mul_n(r[0]).mla_n(y, r[1]).mla_n(z, r[2]).add(F4::splat(r[3]));
        let clip_w = dot(&self.rows[3]);
        let clip_z = dot(&self.rows[2]);
        let behind = F4::behind(clip_z, clip_w);
        // 1 / w per lane on the VFP (exact; NEON's estimate isn't).
        let w = clip_w.to_array();
        let over_w = F4::load(&[1.0 / w[0], 1.0 / w[1], 1.0 / w[2], 1.0 / w[3]]);
        // origin + (scale x clip) x (1 / w).
        let screen_x = F4::splat(self.viewport_origin.0).add(dot(&self.rows[0]).mul_n(self.viewport_scale.0).mul(over_w));
        let screen_y = F4::splat(self.viewport_origin.1).add(dot(&self.rows[1]).mul_n(self.viewport_scale.1).mul(over_w));
        let depth = F4::splat(self.depth_center).add(clip_z.mul_n(self.depth_scale).mul(over_w)).clamp01();
        let fog = match &self.fog_row {
            Some(row) => dot(row),
            None => F4::splat(1.0),
        };
        // `Texgen::uv`: source x matrix, row i as
        // ((s0 m[i] + s1 m[3 + i]) + s2 m[6 + i]) + m[9 + i].
        let s = self.source_scale;
        let source = if self.from_texcoords {
            [F4::load(&input.u).mul_n(s), F4::load(&input.v).mul_n(s), F4::splat(0.0)]
        } else {
            [x.mul_n(s), y.mul_n(s), z.mul_n(s)]
        };
        let m = &self.matrix;
        let row = |i: usize| source[0].mul_n(m[i]).mla_n(source[1], m[3 + i]).mla_n(source[2], m[6 + i]).add(F4::splat(m[9 + i]));
        let (mut u, mut v) = (row(0), row(1));
        if self.projective {
            let (mut us, mut vs, ws) = (u.to_array(), v.to_array(), row(2).to_array());
            for lane in 0..4 {
                if ws[lane] != 0.0 {
                    us[lane] /= ws[lane];
                    vs[lane] /= ws[lane];
                }
            }
            (u, v) = (F4::load(&us), F4::load(&vs));
        }
        let count = out.len().min(4);
        F4::store(out, [screen_x, screen_y, depth, clip_w, u, v, F4::from_bits(&input.color), fog]);
        behind & ((1 << count) - 1)
    }
}

/// Four f32 lanes: NEON on the Vita.
#[cfg(target_arch = "arm")]
#[derive(Clone, Copy)]
struct F4(core::arch::arm::float32x4_t);

// Safe throughout: the Vita's Cortex-A9 has NEON (the target enables it);
// loads and stores touch exactly the four lanes' memory.
#[cfg(target_arch = "arm")]
impl F4 {
    #[inline(always)]
    fn load(lanes: &[f32; 4]) -> F4 {
        unsafe { F4(core::arch::arm::vld1q_f32(lanes.as_ptr())) }
    }

    #[inline(always)]
    fn from_bits(lanes: &[u32; 4]) -> F4 {
        use core::arch::arm::{vld1q_u32, vreinterpretq_f32_u32};
        unsafe { F4(vreinterpretq_f32_u32(vld1q_u32(lanes.as_ptr()))) }
    }

    #[inline(always)]
    fn splat(value: f32) -> F4 {
        unsafe { F4(core::arch::arm::vdupq_n_f32(value)) }
    }

    #[inline(always)]
    fn to_array(self) -> [f32; 4] {
        let mut lanes = [0.0; 4];
        unsafe { core::arch::arm::vst1q_f32(lanes.as_mut_ptr(), self.0) };
        lanes
    }

    #[inline(always)]
    fn add(self, other: F4) -> F4 {
        unsafe { F4(core::arch::arm::vaddq_f32(self.0, other.0)) }
    }

    #[inline(always)]
    fn mul(self, other: F4) -> F4 {
        unsafe { F4(core::arch::arm::vmulq_f32(self.0, other.0)) }
    }

    /// self x n.
    #[inline(always)]
    fn mul_n(self, n: f32) -> F4 {
        unsafe { F4(core::arch::arm::vmulq_n_f32(self.0, n)) }
    }

    /// self + a x n, the product rounded before the add (VMLA).
    #[inline(always)]
    fn mla_n(self, a: F4, n: f32) -> F4 {
        unsafe { F4(core::arch::arm::vmlaq_n_f32(self.0, a.0, n)) }
    }

    /// `f32::clamp(0.0, 1.0)` per lane: below 0 -> 0, then above 1 -> 1
    /// (NaN and -0 kept, unlike VMAX / VMIN).
    #[inline(always)]
    fn clamp01(self) -> F4 {
        use core::arch::arm::{vbslq_f32, vcgtq_f32, vcltq_f32, vdupq_n_f32};
        unsafe {
            let (zero, one) = (vdupq_n_f32(0.0), vdupq_n_f32(1.0));
            let low = vbslq_f32(vcltq_f32(self.0, zero), zero, self.0);
            F4(vbslq_f32(vcgtq_f32(low, one), one, low))
        }
    }

    /// Lanes where !(z >= -w && w > 0): behind the near plane.
    #[inline(always)]
    fn behind(z: F4, w: F4) -> u32 {
        use core::arch::arm::{vandq_u32, vcgeq_f32, vcgtq_f32, vdupq_n_f32, vnegq_f32, vst1q_u32};
        let mut lanes = [0u32; 4];
        unsafe {
            let front = vandq_u32(vcgeq_f32(z.0, vnegq_f32(w.0)), vcgtq_f32(w.0, vdupq_n_f32(0.0)));
            vst1q_u32(lanes.as_mut_ptr(), front);
        }
        lanes.iter().enumerate().fold(0, |mask, (lane, &front)| mask | ((front == 0) as u32) << lane)
    }

    /// `out[n]` = lane n of the eight fields (x, y, z, w, u, v, colour,
    /// fog: `FrameVertex`'s order) for n < out.len(). Four vertices: two
    /// 4x4 transposes in registers and eight 16-byte stores.
    #[inline(always)]
    fn store(out: &mut [FrameVertex], fields: [F4; 8]) {
        use core::arch::arm::{float32x4_t, vcombine_f32, vget_high_f32, vget_low_f32, vst1q_f32, vtrnq_f32};
        // Rows of a 4x4 block from its columns a, b, c, d.
        let transpose = |a: float32x4_t, b: float32x4_t, c: float32x4_t, d: float32x4_t| unsafe {
            let ab = vtrnq_f32(a, b); // (a0 b0 a2 b2), (a1 b1 a3 b3)
            let cd = vtrnq_f32(c, d);
            [
                vcombine_f32(vget_low_f32(ab.0), vget_low_f32(cd.0)),
                vcombine_f32(vget_low_f32(ab.1), vget_low_f32(cd.1)),
                vcombine_f32(vget_high_f32(ab.0), vget_high_f32(cd.0)),
                vcombine_f32(vget_high_f32(ab.1), vget_high_f32(cd.1)),
            ]
        };
        let low = transpose(fields[0].0, fields[1].0, fields[2].0, fields[3].0);
        let high = transpose(fields[4].0, fields[5].0, fields[6].0, fields[7].0);
        for (n, vertex) in out.iter_mut().take(4).enumerate() {
            // Safe: FrameVertex is 8 four-byte fields (repr(C)); the stores
            // write its two halves.
            unsafe {
                let at = (vertex as *mut FrameVertex).cast::<f32>();
                vst1q_f32(at, low[n]);
                vst1q_f32(at.add(4), high[n]);
            }
        }
    }
}

/// Four f32 lanes: the PC, each operation lane by lane (the same IEEE
/// single-precision results, but for denormals: NEON flushes them).
#[cfg(not(target_arch = "arm"))]
#[derive(Clone, Copy)]
struct F4([f32; 4]);

#[cfg(not(target_arch = "arm"))]
impl F4 {
    #[inline(always)]
    fn load(lanes: &[f32; 4]) -> F4 {
        F4(*lanes)
    }

    #[inline(always)]
    fn from_bits(lanes: &[u32; 4]) -> F4 {
        F4(lanes.map(f32::from_bits))
    }

    #[inline(always)]
    fn splat(value: f32) -> F4 {
        F4([value; 4])
    }

    #[inline(always)]
    fn to_array(self) -> [f32; 4] {
        self.0
    }

    #[inline(always)]
    fn zip(self, other: F4, f: impl Fn(f32, f32) -> f32) -> F4 {
        F4(core::array::from_fn(|lane| f(self.0[lane], other.0[lane])))
    }

    #[inline(always)]
    fn add(self, other: F4) -> F4 {
        self.zip(other, |a, b| a + b)
    }

    #[inline(always)]
    fn mul(self, other: F4) -> F4 {
        self.zip(other, |a, b| a * b)
    }

    #[inline(always)]
    fn mul_n(self, n: f32) -> F4 {
        F4(self.0.map(|a| a * n))
    }

    #[inline(always)]
    fn mla_n(self, a: F4, n: f32) -> F4 {
        self.zip(a, |acc, a| acc + a * n)
    }

    #[inline(always)]
    fn clamp01(self) -> F4 {
        F4(self.0.map(|value| value.clamp(0.0, 1.0)))
    }

    #[inline(always)]
    fn behind(z: F4, w: F4) -> u32 {
        (0..4).fold(0, |mask, lane| mask | (!(z.0[lane] >= -w.0[lane] && w.0[lane] > 0.0) as u32) << lane)
    }

    #[inline(always)]
    fn store(out: &mut [FrameVertex], fields: [F4; 8]) {
        for (n, vertex) in out.iter_mut().take(4).enumerate() {
            let [x, y, z, w, u, v, color, fog] = fields.map(|field| field.0[n]);
            *vertex = FrameVertex { x, y, z, w, u, v, color: color.to_bits(), fog };
        }
    }
}
