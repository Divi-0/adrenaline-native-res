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
//! not fused) and always to nearest, flushing denormals to zero; the
//! Vita's threads run the VFP that way too (FPSCR 0x03000000: FZ, DN,
//! round to nearest; debug command `fpscr`). 1 / w and the projective
//! texgen's divides are VFP divides per lane (NEON has no exact divide). On
//! the PC the same steps lane by lane.
//!
//! The Cortex-A9 stalls when NEON results go to the ARM side through
//! memory (and the lanes of a vector built in memory by ARM stores): the
//! fields are loaded into lanes directly (`Fields4::gather`), colours and
//! normals moved into lanes register to register, lanes taken out for the
//! divides the same way (the VFP's registers are NEON's), and the lanes'
//! "in front" flags stored for the caller to read a chunk later.
//!
//! NFS Underground Rivals' main menu draws ~32,000 such vertices per frame
//! (strips of ~160 vertices, s16 positions, u16 texture coordinates, 565
//! colours, texgen from the texture coordinates); the per-vertex general
//! path took ~1,500 Vita cycles per vertex.

use crate::frame::FrameVertex;

/// `Transform::frame_vertex` with `Texgen::uv`'s texture coordinates, for
/// an unskinned draw (`Transform::texgen_kernel`).
#[derive(Clone, Copy, Debug)]
pub struct TexgenKernel {
    /// Clip = rows x (raw x, y, z, 1), row by row (`FastPath::rows`).
    pub(crate) rows: [[f32; 4]; 4],
    pub(crate) fog_row: Option<[f32; 4]>,
    pub(crate) viewport_origin: (f32, f32),
    pub(crate) viewport_scale: (f32, f32),
    pub(crate) depth_center: f32,
    pub(crate) depth_scale: f32,
    /// Texgen source (`Texgen::source`): 0 the position, 1 the texture
    /// coordinates (each times `source_scale`, its format scale), 2 and 3
    /// the normal, normalized or not: `transform4`'s `normal` as it is.
    pub(crate) source: u32,
    pub(crate) source_scale: f32,
    /// The texgen matrix, 4x3, column by column (`Texgen::matrix`).
    pub(crate) matrix: [f32; 12],
    pub(crate) projective: bool,
}

/// Four vertices' raw position and texture coordinates, lane by lane, as
/// `VertexFormat::decode_3d` gives them.
#[derive(Clone, Copy)]
pub struct Fields4 {
    x: F4,
    y: F4,
    z: F4,
    u: F4,
    v: F4,
}

impl Fields4 {
    /// From the four vertices' values.
    #[inline(always)]
    pub fn from_lanes(x: [f32; 4], y: [f32; 4], z: [f32; 4], u: [f32; 4], v: [f32; 4]) -> Fields4 {
        Fields4 { x: F4::from_lanes(x), y: F4::from_lanes(y), z: F4::from_lanes(z), u: F4::from_lanes(u), v: F4::from_lanes(v) }
    }

    /// s16 positions at `position_offset` and u16 texture coordinates at
    /// `texcoord_offset` of the vertices at `first + lane x stride` (lanes
    /// from `count` on: the last vertex again), each value as `as f32`
    /// makes it (exact).
    ///
    /// # Safety
    /// The `count` vertices' fields must be readable, at even addresses.
    #[inline(always)]
    pub unsafe fn gather(first: *const u8, stride: usize, count: usize, position_offset: usize, texcoord_offset: usize) -> Fields4 {
        let vertex = |lane: usize| first.add(lane.min(count - 1) * stride);
        let p = [0, 1, 2, 3].map(|lane| vertex(lane).add(position_offset).cast::<i16>());
        let t = [0, 1, 2, 3].map(|lane| vertex(lane).add(texcoord_offset).cast::<u16>());
        Fields4 {
            x: F4::gather_i16(p),
            y: F4::gather_i16(p.map(|at| at.add(1))),
            z: F4::gather_i16(p.map(|at| at.add(2))),
            u: F4::gather_u16(t),
            v: F4::gather_u16(t.map(|at| at.add(1))),
        }
    }
}

impl TexgenKernel {
    /// Four vertices to the frame, with their finished colours and (texgen
    /// from the normal) source vectors: `out[n]` for lanes n < out.len()
    /// (1..=4; the other lanes are worked out but not stored). `front[n]`:
    /// 0xFFFF if lane n is in front of the near plane, 0 if behind (where
    /// `frame_vertex` gives `None`; its `out` entry is garbage then).
    #[inline(always)]
    pub fn transform4(&self, fields: &Fields4, colors: [u32; 4], normal: &[[f32; 4]; 3], out: &mut [FrameVertex], front: &mut [u16; 4]) {
        let Fields4 { x, y, z, u: raw_u, v: raw_v } = *fields;
        // ((r0 x + r1 y) + r2 z) + r3 per row, as `frame_vertex`'s dot.
        let dot = |r: &[f32; 4]| x.mul_n(r[0]).mla_n(y, r[1]).mla_n(z, r[2]).add(F4::splat(r[3]));
        let clip_w = dot(&self.rows[3]);
        let clip_z = dot(&self.rows[2]);
        F4::store_front(clip_z, clip_w, front);
        // 1 / w per lane on the VFP (exact; NEON's estimate isn't).
        let w = clip_w.lanes();
        let over_w = F4::from_lanes([1.0 / w[0], 1.0 / w[1], 1.0 / w[2], 1.0 / w[3]]);
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
        let source = match self.source {
            0 => [x.mul_n(s), y.mul_n(s), z.mul_n(s)],
            1 => [raw_u.mul_n(s), raw_v.mul_n(s), F4::splat(0.0)],
            _ => normal.map(F4::from_lanes),
        };
        let m = &self.matrix;
        let row = |i: usize| source[0].mul_n(m[i]).mla_n(source[1], m[3 + i]).mla_n(source[2], m[6 + i]).add(F4::splat(m[9 + i]));
        let (mut u, mut v) = (row(0), row(1));
        if self.projective {
            let (mut us, mut vs, ws) = (u.lanes(), v.lanes(), row(2).lanes());
            for lane in 0..4 {
                if ws[lane] != 0.0 {
                    us[lane] /= ws[lane];
                    vs[lane] /= ws[lane];
                }
            }
            (u, v) = (F4::from_lanes(us), F4::from_lanes(vs));
        }
        F4::store(out, [screen_x, screen_y, depth, clip_w, u, v, F4::from_bits(colors), fog]);
    }
}

/// Four f32 lanes: NEON on the Vita.
#[cfg(target_arch = "arm")]
#[derive(Clone, Copy)]
struct F4(core::arch::arm::float32x4_t);

// Safe throughout: the Vita's Cortex-A9 has NEON (the target enables it);
// loads and stores touch exactly the lanes' memory.
#[cfg(target_arch = "arm")]
impl F4 {
    /// Register to register (VMOV into the lanes).
    #[inline(always)]
    fn from_lanes(lanes: [f32; 4]) -> F4 {
        use core::arch::arm::{vdupq_n_f32, vsetq_lane_f32};
        unsafe {
            let v = vsetq_lane_f32::<0>(lanes[0], vdupq_n_f32(0.0));
            let v = vsetq_lane_f32::<1>(lanes[1], v);
            let v = vsetq_lane_f32::<2>(lanes[2], v);
            F4(vsetq_lane_f32::<3>(lanes[3], v))
        }
    }

    /// The bits as lanes (ARM registers into NEON).
    #[inline(always)]
    fn from_bits(lanes: [u32; 4]) -> F4 {
        use core::arch::arm::{vdupq_n_u32, vreinterpretq_f32_u32, vsetq_lane_u32};
        unsafe {
            let v = vsetq_lane_u32::<0>(lanes[0], vdupq_n_u32(0));
            let v = vsetq_lane_u32::<1>(lanes[1], v);
            let v = vsetq_lane_u32::<2>(lanes[2], v);
            F4(vreinterpretq_f32_u32(vsetq_lane_u32::<3>(lanes[3], v)))
        }
    }

    /// The lanes as VFP values (the same registers: no memory).
    #[inline(always)]
    fn lanes(self) -> [f32; 4] {
        use core::arch::arm::vgetq_lane_f32;
        unsafe { [vgetq_lane_f32::<0>(self.0), vgetq_lane_f32::<1>(self.0), vgetq_lane_f32::<2>(self.0), vgetq_lane_f32::<3>(self.0)] }
    }

    /// The i16s at `at` (2-byte aligned), each `as f32`.
    #[inline(always)]
    unsafe fn gather_i16(at: [*const i16; 4]) -> F4 {
        use core::arch::arm::{vcvtq_f32_s32, vdup_n_s16, vld1_lane_s16, vmovl_s16};
        let v = vld1_lane_s16::<0>(at[0], vdup_n_s16(0));
        let v = vld1_lane_s16::<1>(at[1], v);
        let v = vld1_lane_s16::<2>(at[2], v);
        let v = vld1_lane_s16::<3>(at[3], v);
        F4(vcvtq_f32_s32(vmovl_s16(v)))
    }

    /// The u16s at `at` (2-byte aligned), each `as f32`.
    #[inline(always)]
    unsafe fn gather_u16(at: [*const u16; 4]) -> F4 {
        use core::arch::arm::{vcvtq_f32_u32, vdup_n_u16, vld1_lane_u16, vmovl_u16};
        let v = vld1_lane_u16::<0>(at[0], vdup_n_u16(0));
        let v = vld1_lane_u16::<1>(at[1], v);
        let v = vld1_lane_u16::<2>(at[2], v);
        let v = vld1_lane_u16::<3>(at[3], v);
        F4(vcvtq_f32_u32(vmovl_u16(v)))
    }

    #[inline(always)]
    fn splat(value: f32) -> F4 {
        unsafe { F4(core::arch::arm::vdupq_n_f32(value)) }
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

    /// `front[n]` = 0xFFFF where z >= -w && w > 0 (in front of the near
    /// plane), else 0; one NEON store, read by the caller later.
    #[inline(always)]
    fn store_front(z: F4, w: F4, front: &mut [u16; 4]) {
        use core::arch::arm::{vandq_u32, vcgeq_f32, vcgtq_f32, vdupq_n_f32, vmovn_u32, vnegq_f32, vst1_u16};
        unsafe {
            let ahead = vandq_u32(vcgeq_f32(z.0, vnegq_f32(w.0)), vcgtq_f32(w.0, vdupq_n_f32(0.0)));
            vst1_u16(front.as_mut_ptr(), vmovn_u32(ahead));
        }
    }

    /// `out[n]` = lane n of the eight fields (x, y, z, w, u, v, colour,
    /// fog: `FrameVertex`'s order) for n < out.len(): two 4x4 transposes
    /// in registers and two 16-byte stores per vertex.
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
    fn from_lanes(lanes: [f32; 4]) -> F4 {
        F4(lanes)
    }

    #[inline(always)]
    fn from_bits(lanes: [u32; 4]) -> F4 {
        F4(lanes.map(f32::from_bits))
    }

    #[inline(always)]
    fn lanes(self) -> [f32; 4] {
        self.0
    }

    #[inline(always)]
    unsafe fn gather_i16(at: [*const i16; 4]) -> F4 {
        F4(at.map(|at| at.read_unaligned() as f32))
    }

    #[inline(always)]
    unsafe fn gather_u16(at: [*const u16; 4]) -> F4 {
        F4(at.map(|at| at.read_unaligned() as f32))
    }

    #[inline(always)]
    fn splat(value: f32) -> F4 {
        F4([value; 4])
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
    fn store_front(z: F4, w: F4, front: &mut [u16; 4]) {
        for (lane, flag) in front.iter_mut().enumerate() {
            *flag = if z.0[lane] >= -w.0[lane] && w.0[lane] > 0.0 { 0xFFFF } else { 0 };
        }
    }

    #[inline(always)]
    fn store(out: &mut [FrameVertex], fields: [F4; 8]) {
        for (n, vertex) in out.iter_mut().take(4).enumerate() {
            let [x, y, z, w, u, v, color, fog] = fields.map(|field| field.0[n]);
            *vertex = FrameVertex { x, y, z, w, u, v, color: color.to_bits(), fog };
        }
    }
}
