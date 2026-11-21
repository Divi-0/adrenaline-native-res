//! Draws a later draw paints over completely.
//!
//! Need for Speed Underground Rivals' title draws its trail composites,
//! a black quad and two dithering passes over the whole screen, and then an
//! opaque black full-screen layer over all of them (draw 7): nothing of the
//! earlier draws reaches the picture, yet they cost ~20 ms of GPU time per
//! frame (frame reads and blending keep the GPU's own hidden surface
//! removal from dropping them). PPSSPP draws them too; we leave them out.
//!
//! A draw *occludes* everything drawn into the shown framebuffer before it
//! when every pixel it writes is independent of what was there, and it
//! writes every pixel of the screen, colour and alpha (the PSP's stencil),
//! and depth where the earlier draws wrote depth:
//! - screen-space vertices (no transform, raw or cached vertices), no
//!   culling, two or more triangles covering the whole 480x272 screen
//!   (`covers_screen`), the scissor the whole screen;
//! - no blending, or a normal alpha blend of a source whose every fragment
//!   has alpha 255 (core opaque_source: the blend gives the source); the
//!   alpha written (`BLEND_WRITES_ALPHA`, not alpha only, no FIX B);
//! - no stencil test, logic op or colour test; an alpha test only where
//!   every fragment passes it (alpha 255);
//! - a depth test that surely passes everywhere: off, "always", or a draw
//!   at one depth against the depths the screen can hold then (`Depths`,
//!   tracked from the earlier draws' depth writes, unknown at the scene's
//!   start).
//! An earlier draw into the shown framebuffer is then hidden, unless it
//! writes depth and the occluder doesn't overwrite all depths regardless of
//! them (it writes depth, with the test off or "always": else whether it
//! passes depends on those depths; NFS UR's layer passes "<=" only over the
//! clear's depth 1.0), or it has a stencil test (the Vita's stencil isn't
//! the colour's). Draws into other framebuffers
//! (off-screen targets, drawn in scenes of their own) are never hidden.
//!
//! An earlier draw kept for its depth alone (the occluder's depth test reads
//! it) is drawn *depth-only* instead (`is_depth_only`: no colour, no frame
//! read), if it is sure to write the same depths that way: screen-space
//! vertices and no fragment it would discard (alpha test passed by every
//! fragment, no colour test). And one whose depths a later draw overwrites
//! before anything drawn reads them (a full-screen "always" depth write) is
//! hidden after all: NFS UR's title clears depth three times (draws 0, 4,
//! 6); only the last one is drawn, depth-only.
//!
//! The display skips hidden draws; the PC replay too, so the regression
//! pictures check that the picture stays the same.

use crate::blend::{BLEND_ALPHA_ONLY, BLEND_ENABLED, BLEND_FIX_B_ALPHA, BLEND_WRITES_ALPHA};
use crate::frame::{draw_flags, test_flags, Draw, Frame, FrameVertex, MAX_DRAWS};
use crate::opaque_source::samples_only_rows;

/// What becomes of a draw (`find`).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Fate {
    Drawn,
    Hidden,
    /// Drawn for its depth alone.
    DepthOnly,
}

/// Two bits per draw of a frame: hidden, depth-only (`find`).
pub struct HiddenDraws {
    hidden: [u64; MAX_DRAWS / 64],
    depth_only: [u64; MAX_DRAWS / 64],
    count: u32,
    depth_only_count: u32,
}

/// Sets bit `index` of `words` to `on`; returns how the count of set bits
/// changed (-1, 0, 1).
fn set_bit(words: &mut [u64], index: usize, on: bool) -> i32 {
    let Some(word) = words.get_mut(index / 64) else {
        return 0;
    };
    let bit = 1u64 << (index % 64);
    let was = *word & bit != 0;
    if on {
        *word |= bit;
    } else {
        *word &= !bit;
    }
    on as i32 - was as i32
}

fn bit(words: &[u64], index: usize) -> bool {
    words.get(index / 64).is_some_and(|word| word >> (index % 64) & 1 != 0)
}

impl HiddenDraws {
    pub const fn new() -> HiddenDraws {
        HiddenDraws { hidden: [0; MAX_DRAWS / 64], depth_only: [0; MAX_DRAWS / 64], count: 0, depth_only_count: 0 }
    }

    pub fn is_hidden(&self, draw_index: usize) -> bool {
        bit(&self.hidden, draw_index)
    }

    /// Draw `draw_index` is drawn for its depth alone: depth test and
    /// depth writes as given, no colour or alpha written, every fragment
    /// kept (module comment).
    pub fn is_depth_only(&self, draw_index: usize) -> bool {
        bit(&self.depth_only, draw_index)
    }

    /// Hidden draws in the frame.
    pub fn count(&self) -> u32 {
        self.count
    }

    /// Depth-only draws in the frame.
    pub fn depth_only_count(&self) -> u32 {
        self.depth_only_count
    }

    fn set(&mut self, draw_index: usize, fate: Fate) {
        self.count = self.count.wrapping_add_signed(set_bit(&mut self.hidden, draw_index, fate == Fate::Hidden));
        self.depth_only_count =
            self.depth_only_count.wrapping_add_signed(set_bit(&mut self.depth_only, draw_index, fate == Fate::DepthOnly));
    }

    fn clear(&mut self, draw_count: usize) {
        let words = draw_count.div_ceil(64).min(self.hidden.len());
        self.hidden[..words].fill(0);
        self.depth_only[..words].fill(0);
        self.count = 0;
        self.depth_only_count = 0;
    }
}

impl Default for HiddenDraws {
    fn default() -> Self {
        HiddenDraws::new()
    }
}

/// Corners looked at per draw: an occluder is a few triangles.
const MAX_CORNERS: usize = 24;

/// The depths the screen's pixels can hold (screen-space z, 0..1), as far
/// as known.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Depths {
    Unknown,
    /// Each pixel holds one of these.
    OneOf { values: [f32; 4], count: usize },
}

impl Depths {
    fn values(&self) -> Option<&[f32]> {
        match self {
            Depths::Unknown => None,
            Depths::OneOf { values, count } => Some(&values[..*count]),
        }
    }

    fn add(&mut self, z: f32) {
        if let Depths::OneOf { values, count } = self {
            if values[..*count].iter().any(|value| value.to_bits() == z.to_bits()) {
                return;
            }
            if *count < values.len() {
                values[*count] = z;
                *count += 1;
                return;
            }
        }
        *self = Depths::Unknown;
    }
}

/// The depth difference below which two depths may compare either way on
/// the GPU (24-bit depth buffer; ours are the PSP's 16-bit values / 65535).
const DEPTH_MARGIN: f32 = 1.0 / 65536.0;

/// True if depth `z` surely passes compare `function` (the GE's: NEVER,
/// ALWAYS, EQUAL, NOT EQUAL, LESS, LEQUAL, GREATER, GEQUAL) against
/// `stored`: equal values compare equal, others only when apart by more
/// than `DEPTH_MARGIN`.
fn surely_passes(function: u32, z: f32, stored: f32) -> bool {
    let same = z.to_bits() == stored.to_bits();
    match function & 7 {
        0 => false,
        1 => true,
        2 => same,
        3 => !same && (z - stored).abs() > DEPTH_MARGIN,
        4 => z < stored - DEPTH_MARGIN,
        5 => same || z < stored - DEPTH_MARGIN,
        6 => z > stored + DEPTH_MARGIN,
        _ => same || z > stored + DEPTH_MARGIN,
    }
}

/// True if the GE's alpha test of `tests` passes alpha 255.
fn alpha_test_passes_255(tests: u32) -> bool {
    let function = (tests >> test_flags::ALPHA_FUNCTION_SHIFT) & 0xF;
    let reference = (tests >> test_flags::ALPHA_REFERENCE_SHIFT) & 0xFF;
    let mask = (tests >> test_flags::ALPHA_MASK_SHIFT) & 0xFF;
    let (value, reference) = (0xFF & mask, reference & mask);
    match function & 7 {
        0 => false,
        1 => true,
        2 => value == reference,
        3 => value != reference,
        4 => value < reference,
        5 => value <= reference,
        6 => value > reference,
        _ => value >= reference,
    }
}

/// How far inside the screen's edge a covering draw's edge may lie, in PSP
/// pixels: our targets have two pixels per PSP pixel, so the first and last
/// pixel centres are 0.25 inside (NFS UR's layers start at 0.009).
const EDGE_SLACK: f32 = 0.125;

/// True if `corners` (three per triangle) cover the whole screen,
/// 0..480 x 0..272 PSP pixels, every pixel centre of our 960x544 targets
/// (`EDGE_SLACK`): every triangle that isn't degenerate has its
/// corners on the corners of the bounding box, which contains the screen,
/// and two of them split the box along one diagonal (the triangles that
/// leave out opposite box corners).
pub fn covers_screen(corners: &[FrameVertex]) -> bool {
    if corners.len() < 6 || corners.len() % 3 != 0 {
        return false;
    }
    let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for corner in corners {
        if !(corner.x.is_finite() && corner.y.is_finite()) {
            return false;
        }
        x0 = x0.min(corner.x);
        y0 = y0.min(corner.y);
        x1 = x1.max(corner.x);
        y1 = y1.max(corner.y);
    }
    if x0 > EDGE_SLACK || y0 > EDGE_SLACK || x1 < 480.0 - EDGE_SLACK || y1 < 272.0 - EDGE_SLACK {
        return false;
    }
    // Box corners 0 (x0, y0), 1 (x1, y0), 2 (x1, y1), 3 (x0, y1).
    let box_corner = |corner: &FrameVertex| match (corner.x == x0, corner.x == x1, corner.y == y0, corner.y == y1) {
        (true, _, true, _) => Some(0),
        (_, true, true, _) => Some(1),
        (_, true, _, true) => Some(2),
        (true, _, _, true) => Some(3),
        _ => None,
    };
    let mut left_out = 0u8;
    for triangle in corners.chunks_exact(3) {
        let area = (triangle[1].x - triangle[0].x) * (triangle[2].y - triangle[0].y)
            - (triangle[2].x - triangle[0].x) * (triangle[1].y - triangle[0].y);
        if area == 0.0 {
            continue;
        }
        let mut used = 0u8;
        for corner in triangle {
            let Some(at) = box_corner(corner) else {
                return false;
            };
            used |= 1 << at;
        }
        if used.count_ones() != 3 {
            return false;
        }
        left_out |= !used & 0xF;
    }
    left_out & 0b0101 == 0b0101 || left_out & 0b1010 == 0b1010
}

/// A draw's corners in screen space (three per triangle), if it has them
/// on the CPU and not more than `MAX_CORNERS`.
fn screen_space_corners(frame: &Frame, draw: &Draw, out: &mut [FrameVertex; MAX_CORNERS]) -> Option<usize> {
    if draw.flags & (draw_flags::RAW_VERTICES | draw_flags::CACHED_VERTICES) != 0 || draw.transform != 0 {
        return None;
    }
    if frame.draw_index_count(draw) as usize > MAX_CORNERS {
        return None;
    }
    let mut count = 0;
    for corner in frame.draw_corners(draw) {
        out[count] = *corner;
        count += 1;
    }
    Some(count)
}

/// The one depth all `corners` have, if they do.
fn single_depth(corners: &[FrameVertex]) -> Option<f32> {
    let z = corners.first()?.z;
    corners.iter().all(|corner| corner.z.to_bits() == z.to_bits()).then_some(z)
}

fn writes_depth(draw: &Draw) -> bool {
    draw.tests & test_flags::DEPTH_TEST != 0 && draw.tests & test_flags::DEPTH_WRITE_DISABLED == 0
}

/// True if `draw` tests depth with "always" (and so writes it, if it
/// writes, whatever was there).
fn depth_always(draw: &Draw) -> bool {
    draw.tests & test_flags::DEPTH_TEST != 0 && (draw.tests >> test_flags::DEPTH_FUNCTION_SHIFT) & 7 == 1
}

/// True if a draw writes every depth of the screen whatever was there:
/// depth "always" written, no fragment discarded (`keeps_every_fragment`),
/// the full scissor, no culling, `corners` covering the screen.
fn overwrites_every_depth(draw: &Draw, corners: Option<&[FrameVertex]>, keeps_every_fragment: bool) -> bool {
    writes_depth(draw)
        && depth_always(draw)
        && keeps_every_fragment
        && draw.flags & draw_flags::CULL == 0
        && draw.scissor_left == 0
        && draw.scissor_top == 0
        && draw.scissor_right >= 479
        && draw.scissor_bottom >= 271
        && corners.is_some_and(covers_screen)
}

/// If depth writer `draw` (into the shown framebuffer, no stencil test) can
/// be drawn depth-only with the same depths (module comment): whether it
/// then overwrites every depth whatever was there.
fn depth_only_way(frame: &Frame, draw: &Draw, texture_rows: &dyn Fn(usize) -> u16) -> Option<bool> {
    // Screen-space CPU vertices: the display draws their corners.
    if draw.flags & (draw_flags::RAW_VERTICES | draw_flags::CACHED_VERTICES) != 0 || draw.transform != 0 {
        return None;
    }
    if draw.tests & test_flags::COLOR_TEST != 0 {
        return None;
    }
    let mut corners = [FrameVertex::default(); MAX_CORNERS];
    let count = screen_space_corners(frame, draw, &mut corners);
    let corners = count.map(|count| &corners[..count]);
    let keeps_every_fragment = draw.tests & test_flags::ALPHA_TEST == 0
        || alpha_test_passes_always(draw.tests)
        || (alpha_test_passes_255(draw.tests) && corners.is_some_and(|corners| every_fragment_opaque(frame, draw, corners, texture_rows)));
    keeps_every_fragment.then(|| overwrites_every_depth(draw, corners, keeps_every_fragment))
}

/// True if a draw at one depth `z` (None: varying) surely passes its depth
/// test at every pixel, given `depths`.
fn depth_passes_everywhere(draw: &Draw, z: Option<f32>, depths: &Depths) -> bool {
    if draw.tests & test_flags::DEPTH_TEST == 0 {
        return true;
    }
    let function = (draw.tests >> test_flags::DEPTH_FUNCTION_SHIFT) & 7;
    if function == 1 {
        return true;
    }
    match (z, depths.values()) {
        (Some(z), Some(values)) => values.iter().all(|&stored| surely_passes(function, z, stored)),
        _ => false,
    }
}

/// True if every fragment of `draw` (screen-space `corners`) has alpha
/// 255: vertex alphas 255 and, textured, only opaque texel rows sampled
/// (`texture_rows`: `texture::opaque_rows` of a frame texture).
fn every_fragment_opaque(frame: &Frame, draw: &Draw, corners: &[FrameVertex], texture_rows: &dyn Fn(usize) -> u16) -> bool {
    if corners.is_empty() || !corners.iter().all(|corner| corner.color >> 24 == 0xFF) {
        return false;
    }
    if draw.flags & draw_flags::TEXTURED == 0 {
        return true;
    }
    let Some(texture) = frame.textures().get(draw.texture as usize) else {
        return false;
    };
    samples_only_rows(corners, draw.flags, texture.height, texture_rows(draw.texture as usize))
}

/// If `draw` occludes the screen (module comment) given `depths`: whether
/// it writes every depth whatever was there (earlier depth writes may then
/// be hidden too).
fn occluder(frame: &Frame, draw: &Draw, corners: &[FrameVertex], depths: &Depths, texture_rows: &dyn Fn(usize) -> u16) -> Option<bool> {
    let blended = draw.blend & BLEND_ENABLED != 0;
    let tested = draw.tests & test_flags::ALPHA_TEST != 0;
    if !covers_screen(corners) {
        return None;
    }
    if (blended || tested) && !every_fragment_opaque(frame, draw, corners, texture_rows) {
        return None;
    }
    let z = single_depth(corners);
    depth_passes_everywhere(draw, z, depths).then(|| depth_always(draw) && writes_depth(draw))
}

/// True if `draw`'s state lets it occlude the screen (`occluder` checks
/// its corners, opacity and depth test then): a few compares, so that most
/// draws cost nothing more.
fn may_occlude(frame: &Frame, draw: &Draw) -> bool {
    let blend = draw.blend;
    draw.flags & draw_flags::CULL == 0
        && draw.scissor_left == 0
        && draw.scissor_top == 0
        && draw.scissor_right >= 479
        && draw.scissor_bottom >= 271
        && draw.tests & (test_flags::STENCIL_TEST | test_flags::LOGIC_OP | test_flags::COLOR_TEST) == 0
        && blend & BLEND_WRITES_ALPHA != 0
        && blend & (BLEND_ALPHA_ONLY | BLEND_FIX_B_ALPHA) == 0
        // Only a normal alpha blend (of an opaque source) gives the source.
        && (blend & BLEND_ENABLED == 0 || (blend & 0xF, (blend >> 4) & 0xF, (blend >> 8) & 7) == (4, 5, 0))
        && (draw.tests & test_flags::ALPHA_TEST == 0 || alpha_test_passes_255(draw.tests))
        && frame.draw_index_count(draw) as usize <= MAX_CORNERS
}

/// True if `draw` (finished: its own vertices and index count, before any
/// merge) may occlude the screen: `may_occlude`'s state, screen-space
/// vertices around the whole screen, few corners in a plain list, and
/// vertex alphas 255 where the alpha counts. Noted in
/// `Frame::occluders_end` while recording; a draw that isn't is never an
/// occluder (one that is may still not occlude: `find` checks).
pub fn maybe_occluder(draw: &Draw, vertices: &[FrameVertex]) -> bool {
    let blend = draw.blend;
    let alpha_counts = blend & BLEND_ENABLED != 0 || draw.tests & test_flags::ALPHA_TEST != 0;
    draw.flags & (draw_flags::CULL | draw_flags::RAW_VERTICES | draw_flags::CACHED_VERTICES | draw_flags::INDEX_RUNS) == 0
        && draw.transform == 0
        && draw.index_count as usize <= MAX_CORNERS
        && draw.scissor_left == 0
        && draw.scissor_top == 0
        && draw.scissor_right >= 479
        && draw.scissor_bottom >= 271
        && draw.tests & (test_flags::STENCIL_TEST | test_flags::LOGIC_OP | test_flags::COLOR_TEST) == 0
        && blend & BLEND_WRITES_ALPHA != 0
        && blend & (BLEND_ALPHA_ONLY | BLEND_FIX_B_ALPHA) == 0
        && (blend & BLEND_ENABLED == 0 || (blend & 0xF, (blend >> 4) & 0xF, (blend >> 8) & 7) == (4, 5, 0))
        && (!alpha_counts || vertices.iter().all(|vertex| vertex.color >> 24 == 0xFF))
        && around_screen(vertices)
}

/// True if the bounding box of `vertices` contains every pixel centre of
/// the screen (`EDGE_SLACK`).
fn around_screen(vertices: &[FrameVertex]) -> bool {
    let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for vertex in vertices {
        x0 = x0.min(vertex.x);
        y0 = y0.min(vertex.y);
        x1 = x1.max(vertex.x);
        y1 = y1.max(vertex.y);
    }
    x0 <= EDGE_SLACK && y0 <= EDGE_SLACK && x1 >= 480.0 - EDGE_SLACK && y1 >= 272.0 - EDGE_SLACK
}

/// Marks in `out` the draws of `frame` that a later draw paints over
/// completely, hidden or depth-only (module comment). `texture_rows(t)`: how many rows from the
/// top of frame texture `t` have only opaque texels (`texture::opaque_rows`,
/// `ALL_ROWS_OPAQUE`: all; 0 when not known).
pub fn find(frame: &Frame, texture_rows: &dyn Fn(usize) -> u16, out: &mut HiddenDraws) {
    let draws = frame.draws();
    out.clear(draws.len());
    // No draw after this one could be an occluder (`maybe_occluder`).
    let end = (frame.occluders_end as usize).min(draws.len());
    let shown = frame.shown_framebuffer;
    let mut depths = Depths::Unknown;
    let mut corners = [FrameVertex::default(); MAX_CORNERS];
    // Draws before this one are settled: hidden, or kept for good (by an
    // occluder that overwrote every depth).
    let mut settled = 0;
    for (index, draw) in draws.iter().enumerate().take(end) {
        if draw.framebuffer != shown {
            continue;
        }
        // Raw, cached or model-space vertices (most 3D draws): never an
        // occluder, their depths unknown here.
        let screen_space = draw.flags & (draw_flags::RAW_VERTICES | draw_flags::CACHED_VERTICES) == 0 && draw.transform == 0;
        if !screen_space {
            if writes_depth(draw) {
                depths = Depths::Unknown;
            }
            continue;
        }
        let candidate = may_occlude(frame, draw);
        if !candidate && !writes_depth(draw) {
            continue; // most 2D draws: nothing to work out
        }
        let count = screen_space_corners(frame, draw, &mut corners);
        let corners = &corners[..count.unwrap_or(0)];
        if candidate && count.is_some() {
            if let Some(overwrites_depth) = occluder(frame, draw, corners, &depths, texture_rows) {
                settle_before(frame, settled, index, overwrites_depth, texture_rows, out);
                if overwrites_depth {
                    settled = index;
                }
            }
        }
        // The depths the screen can hold after this draw.
        if writes_depth(draw) {
            let Some(z) = count.and_then(|_| single_depth(corners)) else {
                depths = Depths::Unknown;
                continue;
            };
            let discards = draw.tests & test_flags::ALPHA_TEST != 0
                && !(alpha_test_passes_always(draw.tests)
                    || (alpha_test_passes_255(draw.tests) && every_fragment_opaque(frame, draw, corners, texture_rows)));
            if !discards && covers_screen(corners) && depth_passes_everywhere(draw, Some(z), &depths) {
                depths = Depths::OneOf { values: [z, 0.0, 0.0, 0.0], count: 1 };
            } else {
                depths.add(z);
            }
        }
    }
}

/// The fate of the draws `settled..occluder`, painted over by draw
/// `occluder` (it writes every depth whatever was there:
/// `overwrites_depth`). Each occluder decides that whole range anew,
/// from the occluder back, tracking whether the depths right after each
/// draw are still read by a later draw that is drawn (`depth_read`). The
/// colour is the occluder's everywhere; a depth writer is hidden where its
/// depths aren't read, else drawn depth-only (in full where depth-only
/// could change its depths). Draws into other framebuffers and
/// stencil-tested ones are drawn.
fn settle_before(frame: &Frame, settled: usize, occluder: usize, overwrites_depth: bool, texture_rows: &dyn Fn(usize) -> u16, out: &mut HiddenDraws) {
    let draws = frame.draws();
    let shown = frame.shown_framebuffer;
    let mut depth_read = !overwrites_depth;
    for index in (settled..occluder).rev() {
        let draw = &draws[index];
        let fate = if draw.framebuffer != shown {
            Fate::Drawn
        } else if draw.tests & test_flags::STENCIL_TEST != 0 {
            // Drawn in full: what it does to depth depends on the stencil.
            depth_read |= draw.tests & test_flags::DEPTH_TEST != 0;
            Fate::Drawn
        } else if !writes_depth(draw) || !depth_read {
            Fate::Hidden
        } else {
            match depth_only_way(frame, draw, texture_rows) {
                Some(overwrites_every_depth) => {
                    // Earlier depths matter only through its depth test.
                    depth_read = !overwrites_every_depth;
                    Fate::DepthOnly
                }
                None => Fate::Drawn, // reads and writes depth: depth_read stays
            }
        };
        out.set(index, fate);
    }
}

/// True if the alpha test of `tests` passes every alpha ("always").
fn alpha_test_passes_always(tests: u32) -> bool {
    (tests >> test_flags::ALPHA_FUNCTION_SHIFT) & 7 == 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::texture::ALL_ROWS_OPAQUE;

    fn corner(x: f32, y: f32, z: f32, color: u32) -> FrameVertex {
        FrameVertex { x, y, z, w: 1.0, color, ..Default::default() }
    }

    /// Two triangles over (x0, y0)..(x1, y1) along the diagonal from the
    /// top right corner (NFS UR's layers).
    fn quad(x0: f32, y0: f32, x1: f32, y1: f32, z: f32, color: u32) -> [FrameVertex; 6] {
        let c = |x, y| corner(x, y, z, color);
        [c(x0, y0), c(x0, y1), c(x1, y0), c(x1, y0), c(x0, y1), c(x1, y1)]
    }

    #[test]
    fn screen_cover() {
        assert!(covers_screen(&quad(0.0, 0.0, 480.0, 272.0, 0.5, 0)));
        assert!(covers_screen(&quad(-10.0, -1.0, 481.0, 273.0, 0.5, 0)));
        assert!(!covers_screen(&quad(0.0, 0.0, 479.0, 272.0, 0.5, 0)));
        assert!(!covers_screen(&quad(1.0, 0.0, 480.0, 272.0, 0.5, 0)));
        // NFS UR's black layer: a hair inside the edges, still every centre.
        assert!(covers_screen(&quad(0.00881958, 0.0071411133, 480.98636, 272.99414, 0.5, 0)));
        assert!(!covers_screen(&quad(0.2, 0.0, 480.0, 272.0, 0.5, 0)));
        assert!(!covers_screen(&quad(0.0, 0.0, 480.0, 271.8, 0.5, 0)));
        // The same triangle twice: half the box.
        let q = quad(0.0, 0.0, 480.0, 272.0, 0.5, 0);
        assert!(!covers_screen(&[q[0], q[1], q[2], q[0], q[1], q[2]]));
        // Two triangles sharing a side (not the diagonal): three quarters.
        let c = |x, y| corner(x, y, 0.5, 0);
        assert!(!covers_screen(&[c(0.0, 0.0), c(480.0, 0.0), c(480.0, 272.0), c(0.0, 0.0), c(480.0, 0.0), c(0.0, 272.0)]));
        // Degenerate triangles are left out; an inner corner fails.
        let degenerate = [c(0.0, 0.0), c(0.0, 0.0), c(0.0, 272.0)];
        assert!(covers_screen(&[&degenerate[..], &q[..]].concat()));
        assert!(!covers_screen(&[&q[..], &[c(0.0, 0.0), c(240.0, 0.0), c(0.0, 272.0)][..]].concat()));
        assert!(!covers_screen(&q[..3]));
    }

    #[test]
    fn depth_compares_surely() {
        assert!(surely_passes(5, 0.5, 1.0)); // LEQUAL
        assert!(surely_passes(5, 0.5, 0.5));
        assert!(!surely_passes(5, 0.5, 0.5 + DEPTH_MARGIN / 2.0));
        assert!(!surely_passes(4, 0.5, 0.5)); // LESS
        assert!(surely_passes(7, 0.5, 0.25)); // GEQUAL
        assert!(!surely_passes(7, 0.0526, 0.5));
        assert!(surely_passes(1, 0.0, 1.0));
        assert!(!surely_passes(0, 0.0, 1.0));
        let mut depths = Depths::OneOf { values: [1.0, 0.0, 0.0, 0.0], count: 1 };
        depths.add(1.0);
        assert_eq!(depths.values(), Some(&[1.0][..]));
        for z in [0.1, 0.2, 0.3] {
            depths.add(z);
        }
        assert_eq!(depths.values().map(<[f32]>::len), Some(4));
        depths.add(0.4);
        assert_eq!(depths, Depths::Unknown);
    }

    #[test]
    fn alpha_tests() {
        // NFS UR's layers: GREATER 0x10, mask 0xFF.
        assert!(alpha_test_passes_255(0xff10_6503));
        // LESS 0x80: 255 fails.
        assert!(!alpha_test_passes_255(0xff80_4001));
        // Mask 0x7F: 255 & 0x7F = 0x7F, GEQUAL 0x7F passes.
        assert!(alpha_test_passes_255(0x7f7f_7001));
    }

    fn push(frame: &mut Frame, draw: Draw, corners: &[FrameVertex]) {
        let (first_vertex, first_index) = (frame.vertices().len() as u32, frame.indices().len() as u32);
        for corner in corners {
            frame.add_corner(first_vertex, *corner).unwrap();
        }
        frame.finish_draw(Draw { first_vertex, first_index, ..draw });
    }

    fn texture(frame: &mut Frame, height: u16) -> u32 {
        let address = 0x0900_0000 + frame.textures().len() as u32 * 0x1000;
        frame.add_texture(crate::frame::TextureDescriptor { address, width: 16, height, buffer_width: 16, ..Default::default() }).unwrap()
    }

    /// A frame in NFS UR's title order: a clear writing depth 1.0, a
    /// blended quad, an off-screen draw, then the opaque layer at depth 0.5
    /// with "<=", then a logo.
    #[test]
    fn nfs_title_layers() {
        use crate::frame::new_boxed_frame as new_test_frame;
        let shown = 0x0400_0000;
        let mut frame = new_test_frame();
        frame.shown_framebuffer = shown;
        let full = |tests: u32, blend: u32, flags: u32, texture: u32, framebuffer: u32| Draw {
            flags,
            tests,
            blend,
            texture,
            framebuffer,
            scissor_right: 479,
            scissor_bottom: 271,
            ..Default::default()
        };
        let depth_always_write = test_flags::DEPTH_TEST | 1 << test_flags::DEPTH_FUNCTION_SHIFT;
        let normal = BLEND_ENABLED | BLEND_WRITES_ALPHA | 4 | 5 << 4;
        let layer_tests = 0xff10_6503; // alpha GREATER 0x10, depth LEQUAL, writes depth
        push(&mut frame, full(depth_always_write, BLEND_WRITES_ALPHA, 0, 0, shown), &quad(0.0, 0.0, 480.0, 272.0, 1.0, 0xFF00_0000));
        push(&mut frame, full(0, normal, draw_flags::BLEND, 0, shown), &quad(0.0, 0.0, 480.0, 272.0, 0.0, 0x80FF_FFFF));
        push(&mut frame, full(0, normal, draw_flags::BLEND, 0, 0x040c_c000), &quad(0.0, 0.0, 480.0, 272.0, 0.0, 0x80FF_FFFF));
        let textured = draw_flags::TEXTURED | draw_flags::BLEND | draw_flags::CLAMP_U | draw_flags::CLAMP_V;
        let (black, logo) = (texture(&mut frame, 16), texture(&mut frame, 16));
        push(&mut frame, full(layer_tests, normal, textured, black, shown), &quad(0.0, 0.0, 481.0, 273.0, 0.5, 0xFFFF_FFFF));
        push(&mut frame, full(layer_tests, normal, textured, logo, shown), &quad(15.0, 3.0, 200.0, 150.0, 0.5, 0x8FFF_FFFF));
        let mut hidden = HiddenDraws::new();
        find(&frame, &|_| ALL_ROWS_OPAQUE, &mut hidden);
        // The clear stays for its depth: the layer's "<=" passes only over
        // its depth. Depth-only: no colour.
        assert_eq!((0..5).map(|i| hidden.is_hidden(i)).collect::<Vec<_>>(), [false, true, false, false, false]);
        assert_eq!((0..5).map(|i| hidden.is_depth_only(i)).collect::<Vec<_>>(), [true, false, false, false, false]);
        assert_eq!((hidden.count(), hidden.depth_only_count()), (1, 1));
        // A texture with transparent texels: no occluder.
        find(&frame, &|_| 0, &mut hidden);
        assert_eq!((hidden.count(), hidden.depth_only_count()), (0, 0));
        // The clear left out: the screen's depth is unknown, "<=" may fail.
        let mut no_clear = new_test_frame();
        no_clear.shown_framebuffer = shown;
        push(&mut no_clear, full(0, normal, draw_flags::BLEND, 0, shown), &quad(0.0, 0.0, 480.0, 272.0, 0.0, 0x80FF_FFFF));
        let black = texture(&mut no_clear, 16);
        push(&mut no_clear, full(layer_tests, normal, textured, black, shown), &quad(0.0, 0.0, 481.0, 273.0, 0.5, 0xFFFF_FFFF));
        find(&no_clear, &|_| ALL_ROWS_OPAQUE, &mut hidden);
        assert_eq!(hidden.count(), 0);
        // An occluder without depth writes keeps earlier depth writers.
        let mut keeps = new_test_frame();
        keeps.shown_framebuffer = shown;
        push(&mut keeps, full(depth_always_write, BLEND_WRITES_ALPHA, 0, 0, shown), &quad(0.0, 0.0, 480.0, 272.0, 1.0, 0xFF00_0000));
        push(&mut keeps, full(0, normal, draw_flags::BLEND, 0, shown), &quad(0.0, 0.0, 480.0, 272.0, 0.0, 0x80FF_FFFF));
        push(&mut keeps, full(0, BLEND_WRITES_ALPHA, 0, 0, shown), &quad(0.0, 0.0, 480.0, 272.0, 0.0, 0x00FF_FFFF));
        find(&keeps, &|_| ALL_ROWS_OPAQUE, &mut hidden);
        assert_eq!((0..3).map(|i| hidden.is_hidden(i)).collect::<Vec<_>>(), [false, true, false]);
        assert_eq!((0..3).map(|i| hidden.is_depth_only(i)).collect::<Vec<_>>(), [true, false, false]);
        // A second clear ("always", writing depth) overwrites every depth:
        // the first clear and everything before it are hidden.
        push(&mut keeps, full(depth_always_write, BLEND_WRITES_ALPHA, 0, 0, shown), &quad(0.0, 0.0, 480.0, 272.0, 0.25, 0xFF10_2030));
        find(&keeps, &|_| ALL_ROWS_OPAQUE, &mut hidden);
        assert_eq!((0..4).map(|i| hidden.is_hidden(i)).collect::<Vec<_>>(), [true, true, true, false]);
        assert_eq!(hidden.depth_only_count(), 0);
        // An occluder keeping the alpha (the PSP's stencil): nothing hidden.
        let mut alpha_kept = new_test_frame();
        alpha_kept.shown_framebuffer = shown;
        push(&mut alpha_kept, full(0, normal, draw_flags::BLEND, 0, shown), &quad(0.0, 0.0, 480.0, 272.0, 0.0, 0x80FF_FFFF));
        push(&mut alpha_kept, full(0, 0, 0, 0, shown), &quad(0.0, 0.0, 480.0, 272.0, 0.0, 0xFFFF_FFFF));
        find(&alpha_kept, &|_| ALL_ROWS_OPAQUE, &mut hidden);
        assert_eq!(hidden.count(), 0);
    }

    /// The fates of `frame`'s draws: '.' drawn, 'h' hidden, 'd' depth-only.
    fn fates(frame: &Frame) -> String {
        let mut hidden = HiddenDraws::new();
        find(frame, &|_| ALL_ROWS_OPAQUE, &mut hidden);
        let fates: String = (0..frame.draws().len())
            .map(|i| match (hidden.is_hidden(i), hidden.is_depth_only(i)) {
                (true, false) => 'h',
                (false, true) => 'd',
                (false, false) => '.',
                _ => '?',
            })
            .collect();
        assert_eq!(hidden.count() as usize, fates.matches('h').count());
        assert_eq!(hidden.depth_only_count() as usize, fates.matches('d').count());
        fates
    }

    /// NFS UR's title: three full-screen depth clears ("always", 1.0)
    /// between blended layers, then two opaque layers with "<=": only the
    /// last clear's depths are read (depth-only), the first two are
    /// overwritten before (hidden); the first layer is painted over by the
    /// second but its depths are read (depth-only).
    #[test]
    fn depth_clears() {
        use crate::frame::new_boxed_frame as new_test_frame;
        let shown = 0x0400_0000;
        let full = |tests: u32, blend: u32, flags: u32, texture: u32| Draw {
            flags,
            tests,
            blend,
            texture,
            framebuffer: shown,
            scissor_right: 479,
            scissor_bottom: 271,
            ..Default::default()
        };
        let clear = test_flags::DEPTH_TEST | 1 << test_flags::DEPTH_FUNCTION_SHIFT;
        let normal = BLEND_ENABLED | BLEND_WRITES_ALPHA | 4 | 5 << 4;
        let zero_one = BLEND_ENABLED | BLEND_WRITES_ALPHA | 1 << 4; // the target kept
        let layer_tests = 0xff10_6503;
        let textured = draw_flags::TEXTURED | draw_flags::BLEND | draw_flags::CLAMP_U | draw_flags::CLAMP_V;
        let screen = |z: f32, color: u32| quad(0.0, 0.0, 480.0, 272.0, z, color);
        let mut frame = new_test_frame();
        frame.shown_framebuffer = shown;
        // Different textures: draws of the same state in a row are merged.
        let (layer, second, logo) = (texture(&mut frame, 16), texture(&mut frame, 16), texture(&mut frame, 16));
        push(&mut frame, full(clear, BLEND_WRITES_ALPHA, 0, 0), &screen(1.0, 0xFF00_0000));
        push(&mut frame, full(0, normal, draw_flags::BLEND, 0), &screen(0.0, 0x80FF_FFFF));
        push(&mut frame, full(clear, zero_one, draw_flags::BLEND, 0), &screen(1.0, 0));
        push(&mut frame, full(0, normal, draw_flags::BLEND, 0), &screen(0.15, 0xFF00_0000));
        push(&mut frame, full(clear, zero_one, draw_flags::BLEND, 0), &screen(1.0, 0));
        push(&mut frame, full(layer_tests, normal, textured, layer), &quad(0.0, 0.0, 481.0, 273.0, 0.5, 0xFFFF_FFFF));
        push(&mut frame, full(layer_tests, normal, textured, second), &quad(0.0, 0.0, 481.0, 273.0, 0.5, 0xFFFF_FFFF));
        push(&mut frame, full(layer_tests, normal, textured, logo), &quad(15.0, 3.0, 200.0, 150.0, 0.5, 0x8FFF_FFFF));
        assert_eq!(frame.draws().len(), 8);
        assert_eq!(fates(&frame), "hhhhdd..");

        // A stencil-tested draw reading depth between the clears: the first
        // clear's depths are read, so it is drawn depth-only too.
        let mut stencil = new_test_frame();
        stencil.shown_framebuffer = shown;
        let layer = texture(&mut stencil, 16);
        let stencil_tests = test_flags::STENCIL_TEST | test_flags::DEPTH_TEST | 5 << test_flags::DEPTH_FUNCTION_SHIFT;
        push(&mut stencil, full(clear, BLEND_WRITES_ALPHA, 0, 0), &screen(1.0, 0xFF00_0000));
        push(&mut stencil, full(stencil_tests, normal, draw_flags::BLEND, 0), &screen(0.2, 0x80FF_FFFF));
        push(&mut stencil, full(clear, zero_one, draw_flags::BLEND, 0), &screen(1.0, 0));
        push(&mut stencil, full(layer_tests, normal, textured, layer), &quad(0.0, 0.0, 481.0, 273.0, 0.5, 0xFFFF_FFFF));
        assert_eq!(fates(&stencil), "d.d.");

        // A depth writer that may discard (alpha test, alpha 0x80 below
        // 0xFF): depth-only would write more depths, so it is drawn.
        let mut discards = new_test_frame();
        discards.shown_framebuffer = shown;
        let layer = texture(&mut discards, 16);
        let tested_clear = clear | test_flags::ALPHA_TEST | 0xFF << test_flags::ALPHA_MASK_SHIFT | 0xFE << test_flags::ALPHA_REFERENCE_SHIFT | 6 << test_flags::ALPHA_FUNCTION_SHIFT;
        push(&mut discards, full(clear, BLEND_WRITES_ALPHA, 0, 0), &screen(1.0, 0xFF00_0000));
        push(&mut discards, full(tested_clear, normal, draw_flags::BLEND, 0), &screen(1.0, 0x80FF_FFFF));
        push(&mut discards, full(layer_tests, normal, textured, layer), &quad(0.0, 0.0, 481.0, 273.0, 0.5, 0xFFFF_FFFF));
        assert_eq!(fates(&discards), "d..");
        // The same with alpha 255: every fragment passes, its depths overwrite
        // the clear's.
        let mut passes = new_test_frame();
        passes.shown_framebuffer = shown;
        let layer = texture(&mut passes, 16);
        push(&mut passes, full(clear, BLEND_WRITES_ALPHA, 0, 0), &screen(1.0, 0xFF00_0000));
        push(&mut passes, full(tested_clear, normal, draw_flags::BLEND, 0), &screen(1.0, 0xFFFF_FFFF));
        push(&mut passes, full(layer_tests, normal, textured, layer), &quad(0.0, 0.0, 481.0, 273.0, 0.5, 0xFFFF_FFFF));
        assert_eq!(fates(&passes), "hd.");

        // A partial depth write after the full clear: both are read.
        let mut partial = new_test_frame();
        partial.shown_framebuffer = shown;
        let layer = texture(&mut partial, 16);
        push(&mut partial, full(clear, BLEND_WRITES_ALPHA, 0, 0), &screen(1.0, 0xFF00_0000));
        push(&mut partial, full(clear, zero_one, draw_flags::BLEND, 0), &quad(10.0, 10.0, 100.0, 100.0, 0.75, 0));
        push(&mut partial, full(layer_tests, normal, textured, layer), &quad(0.0, 0.0, 481.0, 273.0, 0.5, 0xFFFF_FFFF));
        assert_eq!(fates(&partial), "dd.");
    }
}
