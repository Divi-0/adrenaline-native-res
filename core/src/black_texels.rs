//! Texels known to be black, and the triangles that therefore change nothing.
//!
//! God of War's shadows (2026-11-24, sixty-seventh session): each frame the
//! game clears a strip of seven 64x64 tiles to black, draws a grey blob per
//! character into its tile, and then lays every tile over the ground with a
//! big projected mesh (~200 triangles, ~1/3 of the screen at 2x) blended as
//! "target x (1 - texel)". Where the mesh samples only black texels the
//! target stays as it is, yet the GPU shaded all of it: the fourteen
//! passes cost ~43 ms of a 255 ms gameplay frame. The blobs cover a few
//! texels of their tiles, and the meshes' texture coordinates run hundreds
//! of texels past them, so most triangles sample nothing but black.
//!
//! This module answers, exactly, which triangles those are:
//! - `TargetContents` follows what the display draws into one off-screen
//!   target during a frame: the part a black fill covered (`black_fill`),
//!   and the parts later draws may have changed (`paint_area`, their boxes
//!   cut by the scissor). `lit_box` then gives, for the part of the target a
//!   texture reads, the texels that may not be black. A draw that copies
//!   another off-screen target paints only where that one's non-black
//!   texels land (`Source`): the game blurs the whole strip through a
//!   half-size target and back, which else left every tile painted all
//!   over (2026-11-26, seventy-seventh session).
//! - `zero_source_keeps_target`: a draw for which a fragment of texel colour
//!   0 changes nothing (source colour 0 with an additive blend whose
//!   destination factor is one or one minus the source colour; no depth,
//!   alpha or stencil written, no fog, no dither).
//! - `keeps_triangle` / `cull_triangles`: triangles whose texture
//!   coordinates stay a margin away from the non-black texels on one side,
//!   with clamped coordinates, can only sample black: dropped.
//! - `lit_screen_box`: the box of pixels outside of which a draw's
//!   triangles sample only black: where a copy of a target can paint
//!   (`Source`). (As a scissor for the shadow passes themselves it gave
//!   the GPU 1.1 ms and cost the display thread 2.7: not used so.)
//!
//! Everything is in PSP pixels of the off-screen target (our off-screen
//! targets are 1:1) and in texels of the texture, which reads a part of the
//! target 1:1 (the Vita's cut views and wide copies, frame_loop.rs
//! `screen_view`). The pictures stay bit for bit the same: only triangles
//! whose every fragment keeps the target's colour are left out.

use crate::blend::{Equation, Factor, BLEND_ALPHA_ONLY, BLEND_ENABLED, BLEND_FIX_B_ALPHA, BLEND_WRITES_ALPHA};
use crate::frame::{draw_flags, test_flags, Draw, Frame, FrameVertex};
use crate::ge_math::floor;

/// The smallest whole number >= x (finite x within +-2^23).
fn ceil(x: f32) -> f32 {
    -floor(-x)
}

/// A rectangle of pixels or texels, `left..right` x `top..bottom` (the
/// right and bottom edges not included).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Rect {
    pub left: i32,
    pub top: i32,
    pub right: i32,
    pub bottom: i32,
}

impl Rect {
    pub const EMPTY: Rect = Rect { left: 0, top: 0, right: 0, bottom: 0 };

    pub const fn new(left: i32, top: i32, right: i32, bottom: i32) -> Rect {
        Rect { left, top, right, bottom }
    }

    pub fn is_empty(&self) -> bool {
        self.left >= self.right || self.top >= self.bottom
    }

    /// The pixels in both (`EMPTY` if none).
    pub fn intersect(self, other: Rect) -> Rect {
        let rect = Rect::new(
            self.left.max(other.left),
            self.top.max(other.top),
            self.right.min(other.right),
            self.bottom.min(other.bottom),
        );
        if rect.is_empty() {
            Rect::EMPTY
        } else {
            rect
        }
    }

    /// The smallest rectangle holding both.
    pub fn union(self, other: Rect) -> Rect {
        if self.is_empty() {
            return other;
        }
        if other.is_empty() {
            return self;
        }
        Rect::new(
            self.left.min(other.left),
            self.top.min(other.top),
            self.right.max(other.right),
            self.bottom.max(other.bottom),
        )
    }

    /// True if every pixel of `other` is in this one (an empty one always is).
    pub fn contains(&self, other: Rect) -> bool {
        other.is_empty()
            || (self.left <= other.left && self.top <= other.top && self.right >= other.right && self.bottom >= other.bottom)
    }

    pub fn offset(self, dx: i32, dy: i32) -> Rect {
        if self.is_empty() {
            return self;
        }
        Rect::new(self.left + dx, self.top + dy, self.right + dx, self.bottom + dy)
    }

    pub fn area(&self) -> i64 {
        if self.is_empty() {
            0
        } else {
            (self.right - self.left) as i64 * (self.bottom - self.top) as i64
        }
    }
}

/// How many painted parts `TargetContents` keeps apart; more are merged.
/// (God of War: nine blob draws into seven tiles a frame.)
pub const PAINTED_PARTS: usize = 16;

/// What the draws of the frame so far left in one off-screen target, in its
/// PSP pixels: `black` was filled with colour 0 (RGB), after which only
/// the `painted` parts may have changed. Starts knowing nothing each frame
/// (`UNKNOWN`), and must be told about every draw into the target, in the
/// order the GPU draws them (`note_draw`), and about anything else that
/// changes it (a clear: `forget`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TargetContents {
    black: Rect,
    painted: [Rect; PAINTED_PARTS],
    painted_count: u8,
}

impl Default for TargetContents {
    fn default() -> TargetContents {
        TargetContents::UNKNOWN
    }
}

impl TargetContents {
    pub const UNKNOWN: TargetContents = TargetContents { black: Rect::EMPTY, painted: [Rect::EMPTY; PAINTED_PARTS], painted_count: 0 };

    /// Nothing known any more (a new frame, the target cleared or seeded).
    pub fn forget(&mut self) {
        *self = TargetContents::UNKNOWN;
    }

    /// `draw` was drawn into this target, placed at x `origin_x` (a tile
    /// of a wider framebuffer, core `offscreen::placement`). `source`: what
    /// its texture reads, if that is another off-screen target.
    pub fn note_draw(&mut self, frame: &Frame, draw: &Draw, origin_x: i32, source: Option<Source>) {
        if let Some(fill) = black_fill(frame, draw) {
            self.note_fill(fill.offset(origin_x, 0));
        } else if !self.black.is_empty() {
            // (Without a black part, what a draw paints doesn't matter: a
            // later fill covers it or leaves it outside. That spares the
            // pass over the vertices of God of War's big bloom draws.)
            let area = paint_area(frame, draw);
            let mut lit = [Rect::EMPTY; LIT_AREAS];
            match source.and_then(|source| source.lit_areas(frame, draw, area, &mut lit)) {
                // Black texels of the source leave black pixels black:
                // only where its other texels land can something change.
                Some(count) => lit[..count].iter().for_each(|part| self.note_paint(part.offset(origin_x, 0))),
                None => self.note_paint(area.offset(origin_x, 0)),
            }
        }
    }

    /// `fill` is black now (RGB 0).
    pub fn note_fill(&mut self, fill: Rect) {
        if fill.is_empty() {
            return;
        }
        // The bigger black part wins; what is left of the other one is
        // forgotten (only ever less known: never wrong).
        if fill.contains(self.black) || fill.area() > self.black.area() {
            self.black = fill;
        }
        // Painted parts under the fill are black again.
        let count = self.painted_count as usize;
        let mut kept = 0;
        for k in 0..count {
            let part = self.painted[k];
            if !fill.contains(part) {
                self.painted[kept] = part;
                kept += 1;
            }
        }
        self.painted_count = kept as u8;
    }

    /// `area` may have changed.
    pub fn note_paint(&mut self, area: Rect) {
        if self.black.intersect(area).is_empty() {
            return; // nothing of the black part changed
        }
        let count = self.painted_count as usize;
        if self.painted[..count].iter().any(|part| part.contains(area)) {
            return; // known already
        }
        // The parts inside it are part of it now (a blurred blob over the blob).
        let mut count = 0;
        for k in 0..self.painted_count as usize {
            let part = self.painted[k];
            if !area.contains(part) {
                self.painted[count] = part;
                count += 1;
            }
        }
        self.painted_count = count as u8;
        if count < PAINTED_PARTS {
            self.painted[count] = area;
            self.painted_count += 1;
            return;
        }
        // Full: grown into the part that grows least (a bigger part only
        // means fewer texels known black).
        let growth = |part: &Rect| part.union(area).area() - part.area();
        if let Some(part) = self.painted.iter_mut().min_by_key(|part| growth(part)) {
            *part = part.union(area);
        }
    }

    /// For a texture reading `part` of this target 1:1: the texels (from
    /// the part's top left) that may not be black, `EMPTY` if all are; None
    /// if not all of the part is known (no black fill under it).
    pub fn lit_box(&self, part: Rect) -> Option<Rect> {
        if part.is_empty() || !self.black.contains(part) {
            return None;
        }
        let lit = self.painted[..self.painted_count as usize]
            .iter()
            .fold(Rect::EMPTY, |lit, painted| lit.union(painted.intersect(part)));
        Some(lit.offset(-part.left, -part.top))
    }
}

/// The most parts `Source::lit_areas` gives: one per painted part of the
/// source and one per side of its black part.
pub const LIT_AREAS: usize = PAINTED_PARTS + 4;

/// What the texture of a draw into an off-screen target reads, when that
/// is another off-screen target (`TargetContents::note_draw`): God of War
/// blurs its strip of shadow tiles by drawing it into a half-size target
/// and back.
#[derive(Clone, Copy, Debug)]
pub struct Source<'a> {
    /// What that target holds when the draw is drawn.
    pub contents: &'a TargetContents,
    /// The target's x of the texture's first texel (a tile's place in its
    /// strip; else 0). From there texels and the target's PSP pixels are
    /// the same, whatever size the texture claims: the Vita reads the
    /// target itself, up to its right and bottom ends.
    pub x: i32,
    /// The target's size in PSP pixels.
    pub width: i32,
    pub height: i32,
}

impl Source<'_> {
    /// The parts of `area` (what `draw` may paint, `paint_area`) where
    /// `draw` can turn a black pixel into another colour, written to `out`:
    /// their count. A fragment whose texels are all black is black itself
    /// (`black_texel_gives_black`), and black onto black stays black with
    /// every blend, so only the fragments that can sample a texel not known
    /// black count: for each such part of the source the box
    /// `BlackOutside::lit_screen_box` gives. None if that can't be told (the
    /// caller then takes all of `area`).
    fn lit_areas(&self, frame: &Frame, draw: &Draw, area: Rect, out: &mut [Rect; LIT_AREAS]) -> Option<usize> {
        if !black_texel_gives_black(draw) {
            return None;
        }
        let vertices = own_screen_vertices(frame, draw)?;
        let first = draw.first_index as usize;
        let indices = frame.indices.get(first..first.checked_add(draw.index_count as usize)?).filter(|_| draw.flags & draw_flags::INDEX_RUNS == 0)?;
        let texture = frame.textures().get(draw.texture as usize)?;
        // What the texture can reach of the target, and its black part.
        let visible = Rect::new(self.x, 0, self.width, self.height);
        let black = self.contents.black.intersect(visible);
        if black.is_empty() {
            return None;
        }
        // Not known black: the painted parts, and what lies around the
        // black part (left, right, above, below).
        let around = [
            Rect::new(visible.left, visible.top, black.left, visible.bottom),
            Rect::new(black.right, visible.top, visible.right, visible.bottom),
            Rect::new(black.left, visible.top, black.right, black.top),
            Rect::new(black.left, black.bottom, black.right, visible.bottom),
        ];
        let painted = &self.contents.painted[..self.contents.painted_count as usize];
        let mut count = 0;
        for part in painted.iter().chain(&around) {
            let part = part.intersect(visible);
            if part.is_empty() {
                continue;
            }
            // In texels. A part that ends where the target ends has no
            // black texels beyond it for a clamped coordinate to land on:
            // it counts up to the texture's own end, which `BlackOutside`
            // does not cut at.
            let mut lit = part.offset(-self.x, 0);
            if part.right >= visible.right {
                lit.right = lit.right.max(texture.width as i32);
            }
            if part.bottom >= visible.bottom {
                lit.bottom = lit.bottom.max(texture.height as i32);
            }
            let black = BlackOutside {
                lit,
                width: texture.width as u32,
                height: texture.height as u32,
                clamp_u: draw.flags & draw_flags::CLAMP_U != 0,
                clamp_v: draw.flags & draw_flags::CLAMP_V != 0,
            };
            let shown = black.lit_screen_box(vertices, indices)?.intersect(area);
            if !shown.is_empty() {
                out[count] = shown;
                count += 1;
            }
        }
        Some(count)
    }
}

/// True if a fragment of textured `draw` whose texels are all black (RGB
/// 0) is black itself: the texel times the vertex colour, nothing added
/// (no fog; no dither, which adds to the colour before a 16-bit target's
/// cut; vertices of the frame's own, which carry no second colour).
fn black_texel_gives_black(draw: &Draw) -> bool {
    draw.flags & draw_flags::TEXTURED != 0
        && draw.flags & (draw_flags::FOG | draw_flags::DITHER | draw_flags::RAW_VERTICES | draw_flags::CACHED_VERTICES) == 0
        && draw.transform == 0
        && draw.tests & test_flags::LOGIC_OP == 0
}

/// True if the float with these bits is finite.
#[inline]
fn finite_bits(bits: u32) -> bool {
    bits & 0x7F80_0000 != 0x7F80_0000
}

/// A finite float's bits as an i32 that orders as the float does (-0
/// just below +0): integer compares instead of float ones.
#[inline]
fn order_key(bits: u32) -> i32 {
    let signed = bits as i32;
    if signed < 0 {
        signed ^ 0x7FFF_FFFF
    } else {
        signed
    }
}

/// The float of an `order_key`.
fn from_order_key(key: i32) -> f32 {
    f32::from_bits(order_key(key as u32) as u32)
}

/// The scissor of `draw` as a rectangle.
fn scissor(draw: &Draw) -> Rect {
    Rect::new(
        draw.scissor_left as i32,
        draw.scissor_top as i32,
        draw.scissor_right as i32 + 1,
        draw.scissor_bottom as i32 + 1,
    )
}

/// The vertices of a draw with screen-space vertices of its own (not raw,
/// not cached, no GPU transform); None otherwise.
fn own_screen_vertices<'a>(frame: &'a Frame, draw: &Draw) -> Option<&'a [FrameVertex]> {
    if draw.flags & (draw_flags::RAW_VERTICES | draw_flags::CACHED_VERTICES) != 0 || draw.transform != 0 {
        return None;
    }
    let first = draw.first_vertex as usize;
    frame.vertices.get(first..first.checked_add(draw.vertex_count as usize)?)
}

/// The part of its framebuffer `draw` may change: its vertices' box (one
/// pixel more on each side) inside the scissor, or the whole scissor where
/// its vertices aren't screen-space ones of its own. `EMPTY` for draws
/// that change no colour (core `hidden_draws::colourless`).
pub fn paint_area(frame: &Frame, draw: &Draw) -> Rect {
    if crate::hidden_draws::colourless(draw) {
        return Rect::EMPTY;
    }
    let scissor = scissor(draw);
    let Some(vertices) = own_screen_vertices(frame, draw) else {
        return scissor;
    };
    if vertices.is_empty() {
        return Rect::EMPTY;
    }
    // Compared as `order_key`s: float compares cost the Cortex-A9 a VFP
    // status transfer each (~10 per vertex took God of War's frame ~2 ms).
    let (mut left, mut top, mut right, mut bottom) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
    for vertex in vertices {
        let (x, y) = (vertex.x.to_bits(), vertex.y.to_bits());
        if !(finite_bits(x) && finite_bits(y)) {
            return scissor;
        }
        let (x, y) = (order_key(x), order_key(y));
        left = left.min(x);
        top = top.min(y);
        right = right.max(x);
        bottom = bottom.max(y);
    }
    // (Far outside the screen: the scissor bounds it anyway.)
    let whole = |key: i32| from_order_key(key).clamp(-65536.0, 65536.0);
    let boxed = Rect::new(
        floor(whole(left)) as i32 - 1,
        floor(whole(top)) as i32 - 1,
        ceil(whole(right)) as i32 + 1,
        ceil(whole(bottom)) as i32 + 1,
    );
    boxed.intersect(scissor)
}

/// The rectangle `draw` makes black (RGB 0) in its framebuffer, if it is a
/// black fill: one untextured rectangle (two triangles sharing a diagonal of
/// it) of colour 0, every pixel written as it is (no test that can fail,
/// no dither, no fog; blending off, or a normal blend of alpha 255, or a
/// destination factor of zero), cut by the scissor. Only pixels whose
/// centres lie inside the rectangle count.
pub fn black_fill(frame: &Frame, draw: &Draw) -> Option<Rect> {
    if draw.flags & (draw_flags::TEXTURED | draw_flags::DITHER | draw_flags::FOG) != 0 {
        return None;
    }
    let vertices = own_screen_vertices(frame, draw)?;
    if draw.tests & (test_flags::ALPHA_TEST | test_flags::COLOR_TEST | test_flags::LOGIC_OP) != 0 {
        return None;
    }
    // A depth test that passes everywhere only ("always").
    if draw.tests & test_flags::DEPTH_TEST != 0 && (draw.tests >> test_flags::DEPTH_FUNCTION_SHIFT) & 7 != 1 {
        return None;
    }
    // A stencil test only as an exact state without a test. A 565 target
    // has no stencil: nothing is tested there (core `blend::stencil` gives
    // such a draw no state; God of War clears its half-size blur target
    // with the test's flag still set).
    let no_stencil = (draw.flags & draw_flags::TARGET_FORMAT_MASK) >> draw_flags::TARGET_FORMAT_SHIFT == draw_flags::TARGET_565;
    if draw.tests & test_flags::STENCIL_TEST != 0 && !no_stencil && !frame.stencil_state_of(draw).is_some_and(|state| state.test().is_none()) {
        return None;
    }
    if frame.stencil_state_of(draw).is_some_and(|state| state.test().is_some()) {
        return None;
    }
    if draw.blend & BLEND_ALPHA_ONLY != 0 {
        return None;
    }
    let mut needs_opaque = false;
    if draw.flags & draw_flags::BLEND != 0 && draw.blend & BLEND_ENABLED != 0 {
        if draw.blend & BLEND_FIX_B_ALPHA != 0 {
            return None;
        }
        let (source, destination, equation) = (draw.blend & 0xF, (draw.blend >> 4) & 0xF, (draw.blend >> 8) & 7);
        if equation != Equation::Add as u32 {
            return None;
        }
        if destination == Factor::Zero as u32 {
            // source x factor + 0: 0 for a source of 0.
        } else if (source, destination) == (Factor::SourceAlpha as u32, Factor::OneMinusSourceAlpha as u32) {
            needs_opaque = true; // the GE's terms give the source exactly then
        } else {
            return None;
        }
    }
    // The corners: two triangles.
    let indices = frame.indices.get(draw.first_index as usize..draw.first_index as usize + draw.index_count as usize)?;
    if draw.flags & draw_flags::INDEX_RUNS != 0 || indices.len() != 6 {
        return None;
    }
    let mut corners = [FrameVertex::default(); 6];
    for (corner, &index) in corners.iter_mut().zip(indices) {
        *corner = *vertices.get(index as usize)?;
    }
    if corners.iter().any(|c| c.color & 0x00FF_FFFF != 0 || (needs_opaque && c.color >> 24 != 0xFF)) {
        return None;
    }
    let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for c in &corners {
        if !(c.x.is_finite() && c.y.is_finite()) {
            return None;
        }
        x0 = x0.min(c.x);
        y0 = y0.min(c.y);
        x1 = x1.max(c.x);
        y1 = y1.max(c.y);
    }
    // Each corner at a corner of the box: as numbers 0..3 (bit 0 right, bit 1 bottom).
    let mut numbers = [0u8; 6];
    for (number, c) in numbers.iter_mut().zip(&corners) {
        let right = if c.x == x1 { 1 } else if c.x == x0 { 0 } else { return None };
        let bottom = if c.y == y1 { 2 } else if c.y == y0 { 0 } else { return None };
        *number = right | bottom;
    }
    let mask = |triangle: &[u8]| triangle.iter().fold(0u8, |mask, &n| mask | 1 << n);
    let (first, second) = (mask(&numbers[..3]), mask(&numbers[3..]));
    // Three different corners each, sharing two that lie diagonally (0 and
    // 3, or 1 and 2): together they cover the box.
    let shared = first & second;
    if first.count_ones() != 3 || second.count_ones() != 3 || !(shared == 0b1001 || shared == 0b0110) {
        return None;
    }
    // (Far outside the screen: the scissor bounds it anyway.)
    let clamp = |value: f32| value.clamp(-65536.0, 65536.0);
    let fill = Rect::new(ceil(clamp(x0)) as i32, ceil(clamp(y0)) as i32, floor(clamp(x1)) as i32, floor(clamp(y1)) as i32);
    Some(fill.intersect(scissor(draw)))
}

/// True if a fragment of `draw` whose texel is black (RGB 0) leaves the
/// target exactly as it is: textured, screen-space vertices of its own, an
/// 8-bit target, additive blending with destination factor one or one
/// minus the source colour (the GE's terms then give 0 for the source and
/// the target's colour itself; GXM's blend too), no fog, no dither (each
/// would add to the source), and nothing else written: no depth, no alpha
/// (stencil), no logic op. A source of 0: the texel times the vertex
/// colour (modulate; the other functions are drawn as modulate, recorder
/// `approx::TEXTURE_FUNCTION`), or the texel (replace), doubled or not.
pub fn zero_source_keeps_target(frame: &Frame, draw: &Draw) -> bool {
    let target = (draw.flags & draw_flags::TARGET_FORMAT_MASK) >> draw_flags::TARGET_FORMAT_SHIFT;
    let writes_depth = draw.tests & test_flags::DEPTH_TEST != 0 && draw.tests & test_flags::DEPTH_WRITE_DISABLED == 0;
    let (destination, equation) = ((draw.blend >> 4) & 0xF, (draw.blend >> 8) & 7);
    draw.flags & draw_flags::TEXTURED != 0
        && draw.flags & (draw_flags::FOG | draw_flags::DITHER | draw_flags::RAW_VERTICES | draw_flags::CACHED_VERTICES) == 0
        && draw.flags & draw_flags::BLEND != 0
        && draw.transform == 0
        && target == 0
        && draw.blend & BLEND_ENABLED != 0
        && draw.blend & (BLEND_FIX_B_ALPHA | BLEND_WRITES_ALPHA | BLEND_ALPHA_ONLY) == 0
        && equation == Equation::Add as u32
        && (destination == Factor::One as u32 || destination == Factor::OneMinusSourceColor as u32)
        && draw.tests & test_flags::LOGIC_OP == 0
        && !writes_depth
        && frame.stencil_state_of(draw).is_none_or(|state| state.write().is_none())
}

/// A texture whose texels outside `lit` are black (RGB 0): `lit` in texels
/// (`TargetContents::lit_box`), the texture `width` x `height`, and
/// whether its coordinates are clamped (`draw_flags::CLAMP_U` / `CLAMP_V`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlackOutside {
    pub lit: Rect,
    pub width: u32,
    pub height: u32,
    pub clamp_u: bool,
    pub clamp_v: bool,
}

/// How far (in texels) a triangle's texture coordinates must stay from the
/// non-black texels. A bilinear sample at coordinate c (texels) reads
/// columns floor(c - 1/2) and the one after, so a sample left of column L
/// with c < L - 1/2 reads only columns before L; one more texel covers the
/// interpolation's rounding, at 2x the pixel centres and the GE filter's
/// 1/16 steps.
const MARGIN: f32 = 1.5;

/// `BlackOutside::side_code` bits: a vertex left of / right of / above /
/// below the non-black texels by more than `MARGIN`.
const LEFT: u8 = 1;
const RIGHT: u8 = 2;
const ABOVE: u8 = 4;
const BELOW: u8 = 8;

/// `BlackOutside`'s four limits as `order_key`s of texture coordinates
/// (0..1): a coordinate whose key is below `left` (above `right`, ...) is
/// that far out. Sides that can't be culled have limits nothing passes.
#[derive(Clone, Copy, Debug)]
pub struct Sides {
    left: i32,
    right: i32,
    top: i32,
    bottom: i32,
}

impl BlackOutside {
    /// The limits of `side_code`. A limit is the texel position divided
    /// by the size, as a float: its rounding is far inside the margin.
    pub fn sides(&self) -> Sides {
        let key = |texels: f32, size: u32| order_key((texels / size as f32).to_bits());
        let (lit, width, height) = (self.lit, self.width as i32, self.height as i32);
        Sides {
            left: if self.clamp_u && lit.left >= 1 { key(lit.left as f32 - MARGIN, self.width) } else { i32::MIN },
            right: if self.clamp_u && lit.right < width { key(lit.right as f32 + MARGIN, self.width) } else { i32::MAX },
            top: if self.clamp_v && lit.top >= 1 { key(lit.top as f32 - MARGIN, self.height) } else { i32::MIN },
            bottom: if self.clamp_v && lit.bottom < height { key(lit.bottom as f32 + MARGIN, self.height) } else { i32::MAX },
        }
    }

    /// False if no triangle can go: some texels lit and no side to cull
    /// on (lit up to the edges, or wrapped coordinates).
    pub fn culls_anything(&self) -> bool {
        let sides = self.sides();
        self.lit.is_empty() || sides.left != i32::MIN || sides.right != i32::MAX || sides.top != i32::MIN || sides.bottom != i32::MAX
    }

    /// The sides (`LEFT`, ...) a vertex is out on; 0 if its texture
    /// coordinates aren't finite or its w isn't positive and finite.
    #[inline]
    pub fn side_code(sides: &Sides, vertex: &FrameVertex) -> u8 {
        let (u, v, w) = (vertex.u.to_bits(), vertex.v.to_bits(), vertex.w.to_bits());
        if !(finite_bits(u) && finite_bits(v)) || !(w as i32 > 0 && w < 0x7F80_0000) {
            return 0;
        }
        let (u, v) = (order_key(u), order_key(v));
        (u < sides.left) as u8 | ((u > sides.right) as u8) << 1 | ((v < sides.top) as u8) << 2 | ((v > sides.bottom) as u8) << 3
    }

    /// False if every fragment of the triangle with these corners samples
    /// black texels only: all three corners out on the same side.
    /// Interpolated texture coordinates stay between the corners' (positive
    /// w: perspective weights are positive too), and clamping keeps them
    /// on the same side of a column that isn't the first or the last.
    pub fn keeps_triangle(&self, corners: [&FrameVertex; 3]) -> bool {
        if self.lit.is_empty() {
            return false;
        }
        let sides = self.sides();
        corners.iter().fold(LEFT | RIGHT | ABOVE | BELOW, |out, corner| out & Self::side_code(&sides, corner)) == 0
    }

    /// Copies the triangles of `indices` (three corners each, counted from
    /// the start of `vertices`) that `keeps_triangle` keeps to `out` (as
    /// long as `indices`); returns how many indices it wrote. Triangles
    /// with corners outside `vertices` are kept.
    pub fn cull_triangles(&self, vertices: &[FrameVertex], indices: &[u16], out: &mut [u16]) -> usize {
        if self.lit.is_empty() {
            return 0;
        }
        let sides = self.sides();
        let code = |index: u16| vertices.get(index as usize).map_or(0, |vertex| Self::side_code(&sides, vertex));
        let mut written = 0;
        for triangle in indices.chunks_exact(3) {
            if code(triangle[0]) & code(triangle[1]) & code(triangle[2]) == 0 {
                out[written..written + 3].copy_from_slice(triangle);
                written += 3;
            }
        }
        written
    }

    /// The four limits of `sides` as texture coordinates (0..1), in the
    /// order left, right, top, bottom; None for a side nothing is out on.
    fn limits(&self) -> [Option<f32>; 4] {
        let (lit, width, height) = (self.lit, self.width as i32, self.height as i32);
        let at = |texels: f32, size: u32| texels / size as f32;
        [
            (self.clamp_u && lit.left >= 1).then(|| at(lit.left as f32 - MARGIN, self.width)),
            (self.clamp_u && lit.right < width).then(|| at(lit.right as f32 + MARGIN, self.width)),
            (self.clamp_v && lit.top >= 1).then(|| at(lit.top as f32 - MARGIN, self.height)),
            (self.clamp_v && lit.bottom < height).then(|| at(lit.bottom as f32 + MARGIN, self.height)),
        ]
    }

    /// The pixels of the target (PSP pixels, as the vertices' x and y)
    /// outside of which the triangles of `indices` (three corners each,
    /// counted from the start of `vertices`) sample black texels only.
    /// `EMPTY` if no fragment can sample anything else; None if it can't
    /// be told (a corner outside `vertices`, not finite, w not positive, or
    /// farther out than `SCREEN_LIMIT`).
    ///
    /// Why the box holds every fragment that matters: at a point of a
    /// triangle with screen weights b0, b1, b2 (>= 0) the GPU's texture
    /// coordinate is u = (sum bi ui / wi) / (sum bi / wi), so "left of the
    /// limit L", u < L, is sum bi (ui - L) / wi < 0: a function that is
    /// linear on the screen. Each side's out-region is therefore a half
    /// plane, and what is out on no side is the triangle cut by up to four
    /// of them, a convex polygon. The GPU takes a fragment's texture
    /// coordinate at its pixel's centre: half a pixel in on our 1:1
    /// targets, a quarter or three quarters of a PSP pixel on the 2x ones,
    /// so a PSP pixel p matters only if p + 1/4 .. p + 3/4 reaches the
    /// polygon's box (`CENTRE_SLACK` more on both sides: the float rounding
    /// of the cuts, far below it). A fragment outside the box is out on
    /// some side by more than `MARGIN`, like every fragment of a culled
    /// triangle; of `MARGIN`'s 1.5 texels one is spare for the GPU's own
    /// rounding (half a pixel and more here: the texels are magnified or
    /// at most halved).
    pub fn lit_screen_box(&self, vertices: &[FrameVertex], indices: &[u16]) -> Option<Rect> {
        if self.lit.is_empty() {
            return Some(Rect::EMPTY);
        }
        let limits = self.limits();
        // The box as order keys (integer compares): left, top, right, bottom.
        let mut bounds = [i32::MAX, i32::MAX, i32::MIN, i32::MIN];
        let mut grow = |x: f32, y: f32| {
            let (x, y) = (order_key(x.to_bits()), order_key(y.to_bits()));
            bounds = [bounds[0].min(x), bounds[1].min(y), bounds[2].max(x), bounds[3].max(y)];
        };
        for triangle in indices.chunks_exact(3) {
            let mut polygon = [ClipCorner::default(); CLIPPED_CORNERS];
            let mut inside_all = true;
            for (slot, index) in triangle.iter().enumerate() {
                let vertex = vertices.get(*index as usize)?;
                let corner = ClipCorner::of(vertex, &limits)?;
                inside_all &= corner.out.iter().all(|distance| distance.to_bits() as i32 >= 0);
                polygon[slot] = corner;
            }
            if inside_all {
                // (Most kept triangles near a blob: no cut needed.)
                polygon[..3].iter().for_each(|corner| grow(corner.x, corner.y));
                continue;
            }
            let mut count = 3;
            for side in 0..4 {
                if limits[side].is_some() {
                    count = clip_side(&mut polygon, count, side);
                }
            }
            polygon[..count].iter().for_each(|corner| grow(corner.x, corner.y));
        }
        if bounds[0] > bounds[2] || bounds[1] > bounds[3] {
            return Some(Rect::EMPTY);
        }
        let [left, top, right, bottom] = bounds.map(from_order_key);
        // The pixels with a centre (1/4 .. 3/4) in the box.
        let first = |low: f32| ceil(low - (0.75 + CENTRE_SLACK)) as i32;
        let end = |high: f32| floor(high - (0.25 - CENTRE_SLACK)) as i32 + 1;
        Some(Rect::new(first(left), first(top), end(right), end(bottom)))
    }
}

/// How far (PSP pixels) outside a pixel's 1/4 .. 3/4 `lit_screen_box`
/// still counts a point as its centre.
const CENTRE_SLACK: f32 = 1.0 / 16.0;

/// The corners a triangle can have after `lit_screen_box`'s cuts: one more
/// per side.
const CLIPPED_CORNERS: usize = 7;

/// The largest |x| or |y| (PSP pixels) of a corner `lit_screen_box` works
/// with: up to here an f32 cut point is right to a small fraction of a
/// pixel (the GE's own coordinates end at 4096).
const SCREEN_LIMIT: f32 = 16384.0;

/// A corner of a triangle being cut in `lit_screen_box`: where it is on
/// the screen and, per side (left, right, top, bottom), how far inside
/// that side's limit its texture coordinate is, divided by w (negative:
/// out; linear on the screen; 0 for a side without limit).
#[derive(Clone, Copy, Default)]
struct ClipCorner {
    x: f32,
    y: f32,
    out: [f32; 4],
}

impl ClipCorner {
    /// None for a vertex `lit_screen_box` can't work with.
    fn of(vertex: &FrameVertex, limits: &[Option<f32>; 4]) -> Option<ClipCorner> {
        let (x, y, u, v, w) = (vertex.x.to_bits(), vertex.y.to_bits(), vertex.u.to_bits(), vertex.v.to_bits(), vertex.w.to_bits());
        let on_screen = |bits: u32| bits & 0x7FFF_FFFF <= SCREEN_LIMIT.to_bits();
        if !(on_screen(x) && on_screen(y) && finite_bits(u) && finite_bits(v)) || !(w as i32 > 0 && w < 0x7F80_0000) {
            return None;
        }
        let over_w = 1.0 / vertex.w;
        let inside = |limit: Option<f32>, distance: &dyn Fn(f32) -> f32| limit.map_or(0.0, |limit| distance(limit) * over_w);
        let out = [
            inside(limits[0], &|limit| vertex.u - limit),
            inside(limits[1], &|limit| limit - vertex.u),
            inside(limits[2], &|limit| vertex.v - limit),
            inside(limits[3], &|limit| limit - vertex.v),
        ];
        // (A tiny w can make them infinite: nothing to cut with.)
        out.iter().all(|distance| finite_bits(distance.to_bits())).then_some(ClipCorner { x: vertex.x, y: vertex.y, out })
    }
}

/// Cuts the convex polygon `polygon[..count]` down to its part not out on
/// `side` (`ClipCorner::out[side]` >= 0), in place; returns its new corner
/// count (0: nothing left; at most one corner more than before).
fn clip_side(polygon: &mut [ClipCorner; CLIPPED_CORNERS], count: usize, side: usize) -> usize {
    let source = *polygon;
    let is_in = |corner: &ClipCorner| corner.out[side].to_bits() as i32 >= 0;
    let mut written = 0;
    for k in 0..count {
        let (from, to) = (&source[k], &source[(k + 1) % count]);
        if is_in(from) && written < CLIPPED_CORNERS {
            polygon[written] = *from;
            written += 1;
        }
        if is_in(from) != is_in(to) && written < CLIPPED_CORNERS {
            // Where the edge crosses the limit: the distances have opposite
            // signs, so the difference is the sum of their sizes.
            let t = from.out[side] / (from.out[side] - to.out[side]);
            let mix = |a: f32, b: f32| a + (b - a) * t;
            let mut cut = ClipCorner { x: mix(from.x, to.x), y: mix(from.y, to.y), out: [0.0; 4] };
            for other in 0..4 {
                cut.out[other] = if other == side { 0.0 } else { mix(from.out[other], to.out[other]) };
            }
            polygon[written] = cut;
            written += 1;
        }
    }
    written
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blend::Blend;

    fn vertex(x: f32, y: f32, u: f32, v: f32, color: u32) -> FrameVertex {
        FrameVertex { x, y, z: 0.0, w: 1.0, u, v, color, fog: 1.0 }
    }

    /// A frame with `vertices` and `indices`, and one draw over them.
    fn frame_with(vertices: &[FrameVertex], indices: &[u16], draw: Draw) -> Box<Frame> {
        let mut frame = crate::frame::new_boxed_frame();
        frame.vertices[..vertices.len()].copy_from_slice(vertices);
        frame.vertex_count = vertices.len() as u32;
        frame.indices[..indices.len()].copy_from_slice(indices);
        frame.index_count = indices.len() as u32;
        frame.draws[0] = Draw { vertex_count: vertices.len() as u32, index_count: indices.len() as u32, ..draw };
        frame.draw_count = 1;
        frame
    }

    fn full_scissor() -> Draw {
        Draw { scissor_left: 0, scissor_top: 0, scissor_right: 479, scissor_bottom: 271, ..Default::default() }
    }

    /// God of War's strip clear: two triangles 0..448 x 0..64, colour
    /// ff000000, normal blend (as the frame dump has it).
    fn clear(color: u32) -> (Vec<FrameVertex>, Draw) {
        let corners = [(0.0, 0.0), (448.0, 0.0), (0.0, 64.0), (448.0, 0.0), (448.0, 64.0), (0.0, 64.0)];
        let vertices = corners.iter().map(|&(x, y)| vertex(x, y, 0.0, 0.0, color)).collect();
        let draw = Draw {
            flags: draw_flags::BLEND | draw_flags::RECTANGLES,
            blend: Blend::ALPHA.pack(),
            ..full_scissor()
        };
        (vertices, draw)
    }

    #[test]
    fn a_black_rectangle_is_a_fill() {
        let (vertices, draw) = clear(0xFF00_0000);
        let frame = frame_with(&vertices, &[0, 1, 2, 3, 4, 5], draw);
        assert_eq!(black_fill(&frame, &frame.draws[0]), Some(Rect::new(0, 0, 448, 64)));
    }

    #[test]
    fn fills_need_black_opaque_corners_covering_the_box() {
        // Not black.
        let (vertices, draw) = clear(0xFF00_0001);
        let frame = frame_with(&vertices, &[0, 1, 2, 3, 4, 5], draw);
        assert_eq!(black_fill(&frame, &frame.draws[0]), None);
        // Black but see-through under a normal blend: the old colour stays partly.
        let (vertices, draw) = clear(0x8000_0000);
        let frame = frame_with(&vertices, &[0, 1, 2, 3, 4, 5], draw);
        assert_eq!(black_fill(&frame, &frame.draws[0]), None);
        // Two triangles not sharing a diagonal (corners 0 1 2 and 0 1 3): a gap.
        let (vertices, draw) = clear(0xFF00_0000);
        let frame = frame_with(&vertices, &[0, 1, 2, 0, 1, 4], draw);
        assert_eq!(black_fill(&frame, &frame.draws[0]), None);
        // Blending off: any alpha.
        let (vertices, mut draw) = clear(0x0000_0000);
        draw.flags = draw_flags::RECTANGLES;
        draw.blend = 0;
        let frame = frame_with(&vertices, &[0, 1, 2, 3, 4, 5], draw);
        assert_eq!(black_fill(&frame, &frame.draws[0]), Some(Rect::new(0, 0, 448, 64)));
        // Dither or an alpha test: no.
        for (flags, tests) in [(draw_flags::DITHER, 0), (0, test_flags::ALPHA_TEST)] {
            let (vertices, mut draw) = clear(0xFF00_0000);
            draw.flags |= flags;
            draw.tests |= tests;
            let frame = frame_with(&vertices, &[0, 1, 2, 3, 4, 5], draw);
            assert_eq!(black_fill(&frame, &frame.draws[0]), None);
        }
        // A stencil test without an exact state: not known to pass. On a
        // 565 target there is no stencil to test (God of War's clear of
        // its half-size blur target, flags 2000042, tests 4).
        let (vertices, mut draw) = clear(0xFF00_0000);
        draw.tests |= test_flags::STENCIL_TEST;
        let frame = frame_with(&vertices, &[0, 1, 2, 3, 4, 5], draw);
        assert_eq!(black_fill(&frame, &frame.draws[0]), None);
        draw.flags |= draw_flags::TARGET_565 << draw_flags::TARGET_FORMAT_SHIFT;
        let frame = frame_with(&vertices, &[0, 1, 2, 3, 4, 5], draw);
        assert_eq!(black_fill(&frame, &frame.draws[0]), Some(Rect::new(0, 0, 448, 64)));
    }

    #[test]
    fn fills_are_cut_by_the_scissor_and_take_whole_pixels() {
        let (mut vertices, mut draw) = clear(0xFF00_0000);
        for vertex in vertices.iter_mut() {
            vertex.x = if vertex.x > 0.0 { 447.5 } else { 0.25 };
        }
        draw.scissor_left = 2;
        draw.scissor_top = 2;
        draw.scissor_right = 59;
        draw.scissor_bottom = 59;
        let frame = frame_with(&vertices, &[0, 1, 2, 3, 4, 5], draw);
        assert_eq!(black_fill(&frame, &frame.draws[0]), Some(Rect::new(2, 2, 60, 60)));
    }

    #[test]
    fn contents_follow_fills_and_paints() {
        let mut contents = TargetContents::UNKNOWN;
        let tile = |x: i32| Rect::new(x, 0, x + 64, 64);
        assert_eq!(contents.lit_box(tile(0)), None);
        contents.note_fill(Rect::new(0, 0, 448, 64));
        assert_eq!(contents.lit_box(tile(64)), Some(Rect::EMPTY));
        // A blob in the second tile (target x 64 + 15 .. 64 + 38).
        contents.note_paint(Rect::new(79, 26, 102, 48));
        assert_eq!(contents.lit_box(tile(0)), Some(Rect::EMPTY));
        assert_eq!(contents.lit_box(tile(64)), Some(Rect::new(15, 26, 38, 48)));
        // A part reaching past the black fill: unknown.
        assert_eq!(contents.lit_box(Rect::new(440, 0, 504, 64)), None);
        // A new fill over the tile makes it black again.
        contents.note_fill(tile(64));
        assert_eq!(contents.lit_box(tile(64)), Some(Rect::EMPTY));
        contents.forget();
        assert_eq!(contents.lit_box(tile(64)), None);
    }

    #[test]
    fn too_many_paints_merge_and_stay_covered() {
        let mut contents = TargetContents::UNKNOWN;
        contents.note_fill(Rect::new(0, 0, 480, 272));
        for k in 0..(PAINTED_PARTS as i32 + 8) {
            contents.note_paint(Rect::new(10 * k, 5, 10 * k + 3, 8));
        }
        let lit = contents.lit_box(Rect::new(0, 0, 480, 272)).unwrap();
        for k in 0..(PAINTED_PARTS as i32 + 8) {
            assert!(lit.contains(Rect::new(10 * k, 5, 10 * k + 3, 8)), "{k}: {lit:?}");
        }
    }

    #[test]
    fn paint_areas_are_vertex_boxes_inside_the_scissor() {
        let vertices = [vertex(22.4, 26.0, 0.0, 0.0, 0xFF8C_9195), vertex(40.0, 49.5, 0.0, 0.0, 0), vertex(30.0, 30.0, 0.0, 0.0, 0)];
        let draw = Draw { flags: draw_flags::BLEND, blend: Blend::ALPHA.pack(), scissor_left: 2, scissor_top: 2, scissor_right: 59, scissor_bottom: 59, ..Default::default() };
        let frame = frame_with(&vertices, &[0, 1, 2], draw);
        assert_eq!(paint_area(&frame, &frame.draws[0]), Rect::new(21, 25, 41, 51));
        // Model-space vertices: the whole scissor.
        let frame = frame_with(&vertices, &[0, 1, 2], Draw { transform: 1, ..draw });
        assert_eq!(paint_area(&frame, &frame.draws[0]), Rect::new(2, 2, 60, 60));
    }

    /// The shadow pass: texture x (1 - source colour) onto the screen.
    fn shadow_pass() -> Draw {
        let blend = Blend { source: Factor::Zero, destination: Factor::OneMinusSourceColor, ..Blend::ALPHA }.pack();
        Draw {
            flags: draw_flags::TEXTURED | draw_flags::BLEND | draw_flags::LINEAR_FILTER | draw_flags::CLAMP_U | draw_flags::CLAMP_V,
            blend,
            tests: test_flags::DEPTH_TEST | test_flags::DEPTH_WRITE_DISABLED | 5 << test_flags::DEPTH_FUNCTION_SHIFT,
            ..full_scissor()
        }
    }

    #[test]
    fn shadow_passes_keep_the_target_where_the_texel_is_black() {
        let vertices = [vertex(0.0, 0.0, 0.0, 0.0, !0)];
        let frame = frame_with(&vertices, &[0, 0, 0], shadow_pass());
        assert!(zero_source_keeps_target(&frame, &frame.draws[0]));
        let changed = |change: &dyn Fn(&mut Draw)| {
            let mut draw = shadow_pass();
            change(&mut draw);
            let frame = frame_with(&vertices, &[0, 0, 0], draw);
            zero_source_keeps_target(&frame, &frame.draws[0])
        };
        assert!(!changed(&|d| d.flags |= draw_flags::FOG));
        assert!(!changed(&|d| d.flags |= draw_flags::DITHER));
        assert!(!changed(&|d| d.tests &= !test_flags::DEPTH_WRITE_DISABLED)); // writes depth
        assert!(!changed(&|d| d.blend |= BLEND_WRITES_ALPHA));
        assert!(!changed(&|d| d.blend = Blend::ALPHA.pack())); // normal blend: 0 x a + d x (1 - a)
        assert!(changed(&|d| d.blend = Blend { source: Factor::SourceAlpha, destination: Factor::One, ..Blend::ALPHA }.pack()));
        assert!(changed(&|d| d.tests = 0)); // no depth test: writes nothing either
    }

    fn texture(lit: Rect) -> BlackOutside {
        BlackOutside { lit, width: 64, height: 64, clamp_u: true, clamp_v: true }
    }

    /// A triangle with texture coordinates (in texels) at its corners.
    fn triangle(texels: [(f32, f32); 3]) -> [FrameVertex; 3] {
        texels.map(|(u, v)| vertex(0.0, 0.0, u / 64.0, v / 64.0, !0))
    }

    #[test]
    fn triangles_away_from_the_lit_texels_go() {
        let black = texture(Rect::new(20, 20, 40, 40));
        let keeps = |t: [FrameVertex; 3]| black.keeps_triangle([&t[0], &t[1], &t[2]]);
        // All left of column 20 by more than the margin.
        assert!(!keeps(triangle([(-500.0, 0.0), (18.4, 30.0), (0.0, 64.0)])));
        // Too close: a sample at 18.6 reads columns 18 and 19 only, but the margin keeps it.
        assert!(keeps(triangle([(-500.0, 0.0), (18.6, 30.0), (0.0, 64.0)])));
        // Right of the last lit column (39): from 41.5 on.
        assert!(!keeps(triangle([(41.6, 0.0), (300.0, 30.0), (50.0, 64.0)])));
        assert!(keeps(triangle([(41.4, 0.0), (300.0, 30.0), (50.0, 64.0)])));
        // Above and below.
        assert!(!keeps(triangle([(0.0, -10.0), (64.0, 18.0), (30.0, 0.0)])));
        assert!(!keeps(triangle([(0.0, 42.0), (64.0, 100.0), (30.0, 64.0)])));
        // Spanning the lit texels from left to right: kept.
        assert!(keeps(triangle([(-500.0, 0.0), (500.0, 30.0), (0.0, 64.0)])));
        // Nothing lit: every triangle goes.
        assert!(!texture(Rect::EMPTY).keeps_triangle([&triangle([(30.0, 30.0); 3])[0]; 3]));
        assert!(black.culls_anything() && texture(Rect::EMPTY).culls_anything());
        // Lit up to every edge: nothing to do.
        assert!(!texture(Rect::new(0, 0, 64, 64)).culls_anything());
    }

    #[test]
    fn lit_texels_at_the_edge_keep_that_side() {
        // Lit from column 0: clamped samples far left still read column 0.
        let black = texture(Rect::new(0, 20, 10, 40));
        let t = triangle([(-500.0, 30.0), (-400.0, 30.0), (-450.0, 31.0)]);
        assert!(black.keeps_triangle([&t[0], &t[1], &t[2]]));
        // Wrapped coordinates: only an all-black texture says anything.
        let wrapped = BlackOutside { clamp_u: false, ..texture(Rect::new(20, 20, 40, 40)) };
        let t = triangle([(-500.0, 30.0), (-400.0, 30.0), (-450.0, 31.0)]);
        assert!(wrapped.keeps_triangle([&t[0], &t[1], &t[2]]));
    }

    #[test]
    fn culling_copies_the_kept_triangles() {
        let black = texture(Rect::new(20, 20, 40, 40));
        let mut vertices = triangle([(-500.0, 0.0), (0.0, 30.0), (0.0, 64.0)]).to_vec();
        vertices.extend(triangle([(25.0, 25.0), (30.0, 30.0), (35.0, 25.0)]));
        let indices = [0u16, 1, 2, 3, 4, 5, 0, 1, 9];
        let mut out = [0u16; 9];
        let written = black.cull_triangles(&vertices, &indices, &mut out);
        assert_eq!(&out[..written], &[3, 4, 5, 0, 1, 9]);
    }

    /// A frame with one rectangle drawn (two triangles): target pixels
    /// `to`, reading texels `from` of a `size` texture, clamped, unblended.
    fn copy_frame(to: Rect, from: Rect, size: (u16, u16), scissor: Rect) -> Box<Frame> {
        let corner = |right: bool, bottom: bool| FrameVertex {
            x: if right { to.right } else { to.left } as f32,
            y: if bottom { to.bottom } else { to.top } as f32,
            z: 0.0,
            w: 1.0,
            u: if right { from.right } else { from.left } as f32 / size.0 as f32,
            v: if bottom { from.bottom } else { from.top } as f32 / size.1 as f32,
            color: !0,
            fog: 1.0,
        };
        let vertices = [corner(false, false), corner(true, false), corner(false, true), corner(true, false), corner(true, true), corner(false, true)];
        let draw = Draw {
            flags: draw_flags::TEXTURED | draw_flags::LINEAR_FILTER | draw_flags::CLAMP_U | draw_flags::CLAMP_V | draw_flags::RECTANGLES,
            tests: test_flags::DEPTH_WRITE_DISABLED,
            scissor_left: scissor.left as u16,
            scissor_top: scissor.top as u16,
            scissor_right: scissor.right as u16 - 1,
            scissor_bottom: scissor.bottom as u16 - 1,
            ..Default::default()
        };
        let mut frame = frame_with(&vertices, &[0, 1, 2, 3, 4, 5], draw);
        frame.textures[0] = crate::frame::TextureDescriptor { width: size.0, height: size.1, buffer_width: size.0 as u32, ..Default::default() };
        frame.texture_count = 1;
        frame
    }

    /// God of War's shadow strip (2026-11-26): seven 64x64 tiles cleared
    /// to black, a blob drawn into some, then the whole strip drawn into a
    /// 224x32 target (cleared before) and that one back over the strip.
    #[test]
    fn a_copy_of_a_target_paints_only_where_its_lit_texels_land() {
        let mut strip = TargetContents::UNKNOWN;
        strip.note_fill(Rect::new(0, 0, 448, 64));
        strip.note_paint(Rect::new(23, 32, 41, 53)); // tile 0
        strip.note_paint(Rect::new(64 + 11, 22, 64 + 47, 52)); // tile 1
        strip.note_paint(Rect::new(384 + 20, 34, 384 + 43, 58)); // tile 6
        let mut half = TargetContents::UNKNOWN;
        half.note_fill(Rect::new(0, 0, 224, 32));

        // Down: 448x64 texels of a 512x64 texture onto 224x32 pixels.
        let down = copy_frame(Rect::new(0, 0, 224, 32), Rect::new(0, 0, 448, 64), (512, 64), Rect::new(1, 1, 223, 31));
        let source = Source { contents: &strip, x: 0, width: 448, height: 64 };
        half.note_draw(&down, &down.draws[0], 0, Some(source));
        // Each blob at half its place, 1.5 texels wider: texels 21.5 ..
        // 42.5 are pixels 10.75 .. 21.25, the centres of pixels 10 .. 21.
        assert_eq!(half.painted_count, 3);
        assert_eq!(half.painted[0], Rect::new(10, 15, 22, 28));
        assert_eq!(half.lit_box(Rect::new(0, 0, 224, 32)), Some(Rect::new(10, 10, 215, 30)));

        // Up again: 224x32 texels of a 256x32 texture onto the strip.
        let up = copy_frame(Rect::new(0, 0, 448, 64), Rect::new(0, 0, 224, 32), (256, 32), Rect::new(1, 1, 447, 63));
        let source = Source { contents: &half, x: 0, width: 224, height: 32 };
        strip.note_draw(&up, &up.draws[0], 0, Some(source));
        // The blurred blobs took the blobs' places in the list.
        assert_eq!(strip.painted_count, 3);
        let tile = |k: i32| Rect::new(64 * k, 0, 64 * k + 64, 64);
        assert_eq!(strip.lit_box(tile(0)), Some(Rect::new(17, 27, 47, 59)));
        assert_eq!(strip.lit_box(tile(2)), Some(Rect::EMPTY));
        assert!(strip.lit_box(tile(1)).is_some_and(|lit| lit.contains(Rect::new(11, 22, 47, 52)) && lit.area() < 64 * 64 * 2 / 3));
        assert!(strip.lit_box(tile(6)).is_some_and(|lit| lit.contains(Rect::new(20, 34, 43, 58)) && lit.left > 8));

        // Without the source (or with one that says nothing): all of it.
        let mut plain = TargetContents::UNKNOWN;
        plain.note_fill(Rect::new(0, 0, 448, 64));
        plain.note_draw(&up, &up.draws[0], 0, None);
        assert_eq!(plain.lit_box(tile(2)), Some(Rect::new(0, 1, 64, 63)));
        let mut plain = TargetContents::UNKNOWN;
        plain.note_fill(Rect::new(0, 0, 448, 64));
        plain.note_draw(&up, &up.draws[0], 0, Some(Source { contents: &TargetContents::UNKNOWN, x: 0, width: 224, height: 32 }));
        assert_eq!(plain.lit_box(tile(2)), Some(Rect::new(0, 1, 64, 63)));
        // Fog adds its colour to a black texel: all of it.
        let mut foggy = copy_frame(Rect::new(0, 0, 448, 64), Rect::new(0, 0, 224, 32), (256, 32), Rect::new(1, 1, 447, 63));
        foggy.draws[0].flags |= draw_flags::FOG;
        let mut plain = TargetContents::UNKNOWN;
        plain.note_fill(Rect::new(0, 0, 448, 64));
        plain.note_draw(&foggy, &foggy.draws[0], 0, Some(Source { contents: &half, x: 0, width: 224, height: 32 }));
        assert_eq!(plain.lit_box(tile(2)), Some(Rect::new(0, 1, 64, 63)));
    }

    /// What the source's black fill does not cover may hold anything.
    #[test]
    fn a_copy_paints_where_the_source_is_not_known_black() {
        // The source: 224 wide, black only up to x 200.
        let mut source = TargetContents::UNKNOWN;
        source.note_fill(Rect::new(0, 0, 200, 32));
        let mut target = TargetContents::UNKNOWN;
        target.note_fill(Rect::new(0, 0, 448, 64));
        let up = copy_frame(Rect::new(0, 0, 448, 64), Rect::new(0, 0, 224, 32), (256, 32), Rect::new(0, 0, 448, 64));
        target.note_draw(&up, &up.draws[0], 0, Some(Source { contents: &source, x: 0, width: 224, height: 32 }));
        // Texels from 200 - 1.5 on: pixels from 397.
        assert_eq!(target.lit_box(Rect::new(0, 0, 448, 64)), Some(Rect::new(397, 0, 448, 64)));
        // A tile of the source read from x 64 on: the same texels, 64 lower.
        let mut target = TargetContents::UNKNOWN;
        target.note_fill(Rect::new(0, 0, 448, 64));
        let up = copy_frame(Rect::new(0, 0, 320, 64), Rect::new(0, 0, 160, 32), (256, 32), Rect::new(0, 0, 448, 64));
        target.note_draw(&up, &up.draws[0], 0, Some(Source { contents: &source, x: 64, width: 224, height: 32 }));
        assert_eq!(target.lit_box(Rect::new(0, 0, 448, 64)), Some(Rect::new(269, 0, 320, 64)));
    }

    #[test]
    fn paints_inside_one_another_are_one_part() {
        let mut contents = TargetContents::UNKNOWN;
        contents.note_fill(Rect::new(0, 0, 448, 64));
        contents.note_paint(Rect::new(20, 20, 30, 30));
        contents.note_paint(Rect::new(100, 20, 110, 30));
        contents.note_paint(Rect::new(22, 22, 28, 28)); // inside the first
        assert_eq!(contents.painted_count, 2);
        contents.note_paint(Rect::new(15, 15, 40, 40)); // around the first
        assert_eq!(contents.painted_count, 2);
        assert_eq!(contents.lit_box(Rect::new(0, 0, 64, 64)), Some(Rect::new(15, 15, 40, 40)));
        assert_eq!(contents.lit_box(Rect::new(64, 0, 128, 64)), Some(Rect::new(36, 20, 46, 30)));
    }

    /// A corner at screen (x, y) with texture coordinates in texels and w.
    fn corner(x: f32, y: f32, u: f32, v: f32, w: f32) -> FrameVertex {
        FrameVertex { x, y, z: 0.0, w, u: u / 64.0, v: v / 64.0, color: !0, fog: 1.0 }
    }

    #[test]
    fn the_lit_screen_box_is_where_the_lit_texels_show() {
        // Texels 20..40 lit; one texel per 10 pixels along x and y, so with
        // the margin of 1.5 texels the lit ones show in 185..415: the pixels
        // with their centres there.
        let black = texture(Rect::new(20, 20, 40, 40));
        let big = [corner(-1000.0, -1000.0, -100.0, -100.0, 1.0), corner(3000.0, -1000.0, 300.0, -100.0, 1.0), corner(-1000.0, 3000.0, -100.0, 300.0, 1.0)];
        assert_eq!(black.lit_screen_box(&big, &[0, 1, 2]), Some(Rect::new(185, 185, 415, 415)));
        // A triangle inside the lit texels: its own box.
        let small = [corner(250.0, 250.0, 25.0, 25.0, 1.0), corner(300.5, 250.0, 30.05, 25.0, 1.0), corner(250.0, 290.0, 25.0, 29.0, 1.0)];
        assert_eq!(black.lit_screen_box(&small, &[0, 1, 2]), Some(Rect::new(250, 250, 301, 290)));
        // Both: the union. One that only sees black adds nothing.
        let mut all = big.to_vec();
        all.extend(small);
        all.extend([corner(0.0, 0.0, 0.0, 0.0, 1.0), corner(100.0, 0.0, 10.0, 0.0, 1.0), corner(0.0, 100.0, 0.0, 10.0, 1.0)]);
        assert_eq!(black.lit_screen_box(&all, &[0, 1, 2, 3, 4, 5, 6, 7, 8]), Some(Rect::new(185, 185, 415, 415)));
        assert_eq!(black.lit_screen_box(&all, &[6, 7, 8]), Some(Rect::EMPTY));
        // Around the corner of the lit texels without touching them (no
        // side has all three corners out): nothing either.
        let around = [corner(0.0, 300.0, 0.0, 30.0, 1.0), corner(300.0, 0.0, 30.0, 0.0, 1.0), corner(0.0, 0.0, 0.0, 0.0, 1.0)];
        assert!(black.keeps_triangle([&around[0], &around[1], &around[2]]));
        assert_eq!(black.lit_screen_box(&around, &[0, 1, 2]), Some(Rect::EMPTY));
        // Nothing lit: empty. Lit up to an edge: that side is not cut.
        assert_eq!(texture(Rect::EMPTY).lit_screen_box(&big, &[0, 1, 2]), Some(Rect::EMPTY));
        assert_eq!(texture(Rect::new(0, 20, 40, 40)).lit_screen_box(&big, &[0, 1, 2]), Some(Rect::new(-1000, 185, 415, 415)));
    }

    #[test]
    fn the_lit_screen_box_gives_up_on_corners_it_cannot_place() {
        let black = texture(Rect::new(20, 20, 40, 40));
        let good = [corner(0.0, 0.0, 0.0, 0.0, 1.0), corner(640.0, 0.0, 64.0, 0.0, 1.0), corner(0.0, 640.0, 0.0, 64.0, 1.0)];
        assert!(black.lit_screen_box(&good, &[0, 1, 2]).is_some());
        assert_eq!(black.lit_screen_box(&good, &[0, 1, 3]), None); // a corner that isn't there
        for bad in [
            corner(f32::NAN, 0.0, 0.0, 0.0, 1.0),
            corner(0.0, f32::INFINITY, 0.0, 0.0, 1.0),
            corner(0.0, 20000.0, 0.0, 0.0, 1.0),
            corner(0.0, 0.0, f32::NAN, 0.0, 1.0),
            corner(0.0, 0.0, 0.0, 0.0, 0.0),
            corner(0.0, 0.0, 0.0, 0.0, -1.0),
            corner(0.0, 0.0, 30.0, 30.0, 1e-42),
        ] {
            let vertices = [good[0], good[1], bad];
            assert_eq!(black.lit_screen_box(&vertices, &[0, 1, 2]), None, "{bad:?}");
        }
    }

    /// Random triangles with perspective, shaded as the GPU does at the
    /// centres of the 2x pixels: every pixel whose texture coordinate is
    /// out on no side lies in the box.
    #[test]
    fn the_lit_screen_box_holds_every_fragment_not_out_on_a_side() {
        let mut seed = 0x1234_5678u32;
        let mut random = |low: f32, high: f32| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            low + (high - low) * ((seed >> 8) as f32 / 16777216.0)
        };
        let (mut checked, mut empty) = (0u32, 0u32);
        for round in 0..400 {
            let left = random(0.0, 50.0) as i32;
            let top = random(0.0, 50.0) as i32;
            let lit = Rect::new(left, top, left + 1 + random(0.0, 12.0) as i32, top + 1 + random(0.0, 12.0) as i32);
            let black = BlackOutside { lit, width: 64, height: 64, clamp_u: round % 5 != 0, clamp_v: round % 7 != 0 };
            let vertices: [FrameVertex; 3] =
                core::array::from_fn(|_| corner(random(-80.0, 200.0), random(-80.0, 200.0), random(-150.0, 220.0), random(-150.0, 220.0), random(0.2, 6.0)));
            let Some(found) = black.lit_screen_box(&vertices, &[0, 1, 2]) else { panic!("no box for {vertices:?}") };
            empty += found.is_empty() as u32;
            let limits = black.limits();
            let [a, b, c] = vertices.map(|v| (v.x as f64, v.y as f64, v.u as f64, v.v as f64, v.w as f64));
            let area = (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0);
            if area == 0.0 {
                continue;
            }
            // The 2x pixels of 0..120 x 0..120 PSP pixels.
            for py in 0..240 {
                for px in 0..240 {
                    let (x, y) = ((px as f64 + 0.5) / 2.0, (py as f64 + 0.5) / 2.0);
                    let weight = |p: (f64, f64, f64, f64, f64), q: (f64, f64, f64, f64, f64)| ((q.0 - p.0) * (y - p.1) - (q.1 - p.1) * (x - p.0)) / area;
                    let weights = [weight(b, c), weight(c, a), weight(a, b)];
                    if weights.iter().any(|w| *w < 0.0) {
                        continue;
                    }
                    let over_w: f64 = weights[0] / a.4 + weights[1] / b.4 + weights[2] / c.4;
                    let u = (weights[0] * a.2 / a.4 + weights[1] * b.2 / b.4 + weights[2] * c.2 / c.4) / over_w;
                    let v = (weights[0] * a.3 / a.4 + weights[1] * b.3 / b.4 + weights[2] * c.3 / c.4) / over_w;
                    let out = limits[0].is_some_and(|l| u < l as f64)
                        || limits[1].is_some_and(|l| u > l as f64)
                        || limits[2].is_some_and(|l| v < l as f64)
                        || limits[3].is_some_and(|l| v > l as f64);
                    if out {
                        continue;
                    }
                    checked += 1;
                    let (x, y) = (px / 2, py / 2);
                    assert!(x >= found.left && x < found.right && y >= found.top && y < found.bottom, "round {round}: pixel {x},{y} not in {found:?}");
                }
            }
        }
        // (Enough of both kinds: boxes with fragments, and empty ones.)
        assert!(checked > 20_000 && empty > 20, "{checked} {empty}");
    }
}
