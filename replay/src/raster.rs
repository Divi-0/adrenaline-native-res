//! A small software renderer that draws a frame description the way the
//! Vita side does (Adrenaline/user/nativeres_render.c):
//! - PSP coordinates doubled onto a 960x544 screen cleared to black,
//! - textured draws (our GPU program): texture colour x vertex colour (x2
//!   with colour doubling, then clamped), then fog, both in the GE's integer
//!   arithmetic (`ge_shade`); texture coordinates
//!   interpolated with perspective (by the vertices' w), colours and fog
//!   linearly on the screen as the PSP shades (core `transform.rs`),
//! - untextured draws (vita2d's colour program): vertex colours interpolated
//!   linearly in screen space,
//! - the blend mode recorded per draw (`Draw::blend`, see core `blend.rs`),
//! - scissor rectangle, nearest or bilinear filtering, clamp or repeat,
//! - the depth test and depth writes (`Draw::tests`, the PSP's rules: a draw
//!   writes depth only with the depth test on) and the alpha test, which
//!   comes before them (a discarded pixel writes no depth).
//!
//! Pixels are sampled at their centres with a top-left fill rule, like GPUs
//! do, so triangles sharing an edge don't draw its pixels twice.

use nativeres_core::blend::{channel_mask, BLEND_ENABLED, BLEND_FIX_B_ALPHA, BLEND_FIX_B_SHIFT, BLEND_PREMULTIPLY_SHIFT};
use nativeres_core::frame::{draw_flags, test_flags, texture_flags, Draw, Frame, FrameVertex, TextureDescriptor};
use nativeres_core::texture::{data_size, decode_texture};

use crate::dump_file::{DumpMemory, OriginalPicture};

pub const WIDTH: usize = 960;
pub const HEIGHT: usize = 544;
const SCALE: f32 = 2.0;

/// A decoded texture, RGBA (0xAABBGGRR) row by row.
struct Texture {
    width: usize,
    height: usize,
    pixels: Vec<u32>,
}

/// RGBA as floats 0..1 (may go above 1 with colour doubling).
#[derive(Clone, Copy, Debug)]
struct Color {
    r: f32,
    g: f32,
    b: f32,
    a: f32,
}

impl Color {
    fn from_rgba(rgba: u32) -> Color {
        let channel = |shift: u32| ((rgba >> shift) & 0xFF) as f32 / 255.0;
        Color { r: channel(0), g: channel(8), b: channel(16), a: channel(24) }
    }

    fn to_rgba(self) -> u32 {
        let channel = |value: f32, shift: u32| ((value.clamp(0.0, 1.0) * 255.0 + 0.5) as u32) << shift;
        channel(self.r, 0) | channel(self.g, 8) | channel(self.b, 16) | channel(self.a, 24)
    }

    fn times(self, other: Color) -> Color {
        Color { r: self.r * other.r, g: self.g * other.g, b: self.b * other.b, a: self.a * other.a }
    }

    fn clamped(self) -> Color {
        let clamp = |value: f32| value.clamp(0.0, 1.0);
        Color { r: clamp(self.r), g: clamp(self.g), b: clamp(self.b), a: clamp(self.a) }
    }

    /// `b0 * c0 + b1 * c1 + b2 * c2`.
    fn weighted(colors: [Color; 3], weights: [f32; 3]) -> Color {
        let sum = |channel: fn(&Color) -> f32| (0..3).map(|i| weights[i] * channel(&colors[i])).sum();
        Color { r: sum(|c| c.r), g: sum(|c| c.g), b: sum(|c| c.b), a: sum(|c| c.a) }
    }

    fn mix(a: Color, b: Color, t: f32) -> Color {
        Color {
            r: a.r + (b.r - a.r) * t,
            g: a.g + (b.g - a.g) * t,
            b: a.b + (b.b - a.b) * t,
            a: a.a + (b.a - a.a) * t,
        }
    }
}

/// Renders `frame` and returns 960x544 RGBA pixels. `previous`: the
/// picture of the other framebuffer, if the dump has one (its original
/// picture, see `other_framebuffer`).
pub fn render_frame(frame: &Frame, memory: &mut DumpMemory, previous: Option<&OriginalPicture>) -> Vec<u32> {
    let textures: Vec<Option<Texture>> = frame
        .textures()
        .iter()
        .map(|texture| match previous {
            Some(picture) if other_framebuffer(texture, frame) => Some(framebuffer_texture(texture, picture)),
            _ => decode(texture, frame, memory),
        })
        .collect();
    for (index, texture) in textures.iter().enumerate() {
        if texture.is_none() {
            println!("texture {index}: could not be decoded from the dump");
        }
    }
    // Debugging: REPLAY_SHOW_TEXTURE=n describes texture n and some of its texels.
    if let Some(index) = std::env::var("REPLAY_SHOW_TEXTURE").ok().and_then(|value| value.parse::<usize>().ok()) {
        println!("texture {index}: {:?}", frame.textures().get(index));
        if let Some(Some(texture)) = textures.get(index) {
            let texels: Vec<String> = texture.pixels.iter().step_by(texture.pixels.len() / 8).map(|texel| format!("{texel:08x}")).collect();
            println!("  {}x{} texels {}", texture.width, texture.height, texels.join(" "));
            // REPLAY_TEXTURE_PNG=<path>: the whole decoded texture as a picture.
            if let Ok(path) = std::env::var("REPLAY_TEXTURE_PNG") {
                match crate::png::write_rgba(&path, texture.width, texture.height, &texture.pixels) {
                    Ok(()) => println!("  wrote {path}"),
                    Err(error) => println!("  {path}: {error}"),
                }
            }
        }
    }

    // Debugging: REPLAY_SHOW_DRAW=n prints draw n and its first vertices.
    if let Some(index) = std::env::var("REPLAY_SHOW_DRAW").ok().and_then(|value| value.parse::<usize>().ok()) {
        if let Some(draw) = frame.draws().get(index) {
            println!("draw {index}: {draw:x?}");
            println!("  texture: {:x?}", frame.textures().get(draw.texture as usize));
            for vertex in frame.screen_corners(draw).take(12) {
                println!("  {vertex:?}");
            }
        }
    }

    // The Vita screen is cleared to opaque black before drawing.
    let mut screen = vec![0xFF00_0000u32; WIDTH * HEIGHT];
    // Depth per pixel (0..1). Games clear it before use.
    let mut depth = vec![0.0f32; WIDTH * HEIGHT];
    // Debugging: REPLAY_MAX_DRAWS=n draws only the first n draws.
    let max_draws = std::env::var("REPLAY_MAX_DRAWS").ok().and_then(|value| value.parse().ok()).unwrap_or(usize::MAX);
    // Debugging: REPLAY_PIXEL=x,y (960x544 picture) prints every draw that
    // changes that pixel, with the colour it leaves.
    let watched = std::env::var("REPLAY_PIXEL").ok().and_then(|value| {
        let (x, y) = value.split_once(',')?;
        let (x, y) = (x.trim().parse::<usize>().ok()?, y.trim().parse::<usize>().ok()?);
        (x < WIDTH && y < HEIGHT).then_some(y * WIDTH + x)
    });
    for (index, draw) in frame.draws().iter().enumerate().take(max_draws) {
        let before = watched.map(|at| screen[at]);
        draw_one(&mut screen, &mut depth, frame, &textures, draw);
        if let (Some(at), Some(before)) = (watched, before) {
            if screen[at] != before {
                println!("pixel: draw {index} {before:08x} -> {:08x} texture {} {:x?}", screen[at], draw.texture, frame.textures().get(draw.texture as usize));
            }
        }
    }
    // Shown opaque, as NativeResPresentFrame does: the alpha is the game's
    // stencil, not meant for the screen.
    for pixel in &mut screen {
        *pixel |= 0xFF00_0000;
    }
    screen
}

/// Draws one of the frame's draws into `screen`.
fn draw_one(screen: &mut [u32], depth: &mut [f32], frame: &Frame, textures: &[Option<Texture>], draw: &Draw) {
    // Off-screen rendering isn't supported yet (same as on the Vita).
    if draw.framebuffer != frame.shown_framebuffer {
        return;
    }
    let texture = if draw.flags & draw_flags::TEXTURED != 0 {
        match textures.get(draw.texture as usize) {
            Some(Some(texture)) => Some(texture),
            _ => return, // the Vita skips draws whose texture failed
        }
    } else {
        None
    };
    let mut corners = frame.screen_corners(draw);
    while let (Some(a), Some(b), Some(c)) = (corners.next(), corners.next(), corners.next()) {
        draw_triangle(screen, depth, draw, texture, [a, b, c], frame.dither_matrix);
    }
}

/// A texture that is the PSP framebuffer the frame doesn't draw into: a
/// 32-bit VRAM texture at least the screen's size (MTX's motion blur samples
/// the previous frame this way). The Vita samples its own render target
/// for it, not VRAM; the dump's VRAM copy of it is empty (the plugin isn't
/// asked to copy our targets). The replay uses the dump's original picture
/// instead, Sony's last shown frame, as the Vita seeds a target that holds
/// nothing (SeedTargetFromSony).
fn other_framebuffer(texture: &TextureDescriptor, frame: &Frame) -> bool {
    let address = nativeres_core::memory::fold_vram_mirror(texture.address & 0x0FFF_FFFF);
    (nativeres_core::memory::VRAM_START..nativeres_core::memory::VRAM_END).contains(&address)
        && texture.format == 3
        && texture.width >= 480
        && texture.height >= 272
        && address != frame.shown_framebuffer
}

/// `picture` as `texture` (its size; outside the picture black). Its alpha
/// is the PSP's stencil, not Sony's picture's (always 255): 0, as MTX's
/// clear leaves it; 255 where the draw switches the texture alpha off
/// (`IGNORE_ALPHA`). So a blur that took its weight from the alpha (the
/// slot-8 bug) comes out without blur here and fails the regression.
fn framebuffer_texture(texture: &TextureDescriptor, picture: &OriginalPicture) -> Texture {
    let (width, height) = (texture.width as usize, texture.height as usize);
    let forced_alpha = if texture.flags & texture_flags::IGNORE_ALPHA != 0 { 0xFF00_0000 } else { 0 };
    let mut pixels = vec![0u32; width * height];
    for y in 0..picture.height.min(height) {
        for x in 0..picture.width.min(width) {
            pixels[y * width + x] = (picture.pixels[y * picture.width + x] & 0x00FF_FFFF) | forced_alpha;
        }
    }
    Texture { width, height, pixels }
}

fn decode(texture: &TextureDescriptor, frame: &Frame, memory: &DumpMemory) -> Option<Texture> {
    let data = memory.bytes(texture.address, data_size(texture)?)?;
    let start = texture.palette_offset as usize;
    let palette = frame.palette_words.get(start..start + texture.palette_word_count as usize)?;
    let (width, height) = (texture.width as usize, texture.height as usize);
    let mut pixels = vec![0u32; width * height];
    decode_texture(&data, texture, palette, &mut pixels, width).ok()?;
    Some(Texture { width, height, pixels })
}

/// Scissor rectangle in screen pixels: [left, right) x [top, bottom).
fn scissor(draw: &Draw) -> (i32, i32, i32, i32) {
    let left = draw.scissor_left as i32 * 2;
    let top = draw.scissor_top as i32 * 2;
    let right = ((draw.scissor_right as i32 + 1) * 2).min(WIDTH as i32);
    let bottom = ((draw.scissor_bottom as i32 + 1) * 2).min(HEIGHT as i32);
    (left, top, right, bottom)
}

/// The GE's compare functions (depth and alpha test): NEVER, ALWAYS, EQUAL,
/// NOTEQUAL, LESS, LEQUAL, GREATER, GEQUAL. True if `value` passes against
/// `reference`.
fn passes<T: PartialOrd>(function: u32, value: T, reference: T) -> bool {
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

fn draw_triangle(screen: &mut [u32], depth: &mut [f32], draw: &Draw, texture: Option<&Texture>, corners: [FrameVertex; 3], dither_matrix: u64) {
    let target_bits = nativeres_core::frame::target_channel_bits(draw.flags);
    let position = |vertex: &FrameVertex| (vertex.x * SCALE, vertex.y * SCALE);
    let [p0, p1, p2] = [position(&corners[0]), position(&corners[1]), position(&corners[2])];

    let area = edge(p0, p1, p2);
    if area == 0.0 {
        return;
    }
    // Back-face culling (`draw_flags::CULL`): CULL_CCW keeps area > 0.
    if draw.flags & draw_flags::CULL != 0 && (area > 0.0) != (draw.flags & draw_flags::CULL_CCW != 0) {
        return;
    }
    // Make the triangle counter-clockwise so all edge values are positive inside.
    let (corners, p1, p2, area) = if area < 0.0 {
        ([corners[0], corners[2], corners[1]], p2, p1, -area)
    } else {
        (corners, p1, p2, area)
    };

    let (scissor_left, scissor_top, scissor_right, scissor_bottom) = scissor(draw);
    let min_x = (p0.0.min(p1.0).min(p2.0).floor() as i32).max(scissor_left).max(0);
    let max_x = (p0.0.max(p1.0).max(p2.0).ceil() as i32).min(scissor_right);
    let min_y = (p0.1.min(p1.1).min(p2.1).floor() as i32).max(scissor_top).max(0);
    let max_y = (p0.1.max(p1.1).max(p2.1).ceil() as i32).min(scissor_bottom);

    let colors = corners.map(|corner| Color::from_rgba(corner.color));
    let color_scale = if draw.flags & draw_flags::DOUBLE_COLOR != 0 { 2.0 } else { 1.0 };
    let depth_test = draw.tests & test_flags::DEPTH_TEST != 0;
    let depth_function = draw.tests >> test_flags::DEPTH_FUNCTION_SHIFT;
    let depth_write = depth_test && draw.tests & test_flags::DEPTH_WRITE_DISABLED == 0;
    let alpha_test = draw.tests & test_flags::ALPHA_TEST != 0;
    let alpha_function = draw.tests >> test_flags::ALPHA_FUNCTION_SHIFT;
    let alpha_reference = (draw.tests >> test_flags::ALPHA_REFERENCE_SHIFT) & 0xFF;
    let alpha_mask = draw.tests >> test_flags::ALPHA_MASK_SHIFT;
    for y in min_y..max_y {
        for x in min_x..max_x {
            let center = (x as f32 + 0.5, y as f32 + 0.5);
            let w0 = edge(p1, p2, center);
            let w1 = edge(p2, p0, center);
            let w2 = edge(p0, p1, center);
            if !(inside(w0, p1, p2) && inside(w1, p2, p0) && inside(w2, p0, p1)) {
                continue;
            }
            // Screen-space weights of the three corners.
            let linear = [w0 / area, w1 / area, w2 / area];
            // Depth is z / w, which is linear on the screen.
            let index = y as usize * WIDTH + x as usize;
            let z = linear[0] * corners[0].z + linear[1] * corners[1].z + linear[2] * corners[2].z;
            if depth_test && !passes(depth_function, z, depth[index]) {
                continue;
            }

            let source = match texture {
                Some(texture) => {
                    // Perspective: weights in proportion to 1 / w.
                    let over_w = [0, 1, 2].map(|i| linear[i] / corners[i].w);
                    let total: f32 = over_w.iter().sum();
                    let b = over_w.map(|weight| weight / total);
                    let u = b[0] * corners[0].u + b[1] * corners[1].u + b[2] * corners[2].u;
                    let v = b[0] * corners[0].v + b[1] * corners[1].v + b[2] * corners[2].v;
                    // Colour and fog: linear on the screen, as the PSP shades.
                    ge_shade(sample(texture, draw, u, v), Color::weighted(colors, linear), fog_at(linear, &corners), draw, color_scale)
                }
                None => Color::weighted(colors, linear),
            };

            if alpha_test {
                let alpha = (source.a.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
                if !passes(alpha_function, alpha & alpha_mask, alpha_reference & alpha_mask) {
                    continue;
                }
            }
            if depth_write {
                depth[index] = z;
            }
            let pixel = &mut screen[index];
            *pixel = blend(draw.blend, source, Color::from_rgba(*pixel)).to_rgba();
            if let Some(bits) = target_bits {
                // A 16-bit PSP target: cut as the GE does, after blending
                // (the Vita's fragment programs do the same).
                let dither = if draw.flags & draw_flags::DITHER != 0 {
                    nativeres_core::frame::dither_value(dither_matrix, (index % WIDTH) / 2, (index / WIDTH) / 2)
                } else {
                    0
                };
                let p = *pixel;
                let channel = |shift: u32, bits: u32, dither: i32| {
                    nativeres_core::frame::quantize_like_the_ge(((p >> shift) & 0xFF) as i32, dither, bits) << shift
                };
                let alpha = if bits[3] == 0 { p & 0xFF00_0000 } else { channel(24, bits[3], 0) };
                *pixel = alpha | channel(16, bits[2], dither) | channel(8, bits[1], dither) | channel(0, bits[0], dither);
            }
        }
    }
}

/// Twice the signed area of (a, b, c); positive if c is left of a->b.
/// The fog factor at a pixel (1 without fog): linear on the screen.
fn fog_at(linear: [f32; 3], corners: &[FrameVertex; 3]) -> f32 {
    linear[0] * corners[0].fog + linear[1] * corners[1].fog + linear[2] * corners[2].fog
}

/// The GE's texture function and fog in its integer arithmetic, as the
/// Vita's fragment programs do it (`GeShade` in nativeres_textured_f.cg;
/// ppsspp-ref GPU/Software/Sampler.cpp GetTextureFunctionOutput, DrawPixel.cpp
/// fog): texel and vertex colour truncated to 8 bits, (p + 1) t >> 8 (x 2
/// with colour doubling), fog factor floor(f x 256), (c f + fog (255 - f) +
/// 255) >> 8. Rounding in float instead left MHF about one level brighter.
fn ge_shade(texel: Color, vertex: Color, fog: f32, draw: &Draw, color_scale: f32) -> Color {
    let to8 = |x: f32| (x.clamp(0.0, 1.0) * 255.0 + 0.01) as i32;
    let scale = color_scale as i32;
    let modulate = |p: f32, t: f32, s: i32| (((to8(p) + 1) * to8(t) * s) >> 8).min(255);
    let mut c = [
        modulate(vertex.r, texel.r, scale),
        modulate(vertex.g, texel.g, scale),
        modulate(vertex.b, texel.b, scale),
        modulate(vertex.a, texel.a, 1),
    ];
    if draw.flags & draw_flags::FOG != 0 {
        let f8 = ((fog.clamp(0.0, 1.0) * 256.0 + 0.01).floor() as i32).min(255);
        for (i, channel) in c.iter_mut().take(3).enumerate() {
            let fog_color = ((draw.fog_color >> (8 * i)) & 0xFF) as i32;
            *channel = (*channel * f8 + fog_color * (255 - f8) + 255) >> 8;
        }
    }
    Color { r: c[0] as f32 / 255.0, g: c[1] as f32 / 255.0, b: c[2] as f32 / 255.0, a: c[3] as f32 / 255.0 }
}

fn edge(a: (f32, f32), b: (f32, f32), c: (f32, f32)) -> f32 {
    (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
}

/// Top-left rule: pixels exactly on an edge belong to the triangle only for
/// top edges (horizontal, going right... in screen space with y down) and
/// left edges.
fn inside(w: f32, a: (f32, f32), b: (f32, f32)) -> bool {
    if w != 0.0 {
        return w > 0.0;
    }
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    (dy == 0.0 && dx < 0.0) || dy > 0.0
}

/// GPU blending as the Vita does it: `equation(source * source_factor,
/// destination * destination_factor)`, or the source as is when off; then
/// the colour mask (core `blend::channel_mask`: the alpha, the PSP's
/// stencil, is kept unless the draw writes it).
fn blend(packed: u32, source: Color, destination: Color) -> Color {
    let clamp = |value: f32| value.clamp(0.0, 1.0);
    let source = Color { r: clamp(source.r), g: clamp(source.g), b: clamp(source.b), a: clamp(source.a) };
    let mask = channel_mask(packed);
    if !mask.rgb {
        return Color { a: source.a, ..destination }; // the stencil alone, as is
    }
    let blended = blend_channels(packed, source, destination);
    if mask.alpha {
        blended
    } else {
        Color { a: destination.a, ..blended }
    }
}

fn blend_channels(packed: u32, source: Color, destination: Color) -> Color {
    let clamp = |value: f32| value.clamp(0.0, 1.0);
    if packed & BLEND_ENABLED == 0 {
        return source;
    }
    // FIX B through the alpha (core blend::Blend::fix_b), as the fragment
    // program does it: the source factor applied, FIX B as the alpha.
    let source = if packed & BLEND_FIX_B_ALPHA != 0 {
        let multiplier = match (packed >> BLEND_PREMULTIPLY_SHIFT) & 3 {
            1 => source.a,
            2 => 1.0 - source.a,
            _ => 1.0,
        };
        let fix_b = ((packed >> BLEND_FIX_B_SHIFT) & 0xFF) as f32 / 255.0;
        Color { r: source.r * multiplier, g: source.g * multiplier, b: source.b * multiplier, a: fix_b }
    } else {
        source
    };
    let source_factor = factor(packed & 0xF, source, destination);
    let destination_factor = factor((packed >> 4) & 0xF, source, destination);
    let (s, d) = (source.times(source_factor), destination.times(destination_factor));
    let combine = |s: f32, d: f32| match (packed >> 8) & 7 {
        1 => s - d,     // subtract
        2 => d - s,     // reverse subtract
        3 => s.min(d),  // min
        4 => s.max(d),  // max
        _ => s + d,     // add
    };
    Color {
        r: clamp(combine(s.r, d.r)),
        g: clamp(combine(s.g, d.g)),
        b: clamp(combine(s.b, d.b)),
        a: clamp(combine(s.a, d.a)),
    }
}

/// A blend factor (core `blend::Factor` numbering) as per-channel multipliers.
fn factor(code: u32, source: Color, destination: Color) -> Color {
    let all = |value: f32| Color { r: value, g: value, b: value, a: value };
    let inverse = |c: Color| Color { r: 1.0 - c.r, g: 1.0 - c.g, b: 1.0 - c.b, a: 1.0 - c.a };
    match code {
        0 => all(0.0),
        1 => all(1.0),
        2 => source,
        3 => inverse(source),
        4 => all(source.a),
        5 => all(1.0 - source.a),
        6 => destination,
        7 => inverse(destination),
        8 => all(destination.a),
        _ => all(1.0 - destination.a),
    }
}

fn sample(texture: &Texture, draw: &Draw, u: f32, v: f32) -> Color {
    let clamp_u = draw.flags & draw_flags::CLAMP_U != 0;
    let clamp_v = draw.flags & draw_flags::CLAMP_V != 0;
    let texel = |x: i64, y: i64| -> Color {
        let x = wrap(x, texture.width, clamp_u);
        let y = wrap(y, texture.height, clamp_v);
        Color::from_rgba(texture.pixels[y * texture.width + x])
    };

    let x = u * texture.width as f32;
    let y = v * texture.height as f32;
    if draw.flags & draw_flags::LINEAR_FILTER == 0 {
        return texel(x.floor() as i64, y.floor() as i64);
    }
    // Bilinear: blend the four texels around the sample point.
    let (x, y) = (x - 0.5, y - 0.5);
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let (x0, y0) = (x0 as i64, y0 as i64);
    let top = Color::mix(texel(x0, y0), texel(x0 + 1, y0), fx);
    let bottom = Color::mix(texel(x0, y0 + 1), texel(x0 + 1, y0 + 1), fx);
    Color::mix(top, bottom, fy)
}

fn wrap(coordinate: i64, size: usize, clamp: bool) -> usize {
    let size = size as i64;
    if clamp {
        coordinate.clamp(0, size - 1) as usize
    } else {
        coordinate.rem_euclid(size) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nativeres_core::blend::{Blend, BLEND_ALPHA_ONLY, BLEND_WRITES_ALPHA};

    const SOURCE: Color = Color { r: 1.0, g: 0.0, b: 0.0, a: 0.5 };

    /// MHF (2026-11-18): rounding the texture function and fog in float left
    /// the pictures about one level brighter than the PSP's. The GE
    /// truncates: (150 + 1) x 200 >> 8 = 117 (150 x 200 / 255 = 117.6 would
    /// round to 118); fog factor 0.5 -> 128, (117 x 128 + 0 x 127 + 255) >> 8
    /// = 59; without fog the colour stays.
    #[test]
    fn the_texture_function_and_fog_truncate_like_the_ge() {
        let byte = |value: u32| Color::from_rgba(value * 0x0101_0101);
        let draw = Draw { flags: draw_flags::TEXTURED, ..Draw::default() };
        let plain = ge_shade(byte(200), byte(150), 1.0, &draw, 1.0).to_rgba();
        assert_eq!(plain & 0xFF, 117);
        let fogged = Draw { flags: draw_flags::TEXTURED | draw_flags::FOG, fog_color: 0, ..Draw::default() };
        assert_eq!(ge_shade(byte(200), byte(150), 0.5, &fogged, 1.0).to_rgba() & 0xFF, 59);
        assert_eq!(ge_shade(byte(200), byte(150), 1.0, &fogged, 1.0).to_rgba() & 0xFF, 117);
        let doubled = ge_shade(byte(200), byte(100), 1.0, &draw, 2.0).to_rgba();
        assert_eq!(doubled & 0xFF, 157); // (101 x 200 x 2) >> 8
    }

    /// Monster Hunter Freedom's village ground (2026-11-18, dump
    /// logs/mhf-outside/out.bin): light and shadow edges are vertex colours of
    /// large triangles seen at an angle. The PSP shades linearly on the
    /// screen; interpolated with perspective (by 1 / w) they came out up to 4
    /// levels brighter. A triangle with a near (w 1) black and two far (w 8)
    /// white corners, fogged the same way: half way across the screen the
    /// colour and fog are half way too, not pulled towards the near corner.
    #[test]
    fn colours_and_fog_are_linear_on_the_screen_with_perspective() {
        let corner = |x: f32, y: f32, w: f32, color: u32, fog: f32| FrameVertex { x, y, z: 0.5, w, color, fog, ..FrameVertex::default() };
        let corners = [
            corner(0.0, 0.0, 1.0, 0xFF00_0000, 0.0),
            corner(100.0, 0.0, 8.0, 0xFFFF_FFFF, 1.0),
            corner(0.0, 100.0, 8.0, 0xFFFF_FFFF, 1.0),
        ];
        let white = Texture { width: 1, height: 1, pixels: vec![0xFFFF_FFFF] };
        let mut screen = vec![0u32; WIDTH * HEIGHT];
        let mut depth = vec![1.0f32; WIDTH * HEIGHT];
        // Unfogged: the screen-linear colour, 0.505 white (with perspective: 0.11).
        let whole = Draw { scissor_right: 479, scissor_bottom: 271, ..Draw::default() };
        let draw = Draw { flags: draw_flags::TEXTURED, ..whole };
        draw_triangle(&mut screen, &mut depth, &draw, Some(&white), corners, 0);
        let pixel = |screen: &[u32]| screen[50 * WIDTH + 50] & 0xFF; // PSP (25, 25), weights 1/2, 1/4, 1/4
        assert!((pixel(&screen) as i32 - 129).abs() <= 1, "colour {}", pixel(&screen));
        // Black geometry, white fog: the fog factor alone gives the colour.
        let fogged = corners.map(|corner| FrameVertex { color: 0xFF00_0000, ..corner });
        let draw = Draw { flags: draw_flags::TEXTURED | draw_flags::FOG, fog_color: 0x00FF_FFFF, ..whole };
        screen.fill(0);
        draw_triangle(&mut screen, &mut depth, &draw, Some(&white), fogged, 0);
        // Fog factor about 1/2 half way: about half the fog colour (with
        // perspective: 0.89, near white).
        assert!((pixel(&screen) as i32 - 126).abs() <= 1, "fogged {}", pixel(&screen));
    }
    const TARGET: Color = Color { r: 0.0, g: 0.0, b: 1.0, a: 0.0 };

    /// The target's alpha is the PSP's stencil: blended or not, a draw
    /// without BLEND_WRITES_ALPHA leaves it (MTX's blur weights the sampled
    /// framebuffer by it); stencil clears write only it.
    #[test]
    fn the_alpha_is_kept_unless_the_draw_writes_the_stencil() {
        let unblended = blend(0, SOURCE, TARGET);
        assert_eq!((unblended.r, unblended.b, unblended.a), (1.0, 0.0, 0.0));
        let blended = blend(Blend::ALPHA.pack(), SOURCE, TARGET);
        assert_eq!((blended.r, blended.b, blended.a), (0.5, 0.5, 0.0));
        let written = blend(BLEND_WRITES_ALPHA, SOURCE, TARGET);
        assert_eq!((written.r, written.a), (1.0, 0.5));
        let stencil_only = blend(Blend::KEEP_COLOR.pack() | BLEND_WRITES_ALPHA | BLEND_ALPHA_ONLY, SOURCE, TARGET);
        assert_eq!((stencil_only.r, stencil_only.b, stencil_only.a), (0.0, 1.0, 0.5));
    }
}
