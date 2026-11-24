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

use nativeres_core::blend::{channel_mask, reads_destination_alpha, samples_target_alpha, StencilState, BLEND_ENABLED, BLEND_FIX_B_ALPHA, BLEND_FIX_B_SHIFT, BLEND_PREMULTIPLY_SHIFT};
use nativeres_core::frame::{draw_flags, ge_blend_term, test_flags, texture_flags, wide_variant, Draw, Frame, FrameVertex, TextureDescriptor};
use nativeres_core::hidden_draws::{self, HiddenDraws};
use nativeres_core::offscreen;
use nativeres_core::texture::{data_size, decode_texture, format, ALL_ROWS_OPAQUE};

use crate::dump_file::{DumpMemory, OriginalPicture};

pub const WIDTH: usize = 960;
pub const HEIGHT: usize = 544;
const SCALE: f32 = 2.0;

thread_local! {
    // Debugging: REPLAY_DEPTH_MODE=n draws with another depth model (`psp_depth`).
    static DEPTH_MODE: u8 = std::env::var("REPLAY_DEPTH_MODE").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    // Debugging: REPLAY_TIE_STATS (`draw_triangle`).
    static TIE_STATS: bool = std::env::var_os("REPLAY_TIE_STATS").is_some();
    static DEPTH_CENTER: bool = std::env::var_os("REPLAY_DEPTH_CENTER").is_some();
    static TIE_COUNTS: std::cell::RefCell<std::collections::BTreeMap<usize, u32>> = std::cell::RefCell::new(std::collections::BTreeMap::new());
    static CURRENT_DRAW: std::cell::Cell<usize> = std::cell::Cell::new(0);
    static TESTED: std::cell::Cell<u64> = std::cell::Cell::new(0);
    static FLOAT_DEPTH: std::cell::RefCell<std::collections::HashMap<usize, Vec<f32>>> = std::cell::RefCell::new(std::collections::HashMap::new());
}

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
    nativeres_core::ge_math::GE_EXACT_DEPTH.store(
        match DEPTH_MODE.with(|mode| *mode) {
            8 => 1,
            9 => 2,
            10 => 3,
            _ => 0,
        },
        std::sync::atomic::Ordering::Relaxed,
    );
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
            let t = &frame.textures()[index];
            println!(
                "texture {index}: could not be decoded from the dump (addr={:08x} fmt={} {}x{} bufw={} swz={} bytes={:?} in dump={})",
                t.address,
                t.format,
                t.width,
                t.height,
                t.buffer_width,
                t.swizzled,
                data_size(t),
                data_size(t).is_some_and(|size| memory.bytes(t.address, size).is_some())
            );
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

    // Debugging: REPLAY_SHOW_DRAW=n prints draw n and its first corners.
    if let Some(index) = std::env::var("REPLAY_SHOW_DRAW").ok().and_then(|value| value.parse::<usize>().ok()) {
        if let Some(draw) = frame.draws().get(index) {
            println!("draw {index}: {draw:x?}");
            println!("  texture: {:x?}", frame.textures().get(draw.texture as usize));
            // REPLAY_SHOW_CORNERS=n: the first n corners (12 by default).
            let corners = std::env::var("REPLAY_SHOW_CORNERS").ok().and_then(|value| value.parse().ok()).unwrap_or(12);
            for vertex in frame.screen_corners(draw).take(corners) {
                println!("  {vertex:?}");
            }
        }
    }

    // The Vita screen is cleared to opaque black before drawing.
    let mut screen = vec![0xFF00_0000u32; WIDTH * HEIGHT];
    // Depth per pixel (0..1), far (1) at first as every GXM scene on the
    // Vita starts (the depth surface's background depth): God of War draws
    // its sky before it resets the depth, against the PSP's depth of the
    // frame before; at 0 the sky failed its "<=" test here (2026-11-22).
    let mut depth = vec![1.0f32; WIDTH * HEIGHT];
    // Debugging: REPLAY_MAX_DRAWS=n draws only the first n draws.
    let max_draws = std::env::var("REPLAY_MAX_DRAWS").ok().and_then(|value| value.parse().ok()).unwrap_or(usize::MAX);
    // Debugging: REPLAY_PIXEL=x,y (960x544 picture) prints every draw that
    // changes that pixel, with the colour it leaves.
    let watched = std::env::var("REPLAY_PIXEL").ok().and_then(|value| {
        let (x, y) = value.split_once(',')?;
        let (x, y) = (x.trim().parse::<usize>().ok()?, y.trim().parse::<usize>().ok()?);
        (x < WIDTH && y < HEIGHT).then_some(y * WIDTH + x)
    });
    // Off-screen framebuffers the frame samples (core offscreen.rs): a
    // picture of their own, cleared to opaque black like a new target on the
    // Vita; their draws go there in the frame's order.
    let offscreen_framebuffers = offscreen::sampled(frame.draws(), frame.textures(), frame.shown_framebuffer);
    let mut offscreen: Vec<(u32, Vec<u32>, Vec<f32>)> = offscreen_framebuffers
        .iter()
        .filter(|&&framebuffer| framebuffer != 0)
        .map(|&framebuffer| (framebuffer, vec![0xFF00_0000u32; WIDTH * HEIGHT], vec![1.0f32; WIDTH * HEIGHT]))
        .collect();
    // Normal and additive blends into 8-bit targets blend as the GE does
    // while the game reads its targets' alpha, as the Vita's default
    // (draw.rs `BLEND_8888` 1; its display notes the reads, frame_loop
    // `target_alpha_read`: a target sampled with its alpha, a blend with
    // the destination alpha, a stencil test). REPLAY_BLEND8888=0: never,
    // 2: in every frame (the Vita's `blend8888 0|2`).
    let reads_target_alpha = frame.draws().iter().any(|draw| {
        (draw.blend & BLEND_ENABLED != 0 && reads_destination_alpha(draw.blend))
            || frame.stencil_state_of(draw).is_some_and(|state| state.test().is_some())
    }) || frame.textures().iter().any(|texture| {
        samples_target_alpha(texture.flags)
            && texture.format != format::BGR5650
            && (texture.address == frame.shown_framebuffer
                || other_framebuffer(texture, frame)
                || offscreen_framebuffers.iter().any(|&framebuffer| framebuffer != 0 && offscreen::reads_framebuffer(texture, framebuffer)))
    });
    let blend_mode = std::env::var("REPLAY_BLEND8888").ok().and_then(|value| value.parse::<u8>().ok()).unwrap_or(1);
    GE_BLEND_FRAME.store(blend_mode == 2 || (blend_mode == 1 && reads_target_alpha), std::sync::atomic::Ordering::Relaxed);
    // The screen read only by draws into other framebuffers (God of War's
    // bloom chain): they sample the screen as drawn so far, as the Vita
    // samples its target of it (frame_loop `screen_from_target`).
    let screen_from_target = offscreen::screen_read_only_by_other_runs(frame.draws(), frame.textures(), frame.shown_framebuffer);
    // Draws a later draw paints over completely (core hidden_draws) are
    // left out, as on the Vita; the texels' opacity from our own decoding
    // (textures reading a framebuffer: unknown). Depth-only draws (and
    // those that change no colour) leave the colour. REPLAY_NO_HIDE=1 draws
    // them all in full, for comparing.
    let mut hidden = Box::new(HiddenDraws::new());
    let hiding = std::env::var_os("REPLAY_NO_HIDE").is_none();
    if hiding {
        let rows = |index: usize| -> u16 {
            let descriptor = &frame.textures()[index];
            let reads_framebuffer = other_framebuffer(descriptor, frame)
                || frame.draws().iter().any(|draw| offscreen::reads_framebuffer(descriptor, draw.framebuffer));
            match textures.get(index) {
                Some(Some(texture)) if !reads_framebuffer => opaque_rows(texture),
                _ => 0,
            }
        };
        hidden_draws::find(frame, &rows, &mut hidden);
        if hidden.count() > 0 {
            let list: Vec<usize> = (0..frame.draws().len()).filter(|&index| hidden.is_hidden(index)).collect();
            println!("hidden draws: {} (painted over by a later draw): {list:?}", hidden.count());
        }
        if hidden.depth_only_count() > 0 {
            let list: Vec<usize> = (0..frame.draws().len()).filter(|&index| hidden.is_depth_only(index)).collect();
            println!("depth-only draws: {}: {list:?}", hidden.depth_only_count());
        }
    }
    for (index, draw) in frame.draws().iter().enumerate().take(max_draws) {
        // Debugging: REPLAY_STENCIL_LIST=1 lists the draws with an exact
        // stencil state and the way the Vita does it (core `stencil_way`).
        if std::env::var_os("REPLAY_STENCIL_LIST").is_some() {
            if let Some(state) = frame.stencil_state_of(draw) {
                let textured = draw.flags & draw_flags::TEXTURED != 0;
                println!(
                    "stencil draw {index}: fb {:08x} blend {:08x} textured {textured} hidden {} write {:?} test {:?} way {:?} verts {}",
                    draw.framebuffer, draw.blend, hidden.is_hidden(index), state.write(), state.test(),
                    nativeres_core::blend::stencil_way(Some(state), draw.blend, textured), draw.vertex_count
                );
            }
        }
        CURRENT_DRAW.with(|d| d.set(index));
        // Lab: REPLAY_RAW_VARIANTS=1 tallies the raw vertex programs the
        // Vita would take for the lit draws (vita draw.rs's choice).
        if std::env::var_os("REPLAY_RAW_VARIANTS").is_some() && draw.flags & draw_flags::RAW_VERTICES != 0 && draw.flags & draw_flags::RAW_SPRITES == 0 {
            let colors = match draw.raw_colors {
                0 => None,
                n if n <= frame.raw_colors_count => frame.raw_colors.get(n as usize - 1),
                _ => None,
            };
            let lit = !colors.is_some_and(|colors| colors.rules[0] < 2.5);
            if lit {
                let weights = if (draw.vertex_format >> 9) & 3 != 0 { ((draw.vertex_format >> 14) & 7) + 1 } else { 0 };
                let lights = frame.raw_transform_of(draw).and_then(|(transform, _)| {
                    let set = transform.light_set as u32;
                    (set != 0 && set <= frame.light_set_count).then(|| frame.light_sets[set as usize - 1])
                });
                let point = lights.is_some_and(|l| l.to_light.iter().any(|light| light[3] > 0.5));
                let flat = lights.is_some_and(|l| {
                    let zeros = |values: &[f32; 4]| values.iter().all(|value| value.to_bits() == 0);
                    l.emissive[3] <= 0.0 && zeros(&l.to_light[3]) && zeros(&l.diffuse[3]) && zeros(&l.specular[3])
                });
                let bones = match weights { 0..=1 => "lit1", 2..=4 => "lit4", _ => "full" };
                let kind = if point { "point" } else if flat { "flat" } else { "" };
                let color = if point || bones == "full" { "any".to_string() } else {
                    match (draw.vertex_format >> 2) & 7 { 4 => "5650", 5 => "5551", 6 => "4444", 7 => "8888", _ => "none" }.to_string()
                };
                println!("rawvariant {bones}{kind} {color}");
            }
        }
        // Debugging: REPLAY_SKIP=a,b,... leaves those draws out.
        if std::env::var("REPLAY_SKIP").is_ok_and(|v| v.split(',').any(|d| d.trim().parse::<usize>() == Ok(index))) {
            continue;
        }
        if hidden.is_hidden(index) {
            continue;
        }
        let before = watched.map(|at| screen[at]);
        let depth_before = watched.map(|at| depth[at]);
        // A texture reading an off-screen picture: that picture as drawn so far.
        let from_target = (draw.flags & draw_flags::TEXTURED != 0)
            .then(|| frame.textures().get(draw.texture as usize))
            .flatten()
            .and_then(|texture| {
                if screen_from_target && texture.address == frame.shown_framebuffer {
                    return Some(target_texture(texture, &screen));
                }
                let (_, picture, _) = offscreen.iter().find(|(framebuffer, _, _)| offscreen::reads_framebuffer(texture, *framebuffer))?;
                Some(target_texture(texture, picture))
            });
        // Measuring: REPLAY_NATIVE=1 shades every draw at the PSP's own
        // resolution (each pixel at the centre of its PSP pixel: a 1x
        // picture shown doubled), 2 only the draws into off-screen targets
        // (the Vita's off-screen targets are 1:1, targets.rs), 3 only those
        // of them that read a target (God of War's bloom chain). Against
        // the game's picture this separates resolution from arithmetic
        // (2026-11-23: GoW's bloom at 2x blurred too little).
        // Draws reading our targets filter as the GE does (the Vita's
        // GE_FILTER programs, core `ge_filter_read`); the rest as the Vita's
        // texture unit. Measuring: REPLAY_PSP_FILTER=1 the GE's filter for
        // every draw (2026-11-23: other games' pictures moved further from
        // the PSP's), 0 the GPU's for every draw (before 2026-11-23).
        let stencil = if reads_target_alpha {
            nativeres_core::blend::stencil_way(frame.stencil_state_of(draw), draw.blend, draw.flags & draw_flags::TEXTURED != 0)
        } else {
            nativeres_core::blend::StencilWay::None
        };
        // Measuring: REPLAY_GE_FILTER_PLAIN=1 the GE's filter also for every
        // unblended draw without alpha test (the Vita's plain textured
        // programs; blended draws go through the frame-read programs in GoW).
        let plain_ge = std::env::var_os("REPLAY_GE_FILTER_PLAIN").is_some()
            && draw.flags & draw_flags::TEXTURED != 0
            && draw.flags & draw_flags::LINEAR_FILTER != 0
            && draw.blend & BLEND_ENABLED == 0
            && draw.tests & test_flags::ALPHA_TEST == 0;
        let ge_filter = plain_ge || match psp_filter_mode() {
            Some(all) => all,
            None => nativeres_core::frame::ge_filter_read(
                draw.flags,
                draw.tests,
                draw.blend,
                stencil,
                reads_target_alpha,
                from_target.is_some(),
                // Measuring: REPLAY_GE_FILTER_SCREEN=1 also for draws onto
                // the screen (the Vita's `gefilter 2`).
                draw.framebuffer != frame.shown_framebuffer || std::env::var_os("REPLAY_GE_FILTER_SCREEN").is_some(),
            ),
        };
        GE_FILTER_DRAW.store(ge_filter, std::sync::atomic::Ordering::Relaxed);
        // Debugging: REPLAY_LIST_TARGET_READS=1 lists the draws sampling one
        // of our targets (the GE-filtered ones).
        if from_target.is_some() && std::env::var_os("REPLAY_LIST_TARGET_READS").is_some() {
            let texture = &frame.textures()[draw.texture as usize];
            println!(
                "target read {index}: fb {:08x} tex {:08x} {}x{} fmt {} texflags {:08x} flags {:08x} blend {:08x} tests {:08x} verts {} linear {}",
                draw.framebuffer, texture.address, texture.width, texture.height, texture.format, texture.flags,
                draw.flags, draw.blend, draw.tests, draw.vertex_count, draw.flags & draw_flags::LINEAR_FILTER != 0
            );
        }
        NATIVE_DRAW.store(
            match native_mode() {
                1 => true,
                2 => draw.framebuffer != frame.shown_framebuffer,
                3 => draw.framebuffer != frame.shown_framebuffer && from_target.is_some(),
                _ => false,
            },
            std::sync::atomic::Ordering::Relaxed,
        );
        let mut drawn_textures;
        let draw_textures: &[Option<Texture>] = match from_target {
            Some(texture) => {
                drawn_textures = Vec::with_capacity(textures.len());
                drawn_textures.extend(textures.iter().map(|_| None));
                drawn_textures[draw.texture as usize] = Some(texture);
                &drawn_textures
            }
            None => &textures,
        };
        let colourless = hiding && draw.framebuffer == frame.shown_framebuffer && hidden_draws::colourless_depth_only(draw);
        if hidden.is_depth_only(index) || colourless {
            // Its depths alone, like the Vita's draw without a fragment
            // program (core hidden_draws: it keeps every fragment).
            let colour = screen.clone();
            draw_one(&mut screen, &mut depth, frame, draw_textures, draw);
            screen.copy_from_slice(&colour);
        } else if draw.framebuffer == frame.shown_framebuffer {
            draw_one(&mut screen, &mut depth, frame, draw_textures, draw);
        } else if let Some((_, picture, picture_depth)) = offscreen.iter_mut().find(|(framebuffer, _, _)| *framebuffer == draw.framebuffer) {
            draw_one(picture, picture_depth, frame, draw_textures, draw);
        }
        if let (Some(at), Some(before)) = (watched, before) {
            if screen[at] != before || Some(depth[at]) != depth_before {
                println!("pixel: draw {index} {before:08x} -> {:08x} depth {:?} -> {} tests {:x} texture {} {:x?}", screen[at], depth_before, depth[at], draw.tests, draw.texture, frame.textures().get(draw.texture as usize));
            }
        }
    }
    if TIE_STATS.with(|on| *on) {
        TIE_COUNTS.with(|counts| {
            let counts = counts.borrow();
            let total: u32 = counts.values().sum();
            println!("tie stats: {} of {} draws, {total} fragments decided by whole units, {} depth-tested fragments (960x544)", counts.len(), frame.draws().len(), TESTED.with(|t| t.get()));
            for (draw, count) in counts.iter() {
                println!("  tie draw {draw}: {count}");
            }
        });
    }
    // Debugging: REPLAY_SAVE_DEPTH=<path> saves the screen's depth (960x544
    // little-endian f32, 0..1 = PSP depth / 65535; logs/lab/gow-depth-cmp.py
    // compares it with Sony's depth buffer read from the Vita).
    if let Ok(path) = std::env::var("REPLAY_SAVE_DEPTH") {
        let bytes: Vec<u8> = depth.iter().flat_map(|value| value.to_le_bytes()).collect();
        let _ = std::fs::write(path, bytes);
    }
    // Debugging: REPLAY_SAVE_OFFSCREEN=<hex address>:<png> saves that
    // off-screen picture (as the Vita's debug `showtarget` shows its target).
    if let Some((address, path)) = std::env::var("REPLAY_SAVE_OFFSCREEN").ok().and_then(|value| {
        let (address, path) = value.split_once(':')?;
        Some((u32::from_str_radix(address, 16).ok()?, path.to_string()))
    }) {
        if let Some((_, picture, _)) = offscreen.iter().find(|(framebuffer, _, _)| *framebuffer == address) {
            let opaque: Vec<u32> = picture.iter().map(|pixel| pixel | 0xFF00_0000).collect();
            let _ = crate::png::write_rgba(&path, WIDTH, HEIGHT, &opaque);
        }
    }
    // Shown opaque, as NativeResPresentFrame does: the alpha is the game's
    // stencil, not meant for the screen. Debugging: REPLAY_KEEP_ALPHA=1
    // keeps it (the stencil in the PNG's alpha).
    if std::env::var_os("REPLAY_KEEP_ALPHA").is_none() {
        for pixel in &mut screen {
            *pixel |= 0xFF00_0000;
        }
    }
    screen
}

/// How many rows from the top of `texture` have only texels of alpha 255
/// (`ALL_ROWS_OPAQUE`: all), as core `texture::opaque_rows` says from the
/// source.
fn opaque_rows(texture: &Texture) -> u16 {
    match texture.pixels.chunks(texture.width.max(1)).position(|row| row.iter().any(|texel| texel >> 24 != 0xFF)) {
        Some(row) => row.min(u16::MAX as usize - 1) as u16,
        None => ALL_ROWS_OPAQUE,
    }
}

/// `texture` read from an off-screen picture (960x544, PSP pixels doubled):
/// its texel (x, y) is the picture's pixel (2x, 2y); outside the PSP's
/// 480x272 black. A 16-bit 565 framebuffer has no alpha: read opaque, as
/// with the texture's alpha ignored.
fn target_texture(texture: &TextureDescriptor, picture: &[u32]) -> Texture {
    let (width, height) = (texture.width as usize, texture.height as usize);
    let opaque = texture.format == format::BGR5650 || texture.flags & texture_flags::IGNORE_ALPHA != 0;
    let mut pixels = vec![0u32; width * height];
    for y in 0..height.min(HEIGHT / 2) {
        for x in 0..width.min(WIDTH / 2) {
            let pixel = picture[2 * y * WIDTH + 2 * x];
            pixels[y * width + x] = if opaque { pixel | 0xFF00_0000 } else { pixel };
        }
    }
    Texture { width, height, pixels }
}

/// Draws one of the frame's draws into `screen` (the shown framebuffer's
/// picture or an off-screen one).
fn draw_one(screen: &mut [u32], depth: &mut [f32], frame: &Frame, textures: &[Option<Texture>], draw: &Draw) {
    let texture = if draw.flags & draw_flags::TEXTURED != 0 {
        match textures.get(draw.texture as usize) {
            Some(Some(texture)) => Some(texture),
            _ => return, // the Vita skips draws whose texture failed
        }
    } else {
        None
    };
    let index = (draw as *const Draw as usize - frame.draws().as_ptr() as usize) / core::mem::size_of::<Draw>();
    let mut corners = frame.screen_corners(draw);
    while let (Some(a), Some(b), Some(c)) = (corners.next(), corners.next(), corners.next()) {
        // Debugging: REPLAY_COVER=x,y (960x544) prints every triangle covering
        // that pixel with its depth there, whether it passes or not.
        if let Some((px, py)) = std::env::var("REPLAY_COVER").ok().and_then(|value| {
            let (x, y) = value.split_once(',')?;
            Some((x.trim().parse::<f32>().ok()? + 0.5, y.trim().parse::<f32>().ok()? + 0.5))
        }) {
            let p = |v: &FrameVertex| (v.x * SCALE, v.y * SCALE);
            let (p0, p1, p2) = (p(&a), p(&b), p(&c));
            let area = edge(p0, p1, p2);
            if area != 0.0 {
                let w0 = edge(p1, p2, (px, py)) / area;
                let w1 = edge(p2, p0, (px, py)) / area;
                let w2 = edge(p0, p1, (px, py)) / area;
                if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                    let z = w0 * a.z + w1 * b.z + w2 * c.z;
                    println!("cover: draw {index} z {z:.6} ({:.1}) tests {:x} flags {:x} blend {:x} texture {} area {area:.0}", z * 65535.0, draw.tests, draw.flags, draw.blend, draw.texture);
                }
            }
        }
        draw_triangle(screen, depth, draw, texture, [a, b, c], frame.dither_matrix, frame.stencil_state_of(draw).filter(|_| {
            // Debugging: REPLAY_NO_STENCIL=<first>-<last> draws those draws
            // without their stencil state (which marks or tests matter).
            let range = std::env::var("REPLAY_NO_STENCIL").ok().and_then(|value| {
                let (a, b) = value.split_once('-')?;
                Some(a.parse::<usize>().ok()?..=b.parse::<usize>().ok()?)
            });
            !range.is_some_and(|range| range.contains(&index))
        }));
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

/// `stencil`: the draw's exact stencil state (core `blend::stencil`): its
/// test reads the target's alpha (the PSP's stencil) before anything is
/// written, its value replaces the alpha where the draw writes.
fn draw_triangle(
    screen: &mut [u32],
    depth: &mut [f32],
    draw: &Draw,
    texture: Option<&Texture>,
    corners: [FrameVertex; 3],
    dither_matrix: u64,
    stencil: Option<StencilState>,
) {
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
    // Normal and additive blends into 8-bit targets blend as the GE does,
    // as the Vita's frame-read programs for them (core `wide_variant`; the
    // Vita leaves draws with a stencil state to GXM), in frames that read
    // their targets' alpha (`GE_BLEND_FRAME`).
    let ge_exact = GE_BLEND_FRAME.load(std::sync::atomic::Ordering::Relaxed)
        && target_bits.is_none()
        && stencil.is_none()
        && wide_variant(draw.flags, draw.blend, true).is_some();
    let native = NATIVE_DRAW.load(std::sync::atomic::Ordering::Relaxed);
    for y in min_y..max_y {
        for x in min_x..max_x {
            let center = if native {
                ((x & !1) as f32 + 1.0, (y & !1) as f32 + 1.0) // REPLAY_NATIVE: the PSP pixel's centre
            } else {
                (x as f32 + 0.5, y as f32 + 0.5)
            };
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
            // Debugging: REPLAY_DEPTH_CENTER=1 takes the depth at the PSP pixel's
            // centre (the fragment program's plane from derivatives), coverage at 2x.
            let z = if DEPTH_CENTER.with(|on| *on) {
                let psp_center = ((x & !1) as f32 + 1.0, (y & !1) as f32 + 1.0);
                let weights = [edge(p1, p2, psp_center) / area, edge(p2, p0, psp_center) / area, edge(p0, p1, psp_center) / area];
                psp_depth(weights, &corners)
            } else {
                psp_depth(linear, &corners)
            };
            // Debugging: REPLAY_TIE_STATS=1 counts the fragments whose depth test
            // the PSP's whole units decide differently from the GPU's own float
            // depth (`TIE_COUNTS`, printed per draw at the frame's end).
            // The float model keeps a depth buffer of its own (per target).
            if depth_test && TIE_STATS.with(|on| *on) {
                TESTED.with(|t| t.set(t.get() + 1));
                let float_z = linear[0] * corners[0].z + linear[1] * corners[1].z + linear[2] * corners[2].z;
                FLOAT_DEPTH.with(|buffers| {
                    let mut buffers = buffers.borrow_mut();
                    let buffer = buffers.entry(depth.as_ptr() as usize).or_insert_with(|| vec![1.0f32; WIDTH * HEIGHT]);
                    let float_passes = passes(depth_function, float_z, buffer[index]);
                    if passes(depth_function, z, depth[index]) != float_passes {
                        TIE_COUNTS.with(|counts| *counts.borrow_mut().entry(CURRENT_DRAW.with(|d| d.get())).or_insert(0) += 1);
                    }
                    if float_passes && depth_write {
                        buffer[index] = float_z;
                    }
                });
            }
            if depth_test && !passes(depth_function, z, depth[index]) {
                continue;
            }
            if stencil.is_some_and(|state| !state.passes((screen[index] >> 24) as u8)) {
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
                    ge_shade(sample(texture, draw, u, v, draw_filter()), Color::weighted(colors, linear), fog_at(linear, &corners), draw, color_scale)
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
            let destination = Color::from_rgba(*pixel);
            *pixel = if ge_exact { ge_blend(draw.blend, source, destination) } else { blend(draw.blend, source, destination) }.to_rgba();
            if lab_blend_stats() && target_bits.is_none() && draw.blend & BLEND_ENABLED != 0 && !ge_exact {
                let key = format!(
                    "src{} dst{} eq{} fixb{} stencil{} writesA{} wide{}",
                    draw.blend & 0xF,
                    (draw.blend >> 4) & 0xF,
                    (draw.blend >> 8) & 7,
                    (draw.blend & BLEND_FIX_B_ALPHA != 0) as u8,
                    stencil.is_some() as u8,
                    (draw.blend & nativeres_core::blend::BLEND_WRITES_ALPHA != 0) as u8,
                    wide_variant(draw.flags, draw.blend, true).is_some() as u8
                );
                BLEND_STATS.with(|stats| {
                    let mut stats = stats.borrow_mut();
                    let entry = stats.entry(key).or_insert((0u64, std::collections::BTreeSet::new()));
                    entry.0 += 1;
                    entry.1.insert(CURRENT_DRAW.with(|d| d.get()));
                });
            }
            if let Some(value) = stencil.and_then(|state| state.write()) {
                *pixel = (*pixel & 0x00FF_FFFF) | (value as u32) << 24;
            }
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
/// A pixel's depth (0..1) as the PSP's depth buffer holds it, from the
/// screen weights `linear` of the triangle's corners. The GE (ppsspp-ref
/// TransformUnit `GEScreenZ`, Rasterizer `ComputePlane`) floors each
/// vertex's depth to a whole 16-bit unit, then floors the plane through
/// them at the pixel; draws less than a unit apart tie and the later one
/// passes "<=" (God of War's title fence, NFS UR's title cars over their
/// black quad). The exact GE uses float24 vertex maths and a fixed-point
/// plane: this model matches Sony's own depth buffer on 74 % of GoW's title
/// pixels, the rest within 1 (2026-11-23; rounding the pixel instead: 38 %).
/// Debugging: REPLAY_DEPTH_MODE 0 pixel rounded (the model before), 1 pixel
/// floored (PPSSPP PixelDepthRounding), 2 vertex floored to even and the
/// pixel unquantized (PPSSPP VertexDepthRounding), 4 vertex floored to even
/// and the pixel rounded, 5 unquantized, 6 vertex floored and the pixel
/// rounded, 7 vertex floored and the pixel unquantized (the GPU's own depth);
/// 8 as 3 with raw vertices' depth from the GE's own arithmetic
/// (`ge_math::GE_EXACT_DEPTH`), 9 the same with float clip coordinates and
/// the GE's viewport steps, 10 the GE's clip coordinates and a float
/// viewport; 3 (default) as above.
fn psp_depth(linear: [f32; 3], corners: &[FrameVertex; 3]) -> f32 {
    let mode = DEPTH_MODE.with(|mode| *mode);
    let vertex = |corner: &FrameVertex| {
        let z = corner.z * 65535.0;
        match mode {
            2 | 4 => (z * 0.5).floor() * 2.0,
            3 | 6 | 7 | 8..=10 => z.floor(),
            _ => z,
        }
    };
    let plane = linear[0] * vertex(&corners[0]) + linear[1] * vertex(&corners[1]) + linear[2] * vertex(&corners[2]);
    // + 0.01: a flat draw's plane lands on its whole value, give or take the
    // float weights' rounding (PPSSPP's fragment rounding does the same).
    (match mode {
        1 | 3 | 8..=10 => (plane + 0.01).floor(),
        2 | 5 | 7 => plane,
        _ => plane.round(),
    }) / 65535.0
}

fn fog_at(linear: [f32; 3], corners: &[FrameVertex; 3]) -> f32 {
    linear[0] * corners[0].fog + linear[1] * corners[1].fog + linear[2] * corners[2].fog
}

/// The GE's texture function and fog in its integer arithmetic, as the
/// Vita's fragment programs do it (`GeShade` in nativeres_textured_f.cg;
/// ppsspp-ref GPU/Software/Sampler.cpp GetTextureFunctionOutput, DrawPixel.cpp
/// fog): texel and vertex colour truncated to 8 bits, (p + 1) t >> 8 (x 2
/// with colour doubling), fog factor floor(f x 256), (c f + fog (255 - f) +
/// 255) >> 8. Rounding in float instead left MHF about one level brighter.
/// Rounding only the interpolated vertex colour moved every dumped scene
/// further from Sony's frame (2026-11-23: GoW title 6.07 -> 6.40, MHF village
/// 4.99 -> 5.07): the GE truncates gradients too.
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

thread_local! {
    /// Lab: REPLAY_BLEND_STATS=1 counts the pixels of blends the Vita leaves to GXM, per kind.
    static BLEND_STATS: std::cell::RefCell<std::collections::BTreeMap<String, (u64, std::collections::BTreeSet<usize>)>> = Default::default();
}

fn lab_blend_stats() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("REPLAY_BLEND_STATS").is_some())
}

/// Prints `BLEND_STATS` (REPLAY_BLEND_STATS).
pub fn print_blend_stats() {
    BLEND_STATS.with(|stats| {
        for (key, (pixels, draws)) in stats.borrow().iter() {
            let list: Vec<String> = draws.iter().take(30).map(|d| d.to_string()).collect();
            println!("blendstat {key}: pixels {pixels} draws {} [{}]", draws.len(), list.join(","));
        }
    });
}

/// The GE's blend of 8-bit values (`wide_variant`'s draws), as the Vita's
/// frame-read programs (nativeres_textured_read_f.cg READ_WIDE: normal and
/// additive; the generic one: FIX B) and ppsspp-ref DrawPixel.cpp
/// AlphaBlendingResult: each term ((2c + 1)(2f + 1)) >> 10 (`ge_blend_term`;
/// a factor of 255 is the colour itself), min / max on the plain colours,
/// clamped; FIX B through the alpha as `blend_channels` reads it; the
/// target's alpha kept. GXM's blending rounds the whole blend once instead:
/// ~0.5 levels brighter per normal blend.
fn ge_blend(packed: u32, source: Color, destination: Color) -> Color {
    let to8 = |value: f32| (value.clamp(0.0, 1.0) * 255.0 + 0.5) as i32;
    let (sa, da) = (to8(source.a), to8(destination.a));
    let fix_b = (packed & BLEND_FIX_B_ALPHA != 0).then(|| ((packed >> BLEND_FIX_B_SHIFT) & 0xFF) as i32);
    let factor8 = |code: u32, s: i32, d: i32| -> i32 {
        match code {
            0 => 0,
            1 => 255,
            2 => s,
            3 => 255 - s,
            4 => sa,
            5 => 255 - sa,
            6 => d,
            7 => 255 - d,
            8 => da,
            _ => 255 - da,
        }
    };
    let channel = |s: f32, d: f32| -> f32 {
        let (s, d) = (to8(s), to8(d));
        let (sf, df) = match fix_b {
            Some(grey) => (
                match (packed >> BLEND_PREMULTIPLY_SHIFT) & 3 {
                    1 => sa,
                    2 => 255 - sa,
                    _ => 255,
                },
                grey,
            ),
            None => (factor8(packed & 0xF, s, d), factor8((packed >> 4) & 0xF, s, d)),
        };
        let term = |c: i32, f: i32| ge_blend_term(c as u32, f as u32) as i32;
        let out = match (packed >> 8) & 7 {
            1 => term(s, sf) - term(d, df),
            2 => term(d, df) - term(s, sf),
            3 => s.min(d),
            4 => s.max(d),
            _ => term(s, sf) + term(d, df),
        };
        out.clamp(0, 255) as f32 / 255.0
    };
    Color { r: channel(source.r, destination.r), g: channel(source.g, destination.g), b: channel(source.b, destination.b), a: destination.a }
}

/// Whether this frame's normal and additive blends into 8-bit targets blend
/// as the GE does (`render_frame`, as the Vita's `blend8888`).
static GE_BLEND_FRAME: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// `REPLAY_NATIVE` (see the draw loop), read once.
fn native_mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| std::env::var("REPLAY_NATIVE").ok().and_then(|value| value.parse().ok()).unwrap_or(0))
}

/// Whether the draw being drawn filters as the GE does (`Filter::Ge`).
static GE_FILTER_DRAW: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The filter of the draw being drawn (`GE_FILTER_DRAW`).
fn draw_filter() -> Filter {
    if GE_FILTER_DRAW.load(std::sync::atomic::Ordering::Relaxed) { Filter::Ge } else { Filter::Gpu }
}

/// `REPLAY_PSP_FILTER`, read once: Some(true) the GE's filter for every
/// draw, Some(false) the GPU's for every draw, None as the Vita. `--ge-filter`
/// (the Vita's program set for God of War, core game_quirks.rs
/// `PSP_DEPTH_GAMES`): Some(true).
fn psp_filter_mode() -> Option<bool> {
    static MODE: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("REPLAY_PSP_FILTER") {
        Ok(value) => Some(value != "0"),
        Err(_) => std::env::args().any(|arg| arg == "--ge-filter").then_some(true),
    })
}

/// Whether the draw being drawn is shaded at 1x (`native_mode`).
static NATIVE_DRAW: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

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

/// How a draw's bilinear filter works (`sample`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Filter {
    /// The Vita's texture unit: the mix of the four texels, rounded to 8 bits.
    Gpu,
    /// The GE's (the Vita's GE_FILTER programs, core `ge_filter_read`):
    /// positions in 1/16 texels, each lerp truncated to 8 bits, the
    /// horizontal ones first (ppsspp-ref GPU/Software/Sampler.cpp
    /// SampleLinearLevelT).
    Ge,
}

fn sample(texture: &Texture, draw: &Draw, u: f32, v: f32, filter: Filter) -> Color {
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
    if filter == Filter::Ge {
        let fixed = |value: f32| (value * 16.0) as i64 - 8;
        let (bx, by) = (fixed(x), fixed(y));
        let (fx, fy) = ((bx & 15) as u32, (by & 15) as u32);
        let (x0, y0) = (bx >> 4, by >> 4);
        let raw = |x: i64, y: i64| texture.pixels[wrap(y, texture.height, clamp_v) * texture.width + wrap(x, texture.width, clamp_u)];
        let lerp = |a: u32, b: u32, f: u32| -> u32 {
            (0..4).fold(0, |out, c| out | (((a >> (8 * c)) & 0xFF) * (16 - f) + ((b >> (8 * c)) & 0xFF) * f) >> 4 << (8 * c))
        };
        let top = lerp(raw(x0, y0), raw(x0 + 1, y0), fx);
        let bottom = lerp(raw(x0, y0 + 1), raw(x0 + 1, y0 + 1), fx);
        return Color::from_rgba(lerp(top, bottom, fy));
    }
    // Bilinear: blend the four texels around the sample point.
    let (x, y) = (x - 0.5, y - 0.5);
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let (x0, y0) = (x0 as i64, y0 as i64);
    let top = Color::mix(texel(x0, y0), texel(x0 + 1, y0), fx);
    let bottom = Color::mix(texel(x0, y0 + 1), texel(x0 + 1, y0 + 1), fx);
    // The Vita's texture unit returns an 8-bit texture's filtered value
    // rounded to 8 bits (2026-11-23: God of War's bloom on the Vita = this
    // model within 0.4 levels; the plain mix, which the programs truncate,
    // was ~1.6 darker).
    Color::from_rgba(Color::mix(top, bottom, fy).to_rgba())
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

    /// A 2x1 texture, black then white, bilinear, clamped: half way
    /// between the texel centres the exact mix is 127.5.
    fn black_white() -> (Texture, Draw) {
        let texture = Texture { width: 2, height: 1, pixels: vec![0xFF00_0000, 0xFFFF_FFFF] };
        let flags = draw_flags::TEXTURED | draw_flags::LINEAR_FILTER | draw_flags::CLAMP_U | draw_flags::CLAMP_V;
        (texture, Draw { flags, ..Draw::default() })
    }

    /// God of War's bloom (2026-11-23): the Vita's texture unit returns the
    /// bilinear result of an 8-bit texture rounded to 8 bits (the programs
    /// then take it as it is); modelled as the float mix truncated, the
    /// replay's bloom was ~1.6 levels darker than the Vita's. 127.5 -> 128,
    /// which the program's texture function (white vertices) keeps.
    #[test]
    fn the_vitas_bilinear_filter_rounds_to_8_bits() {
        let (texture, draw) = black_white();
        let white = Color::from_rgba(0xFFFF_FFFF);
        let shade = |filter| ge_shade(sample(&texture, &draw, 0.5, 0.5, filter), white, 1.0, &draw, 1.0).to_rgba() & 0xFF;
        assert_eq!(shade(Filter::Gpu), 128);
        assert_eq!(shade(Filter::Ge), 127);
    }

    /// Draws that read one of our targets filter as the GE does
    /// (`Filter::Ge`, the Vita's GE_FILTER programs; ppsspp-ref
    /// Sampler.cpp SampleLinearLevelT): positions in 1/16 texels, half a
    /// texel back, each lerp truncated. u 0.5 of 2 texels: 16 - 8 = 8 ->
    /// texel 0, fraction 8/16 -> (255 x 8) >> 4 = 127. u 0.65625: 21 - 8 =
    /// 13 -> (255 x 13) >> 4 = 207 (207.2; the GPU: 0.8125 x 255 = 207.2,
    /// 207). u 0.53125: 17 - 8 = 9 -> 143 (143.4).
    #[test]
    fn draws_reading_our_targets_filter_as_the_ge() {
        let (texture, draw) = black_white();
        let red = |u: f32| sample(&texture, &draw, u, 0.5, Filter::Ge).to_rgba() & 0xFF;
        assert_eq!(red(0.5), 127);
        assert_eq!(red(0.65625), 207);
        assert_eq!(red(0.53125), 143);
        // 16.96 truncated to 16: fraction 8, where the exact mix is 142.8.
        assert_eq!(red(0.53), 127);
    }

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
        draw_triangle(&mut screen, &mut depth, &draw, Some(&white), corners, 0, None);
        let pixel = |screen: &[u32]| screen[50 * WIDTH + 50] & 0xFF; // PSP (25, 25), weights 1/2, 1/4, 1/4
        assert!((pixel(&screen) as i32 - 129).abs() <= 1, "colour {}", pixel(&screen));
        // Black geometry, white fog: the fog factor alone gives the colour.
        let fogged = corners.map(|corner| FrameVertex { color: 0xFF00_0000, ..corner });
        let draw = Draw { flags: draw_flags::TEXTURED | draw_flags::FOG, fog_color: 0x00FF_FFFF, ..whole };
        screen.fill(0);
        draw_triangle(&mut screen, &mut depth, &draw, Some(&white), fogged, 0, None);
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

    /// Normal alpha blends into 8-bit targets as the GE does (core
    /// `wide_variant`, the Vita's READ_WIDE programs): each term truncated
    /// on its own, where GXM's blend rounds once (15.69 + 168.63 -> 184);
    /// the target's alpha kept.
    #[test]
    fn eight_bit_targets_blend_as_the_ge() {
        let mut screen = vec![0x3000_00C8u32; WIDTH * HEIGHT]; // red 200, alpha (stencil) 0x30
        let mut depth = vec![1.0f32; WIDTH * HEIGHT];
        let corner = |x: f32, y: f32| FrameVertex { x, y, z: 0.5, w: 1.0, color: 0x2800_0064, ..FrameVertex::default() };
        let triangle = [corner(0.0, 0.0), corner(100.0, 0.0), corner(0.0, 100.0)];
        let draw = Draw { blend: Blend::ALPHA.pack(), scissor_right: 479, scissor_bottom: 271, ..Draw::default() };
        assert!(wide_variant(draw.flags, draw.blend, true).is_some());
        GE_BLEND_FRAME.store(true, std::sync::atomic::Ordering::Relaxed); // a frame that reads its alpha
        draw_triangle(&mut screen, &mut depth, &draw, None, triangle, 0, None);
        GE_BLEND_FRAME.store(false, std::sync::atomic::Ordering::Relaxed);
        // Red 100 at alpha 40 over 200: ((201 x 81) >> 10) + ((401 x 431) >> 10) = 15 + 168.
        assert_eq!(screen[20 * WIDTH + 20], 0x3000_00B7);
        assert_eq!(blend(draw.blend, Color::from_rgba(0x2800_0064), Color::from_rgba(0x3000_00C8)).to_rgba() & 0xFF, 184, "GXM's way");
    }

    /// FIX B into an 8-bit target (God of War's bloom downsampling: the
    /// target x 0x55 plus the texel): the destination term truncated as
    /// the GE's, as the Vita's generic frame-read program (`wide_variant`).
    #[test]
    fn eight_bit_targets_blend_fix_b_as_the_ge() {
        let mut screen = vec![0x3000_00C8u32; WIDTH * HEIGHT]; // red 200, alpha (stencil) 0x30
        let mut depth = vec![1.0f32; WIDTH * HEIGHT];
        let corner = |x: f32, y: f32| FrameVertex { x, y, z: 0.5, w: 1.0, color: 0xFF00_0064, ..FrameVertex::default() };
        let triangle = [corner(0.0, 0.0), corner(100.0, 0.0), corner(0.0, 100.0)];
        let packed = BLEND_ENABLED | BLEND_FIX_B_ALPHA | 1 | 4 << 4 | 0x55 << BLEND_FIX_B_SHIFT;
        let draw = Draw { blend: packed, flags: draw_flags::TEXTURED, scissor_right: 479, scissor_bottom: 271, ..Draw::default() };
        assert!(wide_variant(draw.flags, draw.blend, true).is_some());
        // A white texel: the source is the vertex colour (modulate).
        let texture = Texture { width: 1, height: 1, pixels: vec![0xFFFF_FFFF] };
        GE_BLEND_FRAME.store(true, std::sync::atomic::Ordering::Relaxed);
        draw_triangle(&mut screen, &mut depth, &draw, Some(&texture), triangle, 0, None);
        GE_BLEND_FRAME.store(false, std::sync::atomic::Ordering::Relaxed);
        // 100 + ((401 x 171) >> 10) = 100 + 66; the target's alpha kept.
        assert_eq!(screen[20 * WIDTH + 20], 0x3000_00A6);
        assert_eq!(blend(draw.blend, Color::from_rgba(0xFF00_0064), Color::from_rgba(0x3000_00C8)).to_rgba() & 0xFF, 167, "GXM's way");
    }

    /// God of War's title fence (2026-11-23): rail boxes and planks less
    /// than a depth unit apart. The GE floors the plane at the pixel to 16
    /// bits, so they tie and the later draw wins "<=" (Sony's depth buffer:
    /// 64700 at the post); in floats the earlier box won everywhere.
    #[test]
    fn depth_ties_at_whole_psp_units() {
        let mut screen = vec![0xFF00_0000u32; WIDTH * HEIGHT];
        let mut depth = vec![1.0f32; WIDTH * HEIGHT];
        let flat = |z: f32, color: u32| {
            let corner = |x: f32, y: f32| FrameVertex { x, y, z: z / 65535.0, w: 1.0, color, ..FrameVertex::default() };
            [corner(0.0, 0.0), corner(100.0, 0.0), corner(0.0, 100.0)]
        };
        let draw = Draw { tests: test_flags::DEPTH_TEST | 5 << test_flags::DEPTH_FUNCTION_SHIFT, scissor_right: 479, scissor_bottom: 271, ..Draw::default() };
        draw_triangle(&mut screen, &mut depth, &draw, None, flat(64700.3, 0xFF00_00FF), 0, None);
        draw_triangle(&mut screen, &mut depth, &draw, None, flat(64700.7, 0xFF00_FF00), 0, None);
        assert_eq!(screen[20 * WIDTH + 20], 0xFF00_FF00, "the later draw wins the tie");
    }

    /// The GE floors each vertex's depth before the plane (ppsspp-ref
    /// TransformUnit `GEScreenZ`): a triangle from 64700.9 to 64699.9 is
    /// 64699.5 half way, not 64700.4, so a flat 64700 behind it fails "<=".
    #[test]
    fn vertex_depth_is_floored_before_the_plane() {
        let mut screen = vec![0xFF00_0000u32; WIDTH * HEIGHT];
        let mut depth = vec![1.0f32; WIDTH * HEIGHT];
        let corner = |x: f32, y: f32, z: f32, color: u32| FrameVertex { x, y, z: z / 65535.0, w: 1.0, color, ..FrameVertex::default() };
        let draw = Draw { tests: test_flags::DEPTH_TEST | 5 << test_flags::DEPTH_FUNCTION_SHIFT, scissor_right: 479, scissor_bottom: 271, ..Draw::default() };
        let red = 0xFF00_00FF;
        let sloped = [corner(0.0, 0.0, 64700.9, red), corner(100.0, 0.0, 64699.9, red), corner(0.0, 100.0, 64700.9, red)];
        draw_triangle(&mut screen, &mut depth, &draw, None, sloped, 0, None);
        let green = 0xFF00_FF00;
        let behind = [corner(0.0, 0.0, 64700.0, green), corner(100.0, 0.0, 64700.0, green), corner(0.0, 100.0, 64700.0, green)];
        draw_triangle(&mut screen, &mut depth, &draw, None, behind, 0, None);
        // PSP x 50.25: 64700 - 0.5025 floored = 64699 < 64700.
        assert_eq!(screen[10 * WIDTH + 100], red);
    }

    /// God of War's marks and its blur's test (core `blend::stencil`): a
    /// draw writes its stencil value as the alpha, a tested draw changes
    /// only the pixels whose alpha passes.
    #[test]
    fn stencil_values_are_written_and_tested_on_the_alpha() {
        let mut screen = vec![0x0000_0000u32; WIDTH * HEIGHT];
        let mut depth = vec![1.0f32; WIDTH * HEIGHT];
        let corner = |x: f32, y: f32| FrameVertex { x, y, z: 0.5, w: 1.0, color: 0xFF00_00FF, ..FrameVertex::default() };
        let left = [corner(0.0, 0.0), corner(100.0, 0.0), corner(0.0, 100.0)];
        let whole = [corner(0.0, 0.0), corner(479.0, 0.0), corner(0.0, 271.0)];
        let draw = Draw { blend: BLEND_WRITES_ALPHA, scissor_right: 479, scissor_bottom: 271, ..Draw::default() };
        let mark = StencilState::new(None, Some(0x10));
        draw_triangle(&mut screen, &mut depth, &draw, None, left, 0, Some(mark));
        assert_eq!(screen[20 * WIDTH + 20], 0x1000_00FF, "red, stencil 0x10");
        // Green where (stencil & 0x10) == 0x10, the alpha kept.
        let green = whole.map(|vertex| FrameVertex { color: 0xFF00_FF00, ..vertex });
        let tested = Draw { blend: 0, ..draw };
        draw_triangle(&mut screen, &mut depth, &tested, None, green, 0, Some(StencilState::new(Some((2, 0x10, 0x10)), None)));
        assert_eq!(screen[20 * WIDTH + 20], 0x1000_FF00);
        assert_eq!(screen[20 * WIDTH + 400], 0, "failed the test: untouched");
    }
}
