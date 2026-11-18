//! The display thread's part of the renderer: draws a recorded PSP frame at
//! 960x544.
//!
//! Adrenaline's display loop (menu.c) calls `NativeResRenderFrame` once per
//! loop, before its own scene. It takes the newest recorded frame
//! (frames.rs), looks up the frame's textures (textures.rs: decoded into a
//! cache, keyed by description and a hash of the pixel data, so a texture is
//! only decoded again when the game changes it) and draws the frame into the
//! render target of its PSP framebuffer (targets.rs), in a GXM scene of its
//! own: every draw of the frame becomes one sceGxmDraw (draw.rs). The
//! display's scene then shows that target (`NativeResPresentFrame`). PSP
//! coordinates are simply doubled: the PSP screen is 480x272.

use core::ffi::c_void;
use core::ptr::null;
use core::sync::atomic::{AtomicI32, AtomicU32, AtomicU8, Ordering};

use nativeres_core::blend::{self, BLEND_ENABLED};
use nativeres_core::frame::{draw_flags, Frame, MAX_TEXTURES};
use nativeres_core::opaque_source::palette_texture_opaque;
use nativeres_core::texture_cache::MAX_RENDER_TARGETS;

use crate::capture::{COMPLETED_FRAME_COUNT, GPU_TRANSFORM_READY, set_mip_mode};
use crate::draw::{self, REDRAW_SKIP};
use crate::frames::{acquire_frame, wait_for_frame};
use crate::gpu_textures::TextureObject;
use crate::render_stats::stats;
use crate::targets::{self, Targets, SONY_PICTURE, SONY_PICTURE_STRIDE, TARGET_COUNT};
use crate::textures::{self, FrameInfo};
use crate::{ext, frame_pool, gxm, options, palettes, render_log, sys, SingleThreaded};

/// Menu option "NativeRes Texture Filter" (menu.c points at this byte): 0 =
/// as the game asks (mostly nearest, so every texel becomes a hard 2x2 block
/// at 960x544), 1 = always bilinear (softer text and sprites, closer to the
/// upscaled original).
pub static SMOOTH_TEXTURES: AtomicU8 = AtomicU8::new(0);

/// Only these draws of a frame are drawn (count 0 = off), to find which draw makes a part of the picture (pause
/// the game: the frame is redrawn).
pub static DRAW_ONLY_FIRST: AtomicU32 = AtomicU32::new(0);
pub static DRAW_ONLY_COUNT: AtomicU32 = AtomicU32::new(0);

/// Our targets' alpha holds the PSP's stencil values only while draws keep
/// it (colour mask RGB, `blend::BLEND_WRITES_ALPHA`), which costs the GPU
/// (MTX race frame 12.3 -> 14.6 ms): so only while the game reads it
/// (`blend::keep_target_alpha`: a target sampled with its alpha, or a
/// destination-alpha blend, within the last 120 frames); otherwise every
/// draw writes alpha as before. 0 never kept, 1 (default) while read, 2
/// always.
pub static ALPHA_MASK: AtomicI32 = AtomicI32::new(1);

/// 1: wait for the GPU after each target scene and count the time in the
/// render statistics' gpu=. Slows rendering down; for measuring only.
/// 2: the GPU drained before the scene too.
pub static MEASURE_GPU: AtomicI32 = AtomicI32::new(0);
/// Process time (low 32 bits) when the last drawn frame's rendering
/// started / ended, and when `NativeResRenderFrame` was entered.
pub static RENDER_START_US: AtomicU32 = AtomicU32::new(0);
pub static RENDER_END_US: AtomicU32 = AtomicU32::new(0);
pub static LOOP_START_US: AtomicU32 = AtomicU32::new(0);
/// Draws of the last frame drawn.
pub static DRAWN_DRAW_COUNT: AtomicU32 = AtomicU32::new(0);

/// A PSP texture address (0 = off): the texture
/// cache logs every lookup of the textures at that address: cached or not,
/// decoded, waiting, the data hash and the VRAM copy state.
pub static TRACE_TEXTURE: AtomicU32 = AtomicU32::new(0);

/// Make the arenas (once, at the first drawn
/// frame that wants them; never freed, frames may point into them): 1 = 1
/// MiB of main memory, 2 = phycont memory with the raw arena. 2 by default
/// since 2026-11-06.
pub static VERTEX_CACHE_WANTED: AtomicI32 = AtomicI32::new(2);

/// The frames are captured, recorded and taken as
/// always, but not drawn (the game's own picture is shown): tells what our
/// GPU drawing costs the game (2026-11-09: MTX runs ~27 fps while driving
/// with the renderer on, 30 with it off, its own time between GE sync and
/// flip is what grows).
pub static GPU_DRAW: AtomicI32 = AtomicI32::new(1);

/// 1 (default) or 0. MTX Mototrax blends the previous
/// frame's picture over each new one (motion blur, strong in water: 70 %).
/// That framebuffer's target holds the last frame WE drew into it; when the
/// capture skipped the frame before (it records the newest when behind), it
/// is 2+ frames older than the game's and its old picture showed through
/// (ghost hills and sky over the water, slot 8). Then the target the frame
/// draws into, if it holds a newer frame (normally the one 2 back), is
/// copied into the sampled one first: one frame older than the game's
/// instead of 3, 5, ...
pub static STALE_COPY: AtomicI32 = AtomicI32::new(1);

/// `shows_cpu_written_picture` on (1, default) or off.
pub static CPU_PICTURE_CHECK: AtomicU8 = AtomicU8::new(1);

/// Draw the frame we have once more (paused game, a switch changed what
/// is drawn).
static FORCE_REDRAW: AtomicI32 = AtomicI32::new(0);

pub fn redraw_paused() {
    FORCE_REDRAW.store(1, Ordering::Relaxed);
}

/// See `NativeResRenderFrame`: a PSP frame comes every 16.7 ms.
const FRAME_DUE_US: u64 = 12_000;
const FRAME_WAIT_US: u32 = 5_000;
/// See `NativeResRenderFrame`: how many PSP frames old our last picture may
/// be to be shown again while a frame waits for textures.
const MAX_HELD_FRAMES: u32 = 2;
/// Frames in a row left out because their textures were rewritten.
const MAX_MOVED_ON_HELD: u32 = 2;
/// Frames taking longer than this are logged (a frame lasts 16.7 ms).
const SLOW_FRAME_US: u64 = 20_000;

/// PSP screen pixels to Vita screen pixels.
const SCALE: f32 = 2.0;

/// What `draw_frame` did.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Drawn,
    /// Not yet: textures missing (try again), or no render target.
    NotYet,
    /// Not to be shown (see `MAX_MOVED_ON_HELD`).
    LeftOut,
}

/// The frame loop's state (display thread only).
pub struct FrameLoop {
    /// Frames drawn (`draw_frame`); the texture cache, palette tables and
    /// render targets count ages in these.
    pub render_count: u32,
    /// A new frame not drawn yet: it waited for textures, or was just
    /// acquired.
    frame_pending: bool,
    arena_tried: bool,
    last_new_frame_us: u64,
    /// The frame shown from Sony's picture (`shows_cpu_written_picture`),
    /// remembered while it stays the newest (paused) so it doesn't flip back.
    cpu_picture_frame: u32,
    last_drawn_frame: u32,
    last_drawn_us: u64,
    /// The last two framebuffers frames showed, newest first.
    shown_history: [u32; 2],
    moved_on_held: u32,
    /// `render_count` when the game last read our targets' alpha; 0: never.
    target_alpha_read: u32,
    /// Log lines left for failed scene ends.
    end_logs: u32,
    /// This frame's textures (ours or a target's; null: none), index
    /// textures' colour tables, waiting flags, texture coordinate scales.
    textures: [*const TextureObject; MAX_TEXTURES],
    palettes: [*const c_void; MAX_TEXTURES],
    waiting: [u8; MAX_TEXTURES],
    uv_scales: [[f32; 2]; MAX_TEXTURES],
    /// Index textures whose every texel the GPU can sample is opaque (their
    /// colour table's entries, core `opaque_source`); false for the others.
    opaque_textures: [bool; MAX_TEXTURES],
}

static FRAME_LOOP: SingleThreaded<FrameLoop> = SingleThreaded::new(FrameLoop {
    render_count: 0,
    frame_pending: false,
    arena_tried: false,
    last_new_frame_us: 0,
    cpu_picture_frame: u32::MAX,
    last_drawn_frame: 0,
    last_drawn_us: 0,
    shown_history: [0; 2],
    moved_on_held: 0,
    target_alpha_read: 0,
    end_logs: 5,
    textures: [null(); MAX_TEXTURES],
    palettes: [null(); MAX_TEXTURES],
    waiting: [0; MAX_TEXTURES],
    uv_scales: [[1.0; 2]; MAX_TEXTURES],
    opaque_textures: [false; MAX_TEXTURES],
});

/// True if the GE can have drawn framebuffer `psp_address` (physical, see
/// capture.rs). The GE only renders into VRAM (0x04000000, 2 MiB, plus its
/// mirrors up to 0x04800000): a framebuffer in main RAM was written by the
/// CPU, e.g. MTX Mototrax's logo movies. We have nothing drawn for it.
fn is_ge_framebuffer(psp_address: u32) -> bool {
    (0x0400_0000..0x0480_0000).contains(&psp_address)
}

fn now() -> u64 {
    sys::now_us()
}

/// Display thread, after vita2d_pool_reset and before the display's scene
/// (vita2d_start_drawing): draws a new frame into the render target of its
/// PSP framebuffer, in a GXM scene of its own. Returns 1 if
/// `NativeResPresentFrame` should be used this time, 0 (renderer off, no
/// frame yet, or this frame falls back to the game's own picture) if not.
pub fn render_frame() -> bool {
    // Safe: the display thread's entry point; both references are passed
    // down, never taken again below.
    let (state, targets) = unsafe { (FRAME_LOOP.get(), targets::targets()) };
    state.render_frame(targets)
}

/// Display thread, while the PSP runs its XMB (the game was quit): what
/// the targets hold is the quit game's, never to be shown again (the next
/// game's start showed it until its first frame of ours).
pub fn leave_game() {
    // Safe: the display thread, as `render_frame`.
    let (state, targets) = unsafe { (FRAME_LOOP.get(), targets::targets()) };
    targets.forget();
    state.frame_pending = false;
}

impl FrameLoop {
    fn render_frame(&mut self, targets: &mut Targets) -> bool {

        if ext::before_render(self, targets) {
            return false;
        }

        LOOP_START_US.store(now() as u32, Ordering::Relaxed);
        // Distant textures (options.rs) to the capture (texture::MIP_MODE).
        set_mip_mode(options::mip_mode() as u32);
        if options::renderer_enabled() == 0 {
            targets.forget();
            self.frame_pending = false;
            return false;
        }
        // The vertex cache's arena, once asked for.
        let arenas_wanted = VERTEX_CACHE_WANTED.load(Ordering::Relaxed);
        if arenas_wanted != 0 && !self.arena_tried && GPU_TRANSFORM_READY.load(Ordering::Acquire) != 0 {
            self.arena_tried = true;
            draw::make_arenas(arenas_wanted == 2);
        }
        // The display loop runs at the Vita's refresh, and a PSP frame
        // finishes capturing at varying times within it. Finishing just after
        // the loop looked meant showing the last picture again and then
        // skipping that frame (judder at a steady 60 fps). So when a frame is
        // due but not there yet, wait a little for it: it is shown as soon as
        // it exists, never later. Not waiting longer leaves time to draw it
        // before the refresh.
        if now().wrapping_sub(self.last_new_frame_us) > FRAME_DUE_US {
            wait_for_frame(FRAME_WAIT_US);
        }
        let mut is_new = 0;
        // Safe: display thread; a valid out pointer.
        let frame = unsafe { acquire_frame(&mut is_new) };
        let is_new = is_new != 0;
        if is_new {
            self.last_new_frame_us = now();
        }
        // Safe: the frame handover keeps the frame until the next acquire.
        let Some(frame) = (unsafe { frame.as_ref() }) else {
            return false;
        };
        if GPU_DRAW.load(Ordering::Relaxed) == 0 {
            self.frame_pending = false;
            return false;
        }
        if !is_ge_framebuffer(frame.shown_framebuffer) {
            // CPU-drawn picture (movie): the game's own, upscaled. Our
            // targets stay as they are for when the game draws with the GE
            // again.
            if is_new {
                stats().cpu_frames += 1;
            }
            self.frame_pending = false;
            return false;
        }
        if targets::linear_indices().is_null() || !textures::ready() {
            return false; // no memory for the indices or the staging buffer
        }
        if ext::before_draw(self, targets, frame) {
            return targets.shown.is_some();
        }

        // Nothing of ours shown yet (renderer just switched on): draw the
        // frame we have even if it isn't new, if the game hasn't shown
        // another since (it is paused: we would otherwise only ever show
        // the game's own picture).
        let newest = frame.frame_number.wrapping_add(1) == COMPLETED_FRAME_COUNT.load(Ordering::Acquire);
        self.frame_pending |= is_new || (targets.shown.is_none() && newest);
        if FORCE_REDRAW.swap(0, Ordering::Relaxed) != 0 {
            self.frame_pending = true;
        }
        // A picture the CPU wrote under few draws (MTX's loading screens):
        // Sony's picture for this frame, as for movies. Remembered while
        // the frame stays the newest (paused), so it doesn't flip back.
        if self.frame_pending && shows_cpu_written_picture(frame, targets) {
            self.cpu_picture_frame = frame.frame_number;
            stats().cpu_frames += 1;
            self.frame_pending = false;
            return false;
        }
        if !self.frame_pending && frame.frame_number == self.cpu_picture_frame {
            return false;
        }
        if self.frame_pending {
            match self.draw_frame(targets, frame) {
                Outcome::LeftOut => {
                    // Can't be drawn as the GE drew it (textures rewritten
                    // since): not shown; our last picture stays until the
                    // next frame.
                    self.frame_pending = false;
                    return targets.shown.is_some();
                }
                Outcome::NotYet => {
                    // Textures not there in time (new ones past the work
                    // deadline, a VRAM copy one flip away): keep showing our
                    // last picture, as for a late frame. Showing the game's
                    // own picture instead switched to the upscaled original
                    // for a frame, a visible glitch when Final Fantasy's
                    // battles appear. If our last picture is old (renderer
                    // just switched on, savestate loaded), the game's.
                    return targets.shown.is_some_and(|shown| {
                        frame.frame_number.wrapping_sub(targets.targets[shown].frame_number) <= MAX_HELD_FRAMES
                    });
                }
                Outcome::Drawn => {
                    self.frame_pending = false;
                }
            }
        }
        targets.shown.is_some()
    }

    /// Draws `frame` into the target of its framebuffer.
    pub fn draw_frame(&mut self, targets: &mut Targets, frame: &Frame) -> Outcome {
        self.render_count = self.render_count.wrapping_add(1);
        let render_count = self.render_count;
        let start = now();
        DRAWN_DRAW_COUNT.store(frame.draw_count, Ordering::Relaxed);

        ext::frame_draw_started(frame);

        self.prepare_sampled_targets(targets, frame);
        targets.publish_addresses();

        // Decode (or find in the cache) every texture of the frame once.
        let lookup_start = now();
        let mut info = FrameInfo {
            render_count,
            smooth: SMOOTH_TEXTURES.load(Ordering::Relaxed) != 0,
            trace_address: TRACE_TEXTURE.load(Ordering::Relaxed),
            loop_start_us: LOOP_START_US.load(Ordering::Relaxed),
            render_targets: [0; MAX_RENDER_TARGETS],
        };
        let sampled_targets =
            targets.targets.iter().filter(|t| t.exists() && t.psp_address != 0 && t.psp_address != frame.shown_framebuffer);
        for (slot, target) in info.render_targets.iter_mut().zip(sampled_targets) {
            *slot = target.psp_address;
        }
        textures::for_frame(frame, &info, &mut self.textures, &mut self.waiting);
        let palette_start = now();
        let parts = textures::frame_times();
        {
            let stats = stats();
            stats.lookup_us += palette_start - lookup_start;
            stats.begin_tex_us += parts.trim_us;
            stats.hash_tex_us += parts.hash_us;
        }
        self.use_targets_as_textures(targets, frame, render_count);
        let alpha_mask = ALPHA_MASK.load(Ordering::Relaxed);
        let keep_target_alpha = alpha_mask == 2 || (alpha_mask == 1 && blend::keep_target_alpha(self.target_alpha_read, render_count));
        draw::set_keep_target_alpha(keep_target_alpha);
        self.find_palettes(frame, render_count);
        let textures_us = now() - start;
        {
            let stats = stats();
            stats.palette_us += now() - palette_start;
            stats.textures_us += textures_us;
        }

        if self.waits_for_textures(frame, render_count, textures_us) {
            stats().fallback_frames += 1;
            return Outcome::NotYet;
        }
        // Textures the game rewrote after the GE drew this frame (MTX's
        // menus load the next screen's pictures while the old screen's last
        // frame is on its way; decoded now they showed as noise or wrong
        // tiles): the frame is left out, at most MAX_MOVED_ON_HELD in a row
        // (texture_cache moved_on; `QUICK_CHECKS` off switches the checks
        // off).
        let moved_on = textures::moved_on();
        if moved_on != 0 && self.moved_on_held < MAX_MOVED_ON_HELD {
            self.moved_on_held += 1;
            stats().moved_on_frames += 1;
            render_log::line(format_args!("frame {}: {} textures rewritten since the GE drew it: not shown\n", render_count, moved_on));
            return Outcome::LeftOut;
        }
        self.moved_on_held = 0;

        let Some(target) = targets.target_for(frame.shown_framebuffer, render_count) else {
            render_log::line(format_args!("frame {}: no render target (out of memory)\n", render_count));
            return Outcome::NotYet;
        };
        targets.targets[target].last_used = render_count;
        ext::frame_drawn(frame);
        if MEASURE_GPU.load(Ordering::Relaxed) == 2 {
            // "gputime 2": the GPU drained first (not counted), so gpu_us is
            // this scene's own GPU time after its submission (mostly the
            // fragment phase: SGX shades tiles after sceGxmEndScene), not
            // time the GPU spent on or waiting for earlier scenes. Costs the
            // overlap.
            // Safe: the display thread's context.
            unsafe { gxm::sceGxmFinish(gxm::context()) };
        }
        self.refresh_stale_targets(targets, frame, target);

        let begin_start = now();
        targets.begin_scene(target);
        let loop_start = now();
        stats().begin_us += loop_start - begin_start;
        self.draw_draws(frame);
        draw::draw_tests(); // GPU tests, alpha log
        let end_start = now();
        stats().loop_us += end_start - loop_start;
        // vita2d draws with GXM's defaults (all at z 0.5, no depth test, no culling).
        draw::end_draws();
        // Safe: the display thread's context, inside the target scene.
        let end_result = unsafe { gxm::sceGxmEndScene(gxm::context(), null(), frame_pool::close_segment()) };
        if end_result < 0 {
            frame_pool::scene_failed();
        }
        stats().end_us += now() - end_start;
        if end_result < 0 && self.end_logs > 0 {
            self.end_logs -= 1;
            render_log::line(format_args!("sceGxmEndScene (target) failed: {:08x}\n", end_result));
        }
        targets.shown = Some(target);
        targets.targets[target].frame_number = frame.frame_number;
        self.last_drawn_frame = frame.frame_number;
        self.last_drawn_us = now();
        if MEASURE_GPU.load(Ordering::Relaxed) != 0 {
            let gpu_start = now();
            // Safe: the display thread's context.
            unsafe { gxm::sceGxmFinish(gxm::context()) };
            stats().gpu_us += now() - gpu_start;
        }

        let end = now();
        RENDER_START_US.store(start as u32, Ordering::Relaxed);
        RENDER_END_US.store(end as u32, Ordering::Relaxed);
        let total_us = end - start;
        {
            let stats = stats();
            stats.frames += 1;
            stats.draws += frame.draw_count;
            stats.total_us += total_us;
        }
        if total_us > SLOW_FRAME_US {
            let times = textures::frame_times();
            render_log::line(format_args!(
                "frame {}: slow, {} us (textures {} us: trim {}, hash {}, allocate {}, decode {})\n",
                render_count, total_us, textures_us, times.trim_us, times.hash_us, times.allocate_us, times.decode_us
            ));
        }
        Outcome::Drawn
    }

    /// Render to texture: a game may draw with a framebuffer as texture (MTX
    /// Mototrax samples the previous frame's picture every frame). Targets a
    /// frame samples must hold the game's picture (`seed_from_sony`): after
    /// a savestate load, all of ours hold pictures of before; the other
    /// framebuffer (the last one shown before this frame's) may have no
    /// target yet (renderer just switched on).
    fn prepare_sampled_targets(&mut self, targets: &mut Targets, frame: &Frame) {
        let idle_us = now().wrapping_sub(self.last_drawn_us);
        if nativeres_core::frame::targets_outdated(self.last_drawn_frame, frame.frame_number, idle_us) {
            for target in targets.targets.iter_mut().filter(|t| t.psp_address != 0) {
                target.needs_clear = true;
            }
        }
        if frame.shown_framebuffer != self.shown_history[0] {
            self.shown_history = [frame.shown_framebuffer, self.shown_history[0]];
        }
        let other = self.shown_history[1];
        if other == 0 || other == frame.shown_framebuffer || !is_ge_framebuffer(other) || !frame.samples_framebuffer(other) {
            return;
        }
        let sampled = targets.find(other).or_else(|| targets.target_for(other, self.render_count));
        if let Some(sampled) = sampled {
            let target = &mut targets.targets[sampled];
            target.last_used = self.render_count; // not the one target_for gives this frame
            if target.needs_clear {
                targets.seed_from_sony(sampled, frame.frame_number.wrapping_sub(1));
            }
        }
    }

    /// Textures at a framebuffer we draw into a target of our own come from
    /// that target: our resolution, and no decoding of 1 MiB from VRAM (~20
    /// ms). Not the framebuffer this frame draws into: a target can't be read
    /// while drawn.
    fn use_targets_as_textures(&mut self, targets: &mut Targets, frame: &Frame, render_count: u32) {
        for (i, wanted) in frame.textures().iter().enumerate() {
            self.uv_scales[i] = [1.0, 1.0];
            if !self.textures[i].is_null() || wanted.format >= 4 || wanted.address == frame.shown_framebuffer {
                continue;
            }
            let Some(k) = targets.find(wanted.address) else {
                continue;
            };
            let target = &mut targets.targets[k];
            if blend::samples_target_alpha(wanted.flags) {
                self.target_alpha_read = render_count;
                self.textures[i] = &target.texture;
            } else {
                target.opaque_view.gxm = target.texture.gxm;
                // Safe: a texture description of ours.
                unsafe { gxm::sceGxmTextureSetFormat(&mut target.opaque_view.gxm, gxm::TEXTURE_FORMAT_X8U8U8U8_1BGR) };
                self.textures[i] = &target.opaque_view;
            }
            // Our target holds the PSP's 480x272 screen; the PSP texture
            // coordinates run across width x height PSP pixels.
            self.uv_scales[i] = [wanted.width as f32 / 480.0, wanted.height as f32 / 272.0];
            stats().target_textures += 1;
        }
    }

    /// The colour tables of the frame's index textures: a palette table
    /// (palettes.rs) or, without a free one, the texture's own table (as
    /// before: if the palette changes, a frame still on the GPU may see the
    /// new one).
    fn find_palettes(&mut self, frame: &Frame, render_count: u32) {
        for (i, descriptor) in frame.textures().iter().enumerate() {
            self.palettes[i] = null();
            self.opaque_textures[i] = false;
            // Safe: textures from the cache or targets, alive this frame.
            let Some(texture) = (unsafe { self.textures[i].as_ref() }) else {
                continue;
            };
            // Safe: a texture description of ours.
            if unsafe { gxm::sceGxmTextureGetFormat(&texture.gxm) } != gxm::TEXTURE_FORMAT_P8_ABGR {
                continue;
            }
            let (table, mut opaque_entries) = palettes::palette_for(frame, descriptor, render_count);
            self.palettes[i] = table;
            if table.is_null() {
                // Safe: a palette texture of ours has its 256-word table.
                let own = unsafe { &mut *(gxm::sceGxmTextureGetPalette(&texture.gxm) as *mut [u32; 256]) };
                opaque_entries = palettes::fill_table(frame, descriptor, own);
            }
            self.opaque_textures[i] = palette_texture_opaque(descriptor.format, opaque_entries);
        }
    }

    /// Draws whose texture we don't have. A new VRAM texture arrives one
    /// frame late (see VramState::copy_pending in texture_cache), and new
    /// textures over the work budget wait a frame; drawing without them
    /// leaves holes, which flickered at Final Fantasy's battle start. Such
    /// frames aren't drawn; `NativeResRenderFrame` shows our last picture
    /// again meanwhile. True if the frame waits.
    fn waits_for_textures(&self, frame: &Frame, render_count: u32, textures_us: u64) -> bool {
        let (mut missing, mut waiting_draws) = (0u32, 0u32);
        let mut first_missing = None;
        for draw in frame.draws() {
            if draw.framebuffer != frame.shown_framebuffer || draw.flags & draw_flags::TEXTURED == 0 {
                continue;
            }
            let texture = draw.texture as usize;
            let known = texture < frame.textures().len();
            if known && !self.textures[texture].is_null() {
                continue;
            }
            if first_missing.is_none() && known {
                first_missing = Some(&frame.textures()[texture]);
            }
            missing += 1;
            if known && self.waiting[texture] != 0 {
                waiting_draws += 1;
            }
        }
        if missing != 0 {
            render_log::line(format_args!(
                "frame {}: {} draws without texture, {} waiting (textures {} us, {} us after loop start)\n",
                render_count,
                missing,
                waiting_draws,
                textures_us,
                (now() as u32).wrapping_sub(LOOP_START_US.load(Ordering::Relaxed))
            ));
            if let Some(t) = first_missing {
                render_log::line(format_args!(
                    "  first: addr={:08x} fmt={} {}x{} bufw={} swz={} clut={:06x}\n",
                    t.address, t.format, t.width, t.height, t.buffer_width, t.swizzled, t.clut_format
                ));
            }
        }
        textures::frame_done(waiting_draws != 0);
        waiting_draws != 0
    }

    /// Sampled targets not holding the frame just before this one (see
    /// `STALE_COPY`). A redraw of the same frame (paused, held)
    /// leaves them alone: the one it draws into holds this frame already.
    fn refresh_stale_targets(&self, targets: &mut Targets, frame: &Frame, target: usize) {
        let stale_copy = STALE_COPY.load(Ordering::Relaxed) != 0;
        for texture in self.textures[..frame.textures().len()].iter() {
            for k in 0..TARGET_COUNT {
                let sampled = &targets.targets[k];
                let is_sampled = core::ptr::eq(*texture, &sampled.texture) || core::ptr::eq(*texture, &sampled.opaque_view);
                if k == target || !is_sampled || sampled.frame_number.wrapping_add(1) == frame.frame_number {
                    continue;
                }
                stats().stale_targets += 1;
                let (sampled_frame, drawn) = (sampled.frame_number, &targets.targets[target]);
                let newer = drawn.frame_number.wrapping_sub(sampled_frame) as i32 > 0;
                if stale_copy
                    && drawn.psp_address == frame.shown_framebuffer
                    && !drawn.needs_clear
                    && drawn.frame_number != frame.frame_number
                    && newer
                    && targets.copy(target, k)
                {
                    stats().stale_copies += 1;
                }
            }
        }
    }

    /// The draw loop. Consecutive PSP draws that differ only in their
    /// vertices are already one draw here (the core merges them,
    /// Frame::finish_draw; Final Fantasy's menu: ~1200 glyph draws per
    /// frame), so each becomes one sceGxmDraw.
    fn draw_draws(&mut self, frame: &Frame) {
        let skip = REDRAW_SKIP.load(Ordering::Relaxed);
        let only_first = DRAW_ONLY_FIRST.load(Ordering::Relaxed);
        let only_count = DRAW_ONLY_COUNT.load(Ordering::Relaxed);
        let texture_count = frame.textures().len();
        let mut alpha_read = false;
        for (i, draw) in frame.draws().iter().enumerate() {
            if draw.blend & BLEND_ENABLED != 0 && blend::reads_destination_alpha(draw.blend) {
                alpha_read = true;
            }
            // Off-screen rendering isn't supported yet.
            if draw.framebuffer != frame.shown_framebuffer || skip & 32 != 0 {
                continue;
            }
            if only_count != 0 && (i < only_first as usize || i - only_first as usize >= only_count as usize) {
                continue;
            }
            draw::draw_state(draw); // scissor, depth, cull
            if skip & 16 != 0 {
                continue;
            }
            if draw.flags & draw_flags::TEXTURED != 0 {
                let t = draw.texture as usize;
                if t < texture_count {
                    // Safe: textures from the cache or targets, alive this frame.
                    if let Some(texture) = unsafe { self.textures[t].as_ref() } {
                        draw::draw_textured(frame, draw, i, texture, self.palettes[t], self.uv_scales[t], self.opaque_textures[t]);
                    }
                }
            } else {
                draw::draw_colored(frame, draw, i);
            }
        }
        if alpha_read {
            self.target_alpha_read = self.render_count;
        }
        stats().gxm_draws += frame.draw_count;
    }
}

/// Pictures the CPU wrote into a GE framebuffer.
///
/// MTX Mototrax's loading screens: the game writes a picture into both
/// framebuffers with the CPU and then only draws a progress bar with the GE
/// each frame. We never see the picture (no GE draws, no transfers), so our
/// target kept the last race frame under the bar. PPSSPP learns of such
/// writes from its HLE hooks (memcpy, DMA, video: FramebufferManagerCommon::
/// UpdateFromMemory, NotifyFramebufferCopy); we can't hook the game's own
/// code, but we have Sony's finished picture: for a frame whose draws leave
/// parts of the screen untouched, those parts must look the same in our
/// target and in Sony's picture. If most of them don't, something other
/// than the GE changed them, and the frame is shown from Sony's picture
/// (like CPU-drawn movies) until a frame covers the screen again.
///
/// True if the frame's untouched parts show a picture the CPU wrote.
fn shows_cpu_written_picture(frame: &Frame, targets: &Targets) -> bool {
    /// The screen in cells; a cell any draw's box touches counts as covered.
    const COLUMNS: usize = 16;
    const ROWS: usize = 8;
    /// Only frames with at most this many draws are checked (a loading
    /// screen, a menu over a still background); busier ones cover the screen.
    const MAX_DRAWS: u32 = 64;
    if CPU_PICTURE_CHECK.load(Ordering::Relaxed) == 0 || frame.draw_count > MAX_DRAWS {
        return false;
    }
    let target = targets.find(frame.shown_framebuffer).map(|k| &targets.targets[k]);
    // A target that holds nothing of ours yet is black where we draw
    // nothing: compared as black (display_loop.rs shows_cpu_written_picture;
    // MTX's loading screen flickered when one of its two framebuffers had no
    // picture of ours yet).
    let ours = target.filter(|target| !target.needs_clear);
    let mut covered = [[false; COLUMNS]; ROWS];
    let (cell_width, cell_height) = (480.0 / COLUMNS as f32, 272.0 / ROWS as f32);
    for draw in frame.draws() {
        if draw.framebuffer != frame.shown_framebuffer {
            continue;
        }
        if draw.transform != 0 || draw.flags & draw_flags::CACHED_VERTICES != 0 {
            return false; // 3D: may cover anything; such frames redraw the scene
        }
        let first = draw.first_vertex as usize;
        let vertices = frame.vertices.get(first..first + draw.vertex_count as usize).unwrap_or(&[]);
        let (mut left, mut top, mut right, mut bottom) = (1e9f32, 1e9f32, -1e9f32, -1e9f32);
        for vertex in vertices {
            left = left.min(vertex.x);
            right = right.max(vertex.x);
            top = top.min(vertex.y);
            bottom = bottom.max(vertex.y);
        }
        for (row, cells) in covered.iter_mut().enumerate() {
            for (column, cell) in cells.iter_mut().enumerate() {
                let (x0, y0) = (column as f32 * cell_width, row as f32 * cell_height);
                if right >= x0 && left <= x0 + cell_width && bottom >= y0 && top <= y0 + cell_height {
                    *cell = true;
                }
            }
        }
    }
    let mut sony = [0u32; ROWS * COLUMNS];
    let mut ours_samples = [0u32; ROWS * COLUMNS];
    let mut samples = 0;
    for (row, cells) in covered.iter().enumerate() {
        for (column, _) in cells.iter().enumerate().filter(|(_, covered)| !**covered) {
            let x = ((column as f32 + 0.5) * cell_width) as usize;
            let y = ((row as f32 + 0.5) * cell_height) as usize;
            // Safe: inside Sony's picture (512 x 272 words, mapped).
            sony[samples] = unsafe { core::ptr::read_volatile(SONY_PICTURE.add(y * SONY_PICTURE_STRIDE + x)) };
            ours_samples[samples] = ours.map_or(0, |target| target.pixel((x as f32 * SCALE) as usize, (y as f32 * SCALE) as usize));
            samples += 1;
        }
    }
    let ours_samples = ours.is_some().then_some(&ours_samples[..samples]);
    nativeres_core::display_loop::shows_cpu_written_picture(&sony[..samples], ours_samples)
}

/// Every target can be a texture's (`FrameInfo::render_targets`).
const _: () = assert!(TARGET_COUNT <= MAX_RENDER_TARGETS);
