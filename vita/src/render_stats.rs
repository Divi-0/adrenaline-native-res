//! Where the display thread's time goes; read and reset by `report`.

use core::fmt::Write;

use nativeres_core::report::TextBuffer;

use crate::{frame_pool, palettes, sys, textures, SingleThreaded};

#[derive(Clone, Copy)]
pub struct RenderStats {
    pub frames: u32,
    /// The whole NativeResRenderFrame.
    pub total_us: u64,
    /// The texture cache (hashing + decoding) and palettes.
    pub textures_us: u64,
    /// Palette tables filled.
    pub palettes: u32,
    /// With MEASURE_GPU: GPU time until the target scene is done.
    pub gpu_us: u64,
    pub draws: u32,
    /// sceGxmDraw calls (PSP draws merged by the core, Frame::finish_draw).
    pub gxm_draws: u32,
    /// Shown as the game's own picture.
    pub fallback_frames: u32,
    /// Framebuffer outside VRAM: the game's own picture.
    pub cpu_frames: u32,
    /// Textures taken from our render targets (render to texture).
    pub target_textures: u32,
    /// A target reused although drawn into in the last 2 frames.
    pub target_evictions: u32,
    /// Sampled targets not holding the frame just before.
    pub stale_targets: u32,
    /// Of them, refreshed from the newer target.
    pub stale_copies: u32,
    /// Targets filled from Sony's picture.
    pub seeded_targets: u32,
    /// Frames left out: textures rewritten after the GE drew them.
    pub moved_on_frames: u32,
    /// sceGxmBeginScene (can wait for the GPU).
    pub begin_us: u64,
    /// The draw loop.
    pub loop_us: u64,
    /// In it: vertices and indices copied into the frame pool
    /// (`draw::DRAW_TIMING` only).
    pub convert_us: u64,
    /// Of that: the index copies.
    pub indices_us: u64,
    /// Frame pool bytes written, in KiB.
    pub pool_kb: u32,
    /// Frame pool full: waited for the GPU.
    pub pool_waits: u32,
    /// Textured draws with an alpha test.
    pub alpha_tested: u32,
    /// Of them, drawn without the discard program.
    pub alpha_test_skipped: u32,
    /// Draws whose indices came from the index ring.
    pub ring_draws: u32,
    /// Raw draws with the plain program (no skinning, no lights).
    pub plain_draws: u32,
    /// Raw draws with the lit one- or four-bone programs (RAW_LIT1, RAW_LIT4).
    pub lit_variant_draws: u32,
    /// Of those, with the flat-light programs (RAW_LIT1_FLAT, RAW_LIT4_FLAT).
    pub flat_light_draws: u32,
    /// Draws blended with FIX B through the alpha (BLEND_FIX programs).
    pub fix_b_draws: u32,
    /// Draws into 16-bit PSP targets with the frame-read programs
    /// (draw.rs `TARGET_CUT`).
    pub cut_draws: u32,
    /// Of them, with a fast frame-read program (core frame.rs
    /// `read_variant`).
    pub cut_fast_draws: u32,
    /// Of those, without a frame read (unblended, or `ReadWay::Precut`).
    pub cut_off_draws: u32,
    /// Normal alpha blends drawn unblended: every fragment opaque (draw.rs
    /// `OPAQUE_SOURCES`, core `opaque_source`).
    pub opaque_draws: u32,
    /// Alpha-tested draws whose test was dropped: every fragment opaque
    /// (draw.rs `OPAQUE_UNTESTED`).
    pub opaque_untested_draws: u32,
    /// The display thread's time finding them (vertex alphas read).
    pub opaque_us: u64,
    /// Raw draws left out: no vertex program for their format (all
    /// `programs::RAW_PROGRAMS` places taken, or the shaders not ready).
    pub raw_no_program: u32,
    /// Of textures_us: the cache's lookups.
    pub lookup_us: u64,
    /// Of textures_us: palette tables for the frame's index textures.
    pub palette_us: u64,
    /// Of lookup_us: the cache's begin_frame.
    pub begin_tex_us: u64,
    /// Of lookup_us: hashing texture data.
    pub hash_tex_us: u64,
    /// sceGxmEndScene.
    pub end_us: u64,
}

impl RenderStats {
    const ZERO: RenderStats = RenderStats {
        frames: 0,
        total_us: 0,
        textures_us: 0,
        palettes: 0,
        gpu_us: 0,
        draws: 0,
        gxm_draws: 0,
        fallback_frames: 0,
        cpu_frames: 0,
        target_textures: 0,
        target_evictions: 0,
        stale_targets: 0,
        stale_copies: 0,
        seeded_targets: 0,
        moved_on_frames: 0,
        begin_us: 0,
        loop_us: 0,
        convert_us: 0,
        indices_us: 0,
        pool_kb: 0,
        pool_waits: 0,
        alpha_tested: 0,
        alpha_test_skipped: 0,
        ring_draws: 0,
        plain_draws: 0,
        lit_variant_draws: 0,
        flat_light_draws: 0,
        raw_no_program: 0,
        fix_b_draws: 0,
        cut_draws: 0,
        cut_fast_draws: 0,
        cut_off_draws: 0,
        opaque_draws: 0,
        opaque_untested_draws: 0,
        opaque_us: 0,
        lookup_us: 0,
        palette_us: 0,
        begin_tex_us: 0,
        hash_tex_us: 0,
        end_us: 0,
    };
}

/// The display thread's counters (and `report`'s read-and-reset from
/// another thread, racy as ever: counters only).
static STATS: SingleThreaded<RenderStats> = SingleThreaded::new(RenderStats::ZERO);

/// The display thread's stats. Hold the reference only briefly (no call
/// that may take it again in between).
pub fn stats() -> &'static mut RenderStats {
    // Safe: the display thread's (see the static).
    unsafe { STATS.get() }
}

/// Render timing since the last call, as text into
/// `out` (`size` bytes, NUL-terminated); resets the counters. Returns the
/// length.
///
/// # Safety
/// `out` must hold `size` bytes.
pub unsafe fn report(out: *mut u8, size: i32) -> i32 {
    if out.is_null() || size <= 0 {
        return 0;
    }
    // The display thread may be counting meanwhile (racy, as in C before).
    let shared = STATS.get() as *mut RenderStats;
    let mut s = shared.read_volatile();
    shared.write_volatile(RenderStats::ZERO);
    s.pool_waits = frame_pool::take_waits();
    s.palettes = palettes::take_filled();
    let t = textures::take_counters();
    let frames = s.frames.max(1);
    // Per frame, as the C version printed them (unsigned long: 32 bits).
    let per_frame = |value: u64| (value / frames as u64) as u32;
    let mut memory = sys::SceKernelFreeMemorySizeInfo {
        size: core::mem::size_of::<sys::SceKernelFreeMemorySizeInfo>() as i32,
        size_user: 0,
        size_cdram: 0,
        size_phycont: 0,
    };
    sys::sceKernelGetFreeMemorySize(&mut memory);

    let out = core::slice::from_raw_parts_mut(out, size as usize);
    let mut text = TextBuffer::new(&mut out[..size as usize - 1]);
    let _ = write!(
        text,
        "frames={} avg_total={}us avg_textures={}us avg_decode={}us decodes={} palettes={} refreshes={} avg_refresh={}us ",
        s.frames,
        per_frame(s.total_us),
        per_frame(s.textures_us),
        t.decode_us / frames,
        t.decodes,
        s.palettes,
        t.refreshes,
        if t.refreshes != 0 { t.refresh_us / t.refreshes } else { 0 }
    );
    let _ = write!(
        text,
        "hashed_kb_per_frame={} draws_per_frame={} gxm_draws_per_frame={} ",
        t.hashed_bytes / 1024 / frames,
        s.draws / frames,
        s.gxm_draws / frames
    );
    let _ = write!(
        text,
        "texture_kb={} arena_kb={} evictions={} create_failures={} fallback_frames={} cpu_frames={} target_textures={} free_user_kb={} free_cdram_kb={} target_evictions={} gpu={}us ",
        t.texture_bytes / 1024,
        t.arena_bytes / 1024,
        t.evictions,
        t.create_failures,
        s.fallback_frames,
        s.cpu_frames,
        s.target_textures,
        memory.size_user / 1024,
        memory.size_cdram / 1024,
        s.target_evictions,
        per_frame(s.gpu_us)
    );
    let _ = write!(
        text,
        "begin={}us loop={}us convert={}us indices={}us pool_kb={} pool_waits={} alpha_tested={} alpha_test_skipped={} ring_draws={} plain_draws={} lit_variant_draws={} flat_light_draws={} fix_b_draws={} cut_draws={} cut_fast_draws={} cut_off_draws={} opaque_draws={} opaque_untested_draws={} opaque={}us ",
        per_frame(s.begin_us),
        per_frame(s.loop_us),
        per_frame(s.convert_us),
        per_frame(s.indices_us),
        s.pool_kb / frames,
        s.pool_waits,
        s.alpha_tested / frames,
        s.alpha_test_skipped / frames,
        s.ring_draws / frames,
        s.plain_draws / frames,
        s.lit_variant_draws / frames,
        s.flat_light_draws / frames,
        s.fix_b_draws / frames,
        s.cut_draws / frames,
        s.cut_fast_draws / frames,
        s.cut_off_draws / frames,
        s.opaque_draws / frames,
        s.opaque_untested_draws / frames,
        per_frame(s.opaque_us)
    );
    let _ = write!(
        text,
        "stale_targets={} stale_copies={} seeded={} lookup={}us tex_begin={}us tex_hash={}us palette={}us tex_finds={} tex_scans={} end={}us",
        s.stale_targets,
        s.stale_copies,
        s.seeded_targets,
        per_frame(s.lookup_us),
        per_frame(s.begin_tex_us),
        per_frame(s.hash_tex_us),
        per_frame(s.palette_us),
        t.finds / frames,
        t.scans / frames,
        per_frame(s.end_us)
    );
    if s.raw_no_program > 0 {
        let _ = write!(text, " raw_no_program={}", s.raw_no_program);
    }
    let _ = write!(text, " ");
    crate::heap::report(&mut text);
    let length = text.len();
    out[length] = 0;
    length as i32
}