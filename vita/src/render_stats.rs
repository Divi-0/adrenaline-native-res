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
    /// Frames drawn in several scenes for off-screen targets (frame_loop.rs
    /// `draw_runs`).
    pub offscreen_frames: u32,
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
    /// Of those, with the point-light programs (RAW_LIT1_POINT, RAW_LIT4_POINT).
    pub point_light_draws: u32,
    /// Raw draws with a fixed-colour program (draw.rs `RAW_FIXED_COLOR`).
    pub fixed_color_draws: u32,
    /// Draws blended with FIX B through the alpha (BLEND_FIX programs).
    pub fix_b_draws: u32,
    /// Exact stencil states (draw.rs `stencil_way`): value by a BLEND_FIX
    /// program, 0 by GXM's alpha factors, the generic frame-read program,
    /// and states left approximated for want of that program.
    pub stencil_fix_draws: u32,
    pub stencil_zero_draws: u32,
    pub stencil_read_draws: u32,
    pub stencil_missing_draws: u32,
    /// Textured draws whose alpha test has a mask other than 0xFF (done by
    /// the generic frame-read program, draw.rs `masked_alpha_test_uniform`).
    pub masked_alpha_tests: u32,
    /// Of those, the draws GXM blends that took the alpha-test program with
    /// the mask (draw.rs `MASKED_TEST`) instead of the generic one.
    pub mask_test_draws: u32,
    /// Textured draws that took the program without the specular part
    /// (draw.rs `specular_part`, `Kind::TexturedNoSpec`).
    pub nospec_draws: u32,
    /// Draws into 16-bit PSP targets with the frame-read programs
    /// (draw.rs `TARGET_CUT`).
    pub cut_draws: u32,
    /// Of them, with a fast frame-read program (core frame.rs
    /// `read_variant`).
    pub cut_fast_draws: u32,
    /// Of those, without a frame read (unblended, or `ReadWay::Precut`).
    pub cut_off_draws: u32,
    /// Draws into 8-bit targets blended as the GE does (draw.rs
    /// `BLEND_8888`).
    pub wide_draws: u32,
    /// Normal alpha blends drawn unblended: every fragment opaque (draw.rs
    /// `OPAQUE_SOURCES`, core `opaque_source`).
    pub opaque_draws: u32,
    /// Alpha-tested draws whose test was dropped: every fragment opaque
    /// (draw.rs `OPAQUE_UNTESTED`).
    pub opaque_untested_draws: u32,
    /// Draws found opaque by the rows of the texture they sample (draw.rs
    /// `samples_opaque_rows`), not the whole texture.
    pub opaque_rows_draws: u32,
    /// Raw draws whose used vertices the opaque-source check read (`draw::OPAQUE_USED_ONLY`).
    pub opaque_checked_draws: u32,
    /// Alpha-tested draws drawn without the test: it passes every alpha their
    /// fragments can have (`draw::LOWEST_UNTESTED`).
    pub lowest_untested_draws: u32,
    /// Draws left out: a later draw paints over them (frame_loop.rs
    /// `HIDE_DRAWS`), and the time finding them.
    pub hidden_draws: u32,
    /// Draws drawn for their depths alone (core `hidden_draws`).
    pub depth_only_draws: u32,
    pub hidden_us: u64,
    /// Textured draws through the flat program (draw.rs `FLAT_SHADING`).
    pub flat_draws: u32,
    /// Flat draws of one vertex colour through the constant-colour program
    /// (draw.rs `CONST_COLOR`).
    pub const_draws: u32,
    /// Draws without depth test through their program's sibling without the
    /// PSP depth output (draw.rs `NO_DEPTH_TWINS`).
    pub nodepth_draws: u32,
    /// Draws in the GE's slope level mode through a SLOPE_LOD sibling (setting
    /// `slopemip`), and those without one (level 0).
    pub slope_draws: u32,
    pub slope_missed: u32,
    /// Textured draws reading one of our targets through the GE_FILTER
    /// programs (the GE's bilinear filter; core `ge_filter_read`).
    pub ge_filter_draws: u32,
    /// GE-filtered draws of a wide texture (core `GpuFormat::Wide16`)
    /// through the GE_WIDE twin of their program (draw.rs; DESIGN
    /// "sixty-fifth session"); ...
    pub ge_wide_draws: u32,
    /// ... ones whose program has no usable twin (drawn by the four point
    /// samples: the same pixels, slower);
    pub ge_wide_missed: u32,
    /// ... and wide textures drawn by a program without the GE filter with
    /// the GPU's LINEAR filter: its 16-bit lerps aren't the 8-bit texture's
    /// (should stay 0 in God of War, whose set filters every texture as the GE).
    pub ge_wide_inexact: u32,
    /// GE-filtered draws through the GE_GATHER twin of their program
    /// (draw.rs `GE_GATHER`; DESIGN "seventy-fourth session"), and ones whose
    /// program has no usable twin (four point samples: the same pixels).
    pub ge_gather_draws: u32,
    pub ge_gather_missed: u32,
    /// Textures from off-screen targets drawn wide (wide_targets.rs, DESIGN
    /// "sixty-sixth session"): textures given a wide copy, copies made
    /// and the copy scenes that made them.
    pub wide_target_textures: u32,
    pub wide_copies: u32,
    pub wide_copy_scenes: u32,
    /// Draws onto the screen reading black texels with a blend that keeps
    /// the target there (core black_texels, DESIGN "sixty-seventh
    /// session"), their triangles left out and kept, and the draws left
    /// out whole.
    pub black_draws: u32,
    pub black_culled: u32,
    pub black_kept: u32,
    pub black_skipped: u32,
    /// Blended draws with triangles whose three corners have alpha 0 (core
    /// zero_alpha, DESIGN "seventy-eighth session"), those triangles left
    /// out and the others kept, and the draws left out whole.
    pub alpha0_draws: u32,
    pub alpha0_culled: u32,
    pub alpha0_kept: u32,
    pub alpha0_skipped: u32,
    /// The draws whose vertex alphas were read for it, and the display
    /// thread's time in it (with `drawtiming 1` only, else 0).
    pub alpha0_read: u32,
    pub alpha0_us: u64,
    /// The copies out of PSP RAM made for it (`draw::VertexWindow::fill`)
    /// and their bytes.
    pub alpha0_fills: u32,
    pub alpha0_fill_bytes: u32,
    /// Draws read with a texel test (`draw::texel_test`), and the triangles
    /// it left out (counted in `alpha0_culled` too).
    pub texel0_draws: u32,
    pub texel0_culled: u32,
    /// Strip draws whose cull came out as a triangle list (the kept
    /// triangles scattered: the strip would have been longer).
    pub alpha0_as_list: u32,
    /// Raw draws drawn as one triangle strip (`stripdraws`), their indices
    /// and the triangle-list corners they replaced.
    pub strip_draws: u32,
    pub strip_indices: u32,
    pub strip_corners: u32,
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
        offscreen_frames: 0,
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
        point_light_draws: 0,
        fixed_color_draws: 0,
        raw_no_program: 0,
        fix_b_draws: 0,
        stencil_fix_draws: 0,
        stencil_zero_draws: 0,
        stencil_read_draws: 0,
        stencil_missing_draws: 0,
        masked_alpha_tests: 0,
        mask_test_draws: 0,
        nospec_draws: 0,
        cut_draws: 0,
        cut_fast_draws: 0,
        wide_draws: 0,
        cut_off_draws: 0,
        opaque_draws: 0,
        opaque_untested_draws: 0,
        opaque_rows_draws: 0,
        opaque_checked_draws: 0,
        lowest_untested_draws: 0,
        hidden_draws: 0,
        depth_only_draws: 0,
        hidden_us: 0,
        flat_draws: 0,
        const_draws: 0,
        nodepth_draws: 0,
        slope_draws: 0,
        slope_missed: 0,
        ge_filter_draws: 0,
        ge_wide_draws: 0,
        ge_wide_missed: 0,
        ge_wide_inexact: 0,
        ge_gather_draws: 0,
        ge_gather_missed: 0,
        wide_target_textures: 0,
        wide_copies: 0,
        wide_copy_scenes: 0,
        black_draws: 0,
        black_culled: 0,
        black_kept: 0,
        black_skipped: 0,
        alpha0_draws: 0,
        alpha0_culled: 0,
        alpha0_kept: 0,
        alpha0_skipped: 0,
        alpha0_read: 0,
        alpha0_us: 0,
        alpha0_fills: 0,
        alpha0_fill_bytes: 0,
        texel0_draws: 0,
        texel0_culled: 0,
        alpha0_as_list: 0,
        strip_draws: 0,
        strip_indices: 0,
        strip_corners: 0,
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
        "begin={}us loop={}us convert={}us indices={}us pool_kb={} pool_waits={} alpha_tested={} alpha_test_skipped={} ring_draws={} plain_draws={} lit_variant_draws={} flat_light_draws={} point_light_draws={} fixed_color_draws={} fix_b_draws={} cut_draws={} cut_fast_draws={} cut_off_draws={} wide_draws={} opaque_draws={} opaque_untested_draws={} opaque_rows_draws={} opaque_checked_draws={} lowest_untested_draws={} hidden_draws={} depth_only_draws={} hidden={}us flat_draws={} const_draws={} nodepth_draws={} slope_draws={} slope_missed={} ge_filter_draws={} ge_wide_draws={} ge_wide_missed={} ge_wide_inexact={} ge_gather_draws={} ge_gather_missed={} wide_target_textures={} wide_copies={} wide_copy_scenes={} black_draws={} black_culled={} black_kept={} black_skipped={} alpha0_draws={} alpha0_culled={} alpha0_kept={} alpha0_skipped={} alpha0_read={} alpha0={}us alpha0_fills={} alpha0_fill_bytes={} texel0_draws={} texel0_culled={} alpha0_as_list={} strip_draws={} strip_indices={} strip_corners={} opaque={}us ",
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
        s.point_light_draws / frames,
        s.fixed_color_draws / frames,
        s.fix_b_draws / frames,
        s.cut_draws / frames,
        s.cut_fast_draws / frames,
        s.cut_off_draws / frames,
        s.wide_draws / frames,
        s.opaque_draws / frames,
        s.opaque_untested_draws / frames,
        s.opaque_rows_draws / frames,
        s.opaque_checked_draws / frames,
        s.lowest_untested_draws / frames,
        s.hidden_draws / frames,
        s.depth_only_draws / frames,
        per_frame(s.hidden_us),
        s.flat_draws / frames,
        s.const_draws / frames,
        s.nodepth_draws / frames,
        s.slope_draws / frames,
        s.slope_missed / frames,
        s.ge_filter_draws / frames,
        s.ge_wide_draws / frames,
        s.ge_wide_missed / frames,
        s.ge_wide_inexact,
        s.ge_gather_draws / frames,
        s.ge_gather_missed / frames,
        s.wide_target_textures / frames,
        s.wide_copies / frames,
        s.wide_copy_scenes / frames,
        s.black_draws / frames,
        s.black_culled / frames,
        s.black_kept / frames,
        s.black_skipped / frames,
        s.alpha0_draws / frames,
        s.alpha0_culled / frames,
        s.alpha0_kept / frames,
        s.alpha0_skipped / frames,
        s.alpha0_read / frames,
        per_frame(s.alpha0_us),
        s.alpha0_fills / frames,
        s.alpha0_fill_bytes / frames,
        s.texel0_draws / frames,
        s.texel0_culled / frames,
        s.alpha0_as_list / frames,
        s.strip_draws / frames,
        s.strip_indices / frames,
        s.strip_corners / frames,
        per_frame(s.opaque_us)
    );
    let _ = write!(
        text,
        "stale_targets={} stale_copies={} seeded={} offscreen_frames={} lookup={}us tex_begin={}us tex_hash={}us palette={}us tex_finds={} tex_scans={} end={}us",
        s.stale_targets,
        s.stale_copies,
        s.seeded_targets,
        s.offscreen_frames,
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
    let _ = write!(
        text,
        " stencil_fix_draws={} stencil_zero_draws={} stencil_read_draws={} stencil_missing_draws={} masked_alpha_tests={} mask_test_draws={} nospec_draws={}",
        s.stencil_fix_draws / frames,
        s.stencil_zero_draws / frames,
        s.stencil_read_draws / frames,
        s.stencil_missing_draws / frames,
        s.masked_alpha_tests / frames,
        s.mask_test_draws / frames,
        s.nospec_draws / frames
    );
    let _ = write!(text, " ");
    crate::heap::report(&mut text);
    let length = text.len();
    out[length] = 0;
    length as i32
}