//! When Adrenaline's display loop (Adrenaline/user/menu.c) copies Sony's
//! picture of the PSP screen, and a timing model of what that copy shows.
//!
//! For frames the GE didn't draw (MTX's intro movie, loading screens) the
//! screen shows Sony's picture (`SCE_PSPEMU_FRAMEBUFFER`), which the loop
//! copies with one DMA and draws. Sony rewrites that picture top to bottom
//! across the whole PSP frame, starting at the LCD sync: measured during
//! MTX's intro movie (2026-11-10), rows 0-31 changed
//! 0-3 ms after the sync, rows 240-271 14-17 ms after it. A copy at time `t`
//! after the sync therefore holds the new picture above the row being
//! written at `t` and the old one below it: a seam at that row.
//!
//! Copied after `NativeResRenderFrame` (which may wait up to 5 ms for a
//! frame of ours, `FRAME_WAIT_US`, and is preempted by the capture threads),
//! the copy came 1-7 ms after the sync in more than half the loops
//! (during the movie: 179 of 359 loops 5-7 ms after it): a seam ~90 rows down, "split horizontally in the top section"
//! (user). Copied right after the sync (as the stock loop does), it came
//! within 1 ms (360 of 360 copies): a seam in the top 17 rows at most.

/// One PSP frame (59.94 Hz), the period of Sony's top-to-bottom rewrite.
pub const PSP_FRAME_US: u32 = 16_683;

/// Rows of the PSP screen.
pub const PSP_ROWS: u32 = 272;

/// How long after the LCD sync the loop's early copy has started
/// (measured: all copies in the 0-1 ms bucket).
pub const EARLY_COPY_US: u32 = 1_000;

/// A seam above this row is hidden at the screen's top edge (the first
/// ~6 %: the game's own picture has it there too, and the webcam tear count
/// of the early copy matched Sony's own presentation).
pub const HIDDEN_SEAM_ROWS: u32 = 17;

/// Where the loop copies Sony's picture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CopyPoint {
    /// Right after the LCD sync, before `NativeResRenderFrame`.
    AfterSync,
    /// Only once `NativeResRenderFrame` said the loop shows Sony's picture
    /// (needed then; when our frame is shown, no copy at all).
    AfterRender,
}

/// Where to copy: right after the sync while the screen shows Sony's
/// picture (it was shown in the loop before: CPU-drawn pictures come in
/// runs), otherwise only if `NativeResRenderFrame` falls back to it.
pub fn sony_copy_point(showed_sony_picture_last: bool) -> CopyPoint {
    if showed_sony_picture_last {
        CopyPoint::AfterSync
    } else {
        CopyPoint::AfterRender
    }
}

/// The row a copy starting `copy_at_us` after the LCD sync splits Sony's
/// picture at (0: no seam; `PSP_ROWS`: the whole old picture, no seam
/// either).
pub fn seam_row(copy_at_us: u32) -> u32 {
    let into_frame = copy_at_us % PSP_FRAME_US;
    (into_frame as u64 * PSP_ROWS as u64 / PSP_FRAME_US as u64) as u32
}

/// Whether a copy starting `copy_at_us` after the sync shows no visible
/// seam (above `HIDDEN_SEAM_ROWS`, or none).
pub fn copy_is_clean(copy_at_us: u32) -> bool {
    let row = seam_row(copy_at_us);
    row < HIDDEN_SEAM_ROWS || row >= PSP_ROWS
}

/// When a copy at `point` starts, given when `NativeResRenderFrame` returned
/// (`render_end_us` after the sync).
pub fn copy_time_us(point: CopyPoint, render_end_us: u32) -> u32 {
    match point {
        CopyPoint::AfterSync => EARLY_COPY_US,
        CopyPoint::AfterRender => render_end_us,
    }
}

/// What Adrenaline's shared words say about the PSP screen, read every pass
/// of the display loop.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AdrenalineScreen {
    /// A PS1 game runs (`SceAdrenaline::pops_mode`).
    pub pops_mode: bool,
    /// Homebrew draws at the Vita's resolution itself (DRAW_NATIVE).
    pub draw_native: bool,
    /// A savestate is taken or loaded ("Please wait...").
    pub savestate_screen: bool,
    /// The PSP's HOME menu ("Quit the game?") is open
    /// (`SceAdrenaline::draw_psp_screen_in_pops`: set by Adrenaline's
    /// PAUSE_POPS, which its impose patch sends for every game when the
    /// popup opens, cleared by RESUME_POPS when it closes).
    pub home_menu: bool,
    /// The PSP runs its XMB, no game (`SceAdrenaline::app_type` is
    /// PSP_INIT_KEYCONFIG_VSH: set at every PSP boot, so after "Quit the
    /// game?" > Yes).
    pub xmb: bool,
}

/// Whether the loop may show our picture: none of the above. The HOME menu
/// is drawn by the PSP's kernel (impose), never through the game's display
/// lists we capture: with our picture shown it stayed invisible (user,
/// 2026-11-18), so Sony's picture is shown while it is open. In the XMB
/// no game draws: our last frame of the quit game stayed frozen on the
/// screen (user, 2026-11-18).
pub fn ours_to_draw(screen: AdrenalineScreen) -> bool {
    !(screen.pops_mode || screen.draw_native || screen.savestate_screen || screen.home_menu || screen.xmb)
}

/// A sample of the CPU-picture check differs when its R + G + B differ by
/// more than this (0..765): far above what upscaling and filtering change.
pub const CPU_PICTURE_DIFFERENCE: u32 = 96;

/// A sample also differs when ours is black (R + G + B below
/// `CPU_PICTURE_OURS_BLACK`) and Sony's is not (at least
/// `CPU_PICTURE_SONY_LIT`): a dark picture missing from our target. MTX's
/// boot loading screen is dimmed art (R + G + B ~25..120) that stays under
/// CPU_PICTURE_DIFFERENCE against black in most cells. Above dithered black
/// (a few levels per channel), and only against our black: dark against our
/// own dark picture is still judged by CPU_PICTURE_DIFFERENCE.
pub const CPU_PICTURE_OURS_BLACK: u32 = 16;
pub const CPU_PICTURE_SONY_LIT: u32 = 32;

/// Whether a frame with few draws shows a picture the CPU wrote (loading
/// screens, movies): the screen cells no draw touches, Sony's picture
/// (`sony`) against ours (`ours`; `None` when our target holds nothing yet,
/// new or cleared: black). Then the display shows Sony's picture for it
/// (nativeres_render.c ShowsCpuWrittenPicture).
///
/// MTX's boot loading screen (2026-11-11): the game writes the picture once
/// into both framebuffers and draws only the progress bar, alternating
/// between them. With "our target holds nothing" counting as "don't know",
/// every frame into the still empty target was drawn by us: black with the
/// bar, every other frame (flicker). An empty target is black, which is
/// what we would show.
///
/// 2026-11-18: that picture is dark, so a dark sample over our black counts
/// as different too (`CPU_PICTURE_OURS_BLACK`).
pub fn shows_cpu_written_picture(sony: &[u32], ours: Option<&[u32]>) -> bool {
    const BLACK: u32 = 0xFF00_0000;
    let difference = |a: u32, b: u32| (0..3).map(|channel| ((a >> (8 * channel)) & 0xFF).abs_diff((b >> (8 * channel)) & 0xFF)).sum::<u32>();
    let differs = |sony: u32, ours: u32| {
        difference(sony, ours) > CPU_PICTURE_DIFFERENCE
            || (difference(ours, BLACK) < CPU_PICTURE_OURS_BLACK && difference(sony, BLACK) >= CPU_PICTURE_SONY_LIT)
    };
    let differing = sony
        .iter()
        .enumerate()
        .filter(|&(index, &pixel)| differs(pixel, ours.map_or(BLACK, |ours| ours.get(index).copied().unwrap_or(BLACK))))
        .count();
    sony.len() >= 8 && 2 * differing >= sony.len()
}

#[cfg(test)]
mod tests {
    /// The boot loading screen: Sony shows the picture (here grey), ours
    /// has only the bar, in a target that holds nothing yet (black) or
    /// already shows our previous black frame: Sony's picture both times.
    #[test]
    fn a_cpu_picture_is_found_also_into_an_empty_target() {
        use super::shows_cpu_written_picture;
        let sony = [0xFF80_8080u32; 100];
        assert!(shows_cpu_written_picture(&sony, None), "empty target: compared with black");
        assert!(shows_cpu_written_picture(&sony, Some(&[0xFF00_0000; 100])));
        assert!(!shows_cpu_written_picture(&sony, Some(&[0xFF80_8080; 100])), "ours shows the same");
        assert!(!shows_cpu_written_picture(&[0xFF00_0000; 100], None), "Sony's is black too");
        assert!(!shows_cpu_written_picture(&sony[..4], None), "too few samples");
    }

    /// MTX's boot loading screen (2026-11-18): the game draws the picture
    /// (dimmed art + white text) once, while our renderer was still starting
    /// (its texture waited a frame, then the bar-only frames superseded it),
    /// so our targets hold only the bar. In Sony's picture 75 of the 120
    /// free cells were dark (R + G + B 2..96, the art dimmed), 45 brighter:
    /// against black, under half differed by more than CPU_PICTURE_DIFFERENCE
    /// and the screen stayed black with the bar. Dark against our black
    /// counts as different; dark against our own dark picture doesn't.
    #[test]
    fn a_dark_cpu_picture_is_found_over_an_empty_target() {
        use super::shows_cpu_written_picture;
        // 75 dimmed-art cells (R + G + B = 34 .. 92), 45 bright text cells.
        let mut sony = [0u32; 120];
        for (index, pixel) in sony.iter_mut().enumerate() {
            let level = if index < 75 { 12 + (index as u32 % 20) } else { 110 };
            *pixel = 0xFF00_0000 | (level - 4) << 16 | level << 8 | (level + 4);
        }
        assert!(shows_cpu_written_picture(&sony, None), "empty target: the dark picture is missing from ours");
        assert!(shows_cpu_written_picture(&sony, Some(&[0xFF00_0000; 120])), "ours black with the bar");
        assert!(!shows_cpu_written_picture(&sony, Some(&sony)), "ours shows the same dark picture");
        let ours_slightly_darker: [u32; 120] = core::array::from_fn(|index| {
            let level = (sony[index] >> 8 & 0xFF) * 3 / 4;
            0xFF00_0000 | level << 16 | level << 8 | level
        });
        assert!(!shows_cpu_written_picture(&sony, Some(&ours_slightly_darker)), "filtering differences, not a missing picture");
        let near_black = [0xFF03_0303u32; 120];
        assert!(!shows_cpu_written_picture(&near_black, None), "Sony's near black (dither) over our black");
    }

    use super::*;

    /// PS button in a PSP game, renderer on (2026-11-18): the HOME menu
    /// didn't show, our picture covered Sony's. With the menu open only
    /// Sony's picture is shown; closed, ours again.
    #[test]
    fn the_home_menu_shows_sonys_picture() {
        let game = AdrenalineScreen::default();
        assert!(ours_to_draw(game));
        assert!(!ours_to_draw(AdrenalineScreen { home_menu: true, ..game }));
        assert!(!ours_to_draw(AdrenalineScreen { pops_mode: true, ..game }));
        assert!(!ours_to_draw(AdrenalineScreen { draw_native: true, ..game }));
        assert!(!ours_to_draw(AdrenalineScreen { savestate_screen: true, ..game }));
    }

    /// "Quit the game?" > Yes, renderer on (2026-11-18): the PSP went back
    /// to the XMB, no more frames came, and our last one stayed on the
    /// screen. In the XMB, Sony's picture.
    #[test]
    fn the_xmb_after_quitting_shows_sonys_picture() {
        assert!(!ours_to_draw(AdrenalineScreen { xmb: true, ..AdrenalineScreen::default() }));
    }

    /// Measured during MTX's intro movie, renderer on (2026-11-10): how
    /// long after the LCD sync `NativeResRenderFrame` returned in loops that
    /// showed Sony's picture, by 1 ms bucket. Where the copy used to be.
    const RENDER_END_HISTOGRAM: [(u32, u32); 6] = [(0, 147), (1, 28), (2, 2), (4, 3), (5, 10), (6, 169)];

    /// The torn intro movie: copied after `NativeResRenderFrame`, most loops
    /// split the picture far below the top edge.
    #[test]
    fn copying_after_our_render_step_tore_the_movie() {
        let loops: u32 = RENDER_END_HISTOGRAM.iter().map(|&(_, count)| count).sum();
        let torn: u32 = RENDER_END_HISTOGRAM
            .iter()
            .filter(|&&(bucket_ms, _)| !copy_is_clean(copy_time_us(CopyPoint::AfterRender, bucket_ms * 1000 + 500)))
            .map(|&(_, count)| count)
            .sum();
        assert!(torn * 2 > loops, "{torn} of {loops} loops torn");
        // 6 ms after the sync: ~a third of the way down ("the top section").
        assert_eq!(seam_row(6_500), 105);
    }

    /// The fix: while Sony's picture is shown, the copy comes right after
    /// the sync, whatever our render step does, and its seam stays hidden.
    #[test]
    fn the_early_copy_is_clean_however_long_our_render_step_takes() {
        assert_eq!(sony_copy_point(true), CopyPoint::AfterSync);
        for (bucket_ms, _) in RENDER_END_HISTOGRAM {
            assert!(copy_is_clean(copy_time_us(CopyPoint::AfterSync, bucket_ms * 1000 + 500)));
        }
        assert!(seam_row(EARLY_COPY_US) < HIDDEN_SEAM_ROWS);
    }

    /// While our frames are shown, nothing is copied early (the copy would
    /// be wasted: 557 KiB of DMA per refresh); the first picture of a run of
    /// Sony's is copied late (one possibly torn picture at the run's start).
    #[test]
    fn our_frames_need_no_early_copy() {
        assert_eq!(sony_copy_point(false), CopyPoint::AfterRender);
    }

    /// The model's measured shape: top rows rewritten 0-3 ms after the sync,
    /// bottom rows (240-271) 14-17 ms after it.
    #[test]
    fn the_seam_follows_the_rewrite_down_the_screen() {
        assert!(seam_row(1_500) < 32);
        assert!((220..272).contains(&seam_row(15_000)));
        assert_eq!(seam_row(0), 0);
        assert_eq!(seam_row(PSP_FRAME_US), 0, "a whole frame later: the next rewrite's start");
    }
}
