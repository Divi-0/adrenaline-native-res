//! When Adrenaline's display loop (Adrenaline/user/menu.c) copies Sony's
//! picture of the PSP screen, and a timing model of what that copy shows.
//!
//! For frames the GE didn't draw (MTX's intro movie, loading screens) the
//! screen shows Sony's picture (`SCE_PSPEMU_FRAMEBUFFER`), which the loop
//! copies with one DMA and draws. Sony rewrites that picture top to bottom
//! across the whole PSP frame, starting at the LCD sync: measured with debug
//! `sonywrites` during MTX's intro movie (2026-11-10), rows 0-31 changed
//! 0-3 ms after the sync, rows 240-271 14-17 ms after it. A copy at time `t`
//! after the sync therefore holds the new picture above the row being
//! written at `t` and the old one below it: a seam at that row.
//!
//! Copied after `NativePspRenderFrame` (which may wait up to 5 ms for a
//! frame of ours, `FRAME_WAIT_US`, and is preempted by the capture threads),
//! the copy came 1-7 ms after the sync in more than half the loops
//! (`sonywrites` render_end during the movie: 179 of 359 loops 5-7 ms after
//! it): a seam ~90 rows down, "split horizontally in the top section"
//! (user). Copied right after the sync (as the stock loop does), it came
//! within 1 ms (360 of 360 copies): a seam in the top 17 rows at most.

/// One PSP frame (59.94 Hz), the period of Sony's top-to-bottom rewrite.
pub const PSP_FRAME_US: u32 = 16_683;

/// Rows of the PSP screen.
pub const PSP_ROWS: u32 = 272;

/// How long after the LCD sync the loop's early copy has started
/// (`sonywrites` copies: all in the 0-1 ms bucket).
pub const EARLY_COPY_US: u32 = 1_000;

/// A seam above this row is hidden at the screen's top edge (the first
/// ~6 %: the game's own picture has it there too, and the webcam tear count
/// of the early copy matched Sony's own presentation).
pub const HIDDEN_SEAM_ROWS: u32 = 17;

/// Where the loop copies Sony's picture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CopyPoint {
    /// Right after the LCD sync, before `NativePspRenderFrame`.
    AfterSync,
    /// Only once `NativePspRenderFrame` said the loop shows Sony's picture
    /// (needed then; when our frame is shown, no copy at all).
    AfterRender,
}

/// Where to copy: right after the sync while the screen shows Sony's
/// picture (it was shown in the loop before: CPU-drawn pictures come in
/// runs), otherwise only if `NativePspRenderFrame` falls back to it.
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

/// When a copy at `point` starts, given when `NativePspRenderFrame` returned
/// (`render_end_us` after the sync).
pub fn copy_time_us(point: CopyPoint, render_end_us: u32) -> u32 {
    match point {
        CopyPoint::AfterSync => EARLY_COPY_US,
        CopyPoint::AfterRender => render_end_us,
    }
}

/// A sample of the CPU-picture check differs when its R + G + B differ by
/// more than this (0..765): far above what upscaling and filtering change.
pub const CPU_PICTURE_DIFFERENCE: u32 = 96;

/// Whether a frame with few draws shows a picture the CPU wrote (loading
/// screens, movies): the screen cells no draw touches, Sony's picture
/// (`sony`) against ours (`ours`; `None` when our target holds nothing yet,
/// new or cleared: black). Then the display shows Sony's picture for it
/// (nativepsp_render.c ShowsCpuWrittenPicture).
///
/// MTX's boot loading screen (2026-11-11): the game writes the picture once
/// into both framebuffers and draws only the progress bar, alternating
/// between them. With "our target holds nothing" counting as "don't know",
/// every frame into the still empty target was drawn by us: black with the
/// bar, every other frame (flicker). An empty target is black, which is
/// what we would show.
pub fn shows_cpu_written_picture(sony: &[u32], ours: Option<&[u32]>) -> bool {
    const BLACK: u32 = 0xFF00_0000;
    let difference = |a: u32, b: u32| (0..3).map(|channel| ((a >> (8 * channel)) & 0xFF).abs_diff((b >> (8 * channel)) & 0xFF)).sum::<u32>();
    let differing = sony
        .iter()
        .enumerate()
        .filter(|&(index, &pixel)| difference(pixel, ours.map_or(BLACK, |ours| ours.get(index).copied().unwrap_or(BLACK))) > CPU_PICTURE_DIFFERENCE)
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

    use super::*;

    /// `sonywrites` during MTX's intro movie, renderer on (2026-11-10): how
    /// long after the LCD sync `NativePspRenderFrame` returned in loops that
    /// showed Sony's picture, by 1 ms bucket. Where the copy used to be.
    const RENDER_END_HISTOGRAM: [(u32, u32); 6] = [(0, 147), (1, 28), (2, 2), (4, 3), (5, 10), (6, 169)];

    /// The torn intro movie: copied after `NativePspRenderFrame`, most loops
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
