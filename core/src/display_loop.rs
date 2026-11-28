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
/// runs) or may show it next (the last pass held our last picture while
/// the frame's textures were not ready: past `MAX_HELD_FRAMES` the loop
/// falls back to Sony's picture), otherwise only if `NativeResRenderFrame`
/// falls back to it.
pub fn sony_copy_point(showed_sony_picture_last: bool, held_for_textures_last: bool) -> CopyPoint {
    if showed_sony_picture_last || held_for_textures_last {
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

/// Whether a frame taken from the frame handover (`triple_buffer`) belongs
/// to a game the user quit. Forgetting our render targets in the XMB was
/// not enough: the handover still held the quit game's last frame, and
/// with nothing of ours shown the next game's start drew it again ("the
/// newest frame we have", for a paused game), its textures read from
/// memory the next game had overwritten: Monster Hunter's frozen, garbled
/// frame over MTX's boot (user, 2026-11-19).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AfterQuit {
    /// The XMB was seen and no frame was published since.
    quit: bool,
}

impl AfterQuit {
    pub const fn new() -> Self {
        AfterQuit { quit: false }
    }

    /// The PSP runs its XMB: every frame the display holds is the quit
    /// game's. The caller takes the waiting frame first (one published
    /// while the HOME menu was open, never taken), so the next new one is
    /// the next game's.
    pub fn quit(&mut self) {
        self.quit = true;
    }

    /// A frame was taken (`is_new`: published since the last take): may it
    /// be shown? After a quit only from the first new one on.
    pub fn may_show(&mut self, is_new: bool) -> bool {
        if is_new {
            self.quit = false;
        }
        !self.quit
    }
}

/// When Sony's picture must be put back on the display. With Graphics
/// Filtering Original, Adrenaline's loop draws nothing and the screen shows
/// Sony's picture once Adrenaline's menu exit has put it on the display.
/// Switched to Original outside the menu (our renderer off: SELECT), that
/// never happened and vita2d's last buffer (our picture, with the label)
/// stayed on the screen while the game ran on (user, 2026-11-23: "the game
/// freezes"). vita2d's buffers reach the display at the display queue's
/// flips, which trail the loop by up to ~110 ms: put back any earlier, a
/// late flip covered Sony's picture again. So: once a poll (every 50 ms)
/// sees no new flip after vita2d's buffers were shown, while Original is
/// drawn.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SonyHandBack {
    /// vita2d's flips counted at the last poll.
    last_flips: u32,
    /// vita2d's buffers were on the display since Sony's picture last was.
    loop_shown: bool,
}

impl SonyHandBack {
    pub const fn new() -> Self {
        SonyHandBack { last_flips: 0, loop_shown: false }
    }

    /// `flips`: vita2d's buffers put on the display so far; `original`:
    /// Adrenaline draws nothing (Original, PSP game, not pops / native).
    /// True: put Sony's picture on the display now.
    pub fn poll(&mut self, flips: u32, original: bool) -> bool {
        if flips != self.last_flips {
            self.last_flips = flips;
            self.loop_shown = true;
            return false;
        }
        if !original || !self.loop_shown {
            return false;
        }
        self.loop_shown = false;
        true
    }
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
mod sony_hand_back_tests {
    use super::SonyHandBack;

    /// SELECT in GoW with Original (2026-11-23): our picture flipped every
    /// frame, then the renderer went off. Sony's picture goes back once the
    /// flips have stopped, and only once.
    #[test]
    fn sonys_picture_goes_back_once_the_flips_stop() {
        let mut hand_back = SonyHandBack::new();
        assert!(!hand_back.poll(3, false), "ours shown, renderer on");
        assert!(!hand_back.poll(6, true), "switched off, the last flips still come");
        assert!(hand_back.poll(6, true), "quiet: Sony's picture back");
        assert!(!hand_back.poll(6, true), "once");
    }

    /// The display queue's late flip (two came 36 and 110 ms after the
    /// loop stopped, 2026-11-23): it covers Sony's picture, so it goes back
    /// again after it.
    #[test]
    fn a_late_flip_puts_sonys_picture_back_again() {
        let mut hand_back = SonyHandBack::new();
        hand_back.poll(10, true);
        assert!(hand_back.poll(10, true));
        assert!(!hand_back.poll(11, true), "the late flip");
        assert!(hand_back.poll(11, true), "again after it");
    }

    /// Adrenaline's menu open over Original: its loop flips every frame,
    /// nothing is put on the display until it closes.
    #[test]
    fn nothing_while_the_loop_keeps_flipping() {
        let mut hand_back = SonyHandBack::new();
        for flips in 1..20 {
            assert!(!hand_back.poll(flips * 3, true));
        }
    }

    /// Renderer on (Bilinear drawn): never, even with a still picture of
    /// ours (no flips while nothing changes is not "Original").
    #[test]
    fn never_while_a_filter_draws() {
        let mut hand_back = SonyHandBack::new();
        hand_back.poll(5, false);
        for _ in 0..5 {
            assert!(!hand_back.poll(5, false));
        }
        assert!(hand_back.poll(5, true), "then switched off: back");
    }

    /// Original from the start: Sony's picture is on the display already.
    #[test]
    fn nothing_when_vita2d_never_flipped() {
        let mut hand_back = SonyHandBack::new();
        for _ in 0..5 {
            assert!(!hand_back.poll(0, true));
        }
    }
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

    /// Monster Hunter quit from the HOME menu, then MTX started
    /// (2026-11-19): MHF's last frame, still in the handover, was drawn over
    /// MTX's boot. Also the frame the game published while the HOME menu
    /// was open (the display took none then): taken at the quit, never
    /// shown. The next game's first frame is.
    #[test]
    fn the_quit_games_last_frame_is_not_shown_in_the_next_game() {
        use crate::triple_buffer::TripleBuffer;
        let buffer = TripleBuffer::new();
        let (mut producer, mut consumer) = TripleBuffer::sides();
        let mut after_quit = AfterQuit::new();
        buffer.publish(&mut producer, 1);
        let (_, is_new) = buffer.take(&mut consumer).unwrap();
        assert!(after_quit.may_show(is_new), "MHF running");
        // HOME menu open: the game publishes one more, the display takes none.
        buffer.publish(&mut producer, 2);
        // "Quit the game?" > Yes: the XMB.
        let _ = buffer.take(&mut consumer);
        after_quit.quit();
        // MTX boots: no frame of its own yet, the handover gives MHF's.
        let (_, is_new) = buffer.take(&mut consumer).unwrap();
        assert!(!is_new);
        assert!(!after_quit.may_show(is_new), "MHF's frame over MTX's boot");
        // MTX's first frame.
        buffer.publish(&mut producer, 3);
        let (_, is_new) = buffer.take(&mut consumer).unwrap();
        assert!(after_quit.may_show(is_new), "MTX's own frame");
        let (_, is_new) = buffer.take(&mut consumer).unwrap();
        assert!(after_quit.may_show(is_new), "held while MTX pauses");
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
        assert_eq!(sony_copy_point(true, false), CopyPoint::AfterSync);
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
        assert_eq!(sony_copy_point(false, false), CopyPoint::AfterRender);
    }

    /// Kenka Bancho's train (2026-11-28): its first frame waited for
    /// textures, our last picture was held for 9 passes, then Sony's was
    /// shown for 3; the first of them, copied late, could show a seam (a tear
    /// seen once at the train's start). After a held pass the copy is early,
    /// so a fallback that follows shows a clean picture.
    #[test]
    fn a_pass_held_for_textures_copies_sonys_picture_early() {
        assert_eq!(sony_copy_point(false, true), CopyPoint::AfterSync);
        assert!(copy_is_clean(copy_time_us(sony_copy_point(false, true), 7_000)));
        assert!(!copy_is_clean(copy_time_us(sony_copy_point(false, false), 7_000)), "the late copy tore");
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
