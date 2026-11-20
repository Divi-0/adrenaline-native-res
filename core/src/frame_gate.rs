//! Whether a recorded frame is shown, or the frame before stays on screen
//! (the capture thread's choice at each frame's end, vita capture.rs).
//!
//! A frame is held back when it is known to lack something: events lost
//! from the ring (whole display lists missing: a black flash), VRAM read
//! before the plugin had copied it, a walk that stopped at a bad address or
//! ran away (the rest of its list missing), registers from before a
//! savestate load while the GE's own haven't come, or a GE sync wait the
//! plugin gave up while the frame was walked (the game went on rewriting
//! what the walk read: stretched triangles from MHF's people when the
//! capture was too slow).

/// Recorded frames held back in a row for a reason that may never go away
/// (a walk that always stops early) before one is shown anyway, so such a
/// game still shows.
pub const MAX_HELD_BACK: u32 = 3;

/// What the capture knows about a recorded frame at its end.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameChecks {
    /// Events the ring lost while the frame was walked.
    pub dropped_events: u32,
    /// Reads of VRAM the plugin's copy didn't hold yet.
    pub missing_vram: u32,
    /// The walk stopped at a bad address or ran away.
    pub stopped_early: bool,
    /// After a new session: the GE's own state hasn't come (within its limit).
    pub without_ge_state: bool,
    /// The plugin's count of GE sync waits it gave up
    /// (`event_ring::RingHeader::sync_timeouts`) when the frame's recording
    /// started, and at its end.
    pub sync_timeouts: (u32, u32),
}

/// The choice, frame after frame (it counts frames held back in a row).
#[derive(Default)]
pub struct FrameGate {
    held_back: u32,
}

impl FrameGate {
    pub const fn new() -> FrameGate {
        FrameGate { held_back: 0 }
    }

    /// True: show the frame. False: drop it (the one before stays).
    pub fn should_show(&mut self, checks: &FrameChecks) -> bool {
        // The plugin stopped holding the game in its GE sync while the frame
        // was walked: the game went on and may have rewritten lists,
        // vertices or indices the walk still read (half old, half new).
        // Never shown, however many in a row (user, 2026-11-20): a capture
        // that never keeps up freezes the picture rather than glitch.
        let torn = checks.sync_timeouts.0 != checks.sync_timeouts.1;
        let hold_back = checks.stopped_early && self.held_back < MAX_HELD_BACK;
        let show = checks.dropped_events == 0 && checks.missing_vram == 0 && !hold_back && !checks.without_ge_state && !torn;
        if show {
            self.held_back = 0;
        } else {
            self.held_back += hold_back as u32;
        }
        show
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_frame_without_trouble_is_shown() {
        let mut gate = FrameGate::new();
        assert!(gate.should_show(&FrameChecks::default()));
        assert!(gate.should_show(&FrameChecks::default()));
    }

    #[test]
    fn frames_lacking_lists_vram_or_ge_state_are_dropped_every_time() {
        let mut gate = FrameGate::new();
        for _ in 0..10 {
            assert!(!gate.should_show(&FrameChecks { dropped_events: 1, ..FrameChecks::default() }));
            assert!(!gate.should_show(&FrameChecks { missing_vram: 2, ..FrameChecks::default() }));
            assert!(!gate.should_show(&FrameChecks { without_ge_state: true, ..FrameChecks::default() }));
        }
        assert!(gate.should_show(&FrameChecks::default()));
    }

    #[test]
    fn walks_that_stop_early_are_dropped_at_most_three_in_a_row() {
        let mut gate = FrameGate::new();
        let stopped = FrameChecks { stopped_early: true, ..FrameChecks::default() };
        for _ in 0..MAX_HELD_BACK {
            assert!(!gate.should_show(&stopped));
        }
        assert!(gate.should_show(&stopped), "a game that always ends that way still shows");
        assert!(!gate.should_show(&stopped), "counted again from the shown frame");
    }

    /// MHF's intro fly-over with debug `livecycles` (2026-11-20): the capture
    /// took ~70 ms per frame, the plugin gave up waiting in the game's GE sync
    /// (`gesync=1454/3187ms/23` -> 24 within a frame), the game rewrote its
    /// skinned villagers' vertices while the walk still read them, and the
    /// frame showed stretched triangles from the people.
    #[test]
    fn a_frame_whose_ge_sync_timed_out_is_dropped() {
        let mut gate = FrameGate::new();
        let torn = FrameChecks { sync_timeouts: (23, 24), ..FrameChecks::default() };
        assert!(!gate.should_show(&torn), "the game went on while the frame was walked");
        assert!(gate.should_show(&FrameChecks { sync_timeouts: (24, 24), ..FrameChecks::default() }));
        // The count wraps.
        assert!(!gate.should_show(&FrameChecks { sync_timeouts: (u32::MAX, 0), ..FrameChecks::default() }));
    }

    /// The same run with torn frames dropped at most three in a row: ~21 of
    /// them still came through (every 4th), and the user saw a streak.
    /// User (2026-11-20): drop them all, no limit.
    #[test]
    fn frames_whose_ge_sync_timed_out_are_always_dropped() {
        let mut gate = FrameGate::new();
        let torn = |n: u32| FrameChecks { sync_timeouts: (n, n + 1), ..FrameChecks::default() };
        for n in 0..100 {
            assert!(!gate.should_show(&torn(n)), "torn frame {n} in a row");
        }
        assert!(gate.should_show(&FrameChecks { sync_timeouts: (100, 100), ..FrameChecks::default() }));
    }

    #[test]
    fn torn_frames_leave_the_early_stop_count_alone() {
        let mut gate = FrameGate::new();
        let stopped = FrameChecks { stopped_early: true, ..FrameChecks::default() };
        let torn = FrameChecks { sync_timeouts: (5, 6), ..FrameChecks::default() };
        assert!(!gate.should_show(&stopped));
        assert!(!gate.should_show(&torn));
        assert!(!gate.should_show(&stopped));
        assert!(!gate.should_show(&torn));
        assert!(!gate.should_show(&stopped));
        assert!(gate.should_show(&stopped), "the fourth early stop in a row shows, torn frames between them or not");
    }
}
