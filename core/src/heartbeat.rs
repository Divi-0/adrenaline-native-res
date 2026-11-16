//! The emulated PSP's heartbeat as ScePspemu watches it (the C1-2858-3
//! hunt, DESIGN.md "The real one: a watchdog on the emulated PSP").
//!
//! ScePspemuMain reads shared word 7 (`sceCompatReadShared32(7)`) once per
//! pass of its running loop; when the value stays the same for more than
//! 30 s it gives up with 0x80010089 (the C1-2858-3 dialog). `Heartbeat`
//! follows the same reads and says what is worth a log line: the first
//! value, the first change, a stall (no change for `STALL_US`), the end of
//! a stall, and a progress line every `REPORT_US` during the first
//! `REPORTS_FOR_US` (the boot to the XMB).

/// No change for this long is a stall worth a line (the PSP's boot never
/// stood still that long in a healthy start, to be checked in the logs).
pub const STALL_US: u64 = 2_000_000;
/// ScePspemu's limit (text+0x6018: 0x01C9C380 us).
pub const WATCHDOG_US: u64 = 30_000_000;
/// Progress lines: every 10 s, during the first 60 s after the first read.
pub const REPORT_US: u64 = 10_000_000;
pub const REPORTS_FOR_US: u64 = 60_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Event {
    First { value: u32 },
    FirstChange { value: u32 },
    /// Unchanged since `since_us` (the last change, or the first read).
    Stalled { value: u32, since_us: u64 },
    /// Changed again after standing still for `stalled_us`.
    Moving { value: u32, stalled_us: u64 },
    Report { value: u32, changes: u32 },
}

#[derive(Debug)]
pub struct Heartbeat {
    started: bool,
    value: u32,
    changed_us: u64,
    changes: u32,
    stalled: bool,
    first_us: u64,
    next_report_us: u64,
}

impl Heartbeat {
    pub const fn new() -> Self {
        Heartbeat { started: false, value: 0, changed_us: 0, changes: 0, stalled: false, first_us: 0, next_report_us: 0 }
    }

    /// One read of the heartbeat (`value`) at process time `now_us`.
    pub fn step(&mut self, value: u32, now_us: u64) -> Option<Event> {
        if !self.started {
            self.started = true;
            self.value = value;
            self.changed_us = now_us;
            self.first_us = now_us;
            self.next_report_us = now_us + REPORT_US;
            return Some(Event::First { value });
        }
        if value != self.value {
            let stood_still = now_us.saturating_sub(self.changed_us);
            let was_stalled = self.stalled;
            self.value = value;
            self.changed_us = now_us;
            self.changes = self.changes.saturating_add(1);
            self.stalled = false;
            if was_stalled {
                return Some(Event::Moving { value, stalled_us: stood_still });
            }
            if self.changes == 1 {
                return Some(Event::FirstChange { value });
            }
        } else if !self.stalled && now_us.saturating_sub(self.changed_us) >= STALL_US {
            self.stalled = true;
            return Some(Event::Stalled { value, since_us: self.changed_us });
        }
        if now_us >= self.next_report_us && now_us < self.first_us + REPORTS_FOR_US {
            self.next_report_us = now_us + REPORT_US;
            return Some(Event::Report { value: self.value, changes: self.changes });
        }
        None
    }
}

impl Default for Heartbeat {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reads every 16.7 ms (ScePspemuMain's loop waits for the vblank).
    const PASS_US: u64 = 16_683;

    #[test]
    fn a_steady_beat_gives_first_first_change_and_reports_only() {
        let mut heartbeat = Heartbeat::new();
        let mut events = Vec::new();
        for pass in 0..(70_000_000 / PASS_US) {
            let now = 3_700_000 + pass * PASS_US;
            if let Some(event) = heartbeat.step((pass / 2) as u32, now) {
                events.push(event);
            }
        }
        assert_eq!(events[0], Event::First { value: 0 });
        assert_eq!(events[1], Event::FirstChange { value: 1 });
        let reports = events.iter().filter(|event| matches!(event, Event::Report { .. })).count();
        assert_eq!((events.len(), reports), (2 + 5, 5), "{events:?}");
        assert!(!events.iter().any(|event| matches!(event, Event::Stalled { .. } | Event::Moving { .. })));
    }

    #[test]
    fn a_boot_stall_is_reported_once_with_its_start_and_its_end() {
        // The C1 shape: beating from 3.7 s, standing still from 10 s.
        let mut heartbeat = Heartbeat::new();
        let mut value = 0;
        let mut events = Vec::new();
        let mut now = 3_700_000;
        while now < 40_000_000 {
            if now < 10_000_000 || now > 25_000_000 {
                value += 1;
            }
            if let Some(event) = heartbeat.step(value, now) {
                events.push((now, event));
            }
            now += PASS_US;
        }
        let stalls: Vec<_> = events.iter().filter(|(_, event)| matches!(event, Event::Stalled { .. })).collect();
        assert_eq!(stalls.len(), 1, "{events:?}");
        let (seen_at, Event::Stalled { since_us, .. }) = *stalls[0] else { unreachable!() };
        assert!((9_980_000..10_000_000).contains(&since_us), "{since_us}");
        assert!(seen_at - since_us >= STALL_US && seen_at - since_us < STALL_US + PASS_US);
        let moving: Vec<_> = events.iter().filter_map(|(_, event)| match event {
            Event::Moving { stalled_us, .. } => Some(*stalled_us),
            _ => None,
        }).collect();
        assert_eq!(moving.len(), 1);
        assert!((15_000_000..15_100_000).contains(&moving[0]), "{moving:?}");
    }

    #[test]
    fn a_heartbeat_that_never_moves_stalls_from_its_first_read() {
        let mut heartbeat = Heartbeat::new();
        assert_eq!(heartbeat.step(42, 1_000_000), Some(Event::First { value: 42 }));
        assert_eq!(heartbeat.step(42, 2_999_999), None);
        assert_eq!(heartbeat.step(42, 3_000_000), Some(Event::Stalled { value: 42, since_us: 1_000_000 }));
        assert_eq!(heartbeat.step(42, 9_000_000), None);
    }
}
