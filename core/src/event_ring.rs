//! The event ring the PSP plugin (renderer/psp) writes and the capture
//! reads: what the game sends to the PSP graphics engine (GE), one event
//! per call (layout shared by both sides; the VRAM copy table next to it:
//! [`crate::psp_link`]).
//!
//! Single writer (the plugin), single reader (our capture thread), no
//! locks: the writer fills `events[write_count % CAPACITY]` with its
//! sequence number, then increments `write_count`. The reader remembers how
//! many events it consumed. If `write_count` ran more than `CAPACITY` ahead,
//! the oldest events were overwritten; if an event's sequence number isn't
//! the expected one after copying it, it was overwritten while being read.

/// "RNSP" in memory.
pub const MAGIC: u32 = 0x5053_4E52;
pub const VERSION: u32 = 7;
pub const CAPACITY: u32 = 100;
/// Uncached PSP address of the ring (physical 0x0BCDF000).
pub const RING_ADDRESS: u32 = 0xABCD_F000;
/// Event types (`RawEvent::event_type`) and their args.
/// sceGeListEnQueue: list address, stall address (0 = none), queue id or
/// error, context address (`PspGeListArgs::context`, 0 = none; the GE
/// saves its state there at the list's start and restores it at its end).
/// Version 6 added the context.
pub const EVENT_LIST_ENQUEUE: u32 = 1;
/// sceGeListEnQueueHead: as `EVENT_LIST_ENQUEUE`, but the list runs before
/// queued ones.
pub const EVENT_LIST_ENQUEUE_HEAD: u32 = 2;
/// sceGeListUpdateStallAddr: queue id, new stall address, result. The GE
/// may now execute the list up to (not including) the stall address.
pub const EVENT_STALL_UPDATE: u32 = 3;
/// sceDisplaySetFrameBuf: framebuffer address, buffer width (pixels), pixel
/// format, sync mode. Ends a frame.
pub const EVENT_FRAMEBUFFER_FLIP: u32 = 4;
/// The game's GE sync returned (sceGeDrawSync / sceGeListSync with mode 0;
/// measuring only): 1 DrawSync / 2 ListSync, queue id (ListSync) or 0, us
/// the game waited in Sony's sync (the GE), us it then waited for the Vita
/// side. Written only while the Vita side asks (`vita_sync_events`). Not a
/// GE event: `capture::Event::from_ring` skips it.
pub const EVENT_GE_SYNC: u32 = 5;
/// sceGeSaveContext: context address, result.
pub const EVENT_GE_SAVE_CONTEXT: u32 = 6;
/// sceGeRestoreContext: context address, result.
pub const EVENT_GE_RESTORE_CONTEXT: u32 = 7;
/// The GE's own registers and matrices, read by the plugin when the Vita
/// side asked for them (`RingHeader::vita_state_request`), at a GE sync
/// (the GE idle): address of the block ([`GE_STATE_WORDS`] words, layout
/// below), the request it answers, 0 or the first error of the reads, the
/// microseconds the reads took the game's thread.
/// The Vita side asks after every new session: an Adrenaline savestate
/// load restores PSP RAM (the plugin's too) but the game never sends the
/// registers it set once again (MHF: scissor, fog; 2026-11-18). And after
/// a frame whose walk stopped early: the GE ran the commands we didn't
/// (MHF: a viewport reset missed, the field upside down for good).
/// Version 7 added it.
pub const EVENT_GE_STATE: u32 = 8;

/// The GE state block of [`EVENT_GE_STATE`], in words: every register as
/// sceGeGetCmd gives it (opcode 0..=255; the low 24 bits are the argument),
/// then the matrices as sceGeGetMtx gives them (24-bit floats: the float's
/// bits shifted right by 8, as the DATA commands carry them; PPSSPP
/// sceGe.cpp GetMatrix24): bones 0..7 (12 each), world (12), view (12),
/// projection (16), texture generation (12).
pub const GE_STATE_REGISTERS: usize = 0;
pub const GE_STATE_BONES: usize = 256;
pub const GE_STATE_WORLD: usize = GE_STATE_BONES + 8 * 12;
pub const GE_STATE_VIEW: usize = GE_STATE_WORLD + 12;
pub const GE_STATE_PROJECTION: usize = GE_STATE_VIEW + 12;
pub const GE_STATE_TEXGEN: usize = GE_STATE_PROJECTION + 16;
pub const GE_STATE_WORDS: usize = GE_STATE_TEXGEN + 12;

/// NativeResEvent.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct RawEvent {
    /// `write_count` when this event was written.
    pub sequence: u32,
    /// `EVENT_*` (see `capture::Event::from_ring`).
    pub event_type: u32,
    pub args: [u32; 4],
    /// PSP system time, low 32 bits.
    pub psp_time_us: u32,
    pub reserved: u32,
}

/// NativeResRingHeader.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct RingHeader {
    pub magic: u32,
    pub version: u32,
    pub capacity: u32,
    /// Changes every time the plugin starts (a new game boot).
    pub session: u32,
    pub write_count: u32,
    /// GE sync (see crate::psp_link): events the Vita side has read,
    /// walked and copied (`RingReader::consumed` once handled).
    pub vita_handled: u32,
    /// `SYNC_WANTED` while the Vita side wants the plugin to wait for it.
    pub vita_sync_wanted: u32,
    /// The plugin's counts: syncs that waited, microseconds waited, waits
    /// given up after `SYNC_TIMEOUT_US`.
    pub sync_waits: u32,
    pub sync_wait_us: u32,
    pub sync_timeouts: u32,
    /// `SYNC_WANTED` while the Vita side wants the plugin's GE_SYNC events.
    pub vita_sync_events: u32,
    /// GE state (`EVENT_GE_STATE`): the Vita side asks by making
    /// `vita_state_request` differ from `state_answered`; the plugin answers
    /// at its next GE sync and sets `state_answered` to it.
    pub vita_state_request: u32,
    pub state_answered: u32,
    pub reserved: [u32; 3],
}

/// NATIVERES_SYNC_WANTED.
pub const SYNC_WANTED: u32 = 0x434E_5953;
/// NATIVERES_SYNC_TIMEOUT_US.
pub const SYNC_TIMEOUT_US: u32 = 50_000;

/// NativeResRing.
#[repr(C)]
pub struct Ring {
    pub header: RingHeader,
    pub events: [RawEvent; CAPACITY as usize],
}

const _: () = assert!(core::mem::size_of::<RawEvent>() == 32);
const _: () = assert!(core::mem::size_of::<RingHeader>() == 64);

/// Access to the shared ring. The Vita side reads PSP RAM with volatile
/// loads (the plugin writes it at any time); tests use a plain struct.
pub trait RingSource {
    fn header(&self) -> RingHeader;
    fn event(&self, index: u32) -> RawEvent;
}

impl RingSource for Ring {
    fn header(&self) -> RingHeader {
        self.header
    }
    fn event(&self, index: u32) -> RawEvent {
        self.events[index as usize]
    }
}

/// What [`RingReader::next`] found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Next {
    Event(RawEvent),
    /// This many events were overwritten before we could read them.
    Lost(u32),
    /// Nothing new.
    Empty,
}

#[derive(Default)]
pub struct RingReader {
    read_count: u32,
    /// `write_count` as of the last [`RingReader::catch_up`].
    write_count: u32,
}

impl RingReader {
    pub const fn new() -> Self {
        RingReader { read_count: 0, write_count: 0 }
    }

    /// A new session (the plugin restarted): start from its first event.
    pub fn restart(&mut self) {
        *self = Self::new();
    }

    /// Events taken from the ring so far (counted like the writer's
    /// `write_count`, including events lost by falling behind).
    pub fn consumed(&self) -> u32 {
        self.read_count
    }

    /// Takes the writer's current `write_count`; [`RingReader::next`] then
    /// returns the events up to it. Returns how many events are gone
    /// because we fell more than a ring behind.
    pub fn catch_up(&mut self, write_count: u32) -> u32 {
        self.write_count = write_count;
        let behind = write_count.wrapping_sub(self.read_count);
        if behind > CAPACITY {
            let lost = behind - CAPACITY;
            self.read_count = self.read_count.wrapping_add(lost);
            lost
        } else {
            0
        }
    }

    /// Whether the frame about to be read is already over and the game has
    /// started a newer one: among the events not read yet (up to the
    /// writer's `write_count`, read fresh from the header) there is a frame
    /// flip with at least one more event after it.
    ///
    /// The capture then walks that frame without recording it (as with the
    /// renderer off: GE state and palettes stay right, the expensive vertex
    /// work is skipped) and records a newer one. Recording every frame while
    /// it takes longer than the game's frame made the capture fall a whole
    /// ring (~1.6 s at MTX's 60 events/s) behind, after which the ring
    /// overwrote events before they were read: most frames lost events
    /// (not shown), and the ones recorded read vertex data 1.6 s late.
    pub fn newer_frame_waiting(&self, ring: &impl RingSource) -> bool {
        let write_count = ring.header().write_count;
        let pending = write_count.wrapping_sub(self.read_count).min(CAPACITY);
        // Only the newest `CAPACITY` events are still in the ring.
        let first = write_count.wrapping_sub(pending);
        // The last pending event can't be followed by anything.
        (0..pending.saturating_sub(1)).any(|offset| {
            let sequence = first.wrapping_add(offset);
            let event = ring.event(sequence % CAPACITY);
            event.sequence == sequence && event.event_type == EVENT_FRAMEBUFFER_FLIP
        })
    }

    pub fn next(&mut self, ring: &impl RingSource) -> Next {
        if self.read_count == self.write_count {
            return Next::Empty;
        }
        let event = ring.event(self.read_count % CAPACITY);
        let expected = self.read_count;
        self.read_count = self.read_count.wrapping_add(1);
        if event.sequence != expected {
            Next::Lost(1) // overwritten while we copied it
        } else {
            Next::Event(event)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring() -> Box<Ring> {
        Box::new(Ring {
            header: RingHeader { magic: MAGIC, version: VERSION, capacity: CAPACITY, ..Default::default() },
            events: [RawEvent::default(); CAPACITY as usize],
        })
    }

    fn write(ring: &mut Ring, event_type: u32) {
        let sequence = ring.header.write_count;
        ring.events[(sequence % CAPACITY) as usize] = RawEvent { sequence, event_type, ..Default::default() };
        ring.header.write_count = sequence.wrapping_add(1);
    }

    fn read_all(reader: &mut RingReader, ring: &Ring) -> (Vec<u32>, u32) {
        let mut lost = reader.catch_up(ring.header.write_count);
        let mut types = Vec::new();
        loop {
            match reader.next(ring) {
                Next::Event(event) => types.push(event.event_type),
                Next::Lost(count) => lost += count,
                Next::Empty => return (types, lost),
            }
        }
    }

    #[test]
    fn events_are_read_in_order_once() {
        let mut ring = ring();
        let mut reader = RingReader::new();
        for event_type in 1..=3 {
            write(&mut ring, event_type);
        }
        assert_eq!(read_all(&mut reader, &ring), (vec![1, 2, 3], 0));
        write(&mut ring, 4);
        assert_eq!(read_all(&mut reader, &ring), (vec![4], 0));
        assert_eq!(read_all(&mut reader, &ring), (vec![], 0));
    }

    #[test]
    fn falling_more_than_a_ring_behind_loses_the_oldest() {
        let mut ring = ring();
        let mut reader = RingReader::new();
        for n in 0..CAPACITY + 5 {
            write(&mut ring, n);
        }
        let (types, lost) = read_all(&mut reader, &ring);
        assert_eq!(lost, 5);
        assert_eq!(types.first(), Some(&5));
        assert_eq!(types.len(), CAPACITY as usize);
    }

    #[test]
    fn an_event_overwritten_while_reading_counts_as_lost() {
        let mut ring = ring();
        let mut reader = RingReader::new();
        write(&mut ring, 1);
        write(&mut ring, 2);
        reader.catch_up(ring.header.write_count);
        // The writer laps the reader between catch_up and reading slot 0.
        ring.events[0].sequence = CAPACITY;
        assert_eq!(reader.next(&*ring), Next::Lost(1));
        assert_eq!(reader.next(&*ring), Next::Event(RawEvent { sequence: 1, event_type: 2, ..Default::default() }));
    }

    #[test]
    fn a_newer_frame_waits_only_once_the_next_one_has_started() {
        const ENQUEUE: u32 = 1;
        let mut ring = ring();
        let mut reader = RingReader::new();
        assert!(!reader.newer_frame_waiting(&*ring));
        write(&mut ring, ENQUEUE);
        assert!(!reader.newer_frame_waiting(&*ring)); // a frame in progress
        write(&mut ring, EVENT_FRAMEBUFFER_FLIP);
        assert!(!reader.newer_frame_waiting(&*ring)); // over, but nothing newer yet
        write(&mut ring, ENQUEUE);
        assert!(reader.newer_frame_waiting(&*ring)); // the next one has begun
        // Once the old frame is read, only the new one is left.
        reader.catch_up(ring.header.write_count);
        reader.next(&*ring);
        reader.next(&*ring);
        assert!(!reader.newer_frame_waiting(&*ring));
    }

    #[test]
    fn a_reader_a_ring_behind_still_sees_the_newer_frames() {
        let mut ring = ring();
        let reader = RingReader::new();
        for _ in 0..CAPACITY {
            write(&mut ring, 1);
            write(&mut ring, EVENT_FRAMEBUFFER_FLIP);
        }
        assert!(reader.newer_frame_waiting(&*ring));
    }

    #[test]
    fn counts_wrap_around() {
        let mut ring = ring();
        ring.header.write_count = u32::MAX - 1;
        let mut reader = RingReader { read_count: u32::MAX - 1, write_count: 0 };
        for n in 0..4 {
            write(&mut ring, n);
        }
        // write_count wrapped to 2; the events sit at their sequence % CAPACITY.
        assert_eq!(ring.header.write_count, 2);
        let (types, lost) = read_all(&mut reader, &ring);
        assert_eq!((types, lost), (vec![0, 1, 2, 3], 0));
    }
}
