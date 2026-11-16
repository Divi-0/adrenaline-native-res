//! Recording one frame into a dump (format: `nativepsp_core::dump`).
//!
//! Used from the capture thread only. While a dump is active:
//! - every 1 KiB block of PSP memory that `VitaPspMemory` maps is copied
//!   into the dump the first time it is read,
//! - every event is recorded,
//! - at the end, the frame's textures are read too, so their blocks land in
//!   the dump as well.
//!
//! The dump buffer is allocated by dump_file.rs. Blocks are stored from
//! `BLOCKS_START` on while the frame runs; `finish` then writes header,
//! snapshot and events in front and moves the blocks up behind them.

use core::ptr;

use nativepsp_core::dump::{Header, BLOCK_SIZE, EVENT_WORDS, HEADER_BYTES};
use nativepsp_core::snapshot::Writer;

/// Room for header, snapshot and events at the start of the buffer.
const BLOCKS_START: usize = 256 * 1024;
const MAX_EVENTS: usize = 2048;
const MAX_BLOCKS: usize = 8192;

pub struct DumpRecorder {
    buffer: *mut u8,
    capacity: usize,
    active: bool,
    overflowed: bool,
    snapshot_bytes: usize,
    events: [[u32; EVENT_WORDS]; MAX_EVENTS],
    event_count: usize,
    block_addresses: [u32; MAX_BLOCKS],
    block_count: usize,
}

impl DumpRecorder {
    pub const fn new() -> Self {
        DumpRecorder {
            buffer: ptr::null_mut(),
            capacity: 0,
            active: false,
            overflowed: false,
            snapshot_bytes: 0,
            events: [[0; EVENT_WORDS]; MAX_EVENTS],
            event_count: 0,
            block_addresses: [0; MAX_BLOCKS],
            block_count: 0,
        }
    }

    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Starts a dump into `buffer`; `save_snapshot` writes the capture state.
    ///
    /// # Safety
    /// `buffer` must hold `capacity` writable bytes until `finish`.
    pub unsafe fn begin(&mut self, buffer: *mut u8, capacity: usize, save_snapshot: impl FnOnce(&mut Writer)) -> bool {
        if capacity <= BLOCKS_START {
            return false;
        }
        self.buffer = buffer;
        self.capacity = capacity;
        self.overflowed = false;
        self.event_count = 0;
        self.block_count = 0;

        // The snapshot goes right behind the header.
        let area = core::slice::from_raw_parts_mut(buffer.add(HEADER_BYTES), BLOCKS_START - HEADER_BYTES);
        let mut writer = Writer::new(area);
        save_snapshot(&mut writer);
        if writer.overflowed() {
            return false;
        }
        self.snapshot_bytes = writer.len();
        self.active = true;
        true
    }

    pub fn record_event(&mut self, event_type: u32, args: &[u32]) {
        if self.event_count == MAX_EVENTS {
            self.overflowed = true;
            return;
        }
        self.events[self.event_count] = [event_type, args[0], args[1], args[2], args[3]];
        self.event_count += 1;
    }

    /// Copies the 1 KiB block at `address` (a copy of it at `data`) into the
    /// dump, unless it is already there.
    ///
    /// # Safety
    /// `data` must point to `BLOCK_SIZE` readable bytes.
    pub unsafe fn record_block(&mut self, address: u32, data: *const u8) {
        if self.block_addresses[..self.block_count].contains(&address) {
            return;
        }
        let record_bytes = 4 + BLOCK_SIZE as usize;
        let at = BLOCKS_START + self.block_count * record_bytes;
        if self.block_count == MAX_BLOCKS || at + record_bytes > self.capacity {
            self.overflowed = true;
            return;
        }
        ptr::copy_nonoverlapping(address.to_le_bytes().as_ptr(), self.buffer.add(at), 4);
        ptr::copy_nonoverlapping(data, self.buffer.add(at + 4), BLOCK_SIZE as usize);
        self.block_addresses[self.block_count] = address;
        self.block_count += 1;
    }

    /// Completes the dump. Returns its size in bytes, or 0 if something
    /// didn't fit.
    pub fn finish(&mut self) -> usize {
        self.active = false;
        let events_bytes = self.event_count * EVENT_WORDS * 4;
        if self.overflowed || HEADER_BYTES + self.snapshot_bytes + events_bytes > BLOCKS_START {
            return 0;
        }

        // Safe: everything written below lies inside the buffer given to
        // `begin`; the size checks above and in `record_block` ensure it.
        unsafe {
            let mut at = HEADER_BYTES + self.snapshot_bytes;
            for event in &self.events[..self.event_count] {
                for word in event {
                    ptr::copy_nonoverlapping(word.to_le_bytes().as_ptr(), self.buffer.add(at), 4);
                    at += 4;
                }
            }

            let blocks_bytes = self.block_count * (4 + BLOCK_SIZE as usize);
            ptr::copy(self.buffer.add(BLOCKS_START), self.buffer.add(at), blocks_bytes);

            let header = Header {
                snapshot_bytes: self.snapshot_bytes as u32,
                event_count: self.event_count as u32,
                block_count: self.block_count as u32,
                block_size: BLOCK_SIZE,
            };
            let mut header_bytes = [0u8; HEADER_BYTES];
            header.write(&mut header_bytes);
            ptr::copy_nonoverlapping(header_bytes.as_ptr(), self.buffer, HEADER_BYTES);

            at + blocks_bytes
        }
    }
}
