//! GE contexts: copies of the GE state that the PSP saves and later puts
//! back, so a list's register changes don't reach the lists after it.
//!
//! Two ways, both as PPSSPP does them (GPUCommon.cpp, sceGe.cpp):
//! - a list enqueued with a context (`PspGeListArgs::context`): the GE
//!   saves its state there when the list starts and restores it when the
//!   list finishes;
//! - `sceGeSaveContext` / `sceGeRestoreContext` called by the game or a
//!   system utility.
//!
//! We don't read Sony's context layout from PSP memory: we keep our own
//! `GeState` copy per context address, saved at the moment the PSP saves.
//! A restore of an address we never saw saved is counted
//! (`FrameStats::contexts_missing`) and leaves the state alone.
//!
//! Monster Hunter Freedom: after the name keyboard (sceUtilityOsk) the
//! hunter drew black. Fog is on in every frame but FOG_END / FOG_SCALE are
//! never sent again: Sony's GE got them back from a context, our state held
//! zeros (fully fogged, fog colour black).

use crate::ge_state::GeState;
use crate::memory::physical;

/// Contexts held at once (a game uses one or two; utilities one more).
pub const CONTEXT_SLOTS: usize = 4;

pub struct ContextTable {
    addresses: [u32; CONTEXT_SLOTS],
    used: [bool; CONTEXT_SLOTS],
    states: [GeState; CONTEXT_SLOTS],
    /// The slot a new address takes when all are used (round robin).
    next: usize,
}

impl ContextTable {
    pub const fn new() -> ContextTable {
        ContextTable {
            addresses: [0; CONTEXT_SLOTS],
            used: [false; CONTEXT_SLOTS],
            states: [GeState::new(), GeState::new(), GeState::new(), GeState::new()],
            next: 0,
        }
    }

    pub fn clear(&mut self) {
        self.used = [false; CONTEXT_SLOTS];
        self.next = 0;
    }

    /// Keyed by physical address: the cached and uncached pointers to a
    /// context are the same context.
    fn find(&self, address: u32) -> Option<usize> {
        let address = physical(address);
        (0..CONTEXT_SLOTS).find(|&slot| self.used[slot] && self.addresses[slot] == address)
    }

    /// The PSP saved its GE state at `address`.
    pub fn save(&mut self, address: u32, ge: &GeState) {
        let slot = match self.find(address) {
            Some(slot) => slot,
            None => match (0..CONTEXT_SLOTS).find(|&slot| !self.used[slot]) {
                Some(slot) => slot,
                None => {
                    let slot = self.next;
                    self.next = (self.next + 1) % CONTEXT_SLOTS;
                    slot
                }
            },
        };
        self.addresses[slot] = physical(address);
        self.used[slot] = true;
        self.states[slot].clone_from(ge);
    }

    /// The PSP put back the state saved at `address`. False if we never
    /// saw it saved (`ge` unchanged).
    pub fn restore(&self, address: u32, ge: &mut GeState) -> bool {
        match self.find(address) {
            Some(slot) => {
                ge.restore_context(&self.states[slot]);
                true
            }
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::{op, Command};

    fn set(ge: &mut GeState, opcode: u8, data: u32) {
        ge.set_register(Command(((opcode as u32) << 24) | data));
    }

    #[test]
    fn restore_puts_back_what_was_saved() {
        let mut table = ContextTable::new();
        let mut ge = GeState::new();
        set(&mut ge, op::FOG_END, 0x44_8000);
        set(&mut ge, op::FOG_SCALE, 0x3A_1000);
        table.save(0x0900_1000, &ge);
        // The utility's list: fog parameters zeroed.
        set(&mut ge, op::FOG_END, 0);
        set(&mut ge, op::FOG_SCALE, 0);
        let before = ge.state_changes();
        assert!(table.restore(0x0900_1000, &mut ge));
        assert_eq!(ge.register(op::FOG_END), 0x44_8000);
        assert_eq!(ge.register(op::FOG_SCALE), 0x3A_1000);
        assert_ne!(ge.state_changes(), before, "caches keyed on state_changes must see the restore");
    }

    #[test]
    fn an_unknown_context_changes_nothing() {
        let table = ContextTable::new();
        let mut ge = GeState::new();
        set(&mut ge, op::FOG_END, 0x44_8000);
        assert!(!table.restore(0x0900_2000, &mut ge));
        assert_eq!(ge.register(op::FOG_END), 0x44_8000);
    }

    #[test]
    fn a_full_table_reuses_the_oldest_slot() {
        let mut table = ContextTable::new();
        let mut ge = GeState::new();
        for index in 0..=CONTEXT_SLOTS as u32 {
            set(&mut ge, op::FOG_END, index);
            table.save(0x0900_0000 + index * 0x1000, &ge);
        }
        assert!(!table.restore(0x0900_0000, &mut ge), "the first slot was reused");
        assert!(table.restore(0x0900_0000 + CONTEXT_SLOTS as u32 * 0x1000, &mut ge));
        assert_eq!(ge.register(op::FOG_END), CONTEXT_SLOTS as u32);
    }
}
