//! `--sublists`: what a frame's CALLed sub-lists are, for deciding whether
//! their work could be reused from one frame to the next. Prints one line
//! per sub-list call (in frame order):
//!
//!     call <target> depth <d> hash <hash of its command words> commands <n> prims <n> ops <opcodes>
//!
//! and a summary. Comparing the lines of two dumps of consecutive frames
//! shows which sub-lists stayed the same.

use std::collections::BTreeSet;

use nativeres_core::command::{op, Command};
use nativeres_core::display_list::CommandVisitor;
use nativeres_core::ge_state::GeState;
use nativeres_core::memory::PspMemory;

/// One sub-list call being walked.
struct Open {
    target: u32,
    hash: u64,
    commands: u32,
    prims: u32,
    opcodes: BTreeSet<u8>,
}

pub struct SubLists {
    open: Vec<Open>,
    /// Commands outside any sub-list.
    pub main_commands: u32,
    pub main_prims: u32,
    /// Commands and PRIMs inside sub-lists (innermost counts).
    pub sub_commands: u32,
    pub sub_prims: u32,
    pub calls: u32,
    pub lines: Vec<String>,
}

impl SubLists {
    pub fn new() -> Self {
        SubLists { open: Vec::new(), main_commands: 0, main_prims: 0, sub_commands: 0, sub_prims: 0, calls: 0, lines: Vec::new() }
    }

    pub fn print(&self) {
        for line in &self.lines {
            println!("{line}");
        }
        println!(
            "sublists: {} calls; commands {} in sub-lists, {} outside; prims {} in sub-lists, {} outside",
            self.calls, self.sub_commands, self.main_commands, self.sub_prims, self.main_prims
        );
    }
}

/// FNV-1a over the command words.
fn mix(hash: u64, word: u32) -> u64 {
    (hash ^ word as u64).wrapping_mul(0x0000_0100_0000_01B3)
}

impl CommandVisitor for SubLists {
    fn visit<M: PspMemory>(&mut self, command: Command, ge: &GeState, _memory: &mut M) {
        let is_prim = command.opcode() == op::PRIM;
        match self.open.last_mut() {
            Some(open) => {
                open.hash = mix(open.hash, command.0);
                open.commands += 1;
                open.prims += is_prim as u32;
                open.opcodes.insert(command.opcode());
                self.sub_commands += 1;
                self.sub_prims += is_prim as u32;
            }
            None => {
                self.main_commands += 1;
                self.main_prims += is_prim as u32;
            }
        }
        match command.opcode() {
            op::CALL => {
                self.calls += 1;
                let target = ge.relative_address(command.data() & !3);
                self.open.push(Open { target, hash: 0xCBF2_9CE4_8422_2325, commands: 0, prims: 0, opcodes: BTreeSet::new() });
            }
            op::RET => {
                if let Some(done) = self.open.pop() {
                    let ops: Vec<String> = done.opcodes.iter().map(|op| format!("{op:02x}")).collect();
                    self.lines.push(format!(
                        "call {:08x} depth {} hash {:016x} commands {} prims {} ops {}",
                        done.target,
                        self.open.len(),
                        done.hash,
                        done.commands,
                        done.prims,
                        ops.join(",")
                    ));
                }
            }
            _ => {}
        }
    }
}
