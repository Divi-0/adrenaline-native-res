//! NativeRes core: understands what a PSP game asks the PSP graphics engine
//! (the "GE") to do.
//!
//! This crate has no Vita dependencies, so it builds and is tested on a PC.
//! It never touches memory directly: PSP memory is read through the
//! [`memory::PspMemory`] trait, which the Vita side implements over the
//! emulator's RAM and tests implement over a plain array.
//!
//! How a PSP game draws:
//! 1. It writes a *display list*: an array of 32-bit GE commands in RAM
//!    (see [`command`]).
//! 2. It hands the list to the GE with `sceGeListEnQueue`, optionally with a
//!    *stall address*: the GE executes up to that address and waits. The game
//!    keeps appending commands and moves the stall address forward with
//!    `sceGeListUpdateStallAddr`.
//! 3. When the frame is done, it shows the framebuffer with
//!    `sceDisplaySetFrameBuf`.
//!
//! [`capture::Capture`] follows that sequence of events, walks each list with
//! [`display_list::walk`], and collects per-frame statistics
//! ([`stats::FrameStats`]).

// The NEON intrinsics are still unstable on 32-bit ARM (the Vita build
// uses nightly anyway, for build-std); see transform::RowTransform::clip.
#![cfg_attr(target_arch = "arm", feature(stdarch_arm_neon_intrinsics))]
#![cfg_attr(not(test), no_std)]

pub mod bench;
pub mod blend;
pub mod block_cache;
pub mod capture;
pub mod class_heap;
pub mod color;
pub mod copy_tasks;
pub mod copy_watch;
pub mod cycles;
pub mod command;
pub mod display_list;
pub mod display_loop;
pub mod draw_stream;
pub mod dump;
pub mod event_ring;
pub mod frame;
pub mod frame_builder;
pub mod frame_pool;
pub mod game_fps;
pub mod ge_context;
pub mod ge_state;
pub mod ge_writes;
pub mod heap;
pub mod lighting;
pub mod log_ring;
pub mod memory;
pub mod opaque_source;
pub mod options;

pub mod palette_tables;
pub mod phase;
pub mod psp_copy;
pub mod psp_link;
pub mod raw_arena;
pub mod raw_blocks;
pub mod raw_vertex;
pub mod recorder;
pub mod report;
pub mod snapshot;
pub mod span;
pub mod stats;
pub mod texture;
pub mod texture_cache;
pub mod transform;
pub mod triple_buffer;
pub mod vertex;
pub mod vertex_cache;
pub mod vram_requests;
pub mod walk_stream;
