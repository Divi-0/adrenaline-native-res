//! Access to PSP memory.
//!
//! PSP addresses carry cache/mirror bits in their top nibble (0x08800000,
//! 0x48800000 and 0x88800000 are the same byte). We always strip those with
//! [`physical`] before using an address.

/// Main RAM including Adrenaline's extra RAM: 0x08000000..0x0C000000 (64 MiB).
pub const RAM_START: u32 = 0x0800_0000;
pub const RAM_END: u32 = 0x0C00_0000;

/// Video RAM: 0x04000000..0x04200000 (2 MiB).
pub const VRAM_START: u32 = 0x0400_0000;
pub const VRAM_END: u32 = 0x0420_0000;

/// Reads PSP memory. Returns `None` for addresses that are not readable.
/// The PSP is little-endian, like the Vita.
pub trait PspMemory {
    /// Reads an aligned 32-bit word.
    fn read_u32(&mut self, address: u32) -> Option<u32>;

    fn read_u16(&mut self, address: u32) -> Option<u16> {
        let word = self.read_u32(address & !3)?;
        Some((word >> ((address & 2) * 8)) as u16)
    }

    fn read_u8(&mut self, address: u32) -> Option<u8> {
        let word = self.read_u32(address & !3)?;
        Some((word >> ((address & 3) * 8)) as u8)
    }

    fn read_f32(&mut self, address: u32) -> Option<f32> {
        self.read_u32(address).map(f32::from_bits)
    }

    /// Reads `out.len()` consecutive aligned words starting at `address`.
    /// Memory with a block cache overrides this to look the block up once
    /// rather than for every word (the capture thread reads every vertex
    /// this way; per-word reads were a large part of its time).
    fn read_words(&mut self, address: u32, out: &mut [u32]) -> Option<()> {
        for (index, word) in out.iter_mut().enumerate() {
            *word = self.read_u32(address.wrapping_add(4 * index as u32))?;
        }
        Some(())
    }

    /// The `count` aligned words at `address` without copying them, if this
    /// memory can lend them (the block cache can when they lie in one
    /// block). `None` doesn't mean unreadable: use `read_words` then.
    fn borrow_words(&mut self, _address: u32, _count: usize) -> Option<&[u32]> {
        None
    }

    /// `vertex_cache::data_hash` of exactly the `count` words at `address`,
    /// if whoever copied them hashed them already (`draw_stream`: the
    /// copier, while the words were at hand). `None`: hash them yourself.
    fn copied_hash(&mut self, _address: u32, _count: usize) -> Option<u64> {
        None
    }

    /// Whether the `bytes` bytes at `address` may be read later by another
    /// thread straight from PSP memory instead of now through this memory
    /// (`draw_stream`: the recorder thread copies a draw's vertices itself).
    /// Main RAM qualifies, and VRAM where the plugin's copy holds it (the
    /// memory notes the blocks as needed, as a read now would: hence
    /// `&mut`); not what a GE block transfer wrote (`ge_writes::Overlaid`),
    /// and nothing while a dump runs (the dump records the blocks read).
    /// Default: false.
    fn defer(&mut self, _address: u32, _bytes: u32) -> bool {
        false
    }

    /// Where the `bytes` bytes at `address` came from if a GE block transfer
    /// from main RAM wrote them this frame, all of them from one source
    /// (`ge_writes::Overlaid`, its redirects): reading the source gives the
    /// same bytes. Default: none.
    fn redirect_source(&mut self, _address: u32, _bytes: u32) -> Option<u32> {
        None
    }

    /// Whether the `bytes` bytes at `address` are left in PSP RAM for the GPU
    /// (`raw_arena::ZERO_COPY`): nobody copied them, and a raw run of them
    /// is drawn from PSP RAM. Reading them still works (copied then).
    /// Default: false.
    fn zero_copy(&self, _address: u32, _bytes: u32) -> bool {
        false
    }

    /// The words at `address + 4 * positions[i]` into `out[i]` (the vertex
    /// cache's samples of a run, `vertex_cache::sample_positions`); `None`
    /// if one can't be read. Memories that read PSP memory through a call
    /// (`draw_stream::CopiedMemory`) do it in one call for all of them.
    fn read_samples(&mut self, address: u32, positions: &[usize], out: &mut [u32]) -> Option<()> {
        for (word, &position) in out.iter_mut().zip(positions) {
            *word = self.read_u32(address.wrapping_add(4 * position as u32))?;
        }
        Some(())
    }

    /// Writes an aligned word, as the GE does in a block transfer. Only the
    /// capture's view of memory takes writes (`ge_writes::Overlaid`: kept
    /// until the frame ends); other memories ignore them (false).
    fn write_u32(&mut self, _address: u32, _value: u32) -> bool {
        false
    }

    /// Called before a block transfer writes `bytes` bytes from `address`:
    /// whether this memory can take all of them. A transfer is done whole
    /// or not at all, so a half-done copy never mixes old and new data. The
    /// memory also counts the transfer as done or skipped.
    fn begin_write(&mut self, _address: u32, _bytes: u32) -> bool {
        false
    }

    /// A block transfer of `bytes` contiguous bytes from `source` to
    /// `destination`, taken as a redirect: from now on (this frame) reads of
    /// the destination come from the source, nothing is copied
    /// (`ge_writes`). False: not taken, copy it (`begin_write`). The memory
    /// counts a taken transfer as done.
    fn redirect_write(&mut self, _destination: u32, _source: u32, _bytes: u32) -> bool {
        false
    }
}

/// Removes the cache/mirror bits from a PSP address.
pub fn physical(address: u32) -> u32 {
    address & 0x0FFF_FFFF
}

/// Whether a PSP address is memory at all: main RAM (0x08000000, 32 MiB
/// plus the 64 MiB models' extra), VRAM with its mirrors (0x04000000 ..
/// 0x04800000) or the scratchpad (0x00010000, 16 KiB). As PPSSPP's
/// Memory::IsValidAddress (Core/MemMap.h), which GE CALLs and JUMPs check.
pub fn is_valid_address(address: u32) -> bool {
    (address & 0x3E00_0000) == 0x0800_0000
        || (address & 0x3F80_0000) == 0x0400_0000
        || (address & 0xBFFF_C000) == 0x0001_0000
        || (0x0800_0000..0x0C00_0000).contains(&(address & 0x3F00_0000))
}

/// A physical VRAM address in one of VRAM's mirrors, as the address in the
/// 2 MiB of VRAM itself (other addresses unchanged). The GE sees VRAM at
/// 0x04000000 and mirrored at 0x04200000, 0x04400000 and 0x04600000 (the
/// first and last show depth in a swizzled layout, which we don't emulate),
/// and 0x04800000 and up wrap back (PPSSPP GPUCommon.cpp DoBlockTransfer,
/// TextureCacheCommon.cpp "Decoding texture from VRAM mirror"). Reading a
/// mirror address as it is crashed Adrenaline: nothing is mapped there on
/// the Vita (2026-11-06, MTX race, texture at 0x048066e1).
pub fn fold_vram_mirror(address: u32) -> u32 {
    if address & 0x0F00_0000 == 0x0400_0000 {
        address & !0x00E0_0000
    } else {
        address
    }
}

/// The PSP scratchpad (16 KiB), readable like RAM.
pub const SCRATCHPAD_START: u32 = 0x0001_0000;
pub const SCRATCHPAD_END: u32 = 0x0001_4000;

/// Where a read of PSP memory goes on the Vita (`place_read`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PspPlace {
    /// VRAM, this many bytes from its start (the PSP plugin's copy of it).
    Vram(u32),
    /// RAM or the scratchpad, at this physical address (mapped as it is).
    Mapped(u32),
}

/// Where `size` bytes of PSP memory at physical `address` can be read on
/// the Vita: VRAM (also through its mirrors, `fold_vram_mirror`), RAM or
/// the scratchpad; `None` for anything else: nothing is mapped there, and
/// reading it kills Adrenaline (C2-12828-1; see `fold_vram_mirror`).
pub fn place_read(address: u32, size: u32) -> Option<PspPlace> {
    let address = fold_vram_mirror(address);
    let inside = |start: u32, end: u32| {
        address >= start && address < end && address.checked_add(size).is_some_and(|last| last <= end)
    };
    if inside(VRAM_START, VRAM_END) {
        Some(PspPlace::Vram(address - VRAM_START))
    } else if inside(RAM_START, RAM_END) || inside(SCRATCHPAD_START, SCRATCHPAD_END) {
        Some(PspPlace::Mapped(address))
    } else {
        None
    }
}

/// True if `size` bytes starting at physical `address` are inside RAM or VRAM.
pub fn is_readable(address: u32, size: u32) -> bool {
    let inside = |start: u32, end: u32| address >= start && address.checked_add(size).is_some_and(|last| last <= end);
    inside(RAM_START, RAM_END) || inside(VRAM_START, VRAM_END)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vram_mirrors_fold_into_vram() {
        assert_eq!(fold_vram_mirror(0x0400_1230), 0x0400_1230);
        assert_eq!(fold_vram_mirror(0x0420_1230), 0x0400_1230);
        assert_eq!(fold_vram_mirror(0x0460_1230), 0x0400_1230);
        assert_eq!(fold_vram_mirror(0x0480_66E0), 0x0400_66E0);
        assert_eq!(fold_vram_mirror(0x04E0_0010), 0x0400_0010);
        assert_eq!(fold_vram_mirror(0x0890_0000), 0x0890_0000);
    }

    /// The crash of 2026-11-06 (from the Vita's core dump):
    /// AdrenalineDraw copied 128 bytes of a texture from PSP 0x048066E1 (Vita
    /// 0x6C8066E1, unmapped): a data abort. Such reads must land in VRAM.
    #[test]
    fn crash_20261106_texture_in_vram_mirror_reads_vram() {
        assert_eq!(place_read(0x0480_66E0, 128), Some(PspPlace::Vram(0x66E0)));
        assert_eq!(place_read(0x0480_66E1, 128), Some(PspPlace::Vram(0x66E1)));
    }

    #[test]
    fn reads_outside_psp_memory_are_refused() {
        // Past the end of VRAM, between VRAM and RAM, past RAM, address 0.
        assert_eq!(place_read(0x041F_FFF0, 32), None);
        assert_eq!(place_read(0x0500_0000, 4), None);
        assert_eq!(place_read(0x0C00_0000, 4), None);
        assert_eq!(place_read(0x0BFF_FFFC, 8), None);
        assert_eq!(place_read(0, 4), None);
        assert_eq!(place_read(0xFFFF_FFF0, 32), None);
        // Texture reads refused live in MTX's LA race (2026-11-06, crash-hunt
        // rounds 6-8, `badread=` in the capture log): without the check they
        // read unmapped Vita memory.
        assert_eq!(place_read(0x0E14_D890, 0x1_27D0), None);
        assert_eq!(place_read(0x06E7_6000, 0x48E8), None);
        assert_eq!(place_read(0x0662_DC10, 0x48E8), None);
        // Inside: RAM, the scratchpad, VRAM's last bytes.
        assert_eq!(place_read(0x0890_0000, 1024), Some(PspPlace::Mapped(0x0890_0000)));
        assert_eq!(place_read(0x0001_0000, 16), Some(PspPlace::Mapped(0x0001_0000)));
        assert_eq!(place_read(0x041F_FFF0, 16), Some(PspPlace::Vram(0x1F_FFF0)));
    }
}

#[cfg(test)]
pub mod test_memory {
    use super::*;
    use std::collections::HashMap;

    /// Sparse fake PSP memory for tests: only written words exist.
    #[derive(Default, Clone)]
    pub struct TestMemory {
        words: HashMap<u32, u32>,
    }

    impl TestMemory {
        /// Writes consecutive words starting at `address`.
        pub fn write(&mut self, address: u32, words: &[u32]) {
            for (index, word) in words.iter().enumerate() {
                self.words.insert(physical(address) + 4 * index as u32, *word);
            }
        }
    }

    impl PspMemory for TestMemory {
        fn read_u32(&mut self, address: u32) -> Option<u32> {
            self.words.get(&physical(address)).copied()
        }
    }
}
