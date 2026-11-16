//! Address ranges on the Vita (32-bit), checked without wrapping past the
//! top of the address space.

/// True if `bytes` bytes from `address` lie inside `start..end`.
pub fn inside(address: u32, bytes: u32, start: u32, end: u32) -> bool {
    address >= start && address.checked_add(bytes).is_some_and(|after| after <= end)
}

/// True if any byte of the `bytes` bytes from `address` (at least the one
/// at `address`) lies inside `start..end`; a range running past the top of
/// the address space ends there.
pub fn touches(address: u32, bytes: u32, start: u32, end: u32) -> bool {
    let after = address.saturating_add(bytes.max(1));
    address < end && after > start
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The emulator's window of PSP memory (io_hooks.rs).
    const WINDOW: (u32, u32) = (0x6800_0000, 0x7800_0000);

    #[test]
    fn buffers_in_psp_memory_touch_the_window() {
        // msfs.c's read into PSP RAM, and one ending right at its end.
        assert!(touches(0x6880_0000, 0x10_0000, WINDOW.0, WINDOW.1));
        assert!(touches(0x77FF_F000, 0x1000, WINDOW.0, WINDOW.1));
        // Starting below, reaching in.
        assert!(touches(0x67FF_FFF0, 0x20, WINDOW.0, WINDOW.1));
        // A zero-length call still counts its address.
        assert!(touches(0x6900_0000, 0, WINDOW.0, WINDOW.1));
    }

    #[test]
    fn buffers_outside_do_not() {
        // Adrenaline's settings in its own data segment (2026-11-14).
        assert!(!touches(0x8276_DA28, 36, WINDOW.0, WINDOW.1));
        assert!(!touches(0x67FF_FFF0, 0x10, WINDOW.0, WINDOW.1));
        assert!(!touches(0x7800_0000, 0x1000, WINDOW.0, WINDOW.1));
        // A pointer of all ones with a size that wraps: not inside.
        assert!(!touches(0xFFFF_FFFF, 0x7000_0000, WINDOW.0, WINDOW.1));
    }

    /// AdrenalineUser's code segment on the Vita, 2026-11-14.
    const TEXT: (u32, u32) = (0x8260_0000, 0x8260_0000 + 0xEAA30);

    #[test]
    fn a_word_of_all_ones_is_not_inside() {
        // The crash of 2026-11-14: menu.rs read a
        // data word 0xFFFFFFFF as a pointer to "Settings\0"; 0xFFFFFFFF + 9
        // wrapped to 8 and passed `address + bytes <= end`.
        assert!(!inside(0xFFFF_FFFF, 9, TEXT.0, TEXT.1));
        assert!(!inside(0xFFFF_FFF8, 9, 0, u32::MAX));
    }

    #[test]
    fn edges() {
        assert!(inside(TEXT.0, 9, TEXT.0, TEXT.1));
        assert!(inside(TEXT.1 - 9, 9, TEXT.0, TEXT.1));
        assert!(!inside(TEXT.1 - 8, 9, TEXT.0, TEXT.1));
        assert!(!inside(TEXT.0 - 1, 9, TEXT.0, TEXT.1));
        assert!(inside(TEXT.1, 0, TEXT.0, TEXT.1));
    }
}
