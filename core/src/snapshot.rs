//! Saving and restoring the capture state as bytes.
//!
//! Used for frame dumps: the Vita saves the state at the start of a frame,
//! the PC replay restores it and feeds the frame's events through the same
//! code. Every field is written explicitly as little-endian `u32`s, so the
//! format doesn't depend on how the compiler lays out structs (the Vita and
//! the PC use different targets and Rust versions).

/// Writes `u32`s into a byte buffer. If the buffer is too small, further
/// writes are dropped and `overflowed()` reports it.
pub struct Writer<'a> {
    bytes: &'a mut [u8],
    len: usize,
    overflowed: bool,
}

impl<'a> Writer<'a> {
    pub fn new(bytes: &'a mut [u8]) -> Self {
        Writer { bytes, len: 0, overflowed: false }
    }

    pub fn u32(&mut self, value: u32) {
        if self.len + 4 > self.bytes.len() {
            self.overflowed = true;
            return;
        }
        self.bytes[self.len..self.len + 4].copy_from_slice(&value.to_le_bytes());
        self.len += 4;
    }

    pub fn bool(&mut self, value: bool) {
        self.u32(value as u32);
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn overflowed(&self) -> bool {
        self.overflowed
    }
}

/// Reads `u32`s back. Every read returns `None` once the data runs out.
pub struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Reader { bytes, position: 0 }
    }

    pub fn u32(&mut self) -> Option<u32> {
        let bytes = self.bytes.get(self.position..self.position + 4)?;
        self.position += 4;
        Some(u32::from_le_bytes(bytes.try_into().ok()?))
    }

    pub fn bool(&mut self) -> Option<bool> {
        Some(self.u32()? != 0)
    }

    pub fn position(&self) -> usize {
        self.position
    }
}

/// Bumped whenever the saved layout changes. 2: GE matrices added; 3: bone
/// matrices added; 4: the loaded palette's address instead of its words
/// (`recorder::LoadedPalette`); 5: each list's own offset
/// (`DisplayList::offset`); 6: each list's context (`ge_context`; the
/// saved states themselves are not in the snapshot); 7: the texture
/// generation matrix (`ge_state::Matrices::texgen`). Dumps of older
/// versions still load.
pub const SNAPSHOT_VERSION: u32 = 7;
