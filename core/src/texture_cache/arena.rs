//! The texture arena: one big GPU-mapped memory block, made once by the
//! host (`Host::texture_arena_bytes`), that textures are carved out of in
//! 4 KiB pages.
//!
//! Why: every texture used to be a memory block of its own
//! (sceKernelAllocMemBlock + sceGxmMapMemory) in main memory, of which the
//! PSP emulator leaves us ~3 MiB in MTX's race. The cache was therefore
//! always "short of memory" and dropped every texture the last frame didn't
//! use; at each camera cut or respawn dozens of textures were decoded
//! again and frames weren't drawn (2026-11-10, a 22 s drive: 1,696
//! decodes, 93 textures without memory, 98 frames not drawn). The arena
//! lives in phycont memory, which has room, and handing out pages costs no
//! system call.
//!
//! A first-fit page allocator over a bitmap: simple, and the arena has at
//! most `MAX_PAGES` pages. Texture sizes are whole pages already
//! (`TextureShape::bytes`).

/// The arena hands out memory in pages of this size (GPU textures need far
/// less alignment; textures were whole pages before the arena too).
pub const PAGE_BYTES: u32 = 4096;
/// The biggest arena the bitmap can describe: 8 MiB.
pub const MAX_PAGES: usize = 2048;
const WORDS: usize = MAX_PAGES / 32;

/// Which pages of the arena are in use.
pub struct PageArena {
    /// One bit per page, set while the page is in use.
    used: [u32; WORDS],
    /// Pages in the arena; 0 while there is no arena.
    pages: u32,
    free_pages: u32,
}

impl PageArena {
    pub const fn new() -> PageArena {
        PageArena { used: [0; WORDS], pages: 0, free_pages: 0 }
    }

    /// Once the host made the arena: `bytes` of it (whole pages; more than
    /// `MAX_PAGES` pages are left unused). All pages are free.
    pub fn set_size(&mut self, bytes: u32) {
        self.pages = (bytes / PAGE_BYTES).min(MAX_PAGES as u32);
        self.free_pages = self.pages;
        self.used = [0; WORDS];
    }

    /// True once there is an arena.
    pub fn exists(&self) -> bool {
        self.pages > 0
    }

    pub fn free_bytes(&self) -> u32 {
        self.free_pages * PAGE_BYTES
    }

    pub fn used_bytes(&self) -> u32 {
        (self.pages - self.free_pages) * PAGE_BYTES
    }

    fn is_used(&self, page: u32) -> bool {
        self.used[(page / 32) as usize] & (1 << (page % 32)) != 0
    }

    fn mark(&mut self, first: u32, count: u32, used: bool) {
        for page in first..first + count {
            let (word, bit) = ((page / 32) as usize, 1 << (page % 32));
            if used {
                self.used[word] |= bit;
            } else {
                self.used[word] &= !bit;
            }
        }
    }

    /// The byte offset of `bytes` of free memory (rounded up to whole
    /// pages), now in use; None if no free run of pages is long enough.
    pub fn allocate(&mut self, bytes: u32) -> Option<u32> {
        let count = bytes.div_ceil(PAGE_BYTES).max(1);
        if count > self.free_pages {
            return None;
        }
        // First fit: the first run of `count` free pages.
        let mut run_start = 0;
        let mut run = 0;
        let mut page = 0;
        while page < self.pages {
            // A whole word in use: skip its 32 pages at once.
            if page % 32 == 0 && self.used[(page / 32) as usize] == u32::MAX {
                page += 32;
                run = 0;
                continue;
            }
            if self.is_used(page) {
                run = 0;
            } else {
                if run == 0 {
                    run_start = page;
                }
                run += 1;
                if run == count {
                    self.mark(run_start, count, true);
                    self.free_pages -= count;
                    return Some(run_start * PAGE_BYTES);
                }
            }
            page += 1;
        }
        None
    }

    /// Gives back what `allocate` returned for `bytes` at `offset`.
    pub fn free(&mut self, offset: u32, bytes: u32) {
        let count = bytes.div_ceil(PAGE_BYTES).max(1);
        let first = offset / PAGE_BYTES;
        debug_assert!((first..first + count).all(|page| page < self.pages && self.is_used(page)), "freeing free pages");
        self.mark(first, count, false);
        self.free_pages += count;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_arena_allocates_nothing() {
        let mut arena = PageArena::new();
        assert!(!arena.exists());
        assert_eq!(arena.allocate(4096), None);
    }

    #[test]
    fn first_fit_and_reuse() {
        let mut arena = PageArena::new();
        arena.set_size(16 * PAGE_BYTES);
        let a = arena.allocate(PAGE_BYTES).unwrap();
        let b = arena.allocate(3 * PAGE_BYTES).unwrap();
        let c = arena.allocate(100).unwrap(); // rounded up to a page
        assert_eq!((a, b, c), (0, PAGE_BYTES, 4 * PAGE_BYTES));
        assert_eq!(arena.free_bytes(), 11 * PAGE_BYTES);
        arena.free(b, 3 * PAGE_BYTES);
        // Two pages fit into b's hole; four don't, they go after c.
        assert_eq!(arena.allocate(2 * PAGE_BYTES), Some(PAGE_BYTES));
        assert_eq!(arena.allocate(4 * PAGE_BYTES), Some(5 * PAGE_BYTES));
        assert_eq!(arena.used_bytes(), 8 * PAGE_BYTES);
    }

    #[test]
    fn full_and_fragmented() {
        let mut arena = PageArena::new();
        arena.set_size(64 * PAGE_BYTES);
        let pages: Vec<u32> = (0..64).map(|_| arena.allocate(PAGE_BYTES).unwrap()).collect();
        assert_eq!(arena.allocate(PAGE_BYTES), None);
        // Every second page free: 32 pages free, but no two in a row.
        for offset in pages.iter().step_by(2) {
            arena.free(*offset, PAGE_BYTES);
        }
        assert_eq!(arena.free_bytes(), 32 * PAGE_BYTES);
        assert_eq!(arena.allocate(2 * PAGE_BYTES), None);
        assert_eq!(arena.allocate(PAGE_BYTES), Some(0));
    }

    #[test]
    fn runs_across_full_words() {
        let mut arena = PageArena::new();
        arena.set_size(MAX_PAGES as u32 * PAGE_BYTES + 12345); // more than the bitmap holds
        assert_eq!(arena.free_bytes(), MAX_PAGES as u32 * PAGE_BYTES);
        let first = arena.allocate(40 * PAGE_BYTES).unwrap(); // pages 0-39: word 0 full
        let second = arena.allocate(100 * PAGE_BYTES).unwrap();
        assert_eq!((first, second), (0, 40 * PAGE_BYTES));
        arena.free(first, 40 * PAGE_BYTES);
        assert_eq!(arena.allocate(41 * PAGE_BYTES), Some(140 * PAGE_BYTES));
        assert_eq!(arena.allocate(40 * PAGE_BYTES), Some(0));
        let rest = arena.free_bytes();
        assert!(arena.allocate(rest).is_some());
        assert_eq!(arena.free_bytes(), 0);
    }
}
