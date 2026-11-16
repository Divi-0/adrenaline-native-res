//! Reading a frame dump (format: `nativepsp_core::dump`).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};

use nativepsp_core::block_cache::{BlockSource, BLOCK_SIZE, BLOCK_WORDS};
use nativepsp_core::draw_stream::DeferredReader;
use nativepsp_core::dump::{Header, EVENT_WORDS, HEADER_BYTES, ORIGINAL_MAGIC};
use nativepsp_core::memory::physical;

/// What Sony's PSP emulator showed for the dumped frame.
pub struct OriginalPicture {
    pub width: usize,
    pub height: usize,
    /// RGBA (0xAABBGGRR), `width` pixels per row.
    pub pixels: Vec<u32>,
}

pub struct DumpFile {
    pub snapshot: Vec<u8>,
    pub original: Option<OriginalPicture>,
    /// (event type, arguments) in ring format.
    pub events: Vec<(u32, [u32; 4])>,
    pub memory: DumpMemory,
}

/// The PSP memory the dumped frame read: only those blocks exist, every
/// other address reads as unreadable (`None`), like unmapped memory.
pub struct DumpMemory {
    blocks: HashMap<u32, Vec<u8>>,
}

impl DumpMemory {
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// `size` bytes starting at `address`, if they were all dumped.
    pub fn bytes(&self, address: u32, size: u32) -> Option<Vec<u8>> {
        let mut out = Vec::with_capacity(size as usize);
        let mut at = physical(address);
        let end = at.checked_add(size)?;
        while at < end {
            let block_address = at & !(BLOCK_SIZE - 1);
            let block = self.blocks.get(&block_address)?;
            let start = (at - block_address) as usize;
            let take = ((end - at) as usize).min(block.len() - start);
            out.extend_from_slice(&block[start..start + take]);
            at += take as u32;
        }
        Some(out)
    }
}

/// The dump as the source of a `BlockCache`.
impl BlockSource for &DumpMemory {
    fn copy_block(&mut self, address: u32, out: &mut [u32; BLOCK_WORDS]) -> bool {
        let Some(block) = self.blocks.get(&address) else {
            return false;
        };
        for (word, bytes) in out.iter_mut().zip(block.chunks_exact(4)) {
            *word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        }
        true
    }

    /// Main RAM the dump holds, with `--copier` (`use_for_copier`).
    fn defer(&mut self, address: u32, bytes: u32) -> bool {
        DEFER_MAIN_RAM.load(Ordering::Relaxed) && address >= nativepsp_core::memory::RAM_START && self.bytes(address, bytes).is_some()
    }

    /// As the Vita's direct reads: only the words asked for.
    fn copy_words(&mut self, address: u32, out: &mut [u32]) -> bool {
        let block_address = address & !(BLOCK_SIZE - 1);
        let Some(block) = self.blocks.get(&block_address) else {
            return false;
        };
        let first = (address - block_address) as usize;
        for (word, bytes) in out.iter_mut().zip(block[first..].chunks_exact(4)) {
            *word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        }
        true
    }
}

/// Reads little-endian `u32`s from a byte slice.
struct Cursor<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl Cursor<'_> {
    fn take(&mut self, count: usize) -> Result<&[u8], String> {
        let slice = self
            .bytes
            .get(self.position..self.position + count)
            .ok_or_else(|| format!("dump ends early at byte {}", self.position))?;
        self.position += count;
        Ok(slice)
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
}

/// The optional trailer with the original picture (see `nativepsp_core::dump`).
fn read_original(cursor: &mut Cursor) -> Result<Option<OriginalPicture>, String> {
    if cursor.bytes.len() < cursor.position + 8 || cursor.take(8)? != ORIGINAL_MAGIC {
        return Ok(None);
    }
    let width = cursor.u32()? as usize;
    let height = cursor.u32()? as usize;
    let stride = cursor.u32()? as usize;
    let mut pixels = Vec::with_capacity(width * height);
    for _ in 0..height {
        let row = cursor.take(stride * 4)?;
        for x in 0..width {
            pixels.push(u32::from_le_bytes(row[x * 4..x * 4 + 4].try_into().unwrap()));
        }
    }
    Ok(Some(OriginalPicture { width, height, pixels }))
}

impl DumpFile {
    pub fn parse(bytes: &[u8]) -> Result<DumpFile, String> {
        let header = Header::read(bytes).ok_or("not a NativeRes dump (or another version)")?;
        let mut cursor = Cursor { bytes, position: HEADER_BYTES };

        let snapshot = cursor.take(header.snapshot_bytes as usize)?.to_vec();

        let mut events = Vec::new();
        for _ in 0..header.event_count {
            let mut words = [0u32; EVENT_WORDS];
            for word in words.iter_mut() {
                *word = cursor.u32()?;
            }
            events.push((words[0], [words[1], words[2], words[3], words[4]]));
        }

        if header.block_size != BLOCK_SIZE {
            return Err(format!("the dump has {}-byte blocks, expected {BLOCK_SIZE}", header.block_size));
        }
        let mut blocks = HashMap::new();
        for _ in 0..header.block_count {
            let address = cursor.u32()?;
            let data = cursor.take(header.block_size as usize)?.to_vec();
            blocks.insert(address, data);
        }

        let original = read_original(&mut cursor)?;

        Ok(DumpFile { snapshot, original, events, memory: DumpMemory { blocks } })
    }
}

/// `--copier`: main RAM the dump holds is left to the draw stream's copier
/// (`PspMemory::defer`), read back through `COPIER_READER` from the dump
/// given to `use_for_copier`.
static DEFER_MAIN_RAM: AtomicBool = AtomicBool::new(false);
static COPIER_DUMP: AtomicPtr<DumpMemory> = AtomicPtr::new(std::ptr::null_mut());

/// From now on main RAM is deferred and `COPIER_READER` reads `memory`,
/// which must outlive every use of the reader.
pub fn use_for_copier(memory: &DumpMemory) {
    COPIER_DUMP.store(memory as *const DumpMemory as *mut DumpMemory, Ordering::Release);
    DEFER_MAIN_RAM.store(true, Ordering::Release);
}

/// The draw stream's reader of the dump (`DeferredReader`), as the Vita's
/// copier thread reads PSP RAM.
pub const COPIER_READER: DeferredReader = DeferredReader { read: copier_read, sample: copier_sample };

fn copier_read(address: u32, out: &mut [u32]) -> bool {
    // Safe: set by `use_for_copier`, whose caller keeps the dump alive.
    let Some(memory) = (unsafe { COPIER_DUMP.load(Ordering::Acquire).as_ref() }) else {
        return false;
    };
    let Some(bytes) = memory.bytes(address, 4 * out.len() as u32) else {
        return false;
    };
    for (word, bytes) in out.iter_mut().zip(bytes.chunks_exact(4)) {
        *word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    }
    true
}

fn copier_sample(address: u32, positions: &[usize], out: &mut [u32]) -> bool {
    positions.iter().zip(out.iter_mut()).all(|(&position, word)| copier_read(address + 4 * position as u32, std::slice::from_mut(word)))
}
