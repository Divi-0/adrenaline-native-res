//! Tests of the texture cache against a fake host: PSP memory in plain
//! vectors, GPU textures as numbers whose "contents" are the hash of the
//! data decoded into them, and a worker that runs when the test says so.
//!
//! Several tests replay bugs fixed before on the Vita; their names say which.

use std::collections::HashMap;
use std::fmt;

use super::gpu::{GpuFormat, GpuTexture, TextureShape};
use super::refresh::{run_refresh, RefreshJob, RefreshOutcome, RefreshResult};
use super::staging::STAGING_SIZE;
use super::*;
use crate::frame::TextureDescriptor;
use crate::texture::{self, format, DecodeError};

const RAM_START: u32 = 0x0880_0000;
const RAM_SIZE: u32 = 4 * 1024 * 1024;
const FREE_MEMORY: u32 = 64 * 1024 * 1024;

struct FakeHost {
    /// PSP RAM and VRAM (the fake has no separate VRAM copy: pending copies
    /// are only simulated by `vram`).
    ram: Vec<u32>,
    vram_words: Vec<u32>,
    vram: VramState,
    next_texture: usize,
    /// Textures made and not freed, with their shape.
    live: HashMap<usize, TextureShape>,
    /// The live textures in the arena: their byte offset there.
    in_arena: HashMap<usize, u32>,
    /// Bytes of the texture arena (0: none, as most tests have it).
    arena_bytes: u32,
    /// What each texture shows: the hash of the data decoded into it.
    contents: HashMap<usize, u32>,
    creates: u32,
    /// (texture, level) of every decode, in order.
    decoded_levels: Vec<(usize, u8)>,
    decodes: u32,
    gpu_unswizzles: u32,
    /// Free memory before our textures.
    memory: u32,
    now: u64,
    captured_frame_number: u32,
    worker: bool,
    job: Option<RefreshJob>,
    result: Option<RefreshResult>,
    log: String,
    /// PSP RAM as the staging copies (`Host::copy`) see it, while the game
    /// writes the real `ram` meanwhile (`None`: they see `ram`). On the
    /// Vita the game runs on while the display copies its textures out:
    /// a copy and a later direct read can see different data.
    copies_see: Option<Vec<u32>>,
}

impl FakeHost {
    fn new() -> FakeHost {
        FakeHost {
            ram: vec![0; (RAM_SIZE / 4) as usize],
            vram_words: vec![0; (VRAM_SIZE / 4) as usize],
            vram: VramState::default(),
            next_texture: 1,
            live: HashMap::new(),
            in_arena: HashMap::new(),
            arena_bytes: 0,
            contents: HashMap::new(),
            creates: 0,
            decoded_levels: Vec::new(),
            decodes: 0,
            gpu_unswizzles: 0,
            memory: FREE_MEMORY,
            now: 1_000_000,
            captured_frame_number: 0,
            worker: false,
            job: None,
            result: None,
            log: String::new(),
            copies_see: None,
        }
    }

    fn words(&self, address: u32, size: u32) -> Option<&[u32]> {
        let (words, start) = if (VRAM_START..VRAM_START + VRAM_SIZE).contains(&address) {
            (&self.vram_words, VRAM_START)
        } else if (RAM_START..RAM_START + RAM_SIZE).contains(&address) {
            (&self.ram, RAM_START)
        } else {
            return None;
        };
        let first = ((address - start) / 4) as usize;
        let count = size.div_ceil(4) as usize;
        words.get(first..first + count)
    }

    /// The game writes `value` to the word at `address`.
    fn write(&mut self, address: u32, value: u32) {
        let address = address & 0x0FFF_FFFF;
        if address >= VRAM_START && address < VRAM_START + VRAM_SIZE {
            self.vram_words[((address - VRAM_START) / 4) as usize] = value;
        } else {
            self.ram[((address - RAM_START) / 4) as usize] = value;
        }
    }

    /// What the texture shows (the hash of what was decoded into it).
    fn shows(&self, lookup: Lookup) -> u32 {
        match lookup {
            Lookup::Ready(texture) => self.contents[&texture.handle],
            other => panic!("no texture: {:?}", other),
        }
    }

    /// The hash of the texture's data as it is in PSP memory now.
    fn data_hash(&self, texture: &TextureDescriptor) -> u32 {
        let layout = StagedLayout::of(texture);
        let mut staged = vec![0u8; STAGING_SIZE];
        let size = stage(self, texture, &layout, &mut staged);
        texture::hash_texture(&staged[..size as usize])
    }

    /// The worker runs its job now.
    fn run_worker(&mut self) {
        let job = self.job.take().expect("no refresh job");
        let mut staging = vec![0u8; STAGING_SIZE];
        self.result = Some(run_refresh(self, &job, &mut staging));
    }
}

impl Host for FakeHost {
    fn texture_arena_bytes(&mut self) -> u32 {
        self.arena_bytes
    }

    fn create_texture(&mut self, shape: &TextureShape, arena_offset: Option<u32>) -> Option<GpuTexture> {
        let texture = self.next_texture;
        self.next_texture += 1;
        self.creates += 1;
        if let Some(offset) = arena_offset {
            assert!(offset + shape.bytes() <= self.arena_bytes, "outside the arena");
            // No other live texture's pages overlap these.
            for (other, other_offset) in &self.in_arena {
                let other_end = other_offset + self.live[other].bytes();
                assert!(offset + shape.bytes() <= *other_offset || other_end <= offset, "overlaps texture {other}");
            }
            assert!(!shape.render_target, "render targets need memory of their own");
            self.in_arena.insert(texture, offset);
        }
        self.live.insert(texture, *shape);
        Some(GpuTexture { handle: texture, arena_offset })
    }

    fn free_texture(&mut self, texture: GpuTexture) {
        assert!(self.live.remove(&texture.handle).is_some(), "freed twice: {:?}", texture);
        assert_eq!(self.in_arena.remove(&texture.handle), texture.arena_offset);
    }

    fn decode(
        &mut self,
        texture: GpuTexture,
        gpu_format: GpuFormat,
        as_texture: &TextureDescriptor,
        data: &[u8],
        _palette: &[u32],
        _smooth: bool,
        level: u8,
    ) -> Result<(), DecodeError> {
        let shape = self.live[&texture.handle];
        assert_eq!(shape.format, gpu_format);
        assert!(level < shape.levels.max(1), "a level the texture has");
        let level_size = |size: u16| (size >> level).max(1);
        assert_eq!((level_size(shape.width), level_size(shape.height)), (as_texture.width, as_texture.height));
        self.decoded_levels.push((texture.handle, level));
        if data.is_empty() {
            return Err(DecodeError::BadAddress);
        }
        self.decodes += 1;
        self.contents.insert(texture.handle, texture::hash_texture(data));
        Ok(())
    }

    fn gpu_unswizzle_ready(&mut self, _width: u16, _height: u16) -> bool {
        true
    }

    fn unswizzle_on_gpu(&mut self, wanted: &TextureDescriptor, target: GpuTexture, raw: GpuTexture) -> bool {
        assert!(self.live[&target.handle].render_target && self.live[&raw.handle].render_target);
        // The raw copy holds the PSP's 16-bit texels as they are; the target
        // is what draws sample.
        let raw_shape = self.live[&raw.handle];
        assert!(raw_shape.format.is_16_bit());
        assert_eq!(raw_shape.bytes(), wanted.width as u32 * wanted.height as u32 * 2);
        assert_eq!(self.live[&target.handle].format, GpuFormat::for_texture(wanted, false, false));
        self.gpu_unswizzles += 1;
        let hash = self.data_hash(wanted);
        self.contents.insert(target.handle, hash);
        true
    }

    fn wait_for_gpu(&mut self) {}

    /// Main memory: arena textures don't use any.
    fn free_memory(&self) -> u32 {
        let used: u32 =
            self.live.iter().filter(|(texture, _)| !self.in_arena.contains_key(texture)).map(|(_, shape)| shape.bytes()).sum();
        self.memory.saturating_sub(used)
    }

    fn psp_memory(&self, address: u32, size: u32) -> Option<&[u8]> {
        let words = self.words(address, size)?;
        // Safe: plain words viewed as bytes, within the same allocation.
        let bytes = unsafe { core::slice::from_raw_parts(words.as_ptr() as *const u8, words.len() * 4) };
        Some(&bytes[..size as usize])
    }

    fn psp_words(&self, address: u32, size: u32) -> Option<&[u32]> {
        self.words(address, size)
    }

    fn copy(&self, target: &mut [u8], source: &[u8]) {
        let ram_start = self.ram.as_ptr() as usize;
        let offset = (source.as_ptr() as usize).wrapping_sub(ram_start);
        match &self.copies_see {
            Some(seen) if offset < self.ram.len() * 4 => {
                // Safe: plain words viewed as bytes, within the snapshot.
                let bytes = unsafe { core::slice::from_raw_parts(seen.as_ptr() as *const u8, seen.len() * 4) };
                target.copy_from_slice(&bytes[offset..offset + source.len()]);
            }
            _ => target.copy_from_slice(source),
        }
    }

    fn vram(&self) -> VramState {
        self.vram
    }

    fn captured_frame_number(&self) -> u32 {
        self.captured_frame_number
    }

    fn has_refresh_worker(&self) -> bool {
        self.worker
    }

    fn start_refresh(&mut self, job: &RefreshJob) {
        assert!(self.job.is_none() && self.result.is_none(), "worker busy");
        self.job = Some(*job);
    }

    fn finished_refresh(&self) -> Option<RefreshResult> {
        self.result
    }

    fn end_refresh(&mut self) {
        self.result = None;
    }

    fn now_us(&self) -> u64 {
        self.now
    }

    fn log(&mut self, text: fmt::Arguments) {
        self.log.push_str(&std::format!("{}", text));
    }
}

/// The cache, its host and the display loop around them.
struct Test {
    cache: Box<TextureCache>,
    host: FakeHost,
    staging: Vec<u8>,
    render_count: u32,
    frame_number: u32,
    smooth: bool,
    /// `FrameStart::wide`.
    wide: bool,
    /// How long ago the display loop started when the frame's textures are
    /// looked up.
    late_us: u64,
}

impl Test {
    fn new() -> Test {
        Test {
            cache: Box::new(TextureCache::new()),
            host: FakeHost::new(),
            staging: vec![0; STAGING_SIZE],
            render_count: 0,
            frame_number: 100,
            smooth: false,
            wide: false,
            late_us: 0,
        }
    }

    /// Draws a frame using `textures`; returns what each one got.
    fn frame(&mut self, textures: &[TextureDescriptor]) -> Vec<Lookup> {
        self.render_count += 1;
        self.frame_number += 1;
        let start = FrameStart {
            frame_number: self.frame_number,
            render_count: self.render_count,
            smooth: self.smooth,
            wide: self.wide,
            trace_address: 0,
            loop_start_us: (self.host.now - self.late_us) as u32,
            render_targets: [0; MAX_RENDER_TARGETS],
        };
        self.cache.begin_frame(&mut self.host, start);
        let lookups: Vec<Lookup> =
            textures.iter().map(|wanted| self.cache.texture_for(&mut self.host, &mut self.staging, &[], wanted)).collect();
        self.cache.end_frame(lookups.contains(&Lookup::Waiting));
        lookups
    }

    /// `frame` for one texture.
    fn use_texture(&mut self, wanted: &TextureDescriptor) -> Lookup {
        self.frame(core::slice::from_ref(wanted))[0]
    }

    /// `use_texture` in a frame whose capture sampled the texture's data
    /// as it is now (`Frame::texture_samples`).
    fn use_texture_captured(&mut self, wanted: &TextureDescriptor) -> Lookup {
        self.cache.lookup_sample = staging::quick_hash(&self.host, wanted);
        let lookup = self.use_texture(wanted);
        self.cache.lookup_sample = 0;
        lookup
    }

    /// Frames that don't use the texture.
    fn idle_frames(&mut self, count: u32) {
        for _ in 0..count {
            self.frame(&[]);
        }
    }
}

fn texture(address: u32, width: u16, height: u16, texture_format: u8) -> TextureDescriptor {
    TextureDescriptor { address, buffer_width: width as u32, width, height, format: texture_format, ..Default::default() }
}

/// 64x64 ABGR8888: 16 KiB, small enough for the display thread's re-checks.
fn small_texture(address: u32) -> TextureDescriptor {
    texture(address, 64, 64, format::ABGR8888)
}

#[test]
fn a_texture_is_decoded_once_and_then_cached() {
    let mut test = Test::new();
    let wanted = small_texture(RAM_START);
    let first = test.use_texture(&wanted);
    assert!(matches!(first, Lookup::Ready(_)));
    for _ in 0..20 {
        assert_eq!(test.use_texture(&wanted), first);
    }
    assert_eq!(test.host.decodes, 1);
    assert_eq!(test.cache.bytes(), 16 * 1024);
}

/// MTX's level-select previews (2026-11-10): with the renderer off we drew
/// no frames, so `render_count` stood still while the game loaded the next
/// track's preview into the same RAM. Switched on again, the old decode
/// still counted as just verified, and the new palettes over the old
/// indices showed noise. Game frames passing without ours (more than a
/// re-check interval) make every texture due for a re-check at once.
#[test]
fn textures_are_rechecked_after_game_frames_we_did_not_draw() {
    let mut test = Test::new();
    let wanted = small_texture(RAM_START);
    let old = test.use_texture(&wanted);
    test.host.write(RAM_START + 100, 0x1234_5678);
    test.frame_number += 3 * VERIFY_INTERVAL;
    let new = test.use_texture(&wanted);
    assert!(matches!(new, Lookup::Ready(_)));
    assert_ne!(new, old, "re-checked and decoded again in the first frame after the gap");
    assert_eq!(test.host.decodes, 2);
    // A savestate load starts the frame numbers again: the same.
    test.host.write(RAM_START + 200, 0x1111_2222);
    test.frame_number = 5;
    assert_ne!(test.use_texture(&wanted), new);
    assert_eq!(test.host.decodes, 3);
    // (Ordinary frames don't re-check early: see the next test.)
}

#[test]
fn a_changed_texture_is_found_at_its_next_recheck_and_decoded_into_a_fresh_texture() {
    let mut test = Test::new();
    let wanted = small_texture(RAM_START);
    let old = test.use_texture(&wanted);
    test.host.write(RAM_START + 100, 0x1234_5678);
    // Within VERIFY_INTERVAL frames the old pixels stay (one word the
    // quick samples don't read; a new picture they find at once, see below).
    for _ in 1..VERIFY_INTERVAL {
        assert_eq!(test.use_texture(&wanted), old);
    }
    let new = test.use_texture(&wanted);
    assert_ne!(new, old, "the GPU may still read the old texture");
    assert_eq!(test.host.shows(new), test.host.data_hash(&wanted));
    assert_eq!(test.host.decodes, 2);
}

/// MTX's menus load the next screen's pictures into the same memory (rider
/// -> bike select: tiles of the old screen; level select: the old preview
/// flashing in). The capture's sample of the frame's texture data
/// (`Frame::texture_samples`) shows the new picture in the first frame that
/// draws it, not up to VERIFY_INTERVAL frames later.
#[test]
fn a_new_picture_in_the_same_memory_shows_in_the_first_frame_drawing_it() {
    let mut test = Test::new();
    let wanted = texture(RAM_START, 64, 64, format::CLUT8);
    let old = test.use_texture_captured(&wanted);
    assert_eq!(test.use_texture_captured(&wanted), old);
    load_new_picture(&mut test, 1);
    let new = test.use_texture_captured(&wanted);
    assert_ne!(new, old, "decoded again at once");
    assert_eq!(test.host.shows(new), test.host.data_hash(&wanted));
    assert_eq!(test.host.decodes, 2);
    // Then cached again: no decode while nothing changes.
    for _ in 0..2 * VERIFY_INTERVAL {
        assert_eq!(test.use_texture_captured(&wanted), new);
    }
    assert_eq!(test.host.decodes, 2);
}

/// The other way round (rider -> bike select: one frame of colourful
/// noise): the game draws the last frame of the old screen, then loads the
/// next screen's pictures into the same memory before the display looks at
/// that frame's textures. The frame was drawn with the old pixels (its
/// capture-time sample says so, and its palette is the old one): the cached
/// texture stays, the new pixels are decoded with the first frame drawing
/// them.
#[test]
fn pixels_written_after_the_frame_was_drawn_wait_for_the_next_frame() {
    let mut test = Test::new();
    let wanted = texture(RAM_START, 64, 64, format::CLUT8);
    let old = test.use_texture_captured(&wanted);
    let captured_before_the_load = staging::quick_hash(&test.host, &wanted);
    load_new_picture(&mut test, 2);
    test.cache.lookup_sample = captured_before_the_load;
    assert_eq!(test.use_texture(&wanted), old, "the frame drew the old pixels");
    assert_ne!(test.use_texture_captured(&wanted), old, "the next frame draws the new ones");
    assert_eq!(test.host.decodes, 2);
}

/// rider -> bike select, first visit ("moved on", 20 textures in
/// one frame): the frame drew picture 1, but by the time the display reads
/// the textures the game has loaded picture 2 there. Whatever we decode is
/// not what the frame showed: noticed (`shows_other_pixels`), so the
/// display leaves the frame out. The next frame, drawn with picture 2, is
/// fine.
#[test]
fn textures_rewritten_after_the_frame_was_drawn_are_noticed() {
    let mut test = Test::new();
    let wanted = texture(RAM_START, 64, 64, format::CLUT8);
    test.use_texture_captured(&wanted);
    load_new_picture(&mut test, 1);
    let drawn_with = staging::quick_hash(&test.host, &wanted);
    load_new_picture(&mut test, 2);
    test.cache.lookup_sample = drawn_with;
    test.use_texture(&wanted);
    assert!(test.cache.shows_other_pixels(&wanted), "the frame can't be drawn as the GE drew it");
    test.cache.lookup_sample = staging::quick_hash(&test.host, &wanted);
    test.use_texture(&wanted);
    assert!(!test.cache.shows_other_pixels(&wanted), "the next frame is");
    test.cache.lookup_sample = 0;
    assert!(!test.cache.shows_other_pixels(&wanted), "no capture sample: nothing to say");
}

/// Level select: the track previews are big direct-colour textures, loaded
/// into the same memory whenever the selection moves; single frames showed
/// another track's preview or a mix. They get the same capture-time check.
#[test]
fn big_textures_rewritten_after_the_frame_was_drawn_are_noticed_too() {
    let mut test = Test::new();
    let wanted = big_texture();
    test.use_texture_captured(&wanted);
    let words = (texture::data_size(&wanted).unwrap() / 4) as u32;
    let fill = |test: &mut Test, value: u32| {
        for word in 0..words {
            test.host.write(RAM_START + 4 * word, value ^ word);
        }
    };
    fill(&mut test, 0x1111_1111);
    let drawn_with = staging::quick_hash(&test.host, &wanted);
    fill(&mut test, 0x2222_2222);
    test.cache.lookup_sample = drawn_with;
    test.use_texture(&wanted);
    assert!(test.cache.shows_other_pixels(&wanted));
    test.cache.lookup_sample = staging::quick_hash(&test.host, &wanted);
    test.use_texture(&wanted);
    assert!(!test.cache.shows_other_pixels(&wanted));
}

/// MTX rider <-> bike select (2026-11-17, "graphics glitch again"): for
/// ~8 frames of the wipe, background tiles showed pieces of other pictures
/// (dirt photos, black triangles); Sony's picture clean, the captured
/// frames identical to the clean ones after them. The game loads the next
/// screen's tiles into the same RAM while the display re-checks them: the
/// staged copy we decoded caught the memory mid-load (picture 2), the
/// cached texture's quick sample was read from PSP memory a moment later
/// (picture 3, loaded meanwhile). The next frames, drawn with picture 3,
/// had capture samples equal to that quick sample, so the half-loaded
/// pixels stayed until the regular re-check VERIFY_INTERVAL frames later.
/// A cached texture's quick sample must describe the very bytes decoded.
#[test]
fn a_decode_racing_with_the_game_loading_new_pixels_is_redone_by_the_next_frame() {
    let mut test = Test::new();
    let wanted = texture(RAM_START, 64, 64, format::CLUT8);
    let old = test.use_texture_captured(&wanted);
    // The frame drew picture 2; the display's copy catches picture 2, the
    // game has loaded picture 3 by the time the copy is done.
    load_new_picture(&mut test, 2);
    let drawn_with = staging::quick_hash(&test.host, &wanted);
    let picture_2 = test.host.data_hash(&wanted);
    test.host.copies_see = Some(test.host.ram.clone());
    load_new_picture(&mut test, 3);
    test.cache.lookup_sample = drawn_with;
    let racing = test.use_texture(&wanted);
    assert_ne!(racing, old);
    assert_eq!(test.host.shows(racing), picture_2, "decoded what the copy saw");
    assert!(!test.cache.shows_other_pixels(&wanted), "which is what the frame drew");
    test.cache.lookup_sample = 0;
    test.host.copies_see = None;
    // The next frame drew picture 3: decoded at once, not VERIFY_INTERVAL
    // frames later.
    let next = test.use_texture_captured(&wanted);
    assert_eq!(test.host.shows(next), test.host.data_hash(&wanted), "picture 3, as the frame drew it");
    assert_eq!(test.host.decodes, 3);
}

/// The same race on the path where the re-check finds the old pixels: the
/// copy still sees picture 1 (its hash is the cached one: kept), the game
/// has written picture 2 when the quick sample is read. The cached texture
/// must not take picture 2's quick sample, or the frames drawn with
/// picture 2 keep showing picture 1 (rider <-> bike select, 2026-11-17).
#[test]
fn a_recheck_racing_with_the_game_loading_new_pixels_keeps_the_old_sample() {
    let mut test = Test::new();
    let wanted = texture(RAM_START, 64, 64, format::CLUT8);
    let old = test.use_texture_captured(&wanted);
    test.host.copies_see = Some(test.host.ram.clone());
    load_new_picture(&mut test, 2);
    // The frame drew picture 2; our copy still sees picture 1.
    assert_eq!(test.use_texture_captured(&wanted), old, "the copy found the cached pixels");
    test.host.copies_see = None;
    let next = test.use_texture_captured(&wanted);
    assert_ne!(next, old);
    assert_eq!(test.host.shows(next), test.host.data_hash(&wanted), "picture 2, as the frame drew it");
    assert_eq!(test.host.decodes, 2);
}

/// Every word of the 64x64 CLUT8 texture at RAM_START rewritten.
fn load_new_picture(test: &mut Test, seed: u32) {
    for word in 0..64 * 64 / 4 {
        test.host.write(RAM_START + 4 * word, 0x0101_0101u32.wrapping_mul(word % 7 + seed));
    }
}

/// World map stutter (2026-10-31): FF1 animates its water by switching
/// palettes, and with the palette in the key every map tile counted as new.
#[test]
fn index_textures_are_the_same_texture_with_another_palette() {
    let mut test = Test::new();
    let mut tile = texture(RAM_START, 32, 32, format::CLUT8);
    tile.palette_hash = 1;
    let first = test.use_texture(&tile);
    tile.palette_hash = 2;
    assert_eq!(test.use_texture(&tile), first);
    assert_eq!(test.host.decodes, 1);

    // CLUT16 is decoded to RGBA with its colours: the palette matters.
    let mut wide = texture(RAM_START + 0x10000, 32, 32, format::CLUT16);
    wide.palette_hash = 1;
    let first = test.use_texture(&wide);
    wide.palette_hash = 2;
    assert_ne!(test.use_texture(&wide), first);

    // So does it for CLUT8 with the smooth filter (decoded to RGBA).
    test.smooth = true;
    tile.palette_hash = 1;
    let smooth = test.use_texture(&tile);
    tile.palette_hash = 2;
    assert_ne!(test.use_texture(&tile), smooth);
}

/// The wide mode (God of War, DESIGN "sixty-fifth session"): palette and
/// 8888 textures of power-of-two sizes as `Wide16`, decoded with their
/// palette (so the palette is part of the key); others as before.
#[test]
fn wide_mode_widens_palette_and_8888_textures() {
    let mut test = Test::new();
    test.host.arena_bytes = 6 * 1024 * 1024;
    test.wide = true;
    let mut tile = texture(RAM_START, 32, 64, format::CLUT8);
    tile.palette_hash = 1;
    let Lookup::Ready(first) = test.use_texture(&tile) else { panic!() };
    assert_eq!(test.host.live[&first.handle].format, GpuFormat::Wide16);
    assert_eq!(test.host.live[&first.handle].bytes(), 32 * 64 * 8);
    tile.palette_hash = 2;
    let Lookup::Ready(second) = test.use_texture(&tile) else { panic!() };
    assert_ne!(second, first, "another palette: other colours");
    let mut nibbles = texture(RAM_START + 0x10000, 16, 16, format::CLUT4);
    nibbles.palette_hash = 3;
    let Lookup::Ready(nibbles) = test.use_texture(&nibbles) else { panic!() };
    assert_eq!(test.host.live[&nibbles.handle].format, GpuFormat::Wide16);
    let Lookup::Ready(direct) = test.use_texture(&small_texture(RAM_START + 0x20000)) else { panic!() };
    assert_eq!(test.host.live[&direct.handle].format, GpuFormat::Wide16);
    // 16-bit, sizes that aren't powers of two, mip levels: as before.
    let Lookup::Ready(sixteen) = test.use_texture(&texture(RAM_START + 0x30000, 64, 64, format::ABGR4444)) else { panic!() };
    assert_eq!(test.host.live[&sixteen.handle].format, GpuFormat::Abgr4444);
    let Lookup::Ready(odd) = test.use_texture(&texture(RAM_START + 0x40000, 48, 64, format::CLUT8)) else { panic!() };
    assert_eq!(test.host.live[&odd.handle].format, GpuFormat::Palette8);
    let mut levels = texture(RAM_START + 0x50000, 64, 64, format::CLUT8);
    levels.flags = 1 << crate::frame::texture_flags::LEVELS_SHIFT;
    assert!(!gpu::wide_texture(&levels));
    // Bigger than WIDE_MAX_TEXELS: as before.
    let Lookup::Ready(big) = test.use_texture(&texture(RAM_START + 0x60000, 256, 128, format::CLUT8)) else { panic!() };
    assert_eq!(test.host.live[&big.handle].format, GpuFormat::Palette8);
}

/// Wide textures only while the arena keeps WIDE_ARENA_RESERVE free; then
/// the next ones stay palette textures, and without an arena none is wide.
/// The GPU crash of 2026-11-26 ("GPU Driver detects GPU Crash", twice at the
/// same scene of the lab test debug/lab/gather74, combination 12): the gather
/// twin of the opaque textured program (nativeres_textured_pspgather_f)
/// drawing a 64 x 64 strided view, no mip levels, of a 128 x 64 block. The
/// texture unit does not apply a strided texture's stride to a gather: it
/// reads other memory. Such a texture must never be gathered, whatever else
/// is true of it.
#[test]
fn gpu_crash_gather_of_a_strided_view() {
    // The crashing input: the view as the lab made it (not marked as a target's).
    assert!(!gpu::gather_texture(64, 64, 0, true, false));
    // As the renderer makes them: views of parts of its targets.
    assert!(!gpu::gather_texture(64, 64, 0, true, true));
    assert!(!gpu::gather_texture(256, 256, 0, true, true));
    // A view of a target is kept off the gather even if it were not strided.
    assert!(!gpu::gather_texture(64, 64, 0, false, true));
}

#[test]
fn gather_textures_are_whole_power_of_two_textures_without_levels() {
    assert!(gpu::gather_texture(64, 64, 0, false, false));
    assert!(gpu::gather_texture(128, 32, 1, false, false));
    assert!(gpu::gather_texture(512, 512, 0, false, false));
    // Sizes that are no powers of two, mip levels.
    assert!(!gpu::gather_texture(48, 64, 0, false, false));
    assert!(!gpu::gather_texture(64, 480, 0, false, false));
    assert!(!gpu::gather_texture(64, 64, 2, false, false));
    assert!(!gpu::gather_texture(64, 64, 7, false, false));
}

#[test]
fn wide_textures_leave_room_in_the_arena() {
    let mut test = Test::new();
    test.host.arena_bytes = WIDE_ARENA_RESERVE + 3 * 64 * 128 * 8;
    test.wide = true;
    let formats: Vec<GpuFormat> = (0..5)
        .map(|i| {
            let Lookup::Ready(made) = test.use_texture(&texture(RAM_START + i * 0x10000, 64, 128, format::CLUT8)) else { panic!() };
            test.host.live[&made.handle].format
        })
        .collect();
    assert_eq!(formats[..3], [GpuFormat::Wide16; 3]);
    assert_eq!(formats[3..], [GpuFormat::Palette8; 2]);

    let mut test = Test::new();
    test.wide = true;
    let Lookup::Ready(made) = test.use_texture(&texture(RAM_START, 64, 64, format::CLUT8)) else { panic!() };
    assert_eq!(test.host.live[&made.handle].format, GpuFormat::Palette8, "no arena: no wide textures");
}

/// The host's reservation in the arena: none before the arena exists, then
/// pages no texture gets, counted out of the wide textures' room.
#[test]
fn host_reservations_come_out_of_the_arena() {
    let mut test = Test::new();
    test.host.arena_bytes = 64 * 4096;
    assert_eq!(test.cache.reserve_arena(4096), None, "no arena before the first frame");
    test.idle_frames(1);
    let reserved = test.cache.reserve_arena(60 * 4096).unwrap();
    assert_eq!(test.cache.reserve_arena(8 * 4096), None, "only 4 pages left");
    // Textures get only the pages outside the reservation.
    let Lookup::Ready(made) = test.use_texture(&texture(RAM_START, 64, 64, format::CLUT8)) else { panic!() };
    let offset = test.host.in_arena[&made.handle];
    let bytes = test.host.live[&made.handle].bytes();
    assert!(offset + bytes <= reserved || offset >= reserved + 60 * 4096, "texture inside the reservation");
}

/// A texture kept as a palette texture for lack of room is widened once
/// the arena has room again, at most WIDE_UPGRADES_PER_FRAME a frame.
#[test]
fn textures_kept_for_lack_of_room_are_widened_later() {
    let mut test = Test::new();
    // Room for four wide ones beside the reserve and the palette textures.
    test.host.arena_bytes = WIDE_ARENA_RESERVE + 4 * 64 * 128 * 8 + 32 * 1024;
    test.wide = true;
    let tiles: Vec<TextureDescriptor> = (0..8).map(|i| texture(RAM_START + i * 0x10000, 64, 128, format::CLUT8)).collect();
    let formats = |test: &mut Test, tiles: &[TextureDescriptor]| -> Vec<GpuFormat> {
        test.frame(tiles)
            .iter()
            .map(|lookup| match lookup {
                Lookup::Ready(made) => test.host.live[&made.handle].format,
                _ => panic!(),
            })
            .collect()
    };
    let first = formats(&mut test, &tiles);
    assert_eq!(first[..4], [GpuFormat::Wide16; 4]);
    assert_eq!(first[4..], [GpuFormat::Palette8; 4]);
    // The first half no longer used: once the arena lets them go, the
    // others are widened, at most WIDE_UPGRADES_PER_FRAME a frame, also
    // past the deadline (God of War's loop always is), never waited for.
    test.late_us = TEXTURE_DEADLINE_US as u64 + 1_000;
    let mut wide_before = 0;
    for _ in 0..ARENA_IDLE_FRAMES + RETIRE_FRAMES + 8 {
        let decodes = test.host.decodes;
        let now = formats(&mut test, &tiles[4..]);
        let wide = now.iter().filter(|&&f| f == GpuFormat::Wide16).count();
        assert!(wide - wide_before <= WIDE_UPGRADES_PER_FRAME as usize && (test.host.decodes - decodes) as usize == wide - wide_before);
        wide_before = wide;
    }
    assert_eq!(formats(&mut test, &tiles[4..]), [GpuFormat::Wide16; 4]);
}

/// A wide palette texture is as opaque as its palette table would be on
/// the display side (`find_palettes`): worked out from the palette when
/// decoded, so draws choose the same programs as with the palette texture.
#[test]
fn wide_palette_textures_are_opaque_as_their_palette() {
    let mut test = Test::new();
    test.host.arena_bytes = 6 * 1024 * 1024;
    test.wide = true;
    test.frame(&[]);
    let mut tile = texture(RAM_START, 32, 32, format::CLUT4);
    tile.clut_format = 3 | 0xFF << 8; // 8888 entries
    tile.palette_word_count = 16;
    tile.palette_hash = 1;
    let opaque: Vec<u32> = (0..16).map(|i| 0xFF00_0000 | i * 0x0101).collect();
    let Lookup::Ready(made) = test.cache.texture_for(&mut test.host, &mut test.staging, &opaque, &tile) else { panic!() };
    assert_eq!(test.host.live[&made.handle].format, GpuFormat::Wide16);
    assert_eq!(test.cache.last_opaque_rows, texture::ALL_ROWS_OPAQUE);
    let mut see_through = opaque.clone();
    see_through[7] &= 0x00FF_FFFF;
    tile.palette_hash = 2;
    test.cache.texture_for(&mut test.host, &mut test.staging, &see_through, &tile);
    assert_eq!(test.cache.last_opaque_rows, 0);
}

#[test]
fn switching_the_wide_mode_decodes_again() {
    let mut test = Test::new();
    test.host.arena_bytes = 6 * 1024 * 1024;
    let tile = texture(RAM_START, 32, 32, format::CLUT8);
    let Lookup::Ready(narrow) = test.use_texture(&tile) else { panic!() };
    assert_eq!(test.host.live[&narrow.handle].format, GpuFormat::Palette8);
    test.wide = true;
    let Lookup::Ready(wide) = test.use_texture(&tile) else { panic!() };
    assert_eq!(test.host.live[&wide.handle].format, GpuFormat::Wide16);
    test.wide = false;
    let Lookup::Ready(back) = test.use_texture(&tile) else { panic!() };
    assert_eq!(test.host.live[&back.handle].format, GpuFormat::Palette8);
    assert_eq!(test.host.decodes, 3);
}

#[test]
fn a_filter_change_decodes_again() {
    let mut test = Test::new();
    let tile = texture(RAM_START, 32, 32, format::CLUT8);
    test.use_texture(&tile);
    test.smooth = true;
    let Lookup::Ready(smooth) = test.use_texture(&tile) else { panic!() };
    assert_eq!(test.host.live[&smooth.handle].format, GpuFormat::Rgba8888);
    assert_eq!(test.host.decodes, 2);
}

#[test]
fn a_replaced_texture_is_reused_only_once_the_gpu_is_done_with_it() {
    let mut test = Test::new();
    let a = small_texture(RAM_START);
    let b = small_texture(RAM_START + 0x10000);
    let c = small_texture(RAM_START + 0x20000);
    let Lookup::Ready(old) = test.use_texture(&a) else { panic!() };
    test.host.write(RAM_START, 1);
    test.idle_frames(VERIFY_INTERVAL);
    test.use_texture(&a); // re-checked: decoded into a fresh texture, `old` retired
    assert_eq!(test.host.creates, 2);
    // The next frame: `old` may still be on the GPU, so `b` gets a new one.
    let Lookup::Ready(for_b) = test.use_texture(&b) else { panic!() };
    assert_ne!(for_b, old);
    assert_eq!(test.host.creates, 3);
    test.idle_frames(RETIRE_FRAMES);
    let Lookup::Ready(for_c) = test.use_texture(&c) else { panic!() };
    assert_eq!(for_c, old, "same size, retired long enough: reused");
    assert_eq!(test.host.creates, 3);
}

#[test]
fn retired_textures_nobody_reuses_are_freed() {
    let mut test = Test::new();
    let a = small_texture(RAM_START);
    test.use_texture(&a);
    test.host.write(RAM_START, 1);
    test.idle_frames(VERIFY_INTERVAL);
    test.use_texture(&a);
    assert_eq!(test.host.live.len(), 2);
    test.idle_frames(RETIRE_FRAMES + SPARE_FRAMES);
    assert_eq!(test.host.live.len(), 1);
    assert_eq!(test.cache.bytes(), 16 * 1024);
}

/// Over budget, the least recently used textures go first.
#[test]
fn the_budget_evicts_the_least_recently_used() {
    let mut test = Test::new();
    // 512x512 ABGR8888: 1 MiB each, 20 of them (budget 16 MiB).
    let big: Vec<TextureDescriptor> = (0..20u32).map(|i| texture(RAM_START + i * 0x1000, 512, 512, format::ABGR8888)).collect();
    // (They overlap in memory; each one's first word makes it new.)
    for (i, wanted) in big.iter().enumerate() {
        test.host.write(wanted.address, i as u32 + 1);
        test.use_texture(wanted);
        test.idle_frames(RETIRE_FRAMES); // retired ones can be freed
    }
    assert!(test.cache.bytes() <= BUDGET_BYTES, "{} bytes", test.cache.bytes());
    let decodes = test.host.decodes;
    test.use_texture(&big[19]);
    assert_eq!(test.host.decodes, decodes, "the newest is still cached");
    test.use_texture(&big[0]);
    assert_eq!(test.host.decodes, decodes + 1, "the oldest was evicted");
}

/// Prologue lightning (2026-10-31): a VRAM texture unused for a while was
/// drawn from its old copy while the plugin was still copying the new one.
#[test]
fn a_vram_texture_unused_for_a_while_waits_for_its_copy() {
    let mut test = Test::new();
    let bolt = texture(VRAM_START + 0x10_0000, 64, 64, format::CLUT8);
    let old = test.use_texture(&bolt);
    test.idle_frames(VERIFY_INTERVAL + 1);
    // Meanwhile its range was dropped from the requests; the game wrote new
    // pixels and the range is asked for again: the copy is pending.
    test.host.write(bolt.address, 0xABCD);
    test.host.vram.generation += 1;
    assert_eq!(test.use_texture(&bolt), Lookup::Waiting);
    test.host.vram.copied_generation = test.host.vram.generation;
    let new = test.use_texture(&bolt);
    assert_ne!(new, old);
    assert_eq!(test.host.shows(new), test.host.data_hash(&bolt));
}

#[test]
fn a_vram_texture_used_recently_stays_while_a_copy_is_pending() {
    let mut test = Test::new();
    let tile = texture(VRAM_START, 64, 64, format::CLUT8);
    let old = test.use_texture(&tile);
    test.host.vram.generation += 1;
    for _ in 0..20 {
        assert_eq!(test.use_texture(&tile), old, "no re-check while the copy is pending");
    }
    // New VRAM textures wait for the copy.
    assert_eq!(test.use_texture(&texture(VRAM_START + 0x8000, 64, 64, format::CLUT8)), Lookup::Waiting);
}

/// Menu close (2026-10-30): FF1 loads the field's tiles into VRAM where the
/// menu's graphics were; the plugin reports changed blocks and our VRAM
/// textures are re-checked at once instead of up to 8 frames later.
#[test]
fn changed_vram_blocks_make_vram_textures_recheck_at_once() {
    let mut test = Test::new();
    let tile = texture(VRAM_START, 64, 64, format::CLUT8);
    let in_ram = texture(RAM_START, 64, 64, format::CLUT8);
    test.frame(&[tile, in_ram]);
    test.host.write(tile.address, 7);
    test.host.write(in_ram.address, 7);
    test.host.vram.changes += 3;
    let lookups = test.frame(&[tile, in_ram]);
    assert_eq!(test.host.shows(lookups[0]), test.host.data_hash(&tile), "VRAM texture re-checked");
    assert_ne!(test.host.shows(lookups[1]), test.host.data_hash(&in_ram), "RAM texture waits for its interval");
}

/// Volatile textures (FF1's battle intro): after the data stops changing,
/// the texture must stop being decoded. An unhashed decode used to reset
/// the remembered hash to 0, so the check never found the data unchanged.
#[test]
fn a_volatile_texture_stops_decoding_once_its_data_stays_the_same() {
    let mut test = Test::new();
    let wanted = small_texture(RAM_START);
    for frame in 0..40 {
        test.host.write(RAM_START, frame + 1); // changes every frame
        test.use_texture(&wanted);
    }
    assert!(test.host.decodes > 20, "decoded {} times", test.host.decodes);
    let still = test.host.data_hash(&wanted);
    for _ in 0..12 {
        test.use_texture(&wanted);
    }
    let decodes = test.host.decodes;
    for _ in 0..30 {
        let shown = test.use_texture(&wanted);
        assert_eq!(test.host.shows(shown), still);
    }
    assert_eq!(test.host.decodes, decodes, "still decoding a texture that no longer changes");
}

#[test]
fn rechecks_share_a_budget_per_frame() {
    let mut test = Test::new();
    // 16 textures of 16 KiB: 256 KiB, twice the re-check budget.
    let textures: Vec<TextureDescriptor> = (0..16u32).map(|i| small_texture(RAM_START + i * 0x4000)).collect();
    test.frame(&textures);
    for wanted in &textures {
        test.host.write(wanted.address, 9);
    }
    test.idle_frames(VERIFY_INTERVAL - 1);
    let decodes = test.host.decodes;
    test.frame(&textures);
    assert_eq!(test.host.decodes - decodes, VERIFY_BYTES_PER_FRAME / (16 * 1024));
    test.frame(&textures);
    assert_eq!(test.host.decodes - decodes, 16, "the rest in the next frame");
}

/// FF1's battle start: ~25 new textures in one frame. Past the deadline new
/// textures wait (the frame isn't drawn), but at most MAX_DEFERRED_FRAMES
/// frames in a row.
#[test]
fn past_the_deadline_new_textures_wait_a_few_frames_at_most() {
    let mut test = Test::new();
    let wanted = small_texture(RAM_START);
    test.late_us = TEXTURE_DEADLINE_US as u64 + 1;
    for _ in 0..MAX_DEFERRED_FRAMES {
        assert_eq!(test.use_texture(&wanted), Lookup::Waiting);
    }
    assert!(matches!(test.use_texture(&wanted), Lookup::Ready(_)));
}

/// A texture that failed for lack of memory is not tried again every frame,
/// but after RETRY_FAILED_FRAMES (MTX's HUD stayed missing for good when it
/// was never retried); with changed data at once.
#[test]
fn a_texture_that_failed_for_memory_is_retried_later() {
    let mut test = Test::new();
    let wanted = small_texture(RAM_START);
    test.host.memory = 0; // no memory for it
    assert_eq!(test.use_texture(&wanted), Lookup::Unavailable);
    assert!(test.host.log.contains("out of texture memory"));
    test.host.memory = FREE_MEMORY;
    assert_eq!(test.use_texture(&wanted), Lookup::Unavailable, "same data: not retried at once");
    test.idle_frames(RETRY_FAILED_FRAMES);
    assert!(matches!(test.use_texture(&wanted), Lookup::Ready(_)), "tried again later");

    let other = small_texture(RAM_START + 0x1000);
    test.host.memory = 0;
    assert_eq!(test.use_texture(&other), Lookup::Unavailable);
    test.host.memory = FREE_MEMORY;
    test.host.write(RAM_START + 0x1000, 5);
    assert!(matches!(test.use_texture(&other), Lookup::Ready(_)), "changed data: at once");
}

/// A texture with mip levels (menu "NativeRes Distant Textures") becomes
/// one GPU texture with all its levels, each staged from its own address
/// and decoded into its level; its memory counts every level.
#[test]
fn a_texture_with_mip_levels_decodes_every_level() {
    use crate::frame::{texture_flags, TextureLevels};
    let mut test = Test::new();
    let mut wanted = small_texture(RAM_START);
    wanted.flags |= 2 << texture_flags::LEVELS_SHIFT; // levels 0, 1, 2
    let mut levels = TextureLevels::default();
    let mut address = RAM_START;
    for (level, log2) in [6u8, 5, 4].into_iter().enumerate() {
        levels.address[level] = address;
        levels.buffer_width[level] = 1 << log2;
        levels.log2_size[level] = log2 | (log2 << 4);
        address += (1u32 << log2) * (1 << log2) * 4;
    }
    test.cache.lookup_levels = levels;
    let Lookup::Ready(texture) = test.use_texture(&wanted) else { panic!("decoded") };
    assert_eq!(test.host.decoded_levels, [(texture.handle, 0), (texture.handle, 1), (texture.handle, 2)]);
    let shape = test.host.live[&texture.handle];
    assert_eq!(shape.levels, 3);
    assert_eq!(shape.bytes(), ((64 * 64 + 32 * 32 + 16 * 16) * 4u32).div_ceil(0x1000) * 0x1000);
}

/// 256x256 ABGR8888: 256 KiB, big enough for samples and the worker.
fn big_texture() -> TextureDescriptor {
    texture(RAM_START, 256, 256, format::ABGR8888)
}

#[test]
fn a_big_texture_change_shows_in_its_samples_at_the_next_use() {
    let mut test = Test::new();
    test.host.worker = true;
    let rays = big_texture();
    test.use_texture(&rays);
    test.host.write(RAM_START, 0x55); // the first sampled word
    let shown = test.use_texture(&rays);
    assert_eq!(test.host.shows(shown), test.host.data_hash(&rays));
}

/// 512x512 CLUT8: 256 KiB of indices, Kenka Bancho's cloud wipe mask.
fn big_palette_texture() -> TextureDescriptor {
    texture(RAM_START, 512, 512, format::CLUT8)
}

#[test]
fn a_big_palette_texture_change_shows_in_its_samples_at_the_next_use() {
    let mut test = Test::new();
    test.host.worker = true;
    let mask = big_palette_texture();
    test.use_texture_captured(&mask);
    // The first word: sampled by `sample_hash`, not by the capture's
    // quick sample.
    assert!(!texture::quick_sample_positions(&mask).unwrap().contains(&0));
    test.host.write(RAM_START, 0x55);
    let shown = test.use_texture_captured(&mask);
    assert_eq!(test.host.shows(shown), test.host.data_hash(&mask));
}

/// Kenka Bancho's whiteout after the train (2026-11-28): the game rewrites
/// a 512x512 CLUT8 cloud mask (and its palette) every frame. Its index
/// data was checked only by the capture's 8 sampled words, which missed
/// the rewrites: frames 2-13 of the wipe drew the first frame's mask under
/// the current palette, the hole jumped back and forth, until the 8-frame
/// re-check found it.
#[test]
fn a_palette_texture_rewritten_every_frame_is_never_drawn_stale() {
    let mut test = Test::new();
    test.host.worker = true;
    let mask = big_palette_texture();
    test.use_texture_captured(&mask);
    let quick = texture::quick_sample_positions(&mask).unwrap();
    for frame in 1..=12u32 {
        // Words the capture's samples don't read: in each KiB the one
        // `sample_hash` reads (its column moves 68 bytes a KiB).
        for kib in 0..64 {
            let offset = kib * 1024 + (kib * 68) % 1024;
            assert!(!quick.contains(&((offset / 4) as usize)));
            test.host.write(RAM_START + offset, frame * 0x0101_0101 + kib);
        }
        let shown = test.use_texture_captured(&mask);
        assert_eq!(test.host.shows(shown), test.host.data_hash(&mask), "frame {frame} of the wipe");
    }
}

/// FF1 reuses its font texture for the field right after the menu's last
/// frame. A refresh copied after the change must not be swapped in for a
/// frame captured before it (that frame showed garbage instead of text).
#[test]
fn a_refresh_is_only_used_for_frames_after_its_copy() {
    let mut test = Test::new();
    test.host.worker = true;
    let font = big_texture();
    let old = test.use_texture(&font);
    test.host.write(RAM_START + 4, 0x77); // not a sampled word: only the worker sees it
    test.idle_frames(VERIFY_INTERVAL - 1);
    assert_eq!(test.use_texture(&font), old);
    assert!(test.host.job.is_some(), "re-check handed to the worker");
    // The worker copies the data while the capture has finished frame F.
    test.host.captured_frame_number = test.frame_number + 2;
    test.host.run_worker();
    assert_eq!(test.host.result.unwrap().outcome, RefreshOutcome::Decoded);
    // Frames up to F still drew with the old pixels.
    assert_eq!(test.use_texture(&font), old);
    assert_eq!(test.use_texture(&font), old);
    let new = test.use_texture(&font);
    assert_ne!(new, old);
    assert_eq!(test.host.shows(new), test.host.data_hash(&font));
    assert!(test.host.result.is_none(), "the worker is idle again");
}

#[test]
fn an_unchanged_refresh_keeps_the_texture() {
    let mut test = Test::new();
    test.host.worker = true;
    let font = big_texture();
    test.use_texture(&font);
    // The first re-check decodes anyway: a new big texture isn't hashed.
    test.idle_frames(VERIFY_INTERVAL - 1);
    test.use_texture(&font);
    test.host.run_worker();
    assert_eq!(test.host.result.unwrap().outcome, RefreshOutcome::Decoded);
    let hashed = test.use_texture(&font);
    // The next one finds the same data.
    test.idle_frames(VERIFY_INTERVAL - 1);
    test.use_texture(&font);
    test.host.run_worker();
    assert_eq!(test.host.result.unwrap().outcome, RefreshOutcome::Unchanged);
    assert_eq!(test.use_texture(&font), hashed);
    assert!(test.host.result.is_none());
    // The fresh textures went back into the retired list, and are freed later.
    test.idle_frames(RETIRE_FRAMES + SPARE_FRAMES);
    assert_eq!(test.host.live.len(), 1);
}

#[test]
fn a_refresh_for_an_evicted_entry_is_dropped() {
    let mut test = Test::new();
    test.host.worker = true;
    let font = big_texture();
    test.use_texture(&font);
    test.host.write(RAM_START + 4, 0x77);
    test.idle_frames(VERIFY_INTERVAL - 1);
    test.use_texture(&font);
    // Filter switched meanwhile: the entry is evicted and decoded again.
    test.smooth = true;
    let smooth = test.use_texture(&font);
    test.host.run_worker();
    assert_eq!(test.use_texture(&font), smooth);
    assert!(test.host.result.is_none());
}

#[test]
fn big_swizzled_16_bit_textures_are_unswizzled_on_the_gpu() {
    let mut test = Test::new();
    let mut ground = texture(RAM_START, 512, 256, format::BGR5650);
    ground.swizzled = 1;
    let shown = test.use_texture(&ground);
    assert_eq!(test.host.gpu_unswizzles, 1);
    assert_eq!(test.host.decodes, 0);
    assert_eq!(test.host.shows(shown), test.host.data_hash(&ground));
    // The raw copy for the GPU was retired (render-target textures are
    // kept until memory is short).
    assert_eq!(test.host.live.len(), 2);
    let Lookup::Ready(target) = shown else { panic!() };
    assert!(test.host.live[&target.handle].render_target);
    assert_eq!(test.host.live[&target.handle].format, GpuFormat::Bgr565);
}

/// 16-bit textures stay 16-bit on the GPU: the Vita's texture unit widens
/// their channels as the GE does, also through the linear filter (measured
/// 2026-11-23, see `GpuFormat`); 8888 would only double memory and reads.
/// With the smooth filter they are decoded into 8888 (it changes texels).
#[test]
fn textures_with_16_bit_texels_stay_16_bit() {
    let gpu_format = |psp_format, smooth| GpuFormat::for_texture(&texture(RAM_START, 64, 64, psp_format), smooth, false);
    assert_eq!(gpu_format(format::BGR5650, false), GpuFormat::Bgr565);
    assert_eq!(gpu_format(format::ABGR5551, false), GpuFormat::Abgr1555);
    assert_eq!(gpu_format(format::ABGR4444, false), GpuFormat::Abgr4444);
    assert_eq!(gpu_format(format::BGR5650, true), GpuFormat::Rgba8888);
    // Where an 8888 decode happens (smooth filter), it is the GE's widening.
    for k in 0..32u16 {
        let ge = (k << 3 | k >> 2) as u32;
        assert_eq!(crate::color::rgba_from_5650(k) & 0xFF, ge);
        assert_eq!(crate::color::rgba_from_5551(k << 5) >> 8 & 0xFF, ge);
    }
    for k in 0..64u16 {
        assert_eq!(crate::color::rgba_from_5650(k << 5) >> 8 & 0xFF, (k << 2 | k >> 4) as u32);
    }
}

#[test]
fn a_big_swizzled_5551_texture_is_unswizzled_on_the_gpu() {
    let mut test = Test::new();
    let mut frame_copy = texture(RAM_START, 512, 512, format::ABGR5551);
    frame_copy.swizzled = 1;
    let shown = test.use_texture(&frame_copy);
    assert_eq!((test.host.gpu_unswizzles, test.host.decodes), (1, 0));
    let Lookup::Ready(target) = shown else { panic!() };
    assert_eq!(test.host.live[&target.handle].format, GpuFormat::Abgr1555);
}

#[test]
fn the_smooth_filter_keeps_16_bit_textures_off_the_gpu_unswizzle() {
    let mut test = Test::new();
    test.smooth = true;
    let mut frame_copy = texture(RAM_START, 512, 512, format::ABGR5551);
    frame_copy.swizzled = 1;
    test.use_texture(&frame_copy);
    assert_eq!((test.host.gpu_unswizzles, test.host.decodes), (0, 1));
}

#[test]
fn tracing_logs_every_lookup_of_the_texture() {
    let mut test = Test::new();
    let wanted = small_texture(RAM_START);
    test.render_count += 1;
    let start = FrameStart {
        frame_number: 5,
        render_count: test.render_count,
        trace_address: RAM_START,
        loop_start_us: test.host.now as u32,
        ..Default::default()
    };
    test.cache.begin_frame(&mut test.host, start);
    test.cache.texture_for(&mut test.host, &mut test.staging, &[], &wanted);
    assert!(test.host.log.contains("trace 08800000 64x64 fmt 3 frame 5 (captured 0) render 1: new decoded"), "{}", test.host.log);
}

#[test]
fn stats_count_decodes_and_bytes() {
    let mut test = Test::new();
    test.use_texture(&small_texture(RAM_START));
    let stats = test.cache.take_stats();
    assert_eq!((stats.decodes, stats.hashed_bytes, stats.texture_bytes), (1, 16 * 1024, 16 * 1024));
    let stats = test.cache.take_stats();
    assert_eq!((stats.decodes, stats.texture_bytes), (0, 16 * 1024));
}

#[test]
fn a_framebuffer_we_draw_ourselves_comes_from_its_render_target() {
    let mut test = Test::new();
    let mut frame = crate::frame::new_boxed_frame();
    // MTX Mototrax samples the other framebuffer (512x512 ABGR8888 at
    // 0x04088000) while drawing into 0x04000000.
    let framebuffer = texture(VRAM_START + 0x8_8000, 512, 512, format::ABGR8888);
    let start = FrameStart {
        frame_number: 1,
        render_count: 1,
        loop_start_us: test.host.now as u32,
        render_targets: [VRAM_START + 0x8_8000, 0, 0, 0, 0, 0, 0, 0],
        ..Default::default()
    };
    let mut out = [Lookup::Unavailable; MAX_TEXTURES];
    frame.add_texture(framebuffer);
    test.cache.textures_for_frame(&mut test.host, &mut test.staging, &frame, start, &mut out);
    assert_eq!(out[0], Lookup::RenderTarget);
    assert_eq!(test.host.decodes, 0, "the framebuffer isn't decoded");

    // The same memory read as palette indices is something else: decoded.
    frame.clear();
    frame.add_texture(texture(VRAM_START + 0x8_8000, 64, 64, format::CLUT8));
    test.cache.textures_for_frame(&mut test.host, &mut test.staging, &frame, start, &mut out);
    assert_ne!(out[0], Lookup::RenderTarget);
}

/// God of War's title (2026-11-22) samples the other screen and five
/// off-screen framebuffers we draw ourselves: six render targets. With room
/// for four names, the two smallest bloom levels (0x04162400, 0x04162600)
/// were decoded from VRAM every frame (Sony's pixels, not ours).
#[test]
fn every_target_a_frame_names_comes_from_its_render_target() {
    let mut test = Test::new();
    let mut frame = crate::frame::new_boxed_frame();
    let addresses = [0x0_0000, 0x16_1800, 0x18_1800, 0x16_2000, 0x16_2400, 0x16_2600].map(|offset| VRAM_START + offset);
    let mut start = FrameStart { frame_number: 1, render_count: 1, loop_start_us: test.host.now as u32, ..Default::default() };
    start.render_targets[..addresses.len()].copy_from_slice(&addresses);
    for &address in &addresses {
        frame.add_texture(texture(address, 128, 128, format::ABGR8888));
    }
    let mut out = [Lookup::Unavailable; MAX_TEXTURES];
    test.cache.textures_for_frame(&mut test.host, &mut test.staging, &frame, start, &mut out);
    assert_eq!(out[..addresses.len()], [Lookup::RenderTarget; 6]);
    assert_eq!(test.host.decodes, 0, "no framebuffer is decoded");
}

/// God of War's game (2026-11-23): seven 64x64 shadow tiles drawn as
/// framebuffers 0x04161880, 0x04161900, .. inside the 512x64 strip at
/// 0x04161800 (a render target of ours, `offscreen::placement`), and read
/// one by one by the shadows projected onto the ground. They were decoded
/// from VRAM every frame, where the game's bloom chain (0x04162000, 8888)
/// overwrites the strip at the frame's end: now and then a copy had the
/// bloom's pixels and the ground showed them as a coloured grid.
#[test]
fn tiles_of_a_target_come_from_its_render_target() {
    let mut test = Test::new();
    let mut frame = crate::frame::new_boxed_frame();
    let strip = VRAM_START + 0x16_1800;
    let mut start = FrameStart { frame_number: 1, render_count: 1, loop_start_us: test.host.now as u32, ..Default::default() };
    start.render_targets[0] = strip;
    let wide = |address: u32, width: u16| TextureDescriptor { buffer_width: 1024, ..texture(address, width, 64, format::BGR5650) };
    frame.add_texture(wide(strip, 512));
    for x in 1..7u32 {
        frame.add_texture(wide(strip + 2 * 64 * x, 64));
    }
    let mut out = [Lookup::Unavailable; MAX_TEXTURES];
    test.cache.textures_for_frame(&mut test.host, &mut test.staging, &frame, start, &mut out);
    assert_eq!(out[..7], [Lookup::RenderTarget; 7]);
    assert_eq!(test.host.decodes, 0, "no tile is decoded from VRAM");
}

/// Respawn stutter (2026-11-10): with ~3 MiB of main memory free in MTX's
/// race, below MEMORY_RESERVE, the cache dropped every texture the last
/// frame didn't use, and each camera cut decoded dozens again (1,696
/// decodes in a 22 s drive). Textures in the arena use no main memory, so
/// they stay cached while unused.
#[test]
fn arena_textures_stay_cached_while_main_memory_is_short() {
    let mut test = Test::new();
    test.host.arena_bytes = 4 * 1024 * 1024;
    test.host.memory = 3 * 1024 * 1024;
    let track: Vec<TextureDescriptor> = (0..40u32).map(|i| small_texture(RAM_START + i * 0x4000)).collect();
    let rider = small_texture(RAM_START + 0x10_0000);
    test.frame(&track);
    assert_eq!(test.host.decodes, 40);
    assert!(test.host.in_arena.len() == 40, "all in the arena");
    // A camera cut: only the rider for a while, then the track again.
    for _ in 0..20 {
        test.use_texture(&rider);
    }
    test.frame(&track);
    assert_eq!(test.host.decodes, 41, "the track's textures were still cached");
    assert_eq!(test.cache.take_stats().evictions, 0);
}

/// The arena full: new textures get memory of their own (as before the
/// arena), and textures unused for ARENA_IDLE_FRAMES leave the arena so
/// that new ones find room again.
#[test]
fn a_full_arena_falls_back_to_own_blocks_and_frees_idle_textures() {
    let mut test = Test::new();
    // 64x64 ABGR8888 = 16 KiB = 4 pages; the arena holds 8 of them.
    test.host.arena_bytes = 8 * 16 * 1024;
    let old: Vec<TextureDescriptor> = (0..8u32).map(|i| small_texture(RAM_START + i * 0x4000)).collect();
    test.frame(&old);
    assert_eq!(test.host.in_arena.len(), 8);
    let new = small_texture(RAM_START + 0x10_0000);
    let Lookup::Ready(texture) = test.use_texture(&new) else { panic!("no texture") };
    assert_eq!(texture.arena_offset, None, "arena full: a block of its own");

    // The old ones unused for long enough: they leave the (short) arena.
    test.idle_frames(ARENA_IDLE_FRAMES + RETIRE_FRAMES + 2);
    assert!(test.cache.take_stats().evictions > 0);
    let newer = small_texture(RAM_START + 0x20_0000);
    let Lookup::Ready(texture) = test.use_texture(&newer) else { panic!("no texture") };
    assert!(texture.arena_offset.is_some(), "room in the arena again");
}

/// NFS UR's attract movie (2026-11-21, black with the renderer on): two
/// 1 MiB video buffers drawn every frame, one of them new each frame, while
/// the title's textures (~2.5 MiB) still filled the 6 MiB arena. Each new
/// movie frame needs a fresh texture while the last frames' ones wait as
/// retired (the GPU may read them): the arena ran full, main memory was
/// short, and as the retired pages counted as free the arena never counted
/// as short, so the title's idle textures never left. A full arena now
/// frees textures no frame used for ARENA_IDLE_FRAMES (the GPU is long done
/// with them) to make room. Scaled: 512 KiB textures, a 3 MiB arena.
#[test]
fn a_movie_finds_arena_room_from_idle_textures() {
    let mut test = Test::new();
    test.host.arena_bytes = 3 * 1024 * 1024;
    test.host.memory = 2 * 1024 * 1024; // no room for blocks of their own
    let unit = |offset: u32| texture(RAM_START + offset, 512, 256, format::ABGR8888);
    let title = [unit(0), unit(0x8_0000)];
    let movie = [unit(0x10_0000), unit(0x18_0000)];
    test.frame(&title);
    assert_eq!(test.host.in_arena.len(), 2);
    for frame in 0..120u32 {
        // The video decoder writes one buffer per frame (the first word is
        // sampled), both are drawn.
        test.host.write(movie[(frame % 2) as usize].address, frame + 1);
        let lookups = test.frame(&movie);
        if frame >= ARENA_IDLE_FRAMES + RETIRE_FRAMES + 2 {
            for (k, lookup) in lookups.iter().enumerate() {
                assert!(matches!(lookup, Lookup::Ready(_)), "movie frame {frame}, buffer {k}: {lookup:?}");
                assert_eq!(test.host.shows(*lookup), test.host.data_hash(&movie[k]), "movie frame {frame}, buffer {k}");
            }
        }
    }
}

/// Freed textures give their arena pages back: churning through far more
/// textures than the arena holds never runs it out of pages, and the
/// arena's count matches the live arena textures.
#[test]
fn arena_pages_come_back() {
    let mut test = Test::new();
    test.host.arena_bytes = 1024 * 1024;
    for round in 0..50u32 {
        // A new 128x128 texture (64 KiB) every frame, each used once.
        let wanted = texture(RAM_START + round * 0x1000, 128, 128, format::ABGR8888);
        test.host.write(wanted.address, round + 1);
        let Lookup::Ready(texture) = test.use_texture(&wanted) else { panic!("no texture in round {round}") };
        assert!(texture.arena_offset.is_some(), "round {round}: in the arena");
        // Unused long enough to leave the arena, and the GPU is done with it.
        test.idle_frames(ARENA_IDLE_FRAMES + RETIRE_FRAMES + 2);
    }
    // Everything unused: freed after the idle and spare times.
    test.idle_frames(ARENA_IDLE_FRAMES + RETIRE_FRAMES + SPARE_FRAMES + 10);
    let used_by_cache = test.cache.bytes();
    let in_arena: u32 = test.host.in_arena.keys().map(|texture| test.host.live[texture].bytes()).sum();
    assert_eq!(test.cache.take_stats().arena_bytes, in_arena);
    assert!(used_by_cache >= in_arena);
}

/// `find`'s hints are only hints: all of them pointing at the wrong slot
/// (as after slots were reused by other textures) still finds every
/// texture, and no texture is taken for another.
#[test]
fn wrong_find_hints_still_find_the_right_texture() {
    let mut test = Test::new();
    let textures: Vec<TextureDescriptor> = (0..20u32).map(|i| small_texture(RAM_START + i * 0x4000)).collect();
    let first: Vec<Lookup> = test.frame(&textures);
    for hint in test.cache.hints.iter_mut() {
        *hint = [7, 7];
    }
    assert_eq!(test.frame(&textures), first);
    assert_eq!(test.host.decodes, 20, "nothing decoded again");
    let new = small_texture(RAM_START + 0x10_0000);
    assert!(!first.contains(&test.use_texture(&new)), "a new texture isn't an old one");
}

/// Slow, not wrong (MTX, 2026-11-10): two textures of every frame in the
/// same hint bucket took each other's hint, so both scanned all 512 entries
/// every frame (16-47 scans per frame). With two hints per bucket, neither
/// scans once both were found.
#[test]
fn two_textures_sharing_a_hint_bucket_do_not_scan_every_frame() {
    let mut test = Test::new();
    let first = small_texture(RAM_START);
    let bucket = test.cache.hint_bucket(&first);
    let second = (1..100_000u32)
        .map(|i| small_texture(RAM_START + i * 0x400))
        .find(|texture| test.cache.hint_bucket(texture) == bucket)
        .expect("a texture in the same bucket");
    let pair = [first, second];
    test.frame(&pair);
    test.frame(&pair);
    let _ = test.cache.take_stats();
    for _ in 0..5 {
        test.frame(&pair);
    }
    let stats = test.cache.take_stats();
    assert_eq!((stats.finds, stats.scans), (10, 0));
}

/// The texel mask (`TextureCache::texel_mask`) of the texture the last lookup returned.
fn last_mask(test: &Test) -> Option<texture::TexelMask> {
    test.cache.masks.get(test.cache.last_mask as usize).map(|slot| slot.mask)
}

#[test]
fn a_texture_with_transparent_cells_gets_a_texel_mask_made_from_the_pixels_decoded() {
    let mut test = Test::new();
    // 64x64 8888, all alpha 0: every 8x8 cell transparent.
    let wanted = small_texture(RAM_START);
    test.use_texture(&wanted);
    let mask = last_mask(&test).expect("a mask");
    assert_eq!((mask.cell_shift, mask.grid_shift), ([3, 3], [3, 3]));
    assert!(mask.used[..64].iter().all(|&used| used == 0));
    // A texel of alpha 1 at (9, 0): cell 1 shows, after the re-check that decodes it again.
    test.host.write(RAM_START + 9 * 4, 0x0100_0000);
    test.frame_number += 3 * VERIFY_INTERVAL;
    test.use_texture(&wanted);
    let mask = last_mask(&test).expect("a mask");
    assert_eq!(mask.used[1], 1);
    assert_eq!(mask.used.iter().filter(|&&used| used != 0).count(), 1);
    // All opaque: no cell can go, so no mask.
    let opaque = small_texture(RAM_START + 0x10000);
    for texel in 0..64 * 64 {
        test.host.write(RAM_START + 0x10000 + 4 * texel, 0xFF00_0000);
    }
    test.use_texture(&opaque);
    assert_eq!(last_mask(&test), None);
}

#[test]
fn a_texture_used_every_frame_keeps_its_texel_mask_while_others_come_and_go() {
    let mut test = Test::new();
    let streak = small_texture(RAM_START);
    test.use_texture(&streak);
    let kept = last_mask(&test).expect("a mask");
    for k in 1..=20 {
        let other = small_texture(RAM_START + k * 0x4000);
        test.frame(&[other, streak]);
        assert_eq!(last_mask(&test), Some(kept), "after {k} others");
    }
    // The first others lost theirs (16 slots): used once, long ago.
    test.use_texture(&small_texture(RAM_START + 0x4000));
    assert_eq!(last_mask(&test), None);
}
