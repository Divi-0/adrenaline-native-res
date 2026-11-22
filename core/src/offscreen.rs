//! Off-screen framebuffers: render to texture into memory the display never
//! shows.
//!
//! Need for Speed Underground Rivals keeps a third 480x272 buffer in VRAM
//! (0x040cc000) as a trail / glow buffer: every frame it first darkens it,
//! then adds it onto the screen as a texture. Draws into such a buffer used
//! to be skipped (only the shown framebuffer's draws were drawn), so the
//! texture was waited for as VRAM data that never came, and every frame
//! fell back to the game's own picture.
//!
//! PPSSPP keeps a framebuffer for every address the GE renders into
//! (FramebufferManagerCommon). Our memory is short (each 960x544 target is
//! 2 MiB), so an off-screen framebuffer gets a target of its own only once a
//! frame samples it as a texture (`sampled`); a buffer that is drawn but
//! never read (NFS's 256x256 one at 0x04110000) stays skipped. The frame's
//! draws are then drawn in GE order, one GXM scene per run of draws into the
//! same framebuffer (`Runs`).

use core::ops::Range;

use crate::frame::{Draw, TextureDescriptor};
use crate::memory::fold_vram_mirror;
use crate::texture::{is_dxt_format, is_palette_format};

/// Off-screen framebuffers that can have targets at once. 6 since
/// 2026-11-22 (was 2): God of War's title samples five (a blur pre-pass and
/// a three-level bloom chain); the targets are made in CDRAM only when a
/// game samples that many, each as big as the frame samples (`sampled_size`;
/// GoW's five: 1.25 MiB), and only while CDRAM keeps its reserve for Sony's
/// emulator (the Vita crate's `gpu_memory::CDRAM_RESERVE`).
pub const MAX_OFFSCREEN: usize = 6;

/// Consecutive draws into the same framebuffer: (framebuffer, draw range),
/// in the frame's order.
pub struct Runs<'a> {
    draws: &'a [Draw],
    at: usize,
}

pub fn runs(draws: &[Draw]) -> Runs<'_> {
    Runs { draws, at: 0 }
}

impl Iterator for Runs<'_> {
    type Item = (u32, Range<usize>);

    fn next(&mut self) -> Option<Self::Item> {
        let first = self.draws.get(self.at)?;
        let start = self.at;
        let framebuffer = first.framebuffer;
        while self.draws.get(self.at).is_some_and(|draw| draw.framebuffer == framebuffer) {
            self.at += 1;
        }
        Some((framebuffer, start..self.at))
    }
}

/// How many runs (`runs`: one GXM scene each when drawn in GE order) go
/// into each of `framebuffers`: `counts[i]` for `framebuffers[i]` (the
/// first `framebuffers.len()` places). A render target drawn in more scenes
/// per frame than it was made for makes the second wait until the GPU has
/// finished the first (God of War's bloom chain: ~26 ms per frame), so the
/// display makes such targets with room for more (2026-11-22).
pub fn runs_into(draws: &[Draw], framebuffers: &[u32], counts: &mut [u32], base: impl Fn(u32) -> u32) {
    counts.iter_mut().for_each(|count| *count = 0);
    for (framebuffer, _) in runs(draws) {
        let framebuffer = base(framebuffer);
        if let Some(index) = framebuffers.iter().position(|&known| known == framebuffer) {
            if let Some(count) = counts.get_mut(index) {
                *count += 1;
            }
        }
    }
}

/// True if `texture` reads framebuffer `framebuffer` as its colours (VRAM
/// mirrors folded). Index and DXT formats don't: a framebuffer read as
/// palette indices or blocks means something else than its colours (as
/// texture_cache's `is_render_target`).
pub fn reads_framebuffer(texture: &TextureDescriptor, framebuffer: u32) -> bool {
    framebuffer != 0
        && !is_palette_format(texture.format)
        && !is_dxt_format(texture.format)
        && fold_vram_mirror(texture.address) == fold_vram_mirror(framebuffer)
}

/// Bytes per pixel of a framebuffer read as `format` (16-bit or 32-bit
/// colours); None for index and DXT formats.
fn colour_bytes(format: u8) -> Option<u32> {
    match format {
        0..=2 => Some(2),
        3 => Some(4),
        _ => None,
    }
}

/// Where framebuffer `framebuffer` lies: inside the first row of a wider
/// one the frame samples, as (that framebuffer, x in pixels); else
/// (`framebuffer`, 0). God of War's glow in the game (2026-11-22): seven
/// 64x64 tiles at x 0, 64, .. 384 of a 1024-pixel 16-bit strip at
/// 0x04161800, drawn one by one as framebuffers 0x04161880, 0x04161900, ..
/// and read as one 512x64 texture; as targets of their own the strip read
/// only its first tile (and eleven framebuffers didn't fit into
/// `MAX_OFFSCREEN`: noise). PPSSPP takes such an address as rendering at an
/// x offset too (FramebufferManagerCommon, `SetCurRTOffset`). Where the
/// frame also reads the tile itself, those reads must have the wider one's
/// row stride (same rows). The outermost framebuffer wins where reads nest.
pub fn placement(textures: &[TextureDescriptor], framebuffer: u32) -> (u32, u32) {
    let folded = fold_vram_mirror(framebuffer);
    let same_rows = |stride: u32| {
        textures
            .iter()
            .filter(|texture| colour_bytes(texture.format).is_some() && fold_vram_mirror(texture.address) == folded)
            .all(|texture| texture.buffer_width == stride)
    };
    textures
        .iter()
        .filter_map(|texture| {
            let bytes = colour_bytes(texture.format)?;
            let base = fold_vram_mirror(texture.address);
            let inside = folded > base && folded < base + texture.width as u32 * bytes && (folded - base) % bytes == 0;
            (inside && same_rows(texture.buffer_width)).then(|| (texture.address, (folded - base) / bytes))
        })
        .max_by_key(|&(_, x)| x)
        .unwrap_or((framebuffer, 0))
}

/// How much of off-screen framebuffer `framebuffer` the frame's textures
/// read, in PSP pixels: the widest and the highest of them (0 x 0 if none),
/// counting reads of its tiles (`placement`) from their x. Its target needs
/// no more: God of War's title samples five off-screen framebuffers of
/// 256x256 down to 32x32, which took 768 KiB of CDRAM each as 480x272
/// targets (2026-11-22, `CDRAM_RESERVE` in the Vita crate).
pub fn sampled_size(textures: &[TextureDescriptor], framebuffer: u32) -> (u32, u32) {
    let folded = fold_vram_mirror(framebuffer);
    textures.iter().filter(|texture| colour_bytes(texture.format).is_some()).fold((0, 0), |(width, height), texture| {
        let (base, x) = placement(textures, texture.address);
        if fold_vram_mirror(base) == folded {
            (width.max(x + texture.width as u32), height.max(texture.height as u32))
        } else {
            (width, height)
        }
    })
}

/// The off-screen framebuffers (not `shown`) this frame draws into AND
/// samples as a texture, first drawn first, tiles of a wider one
/// (`placement`) as that one; 0 for unused places.
pub fn sampled(draws: &[Draw], textures: &[TextureDescriptor], shown: u32) -> [u32; MAX_OFFSCREEN] {
    let mut found = [0u32; MAX_OFFSCREEN];
    let mut count = 0;
    for (framebuffer, _) in runs(draws) {
        if count == MAX_OFFSCREEN {
            break;
        }
        let (base, x) = placement(textures, framebuffer);
        if framebuffer == 0
            || fold_vram_mirror(base) == fold_vram_mirror(shown)
            || found[..count].contains(&base)
            || (x == 0 && !textures.iter().any(|texture| reads_framebuffer(texture, framebuffer)))
        {
            continue;
        }
        found[count] = base;
        count += 1;
    }
    found
}

/// Whether some framebuffer the frame draws into is a tile of another
/// (`placement` with an x): such frames are drawn in the GE's order.
pub fn draws_tiles(draws: &[Draw], textures: &[TextureDescriptor]) -> bool {
    runs(draws).any(|(framebuffer, _)| framebuffer != 0 && placement(textures, framebuffer).1 > 0)
}

/// Whether the frame can be drawn as one scene per target: first each
/// off-screen framebuffer's draws (`drawn` says which have targets), then
/// all of the shown framebuffer's in one scene. True when that gives what
/// the GE's order gives: every such framebuffer is drawn in one run, its
/// draws don't sample the shown framebuffer, and no shown draw before its
/// run samples it. Fewer scenes, and each target drawn once per frame: a
/// second scene on the same render target waited for the GPU (sceGxmEndScene
/// ~50 ms per frame on NFS UR's title).
pub fn offscreen_first(draws: &[Draw], textures: &[TextureDescriptor], shown: u32, drawn: impl Fn(u32) -> bool) -> bool {
    let samples = |draw: &Draw, framebuffer: u32| {
        draw.flags & crate::frame::draw_flags::TEXTURED != 0
            && textures.get(draw.texture as usize).is_some_and(|texture| reads_framebuffer(texture, framebuffer))
    };
    let mut seen = [0u32; MAX_OFFSCREEN];
    let mut seen_count = 0;
    for (framebuffer, range) in runs(draws) {
        if framebuffer == shown || !drawn(framebuffer) {
            continue;
        }
        if seen[..seen_count].contains(&framebuffer) || seen_count == MAX_OFFSCREEN {
            return false; // drawn in two runs
        }
        seen[seen_count] = framebuffer;
        seen_count += 1;
        if draws[range.clone()].iter().any(|draw| samples(draw, shown)) {
            return false;
        }
        if draws[..range.start].iter().any(|draw| draw.framebuffer == shown && samples(draw, framebuffer)) {
            return false;
        }
    }
    true
}

/// True if the frame reads the shown framebuffer `shown` as a texture, and
/// only in draws into other framebuffers. Drawn in GE order (one scene per
/// run, `offscreen_first` false), the screen's scene has ended before such a
/// draw, so it can sample our target of the screen as drawn so far, as the
/// GE samples VRAM. God of War's bloom chain samples the screen mid-frame;
/// decoding the 512x512 framebuffer from VRAM instead took ~9 ms per frame
/// and gave Sony's picture, not ours (2026-11-22). A draw into the screen
/// that reads it can't sample a target it draws into: false then.
pub fn screen_read_only_by_other_runs(draws: &[Draw], textures: &[TextureDescriptor], shown: u32) -> bool {
    let reads_screen = |index: u32| textures.get(index as usize).is_some_and(|texture| reads_framebuffer(texture, shown));
    // Cheap first: most frames have no texture at the screen at all.
    if !textures.iter().any(|texture| reads_framebuffer(texture, shown)) {
        return false;
    }
    let mut read = false;
    for draw in draws {
        if draw.flags & crate::frame::draw_flags::TEXTURED == 0 || !reads_screen(draw.texture) {
            continue;
        }
        if fold_vram_mirror(draw.framebuffer) == fold_vram_mirror(shown) {
            return false;
        }
        read = true;
    }
    read
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::texture::format;

    const SHOWN: u32 = 0x0400_0000;
    const TRAIL: u32 = 0x040C_C000;
    const QUAD: u32 = 0x0411_0000;

    /// `first` then zeros, as `sampled` returns them.
    fn slots(first: &[u32]) -> [u32; MAX_OFFSCREEN] {
        let mut slots = [0; MAX_OFFSCREEN];
        slots[..first.len()].copy_from_slice(first);
        slots
    }

    fn draw(framebuffer: u32) -> Draw {
        Draw { framebuffer, ..Default::default() }
    }

    fn texture(address: u32, format: u8) -> TextureDescriptor {
        TextureDescriptor { address, format, width: 512, height: 512, buffer_width: 512, ..Default::default() }
    }

    /// God of War in the game: seven 64x64 tiles drawn into a 1024-pixel
    /// 16-bit strip at 0x04161800, then the strip read as 512x64.
    #[test]
    fn tiles_of_a_wider_framebuffer_are_placed_at_their_x() {
        let strip = 0x0416_1800;
        let tile = |k: u32| strip + 0x80 * k;
        let read = |address, width, height, format| TextureDescriptor { width, height, buffer_width: 1024, ..texture(address, format) };
        let mut textures = vec![read(strip, 512, 64, format::BGR5650)];
        textures.extend((1..7).map(|k| read(tile(k), 64, 64, format::BGR5650)));
        textures.push(TextureDescriptor { width: 256, height: 256, buffer_width: 512, ..texture(0x0416_2000, format::ABGR8888) });
        assert_eq!(placement(&textures, tile(3)), (strip, 192));
        assert_eq!(placement(&textures, strip), (strip, 0));
        assert_eq!(placement(&textures, 0x0416_1C00), (0x0416_1C00, 0), "past the 512 pixels read");
        assert_eq!(placement(&textures, 0x0416_2400), (0x0416_2400, 0), "at the end of a row read, not inside");
        let other_rows = [textures[0], TextureDescriptor { buffer_width: 512, ..textures[3] }];
        assert_eq!(placement(&other_rows, tile(3)), (tile(3), 0), "read itself with other rows: not a tile");
        let mut draws: Vec<Draw> = (0..7).map(|k| draw(tile(k))).collect();
        draws.push(draw(SHOWN));
        assert_eq!(sampled(&draws, &textures, SHOWN), slots(&[strip]));
        assert_eq!(sampled_size(&textures, strip), (512, 64));
        assert!(draws_tiles(&draws, &textures));
        let mut counts = [0; 2];
        runs_into(&draws, &[SHOWN, strip], &mut counts, |framebuffer| placement(&textures, framebuffer).0);
        assert_eq!(counts, [1, 7]);
        // God of War's title: framebuffers one row apart are not tiles.
        let title = [
            TextureDescriptor { width: 256, height: 256, buffer_width: 512, ..texture(0x0416_2000, format::ABGR8888) },
            TextureDescriptor { width: 128, height: 128, buffer_width: 512, ..texture(0x0416_2400, format::ABGR8888) },
        ];
        assert_eq!(placement(&title, 0x0416_2400), (0x0416_2400, 0));
        assert_eq!(placement(&title, 0x0416_2600), (0x0416_2600, 0));
    }

    #[test]
    fn sampled_size_is_the_largest_texture_at_the_framebuffer() {
        // God of War's bloom level at 0x04162400: read as 128x64 and 128x128.
        let level = 0x0416_2400;
        let sized = |address, width, height| TextureDescriptor { width, height, ..texture(address, format::ABGR8888) };
        let textures = [
            sized(level, 128, 64),
            sized(SHOWN, 512, 512),
            sized(level, 128, 128),
            sized(0x0436_2400, 64, 256), // a VRAM mirror of the same address
            TextureDescriptor { width: 512, height: 512, ..texture(level, format::CLUT8) }, // not a framebuffer read
        ];
        assert_eq!(sampled_size(&textures, level), (128, 256));
        assert_eq!(sampled_size(&textures, TRAIL), (0, 0));
    }

    /// NFS UR's title: clear the screen, fade the trail buffer, a quad into
    /// 0x04110000, then the screen with the trail buffer as a texture.
    fn nfs_title() -> (Vec<Draw>, Vec<TextureDescriptor>) {
        let draws = vec![draw(SHOWN), draw(TRAIL), draw(QUAD), draw(SHOWN), draw(SHOWN), draw(SHOWN)];
        let textures = vec![texture(0x09BF_BEE0, format::DXT5), texture(TRAIL, format::BGR5650)];
        (draws, textures)
    }

    #[test]
    fn runs_follow_the_draw_order() {
        let (draws, _) = nfs_title();
        let runs: Vec<_> = runs(&draws).collect();
        assert_eq!(runs, vec![(SHOWN, 0..1), (TRAIL, 1..2), (QUAD, 2..3), (SHOWN, 3..6)]);
        assert_eq!(super::runs(&[]).count(), 0);
    }

    #[test]
    fn only_sampled_off_screen_framebuffers_get_targets() {
        let (draws, textures) = nfs_title();
        // The trail buffer is read back; the quad's buffer isn't.
        assert_eq!(sampled(&draws, &textures, SHOWN), slots(&[TRAIL]));
    }

    #[test]
    fn the_shown_framebuffer_is_never_off_screen() {
        // MTX samples the other framebuffer (motion blur) but draws only the
        // shown one: nothing off-screen.
        let draws = vec![draw(SHOWN), draw(SHOWN)];
        let textures = vec![texture(0x0408_8000, format::ABGR8888), texture(SHOWN, format::ABGR8888)];
        assert_eq!(sampled(&draws, &textures, SHOWN), slots(&[]));
    }

    #[test]
    fn mirrors_count_and_index_or_dxt_reads_do_not() {
        let draws = vec![draw(TRAIL), draw(SHOWN)];
        // The trail buffer sampled through a VRAM mirror.
        assert_eq!(sampled(&draws, &[texture(TRAIL | 0x0020_0000, format::BGR5650)], SHOWN), slots(&[TRAIL]));
        assert_eq!(sampled(&draws, &[texture(TRAIL, format::CLUT8)], SHOWN), slots(&[]));
        assert_eq!(sampled(&draws, &[texture(TRAIL, format::DXT1)], SHOWN), slots(&[]));
    }

    fn textured(framebuffer: u32, texture: u32) -> Draw {
        Draw { framebuffer, texture, flags: crate::frame::draw_flags::TEXTURED, ..Default::default() }
    }

    #[test]
    fn god_of_war_bloom_chain_gets_a_target_per_framebuffer() {
        // God of War's title (2026-11-22): a blur pre-pass ping-pongs two
        // 565 buffers, the scene goes to the screen, then a bloom chain
        // samples the screen into 0x04162000 (256x256), 0x04162400 (128),
        // 0x04162600 (64) and back up, and the screen samples 0x04162000.
        // Each of the five must get a target, else the chain reads VRAM
        // nobody drew (a band of noise across the title).
        const SCREEN: u32 = 0x0408_C000;
        const BLUR_A: u32 = 0x0416_1800;
        const BLUR_B: u32 = 0x0418_1800;
        const LEVEL_0: u32 = 0x0416_2000;
        const LEVEL_1: u32 = 0x0416_2400;
        const LEVEL_2: u32 = 0x0416_2600;
        // Read as in the title's dump (tests/dumps/gow/title.bin).
        let sized = |address, format, width, height, buffer_width| TextureDescriptor { width, height, buffer_width, ..texture(address, format) };
        let textures = vec![
            sized(BLUR_A, format::BGR5650, 64, 64, 1024),
            sized(BLUR_B, format::BGR5650, 32, 32, 1024),
            sized(SCREEN, format::ABGR8888, 512, 512, 512),
            sized(LEVEL_0, format::ABGR8888, 256, 256, 512),
            sized(LEVEL_1, format::ABGR8888, 128, 128, 512),
            sized(LEVEL_2, format::ABGR8888, 64, 64, 512),
        ];
        let draws = vec![
            draw(BLUR_A),
            textured(BLUR_B, 0),
            textured(BLUR_A, 1),
            draw(SCREEN),
            textured(LEVEL_0, 2),
            textured(LEVEL_1, 3),
            textured(LEVEL_2, 4),
            textured(LEVEL_1, 5),
            textured(LEVEL_0, 4),
            textured(SCREEN, 3),
        ];
        assert_eq!(sampled(&draws, &textures, SCREEN), slots(&[BLUR_A, BLUR_B, LEVEL_0, LEVEL_1, LEVEL_2]));
        // The chain samples the screen mid-frame: drawn in GE order.
        assert!(!offscreen_first(&draws, &textures, SCREEN, |framebuffer| framebuffer != SCREEN));
        // Drawn so, the screen, BLUR_A and two levels get two scenes each:
        // their targets need room for two scenes per frame, or the second
        // waits for the GPU to finish the first (~26 ms per frame).
        let framebuffers = [SCREEN, BLUR_A, BLUR_B, LEVEL_0, LEVEL_1, LEVEL_2, 0x0411_0000];
        let mut counts = [0u32; 7];
        runs_into(&draws, &framebuffers, &mut counts, |framebuffer| framebuffer);
        assert_eq!(counts, [2, 2, 1, 2, 2, 1, 0]);
        // Only from a run into another framebuffer: the screen's scene has
        // ended there, so our target of it is the texture.
        assert!(screen_read_only_by_other_runs(&draws, &textures, SCREEN));
    }

    #[test]
    fn the_screen_read_while_drawn_into_is_no_target_texture() {
        let textures = vec![texture(SHOWN, format::ABGR8888), texture(TRAIL, format::BGR5650)];
        // Read by a draw into the screen itself (feedback within one scene).
        let into_itself = vec![draw(SHOWN), textured(TRAIL, 0), textured(SHOWN, 0)];
        assert!(!screen_read_only_by_other_runs(&into_itself, &textures, SHOWN));
        // Not read at all (an untextured draw doesn't read texture 0).
        let not_read = vec![draw(SHOWN), draw(TRAIL), textured(SHOWN, 1)];
        assert!(!screen_read_only_by_other_runs(&not_read, &textures, SHOWN));
        // Through a VRAM mirror, from another run: read from the target.
        let mirrored = [texture(SHOWN | 0x0020_0000, format::ABGR8888)];
        assert!(screen_read_only_by_other_runs(&[draw(SHOWN), textured(TRAIL, 0)], &mirrored, SHOWN));
        // As palette indices it is something else.
        let indices = [texture(SHOWN, format::CLUT8)];
        assert!(!screen_read_only_by_other_runs(&[draw(SHOWN), textured(TRAIL, 0)], &indices, SHOWN));
    }

    #[test]
    fn nfs_title_draws_its_trail_buffer_first() {
        // Clear, fade the trail buffer, quad, screen with the trail buffer.
        let draws = vec![draw(SHOWN), draw(TRAIL), draw(QUAD), textured(SHOWN, 1), draw(SHOWN), textured(SHOWN, 0)];
        let (_, textures) = nfs_title();
        assert!(offscreen_first(&draws, &textures, SHOWN, |framebuffer| framebuffer == TRAIL));
    }

    #[test]
    fn not_first_when_the_order_matters() {
        let textures = vec![texture(TRAIL, format::BGR5650), texture(SHOWN, format::BGR5650)];
        let has_target = |framebuffer| framebuffer == TRAIL;
        // A shown draw reads the trail buffer before the frame draws it.
        let early_read = vec![textured(SHOWN, 0), draw(TRAIL), textured(SHOWN, 0)];
        assert!(!offscreen_first(&early_read, &textures, SHOWN, has_target));
        // The trail buffer's draws read the screen.
        let reads_screen = vec![draw(SHOWN), textured(TRAIL, 1), textured(SHOWN, 0)];
        assert!(!offscreen_first(&reads_screen, &textures, SHOWN, has_target));
        // Drawn in two runs.
        let twice = vec![draw(TRAIL), draw(SHOWN), draw(TRAIL), textured(SHOWN, 0)];
        assert!(!offscreen_first(&twice, &textures, SHOWN, has_target));
        // Framebuffers without targets don't count.
        assert!(offscreen_first(&twice, &textures, SHOWN, |_| false));
    }

    #[test]
    fn at_most_max_offscreen_each_once() {
        // One framebuffer more than there are places, the first drawn twice.
        let framebuffers: Vec<u32> = (0..=MAX_OFFSCREEN as u32).map(|n| 0x0410_0000 + n * 0x2_0000).collect();
        let mut draws = vec![draw(framebuffers[0]), draw(SHOWN)];
        draws.extend(framebuffers.iter().map(|&framebuffer| draw(framebuffer)));
        draws.push(draw(SHOWN));
        let textures: Vec<_> = framebuffers.iter().map(|&address| texture(address, format::ABGR8888)).collect();
        assert_eq!(sampled(&draws, &textures, SHOWN), slots(&framebuffers[..MAX_OFFSCREEN]));
    }
}
