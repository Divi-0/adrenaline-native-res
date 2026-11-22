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
/// game samples that many (512 KiB each; GoW leaves 3 MiB of CDRAM free).
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

/// The off-screen framebuffers (not `shown`) this frame draws into AND
/// samples as a texture, first drawn first; 0 for unused places.
pub fn sampled(draws: &[Draw], textures: &[TextureDescriptor], shown: u32) -> [u32; MAX_OFFSCREEN] {
    let mut found = [0u32; MAX_OFFSCREEN];
    let mut count = 0;
    for (framebuffer, _) in runs(draws) {
        if count == MAX_OFFSCREEN {
            break;
        }
        if framebuffer == 0
            || fold_vram_mirror(framebuffer) == fold_vram_mirror(shown)
            || found[..count].contains(&framebuffer)
            || !textures.iter().any(|texture| reads_framebuffer(texture, framebuffer))
        {
            continue;
        }
        found[count] = framebuffer;
        count += 1;
    }
    found
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
        let textures = vec![
            texture(BLUR_A, format::BGR5650),
            texture(BLUR_B, format::BGR5650),
            texture(SCREEN, format::ABGR8888),
            texture(LEVEL_0, format::ABGR8888),
            texture(LEVEL_1, format::ABGR8888),
            texture(LEVEL_2, format::ABGR8888),
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
