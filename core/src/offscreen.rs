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

/// Off-screen framebuffers that can have targets at once.
pub const MAX_OFFSCREEN: usize = 2;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::texture::format;

    const SHOWN: u32 = 0x0400_0000;
    const TRAIL: u32 = 0x040C_C000;
    const QUAD: u32 = 0x0411_0000;

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
        assert_eq!(sampled(&draws, &textures, SHOWN), [TRAIL, 0]);
    }

    #[test]
    fn the_shown_framebuffer_is_never_off_screen() {
        // MTX samples the other framebuffer (motion blur) but draws only the
        // shown one: nothing off-screen.
        let draws = vec![draw(SHOWN), draw(SHOWN)];
        let textures = vec![texture(0x0408_8000, format::ABGR8888), texture(SHOWN, format::ABGR8888)];
        assert_eq!(sampled(&draws, &textures, SHOWN), [0, 0]);
    }

    #[test]
    fn mirrors_count_and_index_or_dxt_reads_do_not() {
        let draws = vec![draw(TRAIL), draw(SHOWN)];
        // The trail buffer sampled through a VRAM mirror.
        assert_eq!(sampled(&draws, &[texture(TRAIL | 0x0020_0000, format::BGR5650)], SHOWN), [TRAIL, 0]);
        assert_eq!(sampled(&draws, &[texture(TRAIL, format::CLUT8)], SHOWN), [0, 0]);
        assert_eq!(sampled(&draws, &[texture(TRAIL, format::DXT1)], SHOWN), [0, 0]);
    }

    #[test]
    fn at_most_max_offscreen_each_once() {
        let a = 0x0410_0000;
        let b = 0x0412_0000;
        let c = 0x0414_0000;
        let draws = vec![draw(a), draw(SHOWN), draw(a), draw(b), draw(c), draw(SHOWN)];
        let textures: Vec<_> = [a, b, c].iter().map(|&address| texture(address, format::ABGR8888)).collect();
        assert_eq!(sampled(&draws, &textures, SHOWN), [a, b]);
    }
}
