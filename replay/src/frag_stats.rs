//! Debugging: REPLAY_FRAG_STATS=1 counts per draw what its fragments cost a
//! tile-based deferred GPU, and prints one line per drawn draw and one per
//! scene (a run of draws into one framebuffer) after the frame:
//!
//! - `fragments`: covered pixels (inside the scissor), before any test;
//!   `at_start_depth`: of them, depth-tested against the untouched start
//!   depth 1.0; `depth_fail`, `stencil_fail`: failing those tests;
//! - `alpha_kill`: dropped by the alpha test; `written`: reaching the target;
//! - `shaded`: the draw's written fragments a deferred GPU shades. Such a GPU
//!   keeps per pixel the last opaque fragment that passed (a tag) and shades
//!   it only when it must: when a fragment that needs the pixel below it
//!   lands on a tagged pixel, every tag of that 32 x 32 tile is shaded
//!   first, and at the scene's end all that are left. An opaque fragment on
//!   a tagged pixel replaces the tag: the one below is never shaded.
//!   `shaded_px`: the same with only the one pixel's tag shaded (the bound
//!   the other way);
//! - `tiles`: tiles holding a fragment of the draw, `new_tiles`: those no
//!   earlier draw of its scene touched; per scene those of any of its draws,
//!   out of the target's;
//! - `issue`: the indices issued for the draw (index runs: their triangles'
//!   corners, or one stitched strip where all runs of a raw draw are strips);
//! - `scale`: replay pixels per pixel of the Vita's target (4 for draws
//!   shaded at the PSP's resolution, REPLAY_NATIVE): divide every count
//!   but `tiles` by it.
//!
//! Which draws need the pixel below (`class`, a sum): 1 blended or reading
//! the target, 2 may drop fragments, 4 the program computes depth itself (no
//! fragment is decided before it runs), 8 drawn without a fragment program.
//! From the GE state: 1 with blending, 2 with the alpha test.
//! REPLAY_FRAG_CLASS=<file> with lines "<draw> <class>" gives the classes of
//! the programs really taken instead (for the draws it lists).
//!
//! "frag hides: draw n over k count": fragments of draw k that waited to be
//! shaded and never are, because opaque fragments of draw n took their
//! pixels. A frame cut off before draw n (a prefix of its draws) shades them.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::sync::OnceLock;

use nativeres_core::blend::BLEND_ENABLED;
use nativeres_core::frame::{draw_flags, stitched_strip_count, test_flags, Draw, Frame};

use crate::raster::{HEIGHT, WIDTH};

pub const BLENDED: u8 = 1;
pub const DISCARDS: u8 = 2;
pub const COMPUTES_DEPTH: u8 = 4;
pub const NO_PROGRAM: u8 = 8;

/// The GPU's tile, in pixels of its target.
const TILE: usize = 32;

pub fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("REPLAY_FRAG_STATS").is_some())
}

#[derive(Clone, Copy, Default)]
struct Counts {
    fragments: u32,
    at_start: u32,
    depth_fails: u32,
    stencil_fails: u32,
    alpha_kills: u32,
    written: u32,
    shaded: u32,
    shaded_pixel: u32,
    tiles: u32,
    new_tiles: u32,
    scale: u32,
    class: u8,
}

struct Scene {
    framebuffer: u32,
    first: usize,
    last: usize,
    draws: u32,
    issued: u32,
    tiles: u32,
    all_tiles: u32,
    scale: u32,
}

struct State {
    classes: Option<HashMap<usize, u8>>,
    draws: BTreeMap<usize, Counts>,
    /// (the draw that hides, the draw hidden) -> fragments.
    hides: BTreeMap<(usize, usize), u32>,
    scenes: Vec<Scene>,
    current: usize,
    class: u8,
    /// Replay pixels along one side of a tile (32, or 64 at the PSP's resolution).
    tile: usize,
    /// Per pixel the draw (+ 1) whose fragment waits to be shaded; 0 none.
    tags: Vec<u32>,
    tags_pixel: Vec<u32>,
    /// Per tile (64 columns a row): touched in the scene; the draw (+ 1)
    /// that touched it last.
    scene_tiles: Vec<bool>,
    draw_tiles: Vec<usize>,
}

thread_local! {
    static STATE: RefCell<State> = RefCell::new(State {
        classes: None,
        draws: BTreeMap::new(),
        hides: BTreeMap::new(),
        scenes: Vec::new(),
        current: 0,
        class: 0,
        tile: TILE,
        tags: vec![0; WIDTH * HEIGHT],
        tags_pixel: vec![0; WIDTH * HEIGHT],
        scene_tiles: vec![false; 64 * 64],
        draw_tiles: vec![0; 64 * 64],
    });
}

impl State {
    fn tile_of(&self, x: usize, y: usize) -> usize {
        (y / self.tile) * 64 + x / self.tile
    }

    /// Shades every tag of the tile holding (x, y).
    fn shade_tile(&mut self, x: usize, y: usize) {
        let (left, top) = (x / self.tile * self.tile, y / self.tile * self.tile);
        for row in top..(top + self.tile).min(HEIGHT) {
            for at in row * WIDTH + left..row * WIDTH + (left + self.tile).min(WIDTH) {
                let tag = std::mem::take(&mut self.tags[at]);
                if tag != 0 {
                    self.draws.entry(tag as usize - 1).or_default().shaded += 1;
                }
            }
        }
    }

    /// A fragment that needs the pixel below: what waits there is shaded.
    fn needs_below(&mut self, x: usize, y: usize) {
        let at = y * WIDTH + x;
        if self.tags[at] != 0 {
            self.shade_tile(x, y);
        }
        let tag = std::mem::take(&mut self.tags_pixel[at]);
        if tag != 0 {
            self.draws.entry(tag as usize - 1).or_default().shaded_pixel += 1;
        }
    }

    fn end_scene(&mut self) {
        let Some(scene) = self.scenes.last_mut() else { return };
        scene.tiles = self.scene_tiles.iter().filter(|&&touched| touched).count() as u32;
        self.scene_tiles.fill(false);
        for at in 0..WIDTH * HEIGHT {
            let (tag, tag_pixel) = (std::mem::take(&mut self.tags[at]), std::mem::take(&mut self.tags_pixel[at]));
            if tag != 0 {
                self.draws.entry(tag as usize - 1).or_default().shaded += 1;
            }
            if tag_pixel != 0 {
                self.draws.entry(tag_pixel as usize - 1).or_default().shaded_pixel += 1;
            }
        }
    }
}

/// The draw whose fragments follow. `native`: shaded at the PSP's
/// resolution (four replay pixels per pixel of its target); `no_program`:
/// only its depths are drawn.
pub fn begin_draw(index: usize, frame: &Frame, draw: &Draw, native: bool, no_program: bool) {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let state = &mut *state;
        let classes = state.classes.get_or_insert_with(|| {
            let listed = std::env::var("REPLAY_FRAG_CLASS").ok().and_then(|path| std::fs::read_to_string(path).ok()).unwrap_or_default();
            listed
                .lines()
                .filter_map(|line| {
                    let mut words = line.split_whitespace();
                    Some((words.next()?.parse().ok()?, words.next()?.parse().ok()?))
                })
                .collect()
        });
        let from_state = if draw.blend & BLEND_ENABLED != 0 { BLENDED } else { 0 } | if draw.tests & test_flags::ALPHA_TEST != 0 { DISCARDS } else { 0 };
        let class = if no_program { NO_PROGRAM } else { classes.get(&index).copied().unwrap_or(from_state) };
        let scale = if native { 4 } else { 1 };
        if state.scenes.last().map_or(true, |scene| scene.framebuffer != draw.framebuffer) {
            state.end_scene();
            state.tile = if native { 2 * TILE } else { TILE };
            let all_tiles = (WIDTH.div_ceil(state.tile) * HEIGHT.div_ceil(state.tile)) as u32;
            state.scenes.push(Scene { framebuffer: draw.framebuffer, first: index, last: index, draws: 0, issued: 0, tiles: 0, all_tiles, scale });
        }
        let issued = issued_indices(frame, draw);
        if let Some(scene) = state.scenes.last_mut() {
            scene.last = index;
            scene.draws += 1;
            scene.issued += issued;
        }
        state.current = index;
        state.class = class;
        state.draws.insert(index, Counts { scale, class, ..Counts::default() });
    });
}

fn issued_indices(frame: &Frame, draw: &Draw) -> u32 {
    if draw.flags & draw_flags::INDEX_RUNS != 0 && draw.flags & draw_flags::RAW_VERTICES != 0 {
        let runs = &frame.indices()[draw.first_index as usize..(draw.first_index + draw.index_count) as usize];
        if let Some(count) = stitched_strip_count(runs) {
            return count;
        }
    }
    frame.draw_index_count(draw)
}

/// A covered pixel, before the tests: `at_start` depth-tested against the
/// untouched start depth; which test it fails, if any.
pub fn covered(x: usize, y: usize, at_start: bool, depth_fails: bool, stencil_fails: bool) {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let state = &mut *state;
        let (index, tile) = (state.current, state.tile_of(x, y));
        let new_tile = state.draw_tiles[tile] != index + 1;
        state.draw_tiles[tile] = index + 1;
        let new_in_scene = !std::mem::replace(&mut state.scene_tiles[tile], true);
        let counts = state.draws.entry(index).or_default();
        counts.fragments += 1;
        counts.at_start += at_start as u32;
        counts.depth_fails += depth_fails as u32;
        counts.stencil_fails += stencil_fails as u32;
        counts.tiles += new_tile as u32;
        counts.new_tiles += new_in_scene as u32;
        // A program that computes depth runs for every covered fragment; one
        // that may drop or blends, for those passing the tests.
        let runs = state.class & COMPUTES_DEPTH != 0 || (state.class & (BLENDED | DISCARDS) != 0 && !depth_fails && !stencil_fails);
        if runs {
            state.needs_below(x, y);
        }
    });
}

/// The fragment last `covered` was dropped by the alpha test.
pub fn alpha_killed() {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let index = state.current;
        state.draws.entry(index).or_default().alpha_kills += 1;
    });
}

/// The fragment last `covered` reaches the target.
pub fn written(x: usize, y: usize) {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let state = &mut *state;
        let (index, at) = (state.current, y * WIDTH + x);
        state.draws.entry(index).or_default().written += 1;
        if state.class & NO_PROGRAM != 0 {
            return;
        }
        // It waits to be shaded; an opaque one in place of what waited there.
        if state.tags[at] != 0 {
            *state.hides.entry((index, state.tags[at] as usize - 1)).or_default() += 1;
        }
        state.tags[at] = index as u32 + 1;
        state.tags_pixel[at] = index as u32 + 1;
    });
}

pub fn print(frame: &Frame) {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        state.end_scene();
        for (index, counts) in state.draws.iter() {
            let draw = &frame.draws()[*index];
            let texture = frame.textures().get(draw.texture as usize);
            let address = texture.map_or(0, |texture| texture.address);
            let texels = texture.filter(|_| draw.flags & draw_flags::TEXTURED != 0).map_or(0, |texture| texture.width as u32 * texture.height as u32);
            println!(
                "frag stats: draw {index} fragments {} at_start_depth {} depth_fail {} stencil_fail {} fb {:08x} tex {address:08x} blend {:08x} tests {:x} flags {:x} verts {} idx {} alpha_kill {} written {} shaded {} shaded_px {} tiles {} new_tiles {} issue {} texels {texels} scale {} class {}",
                counts.fragments, counts.at_start, counts.depth_fails, counts.stencil_fails,
                draw.framebuffer, draw.blend, draw.tests, draw.flags, draw.vertex_count, draw.index_count,
                counts.alpha_kills, counts.written, counts.shaded, counts.shaded_pixel, counts.tiles, counts.new_tiles, issued_indices(frame, draw), counts.scale, counts.class
            );
        }
        for ((by, hidden), count) in state.hides.iter() {
            println!("frag hides: draw {by} over {hidden} {count}");
        }
        for (number, scene) in state.scenes.iter().enumerate() {
            println!(
                "frag scene: {number} fb {:08x} draws {}-{} drawn {} issue {} tiles {} of {} scale {}",
                scene.framebuffer, scene.first, scene.last, scene.draws, scene.issued, scene.tiles, scene.all_tiles, scene.scale
            );
        }
    });
}
