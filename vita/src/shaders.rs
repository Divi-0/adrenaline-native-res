//! Our GPU programs, embedded as compiled GXP binaries.
//!
//! The sources are `vita/shaders/*.cg` (Cg, the only language the Vita's
//! GPU takes); the NativeRes Shader Compiler app (shadercompiler/)
//! compiles them on the Vita, and the .gxp files go next to them. A changed
//! .gxp is picked up by the next build (`include_bytes!` tracks the file).
//!
//! Each program is a static named like its file (programs.rs registers them).
//! The programs the PSP's whole-unit depth changes come as twins (`Twin`):
//! the normal one and the `_pspdepth_` one, which writes the PSP depth
//! (DESIGN "fifty-seventh session"); which of the two a game gets goes by
//! its title ID (core game_quirks.rs). draw.rs keeps one program set per
//! kind (programs are registered once and never released) and switches
//! between them when the game changes (`Draws::choose_programs`).

use core::sync::atomic::{AtomicBool, Ordering};

/// A compiled program. GXM needs programs 16-byte aligned.
#[repr(C, align(16))]
pub struct Gxp<const BYTES: usize>(pub [u8; BYTES]);

/// Something basic_programs::gxp can hand to GXM.
pub trait Program {
    /// The GXP's first byte.
    fn start(&self) -> *const u8;
}

impl<const BYTES: usize> Program for Gxp<BYTES> {
    fn start(&self) -> *const u8 {
        self.0.as_ptr()
    }
}

/// A program and its PSP-depth twin (scripts/shader-variants.sh `twins`).
#[repr(C)]
pub struct Twin<const BASE: usize, const DEPTH: usize> {
    pub base: Gxp<BASE>,
    pub depth: Gxp<DEPTH>,
}

impl<const BASE: usize, const DEPTH: usize> Program for Twin<BASE, DEPTH> {
    fn start(&self) -> *const u8 {
        if psp_depth() {
            self.depth.start()
        } else {
            self.base.start()
        }
    }
}

/// Whether the program set in use is the PSP-depth one (`Twin` hands out
/// that twin). Changed only by the display thread, together with the set
/// (draw.rs `Draws::choose_programs`), so a set never mixes the two kinds.
static PSP_DEPTH: AtomicBool = AtomicBool::new(false);

/// Whether the programs in use write the PSP's whole-unit depth.
pub fn psp_depth() -> bool {
    PSP_DEPTH.load(Ordering::Relaxed)
}

/// draw.rs `Draws::choose_programs`, display thread only, with the set
/// switched: from now on `Twin`s hand out that kind.
pub fn set_psp_depth(on: bool) {
    PSP_DEPTH.store(on, Ordering::Relaxed);
}

/// Whether the running game wants the PSP-depth programs (core
/// `game_quirks::psp_depth`), and its title ID: the one our PSP plugin read
/// from the disc (capture::GAME_TITLE), else Adrenaline's (empty for games
/// started by title ID). The PSP's own menu and a game whose ID isn't known
/// (yet): no.
pub fn game_wants_psp_depth() -> (bool, [u8; crate::adrenaline::TITLE_ID_BYTES]) {
    if crate::adrenaline::xmb() {
        return (false, [0; crate::adrenaline::TITLE_ID_BYTES]);
    }
    let words = [crate::capture::GAME_TITLE[0].load(Ordering::Relaxed), crate::capture::GAME_TITLE[1].load(Ordering::Relaxed)];
    let title = match nativeres_core::event_ring::unpack_title_id(words) {
        Some(disc) => {
            let mut title = [0u8; crate::adrenaline::TITLE_ID_BYTES];
            title[..disc.len()].copy_from_slice(&disc);
            title
        }
        None => crate::adrenaline::title_id(),
    };
    (nativeres_core::game_quirks::psp_depth(&title), title)
}

/// A title ID as text.
pub fn title_text(title: &[u8]) -> &str {
    let length = title.iter().position(|&byte| byte == 0).unwrap_or(title.len());
    core::str::from_utf8(&title[..length]).unwrap_or("?")
}

/// Embeds `shaders/<name>.gxp` as the static `<name>`.
macro_rules! embed_gxp {
    ($name:ident) => {
        #[allow(non_upper_case_globals)]
        pub static $name: Gxp<{ include_bytes!(concat!("../shaders/", stringify!($name), ".gxp")).len() }> =
            Gxp(*include_bytes!(concat!("../shaders/", stringify!($name), ".gxp")));
    };
}

/// Embeds `shaders/<name>.gxp` and its PSP-depth twin `<depth>.gxp` as the
/// `Twin` static `<name>`.
macro_rules! embed_twin {
    ($name:ident, $depth:ident) => {
        #[allow(non_upper_case_globals)]
        pub static $name: Twin<
            { include_bytes!(concat!("../shaders/", stringify!($name), ".gxp")).len() },
            { include_bytes!(concat!("../shaders/", stringify!($depth), ".gxp")).len() },
        > = Twin {
            base: Gxp(*include_bytes!(concat!("../shaders/", stringify!($name), ".gxp"))),
            depth: Gxp(*include_bytes!(concat!("../shaders/", stringify!($depth), ".gxp"))),
        };
    };
}

// Through-mode (2D) and CPU-transformed draws: position with w, texture
// coordinates, colour and fog per vertex.
embed_twin!(nativeres_textured_v, nativeres_textured_pspdepth_v);
embed_twin!(nativeres_textured_f, nativeres_textured_pspdepth_f);
// The same without fog and specular (most 2D draws): a third of the work.
embed_twin!(nativeres_textured_flat_f, nativeres_textured_flat_pspdepth_f);
// The same with the PSP's alpha test (discard).
embed_twin!(nativeres_textured_test_f, nativeres_textured_test_pspdepth_f);
// Grey FIX B blending through the alpha, without and with the alpha test.
embed_twin!(nativeres_textured_fix_f, nativeres_textured_fix_pspdepth_f);
embed_twin!(nativeres_textured_test_fix_f, nativeres_textured_test_fix_pspdepth_f);
// Both with the GE's bilinear filter, for draws reading our targets.
embed_gxp!(nativeres_textured_ge_f);
embed_gxp!(nativeres_textured_fix_ge_f);
// Draws into the PSP's 16-bit framebuffers: frame-read programs that blend
// themselves and cut the colour as the GE does (textured and untextured).
embed_twin!(nativeres_textured_read_f, nativeres_textured_read_pspdepth_f);
embed_gxp!(nativeres_color_read_f);
// Their fast variants for 5551 targets (one blend each; the keep ones
// leave the target's alpha; "off" reads no frame).
// Their fast variants for 565 targets (green cut to 6 bits, READ_565).
// (textured, without fog: NO_FOG)
embed_twin!(nativeres_textured_read_normal565nofog_f, nativeres_textured_read_normal565nofog_pspdepth_f);
embed_twin!(nativeres_textured_read_normal565nofogtest_f, nativeres_textured_read_normal565nofogtest_pspdepth_f);
embed_twin!(nativeres_textured_read_normal565nofogdither_f, nativeres_textured_read_normal565nofogdither_pspdepth_f);
embed_twin!(nativeres_textured_read_normal565nofogtestdither_f, nativeres_textured_read_normal565nofogtestdither_pspdepth_f);
embed_twin!(nativeres_textured_read_add565nofog_f, nativeres_textured_read_add565nofog_pspdepth_f);
embed_twin!(nativeres_textured_read_add565nofogtest_f, nativeres_textured_read_add565nofogtest_pspdepth_f);
embed_twin!(nativeres_textured_read_add565nofogdither_f, nativeres_textured_read_add565nofogdither_pspdepth_f);
embed_twin!(nativeres_textured_read_add565nofogtestdither_f, nativeres_textured_read_add565nofogtestdither_pspdepth_f);
embed_twin!(nativeres_textured_read_off565nofog_f, nativeres_textured_read_off565nofog_pspdepth_f);
embed_twin!(nativeres_textured_read_off565nofogtest_f, nativeres_textured_read_off565nofogtest_pspdepth_f);
embed_twin!(nativeres_textured_read_off565nofogdither_f, nativeres_textured_read_off565nofogdither_pspdepth_f);
embed_twin!(nativeres_textured_read_off565nofogtestdither_f, nativeres_textured_read_off565nofogtestdither_pspdepth_f);
embed_gxp!(nativeres_color_read_normal565_f);
embed_gxp!(nativeres_color_read_normal565dither_f);
embed_gxp!(nativeres_color_read_add565_f);
embed_gxp!(nativeres_color_read_add565dither_f);
embed_gxp!(nativeres_color_read_off565_f);
embed_gxp!(nativeres_color_read_off565dither_f);
embed_gxp!(nativeres_color_read_invisible565_f);
embed_gxp!(nativeres_color_read_invisible565dither_f);
// One, one, reverse subtract, untextured 565 undithered (ReadWay::Subtract).
embed_gxp!(nativeres_color_read_subtract565_f);
embed_twin!(nativeres_textured_read_normal565_f, nativeres_textured_read_normal565_pspdepth_f);
embed_twin!(nativeres_textured_read_normal565test_f, nativeres_textured_read_normal565test_pspdepth_f);
embed_twin!(nativeres_textured_read_normal565dither_f, nativeres_textured_read_normal565dither_pspdepth_f);
embed_twin!(nativeres_textured_read_normal565testdither_f, nativeres_textured_read_normal565testdither_pspdepth_f);
embed_twin!(nativeres_textured_read_add565_f, nativeres_textured_read_add565_pspdepth_f);
embed_twin!(nativeres_textured_read_add565test_f, nativeres_textured_read_add565test_pspdepth_f);
embed_twin!(nativeres_textured_read_add565dither_f, nativeres_textured_read_add565dither_pspdepth_f);
embed_twin!(nativeres_textured_read_add565testdither_f, nativeres_textured_read_add565testdither_pspdepth_f);
embed_twin!(nativeres_textured_read_off565_f, nativeres_textured_read_off565_pspdepth_f);
embed_twin!(nativeres_textured_read_off565test_f, nativeres_textured_read_off565test_pspdepth_f);
embed_twin!(nativeres_textured_read_off565dither_f, nativeres_textured_read_off565dither_pspdepth_f);
embed_twin!(nativeres_textured_read_off565testdither_f, nativeres_textured_read_off565testdither_pspdepth_f);
embed_twin!(nativeres_textured_read_normal_f, nativeres_textured_read_normal_pspdepth_f);
embed_twin!(nativeres_textured_read_normalkeep_f, nativeres_textured_read_normalkeep_pspdepth_f);
embed_twin!(nativeres_textured_read_add_f, nativeres_textured_read_add_pspdepth_f);
embed_twin!(nativeres_textured_read_addkeep_f, nativeres_textured_read_addkeep_pspdepth_f);
embed_twin!(nativeres_textured_read_off_f, nativeres_textured_read_off_pspdepth_f);
// With the alpha test.
embed_twin!(nativeres_textured_read_normaltest_f, nativeres_textured_read_normaltest_pspdepth_f);
embed_twin!(nativeres_textured_read_normalkeeptest_f, nativeres_textured_read_normalkeeptest_pspdepth_f);
embed_twin!(nativeres_textured_read_addtest_f, nativeres_textured_read_addtest_pspdepth_f);
embed_twin!(nativeres_textured_read_addkeeptest_f, nativeres_textured_read_addkeeptest_pspdepth_f);
embed_twin!(nativeres_textured_read_offtest_f, nativeres_textured_read_offtest_pspdepth_f);
embed_gxp!(nativeres_color_read_normal_f);
embed_gxp!(nativeres_color_read_normalkeep_f);
embed_gxp!(nativeres_color_read_add_f);
embed_gxp!(nativeres_color_read_addkeep_f);
embed_gxp!(nativeres_color_read_off_f);
// With the GE's dither (READ_DITHER).
embed_twin!(nativeres_textured_read_normaldither_f, nativeres_textured_read_normaldither_pspdepth_f);
embed_twin!(nativeres_textured_read_normalkeepdither_f, nativeres_textured_read_normalkeepdither_pspdepth_f);
embed_twin!(nativeres_textured_read_adddither_f, nativeres_textured_read_adddither_pspdepth_f);
embed_twin!(nativeres_textured_read_addkeepdither_f, nativeres_textured_read_addkeepdither_pspdepth_f);
embed_twin!(nativeres_textured_read_offdither_f, nativeres_textured_read_offdither_pspdepth_f);
embed_twin!(nativeres_textured_read_normaltestdither_f, nativeres_textured_read_normaltestdither_pspdepth_f);
embed_twin!(nativeres_textured_read_normalkeeptestdither_f, nativeres_textured_read_normalkeeptestdither_pspdepth_f);
embed_twin!(nativeres_textured_read_addtestdither_f, nativeres_textured_read_addtestdither_pspdepth_f);
embed_twin!(nativeres_textured_read_addkeeptestdither_f, nativeres_textured_read_addkeeptestdither_pspdepth_f);
embed_twin!(nativeres_textured_read_offtestdither_f, nativeres_textured_read_offtestdither_pspdepth_f);
embed_gxp!(nativeres_color_read_normaldither_f);
embed_gxp!(nativeres_color_read_normalkeepdither_f);
embed_gxp!(nativeres_color_read_adddither_f);
embed_gxp!(nativeres_color_read_addkeepdither_f);
embed_gxp!(nativeres_color_read_offdither_f);
// Untextured normal blends of alpha 0 (core frame.rs `ReadWay::Invisible`):
// the target dithered and cut again, alpha written or kept, undithered and
// dithered.
embed_gxp!(nativeres_color_read_invisible_f);
embed_gxp!(nativeres_color_read_invisiblekeep_f);
embed_gxp!(nativeres_color_read_invisibledither_f);
embed_gxp!(nativeres_color_read_invisiblekeepdither_f);
// 8-bit targets (core frame.rs `wide_variant`): the GE's normal / additive
// blend without a cut, the target's alpha kept.
embed_twin!(nativeres_textured_read_normalwidekeep_f, nativeres_textured_read_normalwidekeep_pspdepth_f);
embed_twin!(nativeres_textured_read_normalwidekeeptest_f, nativeres_textured_read_normalwidekeeptest_pspdepth_f);
embed_twin!(nativeres_textured_read_normalwidekeepnofog_f, nativeres_textured_read_normalwidekeepnofog_pspdepth_f);
embed_twin!(nativeres_textured_read_normalwidekeepnofogtest_f, nativeres_textured_read_normalwidekeepnofogtest_pspdepth_f);
embed_gxp!(nativeres_color_read_normalwidekeep_f);
embed_twin!(nativeres_textured_read_addwidekeep_f, nativeres_textured_read_addwidekeep_pspdepth_f);
embed_twin!(nativeres_textured_read_addwidekeeptest_f, nativeres_textured_read_addwidekeeptest_pspdepth_f);
embed_twin!(nativeres_textured_read_addwidekeepnofog_f, nativeres_textured_read_addwidekeepnofog_pspdepth_f);
embed_twin!(nativeres_textured_read_addwidekeepnofogtest_f, nativeres_textured_read_addwidekeepnofogtest_pspdepth_f);
embed_gxp!(nativeres_color_read_addwidekeep_f);
// Decoded vertices in model space: the GPU transforms and lights them.
embed_twin!(nativeres_3d_v, nativeres_3d_pspdepth_v);
// Raw PSP vertices, read by the GPU as the game wrote them: with up to
// eight bones and lights, one and up to four bones without lights, one bone
// with lights, up to four bones with lights, the last two for flat light
// sets (programs.rs `RawVariant::Lit1Flat`), and sprites (two corners per
// rectangle).
embed_twin!(nativeres_raw_v, nativeres_raw_pspdepth_v);
embed_twin!(nativeres_raw_plain_v, nativeres_raw_plain_pspdepth_v);
embed_twin!(nativeres_raw_plain4_v, nativeres_raw_plain4_pspdepth_v);
embed_twin!(nativeres_raw_lit1_v, nativeres_raw_lit1_pspdepth_v);
embed_twin!(nativeres_raw_lit4_v, nativeres_raw_lit4_pspdepth_v);
embed_twin!(nativeres_raw_lit1flat_v, nativeres_raw_lit1flat_pspdepth_v);
embed_twin!(nativeres_raw_lit4flat_v, nativeres_raw_lit4flat_pspdepth_v);
// The lit ones for light sets with point lights (generic colour only).
embed_twin!(nativeres_raw_lit1point_v, nativeres_raw_lit1point_pspdepth_v);
embed_twin!(nativeres_raw_lit4point_v, nativeres_raw_lit4point_pspdepth_v);
// Each of those six with the vertex colour format fixed (programs.rs
// `RawColor`): without colour (the material), 5650, 5551, 4444, 8888.
embed_twin!(nativeres_raw_plain_none_v, nativeres_raw_plain_none_pspdepth_v);
embed_twin!(nativeres_raw_plain_5650_v, nativeres_raw_plain_5650_pspdepth_v);
embed_twin!(nativeres_raw_plain_5551_v, nativeres_raw_plain_5551_pspdepth_v);
embed_twin!(nativeres_raw_plain_4444_v, nativeres_raw_plain_4444_pspdepth_v);
embed_twin!(nativeres_raw_plain_8888_v, nativeres_raw_plain_8888_pspdepth_v);
embed_twin!(nativeres_raw_plain4_none_v, nativeres_raw_plain4_none_pspdepth_v);
embed_twin!(nativeres_raw_plain4_5650_v, nativeres_raw_plain4_5650_pspdepth_v);
embed_twin!(nativeres_raw_plain4_5551_v, nativeres_raw_plain4_5551_pspdepth_v);
embed_twin!(nativeres_raw_plain4_4444_v, nativeres_raw_plain4_4444_pspdepth_v);
embed_twin!(nativeres_raw_plain4_8888_v, nativeres_raw_plain4_8888_pspdepth_v);
embed_twin!(nativeres_raw_lit1_none_v, nativeres_raw_lit1_none_pspdepth_v);
embed_twin!(nativeres_raw_lit1_5650_v, nativeres_raw_lit1_5650_pspdepth_v);
embed_twin!(nativeres_raw_lit1_5551_v, nativeres_raw_lit1_5551_pspdepth_v);
embed_twin!(nativeres_raw_lit1_4444_v, nativeres_raw_lit1_4444_pspdepth_v);
embed_twin!(nativeres_raw_lit1_8888_v, nativeres_raw_lit1_8888_pspdepth_v);
embed_twin!(nativeres_raw_lit4_none_v, nativeres_raw_lit4_none_pspdepth_v);
embed_twin!(nativeres_raw_lit4_5650_v, nativeres_raw_lit4_5650_pspdepth_v);
embed_twin!(nativeres_raw_lit4_5551_v, nativeres_raw_lit4_5551_pspdepth_v);
embed_twin!(nativeres_raw_lit4_4444_v, nativeres_raw_lit4_4444_pspdepth_v);
embed_twin!(nativeres_raw_lit4_8888_v, nativeres_raw_lit4_8888_pspdepth_v);
embed_twin!(nativeres_raw_lit1flat_none_v, nativeres_raw_lit1flat_none_pspdepth_v);
embed_twin!(nativeres_raw_lit1flat_5650_v, nativeres_raw_lit1flat_5650_pspdepth_v);
embed_twin!(nativeres_raw_lit1flat_5551_v, nativeres_raw_lit1flat_5551_pspdepth_v);
embed_twin!(nativeres_raw_lit1flat_4444_v, nativeres_raw_lit1flat_4444_pspdepth_v);
embed_twin!(nativeres_raw_lit1flat_8888_v, nativeres_raw_lit1flat_8888_pspdepth_v);
embed_twin!(nativeres_raw_lit4flat_none_v, nativeres_raw_lit4flat_none_pspdepth_v);
embed_twin!(nativeres_raw_lit4flat_5650_v, nativeres_raw_lit4flat_5650_pspdepth_v);
embed_twin!(nativeres_raw_lit4flat_5551_v, nativeres_raw_lit4flat_5551_pspdepth_v);
embed_twin!(nativeres_raw_lit4flat_4444_v, nativeres_raw_lit4flat_4444_pspdepth_v);
embed_twin!(nativeres_raw_lit4flat_8888_v, nativeres_raw_lit4flat_8888_pspdepth_v);
embed_twin!(nativeres_raw_sprite_v, nativeres_raw_sprite_pspdepth_v);
// The basic programs (basic_programs.rs): untextured draws and clears;
// full-screen quads and textured draws without our textured program.
embed_gxp!(nativeres_color_v);
embed_gxp!(nativeres_color_f);
embed_gxp!(nativeres_texture_v);
embed_gxp!(nativeres_texture_tint_f);
