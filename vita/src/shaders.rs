//! Our GPU programs, embedded as compiled GXP binaries.
//!
//! The sources are `vita/shaders/*.cg` (Cg, the only language the Vita's
//! GPU takes); the NativeRes Shader Compiler app (shadercompiler/)
//! compiles them on the Vita, and the .gxp files go next to them. A changed
//! .gxp is picked up by the next build (`include_bytes!` tracks the file).
//!
//! Each program is a static named like its file (programs.rs registers them).

/// A compiled program. GXM needs programs 16-byte aligned.
#[repr(C, align(16))]
pub struct Gxp<const BYTES: usize>(pub [u8; BYTES]);

/// Embeds `shaders/<name>.gxp` as the static `<name>`.
macro_rules! embed_gxp {
    ($name:ident) => {
        #[allow(non_upper_case_globals)]
        pub static $name: Gxp<{ include_bytes!(concat!("../shaders/", stringify!($name), ".gxp")).len() }> =
            Gxp(*include_bytes!(concat!("../shaders/", stringify!($name), ".gxp")));
    };
}

// Through-mode (2D) and CPU-transformed draws: position with w, texture
// coordinates, colour and fog per vertex.
embed_gxp!(nativeres_textured_v);
embed_gxp!(nativeres_textured_f);
// The same with the PSP's alpha test (discard).
embed_gxp!(nativeres_textured_test_f);
// Grey FIX B blending through the alpha, without and with the alpha test.
embed_gxp!(nativeres_textured_fix_f);
embed_gxp!(nativeres_textured_test_fix_f);
// Decoded vertices in model space: the GPU transforms and lights them.
embed_gxp!(nativeres_3d_v);
// Raw PSP vertices, read by the GPU as the game wrote them: with up to
// eight bones and lights, one bone without lights, one bone with lights, up
// to four bones with lights, and sprites (two corners per rectangle).
embed_gxp!(nativeres_raw_v);
embed_gxp!(nativeres_raw_plain_v);
embed_gxp!(nativeres_raw_lit1_v);
embed_gxp!(nativeres_raw_lit4_v);
embed_gxp!(nativeres_raw_sprite_v);
// The basic programs (basic_programs.rs): untextured draws and clears;
// full-screen quads and textured draws without our textured program.
embed_gxp!(nativeres_color_v);
embed_gxp!(nativeres_color_f);
embed_gxp!(nativeres_texture_v);
embed_gxp!(nativeres_texture_tint_f);
