//! Our GPU programs, embedded as compiled GXP binaries.
//!
//! The sources are `renderer/vita/shaders/*.cg` (Cg, the only language the
//! Vita's GPU takes); `python tools/vita.py shaders` compiles them on the Vita
//! (the NativePSP Shader Compiler app, renderer/shadercompiler) and puts
//! the .gxp files next to them. A changed
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
embed_gxp!(nativepsp_textured_v);
embed_gxp!(nativepsp_textured_f);
// The same with the PSP's alpha test (discard).
embed_gxp!(nativepsp_textured_test_f);
// Grey FIX B blending through the alpha, without and with the alpha test.
embed_gxp!(nativepsp_textured_fix_f);
embed_gxp!(nativepsp_textured_test_fix_f);
// Decoded vertices in model space: the GPU transforms and lights them.
embed_gxp!(nativepsp_3d_v);
// Raw PSP vertices, read by the GPU as the game wrote them: with skinning
// and lights, without either, and sprites (two corners per rectangle).
embed_gxp!(nativepsp_raw_v);
embed_gxp!(nativepsp_raw_plain_v);
embed_gxp!(nativepsp_raw_sprite_v);
// The basic programs (basic_programs.rs): untextured draws and clears;
// full-screen quads and textured draws without our textured program.
embed_gxp!(nativepsp_color_v);
embed_gxp!(nativepsp_color_f);
embed_gxp!(nativepsp_texture_v);
embed_gxp!(nativepsp_texture_tint_f);
