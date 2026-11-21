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
// The same without fog and specular (most 2D draws): a third of the work.
embed_gxp!(nativeres_textured_flat_f);
// The same with the PSP's alpha test (discard).
embed_gxp!(nativeres_textured_test_f);
// Grey FIX B blending through the alpha, without and with the alpha test.
embed_gxp!(nativeres_textured_fix_f);
embed_gxp!(nativeres_textured_test_fix_f);
// Draws into the PSP's 16-bit framebuffers: frame-read programs that blend
// themselves and cut the colour as the GE does (textured and untextured).
embed_gxp!(nativeres_textured_read_f);
embed_gxp!(nativeres_color_read_f);
// Their fast variants for 5551 targets (one blend each; the keep ones
// leave the target's alpha; "off" reads no frame).
// Their fast variants for 565 targets (green cut to 6 bits, READ_565).
embed_gxp!(nativeres_color_read_normal565_f);
embed_gxp!(nativeres_color_read_normal565dither_f);
embed_gxp!(nativeres_color_read_add565_f);
embed_gxp!(nativeres_color_read_add565dither_f);
embed_gxp!(nativeres_color_read_off565_f);
embed_gxp!(nativeres_color_read_off565dither_f);
embed_gxp!(nativeres_color_read_invisible565_f);
embed_gxp!(nativeres_color_read_invisible565dither_f);
embed_gxp!(nativeres_textured_read_normal565_f);
embed_gxp!(nativeres_textured_read_normal565test_f);
embed_gxp!(nativeres_textured_read_normal565dither_f);
embed_gxp!(nativeres_textured_read_normal565testdither_f);
embed_gxp!(nativeres_textured_read_add565_f);
embed_gxp!(nativeres_textured_read_add565test_f);
embed_gxp!(nativeres_textured_read_add565dither_f);
embed_gxp!(nativeres_textured_read_add565testdither_f);
embed_gxp!(nativeres_textured_read_off565_f);
embed_gxp!(nativeres_textured_read_off565test_f);
embed_gxp!(nativeres_textured_read_off565dither_f);
embed_gxp!(nativeres_textured_read_off565testdither_f);
embed_gxp!(nativeres_textured_read_normal_f);
embed_gxp!(nativeres_textured_read_normalkeep_f);
embed_gxp!(nativeres_textured_read_add_f);
embed_gxp!(nativeres_textured_read_addkeep_f);
embed_gxp!(nativeres_textured_read_off_f);
// With the alpha test.
embed_gxp!(nativeres_textured_read_normaltest_f);
embed_gxp!(nativeres_textured_read_normalkeeptest_f);
embed_gxp!(nativeres_textured_read_addtest_f);
embed_gxp!(nativeres_textured_read_addkeeptest_f);
embed_gxp!(nativeres_textured_read_offtest_f);
embed_gxp!(nativeres_color_read_normal_f);
embed_gxp!(nativeres_color_read_normalkeep_f);
embed_gxp!(nativeres_color_read_add_f);
embed_gxp!(nativeres_color_read_addkeep_f);
embed_gxp!(nativeres_color_read_off_f);
// With the GE's dither (READ_DITHER).
embed_gxp!(nativeres_textured_read_normaldither_f);
embed_gxp!(nativeres_textured_read_normalkeepdither_f);
embed_gxp!(nativeres_textured_read_adddither_f);
embed_gxp!(nativeres_textured_read_addkeepdither_f);
embed_gxp!(nativeres_textured_read_offdither_f);
embed_gxp!(nativeres_textured_read_normaltestdither_f);
embed_gxp!(nativeres_textured_read_normalkeeptestdither_f);
embed_gxp!(nativeres_textured_read_addtestdither_f);
embed_gxp!(nativeres_textured_read_addkeeptestdither_f);
embed_gxp!(nativeres_textured_read_offtestdither_f);
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
// Decoded vertices in model space: the GPU transforms and lights them.
embed_gxp!(nativeres_3d_v);
// Raw PSP vertices, read by the GPU as the game wrote them: with up to
// eight bones and lights, one and up to four bones without lights, one bone
// with lights, up to four bones with lights, the last two for flat light
// sets (programs.rs `RawVariant::Lit1Flat`), and sprites (two corners per
// rectangle).
embed_gxp!(nativeres_raw_v);
embed_gxp!(nativeres_raw_plain_v);
embed_gxp!(nativeres_raw_plain4_v);
embed_gxp!(nativeres_raw_lit1_v);
embed_gxp!(nativeres_raw_lit4_v);
embed_gxp!(nativeres_raw_lit1flat_v);
embed_gxp!(nativeres_raw_lit4flat_v);
// Each of those six with the vertex colour format fixed (programs.rs
// `RawColor`): without colour (the material), 5650, 5551, 4444, 8888.
embed_gxp!(nativeres_raw_plain_none_v);
embed_gxp!(nativeres_raw_plain_5650_v);
embed_gxp!(nativeres_raw_plain_5551_v);
embed_gxp!(nativeres_raw_plain_4444_v);
embed_gxp!(nativeres_raw_plain_8888_v);
embed_gxp!(nativeres_raw_plain4_none_v);
embed_gxp!(nativeres_raw_plain4_5650_v);
embed_gxp!(nativeres_raw_plain4_5551_v);
embed_gxp!(nativeres_raw_plain4_4444_v);
embed_gxp!(nativeres_raw_plain4_8888_v);
embed_gxp!(nativeres_raw_lit1_none_v);
embed_gxp!(nativeres_raw_lit1_5650_v);
embed_gxp!(nativeres_raw_lit1_5551_v);
embed_gxp!(nativeres_raw_lit1_4444_v);
embed_gxp!(nativeres_raw_lit1_8888_v);
embed_gxp!(nativeres_raw_lit4_none_v);
embed_gxp!(nativeres_raw_lit4_5650_v);
embed_gxp!(nativeres_raw_lit4_5551_v);
embed_gxp!(nativeres_raw_lit4_4444_v);
embed_gxp!(nativeres_raw_lit4_8888_v);
embed_gxp!(nativeres_raw_lit1flat_none_v);
embed_gxp!(nativeres_raw_lit1flat_5650_v);
embed_gxp!(nativeres_raw_lit1flat_5551_v);
embed_gxp!(nativeres_raw_lit1flat_4444_v);
embed_gxp!(nativeres_raw_lit1flat_8888_v);
embed_gxp!(nativeres_raw_lit4flat_none_v);
embed_gxp!(nativeres_raw_lit4flat_5650_v);
embed_gxp!(nativeres_raw_lit4flat_5551_v);
embed_gxp!(nativeres_raw_lit4flat_4444_v);
embed_gxp!(nativeres_raw_lit4flat_8888_v);
embed_gxp!(nativeres_raw_sprite_v);
// The basic programs (basic_programs.rs): untextured draws and clears;
// full-screen quads and textured draws without our textured program.
embed_gxp!(nativeres_color_v);
embed_gxp!(nativeres_color_f);
embed_gxp!(nativeres_texture_v);
embed_gxp!(nativeres_texture_tint_f);
