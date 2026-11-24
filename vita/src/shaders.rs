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
//!
//! The PSP-depth twins are embedded zlib-compressed as one pack
//! (scripts/pack-depth-twins.py) and unpacked only when a game needs them
//! (`load_depth_pack`): ~64 KB instead of ~250 KB in every other game
//! (DESIGN "sixty-first session").

use core::ffi::c_void;
use core::sync::atomic::{AtomicBool, Ordering};

use crate::{render_log, sys, SingleThreaded};

include!("../shaders/psp_depth_pack.rs");

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

/// A program and its PSP-depth twin (scripts/shader-variants.sh `twins`):
/// the twin by name from the unpacked depth pack (`load_depth_pack`, which made sure
/// every twin is there before `set_psp_depth` turns them on).
#[repr(C)]
pub struct Twin<const BASE: usize> {
    pub base: Gxp<BASE>,
    /// The twin's name (its stem, e.g. b"nativeres_raw_pspdepth_v").
    pub depth: &'static [u8],
}

impl<const BASE: usize> Program for Twin<BASE> {
    fn start(&self) -> *const u8 {
        if psp_depth() {
            depth_program(self.depth).unwrap_or(self.base.start())
        } else {
            self.base.start()
        }
    }
}

/// The PSP-depth twins' pack, zlib-compressed.
static DEPTH_PACK_PACKED: &[u8] = include_bytes!("../shaders/pspdepth.pak.z");
const DEPTH_PACK_NAME_BYTES: usize = 64;
const DEPTH_PACK_ENTRY_BYTES: usize = DEPTH_PACK_NAME_BYTES + 8;

/// Memory plan (tools/memcheck.py): the unpacked pack (and the inflater's
/// state), for PSP-depth games only.
#[no_mangle]
#[used]
pub static nativeres_plan_optin_psp_depth_pack: u32 = sys::pages(DEPTH_PACK_BLOCK_BYTES);

/// The unpacked pack, then the inflater's state (in the same block, freed
/// with it: no room for it on the stack).
const DEPTH_PACK_STATE_OFFSET: usize = DEPTH_PACK_BYTES.next_multiple_of(16);
const DEPTH_PACK_BLOCK_BYTES: usize = DEPTH_PACK_STATE_OFFSET + core::mem::size_of::<miniz_oxide::inflate::core::DecompressorOxide>();

/// The depth pack once unpacked and checked (never freed: GXM keeps
/// pointers into it); None before (or if it couldn't be read).
static DEPTH_PACK: SingleThreaded<Option<&'static [u8]>> = SingleThreaded::new(None);

fn word(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

fn fnv1a(bytes: &[u8]) -> u32 {
    bytes.iter().fold(0x811C_9DC5u32, |hash, &byte| (hash ^ byte as u32).wrapping_mul(0x0100_0193))
}

/// The twin `name` in the loaded pack.
fn depth_program(name: &[u8]) -> Option<*const u8> {
    // Safe: display thread only (programs are made there).
    let pack = unsafe { *DEPTH_PACK.get() }?;
    (0..DEPTH_PACK_COUNT).find_map(|i| {
        let entry = 16 + i * DEPTH_PACK_ENTRY_BYTES;
        let stored = &pack[entry..entry + DEPTH_PACK_NAME_BYTES];
        let length = stored.iter().position(|&byte| byte == 0).unwrap_or(DEPTH_PACK_NAME_BYTES);
        (&stored[..length] == name).then(|| pack[word(pack, entry + DEPTH_PACK_NAME_BYTES) as usize..].as_ptr())
    })
}

/// Unpacks the pack (display thread, once: the first PSP-depth game) and
/// checks it (size, hash). False if that fails (no memory): the float
/// programs stay then.
pub fn load_depth_pack() -> bool {
    // Safe: display thread only.
    let slot = unsafe { DEPTH_PACK.get() };
    if slot.is_some() {
        return true;
    }
    let bytes = DEPTH_PACK_BYTES;
    // Safe: plain system calls; the block is ours, freed if the file is wrong.
    unsafe {
        let block = sys::alloc_block(
            b"NativeResDepthPack\0".as_ptr(),
            sys::SCE_KERNEL_MEMBLOCK_TYPE_USER_RW,
            sys::pages(DEPTH_PACK_BLOCK_BYTES),
            core::ptr::null_mut(),
        );
        if block < 0 {
            render_log::line(format_args!("depth pack: no memory ({bytes} bytes)\n"));
            return false;
        }
        let mut base: *mut c_void = core::ptr::null_mut();
        sys::sceKernelGetMemBlockBase(block, &mut base);
        use miniz_oxide::inflate::core::{decompress, inflate_flags, DecompressorOxide};
        let state = (base as *mut u8).add(DEPTH_PACK_STATE_OFFSET) as *mut DecompressorOxide;
        state.write(DecompressorOxide::new());
        let output = core::slice::from_raw_parts_mut(base as *mut u8, bytes);
        let flags = inflate_flags::TINFL_FLAG_PARSE_ZLIB_HEADER | inflate_flags::TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF;
        let (_, _, read) = decompress(&mut *state, DEPTH_PACK_PACKED, output, 0, flags);
        let pack = core::slice::from_raw_parts(base as *const u8, bytes);
        let fits = read == bytes
            && &pack[..4] == b"NRDP"
            && word(pack, 4) == 1
            && word(pack, 8) as usize == DEPTH_PACK_COUNT
            && word(pack, 12) == DEPTH_PACK_HASH
            && fnv1a(&pack[16..]) == DEPTH_PACK_HASH;
        if !fits {
            render_log::line(format_args!("depth pack: unpacked {read} of {bytes} bytes or wrong hash\n"));
            sys::free_block(block);
            return false;
        }
        *slot = Some(pack);
        render_log::line(format_args!("depth pack: {DEPTH_PACK_COUNT} programs, {bytes} bytes\n"));
        true
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

/// Embeds `shaders/<name>.gxp` as the `Twin` static `<name>`, its PSP-depth
/// twin `<depth>` named (from the depth pack).
macro_rules! embed_twin {
    ($name:ident, $depth:ident) => {
        #[allow(non_upper_case_globals)]
        pub static $name: Twin<{ include_bytes!(concat!("../shaders/", stringify!($name), ".gxp")).len() }> = Twin {
            base: Gxp(*include_bytes!(concat!("../shaders/", stringify!($name), ".gxp"))),
            depth: stringify!($depth).as_bytes(),
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
