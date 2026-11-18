//! The description of one finished PSP frame, handed from the capture thread
//! to the Vita's display thread.
//!
//! A [`Frame`] is a plain `#[repr(C)]` struct with fixed-size arrays: it is
//! handed between threads in fixed memory blocks, and the dumps for the PC
//! replay hold it as bytes. All
//! positions are in PSP screen pixels (480x272); the display side scales
//! them to the Vita screen.

/// Upper limits per frame. Final Fantasy needs at most ~9000 vertices, MTX
/// Mototrax's Los Angeles race ~54,000 (10,400 draws, mostly 4-6 vertex
/// strips; as plain triangles that would be ~93,000, which is why 3D draws
/// are indexed).
pub const MAX_VERTICES: usize = 65536;
/// Three per triangle.
pub const MAX_INDICES: usize = 3 * MAX_VERTICES;
/// A draw's indices are 16 bits, counted from its first vertex (the Vita
/// draws with 16-bit index buffers), so one draw has at most this many.
pub const MAX_DRAW_VERTICES: u32 = 65536;
pub const MAX_DRAWS: usize = 8192;
pub const MAX_TEXTURES: usize = 512;
/// MTX Mototrax loads a palette for nearly every one of its ~5000 index
/// texture draws; ~60 different ones per frame.
pub const MAX_PALETTE_WORDS: usize = 65536;
/// Slots of `Frame::texture_lookup` (twice MAX_TEXTURES: short probes).
pub const TEXTURE_LOOKUP_SLOTS: usize = 2 * MAX_TEXTURES;
/// Palettes one frame can hold (`Frame::palettes`): one per palette address
/// and length its draws use. MTX's Los Angeles race start: 470.
pub const MAX_FRAME_PALETTES: usize = 1024;
/// Slots of `Frame::palette_slots` (twice MAX_FRAME_PALETTES: short probes).
pub const PALETTE_SLOTS: usize = 2 * MAX_FRAME_PALETTES;
/// Entries of `Frame::transforms` (48 KiB; every frame slot has them): only
/// draws decoded on the CPU and left to the GPU's 3D program use them since
/// raw draws have their own (`raw_transforms`, 2026-11-16; MTX's race start
/// used 42 when raw draws were in here too). When the table is full, the CPU
/// transforms.
pub const MAX_TRANSFORMS: usize = 256;
/// Entries of `Frame::raw_transforms` (64 KiB): one per object a frame's raw
/// draws place (world matrix). Monster Hunter Freedom's village ~300+.
/// When the table is full, runs go the old way.
pub const MAX_RAW_TRANSFORMS: usize = 1024;
/// Entries of `Frame::raw_cameras` (11 KiB): view, projection, viewport and
/// the other registers raw draws share; a frame mostly has one or two.
pub const MAX_RAW_CAMERAS: usize = 64;
/// Entries of `Frame::light_sets` (44 KiB; every frame slot has them): the
/// lighting of GPU-lit draws (`GpuLights`), consecutive equal ones stored
/// once. When the table is full, the CPU lights the vertices.
pub const MAX_LIGHT_SETS: usize = 128;
/// Entries of `Frame::raw_colors` (5 KiB; every frame slot has them): the
/// colour rules of raw draws, consecutive equal ones stored once. When the
/// table is full, runs go the old way. MTX's race start uses 3.
pub const MAX_RAW_COLORS: usize = 64;
/// Floats of `Frame::bone_floats` (96 KiB; every frame slot has them): the
/// bones of raw draws skinned in the shader, 12 per bone. MTX's race start
/// changes bones before nearly every rider part (4,584 floats used). When
/// full, runs go the old way: Monster Hunter Freedom's village intro needs
/// 17,280 (2026-11-16: with 8,192, 1,077 PRIMs / 5,919 vertices per frame
/// were decoded on the CPU, `cpu=tables_full`).
pub const MAX_BONE_FLOATS: usize = 24576;

/// One corner of a triangle.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FrameVertex {
    /// PSP screen pixels.
    pub x: f32,
    pub y: f32,
    /// Depth, the PSP's 16-bit depth value / 65535 (0..1, clamped). Only
    /// matters for draws with a depth test or depth writes (`Draw::tests`).
    pub z: f32,
    /// The clip-space w of a 3D vertex (1 for 2D): the GPU interpolates
    /// texture coordinates, colours and fog with it, for perspective.
    pub w: f32,
    /// Texture coordinates, 0..1 across the texture.
    pub u: f32,
    pub v: f32,
    /// RGBA, 0xAABBGGRR. Textured draws: the colour the texture is
    /// multiplied with (the texture function already applied).
    pub color: u32,
    /// PSP fog factor (draws with `draw_flags::FOG`): 1 = no fog, 0 = all
    /// fog colour; not clamped (the GPU interpolates, then clamps).
    pub fog: f32,
}

/// `Draw::flags` bits.
pub mod draw_flags {
    /// Uses `Draw::texture`; otherwise only vertex colours.
    pub const TEXTURED: u32 = 1 << 0;
    /// The draw blends (see `Draw::blend`).
    pub const BLEND: u32 = 1 << 1;
    /// Bilinear texture filtering (otherwise nearest).
    pub const LINEAR_FILTER: u32 = 1 << 2;
    /// Texture coordinates clamp instead of repeating.
    pub const CLAMP_U: u32 = 1 << 3;
    pub const CLAMP_V: u32 = 1 << 4;
    /// Colour doubling: the result of texture x colour is doubled (RGB only).
    pub const DOUBLE_COLOR: u32 = 1 << 5;
    /// Made of rectangles (sprites, tiles): every 6 vertices are one
    /// axis-aligned quad cut from the texture.
    pub const RECTANGLES: u32 = 1 << 6;
    /// 3D draw with fog: mixed toward `Draw::fog_color` by the vertices' `fog`.
    pub const FOG: u32 = 1 << 7;
    /// The vertices are in the vertex cache (`Frame::vertex_cache`), not in
    /// `Frame::vertices`: `first_vertex` and `vertex_count` are 0 and the
    /// indices count from the cache's first vertex. Always with a transform.
    pub const CACHED_VERTICES: u32 = 1 << 8;
    /// The draw's `first_index` / `index_count` are index runs, not
    /// indices: three words per run (primitive type, vertex count, first
    /// vertex), turned into triangle corners by whoever draws them
    /// (`Frame::draw_indices`; the display thread's DrawIndices writes them
    /// straight into the GPU's index buffer). The recorder thread's fast
    /// path writes runs (3 words per PRIM instead of 3 per triangle: the
    /// index writing moved to the display thread, which has time to spare).
    pub const INDEX_RUNS: u32 = 1 << 9;
    /// Back-face culling (CULL_FACE_ENABLE): only triangles of one winding
    /// on the screen are drawn. With `CULL_CCW` (the CULL register's 1) those
    /// whose corners a, b, c turn clockwise on the screen (y down: (b - a) x
    /// (c - a) > 0), without it the others: PPSSPP's software rasterizer
    /// (GPU/Software/TransformUnit.cpp SendTriangle, BinManager::AddTriangle
    /// keeps that cross product > 0). Strip triangles are stored with every
    /// second one's first two corners swapped (`strip_triangle`), so one
    /// winding holds for every triangle. Never on rectangles.
    pub const CULL: u32 = 1 << 10;
    pub const CULL_CCW: u32 = 1 << 11;
    /// The vertices are raw PSP vertices in the raw arena
    /// (`Frame::raw_vertices`, `raw_arena`), in the layout of
    /// `Draw::vertex_format`, read by the GPU as they are
    /// (nativeres_raw_v.cg). `first_vertex` is the byte offset of the
    /// arena window the indices count vertices from (`vertex_count` 0).
    /// The indices are index runs (`INDEX_RUNS`: PRIMs whose vertices follow
    /// each other), or, for indexed PRIMs, a plain triangle list
    /// (`recorder::record_raw_converted`: the PSP's indices converted, each
    /// counted from the window; `Geometry::finish_raw_listed_draw`). Always
    /// with a transform (its `raw_uv`) and an entry of `Frame::raw_colors`
    /// (`Draw::raw_colors`), which finish the decoding. Only vertices in
    /// front of the near plane are drawn: the GPU clips triangles there (the
    /// CPU doesn't look at the positions).
    pub const RAW_VERTICES: u32 = 1 << 12;
    /// With `RAW_VERTICES`: the vertices are in PSP RAM, not the raw arena
    /// (zero copy, `raw_arena::ZERO_COPY`): `first_vertex` is the PSP
    /// address the indices (runs or a list) count vertices from (any
    /// alignment; the display maps PSP RAM for the GPU). The CPU never saw
    /// them, so `Frame::screen_corners` gives none.
    pub const RAW_PSP_RAM: u32 = 1 << 13;
    /// With `RAW_VERTICES`: rectangles (PRIM type 6, 3D) as raw sprites.
    /// Each sprite's two PSP vertices (A, the first corner, and B, the
    /// opposite one) are in the arena four times as A B pairs, one pair per
    /// corner, so vertex k of the window is pair k (A at 2k strides, B one
    /// stride after) and corner k % 4 of its sprite: 0 at A, 1 at (B x, A y),
    /// 2 at (A x, B y), 3 at B (nativeres_raw_sprite_v.cg; the CPU path's
    /// rules: colour and depth from B, flat, `raw_vertex::RawCorners`). Its
    /// index runs are `[RECTANGLES, sprites, first corner]`.
    pub const RAW_SPRITES: u32 = 1 << 14;
    /// The texture has mip levels (`texture_flags::LEVELS_SHIFT`, menu
    /// option "NativeRes Distant Textures" not "Sharp"): sampled with them,
    /// the GPU choosing the level, plus the bias in `MIP_BIAS_SHIFT`.
    pub const MIPS: u32 = 1 << 15;
    /// With `MIPS`: the GPU blends between two levels (the GE's "mip
    /// linear" minification filters).
    pub const MIP_LINEAR: u32 = 1 << 24;
    /// With `MIPS`: a level bias in 1/8 levels, signed (bits 16..24, i8):
    /// the GE's TEXLEVEL offset in "auto" mode, plus one level where the
    /// PSP's choice is wanted (the picture is twice the PSP's size, so the
    /// GPU would pick one level sharper). The display adds it to the GXM
    /// texture's lod bias field (31 + this).
    pub const MIP_BIAS_SHIFT: u32 = 16;
    pub const MIP_BIAS_MASK: u32 = 0xFF << MIP_BIAS_SHIFT;
    /// The PSP target's pixel format when it has 16 bits (bits 25..27,
    /// `TARGET_565` / `TARGET_5551` / `TARGET_4444`; 0 = 8888). The PSP
    /// cuts every pixel it writes there to 4..6 bits per channel, after
    /// blending: faint additive passes (MHF's fire glow) add nothing where
    /// our 8-bit targets add them up. The display then blends in the
    /// fragment program and cuts the colour the same way
    /// (`quantize_like_the_ge`).
    pub const TARGET_FORMAT_SHIFT: u32 = 25;
    pub const TARGET_FORMAT_MASK: u32 = 3 << TARGET_FORMAT_SHIFT;
    pub const TARGET_565: u32 = 1;
    pub const TARGET_5551: u32 = 2;
    pub const TARGET_4444: u32 = 3;
    /// With a 16-bit target: dithering on, `Frame::dither_matrix` added to
    /// the colour before it is cut.
    pub const DITHER: u32 = 1 << 27;
}

/// What the GE writes into a 16-bit target for an 8-bit colour channel
/// (0..255, already blended): `dither` added (-8..7, 0 without), clamped,
/// cut to `bits` bits (the GE drops the low bits), and widened back to 8
/// bits the way a 16-bit framebuffer is read (the top bits repeated below,
/// `color.rs` `expand`), since our targets keep 8 bits per channel.
/// One bit (5551's alpha) widens to 0 or 255.
pub fn quantize_like_the_ge(value: i32, dither: i32, bits: u32) -> u32 {
    let cut = ((value + dither).clamp(0, 255) as u32) >> (8 - bits);
    match bits {
        1 => cut * 255,
        _ => (cut << (8 - bits)) | (cut >> (2 * bits - 8)),
    }
}

/// Bits per channel (R, G, B, A) of a 16-bit target
/// (`draw_flags::TARGET_FORMAT_SHIFT`); None for 8888.
pub fn target_channel_bits(flags: u32) -> Option<[u32; 4]> {
    match (flags & draw_flags::TARGET_FORMAT_MASK) >> draw_flags::TARGET_FORMAT_SHIFT {
        draw_flags::TARGET_565 => Some([5, 6, 5, 0]),
        draw_flags::TARGET_5551 => Some([5, 5, 5, 1]),
        draw_flags::TARGET_4444 => Some([4, 4, 4, 4]),
        _ => None,
    }
}

/// Entry (`x`, `y`) of a dither matrix as the GE keeps it
/// (`Frame::dither_matrix`: row r in bits 16r..16r+16, four signed 4-bit
/// entries, column 0 lowest), for PSP pixel (`x`, `y`).
pub fn dither_value(matrix: u64, x: usize, y: usize) -> i32 {
    let nibble = (matrix >> (16 * (y % 4) + 4 * (x % 4))) & 0xF;
    ((nibble as i32) << 28) >> 28
}

/// The fast frame-read programs' cut of an 8-bit channel to 5 bits,
/// widened back, as bit operations (nativeres_textured_read_f.cg
/// READ_FAST: `v & 0xF8 | v >> 5`, four channels per instruction).
/// The same as `quantize_like_the_ge(v, 0, 5)` (tests).
pub fn cut_to_5_bits(value: u8) -> u8 {
    (value & 0xF8) | (value >> 5)
}

/// The blend of a fast frame-read program (`ReadVariant`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ReadWay {
    /// Source alpha, one minus source alpha, add.
    Normal,
    /// Source alpha, one, add.
    Add,
    /// No blending.
    Off,
    /// Normal alpha blending by GXM, of the source cut to 5 bits first (no
    /// frame read): exact where the source alpha is 255 or 0 (most pixels
    /// of alpha-tested foliage), elsewhere GXM's rounding and no cut after
    /// the blend (one 5-bit step at most). Frame-read draws cost MHF's
    /// intro ~12-16 ms of GPU time (`read_variant`'s `exact_normal`).
    Precut,
}

/// The frame-read program a draw into a 16-bit target gets.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ReadVariant {
    /// The generic program: any format, blend and dithering, in floating
    /// point (uniforms).
    Generic,
    /// A fast one (5551): its blend; `keep_alpha` = the target's alpha
    /// stays (else the alpha is written, as blended); `dither` = the GE's
    /// dither value is added after the blend (`draw_flags::DITHER`).
    Fast { way: ReadWay, keep_alpha: bool, dither: bool },
}

/// The frame-read program for a draw into a 16-bit target with `flags`
/// and packed `blend` (`Draw::blend`), `keep_target_alpha` while a game
/// reads its targets' alpha (the PSP's stencil). As the Vita's 8-bit
/// path: without `keep_target_alpha` every draw writes the alpha (nobody
/// reads it, so its cut doesn't matter). The fast programs only cut to 5
/// bits: a draw that writes an alpha that is read (5551's one bit) gets
/// the generic one; so do BLEND_FIX draws (FIX B through the alpha: the
/// generic program's premultiply). `exact_normal`: normal alpha blends
/// read the frame and blend as the GE does, else `ReadWay::Precut`
/// (dithered ones always read: the dither comes after the blend, which
/// GXM would do). Dithered draws took the generic program until
/// 2026-11-18: MHF's full-screen fade quad cost ~9.6 ms of GPU time there.
pub fn read_variant(flags: u32, blend: u32, keep_target_alpha: bool, exact_normal: bool) -> ReadVariant {
    use crate::blend::{BLEND_ALPHA_ONLY, BLEND_ENABLED, BLEND_FIX_B_ALPHA, BLEND_WRITES_ALPHA};
    let format = (flags & draw_flags::TARGET_FORMAT_MASK) >> draw_flags::TARGET_FORMAT_SHIFT;
    let alpha_only = blend & BLEND_WRITES_ALPHA != 0 && blend & BLEND_ALPHA_ONLY != 0;
    let fix = blend & BLEND_ENABLED != 0 && blend & BLEND_FIX_B_ALPHA != 0;
    if format != draw_flags::TARGET_5551 || alpha_only || fix {
        return ReadVariant::Generic;
    }
    let dither = flags & draw_flags::DITHER != 0;
    let writes_alpha = !keep_target_alpha || blend & BLEND_WRITES_ALPHA != 0;
    if writes_alpha && keep_target_alpha {
        return ReadVariant::Generic;
    }
    let way = if blend & BLEND_ENABLED == 0 {
        ReadWay::Off
    } else {
        match (blend & 0xF, (blend >> 4) & 0xF, (blend >> 8) & 7) {
            (4, 5, 0) if exact_normal || dither => ReadWay::Normal,
            (4, 5, 0) => ReadWay::Precut,
            (4, 1, 0) => ReadWay::Add,
            _ => return ReadVariant::Generic,
        }
    };
    ReadVariant::Fast { way, keep_alpha: !writes_alpha, dither }
}

/// The corners of triangle `t` of a strip, counted from its first vertex:
/// t, t + 1, t + 2, with the first two swapped for odd t, so every triangle
/// of the strip turns the same way on the screen (the GE alternates the
/// winding it expects along a strip: `draw_flags::CULL`). The last corner
/// stays the newest vertex.
#[inline]
pub fn strip_triangle(t: u32) -> [u32; 3] {
    if t % 2 == 0 {
        [t, t + 1, t + 2]
    } else {
        [t + 1, t, t + 2]
    }
}

/// `Draw::tests` bits and fields.
pub mod test_flags {
    pub const ALPHA_TEST: u32 = 1 << 0;
    pub const DEPTH_TEST: u32 = 1 << 1;
    pub const STENCIL_TEST: u32 = 1 << 2;
    pub const LOGIC_OP: u32 = 1 << 3;
    pub const COLOR_TEST: u32 = 1 << 4;
    /// Depth writes off. Like on the PSP, a draw writes depth only when
    /// DEPTH_TEST is set (clears that write depth: DEPTH_TEST with "always").
    pub const DEPTH_WRITE_DISABLED: u32 = 1 << 5;
    /// Depth compare function (ZTEST register) in bits 8..11.
    pub const DEPTH_FUNCTION_SHIFT: u32 = 8;
    /// Alpha compare function (ALPHA_TEST bits 0..3) in bits 12..15,
    /// reference value in bits 16..24, mask in bits 24..32 (both sides
    /// are ANDed with it before the compare).
    pub const ALPHA_FUNCTION_SHIFT: u32 = 12;
    pub const ALPHA_REFERENCE_SHIFT: u32 = 16;
    pub const ALPHA_MASK_SHIFT: u32 = 24;
}

/// `TextureDescriptor::flags` bits.
pub mod texture_flags {
    /// The draw ignores the texture's alpha (texture function "RGB"): decode
    /// it fully opaque so only the vertex alpha counts.
    pub const IGNORE_ALPHA: u16 = 1 << 0;
    /// Mip levels beyond level 0 (bits 8..11, 0 = none): the texture comes
    /// with `levels + 1` levels, described in `Frame::texture_levels` at
    /// the texture's index. Part of the texture's identity, so a texture
    /// used with and without levels is cached twice.
    pub const LEVELS_SHIFT: u16 = 8;
    pub const LEVELS_MASK: u16 = 0x7 << LEVELS_SHIFT;
}

/// The GE's mip levels of a texture (`Frame::texture_levels`), level 0
/// included: where each level's texels are (physical PSP address), its row
/// stride in texels and its size (log2 width in bits 0..4, log2 height in
/// bits 4..8). Only `levels` of `texture_flags::LEVELS_SHIFT` + 1 entries
/// are used; each level is half the one before (the GPU's mip chains are),
/// else fewer levels are kept (`texture::describe_levels`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TextureLevels {
    pub address: [u32; 8],
    pub buffer_width: [u16; 8],
    pub log2_size: [u8; 8],
}

/// The level count of a texture (1: level 0 only).
pub fn texture_level_count(texture: &TextureDescriptor) -> u32 {
    (((texture.flags & texture_flags::LEVELS_MASK) >> texture_flags::LEVELS_SHIFT) + 1) as u32
}

/// A run of triangles (`index_count / 3`) drawn with the same state.
/// Triangle corners are `Frame::indices[first_index..][..index_count]`,
/// each counted from `first_vertex` (so 0 is `vertices[first_vertex]`).
/// Rectangles and 2D draws use their vertices in order (indices 0, 1, 2,
/// ...); 3D strips and fans share vertices between triangles.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Draw {
    pub first_vertex: u32,
    pub vertex_count: u32,
    pub first_index: u32,
    pub index_count: u32,
    /// Index into `Frame::textures` when `TEXTURED` is set.
    pub texture: u32,
    pub flags: u32,
    /// The PSP framebuffer (VRAM address) this was drawn into.
    pub framebuffer: u32,
    /// GPU blending (`blend::Blend::pack`), 0 = blending off.
    pub blend: u32,
    /// Per-pixel tests that were enabled, see `test_flags`.
    pub tests: u32,
    /// Scissor rectangle in PSP pixels, inclusive.
    pub scissor_left: u16,
    pub scissor_top: u16,
    pub scissor_right: u16,
    pub scissor_bottom: u16,
    /// Fog colour (RGB, 0x00BBGGRR) when `FOG` is set.
    pub fog_color: u32,
    /// 0: the vertices are in screen space (PSP pixels, depth, w), as
    /// described at `FrameVertex`. n > 0: the vertices are in model space
    /// and the GPU turns them into screen space with
    /// `Frame::transforms[n - 1]` (`GpuTransform`; their u, v, colour are as
    /// usual, w and fog unused).
    pub transform: u32,
    /// `draw_flags::RAW_VERTICES`: the PSP's VERTEX_TYPE of the vertices
    /// (their layout and field formats); 0 otherwise.
    pub vertex_format: u32,
    /// `draw_flags::RAW_VERTICES`: how its vertices get their colour,
    /// `Frame::raw_colors[n - 1]`; 0 otherwise.
    pub raw_colors: u32,
    /// `draw_flags::RAW_VERTICES` skinned in the shader: where its bones
    /// start in `Frame::bone_floats`, + 1 (0: not skinned). As many bones as
    /// the format has weights, each 3 rows of 4 floats (row r: the bone's
    /// r-th coordinate from the raw x, y, z, 1). The skinned position is the
    /// weighted sum (weights times `VertexFormat::weight_scale`); the
    /// transform's rows (`Transform::skinned_row_transform`) take it on.
    /// Normals skin with the same 3x3 parts. On the draw, not the
    /// transform: bones change for nearly every rider part, and different
    /// transforms filled the table.
    pub bones: u32,
}

/// How the GPU turns a model-space vertex (x, y, z) into the screen-space
/// vertex the CPU used to work out (`transform::RowTransform::apply`, the
/// same formula; `to_screen` is it on the CPU):
///
/// ```text
/// clip  = rows . (x, y, z, 1)            (one dot product per row)
/// x     = viewport[0] + viewport[2] * clip.x / clip.w
/// y     = viewport[1] + viewport[3] * clip.y / clip.w
/// depth = clamp(depth[0] + depth[1] * clip.z / clip.w, 0, 1)
/// fog   = fog_row . (x, y, z, 1)          (draws with draw_flags::FOG)
/// ```
///
/// Only vertices in front of the near plane (clip.z >= -clip.w, clip.w > 0)
/// are drawn this way; the CPU clips the others as before.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GpuTransform {
    /// Projection x view x world (and the bone, the position scale), row
    /// by row.
    pub rows: [[f32; 4]; 4],
    /// Fog factor row; (0, 0, 0, 1) without fog (fog 1 = none).
    pub fog_row: [f32; 4],
    /// Viewport origin x, y and scale x, y.
    pub viewport: [f32; 4],
    /// Depth centre and scale (already divided by 65535), then two zeros.
    pub depth: [f32; 4],
    /// Lit draws lit by the GPU: the 3x3 matrix taking the vertices'
    /// normals to world space (world x bone; negated for REVERSENORMAL), row
    /// by row, w 0. Zeros for other draws.
    pub normal_rows: [[f32; 4]; 3],
    /// Lit draws lit by the GPU: the number of their `Frame::light_sets`
    /// entry + 1 (0: the vertex colour as it is), a float for the shader,
    /// then three zeros. Their vertices hold the raw vertex colour and the
    /// normal packed into `w` (`lighting::pack_normal`).
    pub light_set: [f32; 4],
    /// Raw draws (`draw_flags::RAW_VERTICES`; zeros for others): texture
    /// coordinates are raw x (u, v scale) + (u, v offset): TEXTURE_SCALE and
    /// OFFSET with the format's scale (1/128, 1/32768, 1) folded in, as
    /// `RowTransform` has them. Positions need nothing: `rows` already
    /// hold the position scale. (The colour rules are a table of their own,
    /// `RawColors`: in the transform they made neighbouring entries differ
    /// and filled the table at MTX's race start.)
    pub raw_uv: [f32; 4],
}

/// What a raw draw's vertex program multiplies its vertices with, apart from
/// the object (`RawTransform`): the GE's view and projection matrices and the
/// registers the transform reads, as they are (`raw_blocks`). Shared by every
/// object a camera sees; a new entry only when one of them changes.
///
/// A raw vertex goes as the PSP's GE (and PPSSPP's vertex shaders,
/// VertexShaderGenerator.cpp) takes it: the raw position times the format's
/// scale, skinned (`Draw::bones`), world, view, projection, one after the
/// other (`raw_blocks::clip_vertex`):
///
/// ```text
/// model = skin(position x scale)              (or position x scale)
/// world = world rows . (model, 1)
/// view  = view rows . (world, 1)
/// clip  = projection rows . (view, 1)
/// x     = viewport[0] + viewport[2] * clip.x / clip.w   (as GpuTransform)
/// depth = depth[0] + depth[1] * clip.z / clip.w
/// fog   = (view.z * fog[0] + fog[1]) * fog[2]
/// u, v  = raw u, v x format scale x texture[0..2] + texture[2..4]
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RawCamera {
    /// The view matrix (4x3), row by row.
    pub view: [[f32; 4]; 3],
    /// The projection matrix (4x4), row by row.
    pub projection: [[f32; 4]; 4],
    /// Viewport origin x, y (the screen offset taken off) and scale x, y.
    pub viewport: [f32; 4],
    /// Depth centre and scale (already divided by 65535), then two zeros.
    pub depth: [f32; 4],
    /// With fog: 1, FOG_END, FOG_SCALE, 0, so fog = (view z + end) x scale;
    /// without: 0, 1, 1, 0 (fog 1 = none).
    pub fog: [f32; 4],
    /// TEXTURE_SCALE u, v and TEXTURE_OFFSET u, v.
    pub texture: [f32; 4],
}

/// One object of a frame's raw draws (`Draw::transform` of a draw with
/// `draw_flags::RAW_VERTICES`): its world matrix, the camera that sees it
/// and, for draws lit by the GPU, its lights. Written by `raw_blocks` only
/// when the GE's world matrix, registers or lights changed.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RawTransform {
    /// The world matrix (4x3), row by row.
    pub world: [[f32; 4]; 3],
    /// Its `Frame::raw_cameras` entry (an index, not + 1).
    pub camera: u16,
    /// Its `Frame::light_sets` entry + 1 (0: not lit by the GPU).
    pub light_set: u16,
    /// What it was written from (`raw_blocks`): the transform inputs' change
    /// numbers and the lighting's key, so the next draw with the same inputs
    /// finds it without comparing anything.
    pub inputs_change: u32,
    pub lighting_change: u32,
    pub lighting_key: u32,
}

/// How a raw draw's vertices get their colour (`raw_vertex`,
/// nativeres_raw_v.cg): what the CPU path's `base_color` and `frame_color`
/// do. Colours as whole numbers 0..255 per channel (RGBA), so the shader's
/// integer steps are exact.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RawColors {
    /// The colour of vertices without one, or the material of lit ones.
    pub material: [f32; 4],
    /// Lit without lights: AMBIENT_COLOR and AMBIENT_ALPHA; MATERIAL_EMISSIVE
    /// (RGB, alpha 0).
    pub ambient: [f32; 4],
    pub emissive: [f32; 4],
    /// [lighting (`raw_vertex::lighting`): 0 none, 1 lit without lights with
    /// the vertex colour as material (MATERIAL_UPDATE), 2 lit with
    /// `material`, 3 lit with lights by the GPU (the transform's light set);
    /// 1 = alpha made 255 (unblended draws); texture function "replace": 0
    /// no, 1 RGB white (vertex alpha kept), 2 all white; 1 = alpha doubled
    /// (at most 255; blend factor "doubled source alpha", after the opaque
    /// rule, before replace)].
    pub rules: [f32; 4],
    /// Blend factor FIX A (`blend::Blend::source_scale`): RGB 0..255 and
    /// w 1, last of all the colour becomes `blend::scale_color` of it;
    /// zeros (w 0) = none. Raw like the draws they cover: MTX's puddles
    /// draw water with FIX A over raw ground with an equal-depth GEQUAL, and
    /// the two paths' depths differed by a few ulps (32x32 tile squares).
    pub scale: [f32; 4],
}

/// The lighting of GPU-lit draws (`lighting::Lights::gpu`), world space:
/// what nativeres_3d_v.cg gets as uniforms (same names, same order) and
/// `GpuLights::color` computes on the CPU (the PC replay). Unused lights are
/// all zeros.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GpuLights {
    /// Per light: towards it (normalized), w 0.
    pub to_light: [[f32; 4]; 4],
    /// Per light: the specular half vector (normalized; 0 if none), w 0.
    pub half: [[f32; 4]; 4],
    /// Per light: its diffuse colour (0..1); a: 1 = powered diffuse.
    pub diffuse: [[f32; 4]; 4],
    /// Per light: its specular colour (0..1); a: 1 = it has specular.
    pub specular: [[f32; 4]; 4],
    /// The global ambient light plus every light's ambient colour (RGB),
    /// AMBIENT_ALPHA (a).
    pub ambient: [f32; 4],
    /// MATERIAL_EMISSIVE (RGB); a: the specular coefficient.
    pub emissive: [f32; 4],
    /// MATERIAL_AMBIENT and MATERIAL_ALPHA.
    pub material_ambient: [f32; 4],
    /// MATERIAL_DIFFUSE (RGB); a: 1 = the result's alpha is 1 (unblended
    /// draws, as `recorder::Prepared::frame_color` makes them).
    pub material_diffuse: [f32; 4],
    /// MATERIAL_SPECULAR (RGB); a: 1 = separate specular (LIGHTMODE 1):
    /// the shader adds the specular part after texturing.
    pub material_specular: [f32; 4],
    /// 1 where the ambient, diffuse, specular colour (x, y, z) is the
    /// vertex colour (MATERIAL_UPDATE, vertices with a colour); w 1 =
    /// REVERSE_NORMAL: raw draws turn their normals around (the 3D program
    /// has its normal rows negated instead and doesn't read w).
    pub from_vertex: [f32; 4],
}

/// `Geometry::add_*` check whether an entry repeats the last one by its
/// bits rather than as floats (`PartialEq`: on the A9 each float compare is
/// a VFP compare plus a `vmrs` that stalls the pipeline, up to 44 for a
/// transform, 88 for a light set). Exact for that: equal bits are equal
/// entries; only +0.0 / -0.0 count as different, at worst one more entry.
/// Live A/B 2026-11-08: no measurable difference (recorder busy ~31.5 ms
/// either way; the float compare stops at the first difference).
const GPU_TRANSFORM_WORDS: usize = core::mem::size_of::<GpuTransform>() / 4;
const GPU_LIGHTS_WORDS: usize = core::mem::size_of::<GpuLights>() / 4;
const _: () = assert!(core::mem::size_of::<GpuTransform>() == 48 * 4);
const RAW_COLORS_WORDS: usize = core::mem::size_of::<RawColors>() / 4;
const _: () = assert!(core::mem::size_of::<RawColors>() == 20 * 4);
const _: () = assert!(core::mem::size_of::<GpuLights>() == 88 * 4);

impl GpuTransform {
    /// Its floats as bits.
    pub fn words(&self) -> &[u32; GPU_TRANSFORM_WORDS] {
        // Safe: `repr(C)`, nothing but f32s (no padding, size checked above),
        // and any f32's bits are a valid u32.
        unsafe { &*(self as *const GpuTransform as *const [u32; GPU_TRANSFORM_WORDS]) }
    }
}

impl RawColors {
    /// Its floats as bits.
    pub fn words(&self) -> &[u32; RAW_COLORS_WORDS] {
        // Safe: as `GpuTransform::words`.
        unsafe { &*(self as *const RawColors as *const [u32; RAW_COLORS_WORDS]) }
    }
}

const RAW_CAMERA_WORDS: usize = core::mem::size_of::<RawCamera>() / 4;

impl RawCamera {
    /// Its floats as bits.
    pub fn words(&self) -> &[u32; RAW_CAMERA_WORDS] {
        // Safe: as `GpuTransform::words` (size checked below).
        unsafe { &*(self as *const RawCamera as *const [u32; RAW_CAMERA_WORDS]) }
    }
}

impl GpuLights {
    /// Its floats as bits.
    pub fn words(&self) -> &[u32; GPU_LIGHTS_WORDS] {
        // Safe: as `GpuTransform::words`.
        unsafe { &*(self as *const GpuLights as *const [u32; GPU_LIGHTS_WORDS]) }
    }
}

impl GpuTransform {
    /// The screen-space vertex the GPU makes of `vertex` (model space), with
    /// the same operations in the same order as `RowTransform::apply`, so
    /// the same numbers to the bit (the PC replay draws with it).
    pub fn to_screen(&self, vertex: &FrameVertex) -> FrameVertex {
        let (x, y, z) = (vertex.x, vertex.y, vertex.z);
        let dot = |r: &[f32; 4]| r[0] * x + r[1] * y + r[2] * z + r[3];
        let [clip_x, clip_y, clip_z, clip_w] = self.rows.map(|row| dot(&row));
        let over_w = 1.0 / clip_w;
        FrameVertex {
            x: self.viewport[0] + self.viewport[2] * clip_x * over_w,
            y: self.viewport[1] + self.viewport[3] * clip_y * over_w,
            z: (self.depth[0] + self.depth[1] * clip_z * over_w).clamp(0.0, 1.0),
            w: clip_w,
            u: vertex.u,
            v: vertex.v,
            color: vertex.color,
            fog: dot(&self.fog_row),
        }
    }
}

impl Draw {
    /// Everything but the vertex and index ranges is the same: the two can
    /// be one GPU draw. Field by field rather than comparing copies, as it
    /// runs for every PSP draw (MTX: ~10,000 per frame). Keep it complete
    /// when `Draw` gets a field (the size check below notices a new one).
    #[inline]
    pub fn same_state(&self, other: &Draw) -> bool {
        self.texture == other.texture
            && self.flags == other.flags
            && self.framebuffer == other.framebuffer
            && self.blend == other.blend
            && self.tests == other.tests
            && self.scissor_left == other.scissor_left
            && self.scissor_top == other.scissor_top
            && self.scissor_right == other.scissor_right
            && self.scissor_bottom == other.scissor_bottom
            && self.fog_color == other.fog_color
            && self.transform == other.transform
            && self.vertex_format == other.vertex_format
            && self.raw_colors == other.raw_colors
            && self.bones == other.bones
    }
}

/// Everything needed to decode one PSP texture on the display side.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TextureDescriptor {
    /// Physical PSP address of the pixel data.
    pub address: u32,
    /// Row stride in pixels.
    pub buffer_width: u32,
    pub width: u16,
    pub height: u16,
    /// TEXTURE_FORMAT value (see `command::TEXTURE_FORMAT_NAMES`).
    pub format: u8,
    /// Pixels are stored in 16-byte x 8-row blocks.
    pub swizzled: u8,
    /// `texture_flags` bits.
    pub flags: u16,
    /// The raw CLUT_FORMAT register: palette entry format, index shift,
    /// mask and start. Only used by palette formats.
    pub clut_format: u32,
    /// Where the palette was copied in `Frame::palette_words`.
    pub palette_offset: u32,
    pub palette_word_count: u32,
    /// Identifies the palette contents, for texture caching. While the
    /// frame is recorded: the palette's index in `Frame::palettes` (its
    /// words aren't copied yet); the hash of its words once the frame is
    /// finished (`Frame::resolve_palette_hashes`).
    pub palette_hash: u32,
}

/// One palette of a frame (`Frame::palettes`): the PSP memory a CLUT_LOAD
/// loaded it from and where its words go in `Frame::palette_words`. The
/// capture thread only reserves the place; the words are copied from PSP
/// memory by the draw stream's copier and stored by the recorder thread
/// (`Geometry::fill_palette`), off the capture thread (see
/// `recorder::LoadedPalette`); for the capture thread's own draws the
/// copier stores them itself (`copy_tasks::Layout::Palette`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FramePalette {
    /// Physical address of the loaded words; 0 for words restored from a
    /// saved state (`recorder::LoadedPalette::load_state`).
    pub address: u32,
    pub word_count: u32,
    /// Where its words are in `Frame::palette_words`.
    pub offset: u32,
}

#[repr(C)]
pub struct Frame {
    pub frame_number: u32,
    /// The framebuffer the game displayed; draws into other framebuffers
    /// (off-screen rendering) are not shown.
    pub shown_framebuffer: u32,
    pub vertex_count: u32,
    pub index_count: u32,
    pub draw_count: u32,
    pub texture_count: u32,
    pub palette_word_count: u32,
    /// Set when something didn't fit into the fixed-size arrays.
    pub overflowed: u32,
    /// The index `add_texture` returned last, checked first next time
    /// (consecutive draws mostly use the same texture). Only meaningful
    /// while recording.
    pub last_texture: u32,
    pub vertices: [FrameVertex; MAX_VERTICES],
    pub indices: [u16; MAX_INDICES],
    pub draws: [Draw; MAX_DRAWS],
    pub textures: [TextureDescriptor; MAX_TEXTURES],
    /// The palettes' words (`Frame::palettes`). Part of the geometry: the
    /// recorder thread fills them (the capture thread only reserves them).
    pub palette_words: [u32; MAX_PALETTE_WORDS],
    /// Only used while recording: the frame's textures by a hash of their
    /// descriptor (index + 1, 0 = empty; linear probing), so `add_texture`
    /// finds one without comparing it with every texture of the frame (MTX:
    /// ~400 textures, ~900 lookups per frame; PPSSPP's texture cache is a
    /// hash map too).
    pub texture_lookup: [u16; TEXTURE_LOOKUP_SLOTS],
    /// Set when the vertices, indices or draws didn't fit (written by
    /// `Geometry`, merged into `overflowed` when the frame is finished:
    /// see `merge_overflow`).
    pub geometry_overflowed: u32,
    /// The transforms of draws whose vertices are in model space
    /// (`Draw::transform`). Part of the geometry.
    pub transform_count: u32,
    pub transforms: [GpuTransform; MAX_TRANSFORMS],
    /// The vertex cache's vertices (`vertex_cache::VertexCache`), which
    /// draws with `draw_flags::CACHED_VERTICES` use; null without a cache.
    /// Set when the frame is finished. GPU memory on the Vita.
    pub vertex_cache_len: u32,
    pub vertex_cache: *const FrameVertex,
    /// The lighting of GPU-lit draws (`GpuTransform::light_set`). Part of
    /// the geometry.
    pub light_set_count: u32,
    pub light_sets: [GpuLights; MAX_LIGHT_SETS],
    /// The raw arena (`raw_arena::RawArena`), which draws with
    /// `draw_flags::RAW_VERTICES` read; null without one. Set when the frame
    /// is finished. GPU memory on the Vita.
    pub raw_vertices_len: u32,
    pub raw_vertices: *const u8,
    /// The colour rules of raw draws (`Draw::raw_colors`). Part of the geometry.
    pub raw_colors_count: u32,
    pub raw_colors: [RawColors; MAX_RAW_COLORS],
    /// The bones of raw draws skinned in the shader (`Draw::bones`).
    /// Part of the geometry.
    pub bone_float_count: u32,
    pub bone_floats: [f32; MAX_BONE_FLOATS],
    /// The frame's palettes, one per palette address and length its draws
    /// use (a palette texture's `palette_hash` is its index here while the
    /// frame is recorded). Part of the catalog.
    pub palette_count: u32,
    pub palettes: [FramePalette; MAX_FRAME_PALETTES],
    /// Only used while recording: `palettes` by address and length (index +
    /// 1, 0 = empty; linear probing), so a palette loaded again later in the
    /// frame gets the same place. Part of the catalog.
    pub palette_slots: [u16; PALETTE_SLOTS],
    /// The hash of each palette's words (`texture::hash_words`), worked out
    /// by whoever stored them (`Geometry::fill_palette`). Part of the
    /// geometry.
    pub palette_hashes: [u32; MAX_FRAME_PALETTES],
    /// The mip levels of each texture that has them (`texture_flags::
    /// LEVELS_SHIFT`), at its index in `textures`; other entries are stale.
    /// Part of the catalog.
    pub texture_levels: [TextureLevels; MAX_TEXTURES],
    /// The index ring (`raw_arena::RawArena` with 2-byte steps, owned by the
    /// vertex cache): draws with `draw_flags::INDEX_RUNS` have their corners
    /// written there as a triangle list when the frame is finished
    /// (`write_ring_indices`), so the display passes the GPU a pointer
    /// instead of building the list. Null without a ring. GPU memory on the
    /// Vita. Length in indices.
    pub ring_indices_len: u32,
    pub ring_indices: *const u16,
    /// Per draw: where its triangle list starts in `ring_indices` (in
    /// indices), or `NO_RING_INDICES` (no runs, or the ring was full: the
    /// display builds the list as before). Only meaningful with a ring.
    pub draw_ring_index: [u32; MAX_DRAWS],
    /// Per texture: `texture::quick_sample` of its data, taken by the
    /// capture at the texture's first use in the frame, while the GE sync
    /// holds the game at this frame; 0 = not sampled (VRAM, unreadable).
    /// The texture cache compares it with the data it decoded. Part of the
    /// catalog; entries of textures not in the frame are stale.
    pub texture_samples: [u32; MAX_TEXTURES],
    /// Per texture one bit: its sample was taken in this frame, or handed
    /// to the copier (`Catalog::texture_sample_place`). Only used while
    /// recording; a texture's bit is cleared when it is added. Part of the
    /// catalog.
    pub texture_sampled: [u32; MAX_TEXTURES / 32],
    /// The objects and cameras of raw draws (`Draw::transform` of draws with
    /// `draw_flags::RAW_VERTICES`, `RawTransform::camera`). Part of the
    /// geometry.
    pub raw_transform_count: u32,
    pub raw_transforms: [RawTransform; MAX_RAW_TRANSFORMS],
    pub raw_camera_count: u32,
    pub raw_cameras: [RawCamera; MAX_RAW_CAMERAS],
    /// Only used while recording: the camera change number
    /// (`TransformInputs::camera_change`) of the newest camera lookup
    /// (`RawTables::add_raw_camera_of`) and the camera it gave; `[0, _]`:
    /// none. Part of the geometry's raw tables.
    pub raw_camera_last: [u32; 2],
    /// Only used while recording: the lights and camera change numbers and
    /// the lighting key of the newest light set lookup
    /// (`RawTables::add_light_set_of`) and the light set it gave; `[0, ..]`:
    /// none. Part of the geometry's raw tables.
    pub light_set_last: [u32; 4],
    /// Only used while recording: draws before this one are never merged
    /// with a new one (`Geometry::seal`): the capture thread's draws
    /// (`frame_builder`) go between them when the frame is finished. Part of
    /// the geometry.
    pub draw_merge_floor: u32,
    /// The GE's dither matrix (`dither_value`), for draws with
    /// `draw_flags::DITHER`: the one of the frame's last draw whose state was
    /// worked out with dithering on. Part of the catalog.
    pub dither_matrix: u64,
}

/// `Frame::draw_ring_index` of a draw whose list isn't in the ring.
pub const NO_RING_INDICES: u32 = u32::MAX;

// The sizes the dumps and the GPU uniform layouts rely on.
const _: () = assert!(core::mem::size_of::<FrameVertex>() == 32);
const _: () = assert!(core::mem::size_of::<Draw>() == 64);
const _: () = assert!(core::mem::size_of::<GpuTransform>() == 192);
const _: () = assert!(core::mem::size_of::<GpuLights>() == 352);
const _: () = assert!(core::mem::size_of::<TextureDescriptor>() == 32);
// 9 header words, vertices, indices: draws start here (the header checks the same).
const _: () = assert!(core::mem::offset_of!(Frame, draws) == 2_490_404);
// The transform table at the end (the header checks the same).
const _: () = assert!(core::mem::offset_of!(Frame, transforms) == 3_295_276);
const _: () = assert!(core::mem::offset_of!(Frame, vertex_cache) == 3_344_432);
const _: () = assert!(core::mem::size_of::<RawCamera>() == 44 * 4);
const _: () = assert!(core::mem::size_of::<RawTransform>() == 64);
const _: () = assert!(core::mem::size_of::<FramePalette>() == 12);
const _: () = assert!(core::mem::size_of::<TextureLevels>() == 56);

impl Frame {
    /// Empties the frame so it can be filled again.
    pub fn clear(&mut self) {
        self.frame_number = 0;
        self.shown_framebuffer = 0;
        self.vertex_count = 0;
        self.index_count = 0;
        self.draw_count = 0;
        self.texture_count = 0;
        self.palette_word_count = 0;
        self.overflowed = 0;
        self.last_texture = 0;
        self.texture_lookup = [0; TEXTURE_LOOKUP_SLOTS];
        self.geometry_overflowed = 0;
        self.transform_count = 0;
        self.light_set_count = 0;
        self.vertex_cache_len = 0;
        self.vertex_cache = core::ptr::null();
        self.raw_vertices_len = 0;
        self.raw_vertices = core::ptr::null();
        self.ring_indices_len = 0;
        self.ring_indices = core::ptr::null();
        self.raw_colors_count = 0;
        self.bone_float_count = 0;
        self.raw_transform_count = 0;
        self.raw_camera_count = 0;
        self.raw_camera_last = [0, 0];
        self.light_set_last = [0; 4];
        self.draw_merge_floor = 0;
        self.palette_count = 0;
        self.palette_slots = [0; PALETTE_SLOTS];
    }

    /// The vertex cache's vertices this frame's cached draws use.
    pub fn cached_vertices(&self) -> &[FrameVertex] {
        if self.vertex_cache.is_null() {
            return &[];
        }
        // Safe: set from the cache's arena, which outlives every frame that
        // uses it (`VertexCache`: runs of frames still shown stay put).
        unsafe { core::slice::from_raw_parts(self.vertex_cache, self.vertex_cache_len as usize) }
    }

    /// The raw arena's bytes this frame's raw draws use.
    pub fn raw_bytes(&self) -> &[u8] {
        if self.raw_vertices.is_null() {
            return &[];
        }
        // Safe: set from the raw arena, which outlives every frame that
        // uses it (`RawArena`: runs of recent frames stay put).
        unsafe { core::slice::from_raw_parts(self.raw_vertices, self.raw_vertices_len as usize) }
    }

    /// The frame's two parts while recording: its geometry (vertices,
    /// indices, draws) and its catalog (textures and palettes). Recording
    /// fills them from two halves that may run on different cores (see
    /// `draw_stream`); each part only touches its own fields.
    pub fn split(&mut self) -> (Geometry<'_>, Catalog<'_>) {
        let Frame {
            vertex_count,
            index_count,
            draw_count,
            texture_count,
            palette_word_count,
            overflowed,
            last_texture,
            dither_matrix,
            vertices,
            indices,
            draws,
            textures,
            palette_words,
            texture_lookup,
            geometry_overflowed,
            transform_count,
            transforms,
            light_set_count,
            light_sets,
            raw_colors_count,
            raw_colors,
            bone_float_count,
            bone_floats,
            palette_count,
            palettes,
            palette_slots,
            palette_hashes,
            texture_levels,
            texture_samples,
            texture_sampled,
            raw_transform_count,
            raw_transforms,
            raw_camera_count,
            raw_cameras,
            raw_camera_last,
            light_set_last,
            draw_merge_floor,
            ..
        } = self;
        (
            Geometry {
                vertex_count,
                index_count,
                draw_count,
                overflowed: geometry_overflowed,
                merge_floor: draw_merge_floor,
                vertices,
                indices,
                draws,
                transform_count,
                transforms,
                palette_words,
                palette_hashes,
                raw: Some(RawTables {
                    light_set_count,
                    light_sets,
                    raw_colors_count,
                    raw_colors,
                    bone_float_count,
                    bone_floats,
                    raw_transform_count,
                    raw_transforms,
                    raw_camera_count,
                    raw_cameras,
                    raw_camera_last,
                    light_set_last,
                }),
            },
            Catalog {
                texture_count,
                palette_word_count,
                overflowed,
                last_texture,
                dither_matrix,
                textures,
                texture_lookup,
                palette_count,
                palettes,
                palette_slots,
                texture_levels,
                texture_samples: texture_samples.as_mut_ptr(),
                texture_sampled,
            },
        )
    }

    /// The geometry part of `split`, from a pointer: for the recorder's
    /// consumer thread, while the producer works on the catalog. Without
    /// `with_raw_tables` the geometry has no raw tables (`Geometry::raw`):
    /// the producer owns them then (`raw_tables_raw`, `frame_builder`).
    ///
    /// # Safety
    /// `frame` must point to a frame whose geometry (and with
    /// `with_raw_tables` its raw tables) nobody else uses for `'a` (its
    /// catalog may be in use).
    pub unsafe fn geometry_raw<'a>(frame: *mut Frame, with_raw_tables: bool) -> Geometry<'a> {
        use core::ptr::addr_of_mut;
        Geometry {
            vertex_count: &mut *addr_of_mut!((*frame).vertex_count),
            index_count: &mut *addr_of_mut!((*frame).index_count),
            draw_count: &mut *addr_of_mut!((*frame).draw_count),
            overflowed: &mut *addr_of_mut!((*frame).geometry_overflowed),
            merge_floor: &mut *addr_of_mut!((*frame).draw_merge_floor),
            vertices: &mut *addr_of_mut!((*frame).vertices),
            indices: &mut *addr_of_mut!((*frame).indices),
            draws: &mut *addr_of_mut!((*frame).draws),
            transform_count: &mut *addr_of_mut!((*frame).transform_count),
            transforms: &mut *addr_of_mut!((*frame).transforms),
            palette_words: &mut *addr_of_mut!((*frame).palette_words),
            palette_hashes: &mut *addr_of_mut!((*frame).palette_hashes),
            raw: if with_raw_tables { Some(Frame::raw_tables_raw(frame)) } else { None },
        }
    }

    /// The tables of raw draws (`RawTables`), from a pointer: for whoever
    /// records the raw draws (the recorder, or the capture thread's
    /// `frame_builder`).
    ///
    /// # Safety
    /// `frame` must point to a frame whose raw tables nobody else uses for
    /// `'a`.
    pub unsafe fn raw_tables_raw<'a>(frame: *mut Frame) -> RawTables<'a> {
        use core::ptr::addr_of_mut;
        RawTables {
            light_set_count: &mut *addr_of_mut!((*frame).light_set_count),
            light_sets: &mut *addr_of_mut!((*frame).light_sets),
            raw_colors_count: &mut *addr_of_mut!((*frame).raw_colors_count),
            raw_colors: &mut *addr_of_mut!((*frame).raw_colors),
            bone_float_count: &mut *addr_of_mut!((*frame).bone_float_count),
            bone_floats: &mut *addr_of_mut!((*frame).bone_floats),
            raw_transform_count: &mut *addr_of_mut!((*frame).raw_transform_count),
            raw_transforms: &mut *addr_of_mut!((*frame).raw_transforms),
            raw_camera_count: &mut *addr_of_mut!((*frame).raw_camera_count),
            raw_cameras: &mut *addr_of_mut!((*frame).raw_cameras),
            raw_camera_last: &mut *addr_of_mut!((*frame).raw_camera_last),
            light_set_last: &mut *addr_of_mut!((*frame).light_set_last),
        }
    }

    /// Where palette `place`'s words and its hash go, as pointers for the
    /// copy task that stores them (`copy_tasks::Layout::Palette`, the
    /// capture thread's draws); `None` if the place lies outside the lists.
    /// Nobody else writes them: the place is this palette's alone
    /// (`Catalog::add_palette`), and the recorder stores only the palettes
    /// its jobs carry.
    ///
    /// # Safety
    /// `frame` must point to a frame.
    pub unsafe fn palette_place_raw(frame: *mut Frame, place: PalettePlace) -> Option<(*mut u32, *mut u32)> {
        use core::ptr::addr_of_mut;
        let end = (place.offset as usize).checked_add(place.word_count as usize)?;
        if end > MAX_PALETTE_WORDS || place.index as usize >= MAX_FRAME_PALETTES {
            return None;
        }
        let words = addr_of_mut!((*frame).palette_words) as *mut u32;
        let hashes = addr_of_mut!((*frame).palette_hashes) as *mut u32;
        Some((words.add(place.offset as usize), hashes.add(place.index as usize)))
    }

    /// The catalog part of `split`, from a pointer: for the recorder's
    /// producer, while the consumer works on the geometry.
    ///
    /// # Safety
    /// `frame` must point to a frame whose catalog nobody else uses for `'a`.
    pub unsafe fn catalog_raw<'a>(frame: *mut Frame) -> Catalog<'a> {
        use core::ptr::addr_of_mut;
        Catalog {
            texture_count: &mut *addr_of_mut!((*frame).texture_count),
            palette_word_count: &mut *addr_of_mut!((*frame).palette_word_count),
            overflowed: &mut *addr_of_mut!((*frame).overflowed),
            last_texture: &mut *addr_of_mut!((*frame).last_texture),
            dither_matrix: &mut *addr_of_mut!((*frame).dither_matrix),
            textures: &mut *addr_of_mut!((*frame).textures),
            texture_lookup: &mut *addr_of_mut!((*frame).texture_lookup),
            palette_count: &mut *addr_of_mut!((*frame).palette_count),
            palettes: &mut *addr_of_mut!((*frame).palettes),
            palette_slots: &mut *addr_of_mut!((*frame).palette_slots),
            texture_levels: &mut *addr_of_mut!((*frame).texture_levels),
            texture_samples: addr_of_mut!((*frame).texture_samples) as *mut u32,
            texture_sampled: &mut *addr_of_mut!((*frame).texture_sampled),
        }
    }

    /// True if something didn't fit into the frame.
    pub fn is_overflowed(&self) -> bool {
        self.overflowed != 0 || self.geometry_overflowed != 0
    }

    /// Folds the geometry's overflow flag into `overflowed` (which the
    /// display side reads), when the frame is finished.
    pub fn merge_overflow(&mut self) {
        self.overflowed |= self.geometry_overflowed;
    }

    /// When the frame is finished (every palette stored): gives each
    /// palette texture the hash of its palette's words in place of the
    /// palette's index (`TextureDescriptor::palette_hash`): the display side
    /// tells palettes and textures apart by their contents.
    pub fn resolve_palette_hashes(&mut self) {
        let palette_count = self.palette_count as usize;
        for texture in &mut self.textures[..self.texture_count as usize] {
            if crate::texture::is_palette_format(texture.format) {
                let index = texture.palette_hash as usize;
                texture.palette_hash = if index < palette_count { self.palette_hashes[index] } else { 0 };
            }
        }
    }

    /// The mip levels of the texture at `index` (meaningful when its flags
    /// have `texture_flags::LEVELS_SHIFT`).
    /// `texture_samples` of texture `index` (0 = not sampled).
    pub fn texture_sample(&self, index: usize) -> u32 {
        self.texture_samples.get(index).copied().unwrap_or(0)
    }

    pub fn texture_levels(&self, index: usize) -> &TextureLevels {
        &self.texture_levels[index.min(MAX_TEXTURES - 1)]
    }

    /// The frame's palettes (`Frame::palettes`).
    pub fn palettes(&self) -> &[FramePalette] {
        &self.palettes[..self.palette_count as usize]
    }

    // The parts' methods on the whole frame (tests, the replay).

    pub fn add_vertex(&mut self, first_vertex: u32, vertex: FrameVertex) -> Option<u16> {
        self.split().0.add_vertex(first_vertex, vertex)
    }

    pub fn add_index(&mut self, number: u16) -> Option<()> {
        self.split().0.add_index(number)
    }

    pub fn add_corner(&mut self, first_vertex: u32, vertex: FrameVertex) -> Option<()> {
        self.split().0.add_corner(first_vertex, vertex)
    }

    pub fn undo_draw(&mut self, draw: &Draw) {
        self.split().0.undo_draw(draw)
    }

    pub fn finish_draw(&mut self, draw: Draw) {
        self.split().0.finish_draw(draw)
    }

    pub fn add_texture(&mut self, texture: TextureDescriptor) -> Option<u32> {
        self.split().1.add_texture(texture)
    }

    pub fn add_palette(&mut self, address: u32, word_count: u32) -> Option<PalettePlace> {
        self.split().1.add_palette(address, word_count)
    }

    pub fn fill_palette(&mut self, place: PalettePlace, words: &[u32]) {
        self.split().0.fill_palette(place, words)
    }

    pub fn draws(&self) -> &[Draw] {
        &self.draws[..self.draw_count as usize]
    }

    /// Writes the corners of every draw with index runs or raw vertices into
    /// `ring` as a triangle list (index runs written out in the display's
    /// `DrawIndices` order, `run_indices`; a raw draw's plain indices, the
    /// converted ones of indexed PRIMs, copied as they are), notes where in
    /// `draw_ring_index` and points `ring_indices` at the ring, so the GPU
    /// reads them there. Draws whose list doesn't fit get `NO_RING_INDICES`
    /// (the display copies them then). Returns how many didn't fit.
    pub fn write_ring_indices(&mut self, ring: &mut crate::raw_arena::RawArena) -> u32 {
        let mut full = 0;
        for i in 0..self.draw_count as usize {
            let draw = self.draws[i];
            self.draw_ring_index[i] = NO_RING_INDICES;
            let stored = &self.indices[draw.first_index as usize..][..draw.index_count as usize];
            let runs = draw.flags & draw_flags::INDEX_RUNS != 0;
            if !runs && draw.flags & draw_flags::RAW_VERTICES == 0 {
                continue; // the display copies the CPU path's indices with their vertices
            }
            let corners: u32 =
                if runs { stored.chunks_exact(3).map(|run| 3 * run_triangles(run[0], run[1])).sum() } else { draw.index_count };
            if corners == 0 {
                continue;
            }
            let Some((placed, out)) = ring.reserve(corners * 2, 2) else {
                full += 1;
                continue;
            };
            if runs {
                let mut at = 0;
                for run in stored.chunks_exact(3) {
                    for corner in run_indices(run) {
                        out[at..at + 2].copy_from_slice(&corner.to_le_bytes());
                        at += 2;
                    }
                }
            } else {
                for (bytes, index) in out.chunks_exact_mut(2).zip(stored) {
                    bytes.copy_from_slice(&index.to_le_bytes());
                }
            }
            self.draw_ring_index[i] = placed.offset / 2;
        }
        self.ring_indices = ring.bytes().as_ptr() as *const u16;
        self.ring_indices_len = (ring.bytes().len() / 2) as u32;
        full
    }

    pub fn vertices(&self) -> &[FrameVertex] {
        &self.vertices[..self.vertex_count as usize]
    }

    /// The index words as stored (index runs for draws with
    /// `draw_flags::INDEX_RUNS`; `draw_indices` gives a draw's corners).
    pub fn indices(&self) -> &[u16] {
        &self.indices[..self.index_count as usize]
    }

    /// A draw's triangle corners as vertex numbers (three per triangle),
    /// its index runs written out.
    pub fn draw_indices<'a>(&'a self, draw: &Draw) -> impl Iterator<Item = u16> + 'a {
        let stored = &self.indices[draw.first_index as usize..(draw.first_index + draw.index_count) as usize];
        let (plain, runs) = if draw.flags & draw_flags::INDEX_RUNS != 0 { (&[][..], stored) } else { (stored, &[][..]) };
        plain.iter().copied().chain(runs.chunks_exact(3).flat_map(run_indices))
    }

    /// How many triangle corners a draw has (`index_count`, or its runs'
    /// triangles x 3).
    pub fn draw_index_count(&self, draw: &Draw) -> u32 {
        if draw.flags & draw_flags::INDEX_RUNS == 0 {
            return draw.index_count;
        }
        let runs = &self.indices[draw.first_index as usize..(draw.first_index + draw.index_count) as usize];
        runs.chunks_exact(3).map(|run| 3 * run_triangles(run[0], run[1])).sum()
    }

    /// All draws' triangle corners (as if every index were stored).
    pub fn corner_count(&self) -> u32 {
        self.draws().iter().map(|draw| self.draw_index_count(draw)).sum()
    }

    /// A draw's triangle corners, in order (three per triangle), as stored
    /// (in model space for draws with a `transform`).
    /// Raw draws have no stored corners (see `screen_corners`): none.
    pub fn draw_corners<'a>(&'a self, draw: &Draw) -> impl Iterator<Item = &'a FrameVertex> + 'a {
        let vertices = if draw.flags & draw_flags::RAW_VERTICES != 0 {
            &[][..]
        } else if draw.flags & draw_flags::CACHED_VERTICES != 0 {
            // Indices count from the draw's window (`finish_cached_draw`).
            &self.cached_vertices()[draw.first_vertex as usize..]
        } else {
            &self.vertices[draw.first_vertex as usize..(draw.first_vertex + draw.vertex_count) as usize]
        };
        let raw = draw.flags & draw_flags::RAW_VERTICES != 0;
        self.draw_indices(draw).filter(move |_| !raw).map(move |index| &vertices[index as usize])
    }

    /// A draw's transform (`Draw::transform`), if its vertices are in model
    /// space.
    pub fn transform_of(&self, draw: &Draw) -> Option<&GpuTransform> {
        if draw.flags & draw_flags::RAW_VERTICES != 0 {
            return None; // `raw_transform_of`
        }
        let index = (draw.transform as usize).checked_sub(1)?;
        self.transforms[..self.transform_count as usize].get(index)
    }

    /// A raw draw's object and camera (`Draw::transform`, `RawTransform`).
    pub fn raw_transform_of(&self, draw: &Draw) -> Option<(&RawTransform, &RawCamera)> {
        if draw.flags & draw_flags::RAW_VERTICES == 0 {
            return None;
        }
        let index = (draw.transform as usize).checked_sub(1)?;
        let transform = self.raw_transforms[..self.raw_transform_count as usize].get(index)?;
        let camera = self.raw_cameras[..self.raw_camera_count as usize].get(transform.camera as usize)?;
        Some((transform, camera))
    }

    /// A draw's triangle corners in screen space, as the GPU draws them
    /// (`GpuTransform::to_screen` for draws with a transform; raw draws
    /// decoded and cut at the near plane, `raw_vertex::RawCorners`).
    pub fn screen_corners<'a>(&'a self, draw: &Draw) -> impl Iterator<Item = FrameVertex> + 'a {
        let transform = self.transform_of(draw).copied();
        let raw = match self.raw_transform_of(draw) {
            Some((transform, camera)) if draw.flags & draw_flags::RAW_PSP_RAM == 0 => Some(crate::raw_vertex::RawCorners::new(
                self,
                draw.first_vertex,
                draw.vertex_format,
                (*transform, *camera),
                self.raw_colors_of(draw).copied().unwrap_or_default(),
                self.bones_of(draw),
                self.draw_indices(draw),
                draw.flags & draw_flags::RAW_SPRITES != 0,
            )),
            _ => None,
        };
        let lights = transform.and_then(|transform| self.light_set_of(&transform)).copied();
        let stored = self.draw_corners(draw).map(move |vertex| match &transform {
            Some(transform) => {
                let mut screen = transform.to_screen(vertex);
                // GPU-lit: the colour the 3D program lights (nativeres_3d_v.cg).
                if let Some(lights) = &lights {
                    let normal = crate::lighting::unpack_normal(vertex.w.to_bits());
                    screen.color = lights.color(vertex.color, normal, &transform.normal_rows);
                }
                screen
            }
            None => *vertex,
        });
        stored.chain(raw.into_iter().flatten())
    }

    /// The bones of a raw draw skinned in the shader (`Draw::bones`): 12
    /// floats per bone, one bone per weight of its format; empty if not skinned.
    pub fn bones_of(&self, draw: &Draw) -> &[f32] {
        let Some(first) = (draw.bones as usize).checked_sub(1) else {
            return &[];
        };
        let count = crate::vertex::VertexFormat(draw.vertex_format).weight_count() as usize;
        self.bone_floats[..self.bone_float_count as usize].get(first..first + 12 * count).unwrap_or(&[])
    }

    /// A raw draw's colour rules (`Draw::raw_colors`).
    pub fn raw_colors_of(&self, draw: &Draw) -> Option<&RawColors> {
        let index = (draw.raw_colors as usize).checked_sub(1)?;
        self.raw_colors[..self.raw_colors_count as usize].get(index)
    }

    /// The lighting of a GPU-lit draw's transform (`GpuTransform::light_set`).
    pub fn light_set_of(&self, transform: &GpuTransform) -> Option<&GpuLights> {
        let number = transform.light_set[0] as usize;
        self.light_sets[..self.light_set_count as usize].get(number.checked_sub(1)?)
    }

    pub fn textures(&self) -> &[TextureDescriptor] {
        &self.textures[..self.texture_count as usize]
    }

    /// The framebuffer at `address` is read as colours by one of this
    /// frame's draws (render to texture: MTX blends the other framebuffer,
    /// i.e. the previous frame, over each new one at up to 95 %).
    pub fn samples_framebuffer(&self, address: u32) -> bool {
        address != 0
            && self.textures().iter().any(|texture| texture.address == address && !crate::texture::is_palette_format(texture.format))
    }



    /// Identifies the frame description exactly, to check that a change to
    /// the capture leaves its output the same (FNV-1a over all recorded
    /// data; the regression test's "hash", tests/expected.txt). Hashes the
    /// triangles as drawn (each draw's corners via its indices), so how they
    /// are stored (shared vertices or not) doesn't change it, and in screen
    /// space (`screen_corners`), so neither does whether the CPU or the GPU
    /// transforms them. Leaves out the vertices' depth and fog (see
    /// `vertex_hash`).
    pub fn description_hash(&self) -> u64 {
        let mut hash = 0xCBF2_9CE4_8422_2325u64;
        let mut add = |word: u32| hash = (hash ^ word as u64).wrapping_mul(0x0100_0000_01B3);
        // The corners as if every index were stored: index runs are only a
        // way of storing them (`draw_flags::INDEX_RUNS`).
        let mut first_corner = 0;
        for draw in self.draws() {
            let corners = self.draw_index_count(draw);
            let flags = draw.flags & !draw_flags::INDEX_RUNS;
            let fields = [first_corner, corners, draw.texture, flags, draw.framebuffer, draw.blend, draw.tests];
            first_corner += corners;
            fields.into_iter().for_each(&mut add);
            add((draw.scissor_left as u32) << 16 | draw.scissor_top as u32);
            add((draw.scissor_right as u32) << 16 | draw.scissor_bottom as u32);
        }
        for draw in self.draws() {
            for vertex in self.screen_corners(draw) {
                [vertex.x.to_bits(), vertex.y.to_bits(), vertex.w.to_bits(), vertex.u.to_bits(), vertex.v.to_bits(), vertex.color]
                    .into_iter()
                    .for_each(&mut add);
            }
        }
        for texture in self.textures() {
            let fields = [
                texture.address,
                texture.buffer_width,
                (texture.width as u32) << 16 | texture.height as u32,
                (texture.format as u32) << 24 | (texture.swizzled as u32) << 16 | texture.flags as u32,
                texture.clut_format,
                texture.palette_offset,
                texture.palette_word_count,
                texture.palette_hash,
            ];
            fields.into_iter().for_each(&mut add);
        }
        self.palette_words[..self.palette_word_count as usize].iter().copied().for_each(&mut add);
        hash
    }

    /// Every field of every triangle corner, depth and fog included: with
    /// `description_hash`, a dumped frame captured on the Vita and on the PC
    /// shows that code only the Vita runs (NEON) gives the same frame.
    /// Draws without `draw_flags::FOG` that have a vertex whose fog isn't 1.
    /// Always 0: the Vita copies a draw's vertices to the GPU as they are,
    /// and its program fogs every draw by the vertices' fog (1 = none).
    /// (Draws with a transform get their fog from it: 1 without FOG.)
    pub fn unfogged_draws_with_fog(&self) -> u32 {
        let vertices = self.vertices();
        self.draws()
            .iter()
            .filter(|draw| draw.flags & draw_flags::FOG == 0 && draw.transform == 0)
            .filter(|draw| {
                let start = draw.first_vertex as usize;
                let end = (start + draw.vertex_count as usize).min(vertices.len());
                vertices[start.min(end)..end].iter().any(|vertex| vertex.fog != 1.0)
            })
            .count() as u32
    }

    pub fn vertex_hash(&self) -> u64 {
        let mut hash = 0xCBF2_9CE4_8422_2325u64;
        let mut add = |word: u32| hash = (hash ^ word as u64).wrapping_mul(0x0100_0000_01B3);
        for draw in self.draws() {
            let fogged = draw.flags & draw_flags::FOG != 0;
            for v in self.screen_corners(draw) {
                // Without FOG the GPU ignores the fog (always 1 on the CPU path).
                let fog = if fogged { v.fog } else { 1.0 };
                [v.x, v.y, v.z, v.w, v.u, v.v, fog].into_iter().for_each(|value| add(value.to_bits()));
                add(v.color);
            }
        }
        hash
    }






}

/// A frame's vertices, indices and draws while recording (`Frame::split`).
/// The lists are slices so that the capture thread's draws can be built in
/// lists of their own with the same rules (`frame_builder`): such a geometry
/// has no vertices, transforms or palettes (empty slices).
pub struct Geometry<'a> {
    vertex_count: &'a mut u32,
    index_count: &'a mut u32,
    draw_count: &'a mut u32,
    /// `Frame::geometry_overflowed`.
    overflowed: &'a mut u32,
    /// `Frame::draw_merge_floor`: draws below it are never merged with.
    merge_floor: &'a mut u32,
    vertices: &'a mut [FrameVertex],
    indices: &'a mut [u16],
    draws: &'a mut [Draw],
    transform_count: &'a mut u32,
    transforms: &'a mut [GpuTransform],
    palette_words: &'a mut [u32],
    palette_hashes: &'a mut [u32],
    /// The tables of raw draws, if whoever records into this geometry owns
    /// them: `None` while the capture thread records the raw draws itself
    /// (`frame_builder`); the recorder then draws everything on the CPU path.
    pub raw: Option<RawTables<'a>>,
}

/// The tables raw draws name (`Draw::transform`, `raw_colors`, `bones`, the
/// light sets of `RawTransform` and of GPU-lit CPU-path draws): written by
/// one thread at a time, the recorder or the capture thread's
/// `frame_builder` (`Frame::geometry_raw`, `Frame::raw_tables_raw`).
pub struct RawTables<'a> {
    light_set_count: &'a mut u32,
    light_sets: &'a mut [GpuLights; MAX_LIGHT_SETS],
    raw_colors_count: &'a mut u32,
    raw_colors: &'a mut [RawColors; MAX_RAW_COLORS],
    bone_float_count: &'a mut u32,
    bone_floats: &'a mut [f32; MAX_BONE_FLOATS],
    raw_transform_count: &'a mut u32,
    raw_transforms: &'a mut [RawTransform; MAX_RAW_TRANSFORMS],
    raw_camera_count: &'a mut u32,
    raw_cameras: &'a mut [RawCamera; MAX_RAW_CAMERAS],
    raw_camera_last: &'a mut [u32; 2],
    light_set_last: &'a mut [u32; 4],
}

/// How many of the newest table entries `RawTables::add_raw_colors` checks
/// for the same one (draws alternate between a few materials).
const RAW_COLORS_RECENT: usize = 8;
/// The same for `RawTables::add_raw_camera`.
const RAW_CAMERAS_RECENT: usize = 8;

impl<'a> Geometry<'a> {
    /// A geometry over lists of its own (`frame_builder`): draws and indices
    /// only (no vertices, transforms or palettes), with `raw` as its raw
    /// tables.
    pub fn of_lists(
        counts: &'a mut ListCounts,
        draws: &'a mut [Draw],
        indices: &'a mut [u16],
        raw: Option<RawTables<'a>>,
    ) -> Geometry<'a> {
        let ListCounts { vertex_count, index_count, draw_count, overflowed, merge_floor, transform_count } = counts;
        Geometry {
            vertex_count,
            index_count,
            draw_count,
            overflowed,
            merge_floor,
            vertices: &mut [],
            indices,
            draws,
            transform_count,
            transforms: &mut [],
            palette_words: &mut [],
            palette_hashes: &mut [],
            raw,
        }
    }

    /// The raw tables, if this geometry has them.
    pub fn raw_tables(&mut self) -> Option<&mut RawTables<'a>> {
        self.raw.as_mut()
    }

    /// Draws added from now on are never merged with those before (another
    /// writer's draws will go between them, `frame_builder`).
    pub fn seal(&mut self) {
        *self.merge_floor = *self.draw_count;
    }

    pub fn draw_count(&self) -> u32 {
        *self.draw_count
    }

    pub fn is_overflowed(&self) -> bool {
        *self.overflowed != 0
    }

    /// Stores the words of the palette at `place` (`Catalog::add_palette`)
    /// and their hash: `words` as copied from PSP memory, fewer than
    /// `place.word_count` if the rest couldn't be read (zeros then).
    pub fn fill_palette(&mut self, place: PalettePlace, words: &[u32]) {
        let stored = &mut self.palette_words[place.offset as usize..][..place.word_count as usize];
        let readable = words.len().min(stored.len());
        stored[..readable].copy_from_slice(&words[..readable]);
        stored[readable..].fill(0);
        self.palette_hashes[place.index as usize] = crate::texture::hash_words(stored);
    }

    /// `RawTables::add_light_set`; `None` without raw tables too (the
    /// caller lights on the CPU then).
    pub fn add_light_set(&mut self, lights: &GpuLights) -> Option<u32> {
        self.raw.as_mut()?.add_light_set(lights)
    }
}

/// The newest of the first `count` draws, the one a new draw may be merged
/// with, unless it lies below `floor` (`Geometry::seal`).
fn mergeable(draws: &mut [Draw], floor: u32, count: u32) -> Option<&mut Draw> {
    draws[floor.min(count) as usize..count as usize].last_mut()
}

/// The counts of a `Geometry::of_lists` geometry.
#[derive(Clone, Copy, Debug, Default)]
pub struct ListCounts {
    pub vertex_count: u32,
    pub index_count: u32,
    pub draw_count: u32,
    pub overflowed: u32,
    pub merge_floor: u32,
    pub transform_count: u32,
}

impl ListCounts {
    pub const EMPTY: ListCounts =
        ListCounts { vertex_count: 0, index_count: 0, draw_count: 0, overflowed: 0, merge_floor: 0, transform_count: 0 };
}

impl RawTables<'_> {
    /// Stores `bones` (4x3, column by column, as uploaded) for the vertex
    /// programs, each as 3 rows of 4 floats (`Draw::bones`), and returns
    /// where they start in `Frame::bone_floats`; the last bones stored again
    /// if equal. `None` if the pool is full.
    ///
    /// Written behind the newest ones first, then compared with them by
    /// their bits (one pass; float compares stall the A9's pipeline, see
    /// `GPU_TRANSFORM_WORDS`): MTX's riders change bones before nearly every
    /// part.
    pub fn add_bones(&mut self, bones: &[[f32; 12]]) -> Option<u32> {
        let count = *self.bone_float_count as usize;
        let size = 12 * bones.len();
        if count + size > MAX_BONE_FLOATS {
            // No room behind them: only the newest ones can still be used.
            let same = count >= size
                && bones.iter().enumerate().all(|(b, bone)| {
                    let stored = &self.bone_floats[count - size + 12 * b..][..12];
                    (0..12).all(|i| stored[i].to_bits() == bone[(i % 4) * 3 + i / 4].to_bits())
                });
            return same.then(|| (count - size) as u32);
        }
        for (b, bone) in bones.iter().enumerate() {
            let rows = &mut self.bone_floats[count + 12 * b..][..12];
            for (i, value) in rows.iter_mut().enumerate() {
                // Row i / 4, column i % 4.
                *value = bone[(i % 4) * 3 + i / 4];
            }
        }
        if count >= size {
            let (stored, new) = self.bone_floats[count - size..count + size].split_at(size);
            if stored.iter().zip(new).all(|(a, b)| a.to_bits() == b.to_bits()) {
                return Some((count - size) as u32);
            }
        }
        *self.bone_float_count += size as u32;
        Some(count as u32)
    }

    /// The `Draw::raw_colors` value for `colors`: one of the newest entries
    /// if equal (by bits), else a new one. `None` if the table is full (the
    /// run goes the old way).
    pub fn add_raw_colors(&mut self, colors: &RawColors) -> Option<u32> {
        let count = *self.raw_colors_count as usize;
        let recent = count.saturating_sub(RAW_COLORS_RECENT);
        if let Some(found) = (recent..count).rev().find(|&index| self.raw_colors[index].words() == colors.words()) {
            return Some(found as u32 + 1);
        }
        if count == MAX_RAW_COLORS {
            return None;
        }
        self.raw_colors[count] = *colors;
        *self.raw_colors_count += 1;
        Some(count as u32 + 1)
    }

    /// The `GpuTransform::light_set` number for `lights`: the last one
    /// added if it is the same, else a new entry. `None` if the table is
    /// full (the caller lights on the CPU instead).
    pub fn add_light_set(&mut self, lights: &GpuLights) -> Option<u32> {
        // The table may change: `add_light_set_of`'s note no longer holds.
        self.light_set_last[0] = 0;
        let count = *self.light_set_count as usize;
        if count > 0 && self.light_sets[count - 1].words() == lights.words() {
            return Some(count as u32);
        }
        if count == MAX_LIGHT_SETS {
            return None;
        }
        self.light_sets[count] = *lights;
        *self.light_set_count += 1;
        Some(count as u32 + 1)
    }

    /// The newest raw transform (`Frame::raw_transforms`) and its
    /// `Draw::transform` number, if the frame has one (`raw_blocks`).
    pub fn last_raw_transform(&self) -> Option<(u32, &RawTransform)> {
        let count = *self.raw_transform_count as usize;
        count.checked_sub(1).map(|last| (count as u32, &self.raw_transforms[last]))
    }

    /// Replaces the newest raw transform's bookkeeping (`raw_blocks`: the
    /// same entry found again for other inputs).
    pub fn restamp_last_raw_transform(&mut self, inputs_change: u32, lighting_change: u32) {
        if let Some(last) = (*self.raw_transform_count as usize).checked_sub(1) {
            self.raw_transforms[last].inputs_change = inputs_change;
            self.raw_transforms[last].lighting_change = lighting_change;
        }
    }

    /// Adds a raw transform; its `Draw::transform` number. `None` if the
    /// table is full (the run goes the old way).
    pub fn push_raw_transform(&mut self, transform: RawTransform) -> Option<u32> {
        let count = *self.raw_transform_count as usize;
        if count == MAX_RAW_TRANSFORMS {
            return None;
        }
        self.raw_transforms[count] = transform;
        *self.raw_transform_count += 1;
        Some(count as u32 + 1)
    }

    /// The `RawTransform::camera` index for `camera`: one of the newest
    /// entries if it is the same (by bits; MTX switches between a few texture
    /// scales: 45 entries per race-start frame with only the newest checked),
    /// else a new one. `None` if the table is full.
    pub fn add_raw_camera(&mut self, camera: &RawCamera) -> Option<u16> {
        // The table may change: `add_raw_camera_of`'s note no longer holds.
        self.raw_camera_last[0] = 0;
        let count = *self.raw_camera_count as usize;
        let recent = count.saturating_sub(RAW_CAMERAS_RECENT);
        if let Some(found) = (recent..count).rev().find(|&index| self.raw_cameras[index].words() == camera.words()) {
            return Some(found as u16);
        }
        if count == MAX_RAW_CAMERAS {
            return None;
        }
        self.raw_cameras[count] = *camera;
        *self.raw_camera_count += 1;
        Some(count as u16)
    }

    /// `add_raw_camera(&inputs.raw_camera())`, but a draw whose camera
    /// change number is the one of the newest lookup takes that lookup's
    /// camera without building and comparing a `RawCamera` (equal numbers,
    /// equal view, projection and registers: `TransformCache`). MTX race:
    /// ~230 lookups per frame, ~1,600 cycles each, for one or two cameras.
    pub fn add_raw_camera_of(&mut self, inputs: &crate::transform::TransformInputs) -> Option<u16> {
        let change = inputs.camera_change();
        if change != 0 && self.raw_camera_last[0] == change && !crate::bench::experiment(crate::bench::NO_CAMERA_MEMO) {
            return Some(self.raw_camera_last[1] as u16);
        }
        let number = self.add_raw_camera(&inputs.raw_camera())?;
        *self.raw_camera_last = [change, number as u32];
        Some(number)
    }

    /// `add_light_set` of the light set `work_out` gives for a draw with
    /// `inputs` the GPU lights with lighting `key` (`raw_blocks::lighting_key`),
    /// but with the lights, camera and key of the newest lookup it takes that
    /// lookup's set without working it out or comparing it (equal numbers:
    /// equal lighting registers and view, `TransformCache`). MTX race start:
    /// 88 % of the lit lookups, ~1,900 cycles each.
    pub fn add_light_set_of(
        &mut self,
        inputs: &crate::transform::TransformInputs,
        key: u32,
        work_out: impl FnOnce() -> Option<GpuLights>,
    ) -> Option<u32> {
        let numbers = [inputs.lights_change(), inputs.camera_change(), key];
        if numbers[0] != 0
            && numbers[1] != 0
            && self.light_set_last[..3] == numbers
            && !crate::bench::experiment(crate::bench::NO_LIGHTS_MEMO)
        {
            return Some(self.light_set_last[3]);
        }
        let number = self.add_light_set(&work_out()?)?;
        *self.light_set_last = [numbers[0], numbers[1], numbers[2], number];
        Some(number)
    }

    /// The light set `number` (+ 1, as `GpuTransform::light_set`) holds.
    pub fn light_set(&self, number: u32) -> Option<&GpuLights> {
        self.light_sets[..*self.light_set_count as usize].get((number as usize).checked_sub(1)?)
    }
}

impl Geometry<'_> {
    /// The `Draw::transform` value for `transform`: the last one added if
    /// it is the same (consecutive draws mostly share it), else a new
    /// entry. `None` if the table is full (the caller transforms on the
    /// CPU instead; not an overflow).
    pub fn add_transform(&mut self, transform: &GpuTransform) -> Option<u32> {
        let count = *self.transform_count as usize;
        if count > 0 && self.transforms[count - 1].words() == transform.words() {
            return Some(count as u32);
        }
        if count == self.transforms.len() {
            return None;
        }
        self.transforms[count] = *transform;
        *self.transform_count += 1;
        Some(count as u32 + 1)
    }

    pub fn vertex_count(&self) -> u32 {
        *self.vertex_count
    }

    pub fn index_count(&self) -> u32 {
        *self.index_count
    }

    /// The index words added from `first` on.
    pub fn indices_from(&self, first: u32) -> &[u16] {
        &self.indices[first as usize..*self.index_count as usize]
    }

    /// The vertices added from `start` on.
    pub fn vertices_from(&self, start: u32) -> &[FrameVertex] {
        &self.vertices[start as usize..*self.vertex_count as usize]
    }

    /// Like `finish_draw`, for a draw whose vertices are in the vertex cache
    /// (`draw_flags::CACHED_VERTICES`; its indices count from the cache's
    /// first vertex, so merging needs no shift).
    ///
    /// `window`: the first vertex of the cache's window the indices count
    /// from (`vertex_cache::window_of`), kept in `first_vertex`; draws merge
    /// only within one window.
    pub fn finish_cached_draw(&mut self, mut draw: Draw, window: u32) {
        draw.flags |= draw_flags::CACHED_VERTICES;
        draw.first_vertex = window;
        draw.vertex_count = 0;
        draw.index_count = *self.index_count - draw.first_index;
        if draw.index_count == 0 {
            return;
        }
        if let Some(previous) = mergeable(self.draws, *self.merge_floor, *self.draw_count) {
            if previous.same_state(&draw)
                && previous.first_vertex == draw.first_vertex
                && previous.first_index + previous.index_count == draw.first_index
            {
                previous.index_count += draw.index_count;
                return;
            }
        }
        if *self.draw_count as usize == self.draws.len() {
            *self.overflowed = 1;
            *self.index_count = draw.first_index;
            return;
        }
        self.draws[*self.draw_count as usize] = draw;
        *self.draw_count += 1;
    }

    /// Finishes a raw draw whose vertices stay in PSP RAM (zero copy,
    /// `draw_flags::RAW_PSP_RAM`): its index runs (added since `draw`
    /// started) count from `address`. Joined to the draw before when that
    /// one is the same but for an earlier address a whole number of
    /// `stride`s before (the next chunk of a batch, the next PRIM): its runs
    /// are counted from there instead, while the vertex numbers fit 16 bits.
    pub fn finish_raw_psp_draw(&mut self, mut draw: Draw, address: u32, stride: u32) {
        draw.flags |= draw_flags::RAW_VERTICES | draw_flags::RAW_PSP_RAM | draw_flags::INDEX_RUNS;
        draw.flags &= !draw_flags::CACHED_VERTICES;
        draw.first_vertex = address;
        draw.vertex_count = 0;
        draw.index_count = *self.index_count - draw.first_index;
        if draw.index_count == 0 {
            return;
        }
        if let Some(previous) = mergeable(self.draws, *self.merge_floor, *self.draw_count) {
            let skip = address.wrapping_sub(previous.first_vertex);
            let joinable = previous.flags & draw_flags::RAW_PSP_RAM != 0
                && previous.same_state(&draw)
                && previous.first_index + previous.index_count == draw.first_index
                && address >= previous.first_vertex
                && stride > 0
                && skip % stride == 0;
            if joinable {
                let skipped = skip / stride;
                let runs = &mut self.indices[draw.first_index as usize..(draw.first_index + draw.index_count) as usize];
                // Each run is [primitive type, vertex count, first vertex].
                let fits = runs.chunks_exact(3).all(|run| skipped + run[1] as u32 + run[2] as u32 <= 65_536);
                if fits {
                    for run in runs.chunks_exact_mut(3) {
                        run[2] += skipped as u16;
                    }
                    previous.index_count += draw.index_count;
                    return;
                }
            }
        }
        self.add_raw_draw(draw);
    }

    /// Finishes a raw draw of indexed PRIMs whose indices (added since
    /// `draw` started) are a plain triangle list counted from `place`: the
    /// byte offset of a raw arena window, or with `in_psp_ram` a PSP address
    /// (zero copy, `draw_flags::RAW_PSP_RAM`). Joined to the draw before
    /// when that one is the same, listed too and counts from the same place
    /// (the indices then go on as they are).
    pub fn finish_raw_listed_draw(&mut self, mut draw: Draw, place: u32, in_psp_ram: bool) {
        draw.flags |= draw_flags::RAW_VERTICES;
        if in_psp_ram {
            draw.flags |= draw_flags::RAW_PSP_RAM;
        }
        draw.flags &= !(draw_flags::CACHED_VERTICES | draw_flags::INDEX_RUNS);
        draw.first_vertex = place;
        draw.vertex_count = 0;
        draw.index_count = *self.index_count - draw.first_index;
        if draw.index_count == 0 {
            return;
        }
        if let Some(previous) = mergeable(self.draws, *self.merge_floor, *self.draw_count) {
            if previous.same_state(&draw)
                && previous.first_vertex == draw.first_vertex
                && previous.first_index + previous.index_count == draw.first_index
            {
                previous.index_count += draw.index_count;
                return;
            }
        }
        self.add_raw_draw(draw);
    }

    /// Adds a finished raw draw as a new draw (its vertices aren't in
    /// `Frame::vertices`); drops its indices if the draw list is full.
    fn add_raw_draw(&mut self, draw: Draw) {
        if *self.draw_count as usize == self.draws.len() {
            *self.overflowed = 1;
            *self.index_count = draw.first_index;
            return;
        }
        self.draws[*self.draw_count as usize] = draw;
        *self.draw_count += 1;
    }

    /// Like `finish_cached_draw`, for a draw of raw PSP vertices
    /// (`draw_flags::RAW_VERTICES`): `window` is the byte offset of the raw
    /// arena's window its index runs count from, kept in `first_vertex`;
    /// draws merge within one window (and one vertex format, `same_state`).
    pub fn finish_raw_draw(&mut self, mut draw: Draw, window: u32) {
        draw.flags |= draw_flags::RAW_VERTICES | draw_flags::INDEX_RUNS;
        draw.flags &= !draw_flags::CACHED_VERTICES;
        draw.first_vertex = window;
        draw.vertex_count = 0;
        draw.index_count = *self.index_count - draw.first_index;
        if draw.index_count == 0 {
            return;
        }
        if let Some(previous) = mergeable(self.draws, *self.merge_floor, *self.draw_count) {
            if previous.same_state(&draw)
                && previous.first_vertex == draw.first_vertex
                && previous.first_index + previous.index_count == draw.first_index
            {
                previous.index_count += draw.index_count;
                return;
            }
        }
        self.add_raw_draw(draw);
    }

    /// Drops the vertices from `count` on (a run that went back to the
    /// general path).
    pub fn truncate_vertices(&mut self, count: u32) {
        *self.vertex_count = count.min(*self.vertex_count);
    }

    /// Adds a vertex to the draw that starts at `first_vertex` and returns
    /// its number within that draw. `None` (and `overflowed` set) if the
    /// frame or the draw is full.
    #[inline]
    pub fn add_vertex(&mut self, first_vertex: u32, vertex: FrameVertex) -> Option<u16> {
        let number = *self.vertex_count - first_vertex;
        if *self.vertex_count as usize == self.vertices.len() || number >= MAX_DRAW_VERTICES {
            *self.overflowed = 1;
            return None;
        }
        self.vertices[*self.vertex_count as usize] = vertex;
        *self.vertex_count += 1;
        Some(number as u16)
    }
    /// Adds a triangle corner: vertex `number` of the current draw. `None`
    /// (and `overflowed` set) if the index list is full.
    #[inline]
    pub fn add_index(&mut self, number: u16) -> Option<()> {
        if *self.index_count as usize == self.indices.len() {
            *self.overflowed = 1;
            return None;
        }
        self.indices[*self.index_count as usize] = number;
        *self.index_count += 1;
        Some(())
    }
    /// Room for `count` more triangle corners, to be filled in by the
    /// caller (all of them). `None` (and `overflowed` set) if the index
    /// list is full. One check for a whole draw instead of one per index.
    #[inline]
    pub fn reserve_indices(&mut self, count: usize) -> Option<&mut [u16]> {
        let start = *self.index_count as usize;
        if start + count > self.indices.len() {
            *self.overflowed = 1;
            return None;
        }
        *self.index_count += count as u32;
        Some(&mut self.indices[start..start + count])
    }
    /// Adds a vertex used once, as the next triangle corner (rectangles,
    /// 2D and triangles cut at the near plane).
    #[inline]
    pub fn add_corner(&mut self, first_vertex: u32, vertex: FrameVertex) -> Option<()> {
        let number = self.add_vertex(first_vertex, vertex)?;
        self.add_index(number)
    }
    /// Drops everything added since `draw` started (an unreadable or
    /// overflowing draw).
    pub fn undo_draw(&mut self, draw: &Draw) {
        *self.vertex_count = draw.first_vertex;
        *self.index_count = draw.first_index;
    }
    /// Adds the vertices and indices added since `draw.first_vertex` /
    /// `draw.first_index` as a draw, merging it into the previous draw when
    /// the state is identical, everything follows on directly and the
    /// merged draw stays within `MAX_DRAW_VERTICES`. Fewer, larger draws
    /// are much cheaper on the GPU.
    pub fn finish_draw(&mut self, mut draw: Draw) {
        draw.vertex_count = *self.vertex_count - draw.first_vertex;
        draw.index_count = *self.index_count - draw.first_index;
        if draw.index_count == 0 {
            self.undo_draw(&draw);
            return;
        }

        if let Some(previous) = mergeable(self.draws, *self.merge_floor, *self.draw_count) {
            if previous.same_state(&draw)
                && previous.first_vertex + previous.vertex_count == draw.first_vertex
                && previous.first_index + previous.index_count == draw.first_index
                && previous.vertex_count + draw.vertex_count <= MAX_DRAW_VERTICES
            {
                // The new indices count from the new draw's first vertex;
                // from now on they count from the previous draw's.
                let shift = previous.vertex_count as u16;
                let added = &mut self.indices[draw.first_index as usize..*self.index_count as usize];
                if draw.flags & draw_flags::INDEX_RUNS != 0 {
                    for run in added.chunks_exact_mut(3) {
                        run[2] += shift;
                    }
                } else {
                    for index in added {
                        *index += shift;
                    }
                }
                previous.vertex_count += draw.vertex_count;
                previous.index_count += draw.index_count;
                return;
            }
        }

        if *self.draw_count as usize == self.draws.len() {
            *self.overflowed = 1;
            self.undo_draw(&draw);
            return;
        }
        self.draws[*self.draw_count as usize] = draw;
        *self.draw_count += 1;
    }
}

/// The triangles of an index run (`draw_flags::INDEX_RUNS`) of `count`
/// vertices: lists (3 per triangle), strips and fans.
pub fn run_triangles(primitive_type: u16, count: u16) -> u32 {
    match primitive_type as u32 {
        crate::command::primitive::TRIANGLES => count as u32 / 3,
        // Raw sprites (`draw_flags::RAW_SPRITES`): `count` sprites, two each.
        crate::command::primitive::RECTANGLES => 2 * count as u32,
        _ => (count as u32).saturating_sub(2),
    }
}

/// An index run's triangle corners (`[primitive type, count, first vertex]`),
/// in the GE's order: lists 3t, 3t+1, 3t+2; strips t, t+1, t+2; fans 0,
/// t+1, t+2 (the order the recorder wrote indices in before runs).
pub fn run_indices(run: &[u16]) -> impl Iterator<Item = u16> {
    let (kind, base) = (run[0] as u32, run[2]);
    (0..run_triangles(run[0], run[1]) as u16).flat_map(move |t| {
        let corners = match kind {
            crate::command::primitive::TRIANGLES => [3 * t, 3 * t + 1, 3 * t + 2],
            // Sprite t / 2's corners 0 1 2, then 1 3 2 (the CPU path's order).
            crate::command::primitive::RECTANGLES if t % 2 == 0 => [2 * t, 2 * t + 1, 2 * t + 2],
            crate::command::primitive::RECTANGLES => [2 * t - 1, 2 * t + 1, 2 * t],
            crate::command::primitive::TRIANGLE_STRIP => strip_triangle(t as u32).map(|corner| corner as u16),
            _ => [0, t + 1, t + 2], // TRIANGLE_FAN
        };
        corners.map(|corner| base + corner)
    })
}

/// A frame's textures and palettes while recording (`Frame::split`).
pub struct Catalog<'a> {
    texture_count: &'a mut u32,
    palette_word_count: &'a mut u32,
    overflowed: &'a mut u32,
    last_texture: &'a mut u32,
    dither_matrix: &'a mut u64,
    textures: &'a mut [TextureDescriptor; MAX_TEXTURES],
    texture_lookup: &'a mut [u16; TEXTURE_LOOKUP_SLOTS],
    palette_count: &'a mut u32,
    palettes: &'a mut [FramePalette; MAX_FRAME_PALETTES],
    palette_slots: &'a mut [u16; PALETTE_SLOTS],
    texture_levels: &'a mut [TextureLevels; MAX_TEXTURES],
    /// `Frame::texture_samples` (`MAX_TEXTURES` entries), by pointer: the
    /// copier writes the entries it was handed (`texture_sample_place`)
    /// while the catalog is in use.
    texture_samples: *mut u32,
    texture_sampled: &'a mut [u32; MAX_TEXTURES / 32],
}

/// A palette's place in a frame (`Catalog::add_palette`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PalettePlace {
    /// Its index in `Frame::palettes`.
    pub index: u32,
    /// Where its words go in `Frame::palette_words`.
    pub offset: u32,
    pub word_count: u32,
}

impl Catalog<'_> {
    /// Notes the GE's dither matrix for the frame (`Frame::dither_matrix`).
    pub fn set_dither_matrix(&mut self, matrix: u64) {
        *self.dither_matrix = matrix;
    }

    pub fn textures(&self) -> &[TextureDescriptor] {
        &self.textures[..*self.texture_count as usize]
    }

    /// The mip levels of the texture at `index` (one with
    /// `texture_flags::LEVELS_SHIFT`).
    /// True if texture `index`'s sample was taken in this frame or handed
    /// to the copier (`texture_sample_place`); also for an index out of
    /// range (nothing to sample).
    pub fn is_sampled(&self, index: u32) -> bool {
        let index = index as usize;
        index >= MAX_TEXTURES || self.texture_sampled[index / 32] & (1 << (index % 32)) != 0
    }

    /// Takes back `texture_sample_place`'s note for a sample that wasn't
    /// handed to the copier after all.
    pub fn unmark_sampled(&mut self, index: u32) {
        let index = index as usize;
        if index < MAX_TEXTURES {
            self.texture_sampled[index / 32] &= !(1 << (index % 32));
        }
    }

    /// Stores texture `index`'s sample, taken now (`texture::quick_sample`).
    pub fn set_texture_sample(&mut self, index: u32, sample: u32) {
        if let Some(entry) = self.texture_sample_place(index) {
            // Safe: the entry is in the frame and this texture's alone.
            unsafe { entry.write(sample) };
        }
    }

    /// Texture `index`'s `Frame::texture_samples` entry, noted as sampled:
    /// for a sample task (`copy_tasks::Layout::Sample`), whose copier writes
    /// it. Nobody else writes it until the texture is added again in a later
    /// frame (`add_texture`), when every task is done. `None` out of range.
    pub fn texture_sample_place(&mut self, index: u32) -> Option<*mut u32> {
        let index = index as usize;
        if index >= MAX_TEXTURES {
            return None;
        }
        self.texture_sampled[index / 32] |= 1 << (index % 32);
        // Safe: `texture_samples` points to `MAX_TEXTURES` entries.
        Some(unsafe { self.texture_samples.add(index) })
    }

    pub fn set_texture_levels(&mut self, index: u32, levels: &TextureLevels) {
        if let Some(entry) = self.texture_levels.get_mut(index as usize) {
            *entry = *levels;
        }
    }

    /// Returns the index of an identical texture already in this frame, or
    /// adds it. `None` if the texture table is full.
    pub fn add_texture(&mut self, texture: TextureDescriptor) -> Option<u32> {
        if self.textures().get(*self.last_texture as usize) == Some(&texture) {
            return Some(*self.last_texture);
        }
        let mut slot = texture_slot(&texture);
        loop {
            match self.texture_lookup[slot] {
                0 => break,
                entry => {
                    let index = entry as u32 - 1;
                    if self.textures[index as usize] == texture {
                        *self.last_texture = index;
                        return Some(index);
                    }
                }
            }
            slot = (slot + 1) % TEXTURE_LOOKUP_SLOTS;
        }
        if *self.texture_count as usize == MAX_TEXTURES {
            *self.overflowed = 1;
            return None;
        }
        self.textures[*self.texture_count as usize] = texture;
        // Not sampled yet (`recorder::sample_texture`). Safe: the index is
        // below MAX_TEXTURES (checked above), and no task of this frame has
        // its entry yet.
        let index = *self.texture_count as usize;
        unsafe { self.texture_samples.add(index).write(0) };
        self.texture_sampled[index / 32] &= !(1 << (index % 32));
        *self.texture_count += 1;
        self.texture_lookup[slot] = *self.texture_count as u16;
        *self.last_texture = *self.texture_count - 1;
        Some(*self.last_texture)
    }
    /// The place of the palette of `word_count` words loaded from physical
    /// `address` (0: words restored from a saved state): the one it got
    /// earlier in this frame, or a new one (`palette_count` grows by one:
    /// its words are still to be stored, `Geometry::fill_palette`). Only
    /// the place is reserved here; nothing is read. `None` if the frame's
    /// palette table or words are full.
    pub fn add_palette(&mut self, address: u32, word_count: u32) -> Option<PalettePlace> {
        let mut slot = palette_slot(address, word_count);
        loop {
            match self.palette_slots[slot] {
                0 => break,
                entry => {
                    let index = entry as u32 - 1;
                    let known = self.palettes[index as usize];
                    if known.address == address && known.word_count == word_count {
                        return Some(PalettePlace { index, offset: known.offset, word_count });
                    }
                }
            }
            slot = (slot + 1) % PALETTE_SLOTS;
        }
        let index = *self.palette_count;
        let offset = *self.palette_word_count;
        if index as usize == MAX_FRAME_PALETTES || (offset + word_count) as usize > MAX_PALETTE_WORDS {
            *self.overflowed = 1;
            return None;
        }
        self.palettes[index as usize] = FramePalette { address, word_count, offset };
        *self.palette_count += 1;
        *self.palette_word_count += word_count;
        self.palette_slots[slot] = *self.palette_count as u16;
        Some(PalettePlace { index, offset, word_count })
    }

    /// Palettes placed in this frame so far (`add_palette`).
    pub fn palette_count(&self) -> u32 {
        *self.palette_count
    }
}

/// Where `Frame::palette_slots` starts looking for a palette.
fn palette_slot(address: u32, word_count: u32) -> usize {
    let mixed = (address ^ word_count.rotate_left(24)).wrapping_mul(0x9E37_79B1);
    (mixed >> 20) as usize % PALETTE_SLOTS
}

/// Where `Frame::texture_lookup` starts looking for a texture.
fn texture_slot(texture: &TextureDescriptor) -> usize {
    let mixed = texture.address.wrapping_mul(0x9E37_79B1)
        ^ texture.palette_hash.wrapping_mul(0x85EB_CA6B)
        ^ texture.clut_format.rotate_left(13)
        ^ (texture.format as u32) << 27
        ^ (texture.width as u32) << 16
        ^ texture.height as u32;
    (mixed.wrapping_mul(0x9E37_79B1) >> 22) as usize % TEXTURE_LOOKUP_SLOTS
}

/// No frame drawn for this long: the display's render targets may hold
/// pictures of before. A savestate load stops the game for seconds
/// ("Please wait..."), and its frame numbers go on as before (same capture
/// session), so this is how the display notices one. After a plain pause
/// Sony's picture is the same frame, so re-seeding then does no harm.
pub const TARGETS_OUTDATED_AFTER_US: u64 = 1_000_000;

/// The display's render targets hold pictures of before (see
/// `TARGETS_OUTDATED_AFTER_US`): the frame numbers started again (a new
/// capture session) or no frame was drawn for a second or more.
pub fn targets_outdated(last_drawn: u32, frame_number: u32, idle_us: u64) -> bool {
    last_drawn != 0 && (frame_number < last_drawn || idle_us >= TARGETS_OUTDATED_AFTER_US)
}

#[cfg(test)]
pub fn new_boxed_frame() -> Box<Frame> {
    // Frame is ~2.2 MiB; build it on the heap directly.
    let layout = std::alloc::Layout::new::<Frame>();
    // Safe: all fields are plain integers/floats, for which zero is valid.
    unsafe { Box::from_raw(std::alloc::alloc_zeroed(layout) as *mut Frame) }
}

#[cfg(test)]
mod tests {
    #[test]
    fn strip_triangles_keep_one_winding() {
        use super::{run_indices, strip_triangle};
        assert_eq!([0, 1, 2, 3].map(strip_triangle), [[0, 1, 2], [2, 1, 3], [2, 3, 4], [4, 3, 5]]);
        let run = [crate::command::primitive::TRIANGLE_STRIP as u16, 5, 10];
        assert_eq!(run_indices(&run).collect::<Vec<_>>(), [10, 11, 12, 12, 11, 13, 12, 13, 14]);
    }

    use super::*;

    #[test]
    fn a_faint_additive_pass_adds_nothing_to_a_16_bit_target() {
        // MHF's guild hall (2026-11-17): the hearth glow is drawn as many
        // faint additive passes into a 5551 target. On the PSP each adds a
        // few levels that the cut to 5 bits drops; in an 8-bit target they
        // added up to a grey haze over the room.
        let floor = quantize_like_the_ge(40, 0, 5);
        assert_eq!(floor, 41); // 5 << 3 | 5 >> 2
        let mut psp = floor;
        let mut ours = floor;
        for _ in 0..4 {
            psp = quantize_like_the_ge(psp as i32 + 3, 0, 5);
            ours += 3;
        }
        assert_eq!(psp, 41);
        assert_eq!(ours, 53);
        // A bigger step still gets through.
        assert_eq!(quantize_like_the_ge(41 + 12, 0, 5), 49);
    }

    #[test]
    fn widening_keeps_black_and_white() {
        for bits in [4, 5, 6] {
            assert_eq!(quantize_like_the_ge(255, 0, bits), 255, "{bits} bits");
            assert_eq!(quantize_like_the_ge(0, 0, bits), 0, "{bits} bits");
            assert_eq!(quantize_like_the_ge(300, 7, bits), 255, "{bits} bits, clamped");
            assert_eq!(quantize_like_the_ge(2, -8, bits), 0, "{bits} bits, clamped");
        }
        assert_eq!([quantize_like_the_ge(127, 0, 1), quantize_like_the_ge(128, 0, 1)], [0, 255]);
        // Every widened value cuts back to itself (no drift across passes).
        for bits in [4, 5, 6] {
            for value in 0..256 {
                let once = quantize_like_the_ge(value, 0, bits);
                assert_eq!(quantize_like_the_ge(once as i32, 0, bits), once);
            }
        }
    }

    #[test]
    fn the_fast_cut_is_the_ges_cut() {
        for value in 0..=255u8 {
            assert_eq!(cut_to_5_bits(value) as u32, quantize_like_the_ge(value as i32, 0, 5), "{value}");
        }
    }

    #[test]
    fn draws_into_5551_get_fast_programs_for_common_blends() {
        use crate::blend::{BLEND_ALPHA_ONLY, BLEND_ENABLED, BLEND_FIX_B_ALPHA, BLEND_WRITES_ALPHA};
        let target = |format: u32| format << draw_flags::TARGET_FORMAT_SHIFT;
        let normal = BLEND_ENABLED | 4 | 5 << 4;
        let add = BLEND_ENABLED | 4 | 1 << 4;
        let fast = |way, keep_alpha| ReadVariant::Fast { way, keep_alpha, dither: false };
        let dithered = |way, keep_alpha| ReadVariant::Fast { way, keep_alpha, dither: true };
        let f5551 = target(draw_flags::TARGET_5551);
        // Nobody reads the alpha: every draw writes it.
        assert_eq!(read_variant(f5551, normal, false, true), fast(ReadWay::Normal, false));
        assert_eq!(read_variant(f5551, add, false, true), fast(ReadWay::Add, false));
        assert_eq!(read_variant(f5551, 0, false, true), fast(ReadWay::Off, false));
        // Normal alpha blends without the frame read: GXM over the cut source.
        assert_eq!(read_variant(f5551, normal, false, false), fast(ReadWay::Precut, false));
        assert_eq!(read_variant(f5551, add, false, false), fast(ReadWay::Add, false));
        // The alpha is read: kept by draws that don't write it...
        assert_eq!(read_variant(f5551, normal, true, true), fast(ReadWay::Normal, true));
        assert_eq!(read_variant(f5551, 0, true, true), fast(ReadWay::Off, true));
        // ...and cut to one bit by the generic program where they do.
        assert_eq!(read_variant(f5551, normal | BLEND_WRITES_ALPHA, true, true), ReadVariant::Generic);
        // Dithered: the fast ones with the dither; normal blends read the
        // frame even without `exact_normal` (the dither follows the blend).
        let f5551_dither = f5551 | draw_flags::DITHER;
        assert_eq!(read_variant(f5551_dither, normal, false, true), dithered(ReadWay::Normal, false));
        assert_eq!(read_variant(f5551_dither, normal, false, false), dithered(ReadWay::Normal, false));
        assert_eq!(read_variant(f5551_dither, add, true, true), dithered(ReadWay::Add, true));
        assert_eq!(read_variant(f5551_dither, 0, false, true), dithered(ReadWay::Off, false));
        // Stencil-only clears, other factors and equations, BLEND_FIX and
        // other formats: generic.
        assert_eq!(read_variant(f5551, BLEND_WRITES_ALPHA | BLEND_ALPHA_ONLY, false, true), ReadVariant::Generic);
        assert_eq!(read_variant(f5551, BLEND_ENABLED | 1 | 1 << 4, false, true), ReadVariant::Generic);
        assert_eq!(read_variant(f5551, normal | 1 << 8, false, true), ReadVariant::Generic);
        assert_eq!(read_variant(f5551, normal | BLEND_FIX_B_ALPHA, false, true), ReadVariant::Generic);
        // Without blending the FIX bit means nothing.
        assert_eq!(read_variant(f5551, BLEND_FIX_B_ALPHA, false, true), fast(ReadWay::Off, false));
        assert_eq!(read_variant(target(draw_flags::TARGET_565), normal, false, true), ReadVariant::Generic);
        assert_eq!(read_variant(target(draw_flags::TARGET_4444), normal, false, true), ReadVariant::Generic);
    }

    #[test]
    fn dither_entries_are_signed_nibbles() {
        // MHF's matrix: DITH0..3 = 000f0f, 00f0f1, 000f0f, 0000f1.
        let matrix = 0x0f0f | 0xf0f1 << 16 | 0x0f0f << 32 | 0x00f1 << 48;
        let rows: Vec<Vec<i32>> = (0..4).map(|y| (0..4).map(|x| dither_value(matrix, x, y)).collect()).collect();
        assert_eq!(rows, [[-1, 0, -1, 0], [1, -1, 0, -1], [-1, 0, -1, 0], [1, -1, 0, 0]]);
        assert_eq!(dither_value(matrix, 4, 5), dither_value(matrix, 0, 1), "the matrix repeats");
        assert_eq!(dither_value(0x7 | 0x8 << 4, 1, 0), -8);
        assert_eq!(dither_value(0x7, 0, 0), 7);
    }

    /// MTX's ghost pictures and dark blur (2026-11-10): its motion blur keeps
    /// 95 % of the other framebuffer each frame, so whatever our target for
    /// it held before (the picture before a savestate load, black after the
    /// renderer was switched on) faded only over seconds. The display seeds
    /// such targets from Sony's picture; these say when.
    #[test]
    fn a_frame_sampling_the_other_framebuffer_and_outdated_targets_are_seen() {
        let mut frame = new_boxed_frame();
        frame.textures[0] = TextureDescriptor { address: 0x0408_8000, format: 3, width: 512, height: 512, ..Default::default() };
        frame.textures[1] = TextureDescriptor { address: 0x0400_0000, format: 5, width: 512, height: 512, ..Default::default() };
        frame.texture_count = 2;
        assert!(frame.samples_framebuffer(0x0408_8000));
        assert!(!frame.samples_framebuffer(0x0400_0000), "read as palette indices: not its colours");
        assert!(!frame.samples_framebuffer(0x0411_0000));
        assert!(!frame.samples_framebuffer(0));
        assert!(targets_outdated(8904, 12, 33_000), "frame numbers started again");
        assert!(targets_outdated(8904, 8905, 9_000_000), "seconds without a frame (savestate load)");
        assert!(!targets_outdated(8904, 8905, 33_000));
        assert!(!targets_outdated(8904, 8904, 16_000), "the same frame drawn again");
        assert!(!targets_outdated(0, 12, 9_000_000), "nothing drawn yet");
    }

    /// Adds a triangle (corners with x = 0, 1, 2) as a draw with `texture`.
    fn add_triangle(frame: &mut Frame, texture: u32) {
        let draw = Draw {
            first_vertex: frame.vertex_count,
            first_index: frame.index_count,
            texture,
            flags: draw_flags::TEXTURED,
            ..Draw::default()
        };
        for x in 0..3 {
            if frame.add_corner(draw.first_vertex, FrameVertex { x: x as f32, ..FrameVertex::default() }).is_none() {
                frame.undo_draw(&draw);
                return;
            }
        }
        frame.finish_draw(draw);
    }

    #[test]
    fn identical_consecutive_draws_are_merged() {
        let mut frame = new_boxed_frame();
        add_triangle(&mut frame, 0);
        add_triangle(&mut frame, 0);
        assert_eq!(frame.draws().len(), 1);
        assert_eq!(frame.draws()[0].vertex_count, 6);
        assert_eq!(frame.draws()[0].index_count, 6);
        // The second triangle's indices now count from the merged draw's start.
        assert_eq!(frame.indices(), &[0, 1, 2, 3, 4, 5]);
        let xs: Vec<f32> = frame.draw_corners(&frame.draws()[0]).map(|corner| corner.x).collect();
        assert_eq!(xs, [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]);

        add_triangle(&mut frame, 1);
        assert_eq!(frame.draws().len(), 2, "different texture starts a new draw");
        assert_eq!(frame.draws()[1].first_index, 6);
        assert_eq!(&frame.indices()[6..], &[0, 1, 2], "a new draw counts from its own first vertex");
    }

    /// Zero copy: runs left in PSP RAM join the draw before when they lie a
    /// whole number of vertices after its address; their runs then count
    /// from that address.
    #[test]
    fn psp_ram_runs_join_when_whole_vertices_apart() {
        let mut frame = new_boxed_frame();
        let stride = 12;
        let add_run = |frame: &mut Frame, address: u32, count: u16| {
            let draw = Draw { first_index: frame.index_count, ..Draw::default() };
            let mut geometry = frame.split().0;
            geometry.reserve_indices(3).unwrap().copy_from_slice(&[4, count, 0]);
            geometry.finish_raw_psp_draw(draw, address, stride);
        };
        add_run(&mut frame, 0x0890_0000, 10);
        add_run(&mut frame, 0x0890_0000 + 10 * stride, 6);
        assert_eq!(frame.draws().len(), 1, "the next run of the batch joins");
        assert_eq!(&frame.indices()[3..6], &[4, 6, 10], "counted from the first run's address");
        add_run(&mut frame, 0x0890_0000 + 10 * stride + 2, 4);
        assert_eq!(frame.draws().len(), 2, "not a whole vertex apart: a new draw");
        assert_eq!(frame.draws()[1].first_vertex, 0x0890_0000 + 10 * stride + 2);
        add_run(&mut frame, 0x0890_0000 + 10 * stride + 2 + 70_000 * stride, 4);
        assert_eq!(frame.draws().len(), 3, "beyond 16-bit vertex numbers: a new draw");
        assert!(frame.draws().iter().all(|draw| draw.flags & draw_flags::RAW_PSP_RAM != 0));
        assert_eq!(frame.screen_corners(&frame.draws()[0]).count(), 0, "the CPU never sees them");
    }

    /// The index ring holds each run draw's corners as the display would
    /// build them (`draw_indices`, the CPU's reference), for every run type.
    #[test]
    fn ring_indices_are_the_draws_corners() {
        use crate::command::primitive;
        let mut frame = new_boxed_frame();
        let runs: [u16; 12] = [
            primitive::TRIANGLE_STRIP as u16, 6, 0,
            primitive::TRIANGLES as u16, 6, 10,
            primitive::TRIANGLE_FAN as u16, 5, 20,
            primitive::RECTANGLES as u16, 2, 40,
        ];
        frame.indices[..12].copy_from_slice(&runs);
        frame.indices[12..15].copy_from_slice(&[0, 1, 2]);
        frame.draws[0] = Draw { first_index: 0, index_count: 12, flags: draw_flags::INDEX_RUNS, ..Draw::default() };
        frame.draws[1] = Draw { first_index: 12, index_count: 3, vertex_count: 3, ..Draw::default() };
        frame.draw_count = 2;
        frame.index_count = 15;
        let bytes: &'static mut [u8] =
            Box::leak(vec![0u8; 2 * crate::raw_arena::WINDOW_BYTES as usize].into_boxed_slice());
        let mut ring = crate::raw_arena::RawArena::new(bytes);
        ring.put(&[7; 6], 2).unwrap(); // not at the ring's start
        assert_eq!(frame.write_ring_indices(&mut ring), 0);
        let expected: Vec<u16> = frame.draw_indices(&frame.draws[0]).collect();
        assert_eq!(expected.len(), 3 * (4 + 2 + 3 + 4));
        let at = frame.draw_ring_index[0] as usize;
        let written: Vec<u16> = ring.bytes()[2 * at..2 * (at + expected.len())]
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect();
        assert_eq!(written, expected);
        assert_eq!(frame.draw_ring_index[1], NO_RING_INDICES, "CPU path: not in the ring");
        // A raw draw's plain indices (converted indexed PRIMs) are copied.
        frame.draws[1].flags = draw_flags::RAW_VERTICES;
        assert_eq!(frame.write_ring_indices(&mut ring), 0);
        let at = frame.draw_ring_index[1] as usize;
        assert_ne!(at, NO_RING_INDICES as usize);
        let plain: Vec<u16> = ring.bytes()[2 * at..2 * (at + 3)].chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
        assert_eq!(plain, [0, 1, 2]);
        assert_eq!(frame.ring_indices, ring.bytes().as_ptr() as *const u16);
    }

    #[test]
    fn a_draw_can_use_every_vertex_of_the_frame() {
        // 16-bit indices reach vertex 65535: one draw can fill the frame.
        let mut frame = new_boxed_frame();
        let draw = Draw::default();
        for _ in 0..MAX_VERTICES {
            frame.add_corner(0, FrameVertex::default()).unwrap();
        }
        frame.finish_draw(draw);
        assert_eq!(frame.draws()[0].vertex_count, MAX_DRAW_VERTICES);
        assert_eq!(frame.indices()[MAX_VERTICES - 1], 65535);
        assert!(!frame.is_overflowed());
    }

    #[test]
    fn a_full_frame_drops_the_draw() {
        let mut frame = new_boxed_frame();
        frame.vertex_count = MAX_VERTICES as u32 - 2;
        frame.index_count = 30;
        add_triangle(&mut frame, 0);
        assert!(frame.is_overflowed());
        assert_eq!(frame.draws().len(), 0);
        assert_eq!((frame.vertex_count, frame.index_count), (MAX_VERTICES as u32 - 2, 30), "nothing of it kept");
    }

    #[test]
    fn a_palette_loaded_again_gets_its_place_again() {
        let mut frame = new_boxed_frame();
        let red = frame.add_palette(0x0900_0000, 16).unwrap();
        let blue = frame.add_palette(0x0900_0040, 16).unwrap();
        assert_eq!((red.index, red.offset, blue.index, blue.offset), (0, 0, 1, 16));
        assert_eq!(frame.add_palette(0x0900_0000, 16), Some(red), "not only the last palette is found");
        assert_eq!(frame.add_palette(0x0900_0040, 16), Some(blue));
        // Another length at the same address is another palette.
        let longer = frame.add_palette(0x0900_0000, 32).unwrap();
        assert_eq!((longer.index, longer.offset), (2, 32));
        assert_eq!((frame.palette_count, frame.palette_word_count), (3, 64));
        frame.clear();
        assert_eq!(frame.add_palette(0x0900_0040, 16).map(|place| place.offset), Some(0), "forgotten with the frame");
    }

    #[test]
    fn a_full_palette_table_overflows_the_frame() {
        let mut frame = new_boxed_frame();
        for index in 0..MAX_FRAME_PALETTES as u32 {
            assert!(frame.add_palette(0x0900_0000 + 64 * index, 8).is_some());
        }
        assert!(!frame.is_overflowed());
        assert_eq!(frame.add_palette(0x0880_0000, 8), None);
        assert!(frame.is_overflowed());
        assert!(frame.add_palette(0x0900_0040, 8).is_some(), "the ones placed are still found");
    }

    /// The words are stored where the palette was placed (zeros for what
    /// couldn't be read), and at the frame's end its textures carry the
    /// words' hash instead of the palette's index.
    #[test]
    fn palette_textures_get_the_hash_of_their_words_at_the_end() {
        let mut frame = new_boxed_frame();
        let first = frame.add_palette(0x0900_0000, 4).unwrap();
        let second = frame.add_palette(0x0900_0100, 4).unwrap();
        frame.fill_palette(second, &[5, 6]);
        frame.fill_palette(first, &[1, 2, 3, 4]);
        assert_eq!(frame.palette_words[..8], [1, 2, 3, 4, 5, 6, 0, 0]);
        let clut8 = |palette: u32| TextureDescriptor { address: 0x0910_0000, format: 5, palette_hash: palette, ..Default::default() };
        let direct = TextureDescriptor { address: 0x0920_0000, format: 3, palette_hash: 1, ..Default::default() };
        for texture in [clut8(1), clut8(0), direct] {
            frame.add_texture(texture).unwrap();
        }
        frame.resolve_palette_hashes();
        let hashes: Vec<u32> = frame.textures().iter().map(|texture| texture.palette_hash).collect();
        let hash = crate::texture::hash_words;
        assert_eq!(hashes, [hash(&[5, 6, 0, 0]), hash(&[1, 2, 3, 4]), 1], "direct colour textures keep theirs");
    }

    #[test]
    fn textures_are_found_again_among_many() {
        let mut frame = new_boxed_frame();
        let texture = |n: u32| TextureDescriptor { address: 0x0890_0000 + 0x400 * n, width: 64, height: 32, ..Default::default() };
        for n in 0..MAX_TEXTURES as u32 {
            assert_eq!(frame.add_texture(texture(n)), Some(n));
        }
        for n in (0..MAX_TEXTURES as u32).rev() {
            assert_eq!(frame.add_texture(texture(n)), Some(n), "texture {n} found again");
        }
        assert_eq!(frame.add_texture(texture(MAX_TEXTURES as u32)), None, "table full");
        frame.clear();
        assert_eq!(frame.add_texture(texture(7)), Some(0), "forgotten with the frame");
    }

    #[test]
    fn same_texture_is_stored_once() {
        let mut frame = new_boxed_frame();
        let texture = TextureDescriptor { address: 0x0890_0000, width: 16, height: 16, ..Default::default() };
        assert_eq!(frame.add_texture(texture), Some(0));
        assert_eq!(frame.add_texture(texture), Some(0));
        assert_eq!(frame.textures().len(), 1);
    }
}
