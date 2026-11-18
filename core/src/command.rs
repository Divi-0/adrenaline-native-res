//! GE command words.
//!
//! Every display list entry is one 32-bit word: the top 8 bits select the
//! command (the "opcode"), the low 24 bits are its argument. Most commands
//! just set a GE register (texture address, blend mode, ...); a few control
//! the flow of the list (jump, call, end) or start work (draw, copy).
//!
//! Opcode numbers follow PPSSPP's `GPU/ge_constants.h`.

/// One display list entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Command(pub u32);

impl Command {
    pub fn opcode(self) -> u8 {
        (self.0 >> 24) as u8
    }

    /// The 24-bit argument.
    pub fn data(self) -> u32 {
        self.0 & 0x00FF_FFFF
    }
}

/// Opcodes we care about so far.
pub mod op {
    pub const NOP: u8 = 0x00;
    /// Vertex data address (relative, see `GeState::relative_address`).
    pub const VERTEX_ADDRESS: u8 = 0x01;
    /// Index data address (relative).
    pub const INDEX_ADDRESS: u8 = 0x02;
    /// Draw primitives: bits 0..16 vertex count, bits 16..19 primitive type.
    pub const PRIM: u8 = 0x04;
    pub const BEZIER: u8 = 0x05;
    pub const SPLINE: u8 = 0x06;
    /// Tests whether some vertices are on screen; result used by BJUMP.
    pub const BOUNDING_BOX: u8 = 0x07;
    pub const JUMP: u8 = 0x08;
    /// Jump if the last BOUNDING_BOX test found nothing visible.
    pub const BOUNDING_BOX_JUMP: u8 = 0x09;
    pub const CALL: u8 = 0x0A;
    pub const RET: u8 = 0x0B;
    /// Ends execution; its meaning depends on the command before it
    /// (FINISH or SIGNAL).
    pub const END: u8 = 0x0C;
    /// Raises a signal; together with the following END it can also jump,
    /// call or return (see `SignalBehaviour`).
    pub const SIGNAL: u8 = 0x0E;
    /// Marks the list as complete (followed by END).
    pub const FINISH: u8 = 0x0F;
    /// Upper address bits (24..28) for relative addresses.
    pub const BASE: u8 = 0x10;
    pub const VERTEX_TYPE: u8 = 0x12;
    /// Sets the offset added to relative addresses (argument << 8).
    pub const OFFSET_ADDRESS: u8 = 0x13;
    /// Sets the offset added to relative addresses to the current list position.
    pub const ORIGIN: u8 = 0x14;
    /// Vertex colours computed from lights (3D draws).
    pub const LIGHTING_ENABLE: u8 = 0x17;
    pub const TEXTURE_MAP_ENABLE: u8 = 0x1E;
    /// Back-face culling on (bit 0): triangles of the wrong screen winding
    /// (`CULL`) are dropped; not in clear mode, never rectangles.
    pub const CULL_FACE_ENABLE: u8 = 0x1D;
    /// Which winding culling keeps (bit 0), see `frame::draw_flags::CULL_CCW`.
    pub const CULL: u8 = 0x9B;
    /// Fog (3D draws only): per vertex f = (view-space z + FOG_END) *
    /// FOG_SCALE (24-bit floats), per pixel colour = mix(FOG_COLOR, colour,
    /// clamp(f, 0, 1)) after texturing. Names as in PPSSPP: FOG1, FOG2.
    pub const FOG_ENABLE: u8 = 0x1F;
    pub const FOG_END: u8 = 0xCD;
    pub const FOG_SCALE: u8 = 0xCE;
    /// Fog colour, RGB in bits 0..24.
    pub const FOG_COLOR: u8 = 0xCF;
    /// Matrix uploads for 3D draws: NUMBER says which element the next DATA
    /// writes, each DATA writes one element (a 24-bit float) and moves on.
    /// World and view matrices are 4x3, the projection 4x4, column by column.
    pub const WORLD_MATRIX_NUMBER: u8 = 0x3A;
    pub const WORLD_MATRIX_DATA: u8 = 0x3B;
    pub const VIEW_MATRIX_NUMBER: u8 = 0x3C;
    pub const VIEW_MATRIX_DATA: u8 = 0x3D;
    pub const PROJECTION_MATRIX_NUMBER: u8 = 0x3E;
    pub const PROJECTION_MATRIX_DATA: u8 = 0x3F;
    /// Skinning: 8 bone matrices (4x3 each, 96 elements in a row), uploaded
    /// like the others. A skinned vertex's position is the sum of its
    /// weights times each bone matrix applied to it (before the world matrix).
    pub const BONE_MATRIX_NUMBER: u8 = 0x2A;
    pub const BONE_MATRIX_DATA: u8 = 0x2B;
    /// Viewport (24-bit floats): screen = center + scale * clip / w.
    pub const VIEWPORT_X_SCALE: u8 = 0x42;
    pub const VIEWPORT_Y_SCALE: u8 = 0x43;
    pub const VIEWPORT_X_CENTER: u8 = 0x45;
    pub const VIEWPORT_Y_CENTER: u8 = 0x46;
    /// Depth: depth = Z_CENTER + Z_SCALE * clip z / w, 0..65535 (sceGu
    /// games mostly use 65535 = near, with the depth test GEQUAL).
    pub const VIEWPORT_Z_SCALE: u8 = 0x44;
    pub const VIEWPORT_Z_CENTER: u8 = 0x47;
    /// Texture coordinate scale and offset for 3D draws (24-bit floats).
    pub const TEXTURE_SCALE_U: u8 = 0x48;
    pub const TEXTURE_SCALE_V: u8 = 0x49;
    pub const TEXTURE_OFFSET_U: u8 = 0x4A;
    pub const TEXTURE_OFFSET_V: u8 = 0x4B;
    /// Subtracted from screen positions after the viewport, in 1/16 pixels.
    pub const SCREEN_OFFSET_X: u8 = 0x4C;
    pub const SCREEN_OFFSET_Y: u8 = 0x4D;
    /// How 3D draws get texture coordinates: 0 = from the vertices (scaled
    /// and offset), 1 = projected by a matrix, 2 = environment mapping.
    pub const TEXTURE_MAP_MODE: u8 = 0xC0;
    pub const ALPHA_BLEND_ENABLE: u8 = 0x21;
    pub const ALPHA_TEST_ENABLE: u8 = 0x22;
    pub const DEPTH_TEST_ENABLE: u8 = 0x23;
    pub const STENCIL_TEST_ENABLE: u8 = 0x24;
    pub const COLOR_TEST_ENABLE: u8 = 0x27;
    pub const LOGIC_OP_ENABLE: u8 = 0x28;
    /// Lighting: light n on (bit 0), n = 0..3 (`LIGHT_ENABLE_0 + n`).
    pub const LIGHT_ENABLE_0: u8 = 0x18;
    /// Lighting: which material colours come from the vertex colour
    /// (bit 0 ambient, bit 1 diffuse, bit 2 specular).
    pub const MATERIAL_UPDATE: u8 = 0x53;
    /// Lighting: the material's own light (RGB), added to the result.
    pub const MATERIAL_EMISSIVE: u8 = 0x54;
    /// Lighting: the global ambient light (RGB) and its alpha.
    pub const AMBIENT_COLOR: u8 = 0x5C;
    pub const AMBIENT_ALPHA: u8 = 0x5D;
    /// Colour used when vertices have none (bits 0..24, RGB); with lighting
    /// the material's ambient colour.
    pub const MATERIAL_AMBIENT: u8 = 0x55;
    /// Alpha used when vertices have no colour.
    pub const MATERIAL_ALPHA: u8 = 0x58;
    /// Lighting: the material's diffuse and specular colours (RGB) and the
    /// specular power (a 24-bit float).
    pub const MATERIAL_DIFFUSE: u8 = 0x56;
    pub const MATERIAL_SPECULAR: u8 = 0x57;
    pub const MATERIAL_SPECULAR_COEFFICIENT: u8 = 0x5B;
    /// Lighting: bit 0 = the specular part is a separate colour, added after
    /// texturing.
    pub const LIGHT_MODE: u8 = 0x5E;
    /// Lighting: bit 0 = normals point the other way.
    pub const REVERSE_NORMAL: u8 = 0x51;
    /// Light n's type (`LIGHT_TYPE_0 + n`): bits 0..1 computation (diffuse,
    /// diffuse and specular, powered diffuse), bits 8..9 kind (directional,
    /// point, spot).
    pub const LIGHT_TYPE_0: u8 = 0x5F;
    /// Light n's position (directional: direction to the light), three
    /// 24-bit floats from `LIGHT_X_0 + 3n`.
    pub const LIGHT_X_0: u8 = 0x63;
    /// Light n's spot direction, from `LIGHT_DIRECTION_X_0 + 3n`.
    pub const LIGHT_DIRECTION_X_0: u8 = 0x6F;
    /// Light n's constant, linear and quadratic attenuation, from
    /// `LIGHT_ATTENUATION_A_0 + 3n`.
    pub const LIGHT_ATTENUATION_A_0: u8 = 0x7B;
    /// Light n's spot exponent and cutoff (cosine), `+ n` each.
    pub const LIGHT_SPOT_EXPONENT_0: u8 = 0x87;
    pub const LIGHT_SPOT_CUTOFF_0: u8 = 0x8B;
    /// Light n's ambient, diffuse and specular colours (RGB), from
    /// `LIGHT_AMBIENT_0 + 3n`.
    pub const LIGHT_AMBIENT_0: u8 = 0x8F;
    /// Palette address, low 24 bits (bits 4..24 used).
    pub const CLUT_ADDRESS: u8 = 0xB0;
    /// Palette address bits 24..28 (in bits 16..20).
    pub const CLUT_ADDRESS_UPPER: u8 = 0xB1;
    /// Texture level 0 size: width = 1 << bits 0..4, height = 1 << bits 8..12.
    pub const TEXTURE_SIZE_0: u8 = 0xB8;
    /// Bit 0: texture is swizzled.
    pub const TEXTURE_MODE: u8 = 0xC2;
    /// Copies the palette into the GE: argument = number of 32-byte blocks.
    pub const CLUT_LOAD: u8 = 0xC4;
    /// Palette entry format (bits 0..2), index shift, mask and start.
    pub const CLUT_FORMAT: u8 = 0xC5;
    /// Bit 8: magnification is bilinear (otherwise nearest).
    pub const TEXTURE_FILTER: u8 = 0xC6;
    /// Mip level mode (bits 0..2: auto, const, slope) and offset (bits
    /// 16..24, signed, 1/16 levels).
    pub const TEXTURE_LEVEL: u8 = 0xC8;
    /// Bit 0: clamp U, bit 8: clamp V (otherwise repeat).
    pub const TEXTURE_WRAP: u8 = 0xC7;
    /// How texture and vertex colour combine (bits 0..3) and whether
    /// texture alpha is used (bit 8).
    pub const TEXTURE_FUNCTION: u8 = 0xC9;
    /// Alpha test: function (bits 0..3), reference (bits 8..16), mask (bits 16..24).
    pub const ALPHA_TEST: u8 = 0xDB;
    /// Stencil test: function (bits 0..3), reference (bits 8..16), mask (bits 16..24).
    pub const STENCIL_TEST: u8 = 0xDC;
    /// Stencil ops: on stencil fail (bits 0..8), on depth fail (bits 8..16),
    /// on pass (bits 16..24); 0 KEEP, 1 ZERO, 2 REPLACE, 3 INVERT, 4 INCR, 5 DECR.
    pub const STENCIL_OP: u8 = 0xDD;
    /// Depth compare function (bits 0..3).
    pub const DEPTH_TEST: u8 = 0xDE;
    /// Blending: source factor (bits 0..4), destination factor (bits 4..8),
    /// equation (bits 8..11).
    pub const BLEND_MODE: u8 = 0xDF;
    /// Fixed colour used by the blend factor "FIX" for source (A) and
    /// destination (B), RGB in bits 0..24.
    pub const BLEND_FIXED_A: u8 = 0xE0;
    pub const BLEND_FIXED_B: u8 = 0xE1;
    /// Bit 0: depth writes disabled.
    pub const DEPTH_WRITE_DISABLE: u8 = 0xE7;
    /// Bit 0: "clear mode", draws write colour/depth directly to clear the screen.
    pub const CLEAR_MODE: u8 = 0xD3;
    /// Scissor top-left: x bits 0..10, y bits 10..20.
    pub const SCISSOR_1: u8 = 0xD4;
    /// Scissor bottom-right (inclusive), same layout.
    pub const SCISSOR_2: u8 = 0xD5;
    /// Render target address, low 24 bits.
    pub const FRAMEBUFFER_POINTER: u8 = 0x9C;
    /// Render target stride (bits 0..11) and address bits 24..32 (bits 16..24).
    pub const FRAMEBUFFER_WIDTH: u8 = 0x9D;
    /// Texture level 0 address, low 24 bits.
    pub const TEXTURE_ADDRESS_0: u8 = 0xA0;
    /// Texture level 0 stride (bits 0..11) and address bits 24..28 (bits 16..20).
    pub const TEXTURE_BUFFER_WIDTH_0: u8 = 0xA8;
    pub const TEXTURE_FORMAT: u8 = 0xC3;
    /// Render target pixel format: 0 = 565, 1 = 5551, 2 = 4444, 3 = 8888.
    pub const FRAMEBUFFER_PIXEL_FORMAT: u8 = 0xD2;
    /// Dithering on (bit 0): 16-bit targets get the dither matrix's value
    /// added before their colours are cut to 4..6 bits.
    pub const DITHER_ENABLE: u8 = 0x20;
    /// The dither matrix's rows 0..3 (0xE2..0xE5), four signed 4-bit
    /// entries each (bits 0..16, column 0 lowest).
    pub const DITHER_ROW_0: u8 = 0xE2;
    /// Colour write mask: a set bit keeps that framebuffer bit unchanged
    /// (bits 0..24 = RGB). All set = the draw writes no visible colour.
    pub const COLOR_MASK: u8 = 0xE8;
    /// Same for the alpha channel (bits 0..8).
    pub const ALPHA_MASK: u8 = 0xE9;
    /// Starts a block transfer (memory copy done by the GE); bit 0: 32-bit
    /// pixels (else 16-bit). See `display_list::block_transfer`.
    pub const TRANSFER_START: u8 = 0xEA;
    /// Block transfer source: address bits 4..24.
    pub const TRANSFER_SOURCE: u8 = 0xB2;
    /// Source row width in pixels (bits 3..11) and address bits 24..28 (bits 16..20).
    pub const TRANSFER_SOURCE_WIDTH: u8 = 0xB3;
    /// Block transfer destination, as the source.
    pub const TRANSFER_DESTINATION: u8 = 0xB4;
    pub const TRANSFER_DESTINATION_WIDTH: u8 = 0xB5;
    /// Source x (bits 0..10) and y (bits 10..20) in pixels.
    pub const TRANSFER_SOURCE_POSITION: u8 = 0xEB;
    pub const TRANSFER_DESTINATION_POSITION: u8 = 0xEC;
    /// Width - 1 (bits 0..10) and height - 1 (bits 10..20) in pixels.
    pub const TRANSFER_SIZE: u8 = 0xEE;
}

/// What a SIGNAL + END pair does, from the SIGNAL's bits 16..24.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignalBehaviour {
    /// Call the game's signal handler, then continue (several variants).
    Handler,
    /// Memory barrier; no visible effect for us.
    Sync,
    Jump(TargetKind),
    Call(TargetKind),
    Return,
    Unknown(u8),
}

/// How a signal jump/call target address is interpreted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetKind {
    Absolute,
    /// Relative to the SIGNAL command's own address.
    RelativeToSignal,
    /// Relative to the current origin/offset, like CALL and JUMP.
    RelativeToOrigin,
}

impl SignalBehaviour {
    pub fn from_signal_command(signal: Command) -> SignalBehaviour {
        match (signal.data() >> 16) as u8 {
            0x01..=0x03 => SignalBehaviour::Handler,
            0x08 => SignalBehaviour::Sync,
            0x10 => SignalBehaviour::Jump(TargetKind::Absolute),
            0x11 => SignalBehaviour::Call(TargetKind::Absolute),
            0x12 => SignalBehaviour::Return,
            0x13 => SignalBehaviour::Jump(TargetKind::RelativeToSignal),
            0x14 => SignalBehaviour::Call(TargetKind::RelativeToSignal),
            0x15 => SignalBehaviour::Jump(TargetKind::RelativeToOrigin),
            0x16 => SignalBehaviour::Call(TargetKind::RelativeToOrigin),
            other => SignalBehaviour::Unknown(other),
        }
    }
}

/// Primitive types, PRIM bits 16..19.
pub mod primitive {
    pub const POINTS: u32 = 0;
    pub const LINES: u32 = 1;
    pub const LINE_STRIP: u32 = 2;
    pub const TRIANGLES: u32 = 3;
    pub const TRIANGLE_STRIP: u32 = 4;
    pub const TRIANGLE_FAN: u32 = 5;
    /// Axis-aligned rectangles, two vertices (corners) each.
    pub const RECTANGLES: u32 = 6;
}

/// Texture formats, values of TEXTURE_FORMAT bits 0..4.
pub const TEXTURE_FORMAT_NAMES: [&str; 11] = [
    "5650", "5551", "4444", "8888", "clut4", "clut8", "clut16", "clut32", "dxt1", "dxt3", "dxt5",
];
