//! The GE's register state.
//!
//! Most GE commands simply store their 24-bit argument in a register named by
//! the opcode, and that value stays until the next command with the same
//! opcode, across lists. Like PPSSPP's `gstate.cmdmem`, we keep the last
//! argument of every opcode and decode fields on demand.

use crate::command::{op, Command};
use crate::event_ring::{
    GE_STATE_BONES, GE_STATE_PROJECTION, GE_STATE_REGISTERS, GE_STATE_TEXGEN, GE_STATE_VIEW, GE_STATE_WORDS, GE_STATE_WORLD,
};
use crate::memory::physical;
use crate::snapshot::{Reader, Writer};
use crate::vertex::{IndexFormat, VertexFormat, VertexLayout};

#[derive(Clone)]
pub struct GeState {
    /// Last argument seen for each opcode.
    registers: [u32; 256],
    /// Added to relative addresses. Set by OFFSET_ADDRESS and ORIGIN, and
    /// saved/restored by CALL/RET; not a plain register.
    pub offset: u32,
    /// Where the next draw reads vertices / indices. Set by VERTEX_ADDRESS
    /// and INDEX_ADDRESS (resolved with the base and offset at that moment)
    /// and moved forward by every draw, see `advance_after_draw`.
    pub vertex_address: u32,
    pub index_address: u32,
    /// The layout of VERTEX_TYPE, worked out when the register is written
    /// rather than for every draw (it showed up in the capture's profile).
    vertex_layout: VertexLayout,
    /// Counts changes to the registers a draw's state is worked out from
    /// (`DRAW_STATE_REGISTERS`), so the recorder can tell that a draw has the
    /// same state as the one before and reuse what it worked out for it.
    state_changes: u32,
    /// The 3D transform matrices (see `transform.rs`).
    pub matrices: Matrices,
    /// Count changes to the world/view/projection matrices and to the bone
    /// matrices, so a draw's transform is only worked out again when they
    /// changed (`transform::TransformCache`). An upload of the value already
    /// there is no change (like PPSSPP's Execute_WorldMtxData). Not saved.
    matrix_generation: u32,
    bone_generation: u32,
    /// Counts changes to the other registers the 3D transform reads
    /// (`TRANSFORM_REGISTERS`). Not saved.
    transform_changes: u32,
    /// Counts changes to the lighting registers (`is_lighting_register`),
    /// which `lighting::Lights` is worked out from. Not saved.
    lighting_changes: u32,
}

/// The registers `lighting::Lights` reads: LIGHTING_ENABLE and the four
/// LIGHT_ENABLEs (0x17..=0x1B), REVERSE_NORMAL (0x51), the material colours
/// (0x53..=0x58), and the specular coefficient, the ambient light, the light
/// mode and every light's registers (0x5B..=0x9A).
pub const fn is_lighting_register(opcode: u8) -> bool {
    matches!(opcode, 0x17..=0x1B | 0x51 | 0x53..=0x58 | 0x5B..=0x9A)
}

/// Registers (besides the matrices and VERTEX_TYPE) that `transform::Transform`
/// is worked out from.
pub const TRANSFORM_REGISTERS: [u8; 16] = [
    op::FOG_ENABLE,
    op::FOG_END,
    op::FOG_SCALE,
    op::CLEAR_MODE,
    op::VIEWPORT_X_SCALE,
    op::VIEWPORT_Y_SCALE,
    op::VIEWPORT_Z_SCALE,
    op::VIEWPORT_X_CENTER,
    op::VIEWPORT_Y_CENTER,
    op::VIEWPORT_Z_CENTER,
    op::SCREEN_OFFSET_X,
    op::SCREEN_OFFSET_Y,
    op::TEXTURE_SCALE_U,
    op::TEXTURE_SCALE_V,
    op::TEXTURE_OFFSET_U,
    op::TEXTURE_OFFSET_V,
];

/// What a command's register means for the recorder, looked up once per
/// command (like PPSSPP's per-command flag table, GPUCommonHW.cpp
/// commonCommandTable): `set_register` runs for every command walked.
mod command_flags {
    /// A draw-state input (`DRAW_STATE_REGISTERS`, `GeState::state_changes`).
    pub const DRAW_STATE: u8 = 1 << 0;
    /// An input of the 3D transform (`GeState::transform_changes`).
    pub const TRANSFORM: u8 = 1 << 1;
    /// A matrix upload (NUMBER or DATA), handled by `Matrices::upload`.
    pub const MATRIX: u8 = 1 << 2;
    /// A lighting input (`is_lighting_register`, `GeState::lighting_changes`).
    pub const LIGHTING: u8 = 1 << 3;
}

const COMMAND_FLAGS: [u8; 256] = command_flag_table();

/// `GeState::upload_matrix_run` sets a DATA register without the state
/// counters: none may count in them.
const MATRIX_DATA_FLAGS_ARE_MATRIX_ONLY: () = {
    let data = [op::WORLD_MATRIX_DATA, op::VIEW_MATRIX_DATA, op::PROJECTION_MATRIX_DATA, op::BONE_MATRIX_DATA];
    let mut index = 0;
    while index < data.len() {
        assert!(COMMAND_FLAGS[data[index] as usize] == command_flags::MATRIX);
        index += 1;
    }
};
#[allow(clippy::let_unit_value)]
const _: () = MATRIX_DATA_FLAGS_ARE_MATRIX_ONLY;

const fn command_flag_table() -> [u8; 256] {
    let mut flags = [0u8; 256];
    let mut index = 0;
    while index < DRAW_STATE_REGISTERS.len() {
        flags[DRAW_STATE_REGISTERS[index] as usize] |= command_flags::DRAW_STATE;
        index += 1;
    }
    let mut index = 0;
    while index < TRANSFORM_REGISTERS.len() {
        flags[TRANSFORM_REGISTERS[index] as usize] |= command_flags::TRANSFORM;
        index += 1;
    }
    let mut opcode = 0;
    while opcode < 256 {
        if is_lighting_register(opcode as u8) {
            flags[opcode] |= command_flags::LIGHTING;
        }
        opcode += 1;
    }
    let matrix_commands = [
        op::WORLD_MATRIX_NUMBER,
        op::WORLD_MATRIX_DATA,
        op::VIEW_MATRIX_NUMBER,
        op::VIEW_MATRIX_DATA,
        op::PROJECTION_MATRIX_NUMBER,
        op::PROJECTION_MATRIX_DATA,
        op::BONE_MATRIX_NUMBER,
        op::BONE_MATRIX_DATA,
    ];
    let mut index = 0;
    while index < matrix_commands.len() {
        flags[matrix_commands[index] as usize] |= command_flags::MATRIX;
        index += 1;
    }
    flags
}

/// Registers `GeState::take_hardware_state` leaves alone: commands that do
/// something rather than hold a value (drawing, list flow, CLUT load,
/// texture flush/sync, block transfer), the address commands whose effect
/// isn't the register (offset, origin; the vertex and index addresses move
/// on with every draw), the matrix commands (the matrices come whole) and
/// the unused top range.
fn keeps_own_register(opcode: u8) -> bool {
    matches!(
        opcode,
        0x00..=0x0F
            | op::OFFSET_ADDRESS
            | op::ORIGIN
            | op::BONE_MATRIX_NUMBER
            | op::BONE_MATRIX_DATA
            | 0x3A..=0x41
            | op::CLUT_LOAD
            | 0xCB // texture flush
            | 0xCC // texture sync
            | op::TRANSFER_START
            | 0xF0..=0xFF
    )
}

/// Writes the 24-bit floats `words` into `matrix`; returns how many
/// elements changed.
fn take_matrix(matrix: &mut [f32], words: &[u32]) -> u32 {
    let mut changed = 0;
    for (element, &word) in matrix.iter_mut().zip(words) {
        let value = float24(word & 0x00FF_FFFF);
        changed += (element.to_bits() != value.to_bits()) as u32;
        *element = value;
    }
    changed
}

/// Number of skinning bone matrices.
pub const BONE_COUNT: usize = 8;

/// The world, view, projection and bone matrices, as uploaded element by
/// element (`op::WORLD_MATRIX_NUMBER` and friends). Column by column: the
/// world, view and bone matrices have 4 columns of 3 rows, the projection 4
/// of 4. The bones are 8 such matrices back to back (element 12 is the first
/// of bone 1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Matrices {
    pub world: [f32; 12],
    pub view: [f32; 12],
    pub projection: [f32; 16],
    pub bones: [f32; 12 * BONE_COUNT],
    /// The element the next DATA command of each matrix writes.
    world_next: u32,
    view_next: u32,
    projection_next: u32,
    bone_next: u32,
}

impl Matrices {
    pub const fn new() -> Matrices {
        Matrices {
            world: [0.0; 12],
            view: [0.0; 12],
            projection: [0.0; 16],
            bones: [0.0; 12 * BONE_COUNT],
            world_next: 0,
            view_next: 0,
            projection_next: 0,
            bone_next: 0,
        }
    }

    /// Bone `index` (0..8), 4 columns of 3 rows.
    pub fn bone(&self, index: usize) -> &[f32; 12] {
        self.bones[index * 12..index * 12 + 12].try_into().unwrap()
    }

    /// Handles a matrix upload command: which matrix a DATA command wrote
    /// to, if any.
    fn upload(&mut self, command: Command) -> Uploaded {
        // Elements past the end are ignored; the position keeps counting
        // (PPSSPP's Execute_WorldMtxData). True if the element changed:
        // games upload the same matrices again and again (MTX Mototrax:
        // ~130 world matrices per frame), and an unchanged one needs no new
        // transform (PPSSPP only flushes on a changed word, GPUCommon.cpp).
        fn write(matrix: &mut [f32], next: &mut u32, data: u32) -> bool {
            let mut changed = false;
            if let Some(element) = matrix.get_mut(*next as usize) {
                let value = float24(data);
                changed = element.to_bits() != value.to_bits();
                *element = value;
            }
            *next = (*next + 1).min(matrix.len() as u32);
            changed
        }
        let data = command.data();
        let (changed, uploaded) = match command.opcode() {
            op::WORLD_MATRIX_NUMBER => {
                self.world_next = data & 0xF;
                return Uploaded::Nothing;
            }
            op::VIEW_MATRIX_NUMBER => {
                self.view_next = data & 0xF;
                return Uploaded::Nothing;
            }
            op::PROJECTION_MATRIX_NUMBER => {
                self.projection_next = data & 0xF;
                return Uploaded::Nothing;
            }
            op::BONE_MATRIX_NUMBER => {
                self.bone_next = data & 0x7F;
                return Uploaded::Nothing;
            }
            op::WORLD_MATRIX_DATA => (write(&mut self.world, &mut self.world_next, data), Uploaded::Matrix),
            op::VIEW_MATRIX_DATA => (write(&mut self.view, &mut self.view_next, data), Uploaded::Matrix),
            op::PROJECTION_MATRIX_DATA => (write(&mut self.projection, &mut self.projection_next, data), Uploaded::Matrix),
            op::BONE_MATRIX_DATA => (write(&mut self.bones, &mut self.bone_next, data), Uploaded::Bone),
            _ => return Uploaded::Nothing,
        };
        if changed {
            uploaded
        } else {
            Uploaded::Nothing
        }
    }

    /// The DATA commands at the start of `words` with the first one's
    /// opcode, written as `upload` writes them one by one. Returns how many
    /// it took and how many of them changed their element.
    fn upload_run(&mut self, words: &[u32]) -> (usize, u32) {
        let opcode = Command(words[0]).opcode();
        let (matrix, next): (&mut [f32], &mut u32) = match opcode {
            op::WORLD_MATRIX_DATA => (&mut self.world, &mut self.world_next),
            op::VIEW_MATRIX_DATA => (&mut self.view, &mut self.view_next),
            op::PROJECTION_MATRIX_DATA => (&mut self.projection, &mut self.projection_next),
            op::BONE_MATRIX_DATA => (&mut self.bones, &mut self.bone_next),
            _ => return (0, 0),
        };
        // The position in a local, written back once: through `next` the
        // compiler reloaded and stored it for every element (it may alias
        // the matrix). Elements compared as bits: `word << 8` is the float24
        // value (`float24`), the opcode shifted out.
        let mut position = *next as usize;
        let mut changed = 0;
        let mut taken = 0;
        for &word in words {
            if Command(word).opcode() != opcode {
                break;
            }
            if let Some(element) = matrix.get_mut(position) {
                let value = word << 8;
                changed += (element.to_bits() != value) as u32;
                *element = f32::from_bits(value);
                position += 1;
            } else {
                // Past the end (NUMBER may point there): stays at the end,
                // as `upload` does.
                position = matrix.len();
            }
            taken += 1;
        }
        *next = position as u32;
        (taken, changed)
    }

    fn save(&self, out: &mut Writer) {
        for value in self.world.iter().chain(&self.view).chain(&self.projection) {
            out.u32(value.to_bits());
        }
        out.u32(self.world_next);
        out.u32(self.view_next);
        out.u32(self.projection_next);
        for value in &self.bones {
            out.u32(value.to_bits());
        }
        out.u32(self.bone_next);
    }

    /// `version` of the snapshot: bones since version 3 (older dumps keep
    /// them at 0 until the frame uploads its own, which skinned games do
    /// before every skinned draw).
    fn load(&mut self, input: &mut Reader, version: u32) -> Option<()> {
        for value in self.world.iter_mut().chain(self.view.iter_mut()).chain(self.projection.iter_mut()) {
            *value = f32::from_bits(input.u32()?);
        }
        self.world_next = input.u32()?;
        self.view_next = input.u32()?;
        self.projection_next = input.u32()?;
        if version >= 3 {
            for value in self.bones.iter_mut() {
                *value = f32::from_bits(input.u32()?);
            }
            self.bone_next = input.u32()?;
        }
        Some(())
    }
}

/// What `Matrices::upload` changed.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Uploaded {
    Nothing,
    Matrix,
    Bone,
}

/// The GE's 24-bit floats: the top 24 bits of an IEEE single.
pub fn float24(data: u32) -> f32 {
    f32::from_bits(data << 8)
}

/// The registers a draw's state (`recorder::work_out_draw_state`: render
/// target, tests, scissor, texture, blending) is worked out from: a change
/// to any of them counts in `GeState::state_changes`, so the recorder works
/// the state out again. Registers it doesn't read are left out, so a change
/// of, say, the cull direction or the lighting between two draws doesn't
/// make the recorder redo it (MTX's race: half of all state misses changed
/// only such registers). What a draw reads every time (the texture
/// function's other bits, fog, lighting, the vertex format) needn't be
/// listed. The loaded palette is checked apart (`DrawStateCache::fits`).
/// The test `registers_left_out_of_the_draw_state_do_not_change_it` (in
/// recorder.rs) checks that the list is complete.
pub const DRAW_STATE_REGISTERS: [u8; 60] = [
    // Render target.
    op::FRAMEBUFFER_POINTER,
    op::FRAMEBUFFER_WIDTH,
    op::COLOR_MASK,
    op::CLEAR_MODE,
    // Whether the target's alpha (the PSP's stencil) is written
    // (`blend::alpha_write`).
    op::FRAMEBUFFER_PIXEL_FORMAT,
    op::ALPHA_MASK,
    // 16-bit targets: dithering and its matrix (`draw_flags::DITHER`).
    op::DITHER_ENABLE,
    op::DITHER_ROW_0,
    op::DITHER_ROW_0 + 1,
    op::DITHER_ROW_0 + 2,
    op::DITHER_ROW_0 + 3,
    op::STENCIL_OP,
    // Per-pixel tests and scissor.
    op::ALPHA_TEST_ENABLE,
    op::ALPHA_TEST,
    op::DEPTH_TEST_ENABLE,
    op::DEPTH_TEST,
    op::STENCIL_TEST_ENABLE,
    op::LOGIC_OP_ENABLE,
    op::COLOR_TEST_ENABLE,
    op::DEPTH_WRITE_DISABLE,
    op::SCISSOR_1,
    op::SCISSOR_2,
    // Back-face culling.
    op::CULL_FACE_ENABLE,
    op::CULL,
    // Texture (level 0).
    op::TEXTURE_MAP_ENABLE,
    op::TEXTURE_ADDRESS_0,
    op::TEXTURE_BUFFER_WIDTH_0,
    op::TEXTURE_SIZE_0,
    op::TEXTURE_FORMAT,
    op::TEXTURE_MODE,
    op::TEXTURE_FUNCTION,
    op::CLUT_FORMAT,
    op::TEXTURE_FILTER,
    op::TEXTURE_WRAP,
    // Texture mip levels 1..7 and the level mode (`texture::describe_levels`).
    op::TEXTURE_ADDRESS_0 + 1,
    op::TEXTURE_ADDRESS_0 + 2,
    op::TEXTURE_ADDRESS_0 + 3,
    op::TEXTURE_ADDRESS_0 + 4,
    op::TEXTURE_ADDRESS_0 + 5,
    op::TEXTURE_ADDRESS_0 + 6,
    op::TEXTURE_ADDRESS_0 + 7,
    op::TEXTURE_BUFFER_WIDTH_0 + 1,
    op::TEXTURE_BUFFER_WIDTH_0 + 2,
    op::TEXTURE_BUFFER_WIDTH_0 + 3,
    op::TEXTURE_BUFFER_WIDTH_0 + 4,
    op::TEXTURE_BUFFER_WIDTH_0 + 5,
    op::TEXTURE_BUFFER_WIDTH_0 + 6,
    op::TEXTURE_BUFFER_WIDTH_0 + 7,
    op::TEXTURE_SIZE_0 + 1,
    op::TEXTURE_SIZE_0 + 2,
    op::TEXTURE_SIZE_0 + 3,
    op::TEXTURE_SIZE_0 + 4,
    op::TEXTURE_SIZE_0 + 5,
    op::TEXTURE_SIZE_0 + 6,
    op::TEXTURE_SIZE_0 + 7,
    op::TEXTURE_LEVEL,
    // Blending.
    op::ALPHA_BLEND_ENABLE,
    op::BLEND_MODE,
    op::BLEND_FIXED_A,
    op::BLEND_FIXED_B,
];

/// `VertexFormat(0).layout()`, for `GeState::new` (which is `const`).
/// A test checks that they match.
const LAYOUT_OF_FORMAT_0: VertexLayout =
    VertexLayout { size: 3, texcoord_offset: None, color_offset: None, normal_offset: None, position_offset: 0 };

impl GeState {
    pub const fn new() -> GeState {
        // The texture scale starts at 1.0, as sceGuInit sets it: games set
        // it once at start-up, and when we start watching later (a savestate
        // loaded) 0 made every 3D texture coordinate 0 (Final Fantasy's
        // world map tiles became flat colours).
        let mut registers = [0; 256];
        registers[op::TEXTURE_SCALE_U as usize] = 0x3F_8000;
        registers[op::TEXTURE_SCALE_V as usize] = 0x3F_8000;
        GeState {
            registers,
            offset: 0,
            vertex_address: 0,
            index_address: 0,
            vertex_layout: LAYOUT_OF_FORMAT_0,
            state_changes: 0,
            matrices: Matrices::new(),
            matrix_generation: 0,
            bone_generation: 0,
            transform_changes: 0,
            lighting_changes: 0,
        }
    }

    /// Takes over a saved context's registers, matrices and addresses
    /// (`ge_context`); the change counters move on, so everything worked
    /// out from the state before is worked out again.
    pub fn restore_context(&mut self, saved: &GeState) {
        self.registers = saved.registers;
        self.offset = saved.offset;
        self.vertex_address = saved.vertex_address;
        self.index_address = saved.index_address;
        self.vertex_layout = saved.vertex_layout;
        self.matrices.clone_from(&saved.matrices);
        self.state_changes = self.state_changes.wrapping_add(1);
        self.matrix_generation = self.matrix_generation.wrapping_add(1);
        self.bone_generation = self.bone_generation.wrapping_add(1);
        self.transform_changes = self.transform_changes.wrapping_add(1);
        self.lighting_changes = self.lighting_changes.wrapping_add(1);
    }

    /// Takes over the GE's own registers and matrices (the plugin's
    /// `EVENT_GE_STATE` block, `event_ring::GE_STATE_*` layout). Registers
    /// that act rather than hold a value (`keeps_own_register`) stay as
    /// they are. Returns how many registers and matrix elements differed
    /// from ours: 0 and 0 when our walk had the GE's state (the check
    /// `gestate` makes in a running game).
    pub fn take_hardware_state(&mut self, block: &[u32; GE_STATE_WORDS]) -> (u32, u32) {
        let mut register_differences = 0;
        for opcode in 0..=255u8 {
            if keeps_own_register(opcode) {
                continue;
            }
            let data = block[GE_STATE_REGISTERS + opcode as usize] & 0x00FF_FFFF;
            register_differences += (self.registers[opcode as usize] != data) as u32;
            self.set_register(Command(((opcode as u32) << 24) | data));
        }
        let matrices = &mut self.matrices;
        let matrix_differences = take_matrix(&mut matrices.bones, &block[GE_STATE_BONES..GE_STATE_WORLD])
            + take_matrix(&mut matrices.world, &block[GE_STATE_WORLD..GE_STATE_VIEW])
            + take_matrix(&mut matrices.view, &block[GE_STATE_VIEW..GE_STATE_PROJECTION])
            + take_matrix(&mut matrices.projection, &block[GE_STATE_PROJECTION..GE_STATE_TEXGEN]);
        // The game sends a NUMBER command before its DATA commands.
        matrices.world_next = 0;
        matrices.view_next = 0;
        matrices.projection_next = 0;
        matrices.bone_next = 0;
        self.matrix_generation = self.matrix_generation.wrapping_add(1);
        self.bone_generation = self.bone_generation.wrapping_add(1);
        (register_differences, matrix_differences)
    }

    /// Changes whenever the world, view or projection matrix is uploaded to.
    pub fn matrix_generation(&self) -> u32 {
        self.matrix_generation
    }

    /// Changes whenever a bone matrix is uploaded to.
    pub fn bone_generation(&self) -> u32 {
        self.bone_generation
    }

    /// Changes whenever one of `TRANSFORM_REGISTERS` changes.
    pub fn transform_changes(&self) -> u32 {
        self.transform_changes
    }

    /// Changes whenever a lighting register (`is_lighting_register`) changes.
    pub fn lighting_changes(&self) -> u32 {
        self.lighting_changes
    }

    pub fn vertex_format(&self) -> VertexFormat {
        VertexFormat(self.register(op::VERTEX_TYPE))
    }

    pub fn vertex_layout(&self) -> &VertexLayout {
        &self.vertex_layout
    }

    /// Changes whenever a register that affects how draws look changes.
    pub fn state_changes(&self) -> u32 {
        self.state_changes
    }

    /// After a draw of `count` vertices the GE leaves its read position
    /// just past the data it used, so games can issue several draws from
    /// one buffer without setting the address again.
    pub fn advance_after_draw(&mut self, count: u32) {
        let format = self.vertex_format();
        match format.index_format() {
            IndexFormat::None => self.vertex_address = self.vertex_address.wrapping_add(count * self.vertex_layout.size),
            index => self.index_address = self.index_address.wrapping_add(count * index.size()),
        }
    }

    pub fn is_clear_mode(&self) -> bool {
        self.register(op::CLEAR_MODE) & 1 != 0
    }

    /// In clear mode: whether colour is written (otherwise only depth/stencil).
    pub fn clear_writes_color(&self) -> bool {
        self.register(op::CLEAR_MODE) & (1 << 8) != 0
    }

    /// In clear mode: whether the stencil, the framebuffer's alpha, is
    /// written (with the vertices' alpha).
    pub fn clear_writes_stencil(&self) -> bool {
        self.register(op::CLEAR_MODE) & (1 << 9) != 0
    }

    /// In clear mode: whether depth is written (the vertices' z).
    pub fn clear_writes_depth(&self) -> bool {
        self.register(op::CLEAR_MODE) & (1 << 10) != 0
    }

    pub fn is_blend_enabled(&self) -> bool {
        self.register(op::ALPHA_BLEND_ENABLE) & 1 != 0
    }

    /// The colour of vertices that have none: material ambient + alpha.
    pub fn material_color(&self) -> u32 {
        (self.register(op::MATERIAL_ALPHA) & 0xFF) << 24 | (self.register(op::MATERIAL_AMBIENT) & 0x00FF_FFFF)
    }

    /// Scissor rectangle (left, top, right, bottom), inclusive pixels.
    pub fn scissor(&self) -> (u16, u16, u16, u16) {
        let first = self.register(op::SCISSOR_1);
        let second = self.register(op::SCISSOR_2);
        (
            (first & 0x3FF) as u16,
            ((first >> 10) & 0x3FF) as u16,
            (second & 0x3FF) as u16,
            ((second >> 10) & 0x3FF) as u16,
        )
    }

    pub fn clut_address(&self) -> u32 {
        let low = self.register(op::CLUT_ADDRESS) & 0x00FF_FFF0;
        let high = (self.register(op::CLUT_ADDRESS_UPPER) << 8) & 0x0F00_0000;
        physical(low | high)
    }

    pub fn register(&self, opcode: u8) -> u32 {
        self.registers[opcode as usize]
    }

    /// All registers (for the capture's stage A, `walk_stream`).
    pub fn registers(&self) -> &[u32; 256] {
        &self.registers
    }

    /// Whether `set_register` with the value the register already holds
    /// changes nothing: true for all but the matrix commands (an upload
    /// moves on to the next element even with the same value).
    pub fn same_value_is_no_change(opcode: u8) -> bool {
        COMMAND_FLAGS[opcode as usize] & command_flags::MATRIX == 0
    }

    #[inline]
    pub fn set_register(&mut self, command: Command) {
        let flags = COMMAND_FLAGS[command.opcode() as usize];
        // Before the "unchanged" shortcut below: uploading the same value
        // twice still moves on to the next element.
        if flags & command_flags::MATRIX != 0 {
            match self.matrices.upload(command) {
                Uploaded::Nothing => {}
                Uploaded::Matrix => self.matrix_generation = self.matrix_generation.wrapping_add(1),
                Uploaded::Bone => self.bone_generation = self.bone_generation.wrapping_add(1),
            }
        }
        let register = &mut self.registers[command.opcode() as usize];
        if *register == command.data() {
            return;
        }
        *register = command.data();
        if flags & command_flags::DRAW_STATE != 0 {
            self.state_changes = self.state_changes.wrapping_add(1);
        }
        if flags & command_flags::TRANSFORM != 0 {
            self.transform_changes = self.transform_changes.wrapping_add(1);
        }
        if flags & command_flags::LIGHTING != 0 {
            self.lighting_changes = self.lighting_changes.wrapping_add(1);
        }
        if command.opcode() == op::VERTEX_TYPE {
            self.vertex_layout = self.vertex_format().layout();
        }
    }

    /// `set_register` for the plain commands of a list run
    /// (`display_list::run_plain_commands`), with the same results but
    /// without data-dependent branches for the common case: the counters
    /// are added to, not branched around. About 36,000 such commands per
    /// frame in MTX's race; on the Vita's Cortex-A9 each mispredicted
    /// branch costs ~10 cycles, and the plain runs took 9 ms (sampling
    /// profile, 2026-11-05).
    #[inline(always)]
    pub fn set_plain_register(&mut self, command: Command) {
        let opcode = command.opcode();
        let flags = COMMAND_FLAGS[opcode as usize];
        if flags & command_flags::MATRIX != 0 || opcode == op::VERTEX_TYPE {
            self.set_register(command);
            return;
        }
        let data = command.data();
        let register = &mut self.registers[opcode as usize];
        let changed = (*register != data) as u32;
        *register = data;
        self.state_changes = self.state_changes.wrapping_add(changed & (flags & command_flags::DRAW_STATE) as u32);
        self.transform_changes =
            self.transform_changes.wrapping_add(changed & ((flags & command_flags::TRANSFORM) >> 1) as u32);
        self.lighting_changes = self.lighting_changes.wrapping_add(changed & ((flags & command_flags::LIGHTING) >> 3) as u32);
    }

    /// Runs the matrix DATA commands at the start of `words` that share the
    /// first one's opcode, with the same result as `set_register` for each:
    /// one loop instead of a call and a lookup per command (~13,000 such
    /// commands per frame in MTX's races). Returns how many it ran (>= 1
    /// for a DATA command).
    pub fn upload_matrix_run(&mut self, words: &[u32]) -> usize {
        let (taken, changed) = self.matrices.upload_run(words);
        if taken == 0 {
            return 0;
        }
        let opcode = Command(words[0]).opcode();
        // One step per changed element, as `set_register` would count.
        if opcode == op::BONE_MATRIX_DATA {
            self.bone_generation = self.bone_generation.wrapping_add(changed);
        } else {
            self.matrix_generation = self.matrix_generation.wrapping_add(changed);
        }
        // The register as the commands one by one would leave it: the last
        // one's data. DATA registers count in neither state counter
        // (`MATRIX_DATA_FLAGS_ARE_MATRIX_ONLY`).
        self.registers[opcode as usize] = Command(words[taken - 1]).data();
        taken
    }

    /// What VERTEX_ADDRESS, INDEX_ADDRESS and OFFSET_ADDRESS do besides
    /// setting their register (`walk`'s main loop and the plain runs both
    /// call this, then `set_register`). Other opcodes: nothing.
    #[inline]
    pub fn apply_address(&mut self, command: Command) {
        match command.opcode() {
            op::OFFSET_ADDRESS => self.offset = command.data() << 8,
            op::VERTEX_ADDRESS => self.vertex_address = self.relative_address(command.data()),
            op::INDEX_ADDRESS => self.index_address = self.relative_address(command.data()),
            _ => {}
        }
    }

    /// Turns the 24-bit argument of JUMP, CALL, VERTEX_ADDRESS, ... into a
    /// full address: BASE supplies bits 24..28, then the offset is added.
    pub fn relative_address(&self, data: u32) -> u32 {
        let base_bits = (self.register(op::BASE) & 0x000F_0000) << 8;
        physical((base_bits | data).wrapping_add(self.offset))
    }

    /// Vertices are already in screen coordinates (used for 2D).
    pub fn is_through_mode(&self) -> bool {
        self.register(op::VERTEX_TYPE) & (1 << 23) != 0
    }

    pub fn is_texture_enabled(&self) -> bool {
        self.register(op::TEXTURE_MAP_ENABLE) & 1 != 0
    }

    /// TEXTURE_FORMAT value, an index into `command::TEXTURE_FORMAT_NAMES`.
    pub fn texture_format(&self) -> u32 {
        self.register(op::TEXTURE_FORMAT) & 0xF
    }

    /// Level 0's texture address. The low 4 bits don't count (16-byte
    /// aligned, as PPSSPP's GPUgstate::getTextureAddress).
    pub fn texture_address(&self) -> u32 {
        let low = self.register(op::TEXTURE_ADDRESS_0) & 0x00FF_FFF0;
        let high = self.register(op::TEXTURE_BUFFER_WIDTH_0) & 0x000F_0000;
        physical(low | (high << 8))
    }

    /// Where draws currently go. Render targets live in VRAM, so the
    /// address is reported as a VRAM address (0x04xxxxxx).
    pub fn framebuffer_address(&self) -> u32 {
        let low = self.register(op::FRAMEBUFFER_POINTER) & 0x00FF_FFFF;
        let high = self.register(op::FRAMEBUFFER_WIDTH) & 0x00FF_0000;
        physical(low | (high << 8)) | 0x0400_0000
    }

    /// Render target stride in pixels.
    pub fn framebuffer_stride(&self) -> u32 {
        self.register(op::FRAMEBUFFER_WIDTH) & 0x7FC
    }

    pub fn framebuffer_pixel_format(&self) -> u32 {
        self.register(op::FRAMEBUFFER_PIXEL_FORMAT) & 0x3
    }
}

impl GeState {
    pub fn save(&self, out: &mut Writer) {
        for register in self.registers {
            out.u32(register);
        }
        out.u32(self.offset);
        out.u32(self.vertex_address);
        out.u32(self.index_address);
        self.matrices.save(out);
    }

    /// `version`: of the snapshot (`SNAPSHOT_VERSION`); version 1 had no
    /// matrices (they stay as they are).
    pub fn load(&mut self, input: &mut Reader, version: u32) -> Option<()> {
        for register in self.registers.iter_mut() {
            *register = input.u32()?;
        }
        self.offset = input.u32()?;
        self.vertex_address = input.u32()?;
        self.index_address = input.u32()?;
        if version >= 2 {
            self.matrices.load(input, version)?;
        }
        self.vertex_layout = self.vertex_format().layout();
        self.state_changes = self.state_changes.wrapping_add(1);
        self.matrix_generation = self.matrix_generation.wrapping_add(1);
        self.bone_generation = self.bone_generation.wrapping_add(1);
        self.transform_changes = self.transform_changes.wrapping_add(1);
        self.lighting_changes = self.lighting_changes.wrapping_add(1);
        Some(())
    }
}

impl Default for GeState {
    fn default() -> Self {
        GeState::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Part of the crash of 2026-11-06 (see memory::place_read): the texture
    /// address had low bits set (0x048066E1); the GE ignores them.
    #[test]
    fn crash_20261106_texture_address_ignores_the_low_four_bits() {
        let mut ge = GeState::new();
        ge.set_register(Command(((op::TEXTURE_ADDRESS_0 as u32) << 24) | 0x80_66E1));
        ge.set_register(Command(((op::TEXTURE_BUFFER_WIDTH_0 as u32) << 24) | 0x04_0040));
        assert_eq!(ge.texture_address(), 0x0480_66E0);
    }

    #[test]
    fn plain_registers_are_set_as_set_register_sets_them() {
        let (mut slow, mut fast) = (GeState::new(), GeState::new());
        // Every opcode, each twice with the same value (unchanged the
        // second time) and once with another.
        for round in 0..3u32 {
            for opcode in 0..=255u32 {
                let data = if round == 2 { opcode * 0x1357 + 1 } else { opcode * 0x0101 } & 0xFF_FFFF;
                let command = Command((opcode << 24) | data);
                slow.set_register(command);
                fast.set_plain_register(command);
                assert_eq!(fast.register(opcode as u8), slow.register(opcode as u8), "opcode {opcode:02x}");
                assert_eq!(fast.state_changes(), slow.state_changes(), "opcode {opcode:02x}");
                assert_eq!(fast.transform_changes(), slow.transform_changes(), "opcode {opcode:02x}");
                assert_eq!(fast.lighting_changes(), slow.lighting_changes(), "opcode {opcode:02x}");
                assert_eq!(fast.matrix_generation, slow.matrix_generation);
                assert_eq!(fast.bone_generation, slow.bone_generation);
                assert_eq!(*fast.vertex_layout(), *slow.vertex_layout());
            }
        }
    }

    /// `upload_matrix_run` leaves the state as `set_register` one command
    /// at a time does: matrices, positions (also past a matrix's end),
    /// registers, generations and state counters.
    #[test]
    fn matrix_runs_upload_as_the_commands_one_by_one() {
        let (mut slow, mut fast) = (GeState::new(), GeState::new());
        let mut seed = 0x2468_ACE1u32;
        let mut random = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        let numbers = [op::WORLD_MATRIX_NUMBER, op::VIEW_MATRIX_NUMBER, op::PROJECTION_MATRIX_NUMBER, op::BONE_MATRIX_NUMBER];
        let data = [op::WORLD_MATRIX_DATA, op::VIEW_MATRIX_DATA, op::PROJECTION_MATRIX_DATA, op::BONE_MATRIX_DATA];
        for _ in 0..2000 {
            let kind = (random() % 4) as usize;
            let start = cmd(numbers[kind], random() % 20);
            slow.set_register(start);
            fast.set_register(start);
            // A run of DATA commands (some values repeat, some runs pass the
            // matrix's end), then an unrelated command.
            let length = 1 + (random() % 20) as usize;
            let mut words: Vec<u32> = (0..length).map(|_| cmd(data[kind], random() % 4 * 0x10_0000).0).collect();
            words.push(cmd(op::VERTEX_TYPE, random() & 0xFF_FFFF).0);
            for &word in &words[..length] {
                slow.set_register(Command(word));
            }
            assert_eq!(fast.upload_matrix_run(&words), length);
            assert_eq!(fast.matrices, slow.matrices);
            assert_eq!(fast.registers, slow.registers);
            assert_eq!((fast.matrix_generation, fast.bone_generation), (slow.matrix_generation, slow.bone_generation));
            assert_eq!(
                (fast.state_changes, fast.transform_changes, fast.lighting_changes),
                (slow.state_changes, slow.transform_changes, slow.lighting_changes)
            );
        }
    }

    #[test]
    fn initial_layout_matches_format_0() {
        assert_eq!(*GeState::new().vertex_layout(), VertexFormat(0).layout());
    }

    fn cmd(opcode: u8, data: u32) -> Command {
        Command(((opcode as u32) << 24) | data)
    }

    #[test]
    fn uploading_the_same_matrix_again_is_no_change() {
        let mut ge = GeState::new();
        let upload = |ge: &mut GeState, value: u32| {
            ge.set_register(cmd(op::WORLD_MATRIX_NUMBER, 0));
            for _ in 0..12 {
                ge.set_register(cmd(op::WORLD_MATRIX_DATA, value));
            }
        };
        upload(&mut ge, 0x3F_8000);
        let generation = ge.matrix_generation();
        upload(&mut ge, 0x3F_8000);
        assert_eq!(ge.matrix_generation(), generation, "same values");
        upload(&mut ge, 0x40_0000);
        assert_ne!(ge.matrix_generation(), generation, "new values");
        assert_eq!(ge.matrices.world[11], 2.0);
    }

    #[test]
    fn transform_registers_are_counted_apart_from_the_draw_state() {
        let mut ge = GeState::new();
        let (state, transform) = (ge.state_changes(), ge.transform_changes());
        ge.set_register(cmd(op::VIEWPORT_X_SCALE, 0x43_7000));
        assert_eq!((ge.state_changes(), ge.transform_changes()), (state, transform.wrapping_add(1)));
        // Clear mode is both: it switches the transform off and is part of
        // the draw's state.
        ge.set_register(cmd(op::CLEAR_MODE, 1));
        assert_eq!((ge.state_changes(), ge.transform_changes()), (state.wrapping_add(1), transform.wrapping_add(2)));
        // Writing the value already there changes nothing.
        ge.set_register(cmd(op::CLEAR_MODE, 1));
        assert_eq!((ge.state_changes(), ge.transform_changes()), (state.wrapping_add(1), transform.wrapping_add(2)));
        // Fog is read by every draw: a transform input only.
        ge.set_register(cmd(op::FOG_ENABLE, 1));
        assert_eq!((ge.state_changes(), ge.transform_changes()), (state.wrapping_add(1), transform.wrapping_add(3)));
    }

    #[test]
    fn layout_follows_vertex_type() {
        let mut ge = GeState::new();
        let format = (1 << 23) | 2 | (7 << 2) | (2 << 7); // 2D sprite vertex, 16 bytes
        ge.set_register(Command(((op::VERTEX_TYPE as u32) << 24) | format));
        assert_eq!(ge.vertex_layout().size, 16);

        // Restoring a saved state restores the layout too.
        let mut bytes = [0u8; 2048];
        let mut saved = Writer::new(&mut bytes);
        ge.save(&mut saved);
        assert!(!saved.overflowed());
        let mut restored = GeState::new();
        restored.load(&mut Reader::new(&bytes), crate::snapshot::SNAPSHOT_VERSION).unwrap();
        assert_eq!(restored.vertex_layout(), ge.vertex_layout());
    }

    fn command(opcode: u8, data: u32) -> Command {
        Command(((opcode as u32) << 24) | data)
    }

    #[test]
    fn matrix_uploads() {
        let mut ge = GeState::new();
        let one = 1.0f32.to_bits() >> 8;
        let two = 2.0f32.to_bits() >> 8;
        // Projection from element 14: two values, the same twice (both written).
        ge.set_register(command(op::PROJECTION_MATRIX_NUMBER, 14));
        ge.set_register(command(op::PROJECTION_MATRIX_DATA, two));
        ge.set_register(command(op::PROJECTION_MATRIX_DATA, two));
        ge.set_register(command(op::PROJECTION_MATRIX_DATA, one)); // past the end: ignored
        assert_eq!(&ge.matrices.projection[13..], &[0.0, 2.0, 2.0]);
        // World from element 0.
        ge.set_register(command(op::WORLD_MATRIX_NUMBER, 0));
        ge.set_register(command(op::WORLD_MATRIX_DATA, one));
        ge.set_register(command(op::WORLD_MATRIX_DATA, two));
        assert_eq!(&ge.matrices.world[..3], &[1.0, 2.0, 0.0]);
        assert_eq!(ge.matrices.view, [0.0; 12]);
        // Bone 1, element 2 (the 15th of all bones).
        ge.set_register(command(op::BONE_MATRIX_NUMBER, 14));
        ge.set_register(command(op::BONE_MATRIX_DATA, two));
        assert_eq!(ge.matrices.bone(1)[2], 2.0);

        // Saved and restored.
        let mut bytes = [0u8; 2048];
        let mut saved = Writer::new(&mut bytes);
        ge.save(&mut saved);
        let mut restored = GeState::new();
        restored.load(&mut Reader::new(&bytes), crate::snapshot::SNAPSHOT_VERSION).unwrap();
        assert_eq!(restored.matrices, ge.matrices);
    }
}
