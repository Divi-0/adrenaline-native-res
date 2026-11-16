//! The shader patcher all our GPU programs are made with, our two basic
//! programs and the 960x544 projection. These were vita2d's (Adrenaline's drawing library); a plugin
//! of our own can't reach vita2d, so they are ours now, the same programs:
//!
//! - colour (nativeres_color_v/_f.cg): untextured draws, and the clear of a
//!   new render target (a black triangle over the whole scene);
//! - texture-tint (nativeres_texture_v.cg, nativeres_texture_tint_f.cg):
//!   full-screen quads (render-target copies, presenting our picture, the
//!   GPU unswizzle) and textured draws when our textured program is
//!   missing.
//!
//! Their default fragment programs blend as vita2d's did (source alpha, one
//! minus source alpha); other blend modes come from `blend_fragment_program`
//! (programs.rs keeps them).
//!
//! Display thread only (made on first use there).

use core::ffi::c_void;
use core::ptr::{null, null_mut};

use crate::gxm::{self, PatcherId, ProgramParameter, VertexAttribute, VertexStream};
use crate::{heap, render_log, shaders, sys, SingleThreaded};

/// A vertex of the colour program (16 bytes): position and an RGBA colour.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ColorVertex {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub color: u32,
}

/// A vertex of the texture program (24 bytes): position (w is not read,
/// kept 1), texture coordinates.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct TextureVertex {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
    pub u: f32,
    pub v: f32,
}

/// The 960x544 projection, as vita2d made it (matrix_init_orthographic with
/// left 0, right 960, bottom 544, top 0, near 0, far 1): row-major, applied
/// as position x matrix, so the translation is the last row. Pixel (0, 0)
/// is the top left corner; z 0..1 becomes 1..-1.
pub const ORTHO: [f32; 16] = [
    2.0 / 960.0, 0.0, 0.0, 0.0, //
    0.0, -2.0 / 544.0, 0.0, 0.0, //
    0.0, 0.0, -2.0, 0.0, //
    -1.0, 1.0, 1.0, 1.0,
];

/// The shader patcher's memory, as vita2d gave its own: a buffer for the
/// programs' data and the GPU code (USSE) of vertex and fragment programs,
/// each in uncached main memory mapped for the GPU.
const PATCHER_BUFFER_BYTES: usize = 64 * 1024;
const VERTEX_USSE_BYTES: usize = 64 * 1024;
const FRAGMENT_USSE_BYTES: usize = 64 * 1024;
#[no_mangle]
#[used]
#[allow(non_upper_case_globals)]
pub static nativeres_plan_fixed_shader_patcher: u32 =
    sys::pages(PATCHER_BUFFER_BYTES) + sys::pages(VERTEX_USSE_BYTES) + sys::pages(FRAGMENT_USSE_BYTES);

/// The patcher's bookkeeping lives on our heap.
unsafe extern "C" fn host_alloc(_user_data: *mut c_void, size: u32) -> *mut c_void {
    heap::alloc(size as usize)
}

unsafe extern "C" fn host_free(_user_data: *mut c_void, memory: *mut c_void) {
    heap::free(memory)
}

/// `bytes` of uncached main memory under `name`; null if there is none.
fn usse_block(name: &'static [u8], bytes: usize) -> *mut c_void {
    // Safe: plain system calls; the block is ours alone.
    unsafe {
        let uid = sys::alloc_block(name.as_ptr(), sys::SCE_KERNEL_MEMBLOCK_TYPE_USER_RW_UNCACHE, bytes as u32, null_mut());
        let mut base: *mut c_void = null_mut();
        if uid < 0 || sys::sceKernelGetMemBlockBase(uid, &mut base) < 0 {
            return null_mut();
        }
        base
    }
}

/// Makes the shader patcher; null if it can't be made (logged).
fn make_patcher() -> *mut gxm::ShaderPatcher {
    let buffer = match crate::gpu_memory::for_gpu(b"NativeResPatcherBuffer\0", PATCHER_BUFFER_BYTES as u32) {
        Some(block) => block.base as *mut c_void,
        None => null_mut(),
    };
    let vertex_usse = usse_block(b"NativeResVertexUsse\0", VERTEX_USSE_BYTES);
    let fragment_usse = usse_block(b"NativeResFragmentUsse\0", FRAGMENT_USSE_BYTES);
    if buffer.is_null() || vertex_usse.is_null() || fragment_usse.is_null() {
        render_log::line(format_args!("shader patcher: no memory\n"));
        return null_mut();
    }
    let mut vertex_offset = 0;
    let mut fragment_offset = 0;
    // Safe: our blocks, whole pages.
    let mapped = unsafe {
        gxm::sceGxmMapVertexUsseMemory(vertex_usse, VERTEX_USSE_BYTES as u32, &mut vertex_offset) >= 0
            && gxm::sceGxmMapFragmentUsseMemory(fragment_usse, FRAGMENT_USSE_BYTES as u32, &mut fragment_offset) >= 0
    };
    if !mapped {
        render_log::line(format_args!("shader patcher: USSE memory not mapped\n"));
        return null_mut();
    }
    let params = gxm::ShaderPatcherParams {
        user_data: null_mut(),
        host_alloc: Some(host_alloc),
        host_free: Some(host_free),
        buffer_alloc: 0,
        buffer_free: 0,
        buffer_memory: buffer,
        buffer_bytes: PATCHER_BUFFER_BYTES as u32,
        vertex_usse_alloc: 0,
        vertex_usse_free: 0,
        vertex_usse_memory: vertex_usse,
        vertex_usse_bytes: VERTEX_USSE_BYTES as u32,
        vertex_usse_offset: vertex_offset,
        fragment_usse_alloc: 0,
        fragment_usse_free: 0,
        fragment_usse_memory: fragment_usse,
        fragment_usse_bytes: FRAGMENT_USSE_BYTES as u32,
        fragment_usse_offset: fragment_offset,
    };
    let mut patcher: *mut gxm::ShaderPatcher = null_mut();
    // Safe: valid parameters, memory that stays ours.
    let result = unsafe { gxm::sceGxmShaderPatcherCreate(&params, &mut patcher) };
    if result < 0 {
        render_log::line(format_args!("shader patcher: not made ({:08x})\n", result));
        return null_mut();
    }
    render_log::line(format_args!("shader patcher ready\n"));
    patcher
}

/// The basic programs and their parameters.
pub struct Basic {
    pub color_vertex: *mut gxm::VertexProgram,
    /// Alpha blending, as vita2d's.
    pub color_fragment: *mut gxm::FragmentProgram,
    /// No blending, all channels written (clears).
    clear_fragment: *mut gxm::FragmentProgram,
    /// The clear's 3 vertices (null: no memory, no clears).
    clear_triangle: *mut ColorVertex,
    pub color_wvp: *const ProgramParameter,
    pub texture_vertex: *mut gxm::VertexProgram,
    /// Alpha blending, as vita2d's.
    pub texture_tint_fragment: *mut gxm::FragmentProgram,
    pub texture_wvp: *const ProgramParameter,
    pub tint_color: *const ProgramParameter,
    color_fragment_id: PatcherId,
    texture_tint_fragment_id: PatcherId,
}

struct State {
    patcher_tried: bool,
    patcher: *mut gxm::ShaderPatcher,
    basic_tried: bool,
    basic: Option<Basic>,
}

static STATE: SingleThreaded<State> = SingleThreaded::new(State { patcher_tried: false, patcher: null_mut(), basic_tried: false, basic: None });

fn state() -> &'static mut State {
    // Safe: display thread only (see the module's head).
    unsafe { STATE.get() }
}

/// The shader patcher, made on the first call; null if it can't be made.
pub fn patcher() -> *mut gxm::ShaderPatcher {
    let state = state();
    if !state.patcher_tried {
        state.patcher_tried = true;
        state.patcher = make_patcher();
    }
    state.patcher
}

/// Our programs are made for this (no multisampling: neither vita2d's nor
/// Sony's display buffers use it).
pub const MULTISAMPLE_MODE: u32 = gxm::MULTISAMPLE_NONE as u32;

/// A pointer to a GXP's bytes as GXM wants it.
pub fn gxp<const N: usize>(program: &shaders::Gxp<N>) -> *const gxm::Program {
    program as *const shaders::Gxp<N> as *const gxm::Program
}

/// Checks and registers `program` with the shader patcher.
pub fn register(program: *const gxm::Program) -> Option<PatcherId> {
    let patcher = patcher();
    let mut id: PatcherId = null_mut();
    // Safe: plain calls; the program is embedded and 16-byte aligned.
    unsafe {
        (!patcher.is_null()
            && gxm::sceGxmProgramCheck(program) >= 0
            && gxm::sceGxmShaderPatcherRegisterProgram(patcher, program, &mut id) >= 0)
            .then_some(id)
    }
}

/// `program`'s parameter `name` (NUL-terminated), or null.
pub fn parameter(program: *const gxm::Program, name: &[u8]) -> *const ProgramParameter {
    debug_assert!(name.last() == Some(&0));
    // Safe: a checked program and a NUL-terminated name.
    unsafe { gxm::sceGxmProgramFindParameterByName(program, name.as_ptr()) }
}

/// The register a vertex attribute goes to (0 for a missing parameter).
pub fn resource(parameter: *const ProgramParameter) -> u16 {
    if parameter.is_null() {
        return 0;
    }
    // Safe: a parameter GXM gave us.
    unsafe { gxm::sceGxmProgramParameterGetResourceIndex(parameter) as u16 }
}

/// Makes a vertex program from registered program `id`; null if it fails.
pub fn vertex_program(id: PatcherId, attributes: &[VertexAttribute], streams: &[VertexStream]) -> Result<*mut gxm::VertexProgram, i32> {
    let mut program: *mut gxm::VertexProgram = null_mut();
    // Safe: valid arrays and a registered program.
    let result = unsafe {
        gxm::sceGxmShaderPatcherCreateVertexProgram(
            patcher(),
            id,
            attributes.as_ptr(),
            attributes.len() as u32,
            streams.as_ptr(),
            streams.len() as u32,
            &mut program,
        )
    };
    if result < 0 {
        Err(result)
    } else {
        Ok(program)
    }
}

/// Makes a fragment program from registered program `id`, linked to vertex
/// program `vertex_gxp`, with `blend` (null: blending off, all channels
/// written); null if it fails.
pub fn fragment_program(id: PatcherId, blend: *const gxm::BlendInfo, vertex_gxp: *const gxm::Program) -> *mut gxm::FragmentProgram {
    let mut program: *mut gxm::FragmentProgram = null_mut();
    // Safe: a registered program, a valid (or null) blend info.
    let result = unsafe {
        gxm::sceGxmShaderPatcherCreateFragmentProgram(
            patcher(),
            id,
            gxm::OUTPUT_REGISTER_FORMAT_UCHAR4,
            MULTISAMPLE_MODE,
            blend,
            vertex_gxp,
            &mut program,
        )
    };
    if result < 0 {
        null_mut()
    } else {
        program
    }
}

fn attribute(offset: u32, format: u8, count: u8, parameter: *const ProgramParameter) -> VertexAttribute {
    VertexAttribute { stream_index: 0, offset: offset as u16, format, component_count: count, register_index: resource(parameter) }
}

/// vita2d's blending: source alpha, one minus source alpha, all channels.
const ALPHA_BLEND: gxm::BlendInfo = gxm::BlendInfo::new(
    gxm::COLOR_MASK_ALL,
    gxm::BLEND_FUNC_ADD,
    gxm::BLEND_FACTOR_SRC_ALPHA,
    gxm::BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
);

fn make_basic() -> Option<Basic> {
    let color_v = gxp(&shaders::nativeres_color_v);
    let color_f = gxp(&shaders::nativeres_color_f);
    let texture_v = gxp(&shaders::nativeres_texture_v);
    let texture_tint_f = gxp(&shaders::nativeres_texture_tint_f);
    let (Some(color_vertex_id), Some(color_fragment_id), Some(texture_vertex_id), Some(texture_tint_fragment_id)) =
        (register(color_v), register(color_f), register(texture_v), register(texture_tint_f))
    else {
        render_log::line(format_args!("basic programs: invalid GXP or no shader patcher\n"));
        return None;
    };
    let color_position = parameter(color_v, b"aPosition\0");
    let color_color = parameter(color_v, b"aColor\0");
    let texture_position = parameter(texture_v, b"aPosition\0");
    let texture_texcoord = parameter(texture_v, b"aTexcoord\0");
    let color_wvp = parameter(color_v, b"wvp\0");
    let texture_wvp = parameter(texture_v, b"wvp\0");
    let tint_color = parameter(texture_tint_f, b"uTintColor\0");
    if [color_position, color_color, texture_position, texture_texcoord, color_wvp, texture_wvp, tint_color].iter().any(|p| p.is_null()) {
        render_log::line(format_args!("basic programs: parameters missing\n"));
        return None;
    }
    let color_attributes = [
        attribute(0, gxm::ATTRIBUTE_FORMAT_F32, 3, color_position),
        attribute(12, gxm::ATTRIBUTE_FORMAT_U8N, 4, color_color),
    ];
    let color_stream = [VertexStream { stride: core::mem::size_of::<ColorVertex>() as u16, index_source: gxm::INDEX_SOURCE_INDEX_16BIT }];
    let texture_attributes = [
        attribute(0, gxm::ATTRIBUTE_FORMAT_F32, 3, texture_position),
        attribute(16, gxm::ATTRIBUTE_FORMAT_F32, 2, texture_texcoord),
    ];
    let texture_stream = [VertexStream { stride: core::mem::size_of::<TextureVertex>() as u16, index_source: gxm::INDEX_SOURCE_INDEX_16BIT }];
    let color_vertex = vertex_program(color_vertex_id, &color_attributes, &color_stream);
    let texture_vertex = vertex_program(texture_vertex_id, &texture_attributes, &texture_stream);
    let (Ok(color_vertex), Ok(texture_vertex)) = (color_vertex, texture_vertex) else {
        render_log::line(format_args!("basic programs: vertex programs not made\n"));
        return None;
    };
    let color_fragment = fragment_program(color_fragment_id, &ALPHA_BLEND, color_v);
    let clear_fragment = fragment_program(color_fragment_id, null(), color_v);
    let texture_tint_fragment = fragment_program(texture_tint_fragment_id, &ALPHA_BLEND, texture_v);
    if color_fragment.is_null() || clear_fragment.is_null() || texture_tint_fragment.is_null() {
        render_log::line(format_args!("basic programs: fragment programs not made\n"));
        return None;
    }
    render_log::line(format_args!("basic programs ready\n"));
    Some(Basic {
        color_vertex,
        color_fragment,
        clear_fragment,
        clear_triangle: make_clear_triangle(),
        color_wvp,
        texture_vertex,
        texture_tint_fragment,
        texture_wvp,
        tint_color,
        color_fragment_id,
        texture_tint_fragment_id,
    })
}

/// The basic programs, made on the first call; None if they can't be made.
pub fn get() -> Option<&'static Basic> {
    let state = state();
    if !state.basic_tried {
        state.basic_tried = true;
        state.basic = make_basic();
    }
    state.basic.as_ref()
}

/// The colour (`textured` false) or texture-tint fragment program with
/// `blend` (null: blending off, all channels written); null if it can't be
/// made. Each call makes a new one: programs.rs keeps them.
pub fn blend_fragment_program(textured: bool, blend: *const gxm::BlendInfo) -> *mut gxm::FragmentProgram {
    let Some(basic) = get() else {
        return null_mut();
    };
    if textured {
        fragment_program(basic.texture_tint_fragment_id, blend, gxp(&shaders::nativeres_texture_v))
    } else {
        fragment_program(basic.color_fragment_id, blend, gxp(&shaders::nativeres_color_v))
    }
}

/// Sets the texture programs (the vertex program with the projection,
/// `fragment` with a white tint) for our own passes: unswizzle, target
/// copies, presenting. False if the programs don't exist (draw nothing then).
///
/// # Safety
/// Display thread, inside a scene; `fragment` a texture-tint program.
pub unsafe fn use_texture_programs(fragment: *const gxm::FragmentProgram) -> bool {
    let Some(basic) = get() else {
        return false;
    };
    let context = gxm::context();
    gxm::sceGxmSetVertexProgram(context, basic.texture_vertex);
    gxm::sceGxmSetFragmentProgram(context, fragment);
    let mut vertex_uniforms = null_mut();
    gxm::sceGxmReserveVertexDefaultUniformBuffer(context, &mut vertex_uniforms);
    gxm::sceGxmSetUniformDataF(vertex_uniforms, basic.texture_wvp, 0, 16, ORTHO.as_ptr());
    const WHITE: [f32; 4] = [1.0; 4];
    let mut fragment_uniforms = null_mut();
    gxm::sceGxmReserveFragmentDefaultUniformBuffer(context, &mut fragment_uniforms);
    gxm::sceGxmSetUniformDataF(fragment_uniforms, basic.tint_color, 0, 4, WHITE.as_ptr());
    true
}

/// Fills the current scene with opaque black (alpha 1), no blending, as
/// vita2d_clear_screen did with its clear colour: one triangle over the
/// whole scene at depth 0 (z 1 after the projection).
///
/// # Safety
/// Display thread, inside a scene.
pub unsafe fn clear_screen() {
    let Some(basic) = get() else {
        return;
    };
    let indices = crate::targets::linear_indices();
    if indices.is_null() || basic.clear_triangle.is_null() {
        return;
    }
    let context = gxm::context();
    gxm::sceGxmSetVertexProgram(context, basic.color_vertex);
    gxm::sceGxmSetFragmentProgram(context, basic.clear_fragment);
    let mut vertex_uniforms = null_mut();
    gxm::sceGxmReserveVertexDefaultUniformBuffer(context, &mut vertex_uniforms);
    gxm::sceGxmSetUniformDataF(vertex_uniforms, basic.color_wvp, 0, 16, ORTHO.as_ptr());
    gxm::sceGxmSetVertexStream(context, 0, basic.clear_triangle as *const c_void);
    gxm::sceGxmDraw(context, gxm::PRIMITIVE_TRIANGLES, gxm::INDEX_FORMAT_U16, indices as *const c_void, 3);
}

/// The clear's triangle in GPU memory: (-1, -1), (3, -1), (-1, 3) after
/// the projection, twice the screen in both directions; opaque black.
fn make_clear_triangle() -> *mut ColorVertex {
    let Some(block) = crate::gpu_memory::gpu_input(b"NativeResClearTriangle\0", sys::pages(3 * core::mem::size_of::<ColorVertex>())) else {
        return null_mut();
    };
    let vertices = block.base as *mut ColorVertex;
    const BLACK: u32 = 0xFF00_0000;
    for (i, (x, y)) in [(0.0, 544.0), (1920.0, 544.0), (0.0, -544.0)].into_iter().enumerate() {
        // Safe: inside the block.
        unsafe { vertices.add(i).write(ColorVertex { x, y, z: 0.0, color: BLACK }) };
    }
    vertices
}
