//! The GXM (Vita GPU library) functions and types we call from Rust, declared
//! by hand from vitasdk's psp2/gxm.h.

#![allow(non_snake_case, dead_code)]

use core::ffi::c_void;

/// SceGxmContext: the display thread's drawing context (opaque).
pub type Context = c_void;
/// SceGxmRenderTarget: what a scene draws into, by size (opaque).
pub type RenderTarget = c_void;
/// SceGxmSyncObject, SceGxmValidRegion (opaque; we pass null).
pub type SyncObject = c_void;
pub type ValidRegion = c_void;

/// SceGxmMemoryAttribFlags: what the GPU may do with mapped memory.
pub const MEMORY_ATTRIB_READ: u32 = 1;
pub const MEMORY_ATTRIB_WRITE: u32 = 2;

/// SceGxmTextureFormat values we use.
pub const TEXTURE_FORMAT_U8U8U8U8_ABGR: u32 = 0x0C00_0000;
/// The same texels, alpha read as 1.
pub const TEXTURE_FORMAT_X8U8U8U8_1BGR: u32 = 0x0C00_4000;
pub const TEXTURE_FORMAT_P8_ABGR: u32 = 0x9500_0000;
/// SCE_GXM_TEXTURE_LINEAR_STRIDED (`sceGxmTextureGetType`).
pub const TEXTURE_TYPE_LINEAR_STRIDED: u32 = 0xC000_0000;
pub const TEXTURE_FORMAT_U5U6U5_BGR: u32 = 0x0500_0000;
pub const TEXTURE_FORMAT_U1U5U5U5_ABGR: u32 = 0x0400_0000;
pub const TEXTURE_FORMAT_U4U4U4U4_ABGR: u32 = 0x0200_0000;
/// 16 bits a channel, unsigned normalized (the wide textures, core `GpuFormat::Wide16`).
pub const TEXTURE_FORMAT_U16U16U16U16_ABGR: u32 = 0x1C00_0000;

/// SceGxmColorFormat values we use.
pub const COLOR_FORMAT_A8B8G8R8: u32 = 0x0000_0000;
pub const COLOR_FORMAT_U5U6U5_BGR: u32 = 0x3000_0000;
pub const COLOR_FORMAT_U1U5U5U5_ABGR: u32 = 0x4000_0000;
pub const COLOR_FORMAT_U4U4U4U4_ABGR: u32 = 0x5000_0000;
/// SceGxmColorSurfaceType, SceGxmColorSurfaceScaleMode,
/// SceGxmOutputRegisterSize, SceGxmMultisampleMode.
pub const COLOR_SURFACE_LINEAR: u32 = 0;
pub const COLOR_SURFACE_SCALE_NONE: u32 = 0;
pub const OUTPUT_REGISTER_SIZE_32BIT: u32 = 0;
pub const MULTISAMPLE_NONE: u16 = 0;
/// SceGxmDepthStencilFormat, SceGxmDepthStencilSurfaceType.
pub const DEPTH_STENCIL_FORMAT_S8D24: u32 = 0x0126_6000;
pub const DEPTH_STENCIL_SURFACE_TILED: u32 = 0x0001_1000;

/// SceGxmNotification: when the GPU has finished a scene's vertex or
/// fragment work, it writes `value` to `address` (a word of the
/// notification region).
#[repr(C)]
pub struct Notification {
    pub address: *mut u32,
    pub value: u32,
}

/// SceGxmTexture: a texture's description (where, format, size, filters).
/// GXM copies it when it is set for a draw.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Texture {
    pub control_words: [u32; 4],
}

/// SceGxmColorSurface (0x30 bytes): where and how a scene writes colours.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ColorSurface {
    pub words: [u32; 12],
}

/// SceGxmDepthStencilSurface (0x14 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct DepthStencilSurface {
    pub words: [u32; 5],
}

/// SceGxmRenderTargetParams (0x14 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RenderTargetParams {
    pub flags: u32,
    pub width: u16,
    pub height: u16,
    pub scenes_per_frame: u16,
    pub multisample_mode: u16,
    pub multisample_locations: u32,
    /// -1: GXM allocates the target's memory itself.
    pub driver_mem_block: i32,
}

const _: () = assert!(core::mem::size_of::<Texture>() == 0x10);
const _: () = assert!(core::mem::size_of::<ColorSurface>() == 0x30);
const _: () = assert!(core::mem::size_of::<DepthStencilSurface>() == 0x14);
const _: () = assert!(core::mem::size_of::<RenderTargetParams>() == 0x14);

extern "C" {
    /// Makes `size` bytes at `base` (whole pages of one memory block)
    /// readable or writable by the GPU.
    pub fn sceGxmMapMemory(base: *mut c_void, size: u32, attributes: u32) -> i32;
    pub fn sceGxmUnmapMemory(base: *mut c_void) -> i32;
    /// The process's 512 notification words.
    pub fn sceGxmGetNotificationRegion() -> *mut u32;
    /// Waits until the GPU has finished everything sent so far.
    pub fn sceGxmFinish(context: *mut Context);

    /// `mip_count` 0: no levels.
    pub fn sceGxmTextureInitLinear(
        texture: *mut Texture,
        data: *const c_void,
        format: u32,
        width: u32,
        height: u32,
        mip_count: u32,
    ) -> i32;
    /// A linear texture whose rows lie `byte_stride` bytes apart.
    pub fn sceGxmTextureInitLinearStrided(
        texture: *mut Texture,
        data: *const c_void,
        format: u32,
        width: u32,
        height: u32,
        byte_stride: u32,
    ) -> i32;
    pub fn sceGxmTextureSetPalette(texture: *mut Texture, palette: *const c_void) -> i32;
    pub fn sceGxmTextureGetPalette(texture: *const Texture) -> *mut c_void;
    pub fn sceGxmTextureGetData(texture: *const Texture) -> *mut c_void;
    pub fn sceGxmTextureSetFormat(texture: *mut Texture, format: u32) -> i32;
    pub fn sceGxmTextureGetFormat(texture: *const Texture) -> u32;
    /// `stride_in_samples`: the width for whole tiles.
    pub fn sceGxmDepthStencilSurfaceInit(
        surface: *mut DepthStencilSurface,
        format: u32,
        surface_type: u32,
        stride_in_samples: u32,
        depth_data: *mut c_void,
        stencil_data: *mut c_void,
    ) -> i32;
    /// Starts a scene drawing into `color_surface` (of `target`'s size).
    pub fn sceGxmBeginScene(
        context: *mut Context,
        flags: u32,
        target: *const RenderTarget,
        valid_region: *const ValidRegion,
        vertex_sync: *mut SyncObject,
        fragment_sync: *mut SyncObject,
        color_surface: *const ColorSurface,
        depth_stencil: *const DepthStencilSurface,
    ) -> i32;
    /// Ends the scene; the GPU writes the notifications (null: none) when
    /// it has finished the scene's vertex / fragment work.
    pub fn sceGxmEndScene(context: *mut Context, vertex_notification: *const Notification, fragment_notification: *const Notification) -> i32;
    pub fn sceGxmColorSurfaceInit(
        surface: *mut ColorSurface,
        format: u32,
        surface_type: u32,
        scale_mode: u32,
        output_register_size: u32,
        width: u32,
        height: u32,
        stride_in_pixels: u32,
        data: *mut c_void,
    ) -> i32;
    pub fn sceGxmCreateRenderTarget(params: *const RenderTargetParams, target: *mut *mut RenderTarget) -> i32;
    pub fn sceGxmDestroyRenderTarget(target: *mut RenderTarget) -> i32;
}

// ------------------------------------------------------------------------
// Programs and drawing
// ------------------------------------------------------------------------

/// A compiled program (a GXP), a vertex or fragment program made from one
/// by the shader patcher, one of its parameters, the patcher (all opaque).
pub type Program = c_void;
pub type VertexProgram = c_void;
pub type FragmentProgram = c_void;
pub type ProgramParameter = c_void;
pub type ShaderPatcher = c_void;
/// SceGxmShaderPatcherId: a program registered with the patcher.
pub type PatcherId = *mut c_void;

/// SceGxmBlendInfo (4 bytes): the colour mask, then 4-bit fields in pairs,
/// low nibble first (GCC's bit-field order): colour/alpha function, colour
/// source/destination factor, alpha source/destination factor.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlendInfo {
    pub color_mask: u8,
    pub functions: u8,
    pub color_factors: u8,
    pub alpha_factors: u8,
}

impl BlendInfo {
    /// The same function and factors for colour and alpha.
    pub const fn new(color_mask: u8, function: u8, source: u8, destination: u8) -> BlendInfo {
        BlendInfo {
            color_mask,
            functions: function | (function << 4),
            color_factors: source | (destination << 4),
            alpha_factors: source | (destination << 4),
        }
    }
}

/// SceGxmVertexAttribute (8 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VertexAttribute {
    pub stream_index: u16,
    pub offset: u16,
    pub format: u8,
    pub component_count: u8,
    pub register_index: u16,
}

/// SceGxmVertexStream (4 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VertexStream {
    pub stride: u16,
    pub index_source: u16,
}

const _: () = assert!(core::mem::size_of::<BlendInfo>() == 4);
const _: () = assert!(core::mem::size_of::<VertexAttribute>() == 8);
const _: () = assert!(core::mem::size_of::<VertexStream>() == 4);

/// SceGxmColorMask.
pub const COLOR_MASK_A: u8 = 1;
pub const COLOR_MASK_R: u8 = 2;
pub const COLOR_MASK_G: u8 = 4;
pub const COLOR_MASK_B: u8 = 8;
pub const COLOR_MASK_ALL: u8 = 15;
/// SceGxmBlendFunc (NONE, then ADD, SUBTRACT, REVERSE_SUBTRACT, MIN, MAX:
/// the PSP's equations 0-4 plus one).
pub const BLEND_FUNC_NONE: u8 = 0;
pub const BLEND_FUNC_ADD: u8 = 1;
/// SceGxmBlendFactor: ZERO, ONE, SRC_COLOR, ONE_MINUS_SRC_COLOR, SRC_ALPHA,
/// ONE_MINUS_SRC_ALPHA, DST_COLOR, ONE_MINUS_DST_COLOR, DST_ALPHA,
/// ONE_MINUS_DST_ALPHA: the PSP's factor numbers (blend.rs) as they are.
pub const BLEND_FACTOR_ZERO: u8 = 0;
pub const BLEND_FACTOR_ONE: u8 = 1;
pub const BLEND_FACTOR_SRC_ALPHA: u8 = 4;
pub const BLEND_FACTOR_ONE_MINUS_SRC_ALPHA: u8 = 5;
pub const BLEND_FACTOR_ONE_MINUS_DST_ALPHA: u8 = 9;

/// SceGxmAttributeFormat.
pub const ATTRIBUTE_FORMAT_U8: u8 = 0;
pub const ATTRIBUTE_FORMAT_S8: u8 = 1;
pub const ATTRIBUTE_FORMAT_U16: u8 = 2;
pub const ATTRIBUTE_FORMAT_S16: u8 = 3;
pub const ATTRIBUTE_FORMAT_U8N: u8 = 4;
pub const ATTRIBUTE_FORMAT_S8N: u8 = 5;
pub const ATTRIBUTE_FORMAT_F32: u8 = 9;
pub const INDEX_SOURCE_INDEX_16BIT: u16 = 0;
pub const OUTPUT_REGISTER_FORMAT_UCHAR4: u32 = 1;

/// SceGxmDepthFunc.
pub const DEPTH_FUNC_NEVER: u32 = 0x0000_0000;
pub const DEPTH_FUNC_LESS: u32 = 0x0040_0000;
pub const DEPTH_FUNC_EQUAL: u32 = 0x0080_0000;
pub const DEPTH_FUNC_LESS_EQUAL: u32 = 0x00C0_0000;
pub const DEPTH_FUNC_GREATER: u32 = 0x0100_0000;
pub const DEPTH_FUNC_NOT_EQUAL: u32 = 0x0140_0000;
pub const DEPTH_FUNC_GREATER_EQUAL: u32 = 0x0180_0000;
pub const DEPTH_FUNC_ALWAYS: u32 = 0x01C0_0000;
/// SceGxmDepthWriteMode.
pub const DEPTH_WRITE_ENABLED: u32 = 0x0000_0000;
pub const DEPTH_WRITE_DISABLED: u32 = 0x0010_0000;
/// SceGxmFragmentProgramMode.
pub const FRAGMENT_PROGRAM_ENABLED: u32 = 0x0000_0000;
pub const FRAGMENT_PROGRAM_DISABLED: u32 = 0x0020_0000;
/// SceGxmCullMode.
pub const CULL_NONE: u32 = 0;
pub const CULL_CW: u32 = 1;
pub const CULL_CCW: u32 = 2;
/// SceGxmRegionClipMode.
pub const REGION_CLIP_NONE: u32 = 0x0000_0000;
pub const REGION_CLIP_OUTSIDE: u32 = 0x8000_0000;
/// SceGxmPrimitiveType, SceGxmIndexFormat.
pub const PRIMITIVE_TRIANGLES: u32 = 0x0000_0000;
pub const PRIMITIVE_TRIANGLE_STRIP: u32 = 0x0C00_0000;
pub const INDEX_FORMAT_U16: u32 = 0;
/// SceGxmTextureFilter, SceGxmTextureMipFilter, SceGxmTextureAddrMode.
pub const TEXTURE_FILTER_POINT: u32 = 0;
pub const TEXTURE_FILTER_LINEAR: u32 = 1;
pub const TEXTURE_MIP_FILTER_DISABLED: u32 = 0x0000_0000;
pub const TEXTURE_MIP_FILTER_ENABLED: u32 = 0x0000_0200;
pub const TEXTURE_ADDR_REPEAT: u32 = 0;
pub const TEXTURE_ADDR_CLAMP: u32 = 2;

/// More USSE memory for the shader patcher: `size` bytes, their USSE offset
/// in `usse_offset`; null if there is none.
pub type UsseAlloc = unsafe extern "C" fn(user_data: *mut c_void, size: u32, usse_offset: *mut u32) -> *mut c_void;
pub type UsseFree = unsafe extern "C" fn(user_data: *mut c_void, memory: *mut c_void);

/// SceGxmShaderPatcherParams (0x44 bytes): the patcher's memory. Host
/// memory (its own bookkeeping) through the callbacks; the buffer as a fixed
/// block; the vertex/fragment USSE code memory as fixed blocks, grown
/// through the USSE callbacks when they are full (basic_programs.rs).
#[repr(C)]
pub struct ShaderPatcherParams {
    pub user_data: *mut c_void,
    pub host_alloc: Option<unsafe extern "C" fn(user_data: *mut c_void, size: u32) -> *mut c_void>,
    pub host_free: Option<unsafe extern "C" fn(user_data: *mut c_void, memory: *mut c_void)>,
    pub buffer_alloc: usize,
    pub buffer_free: usize,
    pub buffer_memory: *mut c_void,
    pub buffer_bytes: u32,
    pub vertex_usse_alloc: Option<UsseAlloc>,
    pub vertex_usse_free: Option<UsseFree>,
    pub vertex_usse_memory: *mut c_void,
    pub vertex_usse_bytes: u32,
    pub vertex_usse_offset: u32,
    pub fragment_usse_alloc: Option<UsseAlloc>,
    pub fragment_usse_free: Option<UsseFree>,
    pub fragment_usse_memory: *mut c_void,
    pub fragment_usse_bytes: u32,
    pub fragment_usse_offset: u32,
}

const _: () = assert!(core::mem::size_of::<ShaderPatcherParams>() == 0x44);

extern "C" {
    pub fn sceGxmShaderPatcherCreate(params: *const ShaderPatcherParams, patcher: *mut *mut ShaderPatcher) -> i32;
    /// Maps memory for the GPU's vertex / fragment USSE code; `offset`: where
    /// it lies in the USSE address space.
    pub fn sceGxmMapVertexUsseMemory(base: *mut c_void, size: u32, offset: *mut u32) -> i32;
    pub fn sceGxmMapFragmentUsseMemory(base: *mut c_void, size: u32, offset: *mut u32) -> i32;
    /// Draws only inside (REGION_CLIP_OUTSIDE: outside is clipped) the
    /// rectangle, in pixels, edges included; NONE: everywhere.
    pub fn sceGxmSetRegionClip(context: *mut Context, mode: u32, x_min: u32, y_min: u32, x_max: u32, y_max: u32);
    pub fn sceGxmProgramCheck(program: *const Program) -> i32;
    pub fn sceGxmProgramFindParameterByName(program: *const Program, name: *const u8) -> *const ProgramParameter;
    pub fn sceGxmProgramParameterGetResourceIndex(parameter: *const ProgramParameter) -> u32;
    pub fn sceGxmProgramGetParameterCount(program: *const Program) -> u32;
    pub fn sceGxmProgramGetParameter(program: *const Program, index: u32) -> *const ProgramParameter;
    pub fn sceGxmProgramParameterGetName(parameter: *const ProgramParameter) -> *const u8;
    pub fn sceGxmProgramParameterGetCategory(parameter: *const ProgramParameter) -> u32;
    pub fn sceGxmProgramParameterGetContainerIndex(parameter: *const ProgramParameter) -> u32;
    pub fn sceGxmProgramParameterGetComponentCount(parameter: *const ProgramParameter) -> u32;
    pub fn sceGxmProgramParameterGetArraySize(parameter: *const ProgramParameter) -> u32;
    pub fn sceGxmProgramGetDefaultUniformBufferSize(program: *const Program) -> u32;
    pub fn sceGxmShaderPatcherRegisterProgram(patcher: *mut ShaderPatcher, program: *const Program, id: *mut PatcherId) -> i32;
    pub fn sceGxmShaderPatcherCreateVertexProgram(
        patcher: *mut ShaderPatcher,
        id: PatcherId,
        attributes: *const VertexAttribute,
        attribute_count: u32,
        streams: *const VertexStream,
        stream_count: u32,
        program: *mut *mut VertexProgram,
    ) -> i32;
    pub fn sceGxmShaderPatcherCreateFragmentProgram(
        patcher: *mut ShaderPatcher,
        id: PatcherId,
        output_format: u32,
        multisample_mode: u32,
        blend: *const BlendInfo,
        vertex_program: *const Program,
        program: *mut *mut FragmentProgram,
    ) -> i32;

    pub fn sceGxmSetVertexProgram(context: *mut Context, program: *const VertexProgram);
    pub fn sceGxmSetFragmentProgram(context: *mut Context, program: *const FragmentProgram);
    pub fn sceGxmReserveVertexDefaultUniformBuffer(context: *mut Context, buffer: *mut *mut c_void) -> i32;
    pub fn sceGxmReserveFragmentDefaultUniformBuffer(context: *mut Context, buffer: *mut *mut c_void) -> i32;
    pub fn sceGxmSetUniformDataF(
        buffer: *mut c_void,
        parameter: *const ProgramParameter,
        component_offset: u32,
        component_count: u32,
        data: *const f32,
    ) -> i32;
    pub fn sceGxmSetFrontDepthFunc(context: *mut Context, function: u32);
    pub fn sceGxmSetFrontDepthWriteEnable(context: *mut Context, mode: u32);
    pub fn sceGxmSetFrontFragmentProgramEnable(context: *mut Context, mode: u32);
    pub fn sceGxmSetCullMode(context: *mut Context, mode: u32);
    pub fn sceGxmSetViewport(context: *mut Context, x_offset: f32, x_scale: f32, y_offset: f32, y_scale: f32, z_offset: f32, z_scale: f32);
    pub fn sceGxmSetFragmentTexture(context: *mut Context, index: u32, texture: *const Texture) -> i32;
    pub fn sceGxmSetVertexStream(context: *mut Context, index: u32, data: *const c_void) -> i32;
    pub fn sceGxmSetVertexUniformBuffer(context: *mut Context, buffer_index: u32, data: *const c_void) -> i32;
    pub fn sceGxmDraw(context: *mut Context, primitive: u32, index_format: u32, indices: *const c_void, count: u32) -> i32;

    pub fn sceGxmTextureSetMinFilter(texture: *mut Texture, filter: u32) -> i32;
    pub fn sceGxmTextureSetMagFilter(texture: *mut Texture, filter: u32) -> i32;
    pub fn sceGxmTextureSetMipFilter(texture: *mut Texture, filter: u32) -> i32;
    pub fn sceGxmTextureSetUAddrMode(texture: *mut Texture, mode: u32) -> i32;
    pub fn sceGxmTextureSetVAddrMode(texture: *mut Texture, mode: u32) -> i32;
    pub fn sceGxmTextureSetLodBias(texture: *mut Texture, bias: u32) -> i32;
    pub fn sceGxmTextureGetLodBias(texture: *const Texture) -> u32;
    pub fn sceGxmTextureGetMipmapCount(texture: *const Texture) -> u32;
    pub fn sceGxmTextureSetMipmapCount(texture: *mut Texture, count: u32) -> i32;
    /// SceGxmTextureType: `TEXTURE_TYPE_LINEAR_STRIDED` for a strided texture.
    pub fn sceGxmTextureGetType(texture: *const Texture) -> u32;
    pub fn sceGxmTextureGetWidth(texture: *const Texture) -> u32;
    pub fn sceGxmTextureGetHeight(texture: *const Texture) -> u32;
}

/// The display thread's GXM context (vita2d's, inside AdrenalineUser), as
/// its first sceGxmBeginScene passed it (display_hooks.rs).
///
/// # Safety
/// Only on the display thread, once the display loop has begun a scene.
pub unsafe fn context() -> *mut Context {
    crate::display_hooks::context()
}
