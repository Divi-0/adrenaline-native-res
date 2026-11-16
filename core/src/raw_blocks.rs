//! What raw draws' vertex programs multiply their vertices with: the GE's
//! matrices and registers as they are, copied into small frame tables only
//! when they change (performance refactor step 3, 2026-11-16).
//!
//! Before, every new transform of a raw draw was worked out on the CPU:
//! projection x view x world, the position scale and a shared bone weight
//! folded in, normal rows, fog row, a 192-byte `GpuTransform` compared with
//! the table's newest entry. On the MTX title that was a third of the
//! recorder thread's time (DESIGN "Performance refactor, step 0"). Now a
//! raw draw names a `RawTransform` (the object: its world matrix, which
//! changes per object) whose `RawCamera` (view, projection, viewport, fog,
//! texture scale: once or twice per frame) and light set it shares with the
//! others, and the vertex program does the multiplications, as PPSSPP's
//! vertex shaders do (GPU/Common/VertexShaderGenerator.cpp: u_world, u_view,
//! u_proj, one after the other) and in the GE's order (position scaled,
//! skinned, world, view, projection). Bones go as uploaded (`Draw::bones`):
//! every skinned draw is skinned by the program, a single weight too.
//!
//! The entries carry the transform inputs' change numbers
//! (`TransformInputs::changes`): a draw with the inputs of the draw before
//! finds the newest entry without comparing anything.
//!
//! The functions after `raw_transform` are the vertex programs' arithmetic on
//! the CPU (nativeres_raw_v.cg, _raw_plain_v, _raw_sprite_v), in their order:
//! the PC replay, the frame hashes and the tests draw raw vertices with them
//! (`raw_vertex::RawCorners`). Keep both in step.

use crate::frame::{FrameVertex, Geometry, GpuLights, RawCamera, RawColors, RawTransform};
use crate::transform::{ClipVertex, TransformInputs};
use crate::vertex::VertexFormat;

/// `RawTransform::lighting_key` of a draw: 0 if the GPU doesn't light it,
/// else what its light set depends on besides the inputs (`Lights::gpu`):
/// whether its vertices have a colour and whether it is opaque.
pub fn lighting_key(lit: Option<(bool, bool)>) -> u32 {
    match lit {
        None => 0,
        Some((has_color, opaque)) => 1 | (has_color as u32) << 1 | (opaque as u32) << 2,
    }
}

/// The `Draw::transform` number of a raw draw with `inputs`: the newest
/// `RawTransform` if it was written from the same inputs (or holds the same
/// numbers), else a new one with its camera and, for draws the GPU lights
/// (`lit`: has a vertex colour, opaque), its light set. `None` if a table is
/// full (the run then goes the old way).
pub fn raw_transform(geometry: &mut Geometry, inputs: &TransformInputs, lit: Option<(bool, bool)>) -> Option<u32> {
    let (inputs_change, lighting_change) = inputs.changes();
    let key = lighting_key(lit);
    // The newest entry's camera and world, if it was written from the same
    // matrices and registers (0: inputs without change numbers,
    // `TransformInputs::of`): then only the lighting can differ (MTX's
    // riders change their material colours before nearly every part).
    let mut same_base = None;
    if let Some((number, last)) = geometry.last_raw_transform() {
        if inputs_change != 0 && last.inputs_change == inputs_change {
            if last.lighting_key == key && (key == 0 || last.lighting_change == lighting_change) {
                return Some(number);
            }
            same_base = Some((last.camera, last.world));
        }
    }
    let (camera, world) = match same_base {
        Some(known) => known,
        None => (geometry.add_raw_camera(&inputs.raw_camera())?, inputs.raw_world()),
    };
    let light_set = match lit {
        Some((has_color, opaque)) => geometry.add_light_set(&inputs.gpu_lights(has_color, opaque)?)? as u16,
        None => 0,
    };
    if let Some((number, last)) = geometry.last_raw_transform() {
        // Other inputs, the same numbers (bones or the format changed, or a
        // matrix was uploaded again unchanged in another context): that one,
        // so the draws can still be joined.
        let same_world = (0..3).all(|r| (0..4).all(|c| last.world[r][c].to_bits() == world[r][c].to_bits()));
        if same_world && last.camera == camera && last.light_set == light_set && last.lighting_key == key {
            geometry.restamp_last_raw_transform(inputs_change, lighting_change);
            return Some(number);
        }
    }
    geometry.push_raw_transform(RawTransform { world, camera, light_set, inputs_change, lighting_change, lighting_key: key })
}

/// `((r0 x + r1 y) + r2 z) + r3`: one row of a matrix applied to (x, y, z, 1).
#[inline(always)]
fn apply_row(row: &[f32; 4], [x, y, z]: [f32; 3]) -> f32 {
    row[0] * x + row[1] * y + row[2] * z + row[3]
}

/// A raw vertex's model-space position from its raw position (as read, not
/// scaled): times the format's scale (`VertexFormat::position_scale`),
/// then skinned with `bones` (12 floats per bone, rows, as uploaded; empty:
/// not skinned) and the weights (already scaled, 1.0 = full): the sum of
/// weight x bone applied to (x, y, z, 1). Also the normal, skinned with the
/// bones' 3x3 parts (not normalized).
pub fn model_space(format: VertexFormat, bones: &[f32], position: [f32; 3], weights: &[f32], normal: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let scale = format.position_scale();
    let scaled = position.map(|value| value * scale);
    if bones.is_empty() {
        return (scaled, normal);
    }
    let mut model = [0.0; 3];
    let mut skinned_normal = [0.0; 3];
    for (bone, &weight) in bones.chunks_exact(12).zip(weights) {
        for row in 0..3 {
            let r: &[f32; 4] = bone[4 * row..4 * row + 4].try_into().expect("12 floats per bone");
            model[row] += weight * apply_row(r, scaled);
            skinned_normal[row] += weight * (r[0] * normal[0] + r[1] * normal[1] + r[2] * normal[2]);
        }
    }
    (model, skinned_normal)
}

/// Texture coordinate scale u, v and offset u, v of a raw draw (the
/// program's uRawUv): the format's scale (`VertexFormat::texcoord_scale`,
/// a power of two, so its product with TEXTURE_SCALE is exact) times the
/// camera's texture scale; no scale for formats without texture coordinates
/// (only the offset).
pub fn raw_uv(camera: &RawCamera, format: VertexFormat) -> [f32; 4] {
    let [scale_u, scale_v, offset_u, offset_v] = camera.texture;
    if format.0 & 3 == 0 {
        return [0.0, 0.0, offset_u, offset_v];
    }
    let scale = format.texcoord_scale();
    [scale * scale_u, scale * scale_v, offset_u, offset_v]
}

/// A model-space vertex in clip space with its view-space z: world, view,
/// projection one after the other.
#[inline]
pub fn clip_position(transform: &RawTransform, camera: &RawCamera, model: [f32; 3]) -> ([f32; 4], f32) {
    let world = transform.world.map(|row| apply_row(&row, model));
    let view = camera.view.map(|row| apply_row(&row, world));
    (camera.projection.map(|row| apply_row(&row, view)), view[2])
}

/// A raw vertex (model-space position, raw texture coordinates, its colour
/// from `colors`) in clip space, with texture coordinates (`uv`: `raw_uv`),
/// colour and fog.
pub fn clip_vertex(
    transform: &RawTransform,
    camera: &RawCamera,
    colors: &RawColors,
    uv: &[f32; 4],
    model: [f32; 3],
    texcoord: [f32; 2],
    color: Option<u32>,
) -> ClipVertex {
    let ([x, y, z, w], view_z) = clip_position(transform, camera, model);
    ClipVertex {
        x,
        y,
        z,
        w,
        u: texcoord[0] * uv[0] + uv[2],
        v: texcoord[1] * uv[1] + uv[3],
        color: colors.color(color),
        fog: (view_z * camera.fog[0] + camera.fog[1]) * camera.fog[2],
    }
}

/// A clip-space vertex on the screen, as `GpuTransform::to_screen` puts one.
pub fn clip_to_screen(camera: &RawCamera, clip: &ClipVertex) -> FrameVertex {
    let over_w = 1.0 / clip.w;
    FrameVertex {
        x: camera.viewport[0] + camera.viewport[2] * clip.x * over_w,
        y: camera.viewport[1] + camera.viewport[3] * clip.y * over_w,
        z: (camera.depth[0] + camera.depth[1] * clip.z * over_w).clamp(0.0, 1.0),
        w: clip.w,
        u: clip.u,
        v: clip.v,
        color: clip.color,
        fog: clip.fog,
    }
}

/// The rows taking a raw draw's (skinned) normals to world space for its
/// lights (`GpuLights::color`): the world matrix's 3x3, turned around for
/// REVERSE_NORMAL (`GpuLights::from_vertex` w).
pub fn normal_rows(transform: &RawTransform, lights: &GpuLights) -> [[f32; 4]; 3] {
    let sign = if lights.from_vertex[3] != 0.0 { -1.0 } else { 1.0 };
    transform.world.map(|row| [sign * row[0], sign * row[1], sign * row[2], 0.0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::op;
    use crate::frame::Frame;
    use crate::ge_state::GeState;
    use crate::transform::{Transform, TransformCache};

    fn set(ge: &mut GeState, opcode: u8, data: u32) {
        ge.set_register(crate::command::Command((opcode as u32) << 24 | (data & 0x00FF_FFFF)));
    }

    fn set_float(ge: &mut GeState, opcode: u8, value: f32) {
        set(ge, opcode, value.to_bits() >> 8);
    }

    /// A camera looking at the origin with a perspective projection, fog,
    /// texture scale and offset, and a world matrix that turns and moves.
    fn scene() -> GeState {
        let mut ge = GeState::new();
        ge.matrices.world = [0.8, 0.6, 0.0, -0.6, 0.8, 0.0, 0.0, 0.0, 1.0, 3.0, -2.0, 1.5];
        ge.matrices.view = [1.0, 0.0, 0.0, 0.0, 0.96, -0.28, 0.0, 0.28, 0.96, 0.5, 0.25, -40.0];
        // A perspective projection (column by column): w = -z.
        ge.matrices.projection = [1.7, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, -1.02, -1.0, 0.0, 0.0, -2.02, 0.0];
        for (opcode, value) in [
            (op::VIEWPORT_X_SCALE, 240.0),
            (op::VIEWPORT_Y_SCALE, -136.0),
            (op::VIEWPORT_Z_SCALE, -32000.0),
            (op::VIEWPORT_X_CENTER, 2048.0),
            (op::VIEWPORT_Y_CENTER, 2048.0),
            (op::VIEWPORT_Z_CENTER, 32000.0),
            (op::FOG_END, 60.0),
            (op::FOG_SCALE, 0.025),
            (op::TEXTURE_SCALE_U, 2.0),
            (op::TEXTURE_SCALE_V, 0.5),
            (op::TEXTURE_OFFSET_U, 0.25),
            (op::TEXTURE_OFFSET_V, -0.125),
        ] {
            set_float(&mut ge, opcode, value);
        }
        set(&mut ge, op::SCREEN_OFFSET_X, (2048 - 240) << 4);
        set(&mut ge, op::SCREEN_OFFSET_Y, (2048 - 136) << 4);
        set(&mut ge, op::FOG_ENABLE, 1);
        ge
    }

    /// The program's arithmetic, one multiplication after the other, ends
    /// where the CPU's composed matrix (`Transform`, which the raw draws used
    /// before) ends, up to float rounding: positions within a hundredth of a
    /// pixel, depth, fog and texture coordinates as close.
    #[test]
    fn the_blocks_give_what_the_composed_transform_gives() {
        let ge = scene();
        // 16-bit positions and texture coordinates.
        let format = VertexFormat(2 | 2 << 7);
        let inputs = crate::transform::TransformInputs::of(&ge, format);
        let composed = Transform::from_inputs(&inputs).rows().gpu_transform();
        let camera = inputs.raw_camera();
        let transform = RawTransform { world: inputs.raw_world(), ..RawTransform::default() };
        let uv = raw_uv(&camera, format);
        assert_eq!(uv, composed.raw_uv, "texture coordinates: the same products");
        let colors = RawColors::default();
        for (position, texcoord) in [([1000.0, -2000.0, 300.0], [100.0, 200.0]), ([-32768.0, 32767.0, 0.0], [0.0, 65535.0]), ([5.0, 7.0, -9000.0], [3.0, 4.0])] {
            let (model, _) = model_space(format, &[], position, &[], [0.0; 3]);
            let ours = clip_to_screen(&camera, &clip_vertex(&transform, &camera, &colors, &uv, model, texcoord, None));
            let theirs = composed.clip_to_screen(&composed.raw_clip_vertex(&colors, position, texcoord, None));
            let close = |a: f32, b: f32, limit: f32| (a - b).abs() <= limit * (1.0 + b.abs());
            assert!(close(ours.x, theirs.x, 1e-5) && close(ours.y, theirs.y, 1e-5), "{ours:?} {theirs:?}");
            assert!(close(ours.z, theirs.z, 1e-5) && close(ours.w, theirs.w, 1e-5), "{ours:?} {theirs:?}");
            assert!(close(ours.fog, theirs.fog, 1e-5), "{ours:?} {theirs:?}");
            assert_eq!((ours.u, ours.v), (theirs.u, theirs.v));
        }
    }

    /// Skinned in the program: one bone with a weight gives what folding
    /// weight x bone into the rows gave (`Transform::weighted_rows`).
    #[test]
    fn one_weighted_bone_skins_like_the_folded_rows() {
        let mut ge = scene();
        ge.matrices.bones[..12].copy_from_slice(&[0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 10.0, 0.0, -5.0]);
        // 8-bit weights, one of them; float positions.
        let format = VertexFormat(1 << 9 | 3 << 7);
        let inputs = crate::transform::TransformInputs::of(&ge, format);
        let weight = 248.0 / 128.0;
        let folded = Transform::from_inputs(&inputs).weighted_rows(weight).gpu_transform();
        let camera = inputs.raw_camera();
        let transform = RawTransform { world: inputs.raw_world(), ..RawTransform::default() };
        let bones: Vec<f32> = inputs.bones().iter().flat_map(|bone| crate::transform::rows_of_4x3(bone).into_iter().flatten()).collect();
        let colors = RawColors::default();
        let position = [1.5, -2.0, 0.75];
        let (model, _) = model_space(format, &bones, position, &[weight], [0.0; 3]);
        let ours = clip_position(&transform, &camera, model).0;
        let theirs = folded.raw_clip_vertex(&colors, position, [0.0; 2], None);
        for (a, b) in ours.into_iter().zip([theirs.x, theirs.y, theirs.z, theirs.w]) {
            assert!((a - b).abs() <= 1e-4 * (1.0 + b.abs()), "{ours:?} vs {theirs:?}");
        }
    }

    /// A draw with the inputs of the draw before gets its transform; new
    /// world matrices new ones; a camera shared by several objects is stored
    /// once.
    #[test]
    fn entries_are_written_only_for_new_inputs() {
        let mut frame: Box<Frame> = crate::frame::new_boxed_frame();
        let mut ge = scene();
        let format = VertexFormat(2 | 2 << 7);
        let mut cache = TransformCache::new();
        let mut geometry = frame.split().0;
        let first = raw_transform(&mut geometry, cache.get(&ge, format), None).unwrap();
        assert_eq!(raw_transform(&mut geometry, cache.get(&ge, format), None), Some(first), "same inputs");
        // Another object: a new entry with the same camera.
        let mut command = |opcode: u8, data: u32| ge.set_register(crate::command::Command((opcode as u32) << 24 | data));
        command(op::WORLD_MATRIX_NUMBER, 9);
        command(op::WORLD_MATRIX_DATA, 7.0f32.to_bits() >> 8);
        let second = raw_transform(&mut geometry, cache.get(&ge, format), None).unwrap();
        assert_eq!(second, first + 1);
        assert_eq!(geometry.last_raw_transform().unwrap().1.world[0][3], 7.0);
        drop(geometry);
        assert_eq!(frame.raw_camera_count, 1, "one camera");
        assert_eq!(frame.raw_transforms[0].camera, frame.raw_transforms[1].camera);
    }
}
