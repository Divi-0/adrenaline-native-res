//! `--trace`: prints the GE state at every draw of the replayed frame.

use nativeres_core::command::{op, Command};
use nativeres_core::display_list::CommandVisitor;
use nativeres_core::ge_state::{float24, GeState};
use nativeres_core::memory::PspMemory;
use nativeres_core::transform::Transform;

const TEXTURE_FUNCTIONS: [&str; 8] = ["modulate", "decal", "blend", "replace", "add", "?5", "?6", "?7"];
const BLEND_FACTORS: [&str; 11] = [
    "src/dst colour", "1-src/dst colour", "src alpha", "1-src alpha", "dst alpha",
    "1-dst alpha", "2 src alpha", "1-2 src alpha", "2 dst alpha", "1-2 dst alpha", "fixed",
];
const BLEND_EQUATIONS: [&str; 6] = ["add", "subtract", "reverse subtract", "min", "max", "abs diff"];
const COMPARE: [&str; 8] = ["never", "always", "==", "!=", "<", "<=", ">", ">="];

pub struct DrawTracer {
    enabled: bool,
    draw_number: u32,
    /// The transform inputs last printed: printed again when they change.
    transform_shown: String,
    /// How often each opcode ran (all commands, also with tracing off).
    pub opcode_counts: [u32; 256],
    /// REPLAY_CLUT_LOADS=1: print every CLUT_LOAD (address, 32-byte blocks).
    print_clut_loads: bool,
    /// For `--opcodes`: the opcode before, and which opcodes come right
    /// before a PRIM (how often PRIMs follow each other directly).
    previous_opcode: u8,
    before_prim: [u32; 256],
    /// REPLAY_LIGHTS=1: how the frame's lit draws light (`print_light_survey`).
    survey_lights: bool,
    light_kinds: std::collections::BTreeMap<String, u32>,
    /// Whether it needs to see every command (tracing, `--opcodes`, the
    /// REPLAY_* surveys). Otherwise it wants none, so `walk` runs the plain
    /// commands in its fast runs exactly as on the Vita (where the capture's
    /// tracer is `NoTrace`) and the regression suite tests that path.
    wants_all: bool,
}

impl DrawTracer {
    pub fn new(enabled: bool) -> Self {
        DrawTracer {
            enabled,
            draw_number: 0,
            transform_shown: String::new(),
            opcode_counts: [0; 256],
            print_clut_loads: std::env::var("REPLAY_CLUT_LOADS").is_ok(),
            previous_opcode: 0,
            before_prim: [0; 256],
            survey_lights: std::env::var("REPLAY_LIGHTS").is_ok(),
            light_kinds: std::collections::BTreeMap::new(),
            wants_all: enabled
                || std::env::args().any(|arg| arg == "--opcodes")
                || ["REPLAY_CLUT_LOADS", "REPLAY_LIGHTS", "REPLAY_OPS"].iter().any(|name| std::env::var(name).is_ok()),
        }
    }

    /// REPLAY_LIGHTS=1: the kinds of lighting the frame's PRIMs use, with
    /// how many PRIMs use each (GE registers 0x17..0x1B, 0x50..0x62).
    pub fn print_light_survey(&self) {
        if !self.survey_lights {
            return;
        }
        let total: u32 = self.light_kinds.values().sum();
        println!("lighting survey: {total} PRIMs");
        for (kind, count) in &self.light_kinds {
            println!("  {count:6}  {kind}");
        }
    }

    /// With the light survey: textured PRIMs by their mip levels ("tex ..."
    /// lines): the highest level (TEXMODE 0xC2 bits 16..18), the
    /// minification filter (0xC6 bits 0..2; 4..7 use mip levels) and the
    /// level mode (TEXLEVEL 0xC8 bits 0..1: auto, constant, slope) with its
    /// bias (bits 16..23, signed, 1/16 levels).
    fn survey_mip_levels(&mut self, ge: &GeState) {
        if ge.register(0x1E) & 1 == 0 {
            return; // texturing off
        }
        const FILTERS: [&str; 8] = ["nearest", "linear", "?2", "?3", "nearest+mip nearest", "linear+mip nearest", "nearest+mip linear", "linear+mip linear"];
        const MODES: [&str; 4] = ["auto", "const", "slope", "?3"];
        let max_level = (ge.register(0xC2) >> 16) & 7;
        let min_filter = ge.register(0xC6) & 7;
        let level = ge.register(0xC8);
        let bias = ((level >> 16) & 0xFF) as u8 as i8 as f32 / 16.0;
        let key = format!(
            "tex levels 0..{max_level}, min {}, mode {} bias {bias}",
            FILTERS[min_filter as usize],
            MODES[(level & 3) as usize]
        );
        *self.light_kinds.entry(key).or_default() += 1;
    }

    fn survey_light(&mut self, ge: &GeState) {
        let register = |opcode: u8| ge.register(opcode);
        let format = ge.vertex_format();
        if register(0x17) & 1 == 0 {
            *self.light_kinds.entry(format!("unlit through={}", format.is_through() as u8)).or_default() += 1;
            return;
        }
        const COMPUTATIONS: [&str; 4] = ["diffuse", "diffuse+specular", "powered diffuse", "?3"];
        const TYPES: [&str; 4] = ["directional", "point", "spot", "?3"];
        let mut lights = String::new();
        for light in 0..4u8 {
            if register(0x18 + light) & 1 == 0 {
                continue;
            }
            let kind = register(0x5F + light);
            lights += &format!(
                " L{light}:{}/{}",
                TYPES[((kind >> 8) & 3) as usize],
                COMPUTATIONS[(kind & 3) as usize]
            );
        }
        let kind = format!(
            "lit{lights} | lightmode={} materialupdate={:03b} reversenormal={} shade={} | normals={} colors={} weights={}",
            register(0x5E) & 1,
            register(0x53) & 7,
            register(0x51) & 1,
            register(0x50) & 1,
            format.has_normals() as u8,
            (format.color_format_bits() != 0) as u8,
            format.weight_count(),
        );
        *self.light_kinds.entry(kind).or_default() += 1;
    }
}

impl DrawTracer {
    /// 3D draws: the transform inputs (when they changed) and the first
    /// vertex (all of them for draws of up to 4), raw and on the screen.
    fn trace_transform<M: PspMemory>(&mut self, ge: &GeState, memory: &mut M, count: u32) {
        let float = |opcode: u8| float24(ge.register(opcode));
        let skinning = ge.vertex_format();
        let bones = if skinning.has_weights() {
            (0..skinning.weight_count() as usize)
                .map(|bone| format!("\n    bone {bone}     {:?}", ge.matrices.bone(bone.min(7))))
                .collect::<String>()
        } else {
            String::new()
        };
        let inputs = format!(
            "    world      {:?}{bones}
    view       {:?}
    projection {:?}
    viewport scale {} {} centre {} {}, offset {} {}, tex scale {} {} offset {} {}, lighting {}, map mode {}",
            ge.matrices.world, ge.matrices.view, ge.matrices.projection,
            float(op::VIEWPORT_X_SCALE), float(op::VIEWPORT_Y_SCALE),
            float(op::VIEWPORT_X_CENTER), float(op::VIEWPORT_Y_CENTER),
            (ge.register(op::SCREEN_OFFSET_X) & 0xFFFF) as f32 / 16.0, (ge.register(op::SCREEN_OFFSET_Y) & 0xFFFF) as f32 / 16.0,
            float(op::TEXTURE_SCALE_U), float(op::TEXTURE_SCALE_V), float(op::TEXTURE_OFFSET_U), float(op::TEXTURE_OFFSET_V),
            ge.register(op::LIGHTING_ENABLE) & 1, ge.register(op::TEXTURE_MAP_MODE) & 3,
        );
        // Texgen (map mode 1): its source (bits 8-9) and matrix.
        let inputs = if ge.register(op::TEXTURE_MAP_MODE) & 3 == 1 {
            format!("{inputs}, texgen source {} matrix {:?}", (ge.register(op::TEXTURE_MAP_MODE) >> 8) & 3, ge.matrices.texgen)
        } else {
            inputs
        };
        if inputs != self.transform_shown {
            println!("{inputs}");
            self.transform_shown = inputs;
        }
        let format = ge.vertex_format();
        let layout = ge.vertex_layout();
        let shown = if count <= 4 { count } else { 1 };
        for index in 0..shown {
            let address = ge.vertex_address + index * layout.size;
            if let Some(vertex) = format.read(memory, address, layout) {
                let screen = Transform::new(ge, format).apply(&vertex, vertex.color.unwrap_or_else(|| ge.material_color()));
                println!("    vertex {index} {:?} -> {:?}", vertex, screen);
            }
        }
    }
}

impl CommandVisitor for DrawTracer {
    /// Tracing prints each PRIM with its vertices: one by one then.
    fn batches_prims(&self) -> bool {
        !self.enabled
    }

    fn wants(&self, _opcode: u8) -> bool {
        self.wants_all
    }

    fn visit<M: PspMemory>(&mut self, command: Command, ge: &GeState, memory: &mut M) {
        self.opcode_counts[command.opcode() as usize] += 1;
        if command.opcode() == op::PRIM {
            self.before_prim[self.previous_opcode as usize] += 1;
            if self.survey_lights {
                self.survey_light(ge);
                self.survey_mip_levels(ge);
            }
        }
        self.previous_opcode = command.opcode();
        if self.enabled && std::env::var("REPLAY_OPS").is_ok_and(|ops| {
            ops == "all" || ops.split(',').any(|op| u8::from_str_radix(op.trim(), 16) == Ok(command.opcode()))
        }) {
            println!("op {:08x} (after prim {})", command.0, self.draw_number);
        }
        if self.print_clut_loads && command.opcode() == op::CLUT_LOAD {
            println!("clut_load {:08x} {}", ge.clut_address(), command.data() & 0x3F);
        }
        if !self.enabled || command.opcode() != op::PRIM {
            return;
        }
        let data = command.data();
        let register = |opcode: u8| ge.register(opcode);
        let enabled = |opcode: u8| register(opcode) & 1 != 0;
        let mut line = format!(
            "prim {:4}: type {} x{:<4} vtype {:06x} through {} at {:08x} idx {:08x}",
            self.draw_number,
            (data >> 16) & 7,
            data & 0xFFFF,
            register(op::VERTEX_TYPE),
            ge.is_through_mode() as u8,
            ge.vertex_address,
            ge.index_address,
        );
        if ge.is_clear_mode() {
            line += &format!(" CLEAR({:03x})", register(op::CLEAR_MODE));
        }
        if ge.is_texture_enabled() {
            let function = register(op::TEXTURE_FUNCTION);
            line += &format!(
                " | tex {:08x} fmt {} func {}{}{}",
                ge.texture_address(),
                ge.texture_format(),
                TEXTURE_FUNCTIONS[(function & 7) as usize],
                if function & 0x100 != 0 { " rgba" } else { " rgb" },
                if function & 0x10000 != 0 { " x2" } else { "" },
            );
        }
        if ge.is_blend_enabled() {
            let blend = register(op::BLEND_MODE);
            line += &format!(
                " | blend {} * src {} dst {}",
                BLEND_EQUATIONS.get(((blend >> 8) & 7) as usize).unwrap_or(&"?"),
                BLEND_FACTORS.get((blend & 0xF) as usize).unwrap_or(&"?"),
                BLEND_FACTORS.get(((blend >> 4) & 0xF) as usize).unwrap_or(&"?"),
            );
            if blend & 0xF == 0xA || (blend >> 4) & 0xF == 0xA {
                line += &format!(" fixA {:06x} fixB {:06x}", register(op::BLEND_FIXED_A), register(op::BLEND_FIXED_B));
            }
        }
        if enabled(op::ALPHA_TEST_ENABLE) {
            let test = register(op::ALPHA_TEST);
            line += &format!(" | alpha {} {:02x}", COMPARE[(test & 7) as usize], (test >> 8) & 0xFF);
        }
        if enabled(op::DEPTH_TEST_ENABLE) {
            line += &format!(" | depth {}", COMPARE[(register(op::DEPTH_TEST) & 7) as usize]);
        }
        if enabled(op::FOG_ENABLE) && !ge.is_through_mode() {
            line += &format!(
                " | fog end {} scale {} colour {:06x}",
                float24(register(op::FOG_END)),
                float24(register(op::FOG_SCALE)),
                register(op::FOG_COLOR) & 0xFF_FFFF
            );
        }
        if enabled(op::STENCIL_TEST_ENABLE) {
            line += " | stencil";
        }
        if register(op::COLOR_MASK) & 0xFF_FFFF != 0 {
            line += &format!(" | colour mask {:06x}", register(op::COLOR_MASK) & 0xFF_FFFF);
        }
        line += &format!(" | material {:08x}", ge.material_color());
        println!("{line}");
        if !ge.is_through_mode() {
            self.trace_transform(ge, memory, data & 0xFFFF);
        }
        self.draw_number += 1;
    }
}

impl DrawTracer {
    /// `--opcodes`: the most frequent opcodes of the frame.
    pub fn print_opcode_counts(&self) {
        let mut counts: Vec<(usize, u32)> =
            self.opcode_counts.iter().copied().enumerate().filter(|&(_, count)| count > 0).collect();
        counts.sort_by(|a, b| b.1.cmp(&a.1));
        let total: u32 = counts.iter().map(|&(_, count)| count).sum();
        println!("opcodes: {total} commands");
        for (opcode, count) in counts.iter().take(40) {
            println!("  {opcode:02x}: {count:7} ({:.1}%)", 100.0 * *count as f64 / total.max(1) as f64);
        }
        let mut before: Vec<(usize, u32)> =
            self.before_prim.iter().copied().enumerate().filter(|&(_, count)| count > 0).collect();
        before.sort_by(|a, b| b.1.cmp(&a.1));
        let line: Vec<String> = before.iter().take(12).map(|(opcode, count)| format!("{opcode:02x}:{count}")).collect();
        println!("before PRIM: {}", line.join(" "));
    }
}
