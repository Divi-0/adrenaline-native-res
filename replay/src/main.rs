//! Replays one PSP frame dumped on the Vita and renders it on the PC.
//!
//!     replay <dump.bin> <out.png> [--trace]
//!     replay <dump.bin> --bench <runs> [walk|stats|produce|full]
//!
//! 3D draws are transformed the way the Vita does it, by default on the
//! GPU (`frame::GpuTransform`, here `GpuTransform::to_screen`); with
//! `--cpu-transform` (or `NATIVERES_CPU_TRANSFORM=1`) the capture
//! transforms every vertex itself, as before. Both give the same picture.
//!
//! Draws for the GPU transform go through a vertex cache as on the Vita
//! (`--no-cache` or `NATIVERES_NO_CACHE=1`: none; `--raw` or
//! `NATIVERES_RAW=1`: 3D runs as raw PSP vertices, `raw_arena`). With `--twice` the frame is replayed a second
//! time with the cache already filled, and that second frame is drawn.
//! `--gather`: indexed raw draws gather their vertices (the old way).
//!
//! With `--trace`, prints the GE state of every draw (texture function,
//! blending, tests, ...), to explain differences.
//!
//! If the dump contains the original picture, also writes
//! `<out>-original.png` (enlarged 2x) and `<out>-diff.png` (differences
//! amplified, black = identical), and prints the average difference per
//! screen region. `--reference-vram`: compared with the shown framebuffer
//! in the dump's memory instead (Sony's own picture of the dumped frame,
//! where the dump holds it).
//!
//! `--ge-filter`: every textured draw filters as the GE, as the Vita draws
//! the games of core game_quirks.rs `PSP_DEPTH_GAMES` (God of War).
//!
//! Runs the same `nativeres-core` code as the Vita: restores the capture
//! state from the start of the frame, feeds the frame's events through
//! `Capture`, then draws the resulting frame description with a small
//! software renderer that imitates what the Vita side does (its GPU
//! shaders, alpha blending, 2x scale). Differences between this image and a
//! Vita screenshot point at the Vita-specific code; wrong content in both
//! points at the core.

mod compare;
mod dump_file;
mod png;
mod raster;
mod sublists;
mod trace;

use std::process::ExitCode;

use nativeres_core::bench::{self, Mode};
use nativeres_core::block_cache::{BlockCache, BlockStore, CAPTURE_BLOCKS, SMALL_BLOCKS};
use nativeres_core::capture::{Capture, Event};
use nativeres_core::dump::DumpView;
use nativeres_core::frame::{Frame, FrameVertex};

use nativeres_core::vertex_cache::{VertexCache, ARENA_VERTICES};
use nativeres_core::report;
use nativeres_core::snapshot::Reader;

use dump_file::DumpFile;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let trace = args.iter().any(|arg| arg == "--trace");
    // NATIVERES_LIGHTS=0: lit draws with their vertex colours as they are
    // (`lighting::LIGHTS_ON` off), for comparing costs.
    if std::env::var("NATIVERES_LIGHTS").is_ok_and(|value| value == "0") {
        nativeres_core::lighting::LIGHTS_ON.store(false, std::sync::atomic::Ordering::Relaxed);
    }
    // NATIVERES_TEXGEN_RUNS=0: texgen draws vertex by vertex as before, 2:
    // both ways and compared (`recorder::TEXGEN_RUNS`; --bench prints the
    // counts); the frames must be the same.
    if let Some(mode) = std::env::var("NATIVERES_TEXGEN_RUNS").ok().and_then(|value| value.parse::<u8>().ok()) {
        nativeres_core::recorder::TEXGEN_RUNS.store(mode, std::sync::atomic::Ordering::Relaxed);
    }
    // NATIVERES_CYCLES=<mask>: the core's cycle parts (`cycles::MEASURED`,
    // bit = part; 0 = all) timed with the PC's time stamp counter, printed
    // per frame after --bench (x86 cycles: they only locate, the Vita judges).
    if let Some(mask) = std::env::var("NATIVERES_CYCLES").ok().and_then(|value| value.parse::<u32>().ok()) {
        fn tsc() -> u32 {
            #[cfg(target_arch = "x86_64")]
            // Safe: reads the time stamp counter.
            return unsafe { core::arch::x86_64::_rdtsc() } as u32;
            #[cfg(not(target_arch = "x86_64"))]
            return 0;
        }
        nativeres_core::cycles::set_clock(Some(tsc));
        let mask = if mask == 0 { nativeres_core::cycles::ALL_PARTS } else { mask };
        nativeres_core::cycles::MEASURED.store(mask, std::sync::atomic::Ordering::Relaxed);
    }
    // NATIVERES_EXPERIMENT=<mask>: the debug server's `benchexp` bits
    // (`bench::EXPERIMENT`), for A/Bs of the same frame.
    if let Some(mask) = std::env::var("NATIVERES_EXPERIMENT").ok().and_then(|value| value.parse::<u32>().ok()) {
        nativeres_core::bench::EXPERIMENT.store(mask, std::sync::atomic::Ordering::Relaxed);
    }
    if args.iter().any(|arg| arg == "--sublists") {
        return match print_sublists(&args[1]) {
            Ok(()) => ExitCode::SUCCESS,
            Err(message) => {
                eprintln!("error: {message}");
                ExitCode::FAILURE
            }
        };
    }
    if let Some(at) = args.iter().position(|arg| arg == "--bench") {
        let runs = args.get(at + 1).and_then(|runs| runs.parse().ok()).unwrap_or(100);
        let only = args.get(at + 2).map(String::as_str);
        return match bench(&args[1], runs, only) {
            Ok(()) => ExitCode::SUCCESS,
            Err(message) => {
                eprintln!("error: {message}");
                ExitCode::FAILURE
            }
        };
    }
    let paths: Vec<&String> = args.iter().skip(1).filter(|arg| !arg.starts_with("--")).collect();
    if paths.len() != 2 {
        eprintln!("usage: replay <dump.bin> <out.png> [--trace]");
        return ExitCode::FAILURE;
    }
    match run(paths[0], paths[1], trace) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}

/// Stage B's thread (`--stage-thread`) polls; nothing to wake.
static PC_STAGE_HOOKS: nativeres_core::walk_stream::WalkHooks =
    nativeres_core::walk_stream::WalkHooks { wake_reader: || {}, wait: std::thread::yield_now };

fn stage_thread() -> bool {
    std::env::args().any(|arg| arg == "--stage-thread") || std::env::var("NATIVERES_STAGE_THREAD").is_ok()
}

/// Whether the draw stream's copier copies the vertex memory (as the Vita's
/// copy thread, here in this thread): `--copier` or NATIVERES_COPIER set.
fn copier() -> bool {
    std::env::args().any(|arg| arg == "--copier") || std::env::var("NATIVERES_COPIER").is_ok()
}

/// Whether the capture runs in two stages (`walk_stream`), both in this
/// thread: `--pipeline` or NATIVERES_PIPELINE set.
fn pipelined() -> bool {
    std::env::args().any(|arg| arg == "--pipeline") || std::env::var("NATIVERES_PIPELINE").is_ok()
}

/// Whether 3D draws are left to the GPU transform (see the module comment).
fn gpu_transform() -> bool {
    !std::env::args().any(|arg| arg == "--cpu-transform") && std::env::var("NATIVERES_CPU_TRANSFORM").is_err()
}

/// A vertex cache (`nativeres_core::vertex_cache`) for the session, unless
/// `--no-cache`.
fn new_vertex_cache() -> Option<&'static mut VertexCache> {
    if std::env::args().any(|arg| arg == "--no-cache") || std::env::var("NATIVERES_NO_CACHE").is_ok() {
        return None;
    }
    let arena = Box::leak(vec![FrameVertex::default(); ARENA_VERTICES].into_boxed_slice());
    let cache = Box::leak(Box::new(VertexCache::new(arena)));
    // Raw PSP vertices for the GPU (`--raw` or NATIVERES_RAW=1, as on the
    // Vita by default): a raw arena as big as the Vita's. Without: the old
    // path (the regression hashes are the old path's).
    let raw = std::env::args().any(|arg| arg == "--raw") || std::env::var("NATIVERES_RAW").is_ok_and(|value| value == "1");
    nativeres_core::raw_arena::ENABLED.store(raw, std::sync::atomic::Ordering::Relaxed);
    nativeres_core::raw_arena::SPRITES.store(raw, std::sync::atomic::Ordering::Relaxed);
    // `--gather`: indexed raw draws gather each corner's vertex, as before
    // their indices were kept (`raw_arena::INDEXED_DRAWS`).
    let gather = std::env::args().any(|arg| arg == "--gather");
    nativeres_core::raw_arena::INDEXED_DRAWS.store(!gather, std::sync::atomic::Ordering::Relaxed);
    // NATIVERES_RAW_TEXGEN=1: texgen draws raw where they can be
    // (`raw_arena::TEXGEN`, off by default as on the Vita).
    let raw_texgen = std::env::var("NATIVERES_RAW_TEXGEN").is_ok_and(|value| value == "1");
    nativeres_core::raw_arena::TEXGEN.store(raw_texgen, std::sync::atomic::Ordering::Relaxed);
    // NATIVERES_POINT_LIGHTS=0: point-lit draws lit on the CPU, as before
    // 2026-11-22 (`lighting::POINT_LIGHTS_GPU`, on by default as on the Vita).
    let point_lights = std::env::var("NATIVERES_POINT_LIGHTS").map_or(true, |value| value != "0");
    nativeres_core::lighting::POINT_LIGHTS_GPU.store(point_lights, std::sync::atomic::Ordering::Relaxed);
    // `--ge-light` (or REPLAY_GE_LIGHT=1): GPU-lit colours in the GE's
    // integer arithmetic, as God of War's program set (`lighting::GE_LIGHTING`).
    let ge_light = std::env::args().any(|arg| arg == "--ge-light") || std::env::var_os("REPLAY_GE_LIGHT").is_some();
    nativeres_core::lighting::GE_LIGHTING.store(ge_light, std::sync::atomic::Ordering::Relaxed);
    // Like the Vita once its alpha-test program exists: textured draws are
    // alpha-tested in the shader (this rasterizer tests every draw).
    // NATIVERES_BLEND_ALPHA_TEST=1: blending stands in as before (A/B).
    let shader_alpha_test = std::env::var("NATIVERES_BLEND_ALPHA_TEST").is_err();
    nativeres_core::blend::ALPHA_TEST_IN_SHADER.store(shader_alpha_test, std::sync::atomic::Ordering::Relaxed);
    if raw {
        cache.set_raw_arena(Box::leak(vec![0u8; RAW_ARENA_BYTES].into_boxed_slice()));
    }
    Some(cache)
}

/// The capture thread's own lists for the raw draws it writes itself
/// (`frame_builder`, as on the Vita), unless `--jobs` (or NATIVERES_JOBS):
/// every draw then goes through the draw stream's job queue, as before.
fn new_direct_lists() -> Option<&'static mut nativeres_core::frame_builder::DirectLists> {
    if std::env::args().any(|arg| arg == "--jobs") || std::env::var("NATIVERES_JOBS").is_ok() {
        return None;
    }
    let layout = std::alloc::Layout::new::<nativeres_core::frame_builder::DirectLists>();
    // Safe: the lists are valid as zero bytes (`DirectLists::init_zeroed`)
    // and the layout is non-zero-sized; kept for the whole run.
    unsafe { Some(&mut *(std::alloc::alloc_zeroed(layout) as *mut nativeres_core::frame_builder::DirectLists)) }
}

/// The raw arena's size, as the Vita's (nativeres_render.c RAW_ARENA_BYTES).
const RAW_ARENA_BYTES: usize = 6 * 1024 * 1024;


/// An empty frame description on the heap (it is ~2.2 MiB).
fn new_frame() -> Box<Frame> {
    let layout = std::alloc::Layout::new::<Frame>();
    // Safe: Frame consists of integers and floats only, for which all-zero
    // bytes are valid values, and the layout is non-zero-sized.
    unsafe { Box::from_raw(std::alloc::alloc_zeroed(layout) as *mut Frame) }
}

fn run(dump_path: &str, png_path: &str, trace: bool) -> Result<(), String> {
    let bytes = std::fs::read(dump_path).map_err(|error| format!("{dump_path}: {error}"))?;
    let mut dump = DumpFile::parse(&bytes)?;
    println!(
        "dump: {} events, {} memory blocks of 1 KiB",
        dump.events.len(),
        dump.memory.block_count()
    );

    let mut capture = Capture::new();
    capture.set_gpu_transform(gpu_transform());
    capture.set_vertex_cache(new_vertex_cache());
    capture.set_direct_lists(new_direct_lists());
    capture.set_pipelined(pipelined());
    // `--copier`: the vertex memory goes through the draw stream's copier
    // (copied before the jobs are recorded, as the Vita's copy_thread.rs).
    if copier() {
        static COPIER: nativeres_core::draw_stream::CopierHooks = nativeres_core::draw_stream::CopierHooks { wake: || {} };
        dump_file::use_for_copier(&dump.memory);
        capture.set_stream_reader(Some(dump_file::COPIER_READER));
        capture.set_stream_copier(Some(&COPIER));
    }

    let mut frame = new_frame();
    let mut stats = None;
    let rounds = if std::env::args().any(|arg| arg == "--twice") { 2 } else { 1 };
    // Traces the round that is drawn (the last).
    let mut tracer = trace::DrawTracer::new(trace);
    let mut silent = trace::DrawTracer::new(false);
    // Stage B on a thread of its own (`--stage-thread` or
    // NATIVERES_STAGE_THREAD), as on the Vita.
    let threaded = stage_thread();
    if threaded {
        capture.set_pipelined(true);
        capture.set_stage_hooks(Some(&PC_STAGE_HOOKS));
    }
    let stop = std::sync::atomic::AtomicBool::new(false);
    let capture_address = &mut capture as *mut Capture as usize;
    let shared_memory = &dump.memory;
    std::thread::scope(|scope| -> Result<(), String> {
        if threaded {
            scope.spawn(|| {
                // Stage B's own store, small as on the Vita (stage_thread.rs).
                let mut store = Box::new(BlockStore::<SMALL_BLOCKS>::new());
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    let mut memory = BlockCache::new(&mut *store, shared_memory);
                    // Safe: the capture outlives the scope; this is its only
                    // stage B thread (`Capture::run_stage_b_thread`).
                    let run = unsafe { Capture::run_stage_b_thread(capture_address as *mut Capture, &mut memory) };
                    if run == nativeres_core::capture::StageBRun::Nothing {
                        std::thread::yield_now();
                    }
                }
            });
        }
        let result = (|| -> Result<(), String> {
            let mut store = Box::new(BlockStore::<CAPTURE_BLOCKS>::new());
            for round in 0..rounds {
                capture
                    .load_state(&mut Reader::new(&dump.snapshot))
                    .ok_or("the dump's capture state can't be restored (other version?)")?;
                frame.clear();
                let tracer = if round + 1 == rounds { &mut tracer } else { &mut silent };
                for &(event_type, args) in &dump.events {
                    let Some(event) = Event::from_ring(event_type, args) else {
                        continue;
                    };
                    // A fresh block cache per event, as on the Vita.
                    let mut memory = BlockCache::new(&mut *store, &dump.memory);
                    if let Some(finished) = capture.handle_traced(event, &mut memory, Some(&mut frame), &mut *tracer) {
                        stats = Some(finished);
                    }
                }
                if round + 1 == rounds && rounds > 1 {
                    if let Some(stats) = &stats {
                        let cache = stats.cache;
                        println!(
                            "second round: vertex cache {} hits, {} misses, {} stored, {} for the CPU; raw runs {} ({} lit); transforms {}, light sets {}, raw colours {}, bone floats {}",
                            cache.hits, cache.misses, cache.stored, cache.cpu_only, cache.raw_runs, cache.raw_lit,
                            frame.transform_count, frame.light_set_count, frame.raw_colors_count, frame.bone_float_count
                        );
                    }
                }
            }
            Ok(())
        })();
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        result
    })?;
    let stats = stats.ok_or("the dump contains no complete frame")?;
    if capture.pipelined() {
        let (sent, dropped) = capture.write_filter_counts();
        println!("walk stream: {sent} plain register writes passed on, {dropped} left out (no change)");
    }
    if stats.approximated_draws > 0 {
        let reasons: Vec<String> = nativeres_core::recorder::approx::NAMES
            .iter()
            .zip(stats.approximated_by_reason)
            .filter(|(_, count)| *count > 0)
            .map(|(name, count)| format!("{name} {count}"))
            .collect();
        println!("approximated: {} ({})", stats.approximated_draws, reasons.join(", "));
    }
    if std::env::args().any(|arg| arg == "--vram") {
        // Textures in VRAM (what the Vita side asks the PSP plugin to copy)
        // and the framebuffers the frame draws into.
        let framebuffers: std::collections::BTreeSet<u32> = frame.draws().iter().map(|draw| draw.framebuffer).collect();
        println!("vram: framebuffers drawn {:08x?}", framebuffers);
        for texture in frame.textures() {
            let address = nativeres_core::memory::fold_vram_mirror(texture.address & 0x0FFF_FFFF);
            if (0x0400_0000..0x0420_0000).contains(&address) {
                let size = nativeres_core::texture::data_size(texture).unwrap_or(0);
                println!(
                    "vram: texture {address:08x} {}x{} format {} stride {}: {} KiB",
                    texture.width, texture.height, texture.format, texture.buffer_width, size / 1024
                );
            }
        }
    }
    if std::env::args().any(|arg| arg == "--runs") {
        print_index_runs(&frame);
    }
    if std::env::args().any(|arg| arg == "--palettes") {
        // The palette tables the Vita fills for this frame (one per distinct
        // palette, as nativeres_render.c SamePalette tells them apart).
        let mut tables = std::collections::BTreeMap::<u8, std::collections::BTreeSet<(u32, u32, u32, u16)>>::new();
        for texture in frame.textures().iter().filter(|texture| nativeres_core::texture::is_palette_format(texture.format)) {
            tables.entry(texture.format).or_default().insert((
                texture.clut_format,
                texture.palette_hash,
                texture.palette_word_count,
                texture.flags & nativeres_core::frame::texture_flags::IGNORE_ALPHA,
            ));
        }
        for (format, set) in &tables {
            println!("palette tables: format {format}: {}", set.len());
        }
        println!(
            "frame palettes: {} (by address and length), {} words",
            frame.palettes().len(),
            frame.palette_word_count
        );
    }
    if std::env::args().any(|arg| arg == "--opcodes") {
        tracer.print_opcode_counts();
    }
    {
        tracer.print_light_survey();
    }

    let mut line = String::new();
    report::write_frame(&mut line, &stats).map_err(|_| "report failed")?;
    println!("frame: {line}");
    // How full the frame's tables for the GPU are: a run that finds one full
    // is transformed / lit on the CPU instead.
    println!(
        "gpu tables: transforms {} of {}, light sets {} of {}, raw transforms {} of {}, raw cameras {} of {}, bone floats {} of {}",
        frame.transform_count,
        nativeres_core::frame::MAX_TRANSFORMS,
        frame.light_set_count,
        nativeres_core::frame::MAX_LIGHT_SETS,
        frame.raw_transform_count,
        nativeres_core::frame::MAX_RAW_TRANSFORMS,
        frame.raw_camera_count,
        nativeres_core::frame::MAX_RAW_CAMERAS,
        frame.bone_float_count,
        nativeres_core::frame::MAX_BONE_FLOATS
    );
    let broken = frame.unfogged_draws_with_fog();
    if broken > 0 {
        println!("BROKEN: {broken} draws without fog have vertices with fog (Frame::unfogged_draws_with_fog)");
    }
    println!(
        "description: {} draws, {} vertices, {} indices, {} textures, overflow {}, hash {:016x}",
        frame.draws().len(),
        frame.vertices().len(),
        frame.corner_count(),
        frame.textures().len(),
        frame.is_overflowed() as u32,
        frame.description_hash()
    );
    // The capture's texture samples (`recorder::sample_texture`; not in the
    // description hash): the same with and without a copier thread.
    let samples: Vec<u32> = (0..frame.textures().len()).map(|index| frame.texture_sample(index)).collect();
    let samples_hash = samples.iter().fold(0xCBF2_9CE4_8422_2325u64, |hash, &sample| (hash ^ sample as u64).wrapping_mul(0x100_0000_01B3));
    println!(
        "texture samples: {} of {} textures sampled, hash {:016x}",
        samples.iter().filter(|&&sample| sample != 0).count(),
        samples.len(),
        samples_hash
    );
    let gpu_draws: Vec<_> = frame.draws().iter().filter(|draw| draw.transform != 0).collect();
    println!(
        "gpu transform: {} transforms, {} draws, {} of {} vertices",
        frame.transform_count,
        gpu_draws.len(),
        gpu_draws.iter().map(|draw| draw.vertex_count).sum::<u32>(),
        frame.vertices().len()
    );
    println!("vertex hash: {:016x}", frame.vertex_hash());
    if trace {
        // The frame's textures, as the Vita looks them up.
        for (index, texture) in frame.textures().iter().enumerate() {
            println!(
                "texture {index:3}: {:08x} {}x{} buffer width {} format {} swizzled {} data {} bytes",
                texture.address,
                texture.width,
                texture.height,
                texture.buffer_width,
                texture.format,
                texture.swizzled,
                nativeres_core::texture::data_size(texture).unwrap_or(0)
            );
        }
        // The recorded draws, as the Vita gets them.
        for (index, draw) in frame.draws().iter().enumerate() {
            let corners: Vec<String> = frame.screen_corners(draw).take(3).map(|v| format!("({:.1},{:.1} z {:.6} uv {:.3},{:.3} c {:08x})", v.x, v.y, v.z, v.u, v.v, v.color)).collect();
            println!(
                "draw {index:3}: fb {:08x} tex {} flags {:x} blend {:08x} tests {:x} scissor {},{}-{},{} {} vertices {}",
                draw.framebuffer, draw.texture, draw.flags, draw.blend, draw.tests, draw.scissor_left, draw.scissor_top, draw.scissor_right,
                draw.scissor_bottom, frame.draw_index_count(draw), corners.join(" ")
            );
        }
    }

    let image = raster::render_frame(&frame, &mut dump.memory, dump.original.as_ref());
    raster::print_blend_stats();
    {
        // What debug skinwatch / dump-glitch measure on the Vita.
        let (edge, draw, triangles) = frame.longest_skinned_edge();
        println!("skinned: longest edge {edge} px in draw {} ({triangles} skinned triangles)", draw as i32);
    }
    write_png(png_path, &image)?;

    // `--reference-vram`: compared with the shown framebuffer as the dump's
    // memory holds it (Sony's own picture of this frame, where the dump has
    // it: God of War's screen) instead of the picture the dump stored, which
    // can be another frame where the game animates (God of War's title: the
    // fire glow, the menu highlight; DESIGN "fifty-ninth session").
    let vram_reference = if std::env::args().any(|arg| arg == "--reference-vram") {
        let shown = stats.displayed.ok_or("--reference-vram: the frame shows no framebuffer")?;
        let picture = dump.memory.framebuffer_picture(shown.address, shown.stride, shown.pixel_format, raster::WIDTH / 2, raster::HEIGHT / 2);
        Some(picture.ok_or("--reference-vram: the dump doesn't hold the shown framebuffer")?)
    } else {
        None
    };
    if let Some(original) = vram_reference.as_ref().or(dump.original.as_ref()) {
        let enlarged = compare::enlarge(original, raster::WIDTH, raster::HEIGHT);
        let base = png_path.trim_end_matches(".png");
        write_png(&format!("{base}-original.png"), &enlarged)?;
        write_png(&format!("{base}-diff.png"), &compare::difference_image(&image, &enlarged))?;
        compare::print_region_differences(&image, &enlarged, raster::WIDTH, raster::HEIGHT);
    } else {
        println!("(no original picture in this dump)");
    }
    Ok(())
}

/// `--sublists`: the frame's sub-list calls (see `sublists`).
fn print_sublists(dump_path: &str) -> Result<(), String> {
    let bytes = std::fs::read(dump_path).map_err(|error| format!("{dump_path}: {error}"))?;
    let dump = DumpFile::parse(&bytes)?;
    let mut capture = Capture::new();
    capture.set_gpu_transform(gpu_transform());
    let mut frame = new_frame();
    capture
        .load_state(&mut Reader::new(&dump.snapshot))
        .ok_or("the dump's capture state can't be restored (other version?)")?;
    let mut sublists = sublists::SubLists::new();
    let mut store = Box::new(BlockStore::<CAPTURE_BLOCKS>::new());
    for &(event_type, args) in &dump.events {
        let Some(event) = Event::from_ring(event_type, args) else {
            continue;
        };
        let mut memory = BlockCache::new(&mut *store, &dump.memory);
        capture.handle_traced(event, &mut memory, Some(&mut frame), &mut sublists);
    }
    sublists.print();
    Ok(())
}

/// Times the capture over the dump's events, `runs` times per mode, with
/// the same code a benchmark on the Vita uses (`core::bench`).
/// `--runs`: the frame's index runs (`draw_flags::INDEX_RUNS`) by primitive
/// type, with their triangles, and how many GPU draw calls they would take
/// drawn straight from fixed index tables: list and sprite runs that follow
/// on from the run before (same type, next vertex) join it; strips and fans
/// are one call each.
fn print_index_runs(frame: &nativeres_core::frame::Frame) {
    use nativeres_core::command::primitive;
    use nativeres_core::frame::{draw_flags, run_triangles};
    let mut by_type = std::collections::BTreeMap::<u16, (u32, u32)>::new();
    let (mut draws, mut calls, mut corners) = (0u32, 0u32, 0u32);
    let indices = frame.indices();
    for draw in frame.draws().iter().filter(|draw| draw.flags & draw_flags::INDEX_RUNS != 0) {
        draws += 1;
        let runs = &indices[draw.first_index as usize..][..draw.index_count as usize];
        let mut next: Option<(u16, u32)> = None; // type and vertex a joining run starts at
        for run in runs.chunks_exact(3) {
            let (kind, count, base) = (run[0], run[1], run[2] as u32);
            let triangles = run_triangles(kind, count);
            let entry = by_type.entry(kind).or_default();
            entry.0 += 1;
            entry.1 += triangles;
            corners += 3 * triangles;
            let vertices = match kind as u32 {
                primitive::RECTANGLES => 4 * count as u32,
                _ => count as u32,
            };
            let joinable = matches!(kind as u32, primitive::TRIANGLES | primitive::RECTANGLES);
            if !(joinable && next == Some((kind, base))) {
                calls += 1;
            }
            next = joinable.then_some((kind, base + vertices));
        }
    }
    let name = |kind: u16| match kind as u32 {
        primitive::TRIANGLES => "lists",
        primitive::TRIANGLE_STRIP => "strips",
        primitive::TRIANGLE_FAN => "fans",
        primitive::RECTANGLES => "sprites",
        _ => "other",
    };
    for (kind, (runs, triangles)) in &by_type {
        println!("index runs: {} {runs} runs, {triangles} triangles", name(*kind));
    }
    println!("index runs: {draws} draws, {corners} corners written today, {calls} draw calls from tables");
}

fn bench(dump_path: &str, runs: u32, only: Option<&str>) -> Result<(), String> {
    let bytes = std::fs::read(dump_path).map_err(|error| format!("{dump_path}: {error}"))?;
    let dump = DumpView::parse(&bytes).ok_or("not a complete NativeRes dump (or another version)")?;
    let mut index = vec![0u32; dump.block_count()];
    let sorted = dump.sort_blocks(&mut index).ok_or("block index too small")?;
    let mut capture = Box::new(Capture::new());
    capture.set_gpu_transform(gpu_transform());
    capture.set_vertex_cache(new_vertex_cache());
    // The raw draws written by the capture thread itself, as on the Vita.
    capture.set_direct_lists(new_direct_lists());
    let mut frame = new_frame();
    let mut store = Box::new(BlockStore::new());
    let modes = [("walk", Mode::WalkOnly), ("stats", Mode::Stats), ("produce", Mode::Produce), ("full", Mode::Full)];
    for (name, mode) in modes.into_iter().filter(|(name, _)| only.is_none_or(|only| only == *name)) {
        let start = std::time::Instant::now();
        let totals = bench::run(&dump, sorted, mode, runs, &mut capture, &mut frame, &mut store)
            .ok_or("the dump's capture state can't be restored (other version?)")?;
        let per_run = start.elapsed() / runs;
        println!(
            "{name:>12}: {per_run:>10.1?} per frame, {} copies (blocks + lines), {} draws, {} vertices ({runs} runs)",
            totals.copies, totals.draws, totals.vertices
        );
        if mode == Mode::Full {
            // The hashes the Vita's capture of this dump must give too.
            println!("{:>12}  hashes {:016x} {:016x}", "", frame.description_hash(), frame.vertex_hash());
        }
    }
    if nativeres_core::cycles::MEASURED.load(std::sync::atomic::Ordering::Relaxed) != 0 {
        use nativeres_core::cycles::{ENTRIES, NAMES, TOTALS};
        for (part, name) in NAMES.iter().enumerate() {
            let (total, entries) = (TOTALS[part].load(std::sync::atomic::Ordering::Relaxed), ENTRIES[part].load(std::sync::atomic::Ordering::Relaxed));
            if entries > 0 {
                println!("cycles {name:>12}: {:>10} per frame, {:>6} entries per frame", total / runs, entries / runs);
            }
        }
    }
    if nativeres_core::recorder::TEXGEN_RUNS.load(std::sync::atomic::Ordering::Relaxed) == 2 {
        let [checked, differing, vertices] = nativeres_core::recorder::TEXGEN_CHECKS.each_ref().map(|count| count.load(std::sync::atomic::Ordering::Relaxed));
        println!("texgen check: {checked} draws, {differing} differing, {vertices} vertices");
    }
    Ok(())
}

fn write_png(path: &str, pixels: &[u32]) -> Result<(), String> {
    png::write_rgba(path, raster::WIDTH, raster::HEIGHT, pixels).map_err(|error| format!("{path}: {error}"))?;
    println!("wrote {path}");
    Ok(())
}
