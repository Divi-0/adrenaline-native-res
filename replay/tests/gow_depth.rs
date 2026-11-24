//! God of War's title fence: its rail boxes and planks lie less than one
//! 16-bit depth unit apart, so the PSP's own depth arithmetic decides which
//! one shows (DESIGN "fifty-seventh session"). The Vita's GoW programs take
//! raw vertices' depth from the GE's arithmetic (core `ge_math`, the raw
//! vertex programs' GE_DEPTH twins); the replay models that with
//! `--ge-depth`. Checked against Sony's own depth buffer of the left fence
//! post (read live on the Vita, `dumps/gow_UCES00842/title-fence-depth.u16`: 70 x 100
//! little-endian u16, PSP x 0..70, y 130..230). Skipped without the
//! private test directory's files (see regression.rs).

use std::path::{Path, PathBuf};
use std::process::Command;

fn tests_dir() -> PathBuf {
    match std::env::var_os("NATIVERES_TESTS") {
        Some(dir) => PathBuf::from(dir),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests"),
    }
}

#[test]
fn the_gow_fence_takes_the_ges_own_depth() {
    let dir = tests_dir().join("dumps/gow_UCES00842");
    let (dump, sony) = (dir.join("title.bin"), dir.join("title-fence-depth.u16"));
    if !dump.exists() || !sony.exists() {
        eprintln!("skipped: no {} or {}", dump.display(), sony.display());
        return;
    }
    let out = Path::new(env!("CARGO_MANIFEST_DIR")).join("../target/regression");
    std::fs::create_dir_all(&out).unwrap();
    let depth_file = out.join("gow-fence-depth.f32");
    let output = Command::new(env!("CARGO_BIN_EXE_replay"))
        .arg(&dump)
        .arg(out.join("gow-fence.png"))
        .args(["--raw", "--twice", "--ge-filter", "--ge-light", "--ge-depth"])
        .env("REPLAY_SAVE_DEPTH", &depth_file)
        .output()
        .expect("can't run replay");
    assert!(output.status.success(), "replay failed: {}", String::from_utf8_lossy(&output.stderr));

    let depth: Vec<f32> = std::fs::read(&depth_file).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    let sony: Vec<u16> = std::fs::read(&sony).unwrap().chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
    assert_eq!((depth.len(), sony.len()), (960 * 544, 70 * 100));
    let mut exact = 0;
    for y in 0..100 {
        for x in 0..70 {
            // The PSP pixel's top-left pixel of the 2x picture.
            let ours = (depth[(2 * (130 + y)) * 960 + 2 * x] * 65535.0).round() as u16;
            exact += (ours == sony[y * 70 + x]) as u32;
        }
    }
    // 2026-11-24: 35.1 % with float vertex depth, 60.6 % with the GE's (at
    // 2x positions without the Vita's centre evaluation).
    let share = exact as f64 / 7000.0;
    assert!(share >= 0.5, "only {:.1} % of the fence post's depths are Sony's", share * 100.0);
}
