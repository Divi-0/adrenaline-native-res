//! The GE's "slope" level mode (TEXLEVEL mode 2): the level comes from each
//! pixel's 1/w, so distant geometry blurs on the PSP (core
//! `texture::slope_level`, setting `slopemip`, replay `--slope-mip`). Kenka
//! Bancho's train cutscene draws its whole carriage that way: the PSP's
//! picture of the frame blurs the window scenery, level 0 keeps it sharp.
//! Checked against the dump's stored PSP picture: with the setting the
//! replay must come closer, in the window rows and over the whole frame.
//! Skipped without the private test directory's files (see regression.rs).

use std::path::{Path, PathBuf};
use std::process::Command;

fn tests_dir() -> PathBuf {
    match std::env::var_os("NATIVERES_TESTS") {
        Some(dir) => PathBuf::from(dir),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests"),
    }
}

/// The replay's "average difference per region" grid (4 rows of 6).
fn region_differences(dump: &Path, picture: &Path, extra: &[&str]) -> Vec<Vec<f64>> {
    let output = Command::new(env!("CARGO_BIN_EXE_replay")).arg(dump).arg(picture).args(extra).output().expect("can't run replay");
    let text = String::from_utf8_lossy(&output.stdout).into_owned();
    assert!(output.status.success(), "replay failed: {}", String::from_utf8_lossy(&output.stderr));
    let grid: Vec<Vec<f64>> = text
        .lines()
        .skip_while(|line| !line.starts_with("average difference per region"))
        .skip(1)
        .take(4)
        .map(|line| line.split_whitespace().filter_map(|value| value.parse().ok()).collect())
        .collect();
    assert!(grid.len() == 4 && grid.iter().all(|row| row.len() == 6), "no region grid in:\n{text}");
    grid
}

#[test]
fn slope_mode_blurs_the_train_windows_like_the_psp() {
    let dump = tests_dir().join("dumps/kb_ULUS10442/train-slope.bin");
    if !dump.exists() {
        eprintln!("skipped: no {}", dump.display());
        return;
    }
    let out = Path::new(env!("CARGO_MANIFEST_DIR")).join("../target/regression");
    std::fs::create_dir_all(&out).unwrap();
    let sharp = region_differences(&dump, &out.join("slope-off.png"), &[]);
    let slope = region_differences(&dump, &out.join("slope-on.png"), &["--slope-mip"]);
    let mean = |grid: &[Vec<f64>], rows: std::ops::Range<usize>| {
        let values: Vec<f64> = grid[rows].iter().flatten().copied().collect();
        values.iter().sum::<f64>() / values.len() as f64
    };
    let (windows_sharp, windows_slope) = (mean(&sharp, 1..3), mean(&slope, 1..3));
    let (all_sharp, all_slope) = (mean(&sharp, 0..4), mean(&slope, 0..4));
    println!("window rows {windows_sharp:.2} -> {windows_slope:.2}, frame {all_sharp:.2} -> {all_slope:.2}");
    assert!(windows_slope < windows_sharp - 1.0, "window rows {windows_sharp:.2} -> {windows_slope:.2}");
    assert!(all_slope < all_sharp, "frame {all_sharp:.2} -> {all_slope:.2}");
}
