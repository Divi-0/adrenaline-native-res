//! God of War's shadow passes lay each 64x64 shadow tile over the ground
//! with "target x (1 - texel)": triangles that sample only the tile's black
//! texels change nothing, and the Vita leaves them out (core black_texels,
//! DESIGN "sixty-seventh session"). The replay does the same with
//! `REPLAY_BLACK_TEXELS=1`: the gameplay dump must give the very same
//! picture with and without, and some triangles must go. Skipped without
//! the private test directory's dump (see regression.rs).

use std::path::{Path, PathBuf};
use std::process::Command;

fn tests_dir() -> PathBuf {
    match std::env::var_os("NATIVERES_TESTS") {
        Some(dir) => PathBuf::from(dir),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests"),
    }
}

/// Replays the dump into `picture` with REPLAY_BLACK_TEXELS=`on`; returns the output.
fn replay(dump: &Path, picture: &Path, on: &str) -> String {
    let output = Command::new(env!("CARGO_BIN_EXE_replay"))
        .arg(dump)
        .arg(picture)
        .args(["--raw", "--twice", "--ge-filter", "--ge-light", "--ge-depth"])
        .env("REPLAY_BLACK_TEXELS", on)
        .output()
        .expect("can't run replay");
    assert!(output.status.success(), "replay failed: {}", String::from_utf8_lossy(&output.stderr));
    String::from_utf8_lossy(&output.stdout).into_owned()
}

#[test]
fn black_texel_triangles_change_nothing_in_god_of_wars_shadows() {
    let dump = tests_dir().join("dumps/gow_UCES00842/game.bin");
    if !dump.exists() {
        eprintln!("skipped: no {}", dump.display());
        return;
    }
    let out = Path::new(env!("CARGO_MANIFEST_DIR")).join("../target/regression");
    std::fs::create_dir_all(&out).unwrap();
    let (all, culled) = (out.join("gow-black-all.png"), out.join("gow-black-culled.png"));
    replay(&dump, &all, "0");
    let text = replay(&dump, &culled, "1");
    // "black texels: <kept> triangles kept, <culled> culled"
    let left_out: u32 = text
        .lines()
        .find_map(|line| line.strip_prefix("black texels: ")?.split_once(" triangles kept, ")?.1.strip_suffix(" culled")?.parse().ok())
        .expect("no black texels summary");
    // 2026-11-24: 256 of 836 triangles in the dump's three shadow passes and the bloom add.
    assert!(left_out > 0, "no triangle left out");
    assert!(std::fs::read(&all).unwrap() == std::fs::read(&culled).unwrap(), "the pictures differ");
}
