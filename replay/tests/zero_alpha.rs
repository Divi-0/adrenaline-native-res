//! Triangles whose three corners have alpha 0 draw nothing where the draw
//! blends "source alpha, one minus source alpha or one" and writes only
//! colour (core zero_alpha, DESIGN "seventy-eighth session"): God of War's
//! cloud layer fades to alpha 0 over most of the sky, and its title's too.
//! The Vita leaves such triangles out; the replay does the same with
//! `REPLAY_ZERO_ALPHA=1`: each dump must give the very same picture with
//! and without, and some triangles must go. Skipped without the private
//! test directory's dumps (see regression.rs).

use std::path::{Path, PathBuf};
use std::process::Command;

fn tests_dir() -> PathBuf {
    match std::env::var_os("NATIVERES_TESTS") {
        Some(dir) => PathBuf::from(dir),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests"),
    }
}

/// Replays the dump into `picture` with REPLAY_ZERO_ALPHA=`on` and
/// REPLAY_ZERO_TEXELS=`texels`; returns the output.
fn replay(dump: &Path, picture: &Path, on: &str, texels: &str) -> String {
    let output = Command::new(env!("CARGO_BIN_EXE_replay"))
        .arg(dump)
        .arg(picture)
        .args(["--raw", "--twice", "--ge-filter", "--ge-light", "--ge-depth"])
        .env("REPLAY_ZERO_ALPHA", on)
        .env("REPLAY_ZERO_TEXELS", texels)
        .output()
        .expect("can't run replay");
    assert!(output.status.success(), "replay failed: {}", String::from_utf8_lossy(&output.stderr));
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// The dump replayed with and without the cull: the same picture, and more
/// than `least` triangles left out (with `texels` "1" also those that sample
/// only transparent texels, more than `least_texels` of them).
fn same_picture_with_fewer_triangles(name: &str, texels: &str, least: u32, least_texels: u32) {
    let dump = tests_dir().join(format!("dumps/gow_UCES00842/{name}.bin"));
    if !dump.exists() {
        eprintln!("skipped: no {}", dump.display());
        return;
    }
    let out = Path::new(env!("CARGO_MANIFEST_DIR")).join("../target/regression");
    std::fs::create_dir_all(&out).unwrap();
    let (all, culled) = (out.join(format!("gow-{name}-alpha-all.png")), out.join(format!("gow-{name}-alpha-culled-{texels}.png")));
    replay(&dump, &all, "0", "0");
    let text = replay(&dump, &culled, "1", texels);
    // "zero alpha: <kept> triangles kept, <culled> culled (<n> by their texels)"
    let (left_out, by_texels): (u32, u32) = text
        .lines()
        .find_map(|line| {
            let (culled, texels) = line.strip_prefix("zero alpha: ")?.split_once(" triangles kept, ")?.1.split_once(" culled (")?;
            Some((culled.parse().ok()?, texels.strip_suffix(" by their texels)")?.parse().ok()?))
        })
        .expect("no zero alpha summary");
    assert!(left_out > least, "only {left_out} triangles left out");
    assert!(by_texels > least_texels || texels == "0" && by_texels == 0, "{by_texels} triangles left out by their texels");
    assert!(std::fs::read(&all).unwrap() == std::fs::read(&culled).unwrap(), "the pictures differ");
}

#[test]
fn zero_alpha_triangles_change_nothing_in_god_of_wars_gameplay() {
    same_picture_with_fewer_triangles("game", "0", LEAST_GAME, 0);
}

/// God of War's light streaks (DESIGN "seventy-ninth session"): 256x32 index
/// textures transparent but for a few columns.
#[test]
fn transparent_texel_triangles_change_nothing_in_god_of_wars_gameplay() {
    same_picture_with_fewer_triangles("game", "1", LEAST_GAME + LEAST_GAME_TEXELS, LEAST_GAME_TEXELS);
}

#[test]
fn transparent_texel_triangles_change_nothing_on_god_of_wars_title() {
    same_picture_with_fewer_triangles("title", "1", LEAST_TITLE + LEAST_TITLE_TEXELS, LEAST_TITLE_TEXELS);
}

#[test]
fn zero_alpha_triangles_change_nothing_on_god_of_wars_title() {
    same_picture_with_fewer_triangles("title", "0", LEAST_TITLE, 0);
}

/// 2026-11-26: the triangles left out in the two dumps (the frame drawn twice).
const LEAST_GAME: u32 = 900; // 970 of 5,789
const LEAST_TITLE: u32 = 450; // 507 of 1,471
/// 2026-11-26 (79th): by their texels as well (lab count: 426 and 60, frame drawn twice).
const LEAST_GAME_TEXELS: u32 = 300;
const LEAST_TITLE_TEXELS: u32 = 40;
