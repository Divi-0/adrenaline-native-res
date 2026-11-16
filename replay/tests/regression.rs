//! Regression test over every game and scene checked on the Vita so far.
//!
//! The test directory is `tests/` at the repository root, or the one named
//! by `NATIVERES_TESTS`. In it, `dumps/<game>/<scene>.bin` are frame dumps
//! taken on the Vita (a development build of the plugin records them);
//! `expected.txt` holds, per dump, what the replay made of it when it was
//! last accepted:
//!
//! - `hash`: the hash of the frame description (draws, vertices, textures)
//!   the capture produced. Any change to what would be drawn changes it.
//! - `diff`: the mean difference (0..255) between our picture and the
//!   game's own picture stored in the dump, at 480x272.
//!
//! The test replays every dump with the `replay` binary and fails if a hash
//! changed, saying whether the picture got closer to the game's (an
//! intended improvement: accept it) or further away (a regression), and
//! also if only the picture changed (same hash, `diff` moved by more than
//! `DIFF_TOLERANCE`: a change in how the replay draws, e.g. blending or a
//! sampled framebuffer; mtx/hill-blur is MTX's motion blur, slot 8).
//! Accept the current results with
//!
//! ```text
//! NATIVERES_UPDATE_EXPECTED=1 cargo test --release -p nativeres-replay --test regression
//! ```
//!
//! Pictures go to `renderer/target/regression/<game>-<scene>.png` (plus
//! `-original` and `-diff`) for looking at the differences.
//!
//! Adding a scene: copy its dump to <tests>/dumps/<game>/<scene>.bin and
//! run the update command above.
//!
//! A frame dump is a copy of the game's memory for one frame (its display
//! lists, vertices and textures) plus the game's own picture: dumps of
//! commercial games can't be published, so they live in a private test
//! directory (`NATIVERES_TESTS`). Without any dumps the test passes and
//! says so.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The difference to the original may move this much without counting as
/// better or worse (the dumps' original pictures are from the same frame,
/// but small rounding changes shouldn't be called improvements).
const DIFF_TOLERANCE: f64 = 0.05;

/// `NATIVERES_TESTS`, or `tests/` at the repository root.
fn tests_dir() -> PathBuf {
    match std::env::var_os("NATIVERES_TESTS") {
        Some(dir) => PathBuf::from(dir),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests"),
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Result {
    hash: String,
    diff: f64,
    /// The replay's frame line, for the report only.
    summary: String,
}

/// Replays one dump; `name` is "<game>/<scene>".
fn replay(name: &str, dump: &Path, out_dir: &Path) -> std::result::Result<Result, String> {
    let picture = out_dir.join(format!("{}.png", name.replace('/', "-")));
    let output = Command::new(env!("CARGO_BIN_EXE_replay"))
        .arg(dump)
        .arg(&picture)
        .output()
        .map_err(|error| format!("can't run replay: {error}"))?;
    let text = String::from_utf8_lossy(&output.stdout).into_owned() + &String::from_utf8_lossy(&output.stderr);
    if !output.status.success() {
        return Err(format!("replay failed: {}", text.trim()));
    }

    let hash = text
        .lines()
        .find_map(|line| line.strip_prefix("description: "))
        .and_then(|line| line.rsplit("hash ").next())
        .map(|hash| hash.trim().to_string())
        .ok_or_else(|| format!("no description in the replay output:\n{text}"))?;
    if let Some(broken) = text.lines().find(|line| line.starts_with("BROKEN: ")) {
        return Err(broken.to_string());
    }
    let summary = text.lines().find_map(|line| line.strip_prefix("frame: ")).unwrap_or("").to_string();

    // "average difference per region (0..255):" followed by rows of numbers.
    let mut values = Vec::new();
    let mut in_table = false;
    for line in text.lines() {
        if line.starts_with("average difference") {
            in_table = true;
            continue;
        }
        if in_table {
            let row: Vec<f64> = line.split_whitespace().filter_map(|value| value.parse().ok()).collect();
            if row.is_empty() {
                break;
            }
            values.extend(row);
        }
    }
    if values.is_empty() {
        return Err(format!("no difference table (dump without original picture?):\n{text}"));
    }
    let diff = values.iter().sum::<f64>() / values.len() as f64;
    Ok(Result { hash, diff, summary })
}

/// All dumps under <tests>/dumps, as ("<game>/<scene>", path), sorted.
fn find_dumps(root: &Path) -> Vec<(String, PathBuf)> {
    let mut dumps = Vec::new();
    let Ok(games) = std::fs::read_dir(root) else {
        return dumps;
    };
    for game in games.flatten().filter(|entry| entry.path().is_dir()) {
        for scene in std::fs::read_dir(game.path()).into_iter().flatten().flatten() {
            let path = scene.path();
            if path.extension().is_some_and(|extension| extension == "bin") {
                let name = format!(
                    "{}/{}",
                    game.file_name().to_string_lossy(),
                    path.file_stem().unwrap().to_string_lossy()
                );
                dumps.push((name, path));
            }
        }
    }
    dumps.sort();
    dumps
}

/// Reads <tests>/expected.txt: "<game>/<scene> hash=<hex> diff=<mean>" lines.
fn read_expected(path: &Path) -> BTreeMap<String, Result> {
    let mut expected = BTreeMap::new();
    let text = std::fs::read_to_string(path).unwrap_or_default();
    for line in text.lines().map(str::trim).filter(|line| !line.is_empty() && !line.starts_with('#')) {
        let mut fields = line.split_whitespace();
        let Some(name) = fields.next() else { continue };
        let mut result = Result { hash: String::new(), diff: f64::NAN, summary: String::new() };
        for field in fields {
            if let Some(hash) = field.strip_prefix("hash=") {
                result.hash = hash.to_string();
            } else if let Some(diff) = field.strip_prefix("diff=") {
                result.diff = diff.parse().unwrap_or(f64::NAN);
            }
        }
        expected.insert(name.to_string(), result);
    }
    expected
}

fn write_expected(path: &Path, results: &BTreeMap<String, Result>) {
    let mut text = String::from(
        "# Regression baseline for replay/tests/regression.rs: per dump in\n\
         # dumps/, the frame description hash and the mean difference (0..255)\n\
         # to the game's own picture. Rewritten by\n\
         #   NATIVERES_UPDATE_EXPECTED=1 cargo test --release -p nativeres-replay --test regression\n",
    );
    for (name, result) in results {
        text += &format!("{name} hash={} diff={:.2}\n", result.hash, result.diff);
    }
    std::fs::write(path, text).expect("can't write expected.txt");
}

#[test]
fn every_dumped_scene_replays_as_accepted() {
    let tests = tests_dir();
    let dumps = find_dumps(&tests.join("dumps"));
    if dumps.is_empty() {
        println!("no frame dumps in {}: nothing to replay (set NATIVERES_TESTS to a test directory)", tests.join("dumps").display());
        return;
    }
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../target/regression");
    std::fs::create_dir_all(&out_dir).unwrap();
    let expected_path = tests.join("expected.txt");
    let expected = read_expected(&expected_path);

    let mut results = BTreeMap::new();
    let mut problems = Vec::new();
    for (name, path) in &dumps {
        match replay(name, path, &out_dir) {
            Ok(result) => {
                let line = match expected.get(name) {
                    None => {
                        problems.push(format!("{name}: new scene, no baseline yet"));
                        "NEW".to_string()
                    }
                    // Same frame description, but drawn differently (the
                    // replay's display side: blending, masks, sampled
                    // framebuffers; MTX's hill-blur scene with a blur that
                    // lost its weight): the picture moved.
                    Some(old) if old.hash == result.hash && (result.diff - old.diff).abs() > DIFF_TOLERANCE => {
                        problems.push(format!(
                            "{name}: same frame description, picture changed: difference {:.2} -> {:.2} ({})",
                            old.diff,
                            result.diff,
                            if result.diff < old.diff { "closer to the original" } else { "FURTHER from the original" }
                        ));
                        "PICTURE".to_string()
                    }
                    Some(old) if old.hash == result.hash => "ok".to_string(),
                    Some(old) => {
                        let verdict = if result.diff < old.diff - DIFF_TOLERANCE {
                            "closer to the original"
                        } else if result.diff > old.diff + DIFF_TOLERANCE {
                            "FURTHER from the original"
                        } else {
                            "same difference to the original"
                        };
                        problems.push(format!(
                            "{name}: frame description changed ({} -> {}), difference {:.2} -> {:.2}: {verdict}",
                            old.hash, result.hash, old.diff, result.diff
                        ));
                        "CHANGED".to_string()
                    }
                };
                println!("{line:8} {name:28} diff {:6.2}  {}", result.diff, result.summary);
                results.insert(name.clone(), result);
            }
            Err(error) => problems.push(format!("{name}: {error}")),
        }
    }
    for name in expected.keys().filter(|name| !results.contains_key(*name)) {
        problems.push(format!("{name}: in expected.txt but its dump is missing"));
    }

    if std::env::var_os("NATIVERES_UPDATE_EXPECTED").is_some() {
        write_expected(&expected_path, &results);
        println!("wrote {} ({} scenes)", expected_path.display(), results.len());
        return;
    }
    assert!(
        problems.is_empty(),
        "\n{}\n\nPictures: {}\nIf the changes are intended: NATIVERES_UPDATE_EXPECTED=1 cargo test --release -p nativeres-replay --test regression\n",
        problems.join("\n"),
        out_dir.display()
    );
}
