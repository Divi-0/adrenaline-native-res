//! Per-game choices, by the game's title ID (Adrenaline's, e.g. "UCES00842").
//!
//! Everything here is a way of drawing that is NOT the default for all games,
//! because it costs too much GPU time to give it to games that don't need it
//! (as PPSSPP's compat.ini does for its depth flags).

/// Games drawn with the PSP's whole-unit depth (the depth-writing GPU
/// programs, vita/src/shaders.rs `Twin`; DESIGN "fifty-seventh session"):
/// vertex depth floored, the plane taken at the PSP pixel's centre and
/// floored, written by the fragment programs. Not bit-exact with the PSP
/// (float vertex maths: ~91 % of God of War's title depth exact), but near-
/// coplanar draws tie as on the PSP. It turns off the GPU's early depth test
/// (God of War's title 81 -> 165 ms GPU, 2026-11-23), so only for games whose
/// picture needs it (user, 2026-11-24). Every other game keeps the float depth.
///
/// - God of War: Chains of Olympus: the title fence's rail boxes tie with the
///   planks (z-fighting the PSP decides by whole units). Every version PPSSPP's
///   compat.ini lists (2026-11-24); only UCES00842 (Europe) tested on the Vita.
const PSP_DEPTH_GAMES: &[&[u8]] = &[
    // God of War: Chains of Olympus: discs (Europe, US, Asia, Japan, Korea).
    b"UCES00842", b"UCET00844", b"UCUS98653", b"UCAS40198", b"UCAS40191", b"ULJM05348", b"ULJM05438", b"UCKS45084",
    // ... PSN downloads (EBOOT.PBP: the ID from its PARAM.SFO, psp/src/main.rs
    // sfo_title_id; no EBOOT tested) and demos.
    b"NPUG80325", b"NPEG00023", b"NPHG00027", b"NPHG00028", b"NPJH50170",
    b"UCUS98705", b"UCUS98713", b"UCED00970", b"UCED00971",
];

/// Whether the game `title_id` (NUL-terminated or not) is drawn with the
/// PSP's whole-unit depth (`PSP_DEPTH_GAMES`).
pub fn psp_depth(title_id: &[u8]) -> bool {
    let length = title_id.iter().position(|&byte| byte == 0).unwrap_or(title_id.len());
    PSP_DEPTH_GAMES.contains(&&title_id[..length])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_listed_games_get_the_psp_depth() {
        assert!(psp_depth(b"UCES00842\0"));
        assert!(psp_depth(b"UCES00842"));
        // Every version of God of War: Chains of Olympus PPSSPP's compat.ini lists.
        for gow in [
            &b"UCES00842"[..], b"UCET00844", b"UCUS98653", b"UCAS40198", b"UCAS40191", b"ULJM05348", b"ULJM05438",
            b"UCKS45084", b"NPUG80325", b"NPEG00023", b"NPHG00027", b"NPHG00028", b"NPJH50170", b"UCUS98705",
            b"UCUS98713", b"UCED00970", b"UCED00971",
        ] {
            assert!(psp_depth(gow), "{:?}", gow);
        }
        for other in [&b"ULES00318\0"[..], b"ULES00581\0", b"ULES00986\0", b"ULES00987\0", b"ULES01213\0", b"ULES00025\0", b"\0", b""] {
            assert!(!psp_depth(other), "{:?}", other);
        }
    }
}
