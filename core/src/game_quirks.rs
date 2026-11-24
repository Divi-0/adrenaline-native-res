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
/// - UCES00842 God of War: Chains of Olympus (Europe): the title fence's rail
///   boxes tie with the planks (z-fighting the PSP decides by whole units).
const PSP_DEPTH_GAMES: &[&[u8]] = &[b"UCES00842"];

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
        for other in [&b"ULES00318\0"[..], b"ULES00581\0", b"ULES00986\0", b"ULES00987\0", b"ULES01213\0", b"ULES00025\0", b"\0", b""] {
            assert!(!psp_depth(other), "{:?}", other);
        }
    }
}
