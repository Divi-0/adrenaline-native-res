// The site's data: the tested games (games.html) and the releases (releases.html).
//
// GAMES: one line per game. status: "tested" | "partial" | "broken".
//   fps / of: average frames a second shown with the renderer on, against the game's own rate.
//   low: the 1% low. Both are read from the plugin's diagnostics (DRAWN and LOW) in a named
//   scene; null while nobody has read them (the page shows a dash). img: the 640x363 tile in img/;
//   beside it <name>-native.jpg and <name>-sony.jpg, the 960x544 pair a click on the tile opens.
//   note: a few words on where the game stands.
//
// RELEASES: newest first. title: two to five words. summary: one sentence.
//   games: the games the release changed. id: a title ID from GAMES. was / now: a status,
//          "" = not listed before. fps / of / scene: as above, for the release's build.
//   changes: one short line each, six words at most (the bullet points under the pictures). How to update is said in the GitHub
//          release notes, not here.
window.NR = {
  repo: "https://github.com/Divi-0/adrenaline-native-res",
  games: [
    { title: "Final Fantasy", id: "ULES00986", region: "Europe", status: "tested", fps: 60, of: 60, low: null, img: "img/ff1.jpg", note: "Title, field, battles" },
    { title: "MTX Mototrax", id: "ULES00581", region: "Europe", status: "tested", fps: 30, of: 30, low: null, img: "img/mtx.jpg", note: "Menus and races" },
    { title: "Kenka Bancho: Badass Rumble", id: "ULUS10442", region: "USA", status: "partial", fps: 30, of: 30, low: null, img: "img/kb.jpg", note: "Up to the first area" },
    { title: "Monster Hunter Freedom", id: "ULES00318", region: "Europe", status: "partial", fps: 30, of: 30, low: null, img: "img/mhf.jpg", note: "House and village" },
    { title: "God of War: Chains of Olympus", id: "UCES00842", region: "Europe", status: "broken", fps: 6, of: 30, low: null, img: "img/gow.jpg", note: "Too slow to play" },
    { title: "Need for Speed: Underground Rivals", id: "ULES00025", region: "Europe", status: "broken", fps: null, of: 30, low: null, img: "img/nfs.jpg", note: "Main menu only" },
  ],
  releases: [
    { tag: "v0.3.0", date: "2026-11-27", title: "Truer colors and effects",
      summary: "More games match the colors of the stock image, glow and shadows are drawn in full, and two more games are on the list.",
      games: [
        { id: "ULUS10442", was: "", now: "partial", fps: 30, of: 30, scene: "first area" },
        { id: "UCES00842", was: "", now: "broken", fps: 6, of: 30, scene: "first level" },
      ],
      changes: [
        "More games match the colors of the stock image",
        "Glow and shadows drawn in full",
        "Videos no longer blurred with the Original filter",
        "New option: distance blur like the PSP",
        "Frame-rate label off by default",
      ] },
    { tag: "v0.2.1", date: "2026-11-22", title: "Menu fix and a start-up log",
      summary: "Brings back the missing menu entry and adds a log that shows whether the plugin started.",
      games: [
        { id: "ULES00025", was: "", now: "broken", scene: "" },
      ],
      changes: [
        "NativeRes entry back in Adrenaline's menu",
        "New start-up log in ux0:data/nativeres",
      ] },
    { tag: "v0.2.0", date: "2026-11-20", title: "Monster Hunter Freedom",
      summary: "Adds Monster Hunter Freedom and improves speed and stability.",
      games: [
        { id: "ULES00318", was: "", now: "partial", scene: "" },
        { id: "ULES00986", was: "tested", now: "tested", scene: "" },
      ],
      changes: [
        "Smoother Monster Hunter Freedom and Final Fantasy",
        "Fewer freezes, glitches and crashes",
      ] },
    { tag: "v0.1.0", date: "2026-11-16", title: "First release",
      summary: "PSP games drawn by the Vita's own GPU at 960×544, where the emulator shows 480×272 scaled up.",
      games: [
        { id: "ULES00986", was: "", now: "tested", scene: "" },
        { id: "ULES00581", was: "", now: "tested", scene: "" },
      ],
      changes: [
        "Better picture at the same speed",
      ] },
  ],
};
