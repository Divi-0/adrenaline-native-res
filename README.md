# adrenaline-native-res

A renderer for PSP games in [Adrenaline](https://github.com/isage/Adrenaline) on the PS Vita that
draws them with the Vita's own GPU at the Vita's native resolution, 960x544, in 32-bit colour, instead
of showing the 480x272 picture of Sony's PSP emulator scaled up.

It is one taiHEN plugin, `adrenaline-native-res.suprx`, for stock Adrenaline: no modified
Adrenaline, no changed system files. Inside, it calls itself NativePSP: the menu entry, the settings
folder and the code use that name.

> **Experimental.** Developed and tested on firmware 3.65 (HENkaku Enso) with Final Fantasy
> (ULES00986), MTX Mototrax (ULES00581) and Monster Hunter Freedom (ULES00318). Other games may show
> wrong pictures or run slower than without it; heavy 3D scenes can drop below the game's own frame
> rate. Back up your saves.

## How it works

Adrenaline runs PSP games in Sony's PSP emulator (the one inside the Vita's firmware for PS1/PSP
titles). That emulator renders the PSP's graphics engine (the "GE") at the PSP's resolution, and
Adrenaline shows the result. NativePSP leaves that emulation untouched and draws every frame a second
time, natively:

1. **PSP side** (`psp/`, the kernel module `nativepsp.prx`, runs inside the emulated PSP). It hooks
   the calls a game uses to hand display lists to the GE (`sceGeListEnQueue`, the stall-address
   updates, GE contexts) and to show a finished frame (`sceDisplaySetFrameBuf`). It records each call
   as an event in a ring in PSP memory. Every hook calls the original function, so the game still
   renders as before. The PSP side also copies the VRAM ranges the Vita side asks for.
2. **Capture** (`core/`, `vita/src/capture.rs`, on the Vita's ARM cores). The Vita side reads the
   ring and walks the same display lists in PSP memory, the way the GE would. It keeps the GE state
   (matrices, lighting, textures and palettes, blending, depth and stencil, fog) and turns every draw
   into a draw for the Vita's GPU. The work is split across threads on several cores: one walks the
   lists, one copies the memory the draws need, one records the vertices.
3. **Drawing** (`vita/src/frame_loop.rs`, `draw.rs`, `programs.rs`, `vita/shaders/`). Adrenaline's
   display thread draws the frame with GXM into 960x544 render targets, one per PSP framebuffer. PSP
   vertices go to the GPU mostly as they are: vertex shaders do the transforms, skinning and
   lighting. Textures are decoded once and cached; swizzled ones are unswizzled on the GPU.
4. **Showing it** (`vita/src/display_hooks.rs`). Hooks on Adrenaline's display loop put our picture
   where Sony's would go. Frames the game's CPU drew itself (movies) show Sony's own picture. A frame
   that lost part of its display lists is held back, and the previous one stays on screen.

The plugin carries the PSP side and installs it into Adrenaline's `seplugins` when Adrenaline starts.
`core/` has no Vita code and is tested on a PC. `replay/` renders frames recorded on the Vita with
the same code on a PC, which is the regression test.

## How it differs from GePatch

[GePatch](https://github.com/TheOfficialFloW/GePatch) (TheFloW) was the first plugin to show PSP games
at a higher resolution in Adrenaline. NativePSP's PSP side hooks the GE functions the way GePatch
does, but everything after that is different:

| | GePatch | NativePSP |
|---|---|---|
| Who draws the picture | Sony's PSP emulator: GePatch rewrites the game's display lists in place (scaled viewport and scissor, doubled vertices), so the emulated GE renders a bigger framebuffer | The Vita's GPU (GXM), from our own walk of the display lists |
| Cost of the higher resolution | 4x the pixels for the emulated GE: a big frame-rate loss | On the Vita's GPU, beside the emulation |
| Colour | 16-bit (RGB565) | 32-bit |
| The game's display lists | Modified | Read only: Sony's emulation renders exactly as before |
| Movies (frames the CPU drew) | Some games don't show them (GePatch's known issues) | Sony's own picture, scaled |
| What you install | A PSP plugin (`ge_patch.prx` in `seplugins/game.txt`) | One Vita plugin (taiHEN); it installs its PSP side itself |
| Adrenaline | Adrenaline 7 or higher | Adrenaline 8 (isage), stock |

Because Sony's emulation still runs in full, NativePSP doesn't make a game faster. It makes it look
better at the same speed, as long as the Vita's spare CPU time is enough to capture each frame.

## Installation

You need a PS Vita with HENkaku/taiHEN (tested: firmware 3.65 with Enso) and Adrenaline 8 from
[isage/Adrenaline](https://github.com/isage/Adrenaline) installed and working.

1. Download `adrenaline-native-res.suprx` from the [releases](../../releases).
2. Copy it to `ur0:tai/adrenaline-native-res.suprx` (with VitaShell, for example).
3. Add it to taiHEN's config, `ur0:tai/config.txt`, under Adrenaline's title id. If a
   `ux0:tai/config.txt` exists, taiHEN reads that one instead, so edit that one:
   ```
   *PSPEMUCFW
   ur0:tai/adrenaline-native-res.suprx
   ```
   It only loads inside Adrenaline, never at boot. Keep any other `*PSPEMUCFW` lines you already have.
4. Reboot the Vita, then start Adrenaline. At its first start the plugin writes its PSP side to
   `<Memory Stick>/seplugins/nativepsp.prx` and adds its line to `seplugins/plugins.txt`.
5. Start a PSP game. The renderer is on by default. Switch it in Adrenaline's menu: **Settings →
   NativePSP Renderer**, right below Graphics Filtering. With it on, the picture is drawn with the GPU
   whatever filter is selected; your filter setting is kept.

**Savestates:** Adrenaline's savestates contain the PSP side's code. After installing a new version,
make new savestates; loading an old one runs the old PSP side.

### Settings (optional)

`ux0:data/nativepsp/config.txt`, one `name=value` per line, read when Adrenaline starts:

```
mip=0        # distant textures: 0 sharp (default), 1 blurry like the PSP, 2 mipmapped
smooth=0     # texture filter: 0 the game's own (default), 1 always smooth
gamefps=1    # 1 (default): the game's own frame rate in the top left corner, 0: off
armclock=444 # ARM clock in MHz (default 444; 333 is Sony's default)
```

### Uninstalling

Remove the line from `ur0:tai/config.txt` (or `ux0:tai/config.txt`) and reboot. Then delete
`ur0:tai/adrenaline-native-res.suprx`, `ux0:data/nativepsp/` (if you made a settings file),
`<Memory Stick>/seplugins/nativepsp.prx` and its line in `seplugins/plugins.txt`.

## Building

Everything builds in a Docker image with vitasdk, pspdev and Rust (a pinned nightly: the Vita and
the PSP are tier-3 targets whose `core` is built from source):

```
docker build -t nativepsp-sdk docker
docker run --rm -v "$PWD:/src" -w /src nativepsp-sdk bash -lc "sh scripts/build.sh"
```

This runs the PC tests and writes `target/adrenaline-native-res.suprx`. Pushing a tag `v*` makes the GitHub
release workflow build and publish it (`.github/workflows/release.yml`).

| Directory | What it is |
|---|---|
| `core/` | The GE side: display lists, GE state, transforms, lighting, textures, the capture (no Vita code, `cargo test`) |
| `vita/` | The renderer on the Vita: hooks, capture threads, GXM drawing, texture cache |
| `plugin/` | `adrenaline-native-res.suprx`: `vita/` as a taiHEN plugin |
| `psp/` | The PSP side, `nativepsp.prx` (MIPS) |
| `replay/` | Renders a frame recorded on the Vita on a PC; the regression test (`replay/tests/regression.rs`) |
| `shadercompiler/` | A Vita app that compiles our shaders (only needed to change them) |
| `vita/shaders/` | The GPU programs: `.cg` sources and the compiled `.gxp` files |

`vita/src/ext.rs` lets a development build add its own tools to the plugin (a debug server,
benchmarks, frame dumps). The plugin built here has none.

### Shaders

The compiled `.gxp` files are checked in, so a normal build doesn't compile shaders. Changing one
takes the shader compiler app. Build it with `sh scripts/build-shadercompiler.sh` and install the
VPK. It needs Sony's runtime compiler `libshacccg.suprx` in `ur0:data/`, which isn't included. Run
`scripts/shader-variants.sh` after changing a source that has generated variants.

## Credits

- [Adrenaline](https://github.com/isage/Adrenaline) (TheFloW, isage and contributors), which this
  plugin runs in.
- [GePatch](https://github.com/TheOfficialFloW/GePatch) (TheFloW): how to hook the GE on the PSP side.
- [PPSSPP](https://github.com/hrydgard/ppsspp): the reference for the GE's behaviour. Many parts of
  `core/` follow its semantics, and the comments name where.

## License

GPL-3.0-or-later, see [LICENSE](LICENSE).

---

This project is developed with the help of AI (Claude by Anthropic).
