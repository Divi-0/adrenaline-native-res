"""Writes a section ordering file for the plugin's link: the code the capture
thread runs for every PRIM batch goes first in .text, side by side.

Why: per batch the capture thread runs ~40 KB of code (the walk loop, the
draw state, the direct-draw writer, copy tasks, the block cache, memcpy),
more than the Cortex-A9's 32 KB instruction cache, and spread over the whole
~650 KB of code its lines also collide in the cache and the TLB. Linked
together they miss ~35 % less often (2026-11-17, MTX's paused race: ~120k ->
~78k I-cache refills per frame, the walk ~2 % faster). The code is the same;
only addresses change.

The functions are named by patterns over the demangled section names (crate
hashes left out), so new hashes or generic arguments don't matter; a
pattern that matches nothing is only reported. Functions generic over the
PSP memory are taken in the live capture's instantiation (VitaBlocks) only.

    python3 scripts/hot-order.py <lib.a> <out.ld>
"""
import re
import subprocess
import sys

# In the order they run in, roughly: walk, batch, direct draw, copies, memory.
HOT = [
    r"Capture>::handle_traced::<",            # the walk loop
    r"ListWindow>::refill::<",
    r"capture::Both<.*>::visit::<",
    r"Recorder as .*CommandVisitor>::visit_prims::<",
    r"FrameStats>::count_prims$",
    r"recorder::prepare$",
    r"TransformCache>::get$",
    r"Catalog>::add_texture$",
    r"LightingRegisters>::gpu_ready$",
    r"recorder::count_approximations$",
    r"recorder::sample_texture::<",
    r"texture::quick_sample$",
    r"Queues>::write_direct::<",
    r"FrameBuilder>::",
    r"frame_builder::Writer<",
    r"recorder::raw_colors_if_enabled$",
    r"VertexFormat>::layout$",
    r"RawTables>::add_bones$",
    r"RawArena>::reserve$",
    r"Geometry>::finish_raw_draw$",
    r"DrawStream>::(copy_task|hand_over_task)::<",
    r"copy_tasks::read_stable::<",
    r"copy_thread::wake$",
    r"recorder::queue_draw::<",
    r"display_list::block_transfer::<",
    r"ge_writes::Overlaid<.* as .*PspMemory>::",
    r"block_cache::BlockCache<.*>>?::(fill|slot_for_other)$",
    r"BlockCache<.*> as .*PspMemory>::read_words$",
    r"VitaBlocks as .*BlockSource>::",
    r"nativeres_vita::capture::(read_psp_ram|copy_psp_bytes|sample_psp_ram)$",
    r"^compiler_builtins::(mem|arm)::",
    r"^(__aeabi_mem\w*|memcpy|memset|memmove|memcmp)$",
]


def main():
    library, out = sys.argv[1], sys.argv[2]
    headers = subprocess.run(["arm-vita-eabi-objdump", "-h", library], capture_output=True, text=True, check=True).stdout
    sections = sorted(set(re.findall(r" (\.text\.[^ \n]+)", headers)))
    names = subprocess.run(["arm-vita-eabi-c++filt"], input="\n".join(s[len(".text."):] for s in sections),
                           capture_output=True, text=True, check=True).stdout.split("\n")
    names = [re.sub(r"\[[0-9a-f]+\]", "", name) for name in names]
    placed = []
    for pattern in HOT:
        regex = re.compile(pattern)
        found = [s for s, n in zip(sections, names)
                 if regex.search(n) and ("BlockCache<" not in n or "VitaBlocks" in n) and s not in placed]
        if not found:
            print(f"hot-order: nothing matches {pattern}", file=sys.stderr)
        placed.extend(found)
    with open(out, "w") as file:
        file.write("/* Written by scripts/hot-order.py: the capture thread's per-batch code first. */\n")
        file.write(".text : {\n")
        for section in placed:
            file.write(f"  *({section})\n")
        file.write("}\n")
    print(f"hot-order: {len(placed)} sections first in .text", file=sys.stderr)


main()
