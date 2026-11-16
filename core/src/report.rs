//! Turns [`FrameStats`] into one line of text for the capture log.

use core::fmt::{self, Write};

use crate::command::TEXTURE_FORMAT_NAMES;
use crate::stats::FrameStats;

/// A `fmt::Write` target over a fixed byte buffer. Text that doesn't fit is
/// cut off instead of failing.
pub struct TextBuffer<'a> {
    bytes: &'a mut [u8],
    len: usize,
}

impl<'a> TextBuffer<'a> {
    pub fn new(bytes: &'a mut [u8]) -> Self {
        TextBuffer { bytes, len: 0 }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Write for TextBuffer<'_> {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        let room = self.bytes.len() - self.len;
        let count = text.len().min(room);
        self.bytes[self.len..self.len + count].copy_from_slice(&text.as_bytes()[..count]);
        self.len += count;
        Ok(())
    }
}

/// Writes e.g.
/// `lists=2/2 cmds=5321 draws=210 verts=8000 2d=180 tex=200 [clut4:150 8888:50] textures=35 targets=[04000000] xfer=3 ...`
pub fn write_frame(out: &mut impl Write, stats: &FrameStats) -> fmt::Result {
    write!(
        out,
        "lists={}/{} stalls={} cmds={} draws={} verts={} 2d={} tex={}",
        stats.lists_finished,
        stats.lists_enqueued,
        stats.stall_updates,
        stats.commands,
        stats.draws,
        stats.vertices,
        stats.through_mode_draws,
        stats.textured_draws,
    )?;

    out.write_str(" [")?;
    let mut first = true;
    for (format, &count) in stats.draws_by_texture_format.iter().enumerate() {
        if count == 0 {
            continue;
        }
        if !first {
            out.write_char(' ')?;
        }
        first = false;
        let name = TEXTURE_FORMAT_NAMES.get(format).copied().unwrap_or("?");
        write!(out, "{}:{}", name, count)?;
    }
    out.write_str("]")?;

    write!(out, " textures={} targets=[", stats.textures.count())?;
    for (index, address) in stats.render_targets.as_slice().iter().enumerate() {
        if index > 0 {
            out.write_char(' ')?;
        }
        write!(out, "{:08x}", address)?;
    }
    out.write_str("]")?;

    write!(
        out,
        " xfer={} curves={} bbox={} signals={}",
        stats.block_transfers, stats.curves, stats.bounding_box_jumps, stats.signals
    )?;

    let cache = &stats.cache;
    if cache.hits + cache.misses > 0 {
        // hits/misses/stored/full/cpu-only/changing/quick (hits decided by
        // the mini hash)/repeated (hits on runs used earlier in the frame)
        let numbers =
            [cache.hits, cache.misses, cache.stored, cache.full, cache.cpu_only, cache.changing, cache.quick, cache.repeated];
        write!(
            out,
            " vcache={}/{}/{}/{}/{}/{}/{}/{}",
            numbers[0], numbers[1], numbers[2], numbers[3], numbers[4], numbers[5], numbers[6], numbers[7]
        )?;
    }
    // Raw runs (`raw_arena`): runs/KiB copied/runs that didn't fit/lit ones.
    if cache.raw_runs + cache.raw_full + cache.raw_psp_ram > 0 {
        write!(out, " raw={}/{}k/{}/{}", cache.raw_runs, cache.raw_bytes / 1024, cache.raw_full, cache.raw_lit)?;
        if cache.raw_psp_ram > 0 {
            write!(out, " zerocopy={}/{}k", cache.raw_psp_ram, cache.raw_psp_ram_bytes / 1024)?;
        }
        // Indexed raw runs: indices converted / vertices gathered.
        if cache.raw_converted + cache.raw_gathered > 0 {
            write!(out, " indexed={}/{}", cache.raw_converted, cache.raw_gathered)?;
        }
    }

    // PRIMs decoded on the CPU by reason (`recorder::fallback`):
    // cpu=<reason>:<PRIMs>/<PSP vertices>,...
    if stats.fallback_prims.iter().any(|&prims| prims > 0) {
        out.write_str(" cpu=")?;
        let mut first = true;
        for (reason, name) in crate::recorder::fallback::NAMES.iter().enumerate() {
            if stats.fallback_prims[reason] > 0 {
                if !first {
                    out.write_char(',')?;
                }
                first = false;
                write!(out, "{}:{}/{}", name, stats.fallback_prims[reason], stats.fallback_vertices[reason])?;
            }
        }
    }

    if let Some(shown) = stats.displayed {
        write!(out, " shown={:08x}/{}/{}", shown.address, shown.stride, shown.pixel_format)?;
    }

    // Problems, only when present, so normal lines stay short.
    let problems = [
        ("pending", stats.lists_pending),
        ("bad_addr", stats.bad_addresses),
        ("endless", stats.endless_lists),
        ("abandoned", stats.lists_abandoned),
        ("queue_full", stats.queue_overflows),
        ("dropped", stats.dropped_events),
        ("ctx_save", stats.contexts_saved),
        ("ctx_restore", stats.contexts_restored),
        ("ctx_missing", stats.contexts_missing),
        ("ctx_pipelined", stats.contexts_pipelined),
        ("xfer_skipped", stats.transfers_skipped),
        ("skip3d", stats.skipped_3d_draws),
        ("skip_lines", stats.skipped_point_line_draws),
        ("unreadable", stats.unreadable_draws),
        ("approx", stats.approximated_draws),
        ("frame_full", stats.frame_overflows),
        ("inline", stats.inline_draws),
        ("inline_kb", stats.inline_words / 256),
        ("masked", stats.color_masked_draws),
    ];
    for (name, count) in problems {
        if count > 0 {
            write!(out, " {}={}", name, count)?;
            if name == "bad_addr" {
                write!(out, "@{:08x}", stats.last_bad_address)?;
            }
            if name == "unreadable" {
                write!(out, "@{:08x}", stats.last_unreadable_address)?;
            }
            if name == "approx" {
                // By reason (recorder::approx bits), then the last
                // approximated blend: mode/FIX A/FIX B x states.
                write!(out, "[")?;
                let mut first = true;
                for (bit, &times) in stats.approximated_by_reason.iter().enumerate().filter(|(_, &times)| times > 0) {
                    write!(out, "{}{}:{}", if first { "" } else { "," }, bit, times)?;
                    first = false;
                }
                write!(out, "]")?;
                if stats.approx_blend_states > 0 {
                    let [mode, fix_a, fix_b] = stats.approx_blend;
                    write!(out, " approx_blend={:06x}/{:06x}/{:06x}x{}", mode, fix_a, fix_b, stats.approx_blend_states)?;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_a_frame() {
        let mut stats = FrameStats::new();
        stats.lists_enqueued = 1;
        stats.lists_finished = 1;
        stats.draws = 2;
        stats.textured_draws = 2;
        stats.draws_by_texture_format[4] = 2;
        stats.render_targets.insert(0x0400_0000);
        stats.dropped_events = 3;

        let mut bytes = [0u8; 512];
        let mut text = TextBuffer::new(&mut bytes);
        write_frame(&mut text, &stats).unwrap();
        let len = text.len();
        let line = core::str::from_utf8(&bytes[..len]).unwrap();

        assert!(line.starts_with("lists=1/1 "), "{line}");
        assert!(line.contains("[clut4:2]"), "{line}");
        assert!(line.contains("targets=[04000000]"), "{line}");
        assert!(line.ends_with("dropped=3"), "{line}");
    }

    #[test]
    fn long_text_is_cut_off() {
        let mut bytes = [0u8; 8];
        let mut text = TextBuffer::new(&mut bytes);
        write!(text, "0123456789").unwrap();
        assert_eq!(text.len(), 8);
    }
}
