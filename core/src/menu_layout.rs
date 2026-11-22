//! Where "NativeRes Renderer" goes in Adrenaline's Settings entries
//! (renderer/vita/src/menu.rs reads the entries, this decides).
//!
//! The entry count is a constant of Adrenaline's tab table, so the switch
//! either takes the place of a blank line below Graphics Filtering (the
//! entries between move one down, the count stays: Adrenaline built from
//! master has blank lines there), or, without a blank line (the v8.0.2
//! release, issue #1), the entries are copied into a table one entry longer
//! and the Settings tab is pointed at the copy.

/// One Settings entry as far as the layout cares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Line {
    /// "Graphics Filtering".
    Filter,
    /// A text line with an empty name (the cursor skips it).
    Blank,
    /// Anything else.
    Other,
}

/// Where the switch goes; it is always right below Graphics Filtering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// In Adrenaline's own entries: `filter + 1 ..= blank` move one down
    /// over the blank line at `blank`, the switch goes to `filter + 1`.
    InPlace { filter: usize, blank: usize },
    /// In a copy one entry longer: entries `..= filter` as they are, the
    /// switch at `filter + 1`, the rest one down.
    Grown { filter: usize },
}

/// The layout for `lines`, None without Graphics Filtering.
pub fn plan(lines: &[Line]) -> Option<Layout> {
    let filter = lines.iter().position(|&line| line == Line::Filter)?;
    Some(match lines[filter + 1..].iter().position(|&line| line == Line::Blank) {
        Some(offset) => Layout::InPlace { filter, blank: filter + 1 + offset },
        None => Layout::Grown { filter },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use Line::{Blank as B, Filter as F, Other as O};

    /// Adrenaline v8.0.2 (release): no blank lines in Settings (issue #1:
    /// the switch was missing, Graphics Filtering skipped Original).
    #[test]
    fn release_8_0_2_grows_the_table() {
        let lines = [O, F, O, O, O, O, O, O, O, O, O, O, O];
        assert_eq!(plan(&lines), Some(Layout::Grown { filter: 1 }));
    }

    /// Adrenaline built from master: blank lines after the CFW line and
    /// before Memory Stick Location.
    #[test]
    fn master_takes_the_blank_line() {
        let lines = [O, B, F, O, O, O, O, O, O, B, O, O, O, O, O];
        assert_eq!(plan(&lines), Some(Layout::InPlace { filter: 2, blank: 9 }));
    }

    #[test]
    fn no_filter_no_layout() {
        assert_eq!(plan(&[O, B, O]), None);
    }
}
