//! Comparing our picture with the original from Sony's PSP emulator.

use crate::dump_file::OriginalPicture;

/// The original picture scaled to `width` x `height` by repeating pixels
/// (nearest neighbour), so every PSP pixel becomes a 2x2 block.
pub fn enlarge(original: &OriginalPicture, width: usize, height: usize) -> Vec<u32> {
    let mut out = vec![0xFF00_0000u32; width * height];
    for y in 0..height {
        for x in 0..width {
            let source_x = x * original.width / width;
            let source_y = y * original.height / height;
            out[y * width + x] = original.pixels[source_y * original.width + source_x] | 0xFF00_0000;
        }
    }
    out
}

fn channels(pixel: u32) -> [i32; 3] {
    [(pixel & 0xFF) as i32, ((pixel >> 8) & 0xFF) as i32, ((pixel >> 16) & 0xFF) as i32]
}

/// Black where both pictures agree; the brighter, the bigger the difference
/// (amplified 4x so small differences are visible).
pub fn difference_image(ours: &[u32], original: &[u32]) -> Vec<u32> {
    ours.iter()
        .zip(original)
        .map(|(&a, &b)| {
            let (a, b) = (channels(a), channels(b));
            let d = |i: usize| (((a[i] - b[i]).abs() * 4).min(255)) as u32;
            0xFF00_0000 | d(0) | (d(1) << 8) | (d(2) << 16)
        })
        .collect()
}

/// Average difference (0..255) per region of a 6 x 4 grid.
pub fn print_region_differences(ours: &[u32], original: &[u32], width: usize, height: usize) {
    let (columns, rows) = (6, 4);
    println!("average difference per region (0..255):");
    for row in 0..rows {
        let mut line = String::new();
        for column in 0..columns {
            let (mut total, mut count) = (0i64, 0i64);
            for y in (row * height / rows..(row + 1) * height / rows).step_by(2) {
                for x in (column * width / columns..(column + 1) * width / columns).step_by(2) {
                    let (a, b) = (channels(ours[y * width + x]), channels(original[y * width + x]));
                    total += (0..3).map(|i| (a[i] - b[i]).abs() as i64).sum::<i64>();
                    count += 3;
                }
            }
            line.push_str(&format!("{:6.1}", total as f64 / count as f64));
        }
        println!("{line}");
    }
}
