//! A minimal PNG writer (8-bit RGBA), so the replay needs no dependencies.
//! The image data is stored uncompressed ("stored" deflate blocks): files
//! are big, but any viewer opens them.

use std::fs::File;
use std::io::{self, Write};

pub fn write_rgba(path: &str, width: usize, height: usize, pixels: &[u32]) -> io::Result<()> {
    // Raw scanlines: filter byte 0, then R, G, B, A per pixel. Pixels are
    // 0xAABBGGRR, so the little-endian bytes are already in that order.
    let mut raw = Vec::with_capacity(height * (1 + width * 4));
    for row in pixels.chunks(width) {
        raw.push(0);
        for pixel in row {
            raw.extend_from_slice(&pixel.to_le_bytes());
        }
    }

    let mut header = Vec::new();
    header.extend_from_slice(&(width as u32).to_be_bytes());
    header.extend_from_slice(&(height as u32).to_be_bytes());
    header.extend_from_slice(&[8, 6, 0, 0, 0]); // 8 bit, RGBA, deflate, no filter, no interlace

    let mut file = File::create(path)?;
    file.write_all(b"\x89PNG\r\n\x1a\n")?;
    write_chunk(&mut file, b"IHDR", &header)?;
    write_chunk(&mut file, b"IDAT", &zlib_stored(&raw))?;
    write_chunk(&mut file, b"IEND", &[])
}

fn write_chunk(out: &mut impl Write, kind: &[u8; 4], data: &[u8]) -> io::Result<()> {
    out.write_all(&(data.len() as u32).to_be_bytes())?;
    out.write_all(kind)?;
    out.write_all(data)?;
    let mut crc_input = kind.to_vec();
    crc_input.extend_from_slice(data);
    out.write_all(&crc32(&crc_input).to_be_bytes())
}

/// zlib stream made of uncompressed deflate blocks (at most 65535 bytes each).
fn zlib_stored(data: &[u8]) -> Vec<u8> {
    let mut out = vec![0x78, 0x01];
    let chunks: Vec<&[u8]> = data.chunks(65535).collect();
    for (index, chunk) in chunks.iter().enumerate() {
        let last = index + 1 == chunks.len();
        out.push(last as u8);
        let length = chunk.len() as u16;
        out.extend_from_slice(&length.to_le_bytes());
        out.extend_from_slice(&(!length).to_le_bytes());
        out.extend_from_slice(chunk);
    }
    out.extend_from_slice(&adler32(data).to_be_bytes());
    out
}

fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in data {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ 0xEDB8_8320 } else { crc >> 1 };
        }
    }
    !crc
}
