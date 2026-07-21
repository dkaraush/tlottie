/// Append one PNG chunk (length + type + data + CRC32).
fn png_chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
  out.extend_from_slice(&(data.len() as u32).to_be_bytes());
  out.extend_from_slice(kind);
  out.extend_from_slice(data);
  let mut crc = 0xffff_ffffu32;
  for &b in kind.iter().chain(data.iter()) {
    crc ^= b as u32;
    for _ in 0..8 {
      crc = if crc & 1 != 0 { (crc >> 1) ^ 0xedb8_8320 } else { crc >> 1 };
    }
  }
  out.extend_from_slice(&(!crc).to_be_bytes());
}

/// Minimal PNG writer: 8-bit straight-alpha RGBA, stored (uncompressed)
/// deflate blocks. Dev-tool output only — keeps the zero-dependency rule.
pub fn write_png(path: &str, size: u32, pixels: &[u32]) -> std::io::Result<()> {
  let w = size as usize;
  // scanlines: filter byte 0 + un-premultiplied RGBA
  let mut raw = Vec::with_capacity((w * 4 + 1) * w);
  for row in pixels.chunks(w.max(1)) {
    raw.push(0u8);
    for &pixel in row {
      raw.extend_from_slice(&crate::pixel::premultiplied_rgba_to_straight(pixel));
    }
  }
  // zlib: header, stored blocks (<=65535 bytes), adler32
  let mut idat = vec![0x78, 0x01];
  let mut blocks = raw.chunks(65535).peekable();
  while let Some(block) = blocks.next() {
    idat.push(u8::from(blocks.peek().is_none()));
    let len = block.len() as u16;
    idat.extend_from_slice(&len.to_le_bytes());
    idat.extend_from_slice(&(!len).to_le_bytes());
    idat.extend_from_slice(block);
  }
  let (mut s1, mut s2) = (1u32, 0u32);
  for &b in &raw {
    s1 = (s1 + b as u32) % 65521;
    s2 = (s2 + s1) % 65521;
  }
  idat.extend_from_slice(&((s2 << 16) | s1).to_be_bytes());

  let mut out = Vec::with_capacity(idat.len() + 64);
  out.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]);
  let mut ihdr = Vec::with_capacity(13);
  ihdr.extend_from_slice(&size.to_be_bytes());
  ihdr.extend_from_slice(&size.to_be_bytes());
  ihdr.extend_from_slice(&[8, 6, 0, 0, 0]); // bit depth 8, color type RGBA
  png_chunk(&mut out, b"IHDR", &ihdr);
  png_chunk(&mut out, b"IDAT", &idat);
  png_chunk(&mut out, b"IEND", &[]);
  std::fs::write(path, out)
}
