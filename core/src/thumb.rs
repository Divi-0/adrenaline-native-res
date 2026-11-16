//! Reading Thumb-2 code of Sony's modules: the 32-bit address a
//! `movw rd, #lo` / `movt rd, #hi` pair puts into a register. Sony's code
//! loads its data addresses that way; the loader relocates both immediates,
//! so a variable's address read from the code is right for this start
//! (pspemu_watch.rs finds ScePspemu's suspend flag so).

/// MOVW (T3) and MOVT (T1), first halfword without i and imm4.
const MOVW: u16 = 0xF240;
const MOVT: u16 = 0xF2C0;

/// The 16-bit immediate of a MOVW/MOVT `kind` into register `rd`, given as
/// its two halfwords in memory order; None for any other instruction.
fn mov_immediate(halfwords: [u16; 2], kind: u16, rd: u16) -> Option<u32> {
    let [first, second] = halfwords;
    if first & 0xFBF0 != kind || second & 0x8000 != 0 || (second >> 8) & 0xF != rd {
        return None;
    }
    let imm4 = (first & 0xF) as u32;
    let i = ((first >> 10) & 1) as u32;
    let imm3 = ((second >> 12) & 7) as u32;
    let imm8 = (second & 0xFF) as u32;
    Some(imm4 << 12 | i << 11 | imm3 << 8 | imm8)
}

/// The address `movw rd, #lo` + `movt rd, #hi` load into register `rd`
/// (0-15); None unless both instructions are exactly that.
pub fn movw_movt_value(movw: [u16; 2], movt: [u16; 2], rd: u16) -> Option<u32> {
    Some(mov_immediate(movt, MOVT, rd)? << 16 | mov_immediate(movw, MOVW, rd)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Words from ScePspemu (firmware 3.60), as objdump shows them.

    #[test]
    fn reads_scepspemus_suspend_flag_address() {
        // 8106a82e: f244 66bc movw r6, #0x46bc; 8106a838: f2c8 1610 movt r6, #0x8110
        assert_eq!(movw_movt_value([0xF244, 0x66BC], [0xF2C8, 0x1610], 6), Some(0x8110_46BC));
    }

    #[test]
    fn the_i_bit_and_high_registers() {
        // f643 408c movw r0, #0x3c8c (i set); f2ce 0a01 movt sl, #0xe001
        assert_eq!(mov_immediate([0xF643, 0x408C], MOVW, 0), Some(0x3C8C));
        assert_eq!(mov_immediate([0xF2CE, 0x0A01], MOVT, 10), Some(0xE001));
        // f243 7ad4 movw sl, #0x37d4 + that movt: the stack guard's address
        assert_eq!(movw_movt_value([0xF243, 0x7AD4], [0xF2CE, 0x0A01], 10), Some(0xE001_37D4));
    }

    #[test]
    fn anything_else_is_none() {
        // Wrong register, movw/movt swapped, a 16-bit instruction (movs r0, #0x89).
        assert_eq!(movw_movt_value([0xF244, 0x66BC], [0xF2C8, 0x1610], 7), None);
        assert_eq!(movw_movt_value([0xF2C8, 0x1610], [0xF244, 0x66BC], 6), None);
        assert_eq!(movw_movt_value([0x2089, 0xF2C8], [0xF2C8, 0x1610], 6), None);
    }
}
