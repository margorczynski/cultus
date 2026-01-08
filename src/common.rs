use std::sync::Once;

use bitvec::prelude::*;
use log::LevelFilter;
use simple_logger::SimpleLogger;

static INIT: Once = Once::new();

pub fn setup() {
    INIT.call_once(|| {
        SimpleLogger::new()
            .with_level(LevelFilter::Info)
            .without_timestamps()
            .init()
            .unwrap();
    });
}

/// Convert a bitstring ("01101") to a BitVec
pub fn bitstring_to_bitvec(bitstring: &str) -> BitVec<u64, Lsb0> {
    let mut bv = BitVec::with_capacity(bitstring.len());
    for byte in bitstring.as_bytes() {
        bv.push(*byte == b'1');
    }
    bv
}

/// Convert a BitVec to a bitstring
pub fn bitvec_to_bitstring(bv: &BitVec<u64, Lsb0>) -> String {
    bv.iter().map(|b| if *b { '1' } else { '0' }).collect()
}

/// Convert a bitstring to Vec<bool> (legacy compatibility)
pub fn bitstring_to_bit_vector(bitstring: &str) -> Vec<bool> {
    bitstring
        .chars()
        .map(|c| c != '0')
        .collect()
}

/// Convert Vec<bool> to bitstring (legacy compatibility)
pub fn bit_vector_to_bitstring(bit_vector: &[bool]) -> String {
    bit_vector
        .iter()
        .map(|&bit| if bit { '1' } else { '0' })
        .collect()
}

/// Convert BitVec to Vec<bool>
pub fn bitvec_to_vec_bool(bv: &BitVec<u64, Lsb0>) -> Vec<bool> {
    bv.iter().map(|b| *b).collect()
}

/// Convert Vec<bool> to BitVec
pub fn vec_bool_to_bitvec(v: &[bool]) -> BitVec<u64, Lsb0> {
    v.iter().collect()
}

#[cfg(test)]
mod common_tests {
    use super::*;

    #[test]
    fn bitstring_to_bit_vector_test() {
        setup();
        assert_eq!(
            bitstring_to_bit_vector("00100"),
            vec![false, false, true, false, false]
        );
        assert_eq!(
            bitstring_to_bit_vector("00111"),
            vec![false, false, true, true, true]
        );
        assert_eq!(
            bitstring_to_bit_vector("010101010101"),
            vec![false, true, false, true, false, true, false, true, false, true, false, true]
        );
    }

    #[test]
    fn bit_vector_to_bitstring_test() {
        setup();
        assert_eq!(
            bit_vector_to_bitstring(&vec![false, false, true, false, false]),
            "00100"
        );
        assert_eq!(
            bit_vector_to_bitstring(&vec![false, false, true, true, true]),
            "00111"
        );
        assert_eq!(
            bit_vector_to_bitstring(&vec![
                false, true, false, true, false, true, false, true, false, true, false, true
            ]),
            "010101010101"
        );
    }
}
