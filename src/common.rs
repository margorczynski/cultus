use std::sync::Once;

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

pub fn bitstring_to_bit_vector(bitstring: &str) -> Vec<bool> {
    bitstring
        .chars()
        .map(|c| if c == '0' { false } else { true })
        .collect()
}

pub fn bit_vector_to_bitstring(bit_vector: &Vec<bool>) -> String {
    bit_vector
        .iter()
        .map(|&bit| if bit { '1' } else { '0' })
        .collect()
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
