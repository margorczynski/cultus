#[derive(PartialEq, Debug)]
pub struct Connection {
    pub input: (InputConnectionType, usize),
    pub output: (OutputConnectionType, usize)
}

impl Connection {
    pub fn from_bitstring(s: &str, input_count: usize, output_count: usize, nand_count: usize, nor_count: usize) -> Option<(Connection, usize)> {

        if s.chars().count() < 4 {
            return None;
        }

        //Decode type of input and output
        let input_type_binary = &s[0..2];
        let input_type_decimal = u32::from_str_radix(input_type_binary, 2).unwrap();

        let input_type = match input_type_decimal {
            0 => InputConnectionType::Input,
            1 => InputConnectionType::NAND,
            2 => InputConnectionType::NOR,
            _ => return None
        };

        let output_type_binary = &s[2..4];
        let output_type_decimal = usize::from_str_radix(output_type_binary, 2).unwrap();

        let output_type = match output_type_decimal {
            0 => OutputConnectionType::Output,
            1 => OutputConnectionType::NAND,
            2 => OutputConnectionType::NOR,
            _ => return None
        };

        //Number of bits encoding the indexes
        let input_index_binary_len = match input_type {
            InputConnectionType::Input => get_required_bits_count(input_count),
            InputConnectionType::NAND => get_required_bits_count(nand_count),
            InputConnectionType::NOR => get_required_bits_count(nor_count)
        };

        let output_index_binary_len = match output_type {
            OutputConnectionType::Output => get_required_bits_count(output_count),
            OutputConnectionType::NAND => get_required_bits_count(nand_count),
            OutputConnectionType::NOR => get_required_bits_count(nor_count)
        };

        let total_bits = 4 + input_index_binary_len + output_index_binary_len;

        if s.chars().count() < total_bits {
            return None;
        }

        //Decode the indexes for the input and output
        let input_index_binary = &s[4..input_index_binary_len + 4];
        let input_index = usize::from_str_radix(input_index_binary, 2).unwrap();

        let output_index_binary = &s[input_index_binary_len + 4..input_index_binary_len + output_index_binary_len + 4];
        let output_index = usize::from_str_radix(output_index_binary, 2).unwrap();

        match input_type {
            InputConnectionType::Input => if input_index >= input_count {return None},
            InputConnectionType::NAND => if input_index >= nand_count {return None},
            InputConnectionType::NOR => if input_index >= nor_count {return None},
        };

        match output_type {
            OutputConnectionType::Output => if output_index >= output_count {return None},
            OutputConnectionType::NAND => if output_index >= nand_count {return None},
            OutputConnectionType::NOR => if output_index >= nor_count {return None},
        };

        Some(
            (Connection {
                input: (input_type, input_index),
                output: (output_type, output_index)
            }, total_bits)
        )
    }
}

#[derive(PartialEq, Debug)]
pub enum InputConnectionType {
    Input,
    NAND,
    NOR
}

#[derive(PartialEq, Debug)]
pub enum OutputConnectionType {
    Output,
    NAND,
    NOR
}

fn get_required_bits_count(num: usize) -> usize {
    (num as f32).log2().ceil() as usize
}

#[cfg(test)]
mod connection_tests {
    use super::*;

    #[test]
    fn get_required_bits_count_test() {
        assert_eq!(get_required_bits_count(8), 3);
        assert_eq!(get_required_bits_count(12), 4);
        assert_eq!(get_required_bits_count(2), 1);
        assert_eq!(get_required_bits_count(100), 7);
    }

    #[test]
    fn from_binary_test() {
        let input_count = 7; //3 bits
        let output_count = 6; // 3 bits
        let nand_count = 12; //4 bits
        let nor_count = 4; //2 bits

        let input_index_bits = get_required_bits_count(input_count);
        let output_index_bits = get_required_bits_count(output_count);
        let nand_index_bits = get_required_bits_count(nand_count);
        let nor_index_bits = get_required_bits_count(nor_count);

        //string = 2 Input Type bits | 2 Output Type bits | ? Input index bits | ? Output index bits
        //3 connections - OK
        let expected_1 = Connection {
            input: (InputConnectionType::Input, 4),
            output: (OutputConnectionType::Output, 3),
        };
        let expected_1_str = "0000100011";

        let expected_2 = Connection {
            input: (InputConnectionType::Input, 3),
            output: (OutputConnectionType::NAND, 0),
        };
        let expected_2_str = "00010110000";

        let expected_3 = Connection {
            input: (InputConnectionType::NOR, 2),
            output: (OutputConnectionType::Output, 2),
        };
        let expected_3_str = "100010010";

        //Wrong string
        let wrong_too_short_str = "100";
        let wrong_empty_str = "";
        let input_type_too_big_str = "11010110000";
        let output_type_too_big_str = "00110110000";
        let input_index_too_big_str = "00011110000";
        let output_index_too_big_str = "00010111111";

        //Results - OK
        let result_1 = Connection::from_bitstring(expected_1_str, input_count, output_count, nand_count, nor_count).unwrap();
        let result_2 = Connection::from_bitstring(expected_2_str, input_count, output_count, nand_count, nor_count).unwrap();
        let result_3 = Connection::from_bitstring(expected_3_str, input_count, output_count, nand_count, nor_count).unwrap();
        //Results - Wrong
        let wrong_too_short = Connection::from_bitstring(wrong_too_short_str, input_count, output_count, nand_count, nor_count);
        let wrong_empty = Connection::from_bitstring(wrong_empty_str, input_count, output_count, nand_count, nor_count);
        let input_type_too_big = Connection::from_bitstring(input_type_too_big_str, input_count, output_count, nand_count, nor_count);
        let output_type_too_big = Connection::from_bitstring(output_type_too_big_str, input_count, output_count, nand_count, nor_count);
        let input_index_too_big = Connection::from_bitstring(input_index_too_big_str, input_count, output_count, nand_count, nor_count);
        let output_index_too_big = Connection::from_bitstring(output_index_too_big_str, input_count, output_count, nand_count, nor_count);
        
        assert_eq!(result_1.0, expected_1);
        assert_eq!(result_1.1, expected_1_str.chars().count());

        assert_eq!(result_2.0, expected_2);
        assert_eq!(result_2.1, expected_2_str.chars().count());

        assert_eq!(result_3.0, expected_3);
        assert_eq!(result_3.1, expected_3_str.chars().count());

        assert!(wrong_too_short.is_none());
        assert!(wrong_empty.is_none());
        assert!(input_type_too_big.is_none());
        assert!(output_type_too_big.is_none());
        assert!(input_index_too_big.is_none());
        assert!(output_index_too_big.is_none());
    }
}