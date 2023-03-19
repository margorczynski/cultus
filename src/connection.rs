#[derive(PartialEq, Debug)]
pub struct Connection {
    pub input: (InputConnectionType, usize),
    pub output: (OutputConnectionType, usize)
}

impl Connection {
    pub fn from_bitstring(s: &str, input_count: usize, output_count: usize, nand_count: usize, nor_count: usize, index_bits_count: usize) -> Option<Connection> {

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

        let total_bits = 4 + (2 * index_bits_count);

        if s.chars().count() < total_bits {
            return None;
        }

        //Decode the indexes for the input and output
        let input_index_binary = &s[4..index_bits_count + 4];
        let input_index = usize::from_str_radix(input_index_binary, 2).unwrap();

        let output_index_binary = &s[index_bits_count + 4..2* index_bits_count + 4];
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
            Connection {
                input: (input_type, input_index),
                output: (output_type, output_index)
            }
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

#[cfg(test)]
mod connection_tests {
    use super::*;

    #[test]
    fn from_binary_test() {
        let input_count = 7; //3 bits
        let output_count = 6; // 3 bits
        let nand_count = 12; //4 bits
        let nor_count = 4; //2 bits
        let index_bits_count = 4; //max(3,3,4,2)

        //string = 2 Input Type bits | 2 Output Type bits | ? Input index bits | ? Output index bits
        //3 connections - OK
        let expected_1 = Connection {
            input: (InputConnectionType::Input, 4),
            output: (OutputConnectionType::Output, 3),
        };
        let expected_1_str = "000001000011";

        let expected_2 = Connection {
            input: (InputConnectionType::Input, 3),
            output: (OutputConnectionType::NAND, 0),
        };
        let expected_2_str = "000100110000";

        let expected_3 = Connection {
            input: (InputConnectionType::NOR, 2),
            output: (OutputConnectionType::Output, 2),
        };
        let expected_3_str = "100000100010";

        //Wrong string
        let wrong_too_short_str = "100";
        let wrong_empty_str = "";
        let input_type_too_big_str = "110100000000";
        let output_type_too_big_str = "001100000000";
        let input_index_too_big_str = "000101110000";
        let output_index_too_big_str = "000100111111";

        //Results - OK
        let result_1 = Connection::from_bitstring(expected_1_str, input_count, output_count, nand_count, nor_count, index_bits_count).unwrap();
        let result_2 = Connection::from_bitstring(expected_2_str, input_count, output_count, nand_count, nor_count, index_bits_count).unwrap();
        let result_3 = Connection::from_bitstring(expected_3_str, input_count, output_count, nand_count, nor_count, index_bits_count).unwrap();
        //Results - Wrong
        let wrong_too_short = Connection::from_bitstring(wrong_too_short_str, input_count, output_count, nand_count, nor_count, index_bits_count);
        let wrong_empty = Connection::from_bitstring(wrong_empty_str, input_count, output_count, nand_count, nor_count, index_bits_count);
        let input_type_too_big = Connection::from_bitstring(input_type_too_big_str, input_count, output_count, nand_count, nor_count, index_bits_count);
        let output_type_too_big = Connection::from_bitstring(output_type_too_big_str, input_count, output_count, nand_count, nor_count, index_bits_count);
        let input_index_too_big = Connection::from_bitstring(input_index_too_big_str, input_count, output_count, nand_count, nor_count, index_bits_count);
        let output_index_too_big = Connection::from_bitstring(output_index_too_big_str, input_count, output_count, nand_count, nor_count, index_bits_count);
        
        assert_eq!(result_1, expected_1);
        assert_eq!(result_2, expected_2);
        assert_eq!(result_3, expected_3);

        assert!(wrong_too_short.is_none());
        assert!(wrong_empty.is_none());
        assert!(input_type_too_big.is_none());
        assert!(output_type_too_big.is_none());
        assert!(input_index_too_big.is_none());
        assert!(output_index_too_big.is_none());
    }
}