use std::fmt;

#[derive(Hash, PartialEq, Eq, Clone, Copy, PartialOrd, Ord)]
pub struct Connection {
    pub input: (InputConnectionType, usize),
    pub output: (OutputConnectionType, usize),
}

impl Connection {
    pub fn from_bitstring(
        s: &str,
        input_count: usize,
        output_count: usize,
        nand_count: usize,
        index_bits_count: usize,
    ) -> Option<Connection> {

        //TODO: Try to load the whole thing into memory and directly map it onto a struct?

        if s.chars().count() < 3 {
            return None;
        }

        //Decode type of input and output
        let input_type_binary = &s[0..1];
        let input_type_decimal = u32::from_str_radix(input_type_binary, 2).unwrap();

        let input_type = match input_type_decimal {
            0 => InputConnectionType::Input,
            1 => InputConnectionType::NAND,
            _ => {
                panic!("Non 0/1 value for output type")
            },
        };

        let output_type_binary = &s[1..2];
        let output_type_decimal = usize::from_str_radix(output_type_binary, 2).unwrap();

        let output_type = match output_type_decimal {
            0 => OutputConnectionType::Output,
            1 => OutputConnectionType::NAND,
            _ => {
                panic!("Non 0/1 value for output type")
            },
        };

        let total_bits = 2 + (2 * index_bits_count);

        if s.chars().count() < total_bits {
            return None;
        }

        //Decode the indexes for the input and output
        let input_index_binary = &s[2..index_bits_count + 2];
        let input_index = usize::from_str_radix(input_index_binary, 2).unwrap();

        let output_index_binary = &s[index_bits_count + 2..(2 * index_bits_count + 2)];
        let output_index = usize::from_str_radix(output_index_binary, 2).unwrap();

        match input_type {
            InputConnectionType::Input => {
                if input_index >= input_count {
                    return None;
                }
            }
            InputConnectionType::NAND => {
                if input_index >= nand_count {
                    return None;
                }
            }
        };

        match output_type {
            OutputConnectionType::Output => {
                if output_index >= output_count {
                    return None;
                }
            }
            OutputConnectionType::NAND => {
                if output_index >= nand_count {
                    return None;
                }
            }
        };

        match input_type {
            InputConnectionType::Input => {}
            InputConnectionType::NAND => {
                match output_type {
                    OutputConnectionType::Output => {}
                    OutputConnectionType::NAND => {
                        if input_index >= output_index {
                            return None;
                        }
                    }
                }
            }
        }

        Some(Connection {
            input: (input_type, input_index),
            output: (output_type, output_index),
        })
    }
}

impl fmt::Display for Connection {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let input_type_text = match self.input.0 {
            InputConnectionType::Input => "I".to_string(),
            InputConnectionType::NAND => "NAND".to_string(),
        };

        let output_type_text = match self.output.0 {
            OutputConnectionType::Output => "O".to_string(),
            OutputConnectionType::NAND => "NAND".to_string(),
        };

        write!(
            f,
            "{}({}) -> {}({})",
            input_type_text, self.input.1, output_type_text, self.output.1
        )
    }
}

impl fmt::Debug for Connection {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

#[derive(PartialEq, Eq, Hash, Clone, Debug, Copy, PartialOrd, Ord)]
pub enum InputConnectionType {
    Input,
    NAND,
}

impl PartialEq<OutputConnectionType> for InputConnectionType {
    fn eq(&self, other: &OutputConnectionType) -> bool {
        match self {
            InputConnectionType::Input => false,
            InputConnectionType::NAND => match other {
                OutputConnectionType::Output => false,
                OutputConnectionType::NAND => true,
            },
        }
    }
}

#[derive(PartialEq, Eq, Hash, Clone, Debug, Copy, PartialOrd, Ord)]
pub enum OutputConnectionType {
    Output,
    NAND,
}

impl PartialEq<InputConnectionType> for OutputConnectionType {
    fn eq(&self, other: &InputConnectionType) -> bool {
        match self {
            OutputConnectionType::Output => false,
            OutputConnectionType::NAND => match other {
                InputConnectionType::Input => false,
                InputConnectionType::NAND => true,
            },
        }
    }
}

#[cfg(test)]
mod connection_tests {
    use super::*;

    #[test]
    fn from_binary_test() {
        let input_count = 7; //3 bits
        let output_count = 6; // 3 bits
        let nand_count = 12; //4 bits
        let index_bits_count = 4; //max(3,3,4)

        //string = 1 Input Type bit | 1 Output Type bit | ? Input index bits | ? Output index bits
        //3 connections - OK
        let expected_1 = Connection {
            input: (InputConnectionType::Input, 4),
            output: (OutputConnectionType::Output, 3),
        };
        let expected_1_str = "0001000011";

        let expected_2 = Connection {
            input: (InputConnectionType::Input, 3),
            output: (OutputConnectionType::NAND, 0),
        };
        let expected_2_str = "0100110000";

        let expected_3 = Connection {
            input: (InputConnectionType::NAND, 2),
            output: (OutputConnectionType::Output, 2),
        };
        let expected_3_str = "1000100010";

        //Wrong string
        let wrong_too_short_str = "100";
        let wrong_empty_str = "";
        let input_index_too_big_str = "0101110000";
        let output_index_too_big_str = "0100111111";

        //Results - OK
        let result_1 = Connection::from_bitstring(
            expected_1_str,
            input_count,
            output_count,
            nand_count,
            index_bits_count,
        )
        .unwrap();
        let result_2 = Connection::from_bitstring(
            expected_2_str,
            input_count,
            output_count,
            nand_count,
            index_bits_count,
        )
        .unwrap();
        let result_3 = Connection::from_bitstring(
            expected_3_str,
            input_count,
            output_count,
            nand_count,
            index_bits_count,
        )
        .unwrap();
        //Results - Wrong
        let wrong_too_short = Connection::from_bitstring(
            wrong_too_short_str,
            input_count,
            output_count,
            nand_count,
            index_bits_count,
        );
        let wrong_empty = Connection::from_bitstring(
            wrong_empty_str,
            input_count,
            output_count,
            nand_count,
            index_bits_count,
        );
        let input_index_too_big = Connection::from_bitstring(
            input_index_too_big_str,
            input_count,
            output_count,
            nand_count,
            index_bits_count,
        );
        let output_index_too_big = Connection::from_bitstring(
            output_index_too_big_str,
            input_count,
            output_count,
            nand_count,
            index_bits_count,
        );

        assert_eq!(result_1, expected_1);
        assert_eq!(result_2, expected_2);
        assert_eq!(result_3, expected_3);

        assert!(wrong_too_short.is_none());
        assert!(wrong_empty.is_none());
        assert!(input_index_too_big.is_none());
        assert!(output_index_too_big.is_none());
    }
}
