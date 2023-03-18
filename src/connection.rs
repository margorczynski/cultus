pub struct Connection {
    input: (InputConnectionType, usize),
    output: (OutputConnectionType, usize)
}

impl Connection {
    fn from_bitstring(s: &str, input_count: usize, output_count: usize, nand_count: usize, nor_count: usize) -> Option<Connection> {
        //INPUT
        let input_type_binary = &s[0..1];
        let input_type_decimal = u32::from_str_radix(input_type_binary, 2).unwrap();

        let input_type = match input_type_decimal {
            0 => InputConnectionType::Input,
            1 => InputConnectionType::NAND,
            2 => InputConnectionType::NOR,
            _ => return None
        };

        let input_index_binary_len = match input_type {
            InputConnectionType::Input => get_required_bits_count(&input_count),
            InputConnectionType::NAND => get_required_bits_count(&nand_count),
            InputConnectionType::NOR => get_required_bits_count(&nor_count)
        };

        let input_index_binary = &s[2..input_index_binary_len + 2];
        let input_index = usize::from_str_radix(input_index_binary, 2).unwrap();

        //OUTPUT
        let output_type_binary = &s[input_index_binary_len + 2..input_index_binary_len + 4];
        let output_type_decimal = usize::from_str_radix(output_type_binary, 2).unwrap();

        let output_type = match output_type_decimal {
            0 => OutputConnectionType::Output,
            1 => OutputConnectionType::NAND,
            2 => OutputConnectionType::NOR,
            _ => return None
        };

        let output_index_binary_len = match output_type {
            OutputConnectionType::Output => get_required_bits_count(&output_count),
            OutputConnectionType::NAND => get_required_bits_count(&nand_count),
            OutputConnectionType::NOR => get_required_bits_count(&nor_count)
        };

        let output_index_binary = &s[2..output_index_binary_len + 2];
        let output_index = usize::from_str_radix(output_index_binary, 2).unwrap();

        //4 bits for the input and output types + all the bits for the input/output indexes must equal whole len of string
        if s.chars().count() != 4 + input_index_binary_len + output_index_binary_len {
            return None;
        }

        Some(
            Connection {
                input: (input_type, input_index),
                output: (output_type, output_index)
            }
        )
    }
}

enum InputConnectionType {
    Input,
    NAND,
    NOR
}

enum OutputConnectionType {
    Output,
    NAND,
    NOR
}

fn get_required_bits_count(num: &usize) -> usize {
    ((*num as u32) as f32).log2().ceil() as usize
}