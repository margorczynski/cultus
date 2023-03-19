pub use crate::connection::*;

#[derive(PartialEq, Debug)]
struct Network {
    input_count: usize,
    output_count: usize,
    nand_count: usize,
    nor_count: usize,
    connections: Vec<Connection>
}

//TODO: Verify gate has exactly two inputs, output at most 1 input

impl Network {
    pub fn from_bitstring(s: &str, input_count: usize, output_count: usize, nand_count_bits: usize, nor_count_bits: usize) -> Option<Network> {
        println!("Decoding network from bitstring");

        //Number of bits in whole string not bigger than the sum of sizes of the attributes
        if s.chars().count() <= (nand_count_bits + nor_count_bits) as usize {
            return None
        }

        //s = NAND_COUNT | NOR_COUNT | CONNECTIONS...

        let nand_count_start_index = 0;
        let nor_count_start_index = nand_count_start_index + nand_count_bits;

        let nand_count_end_index = nand_count_start_index + nand_count_bits;
        let nor_count_end_index = nor_count_start_index + nor_count_bits;

        let nand_count_binary = &s[nand_count_start_index..nand_count_end_index];
        let nor_count_binary = &s[nor_count_start_index..nor_count_end_index];

        //Parse from binary string to u32 value. Panic if it fails - the assumption is that the string should always contain 0/1 only
        let nand_count = usize::from_str_radix(nand_count_binary, 2).unwrap();
        let nor_count = usize::from_str_radix(nor_count_binary, 2).unwrap();

        let connection_index_bits_count = *[input_count, output_count, nand_count, nor_count].map(|cnt| get_required_bits_count(cnt)).iter().max().unwrap();
        let connection_bits_count = 4 + (2 * connection_index_bits_count);

        println!("Connection bit size: {}", connection_bits_count);

        let connections_binary = &s[nor_count_end_index..];

        let mut ind = 0;
        let mut connections: Vec<Connection> = Vec::new();
        while ind < connections_binary.len() {
            let connection_with_len = Connection::from_bitstring(&connections_binary[ind..], input_count, output_count, nand_count, nor_count, connection_index_bits_count);
            match connection_with_len {
                Some(connection) => {
                    connections.push(connection);
                    ind = ind + connection_bits_count
                }
                None => {}
            }
        }

        //TODO: Clean connection - remove unused ones

        Some(
            Network {
                input_count,
                output_count,
                nand_count,
                nor_count,
                connections
            }
        )
    }
}

fn get_required_bits_count(num: usize) -> usize {
    (num as f32).log2().ceil() as usize
}


#[cfg(test)]
mod network_tests {
    use super::*;

    #[test]
    fn get_required_bits_count_test() {
        assert_eq!(get_required_bits_count(8), 3);
        assert_eq!(get_required_bits_count(12), 4);
        assert_eq!(get_required_bits_count(2), 1);
        assert_eq!(get_required_bits_count(100), 7);
    }

    #[test]
    fn from_bitstring_test() {
        let input_count = 7; //3 bits
        let output_count = 6; // 3 bits
        let nand_count = 12; //4 bits
        let nor_count = 4; //3 bits
        let index_bits_count = 4;

        let expected_connection = vec![
            Connection {
                input: (InputConnectionType::Input, 0),
                output: (OutputConnectionType::Output, 0),
            },
            Connection {
                input: (InputConnectionType::Input, 0),
                output: (OutputConnectionType::NAND, 0),
            },
            Connection {
                input: (InputConnectionType::Input, 1),
                output: (OutputConnectionType::NAND, 0),
            },
            Connection {
                input: (InputConnectionType::NAND, 0),
                output: (OutputConnectionType::NOR, 0),
            },
            Connection {
                input: (InputConnectionType::Input, 2),
                output: (OutputConnectionType::NOR, 0),
            },
            Connection {
                input: (InputConnectionType::NOR, 0),
                output: (OutputConnectionType::Output, 1),
            }
        ];

        let expected = Network {
            input_count,
            output_count,
            nand_count,
            nor_count,
            connections: expected_connection,
        };

        let expected_network_str = [
            "1100", //12 NANDs
            "100", //4 NORs
            "000000000000", //I0 -> O0
            "000100000000", //I0 -> NAND0
            "000100010000", //I1 -> NAND0
            "011000000000", //NAND0 -> NOR0
            "001000100000", //I2 -> NOR0
            "100000000001" //NOR0 -> O1
        ].join("");

        let result = Network::from_bitstring(&expected_network_str, input_count, output_count, 4, 3).unwrap();

        assert_eq!(result, expected);
    }
}