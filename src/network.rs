pub use crate::connection::Connection;

struct Network {
    input_count: usize,
    output_count: usize,
    nand_count: usize,
    nor_count: usize,
    connections: Vec<Connection>
}

impl Network {
    fn from_bitstring(s: &str, input_count: usize, output_count: usize, nand_count_bits: usize, nor_count_bits: usize) -> Option<Network> {
        //Number of bits in whole string not bigger than the sum of sizes of the attributes
        if s.chars().count() <= (nand_count_bits + nor_count_bits) as usize {
            return None
        }

        //s = NAND_COUNT | NOR_COUNT | CONNECTIONS...

        let nand_count_start_index = 0;
        let nor_count_start_index = nand_count_start_index + nand_count_bits;

        let nand_count_end_index = nand_count_bits - 1;
        let nor_count_end_index = nand_count_end_index + nor_count_bits - 1;

        let nand_count_binary = &s[nand_count_start_index..nand_count_end_index];
        let nor_count_binary = &s[nor_count_start_index..nor_count_end_index];

        //Parse from binary string to u32 value. Panic if it fails - the assumption is that the string should always contain 0/1 only
        let nand_count = usize::from_str_radix(nand_count_binary, 2).unwrap();
        let nor_count = usize::from_str_radix(nor_count_binary, 2).unwrap();



        let connections_binary = &s[nor_count_end_index+1..];

        let mut ind = 0;
        let mut connections: Vec<Connection> = Vec::new();
        while ind < connections_binary.len() {
            let connection_with_len = Connection::from_bitstring(&connections_binary[ind..], input_count, output_count, nand_count, nor_count);
            match connection_with_len {
                Some((connection, bit_len)) => {
                    connections.push(connection);
                    ind = ind + bit_len
                }
                None => {}
            }
        }

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