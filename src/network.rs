pub use crate::connection::Connection;

struct Network {
    input_count: u32,
    output_count: u32,
    nand_count: u32,
    nor_count: u32,
    connections: Vec<Connection>
}

impl Network {
    fn from_bitstring(s: &str, input_count_bits: usize, output_count_bits: usize, nand_count_bits: usize, nor_count_bits: usize) -> Option<Network> {
        //Number of bits in whole string not bigger than the sum of sizes of the attributes
        if s.chars().count() <= (input_count_bits + output_count_bits + nand_count_bits + nor_count_bits) as usize {
            return None
        }

        //s = INPUT_COUNT | OUTPUT_COUNT | NAND_COUNT | NOR_COUNT | CONNECTIONS

        let input_count_start_index = 0;
        let output_count_start_index = input_count_bits;
        let nand_count_start_index = output_count_start_index + output_count_bits;
        let nor_count_start_index = nand_count_start_index + nand_count_bits;

        let input_count_end_index = input_count_bits;
        let output_count_end_index = input_count_end_index + output_count_bits;
        let nand_count_end_index = output_count_end_index + nand_count_bits;
        let nor_count_end_index = nand_count_end_index + nor_count_bits;

        let input_count_binary = &s[input_count_start_index..input_count_end_index];
        let output_count_binary = &s[output_count_start_index..output_count_end_index];
        let nand_count_binary = &s[nand_count_start_index..nand_count_end_index];
        let nor_count_binary = &s[nor_count_start_index..nor_count_end_index];

        //Parse from binary string to u32 value. Panic if it fails - the assumption is that the string should always contain 0/1 only
        let input_count = u32::from_str_radix(input_count_binary, 2).unwrap();
        let output_count = u32::from_str_radix(output_count_binary, 2).unwrap();
        let nand_count = u32::from_str_radix(nand_count_binary, 2).unwrap();
        let nor_count = u32::from_str_radix(nor_count_binary, 2).unwrap();



        let connections_binary = &s[nor_count_bits..];

        None
    }
}