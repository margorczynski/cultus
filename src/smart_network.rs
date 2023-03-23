use std::collections::{HashMap, HashSet};
use log::debug;
pub use crate::network::*;

struct SmartNetwork {
    network: Network,
    memory_addr_input_count: usize,
    memory_rw_input_count: usize,
    get_network_output_fn: Box<dyn Fn(&Vec<bool>) -> Vec<bool>>,
    memory: HashMap<Vec<bool>, Vec<bool>>,
    current_memory_output: Option<Vec<bool>>
}

impl SmartNetwork {
    pub fn from_bitstring(s: &str, input_count: usize, output_count: usize, nand_count_bits: usize, nor_count_bits: usize, memory_addr_bits: usize, memory_rw_bits: usize) -> SmartNetwork {

        let total_input_count = input_count + memory_rw_bits;
        let total_output_count = output_count + (2 * memory_addr_bits) + memory_rw_bits;

        let mut network = Network::from_bitstring(s, total_input_count, total_output_count, nand_count_bits, nor_count_bits).unwrap().clone();

        let output_fn = Box::new(Network::get_outputs_computation_fn(total_output_count, &network.connections));

        network.clean_connections();

        SmartNetwork {
            network,
            memory_addr_input_count: memory_addr_bits,
            memory_rw_input_count: memory_rw_bits,
            get_network_output_fn: output_fn,
            memory: HashMap::new(),
            current_memory_output: None
        }
    }

    pub fn get_required_bits_for_bitstring(input_count: usize, output_count: usize, nand_count_bits: usize, nor_count_bits: usize, memory_addr_input_count: usize, memory_rw_input_count: usize, connection_count: usize) -> usize {
        let total_input_count = input_count + memory_rw_input_count;
        let total_output_count = output_count + (2 * memory_addr_input_count) + memory_rw_input_count;

        debug!("Total input count: {}", total_input_count);
        debug!("Total output count: {}", total_output_count);

        let input_output_max_index_bits_count: usize = [total_input_count, total_output_count].map(|cnt| get_required_bits_count(cnt)).iter().max().unwrap().clone();
        let connection_index_bits_count = [input_output_max_index_bits_count, nand_count_bits, nor_count_bits].iter().max().unwrap().clone();
        let connection_bits_count = 4 + (2 * connection_index_bits_count);

        debug!("IO bit count: {}", input_output_max_index_bits_count);
        debug!("Connection index bits count: {}", connection_index_bits_count);
        debug!("Connection bits count: {}", connection_bits_count);

        nand_count_bits + nor_count_bits + (connection_count * connection_bits_count)
    }

    pub fn compute_output(&mut self, input: &Vec<bool>) -> Vec<bool> {

        let network_output_fn = self.get_network_output_fn.as_ref();
        //TODO: Create full input vec = input + mem output
        let mut full_input: Vec<bool> = Vec::new();

        full_input.extend(input);
        full_input.extend(&self.current_memory_output.clone().unwrap_or(vec![]));

/*        if full_input.len() != self.network.input_count {
            panic!("Input + memory bits len isn't equal to underlying network input len");
        }*/

        let network_output: Vec<bool> = network_output_fn(&full_input);

        let mem_input_start_idx = network_output.len() - (2 * self.memory_addr_input_count) - self.memory_rw_input_count;
        let mem_output_addr_idx = mem_input_start_idx;
        let mem_input_addr_idx = mem_input_start_idx + self.memory_addr_input_count;
        let mem_input_idx = mem_input_addr_idx + self.memory_rw_input_count;

        let mem_output_addr = &network_output[mem_output_addr_idx..(mem_output_addr_idx + self.memory_addr_input_count)];
        let mem_input_addr = &network_output[mem_input_addr_idx..(mem_input_addr_idx + self.memory_addr_input_count)];
        let mem_input = &network_output[mem_input_idx..(mem_input_idx + self.memory_rw_input_count)];

        //Output | Output mem addr | Input mem addr | Mem input
        //TODO: Put output into memory based on addr values, select curr mem output based on address

        self.current_memory_output = self.memory.get(mem_output_addr).cloned();
        self.memory.insert(Vec::from(mem_input_addr), Vec::from(mem_input));

        network_output
    }
}

#[cfg(test)]
mod smart_network_tests {
    use super::*;

    #[test]
    fn from_bitstring_test() {
        let input_count = 12; //4 bits
        let output_count = 12; //4 bits

        let expected_network_str = [
            "1100", //12 NANDs
            "00100", //4 NORs
            "000000000000", //I0 -> O0
            "000100000000", //I0 -> NAND0
            "000100010000", //I1 -> NAND0
            "011000000000", //NAND0 -> NOR0
            "001000100000", //I2 -> NOR0
            "100000000001" //NOR0 -> O1
        ].join("");

        let result = SmartNetwork::from_bitstring(&expected_network_str, input_count, output_count, 4, 5, 16, 64);

        debug!("CONNS: {:?}", result.network.connections);

        //12 + 64 mem
        assert_eq!(result.network.input_count, 76);
        //12 + 16 + 16 + 64
        assert_eq!(result.network.output_count, 108);
        //4 cleaned out
        assert_eq!(result.network.connections.len(), 2);
        assert_eq!(result.memory_addr_input_count, 16);
        assert_eq!(result.memory_rw_input_count, 64);
    }

    #[test]
    fn get_required_bits_for_bitstring_test() {
        let result = SmartNetwork::get_required_bits_for_bitstring(9, 18, 16, 16, 8, 32, 100);

        assert_eq!(result, 3632)
    }

/*    #[test]
    fn compute_output_test() {

        let mut network = Network {
            input_count: 4,
            output_count: 4,
            connections: HashSet::from_iter(vec![

            ]),
        };

        network.clean_connections();

        //TODO: Set the output func
        let total_output_count = 16;
        let output_fn = Box::new(Network::get_outputs_computation_fn(total_output_count, &network.connections));

        let mut smart_network = SmartNetwork {
            network,
            memory_addr_input_count: 2,
            memory_rw_input_count: 4,
            get_network_output_fn: output_fn,
            memory: HashMap::new(),
            current_memory_output: None,
        };

        let input = vec![

        ];

        //let result = smart_network.compute_output()

        assert_eq!(1, 2)
    }*/
}