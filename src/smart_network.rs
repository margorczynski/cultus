use std::collections::HashMap;
pub use crate::network::*;

struct SmartNetwork {
    network: Network,
    memory_addr_bits: usize,
    memory_rw_bits: usize,
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
            memory_addr_bits,
            memory_rw_bits,
            get_network_output_fn: output_fn,
            memory: HashMap::new(),
            current_memory_output: None
        }
    }

    pub fn get_required_bits_for_bitstring(input_count: usize, output_count: usize, nand_count_bits: usize, nor_count_bits: usize, memory_addr_bits: usize, memory_rw_bits: usize, connection_count: usize) -> usize {
        let total_input_count = input_count + memory_rw_bits;
        let total_output_count = output_count + (2 * memory_addr_bits) + memory_rw_bits;

        let input_output_max_index_bits_count: usize = [total_input_count, total_output_count].map(|cnt| get_required_bits_count(cnt)).iter().max().unwrap().clone();
        let connection_index_bits_count = [input_output_max_index_bits_count, nand_count_bits, nor_count_bits].iter().max().unwrap().clone();
        let connection_bits_count = 4 + (2 * connection_index_bits_count);

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

        let mem_input_start_idx = network_output.len() - (2 * self.memory_addr_bits) - self.memory_rw_bits;
        let mem_output_addr_idx = mem_input_start_idx;
        let mem_input_addr_idx = mem_input_start_idx + self.memory_addr_bits;
        let mem_input_idx = mem_input_addr_idx + self.memory_rw_bits;

        let mem_output_addr = &network_output[mem_output_addr_idx..(mem_output_addr_idx + self.memory_addr_bits)];
        let mem_input_addr = &network_output[mem_input_addr_idx..(mem_input_addr_idx + self.memory_addr_bits)];
        let mem_input = &network_output[mem_input_idx..(mem_input_idx + self.memory_rw_bits)];

        //Output | Output mem addr | Input mem addr | Mem input
        //TODO: Put output into memory based on addr values, select curr mem output based on address

        self.current_memory_output = self.memory.get(mem_output_addr).cloned();
        self.memory.insert(Vec::from(mem_input_addr), Vec::from(mem_input));

        network_output
    }
}