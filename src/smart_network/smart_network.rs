use std::collections::{HashMap, HashSet};
use log::debug;

use super::network::*;
use super::connection::*;

struct SmartNetwork {
    network: Network,
    memory_addr_input_count: usize,
    memory_rw_input_count: usize,
    get_network_output_fn: Box<dyn Fn(&Vec<bool>) -> Vec<bool>>,
    memory: HashMap<Vec<bool>, Vec<bool>>,
    current_memory_output: Option<Vec<bool>>
}

impl SmartNetwork {
    pub fn from_bitstring(s: &str, input_count: usize, output_count: usize, nand_count_bits: usize, memory_addr_bits: usize, memory_rw_bits: usize) -> SmartNetwork {

        let total_input_count = input_count + memory_rw_bits;
        let total_output_count = output_count + (2 * memory_addr_bits) + memory_rw_bits;

        let mut network = Network::from_bitstring(s, total_input_count, total_output_count, nand_count_bits).unwrap().clone();

        network.clean_connections();

        let output_fn = Box::new(Network::get_outputs_computation_fn(total_output_count, &network.connections));

        SmartNetwork {
            network,
            memory_addr_input_count: memory_addr_bits,
            memory_rw_input_count: memory_rw_bits,
            get_network_output_fn: output_fn,
            memory: HashMap::new(),
            current_memory_output: None
        }
    }

    pub fn get_required_bits_for_bitstring(input_count: usize, output_count: usize, nand_count_bits: usize, memory_addr_input_count: usize, memory_rw_input_count: usize, connection_count: usize) -> usize {
        let total_input_count = input_count + memory_rw_input_count;
        let total_output_count = output_count + (2 * memory_addr_input_count) + memory_rw_input_count;

        debug!("Total input count: {}", total_input_count);
        debug!("Total output count: {}", total_output_count);

        let input_output_max_index_bits_count: usize = [total_input_count, total_output_count].map(|cnt| get_required_bits_count(cnt)).iter().max().unwrap().clone();
        let connection_index_bits_count = [input_output_max_index_bits_count, nand_count_bits].iter().max().unwrap().clone();
        let connection_bits_count = 2 + (2 * connection_index_bits_count);

        debug!("IO bit count: {}", input_output_max_index_bits_count);
        debug!("Connection index bits count: {}", connection_index_bits_count);
        debug!("Connection bits count: {}", connection_bits_count);

        nand_count_bits + (connection_count * connection_bits_count)
    }

    pub fn compute_output(&mut self, input: &[bool]) -> Vec<bool> {

        let network_output_fn = self.get_network_output_fn.as_ref();
        let mut full_input: Vec<bool> = Vec::new();

        full_input.extend(input);
        full_input.extend(&self.current_memory_output.clone().unwrap_or(vec![false; self.memory_rw_input_count]));

        let network_output: Vec<bool> = network_output_fn(&full_input);

        //The first index of the first output that controls the memory
        let outputs_count_without_memory = network_output.len() - (2 * self.memory_addr_input_count) - self.memory_rw_input_count;
        let mem_inputs_start_idx = outputs_count_without_memory;
        //The indexes of the outputs for
        let mem_output_addr_idx = mem_inputs_start_idx;
        let mem_input_addr_idx = mem_output_addr_idx + self.memory_addr_input_count;
        let mem_input_idx = mem_input_addr_idx + self.memory_addr_input_count;

        let mem_output_addr = &network_output[mem_output_addr_idx..(mem_output_addr_idx + self.memory_addr_input_count)];
        let mem_input_addr = &network_output[mem_input_addr_idx..(mem_input_addr_idx + self.memory_addr_input_count)];
        let mem_input = &network_output[mem_input_idx..(mem_input_idx + self.memory_rw_input_count)];

        //Whole output = Output | Output mem addr | Input mem addr | Mem input
        self.current_memory_output = self.memory.get(mem_output_addr).cloned();
        self.memory.insert(Vec::from(mem_input_addr), Vec::from(mem_input));

        debug!("Memory after compute: {:?}", self.memory);
        debug!("Memory output for next compute: {:?}", self.current_memory_output);

        network_output.iter().take(outputs_count_without_memory).cloned().collect()
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
            "0000000000000000", //I0 -> O0
            "0100000000000000", //I0 -> NAND0
            "0100000010000000", //I1 -> NAND0
            "1100000000000000", //NAND0 -> NAND0
            "0100000100000000", //I2 -> NAND0
            "1000000000000001" //NAND0 -> O1
        ].join("");

        let result = SmartNetwork::from_bitstring(&expected_network_str, input_count, output_count, 4, 16, 64);

        debug!("CONNS: {:?}", result.network.connections);

        //12 + 64 mem
        assert_eq!(result.network.input_count, 76);
        //12 + 16 + 16 + 64
        assert_eq!(result.network.output_count, 108);
        //4 cleaned out
        assert_eq!(result.network.connections.len(), 4);
        assert_eq!(result.memory_addr_input_count, 16);
        assert_eq!(result.memory_rw_input_count, 64);
    }

    #[test]
    fn get_required_bits_for_bitstring_test() {
        let result = SmartNetwork::get_required_bits_for_bitstring(9, 18, 16, 8, 32, 100);

        assert_eq!(result, 3416)
    }

    #[test]
    fn compute_output_test() {

        let mut network = Network {
            input_count: 4,
            output_count: 6,
            connections: HashSet::from_iter(vec![
                Connection {
                    input: (InputConnectionType::Input, 0),
                    output: (OutputConnectionType::NAND, 0),
                },
                Connection {
                    input: (InputConnectionType::Input, 1),
                    output: (OutputConnectionType::NAND, 0),
                },
                Connection {
                    input: (InputConnectionType::Input, 1),
                    output: (OutputConnectionType::NAND, 1),
                },
                Connection {
                    input: (InputConnectionType::Input, 2),
                    output: (OutputConnectionType::NAND, 1),
                },
                Connection {
                    input: (InputConnectionType::Input, 3),
                    output: (OutputConnectionType::Output, 5),
                },
                Connection {
                    input: (InputConnectionType::NAND, 0),
                    output: (OutputConnectionType::Output, 0),
                },
                Connection {
                    input: (InputConnectionType::NAND, 1),
                    output: (OutputConnectionType::Output, 1),
                },
                Connection {
                    input: (InputConnectionType::NAND, 1),
                    output: (OutputConnectionType::Output, 2),
                },
                Connection {
                    input: (InputConnectionType::NAND, 1),
                    output: (OutputConnectionType::Output, 3),
                },
                Connection {
                    input: (InputConnectionType::NAND, 1),
                    output: (OutputConnectionType::Output, 4),
                },
            ]),
        };

        network.clean_connections();

        let total_output_count = 6;
        let output_fn = Box::new(Network::get_outputs_computation_fn(total_output_count, &network.connections));

        let mut smart_network = SmartNetwork {
            network,
            memory_addr_input_count: 1,
            memory_rw_input_count: 2,
            get_network_output_fn: output_fn,
            memory: HashMap::new(),
            current_memory_output: None,
        };

        let input_1 = vec![true, false];
        let input_2 = vec![true, true];
        let input_3 = vec![true, true];

        let expected_1 = vec![true, true];
        let expected_2 = vec![false, true];
        let expected_3 = vec![false, false];

        let result_1 = smart_network.compute_output(&input_1);
        let result_2 = smart_network.compute_output(&input_2);
        let result_3 = smart_network.compute_output(&input_3);

        assert_eq!(result_1, expected_1);
        assert_eq!(result_2, expected_2);
        assert_eq!(result_3, expected_3);
    }
}