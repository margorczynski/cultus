use std::time::Instant;

use log::{debug, trace};

use super::network::*;
use crate::common::get_required_bits_count;

/// Bounded memory storage for the smart network.
/// Uses a fixed-size array indexed by address bits converted to usize.
pub struct SmartNetworkMemory {
    data: Vec<Vec<bool>>,
}

impl SmartNetworkMemory {
    pub fn new(addr_bits: usize, data_width: usize) -> Self {
        let size = 1 << addr_bits; // 2^addr_bits entries
        SmartNetworkMemory {
            data: vec![vec![false; data_width]; size],
        }
    }

    fn addr_to_index(addr: &[bool]) -> usize {
        addr.iter()
            .enumerate()
            .map(|(i, &b)| if b { 1 << i } else { 0 })
            .sum()
    }

    pub fn read(&self, addr: &[bool]) -> &[bool] {
        let idx = Self::addr_to_index(addr);
        &self.data[idx]
    }

    pub fn write(&mut self, addr: &[bool], value: &[bool]) {
        let idx = Self::addr_to_index(addr);
        self.data[idx].copy_from_slice(value);
    }
}

#[allow(dead_code)]
pub struct SmartNetwork {
    network: Network,
    memory_addr_input_count: usize,
    memory_rw_input_count: usize,
    get_network_output_fn: Box<dyn Fn(&Vec<bool>) -> Vec<bool> + Send + Sync>,
    memory: SmartNetworkMemory,
    current_memory_output: Vec<bool>,
}

impl SmartNetwork {
    /// Create a SmartNetwork directly from a bit slice (more efficient than string parsing)
    pub fn from_bits(
        bits: &[bool],
        input_count: usize,
        output_count: usize,
        nand_count_bits: usize,
        memory_addr_bits: usize,
        memory_rw_bits: usize,
    ) -> SmartNetwork {
        let total_input_count = input_count + memory_rw_bits;
        let total_output_count = output_count + (2 * memory_addr_bits) + memory_rw_bits;

        let from_bits_start = Instant::now();
        let mut network =
            Network::from_bits(bits, total_input_count, total_output_count, nand_count_bits)
                .unwrap();
        trace!("from_bits_elapsed={:?}", from_bits_start.elapsed());

        let clean_connections_start = Instant::now();
        network.clean_connections();
        trace!(
            "clean_connections_elapsed={:?}",
            clean_connections_start.elapsed()
        );

        let get_outputs_computation_fn_start = Instant::now();
        let output_fn = Box::new(Network::get_outputs_computation_fn(
            total_output_count,
            &network.connections,
        ));
        trace!(
            "get_outputs_computation_fn_elapsed={:?}",
            get_outputs_computation_fn_start.elapsed()
        );

        SmartNetwork {
            network,
            memory_addr_input_count: memory_addr_bits,
            memory_rw_input_count: memory_rw_bits,
            get_network_output_fn: output_fn,
            memory: SmartNetworkMemory::new(memory_addr_bits, memory_rw_bits),
            current_memory_output: vec![false; memory_rw_bits],
        }
    }

    pub fn from_bitstring(
        s: &str,
        input_count: usize,
        output_count: usize,
        nand_count_bits: usize,
        memory_addr_bits: usize,
        memory_rw_bits: usize,
    ) -> SmartNetwork {
        let total_input_count = input_count + memory_rw_bits;
        let total_output_count = output_count + (2 * memory_addr_bits) + memory_rw_bits;

        let from_bitstring_start = Instant::now();
        let mut network =
            Network::from_bitstring(s, total_input_count, total_output_count, nand_count_bits)
                .unwrap()
                .clone();
        trace!(
            "from_bitstring_elapsed={:?}",
            from_bitstring_start.elapsed()
        );

        let clean_connections_start = Instant::now();
        network.clean_connections();
        trace!(
            "clean_connections_elapsed={:?}",
            clean_connections_start.elapsed()
        );

        let get_outputs_computation_fn_start = Instant::now();
        let output_fn = Box::new(Network::get_outputs_computation_fn(
            total_output_count,
            &network.connections,
        ));
        trace!(
            "get_outputs_computation_fn_elapsed={:?}",
            get_outputs_computation_fn_start.elapsed()
        );

        SmartNetwork {
            network,
            memory_addr_input_count: memory_addr_bits,
            memory_rw_input_count: memory_rw_bits,
            get_network_output_fn: output_fn,
            memory: SmartNetworkMemory::new(memory_addr_bits, memory_rw_bits),
            current_memory_output: vec![false; memory_rw_bits],
        }
    }

    pub fn get_required_bits_for_bitstring(
        input_count: usize,
        output_count: usize,
        nand_count_bits: usize,
        memory_addr_input_count: usize,
        memory_rw_input_count: usize,
        connection_count: usize,
    ) -> usize {
        let total_input_count = input_count + memory_rw_input_count;
        let total_output_count =
            output_count + (2 * memory_addr_input_count) + memory_rw_input_count;

        debug!("Total input count: {}", total_input_count);
        debug!("Total output count: {}", total_output_count);

        let input_output_max_index_bits_count: usize = [total_input_count, total_output_count]
            .map(|cnt| get_required_bits_count(cnt))
            .iter()
            .max()
            .unwrap()
            .clone();
        let connection_index_bits_count = [input_output_max_index_bits_count, nand_count_bits]
            .iter()
            .max()
            .unwrap()
            .clone();
        let connection_bits_count = 2 + (2 * connection_index_bits_count);

        debug!("IO bit count: {}", input_output_max_index_bits_count);
        debug!(
            "Connection index bits count: {}",
            connection_index_bits_count
        );
        debug!("Connection bits count: {}", connection_bits_count);

        nand_count_bits + (connection_count * connection_bits_count)
    }

    pub fn compute_output(&mut self, input: &[bool]) -> Vec<bool> {
        let network_output_fn = self.get_network_output_fn.as_ref();
        let mut full_input: Vec<bool> = Vec::with_capacity(input.len() + self.memory_rw_input_count);

        full_input.extend(input);
        full_input.extend(&self.current_memory_output);

        let network_output: Vec<bool> = network_output_fn(&full_input);

        // The first index of the first output that controls the memory
        let outputs_count_without_memory =
            network_output.len() - (2 * self.memory_addr_input_count) - self.memory_rw_input_count;
        let mem_inputs_start_idx = outputs_count_without_memory;
        // The indexes of the outputs for memory operations
        let mem_output_addr_idx = mem_inputs_start_idx;
        let mem_input_addr_idx = mem_output_addr_idx + self.memory_addr_input_count;
        let mem_input_idx = mem_input_addr_idx + self.memory_addr_input_count;

        let mem_output_addr = &network_output
            [mem_output_addr_idx..(mem_output_addr_idx + self.memory_addr_input_count)];
        let mem_input_addr = &network_output
            [mem_input_addr_idx..(mem_input_addr_idx + self.memory_addr_input_count)];
        let mem_input =
            &network_output[mem_input_idx..(mem_input_idx + self.memory_rw_input_count)];

        // Read from memory at output address
        self.current_memory_output = self.memory.read(mem_output_addr).to_vec();
        // Write to memory at input address
        self.memory.write(mem_input_addr, mem_input);

        debug!("Memory output for next compute: {:?}", self.current_memory_output);

        network_output
            .iter()
            .take(outputs_count_without_memory)
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod smart_network_tests {
    use super::*;
    use crate::smart_network::connection::*;
    use std::collections::HashSet;

    #[test]
    fn from_bitstring_test() {
        let input_count = 12;
        let output_count = 12;

        let expected_network_str = [
            "1100",             //12 NANDs
            "0000000000000000", //I0 -> O0
            "0100000000000000", //I0 -> NAND0
            "0100000010000000", //I1 -> NAND0
            "1100000000000000", //NAND0 -> NAND0
            "0100000100000000", //I2 -> NAND0
            "1000000000000001", //NAND0 -> O1
            "0100000010000001", //O1 -> NAND1
            "1100000010000010", //NAND1 -> NAND2
        ]
        .join("");

        let result = SmartNetwork::from_bitstring(
            &expected_network_str,
            input_count,
            output_count,
            4,
            16,
            64,
        );

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
        let output_fn = Box::new(Network::get_outputs_computation_fn(
            total_output_count,
            &network.connections,
        ));

        let mut smart_network = SmartNetwork {
            network,
            memory_addr_input_count: 1,
            memory_rw_input_count: 2,
            get_network_output_fn: output_fn,
            memory: SmartNetworkMemory::new(1, 2),
            current_memory_output: vec![false; 2],
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
