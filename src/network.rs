use std::collections::{HashMap, HashSet};

use log::debug;
use simple_logger::SimpleLogger;

pub use crate::connection::*;

#[derive(PartialEq, Debug)]
struct Network {
    input_count: usize,
    output_count: usize,
    connections: HashSet<Connection>
}

impl Network {

    /// Create a logic gate network from a string containing binary (0s and 1s)
    /// The amount of inputs and outputs of the network is constant and specified as arguments.
    /// The amount of bits (and thus maximum number) encoding the amount of NAND and NOR gates is constant and specified as arguments.
    /// Connections are deduplicated after decoding.
    /// Incorrect connections will be discarded and connections to a saturated input will be discarded in order of decoding.
    ///
    /// The string is in the form:
    /// NAND amount bits | NOR amount bits | Connection 1 input type | Connection 1 output type | Connection 1 input index | Connection 1 output index | Connection 2 ...
    pub fn from_bitstring(s: &str, input_count: usize, output_count: usize, nand_count_bits: usize, nor_count_bits: usize) -> Option<Network> {
        debug!("Decoding network from bitstring: {}", s);

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

        debug!("Connection bit size: {}", connection_bits_count);

        let connections_binary = &s[nor_count_end_index..];

        let mut ind = 0;
        let mut none_connection_count = 0;
        let mut over_saturated_connection_count = 0;
        let mut connections: HashSet<Connection> = HashSet::new();
        let mut input_counts: HashMap<(OutputConnectionType, usize), usize> = HashMap::new();
        while ind < connections_binary.len() {
            let connection_with_len = Connection::from_bitstring(&connections_binary[ind..], input_count, output_count, nand_count, nor_count, connection_index_bits_count);
            debug!("[IDX: {}] Processing from binary connection: {:?}", ind, connection_with_len);
            match connection_with_len {
                Some(connection) => {
                    let output = connection.output.clone();
                    let max_inputs_cnt = match output.0 {
                        OutputConnectionType::Output => 1,
                        OutputConnectionType::Gate(_) => 2
                    };
                    match input_counts.get(&output) {
                        None => {
                            connections.insert(connection);
                            input_counts.insert(output, 1);
                        }
                        Some(&count) => {
                            let new_input_count = count + 1;
                            if new_input_count <= max_inputs_cnt {
                                connections.insert(connection);
                                input_counts.insert(output, new_input_count);
                            } else {
                                over_saturated_connection_count = over_saturated_connection_count + 1;
                            }
                        }
                    }
                }
                None => {
                    none_connection_count = none_connection_count + 1;
                }
            }
            ind = ind + connection_bits_count;
        }

        debug!("Connections which couldn't be built from binary count: {}", none_connection_count);
        debug!("Connections discarded because of input saturation limit: {}", over_saturated_connection_count);

        Some(
            Network {
                input_count,
                output_count,
                connections
            }
        )
    }

    /// Clean up the network connections
    /// 1. Remove connections which create a cycle
    /// 2. Remove connections which are not computable (no path from input to output going through it, gate hasn't got 2 inputs and 1 output)
    pub fn clean_connections(&mut self) {
        debug!("Cleaning up connections. Starting count: {}", self.connections.len());

        let mut cleaned_up_connections: HashSet<Connection> = self.connections.clone();

        //Go through all the input connections and for each get the connections which cycle when exploring starting from that input
        let cycle_connections = cleaned_up_connections
            .iter()
            .filter(|&conn| conn.input.0 == InputConnectionType::Input)
            .map(|input_conn| Network::get_cycles(input_conn, &cleaned_up_connections))
            .reduce(|acc, e| acc.union(&e).copied().collect())
            .unwrap();

        //Remove the cycling connections
        cleaned_up_connections = cleaned_up_connections.difference(&cycle_connections).cloned().collect();

        //Remove connections of gates where the number of inputs or outputs is incorrect
        //This is done in an iterative manner until there are no more connections to remove
        let mut connections_to_remove: HashSet<Connection> = HashSet::new();
        loop {
            let gates_with_connections = Network::collect_gates_with_connections(&cleaned_up_connections);

            for ((gate, index), connections) in gates_with_connections {
                let input_connections: Vec<_> = connections.iter().filter(|&conn| conn.output.0 == OutputConnectionType::Gate(gate) && conn.output.1 == index).collect();
                let output_connections: Vec<_> = connections.iter().filter(|&conn| conn.input.0 == InputConnectionType::Gate(gate) && conn.input.1 == index).collect();

                if input_connections.len() != 2 || output_connections.len() < 1 {
                    connections_to_remove.extend(connections)
                }
            }

            if connections_to_remove.is_empty() {
                break
            }

            cleaned_up_connections = cleaned_up_connections.difference(&connections_to_remove).cloned().collect();

            connections_to_remove.clear();
        }

        debug!("Connections after cleanup count: {}", cleaned_up_connections.len());

        self.connections = cleaned_up_connections;
    }

    /// Get the closure that computes the output vector given an input vector
    pub fn get_outputs_computation_func(&self) -> impl Fn(Vec<bool>) -> Vec<bool> + '_ {

        let mut output_index_to_input_index_and_gates_stacks_map: HashMap<usize, (Vec<Gate>, Vec<usize>)> = HashMap::new();
        for output_connection in self.connections.iter().filter(|&conn| conn.output.0 == OutputConnectionType::Output) {
            let mut to_explore: Vec<&Connection> = Vec::new();
            let mut gate_stack: Vec<Gate> = Vec::new();
            let mut input_index_stack: Vec<usize> = Vec::new();

            to_explore.push(output_connection);

            while !to_explore.is_empty() {
                let connection = to_explore.pop().unwrap();

                match connection.input.0 {
                    InputConnectionType::Input => {
                        input_index_stack.push(connection.input.1)
                    }
                    InputConnectionType::Gate(gate) => {
                        gate_stack.push(gate)
                    }
                }

                let inputs: Vec<&Connection> =
                    self.connections.iter().filter(|&conn| conn.output.0 == connection.input.0 && conn.output.1 == connection.input.1).collect();

                to_explore.append(&mut inputs.clone());
            }

            debug!("Finished for connection {}, gate stack: {:?}, index stack: {:?}", output_connection, gate_stack, input_index_stack);

            output_index_to_input_index_and_gates_stacks_map.insert(output_connection.output.1, (gate_stack, input_index_stack));
        }

        move |input_bits: Vec<bool>| -> Vec<bool> {

            let mut output: Vec<bool> = Vec::new();
            for output_idx in 0..self.output_count {
                match output_index_to_input_index_and_gates_stacks_map.get(&output_idx) {
                    None => {
                        output.push(false)
                    },
                    Some((gates, input_indexes)) => {

                        debug!("Stacks for output {}: gates: {:?}, indexes: {:?}", output_idx, gates, input_indexes);
                        let mut value_bits_stack: Vec<bool> = input_indexes.iter().map(|&idx| input_bits.get(idx).unwrap().clone()).collect();

                        for gate in gates {
                            //After cleanup it should be impossible for this to throw
                            let first_value = value_bits_stack.pop().unwrap();
                            let second_value = value_bits_stack.pop().unwrap();

                            match gate {
                                Gate::NAND => {
                                    value_bits_stack.push(!(first_value && second_value))
                                }
                                Gate::NOR => {
                                    value_bits_stack.push(!(first_value || second_value))
                                }
                            }
                        }

                        //After cleanup it should be impossible for this to throw
                        output.push(value_bits_stack.first().unwrap().clone());
                    }
                }
            }
            output
        }
    }

    fn collect_gates_with_connections(connections: &HashSet<Connection>) -> HashMap<(Gate, usize), HashSet<Connection>> {
        let mut gates_with_connections: HashMap<(Gate, usize), HashSet<Connection>> = HashMap::new();

        for connection in connections {
            let input = connection.input;
            let output = connection.output;

            let mut add_gate_connection = |gate_type: Gate, index: usize| {
                let gate_id = (gate_type, index);
                match gates_with_connections.get_mut(&gate_id) {
                    None => {
                        let connections_set = HashSet::from_iter(vec![connection.clone()]);
                        gates_with_connections.insert(gate_id, connections_set);
                    }
                    Some(curr_connections_vec) => {
                        curr_connections_vec.insert(connection.clone());
                    }
                }
            };

            match input.0 {
                InputConnectionType::Input => {}
                InputConnectionType::Gate(gate_type) => add_gate_connection(gate_type, input.1)
            }
            match output.0 {
                OutputConnectionType::Output => {}
                OutputConnectionType::Gate(gate_type) => add_gate_connection(gate_type, output.1)
            }
        }

        gates_with_connections
    }

    fn get_cycles(start_connection: &Connection, connections: &HashSet<Connection>) -> HashSet<Connection> {
        let mut to_explore: Vec<&Connection> = Vec::new();
        let mut explored: Vec<&Connection> = Vec::new();
        let mut cycles: HashSet<Connection> = HashSet::new();

        to_explore.push(&start_connection);

        while !to_explore.is_empty() {
            let connection = to_explore.pop().unwrap();

            if explored.contains(&connection) {
                cycles.insert(connection.clone());
            } else {
                explored.push(connection);

                let outputs: Vec<&Connection> =
                    connections.iter().filter(|&conn| conn.input.0 == connection.output.0 && conn.input.1 == connection.output.1).collect();

                to_explore.append(&mut outputs.clone());
            }
        }

        cycles
    }
}

//TODO: Movei nside
fn get_required_bits_count(num: usize) -> usize {
    (num as f32).log2().ceil() as usize
}

#[cfg(test)]
mod network_tests {
    use std::sync::Once;
    use super::*;

    static INIT: Once = Once::new();

    fn setup() {
        INIT.call_once(|| {
            SimpleLogger::new().init().unwrap();
        });
    }

    #[test]
    fn get_required_bits_count_test() {
        setup();
        assert_eq!(get_required_bits_count(8), 3);
        assert_eq!(get_required_bits_count(12), 4);
        assert_eq!(get_required_bits_count(2), 1);
        assert_eq!(get_required_bits_count(100), 7);
    }

    #[test]
    fn from_bitstring_test() {
        setup();
        let input_count = 7; //3 bits
        let output_count = 6; // 3 bits

        let expected_connection = HashSet::from_iter(vec![
            Connection {
                input: (InputConnectionType::Input, 0),
                output: (OutputConnectionType::Output, 0),
            },
            Connection {
                input: (InputConnectionType::Input, 0),
                output: (OutputConnectionType::Gate(Gate::NAND), 0),
            },
            Connection {
                input: (InputConnectionType::Input, 1),
                output: (OutputConnectionType::Gate(Gate::NAND), 0),
            },
            Connection {
                input: (InputConnectionType::Gate(Gate::NAND), 0),
                output: (OutputConnectionType::Gate(Gate::NOR), 0),
            },
            Connection {
                input: (InputConnectionType::Input, 2),
                output: (OutputConnectionType::Gate(Gate::NOR), 0),
            },
            Connection {
                input: (InputConnectionType::Gate(Gate::NOR), 0),
                output: (OutputConnectionType::Output, 1),
            }
        ]);

        let expected = Network {
            input_count,
            output_count,
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

    #[test]
    fn clean_connections_test() {
        setup();

        let input_count = 7;
        let output_count = 5;

        let connections = HashSet::from_iter(vec![
            Connection {
                input: (InputConnectionType::Input, 0),
                output: (OutputConnectionType::Gate(Gate::NAND), 0),
            },
            Connection {
                input: (InputConnectionType::Gate(Gate::NAND), 0),
                output: (OutputConnectionType::Gate(Gate::NOR), 0),
            },
            Connection {
                input: (InputConnectionType::Gate(Gate::NOR), 0),
                output: (OutputConnectionType::Output, 0),
            },
            Connection {
                input: (InputConnectionType::Input, 1),
                output: (OutputConnectionType::Gate(Gate::NAND), 1),
            },
            Connection {
                input: (InputConnectionType::Gate(Gate::NAND), 1),
                output: (OutputConnectionType::Gate(Gate::NOR), 0),
            },
            Connection {
                input: (InputConnectionType::Gate(Gate::NAND), 1),
                output: (OutputConnectionType::Gate(Gate::NOR), 1),
            },
            Connection {
                input: (InputConnectionType::Input, 2),
                output: (OutputConnectionType::Gate(Gate::NAND), 1),
            },
            Connection {
                input: (InputConnectionType::Input, 2),
                output: (OutputConnectionType::Gate(Gate::NAND), 2),
            },
            Connection {
                input: (InputConnectionType::Gate(Gate::NAND), 2),
                output: (OutputConnectionType::Gate(Gate::NOR), 1),
            },
            Connection {
                input: (InputConnectionType::Input, 3),
                output: (OutputConnectionType::Gate(Gate::NAND), 2),
            },
            Connection {
                input: (InputConnectionType::Input, 4),
                output: (OutputConnectionType::Gate(Gate::NOR), 2),
            },
            Connection {
                input: (InputConnectionType::Gate(Gate::NOR), 2),
                output: (OutputConnectionType::Output, 1),
            },
            Connection {
                input: (InputConnectionType::Gate(Gate::NOR), 2),
                output: (OutputConnectionType::Output, 2),
            },
            Connection {
                input: (InputConnectionType::Input, 4),
                output: (OutputConnectionType::Gate(Gate::NAND), 3),
            },
            Connection {
                input: (InputConnectionType::Gate(Gate::NAND), 3),
                output: (OutputConnectionType::Gate(Gate::NOR), 2),
            },
            Connection {
                input: (InputConnectionType::Input, 5),
                output: (OutputConnectionType::Gate(Gate::NAND), 3),
            },
            Connection {
                input: (InputConnectionType::Input, 5),
                output: (OutputConnectionType::Gate(Gate::NAND), 4),
            },
            Connection {
                input: (InputConnectionType::Input, 6),
                output: (OutputConnectionType::Gate(Gate::NAND), 4),
            },
            Connection {
                input: (InputConnectionType::Gate(Gate::NOR), 3),
                output: (OutputConnectionType::Output, 3),
            },
            //Test cycle removal
            Connection {
                input: (InputConnectionType::Gate(Gate::NAND), 4),
                output: (OutputConnectionType::Gate(Gate::NOR), 4),
            },
            Connection {
                input: (InputConnectionType::Gate(Gate::NOR), 4),
                output: (OutputConnectionType::Gate(Gate::NOR), 4),
            },
            Connection {
                input: (InputConnectionType::Gate(Gate::NOR), 4),
                output: (OutputConnectionType::Output, 4),
            },
        ]);

        let cleaned_up_connections: HashSet<Connection> = HashSet::from_iter(vec![
            Connection {
                input: (InputConnectionType::Input, 4),
                output: (OutputConnectionType::Gate(Gate::NOR), 2),
            },
            Connection {
                input: (InputConnectionType::Gate(Gate::NOR), 2),
                output: (OutputConnectionType::Output, 1),
            },
            Connection {
                input: (InputConnectionType::Gate(Gate::NOR), 2),
                output: (OutputConnectionType::Output, 2),
            },
            Connection {
                input: (InputConnectionType::Input, 4),
                output: (OutputConnectionType::Gate(Gate::NAND), 3),
            },
            Connection {
                input: (InputConnectionType::Gate(Gate::NAND), 3),
                output: (OutputConnectionType::Gate(Gate::NOR), 2),
            },
            Connection {
                input: (InputConnectionType::Input, 5),
                output: (OutputConnectionType::Gate(Gate::NAND), 3),
            },
        ]);

        let mut network = Network {
            input_count,
            output_count,
            connections,
        };

        network.clean_connections();

        let network_connections_set = HashSet::from_iter(network.connections.iter().cloned());

        assert_eq!(network_connections_set, cleaned_up_connections);
    }

    #[test]
    fn get_outputs_computation_func_test() {
        setup();

        let inputs = vec![
          true, false, false, true, true
        ];

        let expected = vec![
          false, false, false, true, false
        ];

        let connections = HashSet::from_iter(vec![
            Connection {
                input: (InputConnectionType::Input, 0),
                output: (OutputConnectionType::Gate(Gate::NAND), 0),
            },
            Connection {
                input: (InputConnectionType::Input, 1),
                output: (OutputConnectionType::Gate(Gate::NOR), 0),
            },
            Connection {
                input: (InputConnectionType::Input, 2),
                output: (OutputConnectionType::Gate(Gate::NOR), 0),
            },
            Connection {
                input: (InputConnectionType::Input, 3),
                output: (OutputConnectionType::Gate(Gate::NOR), 1),
            },
            Connection {
                input: (InputConnectionType::Input, 4),
                output: (OutputConnectionType::Output, 3),
            },
            Connection {
                input: (InputConnectionType::Gate(Gate::NOR), 0),
                output: (OutputConnectionType::Gate(Gate::NAND), 0),
            },
            Connection {
                input: (InputConnectionType::Gate(Gate::NAND), 0),
                output: (OutputConnectionType::Output, 0),
            },
            Connection {
                input: (InputConnectionType::Gate(Gate::NAND), 0),
                output: (OutputConnectionType::Gate(Gate::NOR), 1),
            },
            Connection {
                input: (InputConnectionType::Gate(Gate::NOR), 1),
                output: (OutputConnectionType::Output, 1),
            },
        ]);

        let mut network = Network {
            input_count: 5,
            output_count: 5,
            connections,
        };

        //network.clean_connections();

        let output_calc_closures =  network.get_outputs_computation_func();

        let result = output_calc_closures(inputs);

        assert_eq!(result, expected);
    }

    #[test]
    fn collect_gates_with_connections_test() {
        setup();

        let connections = HashSet::from_iter(vec![
            Connection {
                input: (InputConnectionType::Input, 0),
                output: (OutputConnectionType::Output, 0),
            },
            Connection {
                input: (InputConnectionType::Input, 0),
                output: (OutputConnectionType::Output, 1),
            },
            Connection {
                input: (InputConnectionType::Input, 0),
                output: (OutputConnectionType::Output, 2),
            },
            Connection {
                input: (InputConnectionType::Input, 0),
                output: (OutputConnectionType::Gate(Gate::NAND), 0),
            },
            Connection {
                input: (InputConnectionType::Input, 0),
                output: (OutputConnectionType::Gate(Gate::NAND), 1),
            },
            Connection {
                input: (InputConnectionType::Input, 1),
                output: (OutputConnectionType::Gate(Gate::NOR), 0),
            },
            Connection {
                input: (InputConnectionType::Input, 2),
                output: (OutputConnectionType::Gate(Gate::NOR), 0),
            },
            Connection {
                input: (InputConnectionType::Gate(Gate::NOR), 2),
                output: (OutputConnectionType::Gate(Gate::NOR), 0),
            },
        ]);

        let expected = HashMap::from([
            ((Gate::NAND, 0), HashSet::from_iter(vec![
                Connection {
                    input: (InputConnectionType::Input, 0),
                    output: (OutputConnectionType::Gate(Gate::NAND), 0),
                }
            ])),
            ((Gate::NAND, 1), HashSet::from_iter(vec![
                Connection {
                    input: (InputConnectionType::Input, 0),
                    output: (OutputConnectionType::Gate(Gate::NAND), 1),
                }
            ])),
            ((Gate::NOR, 0), HashSet::from_iter(vec![
                Connection {
                    input: (InputConnectionType::Input, 1),
                    output: (OutputConnectionType::Gate(Gate::NOR), 0),
                },
                Connection {
                    input: (InputConnectionType::Input, 2),
                    output: (OutputConnectionType::Gate(Gate::NOR), 0),
                },
                Connection {
                    input: (InputConnectionType::Gate(Gate::NOR), 2),
                    output: (OutputConnectionType::Gate(Gate::NOR), 0),
                }
            ])),
            ((Gate::NOR, 2), HashSet::from_iter(vec![
                Connection {
                    input: (InputConnectionType::Gate(Gate::NOR), 2),
                    output: (OutputConnectionType::Gate(Gate::NOR), 0),
                }
            ])),
        ]);

        let result = Network::collect_gates_with_connections(&connections);

        assert_eq!(result, expected);
    }

    #[test]
    fn explore_connections_test() {
        setup();

        let connections = vec![
            Connection {
                input: (InputConnectionType::Input, 0),
                output: (OutputConnectionType::Gate(Gate::NAND), 0),
            },
            Connection {
                input: (InputConnectionType::Input, 1),
                output: (OutputConnectionType::Gate(Gate::NAND), 1),
            },
            Connection {
                input: (InputConnectionType::Gate(Gate::NAND), 0),
                output: (OutputConnectionType::Gate(Gate::NOR), 0),
            },
            Connection {
                input: (InputConnectionType::Gate(Gate::NOR), 0),
                output: (OutputConnectionType::Gate(Gate::NAND), 1),
            },
            Connection {
                input: (InputConnectionType::Gate(Gate::NAND), 1),
                output: (OutputConnectionType::Gate(Gate::NAND), 0),
            },
            Connection {
                input: (InputConnectionType::Gate(Gate::NAND), 1),
                output: (OutputConnectionType::Gate(Gate::NOR), 0),
            },
        ];

        let connections_set = HashSet::from_iter(connections);

        let start_connection_1 = connections_set.iter().find(|&e| e.input.0 == InputConnectionType::Input && e.input.1 == 0).unwrap();
        let start_connection_2 = connections_set.iter().find(|&e| e.input.0 == InputConnectionType::Input && e.input.1 == 1).unwrap();

        let expected_1 = HashSet::from_iter(vec![
            Connection {
                input: (InputConnectionType::Gate(Gate::NAND), 0),
                output: (OutputConnectionType::Gate(Gate::NOR), 0),
            },
            Connection {
                input: (InputConnectionType::Gate(Gate::NOR), 0),
                output: (OutputConnectionType::Gate(Gate::NAND), 1),
            },
        ]);

        let expected_2 = HashSet::from_iter(vec![
            Connection {
                input: (InputConnectionType::Gate(Gate::NAND), 1),
                output: (OutputConnectionType::Gate(Gate::NAND), 0),
            },
            Connection {
                input: (InputConnectionType::Gate(Gate::NAND), 1),
                output: (OutputConnectionType::Gate(Gate::NOR), 0),
            },
            Connection {
                input: (InputConnectionType::Gate(Gate::NOR), 0),
                output: (OutputConnectionType::Gate(Gate::NAND), 1),
            },
        ]);

        let result_1 = Network::get_cycles(start_connection_1, &connections_set);
        let result_2 = Network::get_cycles(start_connection_2, &connections_set);

        assert_eq!(result_1, expected_1);
        assert_eq!(result_2, expected_2);
    }

    #[test]
    fn from_bitstring_big_test() {
        setup();
        let input_count = 200;
        let output_count = 100;

        let expected_network_str = [
            "10100000",
            "01001100",
            "00101000011010111100",
            "01110110100000100010",
            "10011011011110001010",
            "01001110001010100101",
            "00100110111001011000",
            "10111101011101001100",
            "10010000110100010001",
            "01101110010101111000",
            "01100001001100011000",
            "10011100011110100001",
            "00011011010101001111",
            "01001010110000100100",
            "11001111001010111001",
            "00101111100101000010",
            "10111001101010000000",
            "10100100010111100011",
            "00000011010100100000",
            "10111110110110001010",
            "11111000110101110000",
            "10100100001010101100",
            "11010101010111011100",
            "11011101010100001111",
            "00000111010000010011",
            "01000001101000010001",
            "00010000010100100011",
            "00000010011111001101",
            "10110000011100111111",
            "00100101011011010101",
            "01000111111011100001",
            "01001111001000100110",
            "10001110100101011001",
            "00110010000000011100",
            "11010010000111000011",
            "00010001101001001101",
            "01010101110100110001",
            "10111111001101001110",
            "11011000111101010100",
            "00101001101001001100",
            "11001001110111100011",
            "01000101010111111100"
        ].join("");

        let mut result = Network::from_bitstring(&expected_network_str, input_count, output_count, 8, 8).unwrap();

        result.clean_connections();

        let output_closure = result.get_outputs_computation_func();

        assert_eq!(1, 0);
    }
}