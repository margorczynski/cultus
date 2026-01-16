use std::collections::{HashMap, HashSet};
use std::time::Instant;

use log::{debug, trace};

use super::connection::*;
use crate::common::get_required_bits_count;

#[derive(PartialEq, Debug, Clone)]
pub struct Network {
    pub input_count: usize,
    pub output_count: usize,
    pub connections: HashSet<Connection>,
}

impl Network {
    /// Create a logic gate network from a string containing binary (0s and 1s)
    /// The amount of inputs and outputs of the network is constant and specified as arguments.
    /// The amount of bits (and thus maximum number) encoding the amount of NAND gates is constant and specified as arguments.
    /// Connections are deduplicated after decoding.
    /// Incorrect connections will be discarded and connections to a saturated input will be discarded in order of decoding.
    ///
    /// The string is in the form:
    /// NAND amount bits | Connection 1 input type | Connection 1 output type | Connection 1 input index | Connection 1 output index | Connection 2 ...
    pub fn from_bitstring(
        s: &str,
        input_count: usize,
        output_count: usize,
        nand_count_bits: usize,
    ) -> Option<Network> {
        debug!("Decoding network from bitstring: {}", s);

        //Number of bits in whole string not bigger than the sum of sizes of the attributes
        if s.chars().count() <= nand_count_bits {
            return None;
        }

        //s = NAND_COUNT | CONNECTIONS...
        let nand_count_binary = &s[0..nand_count_bits];

        //Parse from binary string to u32 value. Panic if it fails - the assumption is that the string should always contain 0/1 only
        let nand_count = usize::from_str_radix(nand_count_binary, 2).unwrap();

        let connection_index_bits_count = *[input_count, output_count, nand_count]
            .map(|cnt| get_required_bits_count(cnt))
            .iter()
            .max()
            .unwrap();
        let connection_bits_count = 2 + (2 * connection_index_bits_count);

        debug!(
            "Connection index bits amount: {}",
            connection_index_bits_count
        );
        debug!("Connection bit size: {}", connection_bits_count);

        let connections_binary = &s[nand_count_bits..];

        let mut ind = 0;
        let mut none_connection_count = 0;
        let mut over_saturated_connection_count = 0;
        let mut connections: HashSet<Connection> = HashSet::new();
        let mut input_counts: HashMap<(OutputConnectionType, usize), usize> = HashMap::new();
        let mut output_counts: HashMap<(InputConnectionType, usize), usize> = HashMap::new();
        while ind < connections_binary.len() {
            let connection_with_len = Connection::from_bitstring(
                &connections_binary[ind..],
                input_count,
                output_count,
                nand_count,
                connection_index_bits_count,
            );
            debug!(
                "[IDX: {}] Processing from binary connection: {:?}",
                ind, connection_with_len
            );
            match connection_with_len {
                Some(connection) => {
                    let input = connection.input.clone();
                    let output = connection.output.clone();
                    let max_inputs_cnt = match output.0 {
                        OutputConnectionType::Output => 1,
                        OutputConnectionType::NAND => 2,
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
                                over_saturated_connection_count =
                                    over_saturated_connection_count + 1;
                            }
                        }
                    }
                    match output_counts.get(&input) {
                        None => {
                            output_counts.insert(input, 1);
                        }
                        Some(&count) => {
                            let new_output_count = count + 1;
                            output_counts.insert(input, new_output_count);
                        }
                    }
                }
                None => {
                    none_connection_count = none_connection_count + 1;
                }
            }
            ind = ind + connection_bits_count;
        }

        //TODO: Remove gates where input_cnt != 2 and output_cnt != 1
        let outputs_with_wrong_input_amount: HashSet<(OutputConnectionType, usize)> = input_counts.iter().filter(|(&(conn_type, _), &count)| {
            match conn_type {
                OutputConnectionType::Output => count != 1,
                OutputConnectionType::NAND => count != 2,
            }
        }).map(|(input, _)| input).cloned().collect();

        let inputs_with_not_enough_outputs: HashSet<(InputConnectionType, usize)> = output_counts.iter().filter(|(&(_, _), &count)| {
            count < 1
        }).map(|(output, _)| output).cloned().collect();

        let connections_to_remove: HashSet<Connection> =
            connections
                .iter()
                .filter(|&conn| outputs_with_wrong_input_amount.contains(&conn.output) || inputs_with_not_enough_outputs.contains(&conn.input))
                .cloned()
                .collect();

        connections = connections.difference(&connections_to_remove).cloned().collect();

        debug!(
            "Connections which couldn't be built from binary count: {}",
            none_connection_count
        );
        debug!(
            "Connections discarded because of input saturation limit: {}",
            over_saturated_connection_count
        );
        debug!(
            "Outputs with wrong input amount: {}",
            outputs_with_wrong_input_amount.len()
        );
        debug!(
            "Inputs with not enough outputs: {}",
            inputs_with_not_enough_outputs.len()
        );

        Some(Network {
            input_count,
            output_count,
            connections,
        })
    }

    /// Create a logic gate network directly from a bit slice (more efficient than string parsing)
    pub fn from_bits(
        bits: &[bool],
        input_count: usize,
        output_count: usize,
        nand_count_bits: usize,
    ) -> Option<Network> {
        if bits.len() <= nand_count_bits {
            return None;
        }

        // Parse NAND count from first nand_count_bits bits
        let nand_count: usize = bits[..nand_count_bits]
            .iter()
            .enumerate()
            .filter(|(_, &b)| b)
            .map(|(i, _)| 1 << i)
            .sum();

        let connection_index_bits_count = *[input_count, output_count, nand_count]
            .map(|cnt| get_required_bits_count(cnt))
            .iter()
            .max()
            .unwrap();
        let connection_bits_count = 2 + (2 * connection_index_bits_count);

        let connections_bits = &bits[nand_count_bits..];

        let mut connections: HashSet<Connection> = HashSet::new();
        let mut input_counts: HashMap<(OutputConnectionType, usize), usize> = HashMap::new();
        let mut ind = 0;

        while ind + connection_bits_count <= connections_bits.len() {
            if let Some(connection) = Connection::from_bits(
                &connections_bits[ind..],
                input_count,
                output_count,
                nand_count,
                connection_index_bits_count,
            ) {
                let output = connection.output;
                let max_inputs_cnt = match output.0 {
                    OutputConnectionType::Output => 1,
                    OutputConnectionType::NAND => 2,
                };

                let current_count = input_counts.get(&output).copied().unwrap_or(0);
                if current_count + 1 <= max_inputs_cnt {
                    connections.insert(connection);
                    input_counts.insert(output, current_count + 1);
                }
            }
            ind += connection_bits_count;
        }

        // Remove gates with wrong input amounts
        let outputs_with_wrong_input_amount: HashSet<(OutputConnectionType, usize)> = input_counts
            .iter()
            .filter(|(&(conn_type, _), &count)| match conn_type {
                OutputConnectionType::Output => count != 1,
                OutputConnectionType::NAND => count != 2,
            })
            .map(|(input, _)| *input)
            .collect();

        connections.retain(|conn| !outputs_with_wrong_input_amount.contains(&conn.output));

        Some(Network {
            input_count,
            output_count,
            connections,
        })
    }

    /// Clean up the network connections
    /// Removes connections which are not computable (gates without 2 inputs and 1 output).
    /// Note: Cycles are prevented by construction (input_idx < output_idx constraint).
    pub fn clean_connections(&mut self) {
        debug!(
            "Cleaning up connections. Starting count: {}",
            self.connections.len()
        );

        let mut cleaned_up_connections: HashSet<Connection> = self.connections.clone();

        let remove_incorrect_start = Instant::now();
        //Remove connections of gates where the number of inputs or outputs is incorrect
        //This is done in an iterative manner until there are no more connections to remove
        let mut connections_to_remove: HashSet<Connection> = HashSet::new();
        loop {
            let gates_with_connections =
                Network::collect_gates_with_connections(&cleaned_up_connections);

            for (index, connections) in gates_with_connections {
                let input_connections: Vec<_> = connections
                    .iter()
                    .filter(|&conn| {
                        conn.output.0 == OutputConnectionType::NAND && conn.output.1 == index
                    })
                    .collect();
                let output_connections: Vec<_> = connections
                    .iter()
                    .filter(|&conn| {
                        conn.input.0 == InputConnectionType::NAND && conn.input.1 == index
                    })
                    .collect();

                if input_connections.len() != 2 || output_connections.len() < 1 {
                    connections_to_remove.extend(connections)
                }
            }

            if connections_to_remove.is_empty() {
                break;
            }

            cleaned_up_connections = cleaned_up_connections
                .difference(&connections_to_remove)
                .cloned()
                .collect();

            connections_to_remove.clear();
        }
        trace!("remove_incorrect_elapsed={:?}", remove_incorrect_start.elapsed());

        debug!(
            "Connections after cleanup count: {}",
            cleaned_up_connections.len()
        );

        self.connections = cleaned_up_connections;
    }

    /// Get the closure that computes the output vector given an input vector
    pub fn get_outputs_computation_fn(
        output_count: usize,
        connections: &HashSet<Connection>,
    ) -> impl Fn(&Vec<bool>) -> Vec<bool> {
        let mut output_index_to_input_indexes_and_gates_count_map: HashMap<
            usize,
            (usize, Vec<usize>),
        > = HashMap::new();
        for output_connection in connections
            .iter()
            .filter(|&conn| conn.output.0 == OutputConnectionType::Output)
        {
            let mut to_explore: Vec<&Connection> = Vec::new();
            let mut gate_count: usize = 0;
            let mut input_index_stack: Vec<usize> = Vec::new();

            to_explore.push(output_connection);

            while !to_explore.is_empty() {
                let connection = to_explore.pop().unwrap();

                match connection.input.0 {
                    InputConnectionType::Input => input_index_stack.push(connection.input.1),
                    InputConnectionType::NAND => gate_count += 1,
                }

                let inputs: Vec<&Connection> = connections
                    .iter()
                    .filter(|&conn| {
                        conn.output.0 == connection.input.0 && conn.output.1 == connection.input.1
                    })
                    .collect();

                to_explore.append(&mut inputs.clone());
            }

            debug!("get_outputs_computation_fn: Finished for connection {}, gate count: {:?}, index stack: {:?}", output_connection, gate_count, input_index_stack);

            output_index_to_input_indexes_and_gates_count_map
                .insert(output_connection.output.1, (gate_count, input_index_stack));
        }

        move |input_bits: &Vec<bool>| -> Vec<bool> {
            debug!("Calculate output for input: {:?}", input_bits);

            (0..output_count)
                .map(|output_idx| {
                    match output_index_to_input_indexes_and_gates_count_map.get(&output_idx) {
                        None => false,
                        Some((gate_count, input_indexes)) => {
                            debug!(
                                "Stacks for output {}: gate count: {}, indexes: {:?}",
                                output_idx, gate_count, input_indexes
                            );
                            let mut value_bits_stack: Vec<bool> = input_indexes
                                .iter()
                                .map(|&idx| input_bits.get(idx).unwrap().clone())
                                .collect();

                            for _ in 0..*gate_count {
                                //After cleanup it should be impossible for this to throw
                                let first_value = value_bits_stack.pop().unwrap();
                                let second_value = value_bits_stack.pop().unwrap();

                                value_bits_stack.push(!(first_value && second_value))
                            }

                            //After cleanup it should be impossible for this to throw
                            value_bits_stack.first().unwrap().clone()
                        }
                    }
                })
                .collect()
        }
    }

    fn collect_gates_with_connections(
        connections: &HashSet<Connection>,
    ) -> HashMap<usize, HashSet<Connection>> {
        let mut gates_with_connections: HashMap<usize, HashSet<Connection>> = HashMap::new();

        for connection in connections {
            let input = connection.input;
            let output = connection.output;

            let mut add_gate_connection =
                |index: usize| match gates_with_connections.get_mut(&index) {
                    None => {
                        let connections_set = HashSet::from_iter(vec![connection.clone()]);
                        gates_with_connections.insert(index, connections_set);
                    }
                    Some(curr_connections_vec) => {
                        curr_connections_vec.insert(connection.clone());
                    }
                };

            match input.0 {
                InputConnectionType::Input => {}
                InputConnectionType::NAND => add_gate_connection(input.1),
            }
            match output.0 {
                OutputConnectionType::Output => {}
                OutputConnectionType::NAND => add_gate_connection(output.1),
            }
        }

        gates_with_connections
    }

    #[allow(dead_code)]
    fn get_cycles(
        start_connection: &Connection,
        connections: &HashSet<Connection>,
    ) -> HashSet<Connection> {
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

                let outputs: Vec<&Connection> = connections
                    .iter()
                    .filter(|&conn| {
                        conn.input.0 == connection.output.0 && conn.input.1 == connection.output.1
                    })
                    .collect();

                to_explore.append(&mut outputs.clone());
            }
        }

        cycles
    }
}

#[cfg(test)]
mod network_tests {
    use crate::common::*;

    use super::*;

    #[test]
    fn get_required_bits_count_test() {
        setup();
        assert_eq!(get_required_bits_count(8), 3);
        assert_eq!(get_required_bits_count(12), 4);
        assert_eq!(get_required_bits_count(2), 1);
        assert_eq!(get_required_bits_count(100), 7);
        assert_eq!(get_required_bits_count(66), 7);
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
                output: (OutputConnectionType::NAND, 0),
            },
            Connection {
                input: (InputConnectionType::Input, 1),
                output: (OutputConnectionType::NAND, 0),
            },
            Connection {
                input: (InputConnectionType::NAND, 0),
                output: (OutputConnectionType::NAND, 1),
            },
            Connection {
                input: (InputConnectionType::Input, 2),
                output: (OutputConnectionType::NAND, 1),
            },
            Connection {
                input: (InputConnectionType::NAND, 0),
                output: (OutputConnectionType::Output, 1),
            },
        ]);

        let expected = Network {
            input_count,
            output_count,
            connections: expected_connection,
        };

        let expected_network_str = [
            "1100",       //12 NANDs
            "0000000000", //I0 -> O0
            "0100000000", //I0 -> NAND0
            "0100010000", //I1 -> NAND0
            "1100000001", //NAND0 -> NAND1
            "0100100001", //I2 -> NAND1
            "1000000001", //NAND0 -> O1
        ]
        .join("");

        let result =
            Network::from_bitstring(&expected_network_str, input_count, output_count, 4).unwrap();

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
                output: (OutputConnectionType::NAND, 0),
            },
            Connection {
                input: (InputConnectionType::NAND, 0),
                output: (OutputConnectionType::NAND, 5),
            },
            Connection {
                input: (InputConnectionType::NAND, 5),
                output: (OutputConnectionType::Output, 0),
            },
            Connection {
                input: (InputConnectionType::Input, 1),
                output: (OutputConnectionType::NAND, 1),
            },
            Connection {
                input: (InputConnectionType::NAND, 1),
                output: (OutputConnectionType::NAND, 5),
            },
            Connection {
                input: (InputConnectionType::NAND, 1),
                output: (OutputConnectionType::NAND, 6),
            },
            Connection {
                input: (InputConnectionType::Input, 2),
                output: (OutputConnectionType::NAND, 1),
            },
            Connection {
                input: (InputConnectionType::Input, 2),
                output: (OutputConnectionType::NAND, 2),
            },
            Connection {
                input: (InputConnectionType::NAND, 2),
                output: (OutputConnectionType::NAND, 6),
            },
            Connection {
                input: (InputConnectionType::Input, 3),
                output: (OutputConnectionType::NAND, 2),
            },
            Connection {
                input: (InputConnectionType::Input, 4),
                output: (OutputConnectionType::NAND, 7),
            },
            Connection {
                input: (InputConnectionType::NAND, 7),
                output: (OutputConnectionType::Output, 1),
            },
            Connection {
                input: (InputConnectionType::NAND, 7),
                output: (OutputConnectionType::Output, 2),
            },
            Connection {
                input: (InputConnectionType::Input, 4),
                output: (OutputConnectionType::NAND, 3),
            },
            Connection {
                input: (InputConnectionType::NAND, 3),
                output: (OutputConnectionType::NAND, 7),
            },
            Connection {
                input: (InputConnectionType::Input, 5),
                output: (OutputConnectionType::NAND, 3),
            },
            Connection {
                input: (InputConnectionType::Input, 5),
                output: (OutputConnectionType::NAND, 4),
            },
            Connection {
                input: (InputConnectionType::Input, 6),
                output: (OutputConnectionType::NAND, 4),
            },
            Connection {
                input: (InputConnectionType::NAND, 8),
                output: (OutputConnectionType::Output, 3),
            },
            //Test cycle removal
            Connection {
                input: (InputConnectionType::NAND, 4),
                output: (OutputConnectionType::NAND, 9),
            },
            Connection {
                input: (InputConnectionType::NAND, 9),
                output: (OutputConnectionType::Output, 4),
            },
        ]);

        let cleaned_up_connections: HashSet<Connection> = HashSet::from_iter(vec![
            Connection {
                input: (InputConnectionType::Input, 4),
                output: (OutputConnectionType::NAND, 7),
            },
            Connection {
                input: (InputConnectionType::NAND, 7),
                output: (OutputConnectionType::Output, 1),
            },
            Connection {
                input: (InputConnectionType::NAND, 7),
                output: (OutputConnectionType::Output, 2),
            },
            Connection {
                input: (InputConnectionType::Input, 4),
                output: (OutputConnectionType::NAND, 3),
            },
            Connection {
                input: (InputConnectionType::NAND, 3),
                output: (OutputConnectionType::NAND, 7),
            },
            Connection {
                input: (InputConnectionType::Input, 5),
                output: (OutputConnectionType::NAND, 3),
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

        let inputs = vec![true, false, false, true, true];

        let expected = vec![false, false, false, true, false];

        let connections = HashSet::from_iter(vec![
            Connection {
                input: (InputConnectionType::Input, 0),
                output: (OutputConnectionType::NAND, 0),
            },
            Connection {
                input: (InputConnectionType::Input, 1),
                output: (OutputConnectionType::NAND, 0),
            },
            Connection {
                input: (InputConnectionType::Input, 2),
                output: (OutputConnectionType::NAND, 0),
            },
            Connection {
                input: (InputConnectionType::Input, 3),
                output: (OutputConnectionType::NAND, 1),
            },
            Connection {
                input: (InputConnectionType::Input, 4),
                output: (OutputConnectionType::Output, 3),
            },
            Connection {
                input: (InputConnectionType::NAND, 0),
                output: (OutputConnectionType::NAND, 0),
            },
            Connection {
                input: (InputConnectionType::NAND, 0),
                output: (OutputConnectionType::Output, 0),
            },
            Connection {
                input: (InputConnectionType::NAND, 0),
                output: (OutputConnectionType::NAND, 1),
            },
            Connection {
                input: (InputConnectionType::NAND, 1),
                output: (OutputConnectionType::Output, 1),
            },
        ]);

        let mut network = Network {
            input_count: 5,
            output_count: 5,
            connections,
        };

        network.clean_connections();

        let output_calc_closures = Network::get_outputs_computation_fn(5, &network.connections);

        let result = output_calc_closures(&inputs);

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
                output: (OutputConnectionType::NAND, 0),
            },
            Connection {
                input: (InputConnectionType::Input, 0),
                output: (OutputConnectionType::NAND, 1),
            },
            Connection {
                input: (InputConnectionType::Input, 1),
                output: (OutputConnectionType::NAND, 0),
            },
            Connection {
                input: (InputConnectionType::Input, 2),
                output: (OutputConnectionType::NAND, 0),
            },
            Connection {
                input: (InputConnectionType::NAND, 2),
                output: (OutputConnectionType::NAND, 0),
            },
        ]);

        let expected = HashMap::from([
            (
                0,
                HashSet::from_iter(vec![
                    Connection {
                        input: (InputConnectionType::Input, 0),
                        output: (OutputConnectionType::NAND, 0),
                    },
                    Connection {
                        input: (InputConnectionType::Input, 1),
                        output: (OutputConnectionType::NAND, 0),
                    },
                    Connection {
                        input: (InputConnectionType::Input, 2),
                        output: (OutputConnectionType::NAND, 0),
                    },
                    Connection {
                        input: (InputConnectionType::NAND, 2),
                        output: (OutputConnectionType::NAND, 0),
                    },
                ]),
            ),
            (
                1,
                HashSet::from_iter(vec![Connection {
                    input: (InputConnectionType::Input, 0),
                    output: (OutputConnectionType::NAND, 1),
                }]),
            ),
            (
                2,
                HashSet::from_iter(vec![Connection {
                    input: (InputConnectionType::NAND, 2),
                    output: (OutputConnectionType::NAND, 0),
                }]),
            ),
        ]);

        let result = Network::collect_gates_with_connections(&connections);

        assert_eq!(result, expected);
    }

    #[test]
    fn get_cycles_test() {
        setup();

        let connections = vec![
            Connection {
                input: (InputConnectionType::Input, 0),
                output: (OutputConnectionType::NAND, 0),
            },
            Connection {
                input: (InputConnectionType::Input, 1),
                output: (OutputConnectionType::NAND, 1),
            },
            Connection {
                input: (InputConnectionType::NAND, 0),
                output: (OutputConnectionType::NAND, 2),
            },
            Connection {
                input: (InputConnectionType::NAND, 1),
                output: (OutputConnectionType::NAND, 2),
            },
            Connection {
                input: (InputConnectionType::NAND, 2),
                output: (OutputConnectionType::NAND, 1),
            },
            Connection {
                input: (InputConnectionType::NAND, 1),
                output: (OutputConnectionType::NAND, 0),
            },
        ];

        let connections_set = HashSet::from_iter(connections);

        let start_connection_1 = connections_set
            .iter()
            .find(|&e| e.input.0 == InputConnectionType::Input && e.input.1 == 0)
            .unwrap();
        let start_connection_2 = connections_set
            .iter()
            .find(|&e| e.input.0 == InputConnectionType::Input && e.input.1 == 1)
            .unwrap();

        let expected_1 = HashSet::from_iter(vec![
            Connection {
                input: (InputConnectionType::NAND, 0),
                output: (OutputConnectionType::NAND, 2),
            },
            Connection {
                input: (InputConnectionType::NAND, 2),
                output: (OutputConnectionType::NAND, 1),
            },
        ]);

        let expected_2 = HashSet::from_iter(vec![
            Connection {
                input: (InputConnectionType::NAND, 1),
                output: (OutputConnectionType::NAND, 2),
            },
            Connection {
                input: (InputConnectionType::NAND, 2),
                output: (OutputConnectionType::NAND, 1),
            },
            Connection {
                input: (InputConnectionType::NAND, 1),
                output: (OutputConnectionType::NAND, 0),
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
            "101000011010111100",
            "110110100000100010",
            "011011011110001010",
            "001110001010100101",
            "100110111001011000",
            "111101011101001100",
            "010000110100010001",
            "101110010101111000",
            "100001001100011000",
            "100001001100011000",
            "101000011010111100",
            "110110100000100010",
            "011011011110001010",
            "001110001010100101",
            "100110111001011000",
            "111101011101001100",
            "010000110100010001",
            "101110010101111000",
            "100001001100011000",
            "100001001100011000",
            "101000011010111100",
            "110110100000100010",
            "011011011110001010",
            "001110001010100101",
            "100110111001011000",
            "111101011101001100",
            "010000110100010001",
            "101110010101111000",
            "100001001100011000",
            "100001001100011000",
        ]
        .join("");

        let mut result =
            Network::from_bitstring(&expected_network_str, input_count, output_count, 8).unwrap();

        result.clean_connections();

        let _output_closure =
            Network::get_outputs_computation_fn(output_count, &result.connections);

        assert_eq!(1, 1);
    }
}
