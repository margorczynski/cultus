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

//TODO: Verify gate has exactly two inputs, output at most 1 input

impl Network {
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
                    ind = ind + connection_bits_count
                }
                None => {
                    none_connection_count = none_connection_count + 1;
                }
            }
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
    /// 1. Remove connections which are not computable (no computable path to output, gate hasn't got 2 inputs and 1 output)
    /// 2. Remove connections which create a cycle
    pub fn clean_connections(&mut self) {
        debug!("Cleaning up connections. Starting count: {}", self.connections.len());

        let mut cleaned_up_connections: HashSet<Connection> = self.connections.clone();

        let cycle_connections = cleaned_up_connections
            .iter()
            .filter(|&conn| conn.input.0 == InputConnectionType::Input)
            .map(|input_conn| Network::get_cycles(input_conn, &cleaned_up_connections))
            .reduce(|acc, e| acc.union(&e).copied().collect())
            .unwrap();

        cleaned_up_connections = cleaned_up_connections.difference(&cycle_connections).cloned().collect();

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
}