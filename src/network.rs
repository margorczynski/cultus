use std::collections::{HashMap, HashSet};

use log::debug;
use simple_logger::SimpleLogger;

pub use crate::connection::*;
use crate::connection::InputConnectionType::Input;

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

    /// Clean up the network connections
    /// 1. Remove duplicates
    /// 2. Remove connections where the output is already saturated (e.g. NOR has already 2 input connections)
    /// 3. Remove connections which are not computable (no computable path to output)
    fn clean_connections(&mut self) {
        debug!("Cleaning up connections. Starting count: {}", self.connections.len());

        self.connections.dedup();

        debug!("Connection count after dedup: {}", self.connections.len());

        fn get_required_saturation_cnt(oct: OutputConnectionType) -> usize {
            match &oct {
                OutputConnectionType::Output => 1,
                OutputConnectionType::Gate(_) => 2
            }
        }

        let mut input_counts: HashMap<(OutputConnectionType, usize), usize> = HashMap::new();

        let mut cleaned_up_connections_final: Vec<Connection> = Vec::new();

        //Remove connections where output is already saturated - has to be split to preserve order
        for connection in &self.connections {
            let output = connection.output.clone();
            let max_inputs_cnt = get_required_saturation_cnt(output.0);
            match input_counts.get(&output) {
                None => {
                    cleaned_up_connections_final.push(connection.clone());
                    input_counts.insert(output, 1);
                }
                Some(&count) => {
                    let new_input_count = count + 1;
                    if new_input_count <= max_inputs_cnt {
                        cleaned_up_connections_final.push(connection.clone());
                        input_counts.insert(output, new_input_count);
                    }
                }
            }
        }

        debug!("Connection count after over saturation removal: {}", cleaned_up_connections_final.len());

        let mut connections_to_remove: HashSet<Connection> = HashSet::new();
        loop {
            let gates_with_connections = Network::collect_gates_with_connections(&cleaned_up_connections_final);

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

            for index in (0..cleaned_up_connections_final.len()).rev() {
                if connections_to_remove.contains(&cleaned_up_connections_final[index]) {
                    cleaned_up_connections_final.remove(index);
                }
            }
            connections_to_remove.clear();
        }

        //TODO: Remove cycles? Either trace paths or give every gate a layer - distance from input or output. Gates cannot output into gates from a lower layer

        debug!("Connections after cleanup count: {}", &self.connections.len());

        let gates_with_connections = Network::collect_gates_with_connections(&cleaned_up_connections_final);

        self.nand_count  = gates_with_connections.iter().filter(|&((gate, _), _)| *gate == Gate::NAND).count();
        self.nor_count   = gates_with_connections.iter().filter(|&((gate, _), _)| *gate == Gate::NOR).count();
        self.connections = cleaned_up_connections_final;
    }

    fn collect_gates_with_connections(connections: &Vec<Connection>) -> HashMap<(Gate, usize), Vec<Connection>> {
        let mut gates_with_connections: HashMap<(Gate, usize), Vec<Connection>> = HashMap::new();

        for connection in connections {
            let input = connection.input;
            let output = connection.output;

            let mut add_gate_connection = |gate_type: Gate, index: usize| {
                let gate_id = (gate_type, index);
                match gates_with_connections.get_mut(&gate_id) {
                    None => {
                        let connections_vec = vec![connection.clone()];
                        gates_with_connections.insert(gate_id, connections_vec);
                    }
                    Some(curr_connections_vec) => {
                        curr_connections_vec.push(connection.clone());
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
        let nand_count = 12; //4 bits
        let nor_count = 4; //3 bits

        let expected_connection = vec![
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

    #[test]
    fn collect_gates_with_connections_test() {
        setup();

        let connections = vec![
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
        ];

        let expected = HashMap::from([
            ((Gate::NAND, 0), vec![
                Connection {
                    input: (InputConnectionType::Input, 0),
                    output: (OutputConnectionType::Gate(Gate::NAND), 0),
                }
            ]),
            ((Gate::NAND, 1), vec![
                Connection {
                    input: (InputConnectionType::Input, 0),
                    output: (OutputConnectionType::Gate(Gate::NAND), 1),
                }
            ]),
            ((Gate::NOR, 0), vec![
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
            ]),
            ((Gate::NOR, 2), vec![
                Connection {
                    input: (InputConnectionType::Gate(Gate::NOR), 2),
                    output: (OutputConnectionType::Gate(Gate::NOR), 0),
                }
            ]),
        ]);

        let result = Network::collect_gates_with_connections(&connections);

        assert_eq!(result, expected);
    }

    #[test]
    fn clean_connections_test() {
        setup();

        let input_count = 7;
        let output_count = 4;
        let nand_count = 5;
        let nor_count = 4;

        let connections = vec![
            Connection {
                input: (InputConnectionType::Input, 0),
                output: (OutputConnectionType::Gate(Gate::NAND), 0),
            },
            //Duplicate - removed
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
        ];

        let cleaned_up_connections = vec![
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
        ];

        let mut network = Network {
            input_count,
            output_count,
            nand_count,
            nor_count,
            connections,
        };

        network.clean_connections();

        assert_eq!(network.connections, cleaned_up_connections);
        assert_eq!(network.nand_count, 1);
        assert_eq!(network.nor_count, 1);
    }
}