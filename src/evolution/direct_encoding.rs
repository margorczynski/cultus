//! Direct network encoding for genetic algorithms.
//!
//! This module provides a direct structural representation of logic networks
//! where every encoded gate is valid by construction. This eliminates the
//! wasted bits and invalid connections of the previous bit-string encoding.

use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Source of input for a gate or network output.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InputSource {
    /// Network input at given index
    NetworkInput(u16),
    /// Output of gate at given index (must be < current gate index for DAG)
    GateOutput(u16),
    /// Memory read value at bit index (for SmartNetwork)
    MemoryBit(u8),
    /// Constant value
    Constant(bool),
}

/// Type of logic gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GateType {
    // Basic gates
    Nand,
    And,
    Or,
    Not, // Uses only first input
    Xor,

    // Multiplexer: if input[0] then input[2] else input[1]
    Mux2,

    // Comparison
    Equal,   // input[0] == input[1]

    // Pass-through (identity)
    Buffer, // Uses only first input
}

impl GateType {
    /// Returns the number of inputs required for this gate type.
    pub fn required_inputs(&self) -> usize {
        match self {
            GateType::Not | GateType::Buffer => 1,
            GateType::Nand | GateType::And | GateType::Or | GateType::Xor | GateType::Equal => 2,
            GateType::Mux2 => 3,
        }
    }

    /// Returns all gate types that use the same or fewer inputs.
    pub fn compatible_types(input_count: usize) -> Vec<GateType> {
        let all = vec![
            GateType::Nand,
            GateType::And,
            GateType::Or,
            GateType::Not,
            GateType::Xor,
            GateType::Mux2,
            GateType::Equal,
            GateType::Buffer,
        ];
        all.into_iter()
            .filter(|gt| gt.required_inputs() <= input_count)
            .collect()
    }

    /// Compute the gate output given input values.
    pub fn compute(&self, inputs: &[bool]) -> bool {
        match self {
            GateType::Nand => !(inputs.get(0).copied().unwrap_or(false)
                               && inputs.get(1).copied().unwrap_or(false)),
            GateType::And => inputs.get(0).copied().unwrap_or(false)
                            && inputs.get(1).copied().unwrap_or(false),
            GateType::Or => inputs.get(0).copied().unwrap_or(false)
                           || inputs.get(1).copied().unwrap_or(false),
            GateType::Not => !inputs.get(0).copied().unwrap_or(false),
            GateType::Xor => inputs.get(0).copied().unwrap_or(false)
                            ^ inputs.get(1).copied().unwrap_or(false),
            GateType::Mux2 => {
                let select = inputs.get(0).copied().unwrap_or(false);
                if select {
                    inputs.get(2).copied().unwrap_or(false)
                } else {
                    inputs.get(1).copied().unwrap_or(false)
                }
            }
            GateType::Equal => inputs.get(0).copied().unwrap_or(false)
                              == inputs.get(1).copied().unwrap_or(false),
            GateType::Buffer => inputs.get(0).copied().unwrap_or(false),
        }
    }
}

/// A single gate in the network.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Gate {
    /// Gate type determining the logic operation
    pub gate_type: GateType,
    /// Input sources for this gate
    pub inputs: Vec<InputSource>,
}

impl Gate {
    /// Create a new gate with the given type and inputs.
    pub fn new(gate_type: GateType, inputs: Vec<InputSource>) -> Self {
        Gate { gate_type, inputs }
    }

    /// Compute the gate output given network inputs and previous gate outputs.
    pub fn compute(&self, network_inputs: &[bool], gate_outputs: &[bool], memory_bits: &[bool]) -> bool {
        let input_values: Vec<bool> = self
            .inputs
            .iter()
            .map(|src| resolve_input(src, network_inputs, gate_outputs, memory_bits))
            .collect();
        self.gate_type.compute(&input_values)
    }
}

/// Resolve an input source to its boolean value.
fn resolve_input(
    source: &InputSource,
    network_inputs: &[bool],
    gate_outputs: &[bool],
    memory_bits: &[bool],
) -> bool {
    match source {
        InputSource::NetworkInput(idx) => network_inputs.get(*idx as usize).copied().unwrap_or(false),
        InputSource::GateOutput(idx) => gate_outputs.get(*idx as usize).copied().unwrap_or(false),
        InputSource::MemoryBit(idx) => memory_bits.get(*idx as usize).copied().unwrap_or(false),
        InputSource::Constant(val) => *val,
    }
}

/// Memory configuration for SmartNetwork.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Number of address bits (2^addr_bits memory locations)
    pub addr_bits: u8,
    /// Number of data bits per location
    pub data_bits: u8,
}

/// Direct network representation - every gate is valid by construction.
///
/// Gates are stored in topological order: gate[i] can only reference
/// gate[j] where j < i. This guarantees the network is a DAG.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectNetwork {
    /// Number of network inputs
    pub input_count: u16,
    /// Number of network outputs (not including memory control outputs)
    pub output_count: u16,
    /// Gates in topological order
    pub gates: Vec<Gate>,
    /// Output sources - maps each output to its source
    pub outputs: Vec<InputSource>,
    /// Memory configuration (None = no memory)
    pub memory_config: Option<MemoryConfig>,
}

impl DirectNetwork {
    /// Create an empty network with no gates.
    pub fn new(input_count: u16, output_count: u16) -> Self {
        DirectNetwork {
            input_count,
            output_count,
            gates: Vec::new(),
            outputs: (0..output_count)
                .map(|i| InputSource::NetworkInput(i.min(input_count - 1)))
                .collect(),
            memory_config: None,
        }
    }

    /// Create a random valid network.
    pub fn random(
        input_count: u16,
        output_count: u16,
        gate_count: usize,
        memory_config: Option<MemoryConfig>,
        rng: &mut impl Rng,
    ) -> Self {
        let memory_bits = memory_config.as_ref().map(|m| m.data_bits).unwrap_or(0);
        let mut gates = Vec::with_capacity(gate_count);

        for i in 0..gate_count {
            // Choose random gate type
            let gate_type = *[
                GateType::Nand,
                GateType::And,
                GateType::Or,
                GateType::Xor,
                GateType::Not,
            ]
            .choose(rng)
            .unwrap();

            // Generate valid inputs for this gate
            let num_inputs = gate_type.required_inputs();
            let inputs: Vec<InputSource> = (0..num_inputs)
                .map(|_| Self::random_input_source(input_count, i as u16, memory_bits, rng))
                .collect();

            gates.push(Gate::new(gate_type, inputs));
        }

        // Generate output sources
        let outputs: Vec<InputSource> = (0..output_count)
            .map(|_| Self::random_input_source(input_count, gate_count as u16, memory_bits, rng))
            .collect();

        DirectNetwork {
            input_count,
            output_count,
            gates,
            outputs,
            memory_config,
        }
    }

    /// Generate a random valid input source.
    pub fn random_input_source(
        input_count: u16,
        max_gate_idx: u16,
        memory_bits: u8,
        rng: &mut impl Rng,
    ) -> InputSource {
        let total_sources = input_count as usize + max_gate_idx as usize + memory_bits as usize;

        if total_sources == 0 {
            return InputSource::Constant(rng.gen());
        }

        // Small chance of constant
        if rng.gen::<f32>() < 0.05 {
            return InputSource::Constant(rng.gen());
        }

        let idx = rng.gen_range(0..total_sources);

        if idx < input_count as usize {
            InputSource::NetworkInput(idx as u16)
        } else if idx < (input_count as usize + max_gate_idx as usize) {
            InputSource::GateOutput((idx - input_count as usize) as u16)
        } else {
            InputSource::MemoryBit((idx - input_count as usize - max_gate_idx as usize) as u8)
        }
    }

    /// Compute network output for given inputs.
    pub fn compute(&self, inputs: &[bool]) -> Vec<bool> {
        self.compute_with_memory(inputs, &[])
    }

    /// Compute network output with memory bits available.
    pub fn compute_with_memory(&self, inputs: &[bool], memory_bits: &[bool]) -> Vec<bool> {
        // Evaluate gates in topological order
        let mut gate_outputs: Vec<bool> = Vec::with_capacity(self.gates.len());

        for gate in &self.gates {
            let output = gate.compute(inputs, &gate_outputs, memory_bits);
            gate_outputs.push(output);
        }

        // Generate outputs
        self.outputs
            .iter()
            .map(|src| resolve_input(src, inputs, &gate_outputs, memory_bits))
            .collect()
    }

    /// Get total number of valid input sources for a gate at position `gate_idx`.
    pub fn valid_source_count(&self, gate_idx: usize) -> usize {
        let memory_bits = self.memory_config.as_ref().map(|m| m.data_bits).unwrap_or(0);
        self.input_count as usize + gate_idx + memory_bits as usize
    }

    /// Check if an input source is valid for a gate at the given position.
    pub fn is_valid_source(&self, source: &InputSource, gate_idx: usize) -> bool {
        match source {
            InputSource::NetworkInput(idx) => (*idx as usize) < self.input_count as usize,
            InputSource::GateOutput(idx) => (*idx as usize) < gate_idx,
            InputSource::MemoryBit(idx) => {
                self.memory_config
                    .as_ref()
                    .map(|m| (*idx as usize) < m.data_bits as usize)
                    .unwrap_or(false)
            }
            InputSource::Constant(_) => true,
        }
    }

    /// Validate the network structure (all references are valid).
    pub fn validate(&self) -> bool {
        for (gate_idx, gate) in self.gates.iter().enumerate() {
            // Check gate has correct number of inputs
            if gate.inputs.len() != gate.gate_type.required_inputs() {
                return false;
            }
            // Check all inputs are valid
            for input in &gate.inputs {
                if !self.is_valid_source(input, gate_idx) {
                    return false;
                }
            }
        }
        // Check outputs are valid
        for output in &self.outputs {
            if !self.is_valid_source(output, self.gates.len()) {
                return false;
            }
        }
        true
    }

    /// Repair any invalid references in the network.
    pub fn repair(&mut self, rng: &mut impl Rng) {
        let memory_bits = self.memory_config.as_ref().map(|m| m.data_bits).unwrap_or(0);
        let input_count = self.input_count;
        let gate_count = self.gates.len();

        for gate_idx in 0..gate_count {
            let required_inputs = self.gates[gate_idx].gate_type.required_inputs();

            // Ensure correct number of inputs
            while self.gates[gate_idx].inputs.len() < required_inputs {
                self.gates[gate_idx].inputs.push(Self::random_input_source(
                    input_count,
                    gate_idx as u16,
                    memory_bits,
                    rng,
                ));
            }
            self.gates[gate_idx].inputs.truncate(required_inputs);

            // Collect invalid input indices first
            let invalid_indices: Vec<usize> = self.gates[gate_idx]
                .inputs
                .iter()
                .enumerate()
                .filter(|(_, input)| !Self::is_valid_source_static(input, gate_idx, input_count, &self.memory_config))
                .map(|(i, _)| i)
                .collect();

            // Repair invalid references
            for i in invalid_indices {
                self.gates[gate_idx].inputs[i] = Self::random_input_source(
                    input_count,
                    gate_idx as u16,
                    memory_bits,
                    rng,
                );
            }
        }

        // Repair outputs
        let output_count = self.outputs.len();
        for i in 0..output_count {
            if !Self::is_valid_source_static(&self.outputs[i], gate_count, input_count, &self.memory_config) {
                self.outputs[i] = Self::random_input_source(
                    input_count,
                    gate_count as u16,
                    memory_bits,
                    rng,
                );
            }
        }
    }

    /// Static version of is_valid_source that doesn't borrow self.
    fn is_valid_source_static(source: &InputSource, gate_idx: usize, input_count: u16, memory_config: &Option<MemoryConfig>) -> bool {
        match source {
            InputSource::NetworkInput(idx) => (*idx as usize) < input_count as usize,
            InputSource::GateOutput(idx) => (*idx as usize) < gate_idx,
            InputSource::MemoryBit(idx) => {
                memory_config
                    .as_ref()
                    .map(|m| (*idx as usize) < m.data_bits as usize)
                    .unwrap_or(false)
            }
            InputSource::Constant(_) => true,
        }
    }

    /// Count how many times each gate's output is used.
    pub fn gate_usage_counts(&self) -> HashMap<usize, usize> {
        let mut counts: HashMap<usize, usize> = HashMap::new();

        // Count usage in gates
        for gate in &self.gates {
            for input in &gate.inputs {
                if let InputSource::GateOutput(idx) = input {
                    *counts.entry(*idx as usize).or_insert(0) += 1;
                }
            }
        }

        // Count usage in outputs
        for output in &self.outputs {
            if let InputSource::GateOutput(idx) = output {
                *counts.entry(*idx as usize).or_insert(0) += 1;
            }
        }

        counts
    }

    /// Remove unused gates (dead code elimination).
    pub fn remove_unused_gates(&mut self) {
        loop {
            let usage = self.gate_usage_counts();
            let unused: Vec<usize> = (0..self.gates.len())
                .filter(|idx| !usage.contains_key(idx))
                .collect();

            if unused.is_empty() {
                break;
            }

            // Remove unused gates (in reverse order to preserve indices)
            for &idx in unused.iter().rev() {
                self.remove_gate_at(idx);
            }
        }
    }

    /// Remove gate at index and update all references.
    fn remove_gate_at(&mut self, remove_idx: usize) {
        self.gates.remove(remove_idx);

        // Update all gate references
        for gate in &mut self.gates {
            for input in &mut gate.inputs {
                if let InputSource::GateOutput(ref mut idx) = input {
                    if *idx as usize > remove_idx {
                        *idx -= 1;
                    }
                }
            }
        }

        // Update output references
        for output in &mut self.outputs {
            if let InputSource::GateOutput(ref mut idx) = output {
                if *idx as usize > remove_idx {
                    *idx -= 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_compute_nand() {
        assert!(!GateType::Nand.compute(&[true, true]));
        assert!(GateType::Nand.compute(&[true, false]));
        assert!(GateType::Nand.compute(&[false, true]));
        assert!(GateType::Nand.compute(&[false, false]));
    }

    #[test]
    fn test_gate_compute_and() {
        assert!(GateType::And.compute(&[true, true]));
        assert!(!GateType::And.compute(&[true, false]));
        assert!(!GateType::And.compute(&[false, true]));
        assert!(!GateType::And.compute(&[false, false]));
    }

    #[test]
    fn test_gate_compute_or() {
        assert!(GateType::Or.compute(&[true, true]));
        assert!(GateType::Or.compute(&[true, false]));
        assert!(GateType::Or.compute(&[false, true]));
        assert!(!GateType::Or.compute(&[false, false]));
    }

    #[test]
    fn test_gate_compute_xor() {
        assert!(!GateType::Xor.compute(&[true, true]));
        assert!(GateType::Xor.compute(&[true, false]));
        assert!(GateType::Xor.compute(&[false, true]));
        assert!(!GateType::Xor.compute(&[false, false]));
    }

    #[test]
    fn test_gate_compute_not() {
        assert!(!GateType::Not.compute(&[true]));
        assert!(GateType::Not.compute(&[false]));
    }

    #[test]
    fn test_gate_compute_mux2() {
        // select=false -> return input[1]
        assert!(GateType::Mux2.compute(&[false, true, false]));
        assert!(!GateType::Mux2.compute(&[false, false, true]));
        // select=true -> return input[2]
        assert!(!GateType::Mux2.compute(&[true, true, false]));
        assert!(GateType::Mux2.compute(&[true, false, true]));
    }

    #[test]
    fn test_network_random_creation() {
        let mut rng = rand::thread_rng();
        let network = DirectNetwork::random(10, 2, 50, None, &mut rng);

        assert_eq!(network.input_count, 10);
        assert_eq!(network.output_count, 2);
        assert_eq!(network.gates.len(), 50);
        assert_eq!(network.outputs.len(), 2);
        assert!(network.validate());
    }

    #[test]
    fn test_network_compute_simple() {
        // Create a simple AND gate network: output = input[0] AND input[1]
        let network = DirectNetwork {
            input_count: 2,
            output_count: 1,
            gates: vec![Gate::new(
                GateType::And,
                vec![InputSource::NetworkInput(0), InputSource::NetworkInput(1)],
            )],
            outputs: vec![InputSource::GateOutput(0)],
            memory_config: None,
        };

        assert_eq!(network.compute(&[true, true]), vec![true]);
        assert_eq!(network.compute(&[true, false]), vec![false]);
        assert_eq!(network.compute(&[false, true]), vec![false]);
        assert_eq!(network.compute(&[false, false]), vec![false]);
    }

    #[test]
    fn test_network_compute_chain() {
        // Create a chain: NOT(input[0]) AND input[1]
        let network = DirectNetwork {
            input_count: 2,
            output_count: 1,
            gates: vec![
                Gate::new(GateType::Not, vec![InputSource::NetworkInput(0)]),
                Gate::new(
                    GateType::And,
                    vec![InputSource::GateOutput(0), InputSource::NetworkInput(1)],
                ),
            ],
            outputs: vec![InputSource::GateOutput(1)],
            memory_config: None,
        };

        assert_eq!(network.compute(&[false, true]), vec![true]); // NOT(false) AND true = true
        assert_eq!(network.compute(&[true, true]), vec![false]); // NOT(true) AND true = false
        assert_eq!(network.compute(&[false, false]), vec![false]); // NOT(false) AND false = false
    }

    #[test]
    fn test_network_validate() {
        let valid_network = DirectNetwork {
            input_count: 2,
            output_count: 1,
            gates: vec![Gate::new(
                GateType::And,
                vec![InputSource::NetworkInput(0), InputSource::NetworkInput(1)],
            )],
            outputs: vec![InputSource::GateOutput(0)],
            memory_config: None,
        };
        assert!(valid_network.validate());

        // Invalid: gate references itself
        let invalid_network = DirectNetwork {
            input_count: 2,
            output_count: 1,
            gates: vec![Gate::new(
                GateType::And,
                vec![InputSource::GateOutput(0), InputSource::NetworkInput(1)],
            )],
            outputs: vec![InputSource::GateOutput(0)],
            memory_config: None,
        };
        assert!(!invalid_network.validate());
    }

    #[test]
    fn test_network_with_memory() {
        let network = DirectNetwork {
            input_count: 2,
            output_count: 1,
            gates: vec![Gate::new(
                GateType::And,
                vec![InputSource::NetworkInput(0), InputSource::MemoryBit(0)],
            )],
            outputs: vec![InputSource::GateOutput(0)],
            memory_config: Some(MemoryConfig {
                addr_bits: 2,
                data_bits: 4,
            }),
        };

        // input[0]=true, memory[0]=true -> true
        assert_eq!(network.compute_with_memory(&[true, false], &[true, false, false, false]), vec![true]);
        // input[0]=true, memory[0]=false -> false
        assert_eq!(network.compute_with_memory(&[true, false], &[false, true, false, false]), vec![false]);
    }

    #[test]
    fn test_remove_unused_gates() {
        // Gate 0 is unused, Gate 1 is used by output
        let mut network = DirectNetwork {
            input_count: 2,
            output_count: 1,
            gates: vec![
                Gate::new(GateType::And, vec![InputSource::NetworkInput(0), InputSource::NetworkInput(0)]),
                Gate::new(GateType::Or, vec![InputSource::NetworkInput(0), InputSource::NetworkInput(1)]),
            ],
            outputs: vec![InputSource::GateOutput(1)],
            memory_config: None,
        };

        network.remove_unused_gates();

        // Gate 0 should be removed, Gate 1 (now Gate 0) should remain
        assert_eq!(network.gates.len(), 1);
        assert_eq!(network.gates[0].gate_type, GateType::Or);
        // Output should now reference Gate 0
        assert_eq!(network.outputs[0], InputSource::GateOutput(0));
    }
}
