//! Module detection and module-aware genetic operators.
//!
//! This module provides functionality to identify functional subgraphs (modules)
//! within networks and perform crossover operations that preserve these modules.

use rand::prelude::*;
use std::collections::{HashMap, HashSet};

use crate::evolution::direct_encoding::{DirectNetwork, Gate, GateType, InputSource};

/// A functional subgraph (module) within a network.
#[derive(Clone, Debug)]
pub struct Module {
    /// Gate indices in this module
    pub gate_indices: HashSet<usize>,
    /// Input sources from outside the module
    pub external_inputs: Vec<InputSource>,
    /// Gate indices that provide outputs used outside the module
    pub output_gates: Vec<usize>,
}

impl Module {
    /// Get the size (number of gates) in this module.
    pub fn size(&self) -> usize {
        self.gate_indices.len()
    }
}

impl DirectNetwork {
    /// Detect functional modules in the network.
    ///
    /// A module is a connected subgraph of gates that work together.
    /// Gates are connected if one depends on another's output.
    pub fn detect_modules(&self, min_size: usize) -> Vec<Module> {
        if self.gates.is_empty() {
            return Vec::new();
        }

        // Build dependency graph
        let mut dependencies: HashMap<usize, HashSet<usize>> = HashMap::new();
        let mut dependents: HashMap<usize, HashSet<usize>> = HashMap::new();

        for (gate_idx, gate) in self.gates.iter().enumerate() {
            for input in &gate.inputs {
                if let InputSource::GateOutput(src_idx) = input {
                    dependencies
                        .entry(gate_idx)
                        .or_default()
                        .insert(*src_idx as usize);
                    dependents
                        .entry(*src_idx as usize)
                        .or_default()
                        .insert(gate_idx);
                }
            }
        }

        // Find connected components using union-find style approach
        let mut visited: HashSet<usize> = HashSet::new();
        let mut modules = Vec::new();

        for start_gate in 0..self.gates.len() {
            if visited.contains(&start_gate) {
                continue;
            }

            // BFS to find connected subgraph
            let mut module_gates = HashSet::new();
            let mut queue = vec![start_gate];

            while let Some(idx) = queue.pop() {
                if module_gates.contains(&idx) || visited.contains(&idx) {
                    continue;
                }
                module_gates.insert(idx);

                // Add dependencies (gates this gate depends on)
                if let Some(deps) = dependencies.get(&idx) {
                    for &dep in deps {
                        if !module_gates.contains(&dep) && !visited.contains(&dep) {
                            queue.push(dep);
                        }
                    }
                }

                // Add dependents (gates that depend on this gate)
                if let Some(deps) = dependents.get(&idx) {
                    for &dep in deps {
                        if !module_gates.contains(&dep) && !visited.contains(&dep) {
                            queue.push(dep);
                        }
                    }
                }
            }

            if module_gates.len() >= min_size {
                let module = self.create_module_from_gates(&module_gates, &dependencies);
                modules.push(module);
            }

            visited.extend(&module_gates);
        }

        modules
    }

    /// Create a Module from a set of gate indices.
    fn create_module_from_gates(
        &self,
        gate_indices: &HashSet<usize>,
        dependencies: &HashMap<usize, HashSet<usize>>,
    ) -> Module {
        let mut external_inputs = Vec::new();
        let mut output_gates = Vec::new();

        for &idx in gate_indices {
            let gate = &self.gates[idx];

            // Find external inputs (inputs not from within the module)
            for input in &gate.inputs {
                match input {
                    InputSource::GateOutput(src_idx) => {
                        if !gate_indices.contains(&(*src_idx as usize)) {
                            external_inputs.push(*input);
                        }
                    }
                    _ => external_inputs.push(*input),
                }
            }

            // Check if this gate's output is used outside the module
            let used_outside = self.gates.iter().enumerate().any(|(other_idx, other_gate)| {
                !gate_indices.contains(&other_idx)
                    && other_gate
                        .inputs
                        .iter()
                        .any(|inp| matches!(inp, InputSource::GateOutput(i) if *i as usize == idx))
            });

            // Also check if used by network outputs
            let used_by_output = self
                .outputs
                .iter()
                .any(|out| matches!(out, InputSource::GateOutput(i) if *i as usize == idx));

            if used_outside || used_by_output {
                output_gates.push(idx);
            }
        }

        Module {
            gate_indices: gate_indices.clone(),
            external_inputs,
            output_gates,
        }
    }

    // ==================== Structural Mutations ====================

    /// Add a new gate at a random position.
    pub fn mutate_add_gate(&mut self, rng: &mut impl Rng) {
        let insert_pos = rng.gen_range(0..=self.gates.len());
        let memory_bits = self.memory_config.as_ref().map(|m| m.register_width).unwrap_or(0);

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

        // Create inputs that reference only earlier sources
        let num_inputs = gate_type.required_inputs();
        let inputs: Vec<InputSource> = (0..num_inputs)
            .map(|_| Self::random_input_source(self.input_count, insert_pos as u16, memory_bits, rng))
            .collect();

        let new_gate = Gate::new(gate_type, inputs);
        self.gates.insert(insert_pos, new_gate);

        // Update all references to gates at or after insert position
        self.shift_references_after(insert_pos, 1);
    }

    /// Remove a random gate.
    pub fn mutate_remove_gate(&mut self, rng: &mut impl Rng) {
        if self.gates.is_empty() {
            return;
        }

        let remove_pos = rng.gen_range(0..self.gates.len());
        self.remove_gate_and_repair(remove_pos, rng);
    }

    /// Remove a gate and repair broken references.
    fn remove_gate_and_repair(&mut self, remove_pos: usize, rng: &mut impl Rng) {
        let memory_bits = self.memory_config.as_ref().map(|m| m.register_width).unwrap_or(0);
        self.gates.remove(remove_pos);

        // Repair references in remaining gates
        for gate_idx in 0..self.gates.len() {
            let gate = &mut self.gates[gate_idx];
            for input in &mut gate.inputs {
                if let InputSource::GateOutput(ref mut idx) = input {
                    if *idx as usize == remove_pos {
                        // Reference to removed gate - pick new random source
                        *input = Self::random_input_source(
                            self.input_count,
                            gate_idx as u16,
                            memory_bits,
                            rng,
                        );
                    } else if *idx as usize > remove_pos {
                        *idx -= 1;
                    }
                }
            }
        }

        // Repair output references
        for output in &mut self.outputs {
            if let InputSource::GateOutput(ref mut idx) = output {
                if *idx as usize == remove_pos {
                    *output = Self::random_input_source(
                        self.input_count,
                        self.gates.len() as u16,
                        memory_bits,
                        rng,
                    );
                } else if *idx as usize > remove_pos {
                    *idx -= 1;
                }
            }
        }
    }

    /// Shift all gate references >= position by delta.
    fn shift_references_after(&mut self, position: usize, delta: i32) {
        for gate in &mut self.gates[position + 1..] {
            for input in &mut gate.inputs {
                if let InputSource::GateOutput(ref mut idx) = input {
                    if *idx as usize >= position {
                        *idx = (*idx as i32 + delta) as u16;
                    }
                }
            }
        }

        for output in &mut self.outputs {
            if let InputSource::GateOutput(ref mut idx) = output {
                if *idx as usize >= position {
                    *idx = (*idx as i32 + delta) as u16;
                }
            }
        }
    }

    /// Swap a random connection to a new source.
    pub fn mutate_swap_connection(&mut self, rng: &mut impl Rng) {
        if self.gates.is_empty() {
            return;
        }

        let memory_bits = self.memory_config.as_ref().map(|m| m.register_width).unwrap_or(0);
        let gate_idx = rng.gen_range(0..self.gates.len());
        let gate = &mut self.gates[gate_idx];

        if gate.inputs.is_empty() {
            return;
        }

        let input_idx = rng.gen_range(0..gate.inputs.len());
        gate.inputs[input_idx] =
            Self::random_input_source(self.input_count, gate_idx as u16, memory_bits, rng);
    }

    /// Mutate a random gate's type to a compatible type.
    pub fn mutate_gate_type(&mut self, rng: &mut impl Rng) {
        if self.gates.is_empty() {
            return;
        }

        let gate_idx = rng.gen_range(0..self.gates.len());
        let gate = &mut self.gates[gate_idx];

        let current_inputs = gate.inputs.len();
        let compatible_types = GateType::compatible_types(current_inputs);

        if let Some(&new_type) = compatible_types.choose(rng) {
            gate.gate_type = new_type;
            // Trim inputs if new type needs fewer
            gate.inputs.truncate(new_type.required_inputs());
        }
    }

    /// Duplicate a random subgraph (module).
    pub fn mutate_duplicate_subgraph(&mut self, rng: &mut impl Rng) {
        let modules = self.detect_modules(2);
        if modules.is_empty() {
            return;
        }

        let module = modules.choose(rng).unwrap();

        // Duplicate module gates at end
        let start_idx = self.gates.len();
        let mut index_map: HashMap<usize, usize> = HashMap::new();

        // Create index mapping from old to new positions
        let sorted_indices: Vec<usize> = {
            let mut v: Vec<_> = module.gate_indices.iter().copied().collect();
            v.sort();
            v
        };

        for (i, &old_idx) in sorted_indices.iter().enumerate() {
            index_map.insert(old_idx, start_idx + i);
        }

        // Clone and remap gates
        for &gate_idx in &sorted_indices {
            let mut gate = self.gates[gate_idx].clone();

            // Remap internal references to duplicated gates
            for input in &mut gate.inputs {
                if let InputSource::GateOutput(ref mut idx) = input {
                    if let Some(&new_idx) = index_map.get(&(*idx as usize)) {
                        *idx = new_idx as u16;
                    }
                    // External references stay as-is
                }
            }

            self.gates.push(gate);
        }
    }

    /// Swap inputs within a random gate.
    pub fn mutate_swap_inputs(&mut self, rng: &mut impl Rng) {
        if self.gates.is_empty() {
            return;
        }

        let gate_idx = rng.gen_range(0..self.gates.len());
        let gate = &mut self.gates[gate_idx];

        if gate.inputs.len() >= 2 {
            let i = rng.gen_range(0..gate.inputs.len());
            let j = rng.gen_range(0..gate.inputs.len());
            gate.inputs.swap(i, j);
        }
    }

    /// Redirect a random output to a different source.
    pub fn mutate_output(&mut self, rng: &mut impl Rng) {
        if self.outputs.is_empty() {
            return;
        }

        let memory_bits = self.memory_config.as_ref().map(|m| m.register_width).unwrap_or(0);
        let output_idx = rng.gen_range(0..self.outputs.len());
        self.outputs[output_idx] =
            Self::random_input_source(self.input_count, self.gates.len() as u16, memory_bits, rng);
    }
}

/// Module-level crossover: swap functional subgraphs between parents.
pub fn module_crossover(
    parent_a: &DirectNetwork,
    parent_b: &DirectNetwork,
    rng: &mut impl Rng,
) -> (DirectNetwork, DirectNetwork) {
    let modules_a = parent_a.detect_modules(2);
    let modules_b = parent_b.detect_modules(2);

    if modules_a.is_empty() || modules_b.is_empty() {
        // Fall back to gate-level crossover
        return gate_level_crossover(parent_a, parent_b, rng);
    }

    // Select random module from each parent
    let module_a = modules_a.choose(rng).unwrap();
    let module_b = modules_b.choose(rng).unwrap();

    // Create children by swapping modules
    let child_a = swap_module(parent_a, parent_b, module_a, module_b, rng);
    let child_b = swap_module(parent_b, parent_a, module_b, module_a, rng);

    (child_a, child_b)
}

/// Swap a module between two networks.
fn swap_module(
    recipient: &DirectNetwork,
    donor: &DirectNetwork,
    recipient_module: &Module,
    donor_module: &Module,
    rng: &mut impl Rng,
) -> DirectNetwork {
    let mut child = recipient.clone();
    let memory_bits = child.memory_config.as_ref().map(|m| m.register_width).unwrap_or(0);

    // Remove recipient module gates (in reverse order)
    let mut sorted_remove: Vec<_> = recipient_module.gate_indices.iter().copied().collect();
    sorted_remove.sort();
    sorted_remove.reverse();

    for &idx in &sorted_remove {
        if idx < child.gates.len() {
            child.remove_gate_and_repair(idx, rng);
        }
    }

    // Add donor module gates at end
    let insert_point = child.gates.len();
    let mut index_map: HashMap<usize, usize> = HashMap::new();

    let sorted_donor: Vec<usize> = {
        let mut v: Vec<_> = donor_module.gate_indices.iter().copied().collect();
        v.sort();
        v
    };

    for (i, &old_idx) in sorted_donor.iter().enumerate() {
        index_map.insert(old_idx, insert_point + i);
    }

    for &gate_idx in &sorted_donor {
        if gate_idx >= donor.gates.len() {
            continue;
        }
        let mut gate = donor.gates[gate_idx].clone();

        // Remap input references
        for input in &mut gate.inputs {
            if let InputSource::GateOutput(ref mut idx) = input {
                if let Some(&new_idx) = index_map.get(&(*idx as usize)) {
                    *idx = new_idx as u16;
                } else {
                    // External reference - connect to random valid source
                    *input = DirectNetwork::random_input_source(
                        child.input_count,
                        child.gates.len() as u16,
                        memory_bits,
                        rng,
                    );
                }
            }
        }

        child.gates.push(gate);
    }

    // Repair any invalid output references
    child.repair(rng);

    child
}

/// Simple gate-level crossover (fallback).
pub fn gate_level_crossover(
    parent_a: &DirectNetwork,
    parent_b: &DirectNetwork,
    rng: &mut impl Rng,
) -> (DirectNetwork, DirectNetwork) {
    let mut child_a = parent_a.clone();
    let mut child_b = parent_b.clone();

    if parent_a.gates.is_empty() || parent_b.gates.is_empty() {
        return (child_a, child_b);
    }

    // Single-point crossover on gates
    let cross_point_a = rng.gen_range(0..=parent_a.gates.len());
    let cross_point_b = rng.gen_range(0..=parent_b.gates.len());

    // Child A = first part of A + second part of B
    child_a.gates.truncate(cross_point_a);
    for gate in &parent_b.gates[cross_point_b..] {
        child_a.gates.push(gate.clone());
    }

    // Child B = first part of B + second part of A
    child_b.gates.truncate(cross_point_b);
    for gate in &parent_a.gates[cross_point_a..] {
        child_b.gates.push(gate.clone());
    }

    // Repair both children
    child_a.repair(rng);
    child_b.repair(rng);

    (child_a, child_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_modules() {
        // Create a network with two independent chains
        let network = DirectNetwork {
            input_count: 4,
            output_count: 2,
            gates: vec![
                // Chain 1: gates 0, 1
                Gate::new(GateType::And, vec![InputSource::NetworkInput(0), InputSource::NetworkInput(1)]),
                Gate::new(GateType::Not, vec![InputSource::GateOutput(0)]),
                // Chain 2: gates 2, 3
                Gate::new(GateType::Or, vec![InputSource::NetworkInput(2), InputSource::NetworkInput(3)]),
                Gate::new(GateType::Not, vec![InputSource::GateOutput(2)]),
            ],
            outputs: vec![InputSource::GateOutput(1), InputSource::GateOutput(3)],
            memory_config: None,
        };

        let modules = network.detect_modules(2);
        assert_eq!(modules.len(), 2);
    }

    #[test]
    fn test_mutate_add_gate() {
        let mut rng = rand::thread_rng();
        let mut network = DirectNetwork::random(4, 2, 5, None, &mut rng);

        let initial_count = network.gates.len();
        network.mutate_add_gate(&mut rng);

        assert_eq!(network.gates.len(), initial_count + 1);
        assert!(network.validate());
    }

    #[test]
    fn test_mutate_remove_gate() {
        let mut rng = rand::thread_rng();
        let mut network = DirectNetwork::random(4, 2, 10, None, &mut rng);

        let initial_count = network.gates.len();
        network.mutate_remove_gate(&mut rng);

        assert_eq!(network.gates.len(), initial_count - 1);
        assert!(network.validate());
    }

    #[test]
    fn test_mutate_swap_connection() {
        let mut rng = rand::thread_rng();
        let mut network = DirectNetwork::random(4, 2, 10, None, &mut rng);

        network.mutate_swap_connection(&mut rng);
        assert!(network.validate());
    }

    #[test]
    fn test_gate_level_crossover() {
        let mut rng = rand::thread_rng();
        let parent_a = DirectNetwork::random(4, 2, 10, None, &mut rng);
        let parent_b = DirectNetwork::random(4, 2, 10, None, &mut rng);

        let (child_a, child_b) = gate_level_crossover(&parent_a, &parent_b, &mut rng);

        assert!(child_a.validate());
        assert!(child_b.validate());
    }
}
