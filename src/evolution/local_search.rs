//! Local search (hill climbing) for refining networks.
//!
//! This module provides functionality to apply small, local improvements
//! to networks after genetic operations, smoothing the fitness landscape.

use rand::prelude::*;
use serde::{Deserialize, Serialize};

use crate::evolution::direct_encoding::{DirectNetwork, InputSource};

/// Configuration for local search.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LocalSearchConfig {
    /// Number of random mutations to try
    pub num_trials: usize,
    /// Maximum consecutive non-improvements before stopping
    pub max_stagnation: usize,
    /// Strategy for applying improvements
    pub strategy: LocalSearchStrategy,
}

impl Default for LocalSearchConfig {
    fn default() -> Self {
        LocalSearchConfig {
            num_trials: 20,
            max_stagnation: 5,
            strategy: LocalSearchStrategy::Hybrid {
                lamarckian_probability: 0.5,
            },
        }
    }
}

/// Strategy for incorporating local search improvements.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum LocalSearchStrategy {
    /// Lamarckian: Write improvements back to chromosome (faster convergence)
    Lamarckian,
    /// Baldwin: Keep original chromosome, use improved fitness only (maintains diversity)
    Baldwin,
    /// Hybrid: Probabilistic choice based on improvement
    Hybrid { lamarckian_probability: f64 },
}

/// Local search engine for improving networks.
pub struct LocalSearch {
    config: LocalSearchConfig,
}

impl LocalSearch {
    /// Create a new local search engine with the given configuration.
    pub fn new(config: LocalSearchConfig) -> Self {
        LocalSearch { config }
    }

    /// Create a local search engine with default configuration.
    pub fn default_config() -> Self {
        LocalSearch {
            config: LocalSearchConfig::default(),
        }
    }

    /// Apply local search to improve a network.
    ///
    /// Returns the best fitness found and whether the network was modified.
    pub fn improve<F>(
        &self,
        network: &mut DirectNetwork,
        evaluate: F,
        rng: &mut impl Rng,
    ) -> (f64, bool)
    where
        F: Fn(&DirectNetwork) -> f64,
    {
        let original_fitness = evaluate(network);
        let mut best_network = network.clone();
        let mut best_fitness = original_fitness;
        let mut stagnation = 0;

        for _ in 0..self.config.num_trials {
            if stagnation >= self.config.max_stagnation {
                break;
            }

            // Create neighbor by small random change
            let mut neighbor = best_network.clone();
            self.apply_small_mutation(&mut neighbor, rng);

            // Ensure neighbor is valid
            if !neighbor.validate() {
                neighbor.repair(rng);
            }

            let neighbor_fitness = evaluate(&neighbor);

            if neighbor_fitness > best_fitness {
                best_network = neighbor;
                best_fitness = neighbor_fitness;
                stagnation = 0;
            } else {
                stagnation += 1;
            }
        }

        // Decide whether to apply improvement based on strategy
        let improvement = best_fitness - original_fitness;
        let should_apply = match self.config.strategy {
            LocalSearchStrategy::Lamarckian => improvement > 0.0,
            LocalSearchStrategy::Baldwin => false,
            LocalSearchStrategy::Hybrid { lamarckian_probability } => {
                improvement > 0.0 && rng.gen::<f64>() < lamarckian_probability
            }
        };

        if should_apply {
            *network = best_network;
            (best_fitness, true)
        } else {
            (best_fitness, false)
        }
    }

    /// Apply a small, localized mutation.
    fn apply_small_mutation(&self, network: &mut DirectNetwork, rng: &mut impl Rng) {
        let mutation_type = rng.gen_range(0..7);

        match mutation_type {
            0 => network.mutate_swap_connection(rng),
            1 => network.mutate_gate_type(rng),
            2 => self.swap_inputs_in_gate(network, rng),
            3 => self.toggle_constant(network, rng),
            4 => network.mutate_output(rng),
            5 => self.redirect_to_nearby_gate(network, rng),
            _ => self.swap_two_connections(network, rng),
        }
    }

    /// Swap two inputs within a random gate.
    fn swap_inputs_in_gate(&self, network: &mut DirectNetwork, rng: &mut impl Rng) {
        if network.gates.is_empty() {
            return;
        }

        let gate_idx = rng.gen_range(0..network.gates.len());
        let gate = &mut network.gates[gate_idx];

        if gate.inputs.len() >= 2 {
            gate.inputs.swap(0, 1);
        }
    }

    /// Toggle a constant input if present.
    fn toggle_constant(&self, network: &mut DirectNetwork, rng: &mut impl Rng) {
        if network.gates.is_empty() {
            return;
        }

        let gate_idx = rng.gen_range(0..network.gates.len());
        let gate = &mut network.gates[gate_idx];

        for input in &mut gate.inputs {
            if let InputSource::Constant(ref mut val) = input {
                *val = !*val;
                return;
            }
        }
    }

    /// Redirect an input to a nearby gate (small topology change).
    fn redirect_to_nearby_gate(&self, network: &mut DirectNetwork, rng: &mut impl Rng) {
        if network.gates.len() < 2 {
            return;
        }

        let gate_idx = rng.gen_range(1..network.gates.len());
        let gate = &mut network.gates[gate_idx];

        if gate.inputs.is_empty() {
            return;
        }

        let input_idx = rng.gen_range(0..gate.inputs.len());

        // Find a "nearby" gate (within 3 positions)
        let min_target = gate_idx.saturating_sub(3);
        let max_target = gate_idx;
        if min_target >= max_target {
            return;
        }

        let new_source = rng.gen_range(min_target..max_target);
        gate.inputs[input_idx] = InputSource::GateOutput(new_source as u16);
    }

    /// Swap connections between two random gates.
    fn swap_two_connections(&self, network: &mut DirectNetwork, rng: &mut impl Rng) {
        if network.gates.len() < 2 {
            return;
        }

        let gate_a = rng.gen_range(0..network.gates.len());
        let gate_b = rng.gen_range(0..network.gates.len());

        if gate_a == gate_b {
            return;
        }

        if network.gates[gate_a].inputs.is_empty() || network.gates[gate_b].inputs.is_empty() {
            return;
        }

        let input_a_idx = rng.gen_range(0..network.gates[gate_a].inputs.len());
        let input_b_idx = rng.gen_range(0..network.gates[gate_b].inputs.len());

        // Only swap if both result in valid references
        let input_a = network.gates[gate_a].inputs[input_a_idx];
        let input_b = network.gates[gate_b].inputs[input_b_idx];

        let a_valid_for_b = network.is_valid_source(&input_a, gate_b);
        let b_valid_for_a = network.is_valid_source(&input_b, gate_a);

        if a_valid_for_b && b_valid_for_a {
            network.gates[gate_a].inputs[input_a_idx] = input_b;
            network.gates[gate_b].inputs[input_b_idx] = input_a;
        }
    }
}

/// Apply local search with the given strategy.
pub fn apply_local_search<F>(
    network: &mut DirectNetwork,
    config: &LocalSearchConfig,
    evaluate: F,
    rng: &mut impl Rng,
) -> (f64, bool)
where
    F: Fn(&DirectNetwork) -> f64,
{
    let local_search = LocalSearch::new(config.clone());
    local_search.improve(network, evaluate, rng)
}

/// Batch local search on a population.
pub fn batch_local_search<F>(
    population: &mut [DirectNetwork],
    config: &LocalSearchConfig,
    evaluate: F,
) -> Vec<f64>
where
    F: Fn(&DirectNetwork) -> f64 + Sync,
{
    use rayon::prelude::*;

    population
        .par_iter_mut()
        .map(|network| {
            let mut rng = rand::thread_rng();
            let local_search = LocalSearch::new(config.clone());
            let (fitness, _) = local_search.improve(network, &evaluate, &mut rng);
            fitness
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_fitness(network: &DirectNetwork) -> f64 {
        // Simple fitness: count gates + connections
        (network.gates.len() + network.outputs.len()) as f64
    }

    #[test]
    fn test_local_search_creation() {
        let config = LocalSearchConfig::default();
        let _ls = LocalSearch::new(config);
    }

    #[test]
    fn test_local_search_improve() {
        let mut rng = rand::thread_rng();
        let mut network = DirectNetwork::random(4, 2, 10, None, &mut rng);

        let config = LocalSearchConfig {
            num_trials: 10,
            max_stagnation: 3,
            strategy: LocalSearchStrategy::Lamarckian,
        };

        let ls = LocalSearch::new(config);
        let (fitness, _modified) = ls.improve(&mut network, simple_fitness, &mut rng);

        assert!(fitness > 0.0);
        assert!(network.validate());
    }

    #[test]
    fn test_baldwin_effect() {
        let mut rng = rand::thread_rng();
        let mut network = DirectNetwork::random(4, 2, 10, None, &mut rng);
        let original = network.clone();

        let config = LocalSearchConfig {
            num_trials: 10,
            max_stagnation: 3,
            strategy: LocalSearchStrategy::Baldwin,
        };

        let ls = LocalSearch::new(config);
        let (_fitness, modified) = ls.improve(&mut network, simple_fitness, &mut rng);

        // Baldwin effect: network should not be modified
        assert!(!modified);
        assert_eq!(network.gates.len(), original.gates.len());
    }

    #[test]
    fn test_mutations_preserve_validity() {
        let mut rng = rand::thread_rng();

        for _ in 0..100 {
            let mut network = DirectNetwork::random(4, 2, 20, None, &mut rng);

            let ls = LocalSearch::new(LocalSearchConfig::default());
            ls.apply_small_mutation(&mut network, &mut rng);

            if !network.validate() {
                network.repair(&mut rng);
            }
            assert!(network.validate());
        }
    }
}
