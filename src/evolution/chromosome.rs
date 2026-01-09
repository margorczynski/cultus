use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};

use bitvec::prelude::*;
use rand::prelude::*;
use serde::{Deserialize, Serialize};

use crate::evolution::direct_encoding::{DirectNetwork, MemoryConfig};

/// Legacy chromosome using bit-vector encoding.
/// Kept for backward compatibility.
#[derive(Clone, Serialize, Deserialize)]
pub struct Chromosome {
    pub genes: BitVec<u64, Lsb0>,
}

/// New chromosome type using direct network encoding.
/// Every gate is valid by construction - no wasted bits.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectChromosome {
    pub network: DirectNetwork,
}

impl DirectChromosome {
    /// Create a new chromosome with a random network.
    pub fn random(
        input_count: u16,
        output_count: u16,
        gate_count: usize,
        memory_config: Option<MemoryConfig>,
        rng: &mut impl Rng,
    ) -> Self {
        DirectChromosome {
            network: DirectNetwork::random(input_count, output_count, gate_count, memory_config, rng),
        }
    }

    /// Create a chromosome wrapping an existing network.
    pub fn from_network(network: DirectNetwork) -> Self {
        DirectChromosome { network }
    }

    /// Get the number of gates in this chromosome's network.
    pub fn gate_count(&self) -> usize {
        self.network.gates.len()
    }

    /// Compute the network output for given inputs.
    pub fn compute(&self, inputs: &[bool]) -> Vec<bool> {
        self.network.compute(inputs)
    }

    /// Compute with memory bits available.
    pub fn compute_with_memory(&self, inputs: &[bool], memory_bits: &[bool]) -> Vec<bool> {
        self.network.compute_with_memory(inputs, memory_bits)
    }
}

impl Hash for DirectChromosome {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash gate count and types for quick comparison
        self.network.gates.len().hash(state);
        for gate in &self.network.gates {
            std::mem::discriminant(&gate.gate_type).hash(state);
            for input in &gate.inputs {
                std::mem::discriminant(input).hash(state);
                match input {
                    crate::evolution::direct_encoding::InputSource::NetworkInput(i) => i.hash(state),
                    crate::evolution::direct_encoding::InputSource::GateOutput(i) => i.hash(state),
                    crate::evolution::direct_encoding::InputSource::MemoryBit(i) => i.hash(state),
                    crate::evolution::direct_encoding::InputSource::Constant(b) => b.hash(state),
                }
            }
        }
        for output in &self.network.outputs {
            std::mem::discriminant(output).hash(state);
        }
    }
}

impl Display for DirectChromosome {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DirectChromosome(gates={}, inputs={}, outputs={})",
            self.network.gates.len(),
            self.network.input_count,
            self.network.output_count
        )
    }
}

impl Chromosome {
    pub fn from_bitvec(genes: BitVec<u64, Lsb0>) -> Chromosome {
        Chromosome { genes }
    }

    pub fn from_bool_iter<I: IntoIterator<Item = bool>>(iter: I) -> Chromosome {
        Chromosome {
            genes: iter.into_iter().collect(),
        }
    }

    pub fn len(&self) -> usize {
        self.genes.len()
    }
}

impl PartialEq for Chromosome {
    fn eq(&self, other: &Self) -> bool {
        self.genes == other.genes
    }
}

impl Eq for Chromosome {}

impl Hash for Chromosome {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash the raw storage for efficiency
        self.genes.as_raw_slice().hash(state);
        self.genes.len().hash(state);
    }
}

impl Display for Chromosome {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let result: String = self.genes.iter().map(|b| if *b { '1' } else { '0' }).collect();
        write!(f, "{}", result)
    }
}

impl Debug for Chromosome {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}
