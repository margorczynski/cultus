use std::cmp::Ordering;
use serde::{Deserialize, Serialize};

use crate::evolution::chromosome::Chromosome;

#[derive(Hash, PartialEq, Eq, Clone, Debug, Serialize, Deserialize)]
pub struct ChromosomeWithFitness<T: Clone + Eq> {
    pub chromosome: Chromosome,
    pub fitness: T,
}

impl<T: Clone + Eq> ChromosomeWithFitness<T> {
    pub fn from_chromosome_and_fitness(
        chromosome: Chromosome,
        fitness: T,
    ) -> ChromosomeWithFitness<T> {
        ChromosomeWithFitness {
            chromosome,
            fitness,
        }
    }
}

impl<T: PartialOrd + Clone + Eq> PartialOrd for ChromosomeWithFitness<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        T::partial_cmp(&self.fitness, &other.fitness)
    }
}

impl<T: PartialOrd + Clone + Eq> Ord for ChromosomeWithFitness<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(&other).unwrap_or(Ordering::Equal)
    }
}
