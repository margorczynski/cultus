use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

use serde::{Deserialize, Serialize};

use crate::evolution::chromosome::Chromosome;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChromosomeWithFitness<T: Clone + Eq + Send + Sync> {
    pub chromosome: Chromosome,
    pub fitness: T,
}

impl<T: Clone + Eq + Send + Sync> ChromosomeWithFitness<T> {
    pub fn new(chromosome: Chromosome, fitness: T) -> Self {
        ChromosomeWithFitness {
            chromosome,
            fitness,
        }
    }
}

impl<T: Clone + Eq + Send + Sync> PartialEq for ChromosomeWithFitness<T> {
    fn eq(&self, other: &Self) -> bool {
        self.chromosome == other.chromosome && self.fitness == other.fitness
    }
}

impl<T: Clone + Eq + Send + Sync> Eq for ChromosomeWithFitness<T> {}

impl<T: Clone + Eq + Send + Sync + Hash> Hash for ChromosomeWithFitness<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.chromosome.hash(state);
        self.fitness.hash(state);
    }
}

impl<T: PartialOrd + Clone + Eq + Send + Sync> PartialOrd for ChromosomeWithFitness<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        T::partial_cmp(&self.fitness, &other.fitness)
    }
}

impl<T: PartialOrd + Clone + Eq + Send + Sync> Ord for ChromosomeWithFitness<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}