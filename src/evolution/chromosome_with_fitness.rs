use std::cmp::Ordering;

use crate::evolution::chromosome::Chromosome;

#[derive(Hash, Clone, Debug)]
pub struct ChromosomeWithFitness<T: Clone> {
    pub chromosome: Chromosome,
    pub fitness: T,
}

impl<T: Clone> ChromosomeWithFitness<T> {
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

impl<T: PartialEq + Clone> PartialEq for ChromosomeWithFitness<T> {
    fn eq(&self, other: &Self) -> bool {
        T::eq(&self.fitness, &other.fitness)
    }
}

impl<T: PartialEq + Clone> Eq for ChromosomeWithFitness<T> {}

impl<T: PartialOrd + Clone> PartialOrd for ChromosomeWithFitness<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        T::partial_cmp(&self.fitness, &other.fitness)
    }
}

impl<T: PartialOrd + Clone> Ord for ChromosomeWithFitness<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(&other).unwrap_or(Ordering::Equal)
    }
}
