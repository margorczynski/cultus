use std::cmp::Ordering;
use crate::evolution_logic::chromosome::Chromosome;

pub struct ChromosomeWithFitness<T> {
    chromosome: Chromosome,
    fitness: T
}

impl<T> ChromosomeWithFitness<T> {
    pub fn from_bitstring(s: &str, fitness: T) -> ChromosomeWithFitness<T> {
        let chromosome = Chromosome::from_bitstring(s);

        ChromosomeWithFitness {
            chromosome,
            fitness
        }
    }
}

impl<T: PartialEq> PartialEq for ChromosomeWithFitness<T> {
    fn eq(&self, other: &Self) -> bool {
        T::eq(&self.fitness, &other.fitness)
    }
}

impl<T: PartialEq> Eq for ChromosomeWithFitness<T> {}

impl<T: PartialOrd> PartialOrd  for ChromosomeWithFitness<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        T::partial_cmp(&self.fitness, &other.fitness)
    }
}