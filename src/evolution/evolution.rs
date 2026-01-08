use std::cmp::max;
use std::collections::HashSet;
use std::fmt::Display;

use bitvec::prelude::*;
use log::debug;
use rand::prelude::*;
use rand::{Rng, SeedableRng};
use rand_distr::Binomial;
use rayon::prelude::*;

use rand::distributions::Uniform;

use crate::evolution::chromosome::Chromosome;
use crate::evolution::chromosome_with_fitness::ChromosomeWithFitness;

#[derive(Debug)]
pub enum SelectionStrategy {
    Tournament(usize),
}

pub fn generate_initial_population(
    initial_population_count: usize,
    chromosome_size: usize,
) -> HashSet<Chromosome> {
    debug!(
        "Generating initial population - initial_population_count: {}, chromosome_size: {}",
        initial_population_count, chromosome_size
    );

    let mut population: HashSet<Chromosome> = HashSet::new();

    // Generate chromosomes in parallel using thread-local RNG
    let chromosomes: Vec<Chromosome> = (0..initial_population_count)
        .into_par_iter()
        .map(|_| {
            let mut rng = thread_rng();
            let genes: BitVec<u64, Lsb0> = (0..chromosome_size).map(|_| rng.gen::<bool>()).collect();
            Chromosome::from_bitvec(genes)
        })
        .collect();

    population.extend(chromosomes);

    // Fill any gaps caused by duplicates
    let mut rng = StdRng::from_entropy();
    while population.len() < initial_population_count {
        let genes: BitVec<u64, Lsb0> = (0..chromosome_size).map(|_| rng.gen::<bool>()).collect();
        population.insert(Chromosome::from_bitvec(genes));
    }

    population
}

pub fn evolve<T: PartialEq + PartialOrd + Ord + Clone + Eq + Send + Sync + Into<f64> + Display>(
    chromosomes_with_fitness: &HashSet<ChromosomeWithFitness<T>>,
    selection_strategy: SelectionStrategy,
    elite_factor: f32,
) -> HashSet<Chromosome> {
    let mut new_generation: HashSet<Chromosome> = HashSet::new();

    let elite_amount = ((chromosomes_with_fitness.len() as f32) * elite_factor).floor() as usize;

    debug!("Elite amount: {}", elite_amount);

    let mut chromosomes_with_fitness_ordered: Vec<ChromosomeWithFitness<T>> =
        chromosomes_with_fitness.iter().cloned().collect();

    chromosomes_with_fitness_ordered.sort_unstable();

    let elite = chromosomes_with_fitness_ordered
        .par_iter()
        .rev()
        .take(elite_amount)
        .cloned()
        .map(|cwf| cwf.chromosome);

    new_generation.par_extend(elite);

    let fitness_max: f64 = chromosomes_with_fitness_ordered
        .iter()
        .max()
        .unwrap()
        .clone()
        .fitness
        .into();
    let fitness_avg: f64 = chromosomes_with_fitness_ordered
        .iter()
        .map(|cwf| cwf.fitness.clone().into())
        .sum::<f64>()
        / chromosomes_with_fitness_ordered.len() as f64;

    let offspring = (0..((chromosomes_with_fitness.len() - new_generation.len()) / 2))
        .into_par_iter()
        .map(|_| {
            let parents = select(chromosomes_with_fitness, &selection_strategy);
            let (offspring_1, offspring_2) =
                crossover(parents, 0.7, 1.0, 0.05, 0.5, fitness_avg, fitness_max);
            vec![offspring_1, offspring_2]
        })
        .flatten();

    new_generation.par_extend(offspring);

    // Fallback for duplicates
    while new_generation.len() != chromosomes_with_fitness.len() {
        let parents = select(chromosomes_with_fitness, &selection_strategy);
        let offspring = crossover(parents, 0.7, 1.0, 0.05, 0.5, fitness_avg, fitness_max);

        new_generation.insert(offspring.0);
        if new_generation.len() == chromosomes_with_fitness.len() {
            break;
        }
        new_generation.insert(offspring.1);
    }

    debug!(
        "Total number of chromosomes after crossovers (+ elites retained): {}",
        new_generation.len()
    );

    new_generation
}

fn select<T: PartialEq + PartialOrd + Ord + Clone + Eq + Send + Sync>(
    chromosomes_with_fitness: &HashSet<ChromosomeWithFitness<T>>,
    selection_strategy: &SelectionStrategy,
) -> (ChromosomeWithFitness<T>, ChromosomeWithFitness<T>) {
    match *selection_strategy {
        SelectionStrategy::Tournament(tournament_size) => {
            let mut rng = thread_rng();
            let mut get_winner = |cwf: &HashSet<ChromosomeWithFitness<T>>| {
                cwf.iter()
                    .choose_multiple(&mut rng, tournament_size)
                    .into_iter()
                    .max()
                    .unwrap()
                    .clone()
            };

            let first = get_winner(chromosomes_with_fitness);
            let second = get_winner(chromosomes_with_fitness);

            (first, second)
        }
    }
}

fn crossover<T: PartialEq + PartialOrd + Ord + Clone + Eq + Send + Sync + Into<f64> + Display>(
    parents: (ChromosomeWithFitness<T>, ChromosomeWithFitness<T>),
    crossover_rate_min: f64,
    crossover_rate_max: f64,
    mutation_rate_min: f64,
    mutation_rate_max: f64,
    fitness_avg: f64,
    fitness_max: f64,
) -> (Chromosome, Chromosome) {
    let mut rng = thread_rng();

    let chromosome_len = parents.0.chromosome.len();

    let fitness_parents = max(parents.0.fitness.clone(), parents.1.fitness.clone()).into();
    let fitness_delta = fitness_max - fitness_parents;

    let crossover_rate: f64;
    let mutation_rate: f64;

    if fitness_parents >= fitness_avg || fitness_avg == fitness_max {
        crossover_rate = crossover_rate_max;
    } else {
        crossover_rate = crossover_rate_min * (fitness_delta / (fitness_max - fitness_avg));
    }

    if fitness_parents < fitness_avg || fitness_avg == fitness_max {
        mutation_rate = mutation_rate_max;
    } else {
        mutation_rate = mutation_rate_min * (fitness_delta / (fitness_max - fitness_avg));
    }

    let mut fst_child_genes: BitVec<u64, Lsb0>;
    let mut snd_child_genes: BitVec<u64, Lsb0>;

    if crossover_rate == 1.0f64 || rng.gen::<f64>() <= crossover_rate {
        let crossover_point = rng.gen_range(1..(chromosome_len - 1));

        // Efficient BitVec crossover using slice operations
        fst_child_genes = parents.0.chromosome.genes[..crossover_point].to_bitvec();
        fst_child_genes.extend_from_bitslice(&parents.1.chromosome.genes[crossover_point..]);

        snd_child_genes = parents.1.chromosome.genes[..crossover_point].to_bitvec();
        snd_child_genes.extend_from_bitslice(&parents.0.chromosome.genes[crossover_point..]);
    } else {
        fst_child_genes = parents.0.chromosome.genes.clone();
        snd_child_genes = parents.1.chromosome.genes.clone();
    }

    // Mutation using binomial distribution
    let binomial = Binomial::new(chromosome_len as u64, mutation_rate).unwrap();
    let uniform = Uniform::new(0, chromosome_len);

    let mutated_genes_count_1 = binomial.sample(&mut rng) as usize;
    let mutated_genes_count_2 = binomial.sample(&mut rng) as usize;

    // Apply mutations by flipping bits directly
    for _ in 0..mutated_genes_count_1 {
        let idx = uniform.sample(&mut rng);
        let mut bit = fst_child_genes.get_mut(idx).unwrap();
        *bit = !*bit;
    }
    for _ in 0..mutated_genes_count_2 {
        let idx = uniform.sample(&mut rng);
        let mut bit = snd_child_genes.get_mut(idx).unwrap();
        *bit = !*bit;
    }

    (
        Chromosome::from_bitvec(fst_child_genes),
        Chromosome::from_bitvec(snd_child_genes),
    )
}

#[cfg(test)]
mod evolution_tests {
    use crate::common::*;

    use super::*;

    #[test]
    fn generate_initial_population_test() {
        setup();
        let result = generate_initial_population(100, 50);

        assert_eq!(result.len(), 100);
        assert!(result.iter().all(|c| c.len() == 50))
    }

    #[test]
    fn evolve_test() {
        setup();
        let selection_strategy = SelectionStrategy::Tournament(4);

        let make_chromosome = |bits: &[bool]| Chromosome::from_bool_iter(bits.iter().copied());

        let chromosomes_with_fitness = HashSet::from_iter(vec![
            ChromosomeWithFitness::new(make_chromosome(&[true, true, true, false]), 0),
            ChromosomeWithFitness::new(make_chromosome(&[true, false, false, false]), 10),
            ChromosomeWithFitness::new(make_chromosome(&[false, true, false, false]), 15),
            ChromosomeWithFitness::new(make_chromosome(&[false, false, true, false]), 20),
            ChromosomeWithFitness::new(make_chromosome(&[false, false, false, true]), 25),
            ChromosomeWithFitness::new(make_chromosome(&[true, true, true, true]), 30),
            ChromosomeWithFitness::new(make_chromosome(&[false, false, false, false]), 40),
        ]);

        let result = evolve(&chromosomes_with_fitness, selection_strategy, 0.5);

        debug!("Evo test result: {:?}", result);

        assert_eq!(result.len(), 7);
        assert!(result.contains(&make_chromosome(&[true, true, true, true])));
        assert!(result.contains(&make_chromosome(&[false, false, false, false])));
    }

    #[test]
    fn select_test() {
        setup();
        let selection_strategy = SelectionStrategy::Tournament(5);

        let make_chromosome = |bits: &[bool]| Chromosome::from_bool_iter(bits.iter().copied());

        let chromosomes_with_fitness = HashSet::from_iter(vec![
            ChromosomeWithFitness::new(make_chromosome(&[true, false, false, false]), 10),
            ChromosomeWithFitness::new(make_chromosome(&[false, true, false, false]), 15),
            ChromosomeWithFitness::new(make_chromosome(&[false, false, true, false]), 20),
            ChromosomeWithFitness::new(make_chromosome(&[false, false, false, true]), 25),
            ChromosomeWithFitness::new(make_chromosome(&[true, true, true, true]), 30),
        ]);

        let result = select(&chromosomes_with_fitness, &selection_strategy);

        let results_set: HashSet<Chromosome> =
            HashSet::from_iter(vec![result.0.chromosome, result.1.chromosome]);

        assert!(results_set.contains(&make_chromosome(&[true, true, true, true])));
    }

    #[test]
    fn crossover_test() {
        setup();

        let make_chromosome = |bits: &[bool]| Chromosome::from_bool_iter(bits.iter().copied());

        let parent_1 = ChromosomeWithFitness::new(
            make_chromosome(&[true, true, false, false, true]),
            10,
        );
        let parent_2 = ChromosomeWithFitness::new(
            make_chromosome(&[false, false, true, false, true]),
            20,
        );

        let result = crossover((parent_1, parent_2), 1.0, 1.0, 0.0, 0.0, 15.0, 20.0);

        assert_eq!(result.0.len(), 5);
        assert_eq!(result.1.len(), 5);
    }
}
