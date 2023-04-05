use std::collections::HashSet;
use std::fmt::Display;

use log::{debug, info};
use rand::prelude::*;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::time::{Duration, Instant};
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

    let mut rng = StdRng::from_entropy();
    let mut population: HashSet<Chromosome> = HashSet::new();

    //TODO: Refactor this
    let res = (0..initial_population_count).into_par_iter().map(|_| {
        let mut rng_clone = rng.clone();
        let random_genes= (0..chromosome_size).map(|_| rng_clone.gen::<bool>()).collect();

        Chromosome::from_genes(random_genes)
    });

    population.par_extend(res);

    while population.len() < initial_population_count {
        let random_genes = (0..chromosome_size).map(|_| rng.gen::<bool>()).collect();

        let chromosome = Chromosome::from_genes(random_genes);

        population.insert(chromosome);
    }

    population
}

pub fn evolve<T: PartialEq + PartialOrd + Clone + Eq + Send>(
    chromosomes_with_fitness: &HashSet<ChromosomeWithFitness<T>>,
    selection_strategy: SelectionStrategy,
    mutation_rate: f32,
    elite_factor: f32,
) -> HashSet<Chromosome> {
    debug!("Evolve new generation - chromosomes_with_fitness.len(): {}, selection_strategy: {:?}, mutation_rate: {}, elite_factor: {}", chromosomes_with_fitness.len(), selection_strategy, mutation_rate, elite_factor);
    let mut new_generation: HashSet<Chromosome> = HashSet::new();

    let elite_amount = ((chromosomes_with_fitness.len() as f32) * elite_factor).floor() as usize;

    debug!("Elite amount: {}", elite_amount);

    let mut chromosomes_with_fitness_ordered: Vec<ChromosomeWithFitness<T>> =
        chromosomes_with_fitness.into_iter().cloned().collect();

    chromosomes_with_fitness_ordered.sort_unstable();

    let elite = chromosomes_with_fitness_ordered
        .par_iter()
        .rev()
        .take(elite_amount)
        .cloned()
        .map(|cwf| cwf.chromosome);

    new_generation.par_extend(elite);

    let offspring = (0..((chromosomes_with_fitness.len() - new_generation.len()) / 2))
        .into_par_iter()
        .map(|_| {
            let parents = select(chromosomes_with_fitness, &selection_strategy);
            let (offspring_1, offspring_2) = crossover(parents, 1.0f32, mutation_rate);
            vec![offspring_1, offspring_2]
        }).flatten();

    new_generation.par_extend(offspring);

    //Below is a fallback if the above would generate duplicates of already existing chromosomes
    //TODO: Use Vec instead and allow duplicates?
    while new_generation.len() != chromosomes_with_fitness.len() {
        let parents = select(chromosomes_with_fitness, &selection_strategy);
        let offspring = crossover(parents, 1.0f32, mutation_rate);

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

fn select<T: PartialEq + PartialOrd + Clone + Eq + Send>(
    chromosomes_with_fitness: &HashSet<ChromosomeWithFitness<T>>,
    selection_strategy: &SelectionStrategy,
) -> (Chromosome, Chromosome) {
    match *selection_strategy {
        SelectionStrategy::Tournament(tournament_size) => {
            let mut rng = thread_rng();
            //TODO: If chromosomes.len = 0 OR tournament_size > chromosomes.len -> panic
            let mut get_winner = |cwf: &HashSet<ChromosomeWithFitness<T>>| {
                cwf.iter().choose_multiple(&mut rng, tournament_size).into_iter().max().unwrap().clone()
            };

            let first = get_winner(&chromosomes_with_fitness);
            let second= get_winner(&chromosomes_with_fitness);

            (first.chromosome, second.chromosome)
        }
    }
}

fn crossover(
    parents: (Chromosome, Chromosome),
    _crossover_rate: f32,
    mutation_rate: f32,
) -> (Chromosome, Chromosome) {
    let mut rng = thread_rng();

    let chromosome_len = parents.0.genes.len();

    let crossover_point = rng.gen_range(1..(chromosome_len - 1));

    let (fst_left, fst_right) = parents.0.genes.split_at(crossover_point);
    let (snd_left, snd_right) = parents.1.genes.split_at(crossover_point);

    let mut fst_child_genes: Vec<bool> = Vec::new();
    let mut snd_child_genes: Vec<bool> = Vec::new();

    fst_child_genes.extend(fst_left);
    fst_child_genes.extend(snd_right);

    snd_child_genes.extend(fst_right);
    snd_child_genes.extend(snd_left);

    let mutated_genes_count_1 = (rng.gen_range(0..chromosome_len) as f32 * mutation_rate).ceil() as usize;
    let mutated_genes_count_2 = (rng.gen_range(0..chromosome_len) as f32 * mutation_rate).ceil() as usize;

    if mutated_genes_count_1 > 0 {
        let uniform_1 = Uniform::new(0, mutated_genes_count_1);
        for _ in 0..mutated_genes_count_1 {
            let fst_random_idx = rng.sample(uniform_1);
            fst_child_genes[fst_random_idx] = !fst_child_genes[fst_random_idx];
        }
    }

    if mutated_genes_count_2 > 0 {
        let uniform_2 = Uniform::new(0, mutated_genes_count_2);
        for _ in 0..mutated_genes_count_2 {
            let snd_random_idx = rng.sample(uniform_2);
            snd_child_genes[snd_random_idx] = !snd_child_genes[snd_random_idx];
        }
    }

    (
        Chromosome::from_genes(fst_child_genes),
        Chromosome::from_genes(snd_child_genes),
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
        assert_eq!(result.iter().all(|c| c.genes.len() == 50), true)
    }

    #[test]
    fn evolve_test() {
        setup();
        let selection_strategy = SelectionStrategy::Tournament(4);
        let chromosomes_with_fitness = HashSet::from_iter(vec![
            ChromosomeWithFitness::from_chromosome_and_fitness(
                Chromosome::from_genes(vec![true, true, true, false]),
                0,
            ),
            ChromosomeWithFitness::from_chromosome_and_fitness(
                Chromosome::from_genes(vec![true, false, false, false]),
                10,
            ),
            ChromosomeWithFitness::from_chromosome_and_fitness(
                Chromosome::from_genes(vec![false, true, false, false]),
                15,
            ),
            ChromosomeWithFitness::from_chromosome_and_fitness(
                Chromosome::from_genes(vec![false, false, true, false]),
                20,
            ),
            ChromosomeWithFitness::from_chromosome_and_fitness(
                Chromosome::from_genes(vec![false, false, false, true]),
                25,
            ),
            ChromosomeWithFitness::from_chromosome_and_fitness(
                Chromosome::from_genes(vec![true, true, true, true]),
                30,
            ),
            ChromosomeWithFitness::from_chromosome_and_fitness(
                Chromosome::from_genes(vec![false, false, false, false]),
                40,
            ),
        ]);

        let result = evolve(&chromosomes_with_fitness, selection_strategy, 0.5, 0.35);

        debug!("Evo test result: {:?}", result);

        assert_eq!(result.len(), 7);
        assert!(result.contains(&Chromosome::from_genes(vec![true, true, true, true])));
        assert!(result.contains(&Chromosome::from_genes(vec![false, false, false, false])));
    }

    #[test]
    fn select_test() {
        setup();
        let selection_strategy = SelectionStrategy::Tournament(5);
        let chromosomes_with_fitness = HashSet::from_iter(vec![
            ChromosomeWithFitness::from_chromosome_and_fitness(
                Chromosome::from_genes(vec![true, false, false, false]),
                10,
            ),
            ChromosomeWithFitness::from_chromosome_and_fitness(
                Chromosome::from_genes(vec![false, true, false, false]),
                15,
            ),
            ChromosomeWithFitness::from_chromosome_and_fitness(
                Chromosome::from_genes(vec![false, false, true, false]),
                20,
            ),
            ChromosomeWithFitness::from_chromosome_and_fitness(
                Chromosome::from_genes(vec![false, false, false, true]),
                25,
            ),
            ChromosomeWithFitness::from_chromosome_and_fitness(
                Chromosome::from_genes(vec![true, true, true, true]),
                30,
            ),
        ]);

        let result = select(&chromosomes_with_fitness, &selection_strategy);

        let results_set: HashSet<Chromosome> = HashSet::from_iter(vec![result.0, result.1]);

        assert!(results_set.contains(&Chromosome::from_genes(vec![true, true, true, true])));
        assert!(results_set.contains(&Chromosome::from_genes(vec![true, true, true, true])));
    }

    #[test]
    fn crossover_test() {
        setup();
        let parent_1 = Chromosome::from_genes(vec![true, true, false, false, true]);
        let parent_2 = Chromosome::from_genes(vec![false, false, true, false, true]);

        let result = crossover((parent_1, parent_2), 1.0, 0.05);

        assert_eq!(result.0.genes.len(), 5);
        assert_eq!(result.1.genes.len(), 5);
    }
}
