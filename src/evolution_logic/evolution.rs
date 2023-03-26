use std::collections::HashSet;
use rand::prelude::*;
use log::debug;
use crate::evolution_logic::chromosome::Chromosome;
use crate::evolution_logic::chromosome_with_fitness::ChromosomeWithFitness;

#[derive(Debug)]
pub enum SelectionStrategy {
    Tournament(usize)
}

pub fn generate_initial_population(initial_population_count: usize, chromosome_size: usize) -> HashSet<Chromosome> {
    debug!("Generating initial population - initial_population_count: {}, chromosome_size: {}", initial_population_count, chromosome_size);

    let mut rng = rand::thread_rng();
    let mut population: HashSet<Chromosome> = HashSet::new();

    for _ in 0..initial_population_count {
        let random_genes = (0..chromosome_size).map(|_| rng.gen::<bool>()).collect();

        let chromosome = Chromosome::from_genes(random_genes);

        population.insert(chromosome);
    }

    population
}

pub fn evolve<T: PartialEq + PartialOrd + Clone>(chromosomes_with_fitness: &HashSet<ChromosomeWithFitness<T>>, selection_strategy: SelectionStrategy, mutation_rate: f32, elite_factor: f32) -> HashSet<Chromosome> {
    debug!("Evolve new generation - chromosomes_with_fitness.len(): {}, selection_strategy: {:?}, mutation_rate: {}, elite_factor: {}", chromosomes_with_fitness.len(), selection_strategy, mutation_rate, elite_factor);
    let mut new_generation: HashSet<Chromosome> = HashSet::new();

    let elite_amount = ((chromosomes_with_fitness.len() as f32) * elite_factor).floor() as usize;

    debug!("Elite amount: {}", elite_amount);

    let mut chromosomes_with_fitness_ordered: Vec<ChromosomeWithFitness<T>> = chromosomes_with_fitness.into_iter().cloned().collect();

    chromosomes_with_fitness_ordered.sort();

    let elite: HashSet<Chromosome> = chromosomes_with_fitness_ordered.iter().rev().take(elite_amount).cloned().map(|cwf| cwf.chromosome).collect();

    new_generation.extend(elite);

    while new_generation.len() < chromosomes_with_fitness.len() {
        let parents = select(chromosomes_with_fitness, &selection_strategy);

        let offspring = crossover(parents, 1.0f32, mutation_rate);

        new_generation.insert(offspring.0);
        new_generation.insert(offspring.1);
    }

    debug!("Total number of chromosomes after crossovers (+ elites retained): {}", new_generation.len());

    new_generation.iter().take(chromosomes_with_fitness.len()).cloned().collect()
}

fn select<T: PartialEq + PartialOrd + Clone>(chromosomes_with_fitness: &HashSet<ChromosomeWithFitness<T>>, selection_strategy: &SelectionStrategy) -> (Chromosome, Chromosome) {
    match *selection_strategy {
        SelectionStrategy::Tournament(tournament_size) => {
            //TODO: If chromosomes.len = 0 OR tournament_size > chromosomes.len -> panic
            let get_winner = |chromosomes_with_fitness: &Vec<ChromosomeWithFitness<T>>|
                chromosomes_with_fitness
                    .iter()
                    .take(tournament_size)
                    .max()
                    .unwrap()
                    .clone();

            let mut chromosomes_vec = Vec::from_iter(chromosomes_with_fitness.clone());

            chromosomes_vec.shuffle(&mut thread_rng());
            let first = get_winner(&chromosomes_vec);
            chromosomes_vec.shuffle(&mut thread_rng());
            chromosomes_vec.retain(|ch| *ch != first);
            let second = get_winner(&chromosomes_vec);

            (first.chromosome, second.chromosome)
        }
    }
}

fn crossover(parents: (Chromosome, Chromosome), _crossover_rate: f32, mutation_rate: f32) -> (Chromosome, Chromosome) {
    let mut rng = rand::thread_rng();

    let crossover_point = rng.gen_range(1..(parents.0.genes.len() - 1));

    let (fst_left, fst_right) = parents.0.genes.split_at(crossover_point);
    let (snd_left, snd_right) = parents.1.genes.split_at(crossover_point);

    let mut fst_child_genes: Vec<bool> = Vec::new();
    let mut snd_child_genes: Vec<bool> = Vec::new();

    fst_child_genes.extend(fst_left);
    fst_child_genes.extend(snd_right);

    snd_child_genes.extend(fst_right);
    snd_child_genes.extend(snd_left);

    for idx in 0..fst_child_genes.len() {
        let rnd: f32 = rng.gen();
        if rnd <= mutation_rate {
            fst_child_genes[idx] = !fst_child_genes[idx];
        }
    }

    for idx in 0..snd_child_genes.len() {
        let rnd: f32 = rng.gen();
        if rnd <= mutation_rate {
            snd_child_genes[idx] = !snd_child_genes[idx];
        }
    }

    (Chromosome::from_genes(fst_child_genes), Chromosome::from_genes(snd_child_genes))
}

#[cfg(test)]
mod evolution_tests {
    use super::*;
    use crate::common::*;

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
            ChromosomeWithFitness::from_chromosome_and_fitness(Chromosome::from_genes(vec![true, true, true, false]), 0),
            ChromosomeWithFitness::from_chromosome_and_fitness(Chromosome::from_genes(vec![true, false, false, false]), 10),
            ChromosomeWithFitness::from_chromosome_and_fitness(Chromosome::from_genes(vec![false, true, false, false]), 15),
            ChromosomeWithFitness::from_chromosome_and_fitness(Chromosome::from_genes(vec![false, false, true, false]), 20),
            ChromosomeWithFitness::from_chromosome_and_fitness(Chromosome::from_genes(vec![false, false, false, true]), 25),
            ChromosomeWithFitness::from_chromosome_and_fitness(Chromosome::from_genes(vec![true, true, true, true]), 30),
            ChromosomeWithFitness::from_chromosome_and_fitness(Chromosome::from_genes(vec![false, false, false, false]), 40)
        ]);

        let result = evolve(&chromosomes_with_fitness, selection_strategy, 0.5, 0.35);

        assert_eq!(result.len(), 7);
        assert!(result.contains(&Chromosome::from_genes(vec![true, true, true, true])));
        assert!(result.contains(&Chromosome::from_genes(vec![false, false, false, false])));
    }

    #[test]
    fn select_test() {
        setup();
        let selection_strategy = SelectionStrategy::Tournament(5);
        let chromosomes_with_fitness = HashSet::from_iter(vec![
            ChromosomeWithFitness::from_chromosome_and_fitness(Chromosome::from_genes(vec![true, false, false, false]), 10),
            ChromosomeWithFitness::from_chromosome_and_fitness(Chromosome::from_genes(vec![false, true, false, false]), 15),
            ChromosomeWithFitness::from_chromosome_and_fitness(Chromosome::from_genes(vec![false, false, true, false]), 20),
            ChromosomeWithFitness::from_chromosome_and_fitness(Chromosome::from_genes(vec![false, false, false, true]), 25),
            ChromosomeWithFitness::from_chromosome_and_fitness(Chromosome::from_genes(vec![true, true, true, true]), 30)
        ]);

        let result = select(&chromosomes_with_fitness, &selection_strategy);

        let results_set: HashSet<Chromosome> = HashSet::from_iter(vec![result.0, result.1]);

        assert!(results_set.contains(&Chromosome::from_genes(vec![true, true, true, true])));
        assert!(results_set.contains(&Chromosome::from_genes(vec![false, false, false, true])));
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