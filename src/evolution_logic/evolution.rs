use std::collections::HashSet;
use rand::prelude::*;
use std::cmp::Ordering;
use log::debug;


pub type BinaryChromosome = Vec<bool>;
pub type BinaryChromosomeWithFitness = (BinaryChromosome, f32);

pub enum SelectionStrategy {
    Tournament(usize)
}

pub fn generate_initial_population(initial_population_count: usize, chromosome_size: usize) -> HashSet<BinaryChromosome> {
    debug!("Generating initial population - initial_population_count: {}, chromosome_size: {}", initial_population_count, chromosome_size);

    let mut rng = rand::thread_rng();
    let mut population: HashSet<BinaryChromosome> = HashSet::new();

    for _ in 0..initial_population_count {
        let random_chromosome: BinaryChromosome = (0..chromosome_size).map(|_| rng.gen::<bool>()).collect();

        population.insert(random_chromosome);
    }

    population
}

pub fn evolve(chromosomes_with_fitness: &HashSet<BinaryChromosomeWithFitness>, selection_strategy: SelectionStrategy, mutation_rate: f32, elite_factor: f32) -> HashSet<BinaryChromosome> {
    let mut new_generation: HashSet<BinaryChromosome> = HashSet::new();

    let elite_amount = ((chromosomes_with_fitness.len() as f32) * elite_factor).floor() as usize;

    let mut chromosomes_with_fitness_ordered: Vec<BinaryChromosomeWithFitness> = chromosomes_with_fitness.into_iter().cloned().collect();

    chromosomes_with_fitness_ordered.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    let elite: HashSet<BinaryChromosome> = chromosomes_with_fitness_ordered.iter().rev().take(elite_amount).cloned().map(|cwf| cwf.0).collect();

    new_generation.extend(elite);

    let select_crossover_iteration_amount = (chromosomes_with_fitness.len() - elite_amount) / 2;

    for _ in 0..select_crossover_iteration_amount {
        let parents = select(chromosomes_with_fitness, &selection_strategy);

        let offspring = crossover(parents, 1.0f32, mutation_rate);

        new_generation.insert(offspring.0);
        new_generation.insert(offspring.1);
    }

    new_generation
}

fn select(chromosomes_with_fitness: &HashSet<BinaryChromosomeWithFitness>, selection_strategy: &SelectionStrategy) -> (BinaryChromosome, BinaryChromosome) {
    match *selection_strategy {
        SelectionStrategy::Tournament(tournament_size) => {
            //TODO: If chromosomes.len = 0 OR tournament_size > chromosomes.len -> panic
            let get_winner = |chromosomes_with_fitness: &Vec<BinaryChromosomeWithFitness>|
                chromosomes_with_fitness
                    .iter()
                    .take(tournament_size)
                    .max_by(|&left, &right| left.clone().1.partial_cmp(&right.clone().1).unwrap())
                    .unwrap()
                    .clone();

            let mut chromosomes_vec = Vec::from_iter(chromosomes_with_fitness.clone());

            chromosomes_vec.shuffle(&mut thread_rng());
            let first = get_winner(&chromosomes_vec);
            chromosomes_vec.shuffle(&mut thread_rng());
            let second = get_winner(&chromosomes_vec);

            (first.0, second.0)
        }
    }
}

fn crossover(parents: (BinaryChromosome, BinaryChromosome), _crossover_rate: f32, mutation_rate: f32) -> (BinaryChromosome, BinaryChromosome) {
    let mut rng = rand::thread_rng();

    let crossover_point = rng.gen_range(1..(parents.0.len() - 1));

    let (fst_left, fst_right) = parents.0.split_at(crossover_point);
    let (snd_left, snd_right) = parents.1.split_at(crossover_point);

    let mut fst_child: BinaryChromosome = Vec::new();
    let mut snd_child: BinaryChromosome = Vec::new();

    fst_child.extend(fst_left);
    fst_child.extend(snd_right);

    snd_child.extend(fst_right);
    snd_child.extend(snd_left);

    for idx in 0..fst_child.len() {
        let rnd: f32 = rng.gen();
        if rnd <= mutation_rate {
            fst_child[idx] = !fst_child[idx];
        }
    }

    for idx in 0..snd_child.len() {
        let rnd: f32 = rng.gen();
        if rnd <= mutation_rate {
            snd_child[idx] = !snd_child[idx];
        }
    }

    (fst_child, snd_child)
}

#[cfg(test)]
mod evolution_tests {
    use super::*;

    #[test]
    fn generate_initial_population_test() {
        let result = generate_initial_population(100, 50);

        assert_eq!(result.len(), 100);
        assert!(result.iter().all(|c| c.len() == 50));
    }

    #[test]
    fn evolve_test() {
    }

    #[test]
    fn select_test() {
/*        let selection_strategy = SelectionStrategy::Tournament(5);
        let chromosomes_with_fitness = HashSet::from_iter(vec![
            (vec![true, false, false, false], 10.0),
            (vec![false, true, false, false], 15.0),
            (vec![false, false, true, false], 20.0),
            (vec![false, false, false, true], 25.0),
            (vec![true, true, true, true], 30.0)
        ]);

        let result = select(&chromosomes_with_fitness, &selection_strategy);

        let results_set = HashSet::from_iter(vec![result.0, result.1]);

        assert!(results_set.contains(&vec![true, true, true, true]));
        assert!(results_set.contains(&vec![false, false, false, true]));*/
    }

    #[test]
    fn crossover_test() {
        let parent_1 = vec![true, true, false, false, true];
        let parent_2 = vec![false, false, true, false, true];

        let result = crossover((parent_1, parent_2), 1.0, 0.05);

        assert_eq!(result.0.len(), 5);
        assert_eq!(result.1.len(), 5);
    }
}