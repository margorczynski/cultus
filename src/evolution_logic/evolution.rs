use std::collections::HashSet;
use rand::prelude::*;


pub type BinaryChromosome = Vec<bool>;
pub type BinaryChromosomeWithFitness = (BinaryChromosome, f32);

pub enum SelectionStrategy {
    Tournament(usize)
}

pub fn generate_initial_population(initial_population_count: usize, chromosome_size: usize) -> HashSet<BinaryChromosome> {
    let mut rng = rand::thread_rng();

    let mut population: HashSet<BinaryChromosome> = HashSet::new();

    for i in 0..initial_population_count {
        let random_chromosome: BinaryChromosome = (0..chromosome_size).map(|_| rng.gen::<bool>()).collect();

        population.insert(random_chromosome);
    }

    population
}

pub fn evolve(chromosomes_with_fitness: &HashSet<BinaryChromosomeWithFitness>, generations_count_limit: Option<usize>, selection_strategy: SelectionStrategy) -> HashSet<BinaryChromosome> {
    let mut generation_count: usize = 0;
    let mut new_generation: HashSet<BinaryChromosome> = HashSet::new();
    loop {

        //TODO: Implement - selection, crossover

        generation_count = generation_count + 1;

        match generations_count_limit {
            None => {}
            Some(limit) => if generation_count >= limit {
                break
            }
        }
    }

    new_generation
}

fn select(chromosomes_with_fitness: &HashSet<BinaryChromosomeWithFitness>, selection_strategy: SelectionStrategy) -> (BinaryChromosome, BinaryChromosome) {
    match selection_strategy {
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

            //TODO: Doesn't guarantee that the same element won't be selected twice
            let first = get_winner(&chromosomes_vec);
            chromosomes_vec.shuffle(&mut thread_rng());
            let second = get_winner(&chromosomes_vec);

            (first.0, second.0)
        }
    }
}

fn crossover(parents: (BinaryChromosome, BinaryChromosome), crossover_rate: f32, mutation_rate: f32) -> (BinaryChromosome, BinaryChromosome) {
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