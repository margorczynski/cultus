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

#[derive(Debug, Clone, Copy)]
pub enum SelectionStrategy {
    Tournament(usize),
}

#[derive(Debug, Clone, Copy)]
pub enum CrossoverStrategy {
    SinglePoint,
    TwoPoint,
    Uniform,
}

/// Calculate Hamming distance between two chromosomes (number of differing bits)
fn hamming_distance(c1: &Chromosome, c2: &Chromosome) -> usize {
    c1.genes
        .iter()
        .zip(c2.genes.iter())
        .filter(|(a, b)| *a != *b)
        .count()
}

/// Calculate sharing value between two chromosomes
/// Returns 1.0 if distance is 0, decreases to 0.0 at threshold distance
fn sharing_function(distance: usize, sigma_share: f64, alpha: f64) -> f64 {
    let d = distance as f64;
    if d < sigma_share {
        1.0 - (d / sigma_share).powf(alpha)
    } else {
        0.0
    }
}

/// Apply fitness sharing to maintain population diversity
/// Reduces fitness of similar individuals to prevent premature convergence
pub fn apply_fitness_sharing<T: Clone + Into<f64> + Send + Sync>(
    chromosomes_with_fitness: &[(Chromosome, T)],
    sigma_share: f64,
    alpha: f64,
) -> Vec<(Chromosome, f64)> {
    let n = chromosomes_with_fitness.len();

    chromosomes_with_fitness
        .par_iter()
        .enumerate()
        .map(|(_, (c1, fitness))| {
            // Calculate niche count (sum of sharing values with all other individuals)
            let niche_count: f64 = (0..n)
                .map(|j| {
                    let distance = hamming_distance(c1, &chromosomes_with_fitness[j].0);
                    sharing_function(distance, sigma_share, alpha)
                })
                .sum();

            // Shared fitness = original fitness / niche count
            let original_fitness: f64 = fitness.clone().into();
            let shared_fitness = if niche_count > 0.0 {
                original_fitness / niche_count
            } else {
                original_fitness
            };

            (c1.clone(), shared_fitness)
        })
        .collect()
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
    crossover_strategy: CrossoverStrategy,
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
            let (offspring_1, offspring_2) = crossover(
                parents,
                crossover_strategy,
                0.7,
                1.0,
                0.05,
                0.5,
                fitness_avg,
                fitness_max,
            );
            vec![offspring_1, offspring_2]
        })
        .flatten();

    new_generation.par_extend(offspring);

    // Fallback for duplicates
    while new_generation.len() != chromosomes_with_fitness.len() {
        let parents = select(chromosomes_with_fitness, &selection_strategy);
        let offspring = crossover(
            parents,
            crossover_strategy,
            0.7,
            1.0,
            0.05,
            0.5,
            fitness_avg,
            fitness_max,
        );

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
    crossover_strategy: CrossoverStrategy,
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
        match crossover_strategy {
            CrossoverStrategy::SinglePoint => {
                let crossover_point = rng.gen_range(1..(chromosome_len - 1));

                fst_child_genes = parents.0.chromosome.genes[..crossover_point].to_bitvec();
                fst_child_genes
                    .extend_from_bitslice(&parents.1.chromosome.genes[crossover_point..]);

                snd_child_genes = parents.1.chromosome.genes[..crossover_point].to_bitvec();
                snd_child_genes
                    .extend_from_bitslice(&parents.0.chromosome.genes[crossover_point..]);
            }
            CrossoverStrategy::TwoPoint => {
                let point1 = rng.gen_range(1..(chromosome_len - 2));
                let point2 = rng.gen_range((point1 + 1)..(chromosome_len - 1));

                // Child 1: P1[0..p1] + P2[p1..p2] + P1[p2..]
                fst_child_genes = parents.0.chromosome.genes[..point1].to_bitvec();
                fst_child_genes
                    .extend_from_bitslice(&parents.1.chromosome.genes[point1..point2]);
                fst_child_genes.extend_from_bitslice(&parents.0.chromosome.genes[point2..]);

                // Child 2: P2[0..p1] + P1[p1..p2] + P2[p2..]
                snd_child_genes = parents.1.chromosome.genes[..point1].to_bitvec();
                snd_child_genes
                    .extend_from_bitslice(&parents.0.chromosome.genes[point1..point2]);
                snd_child_genes.extend_from_bitslice(&parents.1.chromosome.genes[point2..]);
            }
            CrossoverStrategy::Uniform => {
                // Each bit is independently chosen from either parent with 50% probability
                fst_child_genes = BitVec::with_capacity(chromosome_len);
                snd_child_genes = BitVec::with_capacity(chromosome_len);

                for i in 0..chromosome_len {
                    if rng.gen::<bool>() {
                        fst_child_genes.push(parents.0.chromosome.genes[i]);
                        snd_child_genes.push(parents.1.chromosome.genes[i]);
                    } else {
                        fst_child_genes.push(parents.1.chromosome.genes[i]);
                        snd_child_genes.push(parents.0.chromosome.genes[i]);
                    }
                }
            }
        }
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

        let result = evolve(
            &chromosomes_with_fitness,
            selection_strategy,
            CrossoverStrategy::SinglePoint,
            0.5,
        );

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

        let result = crossover(
            (parent_1, parent_2),
            CrossoverStrategy::SinglePoint,
            1.0,
            1.0,
            0.0,
            0.0,
            15.0,
            20.0,
        );

        assert_eq!(result.0.len(), 5);
        assert_eq!(result.1.len(), 5);
    }

    #[test]
    fn hamming_distance_test() {
        setup();

        let make_chromosome = |bits: &[bool]| Chromosome::from_bool_iter(bits.iter().copied());

        let c1 = make_chromosome(&[true, true, true, true]);
        let c2 = make_chromosome(&[false, false, false, false]);
        let c3 = make_chromosome(&[true, false, true, false]);

        assert_eq!(hamming_distance(&c1, &c1), 0);
        assert_eq!(hamming_distance(&c1, &c2), 4);
        assert_eq!(hamming_distance(&c1, &c3), 2);
        assert_eq!(hamming_distance(&c2, &c3), 2);
    }

    #[test]
    fn sharing_function_test() {
        setup();

        // Distance 0 -> sharing = 1.0
        assert!((sharing_function(0, 10.0, 1.0) - 1.0).abs() < 0.001);

        // Distance at threshold -> sharing = 0.0
        assert!((sharing_function(10, 10.0, 1.0) - 0.0).abs() < 0.001);

        // Distance beyond threshold -> sharing = 0.0
        assert_eq!(sharing_function(15, 10.0, 1.0), 0.0);

        // Distance in between -> sharing between 0 and 1
        let mid_share = sharing_function(5, 10.0, 1.0);
        assert!(mid_share > 0.0 && mid_share < 1.0);
    }

    #[test]
    fn fitness_sharing_test() {
        setup();

        let make_chromosome = |bits: &[bool]| Chromosome::from_bool_iter(bits.iter().copied());

        // Two identical chromosomes should have their fitness halved
        let identical: Vec<(Chromosome, f64)> = vec![
            (make_chromosome(&[true, true, true, true]), 100.0),
            (make_chromosome(&[true, true, true, true]), 100.0),
        ];

        let shared = apply_fitness_sharing(&identical, 10.0, 1.0);

        // Each identical individual shares with itself (1.0) and the other (1.0)
        // so niche count = 2.0, shared fitness = 100/2 = 50
        assert!((shared[0].1 - 50.0).abs() < 0.1);
        assert!((shared[1].1 - 50.0).abs() < 0.1);

        // Two very different chromosomes should keep most of their fitness
        let different: Vec<(Chromosome, f64)> = vec![
            (make_chromosome(&[true, true, true, true]), 100.0),
            (make_chromosome(&[false, false, false, false]), 100.0),
        ];

        let shared_diff = apply_fitness_sharing(&different, 2.0, 1.0);

        // Distance is 4, threshold is 2, so no sharing between them
        // Each only shares with itself: niche count = 1.0
        assert!((shared_diff[0].1 - 100.0).abs() < 0.1);
        assert!((shared_diff[1].1 - 100.0).abs() < 0.1);
    }

    #[test]
    fn two_point_crossover_test() {
        setup();

        let make_chromosome = |bits: &[bool]| Chromosome::from_bool_iter(bits.iter().copied());

        let parent_1 = ChromosomeWithFitness::new(
            make_chromosome(&[true, true, true, true, true, true]),
            10,
        );
        let parent_2 = ChromosomeWithFitness::new(
            make_chromosome(&[false, false, false, false, false, false]),
            20,
        );

        let result = crossover(
            (parent_1, parent_2),
            CrossoverStrategy::TwoPoint,
            1.0,
            1.0,
            0.0,
            0.0,
            15.0,
            20.0,
        );

        assert_eq!(result.0.len(), 6);
        assert_eq!(result.1.len(), 6);

        // Children should be different from each other (with high probability)
        // and should contain bits from both parents
    }

    #[test]
    fn uniform_crossover_test() {
        setup();

        let make_chromosome = |bits: &[bool]| Chromosome::from_bool_iter(bits.iter().copied());

        let parent_1 = ChromosomeWithFitness::new(
            make_chromosome(&[true, true, true, true, true, true]),
            10,
        );
        let parent_2 = ChromosomeWithFitness::new(
            make_chromosome(&[false, false, false, false, false, false]),
            20,
        );

        let result = crossover(
            (parent_1, parent_2),
            CrossoverStrategy::Uniform,
            1.0,
            1.0,
            0.0,
            0.0,
            15.0,
            20.0,
        );

        assert_eq!(result.0.len(), 6);
        assert_eq!(result.1.len(), 6);

        // For uniform crossover with complementary parents, children should be complementary
        for i in 0..6 {
            assert_ne!(result.0.genes[i], result.1.genes[i]);
        }
    }
}
