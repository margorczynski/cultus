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
use crate::evolution::direct_encoding::{DirectNetwork, MemoryConfig};
use crate::evolution::local_search::{LocalSearch, LocalSearchConfig};
use crate::evolution::modules::{gate_level_crossover, module_crossover};

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

// ==================== DirectNetwork Evolution ====================

/// Configuration for DirectNetwork evolution.
#[derive(Clone, Debug)]
pub struct DirectEvolutionConfig {
    /// Number of gates for new networks
    pub gate_count: usize,
    /// Input count for networks
    pub input_count: u16,
    /// Output count for networks
    pub output_count: u16,
    /// Optional memory configuration
    pub memory_config: Option<MemoryConfig>,
    /// Elite fraction to preserve
    pub elite_factor: f32,
    /// Tournament size for selection
    pub tournament_size: usize,
    /// Crossover probability
    pub crossover_rate: f64,
    /// Base mutation probability
    pub mutation_rate: f64,
    /// Use module-aware crossover
    pub use_module_crossover: bool,
    /// Apply local search after genetic operations
    pub use_local_search: bool,
    /// Local search configuration
    pub local_search_config: Option<LocalSearchConfig>,
}

impl Default for DirectEvolutionConfig {
    fn default() -> Self {
        DirectEvolutionConfig {
            gate_count: 100,
            input_count: 9,
            output_count: 4,
            memory_config: None,
            elite_factor: 0.1,
            tournament_size: 4,
            crossover_rate: 0.8,
            mutation_rate: 0.3,
            use_module_crossover: true,
            use_local_search: false,
            local_search_config: None,
        }
    }
}

/// DirectNetwork paired with its fitness score.
#[derive(Clone, Debug)]
pub struct DirectNetworkWithFitness {
    pub network: DirectNetwork,
    pub fitness: f64,
}

impl DirectNetworkWithFitness {
    pub fn new(network: DirectNetwork, fitness: f64) -> Self {
        DirectNetworkWithFitness { network, fitness }
    }
}

impl PartialEq for DirectNetworkWithFitness {
    fn eq(&self, other: &Self) -> bool {
        self.fitness == other.fitness
    }
}

impl Eq for DirectNetworkWithFitness {}

impl PartialOrd for DirectNetworkWithFitness {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DirectNetworkWithFitness {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.fitness
            .partial_cmp(&other.fitness)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Generate an initial population of random DirectNetworks.
pub fn generate_initial_direct_population(
    population_size: usize,
    config: &DirectEvolutionConfig,
) -> Vec<DirectNetwork> {
    (0..population_size)
        .into_par_iter()
        .map(|_| {
            let mut rng = thread_rng();
            DirectNetwork::random(
                config.input_count,
                config.output_count,
                config.gate_count,
                config.memory_config.clone(),
                &mut rng,
            )
        })
        .collect()
}

/// Evolve a population of DirectNetworks.
///
/// This is the main evolution function for the new direct encoding system.
/// It uses structural mutations and module-aware crossover.
pub fn evolve_direct<F>(
    population_with_fitness: &[DirectNetworkWithFitness],
    config: &DirectEvolutionConfig,
    evaluate: F,
) -> Vec<DirectNetwork>
where
    F: Fn(&DirectNetwork) -> f64 + Sync,
{
    let pop_size = population_with_fitness.len();
    if pop_size == 0 {
        return Vec::new();
    }

    // Sort by fitness (descending)
    let mut sorted: Vec<_> = population_with_fitness.to_vec();
    sorted.sort_by(|a, b| b.cmp(a));

    // Calculate elite count
    let elite_count = ((pop_size as f32) * config.elite_factor).ceil() as usize;
    let elite_count = elite_count.max(1).min(pop_size);

    debug!("Direct evolution: pop_size={}, elite_count={}", pop_size, elite_count);

    // Keep elites
    let mut new_population: Vec<DirectNetwork> = sorted
        .iter()
        .take(elite_count)
        .map(|nwf| nwf.network.clone())
        .collect();

    // Calculate fitness statistics for adaptive rates
    let fitness_max = sorted.first().map(|nwf| nwf.fitness).unwrap_or(0.0);
    let fitness_avg: f64 = sorted.iter().map(|nwf| nwf.fitness).sum::<f64>() / pop_size as f64;

    // Generate offspring
    let offspring_needed = pop_size - elite_count;
    let offspring: Vec<DirectNetwork> = (0..offspring_needed)
        .into_par_iter()
        .map(|_| {
            let mut rng = thread_rng();

            // Select parents via tournament
            let parent1 = select_direct_tournament(&sorted, config.tournament_size, &mut rng);
            let parent2 = select_direct_tournament(&sorted, config.tournament_size, &mut rng);

            // Determine adaptive crossover rate
            let parent_max_fitness = parent1.fitness.max(parent2.fitness);
            let crossover_rate = if parent_max_fitness >= fitness_avg {
                config.crossover_rate
            } else if fitness_max > fitness_avg {
                config.crossover_rate * 0.5
                    + 0.5
                        * config.crossover_rate
                        * ((fitness_max - parent_max_fitness) / (fitness_max - fitness_avg))
            } else {
                config.crossover_rate
            };

            // Crossover
            let mut offspring = if rng.gen::<f64>() < crossover_rate {
                crossover_direct(&parent1.network, &parent2.network, config, &mut rng)
            } else {
                // No crossover - clone fitter parent
                if parent1.fitness >= parent2.fitness {
                    parent1.network.clone()
                } else {
                    parent2.network.clone()
                }
            };

            // Determine adaptive mutation rate
            let mutation_rate = if parent_max_fitness < fitness_avg {
                config.mutation_rate * 1.5 // Increase mutation for below-average individuals
            } else if fitness_max > fitness_avg {
                config.mutation_rate
                    * (1.0
                        + 0.5 * ((fitness_max - parent_max_fitness) / (fitness_max - fitness_avg)))
            } else {
                config.mutation_rate
            };

            // Mutate
            mutate_direct(&mut offspring, mutation_rate, &mut rng);

            // Ensure validity
            if !offspring.validate() {
                offspring.repair(&mut rng);
            }

            // Optional local search
            if config.use_local_search {
                if let Some(ref ls_config) = config.local_search_config {
                    let ls = LocalSearch::new(ls_config.clone());
                    let _ = ls.improve(&mut offspring, &evaluate, &mut rng);
                }
            }

            offspring
        })
        .collect();

    new_population.extend(offspring);

    debug!(
        "Direct evolution complete: new_population_size={}",
        new_population.len()
    );

    new_population
}

/// Tournament selection for DirectNetwork.
fn select_direct_tournament(
    population: &[DirectNetworkWithFitness],
    tournament_size: usize,
    rng: &mut impl Rng,
) -> DirectNetworkWithFitness {
    population
        .iter()
        .choose_multiple(rng, tournament_size)
        .into_iter()
        .max()
        .cloned()
        .unwrap_or_else(|| population[0].clone())
}

/// Crossover two DirectNetworks.
fn crossover_direct(
    parent1: &DirectNetwork,
    parent2: &DirectNetwork,
    config: &DirectEvolutionConfig,
    rng: &mut impl Rng,
) -> DirectNetwork {
    let (child_a, child_b) = if config.use_module_crossover && rng.gen::<f64>() < 0.5 {
        // Module-aware crossover
        module_crossover(parent1, parent2, rng)
    } else {
        // Gate-level crossover
        gate_level_crossover(parent1, parent2, rng)
    };

    // Randomly select one of the two children
    if rng.gen::<bool>() {
        child_a
    } else {
        child_b
    }
}

/// Apply mutations to a DirectNetwork.
fn mutate_direct(network: &mut DirectNetwork, mutation_rate: f64, rng: &mut impl Rng) {
    // Each mutation type has independent probability based on mutation_rate
    let mutation_scale = mutation_rate;

    // Structural mutations (rarer)
    if rng.gen::<f64>() < mutation_scale * 0.1 {
        network.mutate_add_gate(rng);
    }
    if rng.gen::<f64>() < mutation_scale * 0.05 {
        network.mutate_remove_gate(rng);
    }
    if rng.gen::<f64>() < mutation_scale * 0.02 {
        network.mutate_duplicate_subgraph(rng);
    }

    // Connection mutations (more common)
    if rng.gen::<f64>() < mutation_scale * 0.3 {
        network.mutate_swap_connection(rng);
    }
    if rng.gen::<f64>() < mutation_scale * 0.2 {
        network.mutate_gate_type(rng);
    }
    if rng.gen::<f64>() < mutation_scale * 0.15 {
        network.mutate_output(rng);
    }
}

/// Calculate structural distance between two DirectNetworks.
/// Used for fitness sharing with direct encoding.
pub fn direct_network_distance(n1: &DirectNetwork, n2: &DirectNetwork) -> f64 {
    let mut distance = 0.0;

    // Gate count difference
    let gate_diff = (n1.gates.len() as i32 - n2.gates.len() as i32).abs();
    distance += gate_diff as f64 * 0.5;

    // Gate type distribution difference
    let mut type_counts1 = [0u32; 8]; // Assuming 8 gate types
    let mut type_counts2 = [0u32; 8];

    for gate in &n1.gates {
        let idx = gate.gate_type as usize % 8;
        type_counts1[idx] += 1;
    }
    for gate in &n2.gates {
        let idx = gate.gate_type as usize % 8;
        type_counts2[idx] += 1;
    }

    let type_dist: f64 = type_counts1
        .iter()
        .zip(type_counts2.iter())
        .map(|(a, b)| (*a as i32 - *b as i32).abs() as f64)
        .sum();
    distance += type_dist * 0.2;

    // Output connectivity difference
    let min_outputs = n1.outputs.len().min(n2.outputs.len());
    let mut output_diff = 0;
    for i in 0..min_outputs {
        if n1.outputs[i] != n2.outputs[i] {
            output_diff += 1;
        }
    }
    output_diff += (n1.outputs.len() as i32 - n2.outputs.len() as i32).abs() as usize;
    distance += output_diff as f64;

    distance
}

/// Apply fitness sharing to DirectNetwork population.
pub fn apply_direct_fitness_sharing(
    population: &[DirectNetworkWithFitness],
    sigma_share: f64,
    alpha: f64,
) -> Vec<DirectNetworkWithFitness> {
    let n = population.len();

    population
        .par_iter()
        .map(|nwf| {
            // Calculate niche count
            let niche_count: f64 = population
                .iter()
                .map(|other| {
                    let distance = direct_network_distance(&nwf.network, &other.network);
                    if distance < sigma_share {
                        1.0 - (distance / sigma_share).powf(alpha)
                    } else {
                        0.0
                    }
                })
                .sum();

            let shared_fitness = if niche_count > 0.0 {
                nwf.fitness / niche_count
            } else {
                nwf.fitness
            };

            DirectNetworkWithFitness::new(nwf.network.clone(), shared_fitness)
        })
        .collect()
}

/// Evaluate a population of DirectNetworks in parallel.
pub fn evaluate_direct_population<F>(
    population: &[DirectNetwork],
    evaluate: F,
) -> Vec<DirectNetworkWithFitness>
where
    F: Fn(&DirectNetwork) -> f64 + Sync,
{
    population
        .par_iter()
        .map(|network| {
            let fitness = evaluate(network);
            DirectNetworkWithFitness::new(network.clone(), fitness)
        })
        .collect()
}

/// Statistics about a DirectNetwork population.
#[derive(Clone, Debug)]
pub struct DirectPopulationStats {
    pub min_fitness: f64,
    pub max_fitness: f64,
    pub avg_fitness: f64,
    pub median_fitness: f64,
    pub std_dev: f64,
    pub avg_gate_count: f64,
    pub diversity: f64,
}

impl DirectPopulationStats {
    /// Compute statistics for a population.
    pub fn compute(population: &[DirectNetworkWithFitness]) -> Self {
        if population.is_empty() {
            return DirectPopulationStats {
                min_fitness: 0.0,
                max_fitness: 0.0,
                avg_fitness: 0.0,
                median_fitness: 0.0,
                std_dev: 0.0,
                avg_gate_count: 0.0,
                diversity: 0.0,
            };
        }

        let mut fitnesses: Vec<f64> = population.iter().map(|nwf| nwf.fitness).collect();
        fitnesses.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = fitnesses.len();
        let min_fitness = fitnesses[0];
        let max_fitness = fitnesses[n - 1];
        let avg_fitness: f64 = fitnesses.iter().sum::<f64>() / n as f64;
        let median_fitness = fitnesses[n / 2];

        let variance: f64 = fitnesses
            .iter()
            .map(|f| (f - avg_fitness).powi(2))
            .sum::<f64>()
            / n as f64;
        let std_dev = variance.sqrt();

        let avg_gate_count: f64 =
            population.iter().map(|nwf| nwf.network.gates.len()).sum::<usize>() as f64 / n as f64;

        // Diversity: average pairwise distance (sampled for large populations)
        let sample_size = 50.min(n);
        let mut rng = thread_rng();
        let sample: Vec<_> = population.iter().choose_multiple(&mut rng, sample_size);

        let mut total_distance = 0.0;
        let mut pairs = 0;
        for i in 0..sample.len() {
            for j in (i + 1)..sample.len() {
                total_distance += direct_network_distance(&sample[i].network, &sample[j].network);
                pairs += 1;
            }
        }
        let diversity = if pairs > 0 {
            total_distance / pairs as f64
        } else {
            0.0
        };

        DirectPopulationStats {
            min_fitness,
            max_fitness,
            avg_fitness,
            median_fitness,
            std_dev,
            avg_gate_count,
            diversity,
        }
    }
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

    // ==================== DirectNetwork Evolution Tests ====================

    #[test]
    fn generate_initial_direct_population_test() {
        setup();

        let config = DirectEvolutionConfig {
            gate_count: 20,
            input_count: 4,
            output_count: 2,
            ..Default::default()
        };

        let population = generate_initial_direct_population(50, &config);

        assert_eq!(population.len(), 50);
        for network in &population {
            assert!(network.validate());
            assert_eq!(network.input_count, 4);
            assert_eq!(network.output_count, 2);
        }
    }

    #[test]
    fn direct_network_with_fitness_ordering_test() {
        setup();

        let mut rng = rand::thread_rng();
        let net1 = DirectNetwork::random(4, 2, 10, None, &mut rng);
        let net2 = DirectNetwork::random(4, 2, 10, None, &mut rng);

        let nwf1 = DirectNetworkWithFitness::new(net1, 10.0);
        let nwf2 = DirectNetworkWithFitness::new(net2, 20.0);

        assert!(nwf2 > nwf1);
        assert!(nwf1 < nwf2);
    }

    #[test]
    fn evolve_direct_test() {
        setup();

        let config = DirectEvolutionConfig {
            gate_count: 15,
            input_count: 4,
            output_count: 2,
            elite_factor: 0.2,
            tournament_size: 3,
            crossover_rate: 0.8,
            mutation_rate: 0.3,
            use_module_crossover: true,
            use_local_search: false,
            ..Default::default()
        };

        let population = generate_initial_direct_population(20, &config);

        // Simple fitness function: count gates
        let evaluate = |net: &DirectNetwork| net.gates.len() as f64;

        let pop_with_fitness: Vec<_> = population
            .iter()
            .map(|net| DirectNetworkWithFitness::new(net.clone(), evaluate(net)))
            .collect();

        let new_population = evolve_direct(&pop_with_fitness, &config, evaluate);

        assert_eq!(new_population.len(), 20);
        for network in &new_population {
            assert!(network.validate());
        }
    }

    #[test]
    fn direct_network_distance_test() {
        setup();

        let mut rng = rand::thread_rng();
        let net1 = DirectNetwork::random(4, 2, 10, None, &mut rng);
        let net2 = DirectNetwork::random(4, 2, 50, None, &mut rng);

        let self_distance = direct_network_distance(&net1, &net1);
        let different_distance = direct_network_distance(&net1, &net2);

        // Same network should have distance 0
        assert_eq!(self_distance, 0.0);

        // Different networks should have positive distance
        assert!(different_distance > 0.0);
    }

    #[test]
    fn direct_population_stats_test() {
        setup();

        let mut rng = rand::thread_rng();
        let networks: Vec<_> = (0..10)
            .map(|i| {
                let net = DirectNetwork::random(4, 2, 10 + i * 2, None, &mut rng);
                DirectNetworkWithFitness::new(net, (i as f64) * 10.0)
            })
            .collect();

        let stats = DirectPopulationStats::compute(&networks);

        assert_eq!(stats.min_fitness, 0.0);
        assert_eq!(stats.max_fitness, 90.0);
        assert!(stats.avg_fitness > 0.0);
        assert!(stats.std_dev > 0.0);
    }

    #[test]
    fn mutate_direct_test() {
        setup();

        let mut rng = rand::thread_rng();
        let mut network = DirectNetwork::random(4, 2, 20, None, &mut rng);

        // Apply mutations multiple times
        for _ in 0..10 {
            mutate_direct(&mut network, 1.0, &mut rng);
            if !network.validate() {
                network.repair(&mut rng);
            }
            assert!(network.validate());
        }
    }
}
