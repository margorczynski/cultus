use std::collections::HashSet;

use log::info;
use rand::prelude::*;
use rayon::prelude::*;

use common::setup;
use config::cultus_config::CultusConfig;
use evolution::chromosome::Chromosome;
use evolution::chromosome_with_fitness::ChromosomeWithFitness;
use evolution::curriculum::CurriculumManager;
use evolution::direct_encoding::{DirectNetwork, MemoryConfig};
use evolution::evolution::{
    evolve, evolve_direct, generate_initial_direct_population,
    generate_initial_population, CrossoverStrategy, DirectEvolutionConfig,
    DirectNetworkWithFitness, DirectPopulationStats, SelectionStrategy,
};
use evolution::local_search::{LocalSearchConfig, LocalSearchStrategy};
use evolution::novelty::{
    BehavioralSignature, GameTrace, MilestoneTracker, NoveltyArchive, NoveltyConfig, NoveltyFitness,
};
use game::level::Level;
use smart_network::smart_network::SmartNetwork;
use smart_network_game_adapter::{play_game_with_network, play_game_with_network_traced, GameResult};

mod common;
mod config;
mod evolution;
mod game;
mod smart_network;
mod smart_network_game_adapter;

/// Multi-objective fitness result for a chromosome
#[derive(Debug, Clone)]
pub struct FitnessMetrics {
    /// Total points collected across all levels
    pub total_points: f64,
    /// Consistency score (lower variance = higher score)
    pub consistency: f64,
    /// Efficiency (points per step)
    pub efficiency: f64,
    /// Learning score (improvement across plays)
    pub learning: f64,
    /// Combined fitness score
    pub combined: f64,
}

impl FitnessMetrics {
    /// Calculate combined fitness from components
    pub fn calculate_combined(&self) -> f64 {
        // Weighted combination of fitness components
        // Points are the primary objective, but we reward consistency, efficiency, and learning
        let points_weight = 1.0;
        let consistency_weight = 0.2;
        let efficiency_weight = 0.1;
        let learning_weight = 0.3;

        self.total_points * points_weight
            + self.consistency * consistency_weight
            + self.efficiency * efficiency_weight
            + self.learning * learning_weight
    }
}

fn main() {
    setup();

    let config = CultusConfig::new().unwrap();

    // Check if we should use the new direct encoding system
    let use_direct_encoding = config.evolution.use_direct_encoding.unwrap_or(true);

    if use_direct_encoding {
        info!("Using new DirectNetwork-based evolution");
        run_direct_evolution(&config);
    } else {
        info!("Using legacy bit-string encoding");
        run_legacy_evolution(&config);
    }
}

/// Run evolution using the new DirectNetwork encoding system.
fn run_direct_evolution(config: &CultusConfig) {
    let evolution_config = &config.evolution;
    let smart_network_config = &config.smart_network;
    let game_config = &config.game;

    // Load levels
    let levels: Vec<Level> = game_config
        .level_to_times_to_play
        .keys()
        .map(|&lvl_idx| {
            let path = format!("{}level_{}.lvl", game_config.levels_dir_path, lvl_idx);
            Level::from_lvl_file(&path, game_config.max_steps)
        })
        .collect();

    for (idx, lvl) in levels.iter().enumerate() {
        info!(
            "Level {}: size={}x{}, player={:?}, total_points={}",
            idx + 1,
            lvl.get_size_rows(),
            lvl.get_size_column(),
            lvl.get_player_position(),
            lvl.get_point_amount()
        );
    }

    // Configure DirectNetwork evolution
    let memory_config = if smart_network_config.memory_register_count > 0 {
        Some(MemoryConfig {
            register_count: smart_network_config.memory_register_count as u8,
            register_width: smart_network_config.memory_register_width as u8,
        })
    } else {
        None
    };

    // Calculate total output count including memory control signals
    // Memory outputs per register: 1 bit (Write Enable) + width bits (New Data)
    let memory_output_count = if memory_config.is_some() {
        smart_network_config.memory_register_count * (1 + smart_network_config.memory_register_width)
    } else {
        0
    };
    let total_output_count = smart_network_config.output_count + memory_output_count;

    // Total input count includes memory feedback (all registers flattened)
    let total_input_count = smart_network_config.input_count + 
        (smart_network_config.memory_register_count * smart_network_config.memory_register_width);

    let direct_config = DirectEvolutionConfig {
        gate_count: evolution_config.initial_gate_count.unwrap_or(100),
        input_count: total_input_count as u16,
        output_count: total_output_count as u16,
        memory_config: memory_config.clone(),
        elite_factor: evolution_config.elite_factor,
        tournament_size: evolution_config.tournament_size,
        crossover_rate: 0.8,
        mutation_rate: 0.3,
        use_module_crossover: true,
        use_local_search: evolution_config.use_local_search.unwrap_or(false),
        local_search_config: Some(LocalSearchConfig {
            num_trials: 20,
            max_stagnation: 5,
            strategy: LocalSearchStrategy::Hybrid {
                lamarckian_probability: 0.5,
            },
        }),
    };

    // Initialize curriculum manager
    let mut curriculum = CurriculumManager::default_curriculum();

    // Initialize novelty archive and config
    let novelty_config = NoveltyConfig::default();
    let mut novelty_archive = NoveltyArchive::new(
        novelty_config.archive_size,
        novelty_config.k_nearest,
        novelty_config.archive_threshold,
    );

    // Generate initial population
    info!(
        "Generating initial population of {} DirectNetworks with {} gates (inputs={}, outputs={})...",
        evolution_config.initial_population_count, direct_config.gate_count,
        direct_config.input_count, direct_config.output_count
    );
    let mut population =
        generate_initial_direct_population(evolution_config.initial_population_count, &direct_config);
    info!("Initial population generated.");

    // Main evolution loop
    for generation in 0.. {
        // Get current curriculum stage configuration (clone to avoid borrow issues)
        let current_stage = curriculum.current_stage();
        let stage_level_indices = current_stage.level_indices.clone();
        let plays_per_level = current_stage.plays_per_level;

        let stage_levels: Vec<Level> = stage_level_indices
            .iter()
            .filter_map(|&idx| levels.get(idx.saturating_sub(1)).cloned())
            .collect();

        let levels_for_eval: Vec<Level> = if stage_levels.is_empty() {
            info!("Warning: No levels found for current stage, using all levels");
            levels.clone()
        } else {
            stage_levels.clone()
        };

        // Parallel fitness evaluation with novelty
        let evaluated: Vec<(DirectNetwork, f64, Option<BehavioralSignature>)> = population
            .par_iter()
            .map(|network| {
                let (fitness, signature) = evaluate_direct_network_with_novelty(
                    network,
                    &levels_for_eval,
                    plays_per_level,
                    game_config.visibility_distance,
                    novelty_config.use_milestones,
                );
                (network.clone(), fitness, signature)
            })
            .collect();

        // Compute novelty scores
        let population_signatures: Vec<BehavioralSignature> = evaluated
            .iter()
            .filter_map(|(_, _, sig)| sig.clone())
            .collect();

        let evaluated_with_novelty: Vec<DirectNetworkWithFitness> = evaluated
            .iter()
            .map(|(network, objective_fitness, signature)| {
                let novelty_score = signature
                    .as_ref()
                    .map(|sig| novelty_archive.compute_novelty(sig, &population_signatures))
                    .unwrap_or(0.0);

                // Combined fitness
                let combined = NoveltyFitness::compute(
                    *objective_fitness,
                    novelty_score,
                    0.0, // milestone bonus already included in objective
                    novelty_config.objective_weight,
                    novelty_config.novelty_weight,
                );

                DirectNetworkWithFitness::new(network.clone(), combined.combined_fitness)
            })
            .collect();

        // Update novelty archive with best performers
        for (_, _, signature) in &evaluated {
            if let Some(sig) = signature {
                let novelty = novelty_archive.compute_novelty(sig, &population_signatures);
                novelty_archive.maybe_add(sig.clone(), novelty);
            }
        }

        // Calculate statistics
        let stats = DirectPopulationStats::compute(&evaluated_with_novelty);

        // Find best for logging
        let best = evaluated_with_novelty
            .iter()
            .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());

        let best_objective = evaluated
            .iter()
            .map(|(_, f, _)| *f)
            .fold(f64::NEG_INFINITY, f64::max);

        info!(
            "Gen {} [Stage {}]: max={:.1}, avg={:.1}, obj_max={:.1}, gates={:.1}, div={:.2}, archive={}",
            generation,
            curriculum.current_stage_index() + 1,
            stats.max_fitness,
            stats.avg_fitness,
            best_objective,
            stats.avg_gate_count,
            stats.diversity,
            novelty_archive.len()
        );

        // Check for curriculum advancement
        if curriculum.update(best_objective) {
            info!(
                "*** Advanced to curriculum stage {} ***",
                curriculum.current_stage_index() + 1
            );
            // Migrate networks to new stage if needed
            let mut rng = thread_rng();
            for network in &mut population {
                curriculum.migrate_network(network, &mut rng);
            }
        }

        // Persist best if configured
        if evolution_config.persist_top_chromosome {
            if let Some(best_network) = best {
                info!(
                    "Best network: {} gates, fitness={:.1}",
                    best_network.network.gates.len(),
                    best_network.fitness
                );
            }
        }

        // Evolve next generation (capture levels_for_eval and plays_per_level by value)
        let vis_dist = game_config.visibility_distance;
        let evaluate_fn = |network: &DirectNetwork| -> f64 {
            evaluate_direct_network_simple(
                network,
                &levels_for_eval,
                plays_per_level,
                vis_dist,
            )
        };

        population = evolve_direct(&evaluated_with_novelty, &direct_config, evaluate_fn);
    }
}

/// Evaluate a DirectNetwork and return fitness with optional behavioral signature.
fn evaluate_direct_network_with_novelty(
    network: &DirectNetwork,
    levels: &[Level],
    plays_per_level: usize,
    visibility_distance: usize,
    use_milestones: bool,
) -> (f64, Option<BehavioralSignature>) {
    let mut smart_network = SmartNetwork::from_direct_network_auto(network.clone());
    let mut total_points = 0.0;
    let mut all_traces: Vec<GameTrace> = Vec::new();
    let mut milestone_tracker = MilestoneTracker::new();

    for level in levels {
        for _ in 0..plays_per_level {
            let result = play_game_with_network_traced(
                &mut smart_network,
                level.clone(),
                visibility_distance,
                false,
                true, // collect trace
            );

            total_points += result.points as f64;

            if let Some(trace) = result.trace {
                // Check milestones
                if use_milestones {
                    total_points += milestone_tracker.check_milestones(&trace);
                }
                all_traces.push(trace);
            }
        }
    }

    // Combine traces into a single behavioral signature
    let signature = if !all_traces.is_empty() {
        // Use the first trace for simplicity (could combine them)
        Some(BehavioralSignature::from_trace(&all_traces[0]))
    } else {
        None
    };

    let avg_points = total_points / (levels.len() * plays_per_level) as f64;
    (avg_points, signature)
}

/// Simple evaluation without novelty (for local search).
fn evaluate_direct_network_simple(
    network: &DirectNetwork,
    levels: &[Level],
    plays_per_level: usize,
    visibility_distance: usize,
) -> f64 {
    let mut smart_network = SmartNetwork::from_direct_network_auto(network.clone());
    let mut total_points = 0.0;

    for level in levels {
        for _ in 0..plays_per_level {
            let result = play_game_with_network(
                &mut smart_network,
                level.clone(),
                visibility_distance,
                false,
            );
            total_points += result.points as f64;
        }
    }

    total_points / (levels.len() * plays_per_level) as f64
}

/// Run evolution using the legacy bit-string encoding (backward compatibility).
fn run_legacy_evolution(config: &CultusConfig) {
    let evolution_config = &config.evolution;
    let smart_network_config = &config.smart_network;
    let game_config = &config.game;

    // Calculate chromosome size
    let chromosome_size = SmartNetwork::get_required_bits_for_bitstring(
        smart_network_config.input_count,
        smart_network_config.output_count,
        smart_network_config.nand_count_bits,
        smart_network_config.memory_register_count,
        smart_network_config.memory_register_width,
        smart_network_config.connection_count,
    );

    info!("Chromosome size: {} bits", chromosome_size);

    // Load levels once
    let levels: Vec<Level> = game_config
        .level_to_times_to_play
        .keys()
        .map(|&lvl_idx| {
            let path = format!("{}level_{}.lvl", game_config.levels_dir_path, lvl_idx);
            Level::from_lvl_file(&path, game_config.max_steps)
        })
        .collect();

    for (idx, lvl) in levels.iter().enumerate() {
        info!(
            "Level {}: size={}x{}, player={:?}, total_points={}",
            idx + 1,
            lvl.get_size_rows(),
            lvl.get_size_column(),
            lvl.get_player_position(),
            lvl.get_point_amount()
        );
    }

    // Generate initial population
    info!(
        "Generating initial population of {} chromosomes...",
        evolution_config.initial_population_count
    );
    let mut population = generate_initial_population(
        evolution_config.initial_population_count,
        chromosome_size,
    );
    info!("Initial population generated.");

    // Main evolution loop
    for generation in 0.. {
        // Parallel fitness evaluation with multi-objective metrics
        let evaluated: Vec<(Chromosome, FitnessMetrics)> = population
            .par_iter()
            .map(|chromosome| {
                let metrics = evaluate_chromosome_multi_objective(
                    chromosome,
                    &levels,
                    &game_config.level_to_times_to_play,
                    smart_network_config,
                    game_config.visibility_distance,
                );
                (chromosome.clone(), metrics)
            })
            .collect();

        // Calculate statistics
        let fitness_values: Vec<f64> = evaluated.iter().map(|(_, m)| m.combined).collect();
        let fitness_max = fitness_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let fitness_avg = fitness_values.iter().sum::<f64>() / fitness_values.len() as f64;

        // Find best metrics for detailed logging
        let best = evaluated.iter().max_by(|a, b| {
            a.1.combined.partial_cmp(&b.1.combined).unwrap()
        });

        if let Some((_, best_metrics)) = best {
            info!(
                "Gen {}: max={:.1}, avg={:.1} | best: pts={:.0}, cons={:.2}, eff={:.2}, learn={:.2}",
                generation,
                fitness_max,
                fitness_avg,
                best_metrics.total_points,
                best_metrics.consistency,
                best_metrics.efficiency,
                best_metrics.learning
            );
        }

        // Persist top chromosome if configured
        if evolution_config.persist_top_chromosome {
            if let Some((top_chromosome, _)) = best {
                info!("Top chromosome: {}", top_chromosome);
            }
        }

        // Convert to ChromosomeWithFitness for evolution
        let evaluated_with_fitness: Vec<ChromosomeWithFitness<u32>> = evaluated
            .into_iter()
            .map(|(c, m)| ChromosomeWithFitness::new(c, m.combined.floor() as u32))
            .collect();

        let evaluated_set: HashSet<ChromosomeWithFitness<u32>> =
            evaluated_with_fitness.into_iter().collect();

        // Evolve next generation using two-point crossover for better gene mixing
        population = evolve(
            &evaluated_set,
            SelectionStrategy::Tournament(evolution_config.tournament_size),
            CrossoverStrategy::TwoPoint,
            evolution_config.elite_factor,
        );
    }
}

/// Evaluate a chromosome with multi-objective fitness metrics
fn evaluate_chromosome_multi_objective(
    chromosome: &Chromosome,
    levels: &[Level],
    level_to_times: &std::collections::HashMap<usize, usize>,
    smart_network_config: &config::smart_network_config::SmartNetworkConfig,
    visibility_distance: usize,
) -> FitnessMetrics {
    // Convert BitVec to Vec<bool> for SmartNetwork::from_bits
    let bits: Vec<bool> = chromosome.genes.iter().map(|b| *b).collect();

    let mut smart_network = SmartNetwork::from_bits(
        &bits,
        smart_network_config.input_count,
        smart_network_config.output_count,
        smart_network_config.nand_count_bits,
        smart_network_config.memory_register_count,
        smart_network_config.memory_register_width,
    );

    let mut all_results: Vec<GameResult> = Vec::new();
    let mut total_points: f64 = 0.0;
    let mut total_efficiency: f64 = 0.0;
    let mut total_learning: f64 = 0.0;

    for (&level_idx, &times) in level_to_times {
        let level = &levels[level_idx - 1];

        // Play the level multiple times, tracking results for learning detection
        let mut level_results: Vec<GameResult> = Vec::with_capacity(times);
        for _ in 0..times {
            let result = play_game_with_network(
                &mut smart_network,
                level.clone(),
                visibility_distance,
                false,
            );
            level_results.push(result);
        }

        // Calculate learning score: do later plays score better than earlier plays?
        let learning_score = calculate_learning_score(&level_results);
        total_learning += learning_score;

        // Calculate efficiency: points per step
        let level_efficiency: f64 = level_results.iter().map(|r| r.efficiency()).sum::<f64>()
            / level_results.len() as f64;
        total_efficiency += level_efficiency;

        // Aggregate points (use all results, not just top 20%)
        let level_points: f64 = level_results.iter().map(|r| r.points as f64).sum::<f64>()
            / level_results.len() as f64;
        total_points += level_points;

        all_results.extend(level_results);
    }

    // Calculate consistency: inverse of coefficient of variation
    let consistency = calculate_consistency(&all_results);

    let metrics = FitnessMetrics {
        total_points,
        consistency,
        efficiency: total_efficiency,
        learning: total_learning,
        combined: 0.0, // Will be calculated
    };

    FitnessMetrics {
        combined: metrics.calculate_combined(),
        ..metrics
    }
}

/// Calculate learning score based on improvement across sequential plays
/// Returns positive value if performance improves over time
fn calculate_learning_score(results: &[GameResult]) -> f64 {
    if results.len() < 2 {
        return 0.0;
    }

    // Split results into first half and second half
    let mid = results.len() / 2;
    let first_half_avg: f64 = results[..mid].iter().map(|r| r.points as f64).sum::<f64>() / mid as f64;
    let second_half_avg: f64 = results[mid..].iter().map(|r| r.points as f64).sum::<f64>()
        / (results.len() - mid) as f64;

    // Learning score is the improvement (positive if getting better)
    let improvement = second_half_avg - first_half_avg;

    // Also check for trend: count how many consecutive increases we see
    let mut trend_score = 0.0;
    let window_size = 5.min(results.len());
    if results.len() >= window_size * 2 {
        // Start at window_size * 2 to have room for previous window
        for i in (window_size * 2)..=results.len() {
            let current_window: f64 = results[(i - window_size)..i]
                .iter()
                .map(|r| r.points as f64)
                .sum::<f64>();
            let prev_window: f64 = results[(i - window_size * 2)..(i - window_size)]
                .iter()
                .map(|r| r.points as f64)
                .sum::<f64>();
            if current_window > prev_window {
                trend_score += 1.0;
            }
        }
        let num_comparisons = results.len() - window_size * 2 + 1;
        if num_comparisons > 0 {
            trend_score /= num_comparisons as f64;
        }
    }

    // Combine improvement and trend
    improvement.max(0.0) + trend_score * 10.0
}

/// Calculate consistency score based on variance of results
/// Higher score means more consistent performance
fn calculate_consistency(results: &[GameResult]) -> f64 {
    if results.is_empty() {
        return 0.0;
    }

    let points: Vec<f64> = results.iter().map(|r| r.points as f64).collect();
    let mean = points.iter().sum::<f64>() / points.len() as f64;

    if mean == 0.0 {
        return 0.0;
    }

    let variance = points.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / points.len() as f64;
    let std_dev = variance.sqrt();

    // Coefficient of variation (lower is more consistent)
    let cv = std_dev / mean;

    // Convert to consistency score (higher is better)
    // Use exponential decay so high variance is heavily penalized
    let consistency = (-cv).exp() * mean;

    consistency
}

#[cfg(test)]
mod main_tests {
    use super::*;

    #[test]
    fn test_learning_score_improvement() {
        // Simulate improving performance
        let results: Vec<GameResult> = (0..10)
            .map(|i| GameResult {
                points: i * 10,
                steps_taken: 20,
                max_steps: 30,
                trace: None,
            })
            .collect();

        let score = calculate_learning_score(&results);
        assert!(score > 0.0, "Learning score should be positive for improving results");
    }

    #[test]
    fn test_learning_score_no_improvement() {
        // Simulate flat performance
        let results: Vec<GameResult> = (0..10)
            .map(|_| GameResult {
                points: 50,
                steps_taken: 20,
                max_steps: 30,
                trace: None,
            })
            .collect();

        let score = calculate_learning_score(&results);
        assert!(score >= 0.0, "Learning score should be non-negative");
    }

    #[test]
    fn test_consistency_high() {
        // All same scores = high consistency
        let results: Vec<GameResult> = (0..10)
            .map(|_| GameResult {
                points: 100,
                steps_taken: 20,
                max_steps: 30,
                trace: None,
            })
            .collect();

        let score = calculate_consistency(&results);
        assert!(score > 50.0, "Consistency should be high for uniform results");
    }

    #[test]
    fn test_consistency_low() {
        // Highly variable scores = low consistency
        let results: Vec<GameResult> = vec![
            GameResult { points: 0, steps_taken: 30, max_steps: 30, trace: None },
            GameResult { points: 200, steps_taken: 10, max_steps: 30, trace: None },
            GameResult { points: 0, steps_taken: 30, max_steps: 30, trace: None },
            GameResult { points: 200, steps_taken: 10, max_steps: 30, trace: None },
        ];

        let score = calculate_consistency(&results);
        // With high variance, consistency should be lower
        assert!(score < 100.0, "Consistency should be lower for variable results");
    }

    #[test]
    fn test_fitness_metrics_combined() {
        let metrics = FitnessMetrics {
            total_points: 100.0,
            consistency: 50.0,
            efficiency: 5.0,
            learning: 10.0,
            combined: 0.0,
        };

        let combined = metrics.calculate_combined();
        // 100 * 1.0 + 50 * 0.2 + 5 * 0.1 + 10 * 0.3 = 100 + 10 + 0.5 + 3 = 113.5
        assert!((combined - 113.5).abs() < 0.01);
    }
}
