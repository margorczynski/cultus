use std::collections::HashSet;

use log::info;
use rayon::prelude::*;

use common::setup;
use config::cultus_config::CultusConfig;
use evolution::chromosome::Chromosome;
use evolution::chromosome_with_fitness::ChromosomeWithFitness;
use evolution::evolution::{evolve, generate_initial_population, SelectionStrategy};
use game::level::Level;
use smart_network::smart_network::SmartNetwork;
use smart_network_game_adapter::play_game_with_network;

mod common;
mod config;
mod evolution;
mod game;
mod smart_network;
mod smart_network_game_adapter;

fn main() {
    setup();

    let config = CultusConfig::new().unwrap();

    let evolution_config = &config.evolution;
    let smart_network_config = &config.smart_network;
    let game_config = &config.game;

    // Calculate chromosome size
    let chromosome_size = SmartNetwork::get_required_bits_for_bitstring(
        smart_network_config.input_count,
        smart_network_config.output_count,
        smart_network_config.nand_count_bits,
        smart_network_config.mem_addr_bits,
        smart_network_config.mem_rw_bits,
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
        // Parallel fitness evaluation
        let evaluated: Vec<ChromosomeWithFitness<u32>> = population
            .par_iter()
            .map(|chromosome| {
                let fitness = evaluate_chromosome(
                    chromosome,
                    &levels,
                    &game_config.level_to_times_to_play,
                    smart_network_config,
                    game_config.visibility_distance,
                );
                ChromosomeWithFitness::new(chromosome.clone(), fitness)
            })
            .collect();

        // Calculate statistics
        let fitness_sum: u64 = evaluated.iter().map(|c| c.fitness as u64).sum();
        let fitness_max = evaluated.iter().map(|c| c.fitness).max().unwrap_or(0);
        let fitness_avg = fitness_sum as f64 / evaluated.len() as f64;

        info!(
            "Generation {}: max={}, avg={:.2}, population={}",
            generation,
            fitness_max,
            fitness_avg,
            evaluated.len()
        );

        // Persist top chromosome if configured
        if evolution_config.persist_top_chromosome {
            if let Some(top) = evaluated.iter().max() {
                info!("Top chromosome: {}", top.chromosome);
            }
        }

        // Convert to HashSet for evolution
        let evaluated_set: HashSet<ChromosomeWithFitness<u32>> =
            evaluated.into_iter().collect();

        // Evolve next generation
        population = evolve(
            &evaluated_set,
            SelectionStrategy::Tournament(evolution_config.tournament_size),
            evolution_config.elite_factor,
        );
    }
}

fn evaluate_chromosome(
    chromosome: &Chromosome,
    levels: &[Level],
    level_to_times: &std::collections::HashMap<usize, usize>,
    smart_network_config: &config::smart_network_config::SmartNetworkConfig,
    visibility_distance: usize,
) -> u32 {
    // Convert chromosome to bitstring
    let bitstring: String = chromosome.genes.iter().map(|b| if *b { '1' } else { '0' }).collect();

    // Create smart network from chromosome
    let mut smart_network = SmartNetwork::from_bitstring(
        &bitstring,
        smart_network_config.input_count,
        smart_network_config.output_count,
        smart_network_config.nand_count_bits,
        smart_network_config.mem_addr_bits,
        smart_network_config.mem_rw_bits,
    );

    // Play each level multiple times and aggregate fitness
    let mut total_fitness: f64 = 0.0;

    for (&level_idx, &times) in level_to_times {
        let level = &levels[level_idx - 1];

        // Play the level multiple times
        let mut results: Vec<u32> = Vec::with_capacity(times);
        for _ in 0..times {
            let points = play_game_with_network(
                &mut smart_network,
                level.clone(),
                visibility_distance,
                false,
            ) as u32;
            results.push(points);
        }

        // Take Pareto-optimal 20% (best results)
        results.sort_unstable();
        let pareto_count = ((results.len() as f64) * 0.2).ceil() as usize;
        let pareto_sum: u32 = results.iter().rev().take(pareto_count).sum();
        let pareto_avg = pareto_sum as f64 / pareto_count as f64;

        total_fitness += pareto_avg;
    }

    total_fitness.floor() as u32
}
