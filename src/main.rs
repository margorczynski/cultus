extern crate core;

use std::collections::HashSet;
use std::time::Instant;

use log::info;
use rayon::prelude::*;

use common::*;
use evolution::chromosome_with_fitness::ChromosomeWithFitness;
use evolution::evolution::*;
use game::level::Level;
use smart_network::smart_network::*;
use smart_network_game_adapter::play_game_with_network;

use crate::config::cultus_config::CultusConfig;
use crate::evolution::evolution::SelectionStrategy::Tournament;

mod common;
mod config;
mod evolution;
mod game;
mod smart_network;
mod smart_network_game_adapter;

fn main() {
    setup();

    let config = CultusConfig::new().unwrap();

    let evolution_config = config.evolution;
    let smart_network_config = config.smart_network;
    let game_config = config.game;

    let first_level = Level::from_lvl_file(&game_config.level_path, game_config.max_steps);

    info!("Starting smart network architecture search");

    let smart_network_bitstring_len = SmartNetwork::get_required_bits_for_bitstring(
        smart_network_config.input_count,
        smart_network_config.output_count,
        smart_network_config.nand_count_bits,
        smart_network_config.mem_addr_bits,
        smart_network_config.mem_rw_bits,
        smart_network_config.connection_count,
    );
    let mut population = generate_initial_population(100, smart_network_bitstring_len);

    //Generation loop
    let mut generation_counter = 0;
    loop {
        info!("Evaluating generation {}", generation_counter + 1);
        let generation_eval_start = Instant::now();

        //Chromosome loop - create network from chromosome and play the game to calculate the fitness for the chromosomes
        let population_with_fitness: HashSet<ChromosomeWithFitness<usize>> = population
            .par_iter()
            .map(|chromosome| {
                let mut smart_network = SmartNetwork::from_bitstring(
                    &bit_vector_to_bitstring(&chromosome.genes),
                    smart_network_config.input_count,
                    smart_network_config.output_count,
                    smart_network_config.nand_count_bits,
                    smart_network_config.mem_addr_bits,
                    smart_network_config.mem_rw_bits,
                );

                let results: Vec<usize> = (0..20)
                    .map(|_| {
                        play_game_with_network(
                            &mut smart_network,
                            first_level.clone(),
                            game_config.visibility_distance,
                        )
                    })
                    .collect();

                let results_sum: usize = results.iter().sum();
                //TODO: Use max or average?
                let fitness = results_sum as f64 / results.len() as f64;

                ChromosomeWithFitness::from_chromosome_and_fitness(
                    chromosome.clone(),
                    fitness.floor() as usize,
                )
            })
            .collect();

        let fitness_sum = population_with_fitness
            .iter()
            .map(|c| c.fitness)
            .sum::<usize>() as f64;

        let fitness_avg = fitness_sum / population_with_fitness.len() as f64;

        info!(
            "Generated new population. Fitness max: {}, average: {}",
            population_with_fitness
                .iter()
                .max()
                .map(|c| c.fitness as i64)
                .unwrap_or(-9999),
            fitness_avg
        );

        population.clear();
        population.extend(evolve(
            &population_with_fitness,
            Tournament(evolution_config.tournament_size),
            evolution_config.mutation_rate,
            evolution_config.elite_factor,
        ));

        let generation_eval_duration = generation_eval_start.elapsed();
        info!("Evaluation duration: {:?}", generation_eval_duration);
        generation_counter += 1;
    }
}
