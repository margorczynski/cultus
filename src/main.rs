extern crate core;

use std::borrow::Borrow;
use std::collections::HashSet;
use std::time::{Duration, Instant};
use log::info;

use common::*;
use evolution::chromosome_with_fitness::ChromosomeWithFitness;
use evolution::evolution::*;
use game::level::Level;
use smart_network::smart_network::*;
use smart_network_game_adapter::play_game_with_network;

use crate::evolution::evolution::SelectionStrategy::Tournament;

mod smart_network;
mod evolution;
mod common;
mod game;
mod smart_network_game_adapter;

fn main() {
    setup();

    let input_count = 149;
    let output_count = 2;

    let nand_count_bits = 8;

    let mem_addr_bits = 8;
    let mem_rw_bits = 16;

    let connection_count = 5000;

    let visibility_distance = 2;
    let max_steps = 35;
    let first_level_file_path = concat!(env!("CARGO_MANIFEST_DIR"), "/levels/level_1.lvl");
    let first_level = Level::from_lvl_file(first_level_file_path, max_steps);

    let tournament_size = 5;
    let mutation_rate = 0.01;
    let elite_factor = 0.1;

    info!("Starting smart network architecture search");

    let smart_network_bitstring_len = SmartNetwork::get_required_bits_for_bitstring(input_count, output_count, nand_count_bits, mem_addr_bits, mem_rw_bits, connection_count);
    let mut population = generate_initial_population(100, smart_network_bitstring_len);

    //Generation loop
    let mut generation_counter = 0;
    loop {
        info!("Evaluating generation {}", generation_counter + 1);
        let generation_eval_start = Instant::now();

        let mut chromosome_smart_network_build_durations: Vec<Duration> = Vec::new();
        let mut chromosome_fitness_eval_durations: Vec<Duration> = Vec::new();
        let mut chromosomes_with_fitness: HashSet<ChromosomeWithFitness<usize>> = HashSet::new();

        //Chromosome loop - create network from chromosome and play the game to calculate the fitness for the chromosomes
        for chromosome in population.borrow() {

            let chromosome_smart_network_build_start = Instant::now();
            let mut smart_network = SmartNetwork::from_bitstring(&bit_vector_to_bitstring(&chromosome.genes), input_count, output_count, nand_count_bits, mem_addr_bits, mem_rw_bits);
            let chromosome_smart_network_build_duration = chromosome_smart_network_build_start.elapsed();
            chromosome_smart_network_build_durations.push(chromosome_smart_network_build_duration);

            let chromosome_fitness_eval_start = Instant::now();
            let results: Vec<usize> =
                (0..20).map(|idx| play_game_with_network(&mut smart_network, first_level.clone(), visibility_distance)).collect();
            let chromosome_fitness_eval_duration = chromosome_fitness_eval_start.elapsed();
            chromosome_fitness_eval_durations.push(chromosome_fitness_eval_duration);

            let results_sum: usize = results.iter().sum();
            //TODO: Use max or average?
            let fitness = results_sum as f64 / results.len() as f64;

            chromosomes_with_fitness.insert(ChromosomeWithFitness::from_chromosome_and_fitness(chromosome.clone(), fitness.floor() as usize));
        }

        let fitness_avg = chromosomes_with_fitness.iter().map(|c| c.fitness).sum::<usize>() as f64 / chromosomes_with_fitness.len() as f64;

        let chromosome_smart_network_build_duration_max = chromosome_smart_network_build_durations.iter().max();
        let chromosome_fitness_eval_duration_max = chromosome_fitness_eval_durations.iter().max();
        info!("Smart network build max duration: {:?}, fitness eval max duration: {:?}", chromosome_smart_network_build_duration_max, chromosome_fitness_eval_duration_max);
        info!("Generated new population. Fitness max: {}, average: {}", chromosomes_with_fitness.iter().max().map(|c| c.fitness as i64).unwrap_or(-9999), fitness_avg);

        population.clear();
        population.extend(evolve(&chromosomes_with_fitness, Tournament(tournament_size), mutation_rate, elite_factor));

        let generation_eval_duration = generation_eval_start.elapsed();
        info!("Evaluation duration: {:?}", generation_eval_duration);
        generation_counter += 1;
    }
}
