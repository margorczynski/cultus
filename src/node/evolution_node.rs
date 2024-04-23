
use std::collections::HashSet;
use std::str::from_utf8;
use std::time::{Instant};
use std::fs::File;
use std::io::Write;

use futures::stream::StreamExt;
use lapin::message::Delivery;

use lapin::{
    options::{BasicAckOptions, BasicConsumeOptions, BasicPublishOptions},
    types::FieldTable,
    BasicProperties, Connection,
};
use log::{error, info, trace};
use rayon::prelude::*;


use crate::config::amqp_config::AmqpConfig;
use crate::config::evolution_config::EvolutionConfig;
use crate::config::smart_network_config::SmartNetworkConfig;

use crate::evolution::chromosome_with_fitness::ChromosomeWithFitness;
use crate::evolution::evolution::SelectionStrategy::Tournament;
use crate::evolution::evolution::*;
use crate::smart_network::smart_network::SmartNetwork;


pub async fn evolution_node_loop(
    connection: &Connection,
    _smart_network_config: &SmartNetworkConfig,
    evolution_config: &EvolutionConfig,
    amqp_config: &AmqpConfig,
) {
    info!("Starting evolution processing...");

    let mut start = Instant::now();

    let publish_channel = connection.create_channel().await.unwrap();
    let consume_channel = connection.create_channel().await.unwrap();

    let mut deliveries_buffer: Vec<Delivery> = Vec::new();

    let mut consumer = consume_channel
        .basic_consume(
            &amqp_config.chromosome_with_fitness_queue_name,
            "evolution",
            BasicConsumeOptions::default(),
            FieldTable::default(),
        )
        .await
        .unwrap();

    let mut generation_count: usize = 0;

    while let Some(delivery) = consumer.next().await {
        trace!("Received message: {:?}", delivery);
        if let Ok(delivery) = delivery {
            deliveries_buffer.push(delivery);

            //TODO: Not initial pop count always - change
            if deliveries_buffer.len() == evolution_config.initial_population_count {
                let evolve_start = Instant::now();
                info!("------- GENERATION {} -------", generation_count);
                let population_collected: Vec<ChromosomeWithFitness<u32>> = deliveries_buffer
                    .par_iter()
                    .map(|d| {
                        let utf8_payload = from_utf8(d.data.as_slice()).unwrap();

                        serde_json::from_str::<ChromosomeWithFitness<u32>>(utf8_payload).unwrap()
                    })
                    .collect();

                let fitness_sum = population_collected
                    .par_iter()
                    .map(|c| c.fitness)
                    .sum::<u32>() as f64;

                let top_chromosome = population_collected
                    .par_iter()
                    .max();

                let fitness_max = top_chromosome
                    .map(|c| c.fitness as f64)
                    .unwrap_or(-9999f64);

                let fitness_avg = fitness_sum / population_collected.len() as f64;

                info!(
                    "GEN={} ::: fitness_max={}, fitness_average={}, chromosome_count={}",
                    generation_count, fitness_max, fitness_avg, deliveries_buffer.len()
                );

                if evolution_config.persist_top_chromosome {
                    match top_chromosome {
                        None => {
                            error!("GEN={} ::: No max fitness chromosome to persist", generation_count);
                        }
                        Some(chromosome) => {
                            let mut file = File::create("top_chromosomes").unwrap();
                            let chromosome_str = format!("{},{}\n", chromosome.chromosome, chromosome.fitness);
                            file.write(chromosome_str.as_bytes()).unwrap();
                        }
                    }
                }

                let evolve_new_generation_start = Instant::now();
                let evolved = evolve(
                    &HashSet::from_iter(population_collected),
                    Tournament(evolution_config.tournament_size),
                    evolution_config.elite_factor,
                );

                info!("GEN={} ::: evolve_elapsed={:?}. evolve_new_generation={:?}", generation_count, evolve_start.elapsed(), evolve_new_generation_start.elapsed());

                for chromosome in evolved {
                    let serialized = serde_json::to_string(&chromosome).unwrap();

                    publish_channel
                        .basic_publish(
                            "",
                            &amqp_config.chromosome_queue_name,
                            BasicPublishOptions::default(),
                            serialized.as_bytes(),
                            BasicProperties::default(),
                        )
                        .await
                        .unwrap()
                        .await
                        .unwrap();
                }

                publish_channel.wait_for_confirms().await.unwrap();

                info!(
                    "GEN={} ::: since_last_generation_evolve_elapsed={:?}",
                    generation_count, start.elapsed()
                );
                start = Instant::now();

                for buffered_delivery in deliveries_buffer.iter() {
                    buffered_delivery
                        .ack(BasicAckOptions::default())
                        .await
                        .expect("Chromosome with fitness ACK fail");
                }

                publish_channel.wait_for_confirms().await.unwrap();

                deliveries_buffer.clear();

                generation_count += 1;
            }
        }
    }
}

pub async fn evolution_publish_initial_population(
    connection: &Connection,
    smart_network_config: &SmartNetworkConfig,
    evolution_config: &EvolutionConfig,
    amqp_config: &AmqpConfig,
) {
    let channel = connection.create_channel().await.unwrap();

    let smart_network_bitstring_len = SmartNetwork::get_required_bits_for_bitstring(
        smart_network_config.input_count,
        smart_network_config.output_count,
        smart_network_config.nand_count_bits,
        smart_network_config.mem_addr_bits,
        smart_network_config.mem_rw_bits,
        smart_network_config.connection_count,
    );

    let initial_population = generate_initial_population(
        evolution_config.initial_population_count,
        smart_network_bitstring_len,
    );

    for chromosome in &initial_population {
        let serialized = serde_json::to_string(&chromosome).unwrap();

        channel
            .basic_publish(
                "",
                &amqp_config.chromosome_queue_name,
                BasicPublishOptions::default(),
                serialized.as_bytes(),
                BasicProperties::default(),
            )
            .await
            .unwrap();
    }

    channel.wait_for_confirms().await.unwrap();

    info!(
        "Published {} chromosomes as initial population",
        &initial_population.len()
    );
}