use std::collections::HashSet;
use std::str::from_utf8;
use std::time::{Duration, Instant};
use log::{error, info, trace};
use futures::stream::StreamExt;
use lapin::Channel;
use lapin::{
    message::DeliveryResult,
    options::{BasicAckOptions, BasicConsumeOptions, BasicPublishOptions, QueueDeclareOptions},
    types::FieldTable,
    BasicProperties, Connection, ConnectionProperties,
};
use crate::config::evolution_config::EvolutionConfig;

use crate::evolution::evolution::*;
use crate::evolution::evolution::SelectionStrategy::Tournament;
use crate::smart_network::smart_network::SmartNetwork;
use crate::config::smart_network_config::SmartNetworkConfig;
use crate::evolution::chromosome_with_fitness;
use crate::evolution::chromosome_with_fitness::ChromosomeWithFitness;

pub async fn evolution_node_loop(channel: &Channel, smart_network_config: &SmartNetworkConfig, evolution_config: &EvolutionConfig) {

    info!("Starting evolution processing...");

    let mut start = Instant::now();

    let mut population_collected: Vec<ChromosomeWithFitness<usize>> = Vec::new();

    let mut consumer = channel
        .basic_consume(
            "chromosomes_with_fitness",
            "evolution",
            BasicConsumeOptions::default(),
            FieldTable::default(),
        )
        .await
        .unwrap();

    while let Some(delivery) = consumer.next().await {
        trace!("Received message: {:?}", delivery);
        if let Ok(delivery) = delivery {

            let utf8_payload = from_utf8(delivery.data.as_slice()).unwrap();

            let chromosome_with_fitness = serde_json::from_str::<ChromosomeWithFitness<usize>>(utf8_payload).unwrap();

            population_collected.push(chromosome_with_fitness);

            //TODO: Not initial pop count - change
            if population_collected.len() == evolution_config.initial_population_count {
                let fitness_sum = population_collected
                    .iter()
                    .map(|c| c.fitness)
                    .sum::<usize>() as f64;

                let fitness_max = population_collected
                    .iter()
                    .max()
                    .map(|c| c.fitness as i64)
                    .unwrap_or(-9999);

                let fitness_avg = fitness_sum / population_collected.len() as f64;

                info!(
                    "Generating new population from population - fitness max: {}, average: {}",
                    fitness_max,
                    fitness_avg
                );

                let evolved = evolve(
                    &HashSet::from_iter( population_collected.to_owned()),
                    Tournament(evolution_config.tournament_size),
                    evolution_config.mutation_rate,
                    evolution_config.elite_factor,
                );

                population_collected.clear();

                for chromosome in evolved {
                    let serialized = serde_json::to_string(&chromosome).unwrap();

                    channel
                        .basic_publish(
                            "",
                            "chromosomes",
                            BasicPublishOptions::default(),
                            serialized.as_bytes(),
                            BasicProperties::default(),
                        )
                        .await
                        .unwrap()
                        .await
                        .unwrap();
                }

                info!("Published new population to queue. Time elapsed since last publish: {:?}", start.elapsed());
                start = Instant::now();
            }

            delivery
                .ack(BasicAckOptions::default())
                .await
                .expect("basic_ack");
        }
    }
}

pub async fn evolution_publish_initial_population(channel: &Channel, smart_network_config: &SmartNetworkConfig, evolution_config: &EvolutionConfig) {
    let smart_network_bitstring_len = SmartNetwork::get_required_bits_for_bitstring(
        smart_network_config.input_count,
        smart_network_config.output_count,
        smart_network_config.nand_count_bits,
        smart_network_config.mem_addr_bits,
        smart_network_config.mem_rw_bits,
        smart_network_config.connection_count,
    );

    let initial_population = generate_initial_population(evolution_config.initial_population_count, smart_network_bitstring_len);

    for chromosome in &initial_population {
        let serialized = serde_json::to_string(&chromosome).unwrap();

        channel.basic_publish("", "chromosomes", BasicPublishOptions::default(), serialized.as_bytes(), BasicProperties::default()).await.unwrap();
    }

    info!("Published {} chromosomes as initial population", &initial_population.len());
}