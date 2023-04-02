use std::borrow::BorrowMut;
use std::str::from_utf8;

use futures::stream::StreamExt;
use log::{error, info, trace};
use lapin::Channel;
use lapin::{
    message::DeliveryResult,
    options::{BasicAckOptions, BasicConsumeOptions, BasicPublishOptions, QueueDeclareOptions},
    types::FieldTable,
    BasicProperties, Connection, ConnectionProperties,
};

use crate::smart_network::smart_network::SmartNetwork;
use crate::smart_network_game_adapter::play_game_with_network;
use crate::evolution::chromosome_with_fitness::ChromosomeWithFitness;
use crate::common::*;
use crate::config::amqp_config::AmqpConfig;
use crate::game::level::Level;
use crate::config::game_config::GameConfig;
use crate::config::smart_network_config::SmartNetworkConfig;
use crate::evolution::chromosome::Chromosome;

pub async fn fitness_calc_node_loop(channel: &Channel, smart_network_config: &SmartNetworkConfig, game_config: &GameConfig, amqp_config: &AmqpConfig) {
    //TODO: Use config instead of magic values for queue settings

    info!("Starting fitness calculation processing...");

    let level = Level::from_lvl_file(&game_config.level_path, game_config.max_steps);

    let mut consumer = channel
        .basic_consume(
            &amqp_config.chromosome_queue_name,
            "fitness",
            BasicConsumeOptions::default(),
            FieldTable::default(),
        )
        .await
        .unwrap();

    while let Some(delivery) = consumer.next().await {
        if let Ok(delivery) = delivery {
            trace!("Received message: {:?}", delivery);

            let utf8_payload = from_utf8(delivery.data.as_slice()).unwrap();
            let chromosome = serde_json::from_str::<Chromosome>(utf8_payload).unwrap();

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
                        level.clone(),
                        game_config.visibility_distance,
                    )
                })
                .collect();

            let results_sum: usize = results.iter().sum();
            //Use max or average?
            let fitness = results_sum as f64 / results.len() as f64;

            let chromosome_with_fitness = ChromosomeWithFitness::from_chromosome_and_fitness(
                chromosome.clone(),
                fitness.floor() as usize,
            );
            let serialized = serde_json::to_string(&chromosome_with_fitness).unwrap();

            channel
                .basic_publish(
                    "",
                    &amqp_config.chromosome_with_fitness_queue_name,
                    BasicPublishOptions::default(),
                    serialized.as_bytes(),
                    BasicProperties::default(),
                )
                .await
                .unwrap()
                .await
                .unwrap();

            delivery
                .ack(BasicAckOptions::default())
                .await
                .expect("Chromosome with fitness ACK fail");
        }
    }
}