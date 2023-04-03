use std::borrow::BorrowMut;
use std::collections::HashMap;
use std::str::from_utf8;
use std::sync::Arc;

use futures::stream::StreamExt;
use lapin::Channel;
use lapin::{
    message::DeliveryResult,
    options::{BasicAckOptions, BasicConsumeOptions, BasicPublishOptions, QueueDeclareOptions},
    types::FieldTable,
    BasicProperties, Connection, ConnectionProperties,
};
use log::{error, info, trace};

use crate::common::*;
use crate::config::amqp_config::AmqpConfig;
use crate::config::game_config::GameConfig;
use crate::config::smart_network_config::SmartNetworkConfig;
use crate::evolution::chromosome::Chromosome;
use crate::evolution::chromosome_with_fitness::ChromosomeWithFitness;
use crate::game::level::Level;
use crate::smart_network::smart_network::SmartNetwork;
use crate::smart_network_game_adapter::play_game_with_network;

pub async fn fitness_calc_node_loop(
    channel: Arc<Channel>,
    smart_network_config: Arc<SmartNetworkConfig>,
    game_config: Arc<GameConfig>,
    amqp_config: Arc<AmqpConfig>,
) {
    info!("Starting fitness calculation processing...");

    let consumer = channel
        .basic_consume(
            &amqp_config.chromosome_queue_name,
            "fitness",
            BasicConsumeOptions::default(),
            FieldTable::default(),
        )
        .await
        .unwrap();

    let level_paths = (1..4).map(|lvl| {
        vec![
            game_config.levels_dir_path.clone(),
            "level_".to_string(),
            lvl.to_string(),
            ".lvl".to_string(),
        ]
        .concat()
    });

    let levels: Vec<Level> = level_paths
        .map(|path| Level::from_lvl_file(&path, game_config.max_steps))
        .collect();

    let all_points_amount: usize = levels.iter().map(|lvl| lvl.get_point_amount()).sum();

    info!(
        "Loaded {} levels with total amount of points: {}",
        levels.len(),
        all_points_amount
    );

    consumer.set_delegate(move |delivery: DeliveryResult| {
        let channel_clone = channel.clone();
        let smart_network_config_clone = smart_network_config.clone();
        let game_config_clone = game_config.clone();
        let amqp_config_clone = amqp_config.clone();

        let levels_clone = levels.clone();

        async move {
            let delivery = match delivery {
                Ok(Some(delivery)) => delivery,
                Ok(None) => return,
                Err(error) => {
                    error!("Failed to consume queue message {}", error);
                    return;
                }
            };

            trace!("Received message: {:?}", delivery);

            let utf8_payload = from_utf8(delivery.data.as_slice()).unwrap();
            let chromosome = serde_json::from_str::<Chromosome>(utf8_payload).unwrap();

            //TODO: Take this from config
            let levels_idxs_to_times_to_play = HashMap::from([(1, 50), (2, 10), (3, 10)]);

            //TODO: Refactor this
            let results: Vec<usize> = play_levels(
                levels_idxs_to_times_to_play,
                &chromosome,
                &smart_network_config_clone,
                &game_config_clone,
                &levels_clone,
            );

            let results_sum: usize = results.iter().sum();
            let results_len = results.len();
            //Use max or average?
            let fitness = results_sum as f64 / results_len as f64;

            let chromosome_with_fitness = ChromosomeWithFitness::from_chromosome_and_fitness(
                chromosome.clone(),
                fitness.floor() as usize,
            );
            let serialized = serde_json::to_string(&chromosome_with_fitness).unwrap();

            channel_clone
                .basic_publish(
                    "",
                    &amqp_config_clone.chromosome_with_fitness_queue_name,
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
    });

    loop {}
}

fn play_levels(
    level_idxs_to_times: HashMap<usize, usize>,
    chromosome: &Chromosome,
    smart_network_config: &SmartNetworkConfig,
    game_config: &GameConfig,
    levels: &Vec<Level>,
) -> Vec<usize> {
    let mut smart_network = SmartNetwork::from_bitstring(
        &bit_vector_to_bitstring(&chromosome.genes),
        smart_network_config.input_count,
        smart_network_config.output_count,
        smart_network_config.nand_count_bits,
        smart_network_config.mem_addr_bits,
        smart_network_config.mem_rw_bits,
    );

    level_idxs_to_times
        .iter()
        .map(|(&level_idx, &times)| {
            (0..times)
                .map(|_| {
                    play_game_with_network(
                        &mut smart_network,
                        levels[level_idx - 1].clone(),
                        game_config.visibility_distance,
                    )
                })
                .collect::<Vec<usize>>()
        })
        .flatten()
        .collect::<Vec<usize>>()
}
