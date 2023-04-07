use std::borrow::BorrowMut;
use std::collections::HashMap;
use std::str::from_utf8;
use std::sync::Arc;

use futures::stream::StreamExt;
use lapin::{
    message::DeliveryResult,
    options::{BasicAckOptions, BasicConsumeOptions, BasicPublishOptions, QueueDeclareOptions},
    types::FieldTable,
    BasicProperties, Connection, ConnectionProperties,
};
use lapin::options::BasicQosOptions;
use log::{error, info, trace};
use tokio::time::Instant;

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
    connection: &Connection,
    smart_network_config: Arc<SmartNetworkConfig>,
    game_config: Arc<GameConfig>,
    amqp_config: Arc<AmqpConfig>,
) {
    info!("Starting fitness calculation processing...");

    let publish_channel = Arc::new(connection.create_channel().await.unwrap());
    let consume_channel = connection.create_channel().await.unwrap();

    consume_channel.basic_qos(amqp_config.prefetch_count, BasicQosOptions {
        global: true,
    }).await.unwrap();

    let consumer = consume_channel
        .basic_consume(
            &amqp_config.chromosome_queue_name,
            "fitness",
            BasicConsumeOptions::default(),
            FieldTable::default(),
        )
        .await
        .unwrap();

    let level_paths = game_config.level_to_times_to_play.iter().map(|(&lvl_idx, _)| {
        vec![
            game_config.levels_dir_path.clone(),
            "level_".to_string(),
            lvl_idx.to_string(),
            ".lvl".to_string(),
        ]
        .concat()
    });

    let levels: Vec<Level> = level_paths
        .map(|path| Level::from_lvl_file(&path, game_config.max_steps))
        .collect();

    levels.iter().enumerate().for_each(|(idx, lvl)| {
        info!("size_rows={}, size_columns={}, player_pos={:?}, total_points={}", lvl.get_size_rows(), lvl.get_size_column(), lvl.get_player_position() ,lvl.get_point_amount())
    });

    let all_points_amount: usize = levels.iter().map(|lvl| lvl.get_point_amount()).sum();

    consumer.set_delegate(move |delivery: DeliveryResult| {
        let publish_channel_clone = publish_channel.clone();
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

            let results: HashMap<usize, Vec<usize>> = play_levels(
                game_config_clone.level_to_times_to_play.clone(),
                &chromosome,
                &smart_network_config_clone,
                &game_config_clone,
                &levels_clone,
            );

            //Take the averages from the final 20% of plays for each level
            let fitness: f32 = results.iter().map(|(_, results)| {
                let pareto_amount_small = ((results.len() as f32) * 0.2).ceil();
                let sum = results.iter().rev().take(pareto_amount_small as usize).sum::<usize>() as f32 / pareto_amount_small;
                sum / pareto_amount_small
            }).sum::<f32>();

            let chromosome_with_fitness = ChromosomeWithFitness::from_chromosome_and_fitness(
                chromosome.clone(),
                fitness.floor() as usize,
            );
            let serialized = serde_json::to_string(&chromosome_with_fitness).unwrap();

            publish_channel_clone
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
) -> HashMap<usize, Vec<usize>> {

    let smart_network_start = Instant::now();
    let mut smart_network = SmartNetwork::from_bitstring(
        &bit_vector_to_bitstring(&chromosome.genes),
        smart_network_config.input_count,
        smart_network_config.output_count,
        smart_network_config.nand_count_bits,
        smart_network_config.mem_addr_bits,
        smart_network_config.mem_rw_bits,
    );
    trace!("construct_smart_network_elapsed={:?}", smart_network_start.elapsed());
    let playing_start = Instant::now();
    let res = level_idxs_to_times
        .iter()
        .map(|(&level_idx, &times)| {
            let results = (0..times)
                .map(|idx| {
                    let result = play_game_with_network(
                        &mut smart_network,
                        levels[level_idx - 1].clone(),
                        game_config.visibility_distance,
                    );

                    result
                })
                .collect::<Vec<usize>>();

            (level_idx, results)
        })
        .collect::<HashMap<usize, Vec<usize>>>();
    trace!("play_all_games_elapsed={:?}", playing_start.elapsed());
    res
}
