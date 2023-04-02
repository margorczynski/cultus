extern crate core;

use std::borrow::Borrow;
use std::collections::HashSet;
use std::time::Instant;

use log::info;
use rayon::prelude::*;
use lapin::{
    message::DeliveryResult,
    options::{BasicAckOptions, BasicConsumeOptions, BasicPublishOptions, QueueDeclareOptions},
    types::FieldTable,
    BasicProperties, Connection, ConnectionProperties,
};

use common::setup;
use crate::config::cultus_config::CultusConfig;
use game::level::Level;
use crate::node::evolution_node::{evolution_node_loop, evolution_publish_initial_population};
use crate::node::fitness_calc_node::fitness_calc_node_loop;
//use node::evolution_node::evolution_node_loop;

mod common;
mod config;
mod evolution;
mod game;
mod smart_network;
mod smart_network_game_adapter;
mod node;


#[tokio::main]
async fn main() {
    setup();

    let config = CultusConfig::new().unwrap();

    let evolution_config = config.evolution;
    let smart_network_config = config.smart_network;
    let game_config = config.game;

    //TODO: Move to config
    let uri = "amqp://127.0.0.1:5672";
    let options = ConnectionProperties::default()
        .with_executor(tokio_executor_trait::Tokio::current())
        .with_reactor(tokio_reactor_trait::Tokio);

    let connection = Connection::connect(uri, options).await.unwrap();
    let channel = connection.create_channel().await.unwrap();

    let chromosomes_queue = channel
        .queue_declare(
            "chromosomes",
            QueueDeclareOptions::default(),
            FieldTable::default(),
        )
        .await
        .unwrap();

    let chromosomes_with_fitness_queue = channel
        .queue_declare(
            "chromosomes_with_fitness",
            QueueDeclareOptions::default(),
            FieldTable::default(),
        )
        .await
        .unwrap();

    info!("Chromosome queue message count: {}", chromosomes_queue.message_count());
    info!("Chromosome with fitness queue message count: {}", chromosomes_with_fitness_queue.message_count());

    if config.mode == "evolution" {
        if chromosomes_queue.message_count() == 0 && chromosomes_with_fitness_queue.message_count() == 0 {
            evolution_publish_initial_population(&channel, &smart_network_config, &evolution_config).await;
        }

        evolution_node_loop(&channel, &smart_network_config, &evolution_config).await;
    } else {
        fitness_calc_node_loop(&channel, &smart_network_config, &game_config).await;
    }
}
