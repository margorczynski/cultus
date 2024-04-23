extern crate core;



use std::sync::Arc;


use lapin::{
    options::{QueueDeclareOptions},
    types::FieldTable, Connection, ConnectionProperties,
};


use log::info;


use common::setup;


use crate::config::cultus_config::CultusConfig;
use crate::node::evolution_node::{evolution_node_loop, evolution_publish_initial_population};
use crate::node::fitness_calc_node::fitness_calc_node_loop;

//use node::evolution_node::evolution_node_loop;

mod common;
mod config;
mod evolution;
mod game;
mod node;
mod smart_network;
mod smart_network_game_adapter;

#[tokio::main]
async fn main() {
    setup();

    let config = CultusConfig::new().unwrap();

    let evolution_config = config.evolution;
    let smart_network_config = config.smart_network;
    let game_config = config.game;
    let amqp_config = config.amqp;

    let rabbit_uri = format!("amqp://{}:{}@{}:{}/{}", amqp_config.username, amqp_config.password, amqp_config.host, amqp_config.port, amqp_config.vhost);

    let options = ConnectionProperties::default()
        .with_executor(tokio_executor_trait::Tokio::current())
        .with_reactor(tokio_reactor_trait::Tokio);

    let connection = Connection::connect(&rabbit_uri, options)
        .await
        .unwrap();

    let channel = connection.create_channel().await.unwrap();

    let queue_declare_options = QueueDeclareOptions {
        passive: false,
        durable: true,
        exclusive: false,
        auto_delete: false,
        nowait: false,
    };

    let chromosomes_queue = channel
        .queue_declare(
            &amqp_config.chromosome_queue_name,
            queue_declare_options,
            FieldTable::default(),
        )
        .await
        .unwrap();

    let chromosomes_with_fitness_queue = channel
        .queue_declare(
            &amqp_config.chromosome_with_fitness_queue_name,
            queue_declare_options,
            FieldTable::default(),
        )
        .await
        .unwrap();

    info!(
        "Chromosome queue message count: {}",
        chromosomes_queue.message_count()
    );
    info!(
        "Chromosome with fitness queue message count: {}",
        chromosomes_with_fitness_queue.message_count()
    );

    if config.mode == "evolution" {
        if chromosomes_queue.message_count() == 0
            && chromosomes_with_fitness_queue.message_count() == 0
        {
            evolution_publish_initial_population(
                &connection,
                &smart_network_config,
                &evolution_config,
                &amqp_config,
            )
            .await;
        }

        evolution_node_loop(
            &connection,
            &smart_network_config,
            &evolution_config,
            &amqp_config,
        )
        .await;
    } else {
        fitness_calc_node_loop(
            &connection,
            Arc::new(smart_network_config),
            Arc::new(game_config),
            Arc::new(amqp_config),
        )
        .await;
    }
}
