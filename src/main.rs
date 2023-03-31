extern crate core;

mod smart_network;
mod evolution;
mod common;
mod game;
mod smart_network_game_adapter;

use smart_network::smart_network::*;
use evolution::evolution::generate_initial_population;

fn main() {
    let smart_network_bitstring_len = SmartNetwork::get_required_bits_for_bitstring(149, 2, 8, 8, 16, 5000);
    let mut population = generate_initial_population(100, smart_network_bitstring_len);
/*
    loop {

        for chromosome in population {

            let smart_network = SmartNetwork::from_bitstring(chromosome.)
        }
    }*/
}
