# Cultus - distributed GA search for LLNs (Logical Learning Networks)

## Purpose & Idea

Cultus is a framework developed in Rust where the goal is to use an evolution-based search method to find networks composed of NAND gates where the network is paired up with memory to enable learning.

The basic idea is that the NAND gate is functionally complete - meaning any boolean function can be implemented using them and thus, in theory, when paired up with memory a properly constructed network could exhibit the ability to learn new tasks.

Considering the search space even for modest-sized networks is huge and there is no effective gradient to be used a popular metaheuristic algorithm called the Genetic Algorithm was chosen. One of the major factors was also the fact that it is easily parallelizable - given a big enough population and problem size it can easily scale not only to many CPU cores but to many computers.

Here the fitness is calculated against a simple platform game where the goal is to collect the most points using the least amount of moves. Each level can be played multiple times and between each playthrough and each different level the state of the network + memory pair is kept.

The goal of the GA is to find a network which paired up with memory will be able to learn the game after the playthroughs, optimally on a diverse set of levels or even games.

A more detailed description can be found on my blog - https://resethard.io/cultus-framework-to-find-intelligent-logical-circuits/

## Distributed Computation

One of the key features of Cultus is the ability to easily distribute the GA search among different computers within a computer cluster. This is done using a messaging queue - here RabbitMQ was chosen.

The application can be ran in two configurations:
* Fitness - takes the available chromosomes from the chromosome queue and calculates the fitness. Once that is done it sends the result to the chromosome + fitness queue.
* Evolution - takes the available chromosomes with calculated fitness from the queue and process them in batches to generate a new population using the GA method.

## Configuration

The application is configured via the `/config/default.toml` file or environment variables prefixed with `cultus`.

An example config:

```
mode = "fitness"

[evolution]
initial_population_count = 128
tournament_size = 4
mutation_rate = 0.05
elite_factor = 0.1
persist_top_chromosome = false

[smart_network]
input_count = 149
output_count = 2
nand_count_bits = 12
mem_addr_bits = 8
mem_rw_bits = 8
connection_count = 10000

[game]
visibility_distance = 2
max_steps = 30
levels_dir_path = "levels/"
level_to_times_to_play = { 1 = 130 }

[amqp]
host = "0.0.0.0"
port = "5672"
vhost = "cultus"
username = "mac"
password = "pass"
prefetch_count = 30
chromosome_queue_name = "chromosome"
chromosome_with_fitness_queue_name = "chromosome_with_fitness"
```

* mode - fitness/evolution, as explained previously

### evolution

* initial_population_count - the total amount of chromosomes in the population
* tournament_size - the amount of chromosomes that will partake in the tournaments during selection
* mutation_rate - the rate at which mutation happens during crossover
* elite_factor - how much of the top scoring chromosomes will be passed without crossover (0.1 = 10%)
* persist_top_chromosome - if true then save the best performing chromosome of each cycle into a file

### smart_network

* input_count - the amount of inputs the smart network, each input is a bit
* output_count - the amount of outputs of the smart network, each input is a bit
* nand_count_bits - the number of bits to encode the number of NAND gates in the network, e.g. 4 will mean 4 bits will be used and a maximum of 2^4=16 gates can be used in any generated network
* mem_addr_bits - the number of bits that will encode a memory address in the auxiliary memory of the network
* mem_rw_bits - the number of input and outputs of the memory, each being a single bit. Meaning that during a single cycle the network can write 2 bits of data into memory and read 2 bits of data
* connection_count - the maximum amount of connections between elements in the network. This also includes the memory connections and network I/O connections

### game
* visibility_distance - the number of titles the player sees around him. 2 will mean the player will see a maximum of 24 titles around him (5x5 - 1, the one title being the player)
* max_steps - maximum amount of steps made by the player, after this number is reached the game ends.
* levels_dir_path - directory with the level files
level_to_times_to_play - defines the levels to be played by the network along with the number of plays for each level. E.g. 1 = 130 means level 1 will be played 130 times

### amqp
* host - the host on which the AMQP queue is running
* port - the port on which the AMQP queue is running
* vhost - the vhost to be used
* username - the username
* password - the password for the user
* prefetch_count - the amount of messages to be prefetched for processing. Should be the amount of cores available for maximum performance as it will try to process the prefetched messages all at once
* chromosome_queue_name - the name of the queue with the chromosome messages. If it doesn't exist it will be created
* chromosome_with_fitness_queue_name - the name of the queue with the chromosome + fitness messages. If it doesn't exist it will be created


## Running the application

### Prerequistes
* Rust and Cargo installed
* RabbitMQ installed and configuration aligned with the one used above

### Run the application
In the main folder run `cargo run`

### Build the application
In the main folder run `cargo build`

For more information about running and building applications via Cargo please visit https://doc.rust-lang.org/cargo/commands/build-commands.html
