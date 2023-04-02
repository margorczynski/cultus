use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[allow(unused)]
pub struct AmqpConfig {
    pub uri: String,
    pub chromosome_queue_name: String,
    pub chromosome_with_fitness_queue_name: String,
}
