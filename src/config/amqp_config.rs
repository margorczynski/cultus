use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[allow(unused)]
pub struct AmqpConfig {
    pub host: String,
    pub port: u32,
    pub vhost: String,
    pub username: String,
    pub password: String,
    pub chromosome_queue_name: String,
    pub chromosome_with_fitness_queue_name: String,
}
