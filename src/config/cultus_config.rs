use config::{Config, ConfigError, Environment, File};
use log::info;
use serde::Deserialize;

use crate::config::evolution_config::EvolutionConfig;
use crate::config::game_config::GameConfig;
use crate::config::smart_network_config::SmartNetworkConfig;


#[derive(Debug, Deserialize)]
#[allow(unused)]
pub struct CultusConfig {
    pub mode: String,
    pub evolution: EvolutionConfig,
    pub smart_network: SmartNetworkConfig,
    pub game: GameConfig,
}

impl CultusConfig {
    pub fn new() -> Result<Self, ConfigError> {
        let s = Config::builder()
            .add_source(File::with_name("config/default"))
            .add_source(Environment::with_prefix("cultus"))
            .build()?;

        info!("Using config: {:?}", s);

        let deserialized: CultusConfig = s.try_deserialize()?;

        if deserialized.mode == "fitness" || deserialized.mode == "evolution" {
            Ok(deserialized)
        } else {
            Err(ConfigError::Message("Mode must be 'evolution' or 'fitness'".to_string()))
        }
    }
}
