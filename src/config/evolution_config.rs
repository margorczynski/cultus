use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[allow(unused)]
pub struct EvolutionConfig {
    pub initial_population_count: usize,
    pub tournament_size: usize,
    pub mutation_rate: f32,
    pub elite_factor: f32,
    pub persist_top_chromosome: bool,
    /// Use the new direct network encoding (default: true)
    #[serde(default)]
    pub use_direct_encoding: Option<bool>,
    /// Initial gate count for DirectNetwork (default: 100)
    #[serde(default)]
    pub initial_gate_count: Option<usize>,
    /// Use local search after genetic operations (default: false)
    #[serde(default)]
    pub use_local_search: Option<bool>,
}
