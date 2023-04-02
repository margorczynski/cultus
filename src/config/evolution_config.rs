use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[allow(unused)]
pub struct EvolutionConfig {
    pub initial_population_count: usize,
    pub tournament_size: usize,
    pub mutation_rate: f32,
    pub elite_factor: f32,
}
