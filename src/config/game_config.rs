use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[allow(unused)]
pub struct GameConfig {
    pub visibility_distance: usize,
    pub max_steps: usize,
    pub levels_dir_path: String,
}
