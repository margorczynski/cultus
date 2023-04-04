use std::collections::HashMap;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[allow(unused)]
pub struct GameConfig {
    pub visibility_distance: usize,
    pub max_steps: usize,
    pub levels_dir_path: String,
    pub level_to_times_to_play: HashMap<usize, usize>
}
