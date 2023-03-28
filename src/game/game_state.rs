use crate::game::game_state::GameState::{Finished, InProgress};
use super::level::Level;

pub enum GameState {
    InProgress {
        level: Level,
        score: usize,
        step: usize
    },
    Finished {
        final_score: usize
    }
}

impl GameState {

    pub fn from_initial_level(initial_level: Level) -> GameState {
        InProgress {
            level: initial_level,
            score: 0,
            step: 0,
        }
    }

    pub fn next_state() -> GameState {
        todo!("Implement")
    }
}