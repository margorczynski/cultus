use crate::game::game_object::GameObject;
use crate::game::game_state::GameState::{Finished, InProgress};
use super::level::Level;

pub enum GameState {
    InProgress(Level, usize, usize),
    Finished(usize)
}

impl GameState {

    pub fn from_initial_level(initial_level: Level) -> GameState {
        InProgress(initial_level, 0, 0)
    }

    pub fn next_state(&self, game_action: GameAction) -> GameState {
        match self {
            InProgress(current_level, current_step, current_points) => {
                if *current_step >= current_level.max_steps {
                    return Finished(*current_points)
                }

                let mut updated_level = current_level.clone();
                let (row_delta, column_delta) = game_action.get_deltas();

                let game_object_at_new_position = updated_level.move_player_by(row_delta, column_delta);

                let new_points = match game_object_at_new_position {
                    None => *current_points,
                    Some(game_object) => {
                        match game_object {
                            GameObject::Reward(amount) => {
                                *current_points + (amount as usize)
                            }
                            _ => *current_points
                        }
                    }
                };

                if current_level.get_point_amount() == 0 {
                    let bonus_points = current_level.max_steps - current_step;
                    Finished(new_points + bonus_points)
                } else {
                    InProgress(updated_level, current_step + 1, new_points )
                }
            }
            Finished(final_points) => Finished(*final_points)
        }
    }
}

pub enum GameAction {
    MoveUp,
    MoveDown,
    MoveRight,
    MoveLeft
}

impl GameAction {
    pub fn get_deltas(&self) -> (i32, i32) {
        match *self {
            GameAction::MoveUp => (-1, 0),
            GameAction::MoveDown => (1, 0),
            GameAction::MoveRight => (0, 1),
            GameAction::MoveLeft => (0, -1),
        }
    }
}