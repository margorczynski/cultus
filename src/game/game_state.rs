use log::debug;

use crate::game::game_action::GameAction;
use crate::game::game_object::GameObject;
use crate::game::game_state::GameState::{Finished, InProgress};

use super::level::Level;

#[derive(PartialEq, Eq, Clone, Debug)]
pub enum GameState {
    InProgress(Level, usize, usize),
    Finished(usize),
}

impl GameState {
    pub fn from_initial_level(initial_level: Level) -> GameState {
        InProgress(initial_level, 0, 0)
    }

    pub fn next_state(&self, game_action: GameAction) -> GameState {
        match self {
            InProgress(current_level, current_step, current_points) => {
                debug!(
                    "next_state for current InProgress - current_step: {}, current_points: {}",
                    *current_step, *current_points
                );

                if *current_step >= current_level.max_steps {
                    return Finished(*current_points);
                }

                let mut updated_level = current_level.clone();
                let (row_delta, column_delta) = game_action.get_deltas();

                let game_object_at_new_position =
                    updated_level.move_player_by(row_delta, column_delta);

                let new_points = match game_object_at_new_position {
                    None => *current_points,
                    Some(game_object) => match game_object {
                        GameObject::Reward(amount) => *current_points + (amount as usize),
                        _ => *current_points,
                    },
                };

                let new_current_step = current_step + 1;

                debug!(
                    "Level points left: {}, new_points: {}",
                    current_level.get_point_amount(),
                    new_points
                );

                if updated_level.get_point_amount() == 0 {
                    let bonus_points = current_level.max_steps - new_current_step;
                    debug!("Game won - bonus_points: {}", bonus_points);
                    Finished(new_points + bonus_points)
                } else {
                    InProgress(updated_level, new_current_step, new_points)
                }
            }
            Finished(final_points) => Finished(*final_points),
        }
    }
}

#[cfg(test)]
mod game_state_tests {
    use crate::common::*;

    use super::*;

    static TEST_STR: &str = "........\n\
            2...@.##\n\
            #4..3#..\n\
            8..###..";

    #[test]
    fn next_state_too_many_moves_test() {
        setup();

        let max_steps = 15;

        let test_level = Level::from_string(TEST_STR, max_steps);

        let mut exhausted_current_game_state = GameState::from_initial_level(test_level.clone());
        for _ in 0..(max_steps + 1) {
            exhausted_current_game_state =
                exhausted_current_game_state.next_state(GameAction::MoveUp)
        }

        assert_eq!(exhausted_current_game_state, GameState::Finished(0));
    }

    #[test]
    fn next_state_win_test() {
        setup();

        let max_steps = 15;

        let test_level = Level::from_string(TEST_STR, max_steps);

        //4 * left, right, down, down, left, right, up, right, right, right
        let won_current_game_state = GameState::from_initial_level(test_level.clone())
            .next_state(GameAction::MoveLeft)
            .next_state(GameAction::MoveLeft)
            .next_state(GameAction::MoveLeft)
            .next_state(GameAction::MoveLeft)
            .next_state(GameAction::MoveRight)
            .next_state(GameAction::MoveDown)
            .next_state(GameAction::MoveDown)
            .next_state(GameAction::MoveLeft)
            .next_state(GameAction::MoveRight)
            .next_state(GameAction::MoveUp)
            .next_state(GameAction::MoveRight)
            .next_state(GameAction::MoveRight)
            .next_state(GameAction::MoveRight);

        assert_eq!(won_current_game_state, GameState::Finished(19));
    }
}
