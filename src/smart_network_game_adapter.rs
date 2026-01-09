use std::borrow::Borrow;
use log::info;

use crate::common::bitstring_to_bit_vector;
use crate::evolution::novelty::{GameTrace, BehavioralSignature};
use crate::game::game_action::GameAction;
use crate::game::game_action::GameAction::{MoveDown, MoveLeft, MoveRight, MoveUp};
use crate::game::game_state::GameState;
use crate::game::game_state::GameState::Finished;
use crate::game::level::*;
use crate::smart_network::smart_network::SmartNetwork;

use super::game::game_object::*;

/// Result of playing a game, containing metrics for fitness evaluation
#[derive(Debug, Clone)]
pub struct GameResult {
    pub points: usize,
    pub steps_taken: usize,
    pub max_steps: usize,
    /// Optional game trace for novelty computation
    pub trace: Option<GameTrace>,
}

impl GameResult {
    /// Calculate efficiency as points per step (higher is better)
    pub fn efficiency(&self) -> f64 {
        if self.steps_taken == 0 {
            0.0
        } else {
            self.points as f64 / self.steps_taken as f64
        }
    }

    /// Check if the game was won (all points collected before max steps)
    pub fn is_win(&self) -> bool {
        self.steps_taken < self.max_steps && self.points > 0
    }

    /// Get the behavioral signature from this result's trace.
    pub fn behavioral_signature(&self) -> Option<BehavioralSignature> {
        self.trace.as_ref().map(BehavioralSignature::from_trace)
    }
}

/// Play a game with the given network and return detailed results
pub fn play_game_with_network(
    smart_network: &mut SmartNetwork,
    initial_level: Level,
    visibility_distance: usize,
    is_game_logged: bool,
) -> GameResult {
    play_game_with_network_traced(smart_network, initial_level, visibility_distance, is_game_logged, false)
}

/// Play a game with the given network and optionally collect a trace for novelty computation.
pub fn play_game_with_network_traced(
    smart_network: &mut SmartNetwork,
    initial_level: Level,
    visibility_distance: usize,
    is_game_logged: bool,
    collect_trace: bool,
) -> GameResult {
    let max_steps = initial_level.max_steps;
    let mut current_game_state = GameState::from_initial_level(initial_level);
    let mut steps_taken = 0;
    let mut trace = if collect_trace { Some(GameTrace::new()) } else { None };

    loop {
        match current_game_state.borrow() {
            in_progress @ GameState::InProgress(current_level, current_step, current_points) => {
                steps_taken = *current_step;
                let state_bit_vector =
                    game_state_to_bit_vector(in_progress, visibility_distance).unwrap();
                let smart_network_output =
                    smart_network.compute_output(state_bit_vector.as_slice());
                let smart_network_output_as_action =
                    game_action_from_bit_vector(&smart_network_output).unwrap();

                // Record trace step if collecting
                if let Some(ref mut game_trace) = trace {
                    let player_pos = current_level.get_player_position();
                    let action_id = game_action_to_id(&smart_network_output_as_action);
                    game_trace.record_step(
                        (player_pos.row as u16, player_pos.column as u16),
                        action_id,
                        *current_points,
                    );
                }

                current_game_state = current_game_state.next_state(&smart_network_output_as_action);

                if is_game_logged {
                    info!("{}", &smart_network_output_as_action);
                    info!("{}", &current_game_state);
                }
            }
            Finished(final_points) => {
                // Finalize trace if collecting
                if let Some(ref mut game_trace) = trace {
                    // Check if won: collected all rewards (we assume max_points is derived from level)
                    let won = *final_points > 0 && steps_taken < max_steps;
                    game_trace.finalize(*final_points, won);
                }

                return GameResult {
                    points: *final_points,
                    steps_taken,
                    max_steps,
                    trace,
                };
            }
        }
    }
}

/// Convert a game action to its ID (0-3).
fn game_action_to_id(action: &GameAction) -> u8 {
    match action {
        MoveUp => 0,
        MoveDown => 1,
        MoveRight => 2,
        MoveLeft => 3,
    }
}

fn game_action_from_bit_vector(bit_vector: &[bool]) -> Option<GameAction> {
    match bit_vector {
        [false, false] => Some(MoveUp),
        [false, true] => Some(MoveDown),
        [true, false] => Some(MoveRight),
        [true, true] => Some(MoveLeft),
        _ => None,
    }
}

fn game_object_to_bitstring(game_object: Option<&GameObject>) -> String {
    //Max reward is 9 - 5 bits, if first bit is 1 then it encodes reward value
    match game_object {
        None => String::from("00000"),
        Some(go) => match go {
            GameObject::Player => String::from("00001"),
            GameObject::Wall => String::from("00010"),
            GameObject::Reward(amount) => {
                let amount_binary = format!("{amount:b}");
                format!("1{amount_binary:0>4}")
            }
        },
    }
}

fn game_state_to_bit_vector(
    game_state: &GameState,
    visibility_distance: usize,
) -> Option<Vec<bool>> {
    match game_state {
        GameState::InProgress(current_level, current_step, current_points) => {
            let visible_objects = current_level.get_objects_visible_by_player(visibility_distance);
            let player_position = current_level.get_player_position();
            let mut step_score_game_objects_bit_strings: Vec<String> = Vec::new();

            //12 bit (4095 max)
            let current_step_binary = format!("{current_step:b}");
            let current_step_bitstring = format!("{current_step_binary:0>12}");

            //12 bit
            let current_points_binary = format!("{current_points:b}");
            let current_points_bitstring = format!("{current_points_binary:0>12}");

            step_score_game_objects_bit_strings.push(current_step_bitstring);
            step_score_game_objects_bit_strings.push(current_points_bitstring);

            let visible_fields_width_height = 2 * visibility_distance + 1;

            let min_row = (player_position.row as i32) - (visibility_distance as i32);
            let min_column = (player_position.column as i32) - (visibility_distance as i32);

            //5*visible_fields_width_height*visible_fields_width_height bits
            for row in min_row..(min_row + visible_fields_width_height as i32) {
                for column in min_column..(min_column + visible_fields_width_height as i32) {
                    let game_object_at = if row >= 0 && column >= 0 {
                        let row_cast = row as usize;
                        let column_cast = column as usize;
                        visible_objects
                            .get(&Position {
                                row: row_cast,
                                column: column_cast,
                            })
                            .cloned()
                    } else {
                        None
                    };

                    let game_object_bitstring = game_object_to_bitstring(game_object_at);

                    step_score_game_objects_bit_strings.push(game_object_bitstring);
                }
            }

            let final_bitstring: String = step_score_game_objects_bit_strings.concat();

            Some(bitstring_to_bit_vector(&final_bitstring))
        }
        GameState::Finished(_) => None,
    }
}

#[cfg(test)]
mod smart_network_game_adapter_tests {
    use crate::common::*;
    use crate::game::game_object::GameObject::*;

    use super::*;

    #[test]
    fn play_game_with_network_test() {
        setup();

        let input_count = 149; //12 + 12 + 125 = 149, 8 bits
        let output_count = 2; //2 bits

        let network_str = [
            "1100",               //12 NANDs
            "000000000000000000", //I0 -> O0
            "010000000000000000", //I0 -> NAND0
            "010000000100000000", //I1 -> NAND0
            "100000000000000001", //NAND0 -> O1
        ]
        .concat();

        let test_str: &str = "........\n\
            2...@.##\n\
            #4..3#..\n\
            8..###..";

        let mut smart_network =
            SmartNetwork::from_bitstring(&network_str, input_count, output_count, 4, 16, 64);
        let level = Level::from_string(&test_str, 5);

        let result = play_game_with_network(&mut smart_network, level, 2, false);

        assert_eq!(result.points, 3);
    }

    #[test]
    fn game_result_efficiency_test() {
        setup();

        let result = GameResult {
            points: 100,
            steps_taken: 10,
            max_steps: 30,
            trace: None,
        };

        assert_eq!(result.efficiency(), 10.0);
        assert!(result.is_win());
    }

    #[test]
    fn game_object_to_bitstring_test() {
        setup();
        assert_eq!(game_object_to_bitstring(None), "00000");
        assert_eq!(game_object_to_bitstring(Some(&Player)), "00001");
        assert_eq!(game_object_to_bitstring(Some(&Wall)), "00010");
        assert_eq!(game_object_to_bitstring(Some(&Reward(0))), "10000");
        assert_eq!(game_object_to_bitstring(Some(&Reward(5))), "10101");
        assert_eq!(game_object_to_bitstring(Some(&Reward(9))), "11001");
    }

    #[test]
    fn bitstring_to_bit_vector_test() {
        setup();
        assert_eq!(
            bitstring_to_bit_vector("00100"),
            vec![false, false, true, false, false]
        );
        assert_eq!(
            bitstring_to_bit_vector("00111"),
            vec![false, false, true, true, true]
        );
        assert_eq!(
            bitstring_to_bit_vector("010101010101"),
            vec![false, true, false, true, false, true, false, true, false, true, false, true]
        );
    }

    #[test]
    fn game_state_to_bit_vector_test() {
        setup();

        let test_str: &str = "........\n\
            2...@.##\n\
            #4..3#..\n\
            8..###..";

        let expected_bitstring = vec![
            "000000000000",
            "000000000000",
            //First row (out of map)
            "00000",
            "00000",
            "00000",
            "00000",
            "00000",
            //Second row
            "00000",
            "00000",
            "00000",
            "00000",
            "00000",
            //Third row
            "00000",
            "00000",
            "00001",
            "00000",
            "00010",
            //Fourth row
            "00000",
            "00000",
            "10011",
            "00010",
            "00000",
            //Fifth row
            "00000",
            "00010",
            "00010",
            "00010",
            "00000",
        ]
        .concat();

        let test_level = Level::from_string(&test_str, 2);

        let initial_game_state = GameState::from_initial_level(test_level.clone());
        let finished_game_state = GameState::Finished(0);

        let result_in_progress = game_state_to_bit_vector(&initial_game_state, 2).unwrap();
        let result_finished = game_state_to_bit_vector(&finished_game_state, 2);

        assert_eq!(
            bitstring_to_bit_vector(&expected_bitstring),
            result_in_progress
        );
        assert_eq!(None, result_finished);
    }
}
