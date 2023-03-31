use std::borrow::Borrow;
use crate::game::game_action::GameAction;
use crate::game::game_action::GameAction::{MoveDown, MoveLeft, MoveRight, MoveUp};
use crate::game::game_state::GameState;
use crate::game::game_state::GameState::Finished;
use crate::game::level::*;
use crate::smart_network::smart_network::SmartNetwork;
use super::game::game_object::*;


fn play_game_with_network(smart_network: &mut SmartNetwork, initial_level: Level, visibility_distance: usize) -> usize {

    let mut current_game_state = GameState::from_initial_level(initial_level);

    loop {
        match current_game_state.borrow() {
            in_progress @ GameState::InProgress(current_level, current_step, current_points) => {
                let state_bit_vector = game_state_to_bit_vector(&in_progress, visibility_distance).unwrap();
                let smart_network_output = smart_network.compute_output(state_bit_vector.as_slice());
                let smart_network_output_as_action = game_action_from_bit_vector(&smart_network_output).unwrap();

                current_game_state = current_game_state.next_state(smart_network_output_as_action);
            }
            Finished(final_points) => {
                return *final_points;
            }
        }
    }
}

fn game_action_to_bitstring(game_action: GameAction) -> String {
    let str_res = match game_action {
        GameAction::MoveUp => "00",
        GameAction::MoveDown => "01",
        GameAction::MoveRight => "10",
        GameAction::MoveLeft => "11"
    };

    String::from(str_res)
}

fn game_action_from_bit_vector(bit_vector: &Vec<bool>) -> Option<GameAction> {
    match bit_vector.as_slice() {
        [false, false] => Some(MoveUp),
        [false, true] => Some(MoveDown),
        [true, false] => Some(MoveRight),
        [true, true] => Some(MoveLeft),
        _ => None
    }
}

fn game_object_to_bitstring(game_object: Option<&GameObject>) -> String {
    //Max reward is 9 - 5 bits, if first bit is 1 then it encodes reward value
    match game_object {
        None => String::from("00000"),
        Some(go) => {
            match go {
                GameObject::Player => String::from("00001"),
                GameObject::Wall => String::from("00010"),
                GameObject::Reward(amount) => {
                    let amount_binary = format!("{amount:b}");
                    format!("1{amount_binary:0>4}")
                }
            }
        }
    }
}

fn bitstring_to_bit_vector(bitstring: &str) -> Vec<bool> {
    bitstring.chars().map(|c| if c == '0' { false } else { true } ).collect()
}

fn game_state_to_bit_vector(game_state: &GameState, visibility_distance: usize) -> Option<Vec<bool>> {
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
                        visible_objects.get(&Position {row: row_cast, column: column_cast}).cloned()
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
        GameState::Finished(_) => None
    }
}

#[cfg(test)]
mod smart_network_game_adapter_tests {
    use log::debug;
    use super::*;
    use crate::common::*;
    use crate::game::game_object::GameObject::*;

    #[test]
    fn play_game_with_network_test() {
        setup();

        let input_count = 149; //12 + 12 + 125 = 149, 8 bits
        let output_count = 2; //2 bits

        let network_str = [
            "1100", //12 NANDs
            "000000000000000000", //I0 -> O0
            "010000000000000000", //I0 -> NAND0
            "010000000100000000", //I1 -> NAND0
            "100000000000000001" //NAND0 -> O1
        ].concat();

        let test_str: &str =
            "........\n\
            2...@.##\n\
            #4..3#..\n\
            8..###..";

        let mut smart_network = SmartNetwork::from_bitstring(&network_str, input_count, output_count, 4, 16, 64);
        let level = Level::from_string(&test_str, 2);

        let result = play_game_with_network(&mut smart_network, level, 2);

        assert_eq!(result, 3);
    }

    #[test]
    fn game_action_to_bitstring_test() {
        setup();
        assert_eq!(game_action_to_bitstring(GameAction::MoveUp), "00");
        assert_eq!(game_action_to_bitstring(GameAction::MoveDown), "01");
        assert_eq!(game_action_to_bitstring(GameAction::MoveRight), "10");
        assert_eq!(game_action_to_bitstring(GameAction::MoveLeft), "11");
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
        assert_eq!(bitstring_to_bit_vector("00100"), vec![false, false, true, false, false]);
        assert_eq!(bitstring_to_bit_vector("00111"), vec![false, false, true, true, true]);
        assert_eq!(bitstring_to_bit_vector("010101010101"), vec![false, true, false, true, false, true, false, true, false, true, false, true]);
    }

    #[test]
    fn game_state_to_bit_vector_test() {
        setup();

        let test_str: &str =
            "........\n\
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
        ].concat();

        let test_level = Level::from_string(&test_str, 2);

        let initial_game_state = GameState::from_initial_level(test_level.clone());
        let finished_game_state = GameState::Finished(0);

        let result_in_progress = game_state_to_bit_vector(&initial_game_state, 2).unwrap();
        let result_finished = game_state_to_bit_vector(&finished_game_state, 2);

        assert_eq!(bitstring_to_bit_vector(&expected_bitstring), result_in_progress);
        assert_eq!(None, result_finished);
    }
}