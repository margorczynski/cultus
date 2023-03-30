use crate::game::game_state::GameState;
use super::game::level::Position;
use super::game::game_object::*;


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
    use super::*;
    use crate::common::*;
    use crate::game::game_object::GameObject::*;

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
}