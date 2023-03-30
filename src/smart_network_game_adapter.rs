use crate::game::game_state::GameState;
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
            let mut position_to_object_vec: Vec<_> = current_level.get_objects_visible_by_player(visibility_distance).into_iter().collect();

            position_to_object_vec.sort_by(|pair_left, pair_right| pair_left.clone().0.cmp(pair_right.clone().0));

            Some(vec![true])
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
    fn to_bitstring_test() {
        setup();
        assert_eq!(game_object_to_bitstring(None), "00000");
        assert_eq!(game_object_to_bitstring(Some(&Player)), "00001");
        assert_eq!(game_object_to_bitstring(Some(&Wall)), "00010");
        assert_eq!(game_object_to_bitstring(Some(&Reward(0))), "10000");
        assert_eq!(game_object_to_bitstring(Some(&Reward(5))), "10101");
        assert_eq!(game_object_to_bitstring(Some(&Reward(9))), "11001");
    }
}