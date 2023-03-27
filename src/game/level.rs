use std::collections::HashMap;
use super::game_object::*;

#[derive(Hash, PartialEq, Eq, Clone)]
struct Position {
    row: usize,
    column: usize
}

pub struct Level {
    position_to_game_object_map: HashMap<Position, GameObject>,
    max_steps: usize
}

impl Level {
    pub fn from_string(s: &str, max_steps: usize) -> Level {
        let mut position_to_game_object_map: HashMap<Position, GameObject> = HashMap::new();

        for (row, line) in s.lines().enumerate() {
            for (column, title_char) in line.chars().enumerate() {
                let position = Position {
                    row,
                    column,
                };

                match GameObject::from_char(&title_char) {
                    None => {}
                    Some(game_object) => {
                        position_to_game_object_map.insert(position, game_object);
                    }
                }
            }
        }

        Level {
            position_to_game_object_map,
            max_steps
        }
    }

    pub fn get_size_rows(&self) -> usize {
        self.position_to_game_object_map.iter().map(|e| e.0.row).max().unwrap()
    }

    pub fn get_size_column(&self) -> usize {
        self.position_to_game_object_map.iter().map(|e| e.0.column).max().unwrap()
    }

    pub fn get_point_amount(&self) -> usize {
        self.position_to_game_object_map.iter().map(|e| match e.1 {
            GameObject::Player => 0,
            GameObject::Wall => 0,
            GameObject::Reward(amount) => *amount as usize
        }).sum()
    }
}

