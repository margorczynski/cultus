use std::collections::HashMap;
use super::game_object::*;

#[derive(Hash, PartialEq, Eq, Clone, Debug)]
struct Position {
    row: usize,
    column: usize
}

#[derive(PartialEq, Eq, Clone, Debug)]
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

    pub fn get_game_object_at(&self, position: Position) -> Option<&GameObject> {
        self.position_to_game_object_map.get(&position)
    }

    pub fn update_game_object_at(&mut self, position: &Position, new_game_object: &GameObject) -> bool {
        self.position_to_game_object_map.insert(position.clone(), new_game_object.clone()).is_some()
    }

    pub fn remove_game_object_at(&mut self, position: &Position) -> bool {
        self.position_to_game_object_map.remove(&position).is_some()
    }

    pub fn move_player_by(&mut self, row_delta: i32, column_delta: i32) -> bool {
        let old_position = self.position_to_game_object_map.iter().find(|(_, &ref go)| *go == GameObject::Player).map(|e| e.0).unwrap().clone();
        if (old_position.row as i32) + row_delta < 0 || (old_position.column as i32) + column_delta < 0 {
            return false;
        }
        let new_row = ((old_position.row as i32) + row_delta) as usize;
        let new_column = ((old_position.column as i32) + column_delta) as usize;
        let new_position = Position { row: new_row, column: new_column };

        self.remove_game_object_at(&old_position);
        self.update_game_object_at(&new_position, &GameObject::Player);
        true
    }
}

#[cfg(test)]
mod level_tests {
    use super::*;
    use crate::common::*;
    use crate::game::game_object::GameObject::*;

    #[test]
    fn from_string_test() {
        setup();

        let test_str =
            "......\n\
            ....@.\n\
            ##...#\n\
            ...###";

        let result = Level::from_string(test_str, 100);

        let expected_position_to_game_objects = HashMap::from_iter(vec![
            (Position{ row: 1, column: 4 }, Player),
            (Position{ row: 2, column: 0 }, Wall),
            (Position{ row: 2, column: 1 }, Wall),
            (Position{ row: 2, column: 5 }, Wall),
            (Position{ row: 3, column: 3 }, Wall),
            (Position{ row: 3, column: 4 }, Wall),
            (Position{ row: 3, column: 5 }, Wall)
        ]);

        let expected = Level {
            position_to_game_object_map: expected_position_to_game_objects,
            max_steps: 100,
        };

        assert_eq!(result, expected);
    }
}