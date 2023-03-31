use std::collections::HashMap;
use std::fs;

use log::debug;

use super::game_object::*;

#[derive(Hash, PartialEq, Eq, PartialOrd, Ord, Clone, Debug)]
pub struct Position {
    pub row: usize,
    pub column: usize
}

#[derive(PartialEq, Eq, Clone, Debug)]
pub struct Level {
    position_to_game_object_map: HashMap<Position, GameObject>,
    pub max_steps: usize
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

    pub fn from_lvl_file(path: &str, max_steps: usize) -> Level {
        let lvl_string = fs::read_to_string(path).unwrap();
        Level::from_string(&lvl_string, max_steps)
    }

    pub fn get_size_rows(&self) -> usize {
        self.position_to_game_object_map.iter().map(|e| e.0.row).max().unwrap() + 1
    }

    pub fn get_size_column(&self) -> usize {
        self.position_to_game_object_map.iter().map(|e| e.0.column).max().unwrap() + 1
    }

    pub fn get_point_amount(&self) -> usize {
        self.position_to_game_object_map.iter().map(|e| match e.1 {
            GameObject::Player => 0,
            GameObject::Wall => 0,
            GameObject::Reward(amount) => *amount as usize
        }).sum()
    }

    pub fn get_game_object_at(&self, position: &Position) -> Option<&GameObject> {
        self.position_to_game_object_map.get(&position)
    }

    pub fn move_player_by(&mut self, row_delta: i32, column_delta: i32) -> Option<GameObject> {
        let old_position = self.get_player_position();

        debug!("Move player - row_delta={}, column_delta={}, old_position={:?}", row_delta, column_delta, old_position);

        let new_row = (old_position.row as i32) + row_delta;
        let new_column = (old_position.column as i32) + column_delta;

        if new_row < 0 || new_column < 0 || new_row as usize >= self.get_size_rows() || new_column as usize >= self.get_size_column() {
            debug!("Trying to move out of bound - new_row={}, new_column={}", new_row, new_column);
            return None;
        }

        let new_position = Position { row: new_row as usize, column: new_column as usize };

        if self.get_game_object_at(&new_position).cloned() == Some(GameObject::Wall) {
            debug!("Trying to move to position occupied by tree - new_position={:?}", new_position);
            return None;
        }

        self.remove_game_object_at(&old_position);
        self.update_game_object_at(&new_position, &GameObject::Player)
    }

    pub fn get_objects_visible_by_player(&self, visibility_distance: usize) -> HashMap<&Position, &GameObject> {
        let player_position = self.get_player_position();

        self.position_to_game_object_map
            .iter()
            .filter(|(pos, _)| (pos.row as i32 - player_position.row as i32).abs() <= visibility_distance as i32 && (pos.column as i32 - player_position.column as i32).abs() <= visibility_distance as i32)
            .collect()
    }

    pub fn get_player_position(&self) -> Position {
        self.position_to_game_object_map.iter().find(|(_, &ref go)| *go == GameObject::Player).map(|e| e.0).unwrap().clone()
    }

    fn update_game_object_at(&mut self, position: &Position, new_game_object: &GameObject) -> Option<GameObject> {
        self.position_to_game_object_map.insert(position.clone(), new_game_object.clone())
    }

    fn remove_game_object_at(&mut self, position: &Position) -> bool {
        self.position_to_game_object_map.remove(&position).is_some()
    }
}

#[cfg(test)]
mod level_tests {
    use crate::common::*;
    use crate::game::game_object::GameObject::*;

    use super::*;

    static TEST_STR: &str =
           "......\n\
            ....@.\n\
            4#...#\n\
            8..###";

    #[test]
    fn from_string_test() {
        setup();

        let result = Level::from_string(TEST_STR, 100);

        let expected_position_to_game_objects = HashMap::from_iter(vec![
            (Position{ row: 1, column: 4 }, Player),
            (Position{ row: 2, column: 0 }, Reward(4)),
            (Position{ row: 2, column: 1 }, Wall),
            (Position{ row: 2, column: 5 }, Wall),
            (Position{ row: 3, column: 0 }, Reward(8)),
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

    #[test]
    fn from_lvl_file_test() {
        setup();

        let first_level_file_path = concat!(env!("CARGO_MANIFEST_DIR"), "/levels/level_1.lvl");

        let result = Level::from_lvl_file(first_level_file_path, 50);

        assert_eq!(result.position_to_game_object_map.len(), 55);
    }

    #[test]
    fn get_size_rows_test() {
        setup();

        let level = Level::from_string(TEST_STR, 100);

        let result = level.get_size_rows();

        assert_eq!(result, 4);
    }

    #[test]
    fn get_size_columns_test() {
        setup();

        let level = Level::from_string(TEST_STR, 100);

        let result = level.get_size_column();

        assert_eq!(result, 6);
    }

    #[test]
    fn get_point_amount_test() {
        setup();

        let level = Level::from_string(TEST_STR, 100);

        let result = level.get_point_amount();

        assert_eq!(result, 12);
    }

    #[test]
    fn get_game_object_at_test() {
        setup();

        let level = Level::from_string(TEST_STR, 100);

        let result_1 = level.get_game_object_at(&Position { row: 0, column: 0 });
        let result_2 = level.get_game_object_at(&Position { row: 1, column: 4 });
        let result_3 = level.get_game_object_at(&Position { row: 3, column: 0 });

        assert_eq!(result_1, None);
        assert_eq!(result_2.cloned(), Some(Player));
        assert_eq!(result_3.cloned(), Some(Reward(8)));
    }

    #[test]
    fn update_game_object_at_test() {
        setup();

        let mut level = Level::from_string(TEST_STR, 100);

        let position_1 = Position { row: 0, column: 0 };
        let position_2 = Position { row: 1, column: 4 };

        let result_1 = level.update_game_object_at(&position_1, &Reward(5));
        let result_2 = level.update_game_object_at(&position_2, &Wall);

        assert_eq!(result_1, None);
        assert_eq!(result_2, Some(Player));

        assert_eq!(level.get_game_object_at(&position_1).cloned(), Some(Reward(5)));
        assert_eq!(level.get_game_object_at(&position_2).cloned(), Some(Wall));
    }

    #[test]
    fn remove_game_object_at_test() {
        setup();

        let mut level = Level::from_string(TEST_STR, 100);

        let position_1 = Position { row: 0, column: 0 };
        let position_2 = Position { row: 1, column: 4 };

        let result_1 = level.remove_game_object_at(&position_1);
        let result_2 = level.remove_game_object_at(&position_2);

        assert!(!result_1);
        assert!(result_2);

        assert_eq!(level.get_game_object_at(&position_1), None);
        assert_eq!(level.get_game_object_at(&position_2), None);
    }

    #[test]
    fn move_player_by_test() {
        setup();

        let mut level = Level::from_string(TEST_STR, 100);

        let result_1 = level.move_player_by(0, 2);
        let result_2 = level.move_player_by(1, 0);
        let result_3 = level.move_player_by(0, 1);
        let result_4 = level.move_player_by(0, -4);

        assert_eq!(result_1, None);
        assert_eq!(result_2, None);
        assert_eq!(result_3, None);
        assert_eq!(result_4, Some(Reward(4)));
        assert_eq!(level.get_game_object_at(&Position { row: 2, column: 0 }).cloned(), Some(Player));
    }

    #[test]
    fn get_objects_visible_by_player_test() {
        setup();

        let visibility_distance = 2;
        let mut level = Level::from_string(TEST_STR, 100);

        let expected_objects_1 = HashMap::from_iter(vec![
            (Position{ row: 0, column: 0 }, Player),
            (Position{ row: 2, column: 0 }, Reward(4)),
            (Position{ row: 2, column: 1 }, Wall)
        ]);
        let expected_objects_2 = HashMap::from_iter(vec![
            (Position{ row: 1, column: 0 }, Player),
            (Position{ row: 2, column: 0 }, Reward(4)),
            (Position{ row: 2, column: 1 }, Wall),
            (Position{ row: 3, column: 0 }, Reward(8)),
        ]);
        let expected_objects_3 = HashMap::from_iter(vec![
            (Position{ row: 2, column: 2 }, Player),
            (Position{ row: 2, column: 0 }, Reward(4)),
            (Position{ row: 2, column: 1 }, Wall),
            (Position{ row: 3, column: 0 }, Reward(8)),
            (Position{ row: 3, column: 3 }, Wall),
            (Position{ row: 3, column: 4 }, Wall),
        ]);

        level.move_player_by(-1, -4);
        let result_1: HashMap<Position, GameObject> = level.get_objects_visible_by_player(visibility_distance).iter().map(|(&p, &o)| (p.clone(), o.clone())).collect();

        level.move_player_by(1, 0);
        let result_2: HashMap<Position, GameObject> = level.get_objects_visible_by_player(visibility_distance).iter().map(|(&p, &o)| (p.clone(), o.clone())).collect();

        level.move_player_by(1, 2);
        let result_3: HashMap<Position, GameObject> = level.get_objects_visible_by_player(visibility_distance).iter().map(|(&p, &o)| (p.clone(), o.clone())).collect();

        assert_eq!(result_1, expected_objects_1);
        assert_eq!(result_2, expected_objects_2);
        assert_eq!(result_3, expected_objects_3);
    }
}