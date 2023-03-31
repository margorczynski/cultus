#[derive(Hash, PartialEq, Eq, Clone, Debug)]
pub enum GameObject {
    Player,
    Wall,
    Reward(u8),
}

impl GameObject {
    pub fn from_char(c: &char) -> Option<GameObject> {
        match c {
            '@' => Some(GameObject::Player),
            '#' => Some(GameObject::Wall),
            _ => c.to_digit(10).map(|r| GameObject::Reward(r as u8)),
        }
    }

    #[allow(dead_code)]
    pub fn to_char(&self) -> char {
        match *self {
            GameObject::Player => '@',
            GameObject::Wall => '#',
            GameObject::Reward(amount) => (amount + b'0') as char,
        }
    }
}

#[cfg(test)]
mod game_object_tests {
    use crate::common::*;
    use crate::game::game_object::GameObject::*;

    use super::*;

    #[test]
    fn from_char_test() {
        setup();

        let test_str = "@.##012#.9";

        let result: Vec<Option<GameObject>> = test_str
            .chars()
            .map(|c| GameObject::from_char(&c))
            .collect();

        let expected = vec![
            Some(Player),
            None,
            Some(Wall),
            Some(Wall),
            Some(Reward(0)),
            Some(Reward(1)),
            Some(Reward(2)),
            Some(Wall),
            None,
            Some(Reward(9)),
        ];

        assert_eq!(result, expected);
    }

    #[test]
    fn to_char_test() {
        setup();

        let test_game_objects = vec![Wall, Wall, Player, Reward(5), Reward(3)];

        let result: String = test_game_objects.iter().map(|go| go.to_char()).collect();

        let expected = "##@53";

        assert_eq!(result, expected);
    }
}
