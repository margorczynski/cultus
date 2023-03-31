pub enum GameAction {
    MoveUp,
    MoveDown,
    MoveRight,
    MoveLeft,
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
