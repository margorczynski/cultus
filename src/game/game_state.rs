use super::level::Level;

pub enum GameState {
    InProgress(Level),
    Finished(usize)
}