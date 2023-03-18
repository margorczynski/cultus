#[derive(Debug)]
pub enum LogicTree {
    Input(bool),
    NAND(Box<LogicTree>, Box<LogicTree>),
    NOR(Box<LogicTree>, Box<LogicTree>)
}

pub trait Evaluation {
    fn eval(&self) -> bool;
}

impl Evaluation for LogicTree {
    fn eval(&self) -> bool {
        match *self {
            LogicTree::Input(value) => value,
            LogicTree::NAND(ref lt, ref rt) => !(lt.eval() && rt.eval()),
            LogicTree::NOR(ref lt, ref rt) => !(lt.eval() || rt.eval()),
        }
    }
}