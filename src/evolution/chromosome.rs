use std::fmt::{Debug, Display, Formatter};

#[derive(Hash, PartialEq, Eq, Clone)]
pub struct Chromosome {
    pub genes: Vec<bool>
}

impl Chromosome {

    pub fn from_genes(genes: Vec<bool>) -> Chromosome {
        Chromosome {
            genes
        }
    }

    pub fn from_bitstring(s: &str) -> Chromosome {
        let mut genes: Vec<bool> = Vec::new();

        for c in s.chars() {
            if c == '1' {
                genes.push(true)
            } else if c == '0' {
                genes.push(false)
            }
        }

        Chromosome {
            genes
        }
    }
}

impl Display for Chromosome {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let result: String = self.genes.iter().map(|&g| if g {'1'} else {'0'}).collect();

        write!(f, "{}", result)
    }
}

impl Debug for Chromosome {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}