use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};

use bitvec::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct Chromosome {
    pub genes: BitVec<u64, Lsb0>,
}

impl Chromosome {
    pub fn from_bitvec(genes: BitVec<u64, Lsb0>) -> Chromosome {
        Chromosome { genes }
    }

    pub fn from_bool_iter<I: IntoIterator<Item = bool>>(iter: I) -> Chromosome {
        Chromosome {
            genes: iter.into_iter().collect(),
        }
    }

    pub fn len(&self) -> usize {
        self.genes.len()
    }
}

impl PartialEq for Chromosome {
    fn eq(&self, other: &Self) -> bool {
        self.genes == other.genes
    }
}

impl Eq for Chromosome {}

impl Hash for Chromosome {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash the raw storage for efficiency
        self.genes.as_raw_slice().hash(state);
        self.genes.len().hash(state);
    }
}

impl Display for Chromosome {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let result: String = self.genes.iter().map(|b| if *b { '1' } else { '0' }).collect();
        write!(f, "{}", result)
    }
}

impl Debug for Chromosome {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}
