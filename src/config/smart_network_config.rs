use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[allow(unused)]
pub struct SmartNetworkConfig {
    pub input_count: usize,
    pub output_count: usize,
    pub nand_count_bits: usize,
    /// Number of independent memory registers (e.g. 4)
    #[serde(default)]
    pub memory_register_count: usize,
    /// Width of each register in bits (e.g. 4)
    #[serde(rename = "mem_rw_bits", alias = "memory_register_width")]
    pub memory_register_width: usize,
    pub connection_count: usize,
}
