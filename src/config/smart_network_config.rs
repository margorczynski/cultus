use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[allow(unused)]
pub struct SmartNetworkConfig {
    pub input_count: usize,
    pub output_count: usize,
    pub nand_count_bits: usize,
    pub mem_addr_bits: usize,
    pub mem_rw_bits: usize,
    pub connection_count: usize,
}
