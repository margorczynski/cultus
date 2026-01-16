use crate::evolution::direct_encoding::{DirectNetwork, GateType, InputSource};
use serde::{Deserialize, Serialize};

/// Op code for the bytecode interpreter.
/// Each op corresponds to a gate type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum OpCode {
    Nand = 0,
    And = 1,
    Or = 2,
    Not = 3,
    Xor = 4,
    Mux2 = 5,
    Equal = 6,
    Buffer = 7,
}

impl From<GateType> for OpCode {
    fn from(gt: GateType) -> Self {
        match gt {
            GateType::Nand => OpCode::Nand,
            GateType::And => OpCode::And,
            GateType::Or => OpCode::Or,
            GateType::Not => OpCode::Not,
            GateType::Xor => OpCode::Xor,
            GateType::Mux2 => OpCode::Mux2,
            GateType::Equal => OpCode::Equal,
            GateType::Buffer => OpCode::Buffer,
        }
    }
}

/// A single instruction in the bytecode.
/// 
/// We use u32 for indices to keep the struct compact (16 bytes total with padding probably, or packed).
/// Actually, to be cache friendly, we want this to be small.
/// OpCode is u8. output_idx is u32. inputs are u32s.
/// Max 3 inputs (Mux2).
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Instruction {
    pub op: OpCode,
    pub output_idx: u32,
    pub input1_idx: u32,
    pub input2_idx: u32,
    pub input3_idx: u32,
}

/// A compiled network that runs a flat bytecode buffer.
/// 
/// Buffer Layout:
/// [0]: Constant False
/// [1]: Constant True
/// [2..2+inputs]: Network Inputs
/// [...]: Memory Read Bits
/// [...]: Gate Outputs
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompiledNetwork {
    pub instructions: Vec<Instruction>,
    pub output_indices: Vec<u32>,
    pub buffer_size: usize,
    
    // Layout offsets
    pub input_offset: usize,
    pub memory_offset: usize,
    pub gate_offset: usize,
    
    // Quick access sizes
    pub input_count: usize,
    pub memory_count: usize,
}

impl CompiledNetwork {
    /// Compile a DirectNetwork into a CompiledNetwork.
    pub fn compile(network: &DirectNetwork) -> Self {
        let input_count = network.input_count as usize;
        let memory_count = network.memory_config.as_ref()
            .map(|m| m.register_count as usize * m.register_width as usize)
            .unwrap_or(0);
        let gate_count = network.gates.len();

        // Calculate layout
        let const_offset = 0;
        let input_offset = 2;
        let memory_offset = input_offset + input_count;
        let gate_offset = memory_offset + memory_count;
        let buffer_size = gate_offset + gate_count;

        let mut instructions = Vec::with_capacity(gate_count);

        // Map InputSource to buffer index
        let get_index = |source: &InputSource| -> u32 {
            match source {
                InputSource::Constant(val) => if *val { 1 } else { 0 },
                InputSource::NetworkInput(i) => (input_offset + *i as usize) as u32,
                InputSource::MemoryBit(i) => (memory_offset + *i as usize) as u32,
                InputSource::GateOutput(i) => (gate_offset + *i as usize) as u32,
            }
        };

        // Compile gates
        for (i, gate) in network.gates.iter().enumerate() {
            let output_idx = (gate_offset + i) as u32;
            let op = OpCode::from(gate.gate_type);
            
            let i1 = gate.inputs.get(0).map(get_index).unwrap_or(0);
            let i2 = gate.inputs.get(1).map(get_index).unwrap_or(0);
            let i3 = gate.inputs.get(2).map(get_index).unwrap_or(0);

            instructions.push(Instruction {
                op,
                output_idx,
                input1_idx: i1,
                input2_idx: i2,
                input3_idx: i3,
            });
        }

        // Compile outputs
        let output_indices: Vec<u32> = network.outputs.iter()
            .map(get_index)
            .collect();

        CompiledNetwork {
            instructions,
            output_indices,
            buffer_size,
            input_offset,
            memory_offset,
            gate_offset,
            input_count,
            memory_count,
        }
    }

    /// Compute the network output.
    /// 
    /// Returns the output vector.
    pub fn compute(&self, inputs: &[bool], memory: &[bool]) -> Vec<bool> {
        // Allocate buffer on stack if small, or heap. 
        // For performance, we should ideally reuse this buffer, but for now specific allocation per run.
        let mut buffer = vec![false; self.buffer_size];

        // Set constants
        buffer[0] = false;
        buffer[1] = true;

        // Copy inputs
        // Safety: We assume inputs.len() matches input_count usually, but we clamp to be safe or panic?
        // DirectNetwork usually guarantees these match context.
        let copy_len = inputs.len().min(self.input_count);
        buffer[self.input_offset..self.input_offset + copy_len].copy_from_slice(&inputs[0..copy_len]);

        // Copy memory
        let mem_copy_len = memory.len().min(self.memory_count);
        buffer[self.memory_offset..self.memory_offset + mem_copy_len].copy_from_slice(&memory[0..mem_copy_len]);

        // Execute instructions
        for instr in &self.instructions {
            // Using unsafe for unchecked access would be faster, but let's trust the optimizer first.
            let v1 = buffer[instr.input1_idx as usize];
            let v2 = buffer[instr.input2_idx as usize];
            let v3 = buffer[instr.input3_idx as usize];

            let result = match instr.op {
                OpCode::Nand => !(v1 && v2),
                OpCode::And => v1 && v2,
                OpCode::Or => v1 || v2,
                OpCode::Not => !v1,
                OpCode::Xor => v1 ^ v2,
                OpCode::Mux2 => if v1 { v3 } else { v2 }, // Mux2: if in[0] then in[2] else in[1]
                OpCode::Equal => v1 == v2,
                OpCode::Buffer => v1,
            };

            buffer[instr.output_idx as usize] = result;
        }

        // Gather outputs
        self.output_indices.iter()
            .map(|&idx| buffer[idx as usize])
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evolution::direct_encoding::{Gate, InputSource};

    #[test]
    fn test_compile_and_compute_simple() {
        // A AND B
        let network = DirectNetwork {
            input_count: 2,
            output_count: 1,
            gates: vec![Gate::new(
                GateType::And,
                vec![InputSource::NetworkInput(0), InputSource::NetworkInput(1)],
            )],
            outputs: vec![InputSource::GateOutput(0)],
            memory_config: None,
        };

        let compiled = CompiledNetwork::compile(&network);
        
        assert_eq!(compiled.compute(&[true, true], &[]), vec![true]);
        assert_eq!(compiled.compute(&[true, false], &[]), vec![false]);
        assert_eq!(compiled.compute(&[false, true], &[]), vec![false]);
        assert_eq!(compiled.compute(&[false, false], &[]), vec![false]);
    }

    #[test]
    fn test_compile_complex_logic() {
        // (A OR B) XOR C
        let network = DirectNetwork {
            input_count: 3,
            output_count: 1,
            gates: vec![
                Gate::new(GateType::Or, vec![InputSource::NetworkInput(0), InputSource::NetworkInput(1)]),
                Gate::new(GateType::Xor, vec![InputSource::GateOutput(0), InputSource::NetworkInput(2)]),
            ],
            outputs: vec![InputSource::GateOutput(1)],
            memory_config: None,
        };

        let compiled = CompiledNetwork::compile(&network);

        // True OR False = True. True XOR False = True.
        assert_eq!(compiled.compute(&[true, false, false], &[]), vec![true]);
        // False OR False = False. False XOR True = True.
        assert_eq!(compiled.compute(&[false, false, true], &[]), vec![true]);
        // True OR True = True. True XOR True = False.
        assert_eq!(compiled.compute(&[true, true, true], &[]), vec![false]);
    }
}

#[cfg(test)]
mod benchmarks {
    use super::*;
    use crate::evolution::direct_encoding::{DirectNetwork, Gate, GateType, InputSource};
    use std::time::Instant;
    use rand::prelude::*;

    #[test]
    #[ignore]
    fn benchmark_compiled_vs_direct() {
        let mut rng = StdRng::seed_from_u64(42);
        let gate_count = 1000;
        let input_count = 10;
        let memory_bits = 16;
        let output_count = 5;

        // Create random network
        let network = DirectNetwork::random(
            input_count,
            output_count,
            gate_count,
            Some(crate::evolution::direct_encoding::MemoryConfig {
                register_count: 4,
                register_width: 4,
            }),
            &mut rng
        );

        let compiled = CompiledNetwork::compile(&network);
        let iterations = 100_000;
        let inputs: Vec<bool> = (0..input_count).map(|_| rng.gen()).collect();
        let memory: Vec<bool> = (0..memory_bits).map(|_| rng.gen()).collect();

        // Warmup
        for _ in 0..100 {
            network.compute_with_memory(&inputs, &memory);
            compiled.compute(&inputs, &memory);
        }

        // Benchmark Direct
        let start = Instant::now();
        for _ in 0..iterations {
            std::hint::black_box(network.compute_with_memory(&inputs, &memory));
        }
        let duration_direct = start.elapsed();

        // Benchmark Compiled
        let start = Instant::now();
        for _ in 0..iterations {
            std::hint::black_box(compiled.compute(&inputs, &memory));
        }
        let duration_compiled = start.elapsed();

        println!("Direct: {:?}", duration_direct);
        println!("Compiled: {:?}", duration_compiled);
        println!("Speedup: {:.2}x", duration_direct.as_secs_f64() / duration_compiled.as_secs_f64());
    }
}
