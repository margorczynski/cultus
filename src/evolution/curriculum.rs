//! Curriculum learning for staged complexity.
//!
//! This module provides functionality to progressively increase the complexity
//! of the evolution task, starting with simple networks and levels, then
//! unlocking more complexity as fitness thresholds are reached.

use log::info;
use serde::{Deserialize, Serialize};

use crate::evolution::direct_encoding::{DirectNetwork, MemoryConfig};

/// Memory configuration for a curriculum stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStageConfig {
    /// Number of memory registers
    pub register_count: usize,
    /// Bits per register
    pub bits_per_register: usize,
}

impl MemoryStageConfig {
    /// Convert to MemoryConfig for DirectNetwork.
    pub fn to_memory_config(&self) -> MemoryConfig {
        MemoryConfig {
            register_count: self.register_count as u8,
            register_width: self.bits_per_register as u8,
        }
    }
}

/// Configuration for a single curriculum stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurriculumStage {
    /// Human-readable name for the stage
    pub name: String,
    /// Fitness threshold to unlock next stage
    pub unlock_threshold: f64,
    /// Maximum number of gates for networks at this stage
    pub max_gates: usize,
    /// Memory configuration (None = no memory at this stage)
    pub memory_config: Option<MemoryStageConfig>,
    /// Level indices to use at this stage
    pub level_indices: Vec<usize>,
    /// Number of times to play each level
    pub plays_per_level: usize,
}

impl Default for CurriculumStage {
    fn default() -> Self {
        CurriculumStage {
            name: "default".to_string(),
            unlock_threshold: f64::MAX,
            max_gates: 100,
            memory_config: None,
            level_indices: vec![1],
            plays_per_level: 10,
        }
    }
}

/// Manager for curriculum-based progressive learning.
pub struct CurriculumManager {
    stages: Vec<CurriculumStage>,
    current_stage_idx: usize,
    generations_at_stage: usize,
    best_fitness_at_stage: f64,
}

impl CurriculumManager {
    /// Create a new curriculum manager with the given stages.
    pub fn new(stages: Vec<CurriculumStage>) -> Self {
        assert!(!stages.is_empty(), "Curriculum must have at least one stage");
        CurriculumManager {
            stages,
            current_stage_idx: 0,
            generations_at_stage: 0,
            best_fitness_at_stage: 0.0,
        }
    }

    /// Create a default curriculum with 4 stages.
    pub fn default_curriculum() -> Self {
        let stages = vec![
            CurriculumStage {
                name: "stage_1_trivial".to_string(),
                unlock_threshold: 30.0,
                max_gates: 50,
                memory_config: None,
                level_indices: vec![1],
                plays_per_level: 5,
            },
            CurriculumStage {
                name: "stage_2_basic".to_string(),
                unlock_threshold: 100.0,
                max_gates: 150,
                memory_config: None,
                level_indices: vec![1, 2],
                plays_per_level: 15,
            },
            CurriculumStage {
                name: "stage_3_memory".to_string(),
                unlock_threshold: 250.0,
                max_gates: 400,
                memory_config: Some(MemoryStageConfig {
                    register_count: 8,
                    bits_per_register: 4,
                }),
                level_indices: vec![1, 2, 3],
                plays_per_level: 25,
            },
            CurriculumStage {
                name: "stage_4_full".to_string(),
                unlock_threshold: f64::MAX, // Final stage never unlocks
                max_gates: 1000,
                memory_config: Some(MemoryStageConfig {
                    register_count: 256,
                    bits_per_register: 8,
                }),
                level_indices: vec![1, 2, 3],
                plays_per_level: 50,
            },
        ];
        CurriculumManager::new(stages)
    }

    /// Get the current stage.
    pub fn current_stage(&self) -> &CurriculumStage {
        &self.stages[self.current_stage_idx]
    }

    /// Get the current stage index (0-based).
    pub fn current_stage_index(&self) -> usize {
        self.current_stage_idx
    }

    /// Get total number of stages.
    pub fn total_stages(&self) -> usize {
        self.stages.len()
    }

    /// Check if we're at the final stage.
    pub fn is_final_stage(&self) -> bool {
        self.current_stage_idx >= self.stages.len() - 1
    }

    /// Get generations spent at current stage.
    pub fn generations_at_current_stage(&self) -> usize {
        self.generations_at_stage
    }

    /// Get best fitness achieved at current stage.
    pub fn best_fitness_at_current_stage(&self) -> f64 {
        self.best_fitness_at_stage
    }

    /// Update with generation results. Returns true if stage advanced.
    pub fn update(&mut self, best_fitness: f64) -> bool {
        self.generations_at_stage += 1;
        self.best_fitness_at_stage = self.best_fitness_at_stage.max(best_fitness);

        let stage = &self.stages[self.current_stage_idx];
        if best_fitness >= stage.unlock_threshold && !self.is_final_stage() {
            self.advance_stage();
            return true;
        }
        false
    }

    /// Advance to the next stage.
    fn advance_stage(&mut self) {
        let old_stage = &self.stages[self.current_stage_idx];
        self.current_stage_idx += 1;
        let new_stage = &self.stages[self.current_stage_idx];

        info!(
            "Curriculum: Advanced from '{}' to '{}' (stage {}/{})",
            old_stage.name,
            new_stage.name,
            self.current_stage_idx + 1,
            self.stages.len()
        );

        self.generations_at_stage = 0;
        self.best_fitness_at_stage = 0.0;
    }

    /// Get memory config for current stage (if any).
    pub fn current_memory_config(&self) -> Option<MemoryConfig> {
        self.current_stage()
            .memory_config
            .as_ref()
            .map(|mc| mc.to_memory_config())
    }

    /// Migrate a network to the current stage's complexity level.
    ///
    /// This may add gates, enable memory, etc.
    pub fn migrate_network(&self, network: &mut DirectNetwork, rng: &mut impl rand::Rng) {
        let stage = self.current_stage();

        // Update memory config
        network.memory_config = self.current_memory_config();

        // Add gates if needed to reach stage's max_gates
        // (networks can have fewer, but we encourage growth)
        let gates_to_add = (stage.max_gates / 4).saturating_sub(network.gates.len());
        for _ in 0..gates_to_add {
            network.mutate_add_gate(rng);
        }

        // Ensure network is valid after migration
        network.repair(rng);
    }

    /// Check if a network is compatible with the current stage.
    pub fn is_network_compatible(&self, network: &DirectNetwork) -> bool {
        let stage = self.current_stage();

        // Check gate count
        if network.gates.len() > stage.max_gates {
            return false;
        }

        // Check memory config compatibility
        match (&network.memory_config, &stage.memory_config) {
            (Some(_), None) => false, // Network has memory but stage doesn't
            _ => true,
        }
    }
}

/// Create an initial population for the current curriculum stage.
pub fn create_stage_population(
    curriculum: &CurriculumManager,
    population_size: usize,
    input_count: u16,
    output_count: u16,
    rng: &mut impl rand::Rng,
) -> Vec<DirectNetwork> {
    let stage = curriculum.current_stage();
    let memory_config = curriculum.current_memory_config();

    // Start with fewer gates than max to allow growth
    let initial_gates = stage.max_gates / 4;

    (0..population_size)
        .map(|_| DirectNetwork::random(input_count, output_count, initial_gates, memory_config.clone(), rng))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_curriculum_creation() {
        let curriculum = CurriculumManager::default_curriculum();
        assert_eq!(curriculum.total_stages(), 4);
        assert_eq!(curriculum.current_stage_index(), 0);
        assert!(!curriculum.is_final_stage());
    }

    #[test]
    fn test_curriculum_advancement() {
        let mut curriculum = CurriculumManager::default_curriculum();

        // Below threshold - no advancement
        assert!(!curriculum.update(20.0));
        assert_eq!(curriculum.current_stage_index(), 0);

        // At threshold - advance
        assert!(curriculum.update(30.0));
        assert_eq!(curriculum.current_stage_index(), 1);
        assert_eq!(curriculum.current_stage().name, "stage_2_basic");
    }

    #[test]
    fn test_final_stage_no_advance() {
        let stages = vec![CurriculumStage {
            name: "only_stage".to_string(),
            unlock_threshold: 10.0,
            ..Default::default()
        }];
        let mut curriculum = CurriculumManager::new(stages);

        // Even above threshold, final stage doesn't advance
        assert!(!curriculum.update(100.0));
        assert!(curriculum.is_final_stage());
    }

    #[test]
    fn test_memory_config_conversion() {
        let mem_stage = MemoryStageConfig {
            register_count: 16,
            bits_per_register: 8,
        };
        let mem_config = mem_stage.to_memory_config();

        assert_eq!(mem_config.register_count, 16);
        assert_eq!(mem_config.register_width, 8);
    }
}
