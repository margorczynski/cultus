//! Novelty search and behavioral diversity.
//!
//! This module provides functionality to measure behavioral diversity
//! and reward novel behaviors to prevent premature convergence.

use std::collections::HashSet;
use std::hash::Hash;

use serde::{Deserialize, Serialize};

/// A single step in a game trace.
#[derive(Clone, Debug)]
pub struct GameStep {
    /// Position after this step (row, column)
    pub position: (u16, u16),
    /// Action taken (0-3 for the 4 directions)
    pub action: u8,
    /// Cumulative points after this step
    pub cumulative_points: usize,
}

/// Complete trace of a game play.
#[derive(Clone, Debug, Default)]
pub struct GameTrace {
    /// All steps taken during the game
    pub steps: Vec<GameStep>,
    /// Final score
    pub final_score: usize,
    /// Whether the game was won (all rewards collected)
    pub won: bool,
}

impl GameTrace {
    /// Create a new empty trace.
    pub fn new() -> Self {
        GameTrace::default()
    }

    /// Record a step in the trace.
    pub fn record_step(&mut self, position: (u16, u16), action: u8, cumulative_points: usize) {
        self.steps.push(GameStep {
            position,
            action,
            cumulative_points,
        });
    }

    /// Finalize the trace with the final score.
    pub fn finalize(&mut self, final_score: usize, won: bool) {
        self.final_score = final_score;
        self.won = won;
    }
}

/// Behavioral signature for novelty computation.
///
/// Captures the key behavioral characteristics of a network's performance.
#[derive(Clone, Debug)]
pub struct BehavioralSignature {
    /// Cells visited during play (as position tuples)
    pub visited_cells: HashSet<(u16, u16)>,
    /// Final position
    pub final_position: (u16, u16),
    /// Movement pattern: ratio of each action type [up, down, right, left]
    pub action_ratios: [f32; 4],
    /// Exploration score: unique cells / total steps
    pub exploration_ratio: f32,
    /// Points timeline (sampled at regular intervals)
    pub points_timeline: Vec<u16>,
    /// Whether any rewards were collected
    pub collected_rewards: bool,
    /// Total steps taken
    pub total_steps: usize,
}

impl BehavioralSignature {
    /// Create a behavioral signature from a game trace.
    pub fn from_trace(trace: &GameTrace) -> Self {
        let mut visited = HashSet::new();
        let mut action_counts = [0u32; 4];
        let mut points_timeline = Vec::new();

        // Sample points at regular intervals
        let sample_interval = (trace.steps.len() / 10).max(1);

        for (i, step) in trace.steps.iter().enumerate() {
            visited.insert(step.position);
            if step.action < 4 {
                action_counts[step.action as usize] += 1;
            }
            if i % sample_interval == 0 {
                points_timeline.push(step.cumulative_points as u16);
            }
        }

        let total_actions: u32 = action_counts.iter().sum();
        let action_ratios: [f32; 4] = if total_actions > 0 {
            action_counts.map(|c| c as f32 / total_actions as f32)
        } else {
            [0.25; 4]
        };

        let final_position = trace
            .steps
            .last()
            .map(|s| s.position)
            .unwrap_or((0, 0));

        let exploration_ratio = if trace.steps.is_empty() {
            0.0
        } else {
            visited.len() as f32 / trace.steps.len() as f32
        };

        BehavioralSignature {
            visited_cells: visited,
            final_position,
            action_ratios,
            exploration_ratio,
            points_timeline,
            collected_rewards: trace.final_score > 0,
            total_steps: trace.steps.len(),
        }
    }

    /// Compute behavioral distance to another signature.
    pub fn distance(&self, other: &BehavioralSignature) -> f64 {
        let mut dist = 0.0;

        // Jaccard distance on visited cells (weighted heavily)
        let intersection = self.visited_cells.intersection(&other.visited_cells).count();
        let union = self.visited_cells.union(&other.visited_cells).count();
        let jaccard_dist = if union > 0 {
            1.0 - (intersection as f64 / union as f64)
        } else {
            1.0
        };
        dist += jaccard_dist * 10.0;

        // Euclidean distance on action ratios
        let action_dist: f32 = self
            .action_ratios
            .iter()
            .zip(other.action_ratios.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        dist += action_dist as f64 * 5.0;

        // Final position distance
        let pos_dist = ((self.final_position.0 as f64 - other.final_position.0 as f64).powi(2)
            + (self.final_position.1 as f64 - other.final_position.1 as f64).powi(2))
        .sqrt();
        dist += pos_dist;

        // Exploration ratio difference
        dist += (self.exploration_ratio - other.exploration_ratio).abs() as f64 * 3.0;

        // Reward collection difference
        if self.collected_rewards != other.collected_rewards {
            dist += 5.0;
        }

        dist
    }
}

/// Archive of behavioral signatures for novelty computation.
pub struct NoveltyArchive {
    /// Archived signatures
    archive: Vec<BehavioralSignature>,
    /// Maximum archive size
    max_size: usize,
    /// K for k-nearest neighbors
    k_nearest: usize,
    /// Threshold for adding to archive
    add_threshold: f64,
}

impl NoveltyArchive {
    /// Create a new novelty archive.
    pub fn new(max_size: usize, k_nearest: usize, add_threshold: f64) -> Self {
        NoveltyArchive {
            archive: Vec::with_capacity(max_size),
            max_size,
            k_nearest,
            add_threshold,
        }
    }

    /// Get the size of the archive.
    pub fn len(&self) -> usize {
        self.archive.len()
    }

    /// Check if the archive is empty.
    pub fn is_empty(&self) -> bool {
        self.archive.is_empty()
    }

    /// Compute novelty score for a signature against archive + current population.
    pub fn compute_novelty(
        &self,
        signature: &BehavioralSignature,
        population_signatures: &[BehavioralSignature],
    ) -> f64 {
        // Combine archive and population
        let all_signatures: Vec<&BehavioralSignature> = self
            .archive
            .iter()
            .chain(population_signatures.iter())
            .collect();

        if all_signatures.is_empty() {
            return f64::MAX; // Novel by default if no comparison
        }

        // Compute distances to all signatures
        let mut distances: Vec<f64> = all_signatures
            .iter()
            .map(|s| signature.distance(s))
            .collect();

        // Sort and take k-nearest
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Average distance to k-nearest neighbors
        let k = self.k_nearest.min(distances.len());
        if k == 0 {
            return f64::MAX;
        }
        distances[..k].iter().sum::<f64>() / k as f64
    }

    /// Maybe add signature to archive if sufficiently novel.
    pub fn maybe_add(&mut self, signature: BehavioralSignature, novelty: f64) {
        if novelty >= self.add_threshold {
            if self.archive.len() >= self.max_size {
                // Remove oldest
                self.archive.remove(0);
            }
            self.archive.push(signature);
        }
    }

    /// Force add a signature to the archive.
    pub fn add(&mut self, signature: BehavioralSignature) {
        if self.archive.len() >= self.max_size {
            self.archive.remove(0);
        }
        self.archive.push(signature);
    }

    /// Clear the archive.
    pub fn clear(&mut self) {
        self.archive.clear();
    }
}

/// Behavioral milestones that reward specific achievements.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Milestone {
    /// First movement (any direction)
    FirstMove,
    /// Moved in all 4 directions at least once
    AllDirections,
    /// Collected first reward
    FirstReward,
    /// Visited N unique cells
    Exploration(usize),
    /// Approached a reward (got within N cells)
    ApproachedReward,
    /// Completed a level (collected all rewards)
    LevelComplete,
}

/// Tracker for behavioral milestones.
pub struct MilestoneTracker {
    achieved: HashSet<Milestone>,
}

impl MilestoneTracker {
    /// Create a new milestone tracker.
    pub fn new() -> Self {
        MilestoneTracker {
            achieved: HashSet::new(),
        }
    }

    /// Reset the tracker for a new evaluation.
    pub fn reset(&mut self) {
        self.achieved.clear();
    }

    /// Check milestones from a game trace and return bonus points.
    pub fn check_milestones(&mut self, trace: &GameTrace) -> f64 {
        let mut bonus = 0.0;

        if trace.steps.is_empty() {
            return bonus;
        }

        // Check FirstMove
        if !self.achieved.contains(&Milestone::FirstMove) {
            let initial_pos = trace.steps.first().map(|s| s.position);
            let moved = trace.steps.iter().any(|s| Some(s.position) != initial_pos);
            if moved {
                self.achieved.insert(Milestone::FirstMove);
                bonus += 5.0;
            }
        }

        // Check AllDirections
        if !self.achieved.contains(&Milestone::AllDirections) {
            let actions: HashSet<u8> = trace.steps.iter().map(|s| s.action).collect();
            if actions.len() >= 4 {
                self.achieved.insert(Milestone::AllDirections);
                bonus += 10.0;
            }
        }

        // Check FirstReward
        if !self.achieved.contains(&Milestone::FirstReward) {
            if trace.final_score > 0 {
                self.achieved.insert(Milestone::FirstReward);
                bonus += 20.0;
            }
        }

        // Check Exploration milestones
        let unique_cells: HashSet<_> = trace.steps.iter().map(|s| s.position).collect();
        for &threshold in &[5, 10, 20, 50] {
            let milestone = Milestone::Exploration(threshold);
            if !self.achieved.contains(&milestone) && unique_cells.len() >= threshold {
                self.achieved.insert(milestone);
                bonus += threshold as f64;
            }
        }

        // Check LevelComplete
        if !self.achieved.contains(&Milestone::LevelComplete) {
            if trace.won {
                self.achieved.insert(Milestone::LevelComplete);
                bonus += 100.0;
            }
        }

        bonus
    }

    /// Get the number of milestones achieved.
    pub fn achieved_count(&self) -> usize {
        self.achieved.len()
    }
}

impl Default for MilestoneTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Combined fitness with novelty.
#[derive(Clone, Debug)]
pub struct NoveltyFitness {
    /// Objective (game-based) fitness
    pub objective_fitness: f64,
    /// Novelty score
    pub novelty_score: f64,
    /// Milestone bonus
    pub milestone_bonus: f64,
    /// Combined fitness
    pub combined_fitness: f64,
}

impl NoveltyFitness {
    /// Compute combined fitness from components.
    pub fn compute(
        objective: f64,
        novelty: f64,
        milestone: f64,
        objective_weight: f64,
        novelty_weight: f64,
    ) -> Self {
        let combined = objective * objective_weight + novelty * novelty_weight + milestone;

        NoveltyFitness {
            objective_fitness: objective,
            novelty_score: novelty,
            milestone_bonus: milestone,
            combined_fitness: combined,
        }
    }
}

/// Configuration for novelty search.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NoveltyConfig {
    /// Weight for objective fitness (alpha)
    pub objective_weight: f64,
    /// Weight for novelty (beta)
    pub novelty_weight: f64,
    /// K for k-nearest neighbors
    pub k_nearest: usize,
    /// Maximum archive size
    pub archive_size: usize,
    /// Threshold to add to archive
    pub archive_threshold: f64,
    /// Enable behavioral milestones
    pub use_milestones: bool,
}

impl Default for NoveltyConfig {
    fn default() -> Self {
        NoveltyConfig {
            objective_weight: 0.7,
            novelty_weight: 0.3,
            k_nearest: 15,
            archive_size: 500,
            archive_threshold: 10.0,
            use_milestones: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_behavioral_signature_from_trace() {
        let mut trace = GameTrace::new();
        trace.record_step((0, 0), 0, 0);
        trace.record_step((0, 1), 2, 0);
        trace.record_step((0, 2), 2, 10);
        trace.record_step((1, 2), 1, 10);
        trace.finalize(10, false);

        let sig = BehavioralSignature::from_trace(&trace);

        assert_eq!(sig.visited_cells.len(), 4);
        assert_eq!(sig.final_position, (1, 2));
        assert!(sig.collected_rewards);
        assert_eq!(sig.total_steps, 4);
    }

    #[test]
    fn test_behavioral_distance() {
        let mut trace1 = GameTrace::new();
        trace1.record_step((0, 0), 0, 0);
        trace1.record_step((0, 1), 2, 0);
        trace1.finalize(0, false);

        let mut trace2 = GameTrace::new();
        trace2.record_step((5, 5), 1, 0);
        trace2.record_step((5, 6), 3, 10);
        trace2.finalize(10, false);

        let sig1 = BehavioralSignature::from_trace(&trace1);
        let sig2 = BehavioralSignature::from_trace(&trace2);

        let dist = sig1.distance(&sig2);
        assert!(dist > 0.0);
    }

    #[test]
    fn test_novelty_archive() {
        let mut archive = NoveltyArchive::new(10, 3, 5.0);

        let mut trace = GameTrace::new();
        trace.record_step((0, 0), 0, 0);
        trace.finalize(0, false);
        let sig = BehavioralSignature::from_trace(&trace);

        archive.add(sig.clone());
        assert_eq!(archive.len(), 1);

        let novelty = archive.compute_novelty(&sig, &[]);
        assert_eq!(novelty, 0.0); // Same signature = 0 distance
    }

    #[test]
    fn test_milestone_tracker() {
        let mut tracker = MilestoneTracker::new();

        let mut trace = GameTrace::new();
        trace.record_step((0, 0), 0, 0);
        trace.record_step((0, 1), 1, 0);
        trace.record_step((0, 2), 2, 0);
        trace.record_step((0, 3), 3, 0);
        trace.record_step((1, 3), 0, 10);
        trace.finalize(10, false);

        let bonus = tracker.check_milestones(&trace);

        assert!(bonus > 0.0);
        assert!(tracker.achieved_count() >= 3); // FirstMove, AllDirections, FirstReward
    }
}
