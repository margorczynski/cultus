use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use crossbeam_channel::Receiver;
use std::collections::VecDeque;

/// Data packet sent from evolution thread to GUI
#[derive(Debug, Clone)]
pub struct EvolutionMetrics {
    pub generation: usize,
    pub stage_name: String,
    pub stage_index: usize,
    pub best_fitness: f64,
    pub avg_fitness: f64,
    pub objective_fitness: f64,
    pub avg_gates: f64,
    pub diversity: f64,
    pub archive_size: usize,
}

const MAX_PLOT_POINTS: usize = 2000;

pub struct EvolutionApp {
    rx: Receiver<EvolutionMetrics>,
    metrics_history: VecDeque<EvolutionMetrics>,
    // Store processed plot points to avoid re-generating every frame
    plot_fitness_best: Vec<[f64; 2]>,
    plot_fitness_avg: Vec<[f64; 2]>,
    plot_diversity: Vec<[f64; 2]>,
    plot_gates: Vec<[f64; 2]>,
    
    // Optimization: dynamic stride for downsampling
    plot_stride: usize,
}

impl EvolutionApp {
    pub fn new(_cc: &eframe::CreationContext<'_>, rx: Receiver<EvolutionMetrics>) -> Self {
        Self {
            rx,
            metrics_history: VecDeque::new(),
            plot_fitness_best: Vec::new(),
            plot_fitness_avg: Vec::new(),
            plot_diversity: Vec::new(),
            plot_gates: Vec::new(),
            plot_stride: 1,
        }
    }

    fn downsample_vector(vec: &mut Vec<[f64; 2]>) {
        // Keep every second element
        let mut write_idx = 0;
        for read_idx in (0..vec.len()).step_by(2) {
            vec[write_idx] = vec[read_idx];
            write_idx += 1;
        }
        vec.truncate(write_idx);
    }

    fn process_messages(&mut self) {
        // Drain all available messages
        while let Ok(metrics) = self.rx.try_recv() {
            let gen = metrics.generation;
            
            // Limit history size to prevent memory bloat (only need latest for sidebar)
            if self.metrics_history.len() >= 100 {
                self.metrics_history.pop_front();
            }
            // Only update history with latest for current status display
            self.metrics_history.push_back(metrics.clone());
            
            // Plotting update with strided downsampling
            if gen % self.plot_stride == 0 {
                let gen_f = gen as f64;
                self.plot_fitness_best.push([gen_f, metrics.best_fitness]);
                self.plot_fitness_avg.push([gen_f, metrics.avg_fitness]);
                self.plot_diversity.push([gen_f, metrics.diversity]);
                self.plot_gates.push([gen_f, metrics.avg_gates]);

                // Check if we need to downsample
                if self.plot_fitness_best.len() > MAX_PLOT_POINTS {
                    self.plot_stride *= 2;
                    Self::downsample_vector(&mut self.plot_fitness_best);
                    Self::downsample_vector(&mut self.plot_fitness_avg);
                    Self::downsample_vector(&mut self.plot_diversity);
                    Self::downsample_vector(&mut self.plot_gates);
                    // Force a full garbage collect of the plot memory if possible? 
                    // Vec::truncate keeps capacity, but it's fine as it won't grow infinitely now.
                }
            }
        }
    }
}

impl eframe::App for EvolutionApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.process_messages();
        
        // Request constant repaint to show smooth updates if data is coming in fast
        // Or only repaint on new data. 
        // For now, let's request repaint to be responsive.
        ctx.request_repaint();

        let latest = self.metrics_history.back();

        egui::SidePanel::left("side_panel").show(ctx, |ui| {
            ui.heading("Cultus Evolution");
            ui.separator();

            if let Some(m) = latest {
                ui.heading(format!("Gen: {}", m.generation));
                ui.label(format!("Stage: {} ({})", m.stage_name, m.stage_index));
                ui.separator();
                
                ui.label("Fitness:");
                ui.label(format!("  Best: {:.2}", m.best_fitness));
                ui.label(format!("  Avg:  {:.2}", m.avg_fitness));
                ui.label(format!("  Obj:  {:.2}", m.objective_fitness));
                ui.separator();

                ui.label("Diversity:");
                ui.label(format!("  Score:   {:.2}", m.diversity));
                ui.label(format!("  Archive: {}", m.archive_size));
                ui.separator();

                ui.label("Network:");
                ui.label(format!("  Avg Gates: {:.1}", m.avg_gates));
            } else {
                ui.label("Waiting for data...");
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Metrics");
            
            // Tabs via Grid/Layout or just vertical plots
            // Tabs via Grid/Layout or just vertical plots
            let height = ui.available_height() / 3.5;

            ui.label("Fitness (Best & Avg)");
            Plot::new("fitness_plot")
                .view_aspect(2.0)
                .height(height)
                .show(ui, |plot_ui| {
                    plot_ui.line(Line::new(PlotPoints::from(self.plot_fitness_best.clone())).name("Best"));
                    plot_ui.line(Line::new(PlotPoints::from(self.plot_fitness_avg.clone())).name("Avg"));
                });
            
            ui.add_space(8.0);

            ui.label("Behavioral Diversity");
            Plot::new("diversity_plot")
                .view_aspect(2.0)
                .height(height)
                .show(ui, |plot_ui| {
                    plot_ui.line(Line::new(PlotPoints::from(self.plot_diversity.clone())).name("Diversity"));
                });

            ui.add_space(8.0);

            ui.label("Average Gate Count");
            Plot::new("gates_plot")
                .view_aspect(2.0)
                .height(height)
                .show(ui, |plot_ui| {
                    plot_ui.line(Line::new(PlotPoints::from(self.plot_gates.clone())).name("Gates"));
                });
        });
    }
}
