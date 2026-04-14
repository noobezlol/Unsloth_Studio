mod settings;
use settings::AppSettings;
use eframe::egui;
use egui_plot::{Line, Plot};
use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{channel, Receiver};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Debug, PartialEq, Clone)]
enum TrainingMode { SFT, GRPO }

#[derive(Clone)]
struct TrainingConfig {
    model_name: String, learning_rate: f32, batch_size: u32, lora_rank: u32,
    max_steps: u32, dataset_path: String, export_name: String, data_format: String,
    reward_xml: bool, reward_length: bool, preset: String,
}

#[derive(serde::Deserialize, Debug)]
struct LogEvent {
    event: String, step: Option<u32>, max_steps: Option<u32>,
    data: Option<serde_json::Value>, message: Option<String>,
}

#[derive(Clone, PartialEq)]
enum Theme { Dark, Light, Cyberpunk, Nord, Dracula }

#[derive(Clone)]
struct ColorPalette {
    bg_primary: egui::Color32, bg_secondary: egui::Color32, bg_tertiary: egui::Color32,
    bg_card: egui::Color32, text_primary: egui::Color32, text_secondary: egui::Color32,
    text_muted: egui::Color32, accent: egui::Color32, accent_hover: egui::Color32,
    success: egui::Color32, warning: egui::Color32, error: egui::Color32,
    border: egui::Color32, divider: egui::Color32, chart_line: egui::Color32,
    chart_fill: egui::Color32, sidebar_bg: egui::Color32, sidebar_active: egui::Color32,
}

impl ColorPalette {
    fn dark() -> Self {
        Self {
            bg_primary: egui::Color32::from_rgb(10, 10, 12), bg_secondary: egui::Color32::from_rgb(18, 18, 22),
            bg_tertiary: egui::Color32::from_rgb(28, 28, 35), bg_card: egui::Color32::from_rgb(24, 24, 30),
            text_primary: egui::Color32::from_rgb(240, 240, 245), text_secondary: egui::Color32::from_rgb(180, 180, 195),
            text_muted: egui::Color32::from_rgb(110, 110, 125), accent: egui::Color32::from_rgb(85, 140, 255),
            accent_hover: egui::Color32::from_rgb(105, 160, 255), success: egui::Color32::from_rgb(70, 210, 130),
            warning: egui::Color32::from_rgb(255, 180, 70), error: egui::Color32::from_rgb(255, 90, 90),
            border: egui::Color32::from_rgb(50, 50, 60), divider: egui::Color32::from_rgb(40, 40, 50),
            chart_line: egui::Color32::from_rgb(85, 140, 255), chart_fill: egui::Color32::from_rgba_premultiplied(85, 140, 255, 25),
            sidebar_bg: egui::Color32::from_rgb(15, 15, 20), sidebar_active: egui::Color32::from_rgb(35, 35, 45),
        }
    }
    fn light() -> Self {
        Self {
            bg_primary: egui::Color32::from_rgb(250, 250, 252), bg_secondary: egui::Color32::from_rgb(245, 245, 250),
            bg_tertiary: egui::Color32::from_rgb(235, 235, 242), bg_card: egui::Color32::from_rgb(255, 255, 255),
            text_primary: egui::Color32::from_rgb(25, 25, 30), text_secondary: egui::Color32::from_rgb(70, 70, 80),
            text_muted: egui::Color32::from_rgb(120, 120, 135), accent: egui::Color32::from_rgb(55, 115, 230),
            accent_hover: egui::Color32::from_rgb(75, 135, 250), success: egui::Color32::from_rgb(40, 165, 90),
            warning: egui::Color32::from_rgb(220, 145, 30), error: egui::Color32::from_rgb(215, 55, 55),
            border: egui::Color32::from_rgb(215, 215, 220), divider: egui::Color32::from_rgb(230, 230, 235),
            chart_line: egui::Color32::from_rgb(55, 115, 230), chart_fill: egui::Color32::from_rgba_premultiplied(55, 115, 230, 25),
            sidebar_bg: egui::Color32::from_rgb(248, 248, 250), sidebar_active: egui::Color32::from_rgb(230, 230, 238),
        }
    }
    fn cyberpunk() -> Self {
        Self {
            bg_primary: egui::Color32::from_rgb(8, 6, 14), bg_secondary: egui::Color32::from_rgb(12, 8, 22),
            bg_tertiary: egui::Color32::from_rgb(22, 10, 35), bg_card: egui::Color32::from_rgb(16, 10, 28),
            text_primary: egui::Color32::from_rgb(230, 225, 245), text_secondary: egui::Color32::from_rgb(175, 165, 200),
            text_muted: egui::Color32::from_rgb(110, 100, 135), accent: egui::Color32::from_rgb(255, 0, 160),
            accent_hover: egui::Color32::from_rgb(255, 50, 180), success: egui::Color32::from_rgb(0, 255, 195),
            warning: egui::Color32::from_rgb(255, 210, 20), error: egui::Color32::from_rgb(255, 60, 120),
            border: egui::Color32::from_rgb(70, 35, 100), divider: egui::Color32::from_rgb(45, 22, 70),
            chart_line: egui::Color32::from_rgb(255, 0, 190), chart_fill: egui::Color32::from_rgba_premultiplied(255, 0, 190, 30),
            sidebar_bg: egui::Color32::from_rgb(10, 6, 18), sidebar_active: egui::Color32::from_rgb(30, 15, 50),
        }
    }
    fn nord() -> Self {
        Self {
            bg_primary: egui::Color32::from_rgb(32, 38, 52), bg_secondary: egui::Color32::from_rgb(42, 46, 60),
            bg_tertiary: egui::Color32::from_rgb(55, 62, 78), bg_card: egui::Color32::from_rgb(43, 49, 64),
            text_primary: egui::Color32::from_rgb(220, 225, 238), text_secondary: egui::Color32::from_rgb(165, 175, 195),
            text_muted: egui::Color32::from_rgb(120, 130, 150), accent: egui::Color32::from_rgb(130, 185, 210),
            accent_hover: egui::Color32::from_rgb(155, 205, 230), success: egui::Color32::from_rgb(165, 195, 145),
            warning: egui::Color32::from_rgb(240, 210, 145), error: egui::Color32::from_rgb(195, 105, 115),
            border: egui::Color32::from_rgb(60, 68, 85), divider: egui::Color32::from_rgb(55, 62, 78),
            chart_line: egui::Color32::from_rgb(130, 185, 210), chart_fill: egui::Color32::from_rgba_premultiplied(130, 185, 210, 25),
            sidebar_bg: egui::Color32::from_rgb(28, 34, 46), sidebar_active: egui::Color32::from_rgb(55, 62, 78),
        }
    }
    fn dracula() -> Self {
        Self {
            bg_primary: egui::Color32::from_rgb(25, 25, 40), bg_secondary: egui::Color32::from_rgb(35, 35, 52),
            bg_tertiary: egui::Color32::from_rgb(48, 48, 68), bg_card: egui::Color32::from_rgb(38, 38, 56),
            text_primary: egui::Color32::from_rgb(250, 250, 248), text_secondary: egui::Color32::from_rgb(180, 175, 200),
            text_muted: egui::Color32::from_rgb(110, 110, 130), accent: egui::Color32::from_rgb(140, 185, 255),
            accent_hover: egui::Color32::from_rgb(170, 210, 255), success: egui::Color32::from_rgb(170, 230, 165),
            warning: egui::Color32::from_rgb(250, 230, 180), error: egui::Color32::from_rgb(255, 95, 95),
            border: egui::Color32::from_rgb(65, 65, 85), divider: egui::Color32::from_rgb(55, 55, 75),
            chart_line: egui::Color32::from_rgb(140, 185, 255), chart_fill: egui::Color32::from_rgba_premultiplied(140, 185, 255, 25),
            sidebar_bg: egui::Color32::from_rgb(22, 22, 38), sidebar_active: egui::Color32::from_rgb(55, 55, 80),
        }
    }
}

struct Toast { message: String, kind: ToastKind, timestamp: Instant }
#[derive(Clone, PartialEq)] enum ToastKind { Success, Error, Info, Warning }
impl ToastKind { fn color(&self) -> egui::Color32 {
    match self {
        ToastKind::Success => egui::Color32::from_rgb(70, 210, 130),
        ToastKind::Error => egui::Color32::from_rgb(255, 90, 90),
        ToastKind::Warning => egui::Color32::from_rgb(255, 180, 70),
        ToastKind::Info => egui::Color32::from_rgb(85, 140, 255),
    }
}}

#[derive(PartialEq, Clone)]
enum NavTab { Train, Monitor, Logs, Settings }

struct App {
    mode: TrainingMode, config: TrainingConfig, is_training: bool,
    logs: VecDeque<(String, egui::Color32)>, loss_history: Vec<[f64; 2]>,
    reward_history: Vec<[f64; 2]>, lr_history: Vec<[f64; 2]>,
    log_receiver: Option<Receiver<String>>, training_handle: Option<Child>,
    convert_input_path: String, settings: AppSettings, show_settings: bool,
    selected_theme: Theme, active_tab: NavTab, expanded_sections: Vec<&'static str>,
    toasts: Vec<Toast>, auto_scroll: bool, search_text: String,
    current_step: u32, max_steps: u32, eta_seconds: Option<u64>,
    training_start_time: Option<Instant>, gpu_usage: u32, gpu_memory: u32,
    last_gpu_update: Instant, last_frame: Instant,
}

impl Theme {
    fn palette(&self) -> ColorPalette {
        match self {
            Theme::Dark => ColorPalette::dark(),
            Theme::Light => ColorPalette::light(),
            Theme::Cyberpunk => ColorPalette::cyberpunk(),
            Theme::Nord => ColorPalette::nord(),
            Theme::Dracula => ColorPalette::dracula(),
        }
    }
    fn apply(&self, ctx: &egui::Context) {
        let palette = self.palette();
        let mut visuals = match self {
            Theme::Dark | Theme::Cyberpunk | Theme::Nord | Theme::Dracula => egui::Visuals::dark(),
            Theme::Light => egui::Visuals::light(),
        };
        visuals.window_fill = palette.bg_primary;
        visuals.panel_fill = palette.bg_secondary;
        visuals.override_text_color = Some(palette.text_primary);
        visuals.faint_bg_color = palette.bg_tertiary;
        visuals.extreme_bg_color = palette.bg_primary;
        visuals.selection.bg_fill = palette.accent;
        visuals.selection.stroke = egui::Stroke::new(1.5, palette.accent);
        visuals.widgets.noninteractive.bg_fill = palette.bg_tertiary;
        visuals.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, palette.border);
        visuals.widgets.noninteractive.fg_stroke = egui::Stroke::new(1.0, palette.text_primary);
        visuals.widgets.hovered.bg_fill = palette.accent_hover;
        visuals.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, palette.accent);
        visuals.widgets.hovered.fg_stroke = egui::Stroke::new(1.0, palette.text_primary);
        visuals.widgets.active.bg_fill = palette.accent;
        visuals.widgets.active.bg_stroke = egui::Stroke::new(1.0, palette.accent);
        visuals.widgets.active.fg_stroke = egui::Stroke::new(1.0, palette.text_primary);
        visuals.widgets.open.bg_fill = palette.bg_secondary;
        visuals.widgets.open.bg_stroke = egui::Stroke::new(1.0, palette.border);
        visuals.widgets.open.fg_stroke = egui::Stroke::new(1.0, palette.text_primary);
        ctx.set_visuals(visuals);
    }
}

impl Default for App {
    fn default() -> Self {
        let settings = AppSettings::load();
        let show_settings = settings.python_path.trim().is_empty();
        Self {
            mode: TrainingMode::SFT,
            config: TrainingConfig {
                model_name: "unsloth/Llama-3.2-3B-Instruct-bnb-4bit".to_string(),
                learning_rate: 2e-4, batch_size: 2, lora_rank: 16, max_steps: 60,
                dataset_path: "".to_string(), export_name: "my_model".to_string(),
                data_format: "raw".to_string(), reward_xml: false, reward_length: true,
                preset: "3b".to_string(),
            },
            is_training: false,
            logs: VecDeque::from(vec![("System ready. Configure your training parameters to begin.".to_string(), egui::Color32::GRAY)]),
            loss_history: vec![], reward_history: vec![], lr_history: vec![],
            log_receiver: None, training_handle: None, convert_input_path: "".to_string(),
            settings: settings.clone(), show_settings,
            selected_theme: Theme::Dark, active_tab: NavTab::Train,
            expanded_sections: vec!["model", "data", "params"],
            toasts: vec![], auto_scroll: true, search_text: "".to_string(),
            current_step: 0, max_steps: 60, eta_seconds: None,
            training_start_time: None, gpu_usage: 0, gpu_memory: 0,
            last_gpu_update: Instant::now(), last_frame: Instant::now(),
        }
    }
}

impl App {
    fn add_toast(&mut self, message: String, kind: ToastKind) {
        self.toasts.push(Toast { message, kind, timestamp: Instant::now() });
    }
    fn remove_expired_toasts(&mut self) {
        self.toasts.retain(|t| t.timestamp.elapsed() < Duration::from_secs(5));
    }
    fn update_gpu_stats(&mut self) {
        self.gpu_usage = match std::process::Command::new("nvidia-smi")
            .args(&["--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]).output() {
            Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).trim().parse().unwrap_or(0),
            _ => 0,
        };
        self.gpu_memory = match std::process::Command::new("nvidia-smi")
            .args(&["--query-gpu=memory.used", "--format=csv,noheader,nounits"]).output() {
            Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).trim().parse().unwrap_or(0),
            _ => 0,
        };
    }
    fn apply_preset(&mut self, preset: &str) {
        match preset {
            "1b" => { self.config.model_name = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit".to_string(); self.config.lora_rank = 8; self.config.batch_size = 2; self.config.learning_rate = 3e-4; self.config.max_steps = 100; }
            "3b" => { self.config.model_name = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit".to_string(); self.config.lora_rank = 16; self.config.batch_size = 2; self.config.learning_rate = 2e-4; self.config.max_steps = 60; }
            "7b" => { self.config.model_name = "unsloth/Llama-3.2-7B-Instruct-bnb-4bit".to_string(); self.config.lora_rank = 32; self.config.batch_size = 1; self.config.learning_rate = 2e-4; self.config.max_steps = 50; }
            "70b" => { self.config.model_name = "unsloth/Llama-3.1-70B-Instruct-bnb-4bit".to_string(); self.config.lora_rank = 64; self.config.batch_size = 1; self.config.learning_rate = 1.5e-4; self.config.max_steps = 30; }
            _ => {}
        }
        self.add_toast(format!("Applied {} preset", preset), ToastKind::Info);
    }
    fn toggle_section(&mut self, section: &'static str) {
        if self.expanded_sections.contains(&section) {
            self.expanded_sections.retain(|s| *s != section);
        } else { self.expanded_sections.push(section); }
    }
    fn section_expanded(&self, section: &'static str) -> bool {
        self.expanded_sections.contains(&section)
    }
    fn save_run_config(&self) -> Option<String> {
        let config_path = "../configs/active_run.json";
        let json_data = serde_json::json!({
            "model": { "base_model": self.config.model_name, "load_in_4bit": true, "lora_r": self.config.lora_rank, "lora_alpha": 16, "lora_dropout": 0, "max_seq_length": 2048, "target_modules":["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] },
            "training": { "batch_size": self.config.batch_size, "learning_rate": self.config.learning_rate, "max_steps": self.config.max_steps, "output_dir": "outputs", "grad_accum_steps": 4, "warmup_steps": 10, "logging_steps": 1, "export_name": self.config.export_name, "data_format": self.config.data_format, "reward_xml": self.config.reward_xml, "reward_length": self.config.reward_length },
            "dataset_path": self.config.dataset_path,
            "method": match self.mode { TrainingMode::SFT => "sft", TrainingMode::GRPO => "grpo" }
        });
        if let Ok(mut f) = File::create(config_path) {
            if f.write_all(json_data.to_string().as_bytes()).is_ok() { return Some(config_path.to_string()); }
        }
        None
    }
    fn run_conversion(&mut self) {
        if self.convert_input_path.is_empty() {
            self.logs.push_back(("Please select a source file first.".to_string(), egui::Color32::YELLOW));
            return;
        }
        let input_path = self.convert_input_path.clone();
        let output_path = Path::new(&input_path).with_file_name(format!("{}_ready.jsonl", Path::new(&input_path).file_stem().unwrap().to_str().unwrap())).display().to_string();
        self.logs.push_back((format!("Converting: {} -> {}", Path::new(&input_path).file_name().unwrap().to_str().unwrap(), Path::new(&output_path).file_name().unwrap().to_str().unwrap()), egui::Color32::WHITE));
        let (tx, rx) = channel(); self.log_receiver = Some(rx);
        let tx_err = tx.clone(); let python_path = self.settings.python_path.clone();
        thread::spawn(move || {
            let output = Command::new(python_path).arg("../tools/converter.py").arg("-i").arg(&input_path).arg("-o").arg(&output_path).output();
            match output {
                Ok(out) => {
                    if out.status.success() {
                        let _ = tx.send(serde_json::json!({ "event": "finished_conversion", "message": "Conversion completed successfully!", "data": { "new_path": output_path } }).to_string());
                    } else { let _ = tx_err.send(serde_json::json!({ "event": "error", "message": format!("Conversion failed: {}", String::from_utf8_lossy(&out.stderr)) }).to_string()); }
                }
                Err(e) => { let _ = tx_err.send(serde_json::json!({ "event": "error", "message": format!("Tool execution error: {}", e) }).to_string()); }
            }
        });
    }
    fn start_training(&mut self) {
        if self.is_training { return; }
        if self.config.dataset_path.trim().is_empty() { self.logs.push_back(("Dataset path is required!".to_string(), egui::Color32::YELLOW)); self.add_toast("Please select a dataset".to_string(), ToastKind::Warning); return; }
        if self.settings.python_path.trim().is_empty() { self.logs.push_back(("Python path not configured!".to_string(), egui::Color32::YELLOW)); self.show_settings = true; self.add_toast("Configure Python path in Settings".to_string(), ToastKind::Warning); return; }
        self.save_run_config(); self.is_training = true;
        self.logs.push_back(("Initializing training environment...".to_string(), egui::Color32::from_rgb(100, 180, 255)));
        self.loss_history.clear(); self.reward_history.clear(); self.lr_history.clear();
        self.current_step = 0; self.max_steps = self.config.max_steps;
        self.training_start_time = Some(Instant::now());
        self.add_toast("Training started".to_string(), ToastKind::Success);
        let (tx, rx) = channel(); self.log_receiver = Some(rx); let tx_err = tx.clone(); let python_path = self.settings.python_path.clone();
        let child_result = Command::new(python_path).arg("-u").arg("../engine/main.py").arg("--config").arg("../configs/active_run.json").stdout(Stdio::piped()).stderr(Stdio::piped()).spawn();
        match child_result {
            Ok(mut child) => {
                if let Some(stderr) = child.stderr.take() { thread::spawn(move || { let r = BufReader::new(stderr); for l in r.lines() { if let Ok(line) = l { let _ = tx_err.send(serde_json::json!({ "event": "stderr", "message": line }).to_string()); } } }); }
                if let Some(stdout) = child.stdout.take() { thread::spawn(move || { let r = BufReader::new(stdout); for l in r.lines() { if let Ok(line) = l { let _ = tx.send(line); } } }); }
                self.training_handle = Some(child);
            }
            Err(e) => { self.logs.push_back((format!("Failed to start training: {}", e), egui::Color32::RED)); self.is_training = false; self.add_toast(format!("Failed to start: {}", e), ToastKind::Error); }
        }
    }
    fn stop_training(&mut self) {
        if let Some(mut child) = self.training_handle.take() { let _ = child.kill(); let _ = child.wait(); self.logs.push_back(("Training terminated by user".to_string(), egui::Color32::from_rgb(255, 150, 100))); }
        self.is_training = false; self.training_start_time = None; self.add_toast("Training stopped".to_string(), ToastKind::Warning);
    }
    fn parse_log(&mut self, msg: String) {
        if let Ok(event) = serde_json::from_str::<LogEvent>(&msg) {
            match event.event.as_str() {
                "step" => {
                    if let Some(data) = event.data {
                        // Extract all metrics
                        let loss_val = data.get("loss").and_then(|v| v.as_f64());
                        let reward_val = data.get("reward").and_then(|v| v.as_f64());
                        let lr_val = data.get("learning_rate").and_then(|v| v.as_f64()); 
                        
                        let step = event.step.unwrap_or(0); 
                        let max = event.max_steps.unwrap_or(self.config.max_steps);
                        
                        self.current_step = step; 
                        self.max_steps = max;
                        
                        // Push to charts
                        if let Some(v) = loss_val { self.loss_history.push([step as f64, v]); }
                        if let Some(v) = reward_val { self.reward_history.push([step as f64, v]); }
                        if let Some(v) = lr_val { self.lr_history.push([step as f64, v]); } 
                        
                        if let Some(start_time) = self.training_start_time {
                            let elapsed = start_time.elapsed().as_secs_f64();
                            if step > 0 && max > 0 { 
                                let secs_per_step = elapsed / step as f64; 
                                let remaining_steps = max - step; 
                                self.eta_seconds = Some((secs_per_step * remaining_steps as f64) as u64); 
                            }
                        }
                        let progress = if max > 0 { format!("{}/{} ({:.1}%)", step, max, (step as f64 / max as f64) * 100.0) } else { format!("{}", step) };
                        let metric_str = if let Some(loss) = loss_val { format!("Loss: {:.6}", loss) } else if let Some(reward) = reward_val { format!("Reward: {:.4}", reward) } else { "".to_string() };
                        self.logs.push_back((format!("[{}] {}", progress, metric_str), egui::Color32::from_rgb(100, 255, 150)));
                    }
                }
                "status" => self.logs.push_back((format!("{}", event.message.unwrap_or_default()), egui::Color32::from_rgb(100, 180, 255))),
                "finished" => { self.logs.push_back(("Training completed successfully!".to_string(), egui::Color32::from_rgb(100, 255, 150))); if let Some(msg) = event.message { self.logs.push_back((format!("{}", msg), egui::Color32::from_rgb(100, 180, 255))); } self.is_training = false; self.training_handle = None; self.training_start_time = None; self.eta_seconds = None; self.add_toast("Training completed!".to_string(), ToastKind::Success); }
                "finished_conversion" => { self.logs.push_back(("Data conversion completed!".to_string(), egui::Color32::from_rgb(100, 255, 150))); if let Some(new_path) = event.data.and_then(|d| d.get("new_path").map(|s| s.as_str().unwrap().to_string())) { self.config.dataset_path = new_path; } self.add_toast("Conversion complete".to_string(), ToastKind::Success); }
                "error" => self.logs.push_back((format!("Error: {}", event.message.unwrap_or_default()), egui::Color32::from_rgb(255, 100, 100))),
                "stderr" => { let text = event.message.unwrap_or_default(); if text.contains("Error") || text.contains("Traceback") { self.logs.push_back((format!("{}", text), egui::Color32::from_rgb(255, 100, 100))); } else if !text.contains("%") && !text.contains("it/s") && !text.is_empty() { self.logs.push_back((text, egui::Color32::from_rgb(255, 200, 100))); } }
                _ => {}
            }
        }
    }

    fn render_status_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 20.0;
            
            let (status_text, status_color) = if self.is_training {
                ("TRAINING", egui::Color32::GREEN)
            } else {
                ("READY", egui::Color32::GRAY)
            };
            
            ui.label(egui::RichText::new("•").color(status_color).size(16.0));
            ui.label(egui::RichText::new(status_text).color(status_color).strong());
            
            ui.separator();
            
            ui.label(egui::RichText::new(format!("Mode: {:?}", self.mode)).color(egui::Color32::LIGHT_GRAY));
            
            if !self.loss_history.is_empty() {
                let current_step = self.loss_history.last().unwrap()[0] as u32;
                ui.label(egui::RichText::new(format!("Step: {}", current_step)).color(egui::Color32::LIGHT_GRAY));
            }
        });
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.is_training { ctx.request_repaint(); }
        
        self.selected_theme.apply(ctx);
        self.remove_expired_toasts();
        if self.last_gpu_update.elapsed() > Duration::from_secs(1) { self.update_gpu_stats(); self.last_gpu_update = Instant::now(); }
        let palette = self.selected_theme.palette();
        
        let mut messages = Vec::new();
        if let Some(rx) = &self.log_receiver { while let Ok(msg) = rx.try_recv() { messages.push(msg); } }
        for msg in messages { self.parse_log(msg); }

        // --- SETUP SCREEN FIX ---
        if self.show_settings {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.vertical_centered(|ui| {
                    ui.add_space(150.0);
                    ui.heading(egui::RichText::new("Unsloth Studio Setup").size(42.0).strong().color(palette.accent));
                    ui.add_space(10.0);
                    ui.label(egui::RichText::new("Welcome! Please configure your Python environment to begin.").size(18.0).color(palette.text_secondary));
                    ui.add_space(40.0);
                    
                    ui.group(|ui| {
                        ui.set_max_width(600.0);
                        ui.vertical_centered(|ui| {
                            ui.add_space(20.0);
                            ui.label(egui::RichText::new("Python Executable Path").size(16.0).strong().color(palette.text_primary));
                            ui.add_space(10.0);
                            ui.horizontal(|ui| {
                                ui.add(egui::TextEdit::singleline(&mut self.settings.python_path)
                                    .desired_width(400.0)
                                    .hint_text("e.g., /home/user/miniconda3/envs/ai/bin/python3"));
                                if ui.button(egui::RichText::new("Browse").size(14.0)).clicked() { 
                                    if let Some(path) = rfd::FileDialog::new().pick_file() { 
                                        self.settings.python_path = path.display().to_string(); 
                                    } 
                                }
                            });
                            ui.add_space(20.0);
                        });
                    });
                    
                    ui.add_space(30.0);
                    if ui.add_enabled(
                        !self.settings.python_path.trim().is_empty(), 
                        egui::Button::new(egui::RichText::new("Save & Launch").size(18.0).strong())
                            .min_size(egui::vec2(200.0, 50.0))
                            .fill(palette.accent)
                    ).clicked() { 
                        self.settings.save(); 
                        self.show_settings = false; 
                    }
                });
            });
            return;
        }
        // --- END SETUP SCREEN FIX ---

        egui::TopBottomPanel::top("top_bar").exact_height(56.0).show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.add_space(24.0);
                ui.vertical_centered(|ui| {
                    ui.heading(egui::RichText::new("Unsloth Studio").size(22.0).strong().color(palette.accent));
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.add_space(16.0);
                    egui::ComboBox::from_id_salt("theme_selector").selected_text(match self.selected_theme { Theme::Dark => "Dark", Theme::Light => "Light", Theme::Cyberpunk => "Cyberpunk", Theme::Nord => "Nord", Theme::Dracula => "Dracula" }).width(100.0).show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.selected_theme, Theme::Dark, "Dark");
                        ui.selectable_value(&mut self.selected_theme, Theme::Light, "Light");
                        ui.selectable_value(&mut self.selected_theme, Theme::Cyberpunk, "Cyberpunk");
                        ui.selectable_value(&mut self.selected_theme, Theme::Nord, "Nord");
                        ui.selectable_value(&mut self.selected_theme, Theme::Dracula, "Dracula");
                    });
                });
            });
        });

        egui::SidePanel::left("sidebar").resizable(false).default_width(220.0).show(ctx, |ui| {
            ui.add_space(12.0);
            ui.vertical(|ui| {
                for (tab, label) in[(NavTab::Train, "Train"), (NavTab::Monitor, "Monitor"), (NavTab::Logs, "Logs"), (NavTab::Settings, "Settings")] {
                    let is_active = self.active_tab == tab;
                    let resp = ui.add(egui::Button::new(egui::RichText::new(label).size(15.0).color(if is_active { palette.accent } else { palette.text_secondary })).min_size(egui::vec2(0.0, 44.0)).fill(if is_active { palette.sidebar_active } else { egui::Color32::TRANSPARENT }).corner_radius(8.0));
                    if resp.clicked() { self.active_tab = tab; }
                    ui.add_space(4.0);
                }
            });
            ui.with_layout(egui::Layout::bottom_up(egui::Align::Center), |ui| {
                ui.add_space(16.0); ui.separator(); ui.add_space(16.0);
                let status_color = if self.is_training { palette.success } else { palette.text_muted };
                ui.horizontal(|ui| {
                    if self.is_training { ui.painter().circle_filled(ui.cursor().left_center() - egui::vec2(28.0, 0.0), 5.0, status_color); }
                    ui.label(egui::RichText::new(if self.is_training { "Training" } else { "Ready" }).size(13.0).color(status_color));
                });
                ui.add_space(12.0);
            });
        });

        match self.active_tab {
            NavTab::Train => self.render_train_tab(ctx, &palette),
            NavTab::Monitor => self.render_monitor_tab(ctx, &palette),
            NavTab::Logs => self.render_logs_tab(ctx, &palette),
            NavTab::Settings => self.render_settings_tab(ctx, &palette),
        }

        egui::TopBottomPanel::bottom("status_bar").exact_height(28.0).show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.add_space(20.0);
                ui.label(egui::RichText::new(format!("GPU: {}% | {}MB", self.gpu_usage, self.gpu_memory)).size(12.0).color(palette.text_secondary));
                if self.is_training { if let Some(eta) = self.eta_seconds { ui.add_space(16.0); ui.label(egui::RichText::new(format!("ETA: {}m {}s", eta / 60, eta % 60)).size(12.0).color(palette.warning)); } }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| { ui.add_space(20.0); ui.label(egui::RichText::new("v1.0").size(11.0).color(palette.text_muted)); });
            });
        });

        for toast in &self.toasts {
            let age = toast.timestamp.elapsed().as_secs_f32();
            let alpha = if age < 3.5 { 1.0 } else { (4.5 - age).max(0.0) };
            if alpha > 0.0 {
                let rect = egui::Rect::from_center_size(egui::pos2(ctx.available_rect().center().x, 60.0), egui::vec2(320.0, 44.0));
                let painter = ctx.debug_painter();
                painter.rect_filled(rect, 8.0, toast.kind.color().gamma_multiply(alpha * 0.9));
                painter.rect_stroke(rect, 8.0, (1.0, toast.kind.color().gamma_multiply(alpha * 0.5)), egui::StrokeKind::Outside);
                painter.text(rect.center(), egui::Align2::CENTER_CENTER, &toast.message, egui::FontId::proportional(13.0), egui::Color32::WHITE.gamma_multiply(alpha));
            }
        }
    }
}

impl App {
    fn render_train_tab(&mut self, ctx: &egui::Context, palette: &ColorPalette) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add_space(16.0);
            ui.horizontal(|ui| {
                ui.heading(egui::RichText::new("Training Configuration").size(20.0).strong().color(palette.text_primary));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if self.is_training {
                        let progress = if self.max_steps > 0 { (self.current_step as f64 / self.max_steps as f64) * 100.0 } else { 0.0 };
                        ui.label(egui::RichText::new(format!("{:.1}%", progress)).size(26.0).strong().color(palette.success));
                    }
                });
            });
            ui.add_space(20.0);
            ui.horizontal(|ui| {
                ui.set_min_width(340.0);
                ui.vertical(|ui| {
                    ui.group(|ui| {
                        ui.add_space(10.0);
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("Model & Strategy").size(15.0).strong().color(palette.accent));
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if ui.button(if self.section_expanded("model") { "−" } else { "+" }).clicked() { self.toggle_section("model"); }
                            });
                        });
                        ui.add_space(8.0);
                        if self.section_expanded("model") {
                            ui.label("Base Model:"); ui.add(egui::TextEdit::singleline(&mut self.config.model_name).hint_text("HuggingFace model..."));
                            ui.add_space(12.0); ui.separator(); ui.add_space(12.0);
                            ui.label(egui::RichText::new("Training Mode").size(13.0).color(palette.text_secondary));
                            ui.add_space(6.0);
                            egui::ComboBox::from_id_salt("mode_selector").selected_text(match self.mode { TrainingMode::SFT => "SFT", TrainingMode::GRPO => "GRPO" }).width(ui.available_width()).show_ui(ui, |ui| {
                                ui.selectable_value(&mut self.mode, TrainingMode::SFT, "SFT - Supervised Fine-Tuning");
                                ui.selectable_value(&mut self.mode, TrainingMode::GRPO, "GRPO - Reward Optimization");
                            });
                            if self.mode == TrainingMode::GRPO { ui.add_space(10.0); ui.separator(); ui.add_space(10.0); ui.checkbox(&mut self.config.reward_length, "Length-based Rewards"); ui.checkbox(&mut self.config.reward_xml, "DeepSeek XML Format"); }
                        }
                        ui.add_space(8.0);
                    });
                    ui.add_space(16.0);
                    ui.group(|ui| {
                        ui.add_space(10.0);
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("Data Configuration").size(15.0).strong().color(palette.accent));
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if ui.button(if self.section_expanded("data") { "−" } else { "+" }).clicked() { self.toggle_section("data"); }
                            });
                        });
                        ui.add_space(8.0);
                        if self.section_expanded("data") {
                            ui.label("Dataset:");
                            ui.horizontal(|ui| {
                                ui.add(egui::TextEdit::singleline(&mut self.config.dataset_path).hint_text("Path to dataset...").desired_width(ui.available_width() - 85.0));
                                if ui.button("Browse").clicked() { if let Some(path) = rfd::FileDialog::new().add_filter("Dataset", &["jsonl", "parquet", "json"]).pick_file() { self.config.dataset_path = path.display().to_string(); } }
                            });
                            ui.add_space(12.0); ui.separator(); ui.add_space(12.0);
                            ui.label(egui::RichText::new("Data Converter").size(13.0).color(palette.text_secondary));
                            ui.add_space(6.0);
                            ui.horizontal(|ui| {
                                ui.add(egui::TextEdit::singleline(&mut self.convert_input_path).hint_text("Source file...").desired_width(ui.available_width() - 85.0));
                                if ui.button("Browse").clicked() { if let Some(path) = rfd::FileDialog::new().add_filter("Source", &["json", "csv", "txt"]).pick_file() { self.convert_input_path = path.display().to_string(); } }
                            });
                            ui.add_space(8.0);
                            if ui.add(egui::Button::new(egui::RichText::new("Convert").size(13.0)).min_size(egui::vec2(ui.available_width(), 36.0)).fill(palette.bg_tertiary)).clicked() { self.run_conversion(); }
                        }
                        ui.add_space(8.0);
                    });
                    ui.add_space(16.0);
                    ui.group(|ui| {
                        ui.add_space(10.0);
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("Training Parameters").size(15.0).strong().color(palette.accent));
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if ui.button(if self.section_expanded("params") { "−" } else { "+" }).clicked() { self.toggle_section("params"); }
                            });
                        });
                        ui.add_space(8.0);
                        if self.section_expanded("params") {
                            ui.horizontal_wrapped(|ui| { ui.label("Max Steps:"); ui.add(egui::Slider::new(&mut self.config.max_steps, 1..=10000).show_value(true).trailing_fill(true)); });
                            ui.add_space(6.0);
                            ui.horizontal_wrapped(|ui| { ui.label("Learning Rate:"); ui.add(egui::DragValue::new(&mut self.config.learning_rate).speed(1e-5).range(1e-6..=1e-2).prefix("x")); ui.label(egui::RichText::new(format!("{:e}", self.config.learning_rate)).size(11.0).color(palette.text_muted)); });
                            ui.add_space(6.0);
                            ui.horizontal_wrapped(|ui| { ui.label("Batch Size:"); ui.add(egui::Slider::new(&mut self.config.batch_size, 1..=32).show_value(true).trailing_fill(true)); });
                            ui.add_space(6.0);
                            ui.horizontal_wrapped(|ui| { ui.label("LoRA Rank:"); ui.add(egui::Slider::new(&mut self.config.lora_rank, 8..=128).step_by(8.0).show_value(true).trailing_fill(true)); });
                            ui.add_space(8.0); ui.separator(); ui.add_space(8.0);
                            ui.label("Export Name:"); ui.text_edit_singleline(&mut self.config.export_name);
                        }
                        ui.add_space(8.0);
                    });
                });
                ui.add_space(20.0);
                ui.vertical(|ui| {
                    ui.set_min_width(420.0);
                    ui.group(|ui| {
                        ui.add_space(16.0);
                        ui.heading(egui::RichText::new("Training Progress").size(16.0).strong().color(palette.text_primary));
                        ui.add_space(12.0);
                        let progress = if self.max_steps > 0 { self.current_step as f64 / self.max_steps as f64 } else { 0.0 };
                        ui.add(egui::ProgressBar::new(progress as f32).desired_width(ui.available_width()).fill(palette.accent));
                        ui.add_space(8.0);
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new(format!("{}/{}", self.current_step, self.max_steps)).size(13.0).color(palette.text_secondary));
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| { ui.label(egui::RichText::new(format!("{:.1}%", progress * 100.0)).size(13.0).strong().color(palette.accent)); });
                        });
                        if self.is_training {
                            ui.add_space(16.0); ui.separator(); ui.add_space(16.0);
                            ui.heading(egui::RichText::new("Live Metrics").size(15.0).strong().color(palette.text_primary));
                            ui.add_space(10.0);
                            if !self.loss_history.is_empty() {
                                let last_loss = self.loss_history.last().unwrap()[1];
                                ui.horizontal(|ui| { ui.label(egui::RichText::new("Loss:").size(13.0).color(palette.text_secondary)); ui.label(egui::RichText::new(format!("{:.6}", last_loss)).size(15.0).strong().color(palette.success)); });
                                if self.reward_history.len() > 1 { ui.add_space(6.0); ui.horizontal(|ui| { ui.label(egui::RichText::new("Reward:").size(13.0).color(palette.text_secondary)); ui.label(egui::RichText::new(format!("{:.4}", self.reward_history.last().unwrap()[1])).size(15.0).strong().color(palette.accent)); }); }
                            } else { ui.label(egui::RichText::new("Waiting for metrics...").size(13.0).color(palette.text_muted)); }
                            ui.add_space(16.0);
                            ui.heading(egui::RichText::new("Loss Curve").size(15.0).strong().color(palette.text_primary));
                            ui.add_space(8.0);
                            let plot_height = 180.0;
                            egui::Frame::dark_canvas(ui.style()).show(ui, |ui| {
                                ui.allocate_ui_with_layout(egui::Vec2::new(ui.available_width(), plot_height), egui::Layout::top_down(egui::Align::Center), |ui| {
                                    if !self.loss_history.is_empty() {
                                        let line = Line::new("loss", self.loss_history.clone()).color(palette.chart_line).width(2.5);
                                        Plot::new("loss_plot").view_aspect(2.0).auto_bounds(egui::Vec2b::TRUE).show(ui, |p| { p.line(line); });
                                    } else { ui.centered_and_justified(|ui| { ui.label(egui::RichText::new("Training metrics will appear here").size(13.0).color(palette.text_muted)); }); }
                                });
                            });
                        } else { ui.add_space(20.0); ui.centered_and_justified(|ui| { ui.label(egui::RichText::new("Configure settings and press Start").size(13.0).color(palette.text_muted)); }); }
                        ui.add_space(12.0);
                    });
                    ui.add_space(16.0);
                    ui.with_layout(egui::Layout::bottom_up(egui::Align::Center), |ui| {
                        if self.is_training {
                            if ui.add(egui::Button::new(egui::RichText::new("Stop Training").size(16.0).strong()).min_size(egui::vec2(280.0, 50.0)).fill(palette.error).corner_radius(10.0)).clicked() { self.stop_training(); }
                        } else {
                            let can_start = !self.config.dataset_path.trim().is_empty() && !self.settings.python_path.trim().is_empty();
                            if ui.add_enabled(can_start, egui::Button::new(egui::RichText::new("Start Training").size(16.0).strong()).min_size(egui::vec2(280.0, 50.0)).fill(palette.success).corner_radius(10.0)).clicked() { self.start_training(); }
                            if !can_start { ui.add_space(6.0); ui.label(egui::RichText::new("Add a dataset to get started").size(12.0).color(palette.warning)); }
                        }
                    });
                });
            });
        });
    }

    fn render_monitor_tab(&mut self, ctx: &egui::Context, palette: &ColorPalette) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add_space(16.0);
            ui.heading(egui::RichText::new("Training Monitor").size(20.0).strong().color(palette.text_primary));
            ui.add_space(20.0);
            ui.horizontal(|ui| {
                ui.set_min_width(280.0);
                ui.vertical(|ui| {
                    ui.group(|ui| {
                        ui.add_space(14.0);
                        ui.heading(egui::RichText::new("GPU Status").size(15.0).strong().color(palette.accent));
                        ui.add_space(12.0);
                        ui.horizontal(|ui| { ui.label(egui::RichText::new("Usage").size(13.0).color(palette.text_secondary)); ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| { ui.label(egui::RichText::new(format!("{}%", self.gpu_usage)).size(15.0).strong().color(match self.gpu_usage { 0..=50 => palette.success, 51..=80 => palette.warning, _ => palette.error })); }); });
                        ui.add_space(6.0);
                        ui.add(egui::ProgressBar::new((self.gpu_usage as f32 / 100.0).min(1.0)).desired_width(ui.available_width()).fill(match self.gpu_usage { 0..=50 => palette.success, 51..=80 => palette.warning, _ => palette.error }));
                        ui.add_space(12.0);
                        ui.horizontal(|ui| { ui.label(egui::RichText::new("VRAM").size(13.0).color(palette.text_secondary)); ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| { ui.label(egui::RichText::new(format!("{} MB", self.gpu_memory)).size(15.0).strong().color(palette.accent)); }); });
                        ui.add_space(6.0);
                        let vram_pct = (self.gpu_memory as f32 / 16384.0).min(1.0);
                        ui.add(egui::ProgressBar::new(vram_pct).desired_width(ui.available_width()).fill(palette.accent));
                        ui.add_space(6.0);
                        ui.label(egui::RichText::new("of 16 GB").size(11.0).color(palette.text_muted));
                        ui.add_space(12.0);
                    });
                    ui.add_space(16.0);
                    ui.group(|ui| {
                        ui.add_space(14.0);
                        ui.heading(egui::RichText::new("Training Stats").size(15.0).strong().color(palette.accent));
                        ui.add_space(12.0);
                        let stats =[("Step", format!("{}", self.current_step)), ("Total", format!("{}", self.max_steps)), ("Progress", format!("{:.1}%", if self.max_steps > 0 { (self.current_step as f64 / self.max_steps as f64) * 100.0 } else { 0.0 }))];
                        for (label, value) in stats { ui.horizontal(|ui| { ui.label(egui::RichText::new(label).size(12.0).color(palette.text_secondary)); ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| { ui.label(egui::RichText::new(value).size(14.0).strong().color(palette.text_primary)); }); }); ui.add_space(6.0); }
                        if let Some(start) = self.training_start_time { let e = start.elapsed(); ui.horizontal(|ui| { ui.label(egui::RichText::new("Time").size(12.0).color(palette.text_secondary)); ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| { ui.label(egui::RichText::new(format!("{}m {}s", e.as_secs() / 60, e.as_secs() % 60)).size(14.0).strong().color(palette.text_primary)); }); }); ui.add_space(6.0); }
                        if let Some(eta) = self.eta_seconds { ui.horizontal(|ui| { ui.label(egui::RichText::new("ETA").size(12.0).color(palette.text_secondary)); ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| { ui.label(egui::RichText::new(format!("{}m {}s", eta / 60, eta % 60)).size(14.0).strong().color(palette.warning)); }); }); }
                        ui.add_space(8.0);
                    });
                });
                ui.add_space(16.0);
                ui.vertical(|ui| {
                    ui.set_min_width(520.0);
                    ui.group(|ui| {
                        ui.add_space(14.0);
                        ui.heading(egui::RichText::new("Loss").size(15.0).strong().color(palette.text_primary));
                        ui.add_space(8.0);
                        let plot_height = 160.0;
                        egui::Frame::dark_canvas(ui.style()).show(ui, |ui| {
                            ui.allocate_ui_with_layout(egui::Vec2::new(ui.available_width(), plot_height), egui::Layout::top_down(egui::Align::Center), |ui| {
                                if !self.loss_history.is_empty() { let line = Line::new("loss", self.loss_history.clone()).color(palette.chart_line).width(2.5); Plot::new("loss_chart").view_aspect(2.0).auto_bounds(egui::Vec2b::TRUE).show(ui, |p| { p.line(line); }); }
                                else { ui.centered_and_justified(|ui| { ui.label(egui::RichText::new("No training data").size(13.0).color(palette.text_muted)); }); }
                            });
                        });
                        ui.add_space(12.0);
                    });
                    ui.add_space(12.0);
                    if !self.reward_history.is_empty() {
                        ui.group(|ui| {
                            ui.add_space(14.0);
                            ui.heading(egui::RichText::new("Reward").size(15.0).strong().color(palette.text_primary));
                            ui.add_space(8.0);
                            let plot_height = 140.0;
                            egui::Frame::dark_canvas(ui.style()).show(ui, |ui| {
                                ui.allocate_ui_with_layout(egui::Vec2::new(ui.available_width(), plot_height), egui::Layout::top_down(egui::Align::Center), |ui| {
                                    let line = Line::new("reward", self.reward_history.clone()).color(egui::Color32::from_rgb(255, 180, 50)).width(2.5);
                                    Plot::new("reward_chart").view_aspect(2.0).auto_bounds(egui::Vec2b::TRUE).show(ui, |p| { p.line(line); });
                                });
                            });
                            ui.add_space(12.0);
                        });
                    } else { ui.group(|ui| { ui.add_space(50.0); ui.centered_and_justified(|ui| { ui.label(egui::RichText::new("Reward curves appear in GRPO mode").size(13.0).color(palette.text_muted)); }); ui.add_space(50.0); }); }
                    ui.add_space(12.0);
                    if !self.lr_history.is_empty() {
                        ui.group(|ui| {
                            ui.add_space(14.0);
                            ui.heading(egui::RichText::new("Learning Rate").size(15.0).strong().color(palette.text_primary));
                            ui.add_space(8.0);
                            let plot_height = 120.0;
                            egui::Frame::dark_canvas(ui.style()).show(ui, |ui| {
                                ui.allocate_ui_with_layout(egui::Vec2::new(ui.available_width(), plot_height), egui::Layout::top_down(egui::Align::Center), |ui| {
                                    let line = Line::new("lr", self.lr_history.clone()).color(egui::Color32::from_rgb(200, 100, 255)).width(2.0);
                                    Plot::new("lr_chart").view_aspect(2.0).auto_bounds(egui::Vec2b::TRUE).show(ui, |p| { p.line(line); });
                                });
                            });
                            ui.add_space(12.0);
                        });
                    }
                });
            });
        });
    }

    fn render_logs_tab(&mut self, ctx: &egui::Context, palette: &ColorPalette) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add_space(16.0);
            ui.horizontal(|ui| {
                ui.heading(egui::RichText::new("Training Logs").size(20.0).strong().color(palette.text_primary));
                ui.add_space(16.0);
                ui.checkbox(&mut self.auto_scroll, "Auto-scroll");
                if ui.button("Clear").clicked() { self.logs.clear(); self.logs.push_back(("Log cleared".to_string(), egui::Color32::GRAY)); }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| { ui.add_space(8.0); ui.add(egui::TextEdit::singleline(&mut self.search_text).hint_text("Filter logs...").desired_width(150.0)); });
            });
            ui.add_space(12.0);
            ui.group(|ui| {
                let scroll_area = egui::ScrollArea::vertical().stick_to_bottom(self.auto_scroll).max_height(ui.available_height() - 10.0);
                scroll_area.show(ui, |ui| {
                    ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Wrap);
                    let search_lower = self.search_text.to_lowercase();
                    for (log_text, color) in &self.logs {
                        let show = search_lower.is_empty() || log_text.to_lowercase().contains(&search_lower);
                        if show { ui.label(egui::RichText::new(log_text).color(*color).font(egui::FontId::monospace(12.0))); }
                    }
                });
            });
        });
    }

    fn render_settings_tab(&mut self, ctx: &egui::Context, palette: &ColorPalette) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add_space(16.0);
            ui.heading(egui::RichText::new("Settings").size(20.0).strong().color(palette.text_primary));
            ui.add_space(24.0);
            ui.vertical(|ui| {
                ui.set_max_width(520.0);
                ui.group(|ui| {
                    ui.add_space(14.0);
                    ui.heading(egui::RichText::new("Python Configuration").size(15.0).strong().color(palette.accent));
                    ui.add_space(12.0);
                    ui.label("Python Path:");
                    ui.add_space(6.0);
                    ui.horizontal(|ui| {
                        ui.add(egui::TextEdit::singleline(&mut self.settings.python_path).desired_width(ui.available_width() - 90.0).hint_text("/usr/bin/python3"));
                        if ui.button("Browse").clicked() { if let Some(path) = rfd::FileDialog::new().pick_file() { self.settings.python_path = path.display().to_string(); self.settings.save(); } }
                    });
                    ui.add_space(10.0);
                    if self.settings.python_path.trim().is_empty() { ui.label(egui::RichText::new("Required for training operations").size(12.0).color(palette.warning)); } else { ui.label(egui::RichText::new("Configured").size(12.0).color(palette.success)); }
                    ui.add_space(10.0);
                });
                ui.add_space(16.0);
                ui.group(|ui| {
                    ui.add_space(14.0);
                    ui.heading(egui::RichText::new("Theme").size(15.0).strong().color(palette.accent));
                    ui.add_space(12.0);
                    for (theme, name, desc) in[(Theme::Dark, "Dark", "Clean dark theme"), (Theme::Light, "Light", "Bright minimal theme"), (Theme::Cyberpunk, "Cyberpunk", "Neon aesthetics"), (Theme::Nord, "Nord", "Cool blue tones"), (Theme::Dracula, "Dracula", "Dark purple")] {
                        ui.horizontal(|ui| { ui.selectable_value(&mut self.selected_theme, theme.clone(), name); ui.label(egui::RichText::new(desc).size(12.0).color(palette.text_muted)); });
                    }
                    ui.add_space(10.0);
                });
                ui.add_space(16.0);
                ui.group(|ui| {
                    ui.add_space(14.0);
                    ui.heading(egui::RichText::new("About").size(15.0).strong().color(palette.accent));
                    ui.add_space(12.0);
                    ui.label(egui::RichText::new("Unsloth Studio").size(16.0).strong());
                    ui.label(egui::RichText::new("Version 1.0").size(12.0).color(palette.text_secondary));
                    ui.add_space(8.0);
                    ui.label(egui::RichText::new("AI Training Launcher powered by Unsloth").size(12.0).color(palette.text_muted));
                    ui.add_space(10.0);
                });
            });
        });
    }
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1400.0, 900.0]).with_min_inner_size([1100.0, 700.0]).with_title("Unsloth Studio Pro"),
        ..Default::default()
    };
    eframe::run_native("Unsloth Studio Pro", options, Box::new(|cc| { cc.egui_ctx.set_visuals(egui::Visuals::dark()); Ok(Box::new(App::default())) }))
}