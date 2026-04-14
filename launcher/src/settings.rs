use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Serialize, Deserialize, Default, Clone)]
pub struct AppSettings {
    pub python_path: String,
}

impl AppSettings {
    pub fn load() -> Self {
        if let Ok(content) = fs::read_to_string("settings.json") {
            serde_json::from_str(&content).unwrap_or_default()
        } else {
            Self::default()
        }
    }

    pub fn save(&self) {
        if let Ok(json) = serde_json::to_string_pretty(self) {
            let _ = fs::write("settings.json", json);
        }
    }
}