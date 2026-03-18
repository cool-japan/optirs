//! WASM wrapper for the CurriculumScheduler learning rate scheduler.

use optirs_core::schedulers::{CurriculumScheduler, CurriculumStage, TransitionStrategy};

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// WASM-compatible wrapper for the CurriculumScheduler.
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub struct WasmCurriculumScheduler {
    inner: CurriculumScheduler<f64>,
}

/// Parse a JSON string into curriculum stages.
///
/// Expected format: `[{"learning_rate": 0.01, "duration": 100}, ...]`
fn parse_stages(stages_json: &str) -> Result<Vec<CurriculumStage<f64>>, String> {
    // Minimal JSON parsing without serde dependency.
    // Parse an array of objects with "learning_rate" and "duration" fields.
    let trimmed = stages_json.trim();
    if !trimmed.starts_with('[') || !trimmed.ends_with(']') {
        return Err("stages_json must be a JSON array".to_string());
    }

    let inner = &trimmed[1..trimmed.len() - 1];
    if inner.trim().is_empty() {
        return Err("stages_json must contain at least one stage".to_string());
    }

    let mut stages = Vec::new();
    let mut depth = 0;
    let mut start = 0;

    // Split by commas at depth 0 (outside braces)
    for (i, ch) in inner.char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => depth -= 1,
            ',' if depth == 0 => {
                let obj_str = inner[start..i].trim();
                stages.push(parse_single_stage(obj_str)?);
                start = i + 1;
            }
            _ => {}
        }
    }

    // Parse the last (or only) object
    let last_str = inner[start..].trim();
    if !last_str.is_empty() {
        stages.push(parse_single_stage(last_str)?);
    }

    if stages.is_empty() {
        return Err("stages_json must contain at least one stage".to_string());
    }

    Ok(stages)
}

/// Parse a single JSON object into a CurriculumStage.
fn parse_single_stage(obj_str: &str) -> Result<CurriculumStage<f64>, String> {
    let trimmed = obj_str.trim();
    if !trimmed.starts_with('{') || !trimmed.ends_with('}') {
        return Err(format!("Invalid stage object: {}", trimmed));
    }

    let inner = &trimmed[1..trimmed.len() - 1];
    let mut learning_rate: Option<f64> = None;
    let mut duration: Option<usize> = None;
    let mut description: Option<String> = None;

    // Parse key-value pairs
    for pair in inner.split(',') {
        let pair = pair.trim();
        if pair.is_empty() {
            continue;
        }

        let parts: Vec<&str> = pair.splitn(2, ':').collect();
        if parts.len() != 2 {
            continue;
        }

        let key = parts[0].trim().trim_matches('"');
        let value = parts[1].trim();

        match key {
            "learning_rate" | "lr" => {
                learning_rate = Some(
                    value
                        .parse::<f64>()
                        .map_err(|e| format!("Invalid learning_rate: {}", e))?,
                );
            }
            "duration" | "steps" => {
                duration = Some(
                    value
                        .parse::<usize>()
                        .map_err(|e| format!("Invalid duration: {}", e))?,
                );
            }
            "description" | "desc" => {
                description = Some(value.trim_matches('"').to_string());
            }
            _ => {}
        }
    }

    let learning_rate =
        learning_rate.ok_or_else(|| "Missing 'learning_rate' field in stage".to_string())?;
    let duration = duration.ok_or_else(|| "Missing 'duration' field in stage".to_string())?;

    Ok(CurriculumStage {
        learning_rate,
        duration,
        description,
    })
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
impl WasmCurriculumScheduler {
    /// Create a new curriculum scheduler with smooth transitions.
    ///
    /// # Arguments
    ///
    /// * `stages_json` - JSON array of stages, e.g. `[{"learning_rate": 0.01, "duration": 100}, ...]`
    /// * `final_lr` - Learning rate to use after all stages are complete
    #[cfg_attr(feature = "wasm", wasm_bindgen(constructor))]
    pub fn new(stages_json: &str, final_lr: f64) -> Result<WasmCurriculumScheduler, String> {
        let stages = parse_stages(stages_json)?;
        Ok(Self {
            inner: CurriculumScheduler::new(
                stages,
                TransitionStrategy::Smooth { blend_steps: 10 },
                final_lr,
            ),
        })
    }

    /// Create a new curriculum scheduler with immediate transitions.
    ///
    /// # Arguments
    ///
    /// * `stages_json` - JSON array of stages, e.g. `[{"learning_rate": 0.01, "duration": 100}, ...]`
    /// * `final_lr` - Learning rate to use after all stages are complete
    pub fn new_immediate(
        stages_json: &str,
        final_lr: f64,
    ) -> Result<WasmCurriculumScheduler, String> {
        let stages = parse_stages(stages_json)?;
        Ok(Self {
            inner: CurriculumScheduler::new(stages, TransitionStrategy::Immediate, final_lr),
        })
    }

    /// Advance the scheduler by one step and return the new learning rate.
    pub fn step(&mut self) -> f64 {
        use optirs_core::schedulers::LearningRateScheduler;
        self.inner.step()
    }

    /// Get the current learning rate.
    #[cfg_attr(feature = "wasm", wasm_bindgen(getter))]
    pub fn learning_rate(&self) -> f64 {
        use optirs_core::schedulers::LearningRateScheduler;
        self.inner.get_learning_rate()
    }

    /// Reset the scheduler to its initial state.
    pub fn reset(&mut self) {
        use optirs_core::schedulers::LearningRateScheduler;
        self.inner.reset();
    }

    /// Get information about the current stage as a JSON string.
    ///
    /// Returns a JSON object with `learning_rate`, `duration`, and optionally `description`.
    pub fn current_stage_info(&self) -> String {
        let stage = self.inner.current_stage();
        let desc = stage
            .description
            .as_ref()
            .map(|d| format!(", \"description\": \"{}\"", d))
            .unwrap_or_default();
        format!(
            "{{\"learning_rate\": {}, \"duration\": {}{}}}",
            stage.learning_rate, stage.duration, desc
        )
    }

    /// Check if the curriculum has been completed (all stages finished).
    pub fn completed(&self) -> bool {
        self.inner.completed()
    }

    /// Manually advance to the next stage.
    ///
    /// Returns true if successfully advanced, false if there are no more stages.
    pub fn advance_stage(&mut self) -> bool {
        self.inner.advance_stage()
    }

    /// Get the overall progress of the curriculum (0.0 to 1.0).
    pub fn progress(&self) -> f64 {
        self.inner.overall_progress()
    }

    /// Get the scheduler name.
    pub fn name(&self) -> String {
        "CurriculumScheduler".to_string()
    }
}
