// Architecture search space definition and management

use scirs2_core::random::Rng;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use crate::error::Result;
use super::config::{SearchConstraints, LayerType, ArchitectureSpec, LayerSpec};

/// Architecture search space
#[derive(Debug, Clone)]
pub struct ArchitectureSearchSpace {
    /// Available layer types and their configurations
    layer_configurations: HashMap<LayerType, LayerConfiguration>,
    /// Search constraints
    constraints: SearchConstraints,
    /// Architecture templates
    templates: Vec<ArchitectureTemplate>,
    /// Encoding scheme for architectures
    encoding: ArchitectureEncoding,
}

impl ArchitectureSearchSpace {
    /// Create new search space
    pub fn new(constraints: &SearchConstraints) -> Result<Self> {
        let mut layer_configurations = HashMap::new();

        // Define configurations for each layer type
        for &layer_type in &constraints.allowed_layer_types {
            layer_configurations.insert(layer_type, LayerConfiguration::default_for_type(layer_type));
        }

        let templates = Self::create_default_templates(constraints)?;
        let encoding = ArchitectureEncoding::new();

        Ok(Self {
            layer_configurations,
            constraints: constraints.clone(),
            templates,
            encoding,
        })
    }

    /// Sample random architecture from search space
    pub fn sample_random_architecture(&self, rng: &mut impl Rng) -> Result<String> {
        let num_layers = rng.gen_range(self.constraints.min_layers..=self.constraints.max_layers);
        let mut layers = Vec::new();

        for _ in 0..num_layers {
            let layer_type = self.constraints.allowed_layer_types[
                rng.gen_range(0..self.constraints.allowed_layer_types.len())
            ];

            let layer_config = &self.layer_configurations[&layer_type];
            let layer = self.sample_layer(layer_type, layer_config, rng)?;
            layers.push(layer);
        }

        let architecture = ArchitectureSpec {
            layers,
            estimated_memory_mb: 0, // Will be calculated
            estimated_flops: 0,     // Will be calculated
            estimated_params: 0,    // Will be calculated
        };

        self.encoding.encode(&architecture)
    }

    /// Sample layer with given type and configuration
    fn sample_layer(
        &self,
        layer_type: LayerType,
        config: &LayerConfiguration,
        rng: &mut impl Rng,
    ) -> Result<LayerSpec> {
        let mut layer_config = HashMap::new();

        match layer_type {
            LayerType::Dense => {
                let units = rng.gen_range(config.min_units..=config.max_units);
                layer_config.insert("units".to_string(), serde_json::json!(units));

                if config.allow_activation {
                    let activations = &["relu", "tanh", "sigmoid", "gelu"];
                    let activation = activations[rng.gen_range(0..activations.len())];
                    layer_config.insert("activation".to_string(), serde_json::json!(activation));
                }
            }
            LayerType::Conv2D => {
                let filters = rng.gen_range(config.min_filters..=config.max_filters);
                let kernel_sizes = &[1, 3, 5, 7];
                let kernel_size = kernel_sizes[rng.gen_range(0..kernel_sizes.len())];

                layer_config.insert("filters".to_string(), serde_json::json!(filters));
                layer_config.insert("kernel_size".to_string(), serde_json::json!(kernel_size));
            }
            LayerType::LSTM => {
                let units = rng.gen_range(config.min_units..=config.max_units);
                let return_sequences = rng.gen_bool(0.5);

                layer_config.insert("units".to_string(), serde_json::json!(units));
                layer_config.insert("return_sequences".to_string(), serde_json::json!(return_sequences));
            }
            LayerType::Attention => {
                let embed_dim = rng.gen_range(config.min_embed_dim..=config.max_embed_dim);
                let num_heads = [1, 2, 4, 8][rng.gen_range(0..4)];

                layer_config.insert("embed_dim".to_string(), serde_json::json!(embed_dim));
                layer_config.insert("num_heads".to_string(), serde_json::json!(num_heads));
            }
            _ => {
                // Default configurations for other layer types
            }
        }

        let params = self.estimate_layer_params(layer_type, &layer_config)?;

        Ok(LayerSpec {
            layer_type,
            params,
            config: layer_config,
        })
    }

    /// Estimate number of parameters for a layer
    fn estimate_layer_params(
        &self,
        layer_type: LayerType,
        config: &HashMap<String, serde_json::Value>,
    ) -> Result<usize> {
        match layer_type {
            LayerType::Dense => {
                let units = config.get("units")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(128) as usize;
                // Rough estimate: input_dim * units + units (bias)
                // Assume input_dim = 128 for estimation
                Ok(128 * units + units)
            }
            LayerType::Conv2D => {
                let filters = config.get("filters")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(32) as usize;
                let kernel_size = config.get("kernel_size")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(3) as usize;
                // kernel_size^2 * input_channels * filters + filters (bias)
                // Assume input_channels = 32 for estimation
                Ok(kernel_size * kernel_size * 32 * filters + filters)
            }
            LayerType::LSTM => {
                let units = config.get("units")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(128) as usize;
                // LSTM has 4 gates, each with input_dim * units + units * units + units
                // Assume input_dim = 128 for estimation
                Ok(4 * (128 * units + units * units + units))
            }
            LayerType::Attention => {
                let embed_dim = config.get("embed_dim")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(256) as usize;
                // Query, Key, Value projections: 3 * embed_dim * embed_dim
                // Output projection: embed_dim * embed_dim
                Ok(4 * embed_dim * embed_dim)
            }
            _ => Ok(1000), // Default estimate
        }
    }

    /// Mutate architecture
    pub fn mutate_architecture(&self, architecture: &str) -> Result<String> {
        let mut spec = self.encoding.decode(architecture)?;
        let mut rng = scirs2_core::random::thread_rng();

        // Mutation strategies
        match rng.gen_range(0..4) {
            0 => self.mutate_add_layer(&mut spec, &mut rng)?,
            1 => self.mutate_remove_layer(&mut spec, &mut rng)?,
            2 => self.mutate_modify_layer(&mut spec, &mut rng)?,
            3 => self.mutate_swap_layers(&mut spec, &mut rng)?,
            _ => {}
        }

        self.encoding.encode(&spec)
    }

    /// Add layer mutation
    fn mutate_add_layer(&self, spec: &mut ArchitectureSpec, rng: &mut impl Rng) -> Result<()> {
        if spec.layers.len() >= self.constraints.max_layers {
            return Ok(());
        }

        let layer_type = self.constraints.allowed_layer_types[
            rng.gen_range(0..self.constraints.allowed_layer_types.len())
        ];

        let layer_config = &self.layer_configurations[&layer_type];
        let new_layer = self.sample_layer(layer_type, layer_config, rng)?;

        let insert_pos = rng.gen_range(0..=spec.layers.len());
        spec.layers.insert(insert_pos, new_layer);

        Ok(())
    }

    /// Remove layer mutation
    fn mutate_remove_layer(&self, spec: &mut ArchitectureSpec, rng: &mut impl Rng) -> Result<()> {
        if spec.layers.len() <= self.constraints.min_layers {
            return Ok(());
        }

        let remove_pos = rng.gen_range(0..spec.layers.len());
        spec.layers.remove(remove_pos);

        Ok(())
    }

    /// Modify layer mutation
    fn mutate_modify_layer(&self, spec: &mut ArchitectureSpec, rng: &mut impl Rng) -> Result<()> {
        if spec.layers.is_empty() {
            return Ok(());
        }

        let layer_idx = rng.gen_range(0..spec.layers.len());
        let layer_type = spec.layers[layer_idx].layer_type;
        let layer_config = &self.layer_configurations[&layer_type];

        let new_layer = self.sample_layer(layer_type, layer_config, rng)?;
        spec.layers[layer_idx] = new_layer;

        Ok(())
    }

    /// Swap layers mutation
    fn mutate_swap_layers(&self, spec: &mut ArchitectureSpec, rng: &mut impl Rng) -> Result<()> {
        if spec.layers.len() < 2 {
            return Ok(());
        }

        let idx1 = rng.gen_range(0..spec.layers.len());
        let idx2 = rng.gen_range(0..spec.layers.len());

        if idx1 != idx2 {
            spec.layers.swap(idx1, idx2);
        }

        Ok(())
    }

    /// Create default architecture templates
    fn create_default_templates(constraints: &SearchConstraints) -> Result<Vec<ArchitectureTemplate>> {
        let mut templates = Vec::new();

        // Simple feedforward template
        if constraints.allowed_layer_types.contains(&LayerType::Dense) {
            templates.push(ArchitectureTemplate {
                name: "Simple Feedforward".to_string(),
                description: "Basic feedforward network".to_string(),
                layer_pattern: vec![
                    LayerType::Dense,
                    LayerType::Dense,
                    LayerType::Dense,
                ],
            });
        }

        // CNN template
        if constraints.allowed_layer_types.contains(&LayerType::Conv2D) {
            templates.push(ArchitectureTemplate {
                name: "Simple CNN".to_string(),
                description: "Basic convolutional network".to_string(),
                layer_pattern: vec![
                    LayerType::Conv2D,
                    LayerType::Conv2D,
                    LayerType::Dense,
                ],
            });
        }

        // RNN template
        if constraints.allowed_layer_types.contains(&LayerType::LSTM) {
            templates.push(ArchitectureTemplate {
                name: "Simple RNN".to_string(),
                description: "Basic recurrent network".to_string(),
                layer_pattern: vec![
                    LayerType::LSTM,
                    LayerType::LSTM,
                    LayerType::Dense,
                ],
            });
        }

        Ok(templates)
    }

    /// Get available templates
    pub fn get_templates(&self) -> &[ArchitectureTemplate] {
        &self.templates
    }

    /// Validate architecture against constraints
    pub fn validate_architecture(&self, architecture: &ArchitectureSpec) -> bool {
        self.constraints.satisfies(architecture)
    }
}

/// Configuration for a specific layer type
#[derive(Debug, Clone)]
pub struct LayerConfiguration {
    pub min_units: usize,
    pub max_units: usize,
    pub min_filters: usize,
    pub max_filters: usize,
    pub min_embed_dim: usize,
    pub max_embed_dim: usize,
    pub allow_activation: bool,
    pub allow_dropout: bool,
    pub dropout_range: (f32, f32),
}

impl LayerConfiguration {
    pub fn default_for_type(layer_type: LayerType) -> Self {
        match layer_type {
            LayerType::Dense => Self {
                min_units: 16,
                max_units: 1024,
                min_filters: 0,
                max_filters: 0,
                min_embed_dim: 0,
                max_embed_dim: 0,
                allow_activation: true,
                allow_dropout: true,
                dropout_range: (0.0, 0.5),
            },
            LayerType::Conv2D => Self {
                min_units: 0,
                max_units: 0,
                min_filters: 8,
                max_filters: 512,
                min_embed_dim: 0,
                max_embed_dim: 0,
                allow_activation: true,
                allow_dropout: false,
                dropout_range: (0.0, 0.0),
            },
            LayerType::LSTM => Self {
                min_units: 32,
                max_units: 512,
                min_filters: 0,
                max_filters: 0,
                min_embed_dim: 0,
                max_embed_dim: 0,
                allow_activation: false,
                allow_dropout: true,
                dropout_range: (0.0, 0.3),
            },
            LayerType::Attention => Self {
                min_units: 0,
                max_units: 0,
                min_filters: 0,
                max_filters: 0,
                min_embed_dim: 64,
                max_embed_dim: 1024,
                allow_activation: false,
                allow_dropout: true,
                dropout_range: (0.0, 0.2),
            },
            _ => Self::default(),
        }
    }
}

impl Default for LayerConfiguration {
    fn default() -> Self {
        Self {
            min_units: 16,
            max_units: 256,
            min_filters: 8,
            max_filters: 128,
            min_embed_dim: 64,
            max_embed_dim: 512,
            allow_activation: true,
            allow_dropout: true,
            dropout_range: (0.0, 0.3),
        }
    }
}

/// Architecture template
#[derive(Debug, Clone)]
pub struct ArchitectureTemplate {
    pub name: String,
    pub description: String,
    pub layer_pattern: Vec<LayerType>,
}

/// Architecture encoding/decoding
#[derive(Debug, Clone)]
pub struct ArchitectureEncoding;

impl ArchitectureEncoding {
    pub fn new() -> Self {
        Self
    }

    /// Encode architecture to string representation
    pub fn encode(&self, spec: &ArchitectureSpec) -> Result<String> {
        // Simple JSON encoding for now
        serde_json::to_string(spec)
            .map_err(|e| crate::error::OptimError::Other(format!("Encoding error: {}", e)))
    }

    /// Decode string representation to architecture
    pub fn decode(&self, encoded: &str) -> Result<ArchitectureSpec> {
        serde_json::from_str(encoded)
            .map_err(|e| crate::error::OptimError::Other(format!("Decoding error: {}", e)))
    }
}

/// Architecture component abstraction
pub struct ArchitectureComponent {
    pub component_type: ComponentType,
    pub parameters: HashMap<String, f64>,
    pub connections: Vec<usize>,
}

/// Component types
#[derive(Debug, Clone, Copy)]
pub enum ComponentType {
    Input,
    Hidden,
    Output,
    Skip,
    Concat,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_space_creation() {
        let constraints = SearchConstraints::default();
        let space = ArchitectureSearchSpace::new(&constraints);
        assert!(space.is_ok());
    }

    #[test]
    fn test_random_architecture_sampling() {
        let constraints = SearchConstraints::default();
        let space = ArchitectureSearchSpace::new(&constraints).unwrap();
        let mut rng = scirs2_core::random::thread_rng();

        let architecture = space.sample_random_architecture(&mut rng);
        assert!(architecture.is_ok());
    }

    #[test]
    fn test_layer_configuration() {
        let config = LayerConfiguration::default_for_type(LayerType::Dense);
        assert!(config.min_units < config.max_units);
        assert!(config.allow_activation);
    }

    #[test]
    fn test_architecture_encoding() {
        let encoding = ArchitectureEncoding::new();
        let spec = ArchitectureSpec {
            layers: Vec::new(),
            estimated_memory_mb: 100,
            estimated_flops: 1000,
            estimated_params: 5000,
        };

        let encoded = encoding.encode(&spec);
        assert!(encoded.is_ok());

        if let Ok(encoded_str) = encoded {
            let decoded = encoding.decode(&encoded_str);
            assert!(decoded.is_ok());
        }
    }
}