use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Represents a research dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    pub id: String,
    pub name: String,
    pub description: String,
    pub source: DatasetSource,
    pub metadata: DatasetMetadata,
}

/// Source information for a dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetSource {
    pub url: Option<String>,
    pub doi: Option<String>,
    pub repository: Option<String>,
}

/// Metadata for a dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    pub size: usize,
    pub format: String,
    pub features: Vec<String>,
    pub tags: Vec<String>,
    pub license: Option<String>,
}

/// Manager for research datasets
#[derive(Debug, Default)]
pub struct DatasetManager {
    datasets: HashMap<String, Dataset>,
}

impl DatasetManager {
    pub fn new() -> Self {
        Self {
            datasets: HashMap::new(),
        }
    }

    pub fn add_dataset(&mut self, dataset: Dataset) {
        self.datasets.insert(dataset.id.clone(), dataset);
    }

    pub fn get_dataset(&self, id: &str) -> Option<&Dataset> {
        self.datasets.get(id)
    }

    pub fn list_datasets(&self) -> Vec<&Dataset> {
        self.datasets.values().collect()
    }
}
