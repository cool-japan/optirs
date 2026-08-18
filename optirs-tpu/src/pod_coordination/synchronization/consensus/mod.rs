// Mod Module

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Mod {
    pub id: String,
}

#[derive(Debug, Clone, Default)]
pub struct ModManager {
    pub items: Vec<Mod>,
}
