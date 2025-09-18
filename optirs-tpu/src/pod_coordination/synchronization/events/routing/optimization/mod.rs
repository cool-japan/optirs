// Routing optimization modules

pub mod path_optimization;
pub mod cache;
pub mod adaptive;

// Re-export optimization types
pub use path_optimization::*;
pub use cache::*;
pub use adaptive::*;