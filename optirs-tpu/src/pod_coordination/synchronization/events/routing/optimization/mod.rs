// Routing optimization modules

pub mod adaptive;
pub mod cache;
pub mod path_optimization;

// Re-export optimization types
pub use adaptive::*;
pub use cache::*;
pub use path_optimization::*;
