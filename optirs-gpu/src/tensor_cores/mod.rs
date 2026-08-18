//! Auto-generated module structure

pub mod functions;
pub mod pipelineoptimizationconfig_traits;
pub mod tensorcoreconfig_traits;
pub mod tensorcoreperformancebenchmark_traits;
pub mod types;

// Re-export all types.
//
// `functions`, `pipelineoptimizationconfig_traits`, `tensorcoreconfig_traits`
// and `tensorcoreperformancebenchmark_traits` are intentionally not
// glob-reexported here: each contains only a trait `impl` block (plus, for
// `functions`, its own test module) and no free-standing nameable items, so
// `pub use <module>::*;` re-exports nothing and rustc correctly reports it as
// unused. The `impl`s themselves are globally visible wherever the
// implementing type and `Default` are in scope, without needing a re-export,
// and the modules stay `pub mod` (compiled, tests run) below.
pub use types::*;
