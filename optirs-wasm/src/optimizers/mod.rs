//! WASM wrappers for OptiRS optimizers.

mod adabound;
mod adadelta;
mod adagrad;
mod adam;
mod adamw;
mod lamb;
mod lars;
mod lion;
mod radam;
mod ranger;
mod rmsprop;
mod sgd;
mod sparse_adam;

pub use adabound::WasmAdaBound;
pub use adadelta::WasmAdaDelta;
pub use adagrad::WasmAdagrad;
pub use adam::WasmAdam;
pub use adamw::WasmAdamW;
pub use lamb::WasmLAMB;
pub use lars::WasmLARS;
pub use lion::WasmLion;
pub use radam::WasmRAdam;
pub use ranger::WasmRanger;
pub use rmsprop::WasmRMSprop;
pub use sgd::WasmSGD;
pub use sparse_adam::WasmSparseAdam;
