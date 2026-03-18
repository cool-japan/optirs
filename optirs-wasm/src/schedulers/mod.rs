//! WASM wrappers for OptiRS learning rate schedulers.

mod attention_aware;
mod constant;
mod cosine_annealing;
mod cosine_annealing_warm_restarts;
mod curriculum;
mod cyclic_lr;
mod exponential_decay;
mod linear_decay;
mod linear_warmup_decay;
mod noise_injection;
mod one_cycle;
mod reduce_on_plateau;
mod step_decay;
mod vit_layer_decay;

pub use attention_aware::WasmAttentionAwareScheduler;
pub use constant::WasmConstantScheduler;
pub use cosine_annealing::WasmCosineAnnealing;
pub use cosine_annealing_warm_restarts::WasmCosineAnnealingWarmRestarts;
pub use curriculum::WasmCurriculumScheduler;
pub use cyclic_lr::WasmCyclicLR;
pub use exponential_decay::WasmExponentialDecay;
pub use linear_decay::WasmLinearDecay;
pub use linear_warmup_decay::WasmLinearWarmupDecay;
pub use noise_injection::WasmNoiseInjectionScheduler;
pub use one_cycle::WasmOneCycle;
pub use reduce_on_plateau::WasmReduceOnPlateau;
pub use step_decay::WasmStepDecay;
pub use vit_layer_decay::WasmViTLayerDecay;
