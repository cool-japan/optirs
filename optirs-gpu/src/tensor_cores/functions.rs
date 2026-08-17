//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{
    SparseTensorCoreMatrix, TensorCoreBatch, TensorCoreConfig, TensorCoreOperationType,
    TensorCoreOptimizer, TensorCorePrecision,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_cores::types::{
        HardwareUtilizationState, MixedPrecisionTrainer, PerformanceTargets, ResourceRequirements,
        TensorCoreOpType, TensorCoreOperation, TensorCoreWorkload, WorkloadConstraints,
    };
    use scirs2_core::ndarray::Array2;

    fn fused_adam_op(priority: i32) -> TensorCoreOperation<f32> {
        let one = Array2::<f32>::ones((1, 1));
        TensorCoreOperation {
            op_type: TensorCoreOpType::FusedAdam {
                params: one.clone(),
                grads: one.clone(),
                exp_avg: one.clone(),
                exp_avg_sq: one.clone(),
                lr: 1e-3,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.0,
                step: 1,
            },
            output_dims: (1, 1),
            precision: TensorCorePrecision::FP16,
            priority,
            dependencies: Vec::new(),
            compute_cost: 1.0,
            memory_bandwidth: 1.0,
        }
    }

    fn tiny_workload(num_ops: usize) -> TensorCoreWorkload<f32> {
        TensorCoreWorkload {
            operations: (0..num_ops).map(|i| fused_adam_op(i as i32)).collect(),
            resource_requirements: ResourceRequirements {
                memory_bytes: 4,
                compute_flops: 1.0,
                bandwidth_gbps: 1.0,
                tensor_cores: 0,
            },
            performance_targets: PerformanceTargets {
                target_throughput: 1.0,
                max_latency_ms: 1.0,
                target_efficiency: 1.0,
                energy_budget: 1.0,
            },
            constraints: WorkloadConstraints {
                memory_limit: 4096,
                time_limit_ms: 1000,
                power_limit: 100.0,
                precision_requirements: vec![TensorCorePrecision::FP16],
            },
        }
    }

    /// `adaptive_tensor_core_scheduling` must not fabricate hardware
    /// telemetry: the default baseline is honestly all-idle (regression test
    /// for F7 — this used to hardcode 75%/65C/200W under GPU features).
    #[test]
    fn hardware_baseline_is_honestly_idle_not_fabricated() {
        let baseline = HardwareUtilizationState::unknown_baseline();
        assert_eq!(baseline.gpu_utilization, 0.0);
        assert_eq!(baseline.memory_utilization, 0.0);
        assert_eq!(baseline.tensor_core_utilization, 0.0);
        assert_eq!(baseline.bandwidth_utilization, 0.0);
        assert_eq!(baseline.power_consumption, 0.0);

        // The scheduler must still function end-to-end against that baseline.
        let plan = optimizer()
            .adaptive_tensor_core_scheduling(&tiny_workload(5))
            .expect("scheduling against the honest baseline must succeed");
        assert_eq!(plan.operation_order.len(), 5);
        assert_eq!(plan.stream_assignments.len(), 5);
    }

    /// A caller-supplied hardware state must genuinely drive the scheduling
    /// decision (not be accepted and ignored): low GPU utilization uses more
    /// streams than high utilization.
    #[test]
    fn injected_hardware_state_changes_the_schedule() {
        let mut opt = optimizer();
        let idle = HardwareUtilizationState {
            gpu_utilization: 10.0,
            ..HardwareUtilizationState::unknown_baseline()
        };
        let busy = HardwareUtilizationState {
            gpu_utilization: 90.0,
            ..HardwareUtilizationState::unknown_baseline()
        };

        let idle_plan = opt
            .adaptive_tensor_core_scheduling_with_state(&tiny_workload(5), idle)
            .expect("idle scheduling");
        let busy_plan = opt
            .adaptive_tensor_core_scheduling_with_state(&tiny_workload(5), busy)
            .expect("busy scheduling");

        let idle_streams: std::collections::HashSet<usize> =
            idle_plan.stream_assignments.iter().copied().collect();
        let busy_streams: std::collections::HashSet<usize> =
            busy_plan.stream_assignments.iter().copied().collect();
        assert!(
            idle_streams.len() > busy_streams.len(),
            "low utilization ({idle_streams:?}) should spread work over more streams than \
             high utilization ({busy_streams:?}) — the injected state was not used"
        );
    }

    // `TensorCoreOptimizer::new` is now infallible in practice; these helpers
    // fail loudly if that ever regresses, so the tests below always exercise
    // real behaviour instead of silently returning on an `Err`.
    fn optimizer() -> TensorCoreOptimizer {
        match TensorCoreOptimizer::new(TensorCoreConfig::default()) {
            Ok(opt) => opt,
            Err(e) => panic!("TensorCoreOptimizer::new must succeed: {e}"),
        }
    }

    fn trainer() -> MixedPrecisionTrainer {
        match optimizer().create_mixed_precision_trainer() {
            Ok(t) => t,
            Err(e) => panic!("create_mixed_precision_trainer must succeed: {e}"),
        }
    }

    #[test]
    fn test_tensor_core_config_default() {
        let config = TensorCoreConfig::default();
        assert!(config.use_volta_cores);
        assert!(config.use_ampere_cores);
        assert_eq!(config.wmma_tile_m, 16);
        assert!(config.use_tf32);
    }

    #[test]
    fn test_layout_optimization() {
        let mut optimizer = optimizer();
        let layout = optimizer.optimize_layout(100, 200, 64);
        assert!(layout.padding_m <= 16);
        assert!(layout.padding_n <= 16);
        assert!(layout.padding_k <= 16);
        assert!(layout.speedup_factor > 1.0);
    }

    #[test]
    fn test_tensor_core_info() {
        let info = optimizer().get_tensor_core_info();
        assert!(info.max_tensor_ops_per_second >= 0.0);
        // No reachable backend exposes NVIDIA tensor cores, so every capability
        // is honestly negative.
        assert_eq!(info.compute_capability, (0, 0));
        assert!(!info.supports_fp16);
        assert!(!info.supports_bf16);
        assert!(!info.supports_sparse);
    }

    #[test]
    fn test_mixed_precision_trainer() {
        let mut trainer = trainer();
        let initial_scale = trainer.get_loss_scale();
        assert!(initial_scale > 0.0);
        trainer.update_loss_scale(false);
        let stats = trainer.get_statistics();
        assert_eq!(stats.step_count, 1);
        assert_eq!(stats.successful_steps, 1);
        trainer.update_loss_scale(true);
        let new_scale = trainer.get_loss_scale();
        assert!(new_scale < initial_scale);
    }

    #[test]
    fn test_amp_scale_unscale_and_overflow_detection() {
        let mut trainer = trainer();
        let scale = trainer.get_loss_scale();
        // Forward scaling multiplies by the loss scale.
        let mut grads = [1.0f32, -2.0, 0.5];
        trainer.scale(&mut grads);
        assert_eq!(grads, [scale, -2.0 * scale, 0.5 * scale]);
        // Unscaling a finite batch restores the originals and reports no overflow.
        let overflow = trainer.unscale_and_check(&mut grads);
        assert!(!overflow);
        assert!((grads[0] - 1.0).abs() < 1e-6);
        assert!((grads[1] + 2.0).abs() < 1e-6);
        assert!((grads[2] - 0.5).abs() < 1e-6);
        // A non-finite scaled gradient is detected and cuts the loss scale.
        let mut bad = [f32::INFINITY, 1.0];
        assert!(trainer.unscale_and_check(&mut bad));
        assert!(trainer.get_loss_scale() < scale);
    }

    #[test]
    fn test_amp_f16_round_trip() {
        let trainer = trainer();
        let values = [0.0f32, 1.0, -1.0, 0.5, 1024.0];
        let bits = trainer.cast_to_f16(&values);
        let back = trainer.cast_from_f16(&bits);
        for (a, b) in values.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1e-6, "{a} did not round-trip (got {b})");
        }
        // Out-of-range values saturate rather than becoming infinite.
        let saturated = trainer.cast_from_f16(&trainer.cast_to_f16(&[1.0e30f32]));
        assert!(saturated[0].is_finite());
    }

    #[test]
    fn test_precision_selection() {
        let trainer = trainer();
        for op in [
            TensorCoreOperationType::GEMM,
            TensorCoreOperationType::Convolution,
            TensorCoreOperationType::Attention,
        ] {
            let precision = trainer.select_optimal_precision(op);
            assert!(matches!(
                precision,
                TensorCorePrecision::FP16
                    | TensorCorePrecision::BF16
                    | TensorCorePrecision::TF32
                    | TensorCorePrecision::FP8
            ));
        }
    }

    #[test]
    fn test_sparse_tensor_core_matrix() {
        let dense = Array2::from_shape_vec((4, 8), (0..32).map(|x| x as f32).collect())
            .expect("valid shape");
        let sparse = SparseTensorCoreMatrix::from_dense(&dense);
        assert_eq!(sparse.denseshape(), (4, 8));
        assert!(sparse.sparsity_ratio() > 0.0);
        assert!(sparse.sparsity_ratio() <= 1.0);
    }

    #[test]
    fn test_sparse_2_4_round_trips_including_partial_group() {
        // Six columns per row exercise one full 4-group and one partial 2-group,
        // which is the case the old code mishandled.
        let dense = Array2::from_shape_vec(
            (2, 6),
            vec![
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, //
                -9.0, 1.0, -2.0, 8.0, 7.0, -3.0,
            ],
        )
        .expect("valid shape");
        let sparse = SparseTensorCoreMatrix::from_dense(&dense);
        let reconstructed = sparse.to_dense();
        // Row 0 group [1,2,3,4] keeps 3,4 (the two largest); group [5,6] keeps both.
        // Row 1 group [-9,1,-2,8] keeps -9,8; group [7,-3] keeps both.
        let expected = [
            0.0f32, 0.0, 3.0, 4.0, 5.0, 6.0, //
            -9.0, 0.0, 0.0, 8.0, 7.0, -3.0,
        ];
        for (idx, exp) in expected.iter().enumerate() {
            let (r, c) = (idx / 6, idx % 6);
            assert_eq!(reconstructed[[r, c]], *exp, "mismatch at ({r}, {c})");
        }
    }

    #[test]
    fn test_sparse_2_4_survives_nan() {
        // A NaN in a group must not panic the magnitude sort.
        let dense =
            Array2::from_shape_vec((1, 4), vec![f32::NAN, 1.0, 2.0, 3.0]).expect("valid shape");
        let sparse = SparseTensorCoreMatrix::from_dense(&dense);
        assert_eq!(sparse.denseshape(), (1, 4));
    }

    #[test]
    fn test_tensor_core_batch_operations() {
        // Construction succeeds on every backend; the WMMA batch op is an honest
        // `Err` on every backend (no NVIDIA tensor cores are reachable).
        let optimizer = optimizer();
        let batch = TensorCoreBatch {
            a: Array2::ones((16, 16)),
            b: Array2::ones((16, 16)),
            alpha: 1.0f32,
            beta: 0.0f32,
            output_m: 16,
            output_n: 16,
        };
        let batches = vec![batch];
        let result = optimizer.multi_batch_tensor_core_ops(&batches, TensorCorePrecision::FP16);
        assert!(result.is_err());
    }
}
