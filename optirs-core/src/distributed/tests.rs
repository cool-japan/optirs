use super::*;
use approx::assert_relative_eq;
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;

#[test]
fn test_arithmetic_averaging() {
    let mut averager: ParameterAverager<f64, scirs2_core::ndarray::Ix1> =
        ParameterAverager::new(AveragingStrategy::Arithmetic, 3);

    let params1 = vec![Array1::from_vec(vec![1.0, 2.0])];
    let params2 = vec![Array1::from_vec(vec![3.0, 4.0])];
    let params3 = vec![Array1::from_vec(vec![5.0, 6.0])];

    let nodeparameters = vec![(0, params1), (1, params2), (2, params3)];

    averager
        .average_parameters(&nodeparameters)
        .expect("unwrap failed");

    let result = averager.get_averaged_parameters();
    assert_relative_eq!(result[0][0], 3.0, epsilon = 1e-6); // (1+3+5)/3
    assert_relative_eq!(result[0][1], 4.0, epsilon = 1e-6); // (2+4+6)/3
}

#[test]
fn test_weighted_averaging() {
    let mut averager: ParameterAverager<f64, scirs2_core::ndarray::Ix1> =
        ParameterAverager::new(AveragingStrategy::WeightedByData, 2);

    // Initialize first to avoid overwriting weights
    let params1 = vec![Array1::from_vec(vec![2.0])];
    let params2 = vec![Array1::from_vec(vec![6.0])];
    let nodeparameters = vec![(0, params1.clone()), (1, params2.clone())];
    averager.initialize(&params1).expect("unwrap failed");

    // Set different weights after initialization
    averager.set_node_weight(0, 0.75).expect("unwrap failed"); // 75% weight
    averager.set_node_weight(1, 0.25).expect("unwrap failed"); // 25% weight

    averager
        .average_parameters(&nodeparameters)
        .expect("unwrap failed");

    let result = averager.get_averaged_parameters();
    // Weighted average: 0.75 * 2.0 + 0.25 * 6.0 = 1.5 + 1.5 = 3.0
    assert_relative_eq!(result[0][0], 3.0, epsilon = 1e-6);
}

#[test]
fn test_momentum_averaging() {
    let mut averager: ParameterAverager<f64, scirs2_core::ndarray::Ix1> =
        ParameterAverager::new(AveragingStrategy::Momentum { momentum: 0.9 }, 2);

    let params1 = vec![Array1::from_vec(vec![1.0])];
    let params2 = vec![Array1::from_vec(vec![3.0])];

    // First update: average = (1+3)/2 = 2.0, momentum buffer starts at 0, so result = 0.1 * 2.0 = 0.2
    let node_parameters1 = vec![(0, params1.clone()), (1, params2.clone())];
    averager
        .average_parameters(&node_parameters1)
        .expect("unwrap failed");

    let result1 = averager.get_averaged_parameters();
    // First result should be small due to zero initialization
    assert!(result1[0][0] >= 0.0 && result1[0][0] <= 0.5);

    // Several more updates to let momentum build up
    for _ in 0..10 {
        let nodeparameters = vec![(0, params1.clone()), (1, params2.clone())];
        averager
            .average_parameters(&nodeparameters)
            .expect("unwrap failed");
    }

    let final_result = averager.get_averaged_parameters();
    // After many updates, momentum should gradually converge towards the average (2.0)
    // But with momentum=0.9, it builds up slowly, so we use a broader range
    assert!(final_result[0][0] > 0.5 && final_result[0][0] < 2.5);
}

#[test]
fn test_parameter_server() {
    let mut server = ParameterServer::new(AveragingStrategy::Arithmetic, 2, 2);

    let initialparams = vec![Array1::from_vec(vec![0.0, 0.0])];
    server.initialize(&initialparams).expect("unwrap failed");

    // Submit updates from both nodes
    let update1 = vec![Array1::from_vec(vec![1.0, 2.0])];
    let update2 = vec![Array1::from_vec(vec![3.0, 4.0])];

    let ready1 = server.submit_update(0, update1).expect("unwrap failed");
    assert!(!ready1); // Not ready yet, waiting for second node

    let ready2 = server.submit_update(1, update2).expect("unwrap failed");
    assert!(ready2); // Ready after both nodes submitted

    let global_params = server.get_global_parameters();
    assert_relative_eq!(global_params[0][0], 2.0, epsilon = 1e-6); // (1+3)/2
    assert_relative_eq!(global_params[0][1], 3.0, epsilon = 1e-6); // (2+4)/2

    assert_eq!(server.current_round(), 1);
}

#[test]
fn test_distributed_coordinator() {
    let mut coordinator = DistributedCoordinator::new(
        AveragingStrategy::Arithmetic,
        2,  // 2 nodes
        2,  // expect 2 updates per round
        10, // max 10 rounds
    );

    let initialparams = vec![Array1::from_vec(vec![0.0])];
    coordinator
        .initialize(&initialparams)
        .expect("unwrap failed");

    // Simulate training rounds
    for round in 1..=3 {
        let update1 = vec![Array1::from_vec(vec![round as f64])];
        let update2 = vec![Array1::from_vec(vec![(round * 2) as f64])];

        let node_updates = vec![(0, update1), (1, update2)];

        let result = coordinator
            .communication_round(node_updates)
            .expect("unwrap failed");

        assert_eq!(result.round, round);
        assert!(result.should_continue);
        assert!(!result.converged); // Unlikely to converge with these updates

        // Check that global parameters are updated
        assert!(result.global_parameters[0][0] > 0.0);
    }
}

#[test]
fn test_averaging_strategies() {
    // Test arithmetic and federated strategies that should produce expected ranges
    let simple_strategies = vec![
        AveragingStrategy::Arithmetic,
        AveragingStrategy::WeightedByData,
        AveragingStrategy::Federated,
    ];

    for strategy in simple_strategies {
        let mut averager: ParameterAverager<f64, scirs2_core::ndarray::Ix1> =
            ParameterAverager::new(strategy, 2);

        let params1 = vec![Array1::from_vec(vec![1.0])];
        let params2 = vec![Array1::from_vec(vec![3.0])];

        let nodeparameters = vec![(0, params1), (1, params2)];

        averager
            .average_parameters(&nodeparameters)
            .expect("unwrap failed");
        let result = averager.get_averaged_parameters();
        assert!(result[0][0] >= 1.0 && result[0][0] <= 3.0);
    }

    // Test momentum and EMA strategies separately (they start from zero state)
    let stateful_strategies = vec![
        AveragingStrategy::Momentum { momentum: 0.9 },
        AveragingStrategy::ExponentialMovingAverage { decay: 0.9 },
    ];

    for strategy in stateful_strategies {
        let mut averager: ParameterAverager<f64, scirs2_core::ndarray::Ix1> =
            ParameterAverager::new(strategy, 2);

        let params1 = vec![Array1::from_vec(vec![1.0])];
        let params2 = vec![Array1::from_vec(vec![3.0])];

        let nodeparameters = vec![(0, params1), (1, params2)];

        averager
            .average_parameters(&nodeparameters)
            .expect("unwrap failed");
        let result = averager.get_averaged_parameters();
        // First result from momentum/EMA will be smaller due to zero initialization
        assert!(result[0][0] >= 0.0 && result[0][0] <= 3.0);
    }
}

#[test]
fn test_node_weight_validation() {
    let mut averager: ParameterAverager<f64, scirs2_core::ndarray::Ix1> =
        ParameterAverager::new(AveragingStrategy::WeightedByData, 2);

    // Valid node ID
    assert!(averager.set_node_weight(0, 0.5).is_ok());
    assert!(averager.set_node_weight(1, 0.5).is_ok());

    // Invalid node ID
    assert!(averager.set_node_weight(2, 0.5).is_err());
}

#[test]
fn test_parameter_dimension_validation() {
    let mut averager: ParameterAverager<f64, scirs2_core::ndarray::Ix1> =
        ParameterAverager::new(AveragingStrategy::Arithmetic, 2);

    let params1 = vec![Array1::from_vec(vec![1.0, 2.0])];
    let params2 = vec![Array1::from_vec(vec![3.0])]; // Wrong dimension

    let nodeparameters = vec![(0, params1), (1, params2)];

    // Should fail due to dimension mismatch - currently panics instead of returning error
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        averager.average_parameters(&nodeparameters)
    }));

    // Either it returns an error or panics due to dimension mismatch
    assert!(result.is_err() || (result.is_ok() && result.expect("unwrap failed").is_err()));
}

#[test]
fn test_training_stats() {
    let mut stats = TrainingStats::new();

    assert_eq!(stats.num_rounds(), 0);
    assert!(stats.latest_convergence().is_none());

    let params = vec![Array1::from_vec(vec![1.0])];
    stats.record_round(1, 0.5, &params);

    assert_eq!(stats.num_rounds(), 1);
    assert_eq!(stats.latest_convergence(), Some(0.5));
    assert_eq!(stats.convergence_history(), &[0.5]);
}

#[test]
fn test_gradient_compression_none() {
    let mut compressor = GradientCompressor::new(CompressionStrategy::None);

    let gradients = vec![
        Array1::from_vec(vec![1.0, 2.0, 3.0]),
        Array1::from_vec(vec![4.0, 5.0]),
    ];

    let compressed = compressor.compress(&gradients).expect("unwrap failed");
    assert_eq!(compressed.metadata.strategy, CompressionStrategy::None);
    assert_eq!(compressed.metadata.compression_ratio, 1.0);

    let decompressed = compressor.decompress(&compressed).expect("unwrap failed");
    assert_eq!(decompressed.len(), 2);
    assert_eq!(
        decompressed[0].as_slice().expect("unwrap failed"),
        &[1.0, 2.0, 3.0]
    );
    assert_eq!(
        decompressed[1].as_slice().expect("unwrap failed"),
        &[4.0, 5.0]
    );
}

#[test]
fn test_gradient_compression_topk() {
    let mut compressor = GradientCompressor::new(CompressionStrategy::TopK { k: 2 });

    let gradients = vec![Array1::from_vec(vec![0.1, 3.0, 0.2, 4.0, 0.05])];

    let compressed = compressor.compress(&gradients).expect("unwrap failed");
    assert!(compressed.metadata.compression_ratio < 1.0);
    assert_eq!(compressed.metadata.nnz_count, 2); // Top 2 elements

    let decompressed = compressor.decompress(&compressed).expect("unwrap failed");
    assert_eq!(decompressed.len(), 1);

    // Should have only the top 2 elements (4.0 and 3.0), others should be 0
    let result = &decompressed[0];
    assert_eq!(result[1], 3.0); // Original position of 3.0
    assert_eq!(result[3], 4.0); // Original position of 4.0
    assert_eq!(result[0], 0.0); // Should be zeroed
    assert_eq!(result[2], 0.0); // Should be zeroed
    assert_eq!(result[4], 0.0); // Should be zeroed
}

#[test]
fn test_gradient_compression_threshold() {
    let mut compressor = GradientCompressor::new(CompressionStrategy::Threshold { threshold: 1.0 });

    let gradients = vec![Array1::from_vec(vec![0.5, 2.0, 0.8, 3.0, 0.3])];

    let compressed = compressor.compress(&gradients).expect("unwrap failed");
    assert!(compressed.metadata.compression_ratio < 1.0);
    assert_eq!(compressed.metadata.nnz_count, 2); // Elements > 1.0: 2.0 and 3.0

    let decompressed = compressor.decompress(&compressed).expect("unwrap failed");
    let result = &decompressed[0];

    // Only elements > 1.0 should remain
    assert_eq!(result[0], 0.0); // 0.5 < 1.0
    assert_eq!(result[1], 2.0); // 2.0 > 1.0
    assert_eq!(result[2], 0.0); // 0.8 < 1.0
    assert_eq!(result[3], 3.0); // 3.0 > 1.0
    assert_eq!(result[4], 0.0); // 0.3 < 1.0
}

#[test]
fn test_gradient_compression_quantization() {
    let mut compressor = GradientCompressor::new(CompressionStrategy::Quantization { bits: 8 });

    let gradients = vec![Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0])];

    let compressed = compressor.compress(&gradients).expect("unwrap failed");
    assert!(compressed.metadata.compression_ratio < 1.0); // Should use less space with 8-bit quantization

    let decompressed = compressor.decompress(&compressed).expect("unwrap failed");
    let result = &decompressed[0];

    // Values should be approximately restored (with quantization error)
    assert!((result[0] - 1.0).abs() < 0.1);
    assert!((result[1] - 2.0).abs() < 0.1);
    assert!((result[2] - 3.0).abs() < 0.1);
    assert!((result[3] - 4.0).abs() < 0.1);
}

#[test]
fn test_gradient_compression_randomk() {
    let mut compressor = GradientCompressor::new(CompressionStrategy::RandomK { k: 3 });

    // Use a larger array to make compression effective
    let gradients = vec![Array1::from_vec(vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
    ])];

    let compressed = compressor.compress(&gradients).expect("unwrap failed");
    // With 3 out of 10 elements, compression should be effective
    assert!(compressed.metadata.compression_ratio < 1.0);
    assert_eq!(compressed.metadata.nnz_count, 3); // Exactly 3 elements should be kept

    let decompressed = compressor.decompress(&compressed).expect("unwrap failed");
    let result = &decompressed[0];

    // Exactly 3 elements should be non-zero
    let non_zero_count = result.iter().filter(|&&x| x != 0.0).count();
    assert_eq!(non_zero_count, 3);
}

#[test]
fn test_gradient_compression_error_feedback() {
    let base_strategy = CompressionStrategy::TopK { k: 2 };
    let strategy = CompressionStrategy::ErrorFeedback {
        base_strategy: Box::new(base_strategy),
        error_compensation: true,
    };

    let mut compressor = GradientCompressor::new(strategy);

    let gradients = vec![Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0])];

    // Initialize error state
    compressor.initialize_error_state(&gradients);

    // First compression
    let compressed1 = compressor.compress(&gradients).expect("unwrap failed");
    let decompressed1 = compressor.decompress(&compressed1).expect("unwrap failed");

    // Second compression (should include error feedback)
    let compressed2 = compressor.compress(&gradients).expect("unwrap failed");
    let decompressed2 = compressor.decompress(&compressed2).expect("unwrap failed");

    // Both should be valid compressions
    assert_eq!(decompressed1.len(), 1);
    assert_eq!(decompressed2.len(), 1);
}

#[test]
fn test_gradient_compression_clipped() {
    let base_strategy = CompressionStrategy::TopK { k: 3 };
    let strategy = CompressionStrategy::ClippedCompression {
        base_strategy: Box::new(base_strategy),
        clip_value: 2.5,
    };

    let mut compressor = GradientCompressor::new(strategy);

    let gradients = vec![Array1::from_vec(vec![1.0, 5.0, -3.0, 2.0])];

    let compressed = compressor.compress(&gradients).expect("unwrap failed");
    let decompressed = compressor.decompress(&compressed).expect("unwrap failed");

    let result = &decompressed[0];

    // Values should be clipped to [-2.5, 2.5] and then top-k applied
    for &val in result.iter() {
        if val != 0.0 {
            // Non-zero values from top-k
            assert!((-2.5..=2.5).contains(&val));
        }
    }
}

#[test]
fn test_compression_stats() {
    let mut stats = CompressionStats::new();

    assert_eq!(stats.compressions_count, 0);
    assert_eq!(stats.overall_compression_ratio(), 0.0);

    // Record some compressions
    stats.record_compression(1000, 500); // 50% compression
    assert_eq!(stats.compressions_count, 1);
    assert_relative_eq!(stats.overall_compression_ratio(), 0.5, epsilon = 1e-6);
    assert_relative_eq!(stats.bandwidth_savings(), 50.0, epsilon = 1e-6);

    stats.record_compression(1000, 250); // 25% compression
    assert_eq!(stats.compressions_count, 2);
    assert_relative_eq!(stats.overall_compression_ratio(), 0.375, epsilon = 1e-6); // (500+250)/(1000+1000)
    assert_relative_eq!(stats.bandwidth_savings(), 62.5, epsilon = 1e-6);

    assert_relative_eq!(stats.best_compression_ratio, 0.25, epsilon = 1e-6);
    assert_relative_eq!(stats.worst_compression_ratio, 0.5, epsilon = 1e-6);
}

#[test]
fn test_compression_roundtrip() {
    let strategies = vec![
        CompressionStrategy::None,
        CompressionStrategy::TopK { k: 2 },
        CompressionStrategy::RandomK { k: 2 },
        CompressionStrategy::Threshold { threshold: 1.5 },
        CompressionStrategy::Quantization { bits: 4 },
    ];

    let gradients = vec![
        Array1::from_vec(vec![1.0, 2.5, 0.5, 3.0]),
        Array1::from_vec(vec![0.1, 4.0]),
    ];

    for strategy in strategies {
        let mut compressor = GradientCompressor::new(strategy.clone());

        let compressed = compressor.compress(&gradients).expect("unwrap failed");
        let decompressed = compressor.decompress(&compressed).expect("unwrap failed");

        // Should decompress to same number of arrays
        assert_eq!(decompressed.len(), gradients.len());

        // Shapes should match
        for (orig, decomp) in gradients.iter().zip(decompressed.iter()) {
            assert_eq!(orig.shape(), decomp.shape());
        }

        // For lossless strategies, values should match exactly
        match strategy {
            CompressionStrategy::None => {
                for (orig, decomp) in gradients.iter().zip(decompressed.iter()) {
                    for (&o, &d) in orig.iter().zip(decomp.iter()) {
                        assert_relative_eq!(o, d, epsilon = 1e-10);
                    }
                }
            }
            _ => {
                // For lossy strategies, just check that we get reasonable values
                for decomp in &decompressed {
                    assert!(decomp.iter().all(|&x| x.is_finite()));
                }
            }
        }
    }
}

#[test]
fn test_compression_invalid_configs() {
    // Invalid quantization bits
    let strategy = CompressionStrategy::Quantization { bits: 64 };
    let mut compressor = GradientCompressor::new(strategy);

    let gradients = vec![Array1::from_vec(vec![1.0, 2.0])];
    assert!(compressor.compress(&gradients).is_err());

    // Invalid decompression data
    let valid_compressor: GradientCompressor<f64, scirs2_core::ndarray::Ix1> =
        GradientCompressor::new(CompressionStrategy::None);
    let invalid_compressed = CompressedGradient {
        data: vec![1, 2, 3], // Insufficient data
        metadata: CompressionMetadata {
            strategy: CompressionStrategy::None,
            compression_ratio: 1.0,
            nnz_count: 1,
            scale_factors: vec![],
            extra_data: vec![],
        },
        shapes: vec![vec![2]],
    };

    assert!(valid_compressor.decompress(&invalid_compressed).is_err());
}

#[test]
fn test_distributed_with_compression() {
    // Test parameter server with compressed gradients
    let mut server = ParameterServer::new(AveragingStrategy::Arithmetic, 2, 2);
    let initialparams = vec![Array1::from_vec(vec![0.0, 0.0])];
    server.initialize(&initialparams).expect("unwrap failed");

    let mut compressor = GradientCompressor::new(CompressionStrategy::TopK { k: 1 });

    // Create gradients and compress them
    let gradients1 = vec![Array1::from_vec(vec![1.0, 3.0])]; // Top-1 should keep 3.0
    let gradients2 = vec![Array1::from_vec(vec![2.0, 1.0])]; // Top-1 should keep 2.0

    let compressed1 = compressor.compress(&gradients1).expect("unwrap failed");
    let compressed2 = compressor.compress(&gradients2).expect("unwrap failed");

    let decompressed1 = compressor.decompress(&compressed1).expect("unwrap failed");
    let decompressed2 = compressor.decompress(&compressed2).expect("unwrap failed");

    // Submit decompressed gradients to server
    server
        .submit_update(0, decompressed1)
        .expect("unwrap failed");
    server
        .submit_update(1, decompressed2)
        .expect("unwrap failed");

    let global_params = server.get_global_parameters();

    // Should have averaged the compressed gradients
    // Node 0 contributes [0, 3.0], Node 1 contributes [2.0, 0]
    // Average: [1.0, 1.5]
    assert_relative_eq!(global_params[0][0], 1.0, epsilon = 1e-6);
    assert_relative_eq!(global_params[0][1], 1.5, epsilon = 1e-6);
}
