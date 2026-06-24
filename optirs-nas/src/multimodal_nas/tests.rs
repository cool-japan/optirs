//! Unit tests for the multimodal NAS module.

use super::*;
use crate::error::OptimError;
use scirs2_core::random::Random;
use std::collections::HashMap;

fn balanced_arch(id: &str) -> MultimodalArchitecture {
    MultimodalArchitecture {
        modalities: vec![Modality::Image, Modality::Text],
        encoders: vec![
            ModalityEncoder {
                modality: Modality::Image,
                layers: vec![
                    MultimodalLayer::ImageConv {
                        channels: 64,
                        kernel: 3,
                    },
                    MultimodalLayer::Dense { out: 128 },
                ],
            },
            ModalityEncoder {
                modality: Modality::Text,
                layers: vec![
                    MultimodalLayer::TextEmbed { dim: 128 },
                    MultimodalLayer::Dense { out: 64 },
                ],
            },
        ],
        fusion: FusionOp::EarlyConcat,
        head: vec![MultimodalLayer::Dense { out: 64 }],
        parameters: HashMap::new(),
        model_id: id.to_string(),
    }
}

#[test]
fn test_default_search_space_has_layers() {
    let space = MultimodalSearchSpace::default();
    assert!(!space.available_modalities.is_empty());
    assert!(space
        .image_layers
        .iter()
        .any(|l| matches!(l, MultimodalLayer::ImageConv { .. })));
    assert!(space
        .text_layers
        .iter()
        .any(|l| matches!(l, MultimodalLayer::TextTransformer { .. })));
    assert!(space
        .audio_layers
        .iter()
        .any(|l| matches!(l, MultimodalLayer::AudioRecurrent { .. })));
    assert!(space.fusion_ops.contains(&FusionOp::EarlyConcat));
    assert!(space
        .fusion_ops
        .iter()
        .any(|f| matches!(f, FusionOp::Bilinear { .. })));
    assert!(!space.head_layers.is_empty());
    assert_eq!(space.min_encoder_len, 2);
    assert_eq!(space.max_encoder_len, 5);
}

#[test]
fn test_layer_modality_correct() {
    assert_eq!(
        layer_modality(&MultimodalLayer::ImageConv {
            channels: 32,
            kernel: 3
        }),
        Some(Modality::Image)
    );
    assert_eq!(
        layer_modality(&MultimodalLayer::TextTransformer { heads: 4, dim: 64 }),
        Some(Modality::Text)
    );
    assert_eq!(
        layer_modality(&MultimodalLayer::AudioRecurrent { hidden: 128 }),
        Some(Modality::Audio)
    );
    assert_eq!(layer_modality(&MultimodalLayer::Dense { out: 64 }), None);
    assert_eq!(layer_modality(&MultimodalLayer::Norm), None);
}

#[test]
fn test_fusion_supports_helper() {
    assert!(fusion_supports(&FusionOp::EarlyConcat, 1));
    assert!(fusion_supports(&FusionOp::EarlyConcat, 3));
    assert!(!fusion_supports(
        &FusionOp::CrossAttention { heads: 4, dim: 64 },
        1
    ));
    assert!(fusion_supports(
        &FusionOp::CrossAttention { heads: 4, dim: 64 },
        2
    ));
    assert!(fusion_supports(&FusionOp::Bilinear { dim: 128 }, 2));
    assert!(!fusion_supports(&FusionOp::Bilinear { dim: 128 }, 3));
    assert!(!fusion_supports(&FusionOp::Gated { dim: 64 }, 1));
}

#[test]
fn test_propose_returns_valid_architecture() {
    let mut engine = MultimodalNasEngine::with_default_search_space();
    // Loop many seeds: every proposal must validate.
    for seed in 0..200u64 {
        let mut rng = Random::seed(seed);
        let arch = engine.propose(&mut rng).expect("propose");
        assert!(
            arch.validate().is_ok(),
            "seed {} produced invalid arch: {:?}",
            seed,
            arch.validate()
        );
        assert!(!arch.modalities.is_empty());
        assert!(!arch.head.is_empty());
        assert_eq!(arch.encoders.len(), arch.modalities.len());
        for enc in &arch.encoders {
            assert!(enc.layers.len() >= engine.search_space().min_encoder_len);
            assert!(enc.layers.len() <= engine.search_space().max_encoder_len);
        }
    }
}

#[test]
fn test_propose_deterministic_seed() {
    let mut e1 = MultimodalNasEngine::with_default_search_space();
    let mut e2 = MultimodalNasEngine::with_default_search_space();
    let mut r1 = Random::seed(12345);
    let mut r2 = Random::seed(12345);
    let a1 = e1.propose(&mut r1).expect("p1");
    let a2 = e2.propose(&mut r2).expect("p2");
    assert_eq!(a1, a2);
    assert_eq!(a1.model_id, "multimodal_arch_0");
}

#[test]
fn test_proposed_ids_sequential() {
    let mut engine = MultimodalNasEngine::with_default_search_space();
    let mut rng = Random::seed(7);
    let a = engine.propose(&mut rng).expect("a");
    let b = engine.propose(&mut rng).expect("b");
    let c = engine.propose(&mut rng).expect("c");
    assert_eq!(a.model_id, "multimodal_arch_0");
    assert_eq!(b.model_id, "multimodal_arch_1");
    assert_eq!(c.model_id, "multimodal_arch_2");
}

#[test]
fn test_mutate_preserves_validity() {
    let engine = MultimodalNasEngine::with_default_search_space();
    let mut rng = Random::seed(99);
    let base = {
        let mut e = MultimodalNasEngine::with_default_search_space();
        let mut r = Random::seed(3);
        e.propose(&mut r).expect("base")
    };
    for _ in 0..100 {
        let mutated = engine.mutate(&base, &mut rng).expect("mutate");
        assert!(
            mutated.validate().is_ok(),
            "mutation produced invalid arch: {:?}",
            mutated.validate()
        );
        // Modality set is preserved.
        assert_eq!(mutated.modalities, base.modalities);
        assert_ne!(mutated.model_id, base.model_id);
    }
}

#[test]
fn test_mutate_changes_something() {
    let engine = MultimodalNasEngine::with_default_search_space();
    let mut rng = Random::seed(4242);
    let base = {
        let mut e = MultimodalNasEngine::with_default_search_space();
        let mut r = Random::seed(11);
        e.propose(&mut r).expect("base")
    };
    let mut saw_change = false;
    for _ in 0..50 {
        let mutated = engine.mutate(&base, &mut rng).expect("mutate");
        if mutated.encoders != base.encoders
            || mutated.fusion != base.fusion
            || mutated.head != base.head
        {
            saw_change = true;
        }
    }
    assert!(
        saw_change,
        "mutate produced no observable change in 50 tries"
    );
}

#[test]
fn test_crossover_preserves_validity() {
    let engine = MultimodalNasEngine::with_default_search_space();
    for seed in 0..100u64 {
        let mut e = MultimodalNasEngine::with_default_search_space();
        let mut rp = Random::seed(seed.wrapping_mul(7).wrapping_add(1));
        let a = e.propose(&mut rp).expect("a");
        let b = e.propose(&mut rp).expect("b");
        let mut rc = Random::seed(seed);
        let child = engine.crossover(&a, &b, &mut rc).expect("crossover");
        assert!(
            child.validate().is_ok(),
            "seed {} crossover invalid: {:?}",
            seed,
            child.validate()
        );
        assert!(child.model_id.contains("_x_"));
        // Child modalities are the union of the parents.
        for m in &a.modalities {
            assert!(child.modalities.contains(m));
        }
        for m in &b.modalities {
            assert!(child.modalities.contains(m));
        }
    }
}

#[test]
fn test_crossover_seed_reproducibility() {
    let engine = MultimodalNasEngine::with_default_search_space();
    let a = balanced_arch("a");
    let mut b = balanced_arch("b");
    b.encoders[1].layers = vec![
        MultimodalLayer::TextTransformer { heads: 4, dim: 64 },
        MultimodalLayer::Dense { out: 128 },
    ];
    let mut r1 = Random::seed(2024);
    let mut r2 = Random::seed(2024);
    let c1 = engine.crossover(&a, &b, &mut r1).expect("c1");
    let c2 = engine.crossover(&a, &b, &mut r2).expect("c2");
    assert_eq!(c1, c2);
}

#[test]
fn test_validate_accepts_balanced_arch() {
    let arch = balanced_arch("ok");
    assert!(arch.validate().is_ok());
}

#[test]
fn test_malformed_missing_encoder_fails() {
    // Declares Text but provides no Text encoder: fusion would run
    // before Text is encoded.
    let mut arch = balanced_arch("bad");
    arch.encoders.retain(|e| e.modality != Modality::Text);
    assert_eq!(
        arch.validate(),
        Err(MultimodalValidationError::MissingEncoder {
            modality: Modality::Text
        })
    );
}

#[test]
fn test_malformed_fusion_before_encoder_fails() {
    // An empty encoder for a declared modality means fusion executes
    // before that modality is encoded.
    let mut arch = balanced_arch("bad");
    if let Some(enc) = arch
        .encoders
        .iter_mut()
        .find(|e| e.modality == Modality::Text)
    {
        enc.layers.clear();
    }
    assert_eq!(
        arch.validate(),
        Err(MultimodalValidationError::EmptyEncoder {
            modality: Modality::Text
        })
    );
}

#[test]
fn test_malformed_foreign_layer_fails() {
    let mut arch = balanced_arch("bad");
    // Put a Text layer inside the Image encoder.
    arch.encoders[0]
        .layers
        .push(MultimodalLayer::TextEmbed { dim: 64 });
    assert!(matches!(
        arch.validate(),
        Err(MultimodalValidationError::ForeignLayerInEncoder { .. })
    ));
}

#[test]
fn test_malformed_bilinear_three_modalities_fails() {
    let arch = MultimodalArchitecture {
        modalities: vec![Modality::Image, Modality::Text, Modality::Audio],
        encoders: vec![
            ModalityEncoder {
                modality: Modality::Image,
                layers: vec![MultimodalLayer::ImageConv {
                    channels: 64,
                    kernel: 3,
                }],
            },
            ModalityEncoder {
                modality: Modality::Text,
                layers: vec![MultimodalLayer::TextEmbed { dim: 64 }],
            },
            ModalityEncoder {
                modality: Modality::Audio,
                layers: vec![MultimodalLayer::AudioRecurrent { hidden: 64 }],
            },
        ],
        fusion: FusionOp::Bilinear { dim: 128 },
        head: vec![MultimodalLayer::Dense { out: 64 }],
        parameters: HashMap::new(),
        model_id: "tri_bilinear".to_string(),
    };
    assert!(matches!(
        arch.validate(),
        Err(MultimodalValidationError::FusionModalityMismatch { .. })
    ));
}

#[test]
fn test_malformed_empty_head_fails() {
    let mut arch = balanced_arch("bad");
    arch.head.clear();
    assert_eq!(arch.validate(), Err(MultimodalValidationError::EmptyHead));
}

#[test]
fn test_capacity_imbalance_detected() {
    // Image encoder is tiny (Activation x2), Text encoder huge.
    let arch = MultimodalArchitecture {
        modalities: vec![Modality::Image, Modality::Text],
        encoders: vec![
            ModalityEncoder {
                modality: Modality::Image,
                layers: vec![
                    MultimodalLayer::Activation {
                        kind: ActivationKind::Relu,
                    },
                    MultimodalLayer::Activation {
                        kind: ActivationKind::Gelu,
                    },
                ],
            },
            ModalityEncoder {
                modality: Modality::Text,
                layers: vec![
                    MultimodalLayer::TextEmbed { dim: 256 },
                    MultimodalLayer::TextTransformer { heads: 4, dim: 64 },
                ],
            },
        ],
        fusion: FusionOp::EarlyConcat,
        head: vec![MultimodalLayer::Dense { out: 64 }],
        parameters: HashMap::new(),
        model_id: "imbalanced".to_string(),
    };
    assert!(matches!(
        arch.validate(),
        Err(MultimodalValidationError::CapacityImbalance { .. })
    ));
}

#[test]
fn test_no_modalities_fails() {
    let arch = MultimodalArchitecture {
        modalities: vec![],
        encoders: vec![],
        fusion: FusionOp::EarlyConcat,
        head: vec![MultimodalLayer::Dense { out: 64 }],
        parameters: HashMap::new(),
        model_id: "empty".to_string(),
    };
    assert_eq!(
        arch.validate(),
        Err(MultimodalValidationError::NoModalities)
    );
}

#[test]
fn test_record_evaluation_updates_best() {
    let mut engine = MultimodalNasEngine::with_default_search_space();
    let m1 = balanced_arch("m1");
    let m2 = balanced_arch("m2");
    engine
        .record_evaluation(
            m1,
            MultimodalEvaluation {
                accuracy: 0.80,
                latency_ms: 90.0,
                memory_mb: 130.0,
                param_count: 1_000_000,
            },
        )
        .expect("record m1");
    assert_eq!(
        engine.best().map(|m| m.model_id.clone()).expect("best m1"),
        "m1".to_string()
    );
    engine
        .record_evaluation(
            m2,
            MultimodalEvaluation {
                accuracy: 0.90,
                latency_ms: 120.0,
                memory_mb: 180.0,
                param_count: 4_000_000,
            },
        )
        .expect("record m2");
    assert_eq!(
        engine.best().map(|m| m.model_id.clone()).expect("best m2"),
        "m2".to_string()
    );
    let be = engine.best_evaluation().expect("best eval");
    assert!((be.accuracy - 0.90).abs() < 1e-12);
}

#[test]
fn test_record_evaluation_validates_ranges() {
    let mut engine = MultimodalNasEngine::with_default_search_space();
    let m = balanced_arch("bad");
    let neg = engine.record_evaluation(
        m.clone(),
        MultimodalEvaluation {
            accuracy: -0.01,
            latency_ms: 50.0,
            memory_mb: 10.0,
            param_count: 100,
        },
    );
    assert!(matches!(neg, Err(OptimError::InvalidParameter(_))));

    let too_big = engine.record_evaluation(
        m.clone(),
        MultimodalEvaluation {
            accuracy: 1.5,
            latency_ms: 50.0,
            memory_mb: 10.0,
            param_count: 100,
        },
    );
    assert!(matches!(too_big, Err(OptimError::InvalidParameter(_))));

    let bad_latency = engine.record_evaluation(
        m.clone(),
        MultimodalEvaluation {
            accuracy: 0.5,
            latency_ms: -1.0,
            memory_mb: 10.0,
            param_count: 100,
        },
    );
    assert!(matches!(bad_latency, Err(OptimError::InvalidParameter(_))));

    let nan = engine.record_evaluation(
        m,
        MultimodalEvaluation {
            accuracy: f64::NAN,
            latency_ms: 10.0,
            memory_mb: 10.0,
            param_count: 1,
        },
    );
    assert!(matches!(nan, Err(OptimError::InvalidParameter(_))));
}

#[test]
fn test_pareto_front_known_answer() {
    let mut engine = MultimodalNasEngine::with_default_search_space();
    // A: acc 0.90, lat 100, mem 60  (best accuracy -> non-dominated)
    // B: acc 0.80, lat  50, mem 30  (best lat/mem  -> non-dominated)
    // C: acc 0.70, lat 150, mem 80  (dominated by A on every axis)
    let mut a = balanced_arch("a");
    a.model_id = "a".to_string();
    let mut b = balanced_arch("b");
    b.model_id = "b".to_string();
    let mut c = balanced_arch("c");
    c.model_id = "c".to_string();
    engine
        .record_evaluation(
            a,
            MultimodalEvaluation {
                accuracy: 0.90,
                latency_ms: 100.0,
                memory_mb: 60.0,
                param_count: 1,
            },
        )
        .expect("a");
    engine
        .record_evaluation(
            b,
            MultimodalEvaluation {
                accuracy: 0.80,
                latency_ms: 50.0,
                memory_mb: 30.0,
                param_count: 2,
            },
        )
        .expect("b");
    engine
        .record_evaluation(
            c,
            MultimodalEvaluation {
                accuracy: 0.70,
                latency_ms: 150.0,
                memory_mb: 80.0,
                param_count: 3,
            },
        )
        .expect("c");
    let front = engine.pareto_front();
    assert_eq!(front.len(), 2);
    let ids: Vec<String> = front.iter().map(|m| m.model_id.clone()).collect();
    assert!(ids.contains(&"a".to_string()));
    assert!(ids.contains(&"b".to_string()));
    assert!(!ids.contains(&"c".to_string()));
}

#[test]
fn test_pareto_front_all_non_dominated() {
    let mut engine = MultimodalNasEngine::with_default_search_space();
    let mut a = balanced_arch("a");
    a.model_id = "a".to_string();
    let mut b = balanced_arch("b");
    b.model_id = "b".to_string();
    let mut c = balanced_arch("c");
    c.model_id = "c".to_string();
    engine
        .record_evaluation(
            a,
            MultimodalEvaluation {
                accuracy: 0.95,
                latency_ms: 200.0,
                memory_mb: 80.0,
                param_count: 1,
            },
        )
        .expect("a");
    engine
        .record_evaluation(
            b,
            MultimodalEvaluation {
                accuracy: 0.70,
                latency_ms: 40.0,
                memory_mb: 70.0,
                param_count: 2,
            },
        )
        .expect("b");
    engine
        .record_evaluation(
            c,
            MultimodalEvaluation {
                accuracy: 0.80,
                latency_ms: 100.0,
                memory_mb: 20.0,
                param_count: 3,
            },
        )
        .expect("c");
    assert_eq!(engine.pareto_front().len(), 3);
}

#[test]
fn test_pareto_front_empty_initially() {
    let engine = MultimodalNasEngine::with_default_search_space();
    assert!(engine.pareto_front().is_empty());
    assert!(engine.best().is_none());
    assert!(engine.best_evaluation().is_none());
}

#[test]
fn test_reset_clears_state() {
    let mut engine = MultimodalNasEngine::with_default_search_space();
    let mut rng = Random::seed(5);
    let arch = engine.propose(&mut rng).expect("propose");
    engine
        .record_evaluation(
            arch,
            MultimodalEvaluation {
                accuracy: 0.5,
                latency_ms: 50.0,
                memory_mb: 25.0,
                param_count: 1000,
            },
        )
        .expect("record");
    assert!(engine.best().is_some());
    assert!(!engine.evaluated().is_empty());
    engine.reset();
    assert!(engine.best().is_none());
    assert!(engine.evaluated().is_empty());
    let mut rng2 = Random::seed(6);
    let m = engine.propose(&mut rng2).expect("propose after reset");
    assert_eq!(m.model_id, "multimodal_arch_0");
}

#[test]
fn test_invalid_search_space_rejected() {
    let mut space = MultimodalSearchSpace::default();
    space.available_modalities.clear();
    let mut engine = MultimodalNasEngine::new(space);
    let mut rng = Random::seed(1);
    assert!(matches!(
        engine.propose(&mut rng),
        Err(OptimError::SearchSpaceError(_))
    ));
}

#[test]
fn test_layer_hash_eq() {
    use std::collections::HashSet;
    let mut set: HashSet<MultimodalLayer> = HashSet::new();
    set.insert(MultimodalLayer::ImageConv {
        channels: 64,
        kernel: 3,
    });
    set.insert(MultimodalLayer::ImageConv {
        channels: 64,
        kernel: 3,
    });
    set.insert(MultimodalLayer::ImageConv {
        channels: 64,
        kernel: 5,
    });
    assert_eq!(set.len(), 2);
}

#[test]
fn test_serde_roundtrip_in_memory() {
    let arch = balanced_arch("rt");
    let json = serde_json::to_string(&arch).expect("serialize");
    let back: MultimodalArchitecture = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(arch, back);

    let space = MultimodalSearchSpace::default();
    let json_space = serde_json::to_string(&space).expect("serialize space");
    let back_space: MultimodalSearchSpace =
        serde_json::from_str(&json_space).expect("deserialize space");
    assert_eq!(
        back_space.available_modalities.len(),
        space.available_modalities.len()
    );
    assert_eq!(back_space.min_encoder_len, space.min_encoder_len);
}

#[test]
fn test_architecture_serde_file_roundtrip() {
    let mut engine = MultimodalNasEngine::with_default_search_space();
    let mut rng = Random::seed(2024);
    let arch = engine.propose(&mut rng).expect("propose");

    let dir = std::env::temp_dir();
    let path = dir.join(format!(
        "optirs_nas_multimodal_{}_{}.json",
        std::process::id(),
        arch.model_id
    ));
    let json = serde_json::to_string(&arch).expect("serialize");
    std::fs::write(&path, &json).expect("write temp file");
    let read = std::fs::read_to_string(&path).expect("read temp file");
    let back: MultimodalArchitecture = serde_json::from_str(&read).expect("deserialize");
    assert_eq!(arch, back);
    let _ = std::fs::remove_file(&path);
}
