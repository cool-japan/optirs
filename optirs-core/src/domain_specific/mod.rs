// Domain-specific optimization strategies
//
// This module provides specialized optimization strategies tailored for different
// machine learning domains, building on the adaptive selection framework to provide
// domain-aware optimization approaches.

use crate::adaptive_selection::{OptimizerType, ProblemCharacteristics};
use crate::error::{OptimError, Result};
use scirs2_core::ndarray::ScalarOperand;
use scirs2_core::numeric::{Float, ToPrimitive};
use std::collections::HashMap;
use std::fmt::Debug;

/// Fallibly converts an `f64`/`usize` literal or measurement into the
/// selector's generic scalar type `A`.
///
/// Centralizes what used to be `A::from(x).expect("unwrap failed")` call
/// sites across the domain-specific hyperparameter builders: instead of
/// panicking, a type that genuinely cannot represent `x` now produces an
/// honest [`OptimError`].
fn cast_scalar<A: Float, T: ToPrimitive>(value: T) -> Result<A> {
    A::from(value).ok_or_else(|| {
        OptimError::InvalidConfig(
            "failed to convert a numeric value to the domain selector's scalar type".to_string(),
        )
    })
}

/// Domain-specific optimization strategy
#[derive(Debug, Clone)]
pub enum DomainStrategy {
    /// Computer Vision optimization
    ComputerVision {
        /// Image resolution considerations
        resolution_adaptive: bool,
        /// Batch normalization optimization
        batch_norm_tuning: bool,
        /// Data augmentation awareness
        augmentation_aware: bool,
    },
    /// Natural Language Processing optimization
    NaturalLanguage {
        /// Sequence length adaptation
        sequence_adaptive: bool,
        /// Attention mechanism optimization
        attention_optimized: bool,
        /// Vocabulary size considerations
        vocab_aware: bool,
    },
    /// Recommendation Systems optimization
    RecommendationSystems {
        /// Collaborative filtering optimization
        collaborative_filtering: bool,
        /// Matrix factorization tuning
        matrix_factorization: bool,
        /// Cold start handling
        cold_start_aware: bool,
    },
    /// Time Series optimization
    TimeSeries {
        /// Temporal dependency handling
        temporal_aware: bool,
        /// Seasonality consideration
        seasonality_adaptive: bool,
        /// Multi-step ahead optimization
        multi_step: bool,
    },
    /// Reinforcement Learning optimization
    ReinforcementLearning {
        /// Policy gradient optimization
        policy_gradient: bool,
        /// Value function optimization
        value_function: bool,
        /// Exploration-exploitation balance
        exploration_aware: bool,
    },
    /// Scientific Computing optimization
    ScientificComputing {
        /// Numerical stability prioritization
        stability_focused: bool,
        /// High precision requirements
        precision_critical: bool,
        /// Sparse matrix optimization
        sparse_optimized: bool,
    },
}

/// Domain-specific configuration parameters
#[derive(Debug, Clone)]
pub struct DomainConfig<A: Float> {
    /// Base learning rate for the domain
    pub base_learning_rate: A,
    /// Batch size recommendations
    pub recommended_batch_sizes: Vec<usize>,
    /// Gradient clipping thresholds
    pub gradient_clip_values: Vec<A>,
    /// Regularization strengths
    pub regularization_range: (A, A),
    /// Optimizer preferences (ranked by effectiveness)
    pub optimizer_ranking: Vec<OptimizerType>,
    /// Domain-specific hyperparameters
    pub domain_params: HashMap<String, A>,
}

/// Domain-specific optimizer selector
#[derive(Debug)]
pub struct DomainSpecificSelector<A: Float> {
    /// Current domain strategy
    strategy: DomainStrategy,
    /// Domain configuration
    config: DomainConfig<A>,
    /// Performance history per domain
    domain_performance: HashMap<String, Vec<DomainPerformanceMetrics<A>>>,
    /// Cross-domain transfer learning data
    transfer_knowledge: Vec<CrossDomainKnowledge<A>>,
    /// Current optimization context
    currentcontext: Option<OptimizationContext<A>>,
}

/// Performance metrics specific to domains
#[derive(Debug, Clone)]
pub struct DomainPerformanceMetrics<A: Float> {
    /// Standard performance metrics
    pub validation_accuracy: A,
    /// Domain-specific metrics
    pub domain_specific_score: A,
    /// Training stability
    pub stability_score: A,
    /// Convergence speed (epochs to target)
    pub convergence_epochs: usize,
    /// Resource efficiency
    pub resource_efficiency: A,
    /// Transfer learning potential
    pub transfer_score: A,
}

/// Cross-domain knowledge transfer
#[derive(Debug, Clone)]
pub struct CrossDomainKnowledge<A: Float> {
    /// Source domain
    pub source_domain: String,
    /// Target domain
    pub target_domain: String,
    /// Transferable hyperparameters
    pub transferable_params: HashMap<String, A>,
    /// Transfer effectiveness score
    pub transfer_score: A,
    /// Optimization strategy that worked
    pub successful_strategy: OptimizerType,
}

/// Current optimization context
#[derive(Debug, Clone)]
pub struct OptimizationContext<A: Float> {
    /// Problem characteristics
    pub problem_chars: ProblemCharacteristics,
    /// Resource constraints
    pub resource_constraints: ResourceConstraints<A>,
    /// Training characteristics
    pub training_config: TrainingConfiguration<A>,
    /// Domain-specific metadata
    pub domain_metadata: HashMap<String, String>,
}

/// Resource constraints for optimization
#[derive(Debug, Clone)]
pub struct ResourceConstraints<A: Float> {
    /// Maximum memory available (bytes)
    pub max_memory: usize,
    /// Maximum training time (seconds)
    pub max_time: A,
    /// GPU availability and type
    pub gpu_available: bool,
    /// Distributed training capability
    pub distributed_capable: bool,
    /// Energy efficiency requirements
    pub energy_efficient: bool,
}

/// Training configuration parameters
#[derive(Debug, Clone)]
pub struct TrainingConfiguration<A: Float> {
    /// Maximum number of epochs
    pub max_epochs: usize,
    /// Early stopping patience
    pub early_stopping_patience: usize,
    /// Validation frequency
    pub validation_frequency: usize,
    /// Learning rate scheduling
    pub lr_schedule_type: LearningRateScheduleType,
    /// Regularization approach
    pub regularization_approach: RegularizationApproach<A>,
}

/// Learning rate schedule types
#[derive(Debug, Clone)]
pub enum LearningRateScheduleType {
    /// Constant learning rate
    Constant,
    /// Exponential decay
    ExponentialDecay {
        /// Decay rate
        decay_rate: f64,
    },
    /// Cosine annealing
    CosineAnnealing {
        /// Maximum number of iterations
        t_max: usize,
    },
    /// Reduce on plateau
    ReduceOnPlateau {
        /// Number of epochs with no improvement
        patience: usize,
        /// Factor by which learning rate will be reduced
        factor: f64,
    },
    /// One cycle policy
    OneCycle {
        /// Maximum learning rate
        max_lr: f64,
    },
}

/// Regularization approach
#[derive(Debug, Clone)]
pub enum RegularizationApproach<A: Float> {
    /// L2 regularization only
    L2Only {
        /// Regularization weight
        weight: A,
    },
    /// L1 regularization only
    L1Only {
        /// Regularization weight
        weight: A,
    },
    /// Elastic net (L1 + L2)
    ElasticNet {
        /// L1 regularization weight
        l1_weight: A,
        /// L2 regularization weight
        l2_weight: A,
    },
    /// Dropout regularization
    Dropout {
        /// Dropout rate
        dropout_rate: A,
    },
    /// Combined approach
    Combined {
        /// L2 regularization weight
        l2_weight: A,
        /// Dropout rate
        dropout_rate: A,
        /// Additional regularization techniques
        additional_techniques: Vec<String>,
    },
}

impl<A: Float + ScalarOperand + Debug + std::iter::Sum + Send + Sync> DomainSpecificSelector<A> {
    /// Create a new domain-specific selector
    pub fn new(strategy: DomainStrategy) -> Self {
        let config = Self::default_config_for_strategy(&strategy);

        Self {
            strategy,
            config,
            domain_performance: HashMap::new(),
            transfer_knowledge: Vec::new(),
            currentcontext: None,
        }
    }

    /// Set optimization context
    pub fn setcontext(&mut self, context: OptimizationContext<A>) {
        self.currentcontext = Some(context);
    }

    /// Select optimal configuration for the current domain and context
    pub fn select_optimal_config(&mut self) -> Result<DomainOptimizationConfig<A>> {
        let context = self
            .currentcontext
            .as_ref()
            .ok_or_else(|| OptimError::InvalidConfig("No optimization context set".to_string()))?;

        match &self.strategy {
            DomainStrategy::ComputerVision {
                resolution_adaptive,
                batch_norm_tuning,
                augmentation_aware,
            } => self.optimize_computer_vision(
                context,
                *resolution_adaptive,
                *batch_norm_tuning,
                *augmentation_aware,
            ),
            DomainStrategy::NaturalLanguage {
                sequence_adaptive,
                attention_optimized,
                vocab_aware,
            } => self.optimize_natural_language(
                context,
                *sequence_adaptive,
                *attention_optimized,
                *vocab_aware,
            ),
            DomainStrategy::RecommendationSystems {
                collaborative_filtering,
                matrix_factorization,
                cold_start_aware,
            } => self.optimize_recommendation_systems(
                context,
                *collaborative_filtering,
                *matrix_factorization,
                *cold_start_aware,
            ),
            DomainStrategy::TimeSeries {
                temporal_aware,
                seasonality_adaptive,
                multi_step,
            } => self.optimize_time_series(
                context,
                *temporal_aware,
                *seasonality_adaptive,
                *multi_step,
            ),
            DomainStrategy::ReinforcementLearning {
                policy_gradient,
                value_function,
                exploration_aware,
            } => self.optimize_reinforcement_learning(
                context,
                *policy_gradient,
                *value_function,
                *exploration_aware,
            ),
            DomainStrategy::ScientificComputing {
                stability_focused,
                precision_critical,
                sparse_optimized,
            } => self.optimize_scientific_computing(
                context,
                *stability_focused,
                *precision_critical,
                *sparse_optimized,
            ),
        }
    }

    /// Optimize for computer vision tasks
    fn optimize_computer_vision(
        &self,
        context: &OptimizationContext<A>,
        resolution_adaptive: bool,
        batch_norm_tuning: bool,
        augmentation_aware: bool,
    ) -> Result<DomainOptimizationConfig<A>> {
        let mut config = DomainOptimizationConfig::default();

        // Resolution-_adaptive optimization
        if resolution_adaptive {
            let resolution_factor = self.estimate_resolution_factor(&context.problem_chars);
            config.learning_rate = self.config.base_learning_rate * cast_scalar(resolution_factor)?;

            // Larger images need smaller learning rates
            if context.problem_chars.input_dim > 512 * 512 {
                config.learning_rate = config.learning_rate * cast_scalar(0.5)?;
            }
        }

        // Batch normalization _tuning
        if batch_norm_tuning {
            config.optimizer_type = OptimizerType::AdamW; // Better for batch norm
            config
                .specialized_params
                .insert("batch_norm_momentum".to_string(), cast_scalar(0.99)?);
            config
                .specialized_params
                .insert("batch_norm_eps".to_string(), cast_scalar(1e-5)?);
        }

        // Data augmentation awareness
        if augmentation_aware {
            // More aggressive regularization with augmentation
            config.regularization_strength = config.regularization_strength * cast_scalar(1.5)?;
            config
                .specialized_params
                .insert("mixup_alpha".to_string(), cast_scalar(0.2)?);
            config
                .specialized_params
                .insert("cutmix_alpha".to_string(), cast_scalar(1.0)?);
        }

        // CV-specific optimizations
        config.batch_size = self.select_cv_batch_size(&context.resource_constraints);
        config.gradient_clip_norm = Some(cast_scalar(1.0)?);

        // Use cosine annealing for CV tasks
        config.lr_schedule = LearningRateScheduleType::CosineAnnealing {
            t_max: context.training_config.max_epochs,
        };

        Ok(config)
    }

    /// Optimize for natural language processing tasks
    fn optimize_natural_language(
        &self,
        context: &OptimizationContext<A>,
        sequence_adaptive: bool,
        attention_optimized: bool,
        vocab_aware: bool,
    ) -> Result<DomainOptimizationConfig<A>> {
        let mut config = DomainOptimizationConfig::default();

        // Sequence-_adaptive optimization
        if sequence_adaptive {
            let seq_length = context.problem_chars.input_dim; // Assuming input_dim represents sequence length

            // Longer sequences need more careful optimization
            if seq_length > 512 {
                config.learning_rate = self.config.base_learning_rate * cast_scalar(0.7)?;
                config.gradient_clip_norm = Some(cast_scalar(0.5)?);
            } else {
                config.learning_rate = self.config.base_learning_rate;
                config.gradient_clip_norm = Some(cast_scalar(1.0)?);
            }
        }

        // Attention mechanism optimization
        if attention_optimized {
            config.optimizer_type = OptimizerType::AdamW; // Best for transformers
            config
                .specialized_params
                .insert("attention_dropout".to_string(), cast_scalar(0.1)?);
            config
                .specialized_params
                .insert("attention_head_dim".to_string(), cast_scalar(64.0)?);

            // Layer-wise learning rate decay for transformers
            config
                .specialized_params
                .insert("layer_decay_rate".to_string(), cast_scalar(0.95)?);
        }

        // Vocabulary-_aware optimization
        if vocab_aware {
            let vocab_size = context.problem_chars.output_dim; // Assuming output_dim represents vocab size

            // Large vocabularies need special handling
            if vocab_size > 30000 {
                config
                    .specialized_params
                    .insert("tie_embeddings".to_string(), cast_scalar(1.0)?);
                config
                    .specialized_params
                    .insert("embedding_dropout".to_string(), cast_scalar(0.1)?);
            }
        }

        // NLP-specific optimizations
        config.batch_size = self.select_nlp_batch_size(&context.resource_constraints);
        config.lr_schedule = LearningRateScheduleType::OneCycle {
            max_lr: config.learning_rate.to_f64().ok_or_else(|| {
                OptimError::InvalidConfig(
                    "optimize_natural_language: learning_rate must convert to f64".to_string(),
                )
            })?,
        };

        // Warmup for transformers
        config
            .specialized_params
            .insert("warmup_steps".to_string(), cast_scalar(1000.0)?);

        Ok(config)
    }

    /// Optimize for recommendation systems
    fn optimize_recommendation_systems(
        &self,
        context: &OptimizationContext<A>,
        collaborative_filtering: bool,
        matrix_factorization: bool,
        cold_start_aware: bool,
    ) -> Result<DomainOptimizationConfig<A>> {
        let mut config = DomainOptimizationConfig::default();

        // Collaborative _filtering optimization
        if collaborative_filtering {
            config.optimizer_type = OptimizerType::Adam; // Good for sparse data
            config.regularization_strength = cast_scalar(0.01)?; // Prevent overfitting
            config
                .specialized_params
                .insert("negative_sampling_rate".to_string(), cast_scalar(5.0)?);
        }

        // Matrix _factorization tuning
        if matrix_factorization {
            config.learning_rate = cast_scalar(0.01)?; // Lower LR for stability
            config
                .specialized_params
                .insert("embedding_dim".to_string(), cast_scalar(128.0)?);
            config
                .specialized_params
                .insert("factorization_rank".to_string(), cast_scalar(50.0)?);
        }

        // Cold start handling
        if cold_start_aware {
            config
                .specialized_params
                .insert("content_weight".to_string(), cast_scalar(0.3)?);
            config
                .specialized_params
                .insert("popularity_bias".to_string(), cast_scalar(0.1)?);
        }

        // RecSys-specific optimizations
        config.batch_size = self.select_recsys_batch_size(&context.resource_constraints);
        config.gradient_clip_norm = Some(cast_scalar(5.0)?); // Higher clip for sparse gradients

        Ok(config)
    }

    /// Optimize for time series tasks
    fn optimize_time_series(
        &self,
        context: &OptimizationContext<A>,
        temporal_aware: bool,
        seasonality_adaptive: bool,
        multi_step: bool,
    ) -> Result<DomainOptimizationConfig<A>> {
        let mut config = DomainOptimizationConfig::default();

        // Temporal dependency handling
        if temporal_aware {
            config.optimizer_type = OptimizerType::RMSprop; // Good for RNNs
            config.learning_rate = cast_scalar(0.001)?; // Conservative for temporal stability
            config.specialized_params.insert(
                "sequence_length".to_string(),
                cast_scalar(context.problem_chars.input_dim as f64)?,
            );
        }

        // Seasonality consideration
        if seasonality_adaptive {
            config
                .specialized_params
                .insert("seasonal_periods".to_string(), cast_scalar(24.0)?); // Daily pattern
            config
                .specialized_params
                .insert("trend_strength".to_string(), cast_scalar(0.1)?);
        }

        // Multi-_step ahead optimization
        if multi_step {
            config
                .specialized_params
                .insert("prediction_horizon".to_string(), cast_scalar(12.0)?);
            config
                .specialized_params
                .insert("multi_step_loss_weight".to_string(), cast_scalar(0.8)?);
        }

        // Time series-specific optimizations
        config.batch_size = 32; // Smaller batches for temporal consistency
        config.gradient_clip_norm = Some(cast_scalar(1.0)?);
        config.lr_schedule = LearningRateScheduleType::ReduceOnPlateau {
            patience: 10,
            factor: 0.5,
        };

        Ok(config)
    }

    /// Optimize for reinforcement learning tasks
    fn optimize_reinforcement_learning(
        &self,
        context: &OptimizationContext<A>,
        policy_gradient: bool,
        value_function: bool,
        exploration_aware: bool,
    ) -> Result<DomainOptimizationConfig<A>> {
        let mut config = DomainOptimizationConfig::default();

        // Policy _gradient optimization
        if policy_gradient {
            config.optimizer_type = OptimizerType::Adam;
            config.learning_rate = cast_scalar(3e-4)?; // Standard RL learning rate
            config
                .specialized_params
                .insert("entropy_coeff".to_string(), cast_scalar(0.01)?);
        }

        // Value _function optimization
        if value_function {
            config
                .specialized_params
                .insert("value_loss_coeff".to_string(), cast_scalar(0.5)?);
            config
                .specialized_params
                .insert("huber_loss_delta".to_string(), cast_scalar(1.0)?);
        }

        // Exploration-exploitation balance
        if exploration_aware {
            config
                .specialized_params
                .insert("epsilon_start".to_string(), cast_scalar(1.0)?);
            config
                .specialized_params
                .insert("epsilon_end".to_string(), cast_scalar(0.1)?);
            config
                .specialized_params
                .insert("epsilon_decay".to_string(), cast_scalar(0.995)?);
        }

        // RL-specific optimizations
        config.batch_size = self.select_rl_batch_size(&context.resource_constraints);
        config.gradient_clip_norm = Some(cast_scalar(0.5)?); // Important for RL stability
        config.lr_schedule = LearningRateScheduleType::Constant; // Often constant in RL

        Ok(config)
    }

    /// Optimize for scientific computing tasks
    fn optimize_scientific_computing(
        &self,
        context: &OptimizationContext<A>,
        stability_focused: bool,
        precision_critical: bool,
        sparse_optimized: bool,
    ) -> Result<DomainOptimizationConfig<A>> {
        let mut config = DomainOptimizationConfig::default();

        // Numerical stability prioritization
        if stability_focused {
            config.optimizer_type = OptimizerType::LBFGS; // More stable for scientific problems
            config.learning_rate = cast_scalar(0.1)?; // Higher LR for LBFGS
            config
                .specialized_params
                .insert("line_search_tolerance".to_string(), cast_scalar(1e-6)?);
        }

        // High precision requirements
        if precision_critical {
            config
                .specialized_params
                .insert("convergence_tolerance".to_string(), cast_scalar(1e-8)?);
            config
                .specialized_params
                .insert("max_iterations".to_string(), cast_scalar(1000.0)?);
        }

        // Sparse matrix optimization
        if sparse_optimized {
            config.optimizer_type = OptimizerType::Adam;
            config
                .specialized_params
                .insert("sparsity_threshold".to_string(), cast_scalar(1e-6)?);
        }

        // Scientific computing-specific optimizations
        config.batch_size = context.problem_chars.dataset_size.min(1024); // Can use larger batches
        config.gradient_clip_norm = None; // Don't clip for scientific precision
        config.lr_schedule = LearningRateScheduleType::Constant; // Consistent optimization

        Ok(config)
    }

    /// Update performance based on training results
    pub fn update_domain_performance(
        &mut self,
        domain: String,
        metrics: DomainPerformanceMetrics<A>,
    ) {
        self.domain_performance
            .entry(domain)
            .or_default()
            .push(metrics);
    }

    /// Record cross-domain transfer knowledge
    pub fn record_transfer_knowledge(&mut self, knowledge: CrossDomainKnowledge<A>) {
        self.transfer_knowledge.push(knowledge);
    }

    /// Get domain-specific recommendations
    pub fn get_domain_recommendations(&self, domain: &str) -> Vec<DomainRecommendation<A>> {
        let mut recommendations = Vec::new();

        // Analyze historical performance for this domain
        if let Some(history) = self.domain_performance.get(domain) {
            if !history.is_empty() {
                let history_len: A = A::from(history.len())
                    .expect("get_domain_recommendations: history.len() must fit in A");
                let avg_performance =
                    history.iter().map(|m| m.validation_accuracy).sum::<A>() / history_len;

                recommendations.push(DomainRecommendation {
                    recommendation_type: RecommendationType::PerformanceBaseline,
                    description: format!(
                        "Historical average performance: {:.4}",
                        avg_performance.to_f64().expect(
                            "get_domain_recommendations: A (f32/f64) always converts to f64"
                        )
                    ),
                    confidence: A::from(0.8)
                        .expect("get_domain_recommendations: literal 0.8 must fit in A"),
                    action: "Consider this as baseline for improvements".to_string(),
                });
            }
        }

        // Cross-domain transfer recommendations
        for knowledge in &self.transfer_knowledge {
            if knowledge.target_domain == domain {
                recommendations.push(DomainRecommendation {
                    recommendation_type: RecommendationType::TransferLearning,
                    description: format!(
                        "Transfer from {} domain with {:.2} effectiveness",
                        knowledge.source_domain,
                        knowledge.transfer_score.to_f64().expect(
                            "get_domain_recommendations: A (f32/f64) always converts to f64"
                        )
                    ),
                    confidence: knowledge.transfer_score,
                    action: format!("Use {:?} optimizer", knowledge.successful_strategy),
                });
            }
        }

        recommendations
    }

    /// Helper methods for domain-specific optimizations
    fn estimate_resolution_factor(&self, problem_chars: &ProblemCharacteristics) -> f64 {
        let resolution = problem_chars.input_dim as f64;

        if resolution > 1_000_000.0 {
            // Very high resolution
            0.5
        } else if resolution > 250_000.0 {
            // High resolution
            0.7
        } else if resolution > 50_000.0 {
            // Medium resolution
            0.9
        } else {
            // Low resolution
            1.0
        }
    }

    fn select_cv_batch_size(&self, constraints: &ResourceConstraints<A>) -> usize {
        if constraints.max_memory > 16_000_000_000 {
            // 16GB+
            128
        } else if constraints.max_memory > 8_000_000_000 {
            // 8GB+
            64
        } else {
            32
        }
    }

    fn select_nlp_batch_size(&self, constraints: &ResourceConstraints<A>) -> usize {
        if constraints.max_memory > 32_000_000_000 {
            // 32GB+
            64
        } else if constraints.max_memory > 16_000_000_000 {
            // 16GB+
            32
        } else {
            16
        }
    }

    fn select_recsys_batch_size(&self, constraints: &ResourceConstraints<A>) -> usize {
        // RecSys can typically use larger batches due to simpler models
        if constraints.max_memory > 8_000_000_000 {
            512
        } else {
            256
        }
    }

    fn select_rl_batch_size(&self, constraints: &ResourceConstraints<A>) -> usize {
        // RL rollout/replay batches scale with available memory, but more
        // conservatively than supervised CV/NLP batches: most on-policy and
        // off-policy algorithms are sensitive to very large batch sizes.
        if constraints.max_memory > 16_000_000_000 {
            // 16GB+
            128
        } else if constraints.max_memory > 8_000_000_000 {
            // 8GB+
            64
        } else {
            32
        }
    }

    /// Create default configuration for a strategy
    fn default_config_for_strategy(strategy: &DomainStrategy) -> DomainConfig<A> {
        match strategy {
            DomainStrategy::ComputerVision { .. } => DomainConfig {
                base_learning_rate: A::from(0.001)
                    .expect("default_config_for_strategy: literal must fit in A (f32/f64)"),
                recommended_batch_sizes: vec![32, 64, 128],
                gradient_clip_values: vec![
                    A::from(1.0)
                        .expect("default_config_for_strategy: literal must fit in A (f32/f64)"),
                    A::from(2.0)
                        .expect("default_config_for_strategy: literal must fit in A (f32/f64)"),
                ],
                regularization_range: (
                    A::from(1e-5)
                        .expect("default_config_for_strategy: literal must fit in A (f32/f64)"),
                    A::from(1e-2)
                        .expect("default_config_for_strategy: literal must fit in A (f32/f64)"),
                ),
                optimizer_ranking: vec![
                    OptimizerType::AdamW,
                    OptimizerType::SGDMomentum,
                    OptimizerType::Adam,
                ],
                domain_params: HashMap::new(),
            },
            DomainStrategy::NaturalLanguage { .. } => DomainConfig {
                base_learning_rate: A::from(2e-5)
                    .expect("default_config_for_strategy: literal must fit in A (f32/f64)"),
                recommended_batch_sizes: vec![16, 32, 64],
                gradient_clip_values: vec![
                    A::from(0.5)
                        .expect("default_config_for_strategy: literal must fit in A (f32/f64)"),
                    A::from(1.0)
                        .expect("default_config_for_strategy: literal must fit in A (f32/f64)"),
                ],
                regularization_range: (
                    A::from(1e-4)
                        .expect("default_config_for_strategy: literal must fit in A (f32/f64)"),
                    A::from(1e-1)
                        .expect("default_config_for_strategy: literal must fit in A (f32/f64)"),
                ),
                optimizer_ranking: vec![OptimizerType::AdamW, OptimizerType::Adam],
                domain_params: HashMap::new(),
            },
            DomainStrategy::RecommendationSystems { .. } => DomainConfig {
                base_learning_rate: A::from(0.01)
                    .expect("default_config_for_strategy: literal must fit in A (f32/f64)"),
                recommended_batch_sizes: vec![128, 256, 512],
                gradient_clip_values: vec![
                    A::from(5.0)
                        .expect("default_config_for_strategy: literal must fit in A (f32/f64)"),
                    A::from(10.0)
                        .expect("default_config_for_strategy: literal must fit in A (f32/f64)"),
                ],
                regularization_range: (
                    A::from(1e-3)
                        .expect("default_config_for_strategy: literal must fit in A (f32/f64)"),
                    A::from(1e-1)
                        .expect("default_config_for_strategy: literal must fit in A (f32/f64)"),
                ),
                optimizer_ranking: vec![OptimizerType::Adam, OptimizerType::AdaGrad],
                domain_params: HashMap::new(),
            },
            DomainStrategy::TimeSeries { .. } => DomainConfig {
                base_learning_rate: A::from(0.001)
                    .expect("default_config_for_strategy: literal must fit in A (f32/f64)"),
                recommended_batch_sizes: vec![16, 32, 64],
                gradient_clip_values: vec![A::from(1.0)
                    .expect("default_config_for_strategy: literal must fit in A (f32/f64)")],
                regularization_range: (
                    A::from(1e-4)
                        .expect("default_config_for_strategy: literal must fit in A (f32/f64)"),
                    A::from(1e-2)
                        .expect("default_config_for_strategy: literal must fit in A (f32/f64)"),
                ),
                optimizer_ranking: vec![OptimizerType::RMSprop, OptimizerType::Adam],
                domain_params: HashMap::new(),
            },
            DomainStrategy::ReinforcementLearning { .. } => DomainConfig {
                base_learning_rate: A::from(3e-4)
                    .expect("default_config_for_strategy: literal must fit in A (f32/f64)"),
                recommended_batch_sizes: vec![32, 64, 128],
                gradient_clip_values: vec![A::from(0.5)
                    .expect("default_config_for_strategy: literal must fit in A (f32/f64)")],
                regularization_range: (
                    A::from(1e-4)
                        .expect("default_config_for_strategy: literal must fit in A (f32/f64)"),
                    A::from(1e-2)
                        .expect("default_config_for_strategy: literal must fit in A (f32/f64)"),
                ),
                optimizer_ranking: vec![OptimizerType::Adam],
                domain_params: HashMap::new(),
            },
            DomainStrategy::ScientificComputing { .. } => DomainConfig {
                base_learning_rate: A::from(0.1)
                    .expect("default_config_for_strategy: literal must fit in A (f32/f64)"),
                recommended_batch_sizes: vec![64, 128, 256, 512],
                gradient_clip_values: vec![],
                regularization_range: (
                    A::from(1e-6)
                        .expect("default_config_for_strategy: literal must fit in A (f32/f64)"),
                    A::from(1e-3)
                        .expect("default_config_for_strategy: literal must fit in A (f32/f64)"),
                ),
                optimizer_ranking: vec![OptimizerType::LBFGS, OptimizerType::Adam],
                domain_params: HashMap::new(),
            },
        }
    }
}

/// Final domain optimization configuration
#[derive(Debug, Clone)]
pub struct DomainOptimizationConfig<A: Float> {
    /// Selected optimizer type
    pub optimizer_type: OptimizerType,
    /// Optimized learning rate
    pub learning_rate: A,
    /// Optimized batch size
    pub batch_size: usize,
    /// Gradient clipping norm (if applicable)
    pub gradient_clip_norm: Option<A>,
    /// Regularization strength
    pub regularization_strength: A,
    /// Learning rate schedule
    pub lr_schedule: LearningRateScheduleType,
    /// Domain-specific specialized parameters
    pub specialized_params: HashMap<String, A>,
}

impl<A: Float + Send + Sync> Default for DomainOptimizationConfig<A> {
    fn default() -> Self {
        Self {
            optimizer_type: OptimizerType::Adam,
            learning_rate: A::from(0.001)
                .expect("DomainOptimizationConfig: default learning_rate (0.001) must fit in A"),
            batch_size: 32,
            gradient_clip_norm: Some(A::from(1.0).expect(
                "DomainOptimizationConfig: default gradient_clip_norm (1.0) must fit in A",
            )),
            regularization_strength: A::from(1e-4).expect(
                "DomainOptimizationConfig: default regularization_strength (1e-4) must fit in A",
            ),
            lr_schedule: LearningRateScheduleType::Constant,
            specialized_params: HashMap::new(),
        }
    }
}

/// Domain-specific recommendation
#[derive(Debug, Clone)]
pub struct DomainRecommendation<A: Float> {
    /// Type of recommendation
    pub recommendation_type: RecommendationType,
    /// Human-readable description
    pub description: String,
    /// Confidence in the recommendation (0.0-1.0)
    pub confidence: A,
    /// Suggested action
    pub action: String,
}

/// Types of domain recommendations
#[derive(Debug, Clone)]
pub enum RecommendationType {
    /// Performance baseline information
    PerformanceBaseline,
    /// Transfer learning suggestion
    TransferLearning,
    /// Hyperparameter adjustment
    HyperparameterTuning,
    /// Architecture modification
    ArchitectureChange,
    /// Resource optimization
    ResourceOptimization,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adaptive_selection::ProblemType;

    #[test]
    fn test_domain_specific_selector_creation() {
        let strategy = DomainStrategy::ComputerVision {
            resolution_adaptive: true,
            batch_norm_tuning: true,
            augmentation_aware: true,
        };

        let selector = DomainSpecificSelector::<f64>::new(strategy);
        assert_eq!(selector.config.optimizer_ranking[0], OptimizerType::AdamW);
    }

    #[test]
    fn test_computer_vision_optimization() {
        let strategy = DomainStrategy::ComputerVision {
            resolution_adaptive: true,
            batch_norm_tuning: true,
            augmentation_aware: true,
        };

        let mut selector = DomainSpecificSelector::<f64>::new(strategy);

        let context = OptimizationContext {
            problem_chars: ProblemCharacteristics {
                dataset_size: 50000,
                input_dim: 224 * 224 * 3, // Standard ImageNet resolution
                output_dim: 1000,
                problem_type: ProblemType::ComputerVision,
                gradient_sparsity: 0.1,
                gradient_noise: 0.05,
                memory_budget: 8_000_000_000,
                time_budget: 3600.0,
                batch_size: 64,
                lr_sensitivity: 0.5,
                regularization_strength: 0.01,
                architecture_type: Some("ResNet".to_string()),
            },
            resource_constraints: ResourceConstraints {
                max_memory: 17_000_000_000, // Slightly above 16GB to trigger 128 batch size
                max_time: 7200.0,
                gpu_available: true,
                distributed_capable: false,
                energy_efficient: false,
            },
            training_config: TrainingConfiguration {
                max_epochs: 100,
                early_stopping_patience: 10,
                validation_frequency: 1,
                lr_schedule_type: LearningRateScheduleType::CosineAnnealing { t_max: 100 },
                regularization_approach: RegularizationApproach::L2Only { weight: 1e-4 },
            },
            domain_metadata: HashMap::new(),
        };

        selector.setcontext(context);
        let config = selector
            .select_optimal_config()
            .expect("selector.select_optimal_config succeeds in test_computer_vision_optimization");

        assert_eq!(config.optimizer_type, OptimizerType::AdamW);
        assert_eq!(config.batch_size, 128); // Should select larger batch size for high memory
        assert!(config.gradient_clip_norm.is_some());
    }

    #[test]
    fn test_reinforcement_learning_optimization_uses_resource_constraints() {
        let strategy = DomainStrategy::ReinforcementLearning {
            policy_gradient: true,
            value_function: true,
            exploration_aware: true,
        };

        let mut selector = DomainSpecificSelector::<f64>::new(strategy);

        let make_context = |max_memory: usize| OptimizationContext {
            problem_chars: ProblemCharacteristics {
                dataset_size: 10000,
                input_dim: 128,
                output_dim: 4,
                problem_type: ProblemType::ReinforcementLearning,
                gradient_sparsity: 0.0,
                gradient_noise: 0.1,
                memory_budget: max_memory,
                time_budget: 3600.0,
                batch_size: 64,
                lr_sensitivity: 0.5,
                regularization_strength: 0.0,
                architecture_type: None,
            },
            resource_constraints: ResourceConstraints {
                max_memory,
                max_time: 3600.0,
                gpu_available: true,
                distributed_capable: false,
                energy_efficient: false,
            },
            training_config: TrainingConfiguration {
                max_epochs: 1000,
                early_stopping_patience: 50,
                validation_frequency: 10,
                lr_schedule_type: LearningRateScheduleType::Constant,
                regularization_approach: RegularizationApproach::L2Only { weight: 0.0 },
            },
            domain_metadata: HashMap::new(),
        };

        // Low memory: falls back to the smallest RL batch-size tier.
        selector.setcontext(make_context(4_000_000_000));
        let low_mem_config = selector.select_optimal_config().expect("selector.select_optimal_config succeeds in test_reinforcement_learning_optimization_uses_resource_constraints");
        assert_eq!(low_mem_config.batch_size, 32);

        // High memory: batch size should scale up accordingly, proving the
        // resource constraints actually drive the RL batch-size decision
        // rather than a hardcoded constant.
        selector.setcontext(make_context(17_000_000_000));
        let high_mem_config = selector.select_optimal_config().expect("selector.select_optimal_config succeeds in test_reinforcement_learning_optimization_uses_resource_constraints");
        assert_eq!(high_mem_config.batch_size, 128);
        assert!(high_mem_config.gradient_clip_norm.is_some());
    }

    #[test]
    fn test_natural_language_optimization() {
        let strategy = DomainStrategy::NaturalLanguage {
            sequence_adaptive: true,
            attention_optimized: true,
            vocab_aware: true,
        };

        let mut selector = DomainSpecificSelector::<f64>::new(strategy);

        let context = OptimizationContext {
            problem_chars: ProblemCharacteristics {
                dataset_size: 100000,
                input_dim: 512,    // Sequence length
                output_dim: 50000, // Large vocabulary
                problem_type: ProblemType::NaturalLanguage,
                gradient_sparsity: 0.2,
                gradient_noise: 0.1,
                memory_budget: 32_000_000_000,
                time_budget: 7200.0,
                batch_size: 32,
                lr_sensitivity: 0.8,
                regularization_strength: 0.1,
                architecture_type: Some("Transformer".to_string()),
            },
            resource_constraints: ResourceConstraints {
                max_memory: 32_000_000_000,
                max_time: 10800.0,
                gpu_available: true,
                distributed_capable: true,
                energy_efficient: false,
            },
            training_config: TrainingConfiguration {
                max_epochs: 50,
                early_stopping_patience: 5,
                validation_frequency: 1,
                lr_schedule_type: LearningRateScheduleType::OneCycle { max_lr: 2e-5 },
                regularization_approach: RegularizationApproach::Dropout { dropout_rate: 0.1 },
            },
            domain_metadata: HashMap::new(),
        };

        selector.setcontext(context);
        let config = selector.select_optimal_config().expect(
            "selector.select_optimal_config succeeds in test_natural_language_optimization",
        );

        assert_eq!(config.optimizer_type, OptimizerType::AdamW);
        assert!(config.specialized_params.contains_key("warmup_steps"));
        assert!(config.specialized_params.contains_key("tie_embeddings")); // Large vocab
    }

    #[test]
    fn test_time_series_optimization() {
        let strategy = DomainStrategy::TimeSeries {
            temporal_aware: true,
            seasonality_adaptive: true,
            multi_step: true,
        };

        let mut selector = DomainSpecificSelector::<f64>::new(strategy);

        let context = OptimizationContext {
            problem_chars: ProblemCharacteristics {
                dataset_size: 10000,
                input_dim: 168, // One week of hourly data
                output_dim: 24, // Next 24 hours
                problem_type: ProblemType::TimeSeries,
                gradient_sparsity: 0.05,
                gradient_noise: 0.2,
                memory_budget: 4_000_000_000,
                time_budget: 1800.0,
                batch_size: 32,
                lr_sensitivity: 0.7,
                regularization_strength: 0.01,
                architecture_type: Some("LSTM".to_string()),
            },
            resource_constraints: ResourceConstraints {
                max_memory: 8_000_000_000,
                max_time: 3600.0,
                gpu_available: true,
                distributed_capable: false,
                energy_efficient: true,
            },
            training_config: TrainingConfiguration {
                max_epochs: 200,
                early_stopping_patience: 20,
                validation_frequency: 5,
                lr_schedule_type: LearningRateScheduleType::ReduceOnPlateau {
                    patience: 10,
                    factor: 0.5,
                },
                regularization_approach: RegularizationApproach::L2Only { weight: 1e-4 },
            },
            domain_metadata: HashMap::new(),
        };

        selector.setcontext(context);
        let config = selector
            .select_optimal_config()
            .expect("selector.select_optimal_config succeeds in test_time_series_optimization");

        assert_eq!(config.optimizer_type, OptimizerType::RMSprop);
        assert_eq!(config.batch_size, 32);
        assert!(config.specialized_params.contains_key("seasonal_periods"));
        assert!(config.specialized_params.contains_key("prediction_horizon"));
    }

    #[test]
    fn test_performance_tracking() {
        let strategy = DomainStrategy::ComputerVision {
            resolution_adaptive: true,
            batch_norm_tuning: false,
            augmentation_aware: false,
        };

        let mut selector = DomainSpecificSelector::<f64>::new(strategy);

        let metrics = DomainPerformanceMetrics {
            validation_accuracy: 0.95,
            domain_specific_score: 0.92,
            stability_score: 0.88,
            convergence_epochs: 50,
            resource_efficiency: 0.85,
            transfer_score: 0.7,
        };

        selector.update_domain_performance("computer_vision".to_string(), metrics);

        let recommendations = selector.get_domain_recommendations("computer_vision");
        assert!(!recommendations.is_empty());
        assert!(recommendations[0].description.contains("0.95"));
    }

    #[test]
    fn test_cross_domain_transfer() {
        let strategy = DomainStrategy::ComputerVision {
            resolution_adaptive: true,
            batch_norm_tuning: true,
            augmentation_aware: true,
        };

        let mut selector = DomainSpecificSelector::<f64>::new(strategy);

        let transfer_knowledge = CrossDomainKnowledge {
            source_domain: "natural_language".to_string(),
            target_domain: "computer_vision".to_string(),
            transferable_params: HashMap::from([
                ("learning_rate".to_string(), 0.001),
                ("weight_decay".to_string(), 0.01),
            ]),
            transfer_score: 0.8,
            successful_strategy: OptimizerType::AdamW,
        };

        selector.record_transfer_knowledge(transfer_knowledge);

        let recommendations = selector.get_domain_recommendations("computer_vision");
        assert!(recommendations
            .iter()
            .any(|r| matches!(r.recommendation_type, RecommendationType::TransferLearning)));
    }

    #[test]
    fn test_scientific_computing_optimization() {
        let strategy = DomainStrategy::ScientificComputing {
            stability_focused: true,
            precision_critical: true,
            sparse_optimized: false,
        };

        let mut selector = DomainSpecificSelector::<f64>::new(strategy);

        let context = OptimizationContext {
            problem_chars: ProblemCharacteristics {
                dataset_size: 1000,
                input_dim: 100,
                output_dim: 1,
                problem_type: ProblemType::Regression,
                gradient_sparsity: 0.01,
                gradient_noise: 0.01,
                memory_budget: 16_000_000_000,
                time_budget: 7200.0,
                batch_size: 100,
                lr_sensitivity: 0.3,
                regularization_strength: 1e-6,
                architecture_type: Some("MLP".to_string()),
            },
            resource_constraints: ResourceConstraints {
                max_memory: 16_000_000_000,
                max_time: 7200.0,
                gpu_available: false,
                distributed_capable: false,
                energy_efficient: false,
            },
            training_config: TrainingConfiguration {
                max_epochs: 1000,
                early_stopping_patience: 100,
                validation_frequency: 10,
                lr_schedule_type: LearningRateScheduleType::Constant,
                regularization_approach: RegularizationApproach::L2Only { weight: 1e-6 },
            },
            domain_metadata: HashMap::new(),
        };

        selector.setcontext(context);
        let config = selector.select_optimal_config().expect(
            "selector.select_optimal_config succeeds in test_scientific_computing_optimization",
        );

        assert_eq!(config.optimizer_type, OptimizerType::LBFGS);
        assert!(config.gradient_clip_norm.is_none()); // No clipping for precision
        assert!(config
            .specialized_params
            .contains_key("convergence_tolerance"));
    }
}
