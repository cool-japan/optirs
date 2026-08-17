use super::{
    ActorCriticConfig, ActorCriticMethod, ActorCriticOptimizer, Experience, ExperienceReplayBuffer,
    SACConfig,
};
use crate::error::Result;
use crate::reinforcement_learning::{
    ActionDistribution, DistributionType, PolicyEvaluation, PolicyNetwork, QNetwork, ValueNetwork,
};
use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;

// ── Minimal mock networks ─────────────────────────────────────────────

/// Action-value critic returning a constant `q`, with a real (zero) gradient
/// API so the update paths can run end to end.
#[derive(Clone)]
struct MockQ {
    q: f64,
}

impl ValueNetwork<f64> for MockQ {
    fn evaluate_value(&self, _obs: &Array2<f64>) -> Result<Array1<f64>> {
        Err(crate::error::OptimError::UnsupportedOperation(
            "MockQ is an action-value critic".to_string(),
        ))
    }
    fn update_parameters(&mut self, _: &HashMap<String, Array1<f64>>) -> Result<()> {
        Ok(())
    }
    fn get_parameters(&self) -> HashMap<String, Array1<f64>> {
        HashMap::new()
    }
}

impl QNetwork<f64> for MockQ {
    fn evaluate_q(&self, states: &Array2<f64>, _actions: &Array2<f64>) -> Result<Array1<f64>> {
        Ok(Array1::from_elem(states.nrows(), self.q))
    }
    fn q_gradient(
        &self,
        _states: &Array2<f64>,
        _actions: &Array2<f64>,
        _residuals: &Array1<f64>,
    ) -> Result<HashMap<String, Array1<f64>>> {
        Ok(HashMap::new())
    }
    fn action_gradient(&self, states: &Array2<f64>, actions: &Array2<f64>) -> Result<Array2<f64>> {
        Ok(Array2::zeros((states.nrows(), actions.ncols())))
    }
}

#[derive(Clone)]
struct MockPolicy;

impl PolicyNetwork<f64> for MockPolicy {
    fn evaluate_actions(
        &self,
        obs: &Array2<f64>,
        _: &Array2<f64>,
    ) -> Result<PolicyEvaluation<f64>> {
        Ok(PolicyEvaluation {
            log_probs: Array1::zeros(obs.nrows()),
            entropy: Array1::zeros(obs.nrows()),
            metrics: HashMap::new(),
        })
    }
    fn get_action_distribution(&self, obs: &Array2<f64>) -> Result<ActionDistribution<f64>> {
        let n = obs.nrows();
        Ok(ActionDistribution {
            mean: Some(Array2::zeros((n, 2))),
            std: Some(Array2::from_elem((n, 2), 1.0_f64)),
            logits: None,
            distribution_type: DistributionType::Gaussian,
        })
    }
    fn update_parameters(&mut self, _: &HashMap<String, Array1<f64>>) -> Result<()> {
        Ok(())
    }
    fn get_parameters(&self) -> HashMap<String, Array1<f64>> {
        HashMap::new()
    }
}

/// Build an optimizer with `temperature = 0` so the SAC soft target reduces to
/// the plain TD target and the assertions stay deterministic.
fn opt_with(critics: Vec<MockQ>) -> ActorCriticOptimizer<f64, MockPolicy, MockQ> {
    let n = critics.len();
    let cfg = ActorCriticConfig::<f64> {
        n_critics: n,
        sac_config: SACConfig::<f64> {
            temperature: 0.0,
            ..SACConfig::default()
        },
        ..ActorCriticConfig::default()
    };
    ActorCriticOptimizer::new(cfg, MockPolicy, critics).expect("construction")
}

// ── compute_target_q_sac — TD bootstrap r + γ(1−done)·min Q ──────────

#[test]
fn test_target_q_sac_twin_critics_take_minimum() {
    // critics return 3 and 5 → twin-critic min is 3, not 5
    let opt = opt_with(vec![MockQ { q: 3.0 }, MockQ { q: 5.0 }]);
    let states = Array2::zeros((3_usize, 4));
    let rewards = Array1::from_vec(vec![1.0_f64, 2.0, 3.0]);
    let dones = Array1::from_vec(vec![false, true, false]);

    let targets = opt
        .compute_target_q_sac(&states, &rewards, &dones)
        .expect("compute_target_q_sac");

    let gamma = 0.99_f64; // default discount factor
    let q_min = 3.0_f64;
    assert!((targets[0] - (1.0 + gamma * q_min)).abs() < 1e-9);
    assert!(
        (targets[1] - 2.0).abs() < 1e-9,
        "done target[1]={} should equal reward=2.0",
        targets[1]
    );
    assert!((targets[2] - (3.0 + gamma * q_min)).abs() < 1e-9);
}

#[test]
fn test_target_q_sac_single_critic_bootstrap() {
    let opt = opt_with(vec![MockQ { q: 4.0 }]);
    let states = Array2::zeros((2_usize, 4));
    let rewards = Array1::from_vec(vec![2.0_f64, 3.0]);
    let dones = Array1::from_vec(vec![false, false]);

    let targets = opt
        .compute_target_q_sac(&states, &rewards, &dones)
        .expect("compute_target_q_sac");

    let gamma = 0.99_f64;
    assert!((targets[0] - (2.0 + gamma * 4.0)).abs() < 1e-9);
    assert!((targets[1] - (3.0 + gamma * 4.0)).abs() < 1e-9);
}

#[test]
fn test_target_q_sac_all_done_no_bootstrap() {
    let opt = opt_with(vec![MockQ { q: 99.0 }, MockQ { q: 99.0 }]);
    let states = Array2::zeros((3_usize, 2));
    let rewards = Array1::from_vec(vec![5.0_f64, 6.0, 7.0]);
    let dones = Array1::from_vec(vec![true, true, true]);

    let targets = opt
        .compute_target_q_sac(&states, &rewards, &dones)
        .expect("compute_target_q_sac");

    for i in 0..3 {
        assert!(
            (targets[i] - rewards[i]).abs() < 1e-12,
            "done target[{i}]={} should equal reward={}",
            targets[i],
            rewards[i]
        );
    }
}

// ── F48 regression: SAC temperature sign ─────────────────────────────

#[test]
fn test_temperature_decreases_when_entropy_exceeds_target() {
    let cfg = ActorCriticConfig::<f64> {
        method: ActorCriticMethod::SAC,
        sac_config: SACConfig::<f64> {
            temperature: 0.2,
            auto_entropy_tuning: true,
            target_entropy: Some(-2.0),
            temperature_lr: 0.1,
            ..SACConfig::default()
        },
        ..ActorCriticConfig::default()
    };
    let mut opt =
        ActorCriticOptimizer::new(cfg, MockPolicy, vec![MockQ { q: 0.0 }]).expect("build");

    let before = opt.temperature();
    // Current entropy (1.0) far exceeds the target (−2.0) ⇒ α must shrink.
    let loss = opt
        .update_temperature_sac(1.0, 2)
        .expect("temperature update");
    let after = opt.temperature();

    assert!(
        after < before,
        "α must decrease when entropy exceeds the target: {before} -> {after}"
    );
    // dJ/dα = H − H̄ = 3 ⇒ α ← 0.2 − 0.1·3 = −0.1, clamped to the floor.
    assert!(after > 0.0, "α must stay positive, got {after}");
    assert!((loss - 0.2 * 3.0).abs() < 1e-12, "J(α) = α(H − H̄)");
}

#[test]
fn test_temperature_increases_when_entropy_below_target() {
    let cfg = ActorCriticConfig::<f64> {
        method: ActorCriticMethod::SAC,
        sac_config: SACConfig::<f64> {
            temperature: 0.2,
            auto_entropy_tuning: true,
            target_entropy: Some(1.0),
            temperature_lr: 0.05,
            ..SACConfig::default()
        },
        ..ActorCriticConfig::default()
    };
    let mut opt =
        ActorCriticOptimizer::new(cfg, MockPolicy, vec![MockQ { q: 0.0 }]).expect("build");

    let before = opt.temperature();
    let _ = opt
        .update_temperature_sac(-1.0, 2)
        .expect("temperature update");
    let after = opt.temperature();

    assert!(
        after > before,
        "α must increase when entropy is below the target: {before} -> {after}"
    );
    assert!((after - (0.2 + 0.05 * 2.0)).abs() < 1e-12);
}

// ── Gaussian Box–Muller sampling: moments converge ───────────────────

#[test]
fn test_gaussian_sampling_standard_normal_moments() {
    let opt = opt_with(vec![MockQ { q: 0.0 }]);
    let n = 2000_usize;
    let dist = ActionDistribution {
        mean: Some(Array2::zeros((n, 1))),
        std: Some(Array2::from_elem((n, 1), 1.0_f64)),
        logits: None,
        distribution_type: DistributionType::Gaussian,
    };

    let samples = opt
        .sample_actions_from_distribution(&dist)
        .expect("sample_actions");

    let vals: Vec<f64> = samples.iter().copied().collect();
    let mean = vals.iter().sum::<f64>() / n as f64;
    let var = vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    let std = var.sqrt();

    assert!(mean.abs() < 0.12, "N(0,1) mean={mean} (expected ≈0)");
    assert!((std - 1.0).abs() < 0.12, "N(0,1) std={std} (expected ≈1)");
}

#[test]
fn test_gaussian_log_prob_uses_correct_normalizer() {
    // log N(0; 0, 1) = −½ln(2π) ≈ −0.9189385. The old code used −ln(π).
    let opt = opt_with(vec![MockQ { q: 0.0 }]);
    let dist = ActionDistribution {
        mean: Some(Array2::zeros((1, 1))),
        std: Some(Array2::from_elem((1, 1), 1.0_f64)),
        logits: None,
        distribution_type: DistributionType::Gaussian,
    };
    let actions = Array2::zeros((1_usize, 1));
    let log_probs = opt
        .compute_log_probabilities(&dist, &actions)
        .expect("log probs");

    let expected = -0.5 * (2.0 * std::f64::consts::PI).ln();
    assert!(
        (log_probs[0] - expected).abs() < 1e-12,
        "log N(0;0,1) = {} but got {}",
        expected,
        log_probs[0]
    );
}

#[test]
fn test_gaussian_log_prob_survives_zero_sigma() {
    let opt = opt_with(vec![MockQ { q: 0.0 }]);
    let dist = ActionDistribution {
        mean: Some(Array2::zeros((1, 1))),
        std: Some(Array2::zeros((1, 1))),
        logits: None,
        distribution_type: DistributionType::Gaussian,
    };
    let actions = Array2::zeros((1_usize, 1));
    let log_probs = opt
        .compute_log_probabilities(&dist, &actions)
        .expect("log probs");
    assert!(
        log_probs[0].is_finite(),
        "σ = 0 must be clamped, got {}",
        log_probs[0]
    );
}

// ── Categorical inverse-CDF sampling ─────────────────────────────────

#[test]
fn test_categorical_biased_sampling() {
    let opt = opt_with(vec![MockQ { q: 0.0 }]);
    let n = 200_usize;
    let mut logits = Array2::zeros((n, 3_usize));
    for i in 0..n {
        logits[[i, 0]] = 10.0_f64;
    }
    let dist = ActionDistribution {
        mean: None,
        std: None,
        logits: Some(logits),
        distribution_type: DistributionType::Categorical,
    };

    let samples = opt
        .sample_actions_from_distribution(&dist)
        .expect("sample_actions");

    let class0_count = (0..n).filter(|&i| samples[[i, 0]] > 0.5).count();
    assert!(
        class0_count >= 185,
        "biased categorical: class-0 selected {class0_count}/200 (expected ≥185)"
    );
}

#[test]
fn test_categorical_uniform_covers_all_classes() {
    let opt = opt_with(vec![MockQ { q: 0.0 }]);
    let n = 600_usize;
    let dist = ActionDistribution {
        mean: None,
        std: None,
        logits: Some(Array2::zeros((n, 3_usize))),
        distribution_type: DistributionType::Categorical,
    };

    let samples = opt
        .sample_actions_from_distribution(&dist)
        .expect("sample_actions");

    let mut counts = [0_usize; 3];
    for i in 0..n {
        for j in 0..3 {
            if samples[[i, j]] > 0.5 {
                counts[j] += 1;
            }
        }
    }
    for (c, &cnt) in counts.iter().enumerate() {
        assert!(
            cnt >= 100,
            "uniform categorical class {c} appeared {cnt}/600 (expected ≥100)"
        );
    }
}

// ── F71 regression: prioritized replay ───────────────────────────────

fn experience(value: f64, priority: f64) -> Experience<f64> {
    Experience {
        state: Array1::from_elem(1, value),
        action: Array1::from_elem(1, 0.0),
        reward: value,
        next_state: Array1::from_elem(1, value),
        done: false,
        priority,
        info: HashMap::new(),
    }
}

#[test]
fn test_replay_buffer_rejects_zero_capacity() {
    assert!(
        ExperienceReplayBuffer::<f64>::new(0, 0.6, 0.4, false).is_err(),
        "a zero-capacity buffer must be rejected, not panic on `% 0`"
    );
}

#[test]
fn test_replay_buffer_empty_sample_errors() {
    let buffer = ExperienceReplayBuffer::<f64>::new(8, 0.6, 0.4, true).expect("buffer");
    assert!(
        buffer.sample(4).is_err(),
        "sampling an empty buffer must error, not panic inside gen_range(0..0)"
    );
}

#[test]
fn test_prioritized_sampling_favours_high_priority() {
    let mut buffer = ExperienceReplayBuffer::<f64>::new(4, 1.0, 0.4, true).expect("buffer");
    buffer.add(experience(0.0, 1.0));
    buffer.add(experience(1.0, 1.0));
    buffer.add(experience(2.0, 1.0));
    // Index 3 is 100x more likely than any other.
    buffer.add(experience(3.0, 100.0));

    let sample = buffer.sample(200).expect("sample");
    assert_eq!(
        sample.experiences.len(),
        4,
        "batch is capped at buffer size"
    );

    // Draw many batches and count how often index 3 appears.
    let mut hits = 0usize;
    let mut total = 0usize;
    for _ in 0..100 {
        let s = buffer.sample(4).expect("sample");
        for &index in &s.indices {
            total += 1;
            if index == 3 {
                hits += 1;
            }
        }
    }
    // p(3) = 100/103 ≈ 0.97; uniform would give 0.25.
    let ratio = hits as f64 / total as f64;
    assert!(
        ratio > 0.8,
        "high-priority transition drawn {ratio:.2} of the time (uniform would be 0.25)"
    );
}

#[test]
fn test_prioritized_importance_weights_are_normalized() {
    let mut buffer = ExperienceReplayBuffer::<f64>::new(4, 1.0, 1.0, true).expect("buffer");
    buffer.add(experience(0.0, 1.0));
    buffer.add(experience(1.0, 4.0));

    let sample = buffer.sample(2).expect("sample");
    assert_eq!(sample.weights.len(), sample.indices.len());
    for &w in &sample.weights {
        assert!(w > 0.0 && w <= 1.0 + 1e-12, "IS weight out of range: {w}");
    }
    let max = sample.weights.iter().cloned().fold(0.0_f64, f64::max);
    assert!((max - 1.0).abs() < 1e-9, "weights must be max-normalized");
}

#[test]
fn test_uniform_mode_returns_unit_weights() {
    let mut buffer = ExperienceReplayBuffer::<f64>::new(4, 0.6, 0.4, false).expect("buffer");
    buffer.add(experience(0.0, 1.0));
    buffer.add(experience(1.0, 50.0));

    let sample = buffer.sample(2).expect("sample");
    for &w in &sample.weights {
        assert!(
            (w - 1.0).abs() < 1e-12,
            "uniform sampling needs no correction"
        );
    }
}

#[test]
fn test_update_priorities_changes_sampling_mass() {
    let mut buffer = ExperienceReplayBuffer::<f64>::new(4, 1.0, 0.4, true).expect("buffer");
    buffer.add(experience(0.0, 1.0));
    buffer.add(experience(1.0, 1.0));

    let before = buffer.total_priority();
    buffer
        .update_priorities(&[0], &[9.0])
        .expect("priority update");
    let after = buffer.total_priority();

    assert!(
        after > before,
        "raising a TD error must raise the total priority mass: {before} -> {after}"
    );
    assert!(buffer.update_priorities(&[7], &[1.0]).is_err());
    assert!(buffer.update_priorities(&[0], &[1.0, 2.0]).is_err());
}

// ── F72 regression: Ornstein-Uhlenbeck exploration ───────────────────

#[test]
fn test_ou_noise_is_initialized_and_reverts_to_the_mean() {
    let mut opt = opt_with(vec![MockQ { q: 0.0 }]);
    // The state starts uninitialized; one step must create and advance it.
    let first = opt.update_ou_noise(2).expect("ou step");
    assert_eq!(first.len(), 2);
    assert!(
        first.iter().any(|&x| x != 0.0),
        "OU noise must actually move away from zero"
    );

    // With σ = 0 the process is pure mean reversion towards μ = 0.
    opt.config.ddpg_config.ou_noise_sigma = 0.0;
    opt.config.ddpg_config.ou_noise_theta = 0.5;
    opt.config.ddpg_config.ou_noise_dt = 1.0;
    let before: Vec<f64> = opt
        .update_ou_noise(2)
        .expect("ou step")
        .iter()
        .copied()
        .collect();
    let after: Vec<f64> = opt
        .update_ou_noise(2)
        .expect("ou step")
        .iter()
        .copied()
        .collect();
    for (b, a) in before.iter().zip(after.iter()) {
        assert!(
            a.abs() <= b.abs() + 1e-12,
            "mean reversion must shrink |x|: {b} -> {a}"
        );
    }
}

#[test]
fn test_explore_actions_adds_noise_and_respects_bounds() {
    let mut opt = opt_with(vec![MockQ { q: 0.0 }]);
    opt.config.ddpg_config.ou_noise_sigma = 5.0;
    opt.config.ddpg_config.ou_noise_dt = 1.0;
    opt.config.ddpg_config.action_bounds = Some((-1.0, 1.0));

    let states = Array2::zeros((8_usize, 3));
    let actions = opt.explore_actions(&states).expect("explore");

    assert_eq!(actions.dim(), (8, 2));
    assert!(
        actions.iter().any(|&a| a != 0.0),
        "exploration noise must reach the returned actions"
    );
    for &a in actions.iter() {
        assert!((-1.0..=1.0).contains(&a), "action {a} out of bounds");
    }
}

// ── F15 regression: target networks exist and soft-update correctly ──

#[test]
fn test_target_networks_are_populated_when_enabled() {
    let cfg = ActorCriticConfig::<f64> {
        use_target_networks: true,
        ..ActorCriticConfig::default()
    };
    let opt =
        ActorCriticOptimizer::new(cfg, MockPolicy, vec![MockQ { q: 1.0 }]).expect("construction");
    assert!(opt.target_actor.is_some(), "target actor must be created");
    assert!(
        opt.target_critics.is_some(),
        "target critics must be created"
    );
}

#[test]
fn test_soft_update_moves_target_towards_online() {
    /// Critic whose single parameter is observable, to check Polyak averaging.
    #[derive(Clone)]
    struct ParamCritic {
        w: f64,
    }
    impl ValueNetwork<f64> for ParamCritic {
        fn evaluate_value(&self, obs: &Array2<f64>) -> Result<Array1<f64>> {
            Ok(Array1::from_elem(obs.nrows(), self.w))
        }
        fn update_parameters(&mut self, d: &HashMap<String, Array1<f64>>) -> Result<()> {
            if let Some(delta) = d.get("w") {
                self.w += delta[0];
            }
            Ok(())
        }
        fn get_parameters(&self) -> HashMap<String, Array1<f64>> {
            let mut m = HashMap::new();
            m.insert("w".to_string(), Array1::from_elem(1, self.w));
            m
        }
    }

    let cfg = ActorCriticConfig::<f64> {
        use_target_networks: true,
        target_update_rate: 0.25,
        ..ActorCriticConfig::default()
    };
    let mut opt = ActorCriticOptimizer::new(cfg, MockPolicy, vec![ParamCritic { w: 4.0 }])
        .expect("construction");

    // Move the online critic, then Polyak-average the target towards it.
    opt.critics[0].w = 8.0;
    opt.soft_update_targets().expect("soft update");

    let target_w = opt
        .target_critics
        .as_ref()
        .expect("targets")
        .first()
        .expect("critic")
        .w;
    // target ← 0.25·8 + 0.75·4 = 5.0 (NOT 4 + 5 = 9, which is what passing the
    // absolute target parameters into the additive API used to produce).
    assert!(
        (target_w - 5.0).abs() < 1e-12,
        "Polyak average should be 5.0, got {target_w}"
    );
}
