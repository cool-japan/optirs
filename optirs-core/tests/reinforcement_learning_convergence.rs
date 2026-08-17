//! End-to-end convergence tests for the reinforcement-learning optimizers.
//!
//! The in-module unit tests pin the *math* of each update rule (gradient signs,
//! Fisher factors, KL adaptation, replay sampling, …) against finite differences
//! and hand-computed values. This file closes the remaining gap: it drives a real
//! analytic linear policy through many optimizer updates on a tiny reward
//! landscape and asserts that the **policy actually learns** — the average reward
//! climbs from chance to near-optimal.
//!
//! These are the "linear-policy convergence tests" referenced from
//! `policy_gradient.rs`. They only pass when the whole loop is genuine: the score
//! oracle, the advantage/return computation, the learning-rate application, the
//! descent sign and the parameter update all have to be correct simultaneously.
//! Any one of the bugs the unit tests guard (constant gradient, inverted sign,
//! learning rate ignored, advantages never computed) flattens the reward curve
//! and fails these tests.

use scirs2_core::ndarray::{Array1, Array2};

use optirs_core::reinforcement_learning::{
    LinearSoftmaxPolicy, LinearValueFunction, PolicyGradientConfig, PolicyGradientMethod,
    PolicyGradientOptimizer, RLOptimizerConfig, TrajectoryBatch,
};

/// Minimal xorshift64 PRNG: deterministic (so the tests are reproducible) yet
/// statistically independent draws, with no external RNG dependency.
struct Xorshift64(u64);

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        // xorshift64 requires a nonzero seed.
        Self(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform in the open-ish interval (0, 1).
    fn uniform(&mut self) -> f64 {
        // 53-bit mantissa mapped to [0, 1), then nudged off the endpoints.
        let bits = self.next_u64() >> 11;
        let u = (bits as f64) / ((1u64 << 53) as f64);
        u.clamp(1e-12, 1.0 - 1e-12)
    }
}

/// A two-state contextual bandit.
///
/// * State 0 is the one-hot feature `[1, 0]`, state 1 is `[0, 1]`.
/// * The optimal action is **state dependent**: action 0 in state 0, action 1 in
///   state 1. A linear softmax over the one-hot features can represent this
///   exactly, so a converging optimizer must reach it.
/// * Reward is `+1` for the correct action and `-1` otherwise. The `±1` shaping
///   makes the reward zero-mean under a uniform policy, so even baseline-free
///   REINFORCE has an unbiased gradient that points the right way.
struct Bandit {
    n_features: usize,
    n_actions: usize,
}

impl Bandit {
    fn new() -> Self {
        Self {
            n_features: 2,
            n_actions: 2,
        }
    }

    fn correct_action(state: usize) -> usize {
        state
    }

    fn reward(state: usize, action: usize) -> f64 {
        if action == Self::correct_action(state) {
            1.0
        } else {
            -1.0
        }
    }

    /// One-hot feature row for a state.
    fn features(&self, state: usize) -> Vec<f64> {
        let mut row = vec![0.0; self.n_features];
        row[state] = 1.0;
        row
    }

    /// Build a batch of `batch` single-step (terminal) transitions by rolling out
    /// the current policy. Alternating the two states keeps the batch balanced.
    ///
    /// Returns the trajectory and the batch's mean reward (the learning signal we
    /// track for convergence).
    fn rollout(
        &self,
        policy: &LinearSoftmaxPolicy<f64>,
        batch: usize,
        rng: &mut Xorshift64,
    ) -> (TrajectoryBatch<f64>, f64) {
        let mut observations = Array2::<f64>::zeros((batch, self.n_features));
        let states: Vec<usize> = (0..batch).map(|i| i % 2).collect();
        for (i, &s) in states.iter().enumerate() {
            for (f, v) in self.features(s).into_iter().enumerate() {
                observations[[i, f]] = v;
            }
        }

        // Sample one action per row from the current policy.
        let (actions, log_probs, indices) = policy
            .sample_actions_with(&observations, || rng.uniform())
            .expect("sampling from the linear softmax policy");

        // Terminal single-step transitions: each return is exactly its reward.
        let mut rewards = Array1::<f64>::zeros(batch);
        let mut total = 0.0;
        for i in 0..batch {
            let r = Self::reward(states[i], indices[i]);
            rewards[i] = r;
            total += r;
        }

        let values = Array1::<f64>::zeros(batch);
        let dones = Array1::from_vec(vec![true; batch]);

        let trajectory =
            TrajectoryBatch::new(observations, actions, log_probs, rewards, values, dones)
                .expect("valid trajectory batch");

        (trajectory, total / batch as f64)
    }

    /// Probability the policy assigns to the correct action in each of the two
    /// states.
    fn correct_action_probabilities(&self, policy: &LinearSoftmaxPolicy<f64>) -> Vec<f64> {
        (0..2)
            .map(|state| {
                let obs = Array2::from_shape_vec((1, self.n_features), self.features(state))
                    .expect("feature row");
                let probs = policy.probabilities(&obs).expect("probabilities");
                probs[[0, Self::correct_action(state)]]
            })
            .collect()
    }
}

/// Config helper: schedulers disabled (so `policy_lr`/`value_lr` come straight
/// from the base config), gradient clipping disabled, and — for baseline-free
/// REINFORCE — entropy regularization off so the policy can commit.
fn base_config(policy_lr: f64, value_lr: f64, entropy_coeff: f64) -> RLOptimizerConfig<f64> {
    RLOptimizerConfig {
        policy_lr,
        value_lr,
        entropy_coeff,
        // Disable global-norm clipping (max_norm <= 0 is a no-op) so the raw
        // analytic gradient drives learning.
        max_grad_norm: 0.0,
        ..RLOptimizerConfig::default()
    }
}

#[test]
fn reinforce_softmax_policy_improves_bandit_reward() {
    let bandit = Bandit::new();
    let policy = LinearSoftmaxPolicy::<f64>::new(bandit.n_actions, bandit.n_features)
        .expect("softmax policy");

    let config = PolicyGradientConfig::<f64> {
        base_config: base_config(0.5, 1e-3, 0.0),
        method: PolicyGradientMethod::Reinforce,
        // Schedulers would otherwise pin the LR at the 3e-4 default.
        policy_scheduler: None,
        value_scheduler: None,
        use_baseline: false,
        ..PolicyGradientConfig::default()
    };

    // Baseline-free REINFORCE: no value network.
    let mut optimizer =
        PolicyGradientOptimizer::<f64, _, LinearValueFunction<f64>>::new(config, policy, None);

    let mut rng = Xorshift64::new(0x5eed_1234);
    let iterations = 300;
    let batch = 64;

    let mut first_reward = None;
    let mut recent_rewards = Vec::new();

    for it in 0..iterations {
        // Roll out with the *current* policy, then update it.
        let (trajectory, mean_reward) = bandit.rollout(optimizer.policy_network(), batch, &mut rng);
        if first_reward.is_none() {
            first_reward = Some(mean_reward);
        }
        if it >= iterations - 20 {
            recent_rewards.push(mean_reward);
        }
        optimizer.update(trajectory).expect("REINFORCE update");
    }

    let first_reward = first_reward.expect("at least one iteration");
    let final_reward = recent_rewards.iter().sum::<f64>() / recent_rewards.len() as f64;

    // The starting policy is uniform, so the initial mean reward is ≈ 0.
    assert!(
        first_reward.abs() < 0.35,
        "uniform policy should start near zero reward, got {first_reward}"
    );
    // After training the policy must reliably pick the correct action.
    assert!(
        final_reward > 0.8,
        "REINFORCE failed to improve reward: first={first_reward}, final={final_reward}"
    );
    assert!(
        final_reward > first_reward + 0.5,
        "reward must climb substantially: first={first_reward}, final={final_reward}"
    );

    // And the learned policy must be confidently correct in *both* states —
    // proof that the state-dependent optimum was actually represented and found.
    let probs = bandit.correct_action_probabilities(optimizer.policy_network());
    for (state, p) in probs.iter().enumerate() {
        assert!(
            *p > 0.9,
            "correct-action probability in state {state} is only {p} (expected > 0.9)"
        );
    }
}

#[test]
fn a2c_softmax_with_value_baseline_improves_reward() {
    let bandit = Bandit::new();
    let policy = LinearSoftmaxPolicy::<f64>::new(bandit.n_actions, bandit.n_features)
        .expect("softmax policy");
    let value = LinearValueFunction::<f64>::new(bandit.n_features).expect("value function");

    let config = PolicyGradientConfig::<f64> {
        // A small entropy bonus is harmless here and exercises the entropy oracle.
        base_config: base_config(0.3, 0.2, 0.001),
        method: PolicyGradientMethod::ActorCritic,
        policy_scheduler: None,
        value_scheduler: None,
        use_baseline: true,
        ..PolicyGradientConfig::default()
    };

    let mut optimizer = PolicyGradientOptimizer::<f64, _, _>::new(config, policy, Some(value));

    let mut rng = Xorshift64::new(0xa2c_9001);
    let iterations = 400;
    let batch = 64;

    let mut first_reward = None;
    let mut recent_rewards = Vec::new();

    for it in 0..iterations {
        let (trajectory, mean_reward) = bandit.rollout(optimizer.policy_network(), batch, &mut rng);
        if first_reward.is_none() {
            first_reward = Some(mean_reward);
        }
        if it >= iterations - 30 {
            recent_rewards.push(mean_reward);
        }
        optimizer.update(trajectory).expect("A2C update");
    }

    let first_reward = first_reward.expect("at least one iteration");
    let final_reward = recent_rewards.iter().sum::<f64>() / recent_rewards.len() as f64;

    // A2C normalizes its advantages, so learning is a little noisier than raw
    // REINFORCE; the threshold is looser but the improvement must be real.
    assert!(
        final_reward > 0.5,
        "A2C failed to improve reward: first={first_reward}, final={final_reward}"
    );
    assert!(
        final_reward > first_reward + 0.4,
        "reward must climb substantially: first={first_reward}, final={final_reward}"
    );

    let probs = bandit.correct_action_probabilities(optimizer.policy_network());
    for (state, p) in probs.iter().enumerate() {
        assert!(
            *p > 0.75,
            "correct-action probability in state {state} is only {p} (expected > 0.75)"
        );
    }
}
