// Adam optimizer with parameter group support

use crate::error::{OptimError, Result};
use crate::optimizers::Optimizer;
use crate::parameter_groups::{
    GroupManager, GroupedOptimizer, ParameterGroup, ParameterGroupConfig,
};
use scirs2_core::ndarray::{Array, Dimension, ScalarOperand};
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;

/// Adam optimizer with parameter group support
///
/// This optimizer allows different parameter groups to have different
/// hyperparameters (learning rate, weight decay, betas).
///
/// # Example
///
/// ```no_run
/// use scirs2_core::ndarray::Array1;
/// use optirs_core::optimizers::{GroupedAdam, Optimizer};
/// use optirs_core::parameter_groups::{GroupedOptimizer, ParameterGroupConfig};
///
/// // Create grouped optimizer
/// let mut optimizer = GroupedAdam::new(0.001);
///
/// // Add parameter groups with different learning rates
/// let params_fast = vec![Array1::zeros(5)];
/// let config_fast = ParameterGroupConfig::new().with_learning_rate(0.01);
/// let group_fast = optimizer.add_group(params_fast, config_fast).expect("unwrap failed");
///
/// let params_slow = vec![Array1::zeros(3)];
/// let config_slow = ParameterGroupConfig::new().with_learning_rate(0.0001);
/// let group_slow = optimizer.add_group(params_slow, config_slow).expect("unwrap failed");
///
/// // Optimize each group separately
/// let grads_fast = vec![Array1::ones(5)];
/// let updated_fast = optimizer.step_group(group_fast, &grads_fast).expect("unwrap failed");
///
/// let grads_slow = vec![Array1::ones(3)];
/// let updated_slow = optimizer.step_group(group_slow, &grads_slow).expect("unwrap failed");
/// ```
#[derive(Debug)]
pub struct GroupedAdam<A: Float + Send + Sync, D: Dimension> {
    /// Default learning rate
    defaultlr: A,
    /// Default beta1
    default_beta1: A,
    /// Default beta2
    default_beta2: A,
    /// Default weight decay
    default_weight_decay: A,
    /// Epsilon to prevent division by zero
    epsilon: A,
    /// AMSGrad flag
    amsgrad: bool,
    /// Parameter groups
    group_manager: GroupManager<A, D>,
    /// Global step counter (total number of group updates performed)
    step: usize,
    /// Per-group step counters, used for per-group bias correction
    group_steps: HashMap<usize, usize>,
    /// Group used by the plain [`Optimizer::step`] entry point
    ///
    /// Cached so that repeated `step` calls reuse one group instead of appending a
    /// fresh (state-free) group on every call.
    implicit_group: Option<usize>,
}

impl<A: Float + ScalarOperand + Debug + Send + Sync, D: Dimension + Send + Sync> GroupedAdam<A, D> {
    /// Create a new grouped Adam optimizer
    pub fn new(defaultlr: A) -> Self {
        Self {
            defaultlr,
            default_beta1: A::from(0.9).expect("unwrap failed"),
            default_beta2: A::from(0.999).expect("unwrap failed"),
            default_weight_decay: A::zero(),
            epsilon: A::from(1e-8).expect("unwrap failed"),
            amsgrad: false,
            group_manager: GroupManager::new(),
            step: 0,
            group_steps: HashMap::new(),
            implicit_group: None,
        }
    }

    /// Returns the number of update steps applied to `groupid`
    pub fn group_step_count(&self, groupid: usize) -> usize {
        self.group_steps.get(&groupid).copied().unwrap_or(0)
    }

    /// Removes every parameter group and all associated state
    pub fn clear_groups(&mut self) {
        self.group_manager = GroupManager::new();
        self.group_steps.clear();
        self.implicit_group = None;
        self.step = 0;
    }

    /// Set default beta1
    pub fn with_beta1(mut self, beta1: A) -> Self {
        self.default_beta1 = beta1;
        self
    }

    /// Set default beta2
    pub fn with_beta2(mut self, beta2: A) -> Self {
        self.default_beta2 = beta2;
        self
    }

    /// Set default weight decay
    pub fn with_weight_decay(mut self, weight_decay: A) -> Self {
        self.default_weight_decay = weight_decay;
        self
    }

    /// Enable AMSGrad
    pub fn with_amsgrad(mut self) -> Self {
        self.amsgrad = true;
        self
    }

    /// Initialize state for a group
    fn init_group_state(&mut self, groupid: usize) -> Result<()> {
        let group = self.group_manager.get_group_mut(groupid)?;

        if group.state.is_empty() {
            let mut m_t = Vec::new();
            let mut v_t = Vec::new();
            let mut v_hat_max = Vec::new();

            for param in &group.params {
                m_t.push(Array::zeros(param.raw_dim()));
                v_t.push(Array::zeros(param.raw_dim()));
                if self.amsgrad {
                    v_hat_max.push(Array::zeros(param.raw_dim()));
                }
            }

            group.state.insert("m_t".to_string(), m_t);
            group.state.insert("v_t".to_string(), v_t);
            if self.amsgrad {
                group.state.insert("v_hat_max".to_string(), v_hat_max);
            }
        }

        Ok(())
    }

    /// Step for a specific group
    ///
    /// `group_step` is the 1-based update count of this group and drives bias
    /// correction, so each group is bias-corrected against its own history.
    fn step_group_internal(
        &mut self,
        groupid: usize,
        group_step: usize,
        gradients: &[Array<A, D>],
    ) -> Result<Vec<Array<A, D>>> {
        let t = i32::try_from(group_step).map_err(|_| {
            OptimError::InvalidConfig(
                "Timestep too large for bias correction calculation".to_string(),
            )
        })?;

        // Initialize state if needed
        self.init_group_state(groupid)?;

        let group = self.group_manager.get_group_mut(groupid)?;

        if gradients.len() != group.params.len() {
            return Err(OptimError::InvalidConfig(format!(
                "Number of gradients ({}) doesn't match number of parameters ({})",
                gradients.len(),
                group.params.len()
            )));
        }

        // Get hyperparameters for this group
        let lr = group.learning_rate(self.defaultlr);
        let beta1 = group.get_custom_param("beta1", self.default_beta1);
        let beta2 = group.get_custom_param("beta2", self.default_beta2);
        let weightdecay = group.weight_decay(self.default_weight_decay);

        let mut updated_params = Vec::new();

        // Process each parameter
        for i in 0..group.params.len() {
            let param = &group.params[i];
            let grad = &gradients[i];

            // Apply weight decay
            let grad_with_decay = if weightdecay > A::zero() {
                grad + &(param * weightdecay)
            } else {
                grad.clone()
            };

            // Update states and compute new parameters
            let updated = {
                // Update first moment
                let m_t = group.state.get_mut("m_t").ok_or_else(|| {
                    OptimError::InvalidConfig("missing 'm_t' state for group".to_string())
                })?;
                m_t[i] = &m_t[i] * beta1 + &grad_with_decay * (A::one() - beta1);
                let m_hat = &m_t[i] / (A::one() - beta1.powi(t));

                // Update second moment
                let v_t = group.state.get_mut("v_t").ok_or_else(|| {
                    OptimError::InvalidConfig("missing 'v_t' state for group".to_string())
                })?;
                v_t[i] = &v_t[i] * beta2 + &grad_with_decay * &grad_with_decay * (A::one() - beta2);
                let v_hat = &v_t[i] / (A::one() - beta2.powi(t));

                // Update parameters
                if self.amsgrad {
                    let v_hat_max = group.state.get_mut("v_hat_max").ok_or_else(|| {
                        OptimError::InvalidConfig("missing 'v_hat_max' state for group".to_string())
                    })?;
                    v_hat_max[i].zip_mut_with(&v_hat, |a, &b| *a = a.max(b));
                    param - &(&m_hat * lr / (&v_hat_max[i].mapv(|x| x.sqrt()) + self.epsilon))
                } else {
                    param - &(&m_hat * lr / (&v_hat.mapv(|x| x.sqrt()) + self.epsilon))
                }
            };

            updated_params.push(updated);
        }

        // Update group parameters
        group.params = updated_params.clone();

        Ok(updated_params)
    }
}

impl<A: Float + ScalarOperand + Debug + Send + Sync, D: Dimension + Send + Sync>
    GroupedOptimizer<A, D> for GroupedAdam<A, D>
{
    fn add_group(
        &mut self,
        params: Vec<Array<A, D>>,
        config: ParameterGroupConfig<A>,
    ) -> Result<usize> {
        Ok(self.group_manager.add_group(params, config))
    }

    fn get_group(&self, groupid: usize) -> Result<&ParameterGroup<A, D>> {
        self.group_manager.get_group(groupid)
    }

    fn get_group_mut(&mut self, groupid: usize) -> Result<&mut ParameterGroup<A, D>> {
        self.group_manager.get_group_mut(groupid)
    }

    fn groups(&self) -> &[ParameterGroup<A, D>] {
        self.group_manager.groups()
    }

    fn groups_mut(&mut self) -> &mut [ParameterGroup<A, D>] {
        self.group_manager.groups_mut()
    }

    fn step_group(
        &mut self,
        groupid: usize,
        gradients: &[Array<A, D>],
    ) -> Result<Vec<Array<A, D>>> {
        self.step = self.step.saturating_add(1);
        let group_step = {
            let counter = self.group_steps.entry(groupid).or_insert(0);
            *counter = counter.saturating_add(1);
            *counter
        };
        self.step_group_internal(groupid, group_step, gradients)
    }

    fn set_group_learning_rate(&mut self, groupid: usize, lr: A) -> Result<()> {
        let group = self.group_manager.get_group_mut(groupid)?;
        group.config.learning_rate = Some(lr);
        Ok(())
    }

    fn set_group_weight_decay(&mut self, groupid: usize, wd: A) -> Result<()> {
        let group = self.group_manager.get_group_mut(groupid)?;
        group.config.weight_decay = Some(wd);
        Ok(())
    }
}

// Standard optimizer implementation for default behavior
impl<A: Float + ScalarOperand + Debug + Send + Sync, D: Dimension + Send + Sync> Optimizer<A, D>
    for GroupedAdam<A, D>
{
    fn step(&mut self, params: &Array<A, D>, gradients: &Array<A, D>) -> Result<Array<A, D>> {
        if params.shape() != gradients.shape() {
            return Err(OptimError::DimensionMismatch(format!(
                "Incompatible shapes: parameters have shape {:?}, gradients have shape {:?}",
                params.shape(),
                gradients.shape()
            )));
        }

        // Reuse a single implicit group across calls. Creating a new group on every
        // call would reset the moment estimates, grow the group list without bound and
        // make each step cost O(number of previous steps).
        let reusable = self
            .implicit_group
            .filter(|id| self.group_manager.get_group(*id).is_ok());

        let groupid = if let Some(id) = reusable {
            let group = self.group_manager.get_group_mut(id)?;
            let shape_changed =
                group.params.len() != 1 || group.params[0].raw_dim() != params.raw_dim();
            if shape_changed {
                // A different parameter shape means the cached moments are meaningless.
                group.params = vec![params.clone()];
                group.state.clear();
                self.group_steps.insert(id, 0);
            } else {
                group.params[0] = params.clone();
            }
            id
        } else {
            let id = self.add_group(vec![params.clone()], ParameterGroupConfig::new())?;
            self.implicit_group = Some(id);
            self.group_steps.insert(id, 0);
            id
        };

        let result = self.step_group(groupid, std::slice::from_ref(gradients))?;

        result.into_iter().next().ok_or_else(|| {
            OptimError::InvalidConfig("grouped Adam step produced no parameters".to_string())
        })
    }

    fn get_learning_rate(&self) -> A {
        self.defaultlr
    }

    fn set_learning_rate(&mut self, learning_rate: A) {
        self.defaultlr = learning_rate;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_grouped_adam_creation() {
        let optimizer: GroupedAdam<f64, scirs2_core::ndarray::Ix1> = GroupedAdam::new(0.001);
        assert_eq!(optimizer.defaultlr, 0.001);
        assert_eq!(optimizer.default_beta1, 0.9);
        assert_eq!(optimizer.default_beta2, 0.999);
    }

    #[test]
    fn test_grouped_adam_multiple_groups() {
        let mut optimizer = GroupedAdam::new(0.001);

        // Add first group with high learning rate
        let params1 = vec![Array1::from_vec(vec![1.0, 2.0])];
        let config1 = ParameterGroupConfig::new().with_learning_rate(0.01);
        let group1 = optimizer
            .add_group(params1, config1)
            .expect("unwrap failed");

        // Add second group with low learning rate
        let params2 = vec![Array1::from_vec(vec![3.0, 4.0, 5.0])];
        let config2 = ParameterGroupConfig::new().with_learning_rate(0.0001);
        let group2 = optimizer
            .add_group(params2, config2)
            .expect("unwrap failed");

        // Update first group
        let grads1 = vec![Array1::from_vec(vec![0.1, 0.2])];
        let updated1 = optimizer
            .step_group(group1, &grads1)
            .expect("unwrap failed");

        // Update second group
        let grads2 = vec![Array1::from_vec(vec![0.3, 0.4, 0.5])];
        let updated2 = optimizer
            .step_group(group2, &grads2)
            .expect("unwrap failed");

        // Verify different updates due to different learning rates
        assert!(updated1[0][0] < 1.0); // Should decrease more
        assert!(updated2[0][0] > 2.9); // Should decrease less
    }

    #[test]
    fn test_grouped_adam_custom_betas() {
        let mut optimizer = GroupedAdam::new(0.001);

        // Add group with custom betas
        let params = vec![Array1::from_vec(vec![1.0, 2.0])];
        let config = ParameterGroupConfig::new()
            .with_custom_param("beta1".to_string(), 0.8)
            .with_custom_param("beta2".to_string(), 0.99);
        let group = optimizer.add_group(params, config).expect("unwrap failed");

        // Verify custom parameters are used
        let group_ref = optimizer.get_group(group).expect("unwrap failed");
        assert_eq!(group_ref.get_custom_param("beta1", 0.0), 0.8);
        assert_eq!(group_ref.get_custom_param("beta2", 0.0), 0.99);
    }

    #[test]
    fn test_grouped_adam_clear() {
        let mut optimizer = GroupedAdam::new(0.001);

        // Add groups
        let params1 = vec![Array1::zeros(2)];
        let config1 = ParameterGroupConfig::new();
        optimizer
            .add_group(params1, config1)
            .expect("unwrap failed");

        assert_eq!(optimizer.groups().len(), 1);

        // Clear groups
        optimizer.clear_groups();

        assert_eq!(optimizer.groups().len(), 0);
        assert_eq!(optimizer.step, 0);
    }

    /// Regression test: `Optimizer::step` used to append a brand new parameter group on
    /// every call, which reset the Adam moments, leaked memory without bound and made
    /// each step cost O(previous steps).
    #[test]
    fn test_grouped_adam_step_reuses_single_group() {
        let mut optimizer: GroupedAdam<f64, scirs2_core::ndarray::Ix1> = GroupedAdam::new(0.1);

        let mut params = Array1::from_vec(vec![0.0f64]);
        let gradients = Array1::from_vec(vec![1.0f64]);

        for _ in 0..100 {
            params = optimizer
                .step(&params, &gradients)
                .expect("step should succeed");
        }

        // Exactly one implicit group, not one per step.
        assert_eq!(optimizer.groups().len(), 1);
        assert_eq!(optimizer.group_step_count(0), 100);

        // The moments must have been carried across the 100 steps.
        let group = optimizer.get_group(0).expect("implicit group must exist");
        let m_t = group
            .state
            .get("m_t")
            .expect("first moment state must exist");
        // m converges to the (constant) gradient value 1.0
        assert!(
            (m_t[0][0] - 1.0).abs() < 1e-3,
            "first moment did not accumulate: {}",
            m_t[0][0]
        );
    }

    /// The first step of a group must use t = 1 (bias correction), not t = 2.
    #[test]
    fn test_grouped_adam_first_step_uses_t_one() {
        let mut optimizer: GroupedAdam<f64, scirs2_core::ndarray::Ix1> = GroupedAdam::new(0.1);

        let params = Array1::from_vec(vec![0.0f64]);
        let gradients = Array1::from_vec(vec![1.0f64]);

        let updated = optimizer
            .step(&params, &gradients)
            .expect("step should succeed");

        // At t = 1 with a unit gradient Adam moves exactly -lr.
        assert!(
            (updated[0] + 0.1).abs() < 1e-9,
            "expected -0.1, got {}",
            updated[0]
        );
    }

    /// Two explicit groups must keep independent bias-correction clocks.
    #[test]
    fn test_grouped_adam_per_group_step_counters() {
        let mut optimizer = GroupedAdam::new(0.1);

        let group_a = optimizer
            .add_group(
                vec![Array1::from_vec(vec![0.0f64])],
                ParameterGroupConfig::new(),
            )
            .expect("add group a");
        let group_b = optimizer
            .add_group(
                vec![Array1::from_vec(vec![0.0f64])],
                ParameterGroupConfig::new(),
            )
            .expect("add group b");

        let grads = vec![Array1::from_vec(vec![1.0f64])];

        // Step group A five times before group B is touched at all.
        for _ in 0..5 {
            optimizer.step_group(group_a, &grads).expect("step group a");
        }

        let first_b = optimizer.step_group(group_b, &grads).expect("step group b");

        assert_eq!(optimizer.group_step_count(group_a), 5);
        assert_eq!(optimizer.group_step_count(group_b), 1);

        // Group B is on its own first step, so it must move exactly -lr.
        assert!(
            (first_b[0][0] + 0.1).abs() < 1e-9,
            "group B was contaminated by group A's clock: {}",
            first_b[0][0]
        );
    }
}
