// FedProx (Federated Proximal) optimizer implementation
//
// FedProx extends FedAvg by adding a proximal term to the local objective function,
// which helps handle systems heterogeneity (stragglers) and statistical heterogeneity
// (non-IID data) in federated learning.
//
// Reference: Li et al., "Federated Optimization in Heterogeneous Networks" (MLSys 2020)

use crate::error::{OptimError, Result};
use scirs2_core::ndarray::{Array, Array1, Dimension, ScalarOperand, Zip};
use scirs2_core::numeric::Float;
use std::fmt::Debug;

/// Configuration for the FedProx optimizer
#[derive(Debug, Clone)]
pub struct FedProxConfig<A: Float> {
    /// Proximal term coefficient (mu).
    /// Controls the strength of the proximal regularization.
    /// When mu=0, FedProx degenerates to FedAvg.
    pub mu: A,
    /// Number of local training epochs per communication round
    pub local_epochs: usize,
    /// Fraction of clients participating in each round (0.0, 1.0]
    pub participation_rate: A,
    /// Total number of clients in the federation
    pub num_clients: usize,
}

impl<A: Float + ScalarOperand + Debug + Send + Sync + 'static> FedProxConfig<A> {
    /// Create a new FedProxConfig with default values
    pub fn new(num_clients: usize) -> Self {
        Self {
            mu: A::from(0.01).unwrap_or_else(|| A::zero()),
            local_epochs: 5,
            participation_rate: A::one(),
            num_clients,
        }
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        if self.mu < A::zero() {
            return Err(OptimError::InvalidConfig(
                "Proximal term coefficient mu must be non-negative".to_string(),
            ));
        }
        if self.local_epochs == 0 {
            return Err(OptimError::InvalidConfig(
                "local_epochs must be at least 1".to_string(),
            ));
        }
        if self.participation_rate <= A::zero() || self.participation_rate > A::one() {
            return Err(OptimError::InvalidConfig(
                "participation_rate must be in (0.0, 1.0]".to_string(),
            ));
        }
        if self.num_clients == 0 {
            return Err(OptimError::InvalidConfig(
                "num_clients must be at least 1".to_string(),
            ));
        }
        Ok(())
    }
}

/// Builder for FedProxConfig
#[derive(Debug)]
pub struct FedProxConfigBuilder<A: Float> {
    mu: Option<A>,
    local_epochs: Option<usize>,
    participation_rate: Option<A>,
    num_clients: usize,
}

impl<A: Float + ScalarOperand + Debug + Send + Sync + 'static> FedProxConfigBuilder<A> {
    /// Create a new builder with the required number of clients
    pub fn new(num_clients: usize) -> Self {
        Self {
            mu: None,
            local_epochs: None,
            participation_rate: None,
            num_clients,
        }
    }

    /// Set the proximal term coefficient
    pub fn mu(mut self, mu: A) -> Self {
        self.mu = Some(mu);
        self
    }

    /// Set the number of local training epochs
    pub fn local_epochs(mut self, epochs: usize) -> Self {
        self.local_epochs = Some(epochs);
        self
    }

    /// Set the client participation rate
    pub fn participation_rate(mut self, rate: A) -> Self {
        self.participation_rate = Some(rate);
        self
    }

    /// Build the configuration, validating all parameters
    pub fn build(self) -> Result<FedProxConfig<A>> {
        let config = FedProxConfig {
            mu: self
                .mu
                .unwrap_or_else(|| A::from(0.01).unwrap_or_else(|| A::zero())),
            local_epochs: self.local_epochs.unwrap_or(5),
            participation_rate: self.participation_rate.unwrap_or_else(|| A::one()),
            num_clients: self.num_clients,
        };
        config.validate()?;
        Ok(config)
    }
}

/// A client's update submission containing updated parameters and metadata
#[derive(Debug, Clone)]
pub struct ClientUpdate<A: Float, D: Dimension> {
    /// Unique identifier for the client
    pub client_id: usize,
    /// Updated model parameters after local training
    pub parameters: Vec<Array<A, D>>,
    /// Size of the client's local dataset (used for weighted aggregation)
    pub data_size: usize,
}

/// FedProx optimizer for federated learning
///
/// FedProx adds a proximal term mu/2 * ||w - w_global||^2 to each client's
/// local objective, producing a gradient correction of mu * (w - w_global).
/// This encourages local models to stay close to the global model, improving
/// convergence under heterogeneous conditions.
#[derive(Debug)]
pub struct FedProxOptimizer<
    A: Float + ScalarOperand + Debug + Send + Sync + 'static,
    D: Dimension + Send + Sync + Clone,
> {
    /// Configuration
    config: FedProxConfig<A>,
    /// Stored global model parameters (the "anchor" for proximal term)
    global_parameters: Option<Vec<Array<A, D>>>,
    /// Collected client updates for the current round
    client_updates: Vec<ClientUpdate<A, D>>,
    /// Number of completed communication rounds
    round_count: usize,
}

impl<
        A: Float + ScalarOperand + Debug + Send + Sync + 'static,
        D: Dimension + Send + Sync + Clone,
    > FedProxOptimizer<A, D>
{
    /// Create a new FedProxOptimizer with the given configuration
    pub fn new(config: FedProxConfig<A>) -> Self {
        Self {
            config,
            global_parameters: None,
            client_updates: Vec::new(),
            round_count: 0,
        }
    }

    /// Convenience method to create a builder
    pub fn builder(num_clients: usize) -> FedProxConfigBuilder<A> {
        FedProxConfigBuilder::new(num_clients)
    }

    /// Store the current global model parameters
    ///
    /// This must be called before performing local updates, as the proximal
    /// term references the global parameters.
    pub fn set_global_parameters(&mut self, params: &[Array<A, D>]) -> Result<()> {
        if params.is_empty() {
            return Err(OptimError::InvalidParameter(
                "Global parameters cannot be empty".to_string(),
            ));
        }
        self.global_parameters = Some(params.to_vec());
        // Clear any previous client updates when starting a new round
        self.client_updates.clear();
        Ok(())
    }

    /// Perform a local update step with the FedProx proximal gradient
    ///
    /// Computes: w_new = w - lr * (gradient + mu * (w - w_global))
    ///
    /// When mu=0, this reduces to standard SGD: w_new = w - lr * gradient
    pub fn local_update(
        &self,
        params: &[Array<A, D>],
        gradients: &[Array<A, D>],
        lr: A,
    ) -> Result<Vec<Array<A, D>>> {
        if params.len() != gradients.len() {
            return Err(OptimError::DimensionMismatch(format!(
                "Parameters length ({}) does not match gradients length ({})",
                params.len(),
                gradients.len()
            )));
        }

        let proximal_grads = self.compute_proximal_gradient(params)?;

        let mut updated = Vec::with_capacity(params.len());
        for (i, (param, grad)) in params.iter().zip(gradients.iter()).enumerate() {
            if param.shape() != grad.shape() {
                return Err(OptimError::DimensionMismatch(format!(
                    "Parameter shape {:?} does not match gradient shape {:?} at index {}",
                    param.shape(),
                    grad.shape(),
                    i
                )));
            }

            let prox = &proximal_grads[i];
            // w_new = w - lr * (gradient + proximal_gradient)
            let mut new_param = param.clone();
            Zip::from(&mut new_param)
                .and(grad)
                .and(prox)
                .for_each(|w, &g, &p| {
                    *w = *w - lr * (g + p);
                });
            updated.push(new_param);
        }

        Ok(updated)
    }

    /// Compute the proximal gradient: mu * (params - global_params)
    ///
    /// This gradient term pulls local parameters towards the global model,
    /// preventing excessive divergence during local training.
    pub fn compute_proximal_gradient(&self, params: &[Array<A, D>]) -> Result<Vec<Array<A, D>>> {
        let global = self.global_parameters.as_ref().ok_or_else(|| {
            OptimError::InvalidState(
                "Global parameters not set. Call set_global_parameters first.".to_string(),
            )
        })?;

        if params.len() != global.len() {
            return Err(OptimError::DimensionMismatch(format!(
                "Local parameters length ({}) does not match global parameters length ({})",
                params.len(),
                global.len()
            )));
        }

        let mu = self.config.mu;
        let mut prox_grads = Vec::with_capacity(params.len());

        for (i, (local, global_p)) in params.iter().zip(global.iter()).enumerate() {
            if local.shape() != global_p.shape() {
                return Err(OptimError::DimensionMismatch(format!(
                    "Local param shape {:?} != global param shape {:?} at index {}",
                    local.shape(),
                    global_p.shape(),
                    i
                )));
            }

            let mut prox = local.clone();
            Zip::from(&mut prox).and(global_p).for_each(|l, &g| {
                *l = mu * (*l - g);
            });
            prox_grads.push(prox);
        }

        Ok(prox_grads)
    }

    /// Submit a client's updated parameters after local training
    pub fn submit_client_update(
        &mut self,
        client_id: usize,
        params: &[Array<A, D>],
        data_size: usize,
    ) -> Result<()> {
        if params.is_empty() {
            return Err(OptimError::InvalidParameter(
                "Client parameters cannot be empty".to_string(),
            ));
        }
        if data_size == 0 {
            return Err(OptimError::InvalidParameter(
                "Client data_size must be positive".to_string(),
            ));
        }

        // Validate against global parameters shape if available
        if let Some(ref global) = self.global_parameters {
            if params.len() != global.len() {
                return Err(OptimError::DimensionMismatch(format!(
                    "Client {} parameter count ({}) does not match global ({})",
                    client_id,
                    params.len(),
                    global.len()
                )));
            }
            for (i, (cp, gp)) in params.iter().zip(global.iter()).enumerate() {
                if cp.shape() != gp.shape() {
                    return Err(OptimError::DimensionMismatch(format!(
                        "Client {} param shape {:?} != global shape {:?} at index {}",
                        client_id,
                        cp.shape(),
                        gp.shape(),
                        i
                    )));
                }
            }
        }

        self.client_updates.push(ClientUpdate {
            client_id,
            parameters: params.to_vec(),
            data_size,
        });

        Ok(())
    }

    /// Aggregate client updates using weighted averaging (FedAvg-style)
    ///
    /// Each client's contribution is weighted by its data_size relative to
    /// the total data across all participating clients. This produces a new
    /// global model for the next communication round.
    pub fn aggregate_updates(&mut self) -> Result<Vec<Array<A, D>>> {
        if self.client_updates.is_empty() {
            return Err(OptimError::InvalidState(
                "No client updates to aggregate".to_string(),
            ));
        }

        // Compute total data size across all clients
        let total_data: usize = self.client_updates.iter().map(|u| u.data_size).sum();
        if total_data == 0 {
            return Err(OptimError::InvalidState(
                "Total data size across clients is zero".to_string(),
            ));
        }
        let total_data_a = A::from(total_data).ok_or_else(|| {
            OptimError::ComputationError("Cannot convert total data size to float".to_string())
        })?;

        // Determine number of parameter tensors from first client
        let num_params = self.client_updates[0].parameters.len();

        // Initialize aggregated parameters with zeros (same shape as first client)
        let mut aggregated: Vec<Array<A, D>> = self.client_updates[0]
            .parameters
            .iter()
            .map(|p| Array::zeros(p.raw_dim()))
            .collect();

        // Weighted sum
        for update in &self.client_updates {
            if update.parameters.len() != num_params {
                return Err(OptimError::DimensionMismatch(format!(
                    "Client {} has {} parameters, expected {}",
                    update.client_id,
                    update.parameters.len(),
                    num_params
                )));
            }

            let weight = A::from(update.data_size).ok_or_else(|| {
                OptimError::ComputationError("Cannot convert client data size to float".to_string())
            })? / total_data_a;

            for (agg, client_param) in aggregated.iter_mut().zip(update.parameters.iter()) {
                Zip::from(agg).and(client_param).for_each(|a, &c| {
                    *a = *a + weight * c;
                });
            }
        }

        // Update global parameters and increment round
        self.global_parameters = Some(aggregated.clone());
        self.client_updates.clear();
        self.round_count += 1;

        Ok(aggregated)
    }

    /// Get the number of completed communication rounds
    pub fn get_round_count(&self) -> usize {
        self.round_count
    }

    /// Get a reference to the current configuration
    pub fn get_config(&self) -> &FedProxConfig<A> {
        &self.config
    }

    /// Get a reference to the current global parameters, if set
    pub fn get_global_parameters(&self) -> Option<&Vec<Array<A, D>>> {
        self.global_parameters.as_ref()
    }

    /// Get the number of client updates collected so far in this round
    pub fn get_pending_updates_count(&self) -> usize {
        self.client_updates.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, Ix1};

    #[test]
    fn test_fedprox_config_builder() {
        // Test default configuration
        let config: FedProxConfig<f64> =
            FedProxConfigBuilder::new(10).build().expect("build failed");
        assert_eq!(config.num_clients, 10);
        assert!((config.mu - 0.01).abs() < 1e-10);
        assert_eq!(config.local_epochs, 5);
        assert!((config.participation_rate - 1.0).abs() < 1e-10);

        // Test custom configuration
        let config: FedProxConfig<f64> = FedProxConfigBuilder::new(20)
            .mu(0.1)
            .local_epochs(10)
            .participation_rate(0.5)
            .build()
            .expect("build failed");
        assert_eq!(config.num_clients, 20);
        assert!((config.mu - 0.1).abs() < 1e-10);
        assert_eq!(config.local_epochs, 10);
        assert!((config.participation_rate - 0.5).abs() < 1e-10);

        // Test invalid mu
        let result: std::result::Result<FedProxConfig<f64>, _> =
            FedProxConfigBuilder::new(5).mu(-0.1).build();
        assert!(result.is_err());

        // Test invalid participation_rate
        let result: std::result::Result<FedProxConfig<f64>, _> =
            FedProxConfigBuilder::new(5).participation_rate(0.0).build();
        assert!(result.is_err());

        let result: std::result::Result<FedProxConfig<f64>, _> =
            FedProxConfigBuilder::new(5).participation_rate(1.5).build();
        assert!(result.is_err());

        // Test invalid local_epochs
        let result: std::result::Result<FedProxConfig<f64>, _> =
            FedProxConfigBuilder::new(5).local_epochs(0).build();
        assert!(result.is_err());
    }

    #[test]
    fn test_fedprox_set_global_parameters() {
        let config: FedProxConfig<f64> = FedProxConfig::new(3);
        let mut optimizer: FedProxOptimizer<f64, Ix1> = FedProxOptimizer::new(config);

        // Setting valid parameters
        let params = vec![
            Array1::from_vec(vec![1.0, 2.0, 3.0]),
            Array1::from_vec(vec![4.0, 5.0]),
        ];
        assert!(optimizer.set_global_parameters(&params).is_ok());
        assert!(optimizer.get_global_parameters().is_some());

        // Verify stored parameters match
        let stored = optimizer
            .get_global_parameters()
            .expect("should have params");
        assert_eq!(stored.len(), 2);
        assert_eq!(stored[0].len(), 3);
        assert_eq!(stored[1].len(), 2);

        // Setting empty parameters should fail
        let empty: Vec<Array1<f64>> = vec![];
        assert!(optimizer.set_global_parameters(&empty).is_err());
    }

    #[test]
    fn test_local_update_with_proximal_term() {
        let config: FedProxConfig<f64> = FedProxConfigBuilder::new(2)
            .mu(0.1)
            .build()
            .expect("build failed");
        let mut optimizer: FedProxOptimizer<f64, Ix1> = FedProxOptimizer::new(config);

        // Set global parameters
        let global = vec![Array1::from_vec(vec![1.0, 2.0, 3.0])];
        optimizer
            .set_global_parameters(&global)
            .expect("set params failed");

        // Local parameters diverge from global
        let local = vec![Array1::from_vec(vec![1.5, 2.5, 3.5])];
        let grads = vec![Array1::from_vec(vec![0.1, 0.2, 0.3])];
        let lr: f64 = 0.01;

        let updated = optimizer
            .local_update(&local, &grads, lr)
            .expect("local_update failed");

        // Expected: w_new = w - lr * (grad + mu * (w - w_global))
        // For element 0: 1.5 - 0.01 * (0.1 + 0.1 * (1.5 - 1.0))
        //              = 1.5 - 0.01 * (0.1 + 0.05)
        //              = 1.5 - 0.0015 = 1.4985
        assert!((updated[0][0] - 1.4985).abs() < 1e-10);

        // For element 1: 2.5 - 0.01 * (0.2 + 0.1 * (2.5 - 2.0))
        //              = 2.5 - 0.01 * (0.2 + 0.05)
        //              = 2.5 - 0.0025 = 2.4975
        assert!((updated[0][1] - 2.4975).abs() < 1e-10);

        // For element 2: 3.5 - 0.01 * (0.3 + 0.1 * (3.5 - 3.0))
        //              = 3.5 - 0.01 * (0.3 + 0.05)
        //              = 3.5 - 0.0035 = 3.4965
        assert!((updated[0][2] - 3.4965).abs() < 1e-10);
    }

    #[test]
    fn test_proximal_gradient_computation() {
        let config: FedProxConfig<f64> = FedProxConfigBuilder::new(2)
            .mu(0.5)
            .build()
            .expect("build failed");
        let mut optimizer: FedProxOptimizer<f64, Ix1> = FedProxOptimizer::new(config);

        let global = vec![Array1::from_vec(vec![1.0, 2.0, 3.0])];
        optimizer
            .set_global_parameters(&global)
            .expect("set params failed");

        let local = vec![Array1::from_vec(vec![2.0, 4.0, 6.0])];
        let prox = optimizer
            .compute_proximal_gradient(&local)
            .expect("proximal gradient failed");

        // Expected: mu * (local - global) = 0.5 * (local - global)
        // [0.5*(2-1), 0.5*(4-2), 0.5*(6-3)] = [0.5, 1.0, 1.5]
        assert!((prox[0][0] - 0.5).abs() < 1e-10);
        assert!((prox[0][1] - 1.0).abs() < 1e-10);
        assert!((prox[0][2] - 1.5).abs() < 1e-10);

        // Test error when global parameters not set
        let config2: FedProxConfig<f64> = FedProxConfig::new(2);
        let optimizer2: FedProxOptimizer<f64, Ix1> = FedProxOptimizer::new(config2);
        assert!(optimizer2.compute_proximal_gradient(&local).is_err());

        // Test dimension mismatch
        let mismatched = vec![
            Array1::from_vec(vec![1.0, 2.0]),
            Array1::from_vec(vec![3.0]),
        ];
        assert!(optimizer.compute_proximal_gradient(&mismatched).is_err());
    }

    #[test]
    fn test_aggregate_updates_weighted() {
        let config: FedProxConfig<f64> = FedProxConfig::new(3);
        let mut optimizer: FedProxOptimizer<f64, Ix1> = FedProxOptimizer::new(config);

        let global = vec![Array1::from_vec(vec![0.0, 0.0])];
        optimizer
            .set_global_parameters(&global)
            .expect("set params failed");

        // Client 0: params=[2.0, 4.0], data_size=100
        optimizer
            .submit_client_update(0, &[Array1::from_vec(vec![2.0, 4.0])], 100)
            .expect("submit failed");

        // Client 1: params=[4.0, 6.0], data_size=300
        optimizer
            .submit_client_update(1, &[Array1::from_vec(vec![4.0, 6.0])], 300)
            .expect("submit failed");

        assert_eq!(optimizer.get_pending_updates_count(), 2);

        let aggregated = optimizer.aggregate_updates().expect("aggregate failed");

        // Weighted average: (100/400)*[2,4] + (300/400)*[4,6]
        //                 = 0.25*[2,4] + 0.75*[4,6]
        //                 = [0.5+3.0, 1.0+4.5]
        //                 = [3.5, 5.5]
        assert!((aggregated[0][0] - 3.5).abs() < 1e-10);
        assert!((aggregated[0][1] - 5.5).abs() < 1e-10);

        // Round count should have incremented
        assert_eq!(optimizer.get_round_count(), 1);
        // Client updates should be cleared
        assert_eq!(optimizer.get_pending_updates_count(), 0);

        // Aggregating with no updates should fail
        assert!(optimizer.aggregate_updates().is_err());
    }

    #[test]
    fn test_fedprox_mu_zero_is_fedavg() {
        // When mu=0, FedProx should behave identically to FedAvg
        // (the proximal gradient becomes zero)
        let config_prox: FedProxConfig<f64> = FedProxConfigBuilder::new(2)
            .mu(0.0)
            .build()
            .expect("build failed");
        let mut optimizer_prox: FedProxOptimizer<f64, Ix1> = FedProxOptimizer::new(config_prox);

        let global = vec![Array1::from_vec(vec![1.0, 2.0, 3.0])];
        optimizer_prox
            .set_global_parameters(&global)
            .expect("set params failed");

        // Verify proximal gradient is zero when mu=0
        let local = vec![Array1::from_vec(vec![5.0, 10.0, 15.0])];
        let prox_grad = optimizer_prox
            .compute_proximal_gradient(&local)
            .expect("proximal gradient failed");
        for val in prox_grad[0].iter() {
            assert!(
                val.abs() < 1e-15,
                "Proximal gradient should be zero when mu=0"
            );
        }

        // Verify local update with mu=0 is plain SGD
        let grads = vec![Array1::from_vec(vec![0.1, 0.2, 0.3])];
        let lr: f64 = 0.01;
        let updated = optimizer_prox
            .local_update(&local, &grads, lr)
            .expect("local_update failed");

        // Plain SGD: w_new = w - lr * grad
        // [5.0 - 0.01*0.1, 10.0 - 0.01*0.2, 15.0 - 0.01*0.3]
        // = [4.999, 9.998, 14.997]
        assert!((updated[0][0] - 4.999).abs() < 1e-10);
        assert!((updated[0][1] - 9.998).abs() < 1e-10);
        assert!((updated[0][2] - 14.997).abs() < 1e-10);

        // Verify aggregation with mu=0 gives same weighted average as FedAvg
        optimizer_prox
            .submit_client_update(0, &[Array1::from_vec(vec![2.0, 3.0, 4.0])], 200)
            .expect("submit failed");
        optimizer_prox
            .submit_client_update(1, &[Array1::from_vec(vec![4.0, 5.0, 6.0])], 200)
            .expect("submit failed");

        let agg = optimizer_prox
            .aggregate_updates()
            .expect("aggregate failed");

        // Equal weights (200 each) => arithmetic mean
        // [(2+4)/2, (3+5)/2, (4+6)/2] = [3.0, 4.0, 5.0]
        assert!((agg[0][0] - 3.0).abs() < 1e-10);
        assert!((agg[0][1] - 4.0).abs() < 1e-10);
        assert!((agg[0][2] - 5.0).abs() < 1e-10);
    }
}
