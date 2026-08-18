// Dropout regularization applied to gradients

use scirs2_core::ndarray::{Array, Dimension, ScalarOperand, Zip};
use scirs2_core::numeric::Float;
use scirs2_core::random::Rng;
use scirs2_core::Random;
use std::cell::RefCell;
use std::fmt::Debug;

use crate::error::Result;
use crate::regularizers::Regularizer;

/// Dropout regularization for **gradients**
///
/// This type implements the [`Regularizer`] trait, so it operates on the
/// gradient array handed to [`Regularizer::apply`], not on layer activations.
/// While in training mode it zeroes each gradient entry independently with
/// probability `rate` and rescales the survivors by `1 / (1 - rate)` (inverted
/// dropout), which keeps the expected gradient unchanged. In evaluation mode —
/// or when `rate` is zero — gradients pass through untouched.
///
/// Applying the same idea to activations requires a forward/backward pass that
/// this crate's optimizer-side [`Regularizer`] interface does not model; use
/// [`crate::regularizers::SpatialDropout`] or a neural-network layer for that.
///
/// A fresh mask is drawn on every [`Regularizer::apply`] call — masks are never
/// cached or reused across calls, so consecutive calls with identical gradients
/// generally produce different results.
///
/// [`Regularizer::apply`]: crate::regularizers::Regularizer::apply
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array1;
/// use optirs_core::regularizers::Dropout;
/// use scirs2_core::random::SeedableRng;
/// use scirs2_core::random::rngs::SmallRng;
///
/// // Create a dropout regularizer with 0.5 dropout rate
/// let seed = [0u8; 32];
/// let mut rng = SmallRng::from_seed(seed);
/// let mut dropout = Dropout::new(0.5f64, &mut rng);
///
/// // Set to training mode
/// dropout.train();
///
/// // Check the dropout rate
/// assert_eq!(dropout.rate(), 0.5);
///
/// // Set to evaluation mode
/// dropout.eval();
/// assert!(!dropout.is_training());
/// ```
#[derive(Debug)]
pub struct Dropout<A: Float + Debug> {
    /// Dropout rate (fraction of gradient entries that are dropped)
    rate: A,
    /// Random number generator
    rng: RefCell<Random<scirs2_core::random::rngs::StdRng>>,
    /// Boolean indicating whether in training mode
    training: bool,
}

impl<A: Float + Debug + Send + Sync> Dropout<A> {
    /// Create a new dropout regularizer
    ///
    /// # Arguments
    ///
    /// * `rate` - Dropout rate (0.0 to 1.0, fraction of entries that are dropped)
    /// * `rng` - Random number generator used to seed this regularizer's own RNG
    pub fn new<R: Rng>(rate: A, rng: &mut R) -> Self {
        // Ensure rate is between 0 and 1
        let rate = rate.max(A::zero()).min(A::one());

        // Create a new RNG from the provided one
        let mut seed_bytes = [0u8; 8];
        rng.fill_bytes(&mut seed_bytes);
        let seed = u64::from_ne_bytes(seed_bytes);
        let rng = Random::seed(seed);

        Self {
            rate,
            rng: RefCell::new(rng),
            training: true,
        }
    }

    /// Get the dropout rate
    pub fn rate(&self) -> A {
        self.rate
    }

    /// Set the dropout rate
    ///
    /// # Arguments
    ///
    /// * `rate` - Dropout rate (0.0 to 1.0, fraction of entries that are dropped)
    pub fn set_rate(&mut self, rate: A) -> &mut Self {
        // Ensure rate is between 0 and 1
        self.rate = rate.max(A::zero()).min(A::one());
        self
    }

    /// Set to training mode (apply dropout to gradients)
    pub fn train(&mut self) -> &mut Self {
        self.training = true;
        self
    }

    /// Set to inference mode (gradients pass through unchanged)
    pub fn eval(&mut self) -> &mut Self {
        self.training = false;
        self
    }

    /// Get the training mode
    pub fn is_training(&self) -> bool {
        self.training
    }

    /// Draw a fresh dropout mask for the given shape
    ///
    /// During training each entry is independently set to `0` with probability
    /// `rate` and to `1 / (1 - rate)` otherwise, so the mask has unit mean and
    /// masking leaves the expected gradient unchanged. Outside training mode (or
    /// with a zero rate) an all-ones mask is returned.
    fn create_mask<D: Dimension>(&self, shape: D) -> Array<A, D> {
        if !self.training || self.rate <= A::zero() {
            // In eval mode or with 0 dropout rate, no masking is applied
            return Array::ones(shape);
        }

        // The scale factor for the kept entries is 1/(1-rate); this maintains
        // the expected magnitude of the masked gradients.
        let keep_prob = A::one() - self.rate;
        if keep_prob <= A::zero() {
            // rate == 1.0: everything is dropped, no finite rescaling exists.
            return Array::zeros(shape);
        }
        let scale = A::one() / keep_prob;

        // Compare in f64 so no fallible conversion of the random draw is needed.
        let rate = self.rate.to_f64().unwrap_or(0.0);
        let mut rng = self.rng.borrow_mut();
        let mut mask = Array::zeros(shape);
        for elem in mask.iter_mut() {
            let rand_val: f64 = rng.gen_range(0.0..1.0);
            if rand_val > rate {
                *elem = scale;
            }
        }

        mask
    }
}

impl<A, D> Regularizer<A, D> for Dropout<A>
where
    A: Float + ScalarOperand + Debug + Send + Sync,
    D: Dimension<Pattern = D>,
{
    /// Mask the **gradients** in place with a freshly drawn dropout mask.
    ///
    /// `params` is ignored: this regularizer perturbs the gradient signal, not
    /// the parameters. Always returns a zero penalty because dropout adds no
    /// term to the loss.
    fn apply(&self, _params: &Array<A, D>, gradients: &mut Array<A, D>) -> Result<A> {
        if !self.training || self.rate <= A::zero() {
            // In eval mode or with 0 dropout rate, no dropout is applied
            return Ok(A::zero());
        }

        // Draw a fresh mask for this call and apply it to the gradients.
        let mask = self.create_mask(gradients.dim());
        Zip::from(gradients).and(&mask).for_each(|grad, &mask_val| {
            *grad = *grad * mask_val;
        });

        // Dropout doesn't add a penalty term to the loss
        Ok(A::zero())
    }

    fn penalty(&self, _params: &Array<A, D>) -> Result<A> {
        // Dropout doesn't add a penalty term to the loss
        Ok(A::zero())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, ArrayD};
    use scirs2_core::random::rngs::SmallRng;
    use scirs2_core::random::SeedableRng;

    fn make_dropout(rate: f64) -> Dropout<f64> {
        let mut rng = SmallRng::from_seed([7u8; 32]);
        Dropout::new(rate, &mut rng)
    }

    /// The `Regularizer` impl is bound to `D: Dimension<Pattern = D>`, which only
    /// `IxDyn` satisfies, so all fixtures are dynamic-dimension arrays.
    fn dyn_vec(values: &[f64]) -> ArrayD<f64> {
        Array1::from_vec(values.to_vec()).into_dyn()
    }

    fn dyn_ones(len: usize) -> ArrayD<f64> {
        Array1::from_elem(len, 1.0).into_dyn()
    }

    #[test]
    fn eval_mode_leaves_gradients_untouched() {
        let mut dropout = make_dropout(0.5);
        dropout.eval();

        let params = dyn_vec(&[1.0, 2.0, 3.0, 4.0]);
        let original = dyn_vec(&[0.1, 0.2, 0.3, 0.4]);
        let mut gradients = original.clone();

        let penalty = dropout
            .apply(&params, &mut gradients)
            .expect("dropout apply failed");

        assert_eq!(penalty, 0.0);
        assert_eq!(gradients, original);
    }

    #[test]
    fn training_mode_masks_gradients_not_params() {
        let mut dropout = make_dropout(0.5);
        dropout.train();

        let params = dyn_ones(512);
        let params_before = params.clone();
        let mut gradients = dyn_ones(512);

        dropout
            .apply(&params, &mut gradients)
            .expect("dropout apply failed");

        // Parameters are never touched by this regularizer.
        assert_eq!(params, params_before);

        // Gradients are either zeroed or rescaled by 1/(1-rate) = 2.
        assert!(gradients
            .iter()
            .all(|&g| g == 0.0 || (g - 2.0).abs() < 1e-12));
        let dropped = gradients.iter().filter(|&&g| g == 0.0).count();
        assert!(dropped > 0, "no gradient entries were dropped");
        assert!(dropped < 512, "every gradient entry was dropped");

        // Inverted dropout keeps the expected gradient sum near the original.
        let sum: f64 = gradients.sum();
        assert!((sum - 512.0).abs() < 160.0, "unexpected gradient sum {sum}");
    }

    #[test]
    fn mask_is_redrawn_on_every_call() {
        // The mask must not be cached: two applies on identical gradients with a
        // non-trivial rate produce different results with overwhelming odds.
        let mut dropout = make_dropout(0.5);
        dropout.train();

        let params = dyn_ones(256);
        let mut first = dyn_ones(256);
        let mut second = dyn_ones(256);

        dropout
            .apply(&params, &mut first)
            .expect("dropout apply failed");
        dropout
            .apply(&params, &mut second)
            .expect("dropout apply failed");

        assert_ne!(first, second, "dropout mask appears to be cached");
    }

    #[test]
    fn zero_rate_is_identity_and_full_rate_zeroes_everything() {
        let params = dyn_vec(&[1.0, 2.0, 3.0]);
        let original = dyn_vec(&[0.5, -1.5, 2.5]);

        let mut none = make_dropout(0.0);
        none.train();
        let mut gradients = original.clone();
        none.apply(&params, &mut gradients)
            .expect("dropout apply failed");
        assert_eq!(gradients, original);

        let mut all = make_dropout(1.0);
        all.train();
        let mut gradients = original.clone();
        all.apply(&params, &mut gradients)
            .expect("dropout apply failed");
        assert_eq!(gradients, dyn_vec(&[0.0, 0.0, 0.0]));
    }

    #[test]
    fn penalty_is_always_zero() {
        let dropout = make_dropout(0.5);
        let params = dyn_vec(&[1.0, 2.0, 3.0]);
        assert_eq!(
            Regularizer::penalty(&dropout, &params).expect("penalty failed"),
            0.0
        );
    }
}
