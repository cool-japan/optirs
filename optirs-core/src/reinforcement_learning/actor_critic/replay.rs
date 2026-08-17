use crate::error::{OptimError, Result};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;

/// Experience replay buffer entry
#[derive(Debug, Clone)]
pub struct Experience<T: Float + Debug + Send + Sync + 'static> {
    /// State (observation)
    pub state: Array1<T>,

    /// Action taken
    pub action: Array1<T>,

    /// Reward received
    pub reward: T,

    /// Next state
    pub next_state: Array1<T>,

    /// Done flag
    pub done: bool,

    /// Priority (for prioritized replay)
    pub priority: T,

    /// Additional info
    pub info: HashMap<String, T>,
}

/// A sampled replay mini-batch.
///
/// Carries the buffer indices alongside the experiences so their priorities can be
/// refreshed with the TD errors the update produces, plus the importance-sampling
/// weights that correct the bias introduced by non-uniform sampling.
#[derive(Debug, Clone)]
pub struct ReplaySample<T: Float + Debug + Send + Sync + 'static> {
    /// The sampled transitions.
    pub experiences: Vec<Experience<T>>,

    /// Buffer index of each sampled transition.
    pub indices: Vec<usize>,

    /// Importance-sampling weights `w_i = (1 / (N · P(i)))^β`, normalized by their
    /// maximum so they only ever scale gradients *down*. Uniform sampling yields
    /// all-ones.
    pub weights: Vec<T>,
}

/// Experience replay buffer with genuine prioritized sampling.
///
/// When `prioritized` is enabled, transition `i` is drawn with probability
/// `P(i) = p_iᵅ / Σ_k p_kᵅ` using an O(log N) sum tree, and the resulting bias is
/// corrected with importance weights `(1/(N·P(i)))^β` (Schaul et al., 2016). With
/// `prioritized` disabled the buffer samples uniformly and returns unit weights.
///
/// The previous implementation advertised prioritized replay but sampled uniformly
/// and never touched `alpha`/`beta`, and it panicked on an empty buffer
/// (`gen_range(0..0)`) and on `maxsize == 0` (`% 0`). Both are now impossible:
/// `new` rejects a zero capacity and `sample` returns an error on an empty buffer.
pub struct ExperienceReplayBuffer<T: Float + Debug + Send + Sync + 'static> {
    /// Buffer storage
    buffer: Vec<Experience<T>>,

    /// Maximum buffer size
    maxsize: usize,

    /// Next write position (ring buffer)
    position: usize,

    /// Whether the buffer has wrapped at least once
    is_full: bool,

    /// Prioritization exponent (0 = uniform, 1 = fully prioritized)
    alpha: T,

    /// Importance-sampling correction exponent
    beta: T,

    /// Whether prioritized sampling is active
    prioritized: bool,

    /// Sum tree over `p^α`, laid out as a complete binary tree in
    /// `[1, 2·capacity)` with leaves at `[capacity, capacity + maxsize)`.
    priority_tree: Vec<T>,

    /// Power-of-two leaf capacity of the sum tree.
    capacity: usize,

    /// Largest raw priority observed, used to seed new transitions so every
    /// transition is replayed at least once.
    max_priority: T,
}

impl<T: Float + Debug + Send + Sync + 'static> ExperienceReplayBuffer<T> {
    /// Small constant keeping every priority strictly positive.
    fn priority_epsilon() -> T {
        T::from(1e-6).unwrap_or_else(T::epsilon)
    }

    /// Create a new experience replay buffer.
    ///
    /// Returns an error for `maxsize == 0` (a zero-capacity ring buffer cannot
    /// store anything and its modular arithmetic would divide by zero).
    pub fn new(maxsize: usize, alpha: T, beta: T, prioritized: bool) -> Result<Self> {
        if maxsize == 0 {
            return Err(OptimError::InvalidConfig(
                "replay buffer capacity must be greater than zero".to_string(),
            ));
        }
        let capacity = maxsize.next_power_of_two();
        Ok(Self {
            buffer: Vec::with_capacity(maxsize),
            maxsize,
            position: 0,
            is_full: false,
            alpha,
            beta,
            prioritized,
            priority_tree: vec![T::zero(); 2 * capacity],
            capacity,
            max_priority: T::one(),
        })
    }

    /// Whether prioritized sampling is active.
    pub fn is_prioritized(&self) -> bool {
        self.prioritized
    }

    /// Total prioritized mass `Σ p^α` currently stored.
    pub fn total_priority(&self) -> T {
        self.priority_tree[1]
    }

    /// Write `p^α` into leaf `index` and propagate the change to the root.
    fn set_tree_priority(&mut self, index: usize, priority: T) {
        let clamped = priority.max(Self::priority_epsilon());
        let weighted = clamped.powf(self.alpha);

        let mut node = self.capacity + index;
        self.priority_tree[node] = weighted;
        while node > 1 {
            node /= 2;
            self.priority_tree[node] =
                self.priority_tree[2 * node] + self.priority_tree[2 * node + 1];
        }
    }

    /// Locate the leaf whose cumulative interval contains `value`.
    fn find_leaf(&self, mut value: T) -> usize {
        let mut node = 1usize;
        while node < self.capacity {
            let left = 2 * node;
            if value <= self.priority_tree[left] {
                node = left;
            } else {
                value = value - self.priority_tree[left];
                node = left + 1;
            }
        }
        (node - self.capacity).min(self.len().saturating_sub(1))
    }

    /// Add experience to buffer.
    ///
    /// A transition with a non-positive `priority` is inserted at the maximum
    /// priority seen so far, the standard PER convention that guarantees every new
    /// transition is replayed at least once.
    pub fn add(&mut self, experience: Experience<T>) {
        let priority = if experience.priority > T::zero() {
            experience.priority
        } else {
            self.max_priority
        };
        if priority > self.max_priority {
            self.max_priority = priority;
        }

        let index = self.position;
        if self.buffer.len() < self.maxsize {
            self.buffer.push(experience);
        } else {
            self.buffer[index] = experience;
            self.is_full = true;
        }

        self.set_tree_priority(index, priority);
        self.position = (self.position + 1) % self.maxsize;
    }

    /// Sample a mini-batch.
    ///
    /// Prioritized mode uses stratified sampling over `batchsize` equal segments of
    /// the total priority mass (lower variance than independent draws) and returns
    /// max-normalized importance weights. Returns an error when the buffer is
    /// empty or `batchsize` is zero instead of panicking inside the RNG.
    pub fn sample(&self, batchsize: usize) -> Result<ReplaySample<T>> {
        let available = self.len();
        if available == 0 {
            return Err(OptimError::InvalidState(
                "cannot sample from an empty replay buffer".to_string(),
            ));
        }
        if batchsize == 0 {
            return Err(OptimError::InvalidConfig(
                "replay sample size must be greater than zero".to_string(),
            ));
        }

        let sample_size = batchsize.min(available);
        let mut rng = scirs2_core::random::thread_rng();

        let mut indices = Vec::with_capacity(sample_size);
        let mut weights = Vec::with_capacity(sample_size);

        let total = self.total_priority();
        let use_priorities = self.prioritized && total > T::zero();

        if !use_priorities {
            for _ in 0..sample_size {
                indices.push(rng.gen_range(0..available));
                weights.push(T::one());
            }
        } else {
            let n = T::from(available).unwrap_or_else(T::one);
            let segment = total / T::from(sample_size).unwrap_or_else(T::one);
            let mut max_weight = T::zero();

            for k in 0..sample_size {
                let offset = T::from(k as f64 + rng.random::<f64>()).unwrap_or_else(T::zero);
                let value = (segment * offset).min(total);
                let index = self.find_leaf(value);

                // P(i) = p_iᵅ / Σ p^α, w_i = (1 / (N·P(i)))^β
                let leaf = self.priority_tree[self.capacity + index];
                let probability = if total > T::zero() {
                    (leaf / total).max(T::from(1e-12).unwrap_or_else(T::epsilon))
                } else {
                    T::one() / n
                };
                let weight = (T::one() / (n * probability)).powf(self.beta);
                if weight > max_weight {
                    max_weight = weight;
                }

                indices.push(index);
                weights.push(weight);
            }

            if max_weight > T::zero() {
                for weight in weights.iter_mut() {
                    *weight = *weight / max_weight;
                }
            }
        }

        let experiences = indices.iter().map(|&i| self.buffer[i].clone()).collect();

        Ok(ReplaySample {
            experiences,
            indices,
            weights,
        })
    }

    /// Refresh the priorities of previously sampled transitions from their TD errors.
    pub fn update_priorities(&mut self, indices: &[usize], td_errors: &[T]) -> Result<()> {
        if indices.len() != td_errors.len() {
            return Err(OptimError::DimensionMismatch(format!(
                "priority update needs one TD error per index ({} vs {})",
                indices.len(),
                td_errors.len()
            )));
        }

        let available = self.len();
        for (&index, &error) in indices.iter().zip(td_errors.iter()) {
            if index >= available {
                return Err(OptimError::InvalidParameter(format!(
                    "replay index {index} out of range (buffer holds {available} transitions)"
                )));
            }
            let priority = error.abs() + Self::priority_epsilon();
            if priority > self.max_priority {
                self.max_priority = priority;
            }
            self.buffer[index].priority = priority;
            self.set_tree_priority(index, priority);
        }
        Ok(())
    }

    /// Get buffer size
    pub fn len(&self) -> usize {
        if self.is_full {
            self.maxsize
        } else {
            self.buffer.len()
        }
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}
