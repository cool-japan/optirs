//! FoolsGold and FLAME

use crate::error::{OptimError, Result};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::Float;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt::Debug;

use super::aggregator::ByzantineTolerantAggregator;
use super::helpers::{cosine_similarity, from_scalar, l2_norm, to_scalar, FLAME_NOISE_LAMBDA};
use super::types::SplitMix64;

impl<T: Float + Debug + Send + Sync + 'static + scirs2_core::ndarray::ScalarOperand>
    ByzantineTolerantAggregator<T>
{
    /// FoolsGold aggregation (Fung et al., 2020).
    ///
    /// Sybils share an objective, so their *historical* update directions stay
    /// mutually similar while honest clients diverge. Each participant's updates
    /// are accumulated across rounds, the pairwise cosine similarity matrix of
    /// those histories is built, pardoning re-scales the row of a client that is
    /// only similar to a more-similar client, and the resulting `alpha` is
    /// rescaled and passed through the logit `ln(a / (1 - a)) + 0.5` clipped to
    /// `[0, 1]` to obtain the per-client learning rate.
    pub(super) fn fools_gold_aggregation(
        &mut self,
        gradients: &HashMap<String, Array1<T>>,
    ) -> Result<Array1<T>> {
        Self::validate_cohort(gradients)?;
        let ids: Vec<String> = {
            let ordered = Self::ordered_cohort(gradients);
            ordered.iter().map(|(id, _)| (*id).clone()).collect()
        };
        for participant_id in &ids {
            let gradient = gradients.get(participant_id).ok_or_else(|| {
                OptimError::InvalidState(format!("missing gradient for '{participant_id}'"))
            })?;
            match self.fools_gold_history.get_mut(participant_id) {
                Some(history) if history.len() == gradient.len() => {
                    *history = &*history + gradient;
                }
                _ => {
                    self.fools_gold_history
                        .insert(participant_id.clone(), gradient.clone());
                }
            }
        }
        let weights = self.compute_fools_gold_weights(&ids)?;
        let dim = gradients
            .get(&ids[0])
            .map(|g| g.len())
            .ok_or_else(|| OptimError::InvalidState("empty FoolsGold cohort".to_string()))?;
        let mut result: Array1<T> = Array1::zeros(dim);
        let mut total_weight = 0.0f64;
        for (participant_id, weight) in ids.iter().zip(weights.iter()) {
            let gradient = gradients.get(participant_id).ok_or_else(|| {
                OptimError::InvalidState(format!("missing gradient for '{participant_id}'"))
            })?;
            let scaled: T = to_scalar(*weight)?;
            result = result + gradient * scaled;
            total_weight += *weight;
        }
        if total_weight > 0.0 {
            let divisor: T = to_scalar(total_weight)?;
            Ok(result / divisor)
        } else {
            let mut mean: Array1<T> = Array1::zeros(dim);
            for participant_id in &ids {
                let gradient = gradients.get(participant_id).ok_or_else(|| {
                    OptimError::InvalidState(format!("missing gradient for '{participant_id}'"))
                })?;
                mean = mean + gradient;
            }
            let divisor: T = to_scalar(ids.len() as f64)?;
            Ok(mean / divisor)
        }
    }
    /// FoolsGold per-client learning rates in `[0, 1]`, computed from the
    /// accumulated update histories.
    pub(super) fn compute_fools_gold_weights(&self, ids: &[String]) -> Result<Vec<f64>> {
        let n = ids.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        if n == 1 {
            return Ok(vec![1.0]);
        }
        let histories: Vec<&Array1<T>> = ids
            .iter()
            .map(|id| {
                self.fools_gold_history.get(id).ok_or_else(|| {
                    OptimError::InvalidState(format!("missing FoolsGold history for '{id}'"))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut cs = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let similarity = if histories[i].len() == histories[j].len() {
                    from_scalar(cosine_similarity(histories[i], histories[j])?)?
                } else {
                    0.0
                };
                cs[i][j] = similarity;
                cs[j][i] = similarity;
            }
        }
        let row_max = |row: &[f64], skip: usize| -> f64 {
            row.iter()
                .enumerate()
                .filter(|(j, _)| *j != skip)
                .map(|(_, v)| *v)
                .fold(f64::NEG_INFINITY, f64::max)
        };
        let v: Vec<f64> = (0..n).map(|i| row_max(&cs[i], i)).collect();
        let mut pardoned = cs.clone();
        for i in 0..n {
            for j in 0..n {
                if i != j && v[j] > v[i] && v[j] > 0.0 {
                    pardoned[i][j] *= v[i] / v[j];
                }
            }
        }
        let mut alpha: Vec<f64> = (0..n)
            .map(|i| (1.0 - row_max(&pardoned[i], i)).clamp(0.0, 1.0))
            .collect();
        let max_alpha = alpha.iter().copied().fold(0.0f64, f64::max);
        if max_alpha > 0.0 {
            for a in alpha.iter_mut() {
                *a = (*a / max_alpha).clamp(0.0, 1.0);
            }
        } else {
            return Ok(vec![0.0; n]);
        }
        Ok(alpha
            .into_iter()
            .map(|a| {
                let clamped = a.clamp(1e-9, 0.99);
                let logit = (clamped / (1.0 - clamped)).ln() + 0.5;
                if logit.is_finite() {
                    logit.clamp(0.0, 1.0)
                } else {
                    1.0
                }
            })
            .collect())
    }
    /// FLAME aggregation (Nguyen et al., 2022).
    ///
    /// 1. Deterministic average-linkage agglomerative clustering on cosine
    ///    distances, merging until a cluster reaches `min_cluster_size = n/2 + 1`;
    ///    that cluster is admitted and everything else is discarded.
    /// 2. Norm-median clipping: every admitted update is scaled by
    ///    `min(1, S_t / ||g_i||)` where `S_t` is the median L2 norm over the whole
    ///    cohort, so a scaling attack cannot dominate the mean.
    /// 3. Calibrated Gaussian noise with `sigma = lambda * S_t` is added to the
    ///    clipped mean.
    ///
    /// The noise is a backdoor-mitigation measure and carries **no** differential
    /// privacy guarantee; it is drawn from a deterministic generator seeded by the
    /// round counter, so a given sequence of rounds is reproducible.
    pub(super) fn flame_aggregation(
        &self,
        gradients: &HashMap<String, Array1<T>>,
    ) -> Result<Array1<T>> {
        Self::validate_cohort(gradients)?;
        let ordered = Self::ordered_cohort(gradients);
        let grads: Vec<&Array1<T>> = ordered.iter().map(|(_, g)| *g).collect();
        let n = grads.len();
        let dim = grads[0].len();
        let admitted = Self::flame_admitted_indices(&grads)?;
        if admitted.is_empty() {
            return Err(OptimError::InvalidState(
                "FLAME clustering admitted no gradients".to_string(),
            ));
        }
        let mut norms: Vec<T> = grads.iter().map(|g| l2_norm(g)).collect();
        norms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let two: T = to_scalar(2.0)?;
        let median_norm = if norms.len().is_multiple_of(2) {
            let mid = norms.len() / 2;
            (norms[mid - 1] + norms[mid]) / two
        } else {
            norms[norms.len() / 2]
        };
        let mut result: Array1<T> = Array1::zeros(dim);
        for &index in &admitted {
            let gradient = grads[index];
            let norm = l2_norm(gradient);
            let gamma = if norm > T::zero() && norm > median_norm {
                median_norm / norm
            } else {
                T::one()
            };
            result = result + gradient * gamma;
        }
        let divisor: T = to_scalar(admitted.len() as f64)?;
        result = result / divisor;
        let sigma = from_scalar(median_norm)? * FLAME_NOISE_LAMBDA;
        if sigma > 0.0 {
            let seed = self
                .round
                .wrapping_mul(0x0000_0100_0000_01B3)
                .wrapping_add(n as u64)
                .wrapping_add(0x0000_0F1A_3E5E_ED00);
            let mut rng = SplitMix64::new(seed);
            for value in result.iter_mut() {
                let noise: T = to_scalar(rng.next_gaussian() * sigma)?;
                *value = *value + noise;
            }
        }
        Ok(result)
    }
    /// Indices admitted by FLAME's clustering stage, sorted ascending.
    ///
    /// Deterministic average-linkage agglomerative clustering on cosine distances:
    /// the closest pair of clusters is merged repeatedly (ties resolve to the
    /// lowest index pair) until one cluster reaches `n / 2 + 1` members.
    pub(super) fn flame_admitted_indices(grads: &[&Array1<T>]) -> Result<Vec<usize>> {
        let n = grads.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        let min_size = n / 2 + 1;
        let mut distance = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let d = 1.0 - from_scalar(cosine_similarity(grads[i], grads[j])?)?;
                distance[i][j] = d;
                distance[j][i] = d;
            }
        }
        let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
        if clusters[0].len() >= min_size {
            return Ok(clusters[0].clone());
        }
        while clusters.len() > 1 {
            let mut best: Option<(usize, usize, f64)> = None;
            for a in 0..clusters.len() {
                for b in (a + 1)..clusters.len() {
                    let mut sum = 0.0;
                    for &x in &clusters[a] {
                        for &y in &clusters[b] {
                            sum += distance[x][y];
                        }
                    }
                    let linkage = sum / (clusters[a].len() * clusters[b].len()) as f64;
                    if best.is_none_or(|(_, _, current)| linkage < current) {
                        best = Some((a, b, linkage));
                    }
                }
            }
            let Some((a, b, _)) = best else { break };
            let merged = clusters.remove(b);
            clusters[a].extend(merged);
            if clusters[a].len() >= min_size {
                let mut out = clusters[a].clone();
                out.sort_unstable();
                return Ok(out);
            }
        }
        let mut out = clusters.into_iter().next().unwrap_or_default();
        out.sort_unstable();
        Ok(out)
    }
    /// Participant ids admitted by FLAME's clustering stage, sorted ascending.
    pub fn flame_admitted_ids(
        &self,
        gradients: &HashMap<String, Array1<T>>,
    ) -> Result<Vec<String>> {
        Self::validate_cohort(gradients)?;
        let ordered = Self::ordered_cohort(gradients);
        let grads: Vec<&Array1<T>> = ordered.iter().map(|(_, g)| *g).collect();
        let admitted = Self::flame_admitted_indices(&grads)?;
        Ok(admitted
            .into_iter()
            .map(|index| ordered[index].0.clone())
            .collect())
    }
}
